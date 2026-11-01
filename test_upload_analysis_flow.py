#!/usr/bin/env python3
# ============================================================
# test_upload_analysis_flow.py
# 端對端測試：上傳 PCAP -> 規則偵測 -> CNN 推論 -> Grad-CAM -> PDF
# ============================================================
"""
直接呼叫 Django 的 analyzer.tasks.execute_pcap_analysis()，
模擬完整的 PCAP 分析流程，驗證各階段是否正常。

使用方式：
    python test_upload_analysis_flow.py
"""
import os
import sys
import time
import shutil
import traceback

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'network_platform.settings')

import django
django.setup()

from django.conf import settings
from django.contrib.auth import get_user_model

User = get_user_model()

# ── 測試設定 ──────────────────────────────────────────────
PROJECT_ROOT = settings.BASE_DIR

# 可用的模型與對應測試 PCAP
TEST_CASES = [
    {
        'model_key': 'mta_malware',
        'label': '非監督式 MTA (正常流量)',
        'pcap': 'test_pcaps/mta_unsupervised/normal_http_dns.pcap',
        'expect': 'mostly_normal',
    },
    {
        'model_key': 'mta_malware',
        'label': '非監督式 MTA (攻擊流量)',
        'pcap': 'test_pcaps/mta_unsupervised/attack_malware_c2.pcap',
        'expect': 'some_anomaly',
    },
    {
        'model_key': 'semi_mta_malware',
        'label': '半監督式 MTA (正常流量)',
        'pcap': 'test_pcaps/mta_semi/normal_http_dns.pcap',
        'expect': 'mostly_normal',
    },
    {
        'model_key': 'semi_mta_malware',
        'label': '半監督式 MTA (攻擊流量)',
        'pcap': 'test_pcaps/mta_semi/attack_malware_c2.pcap',
        'expect': 'some_anomaly',
    },
]

# 如果 CICIDS2017 模型存在，也加入測試
cicids_vae_path = settings.ANOMALY_MODELS.get('unsupervised_vae', {}).get('path')
cicids_semi_path = settings.ANOMALY_MODELS.get('semi_cicids2017', {}).get('path')

if cicids_vae_path and os.path.exists(str(cicids_vae_path)):
    TEST_CASES += [
        {
            'model_key': 'unsupervised_vae',
            'label': '非監督式 CICIDS2017-VAE (正常流量)',
            'pcap': 'test_pcaps/cicids2017_unsupervised/normal_http_dns.pcap',
            'expect': 'mostly_normal',
        },
        {
            'model_key': 'unsupervised_vae',
            'label': '非監督式 CICIDS2017-VAE (攻擊流量)',
            'pcap': 'test_pcaps/cicids2017_unsupervised/attack_portscan_bruteforce.pcap',
            'expect': 'some_anomaly',
        },
    ]
if cicids_semi_path and os.path.exists(str(cicids_semi_path)):
    TEST_CASES += [
        {
            'model_key': 'semi_cicids2017',
            'label': '半監督式 CICIDS2017 (正常流量)',
            'pcap': 'test_pcaps/cicids2017_semi/normal_http_dns.pcap',
            'expect': 'mostly_normal',
        },
        {
            'model_key': 'semi_cicids2017',
            'label': '半監督式 CICIDS2017 (攻擊流量)',
            'pcap': 'test_pcaps/cicids2017_semi/attack_portscan_bruteforce.pcap',
            'expect': 'some_anomaly',
        },
    ]


def get_or_create_test_user():
    """取得或建立測試用帳號。"""
    username = 'test_analysis_user'
    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        user = User.objects.create_user(
            username=username,
            password='testpass123',
            email='test@vnaap.local',
        )
        # 確保有 profile
        from accounts.models import UserProfile
        UserProfile.objects.get_or_create(
            user=user,
            defaults={
                'role': 'admin',
                'can_upload_pcap': True,
                'can_run_ai_analysis': True,
                'can_export_report': True,
            }
        )
    return user


def run_single_test(test_case, user, test_num, total_tests):
    """執行單一測試案例。"""
    from analyzer.models import AnalysisSession
    from django.core.files.uploadedfile import SimpleUploadedFile

    model_key = test_case['model_key']
    label = test_case['label']
    pcap_rel = test_case['pcap']
    expect = test_case['expect']
    pcap_abs = os.path.join(str(PROJECT_ROOT), pcap_rel)

    print(f'\n{"=" * 64}')
    print(f'  [{test_num}/{total_tests}] {label}')
    print(f'  模型: {model_key}')
    print(f'  PCAP: {pcap_rel}')
    print(f'  預期: {expect}')
    print(f'{"=" * 64}')

    if not os.path.exists(pcap_abs):
        print(f'  [SKIP] PCAP 不存在: {pcap_abs}')
        return {'status': 'SKIP', 'reason': 'PCAP 不存在'}

    # 確認模型檔案存在
    model_cfg = settings.ANOMALY_MODELS.get(model_key)
    if not model_cfg:
        print(f'  [SKIP] settings.ANOMALY_MODELS 沒有 {model_key}')
        return {'status': 'SKIP', 'reason': '模型設定不存在'}
    model_path = str(model_cfg.get('path', ''))
    if not os.path.exists(model_path):
        print(f'  [SKIP] 模型檔案不存在: {model_path}')
        return {'status': 'SKIP', 'reason': '模型檔案不存在'}

    # 建立 AnalysisSession
    with open(pcap_abs, 'rb') as f:
        pcap_content = f.read()

    uploaded = SimpleUploadedFile(
        name=os.path.basename(pcap_rel),
        content=pcap_content,
        content_type='application/octet-stream',
    )

    session = AnalysisSession.objects.create(
        mode='pcap',
        label=f'[TEST] {label}',
        pcap_file=uploaded,
        created_by=user,
        task_status='pending',
        model_key=model_key,
    )
    print(f'  Session #{session.pk} created')

    # 執行分析
    from analyzer.tasks import execute_pcap_analysis

    t0 = time.time()
    result = {}
    try:
        execute_pcap_analysis(
            session.pk,
            celery_task_id='test_sync',
            auto_gradcam=True,
        )
        elapsed = time.time() - t0

        # 重新載入
        session.refresh_from_db()
        print(f'\n  --- 分析結果 (耗時 {elapsed:.1f}s) ---')
        print(f'  狀態:          {session.task_status}')
        print(f'  進度:          {session.progress_percent}%')
        print(f'  階段:          {session.progress_stage}')
        print(f'  封包數:        {session.packet_count}')
        print(f'  規則告警數:    {session.alert_count}')

        # CNN 結果
        cnn = getattr(session, 'cnn_result', None)
        if cnn:
            total_samples = cnn.normal_count + cnn.anomaly_count
            print(f'\n  [CNN 結果]')
            print(f'    Threshold:    {cnn.threshold:.8g}')
            print(f'    樣本總數:     {total_samples}')
            print(f'    正常:         {cnn.normal_count}')
            print(f'    異常:         {cnn.anomaly_count}')
            print(f'    異常比例:     {cnn.detection_rate*100:.2f}%')
            print(f'    平均正常分數: {cnn.avg_normal_error:.8g}')
            print(f'    平均異常分數: {cnn.avg_attack_error:.8g}')
            if cnn.known_attack_count or cnn.unknown_attack_count:
                print(f'    已知攻擊:     {cnn.known_attack_count}')
                print(f'    未知攻擊:     {cnn.unknown_attack_count}')
            has_histogram = bool(cnn.error_histogram and cnn.error_histogram.name)
            print(f'    分數分布圖:   {"有" if has_histogram else "無"}')
            result['cnn'] = {
                'threshold': cnn.threshold,
                'normal': cnn.normal_count,
                'anomaly': cnn.anomaly_count,
                'rate': cnn.detection_rate,
                'has_histogram': has_histogram,
            }
        else:
            print(f'  [CNN] 無結果')
            result['cnn'] = None

        # Grad-CAM
        gcam_count = session.gradcam_images.count()
        gcam_anomaly = session.gradcam_images.filter(is_anomaly=True).count()
        gcam_normal = session.gradcam_images.filter(is_anomaly=False).count()
        print(f'\n  [Grad-CAM]')
        print(f'    影像總數:     {gcam_count}')
        print(f'    異常樣本:     {gcam_anomaly}')
        print(f'    正常參考:     {gcam_normal}')
        # 檢查影像檔案是否實際存在
        gcam_files_ok = 0
        for g in session.gradcam_images.all()[:4]:
            for field_name in ['original_image', 'heatmap_image', 'comparison_image']:
                img_field = getattr(g, field_name)
                if img_field and img_field.name:
                    full_path = os.path.join(str(settings.MEDIA_ROOT), img_field.name)
                    if os.path.exists(full_path):
                        gcam_files_ok += 1
        print(f'    影像檔存在:   {gcam_files_ok} 個')
        result['gradcam'] = {
            'total': gcam_count,
            'anomaly': gcam_anomaly,
            'normal': gcam_normal,
            'files_ok': gcam_files_ok,
        }

        # 告警
        if session.alert_count > 0:
            alerts = session.alerts.all()[:5]
            print(f'\n  [規則告警 前 5 筆]')
            for a in alerts:
                print(f'    {a.severity:8s} | {a.attack_type:20s} | {a.src_ip} -> {a.dst_ip}')
        result['alerts'] = session.alert_count

        # 合理性判斷
        print(f'\n  --- 合理性判斷 ---')
        verdict = 'PASS'
        issues = []

        if session.task_status != 'done':
            issues.append(f'task_status={session.task_status}, 預期 done')
            verdict = 'FAIL'

        if not cnn:
            issues.append('沒有 CNN 結果')
            verdict = 'FAIL'
        else:
            if expect == 'mostly_normal' and cnn.detection_rate > 0.50:
                issues.append(f'正常流量異常率 {cnn.detection_rate*100:.1f}% > 50%, 偏高')
                verdict = 'WARN'
            if expect == 'some_anomaly' and cnn.anomaly_count == 0:
                issues.append('攻擊流量一個異常都沒抓到')
                verdict = 'WARN'

        if gcam_count == 0:
            issues.append('沒有產生 Grad-CAM 影像')
            verdict = 'WARN' if verdict != 'FAIL' else verdict

        if not issues:
            print(f'  [PASS] 所有檢查通過')
        else:
            for iss in issues:
                print(f'  [{verdict}] {iss}')

        result['status'] = verdict
        result['issues'] = issues
        result['elapsed'] = elapsed
        return result

    except Exception as e:
        elapsed = time.time() - t0
        print(f'\n  [FAIL] 分析失敗 ({elapsed:.1f}s): {e}')
        traceback.print_exc()
        # 檢查 session 的錯誤狀態
        session.refresh_from_db()
        if session.error_message:
            print(f'  Session error: {session.error_message[:200]}')
        return {'status': 'FAIL', 'reason': str(e), 'elapsed': elapsed}


def main():
    print('=' * 64)
    print('  VNAAP 端對端分析流程測試')
    print(f'  測試案例: {len(TEST_CASES)}')
    print('=' * 64)

    user = get_or_create_test_user()
    print(f'  測試帳號: {user.username}')

    results = []
    for i, tc in enumerate(TEST_CASES, 1):
        r = run_single_test(tc, user, i, len(TEST_CASES))
        r['label'] = tc['label']
        results.append(r)

    # 總結
    print('\n')
    print('=' * 64)
    print('  測試總結')
    print('=' * 64)
    for r in results:
        status = r.get('status', '?')
        label = r.get('label', '?')
        elapsed = r.get('elapsed', 0)
        marker = {'PASS': '[v]', 'WARN': '[!]', 'FAIL': '[X]', 'SKIP': '[-]'}.get(status, '[?]')
        print(f'  {marker} {label:<45s} {elapsed:>6.1f}s')
        for iss in r.get('issues', []):
            print(f'       -> {iss}')

    n_pass = sum(1 for r in results if r['status'] == 'PASS')
    n_warn = sum(1 for r in results if r['status'] == 'WARN')
    n_fail = sum(1 for r in results if r['status'] == 'FAIL')
    n_skip = sum(1 for r in results if r['status'] == 'SKIP')
    print(f'\n  PASS: {n_pass}  WARN: {n_warn}  FAIL: {n_fail}  SKIP: {n_skip}')


if __name__ == '__main__':
    main()
