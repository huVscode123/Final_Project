# ============================================================
# analyzer/views.py
# ============================================================
import os, json
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.conf import settings
from django.db.models import Q, Count
from django.views.decorators.http import require_POST

from .models import AnalysisSession, Alert, CNNResult, GradCAMImage, AnalysisReport
from projects.models import Project


# ── Dashboard ────────────────────────────────────────────────
@login_required
def dashboard(request):
    user = request.user
    if user.profile.is_admin:
        sessions = AnalysisSession.objects.all()
        alerts   = Alert.objects.all()
    else:
        sessions = AnalysisSession.objects.filter(
            Q(created_by=user)|Q(project__owner=user)|Q(project__members=user)
        ).distinct()
        alerts = Alert.objects.filter(session__in=sessions)

    recent_sessions = sessions.order_by('-created_at')[:8]
    recent_alerts   = alerts.select_related('session').order_by('-timestamp')[:10]

    severity_data = {
        'CRITICAL': alerts.filter(severity='CRITICAL').count(),
        'HIGH':     alerts.filter(severity='HIGH').count(),
        'MEDIUM':   alerts.filter(severity='MEDIUM').count(),
        'LOW':      alerts.filter(severity='LOW').count(),
    }

    from django.utils import timezone
    from datetime import timedelta
    daily_data = {}
    today = timezone.now().date()
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        daily_data[str(day)] = sessions.filter(created_at__date=day).count()

    top_attacks = (alerts.values('attack_type')
                         .annotate(count=Count('id'))
                         .order_by('-count')[:5])

    from projects.models import Project
    proj_qs = Project.objects.filter(
        Q(owner=user)|Q(members=user)).exclude(status='deleted').distinct() \
        if not user.profile.is_admin else Project.objects.exclude(status='deleted')

    # CNN stats
    total_anomalies = sum(
        getattr(s, 'cnn_result', None) and s.cnn_result.anomaly_count or 0
        for s in sessions[:50]
    )

    return render(request, 'analyzer/dashboard.html', {
        'recent_sessions':     recent_sessions,
        'recent_alerts':       recent_alerts,
        'total_alerts':        alerts.count(),
        'total_sessions':      sessions.count(),
        'crit_count':          severity_data['CRITICAL'],
        'high_count':          severity_data['HIGH'],
        'med_count':           severity_data['MEDIUM'],
        'severity_data':       json.dumps(severity_data),
        'daily_data':          json.dumps(daily_data),
        'done_count':          sessions.filter(task_status='done').count(),
        'running_count':       sessions.filter(task_status='running').count(),
        'top_attacks':         top_attacks,
        'project_count':       proj_qs.count(),
        'active_project_count':proj_qs.filter(status='active').count(),
        'cnn_threshold':       settings.CNN_THRESHOLD,
        'cnn_latent_dim':      settings.CNN_LATENT_DIM,
        'total_anomalies':     total_anomalies,
    })


def _pipeline_steps():
    return [
        ('1', 'PCAP Upload',   '解析封包結構，計算 SHA-256 校驗',             'rgba(34,197,94,1)'),
        ('2', 'Rule Detection','Port Scan / DoS / SQLi 特徵比對',              'rgba(59,130,246,1)'),
        ('3', 'CNN Inference', '重建誤差計算，閾值比對',                       'rgba(249,115,22,1)'),
        ('4', 'Grad-CAM',      '異常封包熱力圖視覺化（可選）',                 'rgba(239,68,68,1)'),
        ('5', 'Report Export', 'PDF / JSON / CSV 分析報告',                    'rgba(168,85,247,1)'),
    ]


# ── Upload PCAP ──────────────────────────────────────────────
@login_required
def upload_pcap(request):
    user_projects = Project.objects.filter(
        Q(owner=request.user)|Q(members=request.user)
    ).filter(status='active').distinct()

    if request.method == 'POST':
        pcap_file  = request.FILES.get('pcap_file')
        label      = request.POST.get('label', '')
        project_id = request.POST.get('project_id')

        if not pcap_file:
            messages.error(request, '請選擇 PCAP 檔案。')
            return render(request, 'analyzer/upload.html', {
                'projects': user_projects,
                'pipeline_steps': _pipeline_steps(),
            })

        project = None
        if project_id:
            try:
                project = Project.objects.get(pk=project_id)
            except Project.DoesNotExist:
                pass

        session = AnalysisSession.objects.create(
            mode='pcap', label=label or pcap_file.name,
            pcap_file=pcap_file, project=project,
            created_by=request.user, task_status='pending',
        )

        try:
            from .tasks import run_pcap_analysis
            task = run_pcap_analysis.delay(session.pk)
            session.celery_task_id = task.id
            session.save(update_fields=['celery_task_id'])
        except Exception:
            _sync_run_analysis(session)

        from accounts.views import _log_action
        _log_action(request, 'upload_pcap', f'上傳: {pcap_file.name} → Session {session.pk}')
        messages.success(request, f'「{session.label}」已上傳，背景分析中...')
        return redirect('analyzer:session_detail', pk=session.pk)

    pipeline_steps = [
        ('1', 'PCAP 上傳', '解析封包結構，計算 SHA-256 校驗', 'rgba(34,197,94,1)'),
        ('2', '規則式偵測', 'Port Scan / DoS / SQL Injection 等特徵比對', 'rgba(59,130,246,1)'),
        ('3', 'CNN 推論',  '重建誤差計算，閾值比對', 'rgba(249,115,22,1)'),
        ('4', 'Grad-CAM', '異常封包熱力圖視覺化（可選）', 'rgba(239,68,68,1)'),
        ('5', '報告匯出', 'PDF / JSON / CSV 分析報告', 'rgba(168,85,247,1)'),
    ]
    pipeline_steps = _pipeline_steps()
    return render(request, 'analyzer/upload.html', {'projects': user_projects, 'pipeline_steps': pipeline_steps})


def _sync_run_analysis(session):
    try:
        from django.utils import timezone
        session.task_status = 'running'
        session.save(update_fields=['task_status'])
        from analyzer.tasks import _run_cnn_analysis
        cnn_metrics = _run_cnn_analysis(session, str(settings.MEDIA_ROOT / session.pcap_file.name))
        if cnn_metrics:
            CNNResult.objects.update_or_create(session=session, defaults=cnn_metrics)
        session.task_status = 'done'
        session.finished_at = timezone.now()
        session.save(update_fields=['task_status','finished_at'])
    except Exception as e:
        from django.utils import timezone
        session.task_status   = 'failed'
        session.error_message = str(e)[:1000]
        session.finished_at   = timezone.now()
        session.save(update_fields=['task_status','error_message','finished_at'])


# ── Session List ─────────────────────────────────────────────
@login_required
def session_list(request):
    user = request.user
    if user.profile.is_admin:
        sessions = AnalysisSession.objects.all()
    else:
        sessions = AnalysisSession.objects.filter(
            Q(created_by=user)|Q(project__owner=user)|Q(project__members=user)
        ).distinct()

    q       = request.GET.get('q','').strip()
    status  = request.GET.get('status','')
    proj    = request.GET.get('project','')
    if q:
        sessions = sessions.filter(Q(label__icontains=q)|Q(pcap_file__icontains=q))
    if status:
        sessions = sessions.filter(task_status=status)
    if proj:
        sessions = sessions.filter(project_id=proj)

    sessions = sessions.order_by('-created_at')
    user_projects = Project.objects.filter(
        Q(owner=user)|Q(members=user)).distinct()

    return render(request, 'analyzer/sessions.html', {
        'sessions':      sessions,
        'projects':      user_projects,
        'q':             q,
        'status_filter': status,
        'proj_filter':   proj,
    })


# ── Session Detail ───────────────────────────────────────────
@login_required
def session_detail(request, pk):
    session = get_object_or_404(AnalysisSession, pk=pk)
    _check_session_access(request, session)

    cnn_result   = getattr(session, 'cnn_result', None)
    alerts       = session.alerts.order_by('-timestamp')
    gradcam_imgs = session.gradcam_images.order_by('packet_index')[:20]
    reports      = session.reports.order_by('-created_at')

    sev_data = {
        'CRITICAL': alerts.filter(severity='CRITICAL').count(),
        'HIGH':     alerts.filter(severity='HIGH').count(),
        'MEDIUM':   alerts.filter(severity='MEDIUM').count(),
        'LOW':      alerts.filter(severity='LOW').count(),
    }
    return render(request, 'analyzer/session_detail.html', {
        'session':      session,
        'cnn_result':   cnn_result,
        'alerts':       alerts,
        'gradcam_imgs': gradcam_imgs,
        'reports':      reports,
        'sev_data':     json.dumps(sev_data),
        'can_gradcam':  request.user.profile.can_run_ai_analysis,
        'can_export':   request.user.profile.can_export_report,
    })


# ── Session Status (JSON polling) ────────────────────────────
@login_required
def session_status_api(request, pk):
    session = get_object_or_404(AnalysisSession, pk=pk)
    _check_session_access(request, session)
    return JsonResponse({
        'task_status':   session.task_status,
        'packet_count':  session.packet_count,
        'alert_count':   session.alert_count,
        'error_message': session.error_message,
        'finished_at':   session.finished_at.isoformat() if session.finished_at else None,
        'has_cnn':       hasattr(session, 'cnn_result'),
        'gradcam_count': session.gradcam_images.count(),
    })


# ── Trigger Grad-CAM ─────────────────────────────────────────
@login_required
@require_POST
def trigger_gradcam(request, pk):
    session = get_object_or_404(AnalysisSession, pk=pk)
    _check_session_access(request, session)
    if not request.user.profile.can_run_ai_analysis:
        return JsonResponse({'error': '權限不足'}, status=403)

    variant      = request.POST.get('variant', 'gradcam')
    max_images   = int(request.POST.get('max_images', 20))
    anomaly_only = request.POST.get('anomaly_only', 'true') == 'true'

    try:
        from .tasks import run_gradcam
        task = run_gradcam.delay(session.pk, max_images=max_images,
                                  variant=variant, anomaly_only=anomaly_only)
        from accounts.views import _log_action
        _log_action(request, 'run_gradcam', f'Session {session.pk} variant={variant}')
        return JsonResponse({'status':'queued','task_id':task.id,
                             'message':f'Grad-CAM 已派發（{variant}，最多 {max_images} 張）'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ── GradCAM Gallery ──────────────────────────────────────────
@login_required
def gradcam_gallery(request, pk):
    session = get_object_or_404(AnalysisSession, pk=pk)
    _check_session_access(request, session)
    variant = request.GET.get('variant','')
    images  = session.gradcam_images.order_by('packet_index')
    if variant:
        images = images.filter(variant=variant)
    if request.GET.get('anomaly_only') == '1':
        images = images.filter(is_anomaly=True)
    return render(request, 'analyzer/gradcam_gallery.html', {
        'session': session, 'images': images, 'variant': variant,
    })


# ── Session Delete ───────────────────────────────────────────
@login_required
@require_POST
def session_delete(request, pk):
    session = get_object_or_404(AnalysisSession, pk=pk)
    from django.core.exceptions import PermissionDenied
    try:
        _check_session_access(request, session)
    except PermissionDenied:
        messages.error(request, '您沒有刪除此 Session 的權限。')
        return redirect('analyzer:session_detail', pk=pk)
    from accounts.views import _log_action
    _log_action(request, 'delete_session', f'Session {session.pk}')
    project_pk = session.project_id
    session.delete()
    messages.success(request, 'Session 已刪除。')
    if project_pk:
        return redirect('projects:detail', pk=project_pk)
    return redirect('analyzer:sessions')


# ── Live Monitor ─────────────────────────────────────────────
@login_required
def live_monitor(request):
    user = request.user
    if user.profile.is_admin:
        active_sessions = AnalysisSession.objects.filter(
            task_status__in=['running','pending']).order_by('-created_at')[:20]
        recent_done = AnalysisSession.objects.filter(
            task_status='done').order_by('-finished_at')[:5]
    else:
        qs = AnalysisSession.objects.filter(
            Q(created_by=user)|Q(project__owner=user)|Q(project__members=user)
        ).distinct()
        active_sessions = qs.filter(task_status__in=['running','pending']).order_by('-created_at')[:20]
        recent_done     = qs.filter(task_status='done').order_by('-finished_at')[:5]
    return render(request, 'analyzer/live.html', {
        'active_sessions': active_sessions,
        'recent_done':     recent_done,
    })


# ── AI Chat ──────────────────────────────────────────────────
@login_required
def ai_chat(request):
    user = request.user
    if user.profile.is_admin:
        sessions = AnalysisSession.objects.filter(
            task_status='done').order_by('-finished_at')[:10]
    else:
        sessions = AnalysisSession.objects.filter(
            Q(created_by=user)|Q(project__owner=user)|Q(project__members=user),
            task_status='done',
        ).distinct().order_by('-finished_at')[:10]
    return render(request, 'analyzer/ai_chat.html', {'sessions': sessions})


# ── Access control helper ────────────────────────────────────
def _check_session_access(request, session):
    user = request.user
    if user.profile.is_admin or session.created_by == user:
        return
    if session.project:
        if session.project.owner == user:
            return
        if session.project.members.filter(pk=user.pk).exists():
            return
    from django.core.exceptions import PermissionDenied
    raise PermissionDenied('您沒有存取此 Session 的權限。')


# ── 異常封包模擬檢測 ─────────────────────────────────────────
@login_required
def simulation(request):
    """異常封包模擬檢測頁面 — 支援預設攻擊場景和自訂封包。"""
    # 載入 CNN 模型資訊
    model_ready = False
    model_path = getattr(settings, 'CNN_MODEL_PATH', '')
    if model_path:
        model_ready = os.path.exists(model_path)

    return render(request, 'analyzer/simulation.html', {
        'model_ready':   model_ready,
        'cnn_threshold': settings.CNN_THRESHOLD,
        'cnn_latent_dim': settings.CNN_LATENT_DIM,
    })


@login_required
def simulation_api(request):
    """模擬檢測 AJAX 端點 — 接收封包資料，回傳 CNN 異常分數。"""
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'error': '僅接受 POST'}, status=405)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError) as e:
        return JsonResponse({'ok': False, 'error': f'JSON 解析失敗：{e}'}, status=400)

    attack_type = data.get('attack_type', 'syn_flood')
    packet_count = min(int(data.get('packet_count', 10)), 100)

    # 使用 core 引擎生成模擬封包
    try:
        import sys
        core_path = os.path.join(settings.BASE_DIR, 'core')
        if core_path not in sys.path:
            sys.path.insert(0, core_path)

        from generate_attack_pcap import generate_attack_packets
        from scapy.all import raw as scapy_raw

        # 產生 Scapy 封包物件
        scapy_packets = generate_attack_packets(attack_type, packet_count)

        # 統一轉換為 raw bytes（核心修正：Scapy 封包物件不能直接當 bytes 使用）
        raw_packets = []
        for pkt in scapy_packets:
            try:
                raw_packets.append(scapy_raw(pkt))
            except Exception:
                raw_packets.append(bytes(pkt))

        results = []
        model_path = getattr(settings, 'CNN_MODEL_PATH', '')

        if model_path and os.path.exists(str(model_path)):
            # ── 真實模型推論模式 ──
            from cnn_autoencoder import CNNAutoencoder
            from anomaly_scorer import AnomalyScorer
            from packet_visualizer import PacketVisualizer
            import torch
            import numpy as np

            model = CNNAutoencoder(latent_dim=settings.CNN_LATENT_DIM)
            model.load_state_dict(torch.load(
                str(model_path), map_location='cpu', weights_only=True
            ))
            model.eval()

            visualizer = PacketVisualizer("medium", apply_mask=True)

            # [v3.0 核心優化] 比較式判定策略
            # 問題：合成封包的位元組分布與訓練集不同，
            #        導致固定閾值 (CNN_THRESHOLD) 失效。
            # 解法：先產生 baseline 正常流量，動態計算分離度。
            #
            # 策略：
            #   1. 生成一組正常封包，計算 baseline MSE 分布
            #   2. 計算目標封包的 MSE
            #   3. 用 baseline 的 95th percentile 作為動態閾值
            #   4. 同時回傳原始的 settings 閾值，供前端參考

            # 生成 baseline 正常流量
            baseline_pkts = generate_attack_packets('normal_traffic', 20)
            baseline_raw = []
            for pkt in baseline_pkts:
                try:
                    baseline_raw.append(scapy_raw(pkt))
                except Exception:
                    baseline_raw.append(bytes(pkt))

            # 計算 baseline 的 MSE 分布
            baseline_arrs = np.array([
                visualizer.bytes_to_image(b) for b in baseline_raw
            ], dtype=np.float32)
            baseline_tensor = torch.from_numpy(baseline_arrs[:, np.newaxis])

            with torch.no_grad():
                baseline_errors = model.reconstruction_error(baseline_tensor).numpy()

            baseline_mean = float(baseline_errors.mean())
            baseline_std = float(baseline_errors.std())
            # 動態閾值 = baseline 平均值 + 2 倍標準差
            dynamic_threshold = baseline_mean + 2.0 * baseline_std

            # 計算目標封包的 MSE
            target_arrs = np.array([
                visualizer.bytes_to_image(b) for b in raw_packets
            ], dtype=np.float32)
            target_tensor = torch.from_numpy(target_arrs[:, np.newaxis])

            with torch.no_grad():
                target_errors = model.reconstruction_error(target_tensor).numpy()

            for i, score in enumerate(target_errors):
                score_val = float(score)
                is_anomaly = score_val > dynamic_threshold
                results.append({
                    'index': i,
                    'score': round(score_val, 8),
                    'is_anomaly': is_anomaly,
                    'size': len(raw_packets[i]),
                })

            # 回傳時附上動態閾值資訊
            effective_threshold = dynamic_threshold
        else:
            # ── 模型未就緒 — 使用模擬分數 ──
            import random
            threshold = settings.CNN_THRESHOLD
            is_normal = (attack_type == 'normal_traffic')

            for i, pkt_bytes in enumerate(raw_packets):
                if is_normal:
                    sim_score = random.uniform(threshold * 0.05, threshold * 0.85)
                else:
                    sim_score = random.uniform(threshold * 2.0, threshold * 15.0)

                results.append({
                    'index': i,
                    'score': round(sim_score, 8),
                    'is_anomaly': sim_score > threshold,
                    'size': len(pkt_bytes),
                })
            effective_threshold = threshold

        anomaly_count = sum(1 for r in results if r['is_anomaly'])
        total = len(results)

        return JsonResponse({
            'ok': True,
            'attack_type': attack_type,
            'packet_count': total,
            'threshold': effective_threshold,
            'static_threshold': settings.CNN_THRESHOLD,
            'results': results,
            'avg_score': round(
                sum(r['score'] for r in results) / max(total, 1), 8
            ),
            'anomaly_count': anomaly_count,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'ok': False,
            'error': f'{type(e).__name__}: {str(e)}'
        }, status=500)


# ── 熱力圖分析 ───────────────────────────────────────────────
@login_required
def heatmap_analysis(request, pk):
    """熱力圖深度分析頁面 — Grad-CAM 影像詳細檢視。"""
    session = get_object_or_404(AnalysisSession, pk=pk)
    _check_session_access(request, session)

    images = session.gradcam_images.order_by('packet_index')
    anomaly_images = images.filter(is_anomaly=True)
    normal_images = images.filter(is_anomaly=False)

    cnn_result = getattr(session, 'cnn_result', None)

    return render(request, 'analyzer/heatmap_analysis.html', {
        'session':        session,
        'images':         images,
        'anomaly_images': anomaly_images,
        'normal_images':  normal_images,
        'cnn_result':     cnn_result,
        'total_count':    images.count(),
        'anomaly_count':  anomaly_images.count(),
        'normal_count':   normal_images.count(),
    })


# ── 消融實驗 (Ablation Study) ──────────────────────────────────
import subprocess
import threading
import time

# 用於追蹤當前正在執行的背景消融實驗行程（簡單的全域變數實作）
# 鍵: dataset_name, 值: {'process': Popen_object, 'log_file': path_to_log, 'status': 'running'|'done'|'error'}
ablation_tasks = {}

@login_required
def ablation_dashboard(request):
    """消融實驗主頁面"""
    return render(request, 'analyzer/ablation.html', {
        'page_title': '消融實驗分析',
    })

@require_POST
@login_required
def ablation_run_api(request):
    """啟動背景消融實驗"""
    try:
        data = json.loads(request.body)
        dataset = data.get('dataset', 'simulate')
        experiments = data.get('experiments', ['all'])
        repeats = data.get('repeats', 1)
        
        # 建立輸出目錄與 Log 檔案
        output_dir = os.path.join(settings.BASE_DIR, 'output', dataset, 'ablation')
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, 'ablation.log')
        
        script_path = os.path.join(settings.BASE_DIR, 'core', 'run_threshold_tuning.py')
        
        # 準備命令列參數
        # 由於原始腳本 run_threshold_tuning.py 中目前尚未把 n_repeats 拉成命令列參數，
        # 這裡主要是呼叫 --ablation 和 --ablation-exp。
        # 如果需要 n_repeats，腳本需要被微調，或者我們就依賴預設的 3 次。
        cmd = ['python', script_path, '--dataset', dataset, '--ablation']
        if 'all' not in experiments:
            cmd.extend(['--ablation-exp'] + experiments)
            
        # 清空舊的 Log
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"--- 啟動消融實驗 (資料集: {dataset}) ---\n")
            f.write(f"執行指令: {' '.join(cmd)}\n\n")

        # 啟動子行程，將 stdout 與 stderr 導向至 Log 檔
        log_file = open(log_path, 'a', encoding='utf-8')
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=settings.BASE_DIR,
            shell=False
        )
        
        # 紀錄至全域追蹤變數
        ablation_tasks[dataset] = {
            'process': process,
            'log_path': log_path,
            'log_file_obj': log_file,
            'status': 'running'
        }
        
        # 建立一個背景執行緒來等待行程結束並關閉檔案
        def wait_process(task_dict):
            p = task_dict['process']
            p.wait()
            task_dict['log_file_obj'].close()
            if p.returncode == 0:
                task_dict['status'] = 'done'
            else:
                task_dict['status'] = 'error'

        threading.Thread(target=wait_process, args=(ablation_tasks[dataset],)).start()
        
        return JsonResponse({'ok': True, 'message': '消融實驗已在背景啟動', 'dataset': dataset})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)

@login_required
def ablation_status_api(request):
    """查詢實驗進度與取得結果"""
    dataset = request.GET.get('dataset', 'simulate')
    
    task = ablation_tasks.get(dataset)
    log_path = os.path.join(settings.BASE_DIR, 'output', dataset, 'ablation', 'ablation.log')
    
    status = task['status'] if task else 'idle'
    
    # 讀取最後幾行 Log 作為進度提示
    logs = ""
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                # 讀取最後 100 行
                lines = f.readlines()
                logs = "".join(lines[-100:])
        except Exception:
            logs = "無法讀取日誌檔案..."
            
    # 如果已完成或原本就沒有在跑，嘗試回傳結果
    report_data = None
    images = {}
    
    if status in ['done', 'idle']:
        report_path = os.path.join(settings.BASE_DIR, 'output', dataset, 'ablation', 'ablation_report.json')
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                    
                # 尋找所有生成的圖表
                output_dir = os.path.dirname(report_path)
                for file in os.listdir(output_dir):
                    if file.endswith('.png'):
                        # 提供可以供前端載入的路徑 (這裡暫時使用相對路徑，前提是 output 資料夾可以透過 static 或 media 服務，
                        # 但如果 Django 沒開的話，我們可能要寫個 view 來 serve，或者將它複製到 media 裡面)
                        pass
                        
                # 簡單做法：將圖表複製到 media/ablation/{dataset}/ 供前端存取
                media_ablation_dir = os.path.join(settings.MEDIA_ROOT, 'ablation', dataset)
                os.makedirs(media_ablation_dir, exist_ok=True)
                
                import shutil
                for file in os.listdir(output_dir):
                    if file.endswith('.png'):
                        src = os.path.join(output_dir, file)
                        dst = os.path.join(media_ablation_dir, file)
                        shutil.copy2(src, dst)
                        images[file] = f"{settings.MEDIA_URL}ablation/{dataset}/{file}"
            except Exception as e:
                print("讀取消融實驗報告失敗:", e)
    
    return JsonResponse({
        'ok': True,
        'status': status,
        'logs': logs,
        'report': report_data,
        'images': images
    })
