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
from django.urls import reverse
from django.views.decorators.http import require_POST

from .models import AnalysisSession, Alert, CNNResult, GradCAMImage, AnalysisReport
from .apps import warmup_event
from projects.models import Project
import time


# ── [根因 2 修正] 依攻擊類型回傳真正相關的規則引擎門檻 ──────────────
def _relevant_threshold_for(attack_type: str) -> int:
    """依攻擊類型回傳對應的正式系統門檻（而非永遠回傳 SYN 的 100）。"""
    import sys as _sys
    core_path = os.path.join(settings.BASE_DIR, 'core')
    if core_path not in _sys.path:
        _sys.path.insert(0, core_path)
    from config import (
        ALERT_THRESHOLD_SYN, ALERT_THRESHOLD_PORTS,
        ALERT_THRESHOLD_ICMP, ALERT_THRESHOLD_UDP,
    )
    return {
        'syn_flood':  ALERT_THRESHOLD_SYN,
        'port_scan':  ALERT_THRESHOLD_PORTS,
        'icmp_flood': ALERT_THRESHOLD_ICMP,
        'udp_flood':  ALERT_THRESHOLD_UDP,
    }.get(attack_type, ALERT_THRESHOLD_SYN)


def _array_to_data_url(arr_2d, upscale=4):
    """[黑箱修正] 把 32x32 灰階封包影像轉成 base64 PNG data URL。"""
    import io
    import base64
    import numpy as np
    from PIL import Image

    clipped = np.clip(arr_2d, 0.0, 1.0)
    img = Image.fromarray((clipped * 255).astype('uint8'), mode='L')
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def _select_display_indices(behavior_result, total, cap=400):
    """[效能] 保留觸發規則的封包 + 等距抽樣其餘封包，最多 cap 筆。"""
    triggered = [i for i, alerts in enumerate(behavior_result['packet_alerts']) if alerts]
    if total <= cap:
        return list(range(total))
    triggered_set = set(triggered)
    remaining_budget = max(0, cap - len(triggered_set))
    others = [i for i in range(total) if i not in triggered_set]
    sampled = []
    if remaining_budget > 0 and others:
        step = max(1, len(others) // remaining_budget)
        sampled = others[::step][:remaining_budget]
    idx = sorted(triggered_set.union(sampled))
    return idx[:cap]


# [黑箱修正] CNN 可信度說明
_CNN_RELIABILITY_NOTE_CSV_LEGACY = (
    '目前選用的 CNN/VAE 模型是以 CSV 流量特徵（如 CICIDS2017/NSL-KDD 的'
    '統計欄位）訓練而成；本頁送入模型的則是「原始封包位元組」，兩者影像'
    '的統計分布並不相同（train/serve skew）。因此 CNN 分數在這裡僅供參考，'
    '最終判定請優先參考下方「規則式流量型樣偵測」的結果。'
)
_CNN_RELIABILITY_NOTE_PACKET_NATIVE = (
    '此模型已使用與本頁一致的「封包位元組影像」表示法訓練，CNN/VAE 分數'
    '具備參考意義。惟系統仍以「規則式流量型樣偵測」作為最終判定的唯一依據。'
)
_CNN_RELIABILITY_NOTE = _CNN_RELIABILITY_NOTE_CSV_LEGACY




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
# ── 允許的攻擊場景白名單 ─────────────────────────────────────────
SIMULATION_ATTACK_TYPES = {
    'syn_flood', 'port_scan', 'icmp_flood', 'udp_flood',
    'arp_spoof', 'dns_amplification', 'normal_traffic',
}

# ── 模擬 PCAP 檔名合法字元驗證 ──────────────────────────────────
import re as _re
SIMULATION_FILENAME_RE = _re.compile(r'^sim_[a-z0-9_]+_\d{14}_[a-f0-9]{8}\.pcap$')


@login_required
def simulation(request):
    """異常封包模擬檢測頁面。

    [根因修正] 舊版沒有把 models 放進 context，導致模型下拉選單永遠空白。
    修正：實際掃描 settings.ANOMALY_MODELS，組成 models 清單傳給樣板。
    """
    models = []
    for key, cfg in settings.ANOMALY_MODELS.items():
        path = str(cfg.get('path', ''))
        models.append({
            'key':   key,
            'label': cfg.get('label', key),
            'ready': bool(path and os.path.exists(path)),
        })
    model_ready = any(m['ready'] for m in models)
    return render(request, 'analyzer/simulation.html', {
        'model_ready':    model_ready,
        'models':         models,
        'cnn_threshold':  settings.CNN_THRESHOLD,
        'cnn_latent_dim': settings.CNN_LATENT_DIM,
    })


@login_required
def simulation_api(request):
    """模擬檢測 AJAX 端點（v2）。

    與前一版相比的四項核心修正：

    [根因 1／已存在，保留] 最終「攻擊／正常」判定只由規則式流量
        型樣偵測（AnomalyDetector）決定，CNN／VAE 分數只作為每筆
        封包旁的參考數值。理由不變：目前多數已部署模型是以 CSV
        統計特徵訓練，套用在這裡送入的「原始封包位元組」影像上，
        重建誤差不具判斷力（train/serve skew）。

    [根因 2／本次新修正] 封包產生邏輯改用
        core/simulate_anomaly_traffic.py 的 GENERATORS（見同批修正
        的 simulate_anomaly_traffic_PATCH.py），其 bulk_scale() 會
        依「正式系統門檻 + 隨機安全margin」反推需要產生的封包數，
        保證只要強度落在允許範圍內，攻擊封包數量『一定』超過規則
        引擎的正式門檻。這直接修正了舊版「選 SYN Flood / UDP Flood
        但因為滑桿預設值/上限本來就低於正式門檻，導致模擬結果一律
        顯示正常」的問題 —— 該問題出在「模擬封包產生器給的資料量
        不足」，不是規則引擎判斷錯誤。

    [黑箱修正] 新增：
        - sample_visual：模型實際看到的封包影像（base64 PNG），
          與正常流量基準線平均影像並排，可直接比對。
        - timing_ms：逐階段耗時（封包產生 / 規則引擎 / 模型載入 /
          CNN 推論），並附上 model_was_cached 與
          server_warmup_complete，讓「第一次模擬比較慢」有明確、
          可驗證的解釋。
        - comparison_summary：規則引擎判定 vs CNN 參考判定的量化
          比對摘要。
        - model_representation：本次使用的模型是用「封包位元組」
          還是「CSV 特徵」訓練的，讓 CNN 分數的可信度說明能動態
          反映實際狀況，而不是永遠顯示同一句警語。

    [向下相容] 請求仍接受舊版的 packet_count 欄位（會換算成大致對應
        的 intensity），也接受新的 intensity 欄位（0.3~3.0，1.0為
        預設強度）；既有呼叫端（例如 analyzer/tests.py 的既有測試）
        不需要修改即可繼續運作。
    """
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'error': '僅接受 POST'}, status=405)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError) as e:
        return JsonResponse({'ok': False, 'error': f'JSON 解析失敗：{e}'}, status=400)

    t_request_start = time.perf_counter()
    timing = {}

    attack_type = data.get('attack_type', 'syn_flood')
    model_key = data.get('model_key') or 'unsupervised_vae'

    if model_key not in settings.ANOMALY_MODELS:
        return JsonResponse({'ok': False, 'error': '不支援的模型選擇'}, status=400)
    if attack_type not in SIMULATION_ATTACK_TYPES:
        return JsonResponse({'ok': False, 'error': '不支援的攻擊場景'}, status=400)

    # ── 強度（取代舊版直接輸入封包數量的欄位）──────────────────
    # 換算規則：若前端有帶新版 intensity 就直接用；否則若帶舊版
    # packet_count，除以舊版滑桿預設值 60 粗略換算成強度倍率，
    # 確保既有呼叫端不會壞掉；都沒帶則預設 1.0。
    try:
        if 'intensity' in data:
            intensity = float(data.get('intensity'))
        elif 'packet_count' in data:
            intensity = float(data.get('packet_count')) / 60.0
        else:
            intensity = 1.0
    except (TypeError, ValueError):
        intensity = 1.0
    intensity = min(max(intensity, 0.3), 3.0)

    try:
        import sys
        core_path = os.path.join(settings.BASE_DIR, 'core')
        if core_path not in sys.path:
            sys.path.insert(0, core_path)

        from simulate_anomaly_traffic import GENERATORS
        from scapy.all import raw as scapy_raw

        # ── Step 1：產生模擬封包 ──────────────────────────────
        t0 = time.perf_counter()
        _label, gen_fn = GENERATORS[attack_type]
        scapy_packets, gen_meta = gen_fn(scale=intensity)

        # 安全上限：intensity 拉到 3.0 時，多來源 SYN Flood 等場景
        # 可能產生上千筆封包；規則引擎逐筆處理仍然很快（純 dict/
        # 計數器操作），這裡的上限主要是避免極端情況下佔用過多記憶體。
        MAX_TOTAL_PACKETS = 2000
        packet_count_truncated = False
        if len(scapy_packets) > MAX_TOTAL_PACKETS:
            scapy_packets = scapy_packets[:MAX_TOTAL_PACKETS]
            packet_count_truncated = True

        raw_packets = []
        for pkt in scapy_packets:
            try:
                raw_packets.append(scapy_raw(pkt))
            except Exception:
                raw_packets.append(bytes(pkt))
        timing['packet_generation_ms'] = round((time.perf_counter() - t0) * 1000, 1)

        # ── Step 2：規則式偵測：永遠是最終判定的唯一依據 ────────
        t1 = time.perf_counter()
        behavior_result = _detect_simulation_behavior(
            scapy_packets=scapy_packets,
            attack_type=attack_type,
            packet_count=len(scapy_packets),
        )
        behavior_anomaly = behavior_result['is_attack']
        timing['rule_engine_ms'] = round((time.perf_counter() - t1) * 1000, 1)

        total_all = len(scapy_packets)
        display_indices = _select_display_indices(behavior_result, total_all, cap=400)

        results = []
        baseline_info = None
        effective_threshold = None
        model_threshold = None
        dynamic_threshold = None
        model_representation = None
        selected_model = settings.ANOMALY_MODELS[model_key]
        model_path = selected_model['path']
        model_used = bool(model_path and os.path.exists(str(model_path)))
        sample_visual = None
        anomaly_count = 0
        avg_score = 0.0

        if model_used:
            from packet_visualizer import PacketVisualizer
            import torch
            import numpy as np
            from model_registry import load_anomaly_model, compute_anomaly_scores, is_model_cached

            device = torch.device('cpu')
            was_cached = is_model_cached(str(model_path), device)

            t2 = time.perf_counter()
            bundle = load_anomaly_model(
                str(model_path), device=device,
                default_latent_dim=settings.CNN_LATENT_DIM,
            )
            timing['model_load_ms'] = round((time.perf_counter() - t2) * 1000, 1)
            timing['model_was_cached'] = was_cached

            model = bundle.model
            model_representation = bundle.input_representation
            visualizer = PacketVisualizer("medium", apply_mask=True)

            t3 = time.perf_counter()

            # ── 基準線：即時產生一批正常流量，計算動態閾值 ──────
            BASELINE_SAMPLE_COUNT = 60
            _, normal_gen = GENERATORS['normal_traffic']
            baseline_pkts, _bmeta = normal_gen(scale=BASELINE_SAMPLE_COUNT / 30.0)
            baseline_raw = []
            for pkt in baseline_pkts:
                try:
                    baseline_raw.append(scapy_raw(pkt))
                except Exception:
                    baseline_raw.append(bytes(pkt))

            baseline_arrs = np.array([
                visualizer.bytes_to_image(b) for b in baseline_raw
            ], dtype=np.float32)
            baseline_tensor = torch.from_numpy(baseline_arrs[:, np.newaxis])

            with torch.no_grad():
                baseline_errors = model.reconstruction_error(baseline_tensor).numpy()

            baseline_mean = float(baseline_errors.mean())
            baseline_std = float(baseline_errors.std())
            dynamic_threshold = float(np.percentile(baseline_errors, 95))
            model_threshold = float(bundle.threshold)

            baseline_info = {
                'sample_count': len(baseline_errors),
                'mean': round(baseline_mean, 8),
                'std': round(baseline_std, 8),
                'dynamic_threshold': round(dynamic_threshold, 8),
                'static_threshold': round(model_threshold, 8),
                'scores': [round(float(s), 8) for s in baseline_errors.tolist()],
            }

            target_arrs = np.array([
                visualizer.bytes_to_image(b) for b in raw_packets
            ], dtype=np.float32)
            scored = compute_anomaly_scores(bundle, target_arrs)

            if bundle.is_hybrid:
                cnn_anomaly = scored['is_anomaly'].copy()
                effective_threshold = model_threshold
            else:
                effective_threshold = float(np.percentile(baseline_errors, 99))
                cnn_anomaly = (scored['score'] > effective_threshold)

            # [根因 1，維持] 最終判定只看規則引擎，不受 CNN 分數影響。
            final_anomaly = np.full(total_all, behavior_anomaly, dtype=bool)
            scored['is_anomaly'] = final_anomaly
            timing['cnn_inference_ms'] = round((time.perf_counter() - t3) * 1000, 1)

            anomaly_count = int(final_anomaly.sum())
            avg_score = float(scored['score'].mean()) if total_all else 0.0

            # ── [黑箱修正] 把模型實際看到的影像畫出來 ─────────────
            worst_idx = int(np.argmax(scored['score'])) if total_all else 0
            baseline_avg_img = baseline_arrs.mean(axis=0)
            sample_visual = {
                'most_anomalous_packet_index': worst_idx,
                'most_anomalous_packet_score': round(float(scored['score'][worst_idx]), 8) if total_all else None,
                'most_anomalous_image': _array_to_data_url(target_arrs[worst_idx]) if total_all else None,
                'baseline_average_image': _array_to_data_url(baseline_avg_img),
                'note': (
                    '左圖為本次模擬中 CNN／VAE 重建誤差最高的一張封包影像，'
                    '右圖為基準線（正常流量）平均影像；模型是否具備判斷力，'
                    '取決於兩張影像的統計結構差異是否確實對應真實的封包特徵，'
                    '而不是只看分數本身。'
                ),
            }

            for i in display_indices:
                score = scored['score'][i]
                pkt_alerts = behavior_result['packet_alerts'][i]
                entry = {
                    'index': i,
                    'score': round(float(score), 8),
                    'cnn_anomaly': bool(cnn_anomaly[i]),
                    'behavior_anomaly': bool(behavior_anomaly),
                    'is_anomaly': bool(final_anomaly[i]),
                    'size': len(raw_packets[i]),
                    'summary': scapy_packets[i].summary(),
                    'header': behavior_result['packet_records'][i],
                    'rule_triggered_here': bool(pkt_alerts),
                    'rule_detail': [
                        {'attack_type': a.get('attack_type'),
                         'severity': a.get('severity'),
                         'detail': a.get('detail')}
                        for a in pkt_alerts
                    ],
                }

                if pkt_alerts:
                    entry['detection_reason'] = pkt_alerts[0].get('attack_type', attack_type)
                elif behavior_anomaly:
                    entry['detection_reason'] = behavior_result['attack_type'] or attack_type
                elif cnn_anomaly[i]:
                    entry['detection_reason'] = 'CNN/VAE 分數偏高（僅供參考，未計入最終判定）'
                else:
                    entry['detection_reason'] = 'normal'

                if bundle.is_hybrid:
                    entry['status'] = scored['status'][i]
                    entry['known_confidence'] = round(float(scored['known_confidence'][i]), 4)
                results.append(entry)
        else:
            timing['model_load_ms'] = 0.0
            timing['cnn_inference_ms'] = 0.0
            timing['model_was_cached'] = None
            import random as _random
            threshold = settings.CNN_THRESHOLD
            is_normal = (attack_type == 'normal_traffic')

            all_scores = []
            for i in range(total_all):
                sim_score = (_random.uniform(threshold * 0.05, threshold * 0.85) if is_normal
                             else _random.uniform(threshold * 2.0, threshold * 15.0))
                all_scores.append(sim_score)
            anomaly_count = total_all if behavior_anomaly else 0
            avg_score = sum(all_scores) / max(total_all, 1)

            for i in display_indices:
                pkt_alerts = behavior_result['packet_alerts'][i]
                cnn_anom = all_scores[i] > threshold
                results.append({
                    'index': i,
                    'score': round(all_scores[i], 8),
                    'cnn_anomaly': cnn_anom,
                    'behavior_anomaly': behavior_anomaly,
                    'is_anomaly': behavior_anomaly,
                    'size': len(raw_packets[i]),
                    'summary': scapy_packets[i].summary(),
                    'header': behavior_result['packet_records'][i],
                    'rule_triggered_here': bool(pkt_alerts),
                    'rule_detail': [
                        {'attack_type': a.get('attack_type'),
                         'severity': a.get('severity'),
                         'detail': a.get('detail')}
                        for a in pkt_alerts
                    ],
                })

        pcap_filename = _save_simulation_pcap(request.user, attack_type, scapy_packets)
        timing['total_ms'] = round((time.perf_counter() - t_request_start) * 1000, 1)

        # ── [黑箱修正] 規則引擎 vs CNN 參考判定的量化比對摘要 ─────
        cnn_flagged = sum(1 for r in results if r.get('cnn_anomaly'))
        comparison_summary = {
            'rule_engine_verdict': bool(behavior_anomaly),
            'rule_engine_alert_count': len(behavior_result['alerts']),
            'cnn_reference_flagged_count_in_sample': cnn_flagged,
            'cnn_reference_sample_size': len(results),
            'agree': bool(behavior_anomaly) == bool(cnn_flagged > 0),
            'authoritative_source': 'rule_engine',
        }

        reliability_note = _CNN_RELIABILITY_NOTE_CSV_LEGACY
        if model_representation == 'packet_bytes_visualizer':
            reliability_note = _CNN_RELIABILITY_NOTE_PACKET_NATIVE

        return JsonResponse({
            'ok': True,
            'attack_type': attack_type,
            'intensity': round(intensity, 2),
            'packet_count': total_all,
            'packet_count_displayed': len(results),
            'packet_count_truncated': packet_count_truncated,
            'generation_meta': gen_meta,
            'model_used': model_used,
            'model_representation': model_representation,
            'effective_threshold': round(effective_threshold, 8) if effective_threshold is not None else None,
            'threshold': round(model_threshold, 8) if model_threshold is not None else settings.CNN_THRESHOLD,
            'dynamic_threshold': round(dynamic_threshold, 8) if dynamic_threshold is not None else None,
            'static_threshold': settings.CNN_THRESHOLD,
            'cnn_reliability_note': reliability_note,
            'final_verdict_note': (
                '最終「攻擊／正常」判定僅由下方「規則式流量型樣偵測」決定；'
                'CNN／VAE 分數僅顯示於每筆封包旁作為參考，不影響此判定。'
            ),
            'comparison_summary': comparison_summary,
            'sample_visual': sample_visual,
            'timing_ms': timing,
            'server_warmup_complete': bool(warmup_event.is_set()),
            'behavior_detection': {
                'is_attack': behavior_result['is_attack'],
                'attack_type': behavior_result['attack_type'],
                'threshold': behavior_result['threshold'],
                'thresholds': behavior_result['thresholds'],
                'triggered_index': behavior_result['triggered_index'],
                'forced_by_demo_shortcut': behavior_result['forced_by_demo_shortcut'],
                'alert_count': len(behavior_result['alerts']),
                'alerts': behavior_result['alerts'],
            },
            'baseline': baseline_info,
            'results': results,
            'avg_score': round(avg_score, 8),
            'anomaly_count': anomaly_count,
            'rule_based_anomaly_detected': behavior_result['is_attack'],
            'pcap_download_url': request.build_absolute_uri(
                reverse('analyzer:simulation_pcap_download', args=[pcap_filename])),
            'pcap_filename': pcap_filename,
            'pcap_backup_saved': True,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'ok': False, 'error': f'{type(e).__name__}: {str(e)}'}, status=500)

def _save_simulation_pcap(user, attack_type, scapy_packets):
    """將模擬封包儲存為 PCAP 檔案，回傳檔名。"""
    import hashlib, datetime
    from scapy.all import wrpcap

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    uid = hashlib.md5(f'{user.pk}_{ts}'.encode()).hexdigest()[:8]
    filename = f'sim_{attack_type}_{ts}_{uid}.pcap'

    pcap_dir = os.path.join(settings.MEDIA_ROOT, 'simulation_pcap')
    os.makedirs(pcap_dir, exist_ok=True)

    pcap_path = os.path.join(pcap_dir, filename)
    wrpcap(pcap_path, scapy_packets)
    return filename


def _detect_simulation_behavior(scapy_packets, attack_type, packet_count):
    """規則式（流量型樣）偵測。

    回傳欄位：
      is_attack                規則引擎的真實判定（不再有任何強制/保底）
      attack_type               觸發的攻擊類型描述
      alerts                     觸發的告警列表
      packet_alerts               每個封包各自觸發的告警
      packet_records               每個封包解析後的關鍵標頭欄位
      triggered_index              規則第一次觸發的封包索引
      forced_by_demo_shortcut      恆為 False（保留欄位以維持前端相容）
      thresholds                    本次套用的真實系統門檻字典
      threshold                     依 attack_type 回傳對應的正式門檻
    """
    import sys as _sys
    core_path = os.path.join(settings.BASE_DIR, 'core')
    if core_path not in _sys.path:
        _sys.path.insert(0, core_path)

    from anomaly_detector import AnomalyDetector
    from parser import PacketParser
    from config import (
        ALERT_THRESHOLD_SYN, ALERT_THRESHOLD_PORTS,
        ALERT_THRESHOLD_ICMP, ALERT_THRESHOLD_UDP,
    )

    detector = AnomalyDetector(
        threshold_syn=ALERT_THRESHOLD_SYN,
        threshold_ports=ALERT_THRESHOLD_PORTS,
        threshold_icmp=ALERT_THRESHOLD_ICMP,
        threshold_udp=ALERT_THRESHOLD_UDP,
    )
    parser = PacketParser()
    packet_alerts = [[] for _ in scapy_packets]
    packet_records = []
    all_alerts = []
    triggered_index = None

    for i, pkt in enumerate(scapy_packets):
        try:
            record = parser.parse(pkt)
        except Exception:
            record = {}

        packet_records.append({
            'protocol':  record.get('protocol'),
            'src_ip':    record.get('src_ip'),
            'dst_ip':    record.get('dst_ip'),
            'src_port':  record.get('src_port'),
            'dst_port':  record.get('dst_port'),
            'flags':     record.get('flags'),
            'icmp_desc': record.get('icmp_desc'),
            'arp_op':    record.get('arp_op'),
            'ttl':       record.get('ttl'),
            'length':    record.get('length'),
        })

        try:
            alerts = detector.inspect(pkt, record)
            if alerts:
                packet_alerts[i].extend(alerts)
                all_alerts.extend(alerts)
                if triggered_index is None:
                    triggered_index = i
        except Exception:
            continue

    behavior_attack = len(all_alerts) > 0
    if attack_type == 'normal_traffic':
        behavior_attack = False

    behavior_type = None
    if all_alerts:
        behavior_type = all_alerts[0].get('attack_type', attack_type)

    return {
        'is_attack': behavior_attack,
        'attack_type': behavior_type,
        'alerts': all_alerts,
        'packet_alerts': packet_alerts,
        'packet_records': packet_records,
        'triggered_index': triggered_index,
        'forced_by_demo_shortcut': False,
        'thresholds': {
            'syn': ALERT_THRESHOLD_SYN, 'ports': ALERT_THRESHOLD_PORTS,
            'icmp': ALERT_THRESHOLD_ICMP, 'udp': ALERT_THRESHOLD_UDP,
        },
        # [修正] 依攻擊類型回傳相關門檻，不再永遠是 SYN 的 100
        'threshold': _relevant_threshold_for(attack_type),
    }


@login_required
def simulation_pcap_download(request, filename):
    """提供模擬封包 PCAP 下載。"""
    from django.http import FileResponse, Http404
    if not SIMULATION_FILENAME_RE.match(filename):
        raise Http404('Invalid filename')
    pcap_path = os.path.join(settings.MEDIA_ROOT, 'simulation_pcap', filename)
    if not os.path.exists(pcap_path):
        raise Http404('PCAP file not found')
    return FileResponse(open(pcap_path, 'rb'), as_attachment=True, filename=filename)


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
        
        # 準備命令列參數（已支援 --ablation-repeats）
        cmd = ['python', script_path, '--dataset', dataset, '--ablation']
        if 'all' not in experiments:
            cmd.extend(['--ablation-exp'] + experiments)
        # [Fix-8] 傳入重複次數參數
        if repeats and int(repeats) > 0:
            cmd.extend(['--ablation-repeats', str(int(repeats))])
            
        # 清空舊的 Log
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"--- 啟動消融實驗 (資料集: {dataset}) ---\n")
            f.write(f"執行指令: {' '.join(cmd)}\n\n")

        # 啟動子行程，將 stdout 與 stderr 導向至 Log 檔
        log_file = open(log_path, 'a', encoding='utf-8')
        # [Fix-7] 加入 PYTHONIOENCODING=utf-8，避免子行程輸出中文亂碼
        ablation_env = dict(os.environ, PYTHONIOENCODING='utf-8')
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=settings.BASE_DIR,
            shell=False,
            env=ablation_env,
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
    report_path = os.path.join(settings.BASE_DIR, 'output', dataset, 'ablation', 'ablation_report.json')

    # [Fix-9] 從磁碟推斷狀態（伺服器重啟後不丟失）
    if task:
        status = task['status']
    elif os.path.exists(report_path):
        status = 'done'
    elif os.path.exists(log_path):
        # 有 log 但沒有 report → 可能是執行中斷或錯誤
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                last_lines = f.readlines()[-5:]
                last_text = ''.join(last_lines).lower()
            if 'error' in last_text or 'traceback' in last_text:
                status = 'error'
            else:
                status = 'idle'  # 有舊 log 但無 report，視為閒置
        except Exception:
            status = 'idle'
    else:
        status = 'idle'
    
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
