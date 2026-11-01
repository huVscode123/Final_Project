# ============================================================
# analyzer/views.py
# ============================================================
import os, json, sys, threading
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


def _json_safe(value):
    """Recursively convert NumPy/PyTorch/path values to JSON-safe primitives.

    Simulation metadata can contain numpy.int64/numpy.bool_ values emitted by
    Scapy analysis, flow conversion, and attack generators. DjangoJSONEncoder
    does not serialize NumPy scalar types, so normalize the *entire response*
    immediately before JsonResponse instead of relying on scattered casts.
    """
    type_module = type(value).__module__.split('.', 1)[0]

    # Lazy imports avoid loading numeric/ML libraries for ordinary scalar data.
    if type_module == 'numpy':
        import numpy as np
        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
        if isinstance(value, np.generic):
            return _json_safe(value.item())

    # Support optional tensors if a future response metadata field contains one.
    if type_module == 'torch':
        import torch
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            return value.item() if value.numel() == 1 else _json_safe(value.tolist())

    if isinstance(value, dict):
        # Convert keys too: NumPy integer keys are not JSON-compatible.
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


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


# [可解釋性修正] CNN 可信度說明
_CNN_RELIABILITY_NOTE_CSV_LEGACY = (
    '此模型使用 Flow 統計特徵作為輸入。正式分析時系統會先將 PCAP 轉換成與'
    '訓練期相同的 Flow 特徵欄位、沿用訓練期 scaler/normalization，再送入模型；'
    '因此 CNN/VAE 分數可與訓練時 threshold 直接比較。Grad-CAM 對應的是特徵影像座標。'
)
_CNN_RELIABILITY_NOTE_PACKET_NATIVE = (
    '此模型使用封包位元組影像作為輸入。正式分析時會以與訓練期相同的 PacketVisualizer'
    '設定（遮罩識別欄位、跳過 Ethernet、正規化）產生影像，因此 CNN/VAE 分數與 threshold'
    '使用相同表示法；Grad-CAM 對應的是封包影像局部區域。'
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
def _available_models():
    result = []
    for key, cfg in settings.ANOMALY_MODELS.items():
        if not cfg.get('enabled', True):
            continue
        path = str(cfg.get('path', ''))
        result.append({
            'key': key,
            'label': cfg.get('label', key),
            'exists': os.path.exists(path),
        })
    return result


@login_required
def upload_pcap(request):
    user_projects = Project.objects.filter(
        Q(owner=request.user)|Q(members=request.user)
    ).filter(status='active').distinct()
    available_models = _available_models()

    if request.method == 'POST':
        pcap_file  = request.FILES.get('pcap_file')
        label      = request.POST.get('label', '')
        project_id = request.POST.get('project_id')
        model_key  = request.POST.get('model_key', 'unsup_cicids2017')

        if not pcap_file:
            messages.error(request, '請選擇 PCAP 檔案。')
            return render(request, 'analyzer/upload.html', {
                'projects': user_projects,
                'pipeline_steps': _pipeline_steps(),
                'available_models': available_models,
            })

        selected = settings.ANOMALY_MODELS.get(model_key)
        if not selected or not selected.get('enabled', True):
            messages.error(request, '選擇的模型不存在或已停用。')
            return render(request, 'analyzer/upload.html', {
                'projects': user_projects,
                'pipeline_steps': _pipeline_steps(),
                'available_models': available_models,
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
            model_key=model_key,
        )

        try:
            from .tasks import run_pcap_analysis
            task = run_pcap_analysis.delay(session.pk)
            session.celery_task_id = task.id
            session.save(update_fields=['celery_task_id'])
        except Exception as exc:
            logger = __import__('logging').getLogger(__name__)
            logger.exception('Celery unavailable，改用背景執行緒執行分析: %s', exc)
            # 不在 HTTP upload request 中同步跑長任務；否則瀏覽器要等分析全部完成
            # 才能進入 Session 頁面，使用者看不到後端進度。正式環境仍建議使用 Celery。
            threading.Thread(target=_sync_run_analysis, args=(session,), daemon=True,
                             name=f'pcap-analysis-{session.pk}').start()

        from accounts.views import _log_action
        _log_action(request, 'upload_pcap', f'上傳: {pcap_file.name} → Session {session.pk} → model={model_key}')
        messages.success(request, f'「{session.label}」已上傳，背景分析中...')
        return redirect('analyzer:session_detail', pk=session.pk)

    return render(request, 'analyzer/upload.html', {
        'projects': user_projects,
        'pipeline_steps': _pipeline_steps(),
        'available_models': available_models,
    })


def _sync_run_analysis(session):
    """Celery 不可用時仍執行與非同步完全相同的流程。"""
    from analyzer.tasks import execute_pcap_analysis
    try:
        execute_pcap_analysis(session.pk, celery_task_id='sync', auto_gradcam=True)
    except Exception as e:
        from django.utils import timezone
        session = AnalysisSession.objects.get(pk=session.pk)
        session.task_status = 'failed'
        session.error_message = str(e)[:2000]
        session.progress_stage = '分析失敗'
        session.progress_detail = str(e)[:240]
        session.finished_at = timezone.now()
        session.save(update_fields=['task_status','error_message','finished_at','progress_stage','progress_detail'])


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
    gradcam_imgs = session.gradcam_images.order_by('-is_anomaly', 'packet_index')[:20]
    reports      = session.reports.order_by('-created_at')
    cnn_unit = 'packet'
    cnn_representation = '未知'
    try:
        core_path = str(settings.BASE_DIR / 'core')
        if core_path not in __import__('sys').path:
            __import__('sys').path.insert(0, core_path)
        from model_registry import load_anomaly_model
        selected = settings.ANOMALY_MODELS.get(session.model_key)
        if selected and os.path.exists(str(selected.get('path', ''))):
            bundle = load_anomaly_model(str(selected['path']), default_latent_dim=settings.CNN_LATENT_DIM)
            cnn_representation = bundle.input_representation
            cnn_unit = 'packet' if bundle.is_packet_native else 'flow'
    except Exception:
        pass

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
        'cnn_unit':     cnn_unit,
        'cnn_representation': cnn_representation,
    })


# ── Session Status (JSON polling) ────────────────────────────
@login_required
def session_status_api(request, pk):
    session = get_object_or_404(AnalysisSession, pk=pk)
    _check_session_access(request, session)
    return JsonResponse({
        'task_status':   session.task_status,
        'progress_percent': session.progress_percent,
        'progress_stage': session.progress_stage,
        'progress_detail': session.progress_detail,
        'processed_samples': session.processed_samples,
        'total_samples': session.total_samples,
        'packet_count':  session.packet_count,
        'alert_count':   session.alert_count,
        'error_message': session.error_message,
        'finished_at':   session.finished_at.isoformat() if session.finished_at else None,
        'has_cnn':       hasattr(session, 'cnn_result'),
        'gradcam_count': session.gradcam_images.count(),
        'gradcam_anomaly_count': session.gradcam_images.filter(is_anomaly=True).count(),
    })


# ── Trigger Grad-CAM ─────────────────────────────────────────
@login_required
@require_POST
def trigger_gradcam(request, pk):
    session = get_object_or_404(AnalysisSession, pk=pk)
    _check_session_access(request, session)
    if not request.user.profile.can_run_ai_analysis:
        return JsonResponse({'error': '權限不足'}, status=403)

    variant = request.POST.get('variant', 'gradcam')
    try:
        max_images = max(1, min(int(request.POST.get('max_images', 6)), 100))
    except (TypeError, ValueError):
        return JsonResponse({'error': 'max_images 必須是 1~100 的整數'}, status=400)
    anomaly_only = request.POST.get('anomaly_only', 'false').lower() == 'true'
    if variant not in {'gradcam', 'gradcam++', 'scorecam'}:
        return JsonResponse({'error': '不支援的 Grad-CAM variant'}, status=400)

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
        # 跳過已停用的模型（如 NSL-KDD 因無 PCAP 而停用）
        if not cfg.get('enabled', True):
            continue
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
    """產生真實 Scapy 合成封包，並用使用者選定的 checkpoint 做規則＋CNN 判定。

    不會合成 CNN 分數或在缺少權重時改用隨機值；攻擊情境即使未被偵測，
    也會明確回報「預期攻擊未偵測到」，而不偽稱為正常流量。
    """
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'error': '僅接受 POST'}, status=405)
    try:
        data = json.loads(request.body or '{}')
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'ok': False, 'error': '無法解析請求 JSON'}, status=400)

    attack_type = str(data.get('attack_type', 'port_scan'))
    model_key = str(data.get('model_key', 'unsup_cicids2017'))
    if attack_type not in SIMULATION_ATTACK_TYPES:
        return JsonResponse({'ok': False, 'error': f'不支援的模擬類型：{attack_type}'}, status=400)
    selected = settings.ANOMALY_MODELS.get(model_key)
    if not selected or not selected.get('enabled', True):
        return JsonResponse({'ok': False, 'error': '所選模型不存在或已停用。'}, status=400)
    model_path = str(selected.get('path', ''))
    if not model_path or not os.path.isfile(model_path):
        return JsonResponse({
            'ok': False,
            'error': f'模型權重不存在：{model_path}。請先將對應訓練完成的 .pt 檔放入指定位置。'
        }, status=503)

    try:
        intensity = float(data.get('intensity', float(data.get('packet_count', 60)) / 60.0))
    except (TypeError, ValueError):
        intensity = 1.0
    intensity = min(max(intensity, 0.3), 3.0)
    started = time.perf_counter()
    timing = {}

    try:
        import sys
        import numpy as np
        import torch
        import base64
        from types import SimpleNamespace
        core_path = str(settings.BASE_DIR / 'core')
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
        from simulate_anomaly_traffic import GENERATORS
        from model_registry import load_anomaly_model, compute_anomaly_scores, is_model_cached
        from analyzer.tasks import _iter_model_batches, _heatmap_to_png, _array_to_png
        from cnn_gradcam import GradCAM
        from scapy.all import raw as scapy_raw

        t0 = time.perf_counter()
        attack_label, gen_fn = GENERATORS[attack_type]
        packets, generation_meta = gen_fn(scale=intensity)
        packets = list(packets[:2000])
        if not packets:
            return JsonResponse({'ok': False, 'error': '產生器沒有產生任何封包。'}, status=500)
        raw_packets = []
        for pkt in packets:
            try:
                raw_packets.append(scapy_raw(pkt))
            except Exception:
                raw_packets.append(bytes(pkt))
        timing['packet_generation_ms'] = round((time.perf_counter() - t0) * 1000, 1)

        t1 = time.perf_counter()
        behavior = _detect_simulation_behavior(packets, attack_type, len(packets))
        timing['rule_engine_ms'] = round((time.perf_counter() - t1) * 1000, 1)
        pcap_filename = _save_simulation_pcap(request.user, attack_type, packets)
        pcap_path = os.path.join(settings.MEDIA_ROOT, 'simulation_pcap', pcap_filename)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t2 = time.perf_counter()
        was_cached = bool(is_model_cached(model_path, device))
        bundle = load_anomaly_model(model_path, device=device,
                                    default_latent_dim=settings.CNN_LATENT_DIM)
        timing['model_load_ms'] = round((time.perf_counter() - t2) * 1000, 1)
        timing['model_was_cached'] = was_cached
        model_representation = bundle.input_representation

        t3 = time.perf_counter()
        # 使用跟正式分析相同的輸入轉換器，不可將 flow 模型硬塞入原始 bytes 影像。
        fake_session = SimpleNamespace(model_key=model_key, packet_count=len(packets))
        all_scores, all_flags, all_statuses, scored_sample_indices = [], [], [], []
        top_score_seen = float('-inf')
        top_sample_image = None
        top_original_index = None
        for indices, X, meta in _iter_model_batches(pcap_path, bundle, selected, fake_session):
            scored = compute_anomaly_scores(bundle, X)
            scores = np.asarray(scored['score'], dtype=np.float32).reshape(-1)
            if bundle.is_hybrid:
                statuses = np.asarray(scored.get('status', ['unknown_attack'] * len(scores)), dtype=object)
                flags = statuses != 'normal'
                all_statuses.extend([str(x) for x in statuses.tolist()])
            else:
                flags = scores > float(bundle.threshold)
                statuses = np.where(flags, 'unknown_attack', 'normal')
                all_statuses.extend([str(x) for x in statuses.tolist()])
            all_scores.extend(float(x) for x in scores.tolist())
            all_flags.extend(bool(x) for x in flags.tolist())
            scored_sample_indices.extend(int(x) for x in indices)
            # 只保留目前最高分的單一輸入影像，避免數千個 PCAP 樣本影像佔滿記憶體。
            if len(scores):
                batch_top = int(np.argmax(scores))
                if float(scores[batch_top]) > top_score_seen:
                    top_score_seen = float(scores[batch_top])
                    top_sample_image = np.asarray(X[batch_top], dtype=np.float32).copy()
                    top_original_index = int(indices[batch_top])
        timing['cnn_inference_ms'] = round((time.perf_counter() - t3) * 1000, 1)
        if not all_scores:
            return JsonResponse({'ok': False, 'error': '所選 checkpoint 的前處理沒有產生可推論樣本。'}, status=422)

        scores_np = np.asarray(all_scores, dtype=np.float32)
        flags_np = np.asarray(all_flags, dtype=bool)
        top_i = int(np.argmax(scores_np))
        top_score = float(scores_np[top_i])
        top_original_index = scored_sample_indices[top_i] if top_i < len(scored_sample_indices) else top_original_index
        threshold = float(bundle.threshold)
        cnn_anomaly = bool(flags_np.any())
        rule_anomaly = bool(behavior.get('is_attack'))
        is_anomaly = bool(rule_anomaly or cnn_anomaly)
        expected_attack = attack_type != 'normal_traffic'
        missed_attack = bool(expected_attack and not is_anomaly)
        if missed_attack:
            verdict_code = 'MISSED_ATTACK'
            verdict_label = '未偵測到預期模擬攻擊'
        elif is_anomaly:
            verdict_code = 'ANOMALY'
            verdict_label = '偵測到異常證據'
        else:
            verdict_code = 'NORMAL'
            verdict_label = '本次未觀察到異常證據'

        # 取最高分的實際模型輸入，產生三種 XAI；個別 CAM 失敗會記錄，不會捏造圖像。
        x2d = top_sample_image
        gradcam_variants = []
        cam_errors = []
        if x2d is not None:
            x_tensor = torch.from_numpy(np.asarray(x2d, dtype=np.float32)[None, None, :, :]).to(bundle.device)
            for variant in ('gradcam', 'gradcam++', 'scorecam'):
                try:
                    cam = GradCAM(bundle.model, target_layer=bundle.gradcam_target_layer, variant=variant)
                    heat = np.asarray(cam.generate(x_tensor), dtype=np.float32)
                    heat = np.nan_to_num(heat, nan=0.0, posinf=1.0, neginf=0.0)
                    heat = np.clip(heat, 0.0, 1.0)
                    png = _heatmap_to_png(x_tensor, heat)
                    gradcam_variants.append({
                        'variant': variant,
                        'label': {'gradcam': 'Grad-CAM', 'gradcam++': 'Grad-CAM++', 'scorecam': 'Score-CAM'}[variant],
                        'image': 'data:image/png;base64,' + base64.b64encode(png).decode('ascii'),
                        'note': '熱點表示模型對此輸入區域較敏感，並非單獨的攻擊證明。',
                    })
                except Exception as exc:
                    cam_errors.append(f'{variant}: {type(exc).__name__}: {exc}')

        rule_alerts = behavior.get('alerts', []) or []
        if rule_anomaly:
            reason_summary = f'規則引擎觸發 {len(rule_alerts)} 筆告警。'
        elif cnn_anomaly:
            known_count = sum(x == 'known_attack' for x in all_statuses)
            unknown_count = sum(x == 'unknown_attack' for x in all_statuses)
            if bundle.is_hybrid:
                reason_summary = (f'半監督模型標記異常：已知攻擊分類 {known_count} 筆、'
                                  f'未知異常閘門 {unknown_count} 筆；最高異常分數 {top_score:.8g}，'
                                  f'重建誤差閾值 {threshold:.8g}。')
            else:
                reason_summary = (f'最高異常分數 {top_score:.8g} 高於模型閾值 {threshold:.8g}；'
                                  '至少一個樣本被非監督式模型標記為異常。')
        elif missed_attack:
            reason_summary = ('本次選擇的是攻擊模擬，但規則引擎及所選模型均未觸發。'
                              '這代表本次測試未被偵測，不能解讀為正常流量。')
        else:
            reason_summary = '選擇正常流量情境，規則引擎未告警，模型分數也未超過模型閾值。'

        # 顯示樣本時依模型單位分開：packet-native 可以對應封包；flow 模型不可把 flow index 假裝成 packet index。
        packet_alerts = behavior.get('packet_alerts', [[] for _ in packets])
        result_rows = []
        if bundle.is_packet_native:
            # score array 的位置不一定等於原 PCAP packet index（影像化失敗的封包會被跳過）。
            packet_to_score_pos = {packet_idx: pos for pos, packet_idx in enumerate(scored_sample_indices)}
            alert_positions = [packet_to_score_pos[i] for i, a in enumerate(packet_alerts)
                               if a and i in packet_to_score_pos]
            # 保證最高分樣本一定納入，其他則依規則告警索引納入。
            candidate_positions = list(dict.fromkeys(alert_positions[:99] + [top_i]))[:100]
            for pos in candidate_positions:
                i = scored_sample_indices[pos]
                if i >= len(packets) or i >= len(raw_packets):
                    continue
                row_alerts = packet_alerts[i] if i < len(packet_alerts) else []
                result_rows.append({
                    'index': i, 'sample_unit': 'packet', 'packet_bytes': len(raw_packets[i]),
                    'summary': packets[i].summary(), 'score': round(float(scores_np[pos]), 8),
                    'threshold': threshold, 'cnn_anomaly': bool(flags_np[pos]),
                    'rule_triggered': bool(row_alerts),
                    'rule_detail': [str(x.get('detail', x.get('attack_type', '觸發規則'))) for x in row_alerts],
                })
        else:
            candidate_indices = sorted(set([int(np.argmax(scores_np))] + list(np.argsort(scores_np)[-15:])))
            for i in candidate_indices:
                if i >= len(scores_np):
                    continue
                result_rows.append({
                    'index': i, 'sample_unit': 'flow', 'packet_bytes': None,
                    'summary': '由 PCAP 轉換器產生的 Flow 特徵樣本（非單一封包）',
                    'score': round(float(scores_np[i]), 8), 'threshold': threshold,
                    'cnn_anomaly': bool(flags_np[i]), 'rule_triggered': None, 'rule_detail': [],
                })

        timing['total_ms'] = round((time.perf_counter() - started) * 1000, 1)
        response_payload = {
            'ok': True,
            'model_key': model_key,
            'model_label': selected.get('label', model_key),
            'model_representation': model_representation,
            'attack_type': attack_type,
            'attack_type_label': attack_label,
            'verdict_code': verdict_code,
            'verdict_label': verdict_label,
            'is_anomaly': is_anomaly,
            'missed_attack': missed_attack,
            'reason_summary': reason_summary,
            'rule_based_anomaly_detected': rule_anomaly,
            'rule_alert_count': len(rule_alerts),
            'rule_alerts': rule_alerts[:20],
            'cnn_is_anomaly': cnn_anomaly,
            'cnn_anomaly_count': int(flags_np.sum()),
            'known_attack_count': sum(x == 'known_attack' for x in all_statuses),
            'unknown_attack_count': sum(x == 'unknown_attack' for x in all_statuses),
            'sample_count': len(scores_np),
            'sample_unit': 'packet' if bundle.is_packet_native else 'flow',
            'packet_count': len(packets),
            'avg_packet_size': round(sum(len(x) for x in raw_packets) / max(len(raw_packets), 1), 1),
            'reconstruction_error': top_score,
            'threshold': threshold,
            'anomaly_ratio': round(float(flags_np.mean()), 4),
            'top_sample_index': top_original_index if top_original_index is not None else top_i,
            'top_sample_status': all_statuses[top_i],
            'sample_visual': _array_to_data_url(x2d) if x2d is not None else None,
            'gradcam_variants': gradcam_variants,
            'gradcam_errors': cam_errors,
            'generation_meta': generation_meta,
            'timing_ms': timing,
            'pcap_filename': pcap_filename,
            'pcap_download_url': request.build_absolute_uri(reverse('analyzer:simulation_pcap_download', args=[pcap_filename])),
            'results': result_rows,
            'limitations': '合成流量只能驗證目前產生器與偵測器的互動；未偵測到不代表模型對真實攻擊具有同樣漏報率。',
        }
        # Normalize nested NumPy scalar types (e.g. numpy.int64) from the
        # packet/flow converter, rule detector and simulation metadata.
        return JsonResponse(_json_safe(response_payload))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JsonResponse({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}, status=500)

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
        dataset = data.get('dataset', 'unsup_cicids2017')
        experiments = data.get('experiments', ['all'])
        valid_models = {'unsup_cicids2017', 'semi_cicids2017', 'mta_malware', 'semi_mta_malware'}
        if dataset not in valid_models:
            return JsonResponse({'ok': False, 'error': '不支援的消融模型/資料集組合'}, status=400)
        repeats = data.get('repeats', 1)
        
        # 建立輸出目錄與 Log 檔案
        output_dir = os.path.join(settings.BASE_DIR, 'output', dataset, 'ablation')
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, 'ablation.log')
        
        script_path = os.path.join(settings.BASE_DIR, 'core', 'run_model_ablation.py')

        # 固定使用該模型自己的 .pt 與對應資料表示；此處是推論輸入消融，不重新訓練架構。
        cmd = [sys.executable, script_path, '--model-key', dataset]
        if 'all' in experiments:
            experiments = ['mask10', 'mask25', 'shuffle_rows', 'sample50']
        cmd.extend(['--experiments'] + list(experiments))
            
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
    dataset = request.GET.get('dataset', 'unsup_cicids2017')
    
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
