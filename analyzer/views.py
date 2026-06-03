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
    if session.created_by != request.user and not request.user.profile.is_admin:
        messages.error(request, '只有建立者或管理員可刪除 Session。')
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
