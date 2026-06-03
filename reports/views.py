# ============================================================
# reports/views.py
# 報告匯出：前端觸發與下載
# ============================================================
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import FileResponse, Http404
from django.views.decorators.http import require_POST

from analyzer.models import AnalysisSession, AnalysisReport


@login_required
@require_POST
def export_report(request, session_pk):
    """
    POST /reports/export/<session_pk>/
    觸發報告匯出並導向下載。
    """
    session = get_object_or_404(AnalysisSession, pk=session_pk)

    if not request.user.profile.can_export_report:
        messages.error(request, '您沒有匯出報告的權限。')
        return redirect('analyzer:session_detail', pk=session_pk)

    fmt = request.POST.get('format', 'pdf').lower()
    if fmt not in ('pdf', 'json', 'csv'):
        messages.error(request, '不支援的格式。')
        return redirect('analyzer:session_detail', pk=session_pk)

    try:
        from .generators import ReportGenerator
        gen    = ReportGenerator(session, request.user)
        report = gen.generate(fmt)

        from accounts.views import _log_action
        _log_action(request, 'export_report',
                    f'Session {session_pk} format={fmt}')
        messages.success(request, f'報告已產生（{fmt.upper()}）。')
        return redirect('reports:download', pk=report.pk)

    except ImportError as e:
        messages.error(request, f'缺少必要套件：{e}')
    except Exception as e:
        messages.error(request, f'報告產生失敗：{e}')

    return redirect('analyzer:session_detail', pk=session_pk)


@login_required
def download_report(request, pk):
    """
    GET /reports/download/<pk>/
    下載指定報告檔案（串流輸出）。
    """
    report = get_object_or_404(AnalysisReport, pk=pk)
    session = report.session

    # 存取控制
    user = request.user
    allowed = (
        user.profile.is_admin
        or report.generated_by == user
        or session.created_by == user
        or (session.project and (
            session.project.owner == user or
            session.project.members.filter(pk=user.pk).exists()
        ))
    )
    if not allowed:
        raise Http404

    content_types = {
        'pdf':  'application/pdf',
        'json': 'application/json',
        'csv':  'text/csv; charset=utf-8-sig',
    }
    content_type = content_types.get(report.format, 'application/octet-stream')

    try:
        response = FileResponse(
            report.file.open('rb'),
            content_type=content_type,
        )
        response['Content-Disposition'] = (
            f'attachment; filename="{report.file.name.split("/")[-1]}"'
        )
        return response
    except FileNotFoundError:
        raise Http404('報告檔案不存在，請重新產生。')


@login_required
def report_list(request, session_pk):
    """
    GET /reports/session/<session_pk>/
    列出 Session 的所有報告。
    """
    session = get_object_or_404(AnalysisSession, pk=session_pk)
    reports = session.reports.order_by('-created_at')
    return render(request, 'reports/report_list.html', {
        'session': session,
        'reports': reports,
    })
