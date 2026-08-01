"""API 共用權限檢查工具"""
from rest_framework.response import Response
from analyzer.models import AnalysisSession


def get_session_or_403(request, pk):
    """
    共用的 Session 存取檢查（比照 analyzer/views.py 的 _check_session_access）。
    回傳 (session, None) 或 (None, error_response)。
    """
    try:
        s = AnalysisSession.objects.get(pk=pk)
    except AnalysisSession.DoesNotExist:
        return None, Response({'error': 'Session 不存在'}, status=404)

    user = request.user
    has_access = (
        user.profile.is_admin
        or s.created_by == user
        or (s.project and (
            s.project.owner == user
            or s.project.members.filter(pk=user.pk).exists()
        ))
    )
    if not has_access:
        return None, Response({'error': '權限不足'}, status=403)
    return s, None
