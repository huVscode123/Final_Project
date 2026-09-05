# ============================================================
# api/permissions.py（新建檔案，放在 backend/api/permissions.py）
#
# 共用的 Session 存取權限檢查，供 api/views.py 內所有需要操作
# AnalysisSession 的端點使用（AIChatAPI、SessionStatusAPI、
# CNNResultAPI、GradCAMRunAPI... 等，詳見前次程式碼審查報告
# 「P0-2 — Django REST API 系統性 IDOR 越權漏洞」章節）。
# ============================================================

from rest_framework.response import Response
from analyzer.models import AnalysisSession


def get_session_or_403(request, pk):
    """
    檢查目前使用者是否有權限存取指定的 AnalysisSession。

    權限規則（比照 analyzer/views.py 的 _check_session_access）：
        - 系統管理員（is_admin）
        - Session 建立者本人
        - 所屬專案的擁有者
        - 所屬專案的成員

    回傳:
        (session, None)              — 有權限，session 為該物件
        (None, Response(..., 403))   — 無權限
        (None, Response(..., 404))   — Session 不存在
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
