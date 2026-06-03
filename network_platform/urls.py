# ============================================================
# network_platform/urls.py
# ============================================================
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

# 客製化後台標題
admin.site.site_header  = '網路攻擊分析平台 管理後台'
admin.site.site_title   = 'NetGuard Admin'
admin.site.index_title  = '系統管理'

urlpatterns = [
    # ── 後台管理 ──────────────────────────────────────────────
    path('admin/', admin.site.urls),

    # ── 使用者系統（登入/登出/註冊/個人資料）────────────────
    path('accounts/', include('accounts.urls', namespace='accounts')),

    # ── 專案管理（前端頁面）──────────────────────────────────
    path('projects/', include('projects.urls', namespace='projects')),

    # ── 分析模組（前端頁面）──────────────────────────────────
    path('analyzer/', include('analyzer.urls', namespace='analyzer')),

    # ── 報告匯出 ──────────────────────────────────────────────
    path('reports/', include('reports.urls', namespace='reports')),

    # ── REST API（/api/v1/...）────────────────────────────────
    path('api/v1/', include('api.urls', namespace='api')),

    # ── 根路由導向首頁 ────────────────────────────────────────
    path('', RedirectView.as_view(url='/analyzer/', permanent=False)),
]

# 開發環境 media / static 提供
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,  document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
