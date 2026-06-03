# api/urls.py
from django.urls import path
from . import views

app_name = 'api'

urlpatterns = [
    # ── 認證 ────────────────────────────────────────────────
    path('auth/login/',    views.LoginAPI.as_view(),    name='login'),
    path('auth/logout/',   views.LogoutAPI.as_view(),   name='logout'),
    path('auth/me/',       views.MeAPI.as_view(),       name='me'),
    path('auth/register/', views.RegisterAPI.as_view(), name='register'),

    # ── 專案 ────────────────────────────────────────────────
    path('projects/',         views.ProjectListAPI.as_view(),   name='project_list'),
    path('projects/<int:pk>/',views.ProjectDetailAPI.as_view(), name='project_detail'),
    path('projects/<int:pk>/upload/', views.PacketFileUploadAPI.as_view(), name='packet_upload'),

    # ── Session ──────────────────────────────────────────────
    path('sessions/',              views.SessionListAPI.as_view(),   name='session_list'),
    path('sessions/<int:pk>/',     views.SessionDetailAPI.as_view(), name='session_detail'),
    path('sessions/<int:pk>/status/', views.SessionStatusAPI.as_view(), name='session_status'),

    # ── CNN ──────────────────────────────────────────────────
    path('sessions/<int:pk>/cnn/',       views.CNNResultAPI.as_view(), name='cnn_result'),
    path('sessions/<int:pk>/cnn/run/',   views.CNNRunAPI.as_view(),    name='cnn_run'),

    # ── Grad-CAM ─────────────────────────────────────────────
    path('sessions/<int:pk>/gradcam/',      views.GradCAMListAPI.as_view(), name='gradcam_list'),
    path('sessions/<int:pk>/gradcam/run/',  views.GradCAMRunAPI.as_view(),  name='gradcam_run'),

    # ── 告警 ─────────────────────────────────────────────────
    path('sessions/<int:pk>/alerts/',  views.AlertListAPI.as_view(), name='alert_list'),

    # ── 報告匯出 ─────────────────────────────────────────────
    path('sessions/<int:pk>/reports/',         views.ReportListAPI.as_view(),   name='report_list'),
    path('sessions/<int:pk>/reports/export/',  views.ReportExportAPI.as_view(), name='report_export'),

    # ── AI Chat ──────────────────────────────────────────────
    path('ai/chat/', views.AIChatAPI.as_view(), name='ai_chat'),

    # ── 系統統計（管理員）────────────────────────────────────
    path('admin/stats/', views.SystemStatsAPI.as_view(), name='system_stats'),
]
