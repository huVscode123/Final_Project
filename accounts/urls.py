# ============================================================
# accounts/urls.py
# ============================================================
from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('register/',                views.register,          name='register'),
    path('login/',                   views.user_login,        name='login'),
    path('logout/',                  views.user_logout,       name='logout'),
    path('profile/',                 views.profile,           name='profile'),
    path('activity-logs/',           views.activity_logs,     name='activity_logs'),
    # 管理員面板
    path('admin-panel/',             views.admin_panel,       name='admin_panel'),
    path('admin/user/<int:user_id>/',views.admin_update_user, name='admin_update_user'),
]
