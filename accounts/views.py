# ============================================================
# accounts/views.py
# ============================================================
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
import json

from .forms import RegisterForm, LoginForm, UserProfileForm
from .models import UserActivityLog


def _log_action(request, action, detail=''):
    """記錄使用者操作至稽核日誌。"""
    ip = request.META.get('HTTP_X_FORWARDED_FOR',
                          request.META.get('REMOTE_ADDR', ''))
    if ip and ',' in ip:
        ip = ip.split(',')[0].strip()
    UserActivityLog.objects.create(
        user=request.user, action=action,
        detail=detail, ip_address=ip or None,
    )


def register(request):
    """使用者註冊頁面。"""
    if request.user.is_authenticated:
        return redirect('/')
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'歡迎加入！帳號 {user.username} 已建立。')
            return redirect('/')
    else:
        form = RegisterForm()
    return render(request, 'accounts/register.html', {'form': form})


def user_login(request):
    """使用者登入頁面。"""
    if request.user.is_authenticated:
        return redirect('/')
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            _log_action(request, 'login')
            messages.success(request, f'歡迎回來，{user.get_full_name() or user.username}！')
            return redirect(request.GET.get('next', '/'))
        else:
            messages.error(request, '帳號或密碼錯誤，請重試。')
    else:
        form = LoginForm()
    return render(request, 'accounts/login.html', {'form': form})


@login_required
def user_logout(request):
    """登出。"""
    _log_action(request, 'logout')
    logout(request)
    messages.info(request, '您已安全登出系統。')
    return redirect('accounts:login')


@login_required
def profile(request):
    """使用者個人資料頁面。"""
    profile_obj = request.user.profile
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES,
                               instance=profile_obj, user=request.user)
        if form.is_valid():
            form.save()
            # 同步更新 User 欄位
            u = request.user
            u.first_name = form.cleaned_data.get('first_name', '')
            u.last_name  = form.cleaned_data.get('last_name', '')
            u.email      = form.cleaned_data.get('email', '')
            u.save()
            messages.success(request, '個人資料已更新。')
            return redirect('accounts:profile')
    else:
        form = UserProfileForm(instance=profile_obj, user=request.user)

    # 最近操作紀錄
    logs = UserActivityLog.objects.filter(user=request.user)[:20]
    return render(request, 'accounts/profile.html', {
        'form': form, 'logs': logs,
    })


@login_required
def activity_logs(request):
    """使用者操作稽核日誌（管理員可看全部）。"""
    if request.user.profile.is_admin:
        logs = UserActivityLog.objects.select_related('user').all()[:200]
    else:
        logs = UserActivityLog.objects.filter(user=request.user)[:100]
    return render(request, 'accounts/activity_logs.html', {'logs': logs})


@login_required
def admin_panel(request):
    """管理員管理面板 — 使用者管理、系統統計。"""
    if not request.user.profile.is_admin:
        messages.error(request, '權限不足：僅管理員可存取。')
        return redirect('analyzer:dashboard')

    from django.contrib.auth.models import User
    from analyzer.models import AnalysisSession, Alert
    from projects.models import Project

    users = User.objects.select_related('profile').all().order_by('-date_joined')
    logs = UserActivityLog.objects.select_related('user').all()[:50]

    # 系統統計
    stats = {
        'total_users':    users.count(),
        'admin_count':    users.filter(profile__role='admin').count(),
        'analyst_count':  users.filter(profile__role='analyst').count(),
        'viewer_count':   users.filter(profile__role='viewer').count(),
        'total_sessions': AnalysisSession.objects.count(),
        'done_sessions':  AnalysisSession.objects.filter(task_status='done').count(),
        'total_alerts':   Alert.objects.count(),
        'critical_alerts':Alert.objects.filter(severity='CRITICAL').count(),
        'total_projects': Project.objects.exclude(status='deleted').count(),
    }

    from django.utils import timezone
    from datetime import timedelta
    from analyzer.models import GradCAMImage, AnalysisReport

    now = timezone.now()

    # 任務狀態分布
    task_status_data = {
        'done': AnalysisSession.objects.filter(task_status='done').count(),
        'running': AnalysisSession.objects.filter(task_status='running').count(),
        'pending': AnalysisSession.objects.filter(task_status='pending').count(),
        'failed': AnalysisSession.objects.filter(task_status='failed').count(),
    }

    # 過去 30 天使用者活躍度
    daily_activity = {}
    for i in range(29, -1, -1):
        day = (now - timedelta(days=i)).date()
        daily_activity[str(day)] = UserActivityLog.objects.filter(
            timestamp__date=day
        ).count()

    # 最活躍使用者 Top 5
    from django.db.models import Count as DjCount
    top_users = (UserActivityLog.objects
                 .values('user__username')
                 .annotate(action_count=DjCount('id'))
                 .order_by('-action_count')[:5])

    # 功能使用統計
    feature_usage = {
        '封包分析': AnalysisSession.objects.count(),
        '安全告警': Alert.objects.count(),
        'Grad-CAM': GradCAMImage.objects.count(),
        '報告匯出': AnalysisReport.objects.count(),
    }

    # CNN 模型資訊
    from django.conf import settings
    import os
    model_path = str(settings.CNN_MODEL_PATH)
    model_info = {
        'exists': os.path.exists(model_path),
        'threshold': settings.CNN_THRESHOLD,
        'latent_dim': settings.CNN_LATENT_DIM,
        'file_size_mb': round(os.path.getsize(model_path) / 1024 / 1024, 2) if os.path.exists(model_path) else 0,
    }

    return render(request, 'accounts/admin_panel.html', {
        'users': users,
        'logs': logs,
        'stats': stats,
        'task_status_data': json.dumps(task_status_data),
        'daily_activity': json.dumps(daily_activity),
        'top_users': list(top_users),
        'feature_usage': feature_usage,
        'model_info': model_info,
    })


@login_required
def admin_update_user(request, user_id):
    """管理員更新使用者角色或狀態（AJAX）。"""
    if not request.user.profile.is_admin:
        return JsonResponse({'error': '權限不足'}, status=403)

    import json
    from django.contrib.auth.models import User

    try:
        target = User.objects.get(pk=user_id)
    except User.DoesNotExist:
        return JsonResponse({'error': '使用者不存在'}, status=404)

    if request.method == 'POST':
        data = json.loads(request.body)
        action = data.get('action')

        if action == 'change_role':
            new_role = data.get('role')
            if new_role in ['admin', 'analyst', 'viewer']:
                target.profile.role = new_role
                target.profile.save()
                _log_action(request, 'admin_change_role',
                            f'將 {target.username} 的角色變更為 {new_role}')
                return JsonResponse({'ok': True, 'role': new_role})

        elif action == 'toggle_active':
            target.is_active = not target.is_active
            target.save()
            status = '啟用' if target.is_active else '停用'
            _log_action(request, 'admin_toggle_active',
                        f'{status}使用者 {target.username}')
            return JsonResponse({'ok': True, 'is_active': target.is_active})

    return JsonResponse({'error': '無效的操作'}, status=400)

