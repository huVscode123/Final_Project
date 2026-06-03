# ============================================================
# accounts/views.py
# ============================================================
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse

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
