# ============================================================
# projects/views.py
# ============================================================
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .models import Project, ProjectMembership, PacketFile
from .forms import ProjectForm, PacketFileUploadForm


def _user_projects(user):
    """回傳使用者可見的所有專案（擁有或成員）。"""
    return Project.objects.filter(
        Q(owner=user) | Q(members=user)
    ).exclude(status='deleted').distinct()


@login_required
def project_list(request):
    """專案列表頁。"""
    projects = _user_projects(request.user).order_by('-updated_at')
    return render(request, 'projects/project_list.html', {
        'projects': projects,
    })


@login_required
def project_create(request):
    """建立新專案。"""
    if not request.user.profile.can_manage_projects:
        messages.error(request, '您沒有建立專案的權限。')
        return redirect('projects:list')

    if request.method == 'POST':
        form = ProjectForm(request.POST)
        if form.is_valid():
            project = form.save(commit=False)
            project.owner = request.user
            project.save()
            # 建立者自動成為 owner 成員
            ProjectMembership.objects.create(
                project=project, user=request.user, role='owner'
            )
            from accounts.views import _log_action
            _log_action(request, 'create_project', f'建立專案: {project.name}')
            messages.success(request, f'專案「{project.name}」已建立。')
            return redirect('projects:detail', pk=project.pk)
    else:
        form = ProjectForm()
    return render(request, 'projects/project_form.html', {
        'form': form, 'action': '建立',
    })


@login_required
def project_detail(request, pk):
    """專案詳情頁：列出所有 PacketFile 與 Session。"""
    project = get_object_or_404(Project, pk=pk)
    # 確認使用者有存取權
    if project.owner != request.user and not project.members.filter(pk=request.user.pk).exists():
        messages.error(request, '您沒有存取此專案的權限。')
        return redirect('projects:list')

    packet_files = project.packet_files.order_by('-uploaded_at')
    sessions     = project.sessions.order_by('-created_at')[:20]
    user_role    = project.get_member_role(request.user) or 'owner'

    return render(request, 'projects/project_detail.html', {
        'project':      project,
        'packet_files': packet_files,
        'sessions':     sessions,
        'user_role':    user_role,
    })


@login_required
def project_edit(request, pk):
    """編輯專案基本資料。"""
    project = get_object_or_404(Project, pk=pk)
    if project.owner != request.user:
        messages.error(request, '只有專案擁有者可以編輯設定。')
        return redirect('projects:detail', pk=pk)

    if request.method == 'POST':
        form = ProjectForm(request.POST, instance=project)
        if form.is_valid():
            form.save()
            messages.success(request, '專案資料已更新。')
            return redirect('projects:detail', pk=pk)
    else:
        form = ProjectForm(instance=project)
    return render(request, 'projects/project_form.html', {
        'form': form, 'project': project, 'action': '編輯',
    })


@login_required
def project_archive(request, pk):
    """封存專案（POST）。"""
    project = get_object_or_404(Project, pk=pk)
    if project.owner != request.user:
        messages.error(request, '只有專案擁有者可以封存專案。')
        return redirect('projects:detail', pk=pk)

    if request.method == 'POST':
        project.status = 'archived'
        project.save()
        messages.info(request, f'專案「{project.name}」已封存。')
    return redirect('projects:list')


@require_POST
@login_required
def member_add(request, pk):
    """新增專案成員（POST JSON）。"""
    project = get_object_or_404(Project, pk=pk)
    if project.owner != request.user:
        return JsonResponse({'error': '權限不足'}, status=403)

    from django.contrib.auth.models import User
    username = request.POST.get('username', '').strip()
    role     = request.POST.get('role', 'editor')
    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        messages.error(request, f'找不到使用者 {username}。')
        return redirect('projects:detail', pk=pk)

    membership, created = ProjectMembership.objects.get_or_create(
        project=project, user=user,
        defaults={'role': role},
    )
    if not created:
        membership.role = role
        membership.save()
    messages.success(request, f'已將 {username} 加入專案（{role}）。')
    return redirect('projects:detail', pk=pk)


@require_POST
@login_required
def member_remove(request, pk, user_pk):
    """移除專案成員（POST）。"""
    project = get_object_or_404(Project, pk=pk)
    if project.owner != request.user:
        messages.error(request, '只有擁有者可移除成員。')
        return redirect('projects:detail', pk=pk)
    ProjectMembership.objects.filter(project=project, user_id=user_pk).delete()
    messages.success(request, '已移除成員。')
    return redirect('projects:detail', pk=pk)


@login_required
def packet_file_upload(request, pk):
    """
    非同步封包上傳入口（支援大檔案）。
    上傳成功後，派發 Celery 非同步任務計算 SHA-256 並更新封包數量。
    """
    project = get_object_or_404(Project, pk=pk)
    if project.owner != request.user and \
       not ProjectMembership.objects.filter(
           project=project, user=request.user, role__in=['owner', 'editor']).exists():
        messages.error(request, '您沒有上傳封包的權限。')
        return redirect('projects:detail', pk=pk)

    if request.method == 'POST':
        form = PacketFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            packet_file = form.save(commit=False)
            packet_file.project     = project
            packet_file.uploaded_by = request.user
            packet_file.original_name = request.FILES['file'].name
            packet_file.file_size   = request.FILES['file'].size
            packet_file.save()

            # 派發非同步前處理任務
            try:
                from analyzer.tasks import process_packet_file
                process_packet_file.delay(packet_file.pk)
            except Exception:
                # Celery 未啟動時，同步執行
                _sync_process_packet_file(packet_file)

            from accounts.views import _log_action
            _log_action(request, 'upload_pcap',
                        f'上傳: {packet_file.original_name} → 專案 {project.name}')
            messages.success(request, f'檔案「{packet_file.original_name}」上傳成功，正在背景處理中。')
            return redirect('projects:detail', pk=pk)
    else:
        form = PacketFileUploadForm()
    return render(request, 'projects/packet_upload.html', {
        'form': form, 'project': project,
    })


def _sync_process_packet_file(packet_file):
    """同步版封包前處理（Celery 不可用時的 fallback）。"""
    try:
        packet_file.status = 'processing'
        packet_file.save(update_fields=['status'])
        packet_file.compute_sha256()
        # 嘗試用 scapy 取得封包數
        try:
            from scapy.all import rdpcap
            pkts = rdpcap(packet_file.file.path)
            packet_file.packet_count = len(pkts)
        except Exception:
            pass
        packet_file.status = 'done'
        packet_file.save(update_fields=['status', 'packet_count'])
    except Exception as e:
        packet_file.status = 'failed'
        packet_file.save(update_fields=['status'])


@login_required
def packet_file_delete(request, pk, file_pk):
    """刪除封包檔案（POST）。"""
    project     = get_object_or_404(Project, pk=pk)
    packet_file = get_object_or_404(PacketFile, pk=file_pk, project=project)
    if project.owner != request.user:
        messages.error(request, '只有擁有者可刪除檔案。')
        return redirect('projects:detail', pk=pk)

    if request.method == 'POST':
        name = packet_file.original_name
        try:
            packet_file.file.delete(save=False)
        except Exception:
            pass
        packet_file.delete()
        messages.success(request, f'已刪除「{name}」。')
    return redirect('projects:detail', pk=pk)
