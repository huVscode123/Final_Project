# ============================================================
# projects/models.py
# 專案管理：讓使用者將不同時間點、不同網段的封包分門別類
# ============================================================
import os
from django.db import models
from django.contrib.auth.models import User


def pcap_upload_path(instance, filename):
    """依 project_id / session_id 分目錄儲存 PCAP。"""
    return f'pcap/project_{instance.project.id}/{filename}'


class Project(models.Model):
    """
    一個專案代表一個分析任務群，可對應單一網段、攻防演練或合規稽核等情境。
    擁有者（owner）有完整控制權；成員（members）依角色限制操作。
    """
    STATUS_CHOICES = [
        ('active',   '進行中'),
        ('archived', '已封存'),
        ('deleted',  '已刪除'),
    ]

    name        = models.CharField(max_length=200, verbose_name='專案名稱')
    description = models.TextField(blank=True, verbose_name='專案說明')
    network_segment = models.CharField(max_length=100, blank=True,
                                       verbose_name='網段（如 192.168.1.0/24）')
    status      = models.CharField(max_length=20, choices=STATUS_CHOICES,
                                   default='active')
    owner       = models.ForeignKey(User, on_delete=models.CASCADE,
                                    related_name='owned_projects',
                                    verbose_name='擁有者')
    members     = models.ManyToManyField(User, through='ProjectMembership',
                                         related_name='joined_projects',
                                         blank=True)
    created_at  = models.DateTimeField(auto_now_add=True)
    updated_at  = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        verbose_name      = '專案'
        verbose_name_plural = '專案'

    def __str__(self):
        return f'[{self.get_status_display()}] {self.name}'

    def get_member_role(self, user):
        """回傳指定使用者在本專案的角色（或 None）。"""
        try:
            return self.memberships.get(user=user).role
        except ProjectMembership.DoesNotExist:
            return None

    @property
    def session_count(self):
        return self.sessions.count()

    @property
    def total_alerts(self):
        from analyzer.models import Alert
        return Alert.objects.filter(session__project=self).count()


class ProjectMembership(models.Model):
    """中介表：使用者在專案中的角色。"""
    ROLE_CHOICES = [
        ('owner',    '擁有者'),
        ('editor',   '編輯者'),
        ('viewer',   '檢視者'),
    ]

    project    = models.ForeignKey(Project, on_delete=models.CASCADE,
                                   related_name='memberships')
    user       = models.ForeignKey(User, on_delete=models.CASCADE,
                                   related_name='project_memberships')
    role       = models.CharField(max_length=20, choices=ROLE_CHOICES,
                                  default='editor')
    joined_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('project', 'user')
        verbose_name      = '專案成員'
        verbose_name_plural = '專案成員'

    def __str__(self):
        return f'{self.user.username} @ {self.project.name} ({self.role})'


class PacketFile(models.Model):
    """
    封包檔案管理：每個 PacketFile 代表一個上傳的 PCAP / PCAPNG 檔案。
    與 Project 關聯，並追蹤處理狀態。
    """
    STATUS_CHOICES = [
        ('pending',    '等待處理'),
        ('processing', '處理中'),
        ('done',       '已完成'),
        ('failed',     '處理失敗'),
    ]

    project      = models.ForeignKey(Project, on_delete=models.CASCADE,
                                     related_name='packet_files',
                                     verbose_name='所屬專案')
    uploaded_by  = models.ForeignKey(User, on_delete=models.SET_NULL,
                                     null=True, related_name='uploaded_files',
                                     verbose_name='上傳者')
    file         = models.FileField(upload_to=pcap_upload_path,
                                    verbose_name='PCAP 檔案')
    original_name = models.CharField(max_length=255, verbose_name='原始檔名')
    file_size    = models.BigIntegerField(default=0, verbose_name='檔案大小 (bytes)')
    capture_time = models.DateTimeField(null=True, blank=True,
                                        verbose_name='擷取時間')
    description  = models.TextField(blank=True, verbose_name='檔案備註')
    network_tag  = models.CharField(max_length=100, blank=True,
                                    verbose_name='網段標籤')
    status       = models.CharField(max_length=20, choices=STATUS_CHOICES,
                                    default='pending')
    packet_count = models.IntegerField(default=0, verbose_name='封包數量')
    sha256       = models.CharField(max_length=64, blank=True,
                                    verbose_name='SHA-256 校驗碼')
    uploaded_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']
        verbose_name      = '封包檔案'
        verbose_name_plural = '封包檔案'

    def __str__(self):
        return f'{self.original_name} ({self.project.name})'

    @property
    def file_size_mb(self):
        return round(self.file_size / (1024 * 1024), 2)

    def compute_sha256(self):
        """計算並儲存 SHA-256 校驗碼。"""
        import hashlib
        h = hashlib.sha256()
        with open(self.file.path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        self.sha256 = h.hexdigest()
        self.save(update_fields=['sha256'])
        return self.sha256
