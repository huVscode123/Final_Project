# ============================================================
# accounts/models.py
# 使用者擴充資料（UserProfile）與角色/權限系統
# ============================================================
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    """
    擴充 Django 預設 User 的個人資料。
    透過 OneToOneField 與 User 連結，信號自動建立。
    """
    ROLE_CHOICES = [
        ('admin',    '系統管理員'),
        ('analyst',  '資安分析師'),
        ('viewer',   '唯讀檢視者'),
    ]

    user         = models.OneToOneField(User, on_delete=models.CASCADE,
                                        related_name='profile')
    role         = models.CharField(max_length=20, choices=ROLE_CHOICES,
                                    default='analyst')
    organization = models.CharField(max_length=100, blank=True, verbose_name='所屬組織')
    phone        = models.CharField(max_length=30, blank=True, verbose_name='聯絡電話')
    avatar       = models.ImageField(upload_to='avatars/', null=True, blank=True,
                                     verbose_name='頭像')
    bio          = models.TextField(blank=True, verbose_name='個人簡介')
    created_at   = models.DateTimeField(auto_now_add=True)
    updated_at   = models.DateTimeField(auto_now=True)

    # ── 功能開關 ─────────────────────────────────────────────
    can_upload_pcap      = models.BooleanField(default=True,  verbose_name='可上傳 PCAP')
    can_run_ai_analysis  = models.BooleanField(default=True,  verbose_name='可執行 AI 分析')
    can_export_report    = models.BooleanField(default=True,  verbose_name='可匯出報告')
    can_manage_projects  = models.BooleanField(default=True,  verbose_name='可管理專案')

    class Meta:
        verbose_name      = '使用者資料'
        verbose_name_plural = '使用者資料'

    def __str__(self):
        return f'{self.user.username} ({self.get_role_display()})'

    @property
    def is_admin(self):
        return self.role == 'admin' or self.user.is_staff

    @property
    def is_viewer(self):
        return self.role == 'viewer'


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """User 建立時自動建立 UserProfile。"""
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """User 儲存時同步儲存 Profile。"""
    if hasattr(instance, 'profile'):
        instance.profile.save()


class UserActivityLog(models.Model):
    """
    使用者操作稽核日誌。
    記錄上傳、分析、匯出等關鍵操作。
    """
    ACTION_CHOICES = [
        ('login',           '登入系統'),
        ('logout',          '登出系統'),
        ('upload_pcap',     '上傳 PCAP'),
        ('run_analysis',    '執行分析'),
        ('run_gradcam',     '執行 Grad-CAM'),
        ('export_report',   '匯出報告'),
        ('create_project',  '建立專案'),
        ('delete_session',  '刪除 Session'),
    ]

    user       = models.ForeignKey(User, on_delete=models.CASCADE,
                                   related_name='activity_logs')
    action     = models.CharField(max_length=30, choices=ACTION_CHOICES)
    detail     = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    timestamp  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name      = '操作日誌'
        verbose_name_plural = '操作日誌'

    def __str__(self):
        return f'[{self.timestamp:%Y-%m-%d %H:%M}] {self.user.username} - {self.get_action_display()}'
