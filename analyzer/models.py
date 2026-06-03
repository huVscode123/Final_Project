# ============================================================
# analyzer/models.py
# 分析任務、告警、CNN 結果、視覺化影像儲存
# ============================================================
import os
from django.db import models
from django.contrib.auth.models import User


def gradcam_upload_path(instance, filename):
    return f'gradcam/session_{instance.session_id}/{filename}'


def report_upload_path(instance, filename):
    return f'reports/session_{instance.session_id}/{filename}'


class AnalysisSession(models.Model):
    """
    每次 PCAP 上傳 / 即時擷取對應一個 Session。
    可掛載至 Project（非必填，向下相容舊資料）。
    """
    MODE_CHOICES = [('pcap', 'PCAP 離線分析'), ('live', '即時擷取')]
    TASK_STATUS  = [
        ('pending',    '等待中'),
        ('running',    '分析中'),
        ('done',       '已完成'),
        ('failed',     '分析失敗'),
    ]

    # ── 關聯 ─────────────────────────────────────────────────
    project     = models.ForeignKey(
        'projects.Project', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='sessions',
        verbose_name='所屬專案',
    )
    created_by  = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='analysis_sessions', verbose_name='建立者',
    )
    packet_file = models.ForeignKey(
        'projects.PacketFile', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='sessions',
        verbose_name='來源封包檔',
    )

    # ── 基本欄位 ─────────────────────────────────────────────
    mode        = models.CharField(max_length=10, choices=MODE_CHOICES)
    label       = models.CharField(max_length=200, blank=True)
    pcap_file   = models.FileField(upload_to='uploads/', null=True, blank=True)
    task_status = models.CharField(max_length=20, choices=TASK_STATUS, default='pending')
    celery_task_id = models.CharField(max_length=100, blank=True,
                                      verbose_name='Celery Task ID')
    error_message  = models.TextField(blank=True, verbose_name='錯誤訊息')

    # ── 統計 ─────────────────────────────────────────────────
    packet_count = models.IntegerField(default=0)
    alert_count  = models.IntegerField(default=0)
    created_at   = models.DateTimeField(auto_now_add=True)
    finished_at  = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name      = '分析 Session'
        verbose_name_plural = '分析 Session'

    def __str__(self):
        proj = f' [{self.project.name}]' if self.project else ''
        return f'[{self.mode}]{proj} {self.label} ({self.created_at:%Y-%m-%d %H:%M})'


class Alert(models.Model):
    """規則式攻擊偵測結果（單筆告警）。"""
    SEVERITY_CHOICES = [
        ('LOW',      'Low'),
        ('MEDIUM',   'Medium'),
        ('HIGH',     'High'),
        ('CRITICAL', 'Critical'),
    ]

    session     = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE,
                                    related_name='alerts')
    attack_type = models.CharField(max_length=100)
    severity    = models.CharField(max_length=10, choices=SEVERITY_CHOICES)
    src_ip      = models.CharField(max_length=50)
    dst_ip      = models.CharField(max_length=50, blank=True)
    dst_port    = models.IntegerField(null=True, blank=True)
    detail      = models.TextField(blank=True)
    suggestion  = models.TextField(blank=True)
    timestamp   = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name      = '告警'
        verbose_name_plural = '告警'

    def __str__(self):
        return f'[{self.severity}] {self.attack_type} from {self.src_ip}'

    @property
    def severity_color(self):
        return {
            'LOW': '#3fb950', 'MEDIUM': '#d29922',
            'HIGH': '#f85149', 'CRITICAL': '#ff7b72',
        }.get(self.severity, '#8b949e')


class CNNResult(models.Model):
    """CNN Autoencoder 異常偵測結果。"""
    session          = models.OneToOneField(AnalysisSession, on_delete=models.CASCADE,
                                            related_name='cnn_result')
    threshold        = models.FloatField()
    normal_count     = models.IntegerField(default=0)
    anomaly_count    = models.IntegerField(default=0)
    avg_normal_error = models.FloatField(default=0)
    avg_attack_error = models.FloatField(default=0)
    detection_rate   = models.FloatField(default=0, verbose_name='偵測率')
    # 重建誤差直方圖
    error_histogram  = models.ImageField(upload_to='cnn_results/', null=True, blank=True)
    created_at       = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name      = 'CNN 結果'
        verbose_name_plural = 'CNN 結果'

    def __str__(self):
        return f'CNN Result for Session {self.session_id}'


class GradCAMImage(models.Model):
    """
    Grad-CAM 視覺化影像儲存。
    每張影像對應 Session 中的一個封包（由 packet_index 標識）。
    """
    VARIANT_CHOICES = [
        ('gradcam',   'Grad-CAM'),
        ('gradcam++', 'Grad-CAM++'),
        ('scorecam',  'Score-CAM'),
    ]

    session       = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE,
                                      related_name='gradcam_images')
    packet_index  = models.IntegerField(verbose_name='封包索引')
    variant       = models.CharField(max_length=20, choices=VARIANT_CHOICES,
                                     default='gradcam')
    is_anomaly    = models.BooleanField(default=False, verbose_name='是否為異常封包')
    recon_error   = models.FloatField(default=0, verbose_name='重建誤差')

    # 原始封包影像（灰階 32×32）
    original_image = models.ImageField(upload_to=gradcam_upload_path,
                                       null=True, blank=True,
                                       verbose_name='原始影像')
    # Grad-CAM 熱力圖疊圖
    heatmap_image  = models.ImageField(upload_to=gradcam_upload_path,
                                       null=True, blank=True,
                                       verbose_name='熱力圖')
    # 並排對比圖（原圖 + 熱力圖）
    comparison_image = models.ImageField(upload_to=gradcam_upload_path,
                                         null=True, blank=True,
                                         verbose_name='對比圖')
    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['session', 'packet_index']
        verbose_name      = 'Grad-CAM 影像'
        verbose_name_plural = 'Grad-CAM 影像'
        unique_together = ('session', 'packet_index', 'variant')

    def __str__(self):
        return (f'GradCAM #{self.packet_index} '
                f'({"異常" if self.is_anomaly else "正常"}) '
                f'Session {self.session_id}')


class AnalysisReport(models.Model):
    """
    分析報告：記錄每次報告的匯出紀錄與檔案位置。
    支援 PDF / JSON / CSV 三種格式。
    """
    FORMAT_CHOICES = [
        ('pdf',  'PDF 報告'),
        ('json', 'JSON 資料'),
        ('csv',  'CSV 告警清單'),
    ]

    session     = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE,
                                    related_name='reports')
    generated_by = models.ForeignKey(User, on_delete=models.SET_NULL,
                                     null=True, related_name='generated_reports')
    format      = models.CharField(max_length=10, choices=FORMAT_CHOICES)
    file        = models.FileField(upload_to=report_upload_path, null=True, blank=True)
    file_size   = models.IntegerField(default=0)
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name      = '分析報告'
        verbose_name_plural = '分析報告'

    def __str__(self):
        return f'Report [{self.format.upper()}] Session {self.session_id}'
