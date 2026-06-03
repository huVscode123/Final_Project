# ============================================================
# analyzer/admin.py
# ============================================================
from django.contrib import admin
from django.utils.html import format_html
from .models import AnalysisSession, Alert, CNNResult, GradCAMImage, AnalysisReport


class AlertInline(admin.TabularInline):
    model  = Alert
    extra  = 0
    fields = ('attack_type', 'severity', 'src_ip', 'dst_ip', 'timestamp')
    readonly_fields = ('timestamp',)
    show_change_link = True
    ordering = ('-timestamp',)


class CNNResultInline(admin.StackedInline):
    model   = CNNResult
    extra   = 0
    can_delete = False
    readonly_fields = ('threshold', 'normal_count', 'anomaly_count',
                       'avg_normal_error', 'avg_attack_error',
                       'detection_rate', 'created_at')


@admin.register(AnalysisSession)
class AnalysisSessionAdmin(admin.ModelAdmin):
    list_display  = ('label', 'project', 'mode', 'task_status_badge',
                     'packet_count', 'alert_count', 'created_by', 'created_at')
    list_filter   = ('mode', 'task_status', 'project')
    search_fields = ('label', 'pcap_file', 'created_by__username')
    readonly_fields = ('celery_task_id', 'created_at', 'finished_at', 'error_message')
    inlines       = [AlertInline, CNNResultInline]

    @admin.display(description='狀態')
    def task_status_badge(self, obj):
        colors = {
            'done':    '#3fb950', 'running': '#58a6ff',
            'pending': '#8b949e', 'failed':  '#f85149',
        }
        color = colors.get(obj.task_status, '#8b949e')
        return format_html(
            '<span style="color:{};font-weight:bold">{}</span>',
            color, obj.get_task_status_display()
        )


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display  = ('attack_type', 'severity', 'src_ip', 'dst_ip',
                     'session', 'timestamp')
    list_filter   = ('severity', 'attack_type')
    search_fields = ('src_ip', 'dst_ip', 'attack_type', 'session__label')
    date_hierarchy = 'timestamp'


@admin.register(CNNResult)
class CNNResultAdmin(admin.ModelAdmin):
    list_display = ('session', 'normal_count', 'anomaly_count',
                    'detection_rate_pct', 'threshold', 'created_at')
    readonly_fields = ('created_at',)

    @admin.display(description='偵測率')
    def detection_rate_pct(self, obj):
        return f'{obj.detection_rate * 100:.2f}%'


@admin.register(GradCAMImage)
class GradCAMImageAdmin(admin.ModelAdmin):
    list_display  = ('session', 'packet_index', 'variant', 'is_anomaly',
                     'recon_error', 'preview', 'created_at')
    list_filter   = ('variant', 'is_anomaly', 'session')
    readonly_fields = ('created_at',)

    @admin.display(description='預覽')
    def preview(self, obj):
        if obj.comparison_image:
            return format_html(
                '<img src="{}" width="80" style="border-radius:4px">',
                obj.comparison_image.url
            )
        return '-'


@admin.register(AnalysisReport)
class AnalysisReportAdmin(admin.ModelAdmin):
    list_display  = ('session', 'format', 'generated_by', 'file_size', 'created_at')
    list_filter   = ('format',)
    readonly_fields = ('created_at', 'file_size')
