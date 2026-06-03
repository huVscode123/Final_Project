# ============================================================
# api/serializers.py
# ============================================================
from django.contrib.auth.models import User
from rest_framework import serializers

from accounts.models import UserProfile
from projects.models import Project, ProjectMembership, PacketFile
from analyzer.models import (AnalysisSession, Alert, CNNResult,
                              GradCAMImage, AnalysisReport)


# ── 使用者 / 帳號 ─────────────────────────────────────────────
class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model  = UserProfile
        fields = ('role', 'organization', 'phone', 'bio',
                  'can_upload_pcap', 'can_run_ai_analysis',
                  'can_export_report', 'can_manage_projects')


class UserSerializer(serializers.ModelSerializer):
    profile = UserProfileSerializer(read_only=True)

    class Meta:
        model  = User
        fields = ('id', 'username', 'email', 'first_name', 'last_name',
                  'is_staff', 'date_joined', 'profile')
        read_only_fields = ('id', 'date_joined', 'is_staff')


# ── 專案 ──────────────────────────────────────────────────────
class ProjectMembershipSerializer(serializers.ModelSerializer):
    username = serializers.ReadOnlyField(source='user.username')

    class Meta:
        model  = ProjectMembership
        fields = ('id', 'user', 'username', 'role', 'joined_at')
        read_only_fields = ('joined_at',)


class PacketFileSerializer(serializers.ModelSerializer):
    file_url     = serializers.SerializerMethodField()
    uploaded_by_name = serializers.ReadOnlyField(source='uploaded_by.username')

    class Meta:
        model  = PacketFile
        fields = ('id', 'original_name', 'file_url', 'file_size',
                  'packet_count', 'sha256', 'status', 'network_tag',
                  'description', 'capture_time', 'uploaded_by_name', 'uploaded_at')
        read_only_fields = ('id', 'file_size', 'packet_count', 'sha256',
                            'status', 'uploaded_at')

    def get_file_url(self, obj):
        req = self.context.get('request')
        if obj.file and req:
            return req.build_absolute_uri(obj.file.url)
        return None


class ProjectListSerializer(serializers.ModelSerializer):
    owner_name    = serializers.ReadOnlyField(source='owner.username')
    session_count = serializers.ReadOnlyField()
    total_alerts  = serializers.ReadOnlyField()

    class Meta:
        model  = Project
        fields = ('id', 'name', 'description', 'network_segment', 'status',
                  'owner_name', 'session_count', 'total_alerts',
                  'created_at', 'updated_at')


class ProjectDetailSerializer(ProjectListSerializer):
    memberships  = ProjectMembershipSerializer(many=True, read_only=True)
    packet_files = PacketFileSerializer(many=True, read_only=True)

    class Meta(ProjectListSerializer.Meta):
        fields = ProjectListSerializer.Meta.fields + ('memberships', 'packet_files')


# ── 分析 Session ──────────────────────────────────────────────
class AlertSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Alert
        fields = ('id', 'attack_type', 'severity', 'src_ip', 'dst_ip',
                  'dst_port', 'detail', 'suggestion', 'timestamp')


class CNNResultSerializer(serializers.ModelSerializer):
    error_histogram_url = serializers.SerializerMethodField()

    class Meta:
        model  = CNNResult
        fields = ('threshold', 'normal_count', 'anomaly_count',
                  'avg_normal_error', 'avg_attack_error', 'detection_rate',
                  'error_histogram_url', 'created_at')

    def get_error_histogram_url(self, obj):
        req = self.context.get('request')
        if obj.error_histogram and req:
            return req.build_absolute_uri(obj.error_histogram.url)
        return None


class GradCAMImageSerializer(serializers.ModelSerializer):
    original_url   = serializers.SerializerMethodField()
    heatmap_url    = serializers.SerializerMethodField()
    comparison_url = serializers.SerializerMethodField()

    class Meta:
        model  = GradCAMImage
        fields = ('id', 'packet_index', 'variant', 'is_anomaly',
                  'recon_error', 'original_url', 'heatmap_url',
                  'comparison_url', 'created_at')

    def _url(self, obj, field):
        req = self.context.get('request')
        f   = getattr(obj, field)
        if f and req:
            return req.build_absolute_uri(f.url)
        return None

    def get_original_url(self, obj):   return self._url(obj, 'original_image')
    def get_heatmap_url(self, obj):    return self._url(obj, 'heatmap_image')
    def get_comparison_url(self, obj): return self._url(obj, 'comparison_image')


class AnalysisSessionListSerializer(serializers.ModelSerializer):
    project_name    = serializers.ReadOnlyField(source='project.name')
    created_by_name = serializers.ReadOnlyField(source='created_by.username')

    class Meta:
        model  = AnalysisSession
        fields = ('id', 'label', 'mode', 'task_status', 'packet_count',
                  'alert_count', 'project_name', 'created_by_name',
                  'created_at', 'finished_at')


class AnalysisSessionDetailSerializer(AnalysisSessionListSerializer):
    alerts       = AlertSerializer(many=True, read_only=True)
    cnn_result   = CNNResultSerializer(read_only=True)
    gradcam_images = GradCAMImageSerializer(many=True, read_only=True,
                                             source='gradcam_images.all')
    pcap_url     = serializers.SerializerMethodField()

    class Meta(AnalysisSessionListSerializer.Meta):
        fields = AnalysisSessionListSerializer.Meta.fields + (
            'pcap_url', 'celery_task_id', 'error_message',
            'alerts', 'cnn_result', 'gradcam_images',
        )

    def get_pcap_url(self, obj):
        req = self.context.get('request')
        if obj.pcap_file and req:
            return req.build_absolute_uri(obj.pcap_file.url)
        return None


class AnalysisReportSerializer(serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()

    class Meta:
        model  = AnalysisReport
        fields = ('id', 'format', 'file_url', 'file_size', 'created_at')

    def get_file_url(self, obj):
        req = self.context.get('request')
        if obj.file and req:
            return req.build_absolute_uri(obj.file.url)
        return None
