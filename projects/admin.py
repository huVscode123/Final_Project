# ============================================================
# projects/admin.py
# ============================================================
from django.contrib import admin
from .models import Project, ProjectMembership, PacketFile


class ProjectMembershipInline(admin.TabularInline):
    model  = ProjectMembership
    extra  = 1
    fields = ('user', 'role', 'joined_at')
    readonly_fields = ('joined_at',)


class PacketFileInline(admin.TabularInline):
    model  = PacketFile
    extra  = 0
    fields = ('original_name', 'file_size', 'status', 'network_tag', 'uploaded_at')
    readonly_fields = ('file_size', 'uploaded_at')
    show_change_link = True


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display  = ('name', 'owner', 'network_segment', 'status',
                     'session_count', 'created_at')
    list_filter   = ('status',)
    search_fields = ('name', 'network_segment', 'owner__username')
    inlines       = [ProjectMembershipInline, PacketFileInline]
    readonly_fields = ('created_at', 'updated_at')

    @admin.display(description='Session 數')
    def session_count(self, obj):
        return obj.sessions.count()


@admin.register(PacketFile)
class PacketFileAdmin(admin.ModelAdmin):
    list_display  = ('original_name', 'project', 'uploaded_by', 'file_size_mb',
                     'packet_count', 'status', 'uploaded_at')
    list_filter   = ('status', 'project')
    search_fields = ('original_name', 'network_tag', 'sha256')
    readonly_fields = ('sha256', 'uploaded_at', 'file_size')

    @admin.display(description='大小 (MB)')
    def file_size_mb(self, obj):
        return f'{obj.file_size_mb} MB'
