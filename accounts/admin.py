# ============================================================
# accounts/admin.py
# ============================================================
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from .models import UserProfile, UserActivityLog


class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = '使用者資料'
    fields = ('role', 'organization', 'phone', 'bio',
              'can_upload_pcap', 'can_run_ai_analysis',
              'can_export_report', 'can_manage_projects')


class CustomUserAdmin(BaseUserAdmin):
    inlines = (UserProfileInline,)
    list_display  = ('username', 'email', 'get_role', 'get_org',
                     'is_staff', 'is_active', 'date_joined')
    list_filter   = ('is_staff', 'is_active', 'profile__role')
    search_fields = ('username', 'email', 'profile__organization')

    @admin.display(description='角色', ordering='profile__role')
    def get_role(self, obj):
        return obj.profile.get_role_display() if hasattr(obj, 'profile') else '-'

    @admin.display(description='組織', ordering='profile__organization')
    def get_org(self, obj):
        return obj.profile.organization if hasattr(obj, 'profile') else '-'


admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)


@admin.register(UserActivityLog)
class UserActivityLogAdmin(admin.ModelAdmin):
    list_display  = ('timestamp', 'user', 'action', 'ip_address', 'detail')
    list_filter   = ('action',)
    search_fields = ('user__username', 'detail')
    readonly_fields = ('user', 'action', 'detail', 'ip_address', 'timestamp')
    ordering      = ('-timestamp',)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
