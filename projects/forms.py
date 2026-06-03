# ============================================================
# projects/forms.py
# ============================================================
from django import forms
from .models import Project, PacketFile


class ProjectForm(forms.ModelForm):
    class Meta:
        model  = Project
        fields = ('name', 'description', 'network_segment', 'status')
        labels = {
            'name':            '專案名稱',
            'description':     '說明',
            'network_segment': '網段',
            'status':          '狀態',
        }
        widgets = {
            'description':     forms.Textarea(attrs={'rows': 3}),
            'network_segment': forms.TextInput(attrs={
                'placeholder': '例：192.168.1.0/24'}),
        }


class PacketFileUploadForm(forms.ModelForm):
    class Meta:
        model  = PacketFile
        fields = ('file', 'description', 'network_tag', 'capture_time')
        labels = {
            'file':         'PCAP / PCAPNG 檔案',
            'description':  '備註說明',
            'network_tag':  '網段標籤',
            'capture_time': '擷取時間',
        }
        widgets = {
            'capture_time': forms.DateTimeInput(
                attrs={'type': 'datetime-local'}, format='%Y-%m-%dT%H:%M'),
            'description': forms.Textarea(attrs={'rows': 2}),
        }
