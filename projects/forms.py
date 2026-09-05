# ============================================================
# projects/forms.py
# ============================================================
from django import forms
from django.core.exceptions import ValidationError
from .models import Project, PacketFile

ALLOWED_EXTENSIONS = ('.pcap', '.pcapng', '.cap')
MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200MB


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

    def clean_file(self):
        import os
        f = self.cleaned_data['file']
        # ── [P2-7 修正] 檔名消毒 ──
        f.name = os.path.basename(f.name)
        name_lower = f.name.lower()
        if not name_lower.endswith(ALLOWED_EXTENSIONS):
            raise ValidationError(
                f'不支援的檔案格式，僅接受 {", ".join(ALLOWED_EXTENSIONS)}。')
        if f.size > MAX_UPLOAD_SIZE:
            raise ValidationError(
                f'檔案過大（{f.size/1024/1024:.1f}MB），上限為 '
                f'{MAX_UPLOAD_SIZE/1024/1024:.0f}MB。')
        head = f.read(4)
        f.seek(0)
        valid_magic = head in (
            b'\xa1\xb2\xc3\xd4', b'\xd4\xc3\xb2\xa1',
            b'\x0a\x0d\x0d\x0a',
        )
        if not valid_magic:
            raise ValidationError('檔案內容不像是合法的 PCAP/PCAPNG 檔案。')
        return f
