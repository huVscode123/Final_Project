# ============================================================
# accounts/forms.py
# ============================================================
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import UserProfile


class RegisterForm(UserCreationForm):
    email        = forms.EmailField(required=True, label='電子郵件')
    organization = forms.CharField(max_length=100, required=False, label='所屬組織')
    # [安全修正] 移除 admin 角色自選，防止權限提升漏洞
    role         = forms.ChoiceField(
        choices=[c for c in UserProfile.ROLE_CHOICES if c[0] != 'admin'],
        label='角色',
        initial='analyst',
    )

    class Meta:
        model  = User
        fields = ('username', 'email', 'first_name', 'last_name',
                  'password1', 'password2')
        labels = {
            'username':   '帳號',
            'first_name': '名字',
            'last_name':  '姓氏',
        }

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
            user.profile.organization = self.cleaned_data.get('organization', '')
            user.profile.role = self.cleaned_data.get('role', 'analyst')
            user.profile.save()
        return user


class LoginForm(AuthenticationForm):
    username = forms.CharField(label='帳號', widget=forms.TextInput(
        attrs={'autofocus': True, 'placeholder': '輸入帳號'}))
    password = forms.CharField(label='密碼', widget=forms.PasswordInput(
        attrs={'placeholder': '輸入密碼'}))


class UserProfileForm(forms.ModelForm):
    first_name   = forms.CharField(max_length=50, required=False, label='名字')
    last_name    = forms.CharField(max_length=50, required=False, label='姓氏')
    email        = forms.EmailField(required=False, label='電子郵件')

    class Meta:
        model  = UserProfile
        fields = ('organization', 'phone', 'bio', 'avatar')
        labels = {
            'organization': '所屬組織',
            'phone':        '聯絡電話',
            'bio':          '個人簡介',
            'avatar':       '頭像',
        }

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        if user:
            self.fields['first_name'].initial = user.first_name
            self.fields['last_name'].initial  = user.last_name
            self.fields['email'].initial      = user.email
