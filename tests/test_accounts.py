# ============================================================
# tests/test_accounts.py
# 使用者系統測試：註冊 / 登入 / 登出 / Profile / 稽核日誌
# ============================================================
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from accounts.models import UserProfile, UserActivityLog


class UserRegistrationTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse('accounts:register')

    def test_register_page_loads(self):
        r = self.client.get(self.url)
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, 'VNAAP')

    def test_register_creates_user_and_profile(self):
        r = self.client.post(self.url, {
            'username':     'testanalyst',
            'email':        'analyst@test.com',
            'password1':    'StrongPass123!',
            'password2':    'StrongPass123!',
            'role':         'analyst',
            'organization': 'TestOrg',
        })
        self.assertTrue(User.objects.filter(username='testanalyst').exists())
        u = User.objects.get(username='testanalyst')
        self.assertEqual(u.profile.role, 'analyst')
        self.assertEqual(u.profile.organization, 'TestOrg')

    def test_register_auto_creates_profile(self):
        """post_save signal 自動建立 UserProfile"""
        u = User.objects.create_user('sigtest', password='pass1234')
        self.assertTrue(hasattr(u, 'profile'))
        self.assertIsInstance(u.profile, UserProfile)

    def test_password_mismatch_rejected(self):
        r = self.client.post(self.url, {
            'username':  'bad_user',
            'password1': 'StrongPass123!',
            'password2': 'DifferentPass!',
        })
        self.assertFalse(User.objects.filter(username='bad_user').exists())


class UserLoginLogoutTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user('loginuser', password='testpass99')
        self.login_url  = reverse('accounts:login')
        self.logout_url = reverse('accounts:logout')

    def test_login_page_loads(self):
        r = self.client.get(self.login_url)
        self.assertEqual(r.status_code, 200)

    def test_valid_login_redirects(self):
        r = self.client.post(self.login_url, {
            'username': 'loginuser',
            'password': 'testpass99',
        })
        self.assertIn(r.status_code, [301, 302])

    def test_invalid_login_stays_on_page(self):
        r = self.client.post(self.login_url, {
            'username': 'loginuser',
            'password': 'wrongpassword',
        })
        self.assertEqual(r.status_code, 200)

    def test_logout_requires_login(self):
        r = self.client.get(self.logout_url)
        self.assertIn(r.status_code, [301, 302])

    def test_logout_creates_activity_log(self):
        self.client.login(username='loginuser', password='testpass99')
        self.client.get(self.logout_url)
        self.assertTrue(
            UserActivityLog.objects.filter(
                user=self.user, action='logout').exists()
        )


class UserProfileTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user('profuser', password='pass1234',
                                              email='prof@test.com')
        self.client.login(username='profuser', password='pass1234')
        self.url = reverse('accounts:profile')

    def test_profile_page_loads(self):
        r = self.client.get(self.url)
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, 'profuser')

    def test_profile_update(self):
        r = self.client.post(self.url, {
            'first_name':   'Test',
            'last_name':    'User',
            'email':        'new@test.com',
            'organization': 'NewOrg',
            'phone':        '0912345678',
            'bio':          'Security engineer',
        })
        self.assertIn(r.status_code, [200, 302])
        self.user.refresh_from_db()
        self.assertEqual(self.user.first_name, 'Test')

    def test_profile_requires_login(self):
        self.client.logout()
        r = self.client.get(self.url)
        self.assertIn(r.status_code, [301, 302])
        self.assertIn('/accounts/login/', r['Location'])


class UserPermissionTest(TestCase):
    def test_analyst_default_permissions(self):
        u = User.objects.create_user('analyst1', password='pass1234')
        p = u.profile
        self.assertTrue(p.can_upload_pcap)
        self.assertTrue(p.can_run_ai_analysis)
        self.assertTrue(p.can_export_report)
        self.assertTrue(p.can_manage_projects)

    def test_admin_check(self):
        u = User.objects.create_superuser('sysadmin', 'a@b.com', 'pass1234')
        self.assertTrue(u.profile.is_admin)

    def test_viewer_role_set(self):
        u = User.objects.create_user('viewer1', password='pass1234')
        u.profile.role = 'viewer'
        u.profile.save()
        self.assertEqual(u.profile.get_role_display(), '唯讀檢視者')
