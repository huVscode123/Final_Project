# ============================================================
# tests/test_api.py
# REST API 測試：Token 認證 / Session / CNN / GradCAM / Report
# ============================================================
import json
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

from projects.models import Project, ProjectMembership
from analyzer.models import AnalysisSession, Alert, CNNResult


def make_user(username, password='pass1234', role='analyst'):
    u = User.objects.create_user(username, password=password)
    u.profile.role = role
    u.profile.save()
    return u


def get_token(user):
    token, _ = Token.objects.get_or_create(user=user)
    return token.key


def auth_client(user):
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION='Token ' + get_token(user))
    return client


def make_session(user, status='done', label='API Test Session'):
    return AnalysisSession.objects.create(
        mode='pcap', label=label,
        task_status=status,
        created_by=user,
        packet_count=50,
        alert_count=2,
    )


# ─── Auth API ───────────────────────────────────────────────
class AuthAPITest(TestCase):
    def setUp(self):
        self.user = make_user('apiuser')
        self.client = Client()

    def test_login_returns_token(self):
        r = self.client.post('/api/v1/auth/login/', {
            'username': 'apiuser',
            'password': 'pass1234',
        }, content_type='application/json')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertIn('token', data)
        self.assertIn('username', data)
        self.assertEqual(data['username'], 'apiuser')

    def test_login_wrong_password(self):
        r = self.client.post('/api/v1/auth/login/', {
            'username': 'apiuser',
            'password': 'wrongpassword',
        }, content_type='application/json')
        self.assertEqual(r.status_code, 400)

    def test_me_endpoint_returns_user(self):
        client = auth_client(self.user)
        r = client.get('/api/v1/auth/me/')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertEqual(data['username'], 'apiuser')
        self.assertIn('profile', data)

    def test_me_requires_auth(self):
        r = self.client.get('/api/v1/auth/me/')
        self.assertEqual(r.status_code, 401)

    def test_logout_deletes_token(self):
        token_key = get_token(self.user)
        client = auth_client(self.user)
        r = client.post('/api/v1/auth/logout/')
        self.assertEqual(r.status_code, 200)
        self.assertFalse(Token.objects.filter(key=token_key).exists())

    def test_register_via_api(self):
        r = self.client.post('/api/v1/auth/register/', {
            'username':     'newregister',
            'email':        'new@test.com',
            'password1':    'StrongPass123!',
            'password2':    'StrongPass123!',
            'role':         'analyst',
            'organization': 'TestOrg',
        }, content_type='application/json')
        self.assertEqual(r.status_code, 201)
        data = json.loads(r.content)
        self.assertIn('token', data)


# ─── Project API ─────────────────────────────────────────────
class ProjectAPITest(TestCase):
    def setUp(self):
        self.user   = make_user('projapi')
        self.client = auth_client(self.user)

    def test_list_projects(self):
        Project.objects.create(name='API Project', owner=self.user)
        r = self.client.get('/api/v1/projects/')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertIsInstance(data, list)

    def test_create_project(self):
        r = self.client.post('/api/v1/projects/', {
            'name':            'New API Project',
            'network_segment': '172.16.0.0/12',
            'description':     'Created via API',
            'status':          'active',
        }, format='json')
        self.assertEqual(r.status_code, 201)
        data = json.loads(r.content)
        self.assertEqual(data['name'], 'New API Project')

    def test_get_project_detail(self):
        p = Project.objects.create(name='DetailProj', owner=self.user)
        ProjectMembership.objects.create(project=p, user=self.user, role='owner')
        r = self.client.get(f'/api/v1/projects/{p.pk}/')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertEqual(data['name'], 'DetailProj')
        self.assertIn('memberships', data)

    def test_update_project(self):
        p = Project.objects.create(name='OldName', owner=self.user)
        ProjectMembership.objects.create(project=p, user=self.user, role='owner')
        r = self.client.put(f'/api/v1/projects/{p.pk}/', {
            'name': 'UpdatedName', 'status': 'active',
        }, format='json')
        self.assertEqual(r.status_code, 200)
        p.refresh_from_db()
        self.assertEqual(p.name, 'UpdatedName')

    def test_delete_project(self):
        p = Project.objects.create(name='ToDelete', owner=self.user)
        ProjectMembership.objects.create(project=p, user=self.user, role='owner')
        r = self.client.delete(f'/api/v1/projects/{p.pk}/')
        self.assertEqual(r.status_code, 204)
        p.refresh_from_db()
        self.assertEqual(p.status, 'deleted')

    def test_non_member_cannot_access_project(self):
        other = make_user('other_api')
        p = Project.objects.create(name='Private', owner=other)
        r = self.client.get(f'/api/v1/projects/{p.pk}/')
        self.assertEqual(r.status_code, 403)


# ─── Session API ─────────────────────────────────────────────
class SessionAPITest(TestCase):
    def setUp(self):
        self.user    = make_user('sessapi')
        self.client  = auth_client(self.user)
        self.session = make_session(self.user)

    def test_list_sessions(self):
        r = self.client.get('/api/v1/sessions/')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertIsInstance(data, list)
        ids = [s['id'] for s in data]
        self.assertIn(self.session.pk, ids)

    def test_get_session_detail(self):
        r = self.client.get(f'/api/v1/sessions/{self.session.pk}/')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertEqual(data['id'], self.session.pk)
        self.assertIn('alerts', data)
        self.assertIn('gradcam_images', data)

    def test_get_session_status(self):
        r = self.client.get(f'/api/v1/sessions/{self.session.pk}/status/')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertIn('task_status', data)
        self.assertIn('packet_count', data)
        self.assertIn('gradcam_count', data)

    def test_filter_sessions_by_status(self):
        make_session(self.user, status='failed', label='FailedSession')
        r = self.client.get('/api/v1/sessions/?status=done')
        data = json.loads(r.content)
        statuses = [s['task_status'] for s in data]
        self.assertTrue(all(s == 'done' for s in statuses))

    def test_delete_own_session(self):
        s = make_session(self.user, label='ToDelete')
        pk = s.pk
        r = self.client.delete(f'/api/v1/sessions/{pk}/')
        self.assertEqual(r.status_code, 204)
        self.assertFalse(AnalysisSession.objects.filter(pk=pk).exists())

    def test_cannot_delete_others_session(self):
        other = make_user('othersess')
        s = make_session(other, label='OtherPrivate')
        r = self.client.delete(f'/api/v1/sessions/{s.pk}/')
        self.assertEqual(r.status_code, 403)


# ─── CNN API ────────────────────────────────────────────────
class CNNAPITest(TestCase):
    def setUp(self):
        self.user    = make_user('cnnapi')
        self.client  = auth_client(self.user)
        self.session = make_session(self.user)
        self.cnn     = CNNResult.objects.create(
            session=self.session, threshold=0.000822,
            normal_count=80, anomaly_count=20,
            detection_rate=0.20,
        )

    def test_get_cnn_result(self):
        r = self.client.get(f'/api/v1/sessions/{self.session.pk}/cnn/')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertIn('threshold', data)
        self.assertIn('detection_rate', data)
        self.assertEqual(data['anomaly_count'], 20)

    def test_cnn_result_not_found(self):
        s2 = make_session(self.user, label='NoCNN')
        r  = self.client.get(f'/api/v1/sessions/{s2.pk}/cnn/')
        self.assertEqual(r.status_code, 404)


# ─── Alerts API ──────────────────────────────────────────────
class AlertAPITest(TestCase):
    def setUp(self):
        self.user    = make_user('alertapi')
        self.client  = auth_client(self.user)
        self.session = make_session(self.user)
        Alert.objects.create(session=self.session, attack_type='SYN Flood',
                              severity='CRITICAL', src_ip='1.2.3.4')
        Alert.objects.create(session=self.session, attack_type='Port Scan',
                              severity='MEDIUM',   src_ip='5.6.7.8')

    def test_list_alerts(self):
        r = self.client.get(f'/api/v1/sessions/{self.session.pk}/alerts/')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertEqual(len(data), 2)

    def test_filter_alerts_by_severity(self):
        r = self.client.get(
            f'/api/v1/sessions/{self.session.pk}/alerts/?severity=CRITICAL')
        data = json.loads(r.content)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['severity'], 'CRITICAL')


# ─── GradCAM API ─────────────────────────────────────────────
class GradCAMAPITest(TestCase):
    def setUp(self):
        self.user    = make_user('gcamapi')
        self.client  = auth_client(self.user)
        self.session = make_session(self.user)

    def test_list_gradcam_empty(self):
        r = self.client.get(f'/api/v1/sessions/{self.session.pk}/gradcam/')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertEqual(data, [])

    def test_gradcam_run_blocked_without_permission(self):
        self.user.profile.can_run_ai_analysis = False
        self.user.profile.save()
        r = self.client.post(
            f'/api/v1/sessions/{self.session.pk}/gradcam/run/',
            {'variant': 'gradcam'}, format='json'
        )
        self.assertEqual(r.status_code, 403)


# ─── Report API ──────────────────────────────────────────────
class ReportAPITest(TestCase):
    def setUp(self):
        self.user    = make_user('reportapi')
        self.client  = auth_client(self.user)
        self.session = make_session(self.user)

    def test_list_reports_empty(self):
        r = self.client.get(f'/api/v1/sessions/{self.session.pk}/reports/')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertEqual(data, [])

    def test_export_blocked_without_permission(self):
        self.user.profile.can_export_report = False
        self.user.profile.save()
        r = self.client.post(
            f'/api/v1/sessions/{self.session.pk}/reports/export/',
            {'format': 'json'}, format='json'
        )
        self.assertEqual(r.status_code, 403)


# ─── System Stats API (Admin Only) ───────────────────────────
class SystemStatsAPITest(TestCase):
    def setUp(self):
        self.admin  = make_user('statsadmin')
        self.admin.is_staff = True
        self.admin.is_superuser = True
        self.admin.save()
        self.normal = make_user('normuser')

    def test_admin_can_access_stats(self):
        client = auth_client(self.admin)
        r = client.get('/api/v1/admin/stats/')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertIn('users', data)
        self.assertIn('sessions', data)
        self.assertIn('alerts', data)

    def test_non_admin_blocked(self):
        client = auth_client(self.normal)
        r = client.get('/api/v1/admin/stats/')
        self.assertEqual(r.status_code, 403)
