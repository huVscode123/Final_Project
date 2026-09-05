# ============================================================
# tests/test_analyzer.py
# 分析 Session / CNN 結果 / GradCAM / 告警 / 視圖測試
# ============================================================
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone

from projects.models import Project, ProjectMembership
from analyzer.models import (AnalysisSession, Alert, CNNResult,
                              GradCAMImage, AnalysisReport)


def make_user(username, password='pass1234', staff=False):
    u = User.objects.create_user(username, password=password)
    if staff:
        u.is_staff = True
        u.save()
    return u


def make_session(user, status='done', label='TestSession', project=None):
    s = AnalysisSession.objects.create(
        mode='pcap', label=label,
        task_status=status,
        created_by=user,
        project=project,
        packet_count=100,
        alert_count=3,
    )
    return s


def make_alert(session, severity='HIGH'):
    return Alert.objects.create(
        session=session,
        attack_type='Port Scan',
        severity=severity,
        src_ip='192.168.1.100',
        dst_ip='10.0.0.1',
        dst_port=22,
        detail='Rapid connection attempts',
        suggestion='Block source IP',
    )


def make_cnn_result(session):
    return CNNResult.objects.create(
        session=session,
        threshold=0.000822,
        normal_count=90,
        anomaly_count=10,
        avg_normal_error=0.0003,
        avg_attack_error=0.0025,
        detection_rate=0.10,
    )


# ─── Model Tests ────────────────────────────────────────────
class AnalysisSessionModelTest(TestCase):
    def setUp(self):
        self.user = make_user('modeluser')

    def test_session_str(self):
        s = make_session(self.user, label='Scan Test')
        self.assertIn('Scan Test', str(s))

    def test_session_default_status_pending(self):
        s = AnalysisSession.objects.create(
            mode='pcap', label='New', created_by=self.user)
        self.assertEqual(s.task_status, 'pending')

    def test_alert_severity_color(self):
        s = make_session(self.user)
        a = make_alert(s, 'CRITICAL')
        self.assertIn(a.severity_color, ['#f85149', '#ff7b72'])
        a2 = make_alert(s, 'LOW')
        self.assertEqual(a2.severity_color, '#3fb950')

    def test_cnn_result_linked_to_session(self):
        s = make_session(self.user)
        r = make_cnn_result(s)
        self.assertEqual(r.session.pk, s.pk)
        self.assertEqual(s.cnn_result.pk, r.pk)

    def test_cnn_detection_rate_stored(self):
        s = make_session(self.user)
        r = make_cnn_result(s)
        self.assertAlmostEqual(r.detection_rate, 0.10)


# ─── Dashboard View ──────────────────────────────────────────
class DashboardViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = make_user('dashuser')
        self.client.login(username='dashuser', password='pass1234')

    def test_dashboard_loads(self):
        r = self.client.get(reverse('analyzer:dashboard'))
        self.assertEqual(r.status_code, 200)

    def test_dashboard_contains_kpi_context(self):
        make_session(self.user, status='done')
        r = self.client.get(reverse('analyzer:dashboard'))
        self.assertIn('total_sessions', r.context)
        self.assertIn('total_alerts', r.context)
        self.assertIn('severity_data', r.context)
        self.assertIn('daily_data', r.context)

    def test_dashboard_severity_data_is_valid_json(self):
        r = self.client.get(reverse('analyzer:dashboard'))
        sev = json.loads(r.context['severity_data'])
        self.assertIn('CRITICAL', sev)
        self.assertIn('HIGH', sev)
        self.assertIn('MEDIUM', sev)
        self.assertIn('LOW', sev)

    def test_dashboard_requires_login(self):
        self.client.logout()
        r = self.client.get(reverse('analyzer:dashboard'))
        self.assertIn(r.status_code, [301, 302])


# ─── Upload View ─────────────────────────────────────────────
class UploadViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = make_user('uploader')
        self.client.login(username='uploader', password='pass1234')

    def test_upload_page_loads(self):
        r = self.client.get(reverse('analyzer:upload'))
        self.assertEqual(r.status_code, 200)
        self.assertIn('pipeline_steps', r.context)
        self.assertEqual(len(r.context['pipeline_steps']), 5)

    def test_upload_without_file_shows_error(self):
        r = self.client.post(reverse('analyzer:upload'), {'label': 'NoFile'})
        self.assertEqual(r.status_code, 200)

    def test_upload_creates_session(self):
        content = b'\xd4\xc3\xb2\xa1' + b'\x00' * 512
        pcap = SimpleUploadedFile('test.pcap', content,
                                   content_type='application/octet-stream')
        before = AnalysisSession.objects.count()
        self.client.post(reverse('analyzer:upload'), {
            'pcap_file': pcap, 'label': 'Unit Test Upload',
        })
        self.assertGreater(AnalysisSession.objects.count(), before)

    def test_upload_requires_login(self):
        self.client.logout()
        r = self.client.get(reverse('analyzer:upload'))
        self.assertIn(r.status_code, [301, 302])


# ─── Simulation View ───────────────────────────────────────
class SimulationViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = make_user('simuser')
        self.client.login(username='simuser', password='pass1234')

    def test_simulation_creates_downloadable_pcap_and_backup(self):
        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            with self.settings(
                CNN_MODEL_PATH='',
                MEDIA_ROOT=temp_root / 'media',
                BASE_DIR=temp_root,
            ):
                response = self.client.post(
                    reverse('analyzer:simulation_api'),
                    data=json.dumps({'attack_type': 'syn_flood', 'packet_count': 5}),
                    content_type='application/json',
                )

                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertTrue(payload['ok'])
                self.assertTrue(payload['pcap_backup_saved'])
                self.assertIn('pcap_download_url', payload)
                self.assertTrue((
                    temp_root / 'output' / 'simulation_backups'
                    / f'user_{self.user.pk}' / payload['pcap_filename']
                ).is_file())

                download = self.client.get(payload['pcap_download_url'])
                self.assertEqual(download.status_code, 200)
                self.assertTrue(download['Content-Disposition'].startswith('attachment;'))


# ─── Session List ────────────────────────────────────────────
class SessionListViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = make_user('listuser')
        self.client.login(username='listuser', password='pass1234')

    def test_sessions_page_loads(self):
        r = self.client.get(reverse('analyzer:sessions'))
        self.assertEqual(r.status_code, 200)

    def test_sessions_filter_by_status(self):
        make_session(self.user, status='done',  label='Done1')
        make_session(self.user, status='failed', label='Fail1')
        r = self.client.get(reverse('analyzer:sessions') + '?status=done')
        sessions = list(r.context['sessions'])
        self.assertTrue(all(s.task_status == 'done' for s in sessions))

    def test_sessions_filter_by_query(self):
        make_session(self.user, label='SynFlood2024')
        make_session(self.user, label='NormalTraffic')
        r = self.client.get(reverse('analyzer:sessions') + '?q=SynFlood')
        sessions = list(r.context['sessions'])
        self.assertTrue(all('SynFlood' in s.label for s in sessions))

    def test_user_cannot_see_others_sessions(self):
        other = make_user('other99')
        make_session(other, label='Private Session')
        r = self.client.get(reverse('analyzer:sessions'))
        labels = [s.label for s in r.context['sessions']]
        self.assertNotIn('Private Session', labels)

    def test_admin_sees_all_sessions(self):
        admin = make_user('admin99', staff=True)
        other = make_user('randuser')
        make_session(other, label='OtherUserSession')
        self.client.login(username='admin99', password='pass1234')
        r = self.client.get(reverse('analyzer:sessions'))
        labels = [s.label for s in r.context['sessions']]
        self.assertIn('OtherUserSession', labels)


# ─── Session Detail ──────────────────────────────────────────
class SessionDetailViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = make_user('detailuser')
        self.session = make_session(self.user, status='done')
        self.alert   = make_alert(self.session)
        self.cnn     = make_cnn_result(self.session)
        self.client.login(username='detailuser', password='pass1234')

    def test_detail_page_loads(self):
        r = self.client.get(
            reverse('analyzer:session_detail', args=[self.session.pk]))
        self.assertEqual(r.status_code, 200)

    def test_detail_shows_alerts(self):
        r = self.client.get(
            reverse('analyzer:session_detail', args=[self.session.pk]))
        self.assertIn('alerts', r.context)
        self.assertEqual(r.context['alerts'].count(), 1)

    def test_detail_shows_cnn_result(self):
        r = self.client.get(
            reverse('analyzer:session_detail', args=[self.session.pk]))
        self.assertIsNotNone(r.context['cnn_result'])
        self.assertEqual(r.context['cnn_result'].pk, self.cnn.pk)

    def test_detail_sev_data_json(self):
        r = self.client.get(
            reverse('analyzer:session_detail', args=[self.session.pk]))
        sev = json.loads(r.context['sev_data'])
        self.assertEqual(sev['HIGH'], 1)
        self.assertEqual(sev['CRITICAL'], 0)

    def test_stranger_cannot_view_detail(self):
        stranger = make_user('stranger')
        self.client.login(username='stranger', password='pass1234')
        r = self.client.get(
            reverse('analyzer:session_detail', args=[self.session.pk]))
        self.assertIn(r.status_code, [302, 403, 404])

    def test_project_member_can_view_detail(self):
        p = Project.objects.create(name='SharedProj', owner=self.user)
        member = make_user('projmember')
        ProjectMembership.objects.create(project=p, user=member, role='viewer')
        s2 = make_session(self.user, project=p)
        self.client.login(username='projmember', password='pass1234')
        r = self.client.get(reverse('analyzer:session_detail', args=[s2.pk]))
        self.assertEqual(r.status_code, 200)


# ─── Status API ──────────────────────────────────────────────
class SessionStatusAPITest(TestCase):
    def setUp(self):
        self.client  = Client()
        self.user    = make_user('statususer')
        self.session = make_session(self.user, status='running')
        self.client.login(username='statususer', password='pass1234')

    def test_status_returns_json(self):
        r = self.client.get(
            reverse('analyzer:session_status', args=[self.session.pk]))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r['Content-Type'], 'application/json')

    def test_status_json_fields(self):
        r = self.client.get(
            reverse('analyzer:session_status', args=[self.session.pk]))
        data = json.loads(r.content)
        for field in ('task_status', 'packet_count', 'alert_count',
                      'error_message', 'has_cnn', 'gradcam_count'):
            self.assertIn(field, data)

    def test_status_reflects_actual_status(self):
        r = self.client.get(
            reverse('analyzer:session_status', args=[self.session.pk]))
        data = json.loads(r.content)
        self.assertEqual(data['task_status'], 'running')


# ─── GradCAM Trigger ─────────────────────────────────────────
class GradCAMTriggerTest(TestCase):
    def setUp(self):
        self.client  = Client()
        self.user    = make_user('gcamuser')
        self.session = make_session(self.user, status='done')
        self.client.login(username='gcamuser', password='pass1234')

    def test_gradcam_post_only(self):
        r = self.client.get(
            reverse('analyzer:trigger_gradcam', args=[self.session.pk]))
        self.assertEqual(r.status_code, 405)

    def test_gradcam_trigger_returns_json(self):
        r = self.client.post(
            reverse('analyzer:trigger_gradcam', args=[self.session.pk]),
            {'variant': 'gradcam', 'max_images': '5', 'anomaly_only': 'true'}
        )
        self.assertEqual(r['Content-Type'], 'application/json')

    def test_gradcam_blocked_for_no_ai_permission(self):
        self.user.profile.can_run_ai_analysis = False
        self.user.profile.save()
        r = self.client.post(
            reverse('analyzer:trigger_gradcam', args=[self.session.pk]),
            {'variant': 'gradcam'}
        )
        data = json.loads(r.content)
        self.assertIn('error', data)


# ─── Session Delete ──────────────────────────────────────────
class SessionDeleteTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user   = make_user('deluser')
        self.client.login(username='deluser', password='pass1234')

    def test_owner_can_delete(self):
        s = make_session(self.user)
        pk = s.pk
        self.client.post(reverse('analyzer:session_delete', args=[pk]))
        self.assertFalse(AnalysisSession.objects.filter(pk=pk).exists())

    def test_non_owner_cannot_delete(self):
        owner   = make_user('realowner')
        session = make_session(owner)
        pk = session.pk
        self.client.post(reverse('analyzer:session_delete', args=[pk]))
        # session should still exist
        self.assertTrue(AnalysisSession.objects.filter(pk=pk).exists())


# ─── Live Monitor ────────────────────────────────────────────
class LiveMonitorTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user   = make_user('liveuser')
        self.client.login(username='liveuser', password='pass1234')

    def test_live_page_loads(self):
        r = self.client.get(reverse('analyzer:live'))
        self.assertEqual(r.status_code, 200)

    def test_live_shows_running_sessions(self):
        make_session(self.user, status='running', label='RunningNow')
        make_session(self.user, status='done',    label='Finished')
        r = self.client.get(reverse('analyzer:live'))
        active = list(r.context['active_sessions'])
        labels = [s.label for s in active]
        self.assertIn('RunningNow', labels)
        self.assertNotIn('Finished', labels)

    def test_live_shows_recent_done(self):
        make_session(self.user, status='done', label='RecentDone')
        r = self.client.get(reverse('analyzer:live'))
        done = list(r.context['recent_done'])
        labels = [s.label for s in done]
        self.assertIn('RecentDone', labels)


# ─── AI Chat ─────────────────────────────────────────────────
class AIChatViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user   = make_user('chatuser')
        self.client.login(username='chatuser', password='pass1234')

    def test_ai_chat_page_loads(self):
        r = self.client.get(reverse('analyzer:ai_chat'))
        self.assertEqual(r.status_code, 200)
        self.assertIn('sessions', r.context)

    def test_ai_chat_only_shows_done_sessions(self):
        make_session(self.user, status='done',    label='DoneSession')
        make_session(self.user, status='running', label='StillRunning')
        r = self.client.get(reverse('analyzer:ai_chat'))
        labels = [s.label for s in r.context['sessions']]
        self.assertIn('DoneSession', labels)
        self.assertNotIn('StillRunning', labels)
