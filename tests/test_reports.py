# ============================================================
# tests/test_reports.py
# 報告匯出系統測試：PDF / JSON / CSV 生成 + 下載
# ============================================================
import json
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User

from analyzer.models import AnalysisSession, Alert, CNNResult, AnalysisReport


def make_user(username, password='pass1234'):
    return User.objects.create_user(username, password=password)


def make_full_session(user):
    """建立含告警和 CNN 結果的完整 Session"""
    s = AnalysisSession.objects.create(
        mode='pcap', label='Full Report Session',
        task_status='done', created_by=user,
        packet_count=200, alert_count=5,
    )
    for i, sev in enumerate(['CRITICAL', 'HIGH', 'HIGH', 'MEDIUM', 'LOW']):
        Alert.objects.create(
            session=s, attack_type=f'Attack_{i}',
            severity=sev, src_ip=f'10.0.0.{i+1}',
            detail=f'Detail for attack {i}',
            suggestion=f'Suggestion for attack {i}',
        )
    CNNResult.objects.create(
        session=s, threshold=0.000822,
        normal_count=180, anomaly_count=20,
        avg_normal_error=0.0003,
        avg_attack_error=0.0028,
        detection_rate=0.10,
    )
    return s


class ReportGeneratorTest(TestCase):
    def setUp(self):
        self.user    = make_user('reportgen')
        self.session = make_full_session(self.user)

    def test_json_report_generated(self):
        from reports.generators import ReportGenerator
        gen    = ReportGenerator(self.session, self.user)
        report = gen.generate('json')
        self.assertIsInstance(report, AnalysisReport)
        self.assertEqual(report.format, 'json')
        self.assertGreater(report.file_size, 0)

    def test_json_report_content(self):
        from reports.generators import ReportGenerator
        gen    = ReportGenerator(self.session, self.user)
        report = gen.generate('json')
        with open(report.file.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(data['meta']['session_id'], self.session.pk)
        self.assertEqual(data['meta']['label'], 'Full Report Session')
        self.assertEqual(len(data['alerts']), 5)
        self.assertIsNotNone(data['cnn_result'])
        self.assertEqual(data['cnn_result']['anomaly_count'], 20)

    def test_csv_report_generated(self):
        from reports.generators import ReportGenerator
        gen    = ReportGenerator(self.session, self.user)
        report = gen.generate('csv')
        self.assertEqual(report.format, 'csv')
        with open(report.file.path, 'rb') as f:
            content = f.read().decode('utf-8-sig')
        self.assertIn('Attack_0', content)
        self.assertIn('CRITICAL', content)
        self.assertIn('0.000822', content)

    def test_pdf_report_generated(self):
        try:
            from reports.generators import ReportGenerator
            gen    = ReportGenerator(self.session, self.user)
            report = gen.generate('pdf')
            self.assertEqual(report.format, 'pdf')
            self.assertGreater(report.file_size, 0)
            # PDF magic bytes
            with open(report.file.path, 'rb') as f:
                header = f.read(4)
            self.assertEqual(header, b'%PDF')
        except ImportError:
            self.skipTest('reportlab not installed')

    def test_invalid_format_raises(self):
        from reports.generators import ReportGenerator
        gen = ReportGenerator(self.session, self.user)
        with self.assertRaises(ValueError):
            gen.generate('xlsx')

    def test_report_linked_to_session(self):
        from reports.generators import ReportGenerator
        gen    = ReportGenerator(self.session, self.user)
        report = gen.generate('json')
        self.assertEqual(report.session.pk, self.session.pk)
        self.assertEqual(report.generated_by.pk, self.user.pk)

    def test_multiple_formats_independent(self):
        from reports.generators import ReportGenerator
        gen = ReportGenerator(self.session, self.user)
        r1  = gen.generate('json')
        r2  = gen.generate('csv')
        self.assertNotEqual(r1.pk, r2.pk)
        self.assertEqual(AnalysisReport.objects.filter(
            session=self.session).count(), 2)


class ReportViewTest(TestCase):
    def setUp(self):
        self.client  = Client()
        self.user    = make_user('reportview')
        self.session = make_full_session(self.user)
        self.client.login(username='reportview', password='pass1234')

    def test_export_page_redirects(self):
        r = self.client.post(
            reverse('reports:export', args=[self.session.pk]),
            {'format': 'json'}
        )
        self.assertIn(r.status_code, [200, 301, 302])

    def test_export_creates_report_record(self):
        before = AnalysisReport.objects.count()
        self.client.post(
            reverse('reports:export', args=[self.session.pk]),
            {'format': 'json'}
        )
        self.assertGreater(AnalysisReport.objects.count(), before)

    def test_download_report(self):
        from reports.generators import ReportGenerator
        gen    = ReportGenerator(self.session, self.user)
        report = gen.generate('json')
        r = self.client.get(reverse('reports:download', args=[report.pk]))
        self.assertEqual(r.status_code, 200)
        self.assertIn('application/json', r['Content-Type'])

    def test_download_csv_headers(self):
        from reports.generators import ReportGenerator
        gen    = ReportGenerator(self.session, self.user)
        report = gen.generate('csv')
        r = self.client.get(reverse('reports:download', args=[report.pk]))
        self.assertEqual(r.status_code, 200)
        self.assertIn('csv', r['Content-Type'])
        self.assertIn('attachment', r.get('Content-Disposition', ''))

    def test_stranger_cannot_download(self):
        from reports.generators import ReportGenerator
        gen    = ReportGenerator(self.session, self.user)
        report = gen.generate('json')
        stranger = make_user('stranger')
        self.client.login(username='stranger', password='pass1234')
        r = self.client.get(reverse('reports:download', args=[report.pk]))
        self.assertEqual(r.status_code, 404)

    def test_report_list_page_loads(self):
        r = self.client.get(reverse('reports:list', args=[self.session.pk]))
        self.assertEqual(r.status_code, 200)

    def test_export_blocked_without_permission(self):
        self.user.profile.can_export_report = False
        self.user.profile.save()
        r = self.client.post(
            reverse('reports:export', args=[self.session.pk]),
            {'format': 'json'}
        )
        self.assertIn(r.status_code, [200, 301, 302])
        # 不應建立新報告
        self.assertEqual(AnalysisReport.objects.count(), 0)
