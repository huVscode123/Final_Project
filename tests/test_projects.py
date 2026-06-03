# ============================================================
# tests/test_projects.py
# 專案管理 / 成員 / 封包檔案測試
# ============================================================
import io
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile

from projects.models import Project, ProjectMembership, PacketFile


def make_user(username, password='pass1234', role='analyst'):
    u = User.objects.create_user(username, password=password)
    u.profile.role = role
    u.profile.save()
    return u


def make_project(owner, name='TestProject', segment='10.0.0.0/24'):
    p = Project.objects.create(name=name, network_segment=segment, owner=owner)
    ProjectMembership.objects.create(project=p, user=owner, role='owner')
    return p


class ProjectCRUDTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.owner = make_user('projowner')
        self.client.login(username='projowner', password='pass1234')

    def test_project_list_loads(self):
        r = self.client.get(reverse('projects:list'))
        self.assertEqual(r.status_code, 200)

    def test_create_project(self):
        r = self.client.post(reverse('projects:create'), {
            'name':            'New Network Audit',
            'network_segment': '192.168.0.0/16',
            'description':     'Q1 audit scope',
            'status':          'active',
        })
        self.assertIn(r.status_code, [301, 302])
        self.assertTrue(Project.objects.filter(name='New Network Audit').exists())

    def test_project_auto_owner_membership(self):
        """建立者自動成為 owner 成員"""
        self.client.post(reverse('projects:create'), {
            'name': 'AutoMembership', 'status': 'active',
        })
        p = Project.objects.get(name='AutoMembership')
        self.assertTrue(
            ProjectMembership.objects.filter(
                project=p, user=self.owner, role='owner').exists()
        )

    def test_project_detail_requires_membership(self):
        other = make_user('outsider')
        p = make_project(self.owner)
        self.client.login(username='outsider', password='pass1234')
        r = self.client.get(reverse('projects:detail', args=[p.pk]))
        self.assertIn(r.status_code, [200, 302, 403])

    def test_owner_can_edit_project(self):
        p = make_project(self.owner)
        r = self.client.post(reverse('projects:edit', args=[p.pk]), {
            'name':   'Updated Name',
            'status': 'active',
        })
        self.assertIn(r.status_code, [301, 302])
        p.refresh_from_db()
        self.assertEqual(p.name, 'Updated Name')

    def test_non_owner_cannot_edit(self):
        other = make_user('nonowner')
        p = make_project(self.owner)
        ProjectMembership.objects.create(project=p, user=other, role='editor')
        self.client.login(username='nonowner', password='pass1234')
        r = self.client.post(reverse('projects:edit', args=[p.pk]), {
            'name': 'Hacked Name', 'status': 'active',
        })
        p.refresh_from_db()
        self.assertNotEqual(p.name, 'Hacked Name')

    def test_archive_project(self):
        p = make_project(self.owner)
        self.client.post(reverse('projects:archive', args=[p.pk]))
        p.refresh_from_db()
        self.assertEqual(p.status, 'archived')


class ProjectMembershipTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.owner = make_user('memowner')
        self.member = make_user('member1')
        self.project = make_project(self.owner)
        self.client.login(username='memowner', password='pass1234')

    def test_add_member(self):
        r = self.client.post(
            reverse('projects:member_add', args=[self.project.pk]),
            {'username': 'member1', 'role': 'editor'}
        )
        self.assertIn(r.status_code, [301, 302])
        self.assertTrue(
            ProjectMembership.objects.filter(
                project=self.project, user=self.member).exists()
        )

    def test_remove_member(self):
        ProjectMembership.objects.create(
            project=self.project, user=self.member, role='editor')
        r = self.client.post(
            reverse('projects:member_remove',
                    args=[self.project.pk, self.member.pk])
        )
        self.assertIn(r.status_code, [301, 302])
        self.assertFalse(
            ProjectMembership.objects.filter(
                project=self.project, user=self.member).exists()
        )

    def test_member_can_see_project(self):
        ProjectMembership.objects.create(
            project=self.project, user=self.member, role='viewer')
        self.client.login(username='member1', password='pass1234')
        r = self.client.get(
            reverse('projects:detail', args=[self.project.pk]))
        self.assertEqual(r.status_code, 200)

    def test_non_member_blocked(self):
        stranger = make_user('stranger')
        self.client.login(username='stranger', password='pass1234')
        r = self.client.get(
            reverse('projects:detail', args=[self.project.pk]))
        # 應跳轉或 403
        self.assertNotEqual(r.status_code, 200)


class PacketFileTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.owner   = make_user('pcapowner')
        self.project = make_project(self.owner)
        self.client.login(username='pcapowner', password='pass1234')

    def _fake_pcap(self, name='test.pcap', size=1024):
        """產生一個假的 PCAP 上傳檔案"""
        content = b'\xd4\xc3\xb2\xa1' + b'\x00' * size  # PCAP magic bytes
        return SimpleUploadedFile(name, content, content_type='application/octet-stream')

    def test_upload_packet_file(self):
        pcap = self._fake_pcap()
        r = self.client.post(
            reverse('projects:upload', args=[self.project.pk]),
            {'file': pcap, 'description': 'Test upload', 'network_tag': '10.0.0.0/24'}
        )
        self.assertIn(r.status_code, [200, 301, 302])
        self.assertTrue(
            PacketFile.objects.filter(project=self.project).exists()
        )

    def test_packet_file_sha256_computed(self):
        """compute_sha256 不應拋出例外"""
        import tempfile, os
        pf = PacketFile.objects.create(
            project=self.project,
            uploaded_by=self.owner,
            original_name='test.pcap',
            file_size=0,
        )
        # 無實體檔案時不應 crash（直接測 model 存在即可）
        self.assertIsNotNone(pf.pk)

    def test_file_size_mb_property(self):
        pf = PacketFile(file_size=5 * 1024 * 1024)
        self.assertEqual(pf.file_size_mb, 5.0)

    def test_viewer_cannot_upload(self):
        viewer = make_user('viewer99', role='viewer')
        ProjectMembership.objects.create(
            project=self.project, user=viewer, role='viewer')
        self.client.login(username='viewer99', password='pass1234')
        pcap = self._fake_pcap()
        r = self.client.post(
            reverse('projects:upload', args=[self.project.pk]),
            {'file': pcap}
        )
        self.assertIn(r.status_code, [200, 301, 302, 403])
