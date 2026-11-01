# -*- coding: utf-8 -*-
"""針對 VNAAP 模型選擇、進度欄位與模擬 API 的回歸測試。

執行：python manage.py test analyzer.tests -v 2
需要實際 .pt 與 PCAP 依賴的端到端測試，請另按 README 的驗收步驟執行。
"""
import json
import os
import tempfile
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase, SimpleTestCase, override_settings

from analyzer.models import AnalysisSession, GradCAMImage
from analyzer.views import _available_models


EXPECTED_MODEL_KEYS = {
    'unsup_cicids2017',
    'semi_cicids2017',
    'mta_malware',
    'semi_mta_malware',
}
REMOVED_MODEL_KEYS = {'unsupervised_vae', 'semi_cicddos2019'}


class ModelConfigurationRegressionTests(SimpleTestCase):
    def test_four_expected_models_are_configured(self):
        enabled = {
            key for key, cfg in settings.ANOMALY_MODELS.items()
            if cfg.get('enabled', True)
        }
        self.assertEqual(enabled, EXPECTED_MODEL_KEYS)
        self.assertFalse(REMOVED_MODEL_KEYS.intersection(settings.ANOMALY_MODELS))

    def test_session_default_model_key_is_current(self):
        self.assertEqual(AnalysisSession._meta.get_field('model_key').default,
                         'unsup_cicids2017')

    def test_progress_fields_exist(self):
        names = {field.name for field in AnalysisSession._meta.get_fields()}
        self.assertTrue({
            'progress_percent', 'progress_stage', 'progress_detail',
            'processed_samples', 'total_samples',
        }.issubset(names))

    def test_three_cam_variants_are_supported(self):
        values = {value for value, _label in GradCAMImage.VARIANT_CHOICES}
        self.assertEqual(values, {'gradcam', 'gradcam++', 'scorecam'})

    def test_available_model_selector_has_no_removed_options(self):
        keys = {item['key'] for item in _available_models()}
        self.assertFalse(REMOVED_MODEL_KEYS.intersection(keys))
        self.assertTrue(EXPECTED_MODEL_KEYS.issubset(keys))


class SimulationApiRegressionTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        User = get_user_model()
        cls.user = User.objects.create_superuser(
            username='vnaap_repair_test', email='vnaap_repair_test@example.invalid',
            password='TestPass-Only-Local-123!'
        )

    def setUp(self):
        self.client.force_login(self.user)

    def test_missing_checkpoint_returns_clear_error_instead_of_normal(self):
        model_map = {
            key: dict(value) for key, value in settings.ANOMALY_MODELS.items()
        }
        model_map['unsup_cicids2017']['path'] = Path(tempfile.gettempdir()) / 'vnaap_missing_checkpoint.pt'
        with override_settings(ANOMALY_MODELS=model_map):
            response = self.client.post(
                '/analyzer/simulation/api/',
                data=json.dumps({'attack_type': 'port_scan', 'model_key': 'unsup_cicids2017'}),
                content_type='application/json',
            )
        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertFalse(payload.get('ok', True))
        self.assertIn('模型權重不存在', payload.get('error', ''))

    def test_invalid_attack_type_is_rejected_before_model_inference(self):
        response = self.client.post(
            '/analyzer/simulation/api/',
            data=json.dumps({'attack_type': 'not_a_supported_attack', 'model_key': 'unsup_cicids2017'}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json().get('ok', True))
