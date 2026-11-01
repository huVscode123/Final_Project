# -*- coding: utf-8 -*-
"""修正版關鍵問題回歸測試。執行：python manage.py test analyzer.tests_root_causes -v 2"""
from django.conf import settings
from django.test import SimpleTestCase

from analyzer.models import AnalysisSession, GradCAMImage


class ModelAndExplainabilityRegressionTests(SimpleTestCase):
    def test_model_keys_are_dataset_specific(self):
        required = {
            'unsup_cicids2017', 'semi_cicids2017',
            'mta_malware', 'semi_mta_malware',
        }
        self.assertTrue(required.issubset(settings.ANOMALY_MODELS.keys()))
        self.assertNotIn('unsupervised_vae', settings.ANOMALY_MODELS)
        self.assertNotIn('semi_cicddos2019', settings.ANOMALY_MODELS)

    def test_progress_percent_is_bounded(self):
        field = AnalysisSession._meta.get_field('progress_percent')
        self.assertEqual(field.get_internal_type(), 'PositiveSmallIntegerField')

    def test_all_three_cam_variants_exist(self):
        self.assertEqual(
            {value for value, _label in GradCAMImage.VARIANT_CHOICES},
            {'gradcam', 'gradcam++', 'scorecam'},
        )
