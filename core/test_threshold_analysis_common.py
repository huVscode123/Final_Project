# -*- coding: utf-8 -*-
"""threshold_analysis_common 的小型單元測試，不需要專題資料集或 GPU。"""
import unittest
import numpy as np
from threshold_analysis_common import (
    analyze_scalar_thresholds,
    analyze_hybrid_thresholds,
    split_four_way,
)


class ThresholdAnalysisTests(unittest.TestCase):
    def test_scalar_three_methods_share_test_set(self):
        rng = np.random.default_rng(17)
        cal_n = rng.normal(0, 1, 240)
        cal_a = rng.normal(1.8, 1, 160)
        test_n = rng.normal(0, 1, 120)
        test_a = rng.normal(1.8, 1, 80)
        result, cache = analyze_scalar_thresholds(
            "unit-test", cal_n, cal_a, test_n, test_a, percentile=95
        )
        self.assertEqual(set(result["methods"]), {"percentile", "optimal", "youden"})
        for method in result["methods"].values():
            self.assertEqual(method["test"]["n_normal"], len(test_n))
            self.assertEqual(method["test"]["n_attack"], len(test_a))
        self.assertEqual(len(cache["test_normal"]), len(test_n))

    def test_hybrid_metrics_are_final_hybrid_metrics(self):
        rng = np.random.default_rng(23)
        cal_n = rng.normal(0.2, 0.2, 120)
        cal_a = rng.normal(0.7, 0.3, 80)
        test_n = rng.normal(0.2, 0.2, 60)
        test_a = rng.normal(0.7, 0.3, 40)
        cal_n_p = rng.uniform(0.1, 0.8, len(cal_n))
        cal_a_p = rng.uniform(0.2, 0.95, len(cal_a))
        test_n_p = rng.uniform(0.1, 0.8, len(test_n))
        test_a_p = rng.uniform(0.2, 0.95, len(test_a))
        result, _ = analyze_hybrid_thresholds(
            "semi-unit-test", cal_n, cal_n_p, cal_a, cal_a_p,
            test_n, test_n_p, test_a, test_a_p,
            percentile=95, classifier_threshold=0.5,
        )
        self.assertEqual(set(result["methods"]), {"percentile", "optimal", "youden"})
        for method in result["methods"].values():
            m = method["test"]
            self.assertEqual(m["n_normal"], len(test_n))
            self.assertEqual(m["n_attack"], len(test_a))
            self.assertEqual(m["TN"] + m["FP"], len(test_n))
            self.assertEqual(m["TP"] + m["FN"], len(test_a))

    def test_four_way_split_ratio_and_reproducibility(self):
        x = np.arange(1000)
        s1 = split_four_way(x, 0.6, 0.2, 42)
        s2 = split_four_way(x, 0.6, 0.2, 42)
        self.assertEqual([len(a) for a in s1], [len(a) for a in s2])
        self.assertEqual([len(a) for a in s1], [600, 100, 100, 200])
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a, b)
        merged = np.concatenate(s1)
        self.assertEqual(len(np.unique(merged)), len(x))


if __name__ == "__main__":
    unittest.main(verbosity=2)
