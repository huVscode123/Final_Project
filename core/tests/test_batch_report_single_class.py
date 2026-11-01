import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

_CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)
import batch_pcap_evaluate as batch  # noqa: E402


class TestSingleClassReport(unittest.TestCase):
    def test_benign_only_report_contains_fpr_and_disclaimer(self):
        metrics = {
            "overall": {"n": 10, "n_anomaly": 3, "anomaly_rate": 0.3,
                        "score_quantiles": {"p50": 0.0004, "p90": 0.001, "p95": 0.002,
                                             "p99": 0.003, "p99.9": 0.004}},
            "label_distribution": {"BENIGN": 10},
            "benign_only_or_benign_subset": {
                "n_benign": 10, "false_positive": 3, "true_negative": 7,
                "fpr_at_model_threshold": 0.3, "fpr_ci95": (0.1078, 0.6032)},
            "threshold_used": 0.00082,
        }
        pf = pd.DataFrame([{
            "file": "Monday", "n": 10, "anomaly_rate": .3, "rate_ci_lo": .1,
            "rate_ci_hi": .6, "score_p50": .0004, "score_p95": .002, "score_p99": .003,
            "ks_vs_ref": 0.0,
        }])
        pp = pd.DataFrame([{"proto": 6, "svc": 443, "n": 10, "a": 3, "rate": .3}])
        sp = pd.DataFrame([{"src": "10.0.0.1", "n": 10, "a": 3, "rate": .3}])
        with tempfile.TemporaryDirectory() as tmp:
            batch._write_report(tmp, SimpleNamespace(threshold=None), {"model": "test", "model_type": "VAE", "mode": "flow", "model_threshold": .00082},
                                metrics, pf, [], [], pp, sp, False)
            with open(os.path.join(tmp, "report.md"), encoding="utf-8") as report_file:
                text = report_file.read()
        self.assertIn("單一類別標籤（只有 BENIGN）", text)
        self.assertIn("FPR@threshold：30.0000%", text)
        self.assertIn("不計算 Precision、Recall、F1、ROC-AUC 或 PR-AUC", text)


if __name__ == "__main__":
    unittest.main()
