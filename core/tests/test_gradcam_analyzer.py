# ============================================================
# core/tests/test_gradcam_analyzer.py  - 分析器與報告匯出測試
# ============================================================
"""
測試範圍：
  GradCAMAnalyzer：
    - _compute_recon_scores：輸出形狀、非負值
    - get_reconstructions：輸出形狀與輸入一致
    - analyze：AnalysisReport 結構完整性
    - get_top_anomalies：排序正確、數量限制
    - summarize：必要欄位存在、型別正確
    - plot_feature_importance：產生圖檔
    - plot_mean_cam_comparison：產生圖檔
    - plot_cam_intensity_distribution：產生圖檔

  GradCAMReporter：
    - save_json：JSON 可解析、結構正確
    - save_html：HTML 包含關鍵元素
    - export_all：同時產生兩個檔案
"""

import sys
import os
import json
import unittest
import tempfile
import numpy as np

_CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

# ── 視覺化輸出目錄：output/gradcam_test_output/analyzer/ ──────
# （相對於 final_project/，即 core/ 上兩層）
_OUTPUT_DIR = os.path.normpath(
    os.path.join(_CORE, "..", "output", "gradcam_test_output", "analyzer")
)

from cnn_gradcam import GradCAM, GradCAMAnalyzer, GradCAMReporter
from cnn_gradcam.gradcam_analyzer import AnalysisReport, SampleResult
from tests.gradcam_fixtures import make_model, FakeDataFactory


def _make_fake_report(n_normal=10, n_attack=10, h=32, w=32):
    """建立完整的假 AnalysisReport，不依賴真實 Grad-CAM 執行。"""
    rng = np.random.RandomState(0)

    def _make_results(label, n, score_base):
        results = []
        for i in range(n):
            hm = rng.rand(h, w).astype(np.float32)
            hm /= (hm.max() + 1e-9)
            results.append(SampleResult(
                sample_idx   = i,
                label        = label,
                recon_score  = float(score_base + rng.rand() * 0.01),
                heatmap      = hm,
                cam_mean     = float(hm.mean()),
                cam_max      = float(hm.max()),
                top_k_pixels = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
            ))
        return results

    normal_r = _make_results("Normal", n_normal, 0.001)
    attack_r = _make_results("Attack", n_attack, 0.05)
    normal_cams = np.stack([r.heatmap for r in normal_r])
    attack_cams = np.stack([r.heatmap for r in attack_r])

    feat_names = [f"feature_{i}" for i in range(h)]
    diff_cam   = attack_cams.mean(0) - normal_cams.mean(0)
    feat_imp   = diff_cam.mean(axis=1)

    return AnalysisReport(
        normal_results    = normal_r,
        attack_results    = attack_r,
        normal_mean_cam   = normal_cams.mean(0),
        attack_mean_cam   = attack_cams.mean(0),
        diff_cam          = diff_cam,
        feature_importance= feat_imp,
        feature_names     = feat_names,
    )


# ════════════════════════════════════════════════════════════
class TestGradCAMAnalyzerScores(unittest.TestCase):
    """重建誤差與重建輸出的計算正確性。"""

    def setUp(self):
        self.model = make_model()
        cam = GradCAM(self.model, target_layer="encoder_conv")
        self.analyzer = GradCAMAnalyzer(cam, self.model, batch_size=4)

    def test_recon_scores_shape(self):
        X = FakeDataFactory.normal_batch(n=8)
        scores = self.analyzer._compute_recon_scores(X)
        self.assertEqual(scores.shape, (8,))

    def test_recon_scores_nonnegative(self):
        X = FakeDataFactory.normal_batch(n=8)
        scores = self.analyzer._compute_recon_scores(X)
        self.assertTrue((scores >= 0).all(), "MSE 誤差不可為負值")

    def test_attack_higher_recon_score(self):
        """
        對於未訓練的隨機模型，攻擊樣本（高值異常區塊）
        的重建誤差中位數應 >= 正常樣本（此為弱驗證）。
        """
        X_n = FakeDataFactory.normal_batch(n=32)
        X_a = FakeDataFactory.attack_batch(n=32)
        scores_n = self.analyzer._compute_recon_scores(X_n)
        scores_a = self.analyzer._compute_recon_scores(X_a)
        # 使用中位數避免極端值影響，允許小幅容差
        self.assertGreaterEqual(
            np.median(scores_a) + 1e-6,
            np.median(scores_n) * 0.5,  # 攻擊誤差至少是正常的 50%
            "攻擊樣本重建誤差中位數應不低於正常樣本的 50%"
        )

    def test_get_reconstructions_shape(self):
        X = FakeDataFactory.normal_batch(n=6)
        recs = self.analyzer.get_reconstructions(X)
        self.assertEqual(recs.shape, (6, 32, 32))

    def test_get_reconstructions_range(self):
        """Sigmoid 輸出保證在 [0, 1]。"""
        X = FakeDataFactory.normal_batch(n=4)
        recs = self.analyzer.get_reconstructions(X)
        self.assertGreaterEqual(recs.min(), -1e-6)
        self.assertLessEqual(recs.max(), 1.0 + 1e-6)


class TestGradCAMAnalyzerAnalyze(unittest.TestCase):
    """analyze() 回傳 AnalysisReport 結構完整性。"""

    def setUp(self):
        self.model = make_model()
        cam = GradCAM(self.model, target_layer="encoder_conv")
        self.analyzer = GradCAMAnalyzer(cam, self.model, batch_size=4)

    def test_report_has_correct_counts(self):
        X_n = FakeDataFactory.normal_batch(n=12)
        X_a = FakeDataFactory.attack_batch(n=12)
        report = self.analyzer.analyze(X_n, X_a, max_samples_each=10)
        self.assertEqual(len(report.normal_results), 10)
        self.assertEqual(len(report.attack_results), 10)

    def test_report_mean_cams_not_none(self):
        X_n = FakeDataFactory.normal_batch(n=8)
        X_a = FakeDataFactory.attack_batch(n=8)
        report = self.analyzer.analyze(X_n, X_a, max_samples_each=8)
        self.assertIsNotNone(report.normal_mean_cam)
        self.assertIsNotNone(report.attack_mean_cam)
        self.assertIsNotNone(report.diff_cam)

    def test_report_mean_cam_shapes(self):
        X_n = FakeDataFactory.normal_batch(n=8)
        X_a = FakeDataFactory.attack_batch(n=8)
        report = self.analyzer.analyze(X_n, X_a, max_samples_each=8)
        self.assertEqual(report.normal_mean_cam.shape, (32, 32))
        self.assertEqual(report.diff_cam.shape, (32, 32))

    def test_feature_importance_length_matches_height(self):
        """feature_importance 長度應等於影像高度 H（每行對應一個特徵）。"""
        X_n = FakeDataFactory.normal_batch(n=8)
        X_a = FakeDataFactory.attack_batch(n=8)
        report = self.analyzer.analyze(X_n, X_a, max_samples_each=8,
                                        field_names=[f"f{i}" for i in range(32)])
        self.assertEqual(len(report.feature_importance), 32)

    def test_each_sample_result_fields(self):
        X_n = FakeDataFactory.normal_batch(n=4)
        X_a = FakeDataFactory.attack_batch(n=4)
        report = self.analyzer.analyze(X_n, X_a, max_samples_each=4)
        for r in report.normal_results:
            self.assertEqual(r.label, "Normal")
            self.assertIsInstance(r.recon_score, float)
            self.assertEqual(r.heatmap.shape, (32, 32))
            self.assertGreaterEqual(r.cam_mean, 0.0)
            self.assertEqual(len(r.top_k_pixels), 5)


class TestGradCAMAnalyzerTopAnomalies(unittest.TestCase):
    def test_sorted_descending(self):
        """get_top_anomalies 應按 recon_score 降序排列。"""
        report = _make_fake_report()
        analyzer = GradCAMAnalyzer.__new__(GradCAMAnalyzer)  # 不呼叫 __init__
        top = analyzer.get_top_anomalies(report.attack_results, k=5)
        scores = [r.recon_score for r in top]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_count_limit(self):
        report = _make_fake_report(n_attack=20)
        analyzer = GradCAMAnalyzer.__new__(GradCAMAnalyzer)
        top = analyzer.get_top_anomalies(report.attack_results, k=7)
        self.assertEqual(len(top), 7)

    def test_only_attack_in_results(self):
        report = _make_fake_report()
        all_results = report.normal_results + report.attack_results
        analyzer = GradCAMAnalyzer.__new__(GradCAMAnalyzer)
        top = analyzer.get_top_anomalies(all_results, k=5)
        for r in top:
            self.assertEqual(r.label, "Attack")


class TestGradCAMAnalyzerSummarize(unittest.TestCase):
    def setUp(self):
        self.report = _make_fake_report()

    def _get_summarize(self):
        """直接呼叫 summarize（不需要 __init__）。"""
        return GradCAMAnalyzer.summarize(
            GradCAMAnalyzer.__new__(GradCAMAnalyzer),
            self.report
        )

    def test_required_keys_present(self):
        summary = self._get_summarize()
        self.assertIn("normal", summary)
        self.assertIn("attack", summary)

    def test_stats_fields(self):
        summary = self._get_summarize()
        for key in ["count", "recon_score_mean", "recon_score_std",
                    "cam_intensity_mean"]:
            self.assertIn(key, summary["normal"], f"normal 應有 '{key}' 欄位")
            self.assertIn(key, summary["attack"], f"attack 應有 '{key}' 欄位")

    def test_top5_features_present(self):
        summary = self._get_summarize()
        self.assertIn("top5_important_features", summary)
        self.assertEqual(len(summary["top5_important_features"]), 5)
        for feat in summary["top5_important_features"]:
            self.assertIn("name", feat)
            self.assertIn("importance", feat)

    def test_values_are_json_serializable(self):
        summary = self._get_summarize()
        try:
            import json
            json.dumps(summary)
        except TypeError as e:
            self.fail(f"summary 不可 JSON 序列化：{e}")


class TestGradCAMAnalyzerPlots(unittest.TestCase):
    """
    圖表產生測試：驗證圖檔存在並儲存至可查閱的目錄。

    輸出位置：output/gradcam_test_output/analyzer/
    """

    @classmethod
    def setUpClass(cls):
        """建立輸出目錄並共用 model / report（避免重複初始化）。"""
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        cls.model    = make_model()
        cam          = GradCAM(cls.model, target_layer="encoder_conv")
        cls.analyzer = GradCAMAnalyzer(cam, cls.model, batch_size=4)
        cls.report   = _make_fake_report()
        print(f"\n  [圖表輸出] {os.path.abspath(_OUTPUT_DIR)}")

    def test_plot_feature_importance_creates_file(self):
        self.analyzer.plot_feature_importance(self.report, _OUTPUT_DIR, top_k=10)
        self.assertTrue(
            os.path.exists(os.path.join(_OUTPUT_DIR, "feature_importance.png"))
        )

    def test_plot_mean_cam_comparison_creates_file(self):
        self.analyzer.plot_mean_cam_comparison(self.report, _OUTPUT_DIR)
        self.assertTrue(
            os.path.exists(os.path.join(_OUTPUT_DIR, "mean_cam_comparison.png"))
        )

    def test_plot_cam_intensity_distribution_creates_file(self):
        self.analyzer.plot_cam_intensity_distribution(self.report, _OUTPUT_DIR)
        self.assertTrue(
            os.path.exists(os.path.join(_OUTPUT_DIR, "cam_intensity_distribution.png"))
        )


# ════════════════════════════════════════════════════════════
class TestGradCAMReporterJSON(unittest.TestCase):
    def setUp(self):
        self.report  = _make_fake_report()
        self.analyzer = GradCAMAnalyzer.__new__(GradCAMAnalyzer)
        self.summary  = self.analyzer.summarize(self.report)

    def test_json_is_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = GradCAMReporter(output_dir=tmpdir)
            path = reporter.save_json(self.report, self.summary)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)  # 應不拋出 JSONDecodeError
            self.assertIsInstance(data, dict)

    def test_json_has_required_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = GradCAMReporter(output_dir=tmpdir)
            path = reporter.save_json(self.report, self.summary)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for key in ["meta", "summary", "feature_importance",
                        "normal_samples", "attack_samples"]:
                self.assertIn(key, data, f"JSON 應有 '{key}' 欄位")

    def test_json_meta_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = GradCAMReporter(output_dir=tmpdir,
                                        model_name="TestModel",
                                        dataset_name="simulate")
            path = reporter.save_json(self.report, self.summary,
                                       variant="gradcam++", target_layer="encoder_conv.6")
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["meta"]["model_name"], "TestModel")
            self.assertEqual(data["meta"]["dataset"], "simulate")
            self.assertEqual(data["meta"]["variant"], "gradcam++")

    def test_attack_samples_sorted_descending(self):
        """attack_samples 應按 recon_score 降序排列。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = GradCAMReporter(output_dir=tmpdir)
            path = reporter.save_json(self.report, self.summary)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            scores = [s["recon_score"] for s in data["attack_samples"]]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_sample_count_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = GradCAMReporter(output_dir=tmpdir)
            path = reporter.save_json(self.report, self.summary)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["meta"]["n_normal"], 10)
            self.assertEqual(data["meta"]["n_attack"], 10)


class TestGradCAMReporterHTML(unittest.TestCase):
    def setUp(self):
        self.report  = _make_fake_report()
        self.analyzer = GradCAMAnalyzer.__new__(GradCAMAnalyzer)
        self.summary  = self.analyzer.summarize(self.report)

    def test_html_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = GradCAMReporter(output_dir=tmpdir)
            path = reporter.save_html(self.report, self.summary)
            self.assertTrue(os.path.exists(path))

    def test_html_contains_key_elements(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = GradCAMReporter(output_dir=tmpdir,
                                        model_name="TestModel",
                                        dataset_name="simulate")
            path = reporter.save_html(self.report, self.summary, variant="gradcam")
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Grad-CAM", content)
            self.assertIn("TestModel", content)
            self.assertIn("simulate", content)
            self.assertIn("gradcam", content)

    def test_html_with_embedded_images(self):
        """提供圖片路徑時，HTML 應包含 base64 內嵌圖片。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 建立一個假 PNG（最小合法 PNG 的 bytes）
            png_bytes = (
                b'\x89PNG\r\n\x1a\n'
                b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx'
                b'\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N'
                b'\x00\x00\x00\x00IEND\xaeB`\x82'
            )
            img_path = os.path.join(tmpdir, "test.png")
            with open(img_path, "wb") as f:
                f.write(png_bytes)

            reporter = GradCAMReporter(output_dir=tmpdir)
            path = reporter.save_html(
                self.report, self.summary,
                image_paths={"mean_cam_comparison": img_path},
            )
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("data:image/png;base64,", content)


class TestGradCAMReporterExportAll(unittest.TestCase):
    def test_export_all_creates_both_files(self):
        report   = _make_fake_report()
        analyzer = GradCAMAnalyzer.__new__(GradCAMAnalyzer)
        summary  = analyzer.summarize(report)

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = GradCAMReporter(output_dir=tmpdir)
            paths    = reporter.export_all(report, summary)
            self.assertIn("json", paths)
            self.assertIn("html", paths)
            self.assertTrue(os.path.exists(paths["json"]))
            self.assertTrue(os.path.exists(paths["html"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
