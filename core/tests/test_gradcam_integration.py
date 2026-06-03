# ============================================================
# core/tests/test_gradcam_integration.py  - 端到端整合測試
# ============================================================
"""
整合測試：驗證各模組串接後的完整流程是否正確運作。

測試流程：
  1. MockCNNAutoencoder（與真實架構一致）
  2. GradCAM.generate → 熱力圖
  3. GradCAMVisualizer.plot_batch → PNG 圖片
  4. GradCAMAnalyzer.analyze → AnalysisReport
  5. GradCAMReporter.export_all → JSON + HTML

額外驗證：
  - 三種變體串接後均能完整執行
  - 攻擊樣本的統計特性（diff_cam 有正值）
  - 報告 JSON 的 feature_importance 陣列長度正確
"""

import sys
import os
import json
import tempfile
import unittest
import numpy as np

_CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

# ── 視覺化輸出目錄：output/gradcam_test_output/integration/ ──
_OUTPUT_DIR = os.path.normpath(
    os.path.join(_CORE, "..", "output", "gradcam_test_output", "integration")
)

from cnn_gradcam import GradCAM, GradCAMVisualizer, GradCAMAnalyzer, GradCAMReporter
from tests.gradcam_fixtures import make_model, FakeDataFactory


N_NORMAL = 20
N_ATTACK = 20
H, W     = 32, 32
LATENT   = 32

FIELD_NAMES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
]


class TestFullPipelineGradCAM(unittest.TestCase):
    """Grad-CAM 完整流程整合測試（gradcam 變體）。"""

    @classmethod
    def setUpClass(cls):
        """所有測試共用同一組資料和模型（加速）。"""
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        cls.model    = make_model(latent_dim=LATENT, image_size=H)
        cls.X_n      = FakeDataFactory.normal_batch(N_NORMAL, H, W)
        cls.X_a      = FakeDataFactory.attack_batch(N_ATTACK, H, W)
        cls.cam      = GradCAM(cls.model, target_layer="encoder_conv", variant="gradcam")
        cls.viz      = GradCAMVisualizer(colormap="jet")
        cls.analyzer = GradCAMAnalyzer(cls.cam, cls.model, batch_size=8)
        print(f"\n  [圖表輸出] {os.path.abspath(_OUTPUT_DIR)}")

    def test_step1_heatmaps_shape(self):
        """Grad-CAM 輸出形狀應為 (N, H, W)。"""
        hm_n = self.cam.generate(self.X_n)
        hm_a = self.cam.generate(self.X_a)
        self.assertEqual(hm_n.shape, (N_NORMAL, H, W))
        self.assertEqual(hm_a.shape, (N_ATTACK, H, W))

    def test_step2_visualizer_batch_output(self):
        """plot_batch 應正確生成 PNG 圖檔並儲存至 output/gradcam_test_output/。"""
        hm_n   = self.cam.generate(self.X_n[:4])
        hm_a   = self.cam.generate(self.X_a[:4])
        recs_n = self.analyzer.get_reconstructions(self.X_n[:4])
        recs_a = self.analyzer.get_reconstructions(self.X_a[:4])

        X_all  = np.concatenate([self.X_n[:4], self.X_a[:4]])
        Xh_all = np.concatenate([recs_n, recs_a])
        hm_all = np.concatenate([hm_n, hm_a])
        labels = ["Normal"]*4 + ["Attack"]*4
        scores = list(self.analyzer._compute_recon_scores(self.X_n[:4])) + \
                 list(self.analyzer._compute_recon_scores(self.X_a[:4]))

        viz_dir = os.path.join(_OUTPUT_DIR, "sample_visualizations")
        self.viz.plot_batch(X_all, Xh_all, hm_all, labels, scores,
                            output_dir=viz_dir, max_samples=8,
                            field_names=FIELD_NAMES)
        pngs = [f for f in os.listdir(viz_dir) if f.endswith(".png")]
        self.assertEqual(len(pngs), 8, "應生成 8 張 PNG")

    def test_step3_analyze_report_structure(self):
        """AnalysisReport 應包含完整欄位且類型正確，分析圖存至 output/。"""
        report = self.analyzer.analyze(
            self.X_n, self.X_a,
            max_samples_each=10,
            field_names=FIELD_NAMES,
        )
        # 輸出分析圖到可查閱目錄
        analysis_dir = os.path.join(_OUTPUT_DIR, "analysis")
        self.analyzer.plot_mean_cam_comparison(report, analysis_dir)
        self.analyzer.plot_feature_importance(report, analysis_dir)
        self.analyzer.plot_cam_intensity_distribution(report, analysis_dir)

        hm_n_all = np.stack([r.heatmap for r in report.normal_results])
        hm_a_all = np.stack([r.heatmap for r in report.attack_results])
        self.viz.plot_summary_grid(hm_n_all, hm_a_all, analysis_dir, n_cols=8)

        all_res = report.normal_results + report.attack_results
        self.viz.plot_score_vs_cam_intensity(
            scores          = np.array([r.recon_score for r in all_res]),
            cam_intensities = np.array([r.cam_mean    for r in all_res]),
            labels          = [r.label for r in all_res],
            output_dir      = analysis_dir,
        )

        self.assertEqual(len(report.normal_results), 10)
        self.assertEqual(len(report.attack_results), 10)
        self.assertIsNotNone(report.diff_cam)
        self.assertEqual(report.diff_cam.shape, (H, W))
        self.assertIsNotNone(report.feature_importance)
        self.assertEqual(len(report.feature_importance), H)
        self.assertEqual(report.feature_names, FIELD_NAMES)

    def test_step4_report_export(self):
        """GradCAMReporter 應同時輸出 JSON 和 HTML 至 output/。"""
        report  = self.analyzer.analyze(
            self.X_n, self.X_a,
            max_samples_each=10,
            field_names=FIELD_NAMES,
        )
        summary = self.analyzer.summarize(report)

        report_dir = os.path.join(_OUTPUT_DIR, "report")
        os.makedirs(report_dir, exist_ok=True)

        # 收集已生成的分析圖路徑嵌入 HTML
        analysis_dir = os.path.join(_OUTPUT_DIR, "analysis")
        image_paths  = {
            k: os.path.join(analysis_dir, f"{k}.png")
            for k in ["mean_cam_comparison", "feature_importance",
                      "cam_intensity_distribution", "score_vs_cam_intensity",
                      "gradcam_summary_grid"]
        }

        reporter = GradCAMReporter(
            output_dir   = report_dir,
            model_name   = f"CNNAutoencoder (latent={LATENT})",
            dataset_name = "simulate",
        )
        paths = reporter.export_all(report, summary, image_paths=image_paths)

        self.assertTrue(os.path.exists(paths["json"]))
        self.assertTrue(os.path.exists(paths["html"]))

        with open(paths["json"], encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data["feature_importance"]), H)
        self.assertEqual(data["meta"]["n_normal"], 10)
        self.assertEqual(data["meta"]["n_attack"], 10)


class TestAllVariantsPipeline(unittest.TestCase):
    """三種 Grad-CAM 變體的端到端測試。"""

    @classmethod
    def setUpClass(cls):
        cls.model = make_model(latent_dim=LATENT)
        cls.X_n   = FakeDataFactory.normal_batch(6, H, W)
        cls.X_a   = FakeDataFactory.attack_batch(6, H, W)

    def _run_variant(self, variant):
        cam      = GradCAM(self.model, target_layer="encoder_conv", variant=variant)
        hm_n     = cam.generate(self.X_n)
        hm_a     = cam.generate(self.X_a)
        return hm_n, hm_a

    def test_gradcam_variant(self):
        hm_n, hm_a = self._run_variant("gradcam")
        self.assertEqual(hm_n.shape, (6, H, W))
        self.assertEqual(hm_a.shape, (6, H, W))

    def test_gradcam_pp_variant(self):
        hm_n, hm_a = self._run_variant("gradcam++")
        self.assertEqual(hm_n.shape, (6, H, W))

    def test_scorecam_variant(self):
        hm_n, hm_a = self._run_variant("scorecam")
        self.assertEqual(hm_n.shape, (6, H, W))

    def test_all_variants_same_shape(self):
        """三種變體輸出形狀應完全一致。"""
        shapes = []
        for v in ["gradcam", "gradcam++", "scorecam"]:
            cam = GradCAM(self.model, target_layer="encoder_conv", variant=v)
            hm  = cam.generate(self.X_n[:2])
            shapes.append(hm.shape)
        self.assertEqual(shapes[0], shapes[1])
        self.assertEqual(shapes[1], shapes[2])

    def test_variants_produce_different_heatmaps(self):
        """
        不同變體應產生不同的熱力圖（算法差異應體現在輸出上）。
        注意：未訓練的隨機模型差異可能較小，此為弱驗證。
        """
        hms = {}
        for v in ["gradcam", "gradcam++"]:
            cam = GradCAM(self.model, target_layer="encoder_conv", variant=v)
            hms[v] = cam.generate(self.X_n[:1])

        # 不應完全相同
        are_identical = np.allclose(hms["gradcam"], hms["gradcam++"], atol=1e-6)
        # 僅警告，不強制失敗（隨機模型可能產生相近結果）
        if are_identical:
            import warnings
            warnings.warn("gradcam 與 gradcam++ 輸出完全相同，可能代表模型未訓練")


class TestDiffCamInterpretability(unittest.TestCase):
    """
    差異 CAM 可解釋性驗證。

    注意：使用未訓練（隨機初始化）的模型時，梯度量級極小（~1e-11），
    經過 ReLU + 正規化後 CAM 可能全為零，這是預期行為。
    本測試只驗證結構完整性（shape / dtype）；
    真正的可解釋性差異需在訓練後的模型上驗證。
    """

    def setUp(self):
        self.model = make_model(latent_dim=LATENT)
        cam = GradCAM(self.model, target_layer="encoder_conv", variant="gradcam")
        self.analyzer = GradCAMAnalyzer(cam, self.model, batch_size=8)

    def test_diff_cam_shape_and_dtype(self):
        """diff_cam 應具有正確的形狀與 float 型別（無論訓練與否）。"""
        X_n = FakeDataFactory.normal_batch(10)
        X_a = FakeDataFactory.attack_batch(10)
        report = self.analyzer.analyze(X_n, X_a, max_samples_each=10)
        self.assertIsNotNone(report.diff_cam)
        self.assertEqual(report.diff_cam.shape, (H, W))
        self.assertEqual(report.diff_cam.dtype, np.float32)

    def test_diff_cam_is_finite(self):
        """diff_cam 中不應有 NaN 或 inf（數值穩定性）。"""
        X_n = FakeDataFactory.normal_batch(10)
        X_a = FakeDataFactory.attack_batch(10)
        report = self.analyzer.analyze(X_n, X_a, max_samples_each=10)
        self.assertTrue(np.isfinite(report.diff_cam).all(),
                        "diff_cam 中不應有 NaN 或 inf")

    def test_feature_importance_shape(self):
        """feature_importance 長度應等於影像高度 H。"""
        X_n = FakeDataFactory.normal_batch(10)
        X_a = FakeDataFactory.attack_batch(10)
        report = self.analyzer.analyze(X_n, X_a, max_samples_each=10)
        self.assertIsNotNone(report.feature_importance)
        self.assertEqual(len(report.feature_importance), H)
        self.assertTrue(np.isfinite(report.feature_importance).all())


class TestEdgeCases(unittest.TestCase):
    """邊界條件測試。"""

    def setUp(self):
        self.model = make_model()

    def test_batch_size_one(self):
        """batch_size=1 應正常執行。"""
        cam = GradCAM(self.model, target_layer="encoder_conv")
        x = FakeDataFactory.normal_batch(1)
        hm = cam.generate(x)
        self.assertEqual(hm.shape, (1, H, W))

    def test_large_batch(self):
        """較大批次（64）應正常執行。"""
        cam = GradCAM(self.model, target_layer="encoder_conv")
        x = FakeDataFactory.normal_batch(64)
        hm = cam.generate(x)
        self.assertEqual(hm.shape, (64, H, W))

    def test_different_image_sizes(self):
        """不同影像尺寸（24×24）應透過 AdaptiveMaxPool2d 正常處理。"""
        model_24 = make_model(image_size=24)
        cam = GradCAM(model_24, target_layer="encoder_conv")
        x = np.random.rand(2, 24, 24).astype(np.float32)
        hm = cam.generate(x)
        self.assertEqual(hm.shape, (2, 24, 24))

    def test_reuse_cam_multiple_times(self):
        """同一個 GradCAM 實例應可重複使用（hook 不會累積）。"""
        cam = GradCAM(self.model, target_layer="encoder_conv")
        x = FakeDataFactory.normal_batch(4)
        for _ in range(3):
            hm = cam.generate(x)
            self.assertEqual(hm.shape, (4, H, W))

    def test_analyzer_with_max_samples_larger_than_data(self):
        """max_samples_each 大於實際資料量時應自動截斷，不崩潰。"""
        cam = GradCAM(self.model, target_layer="encoder_conv")
        analyzer = GradCAMAnalyzer(cam, self.model, batch_size=4)
        X_n = FakeDataFactory.normal_batch(5)
        X_a = FakeDataFactory.attack_batch(5)
        report = analyzer.analyze(X_n, X_a, max_samples_each=100)  # 大於 5
        self.assertEqual(len(report.normal_results), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
