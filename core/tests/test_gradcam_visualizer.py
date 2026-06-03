# ============================================================
# core/tests/test_gradcam_visualizer.py  - 視覺化模組單元測試
# ============================================================
"""
測試範圍：
  - cam_to_rgb：形狀、dtype、值域
  - overlay：輸出形狀、save_path 產生檔案
  - plot_comparison：不崩潰、產生檔案
  - plot_batch：正確生成 N 張圖
  - plot_summary_grid：產生摘要網格圖
  - plot_score_vs_cam_intensity：產生散點圖
"""

import sys
import os
import unittest
import tempfile
import numpy as np

_CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from cnn_gradcam.gradcam_visualizer import GradCAMVisualizer
from tests.gradcam_fixtures import FakeDataFactory


def _make_heatmap(h=32, w=32):
    rng = np.random.RandomState(42)
    hm  = rng.rand(h, w).astype(np.float32)
    return hm / hm.max()   # 正規化到 [0,1]


class TestCamToRGB(unittest.TestCase):
    def setUp(self):
        self.viz = GradCAMVisualizer()

    def test_output_shape(self):
        hm = _make_heatmap()
        rgb = self.viz.cam_to_rgb(hm)
        self.assertEqual(rgb.shape, (32, 32, 3))

    def test_output_dtype(self):
        hm = _make_heatmap()
        rgb = self.viz.cam_to_rgb(hm)
        self.assertEqual(rgb.dtype, np.float32)

    def test_output_range(self):
        hm = _make_heatmap()
        rgb = self.viz.cam_to_rgb(hm)
        self.assertGreaterEqual(rgb.min(), 0.0)
        self.assertLessEqual(rgb.max(), 1.0 + 1e-6)

    def test_different_colormaps(self):
        for cmap in ["jet", "plasma", "inferno", "hot"]:
            viz = GradCAMVisualizer(colormap=cmap)
            rgb = viz.cam_to_rgb(_make_heatmap())
            self.assertEqual(rgb.shape, (32, 32, 3))


class TestOverlay(unittest.TestCase):
    def setUp(self):
        self.viz = GradCAMVisualizer()
        self.x   = FakeDataFactory.single_sample()    # (32, 32)
        self.hm  = _make_heatmap()

    def test_output_shape(self):
        blended = self.viz.overlay(self.x, self.hm)
        self.assertEqual(blended.shape, (32, 32, 3))

    def test_output_range(self):
        blended = self.viz.overlay(self.x, self.hm)
        self.assertGreaterEqual(blended.min(), 0.0)
        self.assertLessEqual(blended.max(), 1.0 + 1e-6)

    def test_save_path_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "overlay.png")
            self.viz.overlay(self.x, self.hm, save_path=save_path, score=0.012)
            self.assertTrue(os.path.exists(save_path), "儲存路徑的圖片應存在")

    def test_channel_first_input(self):
        """(1, H, W) 輸入應自動 squeeze。"""
        x = self.x[np.newaxis, ...]   # (1, 32, 32)
        blended = self.viz.overlay(x, self.hm)
        self.assertEqual(blended.shape, (32, 32, 3))


class TestPlotComparison(unittest.TestCase):
    def setUp(self):
        self.viz  = GradCAMVisualizer()
        self.x    = FakeDataFactory.single_sample()
        self.xhat = FakeDataFactory.single_sample(with_anomaly=False)
        self.hm   = _make_heatmap()

    def test_creates_png_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "comp.png")
            self.viz.plot_comparison(
                x=self.x, x_hat=self.xhat, heatmap=self.hm,
                save_path=path, label="Attack", score=0.05,
            )
            self.assertTrue(os.path.exists(path))

    def test_no_crash_without_field_names(self):
        """不傳 field_names 應不崩潰。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "comp_no_names.png")
            self.viz.plot_comparison(
                x=self.x, x_hat=self.xhat, heatmap=self.hm,
                save_path=path, label="Normal",
            )
            self.assertTrue(os.path.exists(path))

    def test_with_field_names(self):
        names = [f"feature_{i}" for i in range(32)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "comp_names.png")
            self.viz.plot_comparison(
                x=self.x, x_hat=self.xhat, heatmap=self.hm,
                save_path=path, field_names=names,
            )
            self.assertTrue(os.path.exists(path))


class TestPlotBatch(unittest.TestCase):
    def setUp(self):
        self.viz = GradCAMVisualizer()

    def test_generates_correct_number_of_files(self):
        n = 6
        X    = FakeDataFactory.normal_batch(n)
        Xhat = FakeDataFactory.normal_batch(n)
        hms  = np.stack([_make_heatmap() for _ in range(n)])
        labels  = ["Normal"] * 3 + ["Attack"] * 3
        scores  = list(np.random.rand(n))

        with tempfile.TemporaryDirectory() as tmpdir:
            self.viz.plot_batch(X, Xhat, hms, labels, scores,
                                output_dir=tmpdir, max_samples=n)
            png_files = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
            self.assertEqual(len(png_files), n)

    def test_max_samples_respected(self):
        n, max_n = 10, 4
        X   = FakeDataFactory.normal_batch(n)
        Xhat = FakeDataFactory.normal_batch(n)
        hms  = np.stack([_make_heatmap() for _ in range(n)])

        with tempfile.TemporaryDirectory() as tmpdir:
            self.viz.plot_batch(X, Xhat, hms,
                                labels=["Normal"]*n, scores=[0.0]*n,
                                output_dir=tmpdir, max_samples=max_n)
            png_files = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
            self.assertEqual(len(png_files), max_n)


class TestPlotSummaryGrid(unittest.TestCase):
    def test_creates_grid_image(self):
        viz = GradCAMVisualizer()
        n_hms = np.stack([_make_heatmap() for _ in range(8)])
        a_hms = np.stack([_make_heatmap() for _ in range(8)])

        with tempfile.TemporaryDirectory() as tmpdir:
            viz.plot_summary_grid(n_hms, a_hms, output_dir=tmpdir, n_cols=4)
            self.assertTrue(
                os.path.exists(os.path.join(tmpdir, "gradcam_summary_grid.png"))
            )


class TestPlotScoreVsCamIntensity(unittest.TestCase):
    def test_creates_scatter_plot(self):
        viz    = GradCAMVisualizer()
        scores = np.random.rand(20).astype(np.float32)
        intens = np.random.rand(20).astype(np.float32)
        labels = ["Normal"] * 10 + ["Attack"] * 10

        with tempfile.TemporaryDirectory() as tmpdir:
            viz.plot_score_vs_cam_intensity(scores, intens, labels, tmpdir)
            self.assertTrue(
                os.path.exists(os.path.join(tmpdir, "score_vs_cam_intensity.png"))
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
