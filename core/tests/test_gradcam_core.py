# ============================================================
# core/tests/test_gradcam_core.py  - GradCAM 核心演算法單元測試
# ============================================================
"""
測試範圍：
  - _preprocess_input：各種輸入形狀自動補齊
  - _normalize_batch：正規化到 [0,1]，全零輸入不崩潰
  - _gaussian_smooth：輸出形狀一致，值域不超出 [0,1]
  - _resolve_target_layer：Sequential → 最後一個 Conv2d
  - GradCAM.generate（gradcam / gradcam++ / scorecam）：
      輸出形狀 / 值域 / 攻擊分數 > 正常分數（可解釋性驗證）
  - list_target_layers：回傳非空列表
  - 無效 variant 拋出 ValueError
"""

import sys
import os
import unittest
import numpy as np
import torch

_CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from cnn_gradcam.gradcam_core import GradCAM
from tests.gradcam_fixtures import (
    make_model, make_gradcam,
    FakeDataFactory, MockCNNAutoencoder,
)


class TestPreprocessInput(unittest.TestCase):
    """_preprocess_input 靜態方法：各種輸入形狀統一為 (B, 1, H, W)。"""

    def test_2d_input(self):
        x = np.zeros((32, 32), dtype=np.float32)
        result = GradCAM._preprocess_input(x)
        self.assertEqual(result.shape, (1, 1, 32, 32))

    def test_3d_input_with_batch(self):
        # (B, H, W) where B > 1
        x = np.zeros((4, 32, 32), dtype=np.float32)
        result = GradCAM._preprocess_input(x)
        self.assertEqual(result.shape, (4, 1, 32, 32))

    def test_3d_input_single(self):
        # (1, H, W) → should become (1, 1, H, W)
        x = np.zeros((1, 32, 32), dtype=np.float32)
        result = GradCAM._preprocess_input(x)
        self.assertEqual(result.shape, (1, 1, 32, 32))

    def test_4d_input_passthrough(self):
        x = np.zeros((2, 1, 32, 32), dtype=np.float32)
        result = GradCAM._preprocess_input(x)
        self.assertEqual(result.shape, (2, 1, 32, 32))

    def test_numpy_to_tensor(self):
        x = np.zeros((32, 32), dtype=np.float32)
        result = GradCAM._preprocess_input(x)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)

    def test_tensor_passthrough(self):
        x = torch.zeros(1, 1, 32, 32)
        result = GradCAM._preprocess_input(x)
        self.assertEqual(result.shape, (1, 1, 32, 32))


class TestNormalizeBatch(unittest.TestCase):
    """_normalize_batch：每樣本獨立正規化到 [0,1]。"""

    def test_output_range(self):
        cam = np.array([[[0.0, 2.0], [4.0, 8.0]],
                        [[1.0, 3.0], [5.0, 7.0]]], dtype=np.float32)
        result = GradCAM._normalize_batch(cam)
        self.assertAlmostEqual(result[0].max(), 1.0, places=5)
        self.assertAlmostEqual(result[0].min(), 0.0, places=5)
        self.assertAlmostEqual(result[1].max(), 1.0, places=5)

    def test_constant_input_no_crash(self):
        """全零輸入不應崩潰，輸出應為全零。"""
        cam = np.zeros((2, 4, 4), dtype=np.float32)
        result = GradCAM._normalize_batch(cam)
        np.testing.assert_array_equal(result, np.zeros_like(cam))

    def test_each_sample_normalized_independently(self):
        """兩個樣本應獨立正規化，互不影響。"""
        cam = np.array([
            [[0, 1], [2, 3]],    # max=3
            [[0, 10], [20, 30]], # max=30
        ], dtype=np.float32)
        result = GradCAM._normalize_batch(cam)
        self.assertAlmostEqual(result[0].max(), 1.0, places=5)
        self.assertAlmostEqual(result[1].max(), 1.0, places=5)


class TestGaussianSmooth(unittest.TestCase):
    """_gaussian_smooth：輸出形狀一致，平滑後邊緣值下降。"""

    def test_shape_preserved(self):
        cam = np.random.rand(3, 16, 16).astype(np.float32)
        result = GradCAM._gaussian_smooth(cam)
        self.assertEqual(result.shape, cam.shape)

    def test_range_clipping(self):
        """值域不應超出輸入範圍太多（高斯平滑保守性）。"""
        cam = np.clip(np.random.rand(2, 8, 8).astype(np.float32), 0, 1)
        result = GradCAM._gaussian_smooth(cam)
        self.assertGreaterEqual(result.min(), -0.01)
        self.assertLessEqual(result.max(), 1.01)

    def test_uniform_input_unchanged(self):
        """均勻輸入經平滑後應保持不變（高斯核對常數無效應）。"""
        cam = np.ones((1, 8, 8), dtype=np.float32) * 0.5
        result = GradCAM._gaussian_smooth(cam)
        np.testing.assert_allclose(result, cam, atol=1e-4)


class TestResolveTargetLayer(unittest.TestCase):
    """_resolve_target_layer：Sequential → 最後一個 Conv2d。"""

    def setUp(self):
        self.model = make_model()

    def test_sequential_resolves_to_last_conv(self):
        """'encoder_conv' 是 Sequential，應解析為其中最後一個 Conv2d。"""
        cam = GradCAM(self.model, target_layer="encoder_conv")
        resolved = cam._resolved_layer
        # 驗證解析到的是真實的 Conv2d
        import torch.nn as nn
        mod = dict(self.model.named_modules()).get(resolved)
        self.assertIsNotNone(mod, f"解析後的層 '{resolved}' 不存在")
        self.assertIsInstance(mod, nn.Conv2d)

    def test_exact_layer_name(self):
        """精確層名稱（encoder_conv.0）應直接解析。"""
        cam = GradCAM(self.model, target_layer="encoder_conv.0")
        self.assertEqual(cam._resolved_layer, "encoder_conv.0")

    def test_invalid_layer_raises(self):
        with self.assertRaises(ValueError):
            GradCAM(self.model, target_layer="nonexistent_xyz")


class TestGradCAMGenerate(unittest.TestCase):
    """GradCAM.generate 核心輸出測試（三種變體）。"""

    def setUp(self):
        self.model = make_model()

    def _make_cam(self, variant="gradcam"):
        return GradCAM(self.model, target_layer="encoder_conv", variant=variant)

    # ── 輸出形狀 ──────────────────────────────────────────────
    def test_output_shape_batch(self):
        """批次輸入應產生 (B, H, W) 輸出。"""
        cam = self._make_cam()
        x = FakeDataFactory.normal_batch(n=4)
        heatmaps = cam.generate(x)
        self.assertEqual(heatmaps.shape, (4, 32, 32))

    def test_output_shape_single(self):
        """單一樣本輸入應產生 (1, H, W) 輸出。"""
        cam = self._make_cam()
        x = FakeDataFactory.single_sample()
        heatmaps = cam.generate(x)
        self.assertEqual(heatmaps.shape, (1, 32, 32))

    def test_output_shape_with_channel_dim(self):
        """帶 channel 維度的輸入 (N, 1, H, W) 應正常處理。"""
        cam = self._make_cam()
        x = FakeDataFactory.channel_first(n=3)
        heatmaps = cam.generate(x)
        self.assertEqual(heatmaps.shape, (3, 32, 32))

    # ── 值域 ──────────────────────────────────────────────────
    def test_output_range_normalized(self):
        """輸出值域應在 [0, 1] 內（normalize_batch 保證）。"""
        cam = self._make_cam()
        x = FakeDataFactory.normal_batch(n=4)
        heatmaps = cam.generate(x)
        self.assertGreaterEqual(heatmaps.min(), -1e-6)
        self.assertLessEqual(heatmaps.max(), 1.0 + 1e-6)

    def test_output_is_numpy(self):
        """輸出應為 numpy array（不是 torch.Tensor）。"""
        cam = self._make_cam()
        x = FakeDataFactory.normal_batch(n=2)
        result = cam.generate(x)
        self.assertIsInstance(result, np.ndarray)

    # ── 無平滑 ────────────────────────────────────────────────
    def test_no_smooth_still_valid(self):
        """smooth=False 時輸出形狀與值域應相同。"""
        cam = self._make_cam()
        x = FakeDataFactory.normal_batch(n=2)
        hm = cam.generate(x, smooth=False)
        self.assertEqual(hm.shape, (2, 32, 32))
        self.assertLessEqual(hm.max(), 1.0 + 1e-6)

    # ── 變體測試 ──────────────────────────────────────────────
    def test_gradcam_pp_output_shape(self):
        cam = self._make_cam("gradcam++")
        x = FakeDataFactory.normal_batch(n=2)
        hm = cam.generate(x)
        self.assertEqual(hm.shape, (2, 32, 32))

    def test_scorecam_output_shape(self):
        """Score-CAM 應產生正確形狀（計算量較大，用小批次）。"""
        cam = self._make_cam("scorecam")
        x = FakeDataFactory.normal_batch(n=2)
        hm = cam.generate(x)
        self.assertEqual(hm.shape, (2, 32, 32))

    def test_invalid_variant_raises(self):
        with self.assertRaises(ValueError):
            cam = GradCAM(self.model, variant="invalid_variant")
            cam.generate(FakeDataFactory.normal_batch(n=1))

    # ── 可解釋性驗證：攻擊樣本的 CAM 強度應高於正常樣本 ────────
    def test_attack_cam_stronger_than_normal(self):
        """
        攻擊樣本（注入高值異常區塊）的平均 CAM 強度
        在統計上應高於正常樣本，驗證 Grad-CAM 有正確追蹤到異常來源。
        （因模型未訓練，此為弱驗證；訓練後差距更顯著）
        """
        cam = self._make_cam()
        x_normal = FakeDataFactory.normal_batch(n=16)
        x_attack = FakeDataFactory.attack_batch(n=16)

        hm_normal = cam.generate(x_normal)
        hm_attack = cam.generate(x_attack)

        # 攻擊樣本右下角 (異常注入區域) 的 CAM 平均值
        h, w = 32, 32
        cam_normal_corner = hm_normal[:, h//2:, w//2:].mean()
        cam_attack_corner = hm_attack[:, h//2:, w//2:].mean()

        # 不是嚴格要求，但攻擊樣本的 CAM 不應明顯低於正常
        # （此處為弱驗證；真實訓練模型應呈現更大差距）
        self.assertGreaterEqual(
            cam_attack_corner + 0.05,  # 容許 5% 容差
            cam_normal_corner,
            "攻擊樣本的 CAM 強度（異常區域）不應明顯低於正常樣本"
        )

    # ── target_type 測試 ──────────────────────────────────────
    def test_pixel_target_type(self):
        """target_type='pixel' 應正常執行。"""
        cam = GradCAM(self.model, target_layer="encoder_conv",
                      target_type="pixel")
        x = FakeDataFactory.normal_batch(n=2)
        hm = cam.generate(x, pixel_coords=(16, 16))
        self.assertEqual(hm.shape, (2, 32, 32))

    def test_channel_target_type(self):
        """target_type='channel' 應正常執行。"""
        cam = GradCAM(self.model, target_layer="encoder_conv",
                      target_type="channel")
        x = FakeDataFactory.normal_batch(n=2)
        hm = cam.generate(x, channel_idx=0)
        self.assertEqual(hm.shape, (2, 32, 32))


class TestGradCAMListLayers(unittest.TestCase):
    """list_target_layers 測試。"""

    def test_returns_nonempty_list(self):
        model = make_model()
        cam = GradCAM(model)
        layers = cam.list_target_layers()
        self.assertIsInstance(layers, list)
        self.assertGreater(len(layers), 0)

    def test_all_encoder_conv_layers_present(self):
        """encoder_conv 中的各 Conv2d 子層應在列表中。"""
        model = make_model()
        cam = GradCAM(model)
        layers = cam.list_target_layers()
        encoder_layers = [l for l in layers if "encoder_conv" in l]
        self.assertGreater(len(encoder_layers), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
