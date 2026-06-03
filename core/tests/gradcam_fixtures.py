# ============================================================
# core/tests/gradcam_fixtures.py  - 共用測試夾具與 Mock 模型（修正版）
#
# 修正清單：
#   [Bug 8] make_gradcam()：原版從 "gradcam" 匯入 GradCAM，
#           但套件實際名稱為 "cnn_gradcam"（core/cnn_gradcam/）。
#           所有使用 make_gradcam 的測試在 import 階段即失敗：
#               ModuleNotFoundError: No module named 'gradcam'
#           修正：改為 from cnn_gradcam import GradCAM。
#
#           注意：core/cnn_gradcam/gradcam_fixtures.py（套件內
#           同名檔）已正確使用 cnn_gradcam，只有此 tests/ 版本
#           遺漏更新。
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ── Mock 模型（與真實 CNNAutoencoderFlex 架構完全一致）────────
class MockCNNAutoencoder(nn.Module):
    """
    與 CNNAutoencoderFlex 結構完全一致的 Mock 模型。
    用於測試時無需載入真實訓練權重，但保留相同的層命名與 forward 邏輯。
    """

    def __init__(
        self,
        latent_dim: int = 32,
        image_size: int = 32,
        ch_scale: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        ch = [max(1, int(c * ch_scale)) for c in [16, 32, 64, 128]]

        def _act():
            return nn.ReLU(inplace=True)

        # ── Encoder ───────────────────────────────────────────
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, ch[0], 3, padding=1), nn.BatchNorm2d(ch[0]), _act(),
            nn.Conv2d(ch[0], ch[1], 3, padding=1), nn.BatchNorm2d(ch[1]), _act(),
            nn.Conv2d(ch[1], ch[2], 3, padding=1), nn.BatchNorm2d(ch[2]), _act(),
            nn.Conv2d(ch[2], ch[3], 3, padding=1), nn.BatchNorm2d(ch[3]), _act(),
            nn.AdaptiveMaxPool2d((4, 4)),
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch[3] * 16, 128), _act(),
            nn.Linear(128, latent_dim),
        )

        # ── Decoder ───────────────────────────────────────────
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128), _act(),
            nn.Linear(128, ch[3] * 16), _act(),
        )
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2), nn.BatchNorm2d(ch[2]), _act(),
            nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2), nn.BatchNorm2d(ch[1]), _act(),
            nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2), nn.BatchNorm2d(ch[0]), _act(),
            nn.Conv2d(ch[0], 1, 3, padding=1), nn.Sigmoid(),
        )
        self._ch3 = ch[3]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_fc(self.encoder_conv(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_fc(z)
        x = x.view(-1, self._ch3, 4, 4)
        return self.decoder_deconv(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z     = self.encode(x)
        x_dec = self.decode(z)
        if x_dec.shape != x.shape:
            x_hat = F.interpolate(x_dec, size=x.shape[2:],
                                   mode="bilinear", align_corners=False)
        else:
            x_hat = x_dec
        return x_hat, z


# ── 假資料產生器 ──────────────────────────────────────────────
class FakeDataFactory:
    """產生各種形狀的假封包影像資料，用於測試輸入處理。"""

    @staticmethod
    def normal_batch(n: int = 8, h: int = 32, w: int = 32,
                     seed: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.uniform(0.0, 0.3, size=(n, h, w)).astype(np.float32)

    @staticmethod
    def attack_batch(n: int = 8, h: int = 32, w: int = 32,
                     seed: int = 1) -> np.ndarray:
        rng  = np.random.RandomState(seed)
        data = rng.uniform(0.0, 0.3, size=(n, h, w)).astype(np.float32)
        data[:, h//2:, w//2:] = rng.uniform(
            0.7, 1.0, size=(n, h - h//2, w - w//2))
        return data

    @staticmethod
    def single_sample(h: int = 32, w: int = 32,
                      with_anomaly: bool = False) -> np.ndarray:
        data = (FakeDataFactory.attack_batch(1, h, w) if with_anomaly
                else FakeDataFactory.normal_batch(1, h, w))
        return data[0]

    @staticmethod
    def channel_first(n: int = 4, h: int = 32, w: int = 32) -> np.ndarray:
        return FakeDataFactory.normal_batch(n, h, w)[:, np.newaxis, :, :]


# ── 共用夾具函式 ──────────────────────────────────────────────
def make_model(latent_dim: int = 32, image_size: int = 32,
               eval_mode: bool = True) -> MockCNNAutoencoder:
    model = MockCNNAutoencoder(latent_dim=latent_dim, image_size=image_size)
    if eval_mode:
        model.eval()
    return model


def make_gradcam(latent_dim: int = 32,
                 variant: str = "gradcam",
                 target_layer: str = "encoder_conv"):
    """
    建立 GradCAM 實例的便利函式。

    [Bug 8 修正] 原版：from gradcam import GradCAM
    套件已重新命名為 cnn_gradcam（core/cnn_gradcam/），
    原名稱導致所有測試在 import 階段即失敗。
    修正後：from cnn_gradcam import GradCAM
    """
    import sys, os
    _core = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _core not in sys.path:
        sys.path.insert(0, _core)

    # [Bug 8 修正] 舊版：from gradcam import GradCAM  ← 錯誤
    from cnn_gradcam import GradCAM               # ← 修正

    model = make_model(latent_dim=latent_dim)
    return GradCAM(model, target_layer=target_layer, variant=variant), model