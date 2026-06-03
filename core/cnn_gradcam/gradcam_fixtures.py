# ============================================================
# core/tests/gradcam_fixtures.py  - 共用測試夾具與 Mock 模型
# ============================================================
"""
測試夾具：提供與真實專案架構完全一致的 Mock 模型與假資料。

真實架構（CNNAutoencoderFlex）：
  encoder_conv: Sequential(
      Conv2d(1→ch0) + BN + Act,
      Conv2d(ch0→ch1) + BN + Act,
      Conv2d(ch1→ch2) + BN + Act,
      Conv2d(ch2→ch3) + BN + Act,
      AdaptiveMaxPool2d((4,4))    ← 固定瓶頸 4×4
  )
  encoder_fc:   Sequential(Linear(ch3*16→128), Act, Linear(128→latent_dim))
  decoder_fc:   Sequential(Linear(latent_dim→128), Act, Linear(128→ch3*16))
  decoder_deconv: Sequential(
      ConvTranspose2d(ch3→ch2,2,stride=2) + BN + Act,
      ConvTranspose2d(ch2→ch1,2,stride=2) + BN + Act,
      ConvTranspose2d(ch1→ch0,2,stride=2) + BN + Act,
      Conv2d(ch0→1,3,padding=1) + Sigmoid
  )
  forward(x) → (x_hat, z)，輸出形狀與輸入一致
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ── 完全對應真實 CNNAutoencoderFlex 的 Mock 模型 ──────────────
class MockCNNAutoencoder(nn.Module):
    """
    與 CNNAutoencoderFlex 結構完全一致的 Mock 模型。
    用於測試時無需載入真實訓練權重，但保留相同的層命名與 forward 邏輯。

    Args:
        latent_dim  : 瓶頸空間維度（預設 32）
        image_size  : 輸入影像尺寸（預設 32，即 32×32）
        ch_scale    : 通道數縮放係數（預設 1.0，對應 [16,32,64,128]）
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
            nn.AdaptiveMaxPool2d((4, 4)),        # 任意輸入尺寸 → 固定 4×4
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
        """
        返回 (x_hat, z)，與真實 CNNAutoencoderFlex.forward 完全一致。
        x_hat 由 F.interpolate 對齊到輸入尺寸。
        """
        z     = self.encode(x)
        x_dec = self.decode(z)
        # 對齊輸入尺寸（真實架構也做了 resize）
        if x_dec.shape != x.shape:
            x_hat = F.interpolate(x_dec, size=x.shape[2:], mode="bilinear", align_corners=False)
        else:
            x_hat = x_dec
        return x_hat, z


# ── 假資料產生器 ──────────────────────────────────────────────
class FakeDataFactory:
    """產生各種形狀的假封包影像資料，用於測試輸入處理。"""

    @staticmethod
    def normal_batch(n: int = 8, h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
        """
        正常流量假資料：值域 [0,1]，低強度隨機雜訊（模擬正常封包的低熵特徵）。
        shape: (N, H, W)
        """
        rng = np.random.RandomState(seed)
        return rng.uniform(0.0, 0.3, size=(n, h, w)).astype(np.float32)

    @staticmethod
    def attack_batch(n: int = 8, h: int = 32, w: int = 32, seed: int = 1) -> np.ndarray:
        """
        攻擊流量假資料：部分區域高強度（模擬攻擊封包的異常特徵分布）。
        shape: (N, H, W)
        """
        rng = np.random.RandomState(seed)
        data = rng.uniform(0.0, 0.3, size=(n, h, w)).astype(np.float32)
        # 注入異常區塊（右下角 1/4 高亮）
        data[:, h//2:, w//2:] = rng.uniform(0.7, 1.0, size=(n, h - h//2, w - w//2))
        return data

    @staticmethod
    def single_sample(h: int = 32, w: int = 32,
                      with_anomaly: bool = False) -> np.ndarray:
        """單筆樣本，shape: (H, W)"""
        data = FakeDataFactory.attack_batch(1, h, w) if with_anomaly \
               else FakeDataFactory.normal_batch(1, h, w)
        return data[0]

    @staticmethod
    def channel_first(n: int = 4, h: int = 32, w: int = 32) -> np.ndarray:
        """shape: (N, 1, H, W)（帶 channel 維度）"""
        return FakeDataFactory.normal_batch(n, h, w)[:, np.newaxis, :, :]


# ── 共用夾具函式 ──────────────────────────────────────────────
def make_model(latent_dim: int = 32, image_size: int = 32,
               eval_mode: bool = True) -> MockCNNAutoencoder:
    """建立並回傳 MockCNNAutoencoder，預設設為 eval 模式。"""
    model = MockCNNAutoencoder(latent_dim=latent_dim, image_size=image_size)
    if eval_mode:
        model.eval()
    return model


def make_gradcam(latent_dim: int = 32,
                 variant: str = "gradcam",
                 target_layer: str = "encoder_conv"):
    """
    建立 GradCAM 實例的便利函式。
    需要先確保 core/ 在 sys.path 中。
    """
    import sys, os
    _core = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _core not in sys.path:
        sys.path.insert(0, _core)

    from cnn_gradcam import GradCAM
    model = make_model(latent_dim=latent_dim)
    return GradCAM(model, target_layer=target_layer, variant=variant), model
