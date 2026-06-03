# ============================================================
# cnn_autoencoder.py - CNN Autoencoder 模型定義
#
# 架構：
#   Input(1x32x32) -> Encoder -> Bottleneck(z) -> Decoder -> Output(1x32x32)
#
# v2.2 修正：
#   將模型從 4.8M 參數縮減至約 300K 參數。
#   原因：4.8M 參數對只有 150 筆訓練資料的小型資料集嚴重過大，
#         模型會學到「重建所有封包」而非「只重建正常封包」，
#         導致攻擊封包的重建誤差與正常封包相同，無法偵測異常。
#   對應關係：參數量應約為訓練樣本數的 2~5 倍，
#             小型資料集（< 500 筆）使用輕量架構。
#
# 參考 GitHub：
#   - https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder
#   - https://github.com/pytorch/examples/tree/main/vae
#   - https://github.com/patrickloeber/pytorchTutorial
# ============================================================

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    編碼器（Encoder）：將 32x32 封包影像壓縮為低維潛在向量

    架構（縮減後）：
      Block 1: Conv(1->16, 3x3) + BN + ReLU                   -> (16, 32, 32)
      Block 2: Conv(16->32, 3x3) + BN + ReLU + MaxPool(2x2)   -> (32, 16, 16)
      Block 3: Conv(32->64, 3x3) + BN + ReLU + MaxPool(2x2)   -> (64, 8, 8)
      Block 4: Conv(64->64, 3x3) + BN + ReLU + MaxPool(2x2)   -> (64, 4, 4)
      Flatten -> FC(64*4*4=1024 -> 128 -> latent_dim)

    縮減原因：
      原架構使用 (32->64->128->256) 通道，產生 4.8M 參數。
      對於 150~500 筆訓練資料，過多參數讓模型學到「重建任何封包」
      而非「只重建正常封包的特徵」，破壞了 Autoencoder 的異常偵測前提。
      縮減至 (16->32->64->64) 通道後，約 300K 參數，
      模型被迫學習更精簡的正常流量表示，異常封包的重建誤差才會明顯偏高。
    """

    def __init__(self, latent_dim: int = 32):
        """
        Args:
            latent_dim: 潛在空間維度，預設 32
                        小型資料集（< 1000 筆）建議 16~32
                        大型資料集（> 10000 筆）可用 64~128
        """
        super().__init__()
        self.latent_dim = latent_dim

        # 卷積特徵提取層（4 個 Block，逐步縮小空間尺寸）
        self.conv_layers = nn.Sequential(
            # Block 1：提取低階特徵，保持解析度
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (1,32,32) -> (16,32,32)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Block 2：增加特徵深度，空間縮為 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (16,32,32) -> (32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # (32,32,32) -> (32,16,16)

            # Block 3：提取中階特徵，空間縮為 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (32,16,16) -> (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # (64,16,16) -> (64,8,8)

            # Block 4：提取高階特徵，空間縮為 4x4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # (64,8,8) -> (64,8,8)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # (64,8,8) -> (64,4,4)
        )

        # 全連接壓縮層：將 1024 維特徵向量壓縮到 latent_dim
        # 中間層 128 維作為過渡，避免直接壓縮導致資訊丟失過多
        self.fc = nn.Sequential(
            nn.Flatten(),                    # (64,4,4) -> 1024
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),                 # Dropout 比例提高至 0.3，加強正則化
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: 輸入封包影像 shape=(batch, 1, 32, 32)
        Returns:
            z: 潛在向量 shape=(batch, latent_dim)
        """
        features = self.conv_layers(x)
        z = self.fc(features)
        return z


class Decoder(nn.Module):
    """
    解碼器（Decoder）：將潛在向量重建回 32x32 封包影像

    架構（Encoder 的鏡像，使用轉置卷積上採樣）：
      FC(latent_dim -> 128 -> 1024) -> Reshape(64, 4, 4)
      Block 4->3: ConvTranspose(64->64, 2x2 stride=2) + BN + ReLU  -> (64, 8, 8)
      Block 3->2: ConvTranspose(64->32, 2x2 stride=2) + BN + ReLU  -> (32, 16, 16)
      Block 2->1: ConvTranspose(32->16, 2x2 stride=2) + BN + ReLU  -> (16, 32, 32)
      Output: ConvTranspose(16->1, 3x3) + Sigmoid                   -> (1, 32, 32)

    最後層使用 Sigmoid 確保輸出值域 [0, 1]，與輸入正規化後的範圍一致。
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()

        # 全連接展開層：從 latent_dim 還原成 4x4 特徵圖所需的維度
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64 * 4 * 4),   # 展開到 1024 維
            nn.ReLU(inplace=True),
        )

        # 轉置卷積重建層（ConvTranspose2d 的 stride=2 讓空間尺寸翻倍）
        self.deconv_layers = nn.Sequential(
            # Block 4->3：4x4 -> 8x8
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3->2：8x8 -> 16x16
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2->1：16x16 -> 32x32
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # 最終輸出：通道數回到 1，Sigmoid 限制輸出在 [0,1]
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            z: 潛在向量 shape=(batch, latent_dim)
        Returns:
            x_hat: 重建影像 shape=(batch, 1, 32, 32)，值域 [0,1]
        """
        x = self.fc(z)
        x = x.view(-1, 64, 4, 4)       # 重新 Reshape：1D -> 3D 特徵圖
        x_hat = self.deconv_layers(x)
        return x_hat


class CNNAutoencoder(nn.Module):
    """
    CNN Autoencoder 完整模型

    訓練方式（非監督式學習）：
      只使用正常流量封包影像訓練，最小化重建誤差（MSE Loss）。
      模型學習「正常封包的典型特徵」，儲存在潛在向量 z 中。

    異常偵測原理：
      - 正常封包：模型見過類似結構，重建誤差低（MSE 小）
      - 攻擊封包：結構與正常不同，重建誤差高（MSE 大）
      - 設定閾值：重建誤差 > 閾值 -> 判定為異常

    小型資料集的限制：
      訓練樣本越少，模型越容易過度泛化（把攻擊也重建得很好）。
      建議至少使用 500 筆以上的正常流量進行訓練。
      使用 CIC-IDS2017 等真實資料集可以顯著提升偵測效果。

    參考：
      - https://github.com/L1aoXingyu/pytorch-beginner
      - https://www.kaggle.com/code/vikasg/autoencoder-anomaly-detection
    """

    def __init__(self, latent_dim: int = 32):
        """
        Args:
            latent_dim: 潛在空間維度（小型資料集建議 16~32）
        """
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor):
        """
        完整前向傳播：input -> encode -> decode -> output

        Args:
            x: 輸入影像 shape=(batch, 1, 32, 32)
        Returns:
            x_hat: 重建影像 shape=(batch, 1, 32, 32)
            z:     潛在向量 shape=(batch, latent_dim)
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """只執行編碼（推論時取得潛在向量）"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """只執行解碼"""
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        計算逐樣本的重建誤差（異常分數）

        對每個樣本計算像素級 MSE（對 C x H x W 維度取平均），
        返回一個一維向量，每個元素代表一個封包的異常分數。
        分數越高 -> 重建越差 -> 越可能是異常封包。

        Args:
            x: 輸入影像 shape=(batch, 1, 32, 32)
        Returns:
            errors: 每個樣本的 MSE shape=(batch,)
        """
        self.eval()
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            # dim=[1,2,3] 對通道、高度、寬度三個維度取平均
            errors = torch.mean((x - x_hat) ** 2, dim=[1, 2, 3])
        return errors

    def get_model_info(self) -> dict:
        """回傳模型基本資訊（參數量、大小）"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable    = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "latent_dim":       self.latent_dim,
            "total_params":     total_params,
            "trainable_params": trainable,
            "model_size_MB":    total_params * 4 / 1024 / 1024,
        }
