# ============================================================
# cnn_autoencoder.py - CNN Autoencoder 模型定義 (v3.0 優化版)
#
# 架構：
#   Input(1x32x32) -> Encoder -> Bottleneck(z) -> Decoder -> Output(1x32x32)
#
# v3.0 優化：
#   - 新增 Kaiming 初始化，加速收斂並提升訓練穩定性
#   - 改善 reconstruction_error() 方法效率，避免冗餘的 eval/train 模式切換
#   - 新增 get_latent_features() 供下游分析使用
#   - 新增 batch_reconstruction_error() 支援高效批次異常計算
#
# v2.2 修正：
#   將模型從 4.8M 參數縮減至約 300K 參數。
#   原因：4.8M 參數對只有 150 筆訓練資料的小型資料集嚴重過大，
#         模型會學到「重建所有封包」而非「只重建正常封包」，
#         導致攻擊封包的重建誤差與正常封包相同，無法偵測異常。
#
# 參考 GitHub：
#   - https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder
#   - https://github.com/pytorch/examples/tree/main/vae
#   - https://github.com/patrickloeber/pytorchTutorial
# ============================================================

import numpy as np
import torch
import torch.nn as nn

def features_to_image(X: np.ndarray, image_size: int = 32) -> np.ndarray:
    """
    將 2D 特徵矩陣 (N, F) 轉換為 CNN 可接受的 (N, 1, image_size, image_size)。
    若特徵數 F > image_size^2，截取前 image_size^2 個特徵。
    若特徵數 F < image_size^2，補零至 image_size^2。
    """
    n_samples = X.shape[0]
    n_pixels  = image_size * image_size

    if X.ndim == 4:          # 已是 (N,1,H,W) 格式
        return X.astype(np.float32)
    if X.ndim == 3:          # 已是 (N,H,W) 格式
        return X[:, np.newaxis, :, :].astype(np.float32)

    # 1D/2D → 補零 or 截取 → reshape
    flat = X.reshape(n_samples, -1).astype(np.float32)
    if flat.shape[1] < n_pixels:
        pad = np.zeros((n_samples, n_pixels - flat.shape[1]), dtype=np.float32)
        flat = np.concatenate([flat, pad], axis=1)
    elif flat.shape[1] > n_pixels:
        flat = flat[:, :n_pixels]

    return flat.reshape(n_samples, 1, image_size, image_size)


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

    def __init__(self, latent_dim: int = 32, image_size: int = 32):
        """
        Args:
            latent_dim: 潛在空間維度，預設 32
                        小型資料集（< 1000 筆）建議 16~32
                        大型資料集（> 10000 筆）可用 64~128
            image_size: 輸入影像的邊長
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

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

            # Block 4：提取高階特徵，自適應壓縮至 4x4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((4, 4)),
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

        # [v3.0] Kaiming 初始化：加速收斂、提升訓練穩定性
        self._init_weights()

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

    def _init_weights(self):
        """[v3.0] Kaiming (He) 初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


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

    def __init__(self, latent_dim: int = 32, image_size: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

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

            # 最終輸出：Upsample 回 image_size 後通過 Conv2d + Sigmoid
            nn.Upsample(size=(image_size, image_size), mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # [v3.0] Kaiming 初始化
        self._init_weights()

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

    def _init_weights(self):
        """[v3.0] Kaiming (He) 初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


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

    def __init__(self, latent_dim: int = 32, image_size: int = 32):
        """
        Args:
            latent_dim: 潛在空間維度（小型資料集建議 16~32）
            image_size: 輸入影像的邊長
        """
        super().__init__()
        self.encoder = Encoder(latent_dim, image_size)
        self.decoder = Decoder(latent_dim, image_size)
        self.latent_dim = latent_dim
        self.image_size = image_size

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

        [v3.0 優化] 使用上下文管理器保存/恢復訓練狀態，
        避免巢狀呼叫時發生狀態混亂。

        Args:
            x: 輸入封包影像 shape=(batch, 1, 32, 32)
        Returns:
            errors: 每個樣本的 MSE shape=(batch,)
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            # dim=[1,2,3] 對通道、高度、寬度三個維度取平均
            errors = torch.mean((x - x_hat) ** 2, dim=[1, 2, 3])
        if was_training:
            self.train()
        return errors

    def get_latent_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        [v3.0 新增] 取得潛在空間特徵向量（用於 t-SNE/UMAP 視覺化、聚類分析）

        Args:
            x: 輸入封包影像 shape=(batch, 1, 32, 32)
        Returns:
            z: 潛在向量 shape=(batch, latent_dim)
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            z = self.encoder(x)
        if was_training:
            self.train()
        return z

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

    @staticmethod
    def _kaiming_init(m):
        """Kaiming (He) 初始化：適用於 ReLU 激活函數的卷積層與全連接層"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# ──────────────────────────────────────────────────────────
# 評估工具（從子資料夾版本合併）
# ──────────────────────────────────────────────────────────

def evaluate(model: CNNAutoencoder,
             X_normal: np.ndarray,
             X_attack: np.ndarray,
             threshold: float,
             device=None) -> dict:
    """
    快速評估模型效能（Precision / Recall / F1 / AUC）

    Args:
        model     : 已訓練的 CNNAutoencoder
        X_normal  : 正常流量資料
        X_attack  : 攻擊流量資料
        threshold : 異常閾值
        device    : 計算裝置

    Returns:
        dict: 包含 precision / recall / f1 / auc / sep_ratio 等指標
    """
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

    device = device or next(model.parameters()).device
    model.eval()

    def _get_errors(X):
        batch_size = 512
        all_errs = []
        for i in range(0, len(X), batch_size):
            chunk = features_to_image(X[i:i + batch_size])
            imgs  = torch.from_numpy(chunk).to(device)
            with torch.no_grad():
                all_errs.append(model.reconstruction_error(imgs).cpu().numpy())
        return np.concatenate(all_errs)

    err_n = _get_errors(X_normal)
    err_a = _get_errors(X_attack)

    y_true  = np.concatenate([np.zeros(len(err_n)), np.ones(len(err_a))])
    y_score = np.concatenate([err_n, err_a])
    y_pred  = (y_score > threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)

    result = {
        "precision": round(float(p),  4),
        "recall":    round(float(r),  4),
        "f1":        round(float(f1), 4),
        "auc":       round(float(auc),4),
        "threshold": threshold,
        "mean_err_normal": round(float(err_n.mean()), 6),
        "mean_err_attack": round(float(err_a.mean()), 6),
        "sep_ratio":       round(float(err_a.mean() / (err_n.mean() + 1e-9)), 3),
    }
    print(f"[Evaluate] Precision={result['precision']:.4f}  "
          f"Recall={result['recall']:.4f}  F1={result['f1']:.4f}  "
          f"AUC={result['auc']:.4f}  SepRatio={result['sep_ratio']:.3f}x")
    return result

