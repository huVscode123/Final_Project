# ============================================================
# cnn_autoencoder.py
# 非監督式 CNN Autoencoder - 異常偵測基礎模型
#
# 策略：純非監督式學習（Unsupervised）
#   - 僅使用正常流量訓練（重建正常樣本）
#   - 推論時：重建誤差 > 閾值 → 判定為攻擊
#
# 輸入格式：(B, 1, 32, 32)  ← 流量特徵 reshape 為 32×32 灰階影像
# 輸出格式：(B, 1, 32, 32)  ← 重建影像
#
# 訓練方式：
#   trainer = UnsupervisedTrainer(config, output_dir="output/unsup")
#   trainer.train(X_normal_npy_path)
# ============================================================

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────
# 模型本體：Encoder / Decoder / CNNAutoencoder
# ──────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """CNN 編碼器：影像 → 潛在向量"""

    def __init__(self, latent_dim: int = 32, image_size: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 自適應池化 → 固定輸出 4×4
            nn.AdaptiveMaxPool2d((4, 4)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class Decoder(nn.Module):
    """CNN 解碼器：潛在向量 → 重建影像"""

    def __init__(self, latent_dim: int = 32, image_size: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),   # 4→8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # 8→16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),   # 16→32
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(image_size, image_size),
                        mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 64, 4, 4)
        return self.deconv(x)


class CNNAutoencoder(nn.Module):
    """
    非監督式 CNN Autoencoder 主體

    用法：
        model = CNNAutoencoder(latent_dim=32)
        x_hat, z = model(x)              # x: (B,1,32,32)
        err = model.reconstruction_error(x)   # (B,) MSE 逐樣本
    """

    def __init__(self, latent_dim: int = 32, image_size: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.encoder = Encoder(latent_dim, image_size)
        self.decoder = Decoder(latent_dim, image_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """逐樣本重建均方誤差 (B,)"""
        self.eval()
        x_hat, _ = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=[1, 2, 3])

    def predict(self,
                X: np.ndarray,
                threshold: float,
                device: Optional[torch.device] = None) -> np.ndarray:
        """
        二元分類預測 (分批處理，避免 GPU OOM)。
        Returns: np.ndarray of int  (0=正常, 1=攻擊)
        """
        if device is None:
            device = next(self.parameters()).device
        
        batch_size = 256
        all_errs = []
        for i in range(0, len(X), batch_size):
            chunk = features_to_image(X[i:i + batch_size], self.image_size)
            imgs  = torch.from_numpy(chunk).to(device)
            with torch.no_grad():
                all_errs.append(self.reconstruction_error(imgs).cpu().numpy())
        
        errs = np.concatenate(all_errs)
        return (errs > threshold).astype(int)


# ──────────────────────────────────────────────────────────
# 非監督訓練器
# ──────────────────────────────────────────────────────────

class UnsupervisedTrainer:
    """
    純非監督式訓練器 - 只使用正常流量訓練 CNNAutoencoder。

    Parameters
    ----------
    config : dict
        latent_dim   : int   (預設 32)
        image_size   : int   (預設 32)
        batch_size   : int   (預設 32)
        epochs       : int   (預設 100)
        learning_rate: float (預設 1e-3)
        percentile   : float (閾值百分位, 預設 95.0)
    output_dir : str
        模型與曲線的輸出目錄
    """

    def __init__(self, config: dict, output_dir: str = "output/model_unsupervised"):
        self.config     = config
        self.output_dir = output_dir
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_dir, exist_ok=True)


        self.model = CNNAutoencoder(
            latent_dim = config.get("latent_dim",  32),
            image_size = config.get("image_size",  32),
        ).to(self.device)

        self.threshold   : Optional[float] = None
        self._train_losses: list = []

    # ── 資料載入 ─────────────────────────────────────────
    def load_data(self, X_normal: np.ndarray):
        """接受 numpy array (N, F) 或 (N, H, W) 或 .npy 路徑"""
        if isinstance(X_normal, str):
            X_normal = np.load(X_normal)
        self._X_normal = features_to_image(
            X_normal, self.config.get("image_size", 32)
        )
        print(f"[UnsupervisedTrainer] 正常樣本載入: {self._X_normal.shape}")

    # ── 訓練主流程 ────────────────────────────────────────
    def train(self) -> list:
        X     = torch.from_numpy(self._X_normal)
        loader = DataLoader(
            TensorDataset(X),
            batch_size = self.config.get("batch_size", 32),
            shuffle    = True,
            drop_last  = False,
        )

        epochs   = self.config.get("epochs", 100)
        lr       = self.config.get("learning_rate", 1e-3)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        criterion = nn.MSELoss()

        self.model.train()
        self._train_losses = []

        print(f"\n[UnsupervisedTrainer] 開始訓練 — 裝置: {self.device}  "
              f"Epochs: {epochs}  LR: {lr}")

        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            for (batch,) in loader:
                batch    = batch.to(self.device)
                x_hat, _ = self.model(batch)
                loss     = criterion(x_hat, batch)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item() * len(batch)
            scheduler.step()
            avg = ep_loss / len(X)
            self._train_losses.append(avg)
            if ep % max(1, epochs // 10) == 0:
                print(f"  Epoch {ep:4d}/{epochs}  Loss: {avg:.6f}")

        self.model.eval()
        self._set_threshold()
        self._save_model()
        return self._train_losses

    # ── 閾值設定 ──────────────────────────────────────────
    def _set_threshold(self):
        # 設定閾值 (分批處理，避免 GPU OOM)
        pct = self.config.get("percentile", 95.0)
        batch_size = self.config.get("batch_size", 256)
        self.model.eval()
        errs_list = []
        with torch.no_grad():
            for i in range(0, len(self._X_normal), batch_size):
                batch = torch.from_numpy(self._X_normal[i:i + batch_size]).to(self.device)
                errs_list.append(self.model.reconstruction_error(batch).cpu().numpy())
        errs = np.concatenate(errs_list)
        self.threshold = float(np.percentile(errs, pct))
        print(f"[UnsupervisedTrainer] 閾值設定 ({pct}th percentile): "
              f"{self.threshold:.6f}")

    # ── 儲存 ──────────────────────────────────────────────
    def _save_model(self):
        path = os.path.join(self.output_dir, "cnn_autoencoder_unsupervised.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "config":      self.config,
            "threshold":   self.threshold,
            "train_losses":self._train_losses,
        }, path)
        print(f"[UnsupervisedTrainer] 模型已儲存: {path}")

    @staticmethod
    def load(path: str, device: Optional[torch.device] = None):
        """載入已儲存的模型"""
        device = device or torch.device("cpu")
        ckpt   = torch.load(path, map_location=device)
        cfg    = ckpt["config"]
        model  = CNNAutoencoder(
            latent_dim = cfg.get("latent_dim", 32),
            image_size = cfg.get("image_size", 32),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt.get("threshold"), cfg


# ──────────────────────────────────────────────────────────
# 評估工具
# ──────────────────────────────────────────────────────────

def evaluate(model: CNNAutoencoder,
             X_normal: np.ndarray,
             X_attack: np.ndarray,
             threshold: float,
             device: Optional[torch.device] = None) -> dict:
    """快速評估模型效能（Precision / Recall / F1 / AUC）"""
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

    device = device or next(model.parameters()).device
    model.eval()

    def _get_errors(X):
        # 改為分批，避免 GPU OOM
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


# ──────────────────────────────────────────────────────────
# CLI 快速訓練入口
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="非監督 CNN Autoencoder 訓練")
    parser.add_argument("--normal",  required=True,  help="正常流量 .npy 路徑")
    parser.add_argument("--attack",  default=None,   help="攻擊流量 .npy 路徑（僅評估用）")
    parser.add_argument("--output",  default="output/unsupervised")
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--latent",  type=int,   default=32)
    parser.add_argument("--batch",   type=int,   default=32)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--pct",     type=float, default=95.0,
                        help="閾值百分位數（預設 95）")
    args = parser.parse_args()

    config = {
        "latent_dim":    args.latent,
        "batch_size":    args.batch,
        "epochs":        args.epochs,
        "learning_rate": args.lr,
        "percentile":    args.pct,
    }

    trainer = UnsupervisedTrainer(config, output_dir=args.output)
    trainer.load_data(np.load(args.normal))
    trainer.train()

    if args.attack:
        X_n = np.load(args.normal)
        X_a = np.load(args.attack)
        result = evaluate(trainer.model, X_n, X_a, trainer.threshold,
                          device=trainer.device)
        with open(os.path.join(args.output, "eval_result.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
