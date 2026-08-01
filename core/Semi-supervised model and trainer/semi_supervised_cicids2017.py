# ============================================================
# semi_supervised_cicids2017.py
# 半監督式 CNN Autoencoder - CICIDS2017 資料集
#
# 資料集：CICIDS2017 (Canadian Institute for Cybersecurity)
#   下載：https://www.unb.ca/cic/datasets/ids-2017.html
#   攻擊類型：
#     DoS Slowloris / Slowhttptest / Hulk / GoldenEye
#     DDoS、PortScan、Brute Force (FTP-Patator / SSH-Patator)
#     Web Attacks (XSS / SQL Injection / Brute Force)
#     Infiltration、Botnet (ARES)
#   特徵數：77 個網路流量特徵（CICFlowMeter 輸出）
#
# 半監督策略：
#   Phase 1 — Pretrain：僅正常流量，MSE 重建損失
#   Phase 2 — Finetune：雙 Dataloader 混合訓練
#     L = α·MSE(x_n, x̂_n)
#       + β·max(0, margin - MSE(x_a, x̂_a))
#       + γ·L_latent_push               ← 潛在空間分離正則項
#
#   潛在空間分離（L_latent_push）：
#     鼓勵攻擊樣本的潛在向量與正常樣本中心距離拉大，
#     提高特徵空間可分性，適合 CICIDS2017 多類別攻擊場景。
#
# 訓練指令：
#   python semi_supervised_cicids2017.py \
#       --data-dir data/cicids2017 \
#       --output   output/model_cicids2017
# ============================================================

from __future__ import annotations

import os
import glob
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, List
from sklearn.preprocessing import RobustScaler
import sys as _sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
_core_dir = os.path.dirname(_this_dir)
if _this_dir not in _sys.path:
    _sys.path.insert(0, _this_dir)
if _core_dir not in _sys.path:
    _sys.path.insert(0, _core_dir)

from cnn_autoencoder import Encoder, Decoder, features_to_image, evaluate
from training_common import EarlyStopping

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ──────────────────────────────────────────────────────────
# CICIDS2017 特徵定義
# ──────────────────────────────────────────────────────────

_CICIDS2017_FEATURES = [
    "Destination Port", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

_BENIGN_LABELS = {"BENIGN", "Benign", "benign", "Normal", "NORMAL"}

# 各攻擊類別權重（稀有攻擊賦予更高權重以平衡訓練）
_ATTACK_WEIGHTS = {
    "Infiltration":    3.0,
    "Web Attack":      2.5,
    "Botnet":          2.5,
    "DoS GoldenEye":   1.5,
    "DoS Slowloris":   1.5,
    "DoS Slowhttptest":1.5,
    "DoS Hulk":        1.0,
    "DDoS":            1.0,
    "FTP-Patator":     2.0,
    "SSH-Patator":     2.0,
    "PortScan":        1.0,
}


# ──────────────────────────────────────────────────────────
# 資料集載入器
# ──────────────────────────────────────────────────────────

class CICIDS2017Loader:
    """
    CICIDS2017 資料集載入與前處理。

    Parameters
    ----------
    data_dir : str
        含 CSV 的目錄（通常包含週一至週五的 CSV 檔案）
    image_size : int
        輸出影像邊長（預設 32）
    max_normal : int
        最大正常樣本數
    max_attack : int
        最大攻擊樣本數（加權抽樣，稀有攻擊多取）
    use_robust_scaler : bool
        使用 RobustScaler（對 CICIDS2017 的高異常值更穩健）
    seed : int
    """

    def __init__(self,
                 data_dir: str,
                 image_size: int = 32,
                 max_normal: int = 60000,
                 max_attack: int = 30000,
                 use_robust_scaler: bool = True,
                 seed: int = 42):
        self.data_dir    = data_dir
        self.image_size  = image_size
        self.max_normal  = max_normal
        self.max_attack  = max_attack
        self.seed        = seed
        self.scaler      = RobustScaler() if use_robust_scaler \
                           else __import__("sklearn.preprocessing",
                                           fromlist=["MinMaxScaler"]).MinMaxScaler()

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        csv_files = sorted(
            glob.glob(os.path.join(self.data_dir, "**/*.csv"), recursive=True)
        ) or sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))

        if not csv_files:
            raise FileNotFoundError(
                f"找不到 CSV，請確認路徑: {self.data_dir}\n"
                "下載: https://www.unb.ca/cic/datasets/ids-2017.html"
            )

        print(f"[CICIDS2017Loader] 找到 {len(csv_files)} 個 CSV")
        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                df.columns = df.columns.str.strip()
                dfs.append(df)
                print(f"  載入: {os.path.basename(f)}  ({len(df):,} 筆)")
            except Exception as e:
                print(f"  [警告] 無法載入 {f}: {e}")

        data = pd.concat(dfs, ignore_index=True)
        label_col = self._detect_label_col(data.columns)

        print(f"[CICIDS2017Loader] 標籤欄: {label_col!r}  "
              f"總計 {len(data):,} 筆")

        # 印出攻擊分佈
        attack_dist = data[label_col].value_counts()
        print(f"  樣本分佈:\n{attack_dist.to_string()}")

        feat_cols = self._select_features(data.columns)
        X_all     = data[feat_cols].copy()
        X_all.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_all.fillna(0.0, inplace=True)
        X_arr  = np.clip(X_all.values.astype(np.float32), -1e9, 1e9)
        labels = data[label_col].astype(str).str.strip()

        is_benign = labels.isin(_BENIGN_LABELS)
        X_normal_raw = X_arr[is_benign.values]
        X_attack_raw = X_arr[~is_benign.values]
        attack_labels_raw = labels[~is_benign.values].values

        print(f"  正常: {len(X_normal_raw):,}  攻擊: {len(X_attack_raw):,}")

        rng = np.random.default_rng(self.seed)

        # 正常樣本抽樣
        if self.max_normal and len(X_normal_raw) > self.max_normal:
            idx = rng.choice(len(X_normal_raw), self.max_normal, replace=False)
            X_normal_raw = X_normal_raw[idx]

        # 攻擊樣本加權抽樣（稀有類別多取）
        X_attack_raw = self._weighted_sample(
            X_attack_raw, attack_labels_raw, self.max_attack, rng
        )

        # RobustScaler 正規化 (fit 僅用正常)
        self.scaler.fit(X_normal_raw)
        X_n_scaled = np.clip(self.scaler.transform(X_normal_raw), 0.0, 1.0)
        X_a_scaled = np.clip(self.scaler.transform(X_attack_raw), 0.0, 1.0)

        X_n_img = features_to_image(X_n_scaled, self.image_size)
        X_a_img = features_to_image(X_a_scaled, self.image_size)

        print(f"[CICIDS2017Loader] 輸出  正常: {X_n_img.shape}  "
              f"攻擊: {X_a_img.shape}")
        return X_n_img, X_a_img

    def _weighted_sample(self,
                         X: np.ndarray,
                         labels: np.ndarray,
                         max_n: int,
                         rng) -> np.ndarray:
        if max_n is None or len(X) <= max_n:
            return X
        # 為每個樣本依類別設定抽樣權重
        weights = np.ones(len(X), dtype=np.float32)
        for lbl, w in _ATTACK_WEIGHTS.items():
            mask = np.array([lbl.lower() in l.lower() for l in labels])
            weights[mask] = w
        weights /= weights.sum()
        idx = rng.choice(len(X), size=max_n, replace=False, p=weights)
        return X[idx]

    def _detect_label_col(self, columns) -> str:
        for c in ["Label", "label", " Label"]:
            if c in columns:
                return c
        for col in columns:
            if "label" in col.lower():
                return col
        raise ValueError(f"找不到標籤欄位，現有: {list(columns)[:10]}")

    def _select_features(self, columns) -> List[str]:
        cols_map = {c.strip(): c for c in columns}
        selected = []
        for feat in _CICIDS2017_FEATURES:
            if feat.strip() in cols_map:
                selected.append(cols_map[feat.strip()])
        if len(selected) < 10:
            exclude = {"label", "timestamp", "src ip", "dst ip",
                       "src port", "dst port", "flow id"}
            selected = [c for c in columns
                        if c.strip().lower() not in exclude][:77]
            print(f"  [警告] 欄位匹配不足，回退使用 {len(selected)} 欄")
        print(f"  使用特徵數: {len(selected)}")
        return selected


# ──────────────────────────────────────────────────────────
# 模型本體：CICIDS2017 半監督 CNN Autoencoder
# ──────────────────────────────────────────────────────────

class SemiSupervisedAE_CICIDS2017(nn.Module):
    """
    CICIDS2017 專用半監督 CNN Autoencoder。

    差異化設計：
      - 加寬 encoder（ch=[24, 48, 96, 96]），強化多類別攻擊特徵萃取
      - Latent Dropout（訓練時隨機遮罩潛在向量，提升泛化）
      - 儲存 normal_centroid 用於 latent push 損失
    """

    def __init__(self,
                 latent_dim: int = 48,
                 image_size: int = 32,
                 latent_dropout: float = 0.2):
        super().__init__()
        self.latent_dim     = latent_dim
        self.image_size     = image_size

        # 加寬通道以應對 CICIDS2017 多元攻擊特徵
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1,  24, 3, padding=1), nn.BatchNorm2d(24),  nn.GELU(),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48),  nn.GELU(),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96),  nn.GELU(),
            nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96),  nn.GELU(),
            nn.AdaptiveMaxPool2d((4, 4)),
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 256), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim),
        )
        self.latent_drop = nn.Dropout(latent_dropout)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(),
            nn.Linear(256, 96 * 4 * 4), nn.GELU(),
        )
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(96, 96, 2, stride=2), nn.BatchNorm2d(96), nn.GELU(),
            nn.ConvTranspose2d(96, 48, 2, stride=2), nn.BatchNorm2d(48), nn.GELU(),
            nn.ConvTranspose2d(48, 24, 2, stride=2), nn.BatchNorm2d(24), nn.GELU(),
            nn.Upsample(size=(image_size, image_size),
                        mode="bilinear", align_corners=False),
            nn.Conv2d(24, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        # 正常樣本潛在中心（微調時計算，用於 latent push 損失）
        self.register_buffer("normal_centroid",
                             torch.zeros(latent_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_fc(self.encoder_conv(x))
        return self.latent_drop(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(-1, 96, 4, 4)
        return self.decoder_deconv(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        x_hat, _ = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=[1, 2, 3])

    def update_normal_centroid(self,
                               X_normal: np.ndarray,
                               device: torch.device,
                               batch_size: int = 256):
        """計算並更新正常流量潛在空間中心點"""
        self.eval()
        zs = []
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_normal)),
            batch_size=batch_size, shuffle=False
        )
        with torch.no_grad():
            for (b,) in loader:
                _, z = self.forward(b.to(device))
                zs.append(z.cpu())
        centroid = torch.cat(zs, dim=0).mean(dim=0)
        self.normal_centroid.copy_(centroid.to(device))
        self.train()


# ──────────────────────────────────────────────────────────
# 半監督訓練器
# ──────────────────────────────────────────────────────────

class SemiSupervisedTrainer_CICIDS2017:
    """
    CICIDS2017 兩階段半監督訓練器。

    Phase 2 Loss（含潛在空間分離項）：
        L = α · MSE(x_n, x̂_n)
          + β · max(0, margin - MSE(x_a, x̂_a))
          + γ · (-distance(z_a, centroid_n))  ← latent push

    Parameters
    ----------
    config : dict
        latent_dim       : int   (預設 48)
        image_size       : int   (預設 32)
        pretrain_epochs  : int   (預設 80)
        finetune_epochs  : int   (預設 50)
        batch_size       : int   (預設 32)
        pretrain_lr      : float (預設 1e-3)
        finetune_lr      : float (預設 3e-4)
        attack_ratio     : float (預設 0.25)
        alpha            : float (重建損失, 預設 1.0)
        beta             : float (Margin 損失, 預設 0.6)
        gamma            : float (Latent push, 預設 0.1)
        margin           : float (預設 0.05)
        percentile       : float (閾值百分位, 預設 95.0)
    """

    def __init__(self, config: dict,
                 output_dir: str = "output/model_cicids2017"):
        self.config     = config
        self.output_dir = output_dir
        self.device     = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        os.makedirs(output_dir, exist_ok=True)

        self.model = SemiSupervisedAE_CICIDS2017(
            latent_dim = config.get("latent_dim", 48),
            image_size = config.get("image_size", 32),
        ).to(self.device)

        self.threshold:        Optional[float] = None
        self._pretrain_losses: list = []
        self._finetune_losses: list = []

    # ── Phase 1 ───────────────────────────────────────────
    def pretrain(self, X_normal: np.ndarray) -> list:
        epochs = self.config.get("pretrain_epochs", 80)
        lr     = self.config.get("pretrain_lr", 1e-3)
        bs     = self.config.get("batch_size", 32)
        val_split = self.config.get("val_split", 0.0)

        # [Fix #11] 可選 val split + Early Stopping
        if val_split > 0:
            n_val = max(1, int(len(X_normal) * val_split))
            indices = np.random.default_rng(42).permutation(len(X_normal))
            X_train = X_normal[indices[n_val:]]
            X_val   = X_normal[indices[:n_val]]
            val_tensor = torch.from_numpy(X_val).to(self.device)
            early_stop = EarlyStopping(
                patience=self.config.get("es_patience", 15),
                path=os.path.join(self.output_dir, "best_pretrain.pt")
            )
        else:
            X_train = X_normal
            early_stop = None

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train)),
            batch_size=bs, shuffle=True, drop_last=False,
        )
        opt   = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                   weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, steps_per_epoch=len(loader), epochs=epochs
        )
        crit  = nn.MSELoss()

        print(f"\n[CICIDS2017 Phase 1] 預訓練 {epochs} epochs  "
              f"裝置={self.device}")
        self.model.train()
        losses = []
        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            for (batch,) in loader:
                batch    = batch.to(self.device)
                x_hat, _ = self.model(batch)
                loss     = crit(x_hat, batch)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                ep_loss += loss.item() * len(batch)
            avg = ep_loss / len(X_train)
            losses.append(avg)
            if ep % max(1, epochs // 8) == 0:
                print(f"  [P1] Epoch {ep:4d}/{epochs}  Loss={avg:.6f}")

            # [Fix #11] 驗證 + Early Stopping
            if early_stop is not None:
                self.model.eval()
                with torch.no_grad():
                    val_hat, _ = self.model(val_tensor)
                    val_loss = crit(val_hat, val_tensor).item()
                self.model.train()
                if early_stop(val_loss, self.model):
                    print(f"  [P1] Early Stopping at epoch {ep}")
                    early_stop.restore_best(self.model, persist=False)
                    break

        if early_stop is not None:
            early_stop.restore_best(self.model, persist=False)
        self.model.eval()
        self._pretrain_losses = losses
        return losses

    # ── Phase 2 ───────────────────────────────────────────
    def finetune(self,
                 X_normal: np.ndarray,
                 X_attack: np.ndarray) -> list:
        epochs    = self.config.get("finetune_epochs", 50)
        lr        = self.config.get("finetune_lr", 3e-4)
        bs        = self.config.get("batch_size", 32)
        atk_ratio = self.config.get("attack_ratio", 0.25)
        alpha     = self.config.get("alpha",  1.0)
        beta      = self.config.get("beta",   0.6)
        gamma     = self.config.get("gamma",  0.1)
        margin    = self.config.get("margin", 0.05)

        # 計算正常樣本潛在中心
        print("  計算正常流量潛在中心 (centroid)...")
        self.model.update_normal_centroid(X_normal, self.device)
        centroid = self.model.normal_centroid.detach()

        n_atk = max(1, int(bs * atk_ratio))
        n_nrm = bs - n_atk

        loader_n = DataLoader(
            TensorDataset(torch.from_numpy(X_normal)),
            batch_size=n_nrm, shuffle=True, drop_last=True,
        )
        loader_a = DataLoader(
            TensorDataset(torch.from_numpy(X_attack)),
            batch_size=n_atk, shuffle=True, drop_last=True,
        )

        opt   = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                   weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit  = nn.MSELoss()

        print(f"\n[CICIDS2017 Phase 2] 微調 {epochs} epochs  "
              f"α={alpha} β={beta} γ={gamma} margin={margin}")
        self.model.train()
        losses = []
        iter_a = iter(loader_a)

        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            for (batch_n,) in loader_n:
                try:
                    (batch_a,) = next(iter_a)
                except StopIteration:
                    iter_a = iter(loader_a)
                    (batch_a,) = next(iter_a)

                batch_n = batch_n.to(self.device)
                batch_a = batch_a.to(self.device)

                # 正常重建損失
                xh_n, _  = self.model(batch_n)
                loss_recon = crit(xh_n, batch_n)

                # 攻擊 Margin 損失
                xh_a, z_a = self.model(batch_a)
                err_a     = torch.mean((batch_a - xh_a) ** 2, dim=[1, 2, 3])
                loss_margin = torch.mean(torch.clamp(margin - err_a, min=0.0))

                # 潛在空間推離損失（攻擊潛在向量遠離正常中心）
                dist_to_center = torch.norm(z_a - centroid.unsqueeze(0), dim=1)
                loss_push = -torch.mean(dist_to_center)  # 最大化距離

                loss = alpha * loss_recon + beta * loss_margin + gamma * loss_push
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item()

            sched.step()
            avg = ep_loss / max(1, len(loader_n))
            losses.append(avg)
            if ep % max(1, epochs // 8) == 0:
                print(f"  [P2] Epoch {ep:4d}/{epochs}  Loss={avg:.6f}")

        self.model.eval()
        self._finetune_losses = losses
        return losses

    # ── 完整訓練 ──────────────────────────────────────────
    def train_full(self,
                   X_normal: np.ndarray,
                   X_attack: np.ndarray):
        t0 = time.time()
        self.pretrain(X_normal)
        self.finetune(X_normal, X_attack)
        train_time = time.time() - t0

        # 設定閾值 (分批處理，避免 GPU OOM)
        pct = self.config.get("percentile", 95.0)
        batch_size = self.config.get("batch_size", 256)
        self.model.eval()
        errs_list = []
        with torch.no_grad():
            for i in range(0, len(X_normal), batch_size):
                batch = torch.from_numpy(X_normal[i:i + batch_size]).to(self.device)
                errs_list.append(self.model.reconstruction_error(batch).cpu().numpy())
        errs = np.concatenate(errs_list)
        self.threshold = float(np.percentile(errs, pct))
        print(f"\n[CICIDS2017] 閾值 ({pct}th pct): {self.threshold:.6f}  "
              f"訓練時間: {train_time:.1f}s")

        self._save(train_time)

    def _save(self, train_time: float = 0.0):
        path = os.path.join(self.output_dir, "semi_supervised_cicids2017.pt")
        torch.save({
            "model_state":     self.model.state_dict(),
            "config":          self.config,
            "threshold":       self.threshold,
            "pretrain_losses": self._pretrain_losses,
            "finetune_losses": self._finetune_losses,
            "train_time":      train_time,
            "dataset":         "cicids2017",
        }, path)
        print(f"[CICIDS2017] 模型已儲存: {path}")

    @staticmethod
    def load(path: str, device: Optional[torch.device] = None):
        device = device or torch.device("cpu")
        ckpt   = torch.load(path, map_location=device)
        cfg    = ckpt["config"]
        model  = SemiSupervisedAE_CICIDS2017(
            latent_dim = cfg.get("latent_dim", 48),
            image_size = cfg.get("image_size", 32),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt.get("threshold"), cfg


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="半監督 CNN Autoencoder — CICIDS2017"
    )
    parser.add_argument("--data-dir", required=True,
                        help="CICIDS2017 CSV 目錄")
    parser.add_argument("--output",   default="output/model_cicids2017")
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--latent",   type=int,   default=48)
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--margin",   type=float, default=0.05)
    parser.add_argument("--alpha",    type=float, default=1.0)
    parser.add_argument("--beta",     type=float, default=0.6)
    parser.add_argument("--gamma",    type=float, default=0.1)
    parser.add_argument("--pct",      type=float, default=95.0)
    parser.add_argument("--max-normal", type=int, default=60000)
    parser.add_argument("--max-attack", type=int, default=30000)
    args = parser.parse_args()

    loader = CICIDS2017Loader(
        data_dir   = args.data_dir,
        max_normal = args.max_normal,
        max_attack = args.max_attack,
    )
    X_normal, X_attack = loader.load()

    import numpy as np
    rng = np.random.default_rng(42)
    idx_n = rng.permutation(len(X_normal))
    split_n = int(len(X_normal) * 0.8)
    X_train_normal = X_normal[idx_n[:split_n]]
    X_test_normal  = X_normal[idx_n[split_n:]]

    idx_a = rng.permutation(len(X_attack))
    split_a = int(len(X_attack) * 0.5)
    X_finetune_attack = X_attack[idx_a[:split_a]]
    X_test_attack     = X_attack[idx_a[split_a:]]

    config = {
        "latent_dim":      args.latent,
        "batch_size":      args.batch,
        "pretrain_epochs": args.pretrain_epochs,
        "finetune_epochs": args.finetune_epochs,
        "alpha":           args.alpha,
        "beta":            args.beta,
        "gamma":           args.gamma,
        "margin":          args.margin,
        "percentile":      args.pct,
    }
    trainer = SemiSupervisedTrainer_CICIDS2017(config, output_dir=args.output)
    trainer.train_full(X_train_normal, X_finetune_attack)

    result = evaluate(trainer.model, X_test_normal, X_test_attack,
                      trainer.threshold, device=trainer.device)
    with open(os.path.join(args.output, "eval_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\n評估結果:", json.dumps(result, indent=2))
