# ============================================================
# semi_supervised_cicddos2019.py
# 半監督式 CNN Autoencoder - CIC-DDoS2019 資料集
#
# 資料集：CIC-DDoS2019
#   下載：https://www.unb.ca/cic/datasets/ddos-2019.html
#   攻擊類型：DNS / LDAP / MSSQL / NetBIOS / NTP / UDP / SYN
#             TFTP Flood 等多種 DDoS 攻擊
#   特徵數：~80 個網路流量特徵
#
# 半監督策略：
#   Phase 1 (Pretrain)  - 僅用正常流量訓練重建
#   Phase 2 (Finetune)  - 混入少量標記攻擊樣本，使用 Margin Loss
#     L = α·MSE(x_n, x̂_n)  +  β·max(0, margin - MSE(x_a, x̂_a))
#
# 目錄放置：core/semi_supervised_cicddos2019.py
# 訓練指令：
#   python semi_supervised_cicddos2019.py \
#       --data-dir data/cicddos2019 \
#       --output   output/model_cicddos2019
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
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, List
from sklearn.preprocessing import MinMaxScaler
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
# 資料集預處理：CIC-DDoS2019
# ──────────────────────────────────────────────────────────

# CIC-DDoS2019 保留的核心特徵（基於資料集說明文件）
_CICDDOS2019_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
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
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
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


class CICDDoS2019Loader:
    """
    CIC-DDoS2019 資料集載入與前處理。

    Parameters
    ----------
    data_dir : str
        包含 CSV 檔案的目錄（訓練日: March 11 / 12, 2019）
    image_size : int
        輸出 2D 影像邊長（預設 32 → 32×32 = 1024 pixels）
    max_normal : int
        最大正常樣本數（None = 全部）
    max_attack : int
        最大攻擊樣本數（None = 全部）
    seed : int
        隨機種子
    """

    def __init__(self,
                 data_dir: str,
                 image_size: int = 32,
                 max_normal: int = 60000,
                 max_attack: int = 30000,
                 seed: int = 42):
        self.data_dir   = data_dir
        self.image_size = image_size
        self.max_normal = max_normal
        self.max_attack = max_attack
        self.seed       = seed
        self.scaler     = MinMaxScaler()
        self._features  = _CICDDOS2019_FEATURES

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        X_normal : np.ndarray  shape (N, 1, image_size, image_size)
        X_attack : np.ndarray  shape (M, 1, image_size, image_size)
        """
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, "**/*.csv"),
                                     recursive=True))
        if not csv_files:
            csv_files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(
                f"找不到 CSV 檔案，請確認路徑: {self.data_dir}\n"
                "下載網址: https://www.unb.ca/cic/datasets/ddos-2019.html"
            )

        print(f"[CICDDoS2019Loader] 找到 {len(csv_files)} 個 CSV 檔案")
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

        # 標籤欄位偵測
        label_col = self._detect_label_col(data.columns)
        print(f"[CICDDoS2019Loader] 標籤欄位: {label_col!r}  "
              f"總計 {len(data):,} 筆")
        print(f"  攻擊類型: {data[label_col].unique()[:8].tolist()}...")

        # 特徵欄位選取
        feat_cols = self._select_features(data.columns)
        X_all     = data[feat_cols].copy()

        # 清理 inf / NaN
        X_all.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_all.fillna(0.0, inplace=True)
        X_arr = X_all.values.astype(np.float32)
        np.clip(X_arr, -1e9, 1e9, out=X_arr)

        labels = data[label_col].astype(str)
        is_benign = labels.apply(lambda x: x.strip() in _BENIGN_LABELS)

        X_normal_raw = X_arr[is_benign.values]
        X_attack_raw = X_arr[~is_benign.values]

        print(f"  正常: {len(X_normal_raw):,} 筆  攻擊: {len(X_attack_raw):,} 筆")

        # 抽樣
        rng = np.random.default_rng(self.seed)
        if self.max_normal and len(X_normal_raw) > self.max_normal:
            idx = rng.choice(len(X_normal_raw), self.max_normal, replace=False)
            X_normal_raw = X_normal_raw[idx]
        if self.max_attack and len(X_attack_raw) > self.max_attack:
            idx = rng.choice(len(X_attack_raw), self.max_attack, replace=False)
            X_attack_raw = X_attack_raw[idx]

        # Min-Max 正規化（僅用正常樣本 fit）
        self.scaler.fit(X_normal_raw)
        X_normal_scaled = self.scaler.transform(X_normal_raw)
        X_attack_scaled = self.scaler.transform(X_attack_raw)

        X_normal_img = features_to_image(X_normal_scaled, self.image_size)
        X_attack_img = features_to_image(X_attack_scaled, self.image_size)

        print(f"[CICDDoS2019Loader] 輸出形狀  正常: {X_normal_img.shape}  "
              f"攻擊: {X_attack_img.shape}")
        return X_normal_img, X_attack_img

    def _detect_label_col(self, columns) -> str:
        candidates = ["Label", "label", "Attack Type", "Class", "class"]
        for c in candidates:
            if c in columns:
                return c
        # 模糊匹配
        for col in columns:
            if "label" in col.lower() or "attack" in col.lower():
                return col
        raise ValueError(f"找不到標籤欄位，現有欄位: {list(columns)[:10]}")

    def _select_features(self, columns) -> List[str]:
        cols_clean = {c.strip(): c for c in columns}
        selected   = []
        for feat in self._features:
            key = feat.strip()
            if key in cols_clean:
                selected.append(cols_clean[key])
        if len(selected) < 10:
            # 回退：使用所有數值欄位
            numeric_cols = [c for c in columns
                            if c.strip() not in ("Label", "label", "Attack Type",
                                                  "Timestamp", "timestamp",
                                                  "Src IP", "Dst IP",
                                                  "Src Port", "Dst Port")]
            selected = numeric_cols[:len(_CICDDOS2019_FEATURES)]
            print(f"  [警告] 特徵欄位匹配不足，回退使用前 {len(selected)} 個欄位")
        print(f"  使用特徵數: {len(selected)}")
        return selected


# ──────────────────────────────────────────────────────────
# 模型本體：CICDDoS2019 半監督 CNN Autoencoder
# ──────────────────────────────────────────────────────────

class SemiSupervisedAE_DDoS2019(nn.Module):
    """
    CIC-DDoS2019 專用半監督 CNN Autoencoder。

    架構與 CNNAutoencoder 相同，但提供：
      - 額外的 bottleneck BN（改善潛在空間分佈）
      - get_reconstruction_error() 方法（供 Trainer 呼叫）
    """

    def __init__(self, latent_dim: int = 32, image_size: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        self.encoder = Encoder(latent_dim, image_size)
        self.decoder = Decoder(latent_dim, image_size)
        # Bottleneck BatchNorm：穩定潛在空間
        self.bottleneck_bn = nn.BatchNorm1d(latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if z.shape[0] > 1:          # BN 需要 batch_size > 1
            z = self.bottleneck_bn(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        x_hat, _ = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=[1, 2, 3])


# ──────────────────────────────────────────────────────────
# 半監督訓練器
# ──────────────────────────────────────────────────────────

class SemiSupervisedTrainer_DDoS2019:
    """
    CIC-DDoS2019 兩階段半監督訓練器。

    Phase 1 — 預訓練（僅正常流量 MSE 重建）
    Phase 2 — 微調（加入標記攻擊樣本的 Margin Loss）

    Loss（Phase 2）：
        L = α · MSE(x_normal, x̂_normal)
          + β · max(0, margin - MSE(x_attack, x̂_attack))

    Parameters
    ----------
    config : dict
        latent_dim      : int   (預設 32)
        image_size      : int   (預設 32)
        pretrain_epochs : int   (預設 80)
        finetune_epochs : int   (預設 40)
        batch_size      : int   (預設 32)
        pretrain_lr     : float (預設 1e-3)
        finetune_lr     : float (預設 5e-4)
        attack_ratio    : float (每批次攻擊樣本比例, 預設 0.2)
        alpha           : float (重建損失權重, 預設 1.0)
        beta            : float (邊距損失權重, 預設 0.5)
        margin          : float (攻擊分離邊距, 預設 0.05)
        percentile      : float (閾值百分位, 預設 95.0)
    output_dir : str
    """

    def __init__(self, config: dict,
                 output_dir: str = "output/model_cicddos2019"):
        self.config     = config
        self.output_dir = output_dir
        self.device     = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        os.makedirs(output_dir, exist_ok=True)

        self.model = SemiSupervisedAE_DDoS2019(
            latent_dim = config.get("latent_dim", 32),
            image_size = config.get("image_size", 32),
        ).to(self.device)

        self.threshold:       Optional[float] = None
        self._pretrain_losses: list = []
        self._finetune_losses: list = []

    # ── Phase 1：預訓練 ────────────────────────────────────
    def pretrain(self, X_normal: np.ndarray) -> list:
        """純重建損失訓練（僅正常流量）"""
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

        self._X_train_normal = X_train  # ── [P2-1 修正] 保存訓練子集 ──

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train)),
            batch_size=bs, shuffle=True, drop_last=False,
        )
        opt  = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.MSELoss()

        print(f"\n[DDoS2019 Phase 1] 預訓練 {epochs} epochs  LR={lr}  "
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
                ep_loss += loss.item() * len(batch)
            sched.step()
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

    # ── Phase 2：微調（半監督 Margin Loss）─────────────────
    def finetune(self,
                 X_normal: np.ndarray,
                 X_attack: np.ndarray) -> list:
        """加入攻擊樣本的 Margin Loss 微調"""
        epochs     = self.config.get("finetune_epochs", 40)
        lr         = self.config.get("finetune_lr", 5e-4)
        bs         = self.config.get("batch_size", 32)
        atk_ratio  = self.config.get("attack_ratio", 0.2)
        alpha      = self.config.get("alpha", 1.0)
        beta       = self.config.get("beta", 0.5)
        margin     = self.config.get("margin", 0.05)

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

        opt  = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.MSELoss()

        print(f"\n[DDoS2019 Phase 2] 微調 {epochs} epochs  "
              f"α={alpha} β={beta} margin={margin}  LR={lr}")
        self.model.train()
        losses = []

        iter_a = iter(loader_a)
        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            for (batch_n,) in loader_n:
                # 攻擊批次（迴圈取用）
                try:
                    (batch_a,) = next(iter_a)
                except StopIteration:
                    iter_a = iter(loader_a)
                    (batch_a,) = next(iter_a)

                batch_n = batch_n.to(self.device)
                batch_a = batch_a.to(self.device)

                # 正常重建損失
                xh_n, _  = self.model(batch_n)
                loss_n   = crit(xh_n, batch_n)

                # 攻擊 Margin 損失
                xh_a, _  = self.model(batch_a)
                err_a    = torch.mean((batch_a - xh_a) ** 2, dim=[1, 2, 3])
                loss_a   = torch.mean(torch.clamp(margin - err_a, min=0.0))

                loss = alpha * loss_n + beta * loss_a
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

    # ── 完整訓練流程 ──────────────────────────────────────
    def train_full(self,
                   X_normal: np.ndarray,
                   X_attack: np.ndarray):
        """
        執行完整兩階段訓練，並設定偵測閾值、儲存模型。
        """
        t_start = time.time()
        self.pretrain(X_normal)
        # ── [P2-1 修正] 僅傳入訓練子集的正常樣本 ──
        self.finetune(self._X_train_normal, X_attack)
        train_time = time.time() - t_start

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
        print(f"\n[DDoS2019] 閾值 ({pct}th pct): {self.threshold:.6f}  "
              f"總訓練時間: {train_time:.1f}s")

        self._save(train_time)

    def _save(self, train_time: float = 0.0):
        path = os.path.join(self.output_dir,
                            "semi_supervised_cicddos2019.pt")
        torch.save({
            "model_state":      self.model.state_dict(),
            "config":           self.config,
            "threshold":        self.threshold,
            "pretrain_losses":  self._pretrain_losses,
            "finetune_losses":  self._finetune_losses,
            "train_time":       train_time,
            "dataset":          "cicddos2019",
        }, path)
        print(f"[DDoS2019] 模型已儲存: {path}")

    @staticmethod
    def load(path: str, device: Optional[torch.device] = None):
        device = device or torch.device("cpu")
        ckpt   = torch.load(path, map_location=device)
        cfg    = ckpt["config"]
        model  = SemiSupervisedAE_DDoS2019(
            latent_dim = cfg.get("latent_dim", 32),
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
        description="半監督 CNN Autoencoder — CIC-DDoS2019"
    )
    parser.add_argument("--data-dir", required=True,
                        help="CIC-DDoS2019 CSV 目錄")
    parser.add_argument("--output",   default="output/model_cicddos2019")
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--latent",   type=int,   default=32)
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--margin",   type=float, default=0.05)
    parser.add_argument("--alpha",    type=float, default=1.0)
    parser.add_argument("--beta",     type=float, default=0.5)
    parser.add_argument("--pct",      type=float, default=95.0)
    parser.add_argument("--max-normal", type=int, default=60000)
    parser.add_argument("--max-attack", type=int, default=30000)
    args = parser.parse_args()

    # 1. 載入資料
    loader = CICDDoS2019Loader(
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

    # 2. 訓練
    config = {
        "latent_dim":      args.latent,
        "batch_size":      args.batch,
        "pretrain_epochs": args.pretrain_epochs,
        "finetune_epochs": args.finetune_epochs,
        "alpha":           args.alpha,
        "beta":            args.beta,
        "margin":          args.margin,
        "percentile":      args.pct,
    }
    trainer = SemiSupervisedTrainer_DDoS2019(config, output_dir=args.output)
    trainer.train_full(X_train_normal, X_finetune_attack)

    # 3. 評估
    result = evaluate(trainer.model, X_test_normal, X_test_attack,
                      trainer.threshold, device=trainer.device)
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "eval_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\n評估結果:", json.dumps(result, indent=2))
