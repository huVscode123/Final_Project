# ============================================================
# semi_supervised_nslkdd.py
# 半監督式 CNN Autoencoder - NSL-KDD 資料集
#
# 資料集：NSL-KDD (改良版 KDD Cup 1999)
#   下載：https://www.unb.ca/cic/datasets/nsl.html
#   訓練檔：KDDTrain+.txt
#   測試檔：KDDTest+.txt （可選）
#   攻擊類型：DoS / Probe / R2L / U2R（四大類）
#   特徵數：41 個（14 連續 + 3 類別 + 其餘二元 / 計數）
#
# 特徵工程：
#   - protocol_type / service / flag 三個類別欄位做 One-Hot
#   - 連續特徵做 Log1p 變換（處理重尾分佈）
#   - 輸出特徵向量 → 補零至 8×8 (=64) 或 16×16 (=256) 影像
#     （NSL-KDD 特徵較少，預設使用 image_size=16）
#
# 半監督策略：
#   Phase 1 — Pretrain：重建正常流量
#   Phase 2 — Finetune：Margin Loss + 攻擊類別感知損失
#     各攻擊類別（DoS/Probe/R2L/U2R）賦予不同 margin，
#     稀有攻擊（U2R/R2L）給予更大 margin，強化偵測。
#
# 訓練指令：
#   python semi_supervised_nslkdd.py \
#       --train data/nslkdd/KDDTrain+.txt \
#       --test  data/nslkdd/KDDTest+.txt  \
#       --output output/model_nslkdd
# ============================================================

from __future__ import annotations

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, Dict, List
from sklearn.preprocessing import MinMaxScaler
import sys as _sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
_core_dir = os.path.dirname(_this_dir)
if _this_dir not in _sys.path:
    _sys.path.insert(0, _this_dir)
if _core_dir not in _sys.path:
    _sys.path.insert(0, _core_dir)

from cnn_autoencoder import features_to_image, evaluate
from training_common import EarlyStopping

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ──────────────────────────────────────────────────────────
# NSL-KDD 特徵定義
# ──────────────────────────────────────────────────────────

# 41 個原始特徵名稱（對應 KDDTrain+.txt 的欄位順序）
_NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty_level",        # 最後兩欄（NSL-KDD 特有）
]

# Log1p 變換的連續特徵（重尾分佈）
_LOG1P_FEATURES = [
    "duration", "src_bytes", "dst_bytes",
    "wrong_fragment", "urgent", "hot",
    "num_compromised", "num_root", "num_file_creations",
    "count", "srv_count",
    "dst_host_count", "dst_host_srv_count",
]

# 類別特徵 → One-Hot
_CAT_FEATURES = ["protocol_type", "service", "flag"]

# 攻擊類別（NSL-KDD 標籤）→ 大類別
_ATTACK_CATEGORY = {
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS",
    "udpstorm": "DoS", "processtable": "DoS", "worm": "DoS",
    # Probe
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L",
    "phf": "R2L", "multihop": "R2L", "warezmaster": "R2L",
    "warezclient": "R2L", "spy": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R",
    "rootkit": "U2R", "perl": "U2R", "sqlattack": "U2R",
    "xterm": "U2R", "ps": "U2R",
}

# 各攻擊類別的 margin（稀有攻擊給更大 margin）
_CATEGORY_MARGIN = {
    "DoS":   0.04,
    "Probe": 0.05,
    "R2L":   0.08,   # 稀有
    "U2R":   0.10,   # 最稀有，最難偵測
}

_BENIGN_LABELS = {"normal", "Normal", "NORMAL"}


# ──────────────────────────────────────────────────────────
# 資料集載入器
# ──────────────────────────────────────────────────────────

class NSLKDDLoader:
    """
    NSL-KDD 資料集載入與特徵工程。

    Parameters
    ----------
    train_path : str
        KDDTrain+.txt 路徑
    test_path  : str | None
        KDDTest+.txt 路徑（可選，None 則不載入）
    image_size : int
        輸出影像大小（預設 16 → 16×16=256 pixels，大於特徵維度）
    use_test_as_attack : bool
        True → 使用測試集攻擊作為微調的攻擊樣本
    seed : int
    """

    def __init__(self,
                 train_path: str,
                 test_path: Optional[str] = None,
                 image_size: int = 16,
                 use_test_as_attack: bool = False,
                 seed: int = 42):
        self.train_path        = train_path
        self.test_path         = test_path
        self.image_size        = image_size
        self.use_test_as_attack = use_test_as_attack
        self.seed              = seed
        self.scaler            = MinMaxScaler()
        self._oh_encoders: Dict = {}   # One-Hot 編碼器（儲存類別值集合）
        self._feature_names: List[str] = []

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        X_normal : np.ndarray  (N, 1, image_size, image_size)
        X_attack : np.ndarray  (M, 1, image_size, image_size)
        attack_categories : np.ndarray  (M,)  字串陣列（"DoS"/"Probe"/"R2L"/"U2R"）
        """
        print(f"[NSLKDDLoader] 載入訓練集: {self.train_path}")
        df_train = self._read_file(self.train_path)
        print(f"  訓練集: {len(df_train):,} 筆")

        df_test = None
        if self.test_path and os.path.exists(self.test_path):
            df_test = self._read_file(self.test_path)
            print(f"  測試集: {len(df_test):,} 筆")

        # 攻擊類別
        df_train["attack_cat"] = df_train["label"].str.lower().map(
            lambda x: _ATTACK_CATEGORY.get(x, "Unknown")
            if x not in _BENIGN_LABELS else "Normal"
        )

        # 取出正常 / 攻擊
        is_normal_train = df_train["label"].str.lower() == "normal"
        df_normal = df_train[is_normal_train].copy()
        df_attack = df_train[~is_normal_train].copy()

        # 若使用測試集攻擊樣本補充
        if df_test is not None and self.use_test_as_attack:
            df_test["attack_cat"] = df_test["label"].str.lower().map(
                lambda x: _ATTACK_CATEGORY.get(x, "Unknown")
                if x not in _BENIGN_LABELS else "Normal"
            )
            df_attack_test = df_test[
                ~df_test["label"].str.lower().isin({"normal"})
            ].copy()
            df_attack = pd.concat([df_attack, df_attack_test],
                                   ignore_index=True)
            print(f"  加入測試集攻擊樣本後: {len(df_attack):,} 筆")

        attack_cats = df_attack["attack_cat"].values

        print(f"  正常: {len(df_normal):,}  攻擊: {len(df_attack):,}")
        cat_dist = pd.Series(attack_cats).value_counts()
        print(f"  攻擊類別分佈:\n{cat_dist.to_string()}")

        # 特徵工程
        X_normal_raw, X_attack_raw = self._engineer_features(df_normal, df_attack)

        X_n_img = features_to_image(X_normal_raw, self.image_size)
        X_a_img = features_to_image(X_attack_raw, self.image_size)

        print(f"[NSLKDDLoader] 輸出  正常: {X_n_img.shape}  "
              f"攻擊: {X_a_img.shape}")
        return X_n_img, X_a_img, attack_cats

    def _read_file(self, path: str) -> pd.DataFrame:
        """讀取 .txt 格式（逗號分隔，無標題行）"""
        try:
            df = pd.read_csv(path, header=None, names=_NSL_KDD_COLUMNS)
        except Exception:
            # 嘗試 ARFF 格式（舊版 NSL-KDD）
            df = pd.read_csv(path, comment="@", header=None,
                             names=_NSL_KDD_COLUMNS)
        # 去除 difficulty_level 欄
        if "difficulty_level" in df.columns:
            df.drop(columns=["difficulty_level"], inplace=True)
        return df

    def _engineer_features(self,
                            df_normal: pd.DataFrame,
                            df_attack: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        特徵工程：
          1. Log1p 對重尾連續特徵
          2. 類別特徵數值化（共用碼表）
          3. MinMaxScaler（fit 僅用正常）

        [修正 #1] 先合併正常+攻擊建立共用碼表，確保同一類別值
        （如 protocol_type="tcp"）在兩個子集中得到相同整數碼。
        原版各自獨立呼叫 pd.Categorical()，會因為各自看到不同
        類別子集而分配不同碼，導致模型看到語意錯位的特徵。
        """
        drop_cols = {"label", "attack_cat", "difficulty_level"}
        feat_cols = [c for c in df_normal.columns if c not in drop_cols]

        # [修正 #1] 建立共用碼表：合併兩個子集的類別全集
        combined = pd.concat([df_normal[feat_cols], df_attack[feat_cols]],
                             ignore_index=True)
        self._cat_categories = {}
        for col in _CAT_FEATURES:
            if col in combined.columns:
                self._cat_categories[col] = sorted(
                    combined[col].astype(str).unique()
                )

        def _prepare(df: pd.DataFrame) -> pd.DataFrame:
            out = df[feat_cols].copy()
            # Log1p 連續特徵
            for col in _LOG1P_FEATURES:
                if col in out.columns:
                    out[col] = np.log1p(pd.to_numeric(out[col],
                                                       errors="coerce")
                                         .fillna(0).clip(lower=0))
            # [修正 #1] 類別特徵數值化 — 使用共用碼表
            for col in _CAT_FEATURES:
                if col in out.columns and col in self._cat_categories:
                    out[col] = pd.Categorical(
                        out[col].astype(str),
                        categories=self._cat_categories[col]
                    ).codes
            # 全部轉 float
            out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            return out

        df_n_feat = _prepare(df_normal)
        df_a_feat = _prepare(df_attack)

        # One-Hot 在這裡已轉為 Categorical codes（整數），
        # 後續 MinMaxScaler 會把它們縮放到 [0,1]
        self.scaler.fit(df_n_feat.values.astype(np.float32))

        X_n = self.scaler.transform(df_n_feat.values.astype(np.float32))
        X_a = self.scaler.transform(df_a_feat.values.astype(np.float32))
        X_n = np.clip(X_n, 0.0, 1.0)
        X_a = np.clip(X_a, 0.0, 1.0)

        self._feature_names = list(df_n_feat.columns)
        return X_n.astype(np.float32), X_a.astype(np.float32)


# ──────────────────────────────────────────────────────────
# 模型本體：NSL-KDD 半監督 CNN Autoencoder
# ──────────────────────────────────────────────────────────

class SemiSupervisedAE_NSLKDD(nn.Module):
    """
    NSL-KDD 專用半監督 CNN Autoencoder。

    差異化設計（針對 NSL-KDD 的特性）：
      - image_size 預設 16（特徵少，小影像即可）
      - 輕量化架構（ch=[8, 16, 32, 32]），避免過擬合小資料集
      - Variational Bottleneck 選項（VAE-style）提升潛在空間分佈
      - 保留重建路徑的 skip-like BN，穩定梯度
    """

    def __init__(self,
                 latent_dim: int = 16,
                 image_size: int = 16,
                 variational: bool = False):
        super().__init__()
        self.latent_dim  = latent_dim
        self.image_size  = image_size
        self.variational = variational

        # ── Encoder ───────────────────────────────────────
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1,  8, 3, padding=1), nn.BatchNorm2d(8),  nn.LeakyReLU(0.1),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.1),
            nn.Conv2d(16,32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.Conv2d(32,32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.AdaptiveMaxPool2d((4, 4)),
        )
        flat_dim = 32 * 4 * 4  # = 512

        if variational:
            self.fc_mu      = nn.Linear(flat_dim, latent_dim)
            self.fc_logvar  = nn.Linear(flat_dim, latent_dim)
        else:
            self.encoder_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, 128), nn.LeakyReLU(0.1),
                nn.Dropout(0.2),
                nn.Linear(128, latent_dim),
            )

        # ── Decoder ───────────────────────────────────────
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, flat_dim),  nn.LeakyReLU(0.1),
        )
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.BatchNorm2d(16), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16,  8, 2, stride=2), nn.BatchNorm2d(8),  nn.LeakyReLU(0.1),
            nn.Upsample(size=(image_size, image_size),
                        mode="bilinear", align_corners=False),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        self.kld_weight = 1e-4    # KL Divergence 權重（variational=True 時生效）
        self._last_logvar = None  # [修正 #2] 快取 logvar，避免重複前向傳播

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_conv(x).flatten(start_dim=1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """[修正 #2] 移除死碼，encoder_fc 只呼叫一次，logvar 快取至 _last_logvar"""
        h = self._flatten(x)
        if self.variational:
            mu     = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            self._last_logvar = logvar
            return mu, logvar
        z = self.encoder_fc(h)          # 只算一次
        self._last_logvar = None
        return z, None

    def reparameterize(self,
                       mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(-1, 32, 4, 4)
        return self.decoder_deconv(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        if self.variational and logvar is not None:
            z     = self.reparameterize(mu, logvar)
            x_hat = self.decode(z)
            return x_hat, mu          # 回傳 mu 作為潛在表示
        x_hat = self.decode(mu)
        return x_hat, mu

    def kl_loss(self, mu: torch.Tensor,
                logvar: torch.Tensor) -> torch.Tensor:
        """KL Divergence 損失（variational=True 時使用）"""
        return -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        ) * self.kld_weight

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        x_hat, _ = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=[1, 2, 3])


# ──────────────────────────────────────────────────────────
# 半監督訓練器
# ──────────────────────────────────────────────────────────

class SemiSupervisedTrainer_NSLKDD:
    """
    NSL-KDD 兩階段半監督訓練器。

    特色：
      - 支援攻擊類別感知的差異化 Margin
        （DoS/Probe 用小 margin，R2L/U2R 用大 margin）
      - 支援 Variational Autoencoder (VAE) 模式

    Phase 2 Loss：
        L = α · (MSE + β_kl · KL)                   ← 正常重建
          + Σ_cat [ β · max(0, margin_cat - err_a) ] ← 類別感知 margin
          + γ · L_regularization                     ← L2 正則

    Parameters
    ----------
    config : dict
        latent_dim         : int   (預設 16)
        image_size         : int   (預設 16)
        variational        : bool  (預設 False)
        pretrain_epochs    : int   (預設 60)
        finetune_epochs    : int   (預設 40)
        batch_size         : int   (預設 64)
        pretrain_lr        : float (預設 1e-3)
        finetune_lr        : float (預設 5e-4)
        attack_ratio       : float (預設 0.3)
        alpha              : float (重建損失, 預設 1.0)
        beta               : float (Margin 損失, 預設 0.8)
        gamma              : float (L2 正則, 預設 0.01)
        use_category_margin: bool  (預設 True)
        default_margin     : float (預設 0.05)
        percentile         : float (預設 95.0)
    """

    def __init__(self, config: dict,
                 output_dir: str = "output/model_nslkdd"):
        self.config     = config
        self.output_dir = output_dir
        self.device     = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        os.makedirs(output_dir, exist_ok=True)

        self.model = SemiSupervisedAE_NSLKDD(
            latent_dim  = config.get("latent_dim",  16),
            image_size  = config.get("image_size",  16),
            variational = config.get("variational", False),
        ).to(self.device)

        self.threshold:        Optional[float] = None
        self._pretrain_losses: list = []
        self._finetune_losses: list = []

    # ── Phase 1 ───────────────────────────────────────────
    def pretrain(self, X_normal: np.ndarray) -> list:
        epochs = self.config.get("pretrain_epochs", 60)
        lr     = self.config.get("pretrain_lr", 1e-3)
        bs     = self.config.get("batch_size", 64)
        var    = self.config.get("variational", False)
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
        opt   = torch.optim.Adam(self.model.parameters(), lr=lr,
                                  weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs//3),
                                                  gamma=0.5)
        crit  = nn.MSELoss()

        print(f"\n[NSLKDD Phase 1] 預訓練 {epochs} epochs  "
              f"variational={var}  裝置={self.device}")
        self.model.train()
        losses = []

        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            for (batch,) in loader:
                batch    = batch.to(self.device)
                x_hat, z = self.model(batch)
                loss     = crit(x_hat, batch)

                # [修正 #2] 讀快取的 logvar，不再重複呼叫 encode()
                if var and self.model._last_logvar is not None:
                    loss = loss + self.model.kl_loss(z, self.model._last_logvar)

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
                    val_hat, val_z = self.model(val_tensor)
                    v_loss = crit(val_hat, val_tensor)
                    if var and self.model._last_logvar is not None:
                        v_loss = v_loss + self.model.kl_loss(val_z, self.model._last_logvar)
                    val_loss = v_loss.item()
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
                 X_attack: np.ndarray,
                 attack_cats: Optional[np.ndarray] = None) -> list:
        """
        Parameters
        ----------
        attack_cats : np.ndarray | None
            攻擊類別字串陣列（"DoS"/"Probe"/"R2L"/"U2R"），
            None 則全部使用 default_margin。
        """
        epochs     = self.config.get("finetune_epochs", 40)
        lr         = self.config.get("finetune_lr", 5e-4)
        bs         = self.config.get("batch_size", 64)
        atk_ratio  = self.config.get("attack_ratio", 0.3)
        alpha      = self.config.get("alpha",  1.0)
        beta       = self.config.get("beta",   0.8)
        gamma      = self.config.get("gamma",  0.01)
        use_cat_m  = self.config.get("use_category_margin", True)
        def_margin = self.config.get("default_margin", 0.05)

        n_atk = max(1, int(bs * atk_ratio))
        n_nrm = bs - n_atk

        # 建立攻擊樣本 Dataset（帶索引，方便取出類別）
        X_a_tensor = torch.from_numpy(X_attack)
        X_n_tensor = torch.from_numpy(X_normal)

        # 對攻擊樣本加入 index，用於取 category
        idx_a = torch.arange(len(X_attack))
        loader_a = DataLoader(
            TensorDataset(X_a_tensor, idx_a),
            batch_size=n_atk, shuffle=True, drop_last=True,
        )
        loader_n = DataLoader(
            TensorDataset(X_n_tensor),
            batch_size=n_nrm, shuffle=True, drop_last=True,
        )

        opt   = torch.optim.Adam(self.model.parameters(), lr=lr,
                                  weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit  = nn.MSELoss()

        print(f"\n[NSLKDD Phase 2] 微調 {epochs} epochs  "
              f"α={alpha} β={beta} γ={gamma}  "
              f"category_margin={use_cat_m}")
        self.model.train()
        losses = []
        iter_a = iter(loader_a)

        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            for (batch_n,) in loader_n:
                try:
                    batch_a, batch_idx = next(iter_a)
                except StopIteration:
                    iter_a = iter(loader_a)
                    batch_a, batch_idx = next(iter_a)

                batch_n = batch_n.to(self.device)
                batch_a = batch_a.to(self.device)

                # 正常重建損失
                xh_n, z_n = self.model(batch_n)
                loss_recon = crit(xh_n, batch_n)

                # 攻擊 Margin 損失（類別感知）
                xh_a, _  = self.model(batch_a)
                err_a    = torch.mean((batch_a - xh_a) ** 2, dim=[1, 2, 3])

                if use_cat_m and attack_cats is not None:
                    # 逐樣本取對應類別 margin
                    cats    = attack_cats[batch_idx.numpy()]
                    margins = torch.tensor(
                        [_CATEGORY_MARGIN.get(c, def_margin) for c in cats],
                        dtype=torch.float32, device=self.device
                    )
                    loss_margin = torch.mean(
                        torch.clamp(margins - err_a, min=0.0)
                    )
                else:
                    loss_margin = torch.mean(
                        torch.clamp(def_margin - err_a, min=0.0)
                    )

                # L2 正則（潛在向量範數）
                loss_reg = gamma * torch.mean(z_n.pow(2))

                loss = alpha * loss_recon + beta * loss_margin + loss_reg
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
                   X_attack: np.ndarray,
                   attack_cats: Optional[np.ndarray] = None):
        t0 = time.time()
        self.pretrain(X_normal)
        # ── [P2-1 修正] 僅傳入訓練子集的正常樣本 ──
        self.finetune(self._X_train_normal, X_attack, attack_cats)
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
        print(f"\n[NSLKDD] 閾值 ({pct}th pct): {self.threshold:.6f}  "
              f"訓練時間: {train_time:.1f}s")

        self._save(train_time)

    def _save(self, train_time: float = 0.0):
        path = os.path.join(self.output_dir, "semi_supervised_nslkdd.pt")
        torch.save({
            "model_state":     self.model.state_dict(),
            "config":          self.config,
            "threshold":       self.threshold,
            "pretrain_losses": self._pretrain_losses,
            "finetune_losses": self._finetune_losses,
            "train_time":      train_time,
            "dataset":         "nslkdd",
        }, path)
        print(f"[NSLKDD] 模型已儲存: {path}")

    @staticmethod
    def load(path: str, device: Optional[torch.device] = None):
        device = device or torch.device("cpu")
        ckpt   = torch.load(path, map_location=device)
        cfg    = ckpt["config"]
        model  = SemiSupervisedAE_NSLKDD(
            latent_dim  = cfg.get("latent_dim",  16),
            image_size  = cfg.get("image_size",  16),
            variational = cfg.get("variational", False),
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
        description="半監督 CNN Autoencoder — NSL-KDD"
    )
    parser.add_argument("--train",   required=True,
                        help="KDDTrain+.txt 路徑")
    parser.add_argument("--test",    default=None,
                        help="KDDTest+.txt 路徑（可選）")
    parser.add_argument("--output",  default="output/model_nslkdd")
    parser.add_argument("--image-size",      type=int,   default=16)
    parser.add_argument("--latent",          type=int,   default=16)
    parser.add_argument("--pretrain-epochs", type=int,   default=60)
    parser.add_argument("--finetune-epochs", type=int,   default=40)
    parser.add_argument("--batch",           type=int,   default=64)
    parser.add_argument("--alpha",           type=float, default=1.0)
    parser.add_argument("--beta",            type=float, default=0.8)
    parser.add_argument("--gamma",           type=float, default=0.01)
    parser.add_argument("--pct",             type=float, default=95.0)
    parser.add_argument("--variational",     action="store_true",
                        help="使用 VAE 模式")
    parser.add_argument("--use-test-attack", action="store_true",
                        help="加入測試集攻擊樣本到微調集")
    args = parser.parse_args()

    loader = NSLKDDLoader(
        train_path          = args.train,
        test_path           = args.test,
        image_size          = args.image_size,
        use_test_as_attack  = args.use_test_attack,
    )
    X_normal, X_attack, attack_cats = loader.load()

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

    if attack_cats is not None:
        attack_cats_finetune = [attack_cats[i] for i in idx_a[:split_a]]
    else:
        attack_cats_finetune = None

    config = {
        "latent_dim":          args.latent,
        "image_size":          args.image_size,
        "variational":         args.variational,
        "batch_size":          args.batch,
        "pretrain_epochs":     args.pretrain_epochs,
        "finetune_epochs":     args.finetune_epochs,
        "alpha":               args.alpha,
        "beta":                args.beta,
        "gamma":               args.gamma,
        "percentile":          args.pct,
        "use_category_margin": True,
    }
    trainer = SemiSupervisedTrainer_NSLKDD(config, output_dir=args.output)
    trainer.train_full(X_train_normal, X_finetune_attack, attack_cats=attack_cats_finetune)

    result = evaluate(trainer.model, X_test_normal, X_test_attack,
                      trainer.threshold, device=trainer.device)
    with open(os.path.join(args.output, "eval_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\n評估結果:", json.dumps(result, indent=2))
