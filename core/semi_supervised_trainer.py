# ============================================================
# semi_supervised_trainer.py - 半監督式 CNN Autoencoder 訓練模組
#
# 半監督式學習策略說明：
#   傳統非監督式 Autoencoder 只用正常流量訓練，模型學會重建正常封包，
#   但無法保證攻擊封包的重建誤差一定高於正常封包。
#
#   本模組採用「邊界損失（Margin Loss）+ 重建損失」的半監督方式：
#
#   ┌─────────────────────────────────────────────────────────┐
#   │  Phase 1：無監督預訓練（Unsupervised Pretraining）        │
#   │    - 只使用正常流量，最小化重建誤差（MSE Loss）            │
#   │    - 讓模型學會「正常封包的典型結構」                      │
#   │                                                         │
#   │  Phase 2：半監督微調（Semi-supervised Fine-tuning）       │
#   │    - 同時使用少量標記的攻擊流量                            │
#   │    - 損失函數 = α × L_recon_normal                       │
#   │              + β × max(0, margin - L_recon_attack)      │
#   │                                                         │
#   │    L_recon_normal：正常封包重建誤差（越低越好）             │
#   │    L_recon_attack ：攻擊封包重建誤差（越高越好，但受 margin）│
#   │    margin         ：希望攻擊誤差超過的目標值                │
#   └─────────────────────────────────────────────────────────┘
#
# 優點（相比純非監督）：
#   1. 攻擊封包的重建誤差被明確推高（Margin Loss）
#   2. 正常/攻擊誤差的分離度（Separability）更大
#   3. 閾值更容易設定，F1 Score 通常提升 5~15%
#   4. 只需少量標記攻擊樣本（通常 10~20% 即有效果）
#
# 執行方式：
#   python run_semi_supervised.py --dataset simulate
#   python run_semi_supervised.py --dataset cicids2017 --data-dir data/cicids2017
#
# 參考文獻：
#   - Ruff et al. (2020) "Deep Semi-Supervised Anomaly Detection" (ICLR 2020)
#   - Ye et al. (2021) "Unsupervised and Semi-supervised Anomaly Detection with LSTM"
# ============================================================

import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    DataLoader, TensorDataset, random_split, ConcatDataset
)

from cnn_autoencoder import CNNAutoencoder
from trainer import PacketDataset, EarlyStopping, DEFAULT_CONFIG
from training_common import compute_optimal_threshold_vectorized  # [修正 #6]


# ── 半監督訓練超參數預設值 ─────────────────────────────────
SEMI_DEFAULT_CONFIG = {
    **DEFAULT_CONFIG,
    # 半監督特有參數
    "pretrain_epochs":   50,     # Phase 1 無監督預訓練輪數
    "finetune_epochs":   50,     # Phase 2 半監督微調輪數
    "alpha":             1.0,    # 正常流量重建損失權重
    "beta":              0.5,    # 攻擊邊界損失權重
    "margin":            0.05,   # 邊界損失目標值（攻擊誤差希望超過此值）
    "attack_ratio":      0.2,    # 用於微調的標記攻擊樣本比例（相對攻擊總量）
    "contrastive_temp":  0.07,   # 對比學習溫度係數（選用）
}


class MarginLoss(nn.Module):
    """
    邊界損失（Margin Loss）

    目的：讓攻擊封包的重建誤差超過設定的 margin 值。

    數學式：
        L_margin = mean(max(0, margin - reconstruction_error_of_attack))

    直觀解釋：
        - 若攻擊樣本誤差 < margin：產生正梯度，推高誤差（懲罰模型重建太好）
        - 若攻擊樣本誤差 >= margin：損失為 0，不更新（已達目標）

    Args:
        margin: 希望攻擊誤差超過的目標值（建議設為正常誤差均值的 2~5 倍）
    """

    def __init__(self, margin: float = 0.05):
        super().__init__()
        self.margin = margin

    def forward(self, errors_attack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            errors_attack: 攻擊封包的重建誤差 shape=(N,)
        Returns:
            scalar: 平均邊界損失
        """

        loss = torch.clamp(self.margin - errors_attack, min=0.0)
        return loss.mean()


class SemiSupervisedDataLoader:
    """
    半監督訓練資料集管理器

    管理正常流量（大量，未標記）與攻擊流量（少量，已標記）的混合載入。

    Args:
        X_normal      : 正常流量影像矩陣 (N, 32, 32)
        X_attack      : 全部攻擊流量影像矩陣 (M, 32, 32)
        attack_ratio  : 用於微調的攻擊樣本比例（0~1）
        batch_size    : 批次大小
        val_split     : 驗證集比例
    """

    def __init__(self,
                 X_normal: np.ndarray,
                 X_attack: np.ndarray,
                 attack_ratio: float = 0.2,
                 batch_size: int = 32,
                 val_split: float = 0.2,
                 seed: int = 42):

        self.batch_size = batch_size
        rng = np.random.default_rng(seed)

        # ── 正常流量：用於 Phase 1 和 Phase 2 ──────────────
        normal_tensor = torch.from_numpy(
            X_normal.astype(np.float32)[:, np.newaxis]
        )  # (N, 1, 32, 32)

        n_total = len(X_normal)
        n_val   = max(1, int(n_total * val_split))
        n_train = n_total - n_val

        normal_ds = TensorDataset(normal_tensor)
        self.train_normal_ds, self.val_normal_ds = random_split(
            normal_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )

        # ── 攻擊流量：只取部分用於半監督微調 ────────────────
        n_attack_labeled = max(1, int(len(X_attack) * attack_ratio))
        attack_indices   = rng.choice(len(X_attack), n_attack_labeled, replace=False)
        X_attack_labeled = X_attack[attack_indices]

        attack_tensor = torch.from_numpy(
            X_attack_labeled.astype(np.float32)[:, np.newaxis]
        )  # (M', 1, 32, 32)

        self.attack_ds    = TensorDataset(attack_tensor)
        self.n_normal_train = n_train
        self.n_normal_val   = n_val
        self.n_attack_labeled = n_attack_labeled

        # [Bug 6 修正 — 2026] 保留「未被用於微調」的攻擊樣本（互補集合），
        # 供 compute_threshold(method="optimal") 使用，避免閾值搜尋
        # 直接在 margin loss 微調用過的同一批攻擊樣本上做 F1 最佳化
        # （屬於資料洩漏：閾值會對已看過的樣本過擬合）。
        holdout_mask = np.ones(len(X_attack), dtype=bool)
        holdout_mask[attack_indices] = False
        self.X_attack_holdout = X_attack[holdout_mask]

        print(f"  [SemiSupervisedDataLoader] 資料分配:")
        print(f"    正常流量 - 訓練: {n_train:,}，驗證: {n_val:,}")
        print(f"    攻擊流量（標記，用於微調）: {n_attack_labeled:,} / {len(X_attack):,} "
              f"（{attack_ratio*100:.0f}%）")
        print(f"    攻擊流量（保留，未參與微調，供閾值搜尋用）: "
              f"{len(self.X_attack_holdout):,}")

    def get_pretrain_loaders(self):
        """Phase 1：只用正常流量的 DataLoader"""
        train_loader = DataLoader(
            self.train_normal_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            self.val_normal_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=0
        )
        return train_loader, val_loader

    def get_finetune_loaders(self):
        """Phase 2：正常流量 DataLoader + 攻擊流量 DataLoader（分開，因損失不同）"""
        normal_loader = DataLoader(
            self.train_normal_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=0
        )
        attack_loader = DataLoader(
            self.attack_ds, batch_size=max(1, self.batch_size // 4),
            shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            self.val_normal_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=0
        )
        return normal_loader, attack_loader, val_loader


class SemiSupervisedTrainer:
    """
    半監督式 CNN Autoencoder 訓練器

    訓練流程：
        Phase 1：無監督預訓練
            使用正常流量訓練基礎 Autoencoder，
            讓模型先學習正常流量的重建能力。

        Phase 2：半監督微調
            引入少量已標記的攻擊樣本，
            使用組合損失函數微調模型，
            進一步拉大正常/攻擊流量的誤差差距。

    使用範例：
        trainer = SemiSupervisedTrainer(config)
        trainer.load_data(X_normal, X_attack)
        trainer.pretrain()     # Phase 1
        trainer.finetune()     # Phase 2
        trainer.plot_training_curve()
        threshold = trainer.compute_threshold(X_normal, X_attack)
    """

    def __init__(self,
                 config: dict = None,
                 output_dir: str = "output/model_semi",
                 model_name: str = ""):
        """
        Args:
            config    : 超參數字典
            output_dir: 輸出目錄
        """
        self.config = {**SEMI_DEFAULT_CONFIG, **(config or {})}
        self.output_dir = output_dir
        self.model_name = model_name
        os.makedirs(output_dir, exist_ok=True)

        # 檔名後綴處理（各資料集模型獨立存放）
        self.suffix = f"_{model_name}" if model_name else ""

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"  [SemiSupervisedTrainer] 使用裝置: {self.device}")
        if model_name:
            print(f"  [SemiSupervisedTrainer] 模型名稱: {model_name} (檔名後綴: {self.suffix})")
        print(f"  [SemiSupervisedTrainer] 訓練策略: Phase1({self.config['pretrain_epochs']} epochs) "
              f"+ Phase2({self.config['finetune_epochs']} epochs)")

        self.model = CNNAutoencoder(
            latent_dim=self.config["latent_dim"]
        ).to(self.device)

        info = self.model.get_model_info()
        print(f"  [SemiSupervisedTrainer] 模型參數: {info['total_params']:,} "
              f"({info['model_size_MB']:.2f} MB)")

        # 損失函數
        self.recon_criterion = nn.MSELoss()
        self.margin_loss     = MarginLoss(margin=self.config["margin"])

        # 記錄
        self.pretrain_losses  = []  # Phase 1 訓練損失
        self.finetune_normal  = []  # Phase 2 正常損失
        self.finetune_attack  = []  # Phase 2 攻擊損失
        self.finetune_total   = []  # Phase 2 總損失
        self.val_losses       = []  # 驗證損失（所有 phase）
        self.threshold        = None

        # 資料載入器（由 load_data 設定）
        self._data_manager    = None

    # ── 資料載入 ──────────────────────────────────────────
    def load_data(self, X_normal: np.ndarray, X_attack: np.ndarray):
        """
        載入正常與攻擊流量

        Args:
            X_normal: 正常流量影像矩陣 (N, 32, 32) float32
            X_attack: 攻擊流量影像矩陣 (M, 32, 32) float32
        """
        print(f"\n  [SemiSupervisedTrainer] 載入資料")
        print(f"    正常流量: {len(X_normal):,} 筆")
        print(f"    攻擊流量: {len(X_attack):,} 筆（只取 {self.config['attack_ratio']*100:.0f}% 用於微調）")

        self.X_normal = X_normal
        self.X_attack = X_attack

        self._data_manager = SemiSupervisedDataLoader(
            X_normal      = X_normal,
            X_attack      = X_attack,
            attack_ratio  = self.config["attack_ratio"],
            batch_size    = self.config["batch_size"],
            val_split     = self.config["val_split"],
        )
        # [Bug 6 修正 — 2026] 記錄「未參與微調」的攻擊樣本，供
        # compute_threshold(method="optimal") 做無洩漏的閾值搜尋。
        self.X_attack_holdout = self._data_manager.X_attack_holdout

    # ── Phase 1：無監督預訓練 ─────────────────────────────
    def pretrain(self):
        """
        Phase 1：無監督預訓練

        只使用正常流量，最小化重建誤差（MSE Loss）。
        目的：讓模型先學會重建正常封包的典型結構。
        """
        print(f"\n{'='*60}")
        print(f"  Phase 1：無監督預訓練（共 {self.config['pretrain_epochs']} epochs）")
        print(f"  損失函數：L = MSE(x, x_hat)  [只有正常流量]")
        print(f"{'='*60}")

        train_loader, val_loader = self._data_manager.get_pretrain_loaders()

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=self.config["lr_factor"],
            patience=self.config["lr_patience"],
        )

        model_path = os.path.join(self.output_dir, f"pretrained_model{self.suffix}.pt")
        early_stop = EarlyStopping(
            patience=self.config["patience"],
            min_delta=self.config["min_delta"],
            path=model_path,
        )

        print(f"  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} {'LR':<12} {'時間(s)'}")
        print("  " + "─" * 58)

        for epoch in range(1, self.config["pretrain_epochs"] + 1):
            t0 = time.time()

            # 訓練
            train_loss = self._train_one_epoch_unsupervised(train_loader, optimizer)
            val_loss   = self._validate_epoch(val_loader)

            self.pretrain_losses.append(train_loss)
            self.val_losses.append(val_loss)

            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]
            print(f"  {epoch:<8} {train_loss:<14.6f} {val_loss:<14.6f} "
                  f"{lr:<12.2e} {time.time()-t0:.1f}")

            if early_stop(val_loss, self.model):
                print(f"\n  [EarlyStopping] 第 {epoch} epoch 停止")
                break

        # 載入並儲存最佳預訓練模型 [修正 #4]
        early_stop.restore_best(self.model, persist=True)
        print(f"\n  Phase 1 完成！最佳驗證損失: {early_stop.best_loss:.6f}")
        print(f"  預訓練模型已儲存: {model_path}")

    # ── Phase 2：半監督微調 ───────────────────────────────
    def finetune(self):
        """
        Phase 2：半監督微調

        同時使用正常流量（重建損失）與少量標記攻擊流量（邊界損失）。

        組合損失函數：
            L_total = α × L_recon_normal + β × L_margin_attack

        其中：
            L_recon_normal = MSE(x_normal, x_hat_normal)
            L_margin_attack = mean(max(0, margin - MSE(x_attack, x_hat_attack)))

            α：正常流量損失權重（通常 1.0）
            β：攻擊邊界損失權重（通常 0.3~0.8，過大會破壞重建能力）
        """
        print(f"\n{'='*60}")
        print(f"  Phase 2：半監督微調（共 {self.config['finetune_epochs']} epochs）")
        print(f"  損失函數：L = {self.config['alpha']}×L_recon + {self.config['beta']}×L_margin")
        print(f"  邊界值（margin）: {self.config['margin']}")
        print(f"{'='*60}")

        normal_loader, attack_loader, val_loader = \
            self._data_manager.get_finetune_loaders()

        # Phase 2 使用較小的學習率（精調，避免破壞預訓練知識）
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"] * 0.1,
            weight_decay=self.config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config["finetune_epochs"],
            eta_min=1e-6,
        )

        model_path = os.path.join(self.output_dir, f"best_model{self.suffix}.pt")
        best_val = float("inf")
        no_improve = 0

        print(f"  {'Epoch':<8} {'Normal Loss':<14} {'Attack Loss':<14} "
              f"{'Total Loss':<14} {'Val Loss':<12}")
        print("  " + "─" * 68)

        for epoch in range(1, self.config["finetune_epochs"] + 1):
            t0 = time.time()

            # 半監督訓練
            loss_n, loss_a, loss_total = self._train_one_epoch_semi(
                normal_loader, attack_loader, optimizer
            )
            val_loss = self._validate_epoch(val_loader)

            self.finetune_normal.append(loss_n)
            self.finetune_attack.append(loss_a)
            self.finetune_total.append(loss_total)
            self.val_losses.append(val_loss)

            scheduler.step()

            print(f"  {epoch:<8} {loss_n:<14.6f} {loss_a:<14.6f} "
                  f"{loss_total:<14.6f} {val_loss:<12.6f} "
                  f"({time.time()-t0:.1f}s)")

            # 儲存最佳模型
            if val_loss < best_val - self.config["min_delta"]:
                best_val = val_loss
                no_improve = 0
                torch.save(self.model.state_dict(), model_path)
            else:
                no_improve += 1
                if no_improve >= self.config["patience"]:
                    print(f"\n  [EarlyStopping] 第 {epoch} epoch 停止")
                    break

        # 載入最佳微調模型
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        print(f"\n  Phase 2 完成！最佳驗證損失: {best_val:.6f}")
        print(f"  最佳模型已儲存: {model_path}")

    # ── 訓練輔助函數 ──────────────────────────────────────
    def _train_one_epoch_unsupervised(self, loader, optimizer) -> float:
        """Phase 1：純重建損失訓練"""
        self.model.train()
        total_loss = 0.0
        for (x,) in loader:
            x = x.to(self.device)
            x_hat, _ = self.model(x)
            loss = self.recon_criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        return total_loss / len(loader.dataset)

    def _train_one_epoch_semi(self, normal_loader, attack_loader, optimizer):
        """
        Phase 2：組合損失訓練

        因正常批次與攻擊批次大小不同，採用輪替方式：
        每個正常批次 + 對應一個攻擊批次（若攻擊批次用完則循環）
        """
        self.model.train()
        total_normal = 0.0
        total_attack = 0.0
        total_combined = 0.0
        n_batches = 0

        alpha = self.config["alpha"]
        beta  = self.config["beta"]

        # 攻擊批次迭代器（循環）
        attack_iter = iter(attack_loader)

        for (x_normal,) in normal_loader:
            x_normal = x_normal.to(self.device)

            # 取下一批攻擊樣本（循環）
            try:
                (x_attack,) = next(attack_iter)
            except StopIteration:
                attack_iter = iter(attack_loader)
                (x_attack,) = next(attack_iter)
            x_attack = x_attack.to(self.device)

            # 正常流量重建損失
            x_hat_n, _ = self.model(x_normal)
            loss_normal = self.recon_criterion(x_hat_n, x_normal)

            # 攻擊流量邊界損失
            # reconstruction_error() 有 no_grad，這裡需要手動計算（帶梯度）
            x_hat_a, _ = self.model(x_attack)
            errors_attack = torch.mean(
                (x_attack - x_hat_a) ** 2, dim=[1, 2, 3]
            )  # (batch,)
            loss_attack = self.margin_loss(errors_attack)

            # 組合損失
            loss_total = alpha * loss_normal + beta * loss_attack

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_normal   += loss_normal.item() * x_normal.size(0)
            total_attack   += loss_attack.item() * x_normal.size(0)
            total_combined += loss_total.item() * x_normal.size(0)
            n_batches      += x_normal.size(0)

        n = max(n_batches, 1)
        return total_normal / n, total_attack / n, total_combined / n

    def _validate_epoch(self, loader) -> float:
        """驗證集評估（只用重建損失）"""
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for (x,) in loader:
                x = x.to(self.device)
                x_hat, _ = self.model(x)
                loss = self.recon_criterion(x_hat, x)
                total += loss.item() * x.size(0)
        return total / len(loader.dataset)

    # ── 閾值計算（半監督增強版）──────────────────────────
    def compute_threshold(self,
                          X_normal: np.ndarray,
                          X_attack: np.ndarray,
                          method: str = "percentile",
                          percentile: int = 95) -> float:
        """
        計算最佳異常偵測閾值

        提供兩種方法：

        method="percentile"（傳統方法）：
            取正常流量誤差的第 percentile 百分位數

        method="optimal"（半監督增強方法）：
            利用已標記的攻擊樣本，掃描多個百分位數，
            選出 F1 Score 最大的閾值

            [Bug 6 修正 — 2026] 舊版直接對呼叫時傳入的 X_attack 做
            F1 最佳化搜尋；但 train_full()/run_semi_supervised.py
            傳入的 X_attack 正是 Phase 2 margin loss 微調所使用的
            「同一批」標記攻擊樣本 —— 等於用訓練時看過的資料去挑
            閾值，屬於資料洩漏，挑出的閾值可能對這批樣本過擬合。
            修正：若已呼叫過 load_data()（因此存在
            self.X_attack_holdout —— 微調時「未使用」的攻擊樣本
            互補集合），method="optimal" 會優先改用這份未參與微調
            的保留樣本做閾值搜尋；只有在沒有可用保留樣本時才退回
            使用傳入的 X_attack，並印出警告提醒使用者這是退化行為。
            注意：最終模型效能評估（evaluate_model）仍應使用完全
            獨立於 load_data() 的測試集（X_test_attack），不受此
            處影響。

        Args:
            X_normal   : 正常流量影像矩陣
            X_attack   : 攻擊流量影像矩陣（method="percentile" 時不使用；
                         method="optimal" 且無可用 holdout 時的備援來源）
            method     : "percentile" 或 "optimal"
            percentile : method="percentile" 時使用

        Returns:
            float: 最佳閾值
        """
        self.model.eval()
        print(f"\n  [閾值計算] 方法: {method}")

        errors_normal = self._compute_errors(X_normal)

        if method == "percentile":
            self.threshold = float(np.percentile(errors_normal, percentile))
            print(f"  正常流量誤差: mean={errors_normal.mean():.6f}, "
                  f"std={errors_normal.std():.6f}")
            print(f"  閾值（{percentile}th 百分位）: {self.threshold:.6f}")

        elif method == "optimal":
            holdout = getattr(self, "X_attack_holdout", None)
            if holdout is not None and len(holdout) >= 10:
                print(f"  [防洩漏] 使用微調時未參與訓練的 {len(holdout):,} "
                      f"筆保留攻擊樣本進行閾值搜尋（非傳入的完整 X_attack）")
                attack_for_search = holdout
            else:
                print(f"  [警告] 無足夠的保留攻擊樣本（需先呼叫 load_data()），"
                      f"退回使用傳入的 X_attack 進行閾值搜尋 —— "
                      f"若此 X_attack 與微調樣本重疊，結果可能對其過擬合")
                attack_for_search = X_attack

            errors_attack = self._compute_errors(attack_for_search)

            print(f"  正常流量誤差: mean={errors_normal.mean():.6f}, "
                  f"std={errors_normal.std():.6f}")
            print(f"  攻擊流量誤差: mean={errors_attack.mean():.6f}, "
                  f"std={errors_attack.std():.6f}")
            print(f"  分離比（攻擊/正常）: "
                  f"{errors_attack.mean() / (errors_normal.mean() + 1e-9):.2f}x")

            # [修正 #6（原）] 向量化閾值搜尋 O(N log N)
            best_thr, best_f1, best_pct = compute_optimal_threshold_vectorized(
                errors_normal, errors_attack, pct_range=(50, 100)
            )
            self.threshold = best_thr
            print(f"  最佳閾值（F1={best_f1:.4f}，{best_pct}th 百分位）: "
                  f"{self.threshold:.6f}")

        else:
            raise ValueError(f"未知的 method: {method}，請使用 'percentile' 或 'optimal'")

        # 儲存閾值到設定
        self._save_config()
        return self.threshold

    def _compute_errors(self, X: np.ndarray) -> np.ndarray:
        """計算影像矩陣的重建誤差"""
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        tensor  = torch.from_numpy(X.astype(np.float32))
        loader  = DataLoader(TensorDataset(tensor),
                             batch_size=128, shuffle=False)
        errors  = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                err   = self.model.reconstruction_error(batch)
                errors.append(err.cpu().numpy())
        return np.concatenate(errors)

    # ── 完整訓練流程（一鍵）──────────────────────────────
    def train_full(self,
                   X_normal: np.ndarray,
                   X_attack: np.ndarray,
                   threshold_method: str = "optimal"):
        """
        執行完整半監督訓練流程：載入資料 → Phase1 → Phase2 → 計算閾值

        Args:
            X_normal          : 正常流量影像矩陣
            X_attack          : 攻擊流量影像矩陣
            threshold_method  : "optimal" 或 "percentile"

        Returns:
            float: 訓練後的最佳閾值
        """
        self.load_data(X_normal, X_attack)
        self.pretrain()
        self.finetune()
        self.plot_training_curve()
        threshold = self.compute_threshold(X_normal, X_attack,
                                           method=threshold_method)
        return threshold

    # ── 視覺化 ────────────────────────────────────────────
    def plot_training_curve(self):
        """
        繪製完整訓練曲線，包含 Phase 1 和 Phase 2 的損失變化
        """
        n_pre  = len(self.pretrain_losses)
        n_fine = len(self.finetune_total)
        total  = n_pre + n_fine

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0d1117")

        # 左圖：Phase 1 預訓練損失
        ax1 = axes[0]
        ax1.set_facecolor("#161b22")
        if self.pretrain_losses:
            ax1.plot(range(1, n_pre+1), self.pretrain_losses,
                     color="#3fb950", linewidth=2, label="訓練損失")
            ax1.plot(range(1, n_pre+1), self.val_losses[:n_pre],
                     color="#58a6ff", linewidth=2, linestyle="--", label="驗證損失")
        ax1.set_title("Phase 1：無監督預訓練損失曲線", color="white", fontsize=11)
        ax1.set_xlabel("Epoch", color="#8b949e")
        ax1.set_ylabel("MSE Loss", color="#8b949e")
        ax1.legend(facecolor="#161b22", labelcolor="white")
        ax1.tick_params(colors="#8b949e")

        # 右圖：Phase 2 半監督微調損失
        ax2 = axes[1]
        ax2.set_facecolor("#161b22")
        if self.finetune_total:
            ep = range(1, n_fine+1)
            ax2.plot(ep, self.finetune_normal, color="#3fb950", linewidth=2,
                     label=f"正常損失（α={self.config['alpha']}）")
            ax2.plot(ep, self.finetune_attack, color="#f85149", linewidth=2,
                     label=f"攻擊邊界損失（β={self.config['beta']}）")
            ax2.plot(ep, self.finetune_total, color="#e3b341", linewidth=2,
                     linestyle="--", label="總損失")
            ax2.plot(ep, self.val_losses[n_pre:n_pre+n_fine],
                     color="#58a6ff", linewidth=2, linestyle=":", label="驗證損失")
        ax2.set_title("Phase 2：半監督微調損失曲線", color="white", fontsize=11)
        ax2.set_xlabel("Epoch", color="#8b949e")
        ax2.set_ylabel("Loss", color="#8b949e")
        ax2.legend(facecolor="#161b22", labelcolor="white", fontsize=8)
        ax2.tick_params(colors="#8b949e")

        fig.suptitle("半監督式 CNN Autoencoder 訓練曲線", color="white", fontsize=13)
        fig.tight_layout()

        path = os.path.join(self.output_dir, "semi_training_curve.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"  [圖表] 半監督訓練曲線已儲存: {path}")

    def plot_error_distribution_comparison(self,
                                           X_normal: np.ndarray,
                                           X_attack: np.ndarray):
        """
        繪製正常 vs 攻擊流量重建誤差分布（半監督訓練後）

        理想狀態：兩個分布完全分離，閾值清晰可辨
        """
        errors_normal = self._compute_errors(X_normal)
        errors_attack = self._compute_errors(X_attack)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0d1117")

        for ax, title_suffix in zip(axes, ["（密度圖）", "（累積分布）"]):
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e")

        # 左圖：密度分布
        ax = axes[0]
        ax.set_facecolor("#161b22")
        ax.hist(errors_normal, bins=60, alpha=0.6, color="#3fb950",
                label=f"正常 (N={len(errors_normal)})", density=True)
        ax.hist(errors_attack, bins=60, alpha=0.6, color="#f85149",
                label=f"攻擊 (N={len(errors_attack)})", density=True)
        if self.threshold:
            ax.axvline(self.threshold, color="#ffd700", linewidth=2.5,
                       linestyle="--", label=f"閾值={self.threshold:.4f}")
        ax.set_xlabel("重建誤差 (MSE)", color="#8b949e")
        ax.set_ylabel("機率密度", color="#8b949e")
        ax.set_title("半監督訓練後：誤差分布（密度）", color="white")
        ax.legend(facecolor="#161b22", labelcolor="white")
        ax.tick_params(colors="#8b949e")

        # 右圖：ECDF（累積分布）
        ax = axes[1]
        ax.set_facecolor("#161b22")
        for errors, label, color in [
            (errors_normal, "正常", "#3fb950"),
            (errors_attack, "攻擊", "#f85149"),
        ]:
            sorted_e = np.sort(errors)
            ecdf = np.arange(1, len(sorted_e)+1) / len(sorted_e)
            ax.plot(sorted_e, ecdf, color=color, linewidth=2, label=label)
        if self.threshold:
            ax.axvline(self.threshold, color="#ffd700", linewidth=2.5,
                       linestyle="--", label=f"閾值={self.threshold:.4f}")
        ax.set_xlabel("重建誤差 (MSE)", color="#8b949e")
        ax.set_ylabel("累積機率", color="#8b949e")
        ax.set_title("半監督訓練後：誤差分布（ECDF）", color="white")
        ax.legend(facecolor="#161b22", labelcolor="white")
        ax.tick_params(colors="#8b949e")

        fig.suptitle("半監督式 Autoencoder：重建誤差分布對比",
                     color="white", fontsize=13)
        fig.tight_layout()

        path = os.path.join(self.output_dir, "semi_error_distribution.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"  [圖表] 誤差分布圖已儲存: {path}")
        return errors_normal, errors_attack

    # ── 儲存設定 ──────────────────────────────────────────
    def _save_config(self):
        """儲存訓練設定與閾值到 JSON"""
        result = {
            "config":         self.config,
            "threshold":      self.threshold,
            "pretrain_epochs_done": len(self.pretrain_losses),
            "finetune_epochs_done": len(self.finetune_total),
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
        }
        path = os.path.join(self.output_dir, "semi_training_result.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  [SemiSupervisedTrainer] 訓練結果已儲存: {path}")

    @classmethod
    def load_model(cls, model_path: str, config_path: str = None,
                   latent_dim: int = 32) -> tuple:
        """
        載入已訓練的半監督模型與閾值

        Args:
            model_path : best_model.pt 路徑
            config_path: semi_training_result.json 路徑
            latent_dim : 若無 config_path，需手動指定
        Returns:
            (CNNAutoencoder, threshold)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        threshold = None

        if config_path and os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                result = json.load(f)
            latent_dim = result["config"].get("latent_dim", latent_dim)
            threshold  = result.get("threshold")

        model = CNNAutoencoder(latent_dim=latent_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        print(f"  [SemiSupervisedTrainer] 模型已載入: {model_path}")
        if threshold:
            print(f"  [SemiSupervisedTrainer] 閾值: {threshold:.6f}")

        return model, threshold

