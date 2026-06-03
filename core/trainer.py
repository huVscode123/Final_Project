# ============================================================
# trainer.py - CNN Autoencoder 訓練模組
#
# 功能：
#   - 從 .npy 資料集載入訓練資料
#   - 訓練 CNN Autoencoder（僅使用正常流量）
#   - Early Stopping 防止過擬合
#   - 自動儲存最佳模型（最低驗證損失）
#   - 繪製訓練曲線與重建影像對比圖
#   - 自動計算最佳異常偵測閾值
#
# 參考 GitHub：
#   - https://github.com/L1aoXingyu/pytorch-beginner
#   - https://github.com/black0017/MedicalZooPytorch（Early Stopping 設計）
#   - https://github.com/pytorch/examples/blob/main/mnist/main.py
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
from torch.utils.data import DataLoader, TensorDataset, random_split
 
from cnn_autoencoder import CNNAutoencoder
 
 
# ── 超參數設定 ────────────────────────────────────────────
# 可依實際資料量與硬體資源調整
DEFAULT_CONFIG = {
    "latent_dim":    64,      # 潛在空間維度
    "batch_size":    32,      # 批次大小（記憶體不足時縮小至 16）
    "epochs":        100,     # 最大訓練輪數
    "learning_rate": 1e-3,    # Adam 初始學習率
    "weight_decay":  1e-5,    # L2 正則化
    "val_split":     0.2,     # 驗證集比例
    "patience":      15,      # Early Stopping 等待輪數
    "min_delta":     1e-6,    # 最小改善量（低於此值視為未改善）
    "lr_patience":   7,       # 學習率降低等待輪數
    "lr_factor":     0.5,     # 學習率降低倍率
    "threshold_percentile": 95,  # 閾值百分位數（95%=只有5%正常流量被誤判）
}
 
 
class EarlyStopping:
    """
    Early Stopping 機制：當驗證損失連續 patience 輪未改善時停止訓練
 
    避免問題：
      - 過擬合：模型過度記憶訓練資料的雜訊
      - 計算浪費：訓練損失下降但驗證損失已停滯
 
    參考：
      https://github.com/Bjarten/early-stopping-pytorch
    """
 
    def __init__(self, patience: int = 15, min_delta: float = 1e-6,
                 path: str = "best_model.pt"):
        """
        Args:
            patience : 等待的輪數上限
            min_delta: 最小改善幅度（低於此值不算改善）
            path     : 最佳模型儲存路徑
        """
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = path
        self.counter    = 0           # 未改善的連續輪數
        self.best_loss  = float("inf")
        self.early_stop = False
 
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        每個 epoch 結束後呼叫
 
        Returns:
            bool: True = 應停止訓練
        """
        if val_loss < self.best_loss - self.min_delta:
            # 有改善：儲存模型並重置計數器
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)
        else:
            # 無改善：計數器 +1
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
 
        return self.early_stop
 
 
class PacketDataset(TensorDataset):
    """
    封包影像資料集包裝器
 
    從 .npy 矩陣載入，轉換為 PyTorch Tensor，
    並將維度從 (N, H, W) 擴展為 (N, 1, H, W) 以符合 Conv2d 輸入格式
    """
 
    @classmethod
    def from_npy(cls, npy_path: str) -> "PacketDataset":
        """
        從 .npy 檔案建立資料集
 
        Args:
            npy_path: X_normal.npy 或 X_all.npy 的路徑
        Returns:
            PacketDataset 物件
        """
        data = np.load(npy_path).astype(np.float32)   # (N, H, W)
        data = data[:, np.newaxis, :, :]               # (N, 1, H, W) 加入通道維度
        tensor = torch.from_numpy(data)
        return cls(tensor)
 
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "PacketDataset":
        """從 numpy array 直接建立"""
        data = arr.astype(np.float32)
        if data.ndim == 3:
            data = data[:, np.newaxis, :, :]           # (N,H,W) → (N,1,H,W)
        return cls(torch.from_numpy(data))
 
 
class Trainer:
    """
    CNN Autoencoder 訓練器
 
    使用流程：
        trainer = Trainer(config)
        trainer.load_data("output/dataset/X_normal.npy")
        trainer.train()
        trainer.plot_training_curve()
        threshold = trainer.compute_threshold()
    """
 
    def __init__(self, config: dict = None, output_dir: str = "output/model", model_name: str = ""):
        """
        Args:
            config    : 超參數字典（None 則使用預設值）
            output_dir: 模型與圖表的輸出目錄
        """
        self.config     = {**DEFAULT_CONFIG, **(config or {})}
        self.output_dir = output_dir
        self.model_name = model_name
        os.makedirs(output_dir, exist_ok=True)
 
        # 檔名後綴處理（各資料集模型獨立存放）
        self.suffix = f"_{model_name}" if model_name else ""
 
        # 自動選擇裝置（GPU 優先）
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"  [Trainer] 使用裝置: {self.device}")
        if model_name:
            print(f"  [Trainer] 模型名稱: {model_name} (檔名後綴: {self.suffix})")
 
        # 初始化模型
        self.model = CNNAutoencoder(
            latent_dim=self.config["latent_dim"]
        ).to(self.device)
 
        # 列印模型資訊
        info = self.model.get_model_info()
        print(f"  [Trainer] 模型參數: {info['total_params']:,} "
              f"({info['model_size_MB']:.2f} MB)")
 
        # 損失函數：MSE（均方誤差）
        # 理由：MSE 對大誤差懲罰更重，利於偵測明顯異常
        self.criterion = nn.MSELoss()
 
        # 優化器：Adam（自適應學習率）
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
 
        # 學習率調度器：驗證損失不改善時自動降低學習率
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config["lr_factor"],
            patience=self.config["lr_patience"],
        )
 
        # 訓練歷史記錄
        self.train_losses = []
        self.val_losses   = []
        self.best_val_loss = float("inf")
        self.threshold    = None   # 異常偵測閾值（訓練後計算）
 
    # ── 資料載入 ──────────────────────────────────────────
    def load_data(self, npy_path: str):
        """
        從 .npy 檔案載入資料並切分訓練/驗證集
 
        只載入正常流量（X_normal.npy），不使用攻擊標籤：
        Autoencoder 是非監督式學習，只需要正常樣本
 
        Args:
            npy_path: 正常流量影像矩陣路徑（X_normal.npy）
        """
        print(f"\n  [Trainer] 載入資料: {npy_path}")
        dataset = PacketDataset.from_npy(npy_path)
        n_total = len(dataset)
 
        # 切分訓練集與驗證集
        n_val   = int(n_total * self.config["val_split"])
        n_train = n_total - n_val
        self.train_set, self.val_set = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)  # 固定隨機種子，確保可重現
        )
 
        # 建立 DataLoader（多執行緒載入加速）
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.config["batch_size"],
            shuffle=True,                # 訓練時打亂順序
            num_workers=0,               # Windows 需設為 0，Linux 可設 2~4
            pin_memory=(self.device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.config["batch_size"],
            shuffle=False,               # 驗證時不需打亂
            num_workers=0,
        )
 
        print(f"  [Trainer] 訓練集: {n_train} 個，驗證集: {n_val} 個")
        print(f"  [Trainer] 影像尺寸: {dataset[0][0].shape}")
 
    # ── 訓練主迴圈 ────────────────────────────────────────
    def train(self):
        """
        執行完整訓練流程
 
        每個 epoch 的步驟：
          1. 前向傳播（輸入 → 重建）
          2. 計算 MSE Loss（重建誤差）
          3. 反向傳播（計算梯度）
          4. 更新權重（Adam step）
          5. 驗證集評估
          6. Early Stopping 判斷
        """
        model_path = os.path.join(self.output_dir, f"best_model{self.suffix}.pt")
        early_stop = EarlyStopping(
            patience=self.config["patience"],
            min_delta=self.config["min_delta"],
            path=model_path,
        )
 
        print(f"\n  [Trainer] 開始訓練（最多 {self.config['epochs']} epochs）")
        print(f"  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} "
              f"{'LR':<12} {'時間(s)'}")
        print("  " + "─" * 60)
 
        total_start = time.time()
 
        for epoch in range(1, self.config["epochs"] + 1):
            epoch_start = time.time()
 
            # ── 訓練階段 ──────────────────────────────────
            train_loss = self._train_epoch()
 
            # ── 驗證階段 ──────────────────────────────────
            val_loss = self._validate_epoch()
 
            # 記錄歷史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
 
            # 更新學習率調度器
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
 
            elapsed = time.time() - epoch_start
            print(f"  {epoch:<8} {train_loss:<14.6f} {val_loss:<14.6f} "
                  f"{current_lr:<12.2e} {elapsed:.1f}")
 
            # ── Early Stopping 判斷 ───────────────────────
            if early_stop(val_loss, self.model):
                print(f"\n  [EarlyStopping] 第 {epoch} epoch 停止"
                      f"（連續 {self.config['patience']} 輪無改善）")
                break
 
        total_time = time.time() - total_start
        print(f"\n  訓練完成！總耗時: {total_time:.1f}s，"
              f"最佳驗證損失: {early_stop.best_loss:.6f}")
 
        # 載入最佳模型權重
        # 修改前
        #self.model.load_state_dict(torch.load(model_path, #map_location=self.device))
 
        # 修改後
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.best_val_loss = early_stop.best_loss
 
        # 儲存訓練設定
        self._save_config()
 
    def _train_epoch(self) -> float:
        """
        執行一個 epoch 的訓練
 
        Returns:
            float: 平均訓練損失
        """
        self.model.train()   # 啟用 Dropout、BatchNorm 訓練模式
        total_loss = 0.0
 
        for batch in self.train_loader:
            x = batch[0].to(self.device)   # (batch, 1, 32, 32)
 
            # 前向傳播
            x_hat, _ = self.model(x)
 
            # 計算重建損失（MSE）
            loss = self.criterion(x_hat, x)
 
            # 反向傳播
            self.optimizer.zero_grad()  # 清除上一步梯度
            loss.backward()             # 計算梯度
 
            # 梯度裁剪：防止梯度爆炸（max_norm=1.0）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
 
            self.optimizer.step()       # 更新權重
 
            total_loss += loss.item() * x.size(0)  # 累積 batch 損失
 
        return total_loss / len(self.train_loader.dataset)
 
    def _validate_epoch(self) -> float:
        """
        執行驗證集評估
 
        Returns:
            float: 平均驗證損失
        """
        self.model.eval()    # 停用 Dropout、BatchNorm 使用移動平均
        total_loss = 0.0
 
        with torch.no_grad():  # 驗證時不計算梯度，節省記憶體
            for batch in self.val_loader:
                x = batch[0].to(self.device)
                x_hat, _ = self.model(x)
                loss = self.criterion(x_hat, x)
                total_loss += loss.item() * x.size(0)
 
        return total_loss / len(self.val_loader.dataset)
 
    # ── 閾值計算 ──────────────────────────────────────────
    def compute_threshold(self, npy_path: str = None,
                          percentile: int = None) -> float:
        """
        計算異常偵測閾值
 
        方法：對所有訓練資料計算重建誤差，取第 percentile 百分位數作為閾值。
        例：percentile=95 → 95% 的正常封包誤差低於閾值，只有 5% 誤報率。
 
        Args:
            npy_path  : 用於計算閾值的正常流量 .npy 路徑（None 則用訓練集）
            percentile: 百分位數（None 則使用 config 值）
        Returns:
            float: 閾值
        """
        pct = percentile or self.config["threshold_percentile"]
        self.model.eval()
 
        errors = []
        loader = self.train_loader  # 預設用訓練集
 
        if npy_path:
            dataset = PacketDataset.from_npy(npy_path)
            loader  = DataLoader(dataset, batch_size=64, shuffle=False)
 
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                err = self.model.reconstruction_error(x)
                errors.extend(err.cpu().numpy().tolist())
 
        errors = np.array(errors)
        self.threshold = float(np.percentile(errors, pct))
 
        print(f"\n  [Threshold] 重建誤差統計:")
        print(f"    Min   : {errors.min():.6f}")
        print(f"    Mean  : {errors.mean():.6f}")
        print(f"    Max   : {errors.max():.6f}")
        print(f"    閾值 ({pct}th percentile): {self.threshold:.6f}")
 
        # 繪製誤差分布圖
        self._plot_error_distribution(errors, pct)
 
        return self.threshold
 
    # ── 視覺化 ────────────────────────────────────────────
    def plot_training_curve(self):
        """繪製訓練 / 驗證損失曲線"""
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")
 
        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, color="#3fb950",
                linewidth=2, label="訓練損失 (Train Loss)")
        ax.plot(epochs, self.val_losses,   color="#58a6ff",
                linewidth=2, label="驗證損失 (Val Loss)")
 
        ax.set_xlabel("Epoch", color="#8b949e")
        ax.set_ylabel("MSE Loss", color="#8b949e")
        ax.set_title(f"CNN Autoencoder 訓練曲線{' (' + self.model_name + ')' if self.model_name else ''}", color="white", fontsize=13)
        ax.legend(facecolor="#161b22", labelcolor="white")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
 
        path = os.path.join(self.output_dir, f"training_curve{self.suffix}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"  [圖表] 訓練曲線已儲存: {path}")
 
    def _plot_error_distribution(self, errors: np.ndarray, pct: int):
        """繪製重建誤差分布直方圖"""
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")
 
        ax.hist(errors, bins=50, color="#3fb950", alpha=0.7, edgecolor="#0d1117")
        ax.axvline(self.threshold, color="#f85149", linewidth=2,
                   linestyle="--", label=f"閾值 ({pct}th pct) = {self.threshold:.4f}")
 
        ax.set_xlabel("重建誤差 (MSE)", color="#8b949e")
        ax.set_ylabel("頻率", color="#8b949e")
        ax.set_title(f"正常流量重建誤差分布{' (' + self.model_name + ')' if self.model_name else ''}", color="white", fontsize=13)
        ax.legend(facecolor="#161b22", labelcolor="white")
        ax.tick_params(colors="#8b949e")
 
        path = os.path.join(self.output_dir, f"error_distribution{self.suffix}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"  [圖表] 誤差分布圖已儲存: {path}")
 
    def plot_reconstruction_samples(self, npy_path: str, n_samples: int = 8):
        """
        繪製原始封包影像 vs 重建影像對比圖
 
        Args:
            npy_path : .npy 檔案路徑
            n_samples: 顯示樣本數
        """
        dataset = PacketDataset.from_npy(npy_path)
        loader  = DataLoader(dataset, batch_size=n_samples, shuffle=True)
        x_batch = next(iter(loader))[0][:n_samples].to(self.device)
 
        self.model.eval()
        with torch.no_grad():
            x_hat, _ = self.model(x_batch)
            errors = self.model.reconstruction_error(x_batch)
 
        x_orig = x_batch.cpu().numpy()[:, 0]    # (n, 32, 32)
        x_recon= x_hat.cpu().numpy()[:, 0]      # (n, 32, 32)
        errs   = errors.cpu().numpy()
 
        fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2, 6))
        fig.patch.set_facecolor("#0d1117")
 
        for i in range(n_samples):
            diff = np.abs(x_orig[i] - x_recon[i])
 
            # 上排：原始影像
            axes[0][i].imshow(x_orig[i], cmap="hot", vmin=0, vmax=1)
            axes[0][i].axis("off")
            if i == 0:
                axes[0][i].set_ylabel("原始", color="white", fontsize=9)
 
            # 中排：重建影像
            axes[1][i].imshow(x_recon[i], cmap="hot", vmin=0, vmax=1)
            axes[1][i].axis("off")
            if i == 0:
                axes[1][i].set_ylabel("重建", color="white", fontsize=9)
 
            # 下排：差異圖（誤差熱力圖）
            axes[2][i].imshow(diff, cmap="jet", vmin=0)
            axes[2][i].axis("off")
            axes[2][i].set_title(f"MSE={errs[i]:.4f}", color="#8b949e", fontsize=7)
            if i == 0:
                axes[2][i].set_ylabel("誤差", color="white", fontsize=9)
 
        fig.suptitle(f"封包影像重建對比（原始 | 重建 | 誤差熱力圖）{' (' + self.model_name + ')' if self.model_name else ''}",
                     color="white", fontsize=12)
        fig.tight_layout()
 
        path = os.path.join(self.output_dir, f"reconstruction_samples{self.suffix}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"  [圖表] 重建對比圖已儲存: {path}")
 
    # ── 儲存 / 載入 ───────────────────────────────────────
    def _save_config(self):
        """儲存訓練設定與結果到 JSON"""
        result = {
            "config":        self.config,
            "best_val_loss": self.best_val_loss,
            "threshold":     self.threshold,
            "train_losses":  self.train_losses,
            "val_losses":    self.val_losses,
            "model_name":    self.model_name,
        }
        path = os.path.join(self.output_dir, f"training_result{self.suffix}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  [Trainer] 訓練結果已儲存: {path}")
 
    @classmethod
    def load_model(cls, model_path: str, config_path: str = None,
                   latent_dim: int = 64) -> tuple:
        """
        載入已訓練的模型與閾值
 
        Args:
            model_path : best_model.pt 路徑
            config_path: training_result.json 路徑
            latent_dim : 若無 config_path，需手動指定
 
        Returns:
            (CNNAutoencoder, threshold)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        # 載入訓練設定
        threshold = None
        if config_path and os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                result = json.load(f)
            latent_dim = result["config"].get("latent_dim", latent_dim)
            threshold  = result.get("threshold")
 
        # 載入模型
        model = CNNAutoencoder(latent_dim=latent_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
 
        print(f"  [Trainer] 模型已載入: {model_path}")
        if threshold:
            print(f"  [Trainer] 閾值: {threshold:.6f}")
 
        return model, threshold