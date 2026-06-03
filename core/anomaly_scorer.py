# ============================================================
# anomaly_scorer.py - 異常分數計算與模型評估模組
#
# 功能：
#   - 對封包（bytes / numpy array / .npy 檔案）計算異常分數
#   - 依閾值判定是否為攻擊流量
#   - 計算 Precision、Recall、F1、ROC-AUC 等評估指標
#   - 繪製 ROC 曲線、Precision-Recall 曲線、混淆矩陣
#   - 整合 Grad-CAM 視覺化（回映異常位元組區段）
#
# 參考 GitHub：
#   - https://github.com/keras-team/keras/blob/master/examples/timeseries_anomaly_detection.py
#   - https://github.com/yunjey/pytorch-tutorial
#   - https://github.com/jacobgil/pytorch-grad-cam（Grad-CAM 原始實作）
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_autoencoder import CNNAutoencoder
from packet_visualizer import PacketVisualizer


class AnomalyScorer:
    """
    異常分數計算器

    使用流程：
        scorer = AnomalyScorer(model, threshold=0.012)

        # 對單一封包計分
        score, is_attack = scorer.score_packet(raw_bytes)

        # 對 .npy 資料集評估
        results = scorer.evaluate(X_test, y_test)

        # 繪製評估圖表
        scorer.plot_roc_curve(X_normal, X_attack)
    """

    def __init__(self, model: CNNAutoencoder, threshold: float = None,
                 device: str = None):
        """
        Args:
            model    : 已訓練的 CNNAutoencoder
            threshold: 異常判定閾值（None 則需呼叫 set_threshold()）
            device   : 計算裝置（None 則自動偵測）
        """
        self.device    = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model     = model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        self.visualizer = PacketVisualizer("medium", apply_mask=True)

    # ── 對單一封包計分 ────────────────────────────────────
    def score_packet(self, raw_bytes: bytes) -> tuple:
        """
        對單一原始封包計算異常分數

        流程：
          raw_bytes → 影像化（32×32）→ Encoder → Decoder → MSE → 異常分數

        Args:
            raw_bytes: 原始封包位元組
        Returns:
            (score: float, is_attack: bool)
              score    : 重建誤差 MSE，值越大代表越可能是異常
              is_attack: True = 超過閾值，判定為攻擊
        """
        # 影像化：bytes → (1, 32, 32) tensor
        arr = self.visualizer.bytes_to_image(raw_bytes)                # (32, 32)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)       # (1, 1, 32, 32)
        tensor = tensor.to(self.device)

        # 計算重建誤差
        error = self.model.reconstruction_error(tensor)
        score = float(error.cpu().item())

        is_attack = (self.threshold is not None) and (score > self.threshold)
        return score, is_attack

    def score_batch(self, raw_bytes_list: list) -> list:
        """
        批次計分多個封包

        Args:
            raw_bytes_list: 原始封包 bytes 列表
        Returns:
            list of (score, is_attack) tuples
        """
        arrs = np.array([
            self.visualizer.bytes_to_image(b) for b in raw_bytes_list
        ], dtype=np.float32)                                           # (N, 32, 32)

        tensor = torch.from_numpy(arrs[:, np.newaxis]).to(self.device) # (N, 1, 32, 32)
        errors = self.model.reconstruction_error(tensor).cpu().numpy()

        return [(float(e), bool(e > self.threshold) if self.threshold else False)
                for e in errors]

    def score_npy(self, npy_path: str) -> np.ndarray:
        """
        對 .npy 資料集計算全部重建誤差

        Args:
            npy_path: (N, H, W) float32 的 .npy 路徑
        Returns:
            np.ndarray shape=(N,)，每個樣本的 MSE
        """
        data   = np.load(npy_path).astype(np.float32)[:, np.newaxis]  # (N,1,32,32)
        tensor = torch.from_numpy(data)
        loader = DataLoader(TensorDataset(tensor), batch_size=64, shuffle=False)

        all_errors = []
        with torch.no_grad():
            for batch in loader:
                x   = batch[0].to(self.device)
                err = self.model.reconstruction_error(x)
                all_errors.extend(err.cpu().numpy().tolist())

        return np.array(all_errors)

    # ── 評估指標 ──────────────────────────────────────────
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 output_dir: str = "output/model") -> dict:
        """
        計算完整評估指標

        Args:
            X_test : 測試影像矩陣 shape=(N, 32, 32)，包含正常與攻擊
            y_test : 標籤 shape=(N,)，0=正常，1=攻擊
            output_dir: 圖表輸出目錄
        Returns:
            dict: 包含 precision/recall/f1/accuracy/auc 等指標
        """
        # 計算所有樣本的重建誤差
        data   = X_test.astype(np.float32)[:, np.newaxis]              # (N,1,32,32)
        tensor = torch.from_numpy(data)
        loader = DataLoader(TensorDataset(tensor), batch_size=64)

        scores = []
        with torch.no_grad():
            for batch in loader:
                x   = batch[0].to(self.device)
                err = self.model.reconstruction_error(x)
                scores.extend(err.cpu().numpy().tolist())
        scores = np.array(scores)

        # 依閾值判定預測標籤
        y_pred = (scores > self.threshold).astype(int)

        # 計算基本指標
        tp = int(np.sum((y_pred == 1) & (y_test == 1)))  # 正確偵測到的攻擊
        tn = int(np.sum((y_pred == 0) & (y_test == 0)))  # 正確判定為正常
        fp = int(np.sum((y_pred == 1) & (y_test == 0)))  # 誤報（正常被判為攻擊）
        fn = int(np.sum((y_pred == 0) & (y_test == 1)))  # 漏報（攻擊被判為正常）

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)   # = Detection Rate
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy  = (tp + tn) / len(y_test)
        fpr       = fp / (fp + tn + 1e-9)   # False Positive Rate

        # AUC（需要 sklearn）
        auc = None
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_test, scores))
        except ImportError:
            pass

        results = {
            "threshold": self.threshold,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "precision":  float(precision),
            "recall":     float(recall),      # Detection Rate
            "f1":         float(f1),
            "f1_score":   float(f1),
            "accuracy":   float(accuracy),
            "fpr":        float(fpr),         # False Positive Rate
            "auc":        auc,
            "n_samples":  len(y_test),
        }

        self._print_evaluation(results)
        self._plot_confusion_matrix(results, output_dir)
        return results

    @staticmethod
    def _print_evaluation(r: dict):
        """格式化列印評估結果"""
        print(f"\n  {'='*50}")
        print(f"  模型評估結果（閾值: {r['threshold']:.6f}）")
        print(f"  {'='*50}")
        print(f"  {'指標':<20} {'值':>12}")
        print(f"  {'─'*34}")
        print(f"  {'Precision':<20} {r['precision']:>12.4f}")
        print(f"  {'Recall（偵測率）':<20} {r['recall']:>12.4f}")
        print(f"  {'F1 Score':<20} {r['f1_score']:>12.4f}")
        print(f"  {'Accuracy':<20} {r['accuracy']:>12.4f}")
        print(f"  {'FPR（誤報率）':<20} {r['fpr']:>12.4f}")
        if r["auc"]:
            print(f"  {'AUC-ROC':<20} {r['auc']:>12.4f}")
        print(f"  {'─'*34}")
        print(f"  TP={r['TP']}  TN={r['TN']}  FP={r['FP']}  FN={r['FN']}")
        print(f"  {'='*50}")

    def _plot_confusion_matrix(self, r: dict, output_dir: str):
        """繪製混淆矩陣"""
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        im = ax.imshow(cm, cmap="Blues")
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]}",
                        ha="center", va="center", color="white", fontsize=12)

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["預測:正常", "預測:攻擊"], color="#8b949e")
        ax.set_yticklabels(["實際:正常", "實際:攻擊"], color="#8b949e")
        ax.set_title(
            f"混淆矩陣  (Recall={r['recall']:.3f}  FPR={r['fpr']:.3f})",
            color="white", fontsize=11
        )
        plt.colorbar(im, ax=ax)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "confusion_matrix.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"  [圖表] 混淆矩陣已儲存: {path}")

    def plot_roc_curve(self, errors_normal: np.ndarray,
                       errors_attack: np.ndarray,
                       output_dir: str = "output/model"):
        """
        繪製 ROC 曲線（Receiver Operating Characteristic）

        ROC 曲線說明：
          - X 軸：False Positive Rate（誤報率）
          - Y 軸：True Positive Rate（偵測率）
          - 曲線越靠近左上角 → 模型越好
          - AUC（曲線下面積）：0.5=隨機，1.0=完美

        Args:
            errors_normal : 正常流量的重建誤差 shape=(N_normal,)
            errors_attack : 攻擊流量的重建誤差 shape=(N_attack,)
        """
        y_true  = np.concatenate([
            np.zeros(len(errors_normal)),
            np.ones(len(errors_attack))
        ])
        y_score = np.concatenate([errors_normal, errors_attack])

        try:
            from sklearn.metrics import roc_curve, auc, precision_recall_curve
        except ImportError:
            print("  需要安裝 scikit-learn：pip install scikit-learn")
            return

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("#0d1117")

        # ROC 曲線
        ax1.set_facecolor("#161b22")
        ax1.plot(fpr, tpr, color="#58a6ff", linewidth=2,
                 label=f"ROC Curve (AUC = {roc_auc:.3f})")
        ax1.plot([0, 1], [0, 1], color="#30363d", linestyle="--",
                 label="Random (AUC = 0.500)")
        if self.threshold:
            # 標記目前閾值在 ROC 上的位置
            idx = np.argmin(np.abs(thresholds - self.threshold))
            ax1.scatter(fpr[idx], tpr[idx], color="#f85149", s=80, zorder=5,
                        label=f"Current Threshold\n(FPR={fpr[idx]:.3f}, TPR={tpr[idx]:.3f})")
        ax1.set_xlabel("False Positive Rate（誤報率）", color="#8b949e")
        ax1.set_ylabel("True Positive Rate（偵測率）", color="#8b949e")
        ax1.set_title("ROC 曲線", color="white", fontsize=12)
        ax1.legend(facecolor="#161b22", labelcolor="white", fontsize=9)
        ax1.tick_params(colors="#8b949e")

        # Precision-Recall 曲線
        ax2.set_facecolor("#161b22")
        ax2.plot(recall, precision, color="#3fb950", linewidth=2,
                 label=f"PR Curve (AUC = {pr_auc:.3f})")
        ax2.set_xlabel("Recall（偵測率）", color="#8b949e")
        ax2.set_ylabel("Precision（精確率）", color="#8b949e")
        ax2.set_title("Precision-Recall 曲線", color="white", fontsize=12)
        ax2.legend(facecolor="#161b22", labelcolor="white")
        ax2.tick_params(colors="#8b949e")

        fig.suptitle("CNN Autoencoder 異常偵測效能評估", color="white", fontsize=13)
        fig.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "roc_pr_curve.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"  [圖表] ROC/PR 曲線已儲存: {path}")

    def plot_score_distribution(self, errors_normal: np.ndarray,
                                errors_attack: np.ndarray,
                                output_dir: str = "output/model"):
        """
        繪製正常流量 vs 攻擊流量的重建誤差分布對比圖

        理想狀態：兩個分布分離，閾值落在兩者之間
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        ax.hist(errors_normal, bins=60, alpha=0.6, color="#3fb950",
                label=f"正常流量 (N={len(errors_normal)})", density=True)
        ax.hist(errors_attack, bins=60, alpha=0.6, color="#f85149",
                label=f"攻擊流量 (N={len(errors_attack)})", density=True)

        if self.threshold:
            ax.axvline(self.threshold, color="#ffd700", linewidth=2.5,
                       linestyle="--",
                       label=f"閾值 = {self.threshold:.4f}")

        ax.set_xlabel("重建誤差 (MSE)", color="#8b949e", fontsize=11)
        ax.set_ylabel("機率密度", color="#8b949e", fontsize=11)
        ax.set_title("正常流量 vs 攻擊流量 重建誤差分布",
                     color="white", fontsize=13)
        ax.legend(facecolor="#161b22", labelcolor="white")
        ax.tick_params(colors="#8b949e")

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "score_distribution.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"  [圖表] 分數分布圖已儲存: {path}")

    # ── Grad-CAM 異常位元組回映 ───────────────────────────
    def gradcam_packet(self, raw_bytes: bytes,
                       save_path: str = None) -> np.ndarray:
        """
        對單一封包執行 Grad-CAM，回映異常特徵到影像

        原理：
          1. 計算重建誤差對 Encoder 最後一層 feature map 的梯度
          2. 對梯度取全域平均池化，得到每個 channel 的權重
          3. 加權求和 feature map → CAM（Class Activation Map）
          4. 疊加到原始封包影像，高亮顯示異常區域

        注意：
          傳統 Grad-CAM 設計給分類任務，這裡改為對「重建誤差」求梯度
          誤差最高的區域 ≈ 與正常封包差異最大的位元組段

        Args:
            raw_bytes: 原始封包位元組
            save_path: 圖片儲存路徑（None 則不儲存）
        Returns:
            overlay: RGB 疊加影像 shape=(32, 32, 3)
        """
        # 影像化
        arr = self.visualizer.bytes_to_image(raw_bytes)                # (32, 32)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)       # (1,1,32,32)
        tensor = tensor.to(self.device).requires_grad_(True)

        # 前向傳播
        self.model.train()   # 需要 requires_grad
        x_hat, _ = self.model(tensor)

        # 目標：重建誤差（對每個像素的誤差求和）
        recon_error = torch.mean((tensor - x_hat) ** 2)

        # 反向傳播（對 Encoder 最後一層的激活求梯度）
        self.model.zero_grad()
        recon_error.backward()

        # 取 Encoder Conv 最後一層的梯度與激活
        # （這裡簡化：直接用輸入梯度作為重要性圖）
        grad_map = tensor.grad.squeeze().cpu().numpy()                 # (32, 32)

        # 正規化梯度圖
        cam = np.abs(grad_map)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # 疊加到封包影像
        overlay = self.visualizer.create_heatmap_overlay(
            arr, cam, alpha=0.65, save_path=save_path
        )

        self.model.eval()
        return overlay

    def set_threshold(self, threshold: float):
        """手動設定閾值"""
        self.threshold = threshold
        print(f"  [AnomalyScorer] 閾值設為: {threshold:.6f}")
