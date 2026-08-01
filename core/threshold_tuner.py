# ============================================================
# threshold_tuner.py - 重建誤差閾值調校模組（優化版）
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from sklearn.metrics import precision_recall_curve, auc as sk_auc

# 自動偵測中文字型
_CJK = ["Microsoft YaHei", "SimHei", "PingFang TC",
        "Heiti TC", "WenQuanYi Zen Hei", "Noto Sans CJK TC"]
_avail = {f.name for f in _fm.fontManager.ttflist}
for _f in _CJK:
    if _f in _avail:
        matplotlib.rcParams["font.family"] = [_f, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ThresholdTuner:
    """
    閾值調校器：支援多百分位數掃描、PR-AUC 計算與最佳閾值自動搜尋。
    """

    def __init__(self, model, device=None):
        self.model  = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def _compute_errors(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3: X = X[:, np.newaxis, :, :]
        tensor = torch.from_numpy(X.astype(np.float32))
        loader = DataLoader(TensorDataset(tensor), batch_size=128, shuffle=False)
        all_errors = []
        with torch.no_grad():
            for (batch,) in loader:
                err = self.model.reconstruction_error(batch.to(self.device))
                all_errors.append(err.cpu().numpy())
        return np.concatenate(all_errors)

    @staticmethod
    def _evaluate_at_threshold(errors_normal: np.ndarray, errors_attack: np.ndarray,
                               threshold: float, pr_auc: float = None) -> dict:
        """在指定閾值下計算評估指標

        Args:
            errors_normal: 正常流量的重建誤差
            errors_attack: 攻擊流量的重建誤差
            threshold    : 判定閾值
            pr_auc       : 預先計算的 PR-AUC（避免重複計算）
        """
        fp = int((errors_normal > threshold).sum()); tn = len(errors_normal) - fp
        tp = int((errors_attack > threshold).sum()); fn = len(errors_attack) - tp
        precision = tp / (tp + fp + 1e-9); recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        fpr = fp / (fp + tn + 1e-9); accuracy = (tp + tn) / (len(errors_normal) + len(errors_attack))

        # [修正] PR-AUC 與 threshold 無關（只取決於 errors 和 labels），
        #        若已從外部傳入則直接使用，不重複計算。
        if pr_auc is None:
            all_errors = np.concatenate([errors_normal, errors_attack])
            all_labels = np.concatenate([np.zeros(len(errors_normal)), np.ones(len(errors_attack))])
            pr_pre, pr_rec, _ = precision_recall_curve(all_labels, all_errors)
            pr_auc = float(sk_auc(pr_rec, pr_pre))

        return {"threshold": threshold, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "accuracy": accuracy, "pr_auc": pr_auc}

    def scan_percentiles(self, X_normal, X_attack, percentiles=None) -> list:
        if percentiles is None: percentiles = [80, 85, 90, 92, 95, 97, 99, 99.5]
        en = self._compute_errors(X_normal); ea = self._compute_errors(X_attack)

        # [修正] PR-AUC 只需計算一次（與 threshold 無關）
        all_errors = np.concatenate([en, ea])
        all_labels = np.concatenate([np.zeros(len(en)), np.ones(len(ea))])
        pr_pre, pr_rec, _ = precision_recall_curve(all_labels, all_errors)
        pr_auc = float(sk_auc(pr_rec, pr_pre))

        results = []
        for pct in percentiles:
            thr = float(np.percentile(en, pct))
            res = self._evaluate_at_threshold(en, ea, thr, pr_auc=pr_auc)
            res["percentile"] = pct; results.append(res)
        return results

    def find_best_threshold(self, results: list, metric: str = "f1") -> dict:
        best = max(results, key=lambda r: r[metric])
        print(f"\n  [最佳閾值] 依 {metric.upper()} 最大化：{best['threshold']:.6f} ({best['percentile']}th pct)")
        return best

    def plot_threshold_curve(self, results: list, output_dir: str):
        """
        繪製各百分位數下的 Precision / Recall / F1 曲線

        Args:
            results   : scan_percentiles() 回傳的結果清單
            output_dir: 圖表儲存目錄
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        percentiles = [r["percentile"]  for r in results]
        precisions  = [r["precision"]   for r in results]
        recalls     = [r["recall"]      for r in results]
        f1s         = [r["f1"]          for r in results]
        thresholds  = [r["threshold"]   for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0d1117")

        # ── 左圖：P / R / F1 vs 百分位數 ──
        ax1.set_facecolor("#161b22")
        ax1.plot(percentiles, precisions, color="#58a6ff", linewidth=2,
                marker="o", markersize=5, label="Precision")
        ax1.plot(percentiles, recalls,    color="#3fb950", linewidth=2,
                marker="s", markersize=5, label="Recall")
        ax1.plot(percentiles, f1s,        color="#f0883e", linewidth=2,
                marker="^", markersize=5, label="F1 Score")

        best_idx = f1s.index(max(f1s))
        ax1.axvline(percentiles[best_idx], color="#f85149",
                    linestyle="--", linewidth=1.5,
                    label=f"Best F1 @ {percentiles[best_idx]}th pct")

        ax1.set_xlabel("百分位數 (Percentile)", color="#8b949e")
        ax1.set_ylabel("分數",                  color="#8b949e")
        ax1.set_title("閾值掃描：P / R / F1",   color="white", fontsize=12)
        ax1.legend(facecolor="#161b22", labelcolor="white")
        ax1.tick_params(colors="#8b949e")
        ax1.set_ylim(0, 1.05)
        for spine in ax1.spines.values():
            spine.set_edgecolor("#30363d")

        # ── 右圖：F1 vs 閾值數值 ──
        ax2.set_facecolor("#161b22")
        ax2.plot(thresholds, f1s, color="#f0883e", linewidth=2,
                marker="^", markersize=5, label="F1 Score")
        ax2.axvline(thresholds[best_idx], color="#f85149",
                    linestyle="--", linewidth=1.5,
                    label=f"Best thr = {thresholds[best_idx]:.5f}")

        ax2.set_xlabel("閾值 (Threshold)",   color="#8b949e")
        ax2.set_ylabel("F1 Score",           color="#8b949e")
        ax2.set_title("F1 Score vs 閾值數值", color="white", fontsize=12)
        ax2.legend(facecolor="#161b22", labelcolor="white")
        ax2.tick_params(colors="#8b949e")
        ax2.set_ylim(0, 1.05)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#30363d")

        fig.tight_layout()
        path = os.path.join(output_dir, "threshold_curve.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"  [圖表] 閾值掃描曲線已儲存: {path}")