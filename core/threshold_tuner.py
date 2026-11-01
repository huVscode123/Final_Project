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

        return {"threshold": threshold, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "accuracy": accuracy, "pr_auc": pr_auc,
                "youden_j": recall - fpr}

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

    # ══════════════════════════════════════════════════════════
    # Youden's Index（J = TPR - FPR）
    # ══════════════════════════════════════════════════════════
    def find_youden_threshold(self, X_normal, X_attack, max_fpr=None) -> dict:
        """
        以 Youden's Index 在 ROC 曲線上自動尋找最佳閾值（連續搜尋，不受百分位粒度限制）。

        請傳入「驗證集」而非測試集，避免閾值對測試資料過擬合。

        Returns:
            dict：與 scan_percentiles() 單一項目相同的欄位（threshold / precision /
            recall / f1 / fpr / youden_j ...），另含 percentile、method、youden_info。
        """
        from training_common import compute_youden_threshold
        en = self._compute_errors(X_normal)
        ea = self._compute_errors(X_attack)
        thr, j, info = compute_youden_threshold(en, ea, max_fpr=max_fpr)
        res = self._evaluate_at_threshold(en, ea, thr)
        res["percentile"] = float((en <= thr).mean() * 100.0)   # 此閾值相當於正常分數的第幾百分位
        res["method"] = "youden" if max_fpr is None else f"youden(max_fpr={max_fpr})"
        res["youden_info"] = info
        print(f"\n  [Youden] 閾值={thr:.6f}  J={j:.4f}  "
              f"TPR={res['recall']:.4f}  FPR={res['fpr']:.4f}  "
              f"（約為正常流量的 {res['percentile']:.1f}th 百分位）")
        self._youden_errors = (en, ea)
        return res

    def plot_youden_roc(self, X_normal=None, X_attack=None, result=None,
                        output_dir="output/model", filename="youden_roc.png"):
        """
        繪製 ROC 曲線並標出 Youden 點：紅色垂直線 = 該點到對角線的距離，
        長度即 J = TPR - FPR。可直接用於論文 4.4 節。

        Args:
            X_normal, X_attack: 若未提供，使用最近一次 find_youden_threshold() 的分數。
            result            : find_youden_threshold() 的回傳值（用來取得 Youden 點）。
        """
        from sklearn.metrics import roc_curve, auc
        if X_normal is not None and X_attack is not None:
            en, ea = self._compute_errors(X_normal), self._compute_errors(X_attack)
        elif hasattr(self, "_youden_errors"):
            en, ea = self._youden_errors
        else:
            raise RuntimeError("請傳入 X_normal / X_attack，或先呼叫 find_youden_threshold()")

        y = np.concatenate([np.zeros(len(en)), np.ones(len(ea))])
        s = np.concatenate([en, ea])
        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = auc(fpr, tpr)

        if result is None:
            result = self.find_youden_threshold(X_normal, X_attack) \
                if X_normal is not None else None
        px, py = result["fpr"], result["recall"]
        j = result["youden_j"]

        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")
        ax.plot(fpr, tpr, color="#58a6ff", linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="#30363d", linestyle="--", label="Random")
        ax.plot([px, px], [px, py], color="#f85149", linewidth=2.5,
                label=f"Youden J = {j:.3f}")
        ax.scatter([px], [py], color="#f85149", s=90, zorder=5,
                   label=f"Youden 點 (FPR={px:.3f}, TPR={py:.3f})")
        ax.set_xlabel("False Positive Rate", color="#8b949e")
        ax.set_ylabel("True Positive Rate", color="#8b949e")
        ax.set_title("ROC 曲線與 Youden's Index 最佳切點", color="white", fontsize=12)
        ax.legend(facecolor="#161b22", labelcolor="white", loc="lower right")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"  [圖表] Youden ROC 圖已儲存: {path}")
        return path

    def compare_threshold_methods(self, X_normal_val, X_attack_val,
                                  X_normal_test, X_attack_test,
                                  percentile=95.0, max_fpr=0.05,
                                  output_dir=None) -> list:
        """
        論文 4.4 節用：在「驗證集」上以四種方法決定閾值，再於「同一份測試集」報告成績。

            1. percentile-<pct>       正常驗證分數的百分位數（完全不用攻擊標籤）
            2. F1-optimal             F1 最大化
            3. Youden                 J = TPR - FPR 最大化
            4. Youden + max_fpr       Youden 並限制 FPR 上限

        Returns:
            list[dict]，每列含 method / threshold / precision / recall / f1 / fpr /
            youden_j（皆為測試集數值）與 val_youden_j（驗證集數值）。
        """
        import json
        from training_common import (compute_optimal_threshold_vectorized,
                                     compute_youden_threshold)
        enV, eaV = self._compute_errors(X_normal_val), self._compute_errors(X_attack_val)
        enT, eaT = self._compute_errors(X_normal_test), self._compute_errors(X_attack_test)

        picks = []
        picks.append((f"percentile-{percentile:g}", float(np.percentile(enV, percentile))))
        f1_thr, _, _ = compute_optimal_threshold_vectorized(enV, eaV, pct_range=(50, 100))
        picks.append(("F1-optimal", f1_thr))
        y_thr, _, _ = compute_youden_threshold(enV, eaV)
        picks.append(("Youden", y_thr))
        ym_thr, _, _ = compute_youden_threshold(enV, eaV, max_fpr=max_fpr)
        picks.append((f"Youden+max_fpr={max_fpr:g}", ym_thr))

        rows = []
        for name, thr in picks:
            rt = self._evaluate_at_threshold(enT, eaT, float(thr))
            rv = self._evaluate_at_threshold(enV, eaV, float(thr))
            rows.append({
                "method": name, "threshold": float(thr),
                "precision": rt["precision"], "recall": rt["recall"], "f1": rt["f1"],
                "fpr": rt["fpr"], "youden_j": rt["youden_j"],
                "val_youden_j": rv["youden_j"],
            })

        print(f"\n  {'方法':<22}{'閾值':>12}{'Precision':>11}{'Recall':>9}"
              f"{'F1':>8}{'FPR':>8}{'J(test)':>9}{'J(val)':>9}")
        print("  " + "-" * 88)
        for r in rows:
            print(f"  {r['method']:<22}{r['threshold']:>12.6f}{r['precision']:>11.4f}"
                  f"{r['recall']:>9.4f}{r['f1']:>8.4f}{r['fpr']:>8.4f}"
                  f"{r['youden_j']:>9.4f}{r['val_youden_j']:>9.4f}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "threshold_method_comparison.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=2, ensure_ascii=False)
            print(f"  [報告] 已儲存: {os.path.join(output_dir, 'threshold_method_comparison.json')}")
        return rows
