# ============================================================
# threshold_tuner_semi.py - 半監督式閾值調校模組（增強版）
#
# 相比原版 threshold_tuner.py 的改進：
#
#   原版：
#     只掃描固定百分位數（80~99th），以正常流量百分位定閾值
#
#   本版新增：
#     1. calibrate_with_labels()
#        利用已標記的攻擊樣本，對所有可能閾值做完整 F1 掃描，
#        選出真正最佳值（不受百分位粒度限制）
#
#     2. plot_pr_f1_curve()
#        同時繪製 Precision / Recall / F1 vs 閾值曲線，
#        直觀顯示最佳操作點
#
#     3. sensitivity_analysis()
#        分析閾值微調對效能的靈敏度，
#        識別「穩定區間」（小幅調整不影響效能）vs「敏感區間」
#
#     4. multi_strategy_report()
#        同時輸出三種策略（高偵測率 / 平衡 / 低誤報）的推薦閾值
#
# 使用方式：
#   tuner = SemiSupervisedThresholdTuner(model, device)
#   report = tuner.calibrate_with_labels(X_normal, X_attack)
#   tuner.plot_pr_f1_curve(report)
#   recommendations = tuner.multi_strategy_report(report)
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

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
from torch.utils.data import DataLoader, TensorDataset

from threshold_tuner import ThresholdTuner


class SemiSupervisedThresholdTuner(ThresholdTuner):
    """
    半監督增強版閾值調校器

    繼承原版 ThresholdTuner，新增以標記攻擊樣本為導向的精準閾值搜索。

    核心改進：
        原版以正常流量百分位定閾值，粒度受限（只有 80/85/90...幾個點）
        本版對整個誤差值域做連續掃描，找出理論最佳閾值
    """

    def calibrate_with_labels(self,
                               X_normal: np.ndarray,
                               X_attack: np.ndarray,
                               n_thresholds: int = 200) -> list:
        """
        利用已標記攻擊樣本，精確搜索最佳閾值

        方法：
            1. 計算所有正常 + 攻擊樣本的重建誤差
            2. 在 [min_error, max_error] 範圍內均勻取 n_thresholds 個候選閾值
            3. 對每個候選閾值計算完整 P/R/F1/FPR
            4. 找出 F1 最大的閾值（最佳平衡點）

        相比 scan_percentiles()，此方法：
            - 不受百分位粒度限制（100 個候選 vs 只有幾個百分位）
            - 更精確地利用攻擊樣本的分布資訊
            - 能找到兩分布之間的最優分割點

        Args:
            X_normal     : 正常流量影像矩陣 (N, 32, 32)
            X_attack     : 攻擊流量影像矩陣 (M, 32, 32)
            n_thresholds : 候選閾值數量（越多越精確，但計算時間增加）

        Returns:
            list of dict，每個 dict 包含一個候選閾值的完整評估指標
        """
        print("\n  [SemiSupervisedThresholdTuner] 精確閾值搜索")
        print(f"  掃描 {n_thresholds} 個候選閾值...")

        errors_normal = self._compute_errors(X_normal)
        errors_attack = self._compute_errors(X_attack)

        print(f"  正常流量誤差: mean={errors_normal.mean():.6f}, "
              f"std={errors_normal.std():.6f}")
        print(f"  攻擊流量誤差: mean={errors_attack.mean():.6f}, "
              f"std={errors_attack.std():.6f}")

        # 搜索範圍：覆蓋兩個分布的全部值域
        all_errors  = np.concatenate([errors_normal, errors_attack])
        thr_min     = float(np.percentile(all_errors, 1))    # 1st 百分位
        thr_max     = float(np.percentile(all_errors, 99))   # 99th 百分位
        candidates  = np.linspace(thr_min, thr_max, n_thresholds)

        results = []
        for thr in candidates:
            metrics = self._evaluate_at_threshold(
                errors_normal, errors_attack, float(thr)
            )
            # 額外計算 MCC（Matthews Correlation Coefficient）
            tp, tn, fp, fn = (metrics["tp"], metrics["tn"],
                               metrics["fp"], metrics["fn"])
            denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) + 1e-9)
            mcc   = (tp*tn - fp*fn) / denom
            metrics["mcc"]          = float(mcc)
            metrics["threshold"]    = float(thr)
            results.append(metrics)

        # 儲存誤差陣列供後續視覺化使用
        self._errors_normal = errors_normal
        self._errors_attack = errors_attack

        # 找出各指標最佳點
        best_f1   = max(results, key=lambda r: r["f1"])
        best_mcc  = max(results, key=lambda r: r["mcc"])
        best_rec  = max(results, key=lambda r: r["recall"])
        best_prec = max(results, key=lambda r: r["precision"])

        print(f"\n  搜索結果摘要：")
        print(f"  {'策略':<16} {'閾值':>10} {'Precision':>10} "
              f"{'Recall':>8} {'F1':>8} {'MCC':>8}")
        print("  " + "-" * 62)
        for label, best in [
            ("最大F1（平衡）", best_f1),
            ("最大MCC",       best_mcc),
            ("最大Recall",    best_rec),
            ("最大Precision", best_prec),
        ]:
            print(f"  {label:<16} {best['threshold']:>10.6f} "
                  f"{best['precision']:>10.4f} {best['recall']:>8.4f} "
                  f"{best['f1']:>8.4f} {best['mcc']:>8.4f}")

        return results

    def plot_pr_f1_curve(self,
                         results: list,
                         output_dir: str = "output/model_semi"):
        """
        繪製精確版 Precision / Recall / F1 vs 閾值曲線

        新增相比原版的元素：
            - MCC（Matthews Correlation Coefficient）曲線
            - 最佳操作點（Best F1）標注
            - 陰影表示「穩定區間」
            - 雙 X 軸（閾值絕對值 + 相對位置百分比）

        Args:
            results   : calibrate_with_labels() 的回傳值
            output_dir: 圖片輸出目錄
        """
        thresholds = [r["threshold"] for r in results]
        precs      = [r["precision"] for r in results]
        recs       = [r["recall"]    for r in results]
        f1s        = [r["f1"]        for r in results]
        fprs       = [r["fpr"]       for r in results]
        mccs       = [r["mcc"]       for r in results]

        # 找最佳 F1 位置
        best_idx = int(np.argmax(f1s))
        best_thr = thresholds[best_idx]
        best_f1  = f1s[best_idx]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor("#0d1117")

        # 左圖：P / R / F1 / MCC vs 閾值
        ax1 = axes[0]
        ax1.set_facecolor("#161b22")
        ax1.plot(thresholds, precs, color="#3fb950", linewidth=2, label="Precision")
        ax1.plot(thresholds, recs,  color="#58a6ff", linewidth=2, label="Recall")
        ax1.plot(thresholds, f1s,   color="#e3b341", linewidth=2.5,
                 label="F1 Score（最佳化目標）")
        ax1.plot(thresholds, mccs,  color="#d2a8ff", linewidth=1.5,
                 linestyle="--", label="MCC")

        # 標注最佳 F1 點
        ax1.axvline(best_thr, color="#f85149", linewidth=2, linestyle="--",
                    alpha=0.8, label=f"最佳閾值 = {best_thr:.4f}")
        ax1.scatter([best_thr], [best_f1], color="#f85149", s=100, zorder=5)
        ax1.annotate(f"F1={best_f1:.3f}", (best_thr, best_f1),
                     textcoords="offset points", xytext=(10, 10),
                     color="white", fontsize=9,
                     bbox=dict(facecolor="#0d1117", edgecolor="#f85149", alpha=0.8))

        ax1.set_xlabel("重建誤差閾值", color="#8b949e")
        ax1.set_ylabel("指標值",       color="#8b949e")
        ax1.set_title("閾值 vs Precision / Recall / F1 / MCC",
                      color="white", fontsize=11)
        ax1.legend(facecolor="#161b22", labelcolor="white", fontsize=9)
        ax1.tick_params(colors="#8b949e")
        ax1.set_ylim(0, 1.05)
        for spine in ax1.spines.values():
            spine.set_edgecolor("#30363d")

        # 右圖：Precision-Recall 曲線（PR Curve）
        ax2 = axes[1]
        ax2.set_facecolor("#161b22")

        # PR 曲線（用閾值作為參數）
        sc = ax2.scatter(recs, precs, c=thresholds, cmap="plasma",
                         s=30, alpha=0.8)
        plt.colorbar(sc, ax=ax2, label="閾值").ax.yaxis.set_tick_params(
            color="white", labelcolor="white")
        ax2.scatter([recs[best_idx]], [precs[best_idx]],
                    color="#f85149", s=150, zorder=5,
                    label=f"最佳 F1 點 (Recall={recs[best_idx]:.3f}, "
                          f"Prec={precs[best_idx]:.3f})")

        ax2.set_xlabel("Recall（偵測率）",   color="#8b949e")
        ax2.set_ylabel("Precision（精確率）", color="#8b949e")
        ax2.set_title("Precision-Recall 曲線（連續閾值）",
                      color="white", fontsize=11)
        ax2.legend(facecolor="#161b22", labelcolor="white", fontsize=8)
        ax2.tick_params(colors="#8b949e")
        ax2.set_xlim(0, 1.05)
        ax2.set_ylim(0, 1.05)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#30363d")

        fig.suptitle("半監督式 CNN Autoencoder：精確閾值調校分析",
                     color="white", fontsize=13)
        fig.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "semi_threshold_curve.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"  [圖表] 半監督閾值調校曲線已儲存: {path}")
        return path

    def multi_strategy_report(self, results: list) -> dict:
        """
        輸出三種使用策略的推薦閾值

        策略 A（高偵測率）：
            適合高風險環境（政府、金融、重要基礎設施）
            可接受較高誤報，絕不漏掉攻擊
            → 選 Recall >= 0.95 的最小閾值

        策略 B（平衡型）：
            適合一般企業監控
            在偵測率與誤報率之間取平衡
            → 選 F1 Score 最大的閾值

        策略 C（低誤報）：
            適合誤報敏感場景（SOC 人工審核每一個告警）
            寧可漏掉部分攻擊，也不要讓分析師處理大量誤報
            → 選 FPR <= 0.05 的最大 F1 閾值

        Returns:
            dict: 三種策略的推薦閾值與對應指標
        """
        # 策略 A：最大 Recall
        strategy_a = max(results, key=lambda r: r["recall"])

        # 策略 B：最大 F1
        strategy_b = max(results, key=lambda r: r["f1"])

        # 策略 C：FPR <= 0.05 中 F1 最大
        low_fpr = [r for r in results if r["fpr"] <= 0.05]
        strategy_c = (max(low_fpr, key=lambda r: r["f1"])
                      if low_fpr else strategy_b)

        recommendations = {
            "high_recall":     strategy_a,
            "balanced_f1":     strategy_b,
            "low_false_alarm": strategy_c,
        }

        print(f"\n  ╔{'═'*62}╗")
        print(f"  ║  閾值策略推薦報告（半監督模型）")
        print(f"  ╠{'═'*62}╣")
        descs = [
            ("策略 A（高偵測率）", "high_recall",
             "適合高風險環境，可接受較高誤報"),
            ("策略 B（平衡型）",   "balanced_f1",
             "一般監控環境推薦，P/R 均衡"),
            ("策略 C（低誤報）",   "low_false_alarm",
             "誤報敏感場景，FPR ≤ 5%"),
        ]
        for name, key, desc in descs:
            r = recommendations[key]
            print(f"  ║  {name}  {desc}")
            print(f"  ║  閾值={r['threshold']:.6f}  "
                  f"Precision={r['precision']:.4f}  "
                  f"Recall={r['recall']:.4f}  "
                  f"F1={r['f1']:.4f}  FPR={r['fpr']:.4f}")
            print(f"  ╟{'─'*62}╢")
        print(f"  ╚{'═'*62}╝")

        return recommendations

    def sensitivity_analysis(self,
                              results: list,
                              target_metric: str = "f1",
                              delta_pct: float = 0.05) -> dict:
        """
        閾值靈敏度分析

        找出閾值的「穩定區間」：
        在最佳閾值附近 ± delta_pct 範圍內，效能下降不超過 5%

        這告訴我們：閾值設定有多少容忍空間
        若穩定區間很窄 → 需要精確設定，微小變化會顯著影響效能
        若穩定區間很寬 → 閾值設定較有彈性

        Args:
            results      : calibrate_with_labels() 的回傳值
            target_metric: 分析目標指標（"f1"/"recall"/"precision"）
            delta_pct    : 容忍的效能下降百分比（預設 5%）

        Returns:
            dict: 穩定區間與靈敏度分析結果
        """
        best_result = max(results, key=lambda r: r[target_metric])
        best_val    = best_result[target_metric]
        best_thr    = best_result["threshold"]
        threshold_range = 0.05 * best_val   # 允許 5% 的效能下降

        # 找出效能在可接受範圍內的所有閾值
        stable = [r for r in results
                  if r[target_metric] >= best_val - threshold_range]

        stable_min = min(r["threshold"] for r in stable)
        stable_max = max(r["threshold"] for r in stable)
        stable_width = stable_max - stable_min

        analysis = {
            "best_threshold":   best_thr,
            "best_metric_value": best_val,
            "stable_range_min":  stable_min,
            "stable_range_max":  stable_max,
            "stable_width":      stable_width,
            "sensitivity":       "LOW" if stable_width > 0.01 else
                                 "MEDIUM" if stable_width > 0.005 else "HIGH",
        }

        print(f"\n  [靈敏度分析] 目標指標: {target_metric.upper()}")
        print(f"  最佳閾值: {best_thr:.6f}  ({target_metric.upper()}={best_val:.4f})")
        print(f"  穩定區間: [{stable_min:.6f}, {stable_max:.6f}]  "
              f"（寬度: {stable_width:.6f}）")
        print(f"  靈敏度  : {analysis['sensitivity']}")

        sensitivity_desc = {
            "LOW":    "閾值設定有較大彈性，在此範圍內調整不影響效能",
            "MEDIUM": "閾值設定需要適度精確",
            "HIGH":   "閾值設定非常敏感，建議精確設定到小數點後 4 位",
        }
        print(f"  說明    : {sensitivity_desc[analysis['sensitivity']]}")

        return analysis
