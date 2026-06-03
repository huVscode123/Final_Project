# ============================================================
# core/gradcam/gradcam_analyzer.py  - 批次分析、特徵重要性排序
# ============================================================
"""
GradCAMAnalyzer：對整批樣本執行 Grad-CAM 並進行統計分析。

功能：
  - 批次計算正常 / 攻擊樣本的 Grad-CAM
  - 統計各像素位置的平均重要性（識別最常被關注的特徵欄位）
  - 比較正常 vs 攻擊樣本的 CAM 分布差異
  - 特徵重要性排序（對應原始封包特徵欄位）
  - Top-K 高異常樣本的精細分析
  - 平均 CAM 對比圖、強度分布盒鬚圖
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

_CJK = ["Microsoft YaHei", "SimHei", "PingFang TC",
        "Heiti TC", "WenQuanYi Zen Hei", "Noto Sans CJK TC"]
_avail = {f.name for f in _fm.fontManager.ttflist}
for _f in _CJK:
    if _f in _avail:
        matplotlib.rcParams["font.family"] = [_f, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break

_DARK_BG    = "#0d1117"
_DARK_AX    = "#161b22"
_TEXT_COLOR = "#e6edf3"


# ── 分析結果資料類 ────────────────────────────────────────────
@dataclass
class SampleResult:
    """單樣本的 Grad-CAM 分析結果。"""
    sample_idx:   int
    label:        str                         # "Normal" / "Attack"
    recon_score:  float                       # 重建誤差
    heatmap:      np.ndarray                  # shape (H, W)
    cam_mean:     float                       # CAM 平均強度
    cam_max:      float                       # CAM 最大值
    top_k_pixels: List[Tuple[int, int]]       # 最高強度像素座標 (row, col)


@dataclass
class AnalysisReport:
    """批次分析報告。"""
    normal_results:    List[SampleResult]         = field(default_factory=list)
    attack_results:    List[SampleResult]         = field(default_factory=list)
    normal_mean_cam:   Optional[np.ndarray]       = None   # 正常樣本 CAM 平均
    attack_mean_cam:   Optional[np.ndarray]       = None   # 攻擊樣本 CAM 平均
    diff_cam:          Optional[np.ndarray]       = None   # 差異 CAM（攻擊 - 正常）
    feature_importance: Optional[np.ndarray]     = None   # 每個特徵欄位的重要性
    feature_names:     Optional[List[str]]        = None


class GradCAMAnalyzer:
    """
    批次 Grad-CAM 分析器。

    Args:
        grad_cam     : 已初始化的 GradCAM 物件（core/gradcam/gradcam_core.py）
        model        : CNN Autoencoder 模型（計算重建誤差）
        device       : 計算設備
        batch_size   : 批次大小（避免 OOM）
        top_k_pixels : 每個樣本保留前 K 個最高強度像素

    使用範例::

        from core.gradcam import GradCAM, GradCAMAnalyzer

        cam      = GradCAM(model, target_layer="encoder_conv")
        analyzer = GradCAMAnalyzer(cam, model)
        report   = analyzer.analyze(X_normal, X_attack,
                                    field_names=["src_port", "dst_port", ...])
        analyzer.plot_feature_importance(report, output_dir="output/gradcam")
    """

    def __init__(
        self,
        grad_cam,
        model: nn.Module,
        device: Optional[torch.device] = None,
        batch_size: int = 64,
        top_k_pixels: int = 5,
    ):
        self.grad_cam     = grad_cam
        self.model        = model
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size   = batch_size
        self.top_k_pixels = top_k_pixels
        self.model.eval()

    # ── 主要分析介面 ──────────────────────────────────────────
    def analyze(
        self,
        X_normal: np.ndarray,
        X_attack: np.ndarray,
        max_samples_each: int = 200,
        field_names: Optional[List[str]] = None,
    ) -> AnalysisReport:
        """
        對正常與攻擊樣本執行批次 Grad-CAM 分析。

        Args:
            X_normal         : 正常流量，shape (N, H, W) 或 (N, 1, H, W)
            X_attack         : 攻擊流量，shape (M, H, W) 或 (M, 1, H, W)
            max_samples_each : 每類最多分析樣本數（避免過長等待）
            field_names      : 特徵欄位名稱（長度需等於影像高度 H）

        Returns:
            AnalysisReport
        """
        n_normal = min(len(X_normal), max_samples_each)
        n_attack = min(len(X_attack), max_samples_each)
        idx_n = np.random.choice(len(X_normal), n_normal, replace=False)
        idx_a = np.random.choice(len(X_attack), n_attack, replace=False)

        print(f"[GradCAMAnalyzer] 分析 {n_normal} 正常 + {n_attack} 攻擊樣本...")

        normal_results = self._analyze_batch(X_normal[idx_n], label="Normal")
        attack_results = self._analyze_batch(X_attack[idx_a], label="Attack")

        normal_cams  = np.stack([r.heatmap for r in normal_results])
        attack_cams  = np.stack([r.heatmap for r in attack_results])
        normal_mean  = normal_cams.mean(axis=0)
        attack_mean  = attack_cams.mean(axis=0)
        diff_cam     = attack_mean - normal_mean

        # 特徵重要性：diff_cam 在列方向取均值（行 = 特徵維度）
        feature_importance = diff_cam.mean(axis=1) if diff_cam.ndim == 2 else None

        report = AnalysisReport(
            normal_results    = normal_results,
            attack_results    = attack_results,
            normal_mean_cam   = normal_mean,
            attack_mean_cam   = attack_mean,
            diff_cam          = diff_cam,
            feature_importance= feature_importance,
            feature_names     = field_names,
        )
        print("[GradCAMAnalyzer] 分析完成。")
        return report

    def _analyze_batch(self, X: np.ndarray, label: str) -> List[SampleResult]:
        """對一批樣本執行 Grad-CAM 並收集結果。"""
        results = []
        for start in range(0, len(X), self.batch_size):
            batch    = X[start:start + self.batch_size]
            heatmaps = self.grad_cam.generate(batch)       # (B, H, W)
            scores   = self._compute_recon_scores(batch)   # (B,)

            for i, (hm, score) in enumerate(zip(heatmaps, scores)):
                flat     = hm.flatten()
                top_idx  = np.argsort(flat)[-self.top_k_pixels:][::-1]
                H, W     = hm.shape
                results.append(SampleResult(
                    sample_idx   = start + i,
                    label        = label,
                    recon_score  = float(score),
                    heatmap      = hm,
                    cam_mean     = float(hm.mean()),
                    cam_max      = float(hm.max()),
                    top_k_pixels = [(idx // W, idx % W) for idx in top_idx],
                ))
        return results

    # ── 重建工具函式 ──────────────────────────────────────────
    def _compute_recon_scores(self, X: np.ndarray) -> np.ndarray:
        """計算每個樣本的重建誤差（MSE）。"""
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            x_hat, _ = self.model(tensor)
            scores   = F.mse_loss(x_hat, tensor, reduction="none").mean(dim=[1, 2, 3])
        return scores.cpu().numpy()

    def get_reconstructions(self, X: np.ndarray) -> np.ndarray:
        """取得批次重建輸出（用於 plot_comparison 的 x_hat 參數）。"""
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        tensor  = torch.from_numpy(X.astype(np.float32))
        all_recs = []
        self.model.eval()
        for start in range(0, len(tensor), self.batch_size):
            batch = tensor[start:start + self.batch_size].to(self.device)
            with torch.no_grad():
                x_hat, _ = self.model(batch)
            all_recs.append(x_hat.squeeze(1).cpu().numpy())
        return np.concatenate(all_recs)

    # ── Top-K 異常樣本 ────────────────────────────────────────
    def get_top_anomalies(self, results: List[SampleResult], k: int = 10) -> List[SampleResult]:
        """回傳重建誤差最高的 k 個攻擊樣本。"""
        return sorted(
            [r for r in results if r.label == "Attack"],
            key=lambda r: r.recon_score, reverse=True
        )[:k]

    # ── 特徵重要性圖 ──────────────────────────────────────────
    def plot_feature_importance(
        self,
        report: AnalysisReport,
        output_dir: str,
        top_k: int = 20,
    ):
        """
        繪製特徵重要性水平長條圖。
        正值（紅色）= 攻擊樣本特別關注的特徵。
        負值（藍色）= 正常樣本特別關注的特徵。
        """
        os.makedirs(output_dir, exist_ok=True)
        if report.feature_importance is None:
            print("[GradCAMAnalyzer] 無特徵重要性資料。")
            return

        n_feat = len(report.feature_importance)
        names  = report.feature_names or [f"Feature_{i}" for i in range(n_feat)]
        k      = min(top_k, n_feat)
        top_idx   = np.argsort(np.abs(report.feature_importance))[-k:][::-1]
        top_imp   = report.feature_importance[top_idx]
        top_names = [names[i] if i < len(names) else f"F{i}" for i in top_idx]

        fig, ax = plt.subplots(figsize=(max(10, k * 0.8), 6))
        fig.patch.set_facecolor(_DARK_BG)
        ax.set_facecolor(_DARK_AX)

        colors = ["#ff6b6b" if v > 0 else "#58a6ff" for v in top_imp]
        ax.barh(range(k), top_imp[::-1], color=colors[::-1])
        ax.set_yticks(range(k))
        ax.set_yticklabels(top_names[::-1], color=_TEXT_COLOR, fontsize=9)
        ax.set_xlabel("重要性分數（攻擊 CAM 均值 − 正常 CAM 均值）", color=_TEXT_COLOR)
        ax.set_title(f"Grad-CAM 特徵重要性 Top-{k}", color=_TEXT_COLOR, fontsize=13)
        ax.tick_params(colors=_TEXT_COLOR)
        ax.axvline(0, color="#444", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        plt.tight_layout()
        save_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(save_path, bbox_inches="tight", facecolor=_DARK_BG, dpi=130)
        plt.close(fig)
        print(f"[GradCAMAnalyzer] 特徵重要性圖已儲存至: {save_path}")

    # ── 平均 CAM 對比圖 ───────────────────────────────────────
    def plot_mean_cam_comparison(self, report: AnalysisReport, output_dir: str):
        """三格圖：正常平均 CAM / 攻擊平均 CAM / 差異圖（攻擊 − 正常）。"""
        os.makedirs(output_dir, exist_ok=True)
        if report.normal_mean_cam is None:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor(_DARK_BG)
        fig.suptitle("Grad-CAM 分析：正常 vs 攻擊", color=_TEXT_COLOR, fontsize=13)

        panels = [
            (report.normal_mean_cam, "正常樣本平均 CAM", "YlOrBr",  False),
            (report.attack_mean_cam, "攻擊樣本平均 CAM", "hot",      False),
            (report.diff_cam,        "差異 CAM（攻擊 − 正常）", "RdBu_r", True),
        ]

        for ax, (data, title, cmap, is_div) in zip(axes, panels):
            ax.set_facecolor(_DARK_AX)
            vmax = np.abs(data).max() if data is not None else 1.0
            if is_div:
                im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
            else:
                im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set_title(title, color=_TEXT_COLOR, fontsize=11)
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors=_TEXT_COLOR)

        plt.tight_layout()
        save_path = os.path.join(output_dir, "mean_cam_comparison.png")
        plt.savefig(save_path, bbox_inches="tight", facecolor=_DARK_BG, dpi=130)
        plt.close(fig)
        print(f"[GradCAMAnalyzer] 平均 CAM 對比圖已儲存至: {save_path}")

    # ── CAM 強度分布圖 ────────────────────────────────────────
    def plot_cam_intensity_distribution(self, report: AnalysisReport, output_dir: str):
        """正常 vs 攻擊樣本的 CAM 平均強度分布（直方圖 + 盒鬚圖）。"""
        os.makedirs(output_dir, exist_ok=True)

        normal_int = np.array([r.cam_mean for r in report.normal_results])
        attack_int = np.array([r.cam_mean for r in report.attack_results])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor(_DARK_BG)
        fig.suptitle("Grad-CAM 強度分布：正常 vs 攻擊",
                     color=_TEXT_COLOR, fontsize=13)

        # 重疊直方圖
        ax1.set_facecolor(_DARK_AX)
        ax1.hist(normal_int, bins=30, alpha=0.7, label="Normal",
                 color="#3fb950", density=True)
        ax1.hist(attack_int, bins=30, alpha=0.7, label="Attack",
                 color="#ff6b6b", density=True)
        ax1.set_xlabel("CAM 平均強度", color=_TEXT_COLOR)
        ax1.set_ylabel("密度", color=_TEXT_COLOR)
        ax1.set_title("強度分布直方圖", color=_TEXT_COLOR)
        ax1.tick_params(colors=_TEXT_COLOR)
        ax1.legend(facecolor=_DARK_AX, labelcolor=_TEXT_COLOR)

        # 盒鬚圖
        ax2.set_facecolor(_DARK_AX)
        bp = ax2.boxplot([normal_int, attack_int],
                         tick_labels=["Normal", "Attack"],
                         patch_artist=True, notch=True)
        for patch, color in zip(bp["boxes"], ["#3fb950", "#ff6b6b"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for elem in ["whiskers", "caps", "medians", "fliers"]:
            for item in bp[elem]:
                item.set_color(_TEXT_COLOR)
        ax2.set_title("強度盒鬚圖", color=_TEXT_COLOR)
        ax2.tick_params(colors=_TEXT_COLOR)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#444")

        plt.tight_layout()
        save_path = os.path.join(output_dir, "cam_intensity_distribution.png")
        plt.savefig(save_path, bbox_inches="tight", facecolor=_DARK_BG, dpi=130)
        plt.close(fig)
        print(f"[GradCAMAnalyzer] 強度分布圖已儲存至: {save_path}")

    # ── 摘要統計 ──────────────────────────────────────────────
    def summarize(self, report: AnalysisReport) -> Dict:
        """從 AnalysisReport 提取可序列化的摘要統計字典。"""
        def _stats(results: List[SampleResult]) -> dict:
            if not results:
                return {}
            scores = np.array([r.recon_score for r in results])
            intens = np.array([r.cam_mean   for r in results])
            return {
                "count":              len(results),
                "recon_score_mean":   float(scores.mean()),
                "recon_score_std":    float(scores.std()),
                "recon_score_p95":    float(np.percentile(scores, 95)),
                "cam_intensity_mean": float(intens.mean()),
                "cam_intensity_std":  float(intens.std()),
            }

        summary = {
            "normal": _stats(report.normal_results),
            "attack": _stats(report.attack_results),
        }

        if report.feature_importance is not None and report.feature_names:
            top5_idx = np.argsort(np.abs(report.feature_importance))[-5:][::-1]
            summary["top5_important_features"] = [
                {
                    "name":       report.feature_names[i] if i < len(report.feature_names) else f"F{i}",
                    "importance": float(report.feature_importance[i]),
                }
                for i in top5_idx
            ]

        return summary
