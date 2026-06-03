# ============================================================
# core/gradcam/gradcam_visualizer.py  - 熱力圖生成與視覺化模組
# ============================================================
"""
GradCAMVisualizer：將 Grad-CAM 產生的熱力圖轉換為視覺化圖像。

功能：
  - 將 CAM 疊加在原始輸入影像上（色彩映射：jet / plasma / inferno）
  - 四格並排對比：原始輸入 / 重建輸出 / 殘差圖 / Grad-CAM 疊圖
  - 批次輸出（一次處理多個樣本）
  - 深色背景風格（與既有專案視覺化風格一致）
  - 正常 vs 攻擊樣本的 Grad-CAM 摘要網格圖
  - 重建誤差 vs CAM 強度散點圖
"""

from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, List

import matplotlib.font_manager as _fm
_CJK = ["Microsoft YaHei", "SimHei", "PingFang TC",
        "Heiti TC", "WenQuanYi Zen Hei", "Noto Sans CJK TC"]
_avail = {f.name for f in _fm.fontManager.ttflist}
for _f in _CJK:
    if _f in _avail:
        matplotlib.rcParams["font.family"] = [_f, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break

_DARK_BG   = "#0d1117"
_DARK_AX   = "#161b22"
_TEXT_COLOR = "#e6edf3"


class GradCAMVisualizer:
    """
    Grad-CAM 視覺化工具。

    Args:
        colormap   (str)  : 熱力圖色彩映射（預設: "jet"）
        alpha      (float): CAM 疊圖透明度，0 = 原圖，1 = 只顯示 CAM
        dark_theme (bool) : 深色主題（預設: True，與專案風格一致）
    """

    def __init__(
        self,
        colormap: str = "jet",
        alpha: float = 0.5,
        dark_theme: bool = True,
    ):
        self.colormap = colormap
        self.alpha    = alpha
        self._bg  = _DARK_BG   if dark_theme else "white"
        self._ax  = _DARK_AX   if dark_theme else "#f0f0f0"
        self._txt = _TEXT_COLOR if dark_theme else "black"

    # ── 基礎疊圖 ──────────────────────────────────────────────
    def cam_to_rgb(self, heatmap: np.ndarray) -> np.ndarray:
        """
        將單張 [0,1] 灰階 CAM 轉換為 RGB 色彩圖。

        Args:
            heatmap: shape (H, W)，值域 [0, 1]

        Returns:
            RGB array, shape (H, W, 3)，dtype float32
        """
        cmap = matplotlib.colormaps.get_cmap(self.colormap)
        return cmap(heatmap)[:, :, :3].astype(np.float32)

    def overlay(
        self,
        x: np.ndarray,
        heatmap: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "",
        score: Optional[float] = None,
    ) -> np.ndarray:
        """
        將 Grad-CAM 熱力圖疊加在原始輸入影像上。

        Args:
            x         : 原始輸入，shape (H, W) 或 (1, H, W)，值域 [0, 1]
            heatmap   : Grad-CAM 輸出，shape (H, W)，值域 [0, 1]
            save_path : 儲存路徑（None 則不儲存）
            title     : 圖表標題
            score     : 異常分數（顯示於標題下方）

        Returns:
            疊圖 RGB numpy array, shape (H, W, 3)
        """
        if x.ndim == 3:
            x = x.squeeze(0)

        img_rgb = np.stack([x, x, x], axis=-1)
        cam_rgb = self.cam_to_rgb(heatmap)
        blended = np.clip((1 - self.alpha) * img_rgb + self.alpha * cam_rgb, 0, 1)

        if save_path:
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor(self._bg)
            ax.set_facecolor(self._ax)
            ax.imshow(blended)
            ax.axis("off")
            full_title = title + (f"\n重建誤差: {score:.5f}" if score is not None else "")
            ax.set_title(full_title, color=self._txt, fontsize=11, pad=10)
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", facecolor=self._bg, dpi=120)
            plt.close(fig)

        return blended

    # ── 四格並排對比圖 ────────────────────────────────────────
    def plot_comparison(
        self,
        x: np.ndarray,
        x_hat: np.ndarray,
        heatmap: np.ndarray,
        save_path: Optional[str] = None,
        sample_idx: int = 0,
        label: str = "Unknown",
        score: Optional[float] = None,
        field_names: Optional[List[str]] = None,
    ):
        """
        四格並排對比圖：
          [1] 原始輸入  [2] 重建輸出  [3] 殘差圖  [4] Grad-CAM 疊圖

        Args:
            x          : 原始輸入，shape (H, W)
            x_hat      : 重建輸出，shape (H, W)
            heatmap    : Grad-CAM，shape (H, W)
            save_path  : 儲存路徑（None 則不儲存）
            sample_idx : 樣本索引（顯示於標題）
            label      : "Normal" / "Attack"
            score      : 重建誤差
            field_names: 特徵欄位名稱（y 軸標注）
        """
        if x.ndim == 3:
            x = x.squeeze(0)
        if x_hat.ndim == 3:
            x_hat = x_hat.squeeze(0)

        residual = np.abs(x - x_hat)
        blended  = self.overlay(x, heatmap)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.patch.set_facecolor(self._bg)

        label_color = "#ff6b6b" if label == "Attack" else "#3fb950"
        score_str   = f"  重建誤差={score:.5f}" if score is not None else ""
        fig.suptitle(
            f"樣本 #{sample_idx} [{label}]{score_str}",
            color=label_color, fontsize=13, fontweight="bold"
        )

        panels = [
            (x,        "原始輸入",                   "gray", False),
            (x_hat,    "重建輸出",                   "gray", False),
            (residual, "殘差圖",                     "hot",  False),
            (blended,  f"Grad-CAM ({self.colormap})", None,  True),
        ]

        for ax, (img, title, cmap, is_rgb) in zip(axes, panels):
            ax.set_facecolor(self._ax)
            if is_rgb:
                ax.imshow(np.clip(img, 0, 1))
            else:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, color=self._txt, fontsize=11)
            ax.axis("off")

            if title == "原始輸入" and field_names and len(field_names) == img.shape[0]:
                ax.set_yticks(range(len(field_names)))
                ax.set_yticklabels(field_names, fontsize=6, color=self._txt)
                ax.yaxis.set_visible(True)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", facecolor=self._bg, dpi=130)
        plt.close(fig)

    # ── 批次輸出 ──────────────────────────────────────────────
    def plot_batch(
        self,
        X: np.ndarray,
        X_hat: np.ndarray,
        heatmaps: np.ndarray,
        labels: List[str],
        scores: List[float],
        output_dir: str,
        max_samples: int = 12,
        field_names: Optional[List[str]] = None,
    ):
        """
        批次生成對比圖並儲存到指定目錄。

        Args:
            X          : 原始輸入，shape (N, H, W) 或 (N, 1, H, W)
            X_hat      : 重建輸出，shape (N, H, W) 或 (N, 1, H, W)
            heatmaps   : Grad-CAM，shape (N, H, W)
            labels     : 每個樣本的標籤
            scores     : 每個樣本的重建誤差
            output_dir : 輸出目錄
            max_samples: 最多生成張數（避免大量輸出）
            field_names: 特徵欄位名稱
        """
        os.makedirs(output_dir, exist_ok=True)
        if X.ndim == 4:
            X = X.squeeze(1)
        if X_hat.ndim == 4:
            X_hat = X_hat.squeeze(1)

        n = min(len(X), max_samples)
        for i in range(n):
            label = labels[i] if i < len(labels) else "Unknown"
            score = float(scores[i]) if i < len(scores) else 0.0
            save_path = os.path.join(output_dir, f"sample_{i:04d}_{label.lower()}.png")
            self.plot_comparison(
                x=X[i], x_hat=X_hat[i], heatmap=heatmaps[i],
                save_path=save_path,
                sample_idx=i, label=label, score=score, field_names=field_names,
            )
        print(f"[GradCAMVisualizer] 已儲存 {n} 張對比圖至: {output_dir}")

    # ── 摘要網格圖 ────────────────────────────────────────────
    def plot_summary_grid(
        self,
        normal_heatmaps: np.ndarray,
        attack_heatmaps: np.ndarray,
        output_dir: str,
        n_cols: int = 8,
    ):
        """
        生成正常 vs 攻擊樣本的 Grad-CAM 摘要網格圖。
        上半部為正常樣本，下半部為攻擊樣本。
        """
        os.makedirs(output_dir, exist_ok=True)
        n_show = min(n_cols, len(normal_heatmaps), len(attack_heatmaps))

        fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 6))
        fig.patch.set_facecolor(self._bg)
        fig.suptitle(
            "Grad-CAM 摘要：正常流量 (上) vs 攻擊流量 (下)",
            color=self._txt, fontsize=13, fontweight="bold"
        )

        row_info = [("正常 (Normal)", "#3fb950", normal_heatmaps[:n_show]),
                    ("攻擊 (Attack)", "#ff6b6b", attack_heatmaps[:n_show])]

        for row_idx, (row_label, row_color, heatmaps) in enumerate(row_info):
            for col_idx in range(n_show):
                ax = axes[row_idx, col_idx] if n_show > 1 else axes[row_idx]
                ax.set_facecolor(self._ax)
                ax.imshow(self.cam_to_rgb(heatmaps[col_idx]))
                ax.axis("off")
                if col_idx == 0:
                    ax.set_ylabel(row_label, color=row_color, fontsize=9, labelpad=5)
                    ax.yaxis.set_visible(True)
                    ax.set_yticks([])

        plt.tight_layout()
        save_path = os.path.join(output_dir, "gradcam_summary_grid.png")
        plt.savefig(save_path, bbox_inches="tight", facecolor=self._bg, dpi=130)
        plt.close(fig)
        print(f"[GradCAMVisualizer] 摘要網格圖已儲存至: {save_path}")

    # ── 重建誤差 vs CAM 強度散點圖 ────────────────────────────
    def plot_score_vs_cam_intensity(
        self,
        scores: np.ndarray,
        cam_intensities: np.ndarray,
        labels: List[str],
        output_dir: str,
    ):
        """
        散點圖：重建誤差（x軸）vs CAM 平均強度（y軸），以標籤著色。
        """
        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(self._bg)
        ax.set_facecolor(self._ax)

        for label_val, color in [("Normal", "#3fb950"), ("Attack", "#ff6b6b")]:
            mask = np.array([l == label_val for l in labels])
            if mask.any():
                ax.scatter(scores[mask], cam_intensities[mask],
                           c=color, label=label_val, alpha=0.6, s=15, edgecolors="none")

        ax.set_xlabel("重建誤差（MSE）", color=self._txt)
        ax.set_ylabel("Grad-CAM 平均強度", color=self._txt)
        ax.set_title("重建誤差 vs Grad-CAM 強度", color=self._txt)
        ax.tick_params(colors=self._txt)
        ax.legend(facecolor=self._ax, labelcolor=self._txt)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        save_path = os.path.join(output_dir, "score_vs_cam_intensity.png")
        plt.savefig(save_path, bbox_inches="tight", facecolor=self._bg, dpi=120)
        plt.close(fig)
        print(f"[GradCAMVisualizer] 散點圖已儲存至: {save_path}")
