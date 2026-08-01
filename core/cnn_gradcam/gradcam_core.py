# ============================================================
# core/cnn_gradcam/gradcam_core.py  - Grad-CAM 核心演算法（修正版）
#
# 修正清單：
#   [Bug 6] _gradcam_pp：缺少 model.zero_grad() 呼叫。
#           若前一次 backward 留有殘餘梯度，Hook 捕捉的
#           gradient 為污染值，alpha 係數計算偏差。
#           修正：在 backward 前加入 self.model.zero_grad()，
#           並移除不必要的 create_graph=True（_gradcam_pp
#           只需一階梯度，create_graph 會增加不必要的計算量）。
# ============================================================

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal

from .gradcam_hooks import HookManager

_DEFAULT_TARGET_LAYER = "encoder_conv"


class GradCAM:
    """
    CNN Autoencoder 的 Grad-CAM 視覺化工具。（修正版）
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str = _DEFAULT_TARGET_LAYER,
        variant: Literal["gradcam", "gradcam++", "scorecam"] = "gradcam",
        target_type: Literal["mse", "pixel", "channel"] = "mse",
        device: Optional[torch.device] = None,
    ):
        self.model        = model
        self.target_layer = target_layer
        self.variant      = variant
        self.target_type  = target_type
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model        = self.model.to(self.device)

        self._resolved_layer = self._resolve_target_layer(target_layer)
        self._hook_manager   = HookManager(self.model)
        self._hook_manager.attach([self._resolved_layer])

    # ── 目標層解析 ────────────────────────────────────────
    def _resolve_target_layer(self, layer_name: str) -> str:
        for name, module in self.model.named_modules():
            if name == layer_name:
                if isinstance(module, nn.Sequential):
                    last_conv = None
                    for sub_name, sub_mod in self.model.named_modules():
                        if sub_name.startswith(layer_name + ".") and isinstance(sub_mod, nn.Conv2d):
                            last_conv = sub_name
                    if last_conv:
                        return last_conv
                return name

        candidates = [
            n for n, m in self.model.named_modules()
            if layer_name in n and isinstance(m, nn.Conv2d)
        ]
        if candidates:
            return candidates[-1]

        raise ValueError(
            f"[GradCAM] 找不到目標層 '{layer_name}'。\n"
            f"可用 Conv2d 層：{HookManager.list_available_layers(self.model)}"
        )

    # ── 主要介面：generate ────────────────────────────────
    def generate(
        self,
        x: torch.Tensor,
        pixel_coords: Optional[Tuple[int, int]] = None,
        channel_idx: int = 0,
        smooth: bool = True,
    ) -> np.ndarray:
        x = self._preprocess_input(x)

        if self.variant == "gradcam":
            cam = self._gradcam(x, pixel_coords, channel_idx)
        elif self.variant == "gradcam++":
            cam = self._gradcam_pp(x, pixel_coords, channel_idx)
        elif self.variant == "scorecam":
            cam = self._scorecam(x)
        else:
            raise ValueError(f"不支援的 variant: {self.variant}")

        H, W = x.shape[-2], x.shape[-1]
        cam_tensor = torch.from_numpy(cam).unsqueeze(1).float()
        cam_up     = F.interpolate(cam_tensor, size=(H, W),
                                   mode="bilinear", align_corners=False)
        cam_np = cam_up.squeeze(1).numpy()
        cam_np = self._normalize_batch(cam_np)

        if smooth:
            cam_np = self._gaussian_smooth(cam_np)

        return cam_np

    # ── Grad-CAM ──────────────────────────────────────────
    def _gradcam(self, x: torch.Tensor,
                 pixel_coords, channel_idx: int) -> np.ndarray:
        """
        原版 Grad-CAM：
          weights = GAP( d(score) / d(A) )
          CAM     = ReLU( sum_k( weights_k * A_k ) )
        """
        self._hook_manager.clear_all()
        self.model.eval()

        x = x.to(self.device).requires_grad_(True)
        x_hat, z = self.model(x)
        score = self._compute_score(x, x_hat, pixel_coords, channel_idx)
        self.model.zero_grad()
        score.backward(retain_graph=True)

        activation = self._hook_manager.get_activation(self._resolved_layer)
        gradient   = self._hook_manager.get_gradient(self._resolved_layer)

        weights = gradient.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * activation.detach()).sum(dim=1)
        cam     = F.relu(cam)
        return cam.cpu().numpy()

    def _gradcam_pp(self, x: torch.Tensor,
                    pixel_coords, channel_idx: int) -> np.ndarray:
        """
        Grad-CAM++（修正版）：
        使用二階梯度（alpha 係數）改善多重激活區域的定位精度。

        [Bug 6 修正] 原版缺少 model.zero_grad()，若前一次
        backward 留有殘餘梯度，Hook 的 gradient 為污染值。
        此外原版使用 create_graph=True，但計算 alpha 只需
        一階梯度，移除 create_graph 可節省記憶體與計算量。
        """
        self._hook_manager.clear_all()
        self.model.eval()

        x = x.to(self.device).requires_grad_(True)
        x_hat, z = self.model(x)
        score = self._compute_score(x, x_hat, pixel_coords, channel_idx)

        # [Bug 6 修正] 清除殘餘梯度，再執行 backward
        self.model.zero_grad()
        # create_graph=True 已移除：_gradcam_pp 只需一階梯度
        score.backward(retain_graph=True)

        activation = self._hook_manager.get_activation(self._resolved_layer).detach()
        gradient   = self._hook_manager.get_gradient(self._resolved_layer)

        if gradient is None:
            return self._gradcam(x.detach(), pixel_coords, channel_idx)

        grad_sq    = gradient ** 2
        grad_cb    = gradient ** 3
        alpha_numer = grad_sq
        alpha_denom = (2 * grad_sq
                       + activation * grad_cb.sum(dim=[2, 3], keepdim=True)
                       + 1e-9)
        alpha   = alpha_numer / alpha_denom
        weights = (alpha * F.relu(gradient)).mean(dim=[2, 3], keepdim=True)

        cam = (weights * activation).sum(dim=1)
        cam = F.relu(cam)
        return cam.cpu().numpy()

    def _scorecam(self, x: torch.Tensor) -> np.ndarray:
        """
        Score-CAM（無梯度，以遮罩擾動估算重要性）。
        計算量 O(C) 次 forward，適合解釋單筆樣本。
        """
        self._hook_manager.clear_all()
        self.model.eval()

        B, C_in, H, W = x.shape
        x = x.to(self.device)

        with torch.no_grad():
            x_hat_base, _ = self.model(x)
            baseline_score = F.mse_loss(x_hat_base, x, reduction="none").mean(dim=[1, 2, 3])

        activation = self._hook_manager.get_activation(self._resolved_layer).detach()

        B, C, h, w = activation.shape
        act_up = F.interpolate(activation, size=(H, W), mode="bilinear", align_corners=False)

        act_min  = act_up.view(B, C, -1).min(dim=2)[0].view(B, C, 1, 1)
        act_max  = act_up.view(B, C, -1).max(dim=2)[0].view(B, C, 1, 1)
        act_norm = (act_up - act_min) / (act_max - act_min + 1e-9)

        weights = torch.zeros(B, C, device=self.device)
        with torch.no_grad():
            for c in range(C):
                mask        = act_norm[:, c:c+1, :, :]
                x_masked    = x * mask
                x_hat_m, _  = self.model(x_masked)
                score_m     = F.mse_loss(x_hat_m, x_masked,
                                          reduction="none").mean(dim=[1, 2, 3])
                weights[:, c] = score_m - baseline_score

        weights = F.relu(weights).view(B, C, 1, 1)
        cam     = (weights * activation).sum(dim=1)
        cam     = F.relu(cam)
        return cam.cpu().numpy()

    # ── 目標函數 ──────────────────────────────────────────
    def _compute_score(self, x, x_hat, pixel_coords, channel_idx) -> torch.Tensor:
        error_map = (x - x_hat) ** 2
        if self.target_type == "pixel" and pixel_coords is not None:
            r, c = pixel_coords
            return error_map[:, :, r, c].mean()
        elif self.target_type == "channel":
            return error_map[:, channel_idx, :, :].mean()
        else:
            return error_map.mean()

    # ── 工具函式 ──────────────────────────────────────────
    @staticmethod
    def _preprocess_input(x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(0) if x.shape[0] == 1 else x.unsqueeze(1)
        return x.float()

    @staticmethod
    def _normalize_batch(cam: np.ndarray) -> np.ndarray:
        B      = cam.shape[0]
        result = np.zeros_like(cam)
        for i in range(B):
            mi, ma = cam[i].min(), cam[i].max()
            result[i] = ((cam[i] - mi) / (ma - mi)
                         if ma - mi > 1e-8 else np.zeros_like(cam[i]))
        return result

    @staticmethod
    def _gaussian_smooth(cam: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """[P3 修正] 用 scipy 取代純 Python 迴圈"""
        if sigma <= 0:
            return cam
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(cam, sigma=sigma, axis=-2, mode='reflect')
        smoothed = gaussian_filter1d(smoothed, sigma=sigma, axis=-1, mode='reflect')
        return smoothed

    def __del__(self):
        try:
            self._hook_manager.remove_all()
        except Exception:
            pass

    def list_target_layers(self) -> list:
        return HookManager.list_available_layers(self.model)