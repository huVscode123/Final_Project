# ============================================================
# core/gradcam/gradcam_hooks.py  - 前向 / 反向傳播 Hook 管理器
# ============================================================
"""
Hook 管理器：在指定層自動附加 forward hook 與 backward hook，
擷取特徵圖（activations）及其對應梯度（gradients）。

設計原則：
  - 支援按「層名稱」或「模組物件」指定目標層
  - 使用 context manager（with 語法）確保 hook 自動清除
  - 支援多層同時掛載（GradCAM++ 等進階方法所需）
  - 線程安全：每個 HookManager 實例獨立管理自己的 hook 列表
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union


class LayerHook:
    """
    單一層的 Hook 容器：同時持有 activation 與 gradient。

    Attributes:
        activation (torch.Tensor | None): forward pass 產生的特徵圖，shape (B, C, H, W)
        gradient   (torch.Tensor | None): backward pass 產生的梯度，shape (B, C, H, W)
        layer_name (str): 目標層的辨識名稱
    """

    def __init__(self, layer_name: str):
        self.layer_name: str = layer_name
        self.activation: Optional[torch.Tensor] = None
        self.gradient: Optional[torch.Tensor] = None
        self._fwd_handle = None
        self._bwd_handle = None

    # ── 前向 Hook ────────────────────────────────────────────
    def _forward_hook(self, module: nn.Module, input: Tuple, output: torch.Tensor):
        """保存 forward pass 輸出的特徵圖。"""
        out = output[0] if isinstance(output, (tuple, list)) else output
        self.activation = out

    # ── 反向 Hook ────────────────────────────────────────────
    def _backward_hook(self, module: nn.Module,
                       grad_input: Tuple, grad_output: Tuple):
        """保存 backward pass 的梯度（grad_output[0] 是該層輸出端的梯度）。"""
        self.gradient = grad_output[0].detach()

    # ── 掛載 / 移除 ──────────────────────────────────────────
    def attach(self, module: nn.Module):
        """在指定模組上附加 forward 與 backward hook。"""
        self._fwd_handle = module.register_forward_hook(self._forward_hook)
        self._bwd_handle = module.register_full_backward_hook(self._backward_hook)

    def remove(self):
        """移除已附加的 hook，釋放資源。"""
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None

    def clear(self):
        """清除暫存的 activation 與 gradient（不移除 hook）。"""
        self.activation = None
        self.gradient = None


# ═══════════════════════════════════════════════════════════
# HookManager：多層 Hook 統一管理
# ═══════════════════════════════════════════════════════════

class HookManager:
    """
    CNN 模型多層 Hook 管理器。

    功能：
      - 按層名稱（如 "encoder_conv.6"）自動找到對應 submodule 並掛載 hook
      - 支援一次掛載多層（GradCAM++ / LayerCAM 需要）
      - 作為 context manager 使用，離開時自動移除所有 hook

    使用範例：
        manager = HookManager(model)
        with manager.attach(["encoder_conv"]):
            output = model(x)
            loss.backward()
            activation = manager.get_activation("encoder_conv")
            gradient   = manager.get_gradient("encoder_conv")

    Args:
        model (nn.Module): 目標模型
    """

    def __init__(self, model: nn.Module):
        self.model: nn.Module = model
        self._hooks: Dict[str, LayerHook] = {}

    # ── 尋找 submodule ────────────────────────────────────────
    def _find_module(self, layer_name: str) -> nn.Module:
        """
        根據名稱取得模型中的 submodule。
        支援：
          - 完整名稱路徑：如 "encoder_conv.6"
          - 部分末尾匹配：如 "encoder_conv"（自動選取最後一層 Conv2d）
          - 傳入 nn.Module 物件（直接回傳）
        """
        if isinstance(layer_name, nn.Module):
            return layer_name

        # 先嘗試精確名稱
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module

        # 再嘗試名稱含有 layer_name 且是 Conv2d 的最後一個
        candidates: List[Tuple[str, nn.Module]] = [
            (n, m) for n, m in self.model.named_modules()
            if layer_name in n and isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))
        ]
        if candidates:
            conv_cands = [(n, m) for n, m in candidates if isinstance(m, nn.Conv2d)]
            chosen_name, chosen_mod = (conv_cands or candidates)[-1]
            return chosen_mod

        raise ValueError(
            f"[HookManager] 找不到層 '{layer_name}'。\n"
            f"可用層：{[n for n, _ in self.model.named_modules() if n]}"
        )

    # ── 掛載 hooks ────────────────────────────────────────────
    def attach(self, layer_names: Union[str, List[str]]) -> "HookManager":
        """
        在指定層掛載 hook。

        Args:
            layer_names: 單一層名稱或層名稱列表

        Returns:
            self（支援鏈式呼叫與 context manager）
        """
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        for name in layer_names:
            module = self._find_module(name)
            hook = LayerHook(layer_name=name)
            hook.attach(module)
            self._hooks[name] = hook

        return self

    # ── Context Manager ───────────────────────────────────────
    def __enter__(self) -> "HookManager":
        return self

    def __exit__(self, *args):
        self.remove_all()

    # ── 移除 hooks ────────────────────────────────────────────
    def remove_all(self):
        """移除所有已掛載的 hook 並清除暫存資料。"""
        for hook in self._hooks.values():
            hook.remove()
        self._hooks.clear()

    def clear_all(self):
        """清除所有暫存的 activation / gradient（保留 hook）。"""
        for hook in self._hooks.values():
            hook.clear()

    # ── 資料存取 ──────────────────────────────────────────────
    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        """取得指定層的 activation。"""
        hook = self._hooks.get(layer_name)
        if hook is None:
            raise KeyError(f"[HookManager] 層 '{layer_name}' 未掛載 hook。")
        return hook.activation

    def get_gradient(self, layer_name: str) -> Optional[torch.Tensor]:
        """取得指定層的 gradient。"""
        hook = self._hooks.get(layer_name)
        if hook is None:
            raise KeyError(f"[HookManager] 層 '{layer_name}' 未掛載 hook。")
        return hook.gradient

    def get_all(self, layer_name: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """同時取得 (activation, gradient)。"""
        return self.get_activation(layer_name), self.get_gradient(layer_name)

    def list_hooked_layers(self) -> List[str]:
        """回傳已掛載 hook 的層名稱列表。"""
        return list(self._hooks.keys())

    # ── 便利方法：列出所有可用層 ──────────────────────────────
    @staticmethod
    def list_available_layers(model: nn.Module,
                               include_types: Tuple = (nn.Conv2d,)) -> List[str]:
        """
        列出模型中所有指定類型的層名稱。

        Args:
            model        : 目標模型
            include_types: 只列出這些類型的層（預設只列 Conv2d）

        Returns:
            層名稱列表
        """
        return [
            name for name, module in model.named_modules()
            if name and isinstance(module, include_types)
        ]
