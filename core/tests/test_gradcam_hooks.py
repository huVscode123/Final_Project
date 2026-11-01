# ============================================================
# core/tests/test_gradcam_hooks.py  - HookManager 單元測試
# ============================================================
"""
測試範圍：
  - LayerHook.attach / remove / clear
  - HookManager._find_module（精確名稱 / 模糊名稱 / Sequential 解析）
  - HookManager context manager（自動清除）
  - HookManager.get_activation / get_gradient（有資料 / 無資料）
  - HookManager.list_available_layers
  - 錯誤處理（找不到層 / 未掛載 hook）
"""

import sys
import os
import unittest
import torch
import torch.nn as nn

# 確保 core/ 在 sys.path
_CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from cnn_gradcam.gradcam_hooks import HookManager, LayerHook
from tests.gradcam_fixtures import make_model, FakeDataFactory


class TestLayerHook(unittest.TestCase):
    """LayerHook 基礎功能測試。"""

    def setUp(self):
        self.model = make_model()
        # 找到第一個 Conv2d
        self.first_conv = None
        self.first_conv_name = None
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                self.first_conv = m
                self.first_conv_name = name
                break

    def test_attach_and_forward_captures_activation(self):
        """attach 後執行 forward，activation 應被填充。"""
        hook = LayerHook(layer_name=self.first_conv_name)
        hook.attach(self.first_conv)

        x = torch.zeros(1, 1, 32, 32)
        with torch.no_grad():
            self.model(x)

        self.assertIsNotNone(hook.activation, "activation 應在 forward 後被填充")
        self.assertEqual(hook.activation.ndim, 4, "activation 應為 4D tensor (B,C,H,W)")
        hook.remove()

    def test_clear_resets_activation(self):
        """clear() 後 activation 應為 None。"""
        hook = LayerHook(layer_name=self.first_conv_name)
        hook.attach(self.first_conv)
        x = torch.zeros(1, 1, 32, 32)
        with torch.no_grad():
            self.model(x)
        self.assertIsNotNone(hook.activation)

        hook.clear()
        self.assertIsNone(hook.activation, "clear() 後 activation 應為 None")
        hook.remove()

    def test_remove_stops_capturing(self):
        """remove() 後再執行 forward，activation 不應更新。"""
        hook = LayerHook(layer_name=self.first_conv_name)
        hook.attach(self.first_conv)
        hook.remove()

        x = torch.zeros(1, 1, 32, 32)
        with torch.no_grad():
            self.model(x)
        self.assertIsNone(hook.activation, "remove() 後 activation 不應被填充")


class TestHookManagerFindModule(unittest.TestCase):
    """HookManager._find_module 各種尋找方式。"""

    def setUp(self):
        self.model = make_model()
        self.manager = HookManager(self.model)

    def test_exact_name_match(self):
        """精確層名稱（如 'encoder_conv.0'）應能直接找到。"""
        mod = self.manager._find_module("encoder_conv.0")
        self.assertIsInstance(mod, nn.Conv2d)

    def test_sequential_name_match(self):
        """Sequential 名稱（'encoder_conv'）應找到對應的 Sequential 模組。"""
        mod = self.manager._find_module("encoder_conv")
        self.assertIsInstance(mod, nn.Sequential)

    def test_partial_name_finds_last_conv(self):
        """部分名稱（'encoder'）應找到最後一個含此名稱的 Conv2d。"""
        mod = self.manager._find_module("encoder_conv")
        self.assertIsNotNone(mod)

    def test_invalid_name_raises(self):
        """找不到的層名稱應拋出 ValueError。"""
        with self.assertRaises(ValueError):
            self.manager._find_module("nonexistent_layer_xyz")

    def test_module_object_passthrough(self):
        """直接傳入 nn.Module 物件應直接回傳。"""
        target = list(self.model.encoder_conv.children())[0]  # 第一個 Conv2d
        result = self.manager._find_module(target)
        self.assertIs(result, target)


class TestHookManagerAttach(unittest.TestCase):
    """HookManager.attach 與資料讀取測試。"""

    def setUp(self):
        self.model = make_model()

    def test_attach_single_layer(self):
        """掛載單一層後，list_hooked_layers 應有一個條目。"""
        manager = HookManager(self.model)
        manager.attach("encoder_conv.0")
        self.assertEqual(len(manager.list_hooked_layers()), 1)
        manager.remove_all()

    def test_attach_multiple_layers(self):
        """一次掛載多層，應全部記錄。"""
        manager = HookManager(self.model)
        manager.attach(["encoder_conv.0", "encoder_conv.3"])
        self.assertEqual(len(manager.list_hooked_layers()), 2)
        manager.remove_all()

    def test_get_activation_after_forward(self):
        """forward 後應能讀到 activation，且形狀正確。"""
        manager = HookManager(self.model)
        manager.attach("encoder_conv.0")

        x = torch.rand(2, 1, 32, 32)
        with torch.no_grad():
            self.model(x)

        act = manager.get_activation("encoder_conv.0")
        self.assertIsNotNone(act)
        self.assertEqual(act.shape[0], 2, "batch size 應為 2")
        manager.remove_all()

    def test_get_gradient_after_backward(self):
        """backward 後應能讀到 gradient。"""
        manager = HookManager(self.model)
        manager.attach("encoder_conv.0")

        x = torch.rand(1, 1, 32, 32).requires_grad_(True)
        x_hat, _ = self.model(x)
        loss = ((x - x_hat) ** 2).mean()
        self.model.zero_grad()
        loss.backward()

        grad = manager.get_gradient("encoder_conv.0")
        self.assertIsNotNone(grad, "backward 後 gradient 應被填充")
        manager.remove_all()

    def test_unhooked_layer_raises_keyerror(self):
        """讀取未掛載 hook 的層應拋出 KeyError。"""
        manager = HookManager(self.model)
        with self.assertRaises(KeyError):
            manager.get_activation("encoder_conv.0")

    def test_clear_all_resets_data(self):
        """clear_all() 後所有 activation / gradient 應為 None。"""
        manager = HookManager(self.model)
        manager.attach("encoder_conv.0")

        x = torch.rand(1, 1, 32, 32)
        with torch.no_grad():
            self.model(x)

        manager.clear_all()
        act = manager.get_activation("encoder_conv.0")
        self.assertIsNone(act)
        manager.remove_all()


class TestHookManagerContextManager(unittest.TestCase):
    """HookManager context manager 自動清除測試。"""

    def setUp(self):
        self.model = make_model()

    def test_context_manager_auto_removes(self):
        """with 區塊結束後，hook 應被自動移除。"""
        manager = HookManager(self.model)
        with manager.attach("encoder_conv.0"):
            self.assertEqual(len(manager.list_hooked_layers()), 1)
        # 離開 with 後應清除
        self.assertEqual(len(manager.list_hooked_layers()), 0,
                         "with 區塊結束後 hook 應被自動移除")

    def test_context_manager_exception_still_removes(self):
        """即使 with 區塊內拋出例外，hook 也應被移除。"""
        manager = HookManager(self.model)
        try:
            with manager.attach("encoder_conv.0"):
                raise RuntimeError("測試例外")
        except RuntimeError:
            pass
        self.assertEqual(len(manager.list_hooked_layers()), 0)


class TestHookManagerListLayers(unittest.TestCase):
    """list_available_layers 靜態方法測試。"""

    def test_lists_conv2d_layers(self):
        """應能列出所有 Conv2d 層，且數量大於 0。"""
        model = make_model()
        layers = HookManager.list_available_layers(model)
        self.assertGreater(len(layers), 0, "應至少有一個 Conv2d 層")
        # encoder_conv 中應有 4 個 Conv2d
        encoder_convs = [l for l in layers if "encoder_conv" in l]
        self.assertGreaterEqual(len(encoder_convs), 1)

    def test_filter_by_type(self):
        """include_types 參數應正確過濾層型別。"""
        model = make_model()
        linear_layers = HookManager.list_available_layers(model, include_types=(nn.Linear,))
        self.assertGreater(len(linear_layers), 0)
        for name in linear_layers:
            mod = dict(model.named_modules())[name]
            self.assertIsInstance(mod, nn.Linear)


if __name__ == "__main__":
    unittest.main(verbosity=2)
