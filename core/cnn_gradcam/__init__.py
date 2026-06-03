# ============================================================
# core/cnn_gradcam/__init__.py  - Grad-CAM 視覺化解釋套件
# ============================================================
"""
Grad-CAM 視覺化解釋套件（針對 CNN Autoencoder 重建誤差）

放置位置：core/cnn_gradcam/
套件命名使用 cnn_gradcam 以避免與 PyPI 的 gradcam 套件衝突。

模組結構：
    gradcam_hooks.py      - 前向 / 梯度 Hook 管理器
    gradcam_core.py       - Grad-CAM 核心演算法（三種變體）
    gradcam_visualizer.py - 熱力圖生成與疊圖視覺化
    gradcam_analyzer.py   - 批次分析、特徵重要性排序
    gradcam_reporter.py   - JSON / HTML 報告匯出

快速使用範例：
    from cnn_gradcam import GradCAM, GradCAMVisualizer, GradCAMAnalyzer

    cam = GradCAM(model, target_layer="encoder_conv")
    heatmap = cam.generate(x_tensor)                  # shape: (B, H, W) numpy

    viz = GradCAMVisualizer()
    viz.overlay(x_tensor, heatmap, save_path="cam.png")
"""

from .gradcam_core import GradCAM
from .gradcam_hooks import HookManager
from .gradcam_visualizer import GradCAMVisualizer
from .gradcam_analyzer import GradCAMAnalyzer
from .gradcam_reporter import GradCAMReporter

__all__ = [
    "GradCAM",
    "HookManager",
    "GradCAMVisualizer",
    "GradCAMAnalyzer",
    "GradCAMReporter",
]

__version__ = "1.0.0"
