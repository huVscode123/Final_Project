# -*- coding: utf-8 -*-
"""Check CNN-VAE checkpoint loading and Grad-CAM score consistency.

Run from project root:
    python core/run_gradcam_vae_check.py
This uses random tensors only for a technical smoke test, not experimental data.
"""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from model_registry import load_anomaly_model
from cnn_gradcam import GradCAM


def main():
    checkpoint = ROOT / "output/threshold_study/unsup_cicids2017/best_vae_cicids2017.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_anomaly_model(str(checkpoint), device=device)
    model = bundle.model.eval()
    x = torch.rand(2, 1, 32, 32, device=device)

    if not callable(getattr(model, "forward_with_scores", None)):
        raise RuntimeError("目前載入的模型沒有 forward_with_scores()，請確認已更新 VAE 程式碼。")

    with torch.no_grad():
        reference = model.reconstruction_error(x)
        _, _, gradcam_scores = model.forward_with_scores(x)
    max_diff = float((reference - gradcam_scores).abs().max().cpu())

    cam = GradCAM(
        model=model,
        target_layer="encoder_conv.13",
        variant="gradcam",
        target_type="mse",
        score_mode="vae",
        device=device,
        smooth_sigma=0.0,
    )
    heatmaps = cam.generate(x, smooth=False)

    print("model_type:", bundle.model_type)
    print("threshold:", bundle.threshold)
    print("resolved Grad-CAM layer:", cam._resolved_layer)
    print("score max absolute difference:", max_diff)
    print("heatmap shape:", tuple(heatmaps.shape))
    print("heatmap finite:", bool(np.isfinite(heatmaps).all()))
    print("heatmap range:", float(heatmaps.min()), float(heatmaps.max()))
    print("gradient diagnostics:", cam.last_diagnostics)

    if bundle.model_type != "cnn_vae":
        raise AssertionError(f"預期 cnn_vae，實際為 {bundle.model_type}")
    if max_diff > 1e-6:
        raise AssertionError(f"Grad-CAM 分數和正式 anomaly score 不一致：max_diff={max_diff}")
    if heatmaps.shape != (2, 32, 32):
        raise AssertionError(f"熱圖形狀不正確：{heatmaps.shape}")
    if not np.isfinite(heatmaps).all():
        raise AssertionError("熱圖含 NaN/Inf")
    print("PASS：技術 smoke test 通過。注意：隨機輸入不屬於正式實驗結果。")


if __name__ == "__main__":
    main()
