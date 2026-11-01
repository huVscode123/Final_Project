# -*- coding: utf-8 -*-
"""Standalone smoke test for the corrected CAM implementation.

Run from the project root:
    python core/tests/test_gradcam_vae_variants_smoke.py

This test uses a tiny synthetic VAE-like model only to verify the CAM code
paths. It is NOT a CICIDS2017 experiment and its maps must not be reported as
research results.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = PROJECT_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from cnn_gradcam.gradcam_core import GradCAM


class TinyVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kl_weight = 1e-3
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.mu = nn.Linear(8 * 16 * 16, 16)
        self.logvar = nn.Linear(8 * 16 * 16, 16)
        self.decode_fc = nn.Linear(16, 8 * 16 * 16)

    def encode_distribution(self, x):
        feature = self.encoder_conv(x).flatten(1)
        return self.mu(feature), self.logvar(feature)

    def forward(self, x):
        mu, _ = self.encode_distribution(x)
        decoded = self.decode_fc(mu).view(-1, 8, 16, 16)
        return self.decoder(decoded), mu

    def loss_per_sample(self, x):
        mu, logvar = self.encode_distribution(x)
        reconstructed = self.decoder(
            self.decode_fc(mu).view(-1, 8, 16, 16)
        )
        recon = (reconstructed - x).pow(2).mean(dim=(1, 2, 3))
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)
        return recon, kl

    def reconstruction_error(self, x):
        recon, kl = self.loss_per_sample(x)
        return recon + self.kl_weight * kl


def main() -> None:
    torch.manual_seed(42)
    model = TinyVAE().eval()
    x = torch.rand(2, 1, 32, 32, dtype=torch.float32)

    for variant in ("gradcam", "gradcam++", "scorecam"):
        with GradCAM(
            model=model,
            target_layer=None,
            variant=variant,
            score_mode="vae",
            device=torch.device("cpu"),
            scorecam_batch_size=8,
            smooth_sigma=0.0,
        ) as cam:
            result = cam.generate(x, smooth=False)
            assert result.shape == (2, 32, 32), (variant, result.shape)
            assert torch.isfinite(torch.from_numpy(result)).all(), variant
            assert result.min() >= -1e-6, variant
            assert result.max() <= 1.0 + 1e-6, variant
            print(
                f"[PASS] {variant:9s} shape={result.shape} "
                f"range=({result.min():.4f}, {result.max():.4f})"
            )

    print("CAM smoke test passed. Synthetic maps are not research results.")


if __name__ == "__main__":
    main()
