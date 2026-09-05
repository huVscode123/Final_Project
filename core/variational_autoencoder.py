"""PyTorch CNN-VAE for normal-traffic anomaly detection.

Implements the VAE approach used by the referenced project while retaining the
32x32 packet-image interface required by this project and its Grad-CAM tools.
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class CNNVariationalAutoencoder(nn.Module):
    model_type = 'cnn_vae'

    def __init__(self, latent_dim=32, image_size=32, kl_weight=1e-3):
        super().__init__()
        self.latent_dim, self.image_size, self.kl_weight = latent_dim, image_size, kl_weight
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.encoder_fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU())
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.decoder_fc = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 64 * 4 * 4), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid(),
        )

    def encode_distribution(self, x):
        h = self.encoder_fc(self.encoder_conv(x))
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, logvar = self.encode_distribution(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        x_hat = self.decoder(self.decoder_fc(z).view(-1, 64, 4, 4))
        return x_hat, mu

    def loss_per_sample(self, x):
        mu, logvar = self.encode_distribution(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        x_hat = self.decoder(self.decoder_fc(z).view(-1, 64, 4, 4))
        recon = ((x - x_hat) ** 2).mean(dim=(1, 2, 3))
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean(dim=1)
        return recon, kl

    def vae_loss(self, x):
        recon, kl = self.loss_per_sample(x)
        return (recon + self.kl_weight * kl).mean()

    def reconstruction_error(self, x):
        """Combined reconstruction/KL anomaly score used for thresholding."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                recon, kl = self.loss_per_sample(x)
                return recon + self.kl_weight * kl
        finally:
            if was_training:
                self.train()


def vae_threshold(scores, percentile=99.0):
    """Calibrate the normal-traffic anomaly boundary used by the VAE reference."""
    if not 0 < percentile < 100:
        raise ValueError('percentile 必須介於 0 與 100 之間')
    return float(np.percentile(np.asarray(scores, dtype=np.float64), percentile))


class VAETrainer:
    """Small training adapter matching the project trainer's public workflow."""
    def __init__(self, config=None, output_dir='output/model_vae', model_name=''):
        defaults = {'latent_dim': 32, 'batch_size': 32, 'epochs': 100,
                    'learning_rate': 1e-3, 'patience': 15, 'threshold_percentile': 99,
                    'kl_weight': 1e-3, 'val_split': .2}
        self.config = {**defaults, **(config or {})}; self.output_dir = output_dir; self.model_name = model_name
        os.makedirs(output_dir, exist_ok=True); self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNNVariationalAutoencoder(self.config['latent_dim'], kl_weight=self.config['kl_weight']).to(self.device)
        self.threshold = None; self.train_losses = []; self.val_losses = []; self.data = None

    def load_data(self, npy_path):
        self.data = np.load(npy_path).astype(np.float32)
        tensor = torch.from_numpy(self.data[:, None]); ds = TensorDataset(tensor)
        n_val = max(1, int(len(ds) * self.config['val_split'])); n_train = len(ds) - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        self.train_loader = DataLoader(train_ds, batch_size=self.config['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.config['batch_size'])

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate']); best = float('inf'); wait = 0
        for _ in range(self.config['epochs']):
            self.model.train(); losses = []
            for (x,) in self.train_loader:
                x = x.to(self.device); optimizer.zero_grad(); loss = self.model.vae_loss(x); loss.backward(); optimizer.step(); losses.append(loss.item())
            self.model.eval()
            with torch.no_grad(): val = [self.model.vae_loss(x.to(self.device)).item() for (x,) in self.val_loader]
            self.train_losses.append(float(np.mean(losses))); self.val_losses.append(float(np.mean(val)))
            if self.val_losses[-1] < best:
                best = self.val_losses[-1]; wait = 0; self._save_checkpoint()
            else:
                wait += 1
                if wait >= self.config['patience']: break
        self._load_checkpoint()

    def compute_threshold(self, npy_path=None, percentile=None):
        X = self.data if npy_path is None else np.load(npy_path).astype(np.float32); scores = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(X), 256): scores.extend(self.model.reconstruction_error(torch.from_numpy(X[start:start+256, None]).to(self.device)).cpu().numpy())
        self.threshold = vae_threshold(scores, percentile or self.config['threshold_percentile']); self._save_checkpoint(); return self.threshold

    def plot_training_curve(self): pass
    def plot_reconstruction_samples(self, _): pass
    def _path(self): return os.path.join(self.output_dir, f'best_vae_{self.model_name or "model"}.pt')
    def _save_checkpoint(self): torch.save({'model_type': 'cnn_vae', 'model_state': self.model.state_dict(), 'config': self.config, 'threshold': self.threshold}, self._path())
    def _load_checkpoint(self): self.model.load_state_dict(torch.load(self._path(), map_location=self.device, weights_only=True)['model_state'])
