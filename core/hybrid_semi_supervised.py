"""CNN-LSTM known-attack classification plus reconstruction-based unknown detection."""
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from variational_autoencoder import CNNVariationalAutoencoder, vae_threshold


class CNNLSTMKnownAttackClassifier(nn.Module):
    """PyTorch equivalent of CyberNeural-IDS' Conv1D → LSTM → softmax path.

    Packet images are treated as a 32-step sequence after CNN feature
    extraction, so the model remains compatible with packet-image Grad-CAM.
    """
    model_type = 'cnn_lstm_known_attack'

    def __init__(self, num_classes=2, hidden_size=64):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=64 * 16, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_classes))

    def forward(self, x):
        features = self.encoder_conv(x)                 # N, 64, 16, 16
        sequence = features.permute(0, 2, 1, 3).flatten(2)  # N, 16, 1024
        _, (hidden, _) = self.lstm(sequence)
        return self.classifier(hidden[-1])


class HybridSemiSupervisedDetector(nn.Module):
    """Loadable website-facing hybrid detector checkpoint."""
    model_type = 'hybrid_semi'

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.config = config
        self.vae = CNNVariationalAutoencoder(
            latent_dim=config.get('latent_dim', 32),
            kl_weight=config.get('kl_weight', 1e-3),
        )
        self.classifier = CNNLSTMKnownAttackClassifier()
        self.threshold = float(config.get('threshold', 0.0))

    def forward(self, x):
        # Keeps Grad-CAM-compatible reconstruction interface.
        return self.vae(x)

    def reconstruction_error(self, x):
        return self.vae.reconstruction_error(x)

    def classify_known(self, x):
        logits = self.classifier(x)
        return torch.softmax(logits, dim=1)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        detector = cls(checkpoint.get('config', {}))
        detector.vae.load_state_dict(checkpoint['vae_state'])
        detector.classifier.load_state_dict(checkpoint['classifier_state'])
        detector.threshold = float(checkpoint['threshold'])
        return detector


def hybrid_decision(class_logits, reconstruction_scores, unknown_threshold):
    """Apply the reference project's priority: unknown anomaly before class label."""
    probabilities = torch.softmax(class_logits, dim=1)
    confidence, labels = probabilities.max(dim=1)
    unknown = reconstruction_scores > unknown_threshold
    return labels, confidence, unknown


class HybridSemiSupervisedTrainer:
    """CyberNeural-style PyTorch trainer: VAE unknown gate + CNN-LSTM classifier."""
    def __init__(self, config=None, output_dir='output/model_semi', model_name=''):
        cfg = {'latent_dim': 32, 'batch_size': 32, 'pretrain_epochs': 50,
               'finetune_epochs': 50, 'learning_rate': 1e-3, 'attack_ratio': .2,
               'unknown_percentile': 95}
        self.config = {**cfg, **(config or {})}; self.output_dir = output_dir; self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNNVariationalAutoencoder(self.config['latent_dim']).to(self.device)
        self.classifier = CNNLSTMKnownAttackClassifier().to(self.device); self.threshold = None

    def _loader(self, X, y=None, shuffle=True):
        tensors = [torch.from_numpy(X.astype(np.float32)[:, None])]
        if y is not None: tensors.append(torch.from_numpy(y.astype(np.int64)))
        return DataLoader(TensorDataset(*tensors), batch_size=self.config['batch_size'], shuffle=shuffle)

    def train_full(self, X_normal, X_attack, threshold_method='optimal'):
        """完整訓練流程：VAE 預訓練（正常基線）→ CNN-LSTM 分類器微調
        （少量標記攻擊）→ 計算未知異常閾值 → 儲存 checkpoint。

        threshold_method:
          'optimal'    - 在「未參與分類器微調」的攻擊樣本（holdout）與
                         正常樣本上做 F1 最佳化閾值搜尋（與舊版
                         SemiSupervisedTrainer / run_training.py 一致）。
          其他任意值    - 退回 config['unknown_percentile']（預設 95）
                         的正常流量重建誤差百分位數。

        [Bug 修正] 舊版此參數雖然存在於函式簽章，但函式本體完全沒有
        使用它，一律採用 percentile 方式，導致
        run_semi_supervised.py 的 --threshold-method optimal
        形同虛設（傳入什麼值結果都相同）。
        """
        # VAE learns only the normal baseline (unknown-attack gate).
        opt = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        for _ in range(self.config['pretrain_epochs']):
            self.model.train()
            for (x,) in self._loader(X_normal):
                x=x.to(self.device); opt.zero_grad(); loss=self.model.vae_loss(x); loss.backward(); opt.step()

        # Only a fraction of labelled attacks are used for classifier fine-tuning;
        # 保留「未被抽中」的攻擊樣本（holdout），供下方 'optimal' 閾值搜尋使用，
        # 避免用同一批已被分類器看過的樣本做搜尋，造成閾值對訓練資料過擬合。
        rng = np.random.default_rng(42)
        n = max(1, int(len(X_attack) * self.config['attack_ratio']))
        attack_idx = rng.choice(len(X_attack), n, replace=False)
        attack_mask = np.zeros(len(X_attack), dtype=bool)
        attack_mask[attack_idx] = True
        attack = X_attack[attack_idx]
        attack_holdout = X_attack[~attack_mask]

        normal = X_normal[rng.choice(len(X_normal), n, replace=False)]
        X=np.concatenate([normal, attack]); y=np.concatenate([np.zeros(n), np.ones(n)])
        opt=torch.optim.Adam(self.classifier.parameters(), lr=self.config['learning_rate']); criterion=nn.CrossEntropyLoss()
        for _ in range(self.config['finetune_epochs']):
            self.classifier.train()
            for x, labels in self._loader(X, y):
                x=x.to(self.device); labels=labels.to(self.device); opt.zero_grad(); loss=criterion(self.classifier(x), labels); loss.backward(); opt.step()

        normal_scores = self._scores(X_normal)
        if threshold_method == 'optimal' and len(attack_holdout) >= 2:
            from training_common import compute_optimal_threshold_vectorized
            attack_scores = self._scores(attack_holdout)
            self.threshold, best_f1, best_pct = compute_optimal_threshold_vectorized(
                normal_scores, attack_scores
            )
            print(f'  [Hybrid] 最佳閾值搜尋（holdout 攻擊樣本={len(attack_holdout)}，'
                  f'F1={best_f1:.4f}，{best_pct}th 百分位）: {self.threshold:.6f}')
        else:
            self.threshold = vae_threshold(normal_scores, self.config['unknown_percentile'])
            print(f'  [Hybrid] Percentile 閾值（{self.config["unknown_percentile"]}th）: '
                  f'{self.threshold:.6f}')

        os.makedirs(self.output_dir, exist_ok=True)
        torch.save({
            'model_type': 'hybrid_semi', 'vae_state': self.model.state_dict(),
            'classifier_state': self.classifier.state_dict(), 'config': self.config,
            'threshold': self.threshold,
        }, os.path.join(self.output_dir, f'semi_{self.model_name}.pt'))
        return self.threshold

    def _scores(self, X):
        self.model.eval(); out=[]
        with torch.no_grad():
            for (x,) in self._loader(X, shuffle=False): out.extend(self.model.reconstruction_error(x.to(self.device)).cpu().numpy())
        return np.asarray(out)

    def predict(self, X):
        scores=self._scores(X); self.classifier.eval()
        with torch.no_grad(): logits=torch.cat([self.classifier(x.to(self.device)).cpu() for (x,) in self._loader(X, shuffle=False)])
        labels, confidence, unknown=hybrid_decision(logits, torch.from_numpy(scores), self.threshold)
        return {'known_label': labels.numpy(), 'confidence': confidence.numpy(), 'unknown': unknown.numpy(), 'score': scores}
