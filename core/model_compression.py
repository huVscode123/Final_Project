# ============================================================
# model_compression.py - 模型壓縮與評估模組（優化版）
# ============================================================

import os
import sys
import copy
import time
import json
import numpy as np
from typing import Optional, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import matplotlib.gridspec as gridspec

from sklearn.metrics import precision_recall_curve, auc as sk_auc

# 自動偵測中文字型
_CJK = ["Microsoft YaHei", "SimHei", "PingFang TC",
        "Heiti TC", "WenQuanYi Zen Hei", "Noto Sans CJK TC"]
_avail = {f.name for f in _fm.fontManager.ttflist}
for _f in _CJK:
    if _f in _avail:
        matplotlib.rcParams["font.family"] = [_f, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from cnn_autoencoder import CNNAutoencoder
from ablation_study import CNNAutoencoderFlex


# ── 工具函式 ─────────────────────────────────────────────────────

def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _count_nonzero_params(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += int((p != 0).sum().item())
    return total

def _model_size_mb(model: nn.Module) -> float:
    n = sum(p.numel() for p in model.parameters())
    return n * 4 / (1024 ** 2)

def _compute_errors_np(model: nn.Module, X: np.ndarray, device, batch_size: int = 128) -> np.ndarray:
    if X.ndim == 3: X = X[:, np.newaxis, :, :]
    tensor  = torch.from_numpy(X.astype(np.float32))
    loader  = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    model.eval()
    errs = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            try: x_hat, _ = model(batch)
            except Exception:
                out = model(batch)
                x_hat = out[0] if isinstance(out, (tuple, list)) else out
            e = torch.mean((batch - x_hat) ** 2, dim=[1, 2, 3])
            errs.append(e.cpu().numpy())
    return np.concatenate(errs)

def _evaluate_threshold(errors_normal: np.ndarray, errors_attack: np.ndarray, pct: float = 95.0) -> dict:
    threshold = float(np.percentile(errors_normal, pct))
    fp = int((errors_normal > threshold).sum()); tn = len(errors_normal) - fp
    tp = int((errors_attack > threshold).sum()); fn = len(errors_attack) - tp
    precision = tp / (tp + fp + 1e-9); recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / (len(errors_normal) + len(errors_attack))
    fpr = fp / (fp + tn + 1e-9)
    all_errors = np.concatenate([errors_normal, errors_attack])
    all_labels = np.concatenate([np.zeros(len(errors_normal)), np.ones(len(errors_attack))])
    # AUC-ROC
    sorted_idx = np.argsort(-all_errors); tprs, fprs = [0.0], [0.0]; tp_c = fp_c = 0
    P = int(all_labels.sum()); N = len(all_labels) - P
    for i in sorted_idx:
        if all_labels[i] == 1: tp_c += 1
        else: fp_c += 1
        tprs.append(tp_c / (P + 1e-9)); fprs.append(fp_c / (N + 1e-9))
    tprs.append(1.0); fprs.append(1.0)
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auc = float(_trapz(tprs, fprs))
    # PR-AUC
    pr_pre, pr_rec, _ = precision_recall_curve(all_labels, all_errors)
    pr_auc = float(sk_auc(pr_rec, pr_pre))
    return {"threshold": threshold, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "accuracy": accuracy, "auc": auc, "pr_auc": pr_auc}

def _measure_latency(model: nn.Module, device, input_shape: tuple = (1, 1, 32, 32), n_runs: int = 200) -> float:
    model.eval(); dummy = torch.randn(*input_shape).to(device)
    for _ in range(10): 
        with torch.no_grad():
            try: model(dummy)
            except Exception: pass
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            try: model(dummy)
            except Exception: pass
        if device.type == "cuda": torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))

def _quick_finetune(model: nn.Module, X_train: np.ndarray, epochs: int = 10, lr: float = 5e-4, device=None):
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_arr = X_train[:, np.newaxis, :, :] if X_train.ndim == 3 else X_train
    loader = DataLoader(TensorDataset(torch.from_numpy(X_arr.astype(np.float32))), batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.MSELoss(); model.train()
    for ep in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device); xh, _ = model(batch); loss = criterion(xh, batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()

# ── 壓縮技術 1-3 ──────────────────────────────────────────────────
class UnstructuredPruner:
    def __init__(self, model): self.original_model = model
    def prune(self, sparsity, finetune_epochs=10, X_train=None, device=None):
        model = copy.deepcopy(self.original_model); modules = []
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)): modules.append((m, "weight"))
        prune.global_unstructured(modules, pruning_method=prune.L1Unstructured, amount=sparsity)
        for m, name in modules: prune.remove(m, name)
        if finetune_epochs > 0 and X_train is not None: _quick_finetune(model, X_train, finetune_epochs, device=device)
        return model

class StructuredPruner:
    def __init__(self, model): self.original_model = model
    def prune(self, ratio, finetune_epochs=10, X_train=None, device=None):
        model = copy.deepcopy(self.original_model); modules = []
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.weight.shape[0]; amt = min(max(1, int(n * ratio)), n - 1)
                if amt > 0: prune.ln_structured(m, name="weight", amount=amt, n=2, dim=0); modules.append((m, "weight"))
        for m, name in modules: prune.remove(m, name)
        if finetune_epochs > 0 and X_train is not None: _quick_finetune(model, X_train, finetune_epochs, device=device)
        return model

class PostTrainingQuantizer:
    def __init__(self, model): self.original_model = model
    def quantize(self, X_calib, backend="fbgemm"):
        try: import torch.quantization as tq
        except ImportError: return None
        model_cpu = copy.deepcopy(self.original_model).cpu().eval()
        class QuantWrapper(nn.Module):
            def __init__(self, inner): super().__init__(); self.quant = tq.QuantStub(); self.model = inner; self.dequant = tq.DeQuantStub()
            def forward(self, x): return self.dequant(self.model(self.quant(x))[0]), None
        wrapped = QuantWrapper(model_cpu); torch.backends.quantized.engine = backend; wrapped.qconfig = tq.get_default_qconfig(backend)
        try: tq.prepare(wrapped, inplace=True)
        except Exception: return self._fallback(model_cpu)
        X_c = X_calib[:, np.newaxis, :, :] if X_calib.ndim == 3 else X_calib
        loader = DataLoader(TensorDataset(torch.from_numpy(X_c[:256].astype(np.float32))), batch_size=64)
        with torch.no_grad():
            for (b,) in loader: wrapped(b)
        try: tq.convert(wrapped, inplace=True); return wrapped
        except Exception: return self._fallback(model_cpu)
    def _fallback(self, model_cpu):
        try: return torch.quantization.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)
        except Exception: return None

# ── 壓縮技術 4：知識蒸餾 ──────────────────────────────────────────
class KnowledgeDistiller:
    def __init__(self, teacher_model, student_latent_dim=16, student_channels=None):
        self.teacher = teacher_model; self.student_latent_dim = student_latent_dim
        self.student_channels = student_channels or [8, 16, 32, 32] # 預設輕量化
        t_latent = getattr(teacher_model, "latent_dim", 32)
        self.projector = nn.Linear(student_latent_dim, t_latent) if t_latent != student_latent_dim else nn.Identity()

    def distill(self, X_train, epochs=50, alpha=0.5, temperature=1.0, lr=1e-3, batch_size=32, device=None):
        if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        student = CNNAutoencoderFlex(latent_dim=self.student_latent_dim, channels=self.student_channels).to(device)
        teacher = copy.deepcopy(self.teacher).to(device).eval(); projector = self.projector.to(device)
        X_arr = X_train[:, np.newaxis, :, :] if X_train.ndim == 3 else X_train
        loader = DataLoader(TensorDataset(torch.from_numpy(X_arr.astype(np.float32))), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(list(student.parameters()) + list(projector.parameters()), lr=lr)
        mse = nn.MSELoss(); student.train(); projector.train()
        for ep in range(epochs):
            for (batch,) in loader:
                batch = batch.to(device)
                with torch.no_grad(): _, z_t = teacher(batch); z_t = z_t / (temperature + 1e-8)
                xh_s, z_s = student(batch); z_s_proj = projector(z_s / (temperature + 1e-8))
                loss = alpha * mse(xh_s, batch) + (1 - alpha) * mse(z_s_proj, z_t)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        student.eval(); return student

# ── 主要評估框架 ──────────────────────────────────────────────────
class ModelCompressor:
    def __init__(self, model, X_normal, X_attack, output_dir="output/compression", percentile=95, device=None):
        self.model = model; self.X_normal = X_normal.astype(np.float32); self.X_attack = X_attack.astype(np.float32)
        self.output_dir = output_dir; self.percentile = percentile; self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_dir, exist_ok=True); self.baseline = self._evaluate_model(model, "Baseline")

    def _evaluate_model(self, model, name):
        m_ev = model.to(self.device).eval()
        en = _compute_errors_np(m_ev, self.X_normal, self.device); ea = _compute_errors_np(m_ev, self.X_attack, self.device)
        mt = _evaluate_threshold(en, ea, self.percentile); np_ = _count_params(m_ev); nnz = _count_nonzero_params(m_ev)
        return {"name": name, "params": np_, "nonzero_params": nnz, "sparsity": round(1 - nnz/(np_+1e-9), 4), "size_mb": round(_model_size_mb(m_ev), 3), "latency_ms": round(_measure_latency(m_ev, self.device), 3), **mt}

    def run_ptq(self, backend="fbgemm"):
        print("\n  ── 訓練後量化 (PTQ) ──")
        if self.device.type != "cpu": print("  [PTQ] 警告：量化僅對 CPU 推論有加速效果")
        res = PostTrainingQuantizer(self.model).quantize(self.X_normal, backend)
        if res is None: return None
        result = self._evaluate_model(res.cpu(), "PTQ INT8"); result["compression_ratio"] = round(self.baseline["size_mb"] / result["size_mb"], 2)
        return result

    def run_knowledge_distillation(self, dims=None, epochs=50):
        dims = dims or [8, 16, 32]; results = []
        for ld in dims:
            s_ch = [8, 16, 32, 32] if ld <= 16 else [12, 24, 48, 48]
            dist = KnowledgeDistiller(self.model, ld, s_ch).distill(self.X_normal, epochs=epochs, device=self.device)
            res = self._evaluate_model(dist, f"KD Student ld={ld}"); results.append(res)
            print(f"    Student ld={ld}: F1={res['f1']:.4f} PR-AUC={res['pr_auc']:.4f}")
        return results

    def run_all(self):
        # Placeholder for full run logic...
        return {"baseline": self.baseline}
