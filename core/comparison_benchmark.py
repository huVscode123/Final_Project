# ============================================================
# comparison_benchmark.py - 半監督 vs 非監督 CNN Autoencoder 完整基準比較
#
# 目的：
#   系統性比較兩種訓練策略在相同資料集、相同架構下的效能差異：
#
#   ┌──────────────────────────────────────────────────────────┐
#   │  Strategy A：純非監督（Unsupervised Baseline）            │
#   │    - 只使用正常流量，最小化重建誤差（MSE）                  │
#   │    - 依賴正常流量的百分位數設定閾值                         │
#   │                                                          │
#   │  Strategy B：半監督（Semi-supervised Fine-tuning）        │
#   │    - Phase 1：純非監督預訓練（同 Strategy A）              │
#   │    - Phase 2：使用少量標記攻擊樣本，邊界損失微調            │
#   │      L = α × MSE(normal) + β × max(0, margin - MSE(atk)) │
#   └──────────────────────────────────────────────────────────┘
#
# 評估指標：
#   - Precision / Recall / F1 Score / AUC-ROC
#   - 誤差分離比（攻擊誤差均值 / 正常誤差均值）
#   - 推論延遲（ms/batch）
#
# 輸出：
#   output/comparison/
#     comparison_metrics_bar.png      四項指標長條對比圖
#     comparison_error_dist.png       誤差分布對比圖（正常 vs 攻擊）
#     comparison_roc_curve.png        ROC 曲線對比圖
#     comparison_radar.png            雷達圖（六維指標）
#     comparison_report.json          完整 JSON 報告
#
# 使用方式：
#   from comparison_benchmark import ComparisonBenchmark
#   bench = ComparisonBenchmark(X_normal, X_attack, output_dir="output/comparison")
#   results = bench.run()
#   bench.plot_all(results)
#   bench.save_report(results)
#
# 命令列：
#   python core/comparison_benchmark.py --dataset simulate
#   python core/comparison_benchmark.py --dataset cicids2017 --data-dir data/cicids2017
#   python core/comparison_benchmark.py \
#       --normal output/dataset_simulate/X_normal.npy \
#       --attack output/dataset_simulate/X_attack.npy
# ============================================================

import os
import sys
import time
import json
import copy
import numpy as np
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import matplotlib.gridspec as gridspec

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
from torch.utils.data import DataLoader, TensorDataset

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from cnn_autoencoder import CNNAutoencoder


# ══════════════════════════════════════════════════════════════════
# 共用工具函式
# ══════════════════════════════════════════════════════════════════

def _compute_errors(model: nn.Module, X: np.ndarray,
                    device, batch_size: int = 128) -> np.ndarray:
    """計算每個樣本的重建誤差（MSE）。"""
    if X.ndim == 3:
        X = X[:, np.newaxis, :, :]
    tensor  = torch.from_numpy(X.astype(np.float32))
    loader  = DataLoader(TensorDataset(tensor),
                         batch_size=batch_size, shuffle=False)
    model.eval()
    errs = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            e = torch.mean((batch - model(batch)[0]) ** 2, dim=[1, 2, 3])
            errs.append(e.cpu().numpy())
    return np.concatenate(errs)


def _full_metrics(errors_normal: np.ndarray,
                  errors_attack: np.ndarray,
                  percentile: int = 95) -> dict:
    """
    給定重建誤差陣列，計算完整評估指標集。

    Returns:
        dict：
          threshold, tp, tn, fp, fn,
          precision, recall, f1, fpr, accuracy, auc,
          separability_ratio,
          avg_normal_error, avg_attack_error
    """
    threshold = float(np.percentile(errors_normal, percentile))
    fp = int((errors_normal > threshold).sum())
    tn = len(errors_normal) - fp
    tp = int((errors_attack > threshold).sum())
    fn = len(errors_attack) - tp

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    fpr       = fp / (fp + tn + 1e-9)
    accuracy  = (tp + tn) / (len(errors_normal) + len(errors_attack))

    # AUC-ROC（梯形積分，不依賴 sklearn）
    all_e = np.concatenate([errors_normal, errors_attack])
    all_y = np.concatenate([np.zeros(len(errors_normal)),
                             np.ones(len(errors_attack))])
    idx   = np.argsort(-all_e)
    tprs, fprs_roc = [0.0], [0.0]
    P = int(all_y.sum()); N = len(all_y) - P
    tpc = fpc = 0
    for i in idx:
        if all_y[i] == 1: tpc += 1
        else:              fpc += 1
        tprs.append(tpc / (P + 1e-9))
        fprs_roc.append(fpc / (N + 1e-9))
    tprs.append(1.0); fprs_roc.append(1.0)
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auc = float(_trapz(tprs, fprs_roc))

    sep = float(errors_attack.mean() / (errors_normal.mean() + 1e-9))

    return {
        "threshold":          round(threshold, 6),
        "percentile":         percentile,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision":          round(precision, 6),
        "recall":             round(recall, 6),
        "f1":                 round(f1, 6),
        "fpr":                round(fpr, 6),
        "accuracy":           round(accuracy, 6),
        "auc":                round(auc, 6),
        "separability_ratio": round(sep, 4),
        "avg_normal_error":   round(float(errors_normal.mean()), 6),
        "avg_attack_error":   round(float(errors_attack.mean()), 6),
        # 儲存原始誤差陣列（用於繪圖，不寫入 JSON）
        "_errors_normal":     errors_normal,
        "_errors_attack":     errors_attack,
        "_fpr_curve":         np.array(fprs_roc),
        "_tpr_curve":         np.array(tprs),
    }


def _quick_train_unsup(model: nn.Module, X_normal: np.ndarray,
                       epochs: int, batch_size: int = 32,
                       lr: float = 1e-3, device=None) -> list:
    """純非監督訓練（只用正常流量，MSE 損失）。"""
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    if X_normal.ndim == 3:
        X_arr = X_normal[:, np.newaxis, :, :]
    else:
        X_arr = X_normal
    tensor   = torch.from_numpy(X_arr.astype(np.float32))
    loader   = DataLoader(TensorDataset(tensor), batch_size=batch_size,
                          shuffle=True, drop_last=False)
    opt      = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit     = nn.MSELoss()
    model.train()
    losses = []
    for _ in range(epochs):
        ep = 0.0
        for (b,) in loader:
            b = b.to(device)
            xh, _ = model(b)
            loss = crit(xh, b)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep += loss.item() * len(b)
        sch.step()
        losses.append(ep / len(tensor))
    model.eval()
    return losses


def _quick_train_semi(model: nn.Module,
                      X_normal: np.ndarray,
                      X_attack: np.ndarray,
                      pretrain_epochs: int,
                      finetune_epochs: int,
                      attack_ratio: float = 0.2,
                      alpha: float = 1.0,
                      beta: float = 0.5,
                      margin: float = 0.05,
                      batch_size: int = 32,
                      lr_pretrain: float = 1e-3,
                      lr_finetune: float = 5e-4,
                      device=None) -> dict:
    """
    兩階段半監督訓練：
      Phase 1：純非監督預訓練（MSE）
      Phase 2：邊界損失微調（alpha × MSE_normal + beta × MarginLoss_attack）

    Returns:
        dict: {'pretrain_losses': [...], 'finetune_normal': [...],
               'finetune_attack': [...], 'finetune_total': [...]}
    """
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)

    def _to_loader(X, bsz, shuffle=True):
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        t = torch.from_numpy(X.astype(np.float32))
        return DataLoader(TensorDataset(t), batch_size=bsz,
                          shuffle=shuffle, drop_last=False)

    crit = nn.MSELoss()

    # ── Phase 1：純非監督預訓練 ──────────────────────────────────
    normal_loader = _to_loader(X_normal, batch_size)
    opt1 = torch.optim.Adam(model.parameters(), lr=lr_pretrain,
                             weight_decay=1e-5)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=pretrain_epochs)
    model.train()
    pretrain_losses = []
    for _ in range(pretrain_epochs):
        ep = 0.0
        for (b,) in normal_loader:
            b = b.to(device)
            xh, _ = model(b)
            loss = crit(xh, b)
            opt1.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()
            ep += loss.item() * len(b)
        sch1.step()
        pretrain_losses.append(ep / len(X_normal))

    # ── Phase 2：邊界損失微調 ────────────────────────────────────
    n_atk = max(8, int(len(X_attack) * attack_ratio))
    idx   = np.random.choice(len(X_attack), n_atk, replace=False)
    attack_loader = _to_loader(X_attack[idx], min(batch_size, n_atk))

    opt2 = torch.optim.Adam(model.parameters(), lr=lr_finetune,
                             weight_decay=1e-5)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=finetune_epochs)

    fine_normal_losses = []
    fine_attack_losses = []
    fine_total_losses  = []

    model.train()
    for _ in range(finetune_epochs):
        ep_n = ep_a = ep_t = 0.0
        n_cnt = 0
        atk_iter = iter(attack_loader)

        for (x_n,) in normal_loader:
            x_n = x_n.to(device)
            try:
                (x_a,) = next(atk_iter)
            except StopIteration:
                atk_iter = iter(attack_loader)
                (x_a,) = next(atk_iter)
            x_a = x_a.to(device)

            # 正常重建
            xh_n, _ = model(x_n)
            L_n = crit(xh_n, x_n)

            # 攻擊邊界損失
            xh_a, _ = model(x_a)
            err_a   = torch.mean((x_a - xh_a) ** 2, dim=[1, 2, 3])
            L_a     = torch.clamp(margin - err_a, min=0.0).mean()

            loss = alpha * L_n + beta * L_a
            opt2.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()

            ep_n += L_n.item() * len(x_n)
            ep_a += L_a.item() * len(x_n)
            ep_t += loss.item() * len(x_n)
            n_cnt += len(x_n)

        sch2.step()
        n_cnt = max(n_cnt, 1)
        fine_normal_losses.append(ep_n / n_cnt)
        fine_attack_losses.append(ep_a / n_cnt)
        fine_total_losses.append(ep_t / n_cnt)

    model.eval()
    return {
        "pretrain_losses":  pretrain_losses,
        "finetune_normal":  fine_normal_losses,
        "finetune_attack":  fine_attack_losses,
        "finetune_total":   fine_total_losses,
    }


# ══════════════════════════════════════════════════════════════════
# 主要比較框架
# ══════════════════════════════════════════════════════════════════

class ComparisonBenchmark:
    """
    半監督 vs 非監督 CNN Autoencoder 完整比較框架。

    兩種策略使用完全相同的：
      - 模型架構（CNNAutoencoder）
      - 訓練資料量（同一批 X_normal / X_attack）
      - 評估資料集（相同 test split）
      - 超參數（學習率、batch size 等）

    唯一差異：訓練策略（有無 Phase 2 Margin Loss 微調）

    Args:
        X_normal        : 正常流量影像 shape=(N, H, W)
        X_attack        : 攻擊流量影像 shape=(M, H, W)
        output_dir      : 輸出目錄
        latent_dim      : CNN Autoencoder 潛在空間維度
        pretrain_epochs : Phase 1 訓練輪數（兩種策略相同）
        finetune_epochs : Phase 2 微調輪數（僅半監督使用）
        attack_ratio    : 用於微調的標記攻擊樣本比例
        percentile      : 閾值百分位數（預設 95）
        test_ratio      : 測試集比例（預設 0.2）
        batch_size      : 批次大小
        device          : torch.device（None 自動選擇）
        seed            : 隨機種子
    """

    _DARK_BG = "#0d1117"
    _DARK_AX = "#161b22"

    def __init__(self,
                 X_normal: np.ndarray,
                 X_attack: np.ndarray,
                 output_dir: str = "output/comparison",
                 latent_dim: int = 32,
                 pretrain_epochs: int = 50,
                 finetune_epochs: int = 50,
                 attack_ratio: float = 0.20,
                 percentile: int = 95,
                 test_ratio: float = 0.20,
                 batch_size: int = 32,
                 device=None,
                 seed: int = 42):

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.latent_dim      = latent_dim
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.attack_ratio    = attack_ratio
        self.percentile      = percentile
        self.batch_size      = batch_size
        self.output_dir      = output_dir
        self.device          = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(output_dir, exist_ok=True)

        # ── 分割訓練集 / 測試集 ──────────────────────────────────
        # 正常流量：80% 訓練，20% 測試
        n_n       = len(X_normal)
        n_n_test  = max(20, int(n_n * test_ratio))
        idx_n     = np.random.permutation(n_n)
        self.X_normal_train = X_normal[idx_n[n_n_test:]].astype(np.float32)
        self.X_normal_test  = X_normal[idx_n[:n_n_test]].astype(np.float32)

        # 攻擊流量：全部用於測試（微調只用 attack_ratio 比例的子集，在訓練內部抽取）
        self.X_attack_train = X_attack.astype(np.float32)   # Phase 2 微調抽樣用
        self.X_attack_test  = X_attack.astype(np.float32)   # 評估時全部使用

        print(f"  [ComparisonBenchmark] 資料切分：")
        print(f"    正常流量訓練: {len(self.X_normal_train):,}  "
              f"測試: {len(self.X_normal_test):,}")
        print(f"    攻擊流量評估: {len(self.X_attack_test):,}  "
              f"（微調使用其中 {attack_ratio:.0%} = "
              f"{max(8, int(len(self.X_attack_train)*attack_ratio)):,} 筆）")
        print(f"  [ComparisonBenchmark] 裝置: {self.device}  "
              f"latent_dim={latent_dim}  percentile={percentile}th")

    # ── Strategy A：純非監督 ─────────────────────────────────────
    def run_unsupervised(self) -> dict:
        """
        訓練純非監督基準模型並評估。

        訓練策略：只使用正常流量（X_normal_train），最小化 MSE 重建誤差。
        閾值：正常流量測試集的第 percentile 百分位數。

        Returns:
            dict 包含所有評估指標與原始誤差陣列
        """
        print(f"\n  ── Strategy A：純非監督（Unsupervised Baseline）──")
        total_ep = self.pretrain_epochs + self.finetune_epochs

        model = CNNAutoencoder(latent_dim=self.latent_dim)

        t0 = time.time()
        losses = _quick_train_unsup(
            model, self.X_normal_train,
            epochs=total_ep,
            batch_size=self.batch_size,
            device=self.device,
        )
        train_time = time.time() - t0

        errors_n = _compute_errors(model, self.X_normal_test, self.device)
        errors_a = _compute_errors(model, self.X_attack_test, self.device)
        metrics  = _full_metrics(errors_n, errors_a, self.percentile)

        print(f"    訓練完成（{total_ep} epochs / {train_time:.1f}s）")
        print(f"    分離比={metrics['separability_ratio']:.3f}x  "
              f"F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}")
        print(f"    Precision={metrics['precision']:.4f}  "
              f"Recall={metrics['recall']:.4f}")

        return {
            "strategy":    "Unsupervised",
            "label":       "非監督（Unsupervised）",
            "train_time":  round(train_time, 2),
            "total_epochs": total_ep,
            "train_losses": losses,
            "model":        model,
            **metrics,
        }

    # ── Strategy B：半監督 ───────────────────────────────────────
    def run_semi_supervised(self) -> dict:
        """
        訓練半監督模型（Phase 1 + Phase 2 Margin Loss）並評估。

        優先使用正式 SemiSupervisedTrainer（若已安裝）；
        否則 fallback 至內建的 _quick_train_semi。

        Returns:
            dict 包含所有評估指標與原始誤差陣列
        """
        print(f"\n  ── Strategy B：半監督（Semi-supervised）──")
        print(f"    Phase 1（pretrain）: {self.pretrain_epochs} epochs")
        print(f"    Phase 2（finetune）: {self.finetune_epochs} epochs  "
              f"attack_ratio={self.attack_ratio:.0%}")

        t0 = time.time()

        # 嘗試使用正式 SemiSupervisedTrainer
        try:
            from semi_supervised_trainer import SemiSupervisedTrainer
            semi_config = {
                "latent_dim":      self.latent_dim,
                "batch_size":      self.batch_size,
                "pretrain_epochs": self.pretrain_epochs,
                "finetune_epochs": self.finetune_epochs,
                "attack_ratio":    self.attack_ratio,
                "alpha":           1.0,
                "beta":            0.5,
                "margin":          0.05,
            }
            semi_trainer = SemiSupervisedTrainer(
                config=semi_config,
                output_dir=os.path.join(self.output_dir, "_semi_ckpt"),
            )
            semi_trainer.train_full(
                self.X_normal_train, self.X_attack_train,
                threshold_method="percentile",
            )
            model = semi_trainer.model
            history = {
                "pretrain_losses": semi_trainer.pretrain_losses,
                "finetune_normal": semi_trainer.finetune_normal,
                "finetune_attack": semi_trainer.finetune_attack,
                "finetune_total":  semi_trainer.finetune_total,
            }
            _from_full_trainer = True
            print("    [使用 SemiSupervisedTrainer]")
        except Exception as e:
            print(f"    [fallback] SemiSupervisedTrainer: {e}")
            model   = CNNAutoencoder(latent_dim=self.latent_dim)
            history = _quick_train_semi(
                model,
                self.X_normal_train,
                self.X_attack_train,
                pretrain_epochs=self.pretrain_epochs,
                finetune_epochs=self.finetune_epochs,
                attack_ratio=self.attack_ratio,
                batch_size=self.batch_size,
                device=self.device,
            )
            _from_full_trainer = False

        train_time = time.time() - t0

        errors_n = _compute_errors(model, self.X_normal_test, self.device)
        errors_a = _compute_errors(model, self.X_attack_test, self.device)
        metrics  = _full_metrics(errors_n, errors_a, self.percentile)

        print(f"    訓練完成（{train_time:.1f}s）")
        print(f"    分離比={metrics['separability_ratio']:.3f}x  "
              f"F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}")
        print(f"    Precision={metrics['precision']:.4f}  "
              f"Recall={metrics['recall']:.4f}")

        return {
            "strategy":          "SemiSupervised",
            "label":             "半監督（Semi-supervised）",
            "train_time":        round(train_time, 2),
            "pretrain_epochs":   self.pretrain_epochs,
            "finetune_epochs":   self.finetune_epochs,
            "from_full_trainer": _from_full_trainer,
            "model":             model,
            **history,
            **metrics,
        }

    # ── 執行完整比較 ──────────────────────────────────────────────
    def run(self) -> dict:
        """
        依序執行兩種策略的訓練與評估。

        Returns:
            dict：
            {
                "unsupervised":  {...},  # Strategy A 結果
                "semi":          {...},  # Strategy B 結果
                "delta":         {...},  # 兩者差值（semi - unsup）
            }
        """
        print("\n" + "=" * 65)
        print("  [ComparisonBenchmark] 開始基準比較實驗")
        print("=" * 65)

        res_u = self.run_unsupervised()
        res_s = self.run_semi_supervised()

        # 計算差值
        metric_keys = ["precision", "recall", "f1", "auc",
                        "accuracy", "fpr", "separability_ratio",
                        "avg_normal_error", "avg_attack_error"]
        delta = {}
        for k in metric_keys:
            delta[k] = round(res_s.get(k, 0) - res_u.get(k, 0), 6)

        print("\n  ── 差值摘要（半監督 - 非監督）──")
        for k in ["precision", "recall", "f1", "auc", "separability_ratio"]:
            sign = "+" if delta[k] >= 0 else ""
            print(f"    Δ {k:<22} {sign}{delta[k]:.4f}")

        return {
            "unsupervised": res_u,
            "semi":         res_s,
            "delta":        delta,
        }

    # ══════════════════════════════════════════════════════════════
    # 繪圖方法
    # ══════════════════════════════════════════════════════════════

    def plot_all(self, results: dict):
        """
        產生四張完整對比圖：
          1. 四項指標長條圖（Precision/Recall/F1/AUC）
          2. 誤差分布對比圖（正常 vs 攻擊，兩種策略並排）
          3. ROC 曲線對比圖
          4. 雷達圖（六維指標）
        """
        self.plot_metrics_bar(results)
        self.plot_error_distribution(results)
        self.plot_roc_curves(results)
        self.plot_radar(results)
        print(f"\n  [ComparisonBenchmark] 四張對比圖已輸出至: "
              f"{os.path.abspath(self.output_dir)}")

    def plot_metrics_bar(self, results: dict, filename: str = "comparison_metrics_bar.png"):
        """
        圖 1：Precision / Recall / F1 / AUC 四項指標並排長條對比圖。

        含「差值標注」（Δ 箭頭），讓讀者直觀看到半監督的提升量。
        """
        res_u = results["unsupervised"]
        res_s = results["semi"]
        delta = results["delta"]

        metrics  = ["Precision", "Recall", "F1 Score", "AUC-ROC"]
        keys     = ["precision", "recall", "f1", "auc"]
        vals_u   = [res_u[k] for k in keys]
        vals_s   = [res_s[k] for k in keys]
        deltas   = [delta[k] for k in keys]

        x = np.arange(len(metrics))
        w = 0.32

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(self._DARK_BG)
        ax.set_facecolor(self._DARK_AX)

        b1 = ax.bar(x - w/2, vals_u, w, color="#58a6ff",
                    alpha=0.90, label=res_u["label"])
        b2 = ax.bar(x + w/2, vals_s, w, color="#3fb950",
                    alpha=0.90, label=res_s["label"])

        # 數值標注 + Δ 差值
        for i, (bu, bs, d) in enumerate(zip(b1, b2, deltas)):
            hu, hs = bu.get_height(), bs.get_height()
            ax.text(bu.get_x() + bu.get_width()/2, hu + 0.008,
                    f"{hu:.3f}", ha="center", va="bottom",
                    color="white", fontsize=9)
            ax.text(bs.get_x() + bs.get_width()/2, hs + 0.008,
                    f"{hs:.3f}", ha="center", va="bottom",
                    color="white", fontsize=9)
            sign = "▲" if d >= 0 else "▼"
            color_d = "#3fb950" if d >= 0 else "#f85149"
            ax.text(x[i], max(hu, hs) + 0.045,
                    f"{sign}{abs(d):.3f}",
                    ha="center", va="bottom",
                    color=color_d, fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, color="#8b949e", fontsize=11)
        ax.set_ylabel("指標值", color="#8b949e", fontsize=11)
        ax.set_ylim(0, 1.20)
        ax.set_title("半監督 vs 非監督：Precision / Recall / F1 / AUC 對比\n"
                     "（▲/▼ 表示半監督相對非監督的提升/下降）",
                     color="white", fontsize=12)
        ax.legend(facecolor=self._DARK_AX, labelcolor="white", fontsize=10)
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

        # 右側摘要文字
        sep_u = res_u["separability_ratio"]
        sep_s = res_s["separability_ratio"]
        d_sep = delta["separability_ratio"]
        sign  = "▲" if d_sep >= 0 else "▼"
        summary = (
            f"分離比\n"
            f"非監督：{sep_u:.2f}x\n"
            f"半監督：{sep_s:.2f}x\n"
            f"{sign} {abs(d_sep):.2f}x"
        )
        ax.text(1.01, 0.97, summary, transform=ax.transAxes,
                color="white", fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor=self._DARK_AX, edgecolor="#30363d"))

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=self._DARK_BG)
        plt.close(fig)
        print(f"  [圖表] 四項指標對比圖已儲存: {path}")

    def plot_error_distribution(self, results: dict,
                                filename: str = "comparison_error_dist.png"):
        """
        圖 2：正常 vs 攻擊重建誤差分布對比（兩種策略並排）。

        理想狀態：兩個分布分離越遠越好（分離比越高越好）。
        """
        res_u = results["unsupervised"]
        res_s = results["semi"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor(self._DARK_BG)
        fig.suptitle("重建誤差分布對比：正常流量 vs 攻擊流量\n"
                     "（兩分布分離越遠 → 閾值越容易設定 → 偵測效能越好）",
                     color="white", fontsize=12, y=1.02)

        for ax, res, title_extra in [
            (axes[0], res_u, "純非監督（Unsupervised）"),
            (axes[1], res_s, "半監督微調（Semi-supervised）"),
        ]:
            ax.set_facecolor(self._DARK_AX)
            en = res["_errors_normal"]
            ea = res["_errors_attack"]
            thr = res["threshold"]
            sep = res["separability_ratio"]

            ax.hist(en, bins=60, alpha=0.65, color="#3fb950", density=True,
                    label=f"正常流量 (N={len(en):,})")
            ax.hist(ea, bins=60, alpha=0.65, color="#f85149", density=True,
                    label=f"攻擊流量 (N={len(ea):,})")
            ax.axvline(thr, color="#ffd700", linewidth=2.5, linestyle="--",
                       label=f"閾值（{self.percentile}th pct）={thr:.5f}")

            # 標注均值
            ax.axvline(en.mean(), color="#3fb950", linewidth=1, linestyle=":",
                       alpha=0.7, label=f"正常均值={en.mean():.5f}")
            ax.axvline(ea.mean(), color="#f85149", linewidth=1, linestyle=":",
                       alpha=0.7, label=f"攻擊均值={ea.mean():.5f}")

            ax.set_xlabel("重建誤差 (MSE)", color="#8b949e", fontsize=10)
            ax.set_ylabel("機率密度", color="#8b949e", fontsize=10)
            ax.set_title(f"{title_extra}\n分離比={sep:.2f}x  "
                         f"F1={res['f1']:.4f}  AUC={res['auc']:.4f}",
                         color="white", fontsize=11)
            ax.legend(facecolor=self._DARK_AX, labelcolor="white", fontsize=8)
            ax.tick_params(colors="#8b949e")
            for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=self._DARK_BG)
        plt.close(fig)
        print(f"  [圖表] 誤差分布對比圖已儲存: {path}")

    def plot_roc_curves(self, results: dict,
                        filename: str = "comparison_roc_curve.png"):
        """
        圖 3：ROC 曲線對比圖。

        同一張圖顯示兩種策略的 ROC 曲線，
        標注各自的 AUC 值與目前閾值的工作點。
        """
        res_u = results["unsupervised"]
        res_s = results["semi"]

        fig, ax = plt.subplots(figsize=(9, 8))
        fig.patch.set_facecolor(self._DARK_BG)
        ax.set_facecolor(self._DARK_AX)

        for res, color, lw in [
            (res_u, "#58a6ff", 2.5),
            (res_s, "#3fb950", 2.5),
        ]:
            fpr_c = res["_fpr_curve"]
            tpr_c = res["_tpr_curve"]
            auc   = res["auc"]
            label = res["label"]
            ax.plot(fpr_c, tpr_c, color=color, linewidth=lw,
                    label=f"{label}（AUC={auc:.4f}）")

            # 標注目前閾值的工作點（FPR, TPR）
            wp_fpr = res["fpr"]
            wp_tpr = res["recall"]
            ax.scatter([wp_fpr], [wp_tpr], s=100, color=color,
                       edgecolors="white", linewidths=1.5, zorder=6)
            ax.annotate(f"  FPR={wp_fpr:.3f}\n  TPR={wp_tpr:.3f}",
                        (wp_fpr, wp_tpr), color=color, fontsize=8)

        ax.plot([0, 1], [0, 1], color="#30363d", linestyle="--",
                linewidth=1.5, label="Random（AUC=0.500）")

        ax.set_xlabel("False Positive Rate（誤報率）", color="#8b949e", fontsize=11)
        ax.set_ylabel("True Positive Rate（偵測率）", color="#8b949e", fontsize=11)
        ax.set_title("ROC 曲線對比：半監督 vs 非監督\n"
                     "（黑點 = 目前閾值的工作點）",
                     color="white", fontsize=12)
        ax.legend(facecolor=self._DARK_AX, labelcolor="white", fontsize=10)
        ax.tick_params(colors="#8b949e")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=self._DARK_BG)
        plt.close(fig)
        print(f"  [圖表] ROC 曲線對比圖已儲存: {path}")

    def plot_radar(self, results: dict,
                   filename: str = "comparison_radar.png"):
        """
        圖 4：六維效能雷達圖。

        六個維度：Precision / Recall / F1 / AUC / Accuracy / 1-FPR
        面積越大代表整體效能越好。
        """
        res_u = results["unsupervised"]
        res_s = results["semi"]

        dims   = ["Precision", "Recall", "F1", "AUC", "Accuracy", "1-FPR"]
        keys_u = [res_u["precision"], res_u["recall"], res_u["f1"],
                  res_u["auc"], res_u["accuracy"], 1 - res_u["fpr"]]
        keys_s = [res_s["precision"], res_s["recall"], res_s["f1"],
                  res_s["auc"], res_s["accuracy"], 1 - res_s["fpr"]]

        n      = len(dims)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        # 閉合多邊形
        vals_u = keys_u + [keys_u[0]]
        vals_s = keys_s + [keys_s[0]]
        angles_c = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        fig.patch.set_facecolor(self._DARK_BG)
        ax.set_facecolor(self._DARK_AX)

        ax.plot(angles_c, vals_u, "o-", color="#58a6ff",
                linewidth=2, label=res_u["label"])
        ax.fill(angles_c, vals_u, color="#58a6ff", alpha=0.15)

        ax.plot(angles_c, vals_s, "o-", color="#3fb950",
                linewidth=2, label=res_s["label"])
        ax.fill(angles_c, vals_s, color="#3fb950", alpha=0.15)

        ax.set_thetagrids(np.degrees(angles), dims,
                          color="white", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                           color="#8b949e", fontsize=8)
        ax.grid(color="#30363d", linewidth=0.8)
        ax.spines["polar"].set_edgecolor("#30363d")
        ax.set_title("六維效能雷達圖\n（面積越大 → 整體效能越好）",
                     color="white", fontsize=12, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15),
                  facecolor=self._DARK_AX, labelcolor="white", fontsize=10)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=self._DARK_BG)
        plt.close(fig)
        print(f"  [圖表] 雷達圖已儲存: {path}")

    # ── 儲存 JSON 報告 ─────────────────────────────────────────────
    def save_report(self, results: dict,
                    filename: str = "comparison_report.json"):
        """
        儲存完整比較報告為 JSON。

        報告結構：
        {
            "meta": { 訓練設定 },
            "unsupervised": { 所有量化指標（不含 numpy 陣列） },
            "semi":         { 所有量化指標 },
            "delta":        { 差值 },
            "conclusion":   { 自動文字摘要 }
        }
        """
        # 排除 numpy 陣列和 nn.Module（無法 JSON 序列化）
        _skip = {"_errors_normal", "_errors_attack",
                 "_fpr_curve", "_tpr_curve", "model",
                 "train_losses", "pretrain_losses",
                 "finetune_normal", "finetune_attack", "finetune_total"}

        def _clean(d: dict) -> dict:
            return {k: (float(v) if isinstance(v, (np.floating, np.integer))
                        else int(v) if isinstance(v, np.integer)
                        else v)
                    for k, v in d.items()
                    if k not in _skip and not isinstance(v, (np.ndarray, nn.Module))}

        res_u = results["unsupervised"]
        res_s = results["semi"]
        delta = results["delta"]

        # 自動結論
        winner   = "半監督" if res_s["f1"] >= res_u["f1"] else "非監督"
        f1_delta = res_s["f1"] - res_u["f1"]
        sep_gain = res_s["separability_ratio"] - res_u["separability_ratio"]
        conclusion = {
            "winner":          winner,
            "f1_improvement":  round(f1_delta, 4),
            "sep_improvement": round(sep_gain, 4),
            "recommendation":  (
                f"半監督微調在分離比上{'提升' if sep_gain>=0 else '降低'} "
                f"{abs(sep_gain):.3f}x，"
                f"F1 Score {'提升' if f1_delta>=0 else '降低'} "
                f"{abs(f1_delta):.4f}。"
                + ("建議採用半監督策略。"
                   if f1_delta >= 0
                   else "本次資料集半監督效益有限，建議增加標記攻擊樣本或調整 margin 值。")
            ),
        }

        report = {
            "meta": {
                "latent_dim":        self.latent_dim,
                "pretrain_epochs":   self.pretrain_epochs,
                "finetune_epochs":   self.finetune_epochs,
                "attack_ratio":      self.attack_ratio,
                "percentile":        self.percentile,
                "device":            str(self.device),
                "n_normal_train":    len(self.X_normal_train),
                "n_normal_test":     len(self.X_normal_test),
                "n_attack_test":     len(self.X_attack_test),
            },
            "unsupervised": _clean(res_u),
            "semi":         _clean(res_s),
            "delta":        delta,
            "conclusion":   conclusion,
        }

        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  [報告] 完整比較報告已儲存: {path}")

        # 印出摘要表格
        self._print_summary_table(res_u, res_s, delta, conclusion)
        return path

    def _print_summary_table(self, res_u, res_s, delta, conclusion):
        """印出純文字摘要比較表。"""
        print("\n  ┌──────────────────────────────────────────────────────────────┐")
        print("  │         半監督 vs 非監督 CNN Autoencoder 基準比較摘要          │")
        print("  ├────────────────────────┬─────────────┬─────────────┬──────────┤")
        print("  │  指標                  │   非監督    │   半監督    │    Δ     │")
        print("  ├────────────────────────┼─────────────┼─────────────┼──────────┤")
        rows = [
            ("Precision",      "precision"),
            ("Recall",         "recall"),
            ("F1 Score",       "f1"),
            ("AUC-ROC",        "auc"),
            ("Accuracy",       "accuracy"),
            ("FPR（誤報率）",  "fpr"),
            ("分離比（x）",    "separability_ratio"),
        ]
        for label, key in rows:
            u = res_u[key]; s = res_s[key]; d = delta[key]
            sign = "▲" if d >= 0 else "▼"
            print(f"  │  {label:<22}  │  {u:>9.4f}  │  {s:>9.4f}  │ "
                  f"{sign}{abs(d):.4f}  │")
        print("  ├────────────────────────┼─────────────┼─────────────┼──────────┤")
        print(f"  │  訓練耗時 (s)          │  {res_u['train_time']:>9.1f}  │"
              f"  {res_s['train_time']:>9.1f}  │          │")
        print("  └────────────────────────┴─────────────┴─────────────┴──────────┘")
        print(f"\n  結論：{conclusion['recommendation']}\n")


# ══════════════════════════════════════════════════════════════════
# 命令列介面
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="半監督 vs 非監督 CNN Autoencoder 基準比較",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  python core/comparison_benchmark.py --dataset simulate
  python core/comparison_benchmark.py --dataset cicids2017 --data-dir data/cicids2017
  python core/comparison_benchmark.py \\
      --normal output/dataset_simulate/X_normal.npy \\
      --attack output/dataset_simulate/X_attack.npy
        """
    )
    parser.add_argument("--dataset",   default=None,
                        choices=["simulate", "nslkdd", "cicids2017", "cicddos2019"],
                        help="使用內建 DatasetFactory 載入資料集")
    parser.add_argument("--data-dir",  default=None)
    parser.add_argument("--normal",    default=None, help="直接指定 .npy 正常流量")
    parser.add_argument("--attack",    default=None, help="直接指定 .npy 攻擊流量")
    parser.add_argument("--output",    default="output/comparison")
    parser.add_argument("--latent",    type=int, default=32)
    parser.add_argument("--pretrain",  type=int, default=50,
                        help="Phase 1 訓練輪數（預設 50）")
    parser.add_argument("--finetune",  type=int, default=50,
                        help="Phase 2 微調輪數（預設 50）")
    parser.add_argument("--attack-ratio", type=float, default=0.20,
                        help="Phase 2 使用的攻擊樣本比例（預設 0.20）")
    parser.add_argument("--pct",       type=int, default=95,
                        help="閾值百分位數（預設 95）")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    # 自動調整輸出目錄，使其包含資料集名稱，避免多個模型結果互相覆蓋
    if args.output == "output/comparison":
        dataset_name = args.dataset or "manual"
        args.output = f"output/{dataset_name}/comparison"

    print("\n" + "=" * 65)
    print("  半監督 vs 非監督 CNN Autoencoder 基準比較")
    print("=" * 65)

    # 載入資料
    if args.normal and os.path.exists(args.normal):
        X_normal = np.load(args.normal).astype(np.float32)
        X_attack = np.load(args.attack).astype(np.float32)
        print(f"  載入資料：正常={X_normal.shape}  攻擊={X_attack.shape}")

    elif args.dataset:
        try:
            from dataset_loader import DatasetFactory
            load_kw = {}
            if args.dataset in ("cicids2017", "cicddos2019"):
                load_kw = {"max_normal": 50000, "max_attack": 30000}
            X_normal, X_attack, _ = DatasetFactory.load(
                args.dataset, data_dir=args.data_dir, **load_kw)
            print(f"  資料集 {args.dataset}：正常={X_normal.shape}  攻擊={X_attack.shape}")
        except FileNotFoundError as e:
            print(f"  錯誤：{e}")
            sys.exit(1)

    else:
        print("  未指定資料集，使用模擬資料（Beta 分布）...")
        np.random.seed(args.seed)
        X_normal = np.random.beta(2, 5, (600, 32, 32)).astype(np.float32)
        X_attack = np.random.beta(5, 2, (250, 32, 32)).astype(np.float32)
        print(f"  模擬：正常={X_normal.shape}  攻擊={X_attack.shape}")

    # 執行比較
    bench = ComparisonBenchmark(
        X_normal        = X_normal,
        X_attack        = X_attack,
        output_dir      = args.output,
        latent_dim      = args.latent,
        pretrain_epochs = args.pretrain,
        finetune_epochs = args.finetune,
        attack_ratio    = args.attack_ratio,
        percentile      = args.pct,
        seed            = args.seed,
    )

    results = bench.run()
    bench.plot_all(results)
    bench.save_report(results)

    print("\n" + "=" * 65)
    print(f"  比較完成！結果輸出：{os.path.abspath(args.output)}")
    print("=" * 65)