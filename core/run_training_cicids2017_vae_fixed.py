# -*- coding: utf-8 -*-
"""
run_training_cicids2017_vae_fixed.py

CICIDS2017「非監督式 VAE」修正版。

重點：
1. 模型維持 CNN-VAE，不改成一般 CNN Autoencoder。
2. VAE 只使用正常流量做模型訓練，攻擊資料不進入 loss。
3. Normal 先切 Train / Validation / Test，再 fit scaler。
4. scaler 只用 Train-Normal fit，避免資料洩漏。
5. Validation normal 可用來決定 percentile threshold；
   也可用 validation attack 做 threshold calibration（模型本身仍是
   unsupervised VAE，attack 不參與模型參數學習）。
6. Final Test 完全不參與 threshold tuning。
7. 匯出 scaler + feature_columns，讓 PCAP->CSV serving 可以沿用相同前處理。
8. Checkpoint 使用 model_type='cnn_vae'，可直接對應現有 model_registry。

執行：
python core/run_training_cicids2017_vae_fixed.py ^
  --data-dir data/cicids2017 ^
  --output output/model_cicids2017 ^
  --epochs 120 ^
  --patience 20 ^
  --latent 32 ^
  --batch 64 ^
  --lr 5e-4 ^
  --kl-weight 1e-3 ^
  --threshold-method percentile ^
  --pct 95 ^
  --max-normal 60000 ^
  --max-attack 30000

若希望只用 validation attack 做 threshold 校準：
  --threshold-method optimal
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 內建評估繪圖工具
# 不依賴 evaluation_plotter.py
# ============================================================
def _configure_plot_font():
    """自動尋找常見 CJK 字型，避免 Windows / Linux 中文亂碼。"""
    import matplotlib
    import matplotlib.font_manager as fm

    cjk_fonts = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "SimHei",
        "PingFang TC",
        "Heiti TC",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK TC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in cjk_fonts:
        if name in available:
            matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


_configure_plot_font()


def _as_float_list(values):
    return [
        float(x)
        for x in np.asarray(values, dtype=np.float64).reshape(-1)
    ]


def add_vae_plot_data(report, normal_scores, attack_scores, threshold):
    """把 VAE 測試分數寫入 JSON，供訓練程式本身重建三張圖。"""
    report["plot_data"] = {
        "version": 1,
        "model_view": "vae",
        "confusion_metrics": {
            "TN": int(report["TN"]),
            "FP": int(report["FP"]),
            "FN": int(report["FN"]),
            "TP": int(report["TP"]),
            "precision": float(report["precision"]),
            "recall": float(report["recall"]),
            "fpr": float(report["fpr"]),
        },
        "roc_pr": {
            "normal_scores": _as_float_list(normal_scores),
            "attack_scores": _as_float_list(attack_scores),
            "score_name": "VAE 異常分數",
            "title": "CNN-VAE 異常偵測效能評估",
        },
        "score_distribution": {
            "normal_scores": _as_float_list(normal_scores),
            "attack_scores": _as_float_list(attack_scores),
            "threshold": float(threshold),
            "score_name": "重建誤差 (MSE)",
            "title": "正常流量 vs 攻擊流量 重建誤差分布",
        },
    }
    return report


def add_hybrid_plot_data(
    report,
    normal_vae_scores,
    attack_vae_scores,
    normal_classifier_scores,
    attack_classifier_scores,
    vae_threshold,
    classifier_threshold,
):
    """把 Hybrid 的 VAE / classifier 連續分數寫入 JSON。"""
    h = report["hybrid_final"]

    report["plot_data"] = {
        "version": 1,
        "model_view": "hybrid_cascade",
        "confusion_metrics": {
            "TN": int(h["TN"]),
            "FP": int(h["FP"]),
            "FN": int(h["FN"]),
            "TP": int(h["TP"]),
            "precision": float(h["precision"]),
            "recall": float(h["recall"]),
            "fpr": float(h["fpr"]),
        },
        "roc_pr": {
            "normal_scores": _as_float_list(normal_classifier_scores),
            "attack_scores": _as_float_list(attack_classifier_scores),
            "score_name": "CNN-LSTM Attack Probability",
            "title": "Hybrid CNN-LSTM Classifier ROC / PR 評估",
        },
        "score_distribution": {
            "normal_scores": _as_float_list(normal_vae_scores),
            "attack_scores": _as_float_list(attack_vae_scores),
            "threshold": float(vae_threshold),
            "score_name": "VAE 重建誤差 (MSE)",
            "title": "Hybrid VAE Novelty Score 分布",
        },
        "thresholds": {
            "classifier_threshold": float(classifier_threshold),
            "vae_threshold": float(vae_threshold),
        },
    }
    return report


def _get_curve_data(section):
    normal = np.asarray(section["normal_scores"], dtype=np.float64)
    attack = np.asarray(section["attack_scores"], dtype=np.float64)

    y_true = np.concatenate([
        np.zeros(len(normal), dtype=np.int32),
        np.ones(len(attack), dtype=np.int32),
    ])
    y_score = np.concatenate([normal, attack])

    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = float(auc(recall, precision))

    return (
        y_true,
        y_score,
        fpr,
        tpr,
        thresholds,
        roc_auc,
        precision,
        recall,
        pr_auc,
    )


def _plot_confusion_matrix(plot_data, output_dir):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    cm_data = plot_data["confusion_metrics"]
    cm = np.array([
        [cm_data["TN"], cm_data["FP"]],
        [cm_data["FN"], cm_data["TP"]],
    ], dtype=np.int64)

    # 白底 + 淡藍色格子 + 黑字，避免文字與背景顏色衝突。
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    vmax = max(int(cm.max()), 1)
    im = ax.imshow(
        cm,
        cmap="Blues",
        norm=Normalize(vmin=0, vmax=max(vmax * 1.35, 1)),
    )

    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{cm[i, j]}",
                ha="center",
                va="center",
                color="black",
                fontsize=12,
                fontweight="bold",
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["預測:正常", "預測:攻擊"], color="black")
    ax.set_yticklabels(["實際:正常", "實際:攻擊"], color="black")
    ax.set_xlabel("預測類別", color="black")
    ax.set_ylabel("實際類別", color="black")
    ax.set_title(
        f"混淆矩陣  (Recall={cm_data['recall']:.3f}  FPR={cm_data['fpr']:.3f})",
        color="black",
        fontsize=11,
    )
    ax.tick_params(axis="both", colors="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="black")

    fig.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_roc_pr(plot_data, output_dir):
    import matplotlib.pyplot as plt

    section = plot_data["roc_pr"]
    _, _, fpr, tpr, _, roc_auc, precision, recall, pr_auc = _get_curve_data(section)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")

    ax1.set_facecolor("#161b22")
    ax1.plot(
        fpr,
        tpr,
        color="#58a6ff",
        linewidth=2,
        label=f"ROC Curve (AUC = {roc_auc:.3f})",
    )
    ax1.plot(
        [0, 1],
        [0, 1],
        color="#30363d",
        linestyle="--",
        label="Random (AUC = 0.500)",
    )
    ax1.set_xlabel("False Positive Rate（誤報率）", color="#8b949e")
    ax1.set_ylabel("True Positive Rate（偵測率）", color="#8b949e")
    ax1.set_title("ROC 曲線", color="white", fontsize=12)
    ax1.legend(facecolor="#161b22", labelcolor="white", fontsize=9)
    ax1.tick_params(colors="#8b949e")

    ax2.set_facecolor("#161b22")
    ax2.plot(
        recall,
        precision,
        color="#3fb950",
        linewidth=2,
        label=f"PR Curve (AUC = {pr_auc:.3f})",
    )
    ax2.set_xlabel("Recall（偵測率）", color="#8b949e")
    ax2.set_ylabel("Precision（精確率）", color="#8b949e")
    ax2.set_title("Precision-Recall 曲線", color="white", fontsize=12)
    ax2.legend(facecolor="#161b22", labelcolor="white")
    ax2.tick_params(colors="#8b949e")

    fig.suptitle(
        section.get("title", "異常偵測效能評估"),
        color="white",
        fontsize=13,
    )
    fig.tight_layout()

    path = os.path.join(output_dir, "roc_pr_curve.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return path


def _plot_score_distribution(plot_data, output_dir):
    import matplotlib.pyplot as plt

    section = plot_data["score_distribution"]
    normal = np.asarray(section["normal_scores"], dtype=np.float64)
    attack = np.asarray(section["attack_scores"], dtype=np.float64)
    threshold = float(section["threshold"])

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    ax.hist(
        normal,
        bins=60,
        alpha=0.6,
        color="#3fb950",
        label=f"正常流量 (N={len(normal)})",
        density=True,
    )
    ax.hist(
        attack,
        bins=60,
        alpha=0.6,
        color="#f85149",
        label=f"攻擊流量 (N={len(attack)})",
        density=True,
    )

    ax.axvline(
        threshold,
        color="#ffd700",
        linewidth=2.5,
        linestyle="--",
        label=f"閾值 = {threshold:.4f}",
    )

    ax.set_xlabel(
        section.get("score_name", "重建誤差 (MSE)"),
        color="#8b949e",
        fontsize=11,
    )
    ax.set_ylabel("機率密度", color="#8b949e", fontsize=11)
    ax.set_title(
        section.get("title", "正常流量 vs 攻擊流量 分數分布"),
        color="white",
        fontsize=13,
    )
    ax.legend(facecolor="#161b22", labelcolor="white")
    ax.tick_params(colors="#8b949e")

    path = os.path.join(output_dir, "score_distribution.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return path


def generate_evaluation_plots_from_json(json_path, output_dir=None):
    """由 JSON 的 plot_data 產生三張評估圖。"""
    from pathlib import Path

    json_path = Path(json_path)
    if output_dir is None:
        output_dir = json_path.parent
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    if "plot_data" not in report:
        raise ValueError(
            f"{json_path} 缺少 plot_data。"
            "舊版 JSON 只有彙總指標，無法由 JSON 還原 ROC/PR 與 score distribution。"
        )

    plot_data = report["plot_data"]
    paths = {
        "confusion_matrix": _plot_confusion_matrix(plot_data, output_dir),
        "roc_pr_curve": _plot_roc_pr(plot_data, output_dir),
        "score_distribution": _plot_score_distribution(plot_data, output_dir),
    }

    print(f"  [圖表] 混淆矩陣: {paths['confusion_matrix']}")
    print(f"  [圖表] ROC/PR : {paths['roc_pr_curve']}")
    print(f"  [圖表] 分數分布: {paths['score_distribution']}")
    return paths



BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from variational_autoencoder import CNNVariationalAutoencoder


IMAGE_SIZE = 32
FEATURE_DIM = IMAGE_SIZE * IMAGE_SIZE

# CICIDS2017 預期特徵順序。
# 保留與專案 dataset_loader 的 feature list 一致。
CICIDS_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="CICIDS2017 非監督式 CNN-VAE 修正版"
    )

    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--output",
        default="output/model_cicids2017",
    )

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-3)

    parser.add_argument(
        "--threshold-method",
        choices=["percentile", "optimal"],
        default="percentile",
        help=(
            "percentile：只使用 validation normal；"
            "optimal：使用 validation normal + attack 搜尋 F1 threshold"
        ),
    )
    parser.add_argument("--pct", type=float, default=95.0)

    parser.add_argument("--max-normal", type=int, default=60000)
    parser.add_argument("--max-attack", type=int, default=30000)

    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def find_label_column(columns):
    candidates = {
        "label",
        "class",
        "attack",
        "attack_type",
        "attack type",
        "target",
    }

    for col in columns:
        if str(col).strip().lower() in candidates:
            return col

    raise ValueError(
        f"找不到 Label 欄位，現有欄位：{list(columns)[:20]}"
    )


def find_feature_columns(columns):
    mapping = {str(c).strip(): c for c in columns}
    selected = []

    for name in CICIDS_FEATURES:
        if name in mapping:
            selected.append(mapping[name])

    # 某些 CICIDS2017 CSV 可能缺少某個重複欄位，
    # 但至少需要絕大多數標準流量特徵。
    if len(selected) < 60:
        raise ValueError(
            f"CICIDS2017 特徵匹配不足：{len(selected)} / {len(CICIDS_FEATURES)}"
        )

    return selected


def clean_numeric(X):
    X = X.astype(np.float64, copy=False)

    for j in range(X.shape[1]):
        col = X[:, j]
        pos_inf = np.isposinf(col)
        neg_inf = np.isneginf(col)

        if not (pos_inf.any() or neg_inf.any()):
            continue

        finite = col[np.isfinite(col)]

        if finite.size == 0:
            col_min = 0.0
            col_max = 0.0
        else:
            col_min = float(np.min(finite))
            col_max = float(np.max(finite))

        X[pos_inf, j] = col_max
        X[neg_inf, j] = col_min

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return X.astype(np.float32)


def load_dataset(data_dir, max_normal, max_attack, seed):
    rng = np.random.default_rng(seed)

    csv_files = sorted(Path(data_dir).rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"在 {data_dir} 找不到 CSV。"
        )

    frames = []

    print(f"[Data] CSV 數量：{len(csv_files)}")

    for path in csv_files:
        try:
            df = pd.read_csv(path, low_memory=False)
            df.columns = df.columns.str.strip()
            frames.append(df)
            print(
                f"  {path.name:<45} {len(df):>10,}"
            )
        except Exception as exc:
            print(f"  [Skip] {path.name}: {exc}")

    if not frames:
        raise RuntimeError("沒有任何 CSV 成功載入。")

    data = pd.concat(frames, ignore_index=True)

    label_col = find_label_column(data.columns)
    feature_cols = find_feature_columns(data.columns)

    labels = data[label_col].astype(str).str.strip()

    normal_mask = labels.str.upper().isin(
        {"BENIGN", "NORMAL", "NORMAL TRAFFIC"}
    )

    normal_df = data.loc[normal_mask, feature_cols]
    attack_df = data.loc[~normal_mask, feature_cols]

    X_normal = clean_numeric(
        normal_df.to_numpy(dtype=np.float64, copy=True)
    )
    X_attack = clean_numeric(
        attack_df.to_numpy(dtype=np.float64, copy=True)
    )

    print(
        f"[Data] 正常={len(X_normal):,}, "
        f"攻擊={len(X_attack):,}, "
        f"features={len(feature_cols)}"
    )

    if max_normal and len(X_normal) > max_normal:
        idx = rng.choice(
            len(X_normal),
            size=max_normal,
            replace=False,
        )
        X_normal = X_normal[idx]

    if max_attack and len(X_attack) > max_attack:
        idx = rng.choice(
            len(X_attack),
            size=max_attack,
            replace=False,
        )
        X_attack = X_attack[idx]

    return X_normal, X_attack, feature_cols


def split_three_way(X, train_ratio, val_ratio, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    n_train = max(1, int(len(X) * train_ratio))
    n_val = max(1, int(len(X) * val_ratio))

    if n_train + n_val >= len(X):
        raise ValueError("資料量不足以切出 train / validation / test。")

    return (
        X[indices[:n_train]],
        X[indices[n_train:n_train + n_val]],
        X[indices[n_train + n_val:]],
    )


def fit_scaler_and_transform(
    train_normal,
    val_normal,
    test_normal,
    val_attack,
    test_attack,
):
    """
    scaler 只用 Train-Normal fit。

    Decoder 最後是 Sigmoid，因此輸入保持在 [0,1]。
    """
    scaler = RobustScaler()
    scaler.fit(train_normal)

    def transform(X):
        X_scaled = scaler.transform(X)
        return np.clip(
            X_scaled,
            0.0,
            1.0,
        ).astype(np.float32)

    return (
        transform(train_normal),
        transform(val_normal),
        transform(test_normal),
        transform(val_attack),
        transform(test_attack),
        scaler,
    )


def to_images(X):
    n = X.shape[0]

    if X.shape[1] > FEATURE_DIM:
        X = X[:, :FEATURE_DIM]

    elif X.shape[1] < FEATURE_DIM:
        pad = np.zeros(
            (n, FEATURE_DIM - X.shape[1]),
            dtype=np.float32,
        )
        X = np.concatenate([X, pad], axis=1)

    return X.reshape(
        n,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ).astype(np.float32)


def compute_vae_scores(model, X, device, batch_size=256):
    tensor = torch.from_numpy(
        X[:, None].astype(np.float32)
    )

    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()

    scores = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            score = model.reconstruction_error(batch)
            scores.append(score.cpu().numpy())

    return np.concatenate(scores)


def optimal_threshold(normal_scores, attack_scores):
    """
    Validation threshold optimization。

    注意：
    VAE 本身仍然完全用 normal train 訓練；
    validation attack 只用來選「部署 threshold」。
    """
    from sklearn.metrics import f1_score

    all_scores = np.concatenate([
        normal_scores,
        attack_scores,
    ])

    candidates = np.unique(
        np.quantile(
            all_scores,
            np.linspace(0.80, 0.999, 500),
        )
    )

    y_true = np.concatenate([
        np.zeros(len(normal_scores), dtype=np.int32),
        np.ones(len(attack_scores), dtype=np.int32),
    ])

    best = None

    for threshold in candidates:
        y_pred = (
            all_scores > threshold
        ).astype(np.int32)

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        if best is None or f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
            }

    return best["threshold"], best["f1"]


def evaluate(normal_scores, attack_scores, threshold):
    tp = int(np.sum(attack_scores > threshold))
    fn = int(np.sum(attack_scores <= threshold))

    fp = int(np.sum(normal_scores > threshold))
    tn = int(np.sum(normal_scores <= threshold))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(
        precision + recall,
        1e-12,
    )
    fpr = fp / max(fp + tn, 1)
    accuracy = (
        (tp + tn)
        / max(tp + tn + fp + fn, 1)
    )

    y_true = np.concatenate([
        np.zeros(len(normal_scores)),
        np.ones(len(attack_scores)),
    ])
    y_score = np.concatenate([
        normal_scores,
        attack_scores,
    ])

    return {
        "threshold": float(threshold),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "accuracy": float(accuracy),
        "roc_auc": float(
            roc_auc_score(y_true, y_score)
        ),
        "pr_auc": float(
            average_precision_score(y_true, y_score)
        ),
    }


class VAETrainerFixed:
    def __init__(
        self,
        latent_dim,
        batch_size,
        lr,
        kl_weight,
        epochs,
        patience,
        device,
        output_dir,
    ):
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model = CNNVariationalAutoencoder(
            latent_dim=latent_dim,
            image_size=IMAGE_SIZE,
            kl_weight=kl_weight,
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

        self.train_losses = []
        self.val_losses = []

    def fit(self, X_train, X_val):
        train_tensor = torch.from_numpy(
            X_train[:, None].astype(np.float32)
        )
        val_tensor = torch.from_numpy(
            X_val[:, None].astype(np.float32)
        )

        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_loader = DataLoader(
            TensorDataset(val_tensor),
            batch_size=self.batch_size,
            shuffle=False,
        )

        # 每 epoch 都有固定 step，因此 OneCycleLR 可直接使用。
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]["lr"],
            steps_per_epoch=max(len(train_loader), 1),
            epochs=self.epochs,
            pct_start=0.15,
            div_factor=10.0,
            final_div_factor=100.0,
        )

        best_val = float("inf")
        best_state = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            train_sum = 0.0
            train_count = 0

            for (batch,) in train_loader:
                batch = batch.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                loss = self.model.vae_loss(batch)

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0,
                )

                self.optimizer.step()
                scheduler.step()

                train_sum += (
                    float(loss.item())
                    * len(batch)
                )
                train_count += len(batch)

            train_loss = train_sum / max(train_count, 1)

            self.model.eval()

            val_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(self.device)

                    loss = self.model.vae_loss(batch)

                    val_sum += (
                        float(loss.item())
                        * len(batch)
                    )
                    val_count += len(batch)

            val_loss = val_sum / max(val_count, 1)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            lr_now = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d}/{self.epochs} "
                f"train={train_loss:.7f} "
                f"val={val_loss:.7f} "
                f"lr={lr_now:.2e}"
            )

            if val_loss < best_val - 1e-7:
                best_val = val_loss
                wait = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value
                    in self.model.state_dict().items()
                }
            else:
                wait += 1

                if wait >= self.patience:
                    print(
                        f"[EarlyStopping] epoch={epoch}"
                    )
                    break

        if best_state is None:
            raise RuntimeError(
                "沒有取得有效的最佳 VAE 權重。"
            )

        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        self.model.eval()

        return best_val


def main():
    args = parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError(
            "train_ratio + val_ratio 必須小於 1。"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("CICIDS2017 非監督式 CNN-VAE 訓練")
    print("=" * 72)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # 1. Load raw features
    # --------------------------------------------------------
    X_normal_raw, X_attack_raw, feature_cols = load_dataset(
        args.data_dir,
        args.max_normal,
        args.max_attack,
        args.seed,
    )

    # --------------------------------------------------------
    # 2. Split before scaling
    # --------------------------------------------------------
    (
        normal_train,
        normal_val,
        normal_test,
    ) = split_three_way(
        X_normal_raw,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    (
        attack_train_unused,
        attack_val,
        attack_test,
    ) = split_three_way(
        X_attack_raw,
        args.train_ratio,
        args.val_ratio,
        args.seed + 1,
    )

    # 關鍵：
    # attack_train_unused 不參與任何 VAE loss。
    # 它只是留在 training partition 中，不使用即可。
    del attack_train_unused

    print()
    print(
        f"[Split] Normal: "
        f"train={len(normal_train):,}, "
        f"val={len(normal_val):,}, "
        f"test={len(normal_test):,}"
    )
    print(
        f"[Split] Attack: "
        f"val={len(attack_val):,}, "
        f"test={len(attack_test):,}"
    )

    # --------------------------------------------------------
    # 3. Fit scaler ONLY on train-normal
    # --------------------------------------------------------
    (
        normal_train,
        normal_val,
        normal_test,
        attack_val,
        attack_test,
        scaler,
    ) = fit_scaler_and_transform(
        normal_train,
        normal_val,
        normal_test,
        attack_val,
        attack_test,
    )

    # --------------------------------------------------------
    # 4. Feature -> image
    # --------------------------------------------------------
    X_train = to_images(normal_train)
    X_val = to_images(normal_val)
    X_val_attack = to_images(attack_val)
    X_test = to_images(normal_test)
    X_test_attack = to_images(attack_test)

    # --------------------------------------------------------
    # 5. Train unsupervised VAE
    # --------------------------------------------------------
    trainer = VAETrainerFixed(
        latent_dim=args.latent,
        batch_size=args.batch,
        lr=args.lr,
        kl_weight=args.kl_weight,
        epochs=args.epochs,
        patience=args.patience,
        device=device,
        output_dir=output_dir,
    )

    best_val_loss = trainer.fit(
        X_train,
        X_val,
    )

    # --------------------------------------------------------
    # 6. Validation threshold
    # --------------------------------------------------------
    print()
    print("[Threshold] Validation calibration")

    val_normal_scores = compute_vae_scores(
        trainer.model,
        X_val,
        device,
    )

    if args.threshold_method == "percentile":
        threshold = float(
            np.percentile(
                val_normal_scores,
                args.pct,
            )
        )

        threshold_info = {
            "method": "percentile",
            "percentile": args.pct,
        }

    else:
        val_attack_scores = compute_vae_scores(
            trainer.model,
            X_val_attack,
            device,
        )

        threshold, best_val_f1 = optimal_threshold(
            val_normal_scores,
            val_attack_scores,
        )

        threshold_info = {
            "method": "optimal",
            "validation_f1": best_val_f1,
        }

    print(
        f"Threshold={threshold:.8f} "
        f"method={args.threshold_method}"
    )

    # --------------------------------------------------------
    # 7. Final test
    # --------------------------------------------------------
    test_normal_scores = compute_vae_scores(
        trainer.model,
        X_test,
        device,
    )

    test_attack_scores = compute_vae_scores(
        trainer.model,
        X_test_attack,
        device,
    )

    result = evaluate(
        test_normal_scores,
        test_attack_scores,
        threshold,
    )

    result["dataset"] = "cicids2017"
    result["model_type"] = "cnn_vae"
    result["representation"] = (
        "csv_features_cicids2017"
    )
    result["best_val_loss"] = float(best_val_loss)
    result["threshold_info"] = threshold_info
    result["config"] = {
        "latent_dim": args.latent,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "kl_weight": args.kl_weight,
        "epochs": args.epochs,
        "patience": args.patience,
        "image_size": IMAGE_SIZE,
    }
    result["split"] = {
        "train_normal": len(X_train),
        "val_normal": len(X_val),
        "test_normal": len(X_test),
        "val_attack": len(X_val_attack),
        "test_attack": len(X_test_attack),
    }

    # --------------------------------------------------------
    # 8. Plot data + JSON
    # --------------------------------------------------------
    # 將 test score 寫入 JSON；訓練結束後的圖表完全由 JSON 讀取產生。
    add_vae_plot_data(
        result,
        normal_scores=test_normal_scores,
        attack_scores=test_attack_scores,
        threshold=threshold,
    )

    # --------------------------------------------------------
    # 9. Save scaler
    # --------------------------------------------------------
    scaler_path = output_dir / "cicids2017_preprocessor.pkl"

    with scaler_path.open("wb") as f:
        pickle.dump(
            {
                "dataset": "cicids2017",
                "scaler": scaler,
                "feature_columns": feature_cols,
                "image_size": IMAGE_SIZE,
                "scaler_type": "RobustScaler",
                "normalization": "clip_0_1",
                "model_type": "cnn_vae",
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # --------------------------------------------------------
    # 9. Save checkpoint
    # --------------------------------------------------------
    checkpoint_path = (
        output_dir
        / "best_vae_cicids2017.pt"
    )

    torch.save(
        {
            "model_type": "cnn_vae",
            "model_state": trainer.model.state_dict(),
            "threshold": threshold,
            "config": {
                "latent_dim": args.latent,
                "batch_size": args.batch,
                "learning_rate": args.lr,
                "kl_weight": args.kl_weight,
                "epochs": args.epochs,
                "patience": args.patience,
                "image_size": IMAGE_SIZE,
                "representation": (
                    "csv_features_cicids2017"
                ),
                "scaler_type": "RobustScaler",
                "feature_columns": feature_cols,
            },
            "train_losses": trainer.train_losses,
            "val_losses": trainer.val_losses,
            "scaler_path": str(scaler_path),
        },
        checkpoint_path,
    )

    # --------------------------------------------------------
    # 10. Save report
    # --------------------------------------------------------
    report_path = (
        output_dir
        / "eval_result.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 與 run_training.py 相同的三張評估圖；圖表來源為剛寫入的 JSON。
    generate_evaluation_plots_from_json(
        report_path,
        output_dir=output_dir,
    )

    print()
    print("=" * 72)
    print("FINAL TEST")
    print("=" * 72)
    print(
        f"Threshold : {result['threshold']:.8f}"
    )
    print(
        f"Precision : {result['precision']:.4f}"
    )
    print(
        f"Recall    : {result['recall']:.4f}"
    )
    print(
        f"F1        : {result['f1']:.4f}"
    )
    print(
        f"FPR       : {result['fpr']:.4f}"
    )
    print(
        f"ROC-AUC   : {result['roc_auc']:.4f}"
    )
    print(
        f"PR-AUC    : {result['pr_auc']:.4f}"
    )
    print(
        f"TN={result['TN']} "
        f"FP={result['FP']} "
        f"FN={result['FN']} "
        f"TP={result['TP']}"
    )
    print()
    print(f"Model : {checkpoint_path}")
    print(f"Scaler: {scaler_path}")
    print(f"Report: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
