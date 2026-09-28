# -*- coding: utf-8 -*-
"""修正版 CICIDS2017 半監督式 CNN-VAE + CNN-LSTM 訓練/評估。"""
import os, sys, json, csv, argparse
import numpy as np


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
    """把 VAE 測試分數寫入 JSON，供訓練程式本身重建三張圖。

    同時支援兩種 report 結構：
    1. 指標直接位於 report 根層。
    2. 指標位於 report["metrics"]，例如 baseline_report。
    """
    # baseline_report 的格式為:
    # {"model_type": "cnn_vae", "metrics": {"TN": ..., ...}}
    # Hybrid/其他報告有時則直接把 TN/FP/FN/TP 放在 report 根層。
    metrics = report.get("metrics", report)

    required = ["TN", "FP", "FN", "TP", "precision", "recall", "fpr"]
    missing = [k for k in required if k not in metrics]
    if missing:
        raise KeyError(
            "VAE plot data 缺少評估欄位: " + ", ".join(missing)
        )

    report["plot_data"] = {
        "version": 1,
        "model_view": "vae",
        "confusion_metrics": {
            "TN": int(metrics["TN"]),
            "FP": int(metrics["FP"]),
            "FN": int(metrics["FN"]),
            "TP": int(metrics["TP"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "fpr": float(metrics["fpr"]),
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



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def set_seed(seed=42):
    import random, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def split_three_way(X, train_ratio=0.6, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = max(1, int(len(X) * train_ratio))
    n_val = max(1, int(len(X) * val_ratio))
    n_train = min(n_train, len(X) - 2)
    n_val = min(n_val, len(X) - n_train - 1)
    return X[idx[:n_train]], X[idx[n_train:n_train+n_val]], X[idx[n_train+n_val:]]


def image_batch(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3: return X[:, None, :, :]
    if X.ndim == 4: return X
    raise ValueError(f"Unsupported shape: {X.shape}")


def vae_scores(model, X, device, batch=128):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(torch.from_numpy(image_batch(X))), batch_size=batch, shuffle=False)
    model.eval(); out=[]
    with torch.no_grad():
        for (x,) in loader:
            out.append(model.reconstruction_error(x.to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float64) if out else np.empty(0)


def model_outputs(trainer, X, batch=128):
    """取得 VAE score 與 classifier attack probability，不直接決定狀態。"""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(
        TensorDataset(torch.from_numpy(image_batch(X))),
        batch_size=batch,
        shuffle=False,
    )
    trainer.model.eval()
    trainer.classifier.eval()

    vae_s, atk_p = [], []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(trainer.device)
            vae_s.append(trainer.model.reconstruction_error(x).cpu().numpy())
            p = torch.softmax(trainer.classifier(x), dim=1)
            atk_p.append(p[:, 1].cpu().numpy())

    vae_s = np.concatenate(vae_s).astype(np.float64)
    atk_p = np.concatenate(atk_p).astype(np.float64)
    return {
        "vae_score": vae_s,
        "classifier_attack_probability": atk_p,
    }


def apply_cascade(outputs, classifier_threshold, vae_threshold):
    """分層 Hybrid：先判已知攻擊，再用 VAE 找 classifier 不確定的異常。"""
    vae_s = outputs["vae_score"]
    atk_p = outputs["classifier_attack_probability"]

    known = atk_p >= float(classifier_threshold)
    unknown = (~known) & (vae_s > float(vae_threshold))
    hybrid = known | unknown

    status = np.full(len(vae_s), "normal", dtype=object)
    status[unknown] = "unknown_attack"
    status[known] = "known_attack"

    # raw overlap 只做分析，不參與最終決策。
    raw_vae_anomaly = vae_s > float(vae_threshold)
    raw_overlap = known & raw_vae_anomaly

    return {
        **outputs,
        "classifier_label": (atk_p >= float(classifier_threshold)).astype(np.int64),
        "known_attack": known,
        "unknown": unknown,
        "hybrid_anomaly": hybrid,
        "status": status,
        "raw_vae_anomaly": raw_vae_anomaly,
        "raw_overlap": raw_overlap,
    }


def calibrate_cascade_thresholds(
    normal_outputs,
    attack_outputs,
    max_fpr=0.05,
):
    """
    Validation-only threshold tuning。

    第一層：CNN-LSTM classifier threshold。
    第二層：VAE threshold。

    以「Hybrid FPR <= max_fpr」為主要約束，再最大化 F1。
    這避免原本單純以 VAE validation F1 最大化導致 threshold 落在
    59th percentile、進而在 test 產生約 42% VAE FPR 的問題。
    """
    y = np.concatenate([
        np.zeros(len(normal_outputs["vae_score"]), dtype=np.int64),
        np.ones(len(attack_outputs["vae_score"]), dtype=np.int64),
    ])
    vae_score = np.concatenate([
        normal_outputs["vae_score"],
        attack_outputs["vae_score"],
    ])
    cls_prob = np.concatenate([
        normal_outputs["classifier_attack_probability"],
        attack_outputs["classifier_attack_probability"],
    ])

    # classifier 閾值：避免只固定 0.5。
    cls_thresholds = np.round(np.arange(0.50, 0.951, 0.01), 2)

    # VAE 閾值：以正常 validation 的高百分位作 novelty gate。
    vae_percentiles = [95.0, 97.0, 98.0, 99.0, 99.5, 99.7, 99.9]
    vae_thresholds = [
        (pct, float(np.percentile(normal_outputs["vae_score"], pct)))
        for pct in vae_percentiles
    ]

    candidates = []
    for cls_th in cls_thresholds:
        known = cls_prob >= cls_th
        for pct, vae_th in vae_thresholds:
            unknown = (~known) & (vae_score > vae_th)
            pred = (known | unknown).astype(np.int64)
            r = metrics(y, pred)
            candidates.append({
                "classifier_threshold": float(cls_th),
                "vae_percentile": float(pct),
                "vae_threshold": float(vae_th),
                **r,
            })

    feasible = [x for x in candidates if x["fpr"] <= max_fpr]
    pool = feasible if feasible else candidates

    # 最大化 F1；同 F1 優先較低 FPR，再優先較高 Recall。
    best = max(
        pool,
        key=lambda x: (
            x["f1"],
            -x["fpr"],
            x["recall"],
        ),
    )

    best["selection"] = (
        "f1_under_fpr_constraint" if feasible
        else "min_fpr_fallback"
    )
    best["max_allowed_fpr"] = float(max_fpr)
    best["feasible_candidates"] = int(len(feasible))
    return best


def hybrid_outputs(trainer, X, batch=128):
    raw = model_outputs(trainer, X, batch=batch)
    return apply_cascade(
        raw,
        classifier_threshold=trainer.classifier_threshold,
        vae_threshold=trainer.threshold,
    )

def metrics(y, pred, score=None):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y=np.asarray(y); pred=np.asarray(pred)
    tn=int(((y==0)&(pred==0)).sum()); fp=int(((y==0)&(pred==1)).sum()); fn=int(((y==1)&(pred==0)).sum()); tp=int(((y==1)&(pred==1)).sum())
    r={"TN":tn,"FP":fp,"FN":fn,"TP":tp,
       "precision":tp/(tp+fp+1e-12),"recall":tp/(tp+fn+1e-12),
       "accuracy":(tp+tn)/max(1,len(y)),"fpr":fp/(fp+tn+1e-12)}
    r["f1"]=2*r["precision"]*r["recall"]/(r["precision"]+r["recall"]+1e-12)
    if score is not None:
        r["roc_auc"]=float(roc_auc_score(y,score)); r["pr_auc"]=float(average_precision_score(y,score))
    return {k: (float(v) if isinstance(v,(np.floating,float)) else int(v) if isinstance(v,(np.integer,)) else v) for k,v in r.items()}


def train_hybrid(
    trainer,
    Xn_tr,
    Xa_tr,
    Xn_val,
    Xa_val,
    threshold_method="optimal",
    pct=95,
    max_hybrid_fpr=0.05,
):
    import torch
    import torch.nn as nn

    print("\n[1/3] VAE pretraining on normal traffic only")
    opt = torch.optim.Adam(
        trainer.model.parameters(),
        lr=trainer.config["learning_rate"],
    )
    for ep in range(trainer.config["pretrain_epochs"]):
        trainer.model.train()
        losses = []
        for (x,) in trainer._loader(Xn_tr, shuffle=True):
            x = x.to(trainer.device)
            opt.zero_grad(set_to_none=True)
            loss = trainer.model.vae_loss(x)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if ep == 0 or (ep + 1) % 10 == 0:
            print(
                f"  epoch {ep+1}/{trainer.config['pretrain_epochs']} "
                f"loss={np.mean(losses):.6f}"
            )

    print("\n[2/3] CNN-LSTM classifier fine-tuning")
    rng = np.random.default_rng(42)
    n = max(1, int(len(Xa_tr) * trainer.config["attack_ratio"]))
    attack_idx = rng.choice(len(Xa_tr), n, replace=False)
    Xa_lab = Xa_tr[attack_idx]
    normal_idx = rng.choice(
        len(Xn_tr),
        n,
        replace=(n > len(Xn_tr)),
    )
    Xn_lab = Xn_tr[normal_idx]

    X = np.concatenate([Xn_lab, Xa_lab])
    y = np.concatenate([
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=np.int64),
    ])

    opt = torch.optim.Adam(
        trainer.classifier.parameters(),
        lr=trainer.config["learning_rate"],
    )
    ce = nn.CrossEntropyLoss()

    for ep in range(trainer.config["finetune_epochs"]):
        trainer.classifier.train()
        losses = []
        for x, lab in trainer._loader(X, y, shuffle=True):
            x = x.to(trainer.device)
            lab = lab.to(trainer.device)
            opt.zero_grad(set_to_none=True)
            loss = ce(trainer.classifier(x), lab)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if ep == 0 or (ep + 1) % 10 == 0:
            print(
                f"  epoch {ep+1}/{trainer.config['finetune_epochs']} "
                f"loss={np.mean(losses):.6f}"
            )

    print("\n[3/3] Validation threshold calibration")

    n_val = model_outputs(trainer, Xn_val)
    a_val = model_outputs(trainer, Xa_val)

    # 不再以 VAE 單獨找 F1 最大值；改成兩層 cascade 聯合校準。
    if threshold_method == "optimal":
        info = calibrate_cascade_thresholds(
            n_val,
            a_val,
            max_fpr=max_hybrid_fpr,
        )
    else:
        # percentile 模式：classifier 用 0.80，VAE 用正常 validation 的高百分位。
        info = {
            "selection": "fixed_cascade_percentile",
            "classifier_threshold": 0.80,
            "vae_percentile": float(pct),
            "vae_threshold": float(np.percentile(n_val["vae_score"], pct)),
            "max_allowed_fpr": float(max_hybrid_fpr),
        }

    trainer.classifier_threshold = float(info["classifier_threshold"])
    trainer.threshold = float(info["vae_threshold"])

    print(
        f"  classifier_threshold="
        f"{trainer.classifier_threshold:.2f}"
    )
    print(
        f"  vae_threshold="
        f"{trainer.threshold:.9f}"
    )
    print(
        f"  vae_percentile={info['vae_percentile']}"
        f"  selection={info.get('selection')}"
        f"  validation_fpr={info.get('fpr', float('nan')):.4f}"
        f"  validation_f1={info.get('f1', float('nan')):.4f}"
    )

    os.makedirs(trainer.output_dir, exist_ok=True)
    torch.save(
        {
            "model_type": "hybrid_semi",
            "vae_model_type": "cnn_vae",
            "classifier_type": "cnn_lstm_known_attack",
            "vae_state": trainer.model.state_dict(),
            "classifier_state": trainer.classifier.state_dict(),
            "config": trainer.config,
            "threshold": trainer.threshold,
            "classifier_threshold": trainer.classifier_threshold,
            "threshold_info": info,
        },
        os.path.join(
            trainer.output_dir,
            f"semi_{trainer.model_name}.pt",
        ),
    )
    return info

def eval_hybrid(trainer, Xn, Xa):
    n = hybrid_outputs(trainer, Xn)
    a = hybrid_outputs(trainer, Xa)

    y = np.concatenate([
        np.zeros(len(Xn), dtype=np.int64),
        np.ones(len(Xa), dtype=np.int64),
    ])

    # calibrated classifier-only：使用 validation 找到的 classifier threshold。
    cls_pred = np.concatenate([
        n["known_attack"],
        a["known_attack"],
    ]).astype(np.int64)
    cls_score = np.concatenate([
        n["classifier_attack_probability"],
        a["classifier_attack_probability"],
    ])

    # raw VAE gate：用於分析 VAE 本身，但不直接決定 Hybrid。
    vae_pred = np.concatenate([
        n["raw_vae_anomaly"],
        a["raw_vae_anomaly"],
    ]).astype(np.int64)
    vae_score = np.concatenate([
        n["vae_score"],
        a["vae_score"],
    ])

    # cascade final
    hyb_pred = np.concatenate([
        n["hybrid_anomaly"],
        a["hybrid_anomaly"],
    ]).astype(np.int64)

    h = metrics(y, hyb_pred)
    h["vae_threshold"] = float(trainer.threshold)
    h["classifier_threshold"] = float(trainer.classifier_threshold)

    status = np.concatenate([
        n["status"],
        a["status"],
    ])

    raw_overlap = np.concatenate([
        n["raw_overlap"],
        a["raw_overlap"],
    ])

    h["status_counts"] = {
        "normal": int((status == "normal").sum()),
        "known_attack": int((status == "known_attack").sum()),
        "unknown_attack": int((status == "unknown_attack").sum()),
    }
    h["raw_classifier_and_vae_overlap"] = int(raw_overlap.sum())

    # 另外保留 argmax classifier-only，方便與舊報告比較。
    argmax_pred = np.concatenate([
        n["classifier_attack_probability"] >= 0.5,
        a["classifier_attack_probability"] >= 0.5,
    ]).astype(np.int64)
    argmax_metrics = metrics(y, argmax_pred, cls_score)

    return {
        "classifier_only": metrics(y, cls_pred, cls_score),
        "classifier_only_argmax": argmax_metrics,
        "vae_gate": {
            **metrics(y, vae_pred, vae_score),
            "threshold": float(trainer.threshold),
        },
        "hybrid_final": h,
        "normal_outputs": n,
        "attack_outputs": a,
    }

def save_predictions(path, Xn, Xa, e):
    fields=["index","true_label","vae_score","classifier_attack_probability","classifier_label","unknown","hybrid_anomaly","status"]
    with open(path,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); idx=0
        for label,out in [(0,e["normal_outputs"]),(1,e["attack_outputs"])]:
            for i in range(len(out["vae_score"])):
                w.writerow({"index":idx,"true_label":label,"vae_score":float(out["vae_score"][i]),"classifier_attack_probability":float(out["classifier_attack_probability"][i]),"classifier_label":int(out["classifier_label"][i]),"unknown":int(out["unknown"][i]),"hybrid_anomaly":int(out["hybrid_anomaly"][i]),"status":str(out["status"][i])}); idx+=1


def print_metrics(name,r):
    print(f"\n[{name}]")
    for k in ["precision","recall","f1","accuracy","fpr","roc_auc","pr_auc"]:
        if k in r: print(f"  {k:10s}: {r[k]:.4f}")
    print(f"  CM: TN={r['TN']} FP={r['FP']} FN={r['FN']} TP={r['TP']}")


def main():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir",default="data/cicids2017"); p.add_argument("--output",default="output/model_semi_cicids2017")
    p.add_argument("--pretrain-epochs",type=int,default=120); p.add_argument("--finetune-epochs",type=int,default=60); p.add_argument("--batch",type=int,default=64)
    p.add_argument("--latent",type=int,default=32); p.add_argument("--lr",type=float,default=5e-4); p.add_argument("--kl-weight",type=float,default=1e-3)
    p.add_argument("--attack-ratio",type=float,default=0.2); p.add_argument("--threshold-method",choices=["percentile","optimal"],default="optimal"); p.add_argument("--pct",type=float,default=95)
    p.add_argument("--max-normal",type=int,default=60000); p.add_argument("--max-attack",type=int,default=30000); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-hybrid-fpr",type=float,default=0.05,
                    help="Validation 時 Hybrid 最大允許 FPR（optimal 模式）")
    p.add_argument("--compare-unsupervised",action="store_true")
    a=p.parse_args(); set_seed(a.seed)
    if not 0<a.attack_ratio<1: p.error("--attack-ratio 必須介於 0 與 1")
    if not 0<a.pct<100: p.error("--pct 必須介於 0 與 100")
    if not 0<a.max_hybrid_fpr<1: p.error("--max-hybrid-fpr 必須介於 0 與 1")
    import torch
    from dataset_loader import DatasetFactory
    from hybrid_semi_supervised import HybridSemiSupervisedTrainer
    print("\n"+"="*72+"\nCICIDS2017 半監督式 CNN-VAE + CNN-LSTM 修正版\n"+"="*72)
    Xn,Xa,_=DatasetFactory.load("cicids2017",data_dir=a.data_dir,max_normal=a.max_normal,max_attack=a.max_attack)
    Xn_tr,Xn_val,Xn_te=split_three_way(Xn,0.6,0.2,a.seed); Xa_tr,Xa_val,Xa_te=split_three_way(Xa,0.6,0.2,a.seed+1)
    print(f"Normal train/val/test = {len(Xn_tr)}/{len(Xn_val)}/{len(Xn_te)}")
    print(f"Attack train/val/test = {len(Xa_tr)}/{len(Xa_val)}/{len(Xa_te)}")
    cfg={"latent_dim":a.latent,"batch_size":a.batch,"pretrain_epochs":a.pretrain_epochs,"finetune_epochs":a.finetune_epochs,"learning_rate":a.lr,"attack_ratio":a.attack_ratio,"kl_weight":a.kl_weight}
    trainer=HybridSemiSupervisedTrainer(config=cfg,output_dir=a.output,model_name="cicids2017"); trainer.model.kl_weight=a.kl_weight
    th_info=train_hybrid(trainer,Xn_tr,Xa_tr,Xn_val,Xa_val,a.threshold_method,a.pct,a.max_hybrid_fpr)
    e=eval_hybrid(trainer,Xn_te,Xa_te)
    print_metrics("Classifier-only (calibrated)",e["classifier_only"]);
    print_metrics("Classifier-only (argmax=0.50)",e["classifier_only_argmax"]);
    print_metrics("VAE gate (raw analysis)",e["vae_gate"]);
    print_metrics("Hybrid final (cascade)",e["hybrid_final"])
    print("\nHybrid status counts:",e["hybrid_final"]["status_counts"],
          "raw classifier+VAE overlap=",e["hybrid_final"]["raw_classifier_and_vae_overlap"])
    os.makedirs(a.output,exist_ok=True)

    report={"dataset":"cicids2017","model_type":"hybrid_semi","vae_model_type":"cnn_vae","classifier_type":"cnn_lstm_known_attack","threshold":float(trainer.threshold),"classifier_threshold":float(trainer.classifier_threshold),"threshold_info":th_info,"config":cfg,"split":{"train_ratio":0.6,"val_ratio":0.2,"test_ratio":0.2,"normal_train":len(Xn_tr),"normal_val":len(Xn_val),"normal_test":len(Xn_te),"attack_train":len(Xa_tr),"attack_val":len(Xa_val),"attack_test":len(Xa_te),"labelled_attack_ratio":a.attack_ratio,"seed":a.seed},"classifier_only":e["classifier_only"],"vae_gate":e["vae_gate"],"hybrid_final":e["hybrid_final"]}

    # 將 Hybrid 需要的連續 score 寫入 JSON。
    # ROC/PR：CNN-LSTM attack probability；
    # Score distribution：VAE novelty / reconstruction score；
    # Confusion matrix：Hybrid final 最終判定。
    add_hybrid_plot_data(
        report,
        normal_vae_scores=e["normal_outputs"]["vae_score"],
        attack_vae_scores=e["attack_outputs"]["vae_score"],
        normal_classifier_scores=e["normal_outputs"]["classifier_attack_probability"],
        attack_classifier_scores=e["attack_outputs"]["classifier_attack_probability"],
        vae_threshold=trainer.threshold,
        classifier_threshold=trainer.classifier_threshold,
    )

    report_path = os.path.join(a.output,"semi_evaluation_report.json")
    with open(report_path,"w",encoding="utf-8") as f: json.dump(report,f,indent=4,ensure_ascii=False)
    save_predictions(os.path.join(a.output,"test_predictions.csv"),Xn_te,Xa_te,e)
    with open(os.path.join(a.output,"hybrid_confusion_matrix.json"),"w",encoding="utf-8") as f: json.dump({k:e["hybrid_final"][k] for k in ["TN","FP","FN","TP"]},f,indent=4)

    # 與 run_training.py 相同的三張圖；這裡刻意重新從 JSON 讀取，
    # 確保圖表與 eval report 使用的是同一份資料。
    generate_evaluation_plots_from_json(report_path, output_dir=a.output)

    if a.compare_unsupervised:
        # 僅使用 VAE baseline；不再呼叫標準 CNN Autoencoder Trainer。
        from variational_autoencoder import VAETrainer, vae_threshold
        from torch.utils.data import DataLoader,TensorDataset
        out=os.path.join(a.output,"unsupervised_vae_baseline"); os.makedirs(out,exist_ok=True)
        vt=VAETrainer(config={"latent_dim":a.latent,"batch_size":a.batch,"epochs":a.pretrain_epochs,"learning_rate":a.lr,"patience":20,"threshold_percentile":a.pct,"kl_weight":a.kl_weight},output_dir=out,model_name="cicids2017_baseline")
        vt.train_loader=DataLoader(TensorDataset(torch.from_numpy(image_batch(Xn_tr))),batch_size=a.batch,shuffle=True); vt.val_loader=DataLoader(TensorDataset(torch.from_numpy(image_batch(Xn_val))),batch_size=a.batch,shuffle=False); vt.data=Xn_tr; vt.train()
        vt.threshold=float(vae_threshold(vae_scores(vt.model,Xn_val,vt.device),a.pct)); vt._save_checkpoint()

        normal_base_scores=vae_scores(vt.model,Xn_te,vt.device)
        attack_base_scores=vae_scores(vt.model,Xa_te,vt.device)
        s=np.concatenate([normal_base_scores,attack_base_scores]); y=np.concatenate([np.zeros(len(Xn_te),dtype=np.int64),np.ones(len(Xa_te),dtype=np.int64)])
        base=metrics(y,(s>vt.threshold).astype(np.int64),s); base["threshold"]=float(vt.threshold)
        baseline_report={"model_type":"cnn_vae","metrics":base}
        add_vae_plot_data(baseline_report,normal_base_scores,attack_base_scores,vt.threshold)
        baseline_report_path=os.path.join(out,"evaluation_report.json")
        with open(baseline_report_path,"w",encoding="utf-8") as f: json.dump(baseline_report,f,indent=4,ensure_ascii=False)
        generate_evaluation_plots_from_json(baseline_report_path,output_dir=out)

        report["unsupervised_vae_baseline"]=baseline_report
        with open(report_path,"w",encoding="utf-8") as f: json.dump(report,f,indent=4,ensure_ascii=False)
        print_metrics("Unsupervised CNN-VAE baseline",base)
    print("\n完成。輸出：",os.path.abspath(a.output))


if __name__=="__main__": main()
