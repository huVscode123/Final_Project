# -*- coding: utf-8 -*-

"""修正版 CICIDS2017 半監督式 CNN-VAE + CNN-LSTM 訓練/評估。"""

import os, sys, json, csv, argparse, pickle

import numpy as np

from threshold_analysis_common import (

    select_hybrid_thresholds, analyze_hybrid_thresholds, write_analysis, split_four_way,

)

METHOD_LABELS = {
    "percentile": "Percentile 95",
    "f1_max": "F1 最大化",
    "youden": "Youden",
}





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

    r["youden_j"]=r["recall"]-r["fpr"]   # Youden's Index = TPR - FPR

    if score is not None:

        r["roc_auc"]=float(roc_auc_score(y,score)); r["pr_auc"]=float(average_precision_score(y,score))

    return {k: (float(v) if isinstance(v,(np.floating,float)) else int(v) if isinstance(v,(np.integer,)) else v) for k,v in r.items()}





def train_hybrid(

    trainer,

    Xn_tr,

    Xa_tr,

    Xn_val,

    Xa_val,

    threshold_method="f1_max",

    pct=95,

    youden_max_fpr=None,

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



    print("\n[3/3] Validation threshold calibration (same classifier policy)")



    n_val = model_outputs(trainer, Xn_val)

    a_val = model_outputs(trainer, Xa_val)



    if threshold_method not in ("percentile", "f1_max", "youden"):

        raise ValueError("threshold_method 必須為 percentile / f1_max / youden")



    # 與 MTA Hybrid 統一：分類器預測門檻固定為 0.5，三種方法只校準 VAE gate。

    # f1_max / youden 的目標使用 Hybrid 最終判定，而非單獨 VAE gate 指標。

    classifier_threshold = 0.5

    thresholds, method_infos = select_hybrid_thresholds(

        cal_normal_vae=n_val["vae_score"],

        cal_normal_prob=n_val["classifier_attack_probability"],

        cal_attack_vae=a_val["vae_score"],

        cal_attack_prob=a_val["classifier_attack_probability"],

        percentile=pct,

        classifier_threshold=classifier_threshold,

        youden_max_fpr=youden_max_fpr,

    )

    if threshold_method not in thresholds:

        raise ValueError(f"無法產生指定閾值方法：{threshold_method}")



    info = {

        "selected_method": threshold_method,

        "threshold": float(thresholds[threshold_method]),

        "classifier_threshold_fixed": float(classifier_threshold),

        "thresholds": {k: float(v) for k, v in thresholds.items()},

        "methods": method_infos,

        "youden_max_fpr": youden_max_fpr,

        "selection_policy": "只調整 VAE gate；classifier threshold 固定 0.5；指標以 Hybrid final 計算",

    }

    trainer.threshold_method_candidates = thresholds

    trainer.threshold_method_infos = method_infos

    trainer.threshold_calibration_outputs = (n_val, a_val)

    trainer.classifier_threshold = float(classifier_threshold)

    trainer.threshold = float(thresholds[threshold_method])



    print(f"  classifier_threshold (fixed) = {trainer.classifier_threshold:.2f}")

    for name in ("percentile", "f1_max", "youden"):

        if name in thresholds:

            print(f"  {METHOD_LABELS.get(name, name):<16} VAE threshold = {thresholds[name]:.9g}")

    print(f"  selected method = {threshold_method}")



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

    for k in ["precision","recall","f1","accuracy","fpr","youden_j","roc_auc","pr_auc"]:

        if k in r: print(f"  {k:10s}: {r[k]:.4f}")

    print(f"  CM: TN={r['TN']} FP={r['FP']} FN={r['FN']} TP={r['TP']}")





def main():

    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--data-dir",default="data/cicids2017"); p.add_argument("--output",default="output/model_semi_cicids2017")

    p.add_argument("--pretrain-epochs",type=int,default=120); p.add_argument("--finetune-epochs",type=int,default=60); p.add_argument("--batch",type=int,default=64)

    p.add_argument("--latent",type=int,default=32); p.add_argument("--lr",type=float,default=5e-4); p.add_argument("--kl-weight",type=float,default=1e-3)

    p.add_argument("--attack-ratio",type=float,default=0.2); p.add_argument("--threshold-method",choices=["percentile","f1_max","youden"],default="f1_max"); p.add_argument("--pct",type=float,default=95)

    p.add_argument("--max-normal",type=int,default=60000); p.add_argument("--max-attack",type=int,default=30000); p.add_argument("--seed",type=int,default=42)

    p.add_argument("--train-ratio", type=float, default=0.60,

                    help="與非監督 CICIDS2017 VAE 相同的訓練比例")

    p.add_argument("--val-ratio", type=float, default=0.20,

                    help="總驗證池比例，平均分為 early-stop 與閾值 calibration")

    p.add_argument("--youden-max-fpr", type=float, default=None,

                    help="選用 Youden 時的可選 FPR 上限（例如 0.05）；不設定則不限制")

    p.add_argument("--compare-unsupervised",action="store_true")

    a=p.parse_args(); set_seed(a.seed)

    if not 0<a.attack_ratio<1: p.error("--attack-ratio 必須介於 0 與 1")

    if not 0<a.pct<100: p.error("--pct 必須介於 0 與 100")

    if not 0<a.train_ratio<1 or not 0<a.val_ratio<1 or a.train_ratio+a.val_ratio>=1:

        p.error("必須符合 train-ratio > 0、val-ratio > 0、train-ratio + val-ratio < 1")

    if a.youden_max_fpr is not None and not 0<a.youden_max_fpr<1: p.error("--youden-max-fpr 必須介於 0 與 1")

    import torch



    # 新增：匯入半監督式 Hybrid 訓練器

    from hybrid_semi_supervised import HybridSemiSupervisedTrainer

    # 與 run_training_cicids2017_vae_fixed.py 共用資料載入、特徵欄位、RobustScaler 與切分。

    # 只要兩支程式使用相同 data-dir / max-normal / max-attack / seed / ratios，

    # CICIDS 的 train / calibration / test 樣本索引和輸入影像就會一致。

    from run_training_cicids2017_vae_fixed import (

        load_dataset as load_cicids_raw,

        fit_scaler_and_transform_fourway,

        to_images,

    )



    print("\n"+"="*72+"\nCICIDS2017 半監督式 CNN-VAE + CNN-LSTM 修正版\n"+"="*72)

    Xn_raw, Xa_raw, feature_cols = load_cicids_raw(

        a.data_dir, a.max_normal, a.max_attack, a.seed

    )

    Xn_tr_raw, Xn_early_raw, Xn_val_raw, Xn_te_raw = split_four_way(

        Xn_raw, a.train_ratio, a.val_ratio, a.seed

    )

    Xa_tr_raw, Xa_early_raw, Xa_val_raw, Xa_te_raw = split_four_way(

        Xa_raw, a.train_ratio, a.val_ratio, a.seed + 1

    )



    (Xn_tr_scaled, Xn_early_scaled, Xn_val_scaled, Xn_te_scaled,

     Xa_tr_scaled, Xa_val_scaled, Xa_te_scaled, scaler) = fit_scaler_and_transform_fourway(

        Xn_tr_raw, Xn_early_raw, Xn_val_raw, Xn_te_raw,

        Xa_tr_raw, Xa_val_raw, Xa_te_raw,

    )



    # 同一套 feature -> 32x32 image 轉換，對齊非監督 VAE 的輸入表示法。

    Xn_tr = to_images(Xn_tr_scaled)

    Xn_early = to_images(Xn_early_scaled)  # 保留但不參與 calibration

    Xn_val = to_images(Xn_val_scaled)      # threshold calibration only

    Xn_te = to_images(Xn_te_scaled)

    Xa_tr = to_images(Xa_tr_scaled)

    Xa_val = to_images(Xa_val_scaled)

    Xa_te = to_images(Xa_te_scaled)



    print(f"Normal train/early/cal/test = {len(Xn_tr)}/{len(Xn_early)}/{len(Xn_val)}/{len(Xn_te)}")

    print(f"Attack train/unused/cal/test = {len(Xa_tr)}/{len(Xa_early_raw)}/{len(Xa_val)}/{len(Xa_te)}")



    os.makedirs(a.output, exist_ok=True)

    scaler_path = os.path.join(a.output, "cicids2017_preprocessor.pkl")

    with open(scaler_path, "wb") as f:

        pickle.dump({

            "dataset": "cicids2017",

            "scaler": scaler,

            "feature_columns": feature_cols,

            "image_size": 32,

            "scaler_type": "RobustScaler",

            "normalization": "clip_0_1",

            "model_type": "hybrid_semi",

        }, f, protocol=pickle.HIGHEST_PROTOCOL)



    cfg={

        "latent_dim":a.latent,"batch_size":a.batch,

        "pretrain_epochs":a.pretrain_epochs,"finetune_epochs":a.finetune_epochs,

        "learning_rate":a.lr,"attack_ratio":a.attack_ratio,"kl_weight":a.kl_weight,

        "representation":"csv_features_cicids2017","scaler_path":scaler_path,

        "scaler_type":"RobustScaler","normalization":"clip_0_1",

        "feature_columns":[str(c) for c in feature_cols],"image_size":32,

        "train_ratio":a.train_ratio,"validation_pool_ratio":a.val_ratio,"seed":a.seed,

    }

    trainer=HybridSemiSupervisedTrainer(config=cfg,output_dir=a.output,model_name="cicids2017"); trainer.model.kl_weight=a.kl_weight

    th_info=train_hybrid(

        trainer, Xn_tr, Xa_tr, Xn_val, Xa_val,

        a.threshold_method, a.pct, a.youden_max_fpr,

    )

    e=eval_hybrid(trainer,Xn_te,Xa_te)



    # 對三種方法在同一份測試輸出上做完整 Hybrid 最終判定評估。

    cal_n_outputs, cal_a_outputs = trainer.threshold_calibration_outputs

    threshold_analysis, threshold_score_cache = analyze_hybrid_thresholds(

        model_key="semi_cicids2017",

        cal_normal_vae=cal_n_outputs["vae_score"],

        cal_normal_prob=cal_n_outputs["classifier_attack_probability"],

        cal_attack_vae=cal_a_outputs["vae_score"],

        cal_attack_prob=cal_a_outputs["classifier_attack_probability"],

        test_normal_vae=e["normal_outputs"]["vae_score"],

        test_normal_prob=e["normal_outputs"]["classifier_attack_probability"],

        test_attack_vae=e["attack_outputs"]["vae_score"],

        test_attack_prob=e["attack_outputs"]["classifier_attack_probability"],

        percentile=a.pct,

        classifier_threshold=trainer.classifier_threshold,

        youden_max_fpr=a.youden_max_fpr,

    )

    threshold_analysis_path, threshold_scores_path = write_analysis(

        threshold_analysis, threshold_score_cache, a.output,

        stem="threshold_analysis",

    )

    print("\n[Threshold comparison — full Hybrid, same test set]")

    print(f"{'method':<16}{'threshold':>14}{'precision':>12}{'recall':>10}{'F1':>9}{'FPR':>9}{'J':>9}")

    for method in ("percentile", "f1_max", "youden"):

        row = threshold_analysis["methods"][method]

        m = row["test"]

        print(f"{METHOD_LABELS.get(method, method):<16}{row['threshold']:>14.7g}{m['precision']:>12.4f}{m['recall']:>10.4f}{m['f1']:>9.4f}{m['fpr']:>9.4f}{m['youden_j']:>9.4f}")

    print(f"[Saved] {threshold_analysis_path}")

    print(f"[Saved] {threshold_scores_path}")

    print_metrics("Classifier-only (calibrated)",e["classifier_only"]);

    print_metrics("Classifier-only (argmax=0.50)",e["classifier_only_argmax"]);

    print_metrics("VAE gate (raw analysis)",e["vae_gate"]);

    print_metrics("Hybrid final (cascade)",e["hybrid_final"])

    print("\nHybrid status counts:",e["hybrid_final"]["status_counts"],

          "raw classifier+VAE overlap=",e["hybrid_final"]["raw_classifier_and_vae_overlap"])

    os.makedirs(a.output,exist_ok=True)



    report={"dataset":"cicids2017","model_type":"hybrid_semi","vae_model_type":"cnn_vae","classifier_type":"cnn_lstm_known_attack","threshold":float(trainer.threshold),"classifier_threshold":float(trainer.classifier_threshold),"threshold_info":th_info,"config":cfg,"preprocessor_path":scaler_path,"split":{"train_ratio":a.train_ratio,"validation_pool_ratio":a.val_ratio,"train_normal":len(Xn_tr),"early_stop_normal_unused":len(Xn_early),"cal_normal":len(Xn_val),"test_normal":len(Xn_te),"train_attack":len(Xa_tr),"unused_attack":len(Xa_early_raw),"cal_attack":len(Xa_val),"test_attack":len(Xa_te),"labelled_attack_ratio":a.attack_ratio,"seed":a.seed,"shared_loader_scaler_and_indices_with_unsupervised_vae":True},"classifier_only":e["classifier_only"],"vae_gate":e["vae_gate"],"hybrid_final":e["hybrid_final"],"threshold_analysis_path":threshold_analysis_path,"threshold_methods_test":{k:v["test"] for k,v in threshold_analysis["methods"].items()},"threshold_auc":threshold_analysis.get("test_auc",{})}



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
