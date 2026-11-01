# -*- coding: utf-8 -*-
"""Re-evaluate CICIDS2017 Hybrid ROC-AUC / PR-AUC from saved test predictions.

This script does NOT train or modify the model. It reads the held-out test
predictions written by run_semi_supervised_cicids2017_hybrid_v2.py.

It reports three clearly separated views:
1) CNN-LSTM classifier-only: attack probability as the ranking score.
2) VAE-only: VAE anomaly score as the ranking score.
3) Full Hybrid cascade ranking: classifier-positive samples are ranked above
   all classifier-negative samples; among classifier-negative samples, VAE
   anomaly score determines the order. This ranking corresponds exactly to
   sweeping the VAE gate threshold while keeping the classifier threshold fixed.

The Hybrid cascade AUC is therefore conditional on the fixed classifier
threshold recorded in semi_evaluation_report.json (normally 0.5). It is not a
new model score used for inference; it is a ranking representation of the
existing two-stage decision policy for ROC/PR evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def get_thresholds(report_path: Path) -> Tuple[float, float]:
    """Read classifier and VAE thresholds from the saved evaluation report."""
    if not report_path.is_file():
        raise FileNotFoundError(f"找不到評估報告：{report_path}")

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    classifier_threshold = report.get("classifier_threshold")
    vae_threshold = report.get("threshold")

    # Also support the thresholds stored in plot_data for compatibility.
    nested = report.get("plot_data", {}).get("thresholds", {})
    if classifier_threshold is None:
        classifier_threshold = nested.get("classifier_threshold")
    if vae_threshold is None:
        vae_threshold = nested.get("vae_threshold")

    if classifier_threshold is None:
        classifier_threshold = report.get("classifier_threshold_fixed")
    if vae_threshold is None:
        raise KeyError("報告中找不到 VAE threshold，請檢查 semi_evaluation_report.json")
    if classifier_threshold is None:
        raise KeyError("報告中找不到 classifier_threshold，不能自行假設 Hybrid 門檻")

    return float(classifier_threshold), float(vae_threshold)


def metrics_from_score(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    """Calculate ROC-AUC, Average Precision and trapezoidal PR-AUC."""
    fpr, tpr, _ = roc_curve(y, score)
    precision, recall, _ = precision_recall_curve(y, score)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        # sklearn Average Precision (AP); this is what the project's
        # threshold-analysis helper stores in its pr_auc field.
        "average_precision": float(average_precision_score(y, score)),
        # Geometric trapezoidal area under the PR curve; may differ from AP.
        "pr_auc_trapezoidal": float(auc(recall, precision)),
    }


def operating_point_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = float(precision_score(y, pred, zero_division=0))
    recall = float(recall_score(y, pred, zero_division=0))
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "precision": precision,
        "recall": recall,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "accuracy": float(accuracy_score(y, pred)),
    }


def make_hybrid_cascade_rank(
    classifier_probability: np.ndarray,
    vae_score: np.ndarray,
    classifier_threshold: float,
) -> np.ndarray:
    """Encode the cascade's ranking while preserving its threshold-sweep order.

    With classifier threshold fixed, any classifier-positive sample is always
    labelled attack regardless of the VAE gate threshold. These samples must
    therefore rank above every classifier-negative sample. The remaining
    samples are ranked by their VAE anomaly scores.
    """
    known_attack = classifier_probability >= classifier_threshold
    rank_score = vae_score.astype(np.float64, copy=True)
    if np.any(known_attack):
        highest = float(np.max(rank_score))
        top_rank = np.nextafter(highest, np.inf)
        # Guard against the rare case where nextafter cannot produce a finite
        # larger number (for example, highest is the largest float64 value).
        if not np.isfinite(top_rank) or top_rank <= highest:
            top_rank = highest + max(abs(highest), 1.0)
        rank_score[known_attack] = top_rank
    return rank_score


def main() -> int:
    parser = argparse.ArgumentParser(
        description="由既有 CICIDS2017 Hybrid test_predictions.csv 計算 ROC-AUC / PR-AUC"
    )
    parser.add_argument(
        "--predictions",
        default="output/threshold_study/semi_cicids2017/test_predictions.csv",
        help="Hybrid 測試集預測 CSV",
    )
    parser.add_argument(
        "--report",
        default="output/threshold_study/semi_cicids2017/semi_evaluation_report.json",
        help="Hybrid 評估 JSON（讀取兩個閾值）",
    )
    parser.add_argument(
        "--output-dir",
        default="output/threshold_study/semi_cicids2017/auc_analysis",
        help="AUC 結果輸出資料夾",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    report_path = Path(args.report)
    out_dir = Path(args.output_dir)

    if not pred_path.is_file():
        raise FileNotFoundError(
            f"找不到測試預測 CSV：{pred_path}\n"
            "請先確認是否有執行過 Hybrid 訓練／評估程式，且輸出目錄正確。"
        )

    classifier_threshold, vae_threshold = get_thresholds(report_path)
    df = pd.read_csv(pred_path)

    required = [
        "true_label",
        "vae_score",
        "classifier_attack_probability",
        "hybrid_anomaly",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            "test_predictions.csv 缺少必要欄位：" + ", ".join(missing)
        )

    # Convert to numeric, then use one common finite mask so all three views
    # evaluate exactly the same held-out observations.
    numeric = df[required].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    valid = np.isfinite(values).all(axis=1)
    dropped = int((~valid).sum())
    numeric = numeric.loc[valid].reset_index(drop=True)

    y = numeric["true_label"].to_numpy(dtype=np.int64)
    vae_score = numeric["vae_score"].to_numpy(dtype=np.float64)
    classifier_probability = numeric["classifier_attack_probability"].to_numpy(dtype=np.float64)
    saved_hybrid_pred = numeric["hybrid_anomaly"].to_numpy(dtype=np.int64)

    unique_labels = set(np.unique(y).tolist())
    if unique_labels != {0, 1}:
        raise ValueError(f"測試標籤必須同時包含 0 和 1，目前為：{sorted(unique_labels)}")
    if not set(np.unique(saved_hybrid_pred).tolist()).issubset({0, 1}):
        raise ValueError("hybrid_anomaly 欄位不是 0/1 二元預測")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("true_label 欄位只能包含 0（正常）與 1（攻擊）")

    # Rebuild the original cascade decision and check it against saved labels.
    # The policy is: known classifier attack OR (classifier-negative AND VAE gate).
    known_attack = classifier_probability >= classifier_threshold
    expected_hybrid_pred = (
        known_attack
        | ((~known_attack) & (vae_score > vae_threshold))
    ).astype(np.int64)
    saved_prediction_agreement = float(np.mean(expected_hybrid_pred == saved_hybrid_pred))

    # A ranking representation of the complete cascade, with classifier
    # threshold fixed and the VAE gate threshold swept over its possible range.
    hybrid_rank_score = make_hybrid_cascade_rank(
        classifier_probability,
        vae_score,
        classifier_threshold,
    )

    results = {
        "dataset": "CICIDS2017",
        "evaluation_split": "saved held-out test_predictions.csv",
        "test_samples": int(len(y)),
        "normal_samples": int((y == 0).sum()),
        "attack_samples": int((y == 1).sum()),
        "dropped_non_finite_rows": dropped,
        "classifier_threshold_fixed": classifier_threshold,
        "vae_threshold_selected_on_calibration": vae_threshold,
        "saved_hybrid_prediction_agreement": saved_prediction_agreement,
        "metrics": {
            "classifier_only": {
                "score_definition": "CNN-LSTM attack probability; higher means more likely attack",
                **metrics_from_score(y, classifier_probability),
            },
            "vae_only": {
                "score_definition": "VAE anomaly score; higher means more anomalous",
                **metrics_from_score(y, vae_score),
            },
            "hybrid_cascade": {
                "score_definition": (
                    "Cascade ranking: classifier-positive samples rank above all others; "
                    "classifier-negative samples are ranked by VAE anomaly score. "
                    "Equivalent to sweeping the VAE gate threshold while keeping "
                    "the classifier threshold fixed."
                ),
                **metrics_from_score(y, hybrid_rank_score),
            },
            "hybrid_saved_operating_point": {
                "score_definition": "Original saved 0/1 Hybrid cascade prediction at calibrated thresholds",
                **operating_point_metrics(y, saved_hybrid_pred),
            },
        },
        "interpretation_note": (
            "Classifier-only AUC and VAE-only AUC evaluate individual branches. "
            "hybrid_cascade AUC evaluates a ranking of the complete cascade under "
            "a fixed classifier threshold and a swept VAE gate threshold. "
            "Do not call classifier-only AUC the AUC of the complete Hybrid."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "hybrid_auc_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save per-sample scores and the cascade ranking for audit/reproducibility.
    audit = pd.DataFrame({
        "true_label": y,
        "classifier_attack_probability": classifier_probability,
        "vae_score": vae_score,
        "saved_hybrid_prediction": saved_hybrid_pred,
        "reconstructed_hybrid_prediction": expected_hybrid_pred,
        "hybrid_cascade_rank_score": hybrid_rank_score,
    })
    audit.to_csv(out_dir / "hybrid_auc_test_scores.csv", index=False, encoding="utf-8-sig")

    # Produce ROC and PR curve data/figures for the three continuous rankings.
    import matplotlib.pyplot as plt

    score_sets = {
        "Classifier-only": classifier_probability,
        "VAE-only": vae_score,
        "Hybrid cascade": hybrid_rank_score,
    }
    curve_rows = []
    roc_fig, roc_ax = plt.subplots(figsize=(7, 6))
    pr_fig, pr_ax = plt.subplots(figsize=(7, 6))

    for name, score in score_sets.items():
        fpr, tpr, roc_thresholds = roc_curve(y, score)
        precision, recall, pr_thresholds = precision_recall_curve(y, score)
        roc_value = float(roc_auc_score(y, score))
        ap_value = float(average_precision_score(y, score))
        trap_pr = float(auc(recall, precision))
        roc_ax.plot(fpr, tpr, label=f"{name} (AUC={roc_value:.4f})")
        pr_ax.plot(recall, precision, label=f"{name} (AP={ap_value:.4f})")
        for i in range(len(fpr)):
            curve_rows.append({
                "view": name,
                "curve": "ROC",
                "x_fpr": float(fpr[i]),
                "y_tpr": float(tpr[i]),
                "threshold": float(roc_thresholds[i]) if i < len(roc_thresholds) else None,
            })
        for i in range(len(recall)):
            curve_rows.append({
                "view": name,
                "curve": "PR",
                "x_recall": float(recall[i]),
                "y_precision": float(precision[i]),
                "threshold": float(pr_thresholds[i]) if i < len(pr_thresholds) else None,
            })
        print(
            f"{name:20s} ROC-AUC={roc_value:.6f}  "
            f"AP={ap_value:.6f}  PR-AUC(trapezoidal)={trap_pr:.6f}"
        )

    roc_ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_title("CICIDS2017 Hybrid: ROC Curves")
    roc_ax.grid(True, alpha=0.3)
    roc_ax.legend(loc="lower right")
    roc_fig.tight_layout()
    roc_fig.savefig(out_dir / "hybrid_roc_curves.png", dpi=180)
    plt.close(roc_fig)

    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("CICIDS2017 Hybrid: Precision-Recall Curves")
    pr_ax.grid(True, alpha=0.3)
    pr_ax.legend(loc="lower left")
    pr_fig.tight_layout()
    pr_fig.savefig(out_dir / "hybrid_pr_curves.png", dpi=180)
    plt.close(pr_fig)

    pd.DataFrame(curve_rows).to_csv(
        out_dir / "hybrid_roc_pr_curve_points.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Hybrid AUC 評估完成 ===")
    print(f"Test samples: {len(y):,} (normal={(y==0).sum():,}, attack={(y==1).sum():,})")
    print(f"Classifier threshold: {classifier_threshold:.8g}")
    print(f"VAE threshold:        {vae_threshold:.8g}")
    print(f"Saved cascade prediction agreement: {saved_prediction_agreement:.4%}")
    print(f"Results: {json_path}")
    print(f"Curves:  {out_dir / 'hybrid_roc_curves.png'}")
    print(f"         {out_dir / 'hybrid_pr_curves.png'}")

    if saved_prediction_agreement < 0.999:
        print(
            "\n警告：依 JSON 閾值重建的級聯預測與 CSV 保存的 hybrid_anomaly 不完全一致。"
            "請先核對 test_predictions.csv 與 semi_evaluation_report.json 是否來自同一次執行，"
            "不要直接將這次結果寫入論文。",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
