# -*- coding: utf-8 -*-
"""彙整四種模型的 percentile / optimal / youden 閾值分析 JSON。

範例：
python core/aggregate_threshold_analysis.py \
  --cicids-vae output/model_cicids2017/threshold_analysis.json \
  --cicids-semi output/model_semi_cicids2017/threshold_analysis.json \
  --mta-vae output/model_mta_malware/vae/threshold_analysis.json \
  --mta-semi output/model_mta_malware/semi/threshold_analysis.json \
  --output output/threshold_comparison_all_models
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

METHODS = ("percentile", "optimal", "youden")
METRIC_COLUMNS = (
    "precision", "recall", "f1", "fpr", "youden_j", "accuracy",
    "TN", "FP", "FN", "TP", "n_normal", "n_attack",
)


def load_report(label: str, path_text: str):
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"[{label}] 找不到閾值分析報告：{path}")
    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    if "methods" not in report:
        raise ValueError(f"[{label}] JSON 缺少 methods 欄位：{path}")
    return path, report


def main():
    ap = argparse.ArgumentParser(
        description="把四個模型各自的閾值分析報告合併成 CSV / JSON。"
    )
    ap.add_argument("--cicids-vae", required=True, help="CICIDS2017 非監督 VAE threshold_analysis.json")
    ap.add_argument("--cicids-semi", required=True, help="CICIDS2017 半監督 Hybrid threshold_analysis.json")
    ap.add_argument("--mta-vae", required=True, help="MTA 非監督 VAE threshold_analysis.json")
    ap.add_argument("--mta-semi", required=True, help="MTA 半監督 Hybrid threshold_analysis.json")
    ap.add_argument("--output", default="output/threshold_comparison_all_models")
    args = ap.parse_args()

    files = [
        ("unsup_cicids2017", args.cicids_vae),
        ("semi_cicids2017", args.cicids_semi),
        ("mta_malware", args.mta_vae),
        ("semi_mta_malware", args.mta_semi),
    ]
    reports = []
    for key, path in files:
        resolved, report = load_report(key, path)
        reports.append((key, resolved, report))

    rows = []
    for model_key, report_path, report in reports:
        dataset = "cicids2017" if "cicids2017" in model_key else "mta_malware"
        model_type = "hybrid_semi" if model_key.startswith("semi_") else "cnn_vae"
        split = report.get("split", {})
        auc = report.get("test_auc", report.get("auc", {})) or {}
        # MTA VAE legacy report uses auc.test_roc_auc; common utility uses test_auc.roc_auc.
        roc_auc = auc.get("roc_auc", auc.get("test_roc_auc", auc.get("classifier_roc_auc")))
        pr_auc = auc.get("pr_auc", auc.get("test_pr_auc", auc.get("classifier_pr_auc")))
        vae_roc_auc = auc.get("vae_score_roc_auc")
        vae_pr_auc = auc.get("vae_score_pr_auc")

        for method in METHODS:
            item = report["methods"].get(method)
            if not item or item.get("available") is False:
                rows.append({
                    "model_key": model_key, "dataset": dataset,
                    "model_type": model_type, "threshold_method": method,
                    "available": False,
                    "unavailable_reason": (item or {}).get("reason", "報告中沒有此方法"),
                    "source_report": str(report_path),
                })
                continue
            test = item.get("test") or {}
            row = {
                "model_key": model_key,
                "dataset": dataset,
                "model_type": model_type,
                "threshold_method": method,
                "available": True,
                "threshold": item.get("threshold"),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "vae_score_roc_auc": vae_roc_auc,
                "vae_score_pr_auc": vae_pr_auc,
                "classifier_threshold": item.get("classifier_threshold", report.get("classifier_threshold_fixed")),
                "cal_normal": split.get("cal_normal"),
                "cal_attack": split.get("cal_attack"),
                "test_normal": split.get("test_normal"),
                "test_attack": split.get("test_attack"),
                "source_report": str(report_path),
            }
            for k in METRIC_COLUMNS:
                row[k] = test.get(k)
            rows.append(row)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "threshold_comparison_all_models.json"
    csv_path = out_dir / "threshold_comparison_all_models.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "models": [k for k, _, _ in reports],
            "methods": list(METHODS),
            "rows": rows,
            "note": (
                "每個模型內的三種方法應使用相同 test split；不同資料集（CICIDS2017 vs MTA）"
                "的絕對指標不可直接視為同一測試分布上的排名。半監督模型指標必須是 Hybrid final。"
            ),
        }, f, ensure_ascii=False, indent=2)

    columns = [
        "model_key", "dataset", "model_type", "threshold_method", "available",
        "threshold", "classifier_threshold", "precision", "recall", "f1", "fpr",
        "youden_j", "accuracy", "TN", "FP", "FN", "TP",
        "roc_auc", "pr_auc", "vae_score_roc_auc", "vae_score_pr_auc",
        "cal_normal", "cal_attack", "test_normal", "test_attack",
        "unavailable_reason", "source_report",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"共整理 {len(rows)} 列（每模型 3 個方法，若缺少方法會明確標示）。")
    print(f"JSON: {json_path.resolve()}")
    print(f"CSV : {csv_path.resolve()}")


if __name__ == "__main__":
    main()
