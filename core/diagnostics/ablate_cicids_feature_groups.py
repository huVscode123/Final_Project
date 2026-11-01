#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic counterfactual test for CICIDS2017 CSV-vs-PCAP feature skew.

Reuses the exact scaler -> clipping -> features_to_image -> model score pipeline from
compare_cicids_csv_vs_pcap_inference.py. For each feature group, it replaces only that
group in sampled PCAP rows with values from shuffled BENIGN training-CSV rows, then
re-scores. This is diagnostic only; it does NOT create a valid repaired PCAP dataset.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# This script is expected to be placed beside compare_cicids_csv_vs_pcap_inference.py.
try:
    import compare_cicids_csv_vs_pcap_inference as cmp
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "請將本程式放在 core/，並確認 core/compare_cicids_csv_vs_pcap_inference.py 存在。"
    ) from exc


def build_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    available = set(feature_cols)

    def exact(names: List[str]) -> List[str]:
        return [n for n in names if n in available]

    flags = [
        n for n in feature_cols
        if ("Flag Count" in n) or ("PSH Flags" in n) or ("URG Flags" in n)
    ]
    act = exact(["act_data_pkt_fwd"])
    minseg = exact(["min_seg_size_forward"])
    packet_lengths = [
        n for n in feature_cols
        if (
            "Packet Length" in n
            or "Packet Size" in n
            or "Segment Size" in n
            or n in {
                "Total Length of Fwd Packets",
                "Total Length of Bwd Packets",
                "Avg Fwd Segment Size",
                "Avg Bwd Segment Size",
            }
        )
    ]
    iat = [n for n in feature_cols if "IAT" in n]
    init_windows = [
        n for n in feature_cols
        if n in {"Init_Win_bytes_forward", "Init_Win_bytes_backward"}
    ]
    subflow = [n for n in feature_cols if n.startswith("Subflow ")]
    rates_duration = [
        n for n in feature_cols
        if n == "Flow Duration"
        or n.endswith("/s")
        or "Packets/s" in n
        or n.startswith("Active ")
        or n.startswith("Idle ")
    ]
    byte_packet_counts = exact([
        "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    ])

    raw_groups = {
        "flags": flags,
        "act_data_pkt_fwd": act,
        "min_seg_size_forward": minseg,
        "packet_length_and_segment_size": packet_lengths,
        "IAT": iat,
        "initial_TCP_window": init_windows,
        "subflow": subflow,
        "rates_duration_active_idle": rates_duration,
        "packet_counts_and_total_bytes": byte_packet_counts,
    }
    return {k: v for k, v in raw_groups.items() if v}


def score_one(X: np.ndarray, payload: dict, bundle, feature_cols: List[str], name: str) -> dict:
    summary, _, _, _ = cmp.score_matrix(X, payload, bundle, feature_cols, name)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="以同一模型與 scaler 做特徵群組反事實替換，定位 PCAP 分數偏高的來源。"
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--pcap-flow-csv", required=True,
                        help="建議使用 compat candidate CSV；原始輸出不會被修改。")
    parser.add_argument("--model", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--sample-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_path = Path(args.train_csv)
    pcap_path = Path(args.pcap_flow_csv)
    model_path = Path(args.model)
    scaler_path = Path(args.scaler)
    for p in [train_path, pcap_path, model_path, scaler_path]:
        if not p.exists():
            raise FileNotFoundError(f"找不到檔案：{p}")
    if args.sample_rows < 100:
        raise ValueError("--sample-rows 至少應為 100；建議 50000")

    with scaler_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "scaler" not in payload or "feature_columns" not in payload:
        raise ValueError("Scaler .pkl 格式不符：需要 dict 且含 scaler / feature_columns")
    feature_cols = [cmp.clean_column(c) for c in payload["feature_columns"]]

    pcap_labels = cmp.count_labels(pcap_path)
    pcap_filter, filter_note = cmp.choose_pcap_label_filter("auto", pcap_labels)
    train_df, train_meta = cmp.sample_csv_uniform(
        train_path, feature_cols, args.sample_rows, args.seed,
        label_filter={"BENIGN", "NORMAL"}, exclude_single_packet_flows=True,
    )
    pcap_df, pcap_meta = cmp.sample_csv_uniform(
        pcap_path, feature_cols, args.sample_rows, args.seed + 1,
        label_filter=pcap_filter, exclude_single_packet_flows=True,
    )
    X_train = cmp.prepare_matrix(train_df, feature_cols)
    X_pcap = cmp.prepare_matrix(pcap_df, feature_cols)

    import torch
    from model_registry import load_anomaly_model

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定 --device cuda，但目前沒有可用 CUDA")
        device = torch.device(args.device)
    bundle = load_anomaly_model(str(model_path), device=device)
    rep = bundle.input_representation
    if rep and not str(rep).startswith("csv_features"):
        raise ValueError(f"checkpoint representation={rep!r} 不是 csv_features")

    baseline_train = score_one(X_train, payload, bundle, feature_cols, "TRAIN_CSV_BENIGN")
    baseline_pcap = score_one(X_pcap, payload, bundle, feature_cols, "PCAP_BASELINE")
    groups = build_groups(feature_cols)
    rng = np.random.default_rng(args.seed + 2026)
    shuffled_train = X_train[rng.permutation(len(X_train))]
    if len(shuffled_train) < len(X_pcap):
        # sample_rows is usually identical; this handles a smaller training sample safely.
        take = rng.integers(0, len(shuffled_train), size=len(X_pcap))
        replacement = shuffled_train[take]
    else:
        replacement = shuffled_train[:len(X_pcap)]

    rows = []
    baseline_rate = float(baseline_pcap["anomaly_rate"])
    baseline_p95 = float(baseline_pcap["score_quantiles"]["p95"])
    baseline_p99 = float(baseline_pcap["score_quantiles"]["p99"])

    # Each run changes one group only. This ranks which group is most associated with
    # the anomaly-score skew; it is not a proposed production-time feature correction.
    for group_name, feature_names in groups.items():
        indices = [feature_cols.index(n) for n in feature_names]
        X_cf = X_pcap.copy()
        X_cf[:, indices] = replacement[:, indices]
        s = score_one(X_cf, payload, bundle, feature_cols, f"PCAP_COUNTERFACTUAL_{group_name}")
        rows.append({
            "scenario": f"replace_{group_name}_with_train_values",
            "group": group_name,
            "features_replaced": "; ".join(feature_names),
            "n_features": len(feature_names),
            "anomaly_rate": float(s["anomaly_rate"]),
            "anomaly_rate_delta_pp_vs_baseline": (float(s["anomaly_rate"]) - baseline_rate) * 100,
            "score_p50": float(s["score_quantiles"]["p50"]),
            "score_p95": float(s["score_quantiles"]["p95"]),
            "score_p95_delta_vs_baseline": float(s["score_quantiles"]["p95"]) - baseline_p95,
            "score_p99": float(s["score_quantiles"]["p99"]),
            "score_p99_delta_vs_baseline": float(s["score_quantiles"]["p99"]) - baseline_p99,
        })

    all_names = sorted(set(n for values in groups.values() for n in values))
    all_indices = [feature_cols.index(n) for n in all_names]
    X_all = X_pcap.copy()
    X_all[:, all_indices] = replacement[:, all_indices]
    s_all = score_one(X_all, payload, bundle, feature_cols, "PCAP_COUNTERFACTUAL_ALL_GROUPS")
    rows.append({
        "scenario": "replace_all_candidate_groups_with_train_values",
        "group": "ALL_GROUPS",
        "features_replaced": "; ".join(all_names),
        "n_features": len(all_names),
        "anomaly_rate": float(s_all["anomaly_rate"]),
        "anomaly_rate_delta_pp_vs_baseline": (float(s_all["anomaly_rate"]) - baseline_rate) * 100,
        "score_p50": float(s_all["score_quantiles"]["p50"]),
        "score_p95": float(s_all["score_quantiles"]["p95"]),
        "score_p95_delta_vs_baseline": float(s_all["score_quantiles"]["p95"]) - baseline_p95,
        "score_p99": float(s_all["score_quantiles"]["p99"]),
        "score_p99_delta_vs_baseline": float(s_all["score_quantiles"]["p99"]) - baseline_p99,
    })

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values("anomaly_rate", ascending=True)
    csv_path = outdir / "feature_group_ablation.csv"
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "counterfactual diagnosis only; never use this modified input as production output",
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "model_type": bundle.model_type,
        "representation": rep,
        "threshold": float(bundle.threshold),
        "normalization": payload.get("normalization", "none"),
        "feature_count": len(feature_cols),
        "train_sample": train_meta,
        "pcap_sample": pcap_meta,
        "pcap_label_counts_full": {str(k): int(v) for k, v in pcap_labels.items()},
        "pcap_label_filter_note": filter_note,
        "baseline_train": baseline_train,
        "baseline_pcap": baseline_pcap,
        "feature_groups": groups,
        "ablation_csv": str(csv_path),
    }
    json_path = outdir / "summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# CICIDS2017 特徵群組反事實診斷",
        "",
        f"- 產生時間：{summary['generated_at']}",
        f"- 模型：`{model_path}`，類型 `{bundle.model_type}`，representation `{rep}`",
        f"- 閾值：`{bundle.threshold:.12g}`",
        f"- 訓練 CSV 抽樣：{len(X_train):,}；PCAP 候選樣本：{len(X_pcap):,}",
        "",
        "> 重要：這是診斷用反事實試驗。將 PCAP 的某組特徵換成訓練資料的值，只用來判斷該特徵群是否與分數差異有關，不是正確的轉換方式，也不可用來宣稱模型效能改善。",
        "",
        "## 基準",
        "",
        "| 資料 | 異常率 | Score p50 | Score p95 | Score p99 |",
        "|---|---:|---:|---:|---:|",
        f"| TRAIN CSV BENIGN | {baseline_train['anomaly_rate']:.2%} | {baseline_train['score_quantiles']['p50']:.8g} | {baseline_train['score_quantiles']['p95']:.8g} | {baseline_train['score_quantiles']['p99']:.8g} |",
        f"| PCAP candidate baseline | {baseline_pcap['anomaly_rate']:.2%} | {baseline_pcap['score_quantiles']['p50']:.8g} | {baseline_pcap['score_quantiles']['p95']:.8g} | {baseline_pcap['score_quantiles']['p99']:.8g} |",
        "",
        "## 單一群組替換結果",
        "",
        "異常率越接近訓練 CSV 不代表可以直接套用這項修正；需回到轉換器與 CICFlowMeter v3.0 定義查證。",
        "",
        "| 情境 | 替換欄位數 | 異常率 | 相對基準變化 (百分點) | Score p95 | Score p99 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in result.iterrows():
        lines.append(
            f"| `{r['scenario']}` | {int(r['n_features'])} | {r['anomaly_rate']:.2%} | "
            f"{r['anomaly_rate_delta_pp_vs_baseline']:+.2f} | {r['score_p95']:.8g} | {r['score_p99']:.8g} |"
        )
    lines += [
        "",
        "## 判讀",
        "",
        "- 若替換某一群組使異常率與 p95 大幅下降，該群組值得優先追查其 PCAP 特徵公式／版本差異。",
        "- 若只有把所有群組一起替換才大幅下降，表示有多組差異共同影響模型，應對照原始 CICFlowMeter 特徵生成器。",
        "- 若全部替換後分數仍顯著偏高，除列出的群組外還存在其他資料表示法／Flow 組成差異。",
        "- 這個試驗不能告訴你訓練資料的數值一定正確，只表示模型目前對哪組輸入偏移較敏感。",
        "",
        "## 檔案",
        "",
        f"- `{csv_path.name}`：完整群組替換結果。",
        f"- `{json_path.name}`：基準資訊、分組欄位與摘要。",
    ]
    report_path = outdir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== 基準 ===")
    print(f"TRAIN CSV: {baseline_train['anomaly_rate']:.2%}; p95={baseline_train['score_quantiles']['p95']:.8g}")
    print(f"PCAP base: {baseline_pcap['anomaly_rate']:.2%}; p95={baseline_pcap['score_quantiles']['p95']:.8g}")
    print("\n=== 替換一組特徵為訓練樣本值後 ===")
    print(result[["scenario", "n_features", "anomaly_rate", "anomaly_rate_delta_pp_vs_baseline", "score_p95", "score_p99"]].to_string(index=False))
    print(f"\n[完成] 報告：{report_path.resolve()}")
    print(f"[完成] CSV：{csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[錯誤] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
