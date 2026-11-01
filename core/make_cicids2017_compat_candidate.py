#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a *diagnostic candidate* Flow CSV matching observed CICIDS2017 CSV conventions.

This does not read/reparse PCAPs and never overwrites the source Flow CSV.
It applies only changes that are checked against the full training CSV:
  1) drop flows with <= 1 packet (CICFlowMeter export convention);
  2) if all training rows have all six Bulk features equal to zero, zero these fields;
  3) if short training flows use Subflow == Total consistently, mirror that convention
     for PCAP flows whose Flow Duration <= 1 second;
  4) use -1 as missing backward TCP-window sentinel for non-TCP / no-backward-packet
     rows, based on the training CSV's observed sentinel convention.

The result is for fast diagnostic comparison only. Do not use it as a final converter
until score/features and labels are validated.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

BULK_COLS = [
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]
SUBFLOW_PAIRS = [
    ("Total Fwd Packets", "Subflow Fwd Packets"),
    ("Total Length of Fwd Packets", "Subflow Fwd Bytes"),
    ("Total Backward Packets", "Subflow Bwd Packets"),
    ("Total Length of Bwd Packets", "Subflow Bwd Bytes"),
]
REQUIRED = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Protocol", "Init_Win_bytes_backward", *BULK_COLS,
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
]


def clean_col(x: object) -> str:
    return str(x).replace("\ufeff", "").strip()


def read_columns(path: str) -> List[str]:
    return [clean_col(c) for c in pd.read_csv(path, nrows=0).columns]


def numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def iter_reference(path: str, chunksize: int):
    cols = read_columns(path)
    missing = [c for c in REQUIRED if c not in cols and c != "Protocol"]
    if missing:
        raise ValueError(f"訓練 CSV 缺少必要欄位：{missing}")
    usecols = [c for c in REQUIRED if c in cols]
    # Keep Label when present so the reference statistics use BENIGN/NORMAL rows only.
    if "Label" in cols:
        usecols.append("Label")
    # Some CICIDS2017 feature CSV variants do not contain Protocol; it is not
    # needed for checking the reference distributions.
    usecols_set = set(usecols)
    for chunk in pd.read_csv(path, usecols=lambda c: clean_col(c) in usecols_set,
                             chunksize=chunksize, low_memory=False):
        chunk.columns = [clean_col(c) for c in chunk.columns]
        if "Label" in chunk.columns:
            labels = chunk["Label"].astype(str).str.strip().str.upper()
            chunk = chunk[labels.isin(["BENIGN", "NORMAL"])].copy()
        numeric(chunk, [c for c in usecols if c in chunk.columns])
        yield chunk


def inspect_training_reference(path: str, chunksize: int) -> Dict[str, object]:
    bulk_nonzero = Counter({c: 0 for c in BULK_COLS})
    bulk_rows = 0
    single_packets = 0
    short_rows = 0
    short_matches = Counter({sub: 0 for _, sub in SUBFLOW_PAIRS})
    short_total = Counter({sub: 0 for _, sub in SUBFLOW_PAIRS})
    bwd_zero_rows = 0
    bwd_zero_init_minus1 = 0
    init_values = Counter()
    row_count = 0

    for df in iter_reference(path, chunksize):
        if df.empty:
            continue
        row_count += len(df)
        bulk_rows += len(df)
        for c in BULK_COLS:
            values = df[c].fillna(0) if c in df.columns else pd.Series(0, index=df.index)
            bulk_nonzero[c] += int((values != 0).sum())

        total_pkts = df["Total Fwd Packets"].fillna(0) + df["Total Backward Packets"].fillna(0)
        single_packets += int((total_pkts <= 1).sum())

        short = df[df["Flow Duration"].fillna(np.inf) <= 1_000_000].copy()
        short_rows += len(short)
        for total_col, sub_col in SUBFLOW_PAIRS:
            totals = short[total_col].fillna(0)
            subs = short[sub_col].fillna(0)
            # Evaluate actual non-zero directional totals; zeros match trivially.
            meaningful = totals != 0
            short_total[sub_col] += int(meaningful.sum())
            if meaningful.any():
                short_matches[sub_col] += int(
                    np.isclose(totals.loc[meaningful].to_numpy(float),
                               subs.loc[meaningful].to_numpy(float)).sum()
                )

        bwd_pkts = df["Total Backward Packets"].fillna(0)
        no_bwd = bwd_pkts <= 0
        bwd_zero_rows += int(no_bwd.sum())
        init_bwd = df["Init_Win_bytes_backward"].fillna(0)
        bwd_zero_init_minus1 += int((no_bwd & np.isclose(init_bwd, -1)).sum())
        vc = init_bwd.value_counts(dropna=False)
        for k, v in vc.items():
            init_values[str(k)] += int(v)

    all_bulk_zero = all(bulk_nonzero[c] == 0 for c in BULK_COLS)
    short_agreement = {
        sub: (short_matches[sub] / short_total[sub] if short_total[sub] else 0.0)
        for _, sub in SUBFLOW_PAIRS
    }
    no_bwd_minus1_ratio = bwd_zero_init_minus1 / bwd_zero_rows if bwd_zero_rows else 0.0
    return {
        "benign_rows": row_count,
        "single_packet_rows_in_reference": single_packets,
        "bulk_nonzero_counts": dict(bulk_nonzero),
        "all_six_bulk_features_zero": all_bulk_zero,
        "short_flow_rows_duration_le_1s": short_rows,
        "short_flow_subflow_equals_total_ratio": short_agreement,
        "rows_without_backward_packets": bwd_zero_rows,
        "no_backward_rows_init_window_minus1_ratio": no_bwd_minus1_ratio,
        "init_window_backward_top_values": init_values.most_common(12),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-csv", required=True, help="CICIDS2017 訓練 CSV，用於核對特徵約定")
    ap.add_argument("--input", required=True, help="既有 PCAP Flow CSV；不會被覆寫")
    ap.add_argument("--output", required=True, help="輸出相容性候選 CSV")
    ap.add_argument("--chunksize", type=int, default=100000)
    args = ap.parse_args()

    for path in [args.train_csv, args.input]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"找不到檔案：{path}")

    print("[1/3] 掃描整份訓練 CSV，確認可套用的相容性規則……")
    ref = inspect_training_reference(args.train_csv, args.chunksize)
    print(f"訓練 BENIGN/NORMAL 筆數：{ref['benign_rows']:,}")
    print(f"訓練 CSV 單封包筆數：{ref['single_packet_rows_in_reference']:,}")
    print("六個 Bulk 欄位非零筆數：")
    for c, n in ref["bulk_nonzero_counts"].items():
        print(f"  {c}: {n:,}")
    print("短 Flow (duration <= 1 秒) 中 Subflow 欄位等於總量的比例：")
    for c, ratio in ref["short_flow_subflow_equals_total_ratio"].items():
        print(f"  {c}: {ratio:.2%}")
    print("訓練資料中 Backward Packets=0 時，Init_Win_bytes_backward=-1 比例："
          f"{ref['no_backward_rows_init_window_minus1_ratio']:.2%}")
    print("Init_Win_bytes_backward 常見值：", ref["init_window_backward_top_values"])

    all_bulk_zero = bool(ref["all_six_bulk_features_zero"])
    # Apply subflow=totals only when the corresponding training feature empirically
    # follows that convention for >=95% of short flows.
    subflow_agreement = ref["short_flow_subflow_equals_total_ratio"]
    subflow_patch_cols = {
        sub for total, sub in SUBFLOW_PAIRS
        if subflow_agreement.get(sub, 0.0) >= 0.95
    }
    patch_missing_window = ref["no_backward_rows_init_window_minus1_ratio"] >= 0.90

    input_cols = read_columns(args.input)
    required_input = [c for c in REQUIRED if c not in input_cols]
    if required_input:
        raise ValueError(f"PCAP Flow CSV 缺少必要欄位：{required_input}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise ValueError("輸出路徑不可與輸入相同，避免覆寫原始 Flow CSV")

    counters = Counter()
    first = True
    readcols = set(input_cols)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as out_f:
        for df in pd.read_csv(args.input, chunksize=args.chunksize, low_memory=False):
            df.columns = [clean_col(c) for c in df.columns]
            numeric_cols = [c for c in REQUIRED if c in df.columns]
            numeric(df, numeric_cols)

            fwd_pkts = df["Total Fwd Packets"].fillna(0)
            bwd_pkts = df["Total Backward Packets"].fillna(0)
            total_pkts = fwd_pkts + bwd_pkts
            keep = total_pkts > 1
            counters["input_rows"] += len(df)
            counters["single_packet_rows_dropped"] += int((~keep).sum())
            df = df.loc[keep].copy()
            if df.empty:
                continue

            if all_bulk_zero:
                for c in BULK_COLS:
                    counters[f"bulk_nonzero_before_{c}"] += int((pd.to_numeric(df[c], errors="coerce").fillna(0) != 0).sum())
                    df[c] = 0.0

            # A flow whose duration is <= 1 second cannot contain a >1s inter-packet
            # gap. When the training CSV confirms Subflow==Total on short flows, use
            # this empirically validated rule for those rows only.
            short_mask = pd.to_numeric(df["Flow Duration"], errors="coerce").fillna(np.inf) <= 1_000_000
            for total_col, sub_col in SUBFLOW_PAIRS:
                if sub_col not in subflow_patch_cols:
                    continue
                vals_total = pd.to_numeric(df[total_col], errors="coerce").fillna(0)
                vals_sub = pd.to_numeric(df[sub_col], errors="coerce").fillna(0)
                mask = short_mask & ~np.isclose(vals_sub, vals_total)
                counters[f"subflow_rows_adjusted_{sub_col}"] += int(mask.sum())
                df.loc[mask, sub_col] = vals_total.loc[mask].to_numpy()

            if patch_missing_window:
                proto = pd.to_numeric(df["Protocol"], errors="coerce")
                bwd_count = pd.to_numeric(df["Total Backward Packets"], errors="coerce").fillna(0)
                mask = (proto != 6) | (bwd_count <= 0)
                old = pd.to_numeric(df["Init_Win_bytes_backward"], errors="coerce")
                counters["backward_window_rows_set_to_minus1"] += int((mask & ~np.isclose(old.fillna(0), -1)).sum())
                df.loc[mask, "Init_Win_bytes_backward"] = -1

            df.to_csv(out_f, index=False, header=first, mode="a" if not first else "w",
                      encoding="utf-8-sig" if first else "utf-8", lineterminator="\n")
            first = False
            counters["output_rows"] += len(df)

    print("\n[2/3] 相容性候選檔已產生（不是正式修正版）")
    print(f"輸入筆數：{counters['input_rows']:,}")
    print(f"排除單封包 Flow：{counters['single_packet_rows_dropped']:,}")
    print(f"輸出筆數：{counters['output_rows']:,}")
    print(f"套用 Bulk 全零：{'是' if all_bulk_zero else '否（訓練資料至少一個 Bulk 欄位存在非零值）'}")
    print(f"套用短 Flow Subflow 對齊欄位：{sorted(subflow_patch_cols)}")
    print(f"套用 backward window 缺失值 -1：{'是' if patch_missing_window else '否（訓練資料未支持此約定）'}")
    print("變更統計：")
    for k, v in sorted(counters.items()):
        if k.startswith("bulk_nonzero_before_") or k.startswith("subflow_rows_adjusted_") or k == "backward_window_rows_set_to_minus1":
            print(f"  {k}: {v:,}")
    print(f"\n[3/3] 輸出：{os.path.abspath(args.output)}")
    print("注意：這是快速診斷候選檔。比較分數若改善，仍須把經驗證的規則正式加入轉換器並重跑完整資料。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
