#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit row-level CICIDS2017 feature identities without re-reading PCAPs.

This is a read-only diagnostic. It does not alter input files, scaler, model, or threshold.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

COLS = [
    "Label", "Protocol", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length", "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
]
NUMERIC = [c for c in COLS if c not in {"Label", "Protocol"}]


def clean_col(value: object) -> str:
    return str(value).replace("\ufeff", "").strip()


def load_sample(path: str, n: int, seed: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    header = [clean_col(c) for c in pd.read_csv(path, nrows=0).columns]
    use = [c for c in COLS if c in header]
    missing_core = [
        c for c in ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                    "Total Length of Fwd Packets", "Total Length of Bwd Packets"]
        if c not in header
    ]
    if missing_core:
        raise ValueError(f"{path} 缺少核心欄位：{missing_core}")
    df = pd.read_csv(path, usecols=lambda c: clean_col(c) in set(use), low_memory=False)
    df.columns = [clean_col(c) for c in df.columns]
    total_rows = len(df)
    if "Label" in df.columns:
        lab = df["Label"].astype(str).str.strip().str.upper()
        mask = lab.isin(["BENIGN", "NORMAL"])
        if mask.any():
            df = df.loc[mask].copy()
    label_rows = len(df)
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    # Fixed-seed random row sample after filtering, for quick and repeatable audit.
    if n > 0 and len(df) > n:
        df = df.sample(n=n, random_state=seed)
    info = {
        "path": path,
        "rows_in_file": total_rows,
        "rows_after_benign_filter_if_available": label_rows,
        "rows_audited": len(df),
        "label_filter_applied": "Label" in header and label_rows < total_rows,
        "columns_present": [c for c in COLS if c in df.columns],
    }
    return df.reset_index(drop=True), info


def finite_pairs(actual: pd.Series, expected: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(expected, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(a) & np.isfinite(e)
    return a[valid], e[valid]


def formula_result(df: pd.DataFrame, label: str, actual_col: str, expected: pd.Series,
                   rtol: float = 1e-3, atol: float = 1e-2) -> Dict[str, object]:
    if actual_col not in df.columns:
        return {"formula": label, "available": False}
    actual, exp = finite_pairs(df[actual_col], expected)
    if len(actual) == 0:
        return {"formula": label, "available": True, "n": 0}
    abs_error = np.abs(actual - exp)
    rel_error = abs_error / np.maximum(np.abs(exp), 1.0)
    close = np.isclose(actual, exp, rtol=rtol, atol=atol)
    return {
        "formula": label,
        "available": True,
        "n": int(len(actual)),
        "match_pct": round(float(close.mean() * 100), 3),
        "median_abs_error": float(np.median(abs_error)),
        "p95_abs_error": float(np.quantile(abs_error, 0.95)),
        "median_relative_error_pct": round(float(np.median(rel_error) * 100), 4),
        "actual_p50": float(np.quantile(actual, 0.50)),
        "expected_p50": float(np.quantile(exp, 0.50)),
        "actual_p95": float(np.quantile(actual, 0.95)),
        "expected_p95": float(np.quantile(exp, 0.95)),
    }


def audit_one(df: pd.DataFrame, info: Dict[str, object]) -> Dict[str, object]:
    needed = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
              "Total Length of Fwd Packets", "Total Length of Bwd Packets"]
    df = df.copy()
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"資料缺少必要欄位 {c}")
    fpk = df["Total Fwd Packets"].fillna(0).clip(lower=0)
    bpk = df["Total Backward Packets"].fillna(0).clip(lower=0)
    fbytes = df["Total Length of Fwd Packets"].fillna(0)
    bbytes = df["Total Length of Bwd Packets"].fillna(0)
    allpk = fpk + bpk
    allbytes = fbytes + bbytes
    dur_s = df["Flow Duration"].where(df["Flow Duration"] > 0) / 1e6

    results: List[Dict[str, object]] = []
    if "Fwd Packet Length Mean" in df:
        results.append(formula_result(df, "Fwd Packet Length Mean == total_fwd_bytes / total_fwd_packets",
                                      "Fwd Packet Length Mean", fbytes / fpk.replace(0, np.nan)))
    if "Bwd Packet Length Mean" in df:
        results.append(formula_result(df, "Bwd Packet Length Mean == total_bwd_bytes / total_bwd_packets",
                                      "Bwd Packet Length Mean", bbytes / bpk.replace(0, np.nan)))
    for col in ["Packet Length Mean", "Average Packet Size"]:
        results.append(formula_result(df, f"{col} == (total_fwd_bytes + total_bwd_bytes) / total_packets",
                                      col, allbytes / allpk.replace(0, np.nan)))
    if "Packet Length Mean" in df.columns and "Average Packet Size" in df.columns:
        results.append(formula_result(
            df, "Packet Length Mean == Average Packet Size",
            "Packet Length Mean", df["Average Packet Size"], rtol=1e-3, atol=1e-2
        ))
    if "Packet Length Variance" in df and "Packet Length Std" in df:
        results.append(formula_result(df, "Packet Length Variance == Packet Length Std ** 2",
                                      "Packet Length Variance", df["Packet Length Std"] ** 2,
                                      rtol=5e-3, atol=1e-1))
    if "Avg Fwd Segment Size" in df:
        results.append(formula_result(df, "Avg Fwd Segment Size == total_fwd_bytes / total_fwd_packets",
                                      "Avg Fwd Segment Size", fbytes / fpk.replace(0, np.nan)))
    if "Avg Bwd Segment Size" in df:
        results.append(formula_result(df, "Avg Bwd Segment Size == total_bwd_bytes / total_bwd_packets",
                                      "Avg Bwd Segment Size", bbytes / bpk.replace(0, np.nan)))
    if "Avg Fwd Segment Size" in df.columns and "Fwd Packet Length Mean" in df.columns:
        results.append(formula_result(
            df, "Avg Fwd Segment Size == Fwd Packet Length Mean",
            "Avg Fwd Segment Size", df["Fwd Packet Length Mean"], rtol=1e-3, atol=1e-2
        ))
    if "Avg Bwd Segment Size" in df.columns and "Bwd Packet Length Mean" in df.columns:
        results.append(formula_result(
            df, "Avg Bwd Segment Size == Bwd Packet Length Mean",
            "Avg Bwd Segment Size", df["Bwd Packet Length Mean"], rtol=1e-3, atol=1e-2
        ))

    # CICFlowMeter CSV Flow Duration is in microseconds; rate formulas use seconds.
    rate_expected = {
        "Flow Bytes/s": allbytes / dur_s,
        "Flow Packets/s": allpk / dur_s,
        "Fwd Packets/s": fpk / dur_s,
        "Bwd Packets/s": bpk / dur_s,
    }
    for col, exp in rate_expected.items():
        if col in df.columns:
            results.append(formula_result(df, f"{col} == count_or_bytes / duration_seconds",
                                          col, exp, rtol=5e-3, atol=1e-2))

    if "Down/Up Ratio" in df.columns:
        float_ratio = bpk / fpk.replace(0, np.nan)
        floor_ratio = np.floor(float_ratio)
        results.append(formula_result(df, "Down/Up Ratio == floating Bwd/Fwd ratio",
                                      "Down/Up Ratio", float_ratio, rtol=1e-3, atol=1e-3))
        results.append(formula_result(df, "Down/Up Ratio == floor(Bwd/Fwd) (legacy integer division candidate)",
                                      "Down/Up Ratio", floor_ratio, rtol=1e-3, atol=1e-3))

    for col in ["Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
                "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"]:
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").fillna(0)
            results.append({
                "formula": f"{col}: rows nonzero",
                "available": True,
                "n": int(len(val)),
                "nonzero_rows": int((val != 0).sum()),
                "nonzero_pct": round(float((val != 0).mean() * 100), 4) if len(val) else 0,
            })

    # Direct identities and diagnostic summary.
    if "Fwd Header Length.1" in df.columns and "Fwd Header Length" in df.columns:
        a, e = finite_pairs(df["Fwd Header Length.1"], df["Fwd Header Length"])
        results.append({
            "formula": "Fwd Header Length.1 == Fwd Header Length",
            "available": True,
            "n": int(len(a)),
            "match_pct": round(float(np.isclose(a, e, atol=0, rtol=0).mean() * 100), 3) if len(a) else 0,
        })
    if "act_data_pkt_fwd" in df.columns:
        act = df["act_data_pkt_fwd"].fillna(0)
        results.append({
            "formula": "act_data_pkt_fwd bounds",
            "available": True,
            "n": int(len(df)),
            "negative_rows": int((act < 0).sum()),
            "greater_than_total_fwd_rows": int((act > fpk).sum()),
            "at_least_total_fwd_rows": int((act >= fpk).sum()),
        })
    if "Init_Win_bytes_backward" in df.columns:
        win = df["Init_Win_bytes_backward"]
        no_bwd = bpk <= 0
        if no_bwd.any():
            results.append({
                "formula": "Init_Win_bytes_backward when Total Backward Packets == 0",
                "available": True,
                "n": int(no_bwd.sum()),
                "minus1_pct": round(float((win[no_bwd] == -1).mean() * 100), 3),
                "zero_pct": round(float((win[no_bwd] == 0).mean() * 100), 3),
            })
    if "Subflow Fwd Bytes" in df.columns:
        short = df["Flow Duration"] <= 1e6
        meaningful = short & (fbytes != 0)
        if meaningful.any():
            sub = pd.to_numeric(df["Subflow Fwd Bytes"], errors="coerce")
            results.append({
                "formula": "Short-flow Subflow Fwd Bytes == Total Length of Fwd Packets",
                "available": True,
                "n": int(meaningful.sum()),
                "match_pct": round(float(np.isclose(sub[meaningful], fbytes[meaningful], atol=0, rtol=0).mean() * 100), 3),
            })

    return {"file": info, "checks": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--pcap-flow-csv", required=True)
    parser.add_argument("--sample-rows", type=int, default=50000,
                        help="最多抽樣筆數；0 表示全檔，預設 50000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="輸出資料夾")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    audits = []
    for key, path in [("TRAIN CSV", args.train_csv), ("PCAP CANDIDATE", args.pcap_flow_csv)]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        df, info = load_sample(path, args.sample_rows, args.seed)
        result = audit_one(df, info)
        result["name"] = key
        audits.append(result)

    json_path = out_dir / "formula_audit.json"
    json_path.write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# CICIDS2017 特徵公式一致性稽核",
        "",
        f"- Sample rows: {args.sample_rows} (0=all)",
        f"- Seed: {args.seed}",
        "- 只讀取 CSV；不會修改輸入、模型、scaler 或閾值。",
        "- 這份稽核檢查特徵之間的逐列數學關係；不代表已證明 PCAP 與 CSV 來源完全相同。",
        "",
    ]
    for audit in audits:
        info = audit["file"]
        lines += [f"## {audit['name']}", "",
                  f"- 檔案：`{info['path']}`",
                  f"- 原始列數：{info['rows_in_file']:,}",
                  f"- 標籤篩選後列數：{info['rows_after_benign_filter_if_available']:,}",
                  f"- 實際稽核列數：{info['rows_audited']:,}", "",
                  "| 公式／檢查 | 可用 | 樣本數 | 符合率／摘要 | 中位絕對誤差 | P95 絕對誤差 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for check in audit["checks"]:
            if not check.get("available"):
                lines.append(f"| {check['formula']} | 否 | - | - | - | - |")
                continue
            n = check.get("n", "-")
            if "match_pct" in check:
                summary = f"{check['match_pct']}%"
            elif "nonzero_rows" in check:
                summary = f"非零 {check['nonzero_rows']:,} ({check.get('nonzero_pct', 0)}%)"
            elif "minus1_pct" in check:
                summary = f"-1={check['minus1_pct']}%; 0={check['zero_pct']}%"
            elif "negative_rows" in check:
                summary = (f"negative={check['negative_rows']}; >Fwd={check['greater_than_total_fwd_rows']}; "
                           f">=Fwd={check['at_least_total_fwd_rows']}")
            else:
                summary = "；".join(f"{k}={v}" for k, v in check.items()
                                    if k not in {"formula", "available", "n", "median_abs_error", "p95_abs_error"})
            med = f"{check['median_abs_error']:.6g}" if "median_abs_error" in check else "-"
            p95 = f"{check['p95_abs_error']:.6g}" if "p95_abs_error" in check else "-"
            lines.append(f"| {check['formula']} | 是 | {n:,} | {summary} | {med} | {p95} |")
        lines.append("")
    lines += [
        "## 如何使用這份稽核",
        "",
        "1. `Down/Up Ratio` 若只符合 floor(Bwd/Fwd)，候選轉換器才應採用整數除法；若更符合浮點比率，維持浮點算法。",
        "2. 若訓練 CSV 的 `Packet Length Mean`、`Average Packet Size` 都不等於總 payload／總封包數，且兩者彼此不相等，需查核目標 CICFlowMeter 版本是否對第一個封包重複累計，再決定是否模仿該歷史行為。",
        "3. 若 `Packet Length Variance` 不等於 `Packet Length Std ** 2`，先確認是統計分母（樣本／母體）差異或 CSV 產生器版本差異。",
        "4. `Fwd Packet Length Mean` 與 `Avg Fwd Segment Size`、`Bwd Packet Length Mean` 與 `Avg Bwd Segment Size` 若不一致，也需先查版本／公式，不要直接強制相等。",
        "5. 修正後仍需以相同 checkpoint、scaler、特徵順序、clipping 和計分函式重新比較分數。不要直接調高閾值。",
        "",
        f"JSON 詳細輸出：`{json_path.name}`",
    ]
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[完成] 報告：", md_path.resolve())
    print("[完成] JSON：", json_path.resolve())
    for audit in audits:
        print("\n===", audit["name"], "===")
        for check in audit["checks"]:
            if check.get("available") and "match_pct" in check:
                print(f"{check['formula']}: {check['match_pct']}% match")
            elif check.get("available") and "nonzero_rows" in check:
                print(f"{check['formula']}: {check['nonzero_rows']} nonzero")
            elif check.get("available"):
                print(check["formula"], {k: v for k, v in check.items() if k not in {"formula", "available"}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
