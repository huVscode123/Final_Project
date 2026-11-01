#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""檢查 Flow CSV 與 scored CSV 的實際標籤分布，辨別 UNKNOWN 是否源自標籤匹配。"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd


def label_counts(path: str, label_col: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0, compression="infer", low_memory=False)
    lookup = {str(c).strip().lower(): c for c in header.columns}
    real_col = lookup.get(label_col.lower())
    if real_col is None:
        raise ValueError(f"{path} 找不到標籤欄 {label_col!r}；現有欄位：{list(header.columns)}")
    labels = pd.read_csv(path, usecols=[real_col], compression="infer", low_memory=False)[real_col]
    labels = labels.astype(str).str.strip().str.upper()
    counts = labels.value_counts(dropna=False)
    total = max(int(counts.sum()), 1)
    print(f"\n{path}\n  標籤欄：{real_col}")
    for lab, n in counts.items():
        print(f"  {lab}: {int(n):,} ({int(n) / total:.2%})")
    if len(counts) == 1 and counts.index[0] == "UNKNOWN":
        print("  結果：所有標籤都是 UNKNOWN；檢查 --labels 與 benign_files / entries 檔名 token。")
    elif "UNKNOWN" in counts.index:
        print("  結果：存在 UNKNOWN；若檔案預期為 Monday 全良性，檢查 benign_files 是否含 monday-workinghours。")
    elif set(counts.index) == {"BENIGN"}:
        print("  結果：標籤為 BENIGN 單一類別；可以計算 FPR，但不能計算 Recall / F1 / ROC-AUC。")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow-csv", help="Flow CSV，標籤欄為 Label")
    ap.add_argument("--scored-csv", help="scored CSV 或 CSV.GZ，標籤欄為 label")
    args = ap.parse_args()
    if not args.flow_csv and not args.scored_csv:
        ap.error("至少指定 --flow-csv 或 --scored-csv")
    try:
        if args.flow_csv:
            label_counts(args.flow_csv, "Label")
        if args.scored_csv:
            label_counts(args.scored_csv, "label")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
