#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a NON-DESTRUCTIVE CICIDS2017 flag-presence counterfactual CSV.

Only use this as a diagnostic candidate, not as a final converter change.
Default columns are those whose training CSV stats showed max=1:
FIN/SYN/RST/PSH/ACK Flag Count.

Example:
  python core/diagnostics/make_cicids_flag_presence_candidate.py \
    --input output/batch_eval/cicids_vae_monday_v4/flows/cicids2017-PCAPs__Monday-WorkingHours.pcap.csv \
    --output output/diagnostics/cicids_flags_presence/monday_flags_presence.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

DEFAULT_FEATURES = [
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
]


def normalize_name(value: object) -> str:
    return str(value).replace("\ufeff", "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="複製 PCAP Flow CSV，將指定旗標計數轉為 0/1 presence；原始檔不會修改。"
    )
    parser.add_argument("--input", required=True, help="原始 PCAP Flow CSV")
    parser.add_argument("--output", required=True, help="另存的候選 CSV 路徑")
    parser.add_argument(
        "--features", default=",".join(DEFAULT_FEATURES),
        help="要轉成 presence 的欄名，以逗號分隔；預設只處理目前已由訓練統計證實 max=1 的五欄。",
    )
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if not src.is_file():
        raise FileNotFoundError(f"找不到輸入檔：{src}")
    if src.resolve() == dst.resolve():
        raise ValueError("輸出路徑不可與輸入路徑相同；原始 Flow CSV 必須保留。")

    requested = [normalize_name(x) for x in args.features.split(",") if normalize_name(x)]
    if not requested:
        raise ValueError("--features 至少要指定一個欄位。")

    df = pd.read_csv(src, low_memory=False)
    # Match by cleaned names while retaining the original column spellings in output.
    matches: dict[str, str] = {}
    for original in df.columns:
        clean = normalize_name(original)
        if clean in requested:
            if clean in matches:
                raise ValueError(f"欄名清理後重複：{clean!r}，請先檢查 CSV header。")
            matches[clean] = original

    missing = [name for name in requested if name not in matches]
    if missing:
        raise ValueError(
            f"輸入 CSV 缺少指定旗標欄位：{missing}\n"
            f"實際欄名（清理後）前 100 欄：{[normalize_name(c) for c in df.columns[:100]]}"
        )

    print(f"[輸入] {src} | rows={len(df):,}")
    changes = []
    for clean in requested:
        col = matches[clean]
        numeric = pd.to_numeric(df[col], errors="coerce")
        bad = int(numeric.isna().sum())
        if bad:
            raise ValueError(f"欄位 {clean!r} 有 {bad:,} 筆非數值/缺值；停止，不自動填補。")
        negative = int((numeric < 0).sum())
        if negative:
            raise ValueError(f"欄位 {clean!r} 有 {negative:,} 筆負值；旗標計數不應為負，停止供檢查。")
        before_counts = numeric.value_counts().sort_index().to_dict()
        presence = (numeric > 0).astype("int8")
        n_changed = int((numeric != presence).sum())
        df[col] = presence
        changes.append((clean, before_counts, presence.value_counts().sort_index().to_dict(), n_changed))

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"[輸出] {dst} | rows={len(df):,} | 原始檔未變更")
    for name, before, after, changed in changes:
        print(f"\n{name}")
        print(f"  before 值分布：{before}")
        print(f"  after  值分布：{after}")
        print(f"  被轉換筆數：{changed:,} ({changed / max(len(df), 1):.2%})")
    print("\n注意：這是特徵語意的反事實診斷，不代表已確認 CICFlowMeter 的所有旗標公式，也不會改變 Flow 分段方式。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
