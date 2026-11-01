#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare FIN and packet-count distributions for CICIDS2017 training CSV and PCAP-derived flows.

The standard CICIDS2017 CSV often does not include a Protocol column. In that
case, statistics for the training file are reported over ALL flows, without
pretending that every row is TCP. If the PCAP-derived CSV has Protocol, its
all-flow statistics are reported for a fair aggregate comparison, followed by
TCP-only diagnostics as supplementary information.
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# Core fields needed for fair aggregate comparison.
CORE_COLS = ["FIN Flag Count", "Total Fwd Packets", "Total Backward Packets"]
# Additional fields are optional and printed when present.
OPTIONAL_COLS = [
    "Protocol", "SYN Flag Count", "RST Flag Count", "ACK Flag Count",
    "PSH Flag Count", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "Fwd Header Length", "Bwd Header Length", "act_data_pkt_fwd",
]
ALL_COLS = CORE_COLS + [c for c in OPTIONAL_COLS if c not in CORE_COLS]

ALIASES = {
    "Protocol": ["protocol", "ip protocol", "protocol number", "transport protocol"],
    "FIN Flag Count": ["fin flag count", "fin flags count"],
    "Total Fwd Packets": ["total fwd packets", "total forward packets"],
    "Total Backward Packets": ["total backward packets", "total bwd packets"],
    "SYN Flag Count": ["syn flag count", "syn flags count"],
    "RST Flag Count": ["rst flag count", "rst flags count"],
    "ACK Flag Count": ["ack flag count", "ack flags count"],
    "PSH Flag Count": ["psh flag count", "psh flags count"],
    "Init_Win_bytes_forward": ["init_win_bytes_forward", "initial window bytes forward"],
    "Init_Win_bytes_backward": ["init_win_bytes_backward", "initial window bytes backward"],
    "Fwd Header Length": ["fwd header length", "forward header length"],
    "Bwd Header Length": ["bwd header length", "backward header length"],
    "act_data_pkt_fwd": ["act_data_pkt_fwd", "active data packets forward"],
}


def _norm(value):
    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("\ufeff", "").replace("\u200b", "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def _load(path):
    header = pd.read_csv(path, nrows=0, low_memory=False, encoding="utf-8-sig")
    raw_columns = list(header.columns)
    mapping = {_norm(col): col for col in raw_columns}
    selected = {}
    missing_core = []

    for canonical in ALL_COLS:
        options = [canonical] + ALIASES.get(canonical, [])
        actual = next((mapping[_norm(opt)] for opt in options if _norm(opt) in mapping), None)
        if actual is not None:
            selected[canonical] = actual
        elif canonical in CORE_COLS:
            missing_core.append(canonical)

    if missing_core:
        print(f"\n[欄位檢查失敗] {path}", file=sys.stderr)
        print(f"缺少必要欄位：{missing_core}", file=sys.stderr)
        print("CSV 實際欄名如下：", file=sys.stderr)
        for i, col in enumerate(raw_columns):
            print(f"  [{i}] {col!r}", file=sys.stderr)
        raise ValueError("缺少必要欄位，無法計算 FIN／封包數分布。")

    df = pd.read_csv(path, usecols=list(dict.fromkeys(selected.values())),
                     low_memory=False, encoding="utf-8-sig")
    df = df.rename(columns={actual: canonical for canonical, actual in selected.items()})
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _dist(values, labels=None):
    values = values.dropna()
    if values.empty:
        return {}, {}
    counts = values.value_counts().sort_index()
    ratios = values.value_counts(normalize=True).sort_index()

    def key(k):
        if labels and k in labels:
            return labels[k]
        return str(k)

    return ({key(k): int(v) for k, v in counts.items()},
            {key(k): round(float(v), 4) for k, v in ratios.items()})


def _profile_frame(path, df, title, subset_name, protocol_filter=None):
    """Print same-scope aggregate stats; caller ensures subset is appropriate."""
    if df.empty:
        print(f"\n--- {title}：{subset_name} ---\n沒有資料。")
        return
    packets = df["Total Fwd Packets"] + df["Total Backward Packets"]
    fin = df["FIN Flag Count"].dropna().clip(lower=0, upper=3).round().astype(int)
    fin_counts, fin_ratios = _dist(fin, {0: "FIN=0", 1: "FIN=1", 2: "FIN=2", 3: "FIN=3+"})
    print(f"\n--- {title}：{subset_name} ---")
    print(f"Flow 數：{len(df):,}")
    print("FIN Flag Count 分布筆數：", fin_counts)
    print("FIN Flag Count 分布比例：", fin_ratios)
    valid_packets = packets.dropna()
    print("總封包數 2/3/4 Flow 筆數：", {k: int((valid_packets == k).sum()) for k in (2, 3, 4)})
    print("總封包數 2/3/4 Flow 佔比（分母為此範圍全部 Flow）：", {
        k: round(float((valid_packets == k).mean()), 4) if len(valid_packets) else 0.0
        for k in (2, 3, 4)
    })
    for col in ("SYN Flag Count", "FIN Flag Count", "RST Flag Count", "ACK Flag Count",
                "PSH Flag Count", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
                "Fwd Header Length", "Bwd Header Length", "act_data_pkt_fwd"):
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals):
            print(f"{col}: p50={vals.quantile(.50):.4g}, p95={vals.quantile(.95):.4g}, "
                  f"p99={vals.quantile(.99):.4g}, min={vals.min():.4g}, max={vals.max():.4g}")
    if "act_data_pkt_fwd" in df.columns:
        fwd = df["Total Fwd Packets"]
        act = df["act_data_pkt_fwd"]
        valid = fwd.notna() & act.notna()
        suspicious = valid & act.ge(fwd)
        print(f"act_data_pkt_fwd >= Total Fwd Packets：{int(suspicious.sum()):,} / {int(valid.sum()):,}")


def _profile(path, df):
    print(f"\n================ {path} ================")
    print(f"全部 Flow：{len(df):,}")
    has_protocol = "Protocol" in df.columns
    if has_protocol:
        proto = df["Protocol"]
        print("Protocol 分布：", {str(k): int(v) for k, v in proto.value_counts(dropna=False).sort_index().items()})
        # All-flow stats make a fair aggregate comparison against training CSVs without Protocol.
        _profile_frame(path, df, "全體統計（可與無 Protocol 的訓練 CSV 做整體對照）", "ALL FLOWS")
        tcp = df.loc[proto.eq(6)].copy()
        _profile_frame(path, tcp, "補充 TCP 專屬統計（訓練 CSV 無 Protocol，無法直接對照）", "TCP ONLY")
    else:
        print("Protocol 欄：不存在。依此 CSV 本身無法可靠篩出 TCP；以下只做全體 Flow 統計。")
        _profile_frame(path, df, "全體統計（不假設協定）", "ALL FLOWS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True, help="訓練用 CICIDS2017 CSV")
    parser.add_argument("--pcap-csv", required=True, help="PCAP 轉換後的 Flow CSV")
    args = parser.parse_args()
    for path in (args.train_csv, args.pcap_csv):
        if not Path(path).is_file():
            print(f"找不到檔案：{path}", file=sys.stderr)
            return 2
    try:
        train = _load(args.train_csv)
        pcap = _load(args.pcap_csv)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        print(f"\n診斷中止：{exc}", file=sys.stderr)
        return 2

    _profile(args.train_csv, train)
    _profile(args.pcap_csv, pcap)
    print("\n=== 判讀提醒 ===")
    print("1. 訓練 CSV 沒有 Protocol 時，只能將訓練全體 Flow 與 PCAP 全體 Flow 做聚合比較。")
    print("2. PCAP 的 TCP-only 統計只能作線索，不能直接當成訓練 CSV 的 TCP 統計。")
    print("3. FIN Flag Count 與 2/3/4 封包比例若全體分布差異很大，可支持進一步檢查 Flow 切分；不能單憑此結果證明 first-FIN 假說。")
    print("4. 不要用目的埠推斷協定，也不要把無 Protocol 的全部訓練資料假設為 TCP。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
