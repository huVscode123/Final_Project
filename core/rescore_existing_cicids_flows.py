#!/usr/bin/env python3
# Re-score already converted CICIDS2017 flow CSVs without parsing the large PCAP files again.
# Drops flows with total forward + backward packet count <= 1, matching CICFlowMeter export rules.
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from batch_pcap_evaluate import score_flow_csv
    from model_registry import load_anomaly_model
except ImportError as exc:
    raise SystemExit(
        "無法匯入 core/batch_pcap_evaluate.py 或 core/model_registry.py。"
        "請將本腳本放在專案的 core 資料夾內執行。"
    ) from exc


def create_filtered_copy(source: Path, target: Path, chunksize: int = 50000) -> tuple[int, int, int]:
    """Write a temporary CSV containing only rows where Total Fwd + Total Backward > 1."""
    required = {"Total Fwd Packets", "Total Backward Packets"}
    first = True
    total_in = 0
    total_kept = 0
    total_dropped = 0

    try:
        for df in pd.read_csv(source, chunksize=chunksize, low_memory=False):
            missing = required.difference(df.columns)
            if missing:
                raise ValueError(f"{source.name} 缺少欄位：{sorted(missing)}")
            fwd = pd.to_numeric(df["Total Fwd Packets"], errors="coerce").fillna(0)
            bwd = pd.to_numeric(df["Total Backward Packets"], errors="coerce").fillna(0)
            keep = (fwd + bwd) > 1
            kept = df.loc[keep].copy()
            total_in += len(df)
            total_kept += len(kept)
            total_dropped += int((~keep).sum())
            kept.to_csv(target, mode="w" if first else "a", header=first,
                        index=False, encoding="utf-8", lineterminator="\n")
            first = False
        if first:
            # Preserve the source header even if the CSV has no data rows.
            header = pd.read_csv(source, nrows=0)
            header.to_csv(target, index=False, encoding="utf-8", lineterminator="\n")
    except Exception:
        try:
            target.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    return total_in, total_kept, total_dropped


def parse_args():
    parser = argparse.ArgumentParser(
        description="對既有 PCAP→Flow CSV 排除單封包 Flow，重新計分，不重新解析 PCAP。"
    )
    parser.add_argument("--model", required=True, help="訓練好的 .pt checkpoint")
    parser.add_argument("--scaler", required=True, help="訓練期前處理器 .pkl")
    parser.add_argument("--flow-dir", required=True, help="既有 Flow CSV 資料夾，例如 output/.../flows")
    parser.add_argument("--scored-dir", required=True, help="計分結果輸出資料夾，例如 output/.../scored")
    parser.add_argument("--include", nargs="*", default=None,
                        help="可選：只處理檔名包含指定字串的 CSV，例如 --include monday")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--chunksize", type=int, default=50000)
    return parser.parse_args()


def main():
    args = parse_args()
    flow_dir = Path(args.flow_dir)
    scored_dir = Path(args.scored_dir)
    if not flow_dir.is_dir():
        raise SystemExit(f"Flow 資料夾不存在：{flow_dir}")
    scored_dir.mkdir(parents=True, exist_ok=True)

    include = [s.lower() for s in (args.include or [])]
    files = sorted(
        p for p in flow_dir.glob("*.csv")
        if not p.name.endswith(".filtered.tmp.csv")
        and (not include or any(s in p.name.lower() for s in include))
    )
    if not files:
        raise SystemExit(f"在 {flow_dir} 找不到符合條件的 *.csv")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("指定 device=cuda，但目前 PyTorch 無法使用 CUDA。")
        device = torch.device(args.device)

    print(f"[模型] 載入 checkpoint：{args.model}")
    bundle = load_anomaly_model(args.model, device=device, default_latent_dim=args.latent_dim)
    with open(args.scaler, "rb") as f:
        payload = pickle.load(f)
    print(f"[模型] type={bundle.model_type}; model_threshold={float(bundle.threshold):.10g}; device={device}")
    print(f"[前處理器] features={len(payload['feature_columns'])}; normalization={payload.get('normalization', 'none')}")
    print(f"[工作] 找到 {len(files)} 個既有 Flow CSV；不會讀取或轉換 PCAP。")

    for idx, source in enumerate(files, 1):
        temporary = source.with_name(source.stem + ".filtered.tmp.csv")
        tag = source.stem
        destination = scored_dir / f"{tag}.csv.gz"
        print(f"\n[{idx}/{len(files)}] {source.name}")
        try:
            n_in, n_kept, n_drop = create_filtered_copy(source, temporary, args.chunksize)
            print(f"  原始 Flow：{n_in:,}; 保留 (>1 封包)：{n_kept:,}; 排除單封包/空封包：{n_drop:,}")
            info = score_flow_csv(str(temporary), str(destination), payload, bundle,
                                  chunk=args.chunksize)
            print(f"  已重新計分：{info.get('n_scored', 0):,} → {destination}")
        finally:
            temporary.unlink(missing_ok=True)

    print("\n全部完成。現在可以對相同 output 資料夾執行 batch_pcap_evaluate.py --analyze-only。")


if __name__ == "__main__":
    main()
