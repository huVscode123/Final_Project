#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Use an independent, BENIGN-only scored Flow CSV to write a new VAE checkpoint threshold.

This script DOES NOT retrain or modify model weights. It copies the checkpoint and updates
only the threshold / calibration metadata. Use only a calibration capture excluded from both
model training and the final evaluation set.

Run from the project root, for example:
  python core/update_checkpoint_threshold.py \
    --checkpoint output/threshold_study/unsup_cicids2017/best_vae_cicids2017.pt \
    --scaler output/threshold_study/unsup_cicids2017/cicids2017_preprocessor.pkl \
    --calibration-csv output/batch_eval/cicids_benign_calibration/scored/FILE.csv.gz \
    --target-fpr 0.01 \
    --output output/threshold_study/unsup_cicids2017/best_vae_cicids2017_fpr1pct.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

BENIGN_LABELS = {"BENIGN", "NORMAL"}
PROFILE_KEYS = (
    "mode",
    "tcp_termination",
    "bulk_mode",
    "flag_mode",
    "missing_window_value",
    "converter_version",
)


def normalized_path(value: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(value)))


def load_checkpoint(path: Path) -> dict:
    # Checkpoints should be your own trusted project files; torch checkpoints use pickle.
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch without weights_only
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError("Checkpoint 內容不是 dict，停止修改。")
    if "model_state" not in data or "threshold" not in data:
        raise ValueError("Checkpoint 缺少 model_state 或 threshold，無法安全更新。")
    return data


def read_calibration_scores(path: Path, chunksize: int = 100_000) -> tuple[np.ndarray, dict]:
    """Read finite scores and require every calibration row to be labeled BENIGN/NORMAL."""
    scores: list[np.ndarray] = []
    label_counts: dict[str, int] = {}
    total_rows = 0
    for chunk in pd.read_csv(path, usecols=lambda c: str(c).strip() in {"score", "label"},
                             chunksize=chunksize, low_memory=False):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        if not {"score", "label"}.issubset(chunk.columns):
            raise ValueError("Calibration CSV 必須包含 score 與 label 欄位。")
        labels = chunk["label"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
        counts = labels.value_counts(dropna=False).to_dict()
        for key, count in counts.items():
            label_counts[str(key)] = label_counts.get(str(key), 0) + int(count)
        bad = ~labels.isin(BENIGN_LABELS)
        if bad.any():
            raise ValueError(
                "校準檔必須只有 BENIGN/NORMAL 樣本；偵測到其他標籤："
                + json.dumps({str(k): int(v) for k, v in labels[bad].value_counts().to_dict().items()}, ensure_ascii=False)
            )
        vals = pd.to_numeric(chunk["score"], errors="coerce").to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            scores.append(vals)
        total_rows += len(chunk)

    if not scores:
        raise ValueError("校準檔沒有有效的數值 score。")
    result = np.concatenate(scores)
    if len(result) < 1000:
        raise ValueError(f"校準資料只有 {len(result):,} 筆有效分數；至少需要 1,000，建議 50,000 以上。")
    return result, {"rows_read": total_rows, "n_valid_scores": int(len(result)), "label_counts": label_counts}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="以獨立 BENIGN Flow 校準集設定目標 FPR，複製 checkpoint 並更新閾值（不重訓）。"
    )
    p.add_argument("--checkpoint", required=True, help="原始訓練好的 .pt checkpoint")
    p.add_argument("--scaler", required=True, help="同一次 Flow 推論使用的 scaler .pkl")
    p.add_argument("--calibration-csv", required=True, help="batch_pcap_evaluate.py 產生、且只含 BENIGN/NORMAL 的 scored CSV 或 CSV.GZ")
    p.add_argument("--target-fpr", type=float, default=0.01, help="目標良性誤報率，例如 0.01 代表 1%%")
    p.add_argument("--output", required=True, help="新 checkpoint 輸出路徑；不覆寫原始 checkpoint")
    p.add_argument("--tcp-termination", choices=["cicflowmeter_original", "mutual_fin_rst"], default="cicflowmeter_original")
    p.add_argument("--bulk-mode", choices=["cicids2017_zero", "state_machine"], default="cicids2017_zero")
    p.add_argument("--flag-mode", choices=["packet_count", "cicids2017_legacy_first_packet"], default="cicids2017_legacy_first_packet")
    p.add_argument("--missing-window-value", type=float, choices=[-1.0, 0.0], default=-1.0)
    p.add_argument("--converter-version", default="5.0.0-cicids2017-legacy-compat")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.target_fpr < 1.0:
        raise ValueError("--target-fpr 必須介於 0 和 1 之間，例如 0.01。")

    checkpoint_path = Path(args.checkpoint).resolve()
    scaler_path = Path(args.scaler).resolve()
    calibration_path = Path(args.calibration_csv).resolve()
    output_path = Path(args.output).resolve()
    for path, label in ((checkpoint_path, "Checkpoint"), (scaler_path, "Scaler"), (calibration_path, "Calibration CSV")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} 不存在：{path}")
    if output_path == checkpoint_path:
        raise ValueError("輸出路徑不可覆寫原始 checkpoint；請另存新檔。")

    # batch_pcap_evaluate.py writes <output>/scored/<file>.csv.gz, with run_meta.json at <output>/.
    run_meta_path = calibration_path.parent.parent / "run_meta.json"
    if not run_meta_path.is_file():
        raise FileNotFoundError(
            f"找不到校準批次對應的 run_meta.json：{run_meta_path}。"
            "請使用 batch_pcap_evaluate.py 完整產生校準輸出。"
        )
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

    expected_model = normalized_path(checkpoint_path)
    recorded_model = run_meta.get("model")
    if not recorded_model or normalized_path(recorded_model) != expected_model:
        raise ValueError(
            "校準批次使用的模型路徑與 --checkpoint 不同。請先用原始 checkpoint 產生校準分數。"
            f"\nrun_meta model={recorded_model!r}\ncheckpoint={str(checkpoint_path)!r}"
        )
    recorded_scaler = run_meta.get("scaler")
    if not recorded_scaler or normalized_path(recorded_scaler) != normalized_path(scaler_path):
        raise ValueError(
            "校準批次使用的 scaler 與 --scaler 不同。"
            f"\nrun_meta scaler={recorded_scaler!r}\nscaler={str(scaler_path)!r}"
        )

    expected_profile = {
        "mode": "flow",
        "tcp_termination": args.tcp_termination,
        "bulk_mode": args.bulk_mode,
        "flag_mode": args.flag_mode,
        "missing_window_value": args.missing_window_value,
        "converter_version": args.converter_version,
    }
    mismatch = {
        key: {"run_meta": run_meta.get(key), "expected": value}
        for key, value in expected_profile.items()
        if run_meta.get(key) != value
    }
    if mismatch:
        raise ValueError(
            "校準分數的推論 profile 與目前設定不一致，拒絕套用閾值：\n"
            + json.dumps(mismatch, ensure_ascii=False, indent=2)
        )

    scores, score_info = read_calibration_scores(calibration_path)
    # 'higher' picks an observed calibration score at the requested upper quantile.
    # Since anomaly means score > threshold, ties can make the achieved FPR lower than target.
    q = 1.0 - args.target_fpr
    try:
        threshold = float(np.quantile(scores, q, method="higher"))
    except TypeError:  # NumPy < 1.22
        threshold = float(np.quantile(scores, q, interpolation="higher"))
    if not math.isfinite(threshold):
        raise ValueError(f"校準得到非有限閾值：{threshold}")
    observed_fpr = float(np.mean(scores > threshold))

    checkpoint = load_checkpoint(checkpoint_path)
    updated = copy.deepcopy(checkpoint)
    old_threshold = float(updated["threshold"])
    method = "independent_benign_pcaps_fpr_{:g}".format(args.target_fpr)
    updated["threshold"] = threshold
    updated["threshold_method"] = method
    updated["threshold_info"] = {
        "method": method,
        "threshold": threshold,
        "target_fpr": float(args.target_fpr),
        "observed_calibration_fpr": observed_fpr,
        "quantile": q,
        "quantile_method": "higher",
        "n_benign_scores": int(len(scores)),
        "calibration_csv": str(calibration_path),
        "calibration_run_meta": str(run_meta_path),
        "profile": expected_profile,
        "note": "只更新 checkpoint 閾值；model_state 權重未變更。最終效能仍須在另一份獨立標記測試集驗證。",
    }
    config = dict(updated.get("config") or {})
    config["threshold_method"] = method
    updated["config"] = config
    updated["calibration_info"] = updated["threshold_info"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(updated, output_path)
    report_path = output_path.with_suffix(output_path.suffix + ".calibration.json")
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_checkpoint": str(checkpoint_path),
        "output_checkpoint": str(output_path),
        "scaler": str(scaler_path),
        "calibration_csv": str(calibration_path),
        "old_threshold": old_threshold,
        "new_threshold": threshold,
        "target_fpr": float(args.target_fpr),
        "observed_calibration_fpr": observed_fpr,
        "n_benign_scores": int(len(scores)),
        "quantile_method": "higher",
        "score_info": score_info,
        "profile": expected_profile,
        "model_weights_changed": False,
        "warning": "校準集必須與模型訓練資料及最終測試資料分離；run_meta 只能驗證推論 profile，不能證明統計獨立性。",
    }
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Threshold calibration complete ===")
    print(f"old threshold             : {old_threshold:.12g}")
    print(f"new threshold             : {threshold:.12g}")
    print(f"target FPR                : {args.target_fpr:.4%}")
    print(f"observed calibration FPR  : {observed_fpr:.4%}")
    print(f"valid BENIGN scores       : {len(scores):,}")
    print(f"checkpoint (new copy)     : {output_path}")
    print(f"calibration report        : {report_path}")
    print("模型權重沒有重新訓練或更動。請用新 checkpoint 在獨立測試集重新執行批次評估。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[錯誤] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
