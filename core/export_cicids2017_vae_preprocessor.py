#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild the CICIDS2017 unsupervised VAE preprocessor without retraining.

IMPORTANT: This reproduces the scaler-fitting pipeline used by
core/run_training_cicids2017_vae_fixed.py. It is only equivalent to the
checkpoint's training scaler if the original CSV files and training options
(seed, sample limits, split ratios) are identical.
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import RobustScaler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="匯出非監督式 CICIDS2017 CNN-VAE 的 RobustScaler 前處理器"
    )
    parser.add_argument("--data-dir", default="data/cicids2017")
    parser.add_argument(
        "--output",
        default="output/model_cicids2017_vae_fixed/cicids2017_preprocessor.pkl",
    )
    parser.add_argument("--max-normal", type=int, default=60000)
    parser.add_argument("--max-attack", type=int, default=30000)
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_normal < 1 or args.max_attack < 1:
        parser.error("--max-normal 與 --max-attack 必須大於 0")
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        parser.error("必須符合 train_ratio > 0、val_ratio >= 0、train_ratio + val_ratio < 1")

    # Resolve the project root whether this script is placed in project root or core/.
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "core" / "run_training_cicids2017_vae_fixed.py").exists():
        project_root = script_dir
        core_dir = project_root / "core"
    elif (script_dir / "run_training_cicids2017_vae_fixed.py").exists():
        core_dir = script_dir
        project_root = script_dir.parent
    else:
        raise FileNotFoundError(
            "找不到 core/run_training_cicids2017_vae_fixed.py。請把此腳本放到專案根目錄或 core/ 目錄。"
        )
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))

    # Reuse the exact feature selection, cleaning, sampling and split functions
    # from the training implementation rather than fitting an unrelated scaler.
    from run_training_cicids2017_vae_fixed import load_dataset, split_three_way, IMAGE_SIZE

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = project_root / out_path

    print("[1/4] 使用訓練腳本相同的 CSV 載入、欄位選取、清理與取樣流程…")
    X_normal_raw, _X_attack_raw, feature_columns = load_dataset(
        str(data_dir), args.max_normal, args.max_attack, args.seed
    )
    print(f"[2/4] 正常樣本總數：{len(X_normal_raw):,}；特徵欄位：{len(feature_columns)}")
    print("[3/4] 使用相同 train-ratio/val-ratio/seed 重現正常樣本切分，僅以 Train-Normal fit RobustScaler…")
    X_train_normal, _X_val_normal, _X_test_normal = split_three_way(
        X_normal_raw, args.train_ratio, args.val_ratio, args.seed
    )
    scaler = RobustScaler()
    scaler.fit(X_train_normal)

    payload = {
        "dataset": "cicids2017",
        "scaler": scaler,
        "feature_columns": feature_columns,
        "image_size": IMAGE_SIZE,
        "scaler_type": "RobustScaler",
        "normalization": "clip_0_1",
        "model_type": "cnn_vae",
        "representation": "csv_features_cicids2017",
        "training_data_dir": str(data_dir.resolve()),
        "max_normal": int(args.max_normal),
        "max_attack": int(args.max_attack),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "note": (
            "重建的 scaler 僅在原始 CSV、清理後的特徵、取樣參數、seed、切分比例及訓練程式一致時，"
            "才可視為與既有 checkpoint 訓練時的前處理器一致。"
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[4/4] 已儲存：{out_path}")
    print(f"  scaler：{type(scaler).__name__}")
    print(f"  欄位數：{len(feature_columns)}")
    print(f"  Image size：{IMAGE_SIZE}x{IMAGE_SIZE}")
    print("提醒：此工具不會重新訓練或修改 .pt。若訓練時參數或 CSV 版本不同，請不要把重建檔當作完全相同的 scaler。")


if __name__ == "__main__":
    main()
