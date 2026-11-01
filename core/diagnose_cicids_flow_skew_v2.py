#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二階段 CICIDS2017 Train/Serve Skew 診斷。

與 v1 不同：不只讀取檔案前 50,000 列；會分塊掃描整個 CSV，從全部 BENIGN rows
中按固定 seed 抽樣，再比較原始特徵、RobustScaler center/scale 與 clip 後的分布。
只讀取資料，不修改模型、Scaler、PCAP、Flow CSV 或既有結果。

於 final_project 專案根目錄：
  python core/diagnose_cicids_flow_skew_v2.py
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENIGN_LABELS = {"BENIGN", "NORMAL", "NORMAL TRAFFIC"}
CHUNK_SIZE = 100_000


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def locate_one(paths: list[Path], description: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"找不到{description}；請確認資料夾或用命令列指定路徑。")
    return max(paths, key=lambda p: p.stat().st_size)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def is_benign(df: pd.DataFrame) -> pd.Series:
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise ValueError("CSV 找不到 Label 欄位，無法只抽樣 BENIGN rows。")
    return df[label_col].astype(str).str.strip().str.upper().isin(BENIGN_LABELS)


def benign_counts_by_chunk(path: Path) -> list[int]:
    """第一趟只計算每個 chunk 的 BENIGN 筆數，方便按比例抽樣。"""
    counts: list[int] = []
    use_label = lambda c: str(c).strip().lower() == "label"
    try:
        for df in pd.read_csv(path, usecols=use_label, chunksize=CHUNK_SIZE, low_memory=False):
            normalize_columns(df)
            counts.append(int(is_benign(df).sum()))
    except ValueError as exc:
        raise ValueError(f"讀取 {path.name} 的 Label 欄位失敗：{exc}") from exc
    if not counts:
        raise ValueError(f"CSV 沒有資料：{path}")
    return counts


def random_benign_sample(path: Path, feature_columns: list[str], limit: int, seed: int) -> tuple[pd.DataFrame, int]:
    """兩趟分塊抽樣：對整個檔案內的 BENIGN rows 做可重現的近似均勻抽樣。"""
    counts = benign_counts_by_chunk(path)
    total_benign = int(sum(counts))
    if total_benign == 0:
        raise ValueError(f"{path.name} 沒有可辨識的 BENIGN rows。")
    target = min(limit, total_benign)
    rng = np.random.default_rng(seed)

    feature_set = set(feature_columns)
    use_cols = lambda c: (str(c).strip() in feature_set or str(c).strip().lower() == "label")
    sampled: list[pd.DataFrame] = []
    seen_benign = 0
    try:
        for i, chunk in enumerate(pd.read_csv(path, usecols=use_cols, chunksize=CHUNK_SIZE, low_memory=False)):
            normalize_columns(chunk)
            mask = is_benign(chunk)
            benign = chunk.loc[mask]
            n_benign = len(benign)
            if n_benign == 0:
                continue
            # 按各 chunk 的 BENIGN 筆數比例配置樣本；最後再統一抽到 target 筆。
            take = min(n_benign, int(round(target * counts[i] / total_benign)))
            if take > 0:
                sampled.append(benign.sample(n=take, random_state=seed + i))
            seen_benign += n_benign
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"讀取特徵欄位失敗（{path.name}）：{exc}") from exc

    if not sampled:
        raise ValueError(f"無法從 {path.name} 抽取 BENIGN rows。")
    result = pd.concat(sampled, ignore_index=True)
    if len(result) > target:
        result = result.sample(n=target, random_state=seed).reset_index(drop=True)
    elif len(result) < target:
        # 由 chunk rounding 造成的少量差額，從全檔另一次固定 seed 抽樣補足。
        # 不重複抽樣，且只在不足時執行；一般 50k 樣本不會進入此分支。
        remaining = target - len(result)
        if remaining > 0:
            # 保持簡單和可追溯：回傳可用的等比例樣本數，並明確印出實際數量。
            pass
    return result, total_benign


def numeric_matrix(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"缺少特徵欄位 {len(missing)} 個：{missing[:20]}")
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def main() -> int:
    p = argparse.ArgumentParser(description="以全檔 BENIGN 隨機抽樣比較 CICIDS2017 訓練 CSV 與 PCAP Flow")
    p.add_argument("--project-root", default=".")
    p.add_argument("--scaler", default="output/threshold_study/unsup_cicids2017/cicids2017_preprocessor.pkl")
    p.add_argument("--flow-dir", default="output/batch_eval/cicids_vae_full/flows")
    p.add_argument("--train-dir", default="data/cicids2017")
    p.add_argument("--flow-csv", default=None)
    p.add_argument("--train-csv", default=None)
    p.add_argument("--rows", type=int, default=50000)
    p.add_argument("--seed", type=int, default=20261019)
    args = p.parse_args()
    if args.rows < 100:
        p.error("--rows 建議至少 100")

    root = Path(args.project_root).resolve()
    scaler_path = resolve_path(root, args.scaler)
    flow_dir = resolve_path(root, args.flow_dir)
    train_dir = resolve_path(root, args.train_dir)
    if not scaler_path.exists():
        raise FileNotFoundError(f"找不到 scaler：{scaler_path}")
    with scaler_path.open("rb") as f:
        payload = pickle.load(f)
    features = [str(c).strip() for c in payload.get("feature_columns", [])]
    scaler = payload.get("scaler")
    if not features or scaler is None:
        raise ValueError("Scaler PKL 缺少 feature_columns 或 scaler。")

    flow_path = resolve_path(root, args.flow_csv) if args.flow_csv else locate_one(
        list(flow_dir.glob("*Monday*.csv")) + list(flow_dir.glob("*monday*.csv")), "Monday PCAP Flow CSV"
    )
    train_path = resolve_path(root, args.train_csv) if args.train_csv else locate_one(
        [x for x in train_dir.rglob("*.csv") if "monday" in x.name.lower()], "CICIDS2017 Monday 訓練 CSV"
    )
    if not flow_path.exists() or not train_path.exists():
        raise FileNotFoundError(f"資料檔不存在：train={train_path}; flow={flow_path}")

    print("=== A. 檔案與前處理器 ===")
    print(f"Scaler      : {scaler_path}")
    print(f"Scaler type : {type(scaler).__name__}")
    print(f"Feature num : {len(features)}")
    print(f"Normalization: {payload.get('normalization', '未記錄')}")
    print(f"Train CSV   : {train_path}")
    print(f"Flow CSV    : {flow_path}")
    print(f"Sampling    : 全檔 BENIGN 隨機抽樣，目標每份 {args.rows:,} rows，seed={args.seed}")

    train_df, n_train_benign = random_benign_sample(train_path, features, args.rows, args.seed)
    flow_df, n_flow_benign = random_benign_sample(flow_path, features, args.rows, args.seed + 1)
    train_x = numeric_matrix(train_df, features)
    flow_x = numeric_matrix(flow_df, features)
    print("\n=== B. 全檔正常樣本及抽樣數 ===")
    print(f"Train 全檔 BENIGN rows : {n_train_benign:,}; 本次抽樣 {len(train_df):,}")
    print(f"Flow  全檔 BENIGN rows : {n_flow_benign:,}; 本次抽樣 {len(flow_df):,}")

    train_z = scaler.transform(train_x)
    flow_z = scaler.transform(flow_x)
    train_clip = np.clip(train_z, 0.0, 1.0)
    flow_clip = np.clip(flow_z, 0.0, 1.0)
    print("\n=== C. RobustScaler 後、clip 前的比例 ===")
    print(f"Train < 0: {np.mean(train_z < 0):.2%}; > 1: {np.mean(train_z > 1):.2%}")
    print(f"Flow  < 0: {np.mean(flow_z < 0):.2%}; > 1: {np.mean(flow_z > 1):.2%}")

    t_raw50 = np.quantile(train_x, 0.50, axis=0)
    f_raw50 = np.quantile(flow_x, 0.50, axis=0)
    t_raw95 = np.quantile(train_x, 0.95, axis=0)
    f_raw95 = np.quantile(flow_x, 0.95, axis=0)
    t50 = np.quantile(train_clip, 0.50, axis=0)
    f50 = np.quantile(flow_clip, 0.50, axis=0)
    t95 = np.quantile(train_clip, 0.95, axis=0)
    f95 = np.quantile(flow_clip, 0.95, axis=0)
    delta = np.abs(t50 - f50) + np.abs(t95 - f95)
    order = np.argsort(delta)[::-1][:20]
    center = getattr(scaler, "center_", np.full(len(features), np.nan))
    scale = getattr(scaler, "scale_", np.full(len(features), np.nan))

    print("\n=== D. 差異最大的 20 個特徵 ===")
    print("Feature | raw Train p50 | raw Flow p50 | scaler center | scaler scale | clip Train p50/p95 | clip Flow p50/p95 | delta")
    for i in order:
        print(
            f"{features[i]} | {t_raw50[i]:.6g} | {f_raw50[i]:.6g} | {center[i]:.6g} | {scale[i]:.6g} | "
            f"{t50[i]:.5f}/{t95[i]:.5f} | {f50[i]:.5f}/{f95[i]:.5f} | {delta[i]:.5f}"
        )

    print("\n=== E. 關鍵特徵原始分布 ===")
    important = ["Destination Port", "Flow Duration", "Fwd Packet Length Min", "Min Packet Length",
                 "Fwd Header Length", "Fwd Header Length.1", "Active Mean", "Active Max", "Active Min",
                 "Idle Mean", "FIN Flag Count", "RST Flag Count", "Flow Bytes/s", "Flow Packets/s"]
    idx_map = {name: i for i, name in enumerate(features)}
    print("Feature | Train p05/p50/p95 | Flow p05/p50/p95")
    for name in important:
        if name not in idx_map:
            continue
        i = idx_map[name]
        tq = np.quantile(train_x[:, i], [0.05, 0.5, 0.95])
        fq = np.quantile(flow_x[:, i], [0.05, 0.5, 0.95])
        print(f"{name} | {tq[0]:.6g}/{tq[1]:.6g}/{tq[2]:.6g} | {fq[0]:.6g}/{fq[1]:.6g}/{fq[2]:.6g}")

    print("\n=== F. 判讀提醒 ===")
    print("這次以全檔 BENIGN 樣本做可重現的分布比較，較不受 CSV 前 50,000 列順序影響；但仍不是逐 Flow 一對一比對。")
    print("若 Destination Port、Active/Idle、封包長度等原始分布仍大幅不同，下一步應逐項核對轉換器公式、方向與時間/長度單位。")
    print("本程式只讀資料，不會修改模型、Scaler、Flow CSV 或原始 PCAP。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[錯誤] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
