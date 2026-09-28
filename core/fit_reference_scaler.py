# ============================================================
# core/fit_reference_scaler.py
# 匯出「訓練期使用的正規化器（scaler）與特徵欄位名單」
#
# 為什麼需要這支腳本：
#   core/dataset_loader.py 的 CICIDSLoader / CICDDoS2019Loader 在
#   訓練時會用「當次載入的正常流量」現場 fit 一個 MinMaxScaler，
#   但這個 scaler 物件只存在於 loader 的記憶體中，訓練結束後從未
#   被保存到磁碟。
#
#   若之後想替 convert_test_pcap.py 轉換出的測試流量特徵計分，
#   若直接把這批新資料丟進 DatasetFactory 重新 fit 一個 scaler，
#   等於是用測試資料自己的統計量正規化自己 —— 這會產生新一輪的
#   train/serve 不一致（值域基準不同於訓練時的基準），使計分結果
#   失去比較意義。
#
#   本腳本刻意「不修改」core/dataset_loader.py 本身（訓練流程完全
#   維持原樣），而是在旁路重新掃描一次原始訓練 CSV，重現與該檔案
#   幾乎相同的前處理邏輯（CSV 讀取 → 標籤欄位偵測 → 正常流量篩選
#   → 數值欄位選取 → MinMaxScaler.fit），取得與訓練時同一份 scaler
#   設定與特徵欄位順序，序列化保存供 score_converted_pcap.py 使用。
#
# 用法：
#   python core/fit_reference_scaler.py \
#       --dataset cicids2017 --data-dir data/cicids2017 \
#       --output  output/scalers/cicids2017_scaler.pkl
#
#   python core/fit_reference_scaler.py \
#       --dataset cicddos2019 --data-dir data/cicddos2019 \
#       --output  output/scalers/cicddos2019_scaler.pkl
#
# 注意：
#   請使用「與當初訓練模型時完全相同」的 --data-dir，才能重現出
#   同一份 scaler。若訓練資料之後有更新，請重新執行本腳本，
#   否則舊 scaler 與新模型會不一致。
# ============================================================

import os
import sys
import glob
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

_NORMAL_LABELS = {"BENIGN", "NORMAL", "NORMAL TRAFFIC"}


def _find_label_column(columns) -> str:
    candidates = {"label", "class", "attack", "attack_type", "attack type", "target"}
    for col in columns:
        if str(col).strip().lower() in candidates:
            return col
    raise ValueError(f"找不到標籤欄位，現有欄位: {list(columns)[:10]}")


def _load_normal_matrix(data_dir: str, recursive: bool):
    pattern = os.path.join(data_dir, "**", "*.csv") if recursive else os.path.join(data_dir, "*.csv")
    csv_files = sorted(glob.glob(pattern, recursive=recursive))
    if not csv_files:
        raise FileNotFoundError(f"在 {data_dir} 找不到任何 CSV（無法重現訓練期前處理）")

    print(f"  找到 {len(csv_files)} 個 CSV 檔案")
    frames = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, low_memory=False)
            df.columns = df.columns.str.strip()
            frames.append(df)
            print(f"    載入: {os.path.basename(path)}  ({len(df):,} 筆)")
        except Exception as e:
            print(f"    [警告] 無法載入 {path}: {e}")

    data = pd.concat(frames, ignore_index=True)
    label_col = _find_label_column(data.columns)

    normal_mask = data[label_col].astype(str).str.strip().str.upper().isin(_NORMAL_LABELS)
    df_normal = data[normal_mask].copy()
    print(f"  正常流量筆數: {len(df_normal):,} / {len(data):,}  （標籤欄位: {label_col!r}）")

    drop_cols = [c for c in df_normal.columns if c.strip() in ("Label", "label") or c == label_col]
    df_normal = df_normal.drop(columns=drop_cols, errors="ignore")
    df_numeric = df_normal.select_dtypes(include=[np.number])

    X = df_numeric.values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, list(df_numeric.columns)


def main():
    parser = argparse.ArgumentParser(description="匯出訓練期 scaler，供轉換後的測試 PCAP 特徵重複使用")
    parser.add_argument("--dataset", required=True, choices=["cicids2017", "cicddos2019"])
    parser.add_argument("--data-dir", required=True, help="原始訓練 CSV 目錄（需與訓練模型時相同）")
    parser.add_argument("--output", required=True, help="輸出 .pkl 路徑")
    parser.add_argument("--image-size", type=int, default=32,
                        help="訓練時使用的 CNN 輸入影像邊長（需與模型的 latent/image 設定一致，預設 32）")
    args = parser.parse_args()

    print(f"[fit_reference_scaler] 重現 {args.dataset} 訓練期前處理...")
    X_normal, feature_columns = _load_normal_matrix(
        args.data_dir, recursive=(args.dataset == "cicddos2019")
    )
    print(f"  數值特徵欄位數: {len(feature_columns)}")

    scaler = MinMaxScaler()
    scaler.fit(X_normal)

    payload = {
        "dataset": args.dataset,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "image_size": args.image_size,
    }
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(payload, f)

    print(f"[fit_reference_scaler] 已儲存: {args.output}")
    print("  請留存此檔案對應的資料集版本紀錄；訓練資料更新後請重新產生。")


if __name__ == "__main__":
    main()
