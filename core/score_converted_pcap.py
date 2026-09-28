# ============================================================
# core/score_converted_pcap.py
# 使用「訓練期 scaler」與既有已訓練模型，為 convert_test_pcap.py
# 產生的流量特徵 CSV 計算異常分數。
#
# 前置需求：
#   1. 已用 convert_test_pcap.py 把測試 PCAP 轉成流量特徵 CSV
#   2. 已用 fit_reference_scaler.py 匯出對應資料集的 scaler .pkl
#   3. 已有一個以該資料集 CSV 訓練好的模型 checkpoint
#      （settings.ANOMALY_MODELS 中的 semi_cicids2017.pt /
#      semi_cicddos2019.pt，或自行以 run_training.py 訓練的模型）
#
# 用法：
#   python core/score_converted_pcap.py \
#       --features output/test_flows/wed_cicids2017.csv \
#       --scaler   output/scalers/cicids2017_scaler.pkl \
#       --model    media/model/semi_cicids2017.pt \
#       --output   output/test_flows/wed_cicids2017_scored.csv
#
# 輸出：
#   在原始特徵 CSV 的每一列後方附加 anomaly_score / is_anomaly
#   （hybrid_semi 模型另附加 status 欄位：known_attack / unknown_attack / normal），
#   另印出整體摘要統計。
# ============================================================

import os
import sys
import argparse
import pickle

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import pandas as pd
import torch

from cnn_autoencoder import features_to_image
from model_registry import load_anomaly_model, compute_anomaly_scores


def main():
    parser = argparse.ArgumentParser(description="為轉換後的測試流量特徵計分（沿用訓練期 scaler，維持 train/serve 一致）")
    parser.add_argument("--features", required=True, help="convert_test_pcap.py 輸出的 CSV")
    parser.add_argument("--scaler", required=True, help="fit_reference_scaler.py 輸出的 .pkl")
    parser.add_argument("--model", required=True, help="已訓練模型 checkpoint 路徑")
    parser.add_argument("--output", required=True, help="輸出含分數的 CSV 路徑")
    parser.add_argument("--latent-dim", type=int, default=32,
                        help="模型 checkpoint 未內含 config 時的備援 latent_dim")
    args = parser.parse_args()

    with open(args.scaler, "rb") as f:
        payload = pickle.load(f)
    scaler = payload["scaler"]
    feature_cols = payload["feature_columns"]
    image_size = payload.get("image_size", 32)

    df = pd.read_csv(args.features)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  [警告] 轉換後的 CSV 缺少 {len(missing)} 個訓練期使用過的欄位，將以 0 補齊："
              f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")
        for c in missing:
            df[c] = 0.0

    X_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float64)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # 與訓練時「攻擊/測試資料」路徑相同：只 transform、不 clip。
    # 測試資料的值可能超出訓練時觀察到的值域，這是預期行為
    # （超出範圍本身就是有意義的異常訊號，不應該被裁切掉）。
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    # 與 dataset_loader.py 內部相同的 pad/截斷邏輯（features_to_image 內建）
    X_img = features_to_image(X_scaled, image_size)  # (N, 1, H, W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_anomaly_model(args.model, device=device, default_latent_dim=args.latent_dim)
    scored = compute_anomaly_scores(bundle, X_img.squeeze(1))

    df["anomaly_score"] = scored["score"]
    df["is_anomaly"] = scored["is_anomaly"]
    if bundle.is_hybrid:
        df["status"] = scored["status"]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output, index=False)

    n_total = len(df)
    n_anom = int(df["is_anomaly"].sum())
    print(f"[score_converted_pcap] 完成，共 {n_total:,} 條 Flow 記錄")
    print(f"  模型判定異常: {n_anom:,} 筆（{n_anom / max(n_total, 1) * 100:.1f}%）")
    print(f"  模型輸入表示法: {bundle.input_representation}")
    if bundle.input_representation != "csv_features_legacy":
        print("  [警告] 此模型並非以 CSV 統計特徵訓練，與本腳本產生的流量特徵不匹配，"
              "分數不具參考意義（請確認 --model 對應的是 CSV 訓練管線產生的 checkpoint）。")
    if "Label" in df.columns and df["Label"].nunique() > 1:
        summary = df.groupby("Label")["is_anomaly"].mean().sort_values(ascending=False)
        print("  各標籤的模型判定異常比例：")
        for label, rate in summary.items():
            print(f"    {label:<12} {rate * 100:6.1f}%")
    print(f"  輸出: {args.output}")


if __name__ == "__main__":
    main()
