#!/usr/bin/env python3
# ============================================================
# core/train_pcap_native.py
#
# CIC-IDS2017 / CIC-DDoS2019 原始 PCAP → 封包位元組影像訓練腳本
#
# 解決問題：
#   train/serve skew — 舊版 run_training.py 用 CSV 統計特徵訓練，
#   但推論時用 PacketVisualizer.bytes_to_image() 讀原始位元組，
#   兩者表示法完全不同。本腳本直接讀原始 PCAP，用與推論一致的
#   PacketVisualizer 產生訓練影像，徹底消除 skew。
#
# PCAP 目錄結構：
#   <pcap-dir>/
#     normal/     ← 放入正常流量 PCAP（如 Monday 全天）
#     attack/     ← 放入攻擊流量 PCAP（已篩選的攻擊段落）
#
# 使用方式：
#   python core/train_pcap_native.py \
#     --dataset cicids2017 \
#     --pcap-dir PCAPs/cicids2017-PCAPs \
#     --epochs 300 --patience 25
#
#   python core/train_pcap_native.py \
#     --dataset cicddos2019 \
#     --pcap-dir PCAPs/cicddos2019-PCAPs \
#     --epochs 300 --patience 25
#
# Google Colab 用法：
#   !python core/train_pcap_native.py \
#     --dataset cicids2017 \
#     --pcap-dir /content/drive/MyDrive/PCAPs/cicids2017-PCAPs \
#     --epochs 300 --batch-size 128
# ============================================================

import os
import sys
import json
import time
import glob
import argparse
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_PROJECT_ROOT = os.path.dirname(_THIS_DIR)


def find_pcap_files(directory):
    """遞迴搜尋目錄下所有 .pcap / .pcapng / .cap 檔案"""
    extensions = ['*.pcap', '*.pcapng', '*.cap']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    return sorted(files)


def build_dataset_from_pcap(pcap_dir, output_dir, max_packets_per_file=0):
    """
    讀取 pcap_dir/normal/ 和 pcap_dir/attack/ 下的 PCAP，
    透過 DatasetBuilder 轉換為 X_normal.npy / X_attack.npy
    """
    from dataset_builder import DatasetBuilder

    normal_dir = os.path.join(pcap_dir, 'normal')
    attack_dir = os.path.join(pcap_dir, 'attack')

    if not os.path.isdir(normal_dir):
        raise FileNotFoundError(
            f"找不到 normal 目錄：{normal_dir}\n"
            f"請在 {pcap_dir} 下建立 normal/ 子目錄，放入正常流量 PCAP。"
        )

    normal_pcaps = find_pcap_files(normal_dir)
    attack_pcaps = find_pcap_files(attack_dir) if os.path.isdir(attack_dir) else []

    if not normal_pcaps:
        raise FileNotFoundError(
            f"normal 目錄中沒有 PCAP 檔案：{normal_dir}\n"
            f"支援副檔名：.pcap / .pcapng / .cap"
        )

    print(f"\n[資料集建構] PCAP 目錄：{pcap_dir}")
    print(f"  正常流量 PCAP：{len(normal_pcaps)} 個檔案")
    print(f"  攻擊流量 PCAP：{len(attack_pcaps)} 個檔案")
    if not attack_pcaps:
        print("  [提示] 未找到攻擊流量 PCAP，將僅訓練非監督模型（無法訓練半監督模型）")

    builder = DatasetBuilder(
        output_dir=output_dir,
        image_size="medium",
        apply_mask=True,
        max_packets_per_file=max_packets_per_file,
    )

    # 處理正常流量
    for pcap_path in normal_pcaps:
        print(f"\n  處理正常流量：{os.path.basename(pcap_path)}")
        builder.build_from_pcap(pcap_path, label="normal", save_png=False)

    # 處理攻擊流量
    for pcap_path in attack_pcaps:
        print(f"\n  處理攻擊流量：{os.path.basename(pcap_path)}")
        builder.build_from_pcap(pcap_path, label="attack", save_png=False)

    # 儲存為 numpy
    saved = builder.save_numpy_arrays()
    builder.save_index_csv()

    return saved


def train_vae(X_normal, X_attack, epochs, patience, batch_size,
              latent_dim, pct, output_dir, model_name):
    """訓練非監督式 VAE"""
    from variational_autoencoder import VAETrainer
    import torch

    print("\n" + "═" * 62)
    print(f"  [{model_name}] 非監督式 CNN-VAE（PCAP 原生表示法）")
    print(f"  Epochs 上限：{epochs}（EarlyStopping patience={patience}）")
    print("═" * 62)

    out_dir = os.path.join(output_dir, f"vae_{model_name}")
    os.makedirs(out_dir, exist_ok=True)

    # 切分 85% 訓練 / 15% holdout
    n = len(X_normal)
    idx = np.random.default_rng(42).permutation(n)
    split = max(1, int(n * 0.85))
    X_train = X_normal[idx[:split]]
    X_holdout = X_normal[idx[split:]] if split < n else X_normal[idx[-max(1, n // 20):]:]

    train_npy = os.path.join(out_dir, "_X_train.npy")
    np.save(train_npy, X_train)

    config = {
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": 1e-3,
        "patience": patience,
        "threshold_percentile": pct,
        "kl_weight": 1e-3,
        "val_split": 0.2,
    }
    trainer = VAETrainer(config=config, output_dir=out_dir, model_name=model_name)
    trainer.load_data(train_npy)

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    threshold = trainer.compute_threshold(train_npy, pct)
    trainer.plot_training_curve()

    # 估算分離比
    sep_ratio = 1.0
    if X_attack is not None and len(X_attack) > 0:
        device = trainer.device
        errors_n, errors_a = [], []
        with torch.no_grad():
            for i in range(0, len(X_holdout), 256):
                batch = torch.from_numpy(
                    X_holdout[i:i+256, None].astype(np.float32)).to(device)
                errors_n.append(trainer.model.reconstruction_error(batch).cpu().numpy())
            for i in range(0, len(X_attack), 256):
                batch = torch.from_numpy(
                    X_attack[i:i+256, None].astype(np.float32)).to(device)
                errors_a.append(trainer.model.reconstruction_error(batch).cpu().numpy())
        errors_n = np.concatenate(errors_n) if errors_n else np.array([0.0])
        errors_a = np.concatenate(errors_a) if errors_a else np.array([0.0])
        sep_ratio = float(errors_a.mean() / (errors_n.mean() + 1e-9))

    epochs_ran = len(trainer.train_losses)
    ckpt_path = trainer._path()

    print(f"  訓練完成：{epochs_ran} epochs，耗時 {train_time:.1f}s")
    print(f"  閾值（{pct}th pct）：{threshold:.6f}")
    print(f"  分離比：{sep_ratio:.2f}x")

    return {
        "model_key": f"semi_{model_name}",
        "checkpoint": ckpt_path,
        "epochs_ran": epochs_ran,
        "train_time_sec": round(train_time, 1),
        "threshold": threshold,
        "separability_ratio": round(sep_ratio, 4),
    }


def train_semi(X_normal, X_attack, pretrain_epochs, finetune_epochs,
               batch_size, latent_dim, output_dir, model_name):
    """訓練半監督式 Hybrid 模型"""
    from hybrid_semi_supervised import HybridSemiSupervisedTrainer

    print("\n" + "═" * 62)
    print(f"  [semi_{model_name}] 半監督 CNN-VAE + CNN-LSTM（PCAP 原生表示法）")
    print(f"  Pretrain：{pretrain_epochs}  Finetune：{finetune_epochs}")
    print("═" * 62)

    out_dir = os.path.join(output_dir, f"semi_{model_name}")
    os.makedirs(out_dir, exist_ok=True)

    n = len(X_normal)
    idx = np.random.default_rng(42).permutation(n)
    split = max(1, int(n * 0.85))
    X_train = X_normal[idx[:split]]

    config = {
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "pretrain_epochs": pretrain_epochs,
        "finetune_epochs": finetune_epochs,
        "learning_rate": 1e-3,
        "attack_ratio": 0.3,
        "unknown_percentile": 95,
    }
    trainer = HybridSemiSupervisedTrainer(
        config=config, output_dir=out_dir, model_name=model_name)

    t0 = time.time()
    threshold = trainer.train_full(X_train, X_attack, threshold_method="optimal")
    train_time = time.time() - t0

    print(f"  訓練完成，耗時 {train_time:.1f}s，閾值：{threshold:.6f}")

    ckpt_path = os.path.join(out_dir, f"semi_{model_name}.pt")
    return {
        "model_key": f"semi_{model_name}",
        "checkpoint": ckpt_path,
        "train_time_sec": round(train_time, 1),
        "threshold": float(threshold),
    }


def main():
    parser = argparse.ArgumentParser(
        description="用 CIC-IDS2017 / CIC-DDoS2019 原始 PCAP 訓練異常偵測模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python core/train_pcap_native.py --dataset cicids2017 --pcap-dir PCAPs/cicids2017-PCAPs
  python core/train_pcap_native.py --dataset cicddos2019 --pcap-dir PCAPs/cicddos2019-PCAPs
  python core/train_pcap_native.py --dataset cicids2017 --pcap-dir PCAPs/cicids2017-PCAPs --only vae
        """,
    )

    parser.add_argument("--dataset", required=True,
                        choices=["cicids2017", "cicddos2019"],
                        help="資料集名稱")
    parser.add_argument("--pcap-dir", required=True,
                        help="PCAP 目錄（需含 normal/ 子目錄，可選 attack/ 子目錄）")
    parser.add_argument("--only", choices=["all", "vae", "semi"], default="all",
                        help="只訓練指定類型（預設 all）")

    # 資料集
    parser.add_argument("--max-packets", type=int, default=0,
                        help="每個 PCAP 最多處理多少封包（0=全部）")

    # 訓練參數
    parser.add_argument("--epochs", type=int, default=300,
                        help="VAE 訓練 epoch 上限（EarlyStopping 可提前停止）")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--pct", type=float, default=95.0,
                        help="異常閾值百分位數")
    parser.add_argument("--semi-pretrain-epochs", type=int, default=150)
    parser.add_argument("--semi-finetune-epochs", type=int, default=60)

    # 輸出
    parser.add_argument("--output", default=None,
                        help="輸出目錄（預設 output/model_<dataset>_pcap_native）")
    parser.add_argument("--no-deploy", action="store_true",
                        help="不自動部署到 media/model/")

    args = parser.parse_args()

    if args.output is None:
        args.output = f"output/model_{args.dataset}_pcap_native"

    print("\n" + "═" * 62)
    print(f"  PCAP 原生訓練 — {args.dataset}")
    print(f"  PCAP 目錄：{os.path.abspath(args.pcap_dir)}")
    print("═" * 62)

    # ── Step 1：建構資料集 ──────────────────────────────────
    dataset_dir = os.path.join(args.output, "dataset")
    normal_npy = os.path.join(dataset_dir, "X_normal.npy")
    attack_npy = os.path.join(dataset_dir, "X_attack.npy")

    if os.path.exists(normal_npy):
        print(f"\n[Step 1] 使用既有資料集：{dataset_dir}")
        X_normal = np.load(normal_npy).astype(np.float32)
        X_attack = np.load(attack_npy).astype(np.float32) if os.path.exists(attack_npy) else None
    else:
        print(f"\n[Step 1] 從 PCAP 建構資料集...")
        build_dataset_from_pcap(args.pcap_dir, dataset_dir, args.max_packets)
        X_normal = np.load(normal_npy).astype(np.float32)
        X_attack = np.load(attack_npy).astype(np.float32) if os.path.exists(attack_npy) else None

    print(f"  X_normal: {X_normal.shape}")
    if X_attack is not None:
        print(f"  X_attack: {X_attack.shape}")
    else:
        print("  X_attack: 無（僅訓練非監督模型）")

    summary = {"meta": vars(args), "results": {}}
    t_start = time.time()

    # ── Step 2：訓練 VAE ───────────────────────────────────
    if args.only in ("all", "vae"):
        res = train_vae(
            X_normal, X_attack,
            epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size, latent_dim=args.latent_dim,
            pct=args.pct, output_dir=args.output,
            model_name=args.dataset,
        )
        summary["results"]["vae"] = res
        if not args.no_deploy:
            from deploy_model import deploy
            # 部署為對應的半監督模型 key
            key = f"semi_{args.dataset}"
            try:
                deploy(key, res["checkpoint"], _PROJECT_ROOT)
                print(f"  [部署] {key} → media/model/")
            except Exception as e:
                print(f"  [部署失敗] {e}")

    # ── Step 3：訓練半監督模型（需要攻擊資料）────────────────
    if args.only in ("all", "semi") and X_attack is not None and len(X_attack) >= 50:
        res = train_semi(
            X_normal, X_attack,
            pretrain_epochs=args.semi_pretrain_epochs,
            finetune_epochs=args.semi_finetune_epochs,
            batch_size=args.batch_size, latent_dim=args.latent_dim,
            output_dir=args.output, model_name=args.dataset,
        )
        summary["results"]["semi"] = res
        if not args.no_deploy:
            from deploy_model import deploy
            key = f"semi_{args.dataset}"
            try:
                deploy(key, res["checkpoint"], _PROJECT_ROOT)
                print(f"  [部署] {key} → media/model/")
            except Exception as e:
                print(f"  [部署失敗] {e}")
    elif args.only in ("all", "semi") and (X_attack is None or len(X_attack) < 50):
        print("\n[跳過] 半監督模型：攻擊資料不足（需要至少 50 筆）")

    total_time = time.time() - t_start

    # ── 摘要 ──────────────────────────────────────────────
    summary_path = os.path.join(args.output, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "═" * 62)
    print(f"  訓練完成，總耗時：{total_time/60:.1f} 分鐘")
    print(f"  摘要報告：{os.path.abspath(summary_path)}")
    print("═" * 62)


if __name__ == "__main__":
    main()
