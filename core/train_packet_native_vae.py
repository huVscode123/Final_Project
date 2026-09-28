#!/usr/bin/env python3
# ============================================================
# core/train_packet_native_vae.py（新增檔案 — 補上缺失的最後一塊拼圖）
#
# ── 這支腳本要解決什麼 ──────────────────────────────────────
# core/model_registry.py 的 clear_model_cache() docstring 已經寫著：
#     「若你透過 core/train_packet_native_vae.py 重新訓練並覆蓋了
#       media/model/best_vae_model.pt ...」
# core/packet_dataset_builder.py 的檔頭註解也詳細診斷出根因：
#   - 正式訓練腳本（run_training.py / run_semi_supervised.py）讀的是
#     core/dataset_loader.py::DatasetFactory，也就是 CIC-IDS2017／
#     NSL-KDD 的「CSV 統計特徵」reshape 成的影像。
#   - 但正式 PCAP 分析（analyzer/tasks.py）與模擬檢測
#     （analyzer/views.py::simulation_api）餵給模型「推論」時，用的是
#     core/packet_visualizer.py::PacketVisualizer 把「封包原始位元組」
#     轉成的影像。
#   - 兩者是完全不同的資料分布（train/serve skew），這才是「CNN／VAE
#     分數看起來不合理、閾值看起來抓不準」的根本原因 —— 不是閾值算
#     錯、也不是 epoch 數不夠，而是模型從頭到尾就沒有學過怎麼重建
#     「封包位元組影像」。
#   - packet_dataset_builder.py 已經把「正確表示法」的資料集做出來了
#     （呼叫 core/simulate_anomaly_traffic.py 的封包產生器 + 與推論時
#     完全相同的 PacketVisualizer），但專案裡從缺「拿這份資料集去
#     訓練模型」這一步的腳本本身 —— 也就是本檔案。
#
# 換句話說：「目前所有模型只訓練 50 次」這個症狀，藥不在「調高
# epoch 數」，而在於本檔案：先用對的資料表示法訓練，epoch 數才開始
# 有意義。這裡順便把 epoch 上限也一併拉高（並非盲目調大，而是搭配
# 既有的 EarlyStopping／驗證集監控機制，讓模型練到收斂才停，而不是
# 練不夠就被 epoch 上限硬性打斷）。
#
# ── 這支腳本做的事 ──────────────────────────────────────────
#   Step 1  呼叫 core/packet_dataset_builder.py 建立（或讀取已存在的）
#           X_normal.npy / X_attack.npy —— 與 analyzer/tasks.py、
#           analyzer/views.py::simulation_api() 推論時完全相同的
#           「封包位元組影像」表示法。
#   Step 2  用 core/variational_autoencoder.py::VAETrainer 訓練
#           settings.ANOMALY_MODELS['unsupervised_vae'] 對應的
#           非監督式 CNN-VAE（epoch 上限 300，EarlyStopping 已存在
#           於 VAETrainer.train() 內，patience 可調，預設 25）。
#   Step 3  用 core/hybrid_semi_supervised.py::HybridSemiSupervisedTrainer
#           訓練 settings.ANOMALY_MODELS 中 3 個半監督模型
#           （semi_cicddos2019 / semi_cicids2017 / semi_nslkdd），
#           pretrain/finetune epoch 上限由原本各腳本預設的 50 拉高至
#           150 / 60（此類別目前沒有內建 EarlyStopping，故不像 VAE
#           一樣直接調到很高，避免單次訓練時間過度膨脹；如需要更精確
#           的提前停止，可在 hybrid_semi_supervised.py 補上驗證集監控
#           後再調高，這是本次範圍外的後續優化建議）。
#   Step 4  每個模型訓練完成後自動呼叫既有的 core/deploy_model.py，
#           部署到 media/model/ 對應路徑，不需要再手動複製改檔名。
#   Step 5  輸出 training_summary.json，記錄每個模型的閾值、分離比、
#           訓練耗時，方便追蹤「這次重訓有沒有變好」。
#
# ── 使用方式 ────────────────────────────────────────────────
#   # 全部 4 個模型（資料集不存在時會先自動建立）
#   python core/train_packet_native_vae.py
#
#   # 只訓練其中一個
#   python core/train_packet_native_vae.py --only unsupervised_vae
#   python core/train_packet_native_vae.py --only semi_cicids2017
#
#   # 資料集已經建立過，跳過重建（省時間）
#   python core/train_packet_native_vae.py --skip-build
#
#   # 資料集想強制重新產生（例如 simulate_anomaly_traffic.py 有更新）
#   python core/train_packet_native_vae.py --rebuild
#
#   # 調整訓練規模
#   python core/train_packet_native_vae.py \
#       --vae-epochs 300 --vae-patience 25 \
#       --semi-pretrain-epochs 150 --semi-finetune-epochs 60
#
#   # 不要自動部署到 media/model/（只想看訓練結果）
#   python core/train_packet_native_vae.py --no-deploy
#
# 訓練完成後，若 Django / Celery worker 是長駐執行的行程，記得依
# core/model_registry.py::clear_model_cache() 的說明清除行程內快取，
# 或直接重新啟動伺服器，新模型才會生效。
# ============================================================

import os
import sys
import json
import time
import argparse
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_PROJECT_ROOT_DEFAULT = os.path.dirname(_THIS_DIR)

DATASET_DIR_DEFAULT = os.path.join("output", "dataset_packet_native")
OUTPUT_ROOT_DEFAULT = os.path.join("output", "model_packet_native")

SEMI_MODEL_NAMES = ["cicddos2019", "cicids2017", "nslkdd"]
ALL_MODEL_KEYS = ["unsupervised_vae"] + [f"semi_{n}" for n in SEMI_MODEL_NAMES]


# ──────────────────────────────────────────────────────────
# Step 1：資料集建立 / 讀取
# ──────────────────────────────────────────────────────────
def _ensure_dataset(dataset_dir: str, rebuild: bool, **build_kwargs):
    """
    確保 dataset_dir 下存在 X_normal.npy / X_attack.npy（封包位元組影像
    表示法）。不存在或 rebuild=True 時，呼叫 packet_dataset_builder.py
    重新產生 —— 產生邏輯與 analyzer 推論時使用的 PacketVisualizer 完全
    一致，這正是本次要修正的 train/serve 表示法不一致問題的解方。
    """
    normal_path = os.path.join(dataset_dir, "X_normal.npy")
    attack_path = os.path.join(dataset_dir, "X_attack.npy")

    if rebuild or not (os.path.exists(normal_path) and os.path.exists(attack_path)):
        from packet_dataset_builder import build_dataset
        print(f"\n[train_packet_native_vae] 建立封包位元組原生資料集：{dataset_dir}")
        build_dataset(output_dir=dataset_dir, **build_kwargs)
    else:
        print(f"\n[train_packet_native_vae] 使用既有資料集（--rebuild 可強制重建）：{dataset_dir}")

    X_normal = np.load(normal_path).astype(np.float32)
    X_attack = np.load(attack_path).astype(np.float32)
    print(f"  X_normal: {X_normal.shape}   X_attack: {X_attack.shape}")

    if len(X_normal) < 50 or len(X_attack) < 50:
        raise RuntimeError(
            f"資料集樣本數過少（正常={len(X_normal)}，攻擊={len(X_attack)}），"
            f"請確認 simulate_anomaly_traffic.py 的產生器正常運作，"
            f"或加大 --n-normal-batches / --n-attack-batches。"
        )
    return X_normal, X_attack


# ──────────────────────────────────────────────────────────
# Step 2：unsupervised_vae
# ──────────────────────────────────────────────────────────
def train_unsupervised_vae(X_normal: np.ndarray,
                            X_attack: np.ndarray,
                            epochs: int,
                            patience: int,
                            output_root: str,
                            latent_dim: int,
                            pct: float) -> dict:
    from variational_autoencoder import VAETrainer
    import torch

    print("\n" + "═" * 62)
    print("  [unsupervised_vae] 非監督式 CNN-VAE")
    print("  資料表示法：packet_bytes_visualizer（與推論時一致）")
    print(f"  Epoch 上限：{epochs}（EarlyStopping patience={patience}）")
    print("═" * 62)

    out_dir = os.path.join(output_root, "unsupervised_vae")
    os.makedirs(out_dir, exist_ok=True)

    # 正常流量另切一份「訓練前保留的測試集」，只用來估算分離比，
    # 不參與訓練 / 驗證，避免用同一批資料自我評估造成錯覺。
    n = len(X_normal)
    idx = np.random.default_rng(42).permutation(n)
    split = max(1, int(n * 0.85))
    X_train_normal = X_normal[idx[:split]]
    X_holdout_normal = X_normal[idx[split:]] if split < n else X_normal[idx[-max(1, n // 20):]]

    train_npy = os.path.join(out_dir, "_X_train_normal.npy")
    np.save(train_npy, X_train_normal)

    config = {
        "latent_dim": latent_dim,
        "batch_size": 32,
        "epochs": epochs,
        "learning_rate": 1e-3,
        "patience": patience,
        "threshold_percentile": pct,
        "kl_weight": 1e-3,
        "val_split": 0.2,
    }
    trainer = VAETrainer(config=config, output_dir=out_dir, model_name="packet_native")
    trainer.load_data(train_npy)

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    threshold = trainer.compute_threshold(train_npy, pct)

    # 用保留的測試集 + 全部攻擊樣本估算分離比，做為訓練成效的參考指標
    device = trainer.device
    errors_n, errors_a = [], []
    with torch.no_grad():
        for i in range(0, len(X_holdout_normal), 256):
            batch = torch.from_numpy(
                X_holdout_normal[i:i + 256, None].astype(np.float32)).to(device)
            errors_n.append(trainer.model.reconstruction_error(batch).cpu().numpy())
        for i in range(0, len(X_attack), 256):
            batch = torch.from_numpy(
                X_attack[i:i + 256, None].astype(np.float32)).to(device)
            errors_a.append(trainer.model.reconstruction_error(batch).cpu().numpy())
    errors_n = np.concatenate(errors_n) if errors_n else np.array([0.0])
    errors_a = np.concatenate(errors_a) if errors_a else np.array([0.0])
    sep_ratio = float(errors_a.mean() / (errors_n.mean() + 1e-9))
    epochs_ran = len(trainer.train_losses)

    print(f"  訓練完成：{epochs_ran} epochs（上限 {epochs}），耗時 {train_time:.1f}s")
    print(f"  閾值（{pct}th pct）：{threshold:.6f}")
    print(f"  分離比（攻擊平均誤差 / 正常平均誤差）：{sep_ratio:.2f}x"
          f"（越大代表模型越能分辨攻擊，1.0x 代表完全分不出來）")

    ckpt_path = trainer._path()
    return {
        "model_key": "unsupervised_vae",
        "checkpoint": ckpt_path,
        "epochs_ran": epochs_ran,
        "epochs_cap": epochs,
        "train_time_sec": round(train_time, 1),
        "threshold": threshold,
        "separability_ratio": round(sep_ratio, 4),
        "n_train_normal": int(len(X_train_normal)),
    }


# ──────────────────────────────────────────────────────────
# Step 3：半監督模型（semi_cicddos2019 / semi_cicids2017 / semi_nslkdd）
# ──────────────────────────────────────────────────────────
def train_hybrid_semi(model_name: str,
                       X_normal: np.ndarray,
                       X_attack: np.ndarray,
                       epochs_pretrain: int,
                       epochs_finetune: int,
                       output_root: str,
                       latent_dim: int) -> dict:
    from hybrid_semi_supervised import HybridSemiSupervisedTrainer

    print("\n" + "═" * 62)
    print(f"  [semi_{model_name}] 半監督 CNN-VAE + CNN-LSTM 已知攻擊分類器")
    print("  資料表示法：packet_bytes_visualizer（與推論時一致）")
    print(f"  Pretrain 上限：{epochs_pretrain}  Finetune 上限：{epochs_finetune}")
    print("  [注意] HybridSemiSupervisedTrainer 目前沒有內建 EarlyStopping，"
          "epoch 數為固定值，非「上限」")
    print("═" * 62)

    out_dir = os.path.join(output_root, f"semi_{model_name}")
    os.makedirs(out_dir, exist_ok=True)

    n = len(X_normal)
    idx = np.random.default_rng(42).permutation(n)
    split = max(1, int(n * 0.85))
    X_train_normal = X_normal[idx[:split]]

    config = {
        "latent_dim": latent_dim,
        "batch_size": 32,
        "pretrain_epochs": epochs_pretrain,
        "finetune_epochs": epochs_finetune,
        "learning_rate": 1e-3,
        "attack_ratio": 0.3,
        "unknown_percentile": 95,
    }
    trainer = HybridSemiSupervisedTrainer(
        config=config, output_dir=out_dir, model_name=model_name)

    t0 = time.time()
    # threshold_method="optimal"：在未參與分類器微調的 holdout 攻擊樣本
    # 上做 F1 最佳化閾值搜尋（見 hybrid_semi_supervised.py 的說明），
    # 比固定百分位數更貼近實際分離狀況。
    threshold = trainer.train_full(X_train_normal, X_attack, threshold_method="optimal")
    train_time = time.time() - t0

    print(f"  訓練完成，耗時 {train_time:.1f}s")
    print(f"  閾值：{threshold:.6f}")

    ckpt_path = os.path.join(out_dir, f"semi_{model_name}.pt")
    return {
        "model_key": f"semi_{model_name}",
        "checkpoint": ckpt_path,
        "epochs_pretrain": epochs_pretrain,
        "epochs_finetune": epochs_finetune,
        "train_time_sec": round(train_time, 1),
        "threshold": float(threshold),
        "n_train_normal": int(len(X_train_normal)),
        "n_attack": int(len(X_attack)),
    }


# ──────────────────────────────────────────────────────────
# Step 4：部署（沿用既有的 core/deploy_model.py）
# ──────────────────────────────────────────────────────────
def _deploy(model_key: str, checkpoint_path: str, project_root: str) -> str:
    from deploy_model import deploy
    dst = deploy(model_key, checkpoint_path, project_root=project_root)
    print(f"  [部署] {model_key} → {dst}")
    return dst


# ──────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="用與推論一致的封包位元組影像表示法，重新訓練 4 個異常偵測模型，"
                     "修正 train/serve 資料表示法不一致（CSV 特徵 vs 封包位元組）的根本問題。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--only", choices=["all"] + ALL_MODEL_KEYS, default="all",
                        help="只訓練指定模型（預設 all，訓練全部 4 個）")

    # 資料集
    parser.add_argument("--dataset-dir", default=DATASET_DIR_DEFAULT,
                        help=f"封包原生資料集目錄（預設 {DATASET_DIR_DEFAULT}）")
    parser.add_argument("--skip-build", action="store_true",
                        help="（預設行為）資料集已存在時不重建，此旗標僅為明確語意保留")
    parser.add_argument("--rebuild", action="store_true",
                        help="強制重新產生資料集，即使已存在")
    parser.add_argument("--n-normal-batches", type=int, default=400)
    parser.add_argument("--n-attack-batches", type=int, default=80,
                        help="每種攻擊類型各自的批次數")
    parser.add_argument("--max-per-attack-type", type=int, default=3000)
    parser.add_argument("--scale-min", type=float, default=0.4)
    parser.add_argument("--scale-max", type=float, default=2.5)
    parser.add_argument("--image-size", default="medium",
                        choices=["small", "medium", "large"])
    parser.add_argument("--seed", type=int, default=42)

    # unsupervised_vae
    parser.add_argument("--vae-epochs", type=int, default=300,
                        help="VAE 訓練 epoch 上限（實際會由 EarlyStopping 提前停止）")
    parser.add_argument("--vae-patience", type=int, default=25,
                        help="VAE EarlyStopping 的 patience（連續幾輪驗證損失無改善才停止）")
    parser.add_argument("--vae-pct", type=float, default=95.0,
                        help="VAE 異常閾值百分位數")

    # semi_* 三個模型
    parser.add_argument("--semi-pretrain-epochs", type=int, default=150,
                        help="半監督模型 Phase 1（無監督預訓練）epoch 數")
    parser.add_argument("--semi-finetune-epochs", type=int, default=60,
                        help="半監督模型 Phase 2（半監督微調）epoch 數")

    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT,
                        help=f"訓練輸出根目錄（預設 {OUTPUT_ROOT_DEFAULT}）")
    parser.add_argument("--project-root", default=_PROJECT_ROOT_DEFAULT,
                        help="Django 專案根目錄（含 media/），用於自動部署")
    parser.add_argument("--no-deploy", action="store_true",
                        help="訓練完成後不自動部署到 media/model/")

    args = parser.parse_args()

    print("\n" + "═" * 62)
    print("  封包位元組原生表示法 — 異常偵測模型訓練")
    print("  修正對象：train/serve 資料表示法不一致（CSV 特徵 vs 封包位元組）")
    print("═" * 62)

    # ── Step 1：資料集 ──────────────────────────────────────
    X_normal, X_attack = _ensure_dataset(
        dataset_dir=args.dataset_dir,
        rebuild=args.rebuild,
        n_normal_batches=args.n_normal_batches,
        n_attack_batches_per_type=args.n_attack_batches,
        max_per_attack_type=args.max_per_attack_type,
        scale_range=(args.scale_min, args.scale_max),
        image_size=args.image_size,
        seed=args.seed,
    )

    targets = ALL_MODEL_KEYS if args.only == "all" else [args.only]
    os.makedirs(args.output_root, exist_ok=True)
    summary = {"meta": vars(args), "results": {}}
    t_all_start = time.time()

    # ── Step 2：unsupervised_vae ─────────────────────────────
    if "unsupervised_vae" in targets:
        res = train_unsupervised_vae(
            X_normal, X_attack,
            epochs=args.vae_epochs, patience=args.vae_patience,
            output_root=args.output_root, latent_dim=args.latent_dim,
            pct=args.vae_pct,
        )
        summary["results"]["unsupervised_vae"] = res
        if not args.no_deploy:
            res["deployed_to"] = _deploy("unsupervised_vae", res["checkpoint"],
                                         args.project_root)

    # ── Step 3：三個半監督模型 ────────────────────────────────
    for name in SEMI_MODEL_NAMES:
        key = f"semi_{name}"
        if key not in targets:
            continue
        res = train_hybrid_semi(
            name, X_normal, X_attack,
            epochs_pretrain=args.semi_pretrain_epochs,
            epochs_finetune=args.semi_finetune_epochs,
            output_root=args.output_root, latent_dim=args.latent_dim,
        )
        summary["results"][key] = res
        if not args.no_deploy:
            res["deployed_to"] = _deploy(key, res["checkpoint"], args.project_root)

    total_time = time.time() - t_all_start

    # ── Step 5：摘要報告 ─────────────────────────────────────
    summary_path = os.path.join(args.output_root, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "═" * 62)
    print("  訓練完成摘要")
    print("═" * 62)
    header = f"  {'模型':<20} {'閾值':>14} {'耗時(s)':>10} {'備註'}"
    print(header)
    print("  " + "-" * 58)
    for key, res in summary["results"].items():
        note = ""
        if "separability_ratio" in res:
            note = f"分離比 {res['separability_ratio']}x"
        print(f"  {key:<20} {res['threshold']:>14.6f} "
              f"{res['train_time_sec']:>10.1f} {note}")
    print("  " + "-" * 58)
    print(f"  總耗時：{total_time/60:.1f} 分鐘")
    print(f"  摘要報告：{os.path.abspath(summary_path)}")
    if args.no_deploy:
        print("\n  [提醒] 已加 --no-deploy，模型尚未部署到 media/model/，"
              "請手動執行 core/deploy_model.py 或重新執行本腳本不加此旗標。")
    else:
        print("\n  [提醒] 若 Django / Celery worker 為長駐執行的行程，"
              "請依 core/model_registry.py::clear_model_cache() 的說明清除快取，"
              "或直接重新啟動伺服器，新模型才會生效。")
    print("═" * 62)


if __name__ == "__main__":
    main()
