# ============================================================
# core/train_packet_native_vae.py（新增檔案）
#
# 使用 core/packet_dataset_builder.py 產生的「封包位元組表示法」
# 資料集，訓練 unsupervised_vae 模型（settings.ANOMALY_MODELS 的
# 'unsupervised_vae'，實際檔案是 media/model/best_vae_model.pt）。
#
# 與 core/run_training.py 的差異：
#   run_training.py 呼叫 dataset_loader.DatasetFactory.load()，讀的
#   是 CICIDS2017/NSL-KDD 等『CSV 統計特徵』資料集，與正式 PCAP
#   分析、模擬檢測實際送入模型的『封包原始位元組影像』表示法不同
#   （詳見 packet_dataset_builder.py 檔頭說明）。
#
#   本腳本改讀 packet_dataset_builder.py 產生的 .npy，並且會在
#   儲存的 checkpoint 的 config 字典裡多寫入
#       'representation': 'packet_bytes_visualizer'
#   這個標記。core/model_registry.py::AnomalyModelBundle
#   .input_representation 會讀出這個標記，analyzer/views.py
#   ::simulation_api() 就能依此動態切換 CNN 可信度說明文字
#   （從「僅供參考、不宜採信」變成「具備參考意義」），而不是對所有
#   模型都顯示同一句制式警語。
#
# ── 使用方式 ────────────────────────────────────────────────
#   # Step 1：先建立資料集
#   python core/packet_dataset_builder.py --output output/dataset_packet_native
#
#   # Step 2：訓練
#   python core/train_packet_native_vae.py \
#       --dataset-dir output/dataset_packet_native \
#       --output output/model_packet_native \
#       --epochs 150
#
#   # Step 3（選用）：訓練完成後直接部署（複製到 Django 讀取的路徑）
#   python core/train_packet_native_vae.py \
#       --dataset-dir output/dataset_packet_native \
#       --output output/model_packet_native \
#       --epochs 150 \
#       --deploy-to ../media/model/best_vae_model.pt
#
#   部署後請重新啟動 Django（或在 Django shell 呼叫
#   model_registry.clear_model_cache()），否則長駐中的行程仍會
#   使用行程內快取的舊模型。
#
# ── 關於「50 epoch」───────────────────────────────────────────
#   在資料表示法已經修正為與推論一致的前提下，epoch 數量才開始有
#   意義：訓練資料與推論資料現在是同一種分布，模型才可能透過更多
#   訓練確實學到「正常封包長什麼樣子」。這裡把預設值提高到 150，
#   並沿用 VAETrainer 既有的 EarlyStopping（patience=20）機制，
#   不會因為 epoch 設太高而過擬合過頭，訓練會在驗證損失不再下降
#   時自動停止。若在表示法修正『之前』就把 epoch 從 50 調高，並不
#   會讓 CNN 分數變得更有意義，只是在錯誤的資料上訓練更久。
# ============================================================

import os
import sys
import json
import shutil
import argparse
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="使用封包位元組表示法資料集訓練 CNN-VAE 異常偵測模型"
    )
    parser.add_argument("--dataset-dir", default="output/dataset_packet_native")
    parser.add_argument("--output", default="output/model_packet_native")
    parser.add_argument("--epochs", type=int, default=150,
                         help="訓練輪數（預設 150；建議在表示法修正後才調高，"
                              "見檔案開頭說明）")
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--pct", type=int, default=97,
                         help="閾值百分位數（預設 97；封包位元組影像的重建誤差"
                              "變異通常比 CSV 特徵影像大，稍微提高百分位數"
                              "可降低誤報率）")
    parser.add_argument("--deploy-to", default=None,
                         help="若指定路徑，訓練完成後自動複製模型檔案到該路徑"
                              "（例如 ../media/model/best_vae_model.pt，"
                              "對應 settings.ANOMALY_MODELS['unsupervised_vae']"
                              "['path']）。部署後仍須重啟 Django 或清除模型快取"
                              "才會生效。")
    args = parser.parse_args()

    from variational_autoencoder import VAETrainer
    from anomaly_scorer import AnomalyScorer

    normal_path = os.path.join(args.dataset_dir, "X_normal.npy")
    attack_path = os.path.join(args.dataset_dir, "X_attack.npy")
    if not os.path.exists(normal_path) or not os.path.exists(attack_path):
        print("[錯誤] 找不到資料集，請先執行：")
        print(f"  python core/packet_dataset_builder.py --output {args.dataset_dir}")
        sys.exit(1)

    X_normal = np.load(normal_path).astype(np.float32)
    X_attack = np.load(attack_path).astype(np.float32)
    print(f"[train_packet_native_vae] 正常影像 {X_normal.shape}，攻擊影像 {X_attack.shape}")

    if len(X_normal) < 50:
        print("[警告] 正常樣本數過少，建議調高 packet_dataset_builder.py 的 "
              "--n-normal-batches 重新產生資料集。")

    # 80/20 切分，避免資料洩漏（閾值/評估都只用模型『沒看過』的正常樣本）
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_normal))
    split = int(len(X_normal) * 0.8)
    X_train = X_normal[idx[:split]]
    X_test_normal = X_normal[idx[split:]]

    idx_a = rng.permutation(len(X_attack))
    split_a = int(len(X_attack) * 0.5)
    X_test_attack = X_attack[idx_a[split_a:]]  # 只用一半做最終評估

    os.makedirs(args.output, exist_ok=True)
    train_npy_path = os.path.join(args.output, "_X_train_normal.npy")
    np.save(train_npy_path, X_train)

    config = {
        "latent_dim": args.latent,
        "batch_size": args.batch,
        "epochs": args.epochs,
        "learning_rate": 1e-3,
        "patience": 20,
        "threshold_percentile": args.pct,
        "kl_weight": 1e-3,
        # [核心標記] 讓 model_registry.AnomalyModelBundle.input_representation
        # 能正確判斷這個模型是用封包位元組表示法訓練的。
        "representation": "packet_bytes_visualizer",
    }

    trainer = VAETrainer(config=config, output_dir=args.output, model_name="packet_native")
    trainer.load_data(train_npy_path)
    trainer.train()
    threshold = trainer.compute_threshold(train_npy_path, args.pct)

    scorer = AnomalyScorer(trainer.model, threshold=threshold)
    X_eval = np.concatenate([X_test_normal, X_test_attack])
    y_eval = np.concatenate([
        np.zeros(len(X_test_normal), dtype=int),
        np.ones(len(X_test_attack), dtype=int),
    ])
    print(f"\n[train_packet_native_vae] 評估集：{len(X_test_normal)} 正常 + "
          f"{len(X_test_attack)} 攻擊 = {len(X_eval)} 筆")
    results = scorer.evaluate(X_eval, y_eval, output_dir=args.output)

    report = {
        "representation": "packet_bytes_visualizer",
        "epochs": args.epochs,
        "threshold": threshold,
        "eval": results,
    }
    with open(os.path.join(args.output, "packet_native_training_report.json"),
              "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print("\n[train_packet_native_vae] 完成。")
    print(f"  Precision={results['precision']:.4f}  Recall={results['recall']:.4f}  "
          f"F1={results['f1_score']:.4f}  AUC={results.get('auc')}")

    src = os.path.join(args.output, "best_vae_packet_native.pt")
    if args.deploy_to:
        if not os.path.exists(src):
            print(f"[錯誤] 找不到訓練輸出的模型檔案：{src}")
            sys.exit(1)
        os.makedirs(os.path.dirname(args.deploy_to) or ".", exist_ok=True)
        shutil.copy2(src, args.deploy_to)
        print(f"\n  已部署至：{os.path.abspath(args.deploy_to)}")
        print("  [重要] 部署後請重新啟動 Django（或在 Django shell 呼叫 "
              "model_registry.clear_model_cache()），否則長駐執行中的伺服器"
              "行程仍會使用快取住的舊模型。")
    else:
        print(f"\n  模型已儲存於：{os.path.abspath(src)}")
        print("  若要讓 Django 使用這個模型，請手動複製到 "
              "media/model/best_vae_model.pt，並重啟伺服器或清除模型快取。")


if __name__ == "__main__":
    main()
