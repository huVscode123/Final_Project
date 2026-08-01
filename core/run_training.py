# ============================================================
# run_training.py - CNN Autoencoder 端對端訓練腳本
#
# v2.3 更新：整合真實資料集支援
#   新增 --dataset 參數，可切換三種真實資料集：
#     simulate    使用模擬封包（預設，無需下載）
#     nslkdd      NSL-KDD （CSV 格式，41 特徵）
#     cicids2017  CIC-IDS2017（CSV 格式，78 特徵）
#     cicddos2019 CIC-DDoS2019（CSV 格式，78 特徵）
#
#   核心修改：
#     原本直接呼叫 generate_test_pcap() + DatasetBuilder，
#     現在改為呼叫 DatasetFactory.load()，統一資料來源介面。
#     特徵向量轉影像的邏輯移至 dataset_loader.py，
#     不再依賴 packet_visualizer.py 的 PCAP 解析路徑。
#
# 執行方式：
#   python core/run_training.py                          使用模擬資料
#   python core/run_training.py --dataset nslkdd --data-dir data/nslkdd
#   python core/run_training.py --dataset cicids2017 --data-dir data/cicids2017
#   python core/run_training.py --dataset cicddos2019 --data-dir data/cicddos2019
#   python core/run_training.py --eval-only
#
# 資料集下載連結：
#   NSL-KDD:     https://www.unb.ca/cic/datasets/nsl.html
#   CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
#   CIC-DDoS2019:https://www.unb.ca/cic/datasets/ddos-2019.html
# ============================================================
 
import os
import sys
import argparse
import numpy as np
 
# 確保 core 目錄在 sys.path 中，以便直接匯入同目錄下的模組
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
 
 
def check_torch():
    """確認 PyTorch 已安裝"""
    try:
        import torch
        print(f"  PyTorch {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        return True
    except ImportError:
        print("  PyTorch 未安裝")
        print("  請執行: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return False
 
 
def main():
    parser = argparse.ArgumentParser(
        description="CNN Autoencoder 訓練腳本（支援真實資料集）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python core/run_training.py
  python core/run_training.py --dataset nslkdd --data-dir data/nslkdd
  python core/run_training.py --dataset cicids2017 --data-dir data/cicids2017
  python core/run_training.py --dataset cicddos2019 --data-dir data/cicddos2019
  python core/run_training.py --epochs 100 --latent 32
  python core/run_training.py --eval-only
        """
    )
 
    # 資料集相關
    parser.add_argument("--dataset",    default="simulate",
                        choices=["simulate", "nslkdd", "cicids2017", "cicddos2019", "cic-ddos2019"],
                        help="資料集名稱（預設 simulate）")
    parser.add_argument("--data-dir",   default=None,
                        help="資料集目錄（未指定則使用預設路徑）")
    parser.add_argument("--max-normal", type=int, default=50000,
                        help="最大正常樣本數（真實資料集，預設 50000）")
    parser.add_argument("--max-attack", type=int, default=30000,
                        help="最大攻擊樣本數（真實資料集，預設 30000）")
 
    # 訓練相關
    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--latent",   type=int,   default=32)
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--patience", type=int,   default=15)
    parser.add_argument("--pct",      type=int,   default=95,
                        help="閾值百分位數（預設 95）")
    parser.add_argument("--output",   default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model",     default=None)
    args = parser.parse_args()
 
    # 各資料集使用獨立輸出目錄與模型後綴
    DATASET_SUFFIX = {
        "simulate":    "simulate",
        "nslkdd":      "nslkdd",
        "cicids2017":  "cicids2017",
        "cicddos2019": "cicddos2019",
        "cic-ddos2019": "cicddos2019",
    }
    model_name = DATASET_SUFFIX.get(args.dataset, args.dataset)
    if args.output is None:
        args.output = f"output/model_{model_name}"
 
    print("\n" + "=" * 60)
    print(f"  CNN Autoencoder 訓練流程（資料集：{args.dataset}）")
    print("=" * 60)
 
    if not check_torch():
        sys.exit(1)
 
    # 注意：移除 os.chdir(BASE_DIR)，保持在專案根目錄執行，
    # 這樣傳入的相對路徑 (如 data/cicids2017) 才能正確對應。
    
    from trainer import Trainer
    from anomaly_scorer import AnomalyScorer
    from dataset_loader import DatasetFactory
 
    # ── Step 1：載入資料集（統一介面）────────────────────
    print(f"\n[Step 1] 載入資料集：{args.dataset}")
 
    # DatasetFactory.load() 處理所有資料集的差異：
    #   simulate    -> 直接產生 numpy 模擬資料
    #   nslkdd      -> 讀取 KDDTrain+.txt，解析 41 特徵，轉 32x32 影像
    #   cicids2017  -> 讀取 CSV，解析 78 特徵，處理 inf/NaN，轉 32x32 影像
    #   cicddos2019 -> 同 cicids2017，遞迴掃描子目錄
    load_kwargs = {}
    if args.dataset in ("cicids2017", "cicddos2019"):
        load_kwargs["max_normal"] = args.max_normal
        load_kwargs["max_attack"] = args.max_attack
 
    try:
        X_normal, X_attack, y = DatasetFactory.load(
            args.dataset,
            data_dir=args.data_dir,
            **load_kwargs
        )
    except FileNotFoundError as e:
        print(f"\n  錯誤：{e}")
        sys.exit(1)
 
    print(f"  正常流量: {len(X_normal):,} 筆  攻擊流量: {len(X_attack):,} 筆")
 
    # 儲存為 .npy（供 Trainer 使用）
    print("\n[Step 2] 儲存影像矩陣")
    dataset_dir = f"output/dataset_{args.dataset}"
    paths = DatasetFactory.save_as_npy(X_normal, X_attack, y, dataset_dir)
    normal_npy = paths["normal"]
    attack_npy = paths["attack"]
 
    # ── Step 3：訓練或載入模型 ────────────────────────────
    model_path  = args.model or os.path.join(args.output, f"best_model_{model_name}.pt")
    config_path = os.path.join(args.output, f"training_result_{model_name}.json")
 
    if args.eval_only:
        print(f"\n[Step 3] 載入模型: {args.model or model_path}")
        model, threshold = Trainer.load_model(
            args.model or model_path, config_path, args.latent
        )
        scorer = AnomalyScorer(model, threshold=threshold)
    else:
        print("\n[Step 3] 訓練 CNN Autoencoder")
        config = {
            "latent_dim":           args.latent,
            "batch_size":           args.batch,
            "epochs":               args.epochs,
            "learning_rate":        args.lr,
            "patience":             args.patience,
            "threshold_percentile": args.pct,
        }
        trainer = Trainer(config=config, output_dir=args.output, model_name=model_name)
        trainer.load_data(normal_npy)
        trainer.train()
        trainer.plot_training_curve()
 
        print("\n[Step 4] 計算異常偵測閾值")
        threshold = trainer.compute_threshold(normal_npy, args.pct)
 
        print("\n[Step 5] 繪製重建對比圖")
        trainer.plot_reconstruction_samples(normal_npy)
 
        scorer = AnomalyScorer(trainer.model, threshold=threshold)
 
    # ── Step 6：評估 ──────────────────────────────────────
    if os.path.exists(attack_npy):
        print("\n[Step 6] 模型效能評估")

        # [修正] 分離測試集，避免資料洩漏
        # 原版使用完整的 X_normal（包含訓練集）進行評估，
        # 模型在自己見過的資料上被評估，導致指標過度樂觀。
        # 現在只使用 20% 的正常流量作為測試集（與訓練時的 val_split 一致）。
        n_test = max(1, int(len(X_normal) * 0.2))
        X_normal_test = X_normal[-n_test:]    # 取最後 20% 作為測試（不與訓練集重疊）
        X_test = np.concatenate([X_normal_test, X_attack])
        y_test = np.concatenate([
            np.zeros(len(X_normal_test), dtype=int),
            np.ones(len(X_attack),       dtype=int)
        ])
        print(f"  測試集: {len(X_normal_test)} 正常 + {len(X_attack)} 攻擊 = {len(X_test)} 樣本")
        results = scorer.evaluate(X_test, y_test, output_dir=args.output)
 
        errors_normal = scorer.score_npy(normal_npy)
        errors_attack = scorer.score_npy(attack_npy)
        print(f"\n  正常流量平均誤差: {errors_normal.mean():.6f}")
        print(f"  攻擊流量平均誤差: {errors_attack.mean():.6f}")
        ratio = errors_attack.mean() / (errors_normal.mean() + 1e-9)
        print(f"  兩者誤差比值    : {ratio:.2f}x")
 
        scorer.plot_roc_curve(errors_normal, errors_attack, args.output)
        scorer.plot_score_distribution(errors_normal, errors_attack, args.output)
 
    # ── 完成 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  完成！輸出目錄:", os.path.abspath(args.output))
    if os.path.exists(args.output):
        for fname in sorted(os.listdir(args.output)):
            fpath = os.path.join(args.output, fname)
            print(f"    {fname:<40} {os.path.getsize(fpath)/1024:.1f} KB")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()