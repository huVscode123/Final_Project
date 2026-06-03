# ============================================================
# run_full_pipeline.py - 從資料到部署的完整訓練流程
#
# 這個腳本整合所有步驟：
#   Step 1  載入資料集（真實 / 模擬）
#   Step 2  資料擴增（增加訓練量）
#   Step 3  合併多個資料集（選用）
#   Step 4  訓練 CNN Autoencoder
#   Step 5  掃描最佳閾值
#   Step 6  自動更新 settings.py 的 CNN_THRESHOLD
#   Step 7  輸出完整評估報告
#
# 執行方式：
#   # 模擬資料快速測試（不需下載）
#   python core/run_full_pipeline.py
#
#   # 真實資料集
#   python core/run_full_pipeline.py \
#       --dataset cicids2017 --data-dir data/cicids2017 \
#       --augment --multiplier 3 \
#       --merge-nslkdd data/nslkdd
#
#   # 使用 GPU 加速
#   python core/run_full_pipeline.py --dataset cicids2017 --gpu
# ============================================================
 
import os
import sys
import json
import argparse
import re
import numpy as np
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)   # network_platform/
sys.path.insert(0, BASE_DIR)
os.chdir(PROJECT_DIR)
 
 
def check_gpu():
    """確認 GPU 狀態，回傳裝置字串"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            print(f"  GPU: {name}（{mem} MB VRAM）")
            return "cuda"
        else:
            print("  GPU 未偵測到，使用 CPU 訓練")
            return "cpu"
    except ImportError:
        print("  PyTorch 未安裝")
        sys.exit(1)
 
 
def update_settings_threshold(threshold: float,
                               settings_path: str = None):
    """
    自動更新 settings.py 的 CNN_THRESHOLD 數值
 
    訓練完成後不需要手動修改 settings.py，
    本函式直接讀取並替換 CNN_THRESHOLD 這行。
 
    Args:
        threshold    : 新的閾值數值
        settings_path: settings.py 的路徑（None 則自動尋找）
    """
    if settings_path is None:
        settings_path = os.path.join(
            PROJECT_DIR, "network_platform", "settings.py"
        )
 
    if not os.path.exists(settings_path):
        print(f"  [警告] 找不到 settings.py：{settings_path}")
        print(f"  請手動將 CNN_THRESHOLD 設為 {threshold:.6f}")
        return
 
    content = open(settings_path, encoding="utf-8").read()
 
    # 用正則表達式找到並替換 CNN_THRESHOLD 這行
    pattern = r"CNN_THRESHOLD\s*=\s*[\d\.]+"
    new_line = f"CNN_THRESHOLD  = {threshold:.6f}"
 
    if re.search(pattern, content):
        new_content = re.sub(pattern, new_line, content)
        open(settings_path, "w", encoding="utf-8").write(new_content)
        print(f"  [已更新] settings.py CNN_THRESHOLD = {threshold:.6f}")
    else:
        print(f"  [警告] settings.py 中找不到 CNN_THRESHOLD")
        print(f"  請手動加入：CNN_THRESHOLD = {threshold:.6f}")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="完整訓練流程（資料擴增 → 訓練 → 閾值更新）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python core/run_full_pipeline.py
  python core/run_full_pipeline.py --dataset cicids2017 --data-dir data/cicids2017
  python core/run_full_pipeline.py --dataset cicids2017 --augment --multiplier 3
  python core/run_full_pipeline.py --merge-nslkdd data/nslkdd
  python core/run_full_pipeline.py --gpu --batch 64 --epochs 200
        """
    )
 
    # 資料集設定
    parser.add_argument("--dataset",  default="simulate",
                        choices=["simulate","nslkdd","cicids2017","cicddos2019","cic-ddos2019"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--max-normal", type=int, default=50000)
    parser.add_argument("--max-attack", type=int, default=30000)
 
    # 資料擴增
    parser.add_argument("--augment",    action="store_true",
                        help="啟用資料擴增（當正常流量 < 2000 筆時建議開啟）")
    parser.add_argument("--multiplier", type=int, default=3,
                        help="擴增倍數（預設 3，即 1000 筆 → 3000 筆）")
 
    # 合併多個資料集
    parser.add_argument("--merge-nslkdd", default=None, metavar="DIR",
                        help="同時使用 NSL-KDD 正常流量合併訓練（填入資料目錄）")
 
    # 訓練設定
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--latent",  type=int,   default=32)
    parser.add_argument("--batch",   type=int,   default=32,
                        help="GPU 可用時建議設為 64 或 128")
    parser.add_argument("--gpu",     action="store_true",
                        help="強制使用 GPU（若不可用則報錯）")
    parser.add_argument("--output",  default=None,
                        help="模型輸出目錄（預設: output/model_<dataset>）")
 
    # 閾值設定
    parser.add_argument("--metric",  default="f1",
                        choices=["f1","recall","precision"],
                        help="自動選閾值的最佳化目標（預設 f1）")
    parser.add_argument("--no-update-settings", action="store_true",
                        help="訓練完成後不自動更新 settings.py")
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
 
    print("\n" + "=" * 65)
    print("  完整訓練流程")
    print("=" * 65)
 
    # ── GPU 確認 ──────────────────────────────────────────
    import torch
    device_str = check_gpu()
    if args.gpu and device_str != "cuda":
        print("  錯誤：指定 --gpu 但找不到 GPU")
        sys.exit(1)
 
    from trainer import Trainer
    from dataset_loader import DatasetFactory
    from threshold_tuner import ThresholdTuner
    from data_augmentor import DataAugmentor
 
    # ══════════════════════════════════════════════════════
    # Step 1：載入主要資料集
    # ══════════════════════════════════════════════════════
    print(f"\n[Step 1] 載入資料集：{args.dataset}")
 
    load_kwargs = {}
    if args.dataset in ("cicids2017", "cicddos2019"):
        load_kwargs["max_normal"] = args.max_normal
        load_kwargs["max_attack"] = args.max_attack
 
    try:
        X_normal, X_attack, y = DatasetFactory.load(
            args.dataset, data_dir=args.data_dir, **load_kwargs
        )
    except FileNotFoundError as e:
        print(f"\n  錯誤：{e}")
        sys.exit(1)
 
    print(f"  正常流量: {len(X_normal):,} 筆")
    print(f"  攻擊流量: {len(X_attack):,} 筆")
 
    # ══════════════════════════════════════════════════════
    # Step 2：合併其他資料集（選用）
    # ══════════════════════════════════════════════════════
    if args.merge_nslkdd:
        print(f"\n[Step 2] 合併 NSL-KDD 正常流量：{args.merge_nslkdd}")
        try:
            from dataset_loader import NSLKDDLoader
            loader_kdd = NSLKDDLoader(args.merge_nslkdd)
            X_kdd_normal, _, _ = loader_kdd.load()
            print(f"  NSL-KDD 正常流量: {len(X_kdd_normal):,} 筆")
 
            # 合併兩個資料集
            X_normal = DataAugmentor.merge_datasets.__func__(
                None
            ) if False else np.concatenate([X_normal, X_kdd_normal], axis=0)
            print(f"  合併後正常流量: {len(X_normal):,} 筆")
        except Exception as e:
            print(f"  [警告] NSL-KDD 合併失敗：{e}，繼續使用原資料集")
    else:
        print(f"\n[Step 2] 跳過資料集合併")
 
    # ══════════════════════════════════════════════════════
    # Step 3：資料擴增
    # ══════════════════════════════════════════════════════
    if args.augment:
        print(f"\n[Step 3] 資料擴增（倍數：{args.multiplier}x）")
        print(f"  說明：正常流量 {len(X_normal):,} 筆 → "
              f"{len(X_normal) * args.multiplier:,} 筆")
 
        aug = DataAugmentor(
            noise_std   = 0.02,
            mask_ratio  = 0.10,
            scale_range = (0.90, 1.10),
            use_mixup   = True,
        )
        X_normal = aug.augment(X_normal, multiplier=args.multiplier)
        print(f"  擴增完成：{len(X_normal):,} 筆正常流量")
    else:
        print(f"\n[Step 3] 跳過資料擴增（加 --augment 可啟用）")
 
    # ══════════════════════════════════════════════════════
    # Step 4：儲存並訓練
    # ══════════════════════════════════════════════════════
    print(f"\n[Step 4] 儲存影像矩陣並開始訓練")
 
    dataset_dir = f"output/dataset_{args.dataset}"
    os.makedirs(dataset_dir, exist_ok=True)
    normal_npy = os.path.join(dataset_dir, "X_normal.npy")
    attack_npy = os.path.join(dataset_dir, "X_attack.npy")
 
    np.save(normal_npy, X_normal)
    np.save(attack_npy, X_attack)
    print(f"  X_normal.npy → {len(X_normal):,} 筆")
    print(f"  X_attack.npy → {len(X_attack):,} 筆")
 
    # 訓練設定
    # GPU 可用時 batch_size 可以加大，加快訓練速度
    effective_batch = args.batch
    if device_str == "cuda" and args.batch < 64:
        effective_batch = 64
        print(f"  GPU 模式：batch_size 自動調整為 {effective_batch}")
 
    config = {
        "latent_dim":    args.latent,
        "batch_size":    effective_batch,
        "epochs":        args.epochs,
        "learning_rate": 1e-3,
        "patience":      15,
        "threshold_percentile": 95,
    }
 
    trainer = Trainer(config=config, output_dir=args.output, model_name=model_name)
    trainer.load_data(normal_npy)
    trainer.train()
    trainer.plot_training_curve()
    trainer.plot_reconstruction_samples(normal_npy)
 
    # ══════════════════════════════════════════════════════
    # Step 5：掃描最佳閾值
    # ══════════════════════════════════════════════════════
    print(f"\n[Step 5] 掃描最佳閾值（目標：最大化 {args.metric.upper()}）")
 
    import torch
    device = torch.device(device_str)
    tuner  = ThresholdTuner(trainer.model, device)
 
    scan_results = tuner.scan_percentiles(
        X_normal, X_attack,
        percentiles=[80, 85, 90, 92, 95, 97, 99]
    )
    tuner.plot_threshold_curve(scan_results, args.output)
 
    best = tuner.find_best_threshold(scan_results, metric=args.metric)
    best_threshold = best["threshold"]
 
    # ══════════════════════════════════════════════════════
    # Step 6：自動更新 settings.py
    # ══════════════════════════════════════════════════════
    print(f"\n[Step 6] 更新 Django 設定")
 
    if not args.no_update_settings:
        update_settings_threshold(best_threshold)
 
        # 同時更新 CNN_LATENT_DIM（若與現有設定不同）
        settings_path = os.path.join(
            PROJECT_DIR, "network_platform", "settings.py"
        )
        if os.path.exists(settings_path):
            content = open(settings_path, encoding="utf-8").read()
            pattern_dim = r"CNN_LATENT_DIM\s*=\s*\d+"
            new_dim = f"CNN_LATENT_DIM  = {args.latent}"
            if re.search(pattern_dim, content):
                new_content = re.sub(pattern_dim, new_dim, content)
                open(settings_path, "w", encoding="utf-8").write(new_content)
                print(f"  [已更新] settings.py CNN_LATENT_DIM = {args.latent}")
 
    # ══════════════════════════════════════════════════════
    # Step 7：輸出完整報告
    # ══════════════════════════════════════════════════════
    print(f"\n[Step 7] 產生完整評估報告")
 
    report = {
        "dataset":         args.dataset,
        "augmented":       args.augment,
        "multiplier":      args.multiplier if args.augment else 1,
        "normal_count":    int(len(X_normal)),
        "attack_count":    int(len(X_attack)),
        "latent_dim":      args.latent,
        "device":          device_str,
        "best_threshold":  best_threshold,
        "best_metric":     args.metric,
        "best_f1":         best["f1"],
        "best_recall":     best["recall"],
        "best_precision":  best["precision"],
        "scan_results":    scan_results,
    }
 
    report_path = os.path.join(args.output, "pipeline_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
 
    print(f"  報告已儲存：{report_path}")
 
    # ── 完成摘要 ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  訓練完成摘要")
    print("=" * 65)
    print(f"  資料集      : {args.dataset}")
    print(f"  訓練筆數    : {len(X_normal):,} 筆正常流量")
    print(f"  裝置        : {device_str.upper()}")
    print(f"  最佳閾值    : {best_threshold:.6f}（{best['percentile']}th 百分位）")
    print(f"  Precision   : {best['precision']:.4f}")
    print(f"  Recall      : {best['recall']:.4f}")
    print(f"  F1 Score    : {best['f1']:.4f}")
    print()
    print("  下一步：")
    print(f"    1. 將訓練好的模型複製到 media/model/")
    print(f"       cp {args.output}/best_model_{model_name}.pt media/model/best_model_{model_name}.pt")
    print(f"    2. settings.py 的 CNN_THRESHOLD 已自動更新為 {best_threshold:.6f}")
    print(f"    3. 重新啟動 Django：python manage.py runserver")
    print("=" * 65)
 
 
if __name__ == "__main__":
    main()