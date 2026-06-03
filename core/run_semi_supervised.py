# ============================================================
# run_semi_supervised.py - 半監督式訓練主程式
#
# 此腳本整合以下流程：
#   Step 1  載入資料集（真實 / 模擬）
#   Step 2  資料擴增（選用）
#   Step 3  Phase 1：無監督預訓練
#   Step 4  Phase 2：半監督微調
#   Step 5  計算最佳閾值（optimal / percentile）
#   Step 6  效能評估（P / R / F1 / AUC）
#   Step 7  產生對比報告（半監督 vs 非監督）
#
# 執行方式：
#   # 快速測試（模擬資料，無需下載）
#   python core/run_semi_supervised.py
#
#   # 真實資料集
#   python core/run_semi_supervised.py \
#       --dataset cicids2017 --data-dir data/cicids2017
#
#   # 完整選項
#   python core/run_semi_supervised.py \
#       --dataset cicids2017 \
#       --data-dir data/cicids2017 \
#       --pretrain-epochs 80 \
#       --finetune-epochs 60 \
#       --alpha 1.0 --beta 0.5 \
#       --margin 0.05 \
#       --attack-ratio 0.2 \
#       --threshold-method optimal \
#       --compare-unsupervised
#
# 輸出：
#   output/model_semi/best_model.pt          最佳半監督模型
#   output/model_semi/semi_training_curve.png 訓練曲線
#   output/model_semi/semi_error_distribution.png 誤差分布
#   output/model_semi/semi_evaluation_report.json 評估報告
#   output/model_semi/comparison_report.png  對比圖（加 --compare-unsupervised）
# ============================================================
 
import os
import sys
import json
import argparse
import numpy as np
 
# 確保 core 目錄在 sys.path 中
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
 
 
def check_torch():
    try:
        import torch
        print(f"  PyTorch {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        return True
    except ImportError:
        print("  PyTorch 未安裝")
        print("  請執行: pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
        return False
 
 
def evaluate_model(model, X_normal, X_attack, threshold, device,
                   output_dir="output/model_semi"):
    """
    完整評估模型效能
 
    Returns:
        dict: 包含 precision/recall/f1/auc 等指標
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
 
    def compute_errors(X):
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        t = torch.from_numpy(X.astype(np.float32))
        loader = DataLoader(TensorDataset(t), batch_size=128)
        errors = []
        with torch.no_grad():
            for (b,) in loader:
                b = b.to(device)
                e = model.reconstruction_error(b)
                errors.append(e.cpu().numpy())
        return np.concatenate(errors)
 
    errors_n = compute_errors(X_normal)
    errors_a = compute_errors(X_attack)
 
    fp = int((errors_n > threshold).sum())
    tn = len(errors_n) - fp
    tp = int((errors_a > threshold).sum())
    fn = len(errors_a) - tp
 
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    
    # 計算 AUC
    from sklearn.metrics import roc_auc_score
    y_true = np.concatenate([np.zeros(len(errors_n)), np.ones(len(errors_a))])
    y_scores = np.concatenate([errors_n, errors_a])
    auc = roc_auc_score(y_true, y_scores)
 
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "auc": auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": float(threshold)
    }
 
 
def main():
    parser = argparse.ArgumentParser(
        description="半監督式 CNN Autoencoder 訓練與評估",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
 
    # 資料相關
    parser.add_argument("--dataset",   default="simulate",
                        choices=["simulate", "nslkdd", "cicids2017", "cicddos2019"])
    parser.add_argument("--data-dir",  default=None)
    parser.add_argument("--attack-ratio", type=float, default=0.1,
                        help="訓練集中包含的攻擊流量比例 (0.0~0.5)")
    parser.add_argument("--augment",   action="store_true", help="啟用資料擴增")
 
    # 訓練相關
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--batch",           type=int, default=32)
    parser.add_argument("--latent",          type=int, default=32)
    parser.add_argument("--alpha",           type=float, default=1.0, help="重構損失權重")
    parser.add_argument("--beta",            type=float, default=0.5, help="對比損失權重")
    parser.add_argument("--margin",          type=float, default=0.1, help="對比損失 Margin")
 
    # 評估相關
    parser.add_argument("--threshold-method", default="percentile",
                        choices=["percentile", "optimal"], help="閾值計算方法")
    parser.add_argument("--pct",              type=int, default=95)
    parser.add_argument("--output",           default=None,
                        help="模型輸出目錄（預設: output/model_semi_<dataset>）")
    parser.add_argument("--compare-unsupervised", action="store_true", 
                        help="同時執行純無監督訓練以進行對比")
 
    args = parser.parse_args()
 
    # 各資料集使用獨立輸出目錄與模型後綴
    DATASET_SUFFIX = {
        "simulate":    "simulate",
        "nslkdd":      "nslkdd",
        "cicids2017":  "cicids2017",
        "cicddos2019": "cicddos2019",
    }
    model_name = DATASET_SUFFIX.get(args.dataset, args.dataset)
    if args.output is None:
        args.output = f"output/model_semi_{model_name}"
 
    print("\n" + "=" * 60)
    print(f"  半監督式訓練流程（資料集：{args.dataset}）")
    print("=" * 60)
 
    if not check_torch():
        sys.exit(1)
 
    import torch
    from dataset_loader import DatasetFactory
    from data_augmentor import DataAugmentor
    from semi_supervised_trainer import SemiSupervisedTrainer
    from trainer import Trainer
    from anomaly_scorer import AnomalyScorer
 
    # ══════════════════════════════════════════════════════
    # Step 1：載入資料集
    # ══════════════════════════════════════════════════════
    print(f"\n[Step 1] 載入資料集：{args.dataset}")
    
    load_kwargs = {}
    if args.dataset in ("cicids2017", "cicddos2019"):
        load_kwargs["max_normal"] = 30000
        load_kwargs["max_attack"] = 20000
 
    try:
        X_normal, X_attack, _ = DatasetFactory.load(
            args.dataset, data_dir=args.data_dir, **load_kwargs
        )
    except FileNotFoundError as e:
        print(f"\n  錯誤：{e}")
        sys.exit(1)
 
    # 切分訓練與測試集
    np.random.seed(42)
    idx_n = np.random.permutation(len(X_normal))
    split_n = int(len(X_normal) * 0.8)
    X_train_normal = X_normal[idx_n[:split_n]]
    X_test_normal  = X_normal[idx_n[split_n:]]
 
    idx_a = np.random.permutation(len(X_attack))
    split_a = int(len(X_attack) * 0.5)
    X_train_attack_pool = X_attack[idx_a[:split_a]]
    X_test_attack       = X_attack[idx_a[split_a:]]
 
    # 從攻擊池中挑選一部分加入訓練集（半監督關鍵）
    n_attack_in_train = int(len(X_train_normal) * args.attack_ratio)
    n_attack_in_train = min(n_attack_in_train, len(X_train_attack_pool))
    X_train_attack = X_train_attack_pool[:n_attack_in_train]
 
    print(f"  訓練集: {len(X_train_normal):,} 正常 + {len(X_train_attack):,} 攻擊")
    print(f"  測試集: {len(X_test_normal):,} 正常 + {len(X_test_attack):,} 攻擊")
 
    # ══════════════════════════════════════════════════════
    # Step 2：資料擴增
    # ══════════════════════════════════════════════════════
    if args.augment:
        print("\n[Step 2] 執行資料擴增")
        augmentor = DataAugmentor()
        X_train_normal = augmentor.augment_batch(X_train_normal, multiplier=2)
        print(f"  擴增後正常訓練樣本: {len(X_train_normal):,}")
 
    # ══════════════════════════════════════════════════════
    # Step 3 & 4：半監督訓練
    # ══════════════════════════════════════════════════════
    print("\n[Step 3/4] 執行半監督訓練")
    config = {
        "latent_dim": args.latent,
        "batch_size": args.batch,
        "pretrain_epochs": args.pretrain_epochs,
        "finetune_epochs": args.finetune_epochs,
        "alpha": args.alpha,
        "beta":  args.beta,
        "margin": args.margin
    }
    
    semi_trainer = SemiSupervisedTrainer(config=config, output_dir=args.output, model_name=model_name)
    
    # train_full 內部會依序執行 load_data -> pretrain -> finetune -> plot_training_curve -> compute_threshold
    threshold = semi_trainer.train_full(
        X_normal=X_train_normal, 
        X_attack=X_train_attack,
        threshold_method=args.threshold_method
    )
 
    # ══════════════════════════════════════════════════════
    # Step 5：閾值計算與評估
    # ══════════════════════════════════════════════════════
    print("\n[Step 5] 計算最佳閾值並評估")
    
    # 使用驗證集（從訓練集中再切一小塊或直接用測試集中的正常部分）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_semi = semi_trainer.model
 
    results_semi = evaluate_model(model_semi, X_test_normal, X_test_attack, 
                                 threshold, device, args.output)
    
    print(f"\n  [半監督結果]")
    print(f"    Precision: {results_semi['precision']:.4f}")
    print(f"    Recall:    {results_semi['recall']:.4f}")
    print(f"    F1 Score:  {results_semi['f1']:.4f}")
    print(f"    AUC-ROC:   {results_semi['auc']:.4f}")
 
    # ══════════════════════════════════════════════════════
    # Step 6：對比實驗（選用）
    # ══════════════════════════════════════════════════════
    if args.compare_unsupervised:
        print("\n[Step 6] 執行對比實驗（純無監督）")
        unsup_config = {
            "latent_dim": args.latent,
            "batch_size": args.batch,
            "epochs": args.pretrain_epochs + args.finetune_epochs
        }
        unsup_output = os.path.join(args.output, "unsupervised_baseline")
        unsup_trainer = Trainer(config=unsup_config, output_dir=unsup_output)
        
        # 只用正常流量訓練
        unsup_trainer.from_numpy(X_train_normal)
        unsup_trainer.train()
        
        # 評估
        unsup_threshold = unsup_trainer.compute_threshold_from_numpy(X_test_normal, args.pct)
        results_unsup = evaluate_model(unsup_trainer.model, X_test_normal, X_test_attack,
                                      unsup_threshold, device, unsup_output)
        
        print(f"\n  [無監督基準]")
        print(f"    Precision: {results_unsup['precision']:.4f}")
        print(f"    Recall:    {results_unsup['recall']:.4f}")
        print(f"    F1 Score:  {results_unsup['f1']:.4f}")
        
        # 繪製對比圖
        import matplotlib.pyplot as plt
        metrics = ["Precision", "Recall", "F1 Score", "AUC"]
        semi_vals = [results_semi["precision"], results_semi["recall"], 
                     results_semi["f1"], results_semi["auc"]]
        unsup_vals = [results_unsup["precision"], results_unsup["recall"], 
                      results_unsup["f1"], results_unsup["auc"]]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, semi_vals, width, label="Semi-Supervised")
        plt.bar(x + width/2, unsup_vals, width, label="Unsupervised")
        plt.ylabel("Score")
        plt.title("Performance Comparison: Semi-Supervised vs Unsupervised")
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(args.output, "comparison_report.png"))
        print(f"\n  對比報告已儲存至 {args.output}/comparison_report.png")
 
    # 儲存最終結果
    with open(os.path.join(args.output, "semi_evaluation_report.json"), "w") as f:
        json.dump(results_semi, f, indent=4)
 
    print("\n" + "=" * 60)
    print("  半監督式訓練流程完成！")
    print(f"  輸出目錄：{os.path.abspath(args.output)}")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()