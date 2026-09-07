# ============================================================
# run_threshold_tuning.py - 模型訓練、閾值調校、消融實驗與模型壓縮評估
#
# 使用資料集：
#   CIC-IDS2017（主要）：真實 DoS / DDoS / PortScan 攻擊流量
#     下載：https://www.unb.ca/cic/datasets/ids-2017.html
#     放置：data/cicids2017/*.csv
#
#   NSL-KDD（輔助對照）：較小、格式簡單、適合快速驗證
#     下載：https://www.unb.ca/cic/datasets/nsl.html
#     放置：data/nslkdd/KDDTrain+.txt
#
#   模擬資料（開發測試，無需下載）：
#     python core/run_threshold_tuning.py --dataset simulate
#
# 執行方式：
#   # 完整流程（訓練 + 閾值調校）
#   python core/run_threshold_tuning.py --dataset simulate
#
#   # 使用真實資料集
#   python core/run_threshold_tuning.py --dataset cicids2017 --data-dir data/cicids2017
#   python core/run_threshold_tuning.py --dataset nslkdd --data-dir data/nslkdd
#
#   # 消融實驗（完整六維度）
#   python core/run_threshold_tuning.py --dataset simulate --ablation
#
#   # 僅執行特定消融實驗
#   python core/run_threshold_tuning.py --dataset simulate --ablation --ablation-exp latent arch
#
#   # 模型壓縮評估
#   python core/run_threshold_tuning.py --dataset simulate --compress
#
#   # 僅執行特定壓縮方法
#   python core/run_threshold_tuning.py --dataset simulate --compress --compress-method unstructured distill
#
#   # 全部（訓練 + 消融實驗 + 壓縮評估）
#   python core/run_threshold_tuning.py --dataset simulate --ablation --compress
#
#   # 只評估現有模型（跳過訓練）
#   python core/run_threshold_tuning.py --eval-only --model output/model/best_model.pt
#
# 輸出目錄結構：
#   output/model/
#     best_model.pt                     最佳模型權重
#     training_curve.png                訓練曲線
#     threshold_curve.png               閾值調校曲線（Precision/Recall/F1 vs 百分位）
#     threshold_report.json             各閾值完整評估結果
#
#   output/ablation/                    消融實驗（--ablation 時）
#     ablation_latent_dim.png           Latent Dimension 實驗
#     ablation_arch_scale.png           架構寬度實驗
#     ablation_image_size.png           影像尺寸實驗
#     ablation_field_mask.png           欄位遮罩實驗
#     ablation_data_size.png            訓練資料量實驗
#     ablation_activation.png           激活函數實驗
#     ablation_summary.png              全部實驗摘要圖
#     ablation_report.json              完整實驗結果
#
#   output/compression/                 模型壓縮評估（--compress 時）
#     pruned_unstructured_*.pt          非結構化剪枝模型
#     pruned_structured_*.pt            結構化剪枝模型
#     quantized_model.pt                PTQ INT8 量化模型
#     student_ld*.pt                    知識蒸餾 Student 模型
#     compression_tradeoff.png          壓縮-效能 Trade-off 圖
#     compression_report.json           完整壓縮評估結果
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
        print("  請執行: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="模型訓練、閾值調校、消融實驗與模型壓縮評估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  # 完整流程（訓練 + 閾值調校）
  python core/run_threshold_tuning.py --dataset simulate

  # 消融實驗（六個維度）
  python core/run_threshold_tuning.py --dataset simulate --ablation

  # 僅執行特定消融實驗
  python core/run_threshold_tuning.py --dataset simulate --ablation --ablation-exp latent arch

  # 模型壓縮評估（全部方法）
  python core/run_threshold_tuning.py --dataset simulate --compress

  # 僅執行特定壓縮方法
  python core/run_threshold_tuning.py --dataset simulate --compress --compress-method unstructured distill

  # 全部（訓練 + 消融 + 壓縮）
  python core/run_threshold_tuning.py --dataset simulate --ablation --compress

  # 只評估現有模型（跳過訓練）
  python core/run_threshold_tuning.py --eval-only --model output/model/best_model.pt
        """
    )

    # ── 資料集參數 ────────────────────────────────────────────────
    parser.add_argument("--dataset",   default="simulate",
                        choices=["simulate", "nslkdd", "cicids2017", "cicddos2019", "cic-ddos2019"],
                        help="使用的資料集（預設: simulate）")
    parser.add_argument("--data-dir",  default=None,
                        help="真實資料集目錄（cicids2017/nslkdd 時需指定）")

    # ── 訓練參數 ──────────────────────────────────────────────────
    parser.add_argument("--epochs",    type=int,   default=100,
                        help="訓練輪數（預設: 100）")
    parser.add_argument("--latent",    type=int,   default=32,
                        help="CNN Autoencoder 潛在空間維度（預設: 32）")
    parser.add_argument("--batch",     type=int,   default=32,
                        help="批次大小（預設: 32）")
    parser.add_argument("--output",    default="output/model",
                        help="主要輸出目錄（預設: output/model）")
    parser.add_argument("--eval-only", action="store_true",
                        help="跳過訓練，只載入現有模型進行評估")
    parser.add_argument("--model",     default=None,
                        help="--eval-only 時指定的模型路徑")

    # ── 消融實驗參數 ──────────────────────────────────────────────
    parser.add_argument("--ablation",  action="store_true",
                        help="執行完整消融實驗（六個設計決策維度）")
    parser.add_argument("--ablation-exp",
                        nargs="+",
                        default=["all"],
                        choices=["all", "latent", "arch", "size", "mask", "data", "act"],
                        help="選擇要執行的消融實驗子集（預設: all）")
    parser.add_argument("--ablation-epochs", type=int, default=25,
                        help="消融實驗每次訓練的 epoch 數（預設: 25，較少以節省時間）")
    parser.add_argument("--ablation-repeats", type=int, default=3,
                        help="每個設定重複訓練的次數（預設: 3，用於估計 mean/std/CI）")
    parser.add_argument("--ablation-output", default="output/ablation",
                        help="消融實驗輸出目錄（預設: output/ablation）")

    # ── 模型壓縮參數 ──────────────────────────────────────────────
    parser.add_argument("--compress",  action="store_true",
                        help="執行模型壓縮評估（剪枝 / 量化 / 知識蒸餾）")
    parser.add_argument("--compress-method",
                        nargs="+",
                        default=["all"],
                        choices=["all", "unstructured", "structured", "ptq", "distill"],
                        help="選擇要執行的壓縮方法（預設: all）")
    parser.add_argument("--compress-output", default="output/compression",
                        help="壓縮評估輸出目錄（預設: output/compression）")
    parser.add_argument("--finetune-epochs", type=int, default=10,
                        help="剪枝後微調的 epoch 數（預設: 10）")
    parser.add_argument("--distill-epochs",  type=int, default=50,
                        help="知識蒸餾的 epoch 數（預設: 50）")

    args = parser.parse_args()

    # 自動調整輸出目錄，使其包含資料集名稱，避免多個模型結果互相覆蓋
    if args.output == "output/model":
        args.output = f"output/{args.dataset}/model"
    if args.ablation_output == "output/ablation":
        args.ablation_output = f"output/{args.dataset}/ablation"
    if args.compress_output == "output/compression":
        args.compress_output = f"output/{args.dataset}/compression"

    print("\n" + "=" * 65)
    print(f"  模型訓練、閾值調校與評估（資料集：{args.dataset}）")
    print("=" * 65)

    if not check_torch():
        sys.exit(1)

    import torch
    from trainer import Trainer
    from dataset_loader import DatasetFactory
    from threshold_tuner import ThresholdTuner
    from cnn_autoencoder import CNNAutoencoder

    # ══════════════════════════════════════════════════════════════
    # Step 1：載入資料集
    # ══════════════════════════════════════════════════════════════
    print(f"\n[Step 1] 載入資料集：{args.dataset}")

    load_kwargs = {}
    if args.dataset in ("cicids2017", "cicddos2019"):
        load_kwargs["max_normal"] = 50000
        load_kwargs["max_attack"] = 30000

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

    # 儲存為 .npy 供消融實驗與壓縮評估使用
    dataset_dir = f"output/dataset_{args.dataset}"
    paths = DatasetFactory.save_as_npy(X_normal, X_attack, y, dataset_dir)
    normal_npy = paths["normal"]
    attack_npy = paths["attack"]

    # ══════════════════════════════════════════════════════════════
    # Step 2：訓練模型
    # ══════════════════════════════════════════════════════════════
    model_path  = args.model or os.path.join(args.output, "best_model.pt")
    config_path = os.path.join(args.output, "training_result.json")

    if args.eval_only:
        print(f"\n[Step 2] 載入現有模型: {model_path}")
        model, threshold = Trainer.load_model(model_path, config_path, args.latent)
    else:
        print("\n[Step 2] 訓練 CNN Autoencoder")
        config = {
            "latent_dim":    args.latent,
            "batch_size":    args.batch,
            "epochs":        args.epochs,
            "learning_rate": 1e-3,
        }
        trainer = Trainer(config=config, output_dir=args.output)
        trainer.load_data(normal_npy)
        trainer.train()
        trainer.plot_training_curve()
        model = trainer.model

    # ══════════════════════════════════════════════════════════════
    # Step 3：閾值調校與分析
    # ══════════════════════════════════════════════════════════════
    print("\n[Step 3] 執行閾值調校分析 (ThresholdTuner)")
    tuner = ThresholdTuner(model)

    # ── [修正 P1-3] 閾值校準只使用驗證集 ──
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_normal))
    split = int(len(X_normal) * 0.8)
    X_normal_val = X_normal[idx[split:]]

    # 掃描不同百分位數的成效
    report = tuner.scan_percentiles(X_normal_val, X_attack)

    # 找出最佳閾值
    best = tuner.find_best_threshold(report, metric="f1")

    # 儲存閾值掃描報告
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "threshold_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    # 繪製閾值曲線（Precision/Recall/F1 vs 百分位）
    tuner.plot_threshold_curve(report, output_dir=args.output)

    print(f"\n  最佳閾值 (by F1)：{best['threshold']:.6f}  "
          f"F1={best['f1']:.4f}  Recall={best['recall']:.4f}  "
          f"Precision={best['precision']:.4f}")

    # ══════════════════════════════════════════════════════════════
    # Step 4：消融實驗（--ablation 旗標觸發）
    # ══════════════════════════════════════════════════════════════
    if args.ablation and not args.eval_only:
        print(f"\n[Step 4] 執行消融實驗 (Ablation Study)")
        print(f"  輸出目錄：{os.path.abspath(args.ablation_output)}")
        print(f"  每次實驗 Epoch：{args.ablation_epochs}")
        print(f"  實驗範圍：{args.ablation_exp}")

        from ablation_study import AblationStudy

        study = AblationStudy(
            X_normal    = X_normal,
            X_attack    = X_attack,
            output_dir  = args.ablation_output,
            epochs      = args.ablation_epochs,
            n_repeats   = args.ablation_repeats,
        )

        run_all_exp = "all" in args.ablation_exp
        all_ablation = {}

        if run_all_exp or "latent" in args.ablation_exp:
            all_ablation["latent_dim"]    = study.run_latent_dim()
        if run_all_exp or "arch" in args.ablation_exp:
            all_ablation["arch_scale"]    = study.run_arch_scale()
        if run_all_exp or "size" in args.ablation_exp:
            all_ablation["image_size"]    = study.run_image_size()
        if run_all_exp or "mask" in args.ablation_exp:
            all_ablation["field_masking"] = study.run_field_masking()
        if run_all_exp or "data" in args.ablation_exp:
            all_ablation["data_size"]     = study.run_training_data_size()
        if run_all_exp or "act" in args.ablation_exp:
            all_ablation["activation"]    = study.run_activation_fn()

        # 繪製摘要圖與儲存報告
        if all_ablation:
            study.plot_summary(all_ablation)
            study.save_report(all_ablation)

        print(f"\n  [Step 4] 消融實驗完成")
        print(f"  結果輸出：{os.path.abspath(args.ablation_output)}")

    elif args.ablation and args.eval_only:
        print("\n[Step 4] 注意：--eval-only 模式下不執行消融實驗（需要重新訓練）")

    # ══════════════════════════════════════════════════════════════
    # Step 5：模型壓縮評估（--compress 旗標觸發）
    # ══════════════════════════════════════════════════════════════
    if args.compress:
        print(f"\n[Step 5] 執行模型壓縮評估 (Model Compression)")
        print(f"  輸出目錄：{os.path.abspath(args.compress_output)}")
        print(f"  壓縮方法：{args.compress_method}")
        print(f"  剪枝微調 Epoch：{args.finetune_epochs}")
        print(f"  知識蒸餾 Epoch：{args.distill_epochs}")

        from model_compression import ModelCompressor

        compressor = ModelCompressor(
            model      = model,
            X_normal   = X_normal,
            X_attack   = X_attack,
            output_dir = args.compress_output,
        )

        run_all_comp = "all" in args.compress_method
        all_compress = {"baseline": compressor.baseline}

        if run_all_comp or "unstructured" in args.compress_method:
            all_compress["unstructured"] = compressor.run_unstructured_pruning(
                finetune_epochs=args.finetune_epochs
            )
        if run_all_comp or "structured" in args.compress_method:
            all_compress["structured"] = compressor.run_structured_pruning(
                finetune_epochs=args.finetune_epochs
            )
        if run_all_comp or "ptq" in args.compress_method:
            all_compress["ptq"] = compressor.run_ptq()
        if run_all_comp or "distill" in args.compress_method:
            all_compress["distillation"] = compressor.run_knowledge_distillation(
                epochs=args.distill_epochs
            )

        # 繪製 Trade-off 圖與儲存報告
        compressor.plot_tradeoff(all_compress)
        compressor.save_report(all_compress)

        print(f"\n  [Step 5] 模型壓縮評估完成")
        print(f"  結果輸出：{os.path.abspath(args.compress_output)}")

    # ══════════════════════════════════════════════════════════════
    # 完成摘要
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  流程完成！")
    print(f"  模型輸出  ：{os.path.abspath(args.output)}")
    if args.ablation:
        print(f"  消融實驗  ：{os.path.abspath(args.ablation_output)}")
    if args.compress:
        print(f"  壓縮評估  ：{os.path.abspath(args.compress_output)}")
    print("=" * 65)


if __name__ == "__main__":
    main()