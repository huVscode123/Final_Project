# ============================================================
# train_all_models.py
# 四模型統一訓練腳本
#
# 功能：
#   在一次執行中，依序訓練以下 4 個模型：
#   [1] 非監督  CNN Autoencoder  (simulate / 任意 .npy 資料)
#   [2] 半監督  CIC-DDoS2019
#   [3] 半監督  CICIDS2017
#   [4] 半監督  NSL-KDD
#
# 使用方式（最簡單，使用模擬資料測試所有流程）：
#   python train_all_models.py --mode simulate
#
# 使用真實資料集：
#   python train_all_models.py --mode real \
#       --unsup-normal    data/any_normal.npy \
#       --ddos-dir        data/cicddos2019 \
#       --ids-dir         data/cicids2017 \
#       --kdd-train       data/nslkdd/KDDTrain+.txt \
#       --kdd-test        data/nslkdd/KDDTest+.txt
#
# 單獨訓練某個模型：
#   python train_all_models.py --mode real --only unsup
#   python train_all_models.py --mode real --only ddos  --ddos-dir data/cicddos2019
#   python train_all_models.py --mode real --only ids   --ids-dir  data/cicids2017
#   python train_all_models.py --mode real --only kdd   --kdd-train data/nslkdd/KDDTrain+.txt
# ============================================================

import os
import sys
import json
import time
import argparse
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)


# ──────────────────────────────────────────────────────────
# 模擬資料產生器（無須下載，用於快速驗證流程）
# ──────────────────────────────────────────────────────────

def generate_simulate_data(n_normal: int = 3000,
                           n_attack: int = 1500,
                           n_features: int = 77,
                           image_size: int = 32,
                           seed: int = 42) -> tuple:
    """
    產生模擬正常 / 攻擊流量影像。
    正常流量：以低均值高斯分佈模擬規律流量。
    攻擊流量：在部分特徵維度上加入脈衝訊號模擬攻擊行為。
    """
    rng = np.random.default_rng(seed)
    n_px = image_size * image_size

    # 正常流量
    X_n_feat = rng.normal(0.3, 0.1, size=(n_normal, n_features)).clip(0, 1)
    pad_n    = np.zeros((n_normal, n_px - n_features), dtype=np.float32)
    X_n_flat = np.concatenate([X_n_feat, pad_n], axis=1).astype(np.float32)
    X_normal = X_n_flat.reshape(n_normal, 1, image_size, image_size)

    # 攻擊流量（在特定區域注入異常訊號）
    X_a_feat = rng.normal(0.3, 0.1, size=(n_attack, n_features)).clip(0, 1)
    attack_cols = rng.choice(n_features, size=n_features // 3, replace=False)
    X_a_feat[:, attack_cols] += rng.uniform(0.4, 0.7,
                                             size=(n_attack, len(attack_cols)))
    X_a_feat = X_a_feat.clip(0, 1)
    pad_a    = np.zeros((n_attack, n_px - n_features), dtype=np.float32)
    X_a_flat = np.concatenate([X_a_feat, pad_a], axis=1).astype(np.float32)
    X_attack = X_a_flat.reshape(n_attack, 1, image_size, image_size)

    return X_normal.astype(np.float32), X_attack.astype(np.float32)


def generate_simulate_data_small(image_size: int = 16,
                                  seed: int = 42) -> tuple:
    """NSL-KDD 模擬資料（小影像版）"""
    X_n, X_a = generate_simulate_data(
        n_normal=2000, n_attack=1000,
        n_features=41, image_size=image_size, seed=seed
    )
    cats = np.array(["DoS"] * 400 + ["Probe"] * 300 +
                    ["R2L"] * 200 + ["U2R"] * 100)
    # 重複至 n_attack 筆
    cats = np.resize(cats, 1000)
    return X_n, X_a, cats


# ──────────────────────────────────────────────────────────
# 各模型訓練函式
# ──────────────────────────────────────────────────────────

def train_unsupervised(args, X_normal, X_attack, output_root):
    """模型 [1]：非監督 CNN Autoencoder"""
    from cnn_autoencoder import UnsupervisedTrainer, evaluate

    print("\n" + "═" * 60)
    print("  [1/4] 非監督 CNN Autoencoder 訓練")
    print("═" * 60)

    config = {
        "latent_dim":    args.latent_dim,
        "image_size":    32,
        "batch_size":    args.batch_size,
        "epochs":        args.unsup_epochs,
        "learning_rate": 1e-3,
        "percentile":    args.percentile,
    }
    out_dir = os.path.join(output_root, "model_unsupervised")
    trainer = UnsupervisedTrainer(config, output_dir=out_dir)
    trainer.load_data(X_normal)
    trainer.train()

    result = evaluate(trainer.model, X_normal, X_attack,
                      trainer.threshold, device=trainer.device)
    _save_result(result, out_dir, "unsupervised")
    return result


def train_ddos2019(args, X_normal, X_attack, output_root):
    """模型 [2]：半監督 CIC-DDoS2019"""
    from semi_supervised_cicddos2019 import (
        SemiSupervisedTrainer_DDoS2019, CICDDoS2019Loader
    )
    from cnn_autoencoder import evaluate

    print("\n" + "═" * 60)
    print("  [2/4] 半監督 CIC-DDoS2019 訓練")
    print("═" * 60)

    if X_normal is None:
        loader = CICDDoS2019Loader(
            data_dir   = args.ddos_dir,
            max_normal = args.max_normal,
            max_attack = args.max_attack,
        )
        X_normal, X_attack = loader.load()

    config = {
        "latent_dim":      args.latent_dim,
        "image_size":      32,
        "batch_size":      args.batch_size,
        "pretrain_epochs": args.pretrain_epochs,
        "finetune_epochs": args.finetune_epochs,
        "alpha":           1.0,
        "beta":            0.5,
        "margin":          0.05,
        "percentile":      args.percentile,
    }
    out_dir = os.path.join(output_root, "model_cicddos2019")
    trainer = SemiSupervisedTrainer_DDoS2019(config, output_dir=out_dir)

    import numpy as np
    rng = np.random.default_rng(42)
    idx_n = rng.permutation(len(X_normal))
    split_n = int(len(X_normal) * 0.8)
    X_train_normal = X_normal[idx_n[:split_n]]
    X_test_normal  = X_normal[idx_n[split_n:]]

    idx_a = rng.permutation(len(X_attack))
    split_a = int(len(X_attack) * 0.5)
    X_finetune_attack = X_attack[idx_a[:split_a]]
    X_test_attack     = X_attack[idx_a[split_a:]]

    trainer.train_full(X_train_normal, X_finetune_attack)

    result = evaluate(trainer.model, X_test_normal, X_test_attack,
                      trainer.threshold, device=trainer.device)
    _save_result(result, out_dir, "cicddos2019")
    return result


def train_cicids2017(args, X_normal, X_attack, output_root):
    """模型 [3]：半監督 CICIDS2017"""
    from semi_supervised_cicids2017 import (
        SemiSupervisedTrainer_CICIDS2017, CICIDS2017Loader
    )
    from cnn_autoencoder import evaluate

    print("\n" + "═" * 60)
    print("  [3/4] 半監督 CICIDS2017 訓練")
    print("═" * 60)

    if X_normal is None:
        loader = CICIDS2017Loader(
            data_dir   = args.ids_dir,
            max_normal = args.max_normal,
            max_attack = args.max_attack,
        )
        X_normal, X_attack = loader.load()

    config = {
        "latent_dim":      48,
        "image_size":      32,
        "batch_size":      args.batch_size,
        "pretrain_epochs": args.pretrain_epochs,
        "finetune_epochs": args.finetune_epochs,
        "alpha":           1.0,
        "beta":            0.6,
        "gamma":           0.1,
        "margin":          0.05,
        "percentile":      args.percentile,
    }
    out_dir = os.path.join(output_root, "model_cicids2017")
    trainer = SemiSupervisedTrainer_CICIDS2017(config, output_dir=out_dir)

    import numpy as np
    rng = np.random.default_rng(42)
    idx_n = rng.permutation(len(X_normal))
    split_n = int(len(X_normal) * 0.8)
    X_train_normal = X_normal[idx_n[:split_n]]
    X_test_normal  = X_normal[idx_n[split_n:]]

    idx_a = rng.permutation(len(X_attack))
    split_a = int(len(X_attack) * 0.5)
    X_finetune_attack = X_attack[idx_a[:split_a]]
    X_test_attack     = X_attack[idx_a[split_a:]]

    trainer.train_full(X_train_normal, X_finetune_attack)

    result = evaluate(trainer.model, X_test_normal, X_test_attack,
                      trainer.threshold, device=trainer.device)
    _save_result(result, out_dir, "cicids2017")
    return result


def train_nslkdd(args, X_normal, X_attack, attack_cats, output_root):
    """模型 [4]：半監督 NSL-KDD"""
    from semi_supervised_nslkdd import (
        SemiSupervisedTrainer_NSLKDD, NSLKDDLoader
    )
    from cnn_autoencoder import evaluate

    print("\n" + "═" * 60)
    print("  [4/4] 半監督 NSL-KDD 訓練")
    print("═" * 60)

    if X_normal is None:
        loader = NSLKDDLoader(
            train_path = args.kdd_train,
            test_path  = args.kdd_test,
            image_size = 16,
        )
        X_normal, X_attack, attack_cats = loader.load()

    config = {
        "latent_dim":          16,
        "image_size":          16,
        "variational":         False,
        "batch_size":          64,
        "pretrain_epochs":     args.pretrain_epochs,
        "finetune_epochs":     args.finetune_epochs,
        "alpha":               1.0,
        "beta":                0.8,
        "gamma":               0.01,
        "percentile":          args.percentile,
        "use_category_margin": True,
    }
    out_dir = os.path.join(output_root, "model_nslkdd")
    trainer = SemiSupervisedTrainer_NSLKDD(config, output_dir=out_dir)

    import numpy as np
    rng = np.random.default_rng(42)
    idx_n = rng.permutation(len(X_normal))
    split_n = int(len(X_normal) * 0.8)
    X_train_normal = X_normal[idx_n[:split_n]]
    X_test_normal  = X_normal[idx_n[split_n:]]

    idx_a = rng.permutation(len(X_attack))
    split_a = int(len(X_attack) * 0.5)
    X_finetune_attack = X_attack[idx_a[:split_a]]
    X_test_attack     = X_attack[idx_a[split_a:]]

    if attack_cats is not None:
        attack_cats_finetune = [attack_cats[i] for i in idx_a[:split_a]]
    else:
        attack_cats_finetune = None

    trainer.train_full(X_train_normal, X_finetune_attack, attack_cats=attack_cats_finetune)

    result = evaluate(trainer.model, X_test_normal, X_test_attack,
                      trainer.threshold, device=trainer.device)
    _save_result(result, out_dir, "nslkdd")
    return result


# ──────────────────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────────────────

def _save_result(result: dict, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "eval_result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  評估結果已儲存: {path}")


def _print_summary(all_results: dict):
    print("\n" + "═" * 60)
    print("  訓練完成摘要")
    print("═" * 60)
    header = f"  {'模型':<22} {'F1':>6} {'AUC':>6} {'Precision':>10} {'Recall':>8}"
    print(header)
    print("  " + "-" * 56)
    for name, res in all_results.items():
        if res:
            print(f"  {name:<22} "
                  f"{res.get('f1', 0):>6.4f} "
                  f"{res.get('auc', 0):>6.4f} "
                  f"{res.get('precision', 0):>10.4f} "
                  f"{res.get('recall', 0):>8.4f}")
    print("═" * 60)


# ──────────────────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="四模型統一訓練腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 模擬資料（快速驗證，不需下載資料集）
  python train_all_models.py --mode simulate

  # 真實資料（全部）
  python train_all_models.py --mode real \\
      --ddos-dir  data/cicddos2019 \\
      --ids-dir   data/cicids2017 \\
      --kdd-train data/nslkdd/KDDTrain+.txt \\
      --kdd-test  data/nslkdd/KDDTest+.txt

  # 只訓練 NSL-KDD
  python train_all_models.py --mode real --only kdd \\
      --kdd-train data/nslkdd/KDDTrain+.txt
        """
    )

    # 模式
    parser.add_argument("--mode", default="simulate",
                        choices=["simulate", "real"],
                        help="simulate=使用模擬資料  real=使用真實資料集")
    parser.add_argument("--only", default="all",
                        choices=["all", "unsup", "ddos", "ids", "kdd"],
                        help="只訓練指定模型（預設: all）")

    # 真實資料路徑
    parser.add_argument("--unsup-normal", default=None,
                        help="[1] 非監督正常流量 .npy 路徑")
    parser.add_argument("--unsup-attack",  default=None,
                        help="[1] 非監督攻擊流量 .npy（僅評估用）")
    parser.add_argument("--ddos-dir",  default=None,
                        help="[2] CIC-DDoS2019 CSV 目錄")
    parser.add_argument("--ids-dir",   default=None,
                        help="[3] CICIDS2017 CSV 目錄")
    parser.add_argument("--kdd-train", default=None,
                        help="[4] NSL-KDD KDDTrain+.txt 路徑")
    parser.add_argument("--kdd-test",  default=None,
                        help="[4] NSL-KDD KDDTest+.txt 路徑（可選）")

    # 共用超參數
    parser.add_argument("--output",          default="output")
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--latent-dim",      type=int,   default=32)
    parser.add_argument("--unsup-epochs",    type=int,   default=60,
                        help="非監督模型訓練 epochs")
    parser.add_argument("--pretrain-epochs", type=int,   default=50,
                        help="半監督 Phase 1 epochs")
    parser.add_argument("--finetune-epochs", type=int,   default=30,
                        help="半監督 Phase 2 epochs")
    parser.add_argument("--percentile",      type=float, default=95.0)
    parser.add_argument("--max-normal",      type=int,   default=50000)
    parser.add_argument("--max-attack",      type=int,   default=25000)
    parser.add_argument("--seed",            type=int,   default=42)

    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  網路攻擊異常偵測 — 四模型訓練腳本")
    print(f"  模式: {args.mode.upper()}   目標: {args.only.upper()}")
    print("═" * 60)

    t_total = time.time()
    all_results = {}

    # ── 模式 A：模擬資料 ──────────────────────────────────
    if args.mode == "simulate":
        print("\n[模擬模式] 產生模擬訓練資料...")
        # 32×32 模擬資料（[1][2][3] 共用）
        X_sim_n32, X_sim_a32 = generate_simulate_data(
            n_normal=3000, n_attack=1500,
            n_features=77, image_size=32, seed=args.seed
        )
        # 16×16 模擬資料（[4] NSL-KDD 用）
        X_sim_n16, X_sim_a16, sim_cats = generate_simulate_data_small(
            image_size=16, seed=args.seed
        )
        print(f"  32×32 正常: {X_sim_n32.shape}  攻擊: {X_sim_a32.shape}")
        print(f"  16×16 正常: {X_sim_n16.shape}  攻擊: {X_sim_a16.shape}")

        run_all = (args.only == "all")

        if run_all or args.only == "unsup":
            all_results["[1] Unsupervised"] = train_unsupervised(
                args, X_sim_n32, X_sim_a32, args.output
            )
        if run_all or args.only == "ddos":
            all_results["[2] DDoS2019 (Semi)"] = train_ddos2019(
                args, X_sim_n32, X_sim_a32, args.output
            )
        if run_all or args.only == "ids":
            all_results["[3] CICIDS2017 (Semi)"] = train_cicids2017(
                args, X_sim_n32, X_sim_a32, args.output
            )
        if run_all or args.only == "kdd":
            all_results["[4] NSL-KDD (Semi)"] = train_nslkdd(
                args, X_sim_n16, X_sim_a16, sim_cats, args.output
            )

    # ── 模式 B：真實資料 ──────────────────────────────────
    else:
        run_all = (args.only == "all")

        if run_all or args.only == "unsup":
            if not args.unsup_normal:
                print("[警告] --unsup-normal 未指定，跳過非監督模型")
                all_results["[1] Unsupervised"] = None
            else:
                X_n = np.load(args.unsup_normal)
                X_a = np.load(args.unsup_attack) \
                      if args.unsup_attack else np.zeros((1, 1, 32, 32),
                                                          dtype=np.float32)
                all_results["[1] Unsupervised"] = train_unsupervised(
                    args, X_n, X_a, args.output
                )

        if run_all or args.only == "ddos":
            if not args.ddos_dir:
                print("[警告] --ddos-dir 未指定，跳過 DDoS2019")
                all_results["[2] DDoS2019 (Semi)"] = None
            else:
                all_results["[2] DDoS2019 (Semi)"] = train_ddos2019(
                    args, None, None, args.output
                )

        if run_all or args.only == "ids":
            if not args.ids_dir:
                print("[警告] --ids-dir 未指定，跳過 CICIDS2017")
                all_results["[3] CICIDS2017 (Semi)"] = None
            else:
                all_results["[3] CICIDS2017 (Semi)"] = train_cicids2017(
                    args, None, None, args.output
                )

        if run_all or args.only == "kdd":
            if not args.kdd_train:
                print("[警告] --kdd-train 未指定，跳過 NSL-KDD")
                all_results["[4] NSL-KDD (Semi)"] = None
            else:
                all_results["[4] NSL-KDD (Semi)"] = train_nslkdd(
                    args, None, None, None, args.output
                )

    # ── 摘要輸出 ───────────────────────────────────────────
    total_time = time.time() - t_total
    _print_summary({k: v for k, v in all_results.items() if v})

    # 儲存總覽報告
    report = {k: v for k, v in all_results.items() if v}
    with open(os.path.join(args.output, "summary_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  總訓練時間: {total_time/60:.1f} 分鐘")
    print(f"  輸出目錄  : {os.path.abspath(args.output)}")
    print("═" * 60)


if __name__ == "__main__":
    main()
