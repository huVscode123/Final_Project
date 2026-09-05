#!/usr/bin/env python3
# ============================================================
# core/run_gradcam.py  - Grad-CAM 視覺化解釋整合執行腳本
# ============================================================
"""
Grad-CAM 視覺化解釋主程式

放置位置：core/run_gradcam.py（與 run_training.py、run_full_pipeline.py 同層）

整合流程：
  1. 載入訓練好的 CNN Autoencoder 模型（output/model/best_model.pt）
  2. 載入資料集（正常 / 攻擊流量），支援 simulate / nslkdd / cicids2017 / cicddos2019
  3. 執行指定 Grad-CAM 變體（gradcam / gradcam++ / scorecam）
  4. 批次視覺化 + 統計分析
  5. 匯出 JSON / HTML 報告到 output/gradcam/

使用範例：
  # 完整分析（預設 gradcam 變體）
  python core/run_gradcam.py --dataset simulate

  # 指定模型路徑與變體
  python core/run_gradcam.py --model output/model/best_model.pt \\
                              --dataset simulate --variant gradcam++

  # 指定目標層（對應 CNNAutoencoderFlex 架構）
  python core/run_gradcam.py --dataset simulate --layer encoder_conv.6

  # 三種變體同時執行並比較
  python core/run_gradcam.py --dataset simulate --all-variants

  # 快速預覽（只生成個別樣本圖，跳過批次統計）
  python core/run_gradcam.py --dataset simulate --viz-only --n-viz 20

  # Score-CAM（無梯度，適用於解釋困難樣本）
  python core/run_gradcam.py --dataset simulate --variant scorecam
"""

import os
import sys
import json
import argparse
import numpy as np

# ── [P1-1 修正] 往上一層才是 core/ ──
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)


# ── 預設特徵名稱（NSL-KDD 風格，對應 32×32 影像的行維度）────────
_DEFAULT_FEATURE_NAMES = [
    "duration",       "protocol_type",   "service",          "flag",
    "src_bytes",      "dst_bytes",       "land",             "wrong_fragment",
    "urgent",         "hot",             "num_failed_logins","logged_in",
    "num_compromised","root_shell",      "su_attempted",     "num_root",
    "num_file_creations","num_shells",   "num_access_files", "num_outbound_cmds",
    "is_host_login",  "is_guest_login",  "count",            "srv_count",
    "serror_rate",    "srv_serror_rate", "rerror_rate",      "srv_rerror_rate",
    "same_srv_rate",  "diff_srv_rate",   "srv_diff_host_rate","dst_host_count",
]


def check_torch():
    try:
        import torch
        print(f"  PyTorch {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        return True
    except ImportError:
        print("  PyTorch 未安裝，請執行: pip install torch")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grad-CAM 視覺化解釋整合執行腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 資料與模型 ──────────────────────────────────────────
    parser.add_argument("--dataset", default="simulate",
                        choices=["simulate", "nslkdd", "cicids2017", "cicddos2019"],
                        help="使用的資料集（預設: simulate）")
    parser.add_argument("--data-dir", default=None,
                        help="真實資料集目錄（cicids2017/nslkdd 時需指定）")
    parser.add_argument("--model",   default="output/model/best_model.pt",
                        help="模型權重路徑（預設: output/model/best_model.pt）")
    parser.add_argument("--latent",  type=int, default=32,
                        help="CNN Autoencoder latent_dim（需與訓練時一致，預設: 32）")

    # ── Grad-CAM 設定 ───────────────────────────────────────
    parser.add_argument("--variant", default="gradcam",
                        choices=["gradcam", "gradcam++", "scorecam"],
                        help="Grad-CAM 變體（預設: gradcam）")
    parser.add_argument("--layer",   default=None,
                        help="目標層名稱（預設: None，自動偵測 Encoder 最後一個 Conv2d 層；"
                             "同時相容 CNNAutoencoder 巢狀命名與 CNNAutoencoderFlex 扁平命名）")
    parser.add_argument("--target-type", default="mse",
                        choices=["mse", "pixel", "channel"],
                        help="目標函數類型（預設: mse）")
    parser.add_argument("--all-variants", action="store_true",
                        help="同時執行三種 Grad-CAM 變體並比較特徵重要性")

    # ── 分析設定 ────────────────────────────────────────────
    parser.add_argument("--n-samples",  type=int, default=200,
                        help="每類最多分析樣本數（預設: 200）")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批次大小（預設: 32）")
    parser.add_argument("--viz-only",   action="store_true",
                        help="只生成個別樣本視覺化圖，跳過批次統計分析")
    parser.add_argument("--n-viz",      type=int, default=12,
                        help="視覺化個別樣本數（預設: 12）")
    parser.add_argument("--smooth",     action="store_true", default=True,
                        help="對 CAM 套用高斯平滑（預設: 啟用）")
    parser.add_argument("--no-smooth",  dest="smooth", action="store_false",
                        help="停用高斯平滑")
    parser.add_argument("--colormap",   default="jet",
                        help="熱力圖色彩映射（預設: jet）")

    # ── 輸出 ────────────────────────────────────────────────
    parser.add_argument("--output",   default=None,
                        help="輸出根目錄（預設: output/<dataset>/gradcam）")
    parser.add_argument("--no-html",  action="store_true",
                        help="不生成 HTML 報告（只生成 JSON）")

    args = parser.parse_args()

    # 自動調整輸出目錄，使其包含資料集名稱
    if args.output is None:
        args.output = f"output/{args.dataset}/gradcam"

    return args


def load_model(model_path: str, latent_dim: int, device):
    """
    載入 CNN Autoencoder 模型。

    [Bug 1 修正 — 2026] 舊版無條件優先 try `from ablation_study import
    CNNAutoencoderFlex`，但 ablation_study.py 在同目錄下必定 import 成功，
    導致 except ImportError 永遠不會觸發 —— 程式因此永遠使用
    CNNAutoencoderFlex（扁平層命名 "encoder_conv.*"），而非
    trainer.py / run_training.py 實際訓練並存檔的 CNNAutoencoder
    （巢狀層命名 "encoder.conv_layers.*"）。兩者 state_dict 的 key
    完全不同，load_state_dict() 必定 RuntimeError，被 except 捕捉後
    悄悄回退成「隨機初始化」的模型，導致所有後續 Grad-CAM 熱力圖、
    特徵重要性分析都是在未訓練模型上跑出來的無意義結果。

    修正：改為「先讀取 checkpoint，再依 state_dict 的 key 前綴判斷
    對應的模型類別」，確保載入的架構與訓練時儲存的架構一致。
    """
    import torch
    from cnn_autoencoder import CNNAutoencoder

    if not os.path.exists(model_path):
        print(f"  [警告] 找不到 {model_path}，使用隨機初始化的 CNNAutoencoder（僅用於測試）")
        return CNNAutoencoder(latent_dim=latent_dim).to(device).eval()

    state = torch.load(model_path, map_location=device)

    # 相容多種儲存格式（trainer.py / EarlyStopping.restore_best 輸出）
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    # ── 依 checkpoint 的實際 key 前綴判斷架構，而非盲目猜測 ──
    keys = list(state_dict.keys())
    is_flex_arch = any(
        k.startswith(("encoder_conv", "encoder_fc", "decoder_fc", "decoder_deconv"))
        for k in keys
    )

    if is_flex_arch:
        from ablation_study import CNNAutoencoderFlex
        model = CNNAutoencoderFlex(latent_dim=latent_dim)
        print("  偵測到 CNNAutoencoderFlex（消融實驗）架構的權重")
    else:
        model = CNNAutoencoder(latent_dim=latent_dim)
        print("  偵測到標準 CNNAutoencoder 架構的權重")

    try:
        model.load_state_dict(state_dict)
        print(f"  模型載入成功：{model_path}")
    except RuntimeError as e:
        print(f"  [錯誤] 模型權重載入失敗（架構仍不相符）：{e}")
        print(f"  [警告] 使用隨機初始化的模型 — 後續 Grad-CAM 結果將不具參考價值！")

    return model.to(device).eval()


def run_single_variant(model, X_normal, X_attack, args, variant, output_dir, device, feature_names):
    """
    執行單一 Grad-CAM 變體的完整流程：
      視覺化 → 批次分析 → 匯出圖表
    """
    # ← 從 core/gradcam/ 套件 import（對應新的模組名稱）
    from cnn_gradcam import GradCAM, GradCAMVisualizer, GradCAMAnalyzer, GradCAMReporter

    print(f"\n{'─'*60}")
    print(f"  Grad-CAM 變體：{variant}  |  目標層：{args.layer}")
    print(f"{'─'*60}")

    os.makedirs(output_dir, exist_ok=True)

    # ── 初始化 ──────────────────────────────────────────────
    cam = GradCAM(
        model        = model,
        target_layer = args.layer,
        variant      = variant,
        target_type  = args.target_type,
        device       = device,
    )
    visualizer = GradCAMVisualizer(colormap=args.colormap, alpha=0.5)

    print(f"  目標層已解析：{cam._resolved_layer}")
    print(f"  可用 Conv2d 層：{cam.list_target_layers()}")

    # ── Step 1：個別樣本視覺化 ──────────────────────────────
    print(f"\n  [Step 1] 個別樣本視覺化（各 {args.n_viz} 張）...")

    n_viz      = min(args.n_viz, len(X_normal), len(X_attack))
    viz_normal = X_normal[:n_viz]
    viz_attack = X_attack[:n_viz]

    hm_normal  = cam.generate(viz_normal, smooth=args.smooth)
    hm_attack  = cam.generate(viz_attack, smooth=args.smooth)

    # 取得重建輸出（供 plot_comparison 使用）
    analyzer_tmp  = GradCAMAnalyzer(cam, model, device=device, batch_size=args.batch_size)
    rec_normal    = analyzer_tmp.get_reconstructions(viz_normal)
    rec_attack    = analyzer_tmp.get_reconstructions(viz_attack)
    score_normal  = analyzer_tmp._compute_recon_scores(viz_normal)
    score_attack  = analyzer_tmp._compute_recon_scores(viz_attack)

    visualizer.plot_batch(
        X          = np.concatenate([viz_normal, viz_attack]),
        X_hat      = np.concatenate([rec_normal,  rec_attack]),
        heatmaps   = np.concatenate([hm_normal,   hm_attack]),
        labels     = ["Normal"] * n_viz + ["Attack"] * n_viz,
        scores     = list(score_normal) + list(score_attack),
        output_dir = os.path.join(output_dir, "sample_visualizations"),
        max_samples= n_viz * 2,
        field_names= feature_names,
    )

    if args.viz_only:
        print("  [viz-only 模式] 跳過批次統計分析。")
        return None, None, {}

    # ── Step 2：批次統計分析 ────────────────────────────────
    print(f"\n  [Step 2] 批次統計分析（最多各 {args.n_samples} 樣本）...")

    analyzer = GradCAMAnalyzer(
        grad_cam   = cam,
        model      = model,
        device     = device,
        batch_size = args.batch_size,
    )
    report  = analyzer.analyze(X_normal, X_attack,
                                max_samples_each=args.n_samples,
                                field_names=feature_names)
    summary = analyzer.summarize(report)

    # ── Step 3：繪製分析圖表 ────────────────────────────────
    print(f"\n  [Step 3] 繪製分析圖表...")
    analysis_dir = os.path.join(output_dir, "analysis")

    analyzer.plot_mean_cam_comparison(report, analysis_dir)
    analyzer.plot_feature_importance(report, analysis_dir)
    analyzer.plot_cam_intensity_distribution(report, analysis_dir)

    all_results  = report.normal_results + report.attack_results
    visualizer.plot_score_vs_cam_intensity(
        scores         = np.array([r.recon_score for r in all_results]),
        cam_intensities= np.array([r.cam_mean    for r in all_results]),
        labels         = [r.label for r in all_results],
        output_dir     = analysis_dir,
    )
    visualizer.plot_summary_grid(
        normal_heatmaps= np.stack([r.heatmap for r in report.normal_results]),
        attack_heatmaps= np.stack([r.heatmap for r in report.attack_results]),
        output_dir     = analysis_dir,
        n_cols         = 8,
    )

    # ── 統計輸出 ────────────────────────────────────────────
    ns, as_ = summary.get("normal", {}), summary.get("attack", {})
    print(f"\n  [{variant}] 統計摘要：")
    print(f"    正常 - 平均重建誤差: {ns.get('recon_score_mean', 0):.5f} ± {ns.get('recon_score_std', 0):.5f}")
    print(f"    攻擊 - 平均重建誤差: {as_.get('recon_score_mean', 0):.5f} ± {as_.get('recon_score_std', 0):.5f}")
    sep_ratio = as_.get("recon_score_mean", 1e-9) / (ns.get("recon_score_mean", 1e-9) + 1e-9)
    print(f"    分離比:              {sep_ratio:.2f}x")
    for feat in summary.get("top5_important_features", []):
        print(f"    [{feat['name']:30s}] {feat['importance']:+.5f}")

    image_paths = {
        k: os.path.join(analysis_dir, f"{k}.png")
        for k in ["mean_cam_comparison", "feature_importance",
                  "cam_intensity_distribution", "score_vs_cam_intensity",
                  "gradcam_summary_grid"]
    }
    return report, summary, image_paths


def compare_variants(results_dict: dict, output_dir: str):
    """三種變體的特徵重要性並排比較圖。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    variants = list(results_dict.keys())
    if len(variants) < 2:
        return

    fig, axes = plt.subplots(1, len(variants), figsize=(8 * len(variants), 6), sharey=True)
    if len(variants) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Grad-CAM 變體比較：特徵重要性 Top-10", color="#e6edf3", fontsize=14)

    for ax, (variant, (report, _, _)) in zip(axes, results_dict.items()):
        ax.set_facecolor("#161b22")
        if report is None or report.feature_importance is None:
            continue
        importance = report.feature_importance
        names  = report.feature_names or [f"F{i}" for i in range(len(importance))]
        top10  = np.argsort(np.abs(importance))[-10:][::-1]
        vals   = importance[top10]
        lbls   = [names[i] if i < len(names) else f"F{i}" for i in top10]
        colors = ["#ff6b6b" if v > 0 else "#58a6ff" for v in vals]

        ax.barh(range(10), vals[::-1], color=colors[::-1])
        ax.set_yticks(range(10))
        ax.set_yticklabels(lbls[::-1], color="#e6edf3", fontsize=9)
        ax.set_title(variant, color="#e6edf3", fontsize=12)
        ax.tick_params(colors="#e6edf3")
        ax.axvline(0, color="#444", linewidth=0.8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "variant_comparison.png")
    plt.savefig(save_path, bbox_inches="tight", facecolor="#0d1117", dpi=130)
    plt.close(fig)
    print(f"\n[Compare] 變體比較圖已儲存至: {save_path}")


def run_gradcam_analysis(pcap_path, output_dir, model_path="output/model/best_model.pt"):
    """
    為 Django Task 提供整合式的 Grad-CAM 分析進入點。

    [Bug 4 修正 — 2026] 舊版呼叫 `AnomalyScorer(model_path=model_path,
    device=device)`，但 AnomalyScorer.__init__ 的簽名是
    `(self, model: CNNAutoencoder, threshold=None, device=None)`，
    並不接受 `model_path` 參數、也不會自行從路徑載入模型 ——
    只要呼叫這個函式就會直接拋出 TypeError。
    修正：先用 Trainer.load_model() 從路徑載入模型與訓練時計算好的
    閾值，再把「已實例化的模型物件」正確地傳給 AnomalyScorer。
    """
    import torch
    from pcap_analyzer import PcapAnalyzer
    from anomaly_scorer import AnomalyScorer
    from trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 解析 PCAP
    analyzer = PcapAnalyzer(pcap_path)
    analyzer.load()
    packets = analyzer.packets

    # 2. 載入已訓練模型與閾值，再建立 Scorer
    config_path = os.path.join(os.path.dirname(model_path), "training_result.json")
    model, threshold = Trainer.load_model(
        model_path,
        config_path=config_path if os.path.exists(config_path) else None,
    )
    model = model.to(device)
    scorer = AnomalyScorer(model, threshold=threshold, device=device)

    # 3. 執行分析：對每個封包計算異常分數
    anomalies = []
    if packets:
        raw_bytes_list = [bytes(p) for p in packets]
        results_list = scorer.score_batch(raw_bytes_list)
        for i, (score, is_attack) in enumerate(results_list):
            if is_attack:
                anomalies.append({"packet_index": i, "score": score})

    results = {
        'total_packets': len(packets) if packets else 0,
        'anomalies': anomalies,
    }
    return results

def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print(f"  Grad-CAM 視覺化解釋分析（資料集：{args.dataset}）")
    print("=" * 65)

    if not check_torch():
        sys.exit(1)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  計算設備: {device}")

    # ── Step 1：載入資料集 ──────────────────────────────────
    print(f"\n[Step 1] 載入資料集：{args.dataset}")
    from dataset_loader import DatasetFactory

    load_kwargs = {}
    if args.dataset in ("cicids2017", "cicddos2019"):
        load_kwargs = {"max_normal": 50000, "max_attack": 30000}

    try:
        X_normal, X_attack, _ = DatasetFactory.load(
            args.dataset, data_dir=args.data_dir, **load_kwargs
        )
    except FileNotFoundError as e:
        print(f"  錯誤：{e}")
        sys.exit(1)

    print(f"  正常流量: {len(X_normal):,} 筆  攻擊流量: {len(X_attack):,} 筆")

    # 特徵名稱（長度對應影像高度 H）
    H = X_normal.shape[-2] if X_normal.ndim >= 3 else X_normal.shape[-1]
    feature_names = _DEFAULT_FEATURE_NAMES[:H]
    if len(feature_names) < H:
        feature_names += [f"F{i}" for i in range(len(feature_names), H)]

    # ── Step 2：載入模型 ────────────────────────────────────
    print(f"\n[Step 2] 載入模型：{args.model}")
    model = load_model(args.model, args.latent, device)

    # ── Step 3：執行 Grad-CAM ────────────────────────────────
    from cnn_gradcam import GradCAMReporter

    os.makedirs(args.output, exist_ok=True)
    variants = ["gradcam", "gradcam++", "scorecam"] if args.all_variants else [args.variant]

    results_dict = {}
    for variant in variants:
        var_dir = os.path.join(args.output, variant.replace("+", "p"))
        report, summary, image_paths = run_single_variant(
            model=model, X_normal=X_normal, X_attack=X_attack,
            args=args, variant=variant,
            output_dir=var_dir, device=device, feature_names=feature_names,
        )
        results_dict[variant] = (report, summary, image_paths)

        # ── Step 4：匯出報告 ────────────────────────────────
        if not args.viz_only and report is not None:
            reporter = GradCAMReporter(
                output_dir   = var_dir,
                model_name   = f"CNNAutoencoder (latent={args.latent})",
                dataset_name = args.dataset,
            )
            paths = reporter.export_all(
                report=report, summary=summary,
                image_paths=image_paths,
                variant=variant, target_layer=args.layer,
            )
            print(f"\n  報告已儲存：")
            print(f"    JSON : {paths['json']}")
            if not args.no_html:
                print(f"    HTML : {paths['html']}")

    # ── 多變體比較 ──────────────────────────────────────────
    if len(variants) > 1 and not args.viz_only:
        compare_variants(results_dict, args.output)

    # ── 完成 ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Grad-CAM 分析完成！")
    print(f"  輸出目錄：{os.path.abspath(args.output)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
