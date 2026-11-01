#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare CICIDS2017 training-CSV and converted-PCAP Flow through one inference path.

Run from project root, e.g.:
  python core/compare_cicids_csv_vs_pcap_inference.py \
    --train-csv data/cicids2017/Monday-WorkingHours.pcap_ISCX.csv \
    --pcap-flow-csv output/batch_eval/cicids_vae_monday_converter_fix/flows/cicids2017-PCAPs__Monday-WorkingHours.pcap.csv \
    --model output/threshold_study/unsup_cicids2017/best_vae_cicids2017.pt \
    --scaler output/threshold_study/unsup_cicids2017/cicids2017_preprocessor.pkl \
    --sample-rows 50000 \
    --output output/diagnostics/cicids_csv_vs_pcap_monday

This is a representation / pipeline sanity check, not a held-out performance estimate.
The training CSV may overlap the data used to train the model.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# This script lives in core/diagnostics/, while project modules such as
# model_registry.py and cnn_autoencoder.py live directly in core/.
# Add core/ to sys.path so running this script from the project root works.
CORE_DIR = Path(__file__).resolve().parents[1]
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import numpy as np
import pandas as pd


def clean_column(value: object) -> str:
    return str(value).replace("\ufeff", "").strip()


def read_header(path: Path) -> List[str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    return [clean_column(c) for c in header]


def count_labels(path: Path, label_col: str = "Label", chunksize: int = 100_000) -> Counter:
    """Read only the label column to decide whether PCAP labels can filter BENIGN rows."""
    header = read_header(path)
    if label_col not in header:
        return Counter({"<NO_LABEL_COLUMN>": -1})
    counts: Counter = Counter()
    for chunk in pd.read_csv(
        path,
        usecols=lambda c: clean_column(c) == label_col,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk.columns = [clean_column(c) for c in chunk.columns]
        labels = chunk[label_col].fillna("UNKNOWN").astype(str).str.strip().str.upper()
        counts.update(labels.value_counts(dropna=False).to_dict())
    return counts


def choose_pcap_label_filter(label_mode: str, label_counts: Counter) -> Tuple[Optional[set], str]:
    normal = {"BENIGN", "NORMAL"}
    known_normal_count = sum(int(label_counts.get(x, 0)) for x in normal)
    if label_mode == "all":
        return None, "使用 PCAP Flow 全部列；不以 Label 過濾。"
    if label_mode == "benign":
        if known_normal_count <= 0:
            raise ValueError(
                "指定 --pcap-label-filter benign，但 PCAP Flow CSV 沒有 BENIGN/NORMAL 標籤。"
                "請先修正 labels JSON／檔名匹配，或暫時改用 auto 進行未標記分布診斷。"
            )
        return normal, "只取 PCAP Flow 中 Label=BENIGN/NORMAL 的列。"
    # auto: when BENIGN labels exist use only them; if not, retain all rows as a clearly
    # labelled, unverified distribution comparison rather than silently pretending labels match.
    if known_normal_count > 0:
        return normal, "auto：找到 BENIGN/NORMAL 標籤，只取正常標籤列。"
    return None, (
        "警告：PCAP Flow CSV 沒有 BENIGN/NORMAL 標籤；本次將所有列作分布診斷。"
        "若該檔案確實是 Monday 全良性 PCAP，需另行用 --inspect 修正標籤匹配；"
        "此比較不能當作有標籤的良性評估。"
    )


def sample_csv_uniform(
    path: Path,
    feature_cols: List[str],
    sample_rows: int,
    seed: int,
    label_filter: Optional[set],
    chunksize: int = 50_000,
    exclude_single_packet_flows: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Uniform sample by assigning random priorities and retaining the lowest N.

    Reads only the model features plus Label. It checks feature header completeness before
    reading any data and never substitutes a missing feature with zero.
    """
    header = read_header(path)
    header_set = set(header)
    missing = [c for c in feature_cols if c not in header_set]
    if missing:
        raise ValueError(
            f"{path} 缺少 {len(missing)} 個 scaler 訓練特徵欄位，停止比較：{missing[:25]}"
        )
    has_label = "Label" in header_set
    if label_filter is not None and not has_label:
        raise ValueError(f"{path} 沒有 Label 欄，無法套用標籤篩選。")

    packet_count_cols = ["Total Fwd Packets", "Total Backward Packets"]
    if exclude_single_packet_flows:
        missing_counts = [c for c in packet_count_cols if c not in header_set]
        if missing_counts:
            raise ValueError(
                f"{path} 缺少 Flow 封包計數欄位 {missing_counts}，不能可靠排除單封包 Flow。"
            )

    selected_columns = set(feature_cols)
    if has_label:
        selected_columns.add("Label")
    if exclude_single_packet_flows:
        selected_columns.update(packet_count_cols)

    rng = np.random.default_rng(seed)
    reservoir = pd.DataFrame()
    n_seen = 0
    labels_seen: Counter = Counter()
    n_after_filter = 0
    n_bad_numeric = 0
    n_single_packet_filtered = 0

    for chunk in pd.read_csv(
        path,
        usecols=lambda c: clean_column(c) in selected_columns,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk.columns = [clean_column(c) for c in chunk.columns]
        if has_label:
            labels = chunk["Label"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
            labels_seen.update(labels.value_counts(dropna=False).to_dict())
            if label_filter is not None:
                chunk = chunk[labels.isin(label_filter)].copy()
        if chunk.empty:
            continue

        if exclude_single_packet_flows:
            fwd = pd.to_numeric(chunk["Total Fwd Packets"], errors="coerce").fillna(0.0)
            bwd = pd.to_numeric(chunk["Total Backward Packets"], errors="coerce").fillna(0.0)
            keep_multi = (fwd + bwd) > 1
            n_single_packet_filtered += int((~keep_multi).sum())
            chunk = chunk[keep_multi].copy()
        if chunk.empty:
            continue

        # Keep only exact model input fields plus label; numeric conversion matches the
        # production path's errors='coerce' followed by fillna(0).
        numeric = chunk[feature_cols].apply(pd.to_numeric, errors="coerce")
        n_bad_numeric += int(numeric.isna().sum().sum())
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if has_label:
            candidate = numeric.copy()
            candidate["Label"] = chunk["Label"].fillna("UNKNOWN").astype(str).str.strip()
        else:
            candidate = numeric
        candidate["__rand_priority"] = rng.random(len(candidate))

        n_seen += len(candidate)
        n_after_filter += len(candidate)
        if len(reservoir):
            candidate = pd.concat([reservoir, candidate], ignore_index=True)
        if len(candidate) > sample_rows:
            reservoir = candidate.nsmallest(sample_rows, "__rand_priority").copy()
        else:
            reservoir = candidate.copy()

    if reservoir.empty:
        raise ValueError(f"{path} 篩選後沒有可比較的資料列。")

    reservoir = reservoir.sort_values("__rand_priority").drop(columns=["__rand_priority"])
    reservoir = reservoir.reset_index(drop=True)
    meta: Dict[str, object] = {
        "path": str(path),
        "file_rows_after_filter": int(n_after_filter),
        "sample_rows": int(len(reservoir)),
        "label_counts_seen": {str(k): int(v) for k, v in labels_seen.items()},
        "feature_values_coerced_to_zero_count": int(n_bad_numeric),
        "has_label_column": bool(has_label),
        "label_filter": sorted(label_filter) if label_filter is not None else "ALL",
        "exclude_single_packet_flows": bool(exclude_single_packet_flows),
        "single_packet_rows_filtered": int(n_single_packet_filtered),
    }
    return reservoir, meta


def safe_quantiles(values: np.ndarray) -> Dict[str, float]:
    q = np.quantile(values, [0.50, 0.90, 0.95, 0.99, 0.999])
    return {
        "p50": float(q[0]),
        "p90": float(q[1]),
        "p95": float(q[2]),
        "p99": float(q[3]),
        "p99_9": float(q[4]),
    }


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(a, b, method="auto").statistic)
    except Exception:
        return float("nan")


def prepare_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def score_matrix(X_raw: np.ndarray, payload: dict, bundle, feature_cols: List[str], dataset_name: str) -> Tuple[dict, dict, np.ndarray, np.ndarray]:
    from cnn_autoencoder import features_to_image
    from model_registry import compute_anomaly_scores

    scaler = payload["scaler"]
    image_size = int(payload.get("image_size", 32))
    clip = payload.get("normalization") == "clip_0_1"
    X_preclip = scaler.transform(X_raw).astype(np.float32)
    preclip_lt0 = float(np.mean(X_preclip < 0.0))
    preclip_gt1 = float(np.mean(X_preclip > 1.0))
    X_input = np.clip(X_preclip, 0.0, 1.0) if clip else X_preclip

    image = features_to_image(X_input, image_size).squeeze(1)
    result = compute_anomaly_scores(bundle, image)
    scores = np.asarray(result["score"], dtype=np.float64).reshape(-1)
    anomalies = np.asarray(result["is_anomaly"], dtype=bool).reshape(-1)

    summary = {
        "dataset": dataset_name,
        "n_scored": int(len(scores)),
        "threshold": float(bundle.threshold),
        "anomalies": int(anomalies.sum()),
        "anomaly_rate": float(anomalies.mean()) if len(anomalies) else float("nan"),
        "score_quantiles": safe_quantiles(scores),
        "normalization": payload.get("normalization", "none"),
        "image_size": image_size,
        "scaled_fraction_lt_0_before_clip": preclip_lt0,
        "scaled_fraction_gt_1_before_clip": preclip_gt1,
        "clip_applied": bool(clip),
    }
    return summary, {"scores": scores, "anomalies": anomalies}, X_preclip, X_input


def build_feature_comparison(
    train_raw: np.ndarray,
    pcap_raw: np.ndarray,
    train_preclip: np.ndarray,
    pcap_preclip: np.ndarray,
    train_input: np.ndarray,
    pcap_input: np.ndarray,
    feature_cols: List[str],
) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(feature_cols):
        tr = train_raw[:, i]
        pc = pcap_raw[:, i]
        tr_s = train_preclip[:, i]
        pc_s = pcap_preclip[:, i]
        tr_c = train_input[:, i]
        pc_c = pcap_input[:, i]
        rows.append({
            "feature": name,
            "train_raw_p50": float(np.quantile(tr, 0.50)),
            "pcap_raw_p50": float(np.quantile(pc, 0.50)),
            "train_raw_p95": float(np.quantile(tr, 0.95)),
            "pcap_raw_p95": float(np.quantile(pc, 0.95)),
            "train_raw_p99": float(np.quantile(tr, 0.99)),
            "pcap_raw_p99": float(np.quantile(pc, 0.99)),
            "raw_ks": ks_statistic(tr, pc),
            "train_scaled_lt0_pct": float(np.mean(tr_s < 0) * 100),
            "pcap_scaled_lt0_pct": float(np.mean(pc_s < 0) * 100),
            "train_scaled_gt1_pct": float(np.mean(tr_s > 1) * 100),
            "pcap_scaled_gt1_pct": float(np.mean(pc_s > 1) * 100),
            "train_model_input_p50": float(np.quantile(tr_c, 0.50)),
            "pcap_model_input_p50": float(np.quantile(pc_c, 0.50)),
            "train_model_input_p95": float(np.quantile(tr_c, 0.95)),
            "pcap_model_input_p95": float(np.quantile(pc_c, 0.95)),
            "train_model_input_p99": float(np.quantile(tr_c, 0.99)),
            "pcap_model_input_p99": float(np.quantile(pc_c, 0.99)),
            "model_input_ks": ks_statistic(tr_c, pc_c),
            "train_model_input_zero_pct": float(np.mean(tr_c == 0) * 100),
            "pcap_model_input_zero_pct": float(np.mean(pc_c == 0) * 100),
            "train_model_input_one_pct": float(np.mean(tr_c == 1) * 100),
            "pcap_model_input_one_pct": float(np.mean(pc_c == 1) * 100),
        })
    result = pd.DataFrame(rows)
    return result.sort_values(["model_input_ks", "raw_ks"], ascending=False, na_position="last")


def main() -> int:
    parser = argparse.ArgumentParser(description="用同一模型與 scaler 對照訓練 CSV vs PCAP Flow 分布")
    parser.add_argument("--train-csv", required=True, help="CICIDS2017 訓練 CSV")
    parser.add_argument("--pcap-flow-csv", required=True, help="由 PCAP 轉換出的 Flow CSV")
    parser.add_argument("--model", required=True, help="VAE checkpoint .pt")
    parser.add_argument("--scaler", required=True, help="訓練期前處理器 .pkl")
    parser.add_argument("--sample-rows", type=int, default=50000, help="每份資料最大抽樣筆數，預設 50000")
    parser.add_argument("--chunksize", type=int, default=50000, help="CSV 讀取 chunk 大小")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pcap-label-filter", choices=["auto", "benign", "all"], default="auto",
                        help="auto=有 BENIGN 標籤就只取正常列；無正常標籤時用全部列並明確警告")
    parser.add_argument("--include-single-packet-flows", action="store_true",
                        help="診斷時也保留總封包數<=1 的 Flow；預設排除，以符合訓練 CSV 的匯出規則")
    parser.add_argument("--output", required=True, help="診斷輸出資料夾")
    args = parser.parse_args()

    train_path = Path(args.train_csv)
    pcap_path = Path(args.pcap_flow_csv)
    model_path = Path(args.model)
    scaler_path = Path(args.scaler)
    for p in [train_path, pcap_path, model_path, scaler_path]:
        if not p.exists():
            raise FileNotFoundError(f"找不到檔案：{p}")
    conversion_meta_path = Path(str(pcap_path) + ".params.json")
    conversion_meta = None
    if conversion_meta_path.is_file():
        try:
            conversion_meta = json.loads(conversion_meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            warnings.warn(f"Flow 轉換參數 sidecar 無法讀取：{exc}")
    else:
        warnings.warn(
            f"找不到 {conversion_meta_path}；比較結果仍可計算，但無法追溯此 Flow CSV 的轉換規則。"
        )
    if args.sample_rows < 100:
        raise ValueError("--sample-rows 至少應為 100，建議 50000")

    with scaler_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "scaler" not in payload or "feature_columns" not in payload:
        raise ValueError("Scaler .pkl 格式不符：需要 dict 且含 scaler / feature_columns")
    feature_cols = [clean_column(c) for c in payload["feature_columns"]]
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("feature_columns 含有清理名稱後的重複欄位，需先檢查 scaler payload")

    print(f"[模型特徵] {len(feature_cols)} 欄 | normalization={payload.get('normalization', 'none')} | image_size={payload.get('image_size', 32)}")
    pcap_label_counts = count_labels(pcap_path)
    pcap_filter, pcap_filter_note = choose_pcap_label_filter(args.pcap_label_filter, pcap_label_counts)
    if pcap_filter_note.startswith("警告"):
        print("\n" + pcap_filter_note)
    else:
        print("[PCAP 標籤篩選] " + pcap_filter_note)
    print("[PCAP Label 分布]", dict(pcap_label_counts))

    train_df, train_meta = sample_csv_uniform(
        train_path, feature_cols, args.sample_rows, args.seed,
        label_filter={"BENIGN", "NORMAL"}, chunksize=args.chunksize,
        exclude_single_packet_flows=not args.include_single_packet_flows,
    )
    pcap_df, pcap_meta = sample_csv_uniform(
        pcap_path, feature_cols, args.sample_rows, args.seed + 1,
        label_filter=pcap_filter, chunksize=args.chunksize,
        exclude_single_packet_flows=not args.include_single_packet_flows,
    )
    if "Label" in train_df:
        print("[訓練 CSV 樣本] Label 分布：", train_df["Label"].astype(str).str.upper().value_counts().to_dict())
    if "Label" in pcap_df:
        print("[PCAP Flow 樣本] Label 分布：", pcap_df["Label"].astype(str).str.upper().value_counts().to_dict())

    train_raw = prepare_matrix(train_df, feature_cols)
    pcap_raw = prepare_matrix(pcap_df, feature_cols)

    import torch
    from model_registry import load_anomaly_model

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定 --device cuda，但目前環境沒有可用 CUDA")
        device = torch.device(args.device)

    bundle = load_anomaly_model(str(model_path), device=device)
    rep = bundle.input_representation
    if rep and not str(rep).startswith("csv_features"):
        raise ValueError(f"checkpoint representation={rep!r} 不是 csv_features；請確認模型與 scaler 配對")
    print(f"[載入模型] type={bundle.model_type} representation={rep} threshold={bundle.threshold:.12g} device={device}")

    print("[計分] 依 batch_pcap_evaluate.py 相同的 scaler.transform → clip_0_1 → features_to_image → compute_anomaly_scores 流程計分")
    train_summary, train_scored, train_preclip, train_input = score_matrix(
        train_raw, payload, bundle, feature_cols, "TRAIN_CSV_BENIGN_SAMPLE"
    )
    pcap_summary, pcap_scored, pcap_preclip, pcap_input = score_matrix(
        pcap_raw, payload, bundle, feature_cols, "PCAP_FLOW_SAMPLE"
    )

    feature_cmp = build_feature_comparison(
        train_raw, pcap_raw, train_preclip, pcap_preclip,
        train_input, pcap_input, feature_cols,
    )

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    feature_path = outdir / "feature_comparison.csv"
    feature_cmp.to_csv(feature_path, index=False, encoding="utf-8-sig")

    score_rows = []
    for dataset, scored in [("TRAIN_CSV_BENIGN_SAMPLE", train_scored), ("PCAP_FLOW_SAMPLE", pcap_scored)]:
        score_rows.append(pd.DataFrame({
            "dataset": dataset,
            "score": scored["scores"].astype(float),
            "is_anomaly": scored["anomalies"].astype(bool),
        }))
    score_samples_path = outdir / "score_samples.csv"
    pd.concat(score_rows, ignore_index=True).to_csv(score_samples_path, index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "representation / inference-pipeline sanity check; not a held-out model performance estimate",
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "pcap_flow_csv": str(pcap_path),
        "pcap_conversion_params_path": str(conversion_meta_path) if conversion_meta else None,
        "pcap_conversion_params": conversion_meta,
        "model_type": bundle.model_type,
        "representation": rep,
        "threshold": float(bundle.threshold),
        "n_model_features": len(feature_cols),
        "normalization": payload.get("normalization", "none"),
        "image_size": int(payload.get("image_size", 32)),
        "sample_rows_requested_per_dataset": args.sample_rows,
        "train_sampling": train_meta,
        "pcap_sampling": pcap_meta,
        "pcap_label_counts_full_file": {str(k): int(v) for k, v in pcap_label_counts.items()},
        "pcap_filter_note": pcap_filter_note,
        "train_score_summary": train_summary,
        "pcap_score_summary": pcap_summary,
        "score_anomaly_rate_delta_pcap_minus_train": float(pcap_summary["anomaly_rate"] - train_summary["anomaly_rate"]),
        "score_p50_ratio_pcap_over_train": float(pcap_summary["score_quantiles"]["p50"] / max(train_summary["score_quantiles"]["p50"], 1e-15)),
        "feature_comparison_csv": str(feature_path),
        "score_samples_csv": str(score_samples_path),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    top = feature_cmp.head(15)
    lines = [
        "# CICIDS2017 訓練 CSV vs PCAP Flow 同管線對照",
        "",
        f"- 產生時間：{summary['generated_at']}",
        f"- 模型：`{model_path}`（{bundle.model_type}）",
        f"- Representation：`{rep}`",
        f"- Scaler：`{scaler_path}`",
        f"- 特徵欄位數：{len(feature_cols)}",
        f"- 原始模型閾值：`{bundle.threshold:.12g}`",
        f"- 抽樣：每組最多 {args.sample_rows:,} 筆；固定 seed={args.seed}",
        f"- PCAP Flow 轉換參數：`{json.dumps(conversion_meta, ensure_ascii=False) if conversion_meta else '缺少 .params.json，無法追溯'}`",
        "",
        "> 注意：訓練 CSV 樣本可能與模型訓練資料重疊。這份報告用於檢查資料表示法／推論流程，不是獨立測試集的泛化效能評估。",
        "",
        "## 一、同管線計分比較",
        "",
        "| 指標 | 訓練 CSV BENIGN 樣本 | PCAP Flow 樣本 |",
        "|---|---:|---:|",
        f"| 實際計分筆數 | {train_summary['n_scored']:,} | {pcap_summary['n_scored']:,} |",
        f"| 異常數 | {train_summary['anomalies']:,} | {pcap_summary['anomalies']:,} |",
        f"| 異常率 | {train_summary['anomaly_rate']:.4%} | {pcap_summary['anomaly_rate']:.4%} |",
        f"| Score p50 | {train_summary['score_quantiles']['p50']:.8g} | {pcap_summary['score_quantiles']['p50']:.8g} |",
        f"| Score p90 | {train_summary['score_quantiles']['p90']:.8g} | {pcap_summary['score_quantiles']['p90']:.8g} |",
        f"| Score p95 | {train_summary['score_quantiles']['p95']:.8g} | {pcap_summary['score_quantiles']['p95']:.8g} |",
        f"| Score p99 | {train_summary['score_quantiles']['p99']:.8g} | {pcap_summary['score_quantiles']['p99']:.8g} |",
        f"| Score p99.9 | {train_summary['score_quantiles']['p99_9']:.8g} | {pcap_summary['score_quantiles']['p99_9']:.8g} |",
        f"| scaler 後、clip 前 < 0 比例 | {train_summary['scaled_fraction_lt_0_before_clip']:.2%} | {pcap_summary['scaled_fraction_lt_0_before_clip']:.2%} |",
        f"| scaler 後、clip 前 > 1 比例 | {train_summary['scaled_fraction_gt_1_before_clip']:.2%} | {pcap_summary['scaled_fraction_gt_1_before_clip']:.2%} |",
        "",
        "## 二、差異最大的特徵（依 model_input_ks 排序）",
        "",
        "`model_input_ks` 是 scaler 與 clipping 後，該特徵在兩組樣本的 KS 統計量；越接近 1 表示經驗分布差異越大。KS 僅作特徵分布診斷，不能單獨證明計算公式錯誤。",
        "",
        "| Feature | Raw KS | Model input KS | Train p50 | PCAP p50 | Train p95 | PCAP p95 | Train clip=0 % | PCAP clip=0 % | Train clip=1 % | PCAP clip=1 % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['raw_ks']:.4f} | {r['model_input_ks']:.4f} | "
            f"{r['train_raw_p50']:.6g} | {r['pcap_raw_p50']:.6g} | "
            f"{r['train_raw_p95']:.6g} | {r['pcap_raw_p95']:.6g} | "
            f"{r['train_model_input_zero_pct']:.2f}% | {r['pcap_model_input_zero_pct']:.2f}% | "
            f"{r['train_model_input_one_pct']:.2f}% | {r['pcap_model_input_one_pct']:.2f}% |"
        )
    lines += [
        "",
        "## 三、標籤／抽樣注意事項",
        "",
        f"- 訓練 CSV 抽樣：{train_meta['sample_rows']:,} 筆，檔案中篩選後總列數約 {train_meta['file_rows_after_filter']:,}。",
        f"- PCAP Flow 抽樣：{pcap_meta['sample_rows']:,} 筆，篩選後總列數約 {pcap_meta['file_rows_after_filter']:,}。",
        f"- 訓練 CSV 排除的單封包 Flow：{train_meta['single_packet_rows_filtered']:,} 筆。",
        f"- PCAP Flow 排除的單封包 Flow：{pcap_meta['single_packet_rows_filtered']:,} 筆。",
        f"- PCAP Label 篩選說明：{pcap_filter_note}",
        f"- PCAP Label 全檔分布：`{json.dumps(summary['pcap_label_counts_full_file'], ensure_ascii=False)}`",
        "",
        "## 四、輸出檔案",
        "",
        f"- `feature_comparison.csv`：所有模型特徵的分位數、clip 比例與 KS 排序。",
        f"- `score_samples.csv`：兩組抽樣資料的 score 與異常判定。",
        f"- `summary.json`：機器可讀的完整摘要。",
        "",
        "## 五、判讀規則",
        "",
        "1. 若訓練 CSV BENIGN 樣本的異常率也很高，檢查 checkpoint、scaler、representation、特徵順序和 score pipeline 是否配對。",
        "2. 若訓練 CSV 分數相對低，但 PCAP Flow 分數明顯偏高，且多個 `model_input_ks` 很大，優先校正 PCAP→Flow 計算規則。",
        "3. 若原始特徵 KS 高，表示原始值已偏移；若原始 KS 尚可但 model-input KS 高，優先檢查 scaler 與 clip 飽和（0/1 比例）。",
        "4. 不要用調高閾值取代特徵對齊；完成表示法驗證後才評估閾值。",
    ]
    report_path = outdir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== 同管線計分結果 ===")
    for s in [train_summary, pcap_summary]:
        q = s["score_quantiles"]
        print(
            f"{s['dataset']}: n={s['n_scored']:,}, anomalies={s['anomalies']:,} "
            f"({s['anomaly_rate']:.2%}), score p50={q['p50']:.8g}, "
            f"p95={q['p95']:.8g}, p99={q['p99']:.8g}"
        )
    print(f"\n[完成] 報告：{report_path.resolve()}")
    print(f"[完成] 特徵差異：{feature_path.resolve()}")
    print(f"[完成] 分數樣本：{score_samples_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[錯誤] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
