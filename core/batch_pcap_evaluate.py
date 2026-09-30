#!/usr/bin/env python3
# ============================================================
# core/batch_pcap_evaluate.py
# 大量 PCAP 批次測試 + 統計分析（CICIDS2017 / Malcolm-PCAP / 任意 PCAP 目錄）
#
# 兩種推論模式（必須與模型「訓練時」的資料表示法一致）：
#   packet : 封包位元組影像（PacketVisualizer）。適用 train_pcap_native.py /
#            train_packet_native_vae.py / train_mta_malware.py 訓練出的模型。
#            輸出單位 = 封包。
#   flow   : CICFlowMeter 風格流量統計特徵（PcapFlowConverter + 訓練期 scaler）。
#            適用 run_training_cicids2017_vae_fixed.py / run_semi_supervised*.py /
#            run_training.py 等以 CSV 特徵訓練出的模型。輸出單位 = Flow。
#   auto   : 依 checkpoint 內 config["representation"] 判斷（見腳本說明）。
#
# 流程：
#   1. 遞迴掃描 --pcap-dir 下所有 .pcap/.pcapng/.cap
#   2. 逐檔轉換 + 計分 → <output>/scored/<tag>.csv.gz（每檔獨立，可中斷續跑）
#   3. 彙整統計：每檔異常率、分數分布、（有標籤時）Precision/Recall/F1/AUC/
#      各攻擊類型偵測率/固定 FPR 下的 Recall、協定/埠分布、時間軸、圖表、report.md
#
# 範例見檔案最下方或直接 --help。
# ============================================================
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

PCAP_EXTS = (".pcap", ".pcapng", ".cap")
SCORED_COLS = ["ts", "src", "dst", "sport", "dport", "proto", "label", "score", "is_anomaly"]


# ============================================================
# 共用小工具
# ============================================================
def _setup_stdout():
    """Windows cp950 終端無法輸出部分字元，強制 UTF-8。"""
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if s is not None and hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def wilson(k: int, n: int, z: float = 1.96):
    """比例的 Wilson 95% 信賴區間。"""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def md_table(headers: List[str], rows: List[list]) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 1000 else f"{v:,.1f}"
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        return str(v)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(x) for x in r) + " |")
    return "\n".join(out)


def find_pcaps(roots: List[str], include: Optional[List[str]], exclude: Optional[List[str]]):
    """回傳 [(root, path)]，遞迴掃描並依檔名子字串過濾。"""
    found = []
    for root in roots:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(PCAP_EXTS):
                    found.append((root, os.path.join(dp, fn)))
    inc = [s.lower() for s in (include or [])]
    exc = [s.lower() for s in (exclude or [])]

    def ok(p):
        name = os.path.basename(p).lower()
        if inc and not any(s in name for s in inc):
            return False
        if exc and any(s in name for s in exc):
            return False
        return True

    return sorted([(r, p) for r, p in found if ok(p)], key=lambda x: x[1])


def make_tag(root: str, path: str) -> str:
    rel = os.path.relpath(path, root)
    base = os.path.basename(os.path.normpath(root))
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{base}__{rel}")


# ============================================================
# 標籤對照表（IP + 時間窗）
# ============================================================
class LabelSchedule:
    """讀取 cicids2017_labels.json，為每個 PCAP 產生 labeler(ip_a, ip_b, t0, t1)。"""

    def __init__(self, spec: dict):
        self.tz = float(spec.get("timezone_offset_hours", 0))
        self.benign_files = [s.lower() for s in spec.get("benign_files", [])]
        tzinfo = timezone(timedelta(hours=self.tz))

        def to_epoch(date: str, hhmm: str) -> float:
            fmt = "%Y-%m-%d %H:%M:%S" if hhmm.count(":") == 2 else "%Y-%m-%d %H:%M"
            return datetime.strptime(f"{date} {hhmm}", fmt).replace(tzinfo=tzinfo).timestamp()

        self.entries = []
        for e in spec.get("entries", []):
            wins = e.get("windows") or [[e["start"], e["end"]]]
            self.entries.append({
                "file_contains": e.get("file_contains", "").lower(),
                "label": e["label"],
                "attackers": set(e.get("attacker_ips", [])),
                "victims": set(e.get("victim_ips", [])),
                # 官方時間只到分鐘 → 結束時間 +60 秒
                "windows": [(to_epoch(e["date"], s), to_epoch(e["date"], t) + 60.0) for s, t in wins],
            })

    def describe(self, pcap_name: str) -> str:
        ents = self._applicable(pcap_name)
        if ents is None:
            return "UNKNOWN（檔名未匹配任何 benign_files / entries）"
        if not ents:
            return "全部 BENIGN"
        return "、".join(sorted({e["label"] for e in ents}))

    def _applicable(self, pcap_name: str):
        name = pcap_name.lower()
        if any(b in name for b in self.benign_files):
            return []
        ents = [e for e in self.entries if e["file_contains"] and e["file_contains"] in name]
        return ents if ents else None

    def make_labeler(self, pcap_name: str):
        ents = self._applicable(pcap_name)

        def label(ip_a: str, ip_b: str, t0: float, t1: float) -> str:
            if ents is None:
                return "UNKNOWN"
            for e in ents:
                a, v = e["attackers"], e["victims"]
                hit = (ip_a in a and (not v or ip_b in v)) or (ip_b in a and (not v or ip_a in v))
                if not hit:
                    continue
                for ws, we in e["windows"]:
                    if t1 >= ws and t0 <= we:
                        return e["label"]
            return "BENIGN"

        return label


def load_schedule(path: Optional[str]) -> Optional[LabelSchedule]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return LabelSchedule(json.load(f))


# ============================================================
# Flow 模式 stage 1：PCAP → 流量特徵 CSV（可在子行程平行執行；只 import scapy）
# ============================================================
def _install_packet_limit(max_packets: int):
    """讓 PcapFlowConverter 只讀前 N 個封包（converter 本身沒有此參數）。"""
    import pcap_flow_converter as pfc
    from scapy.all import PcapReader as _Orig

    class _Limited:
        def __init__(self, path):
            self._r = _Orig(path)

        def __enter__(self):
            self._r.__enter__()
            return self

        def __exit__(self, *a):
            return self._r.__exit__(*a)

        def __iter__(self):
            for i, p in enumerate(self._r):
                if i >= max_packets:
                    break
                yield p

    pfc.PcapReader = _Limited


def convert_worker(pcap_path: str, csv_path: str, spec: Optional[dict], params: dict) -> dict:
    t0 = time.time()
    try:
        from pcap_flow_converter import PcapFlowConverter
        if params.get("max_packets"):
            _install_packet_limit(params["max_packets"])
        label_fn = None
        if spec is not None:
            lab = LabelSchedule(spec).make_labeler(os.path.basename(pcap_path))
            label_fn = lambda key, t0_, t1_: lab(key[0][0], key[1][0], t0_, t1_)  # noqa: E731
        tmp = csv_path + ".tmp"
        PcapFlowConverter(
            flow_timeout=params["flow_timeout"],
            idle_threshold=params["idle_threshold"],
            bulk_threshold=params["bulk_threshold"],
        ).convert(pcap_path, tmp, label_fn=label_fn, default_label="UNKNOWN",
                  progress_every=1_000_000)
        os.replace(tmp, csv_path)
        return {"ok": True, "seconds": round(time.time() - t0, 1), "csv": csv_path}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "seconds": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"}


# ============================================================
# 推論：Flow CSV → 分數
# ============================================================
def score_flow_csv(csv_path: str, out_gz: str, payload: dict, bundle, chunk: int = 50000) -> dict:
    import pandas as pd
    from cnn_autoencoder import features_to_image
    from model_registry import compute_anomaly_scores

    feature_cols = payload["feature_columns"]
    scaler = payload["scaler"]
    image_size = payload.get("image_size", 32)
    # run_training_cicids2017_vae_fixed.py 的訓練資料全部 clip 到 [0,1]，推論必須一致；
    # 舊版 fit_reference_scaler.py（MinMaxScaler）則刻意不 clip（見 score_converted_pcap.py）。
    clip = payload.get("normalization") == "clip_0_1"

    tmp = out_gz + ".tmp"
    first, n_total, warned = True, 0, False
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
        for df in pd.read_csv(csv_path, chunksize=chunk, low_memory=False):
            missing = [c for c in feature_cols if c not in df.columns]
            if missing and not warned:
                print(f"    [警告] 缺少 {len(missing)} 個訓練期欄位，以 0 補齊：{missing[:5]}")
                warned = True
            for c in missing:
                df[c] = 0.0
            X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            Xs = scaler.transform(X).astype(np.float32)
            if clip:
                Xs = np.clip(Xs, 0.0, 1.0)
            img = features_to_image(Xs, image_size).squeeze(1)
            res = compute_anomaly_scores(bundle, img)
            out = pd.DataFrame({
                "ts": df["Timestamp"].astype(float), "src": df["Src IP"], "dst": df["Dst IP"],
                "sport": df["Src Port"].astype(int), "dport": df["Dst Port"].astype(int),
                "proto": df["Protocol"].astype(int), "label": df["Label"],
                "score": res["score"].astype(np.float32), "is_anomaly": res["is_anomaly"].astype(bool),
            })
            if "status" in res:
                out["status"] = res["status"]
            out.to_csv(fh, header=first, index=False)
            first = False
            n_total += len(out)
    os.replace(tmp, out_gz)
    return {"n_scored": n_total}


# ============================================================
# 推論：PCAP → 封包位元組影像 → 分數（串流，不會把整個 PCAP 載入記憶體）
# ============================================================
def score_packet_file(pcap_path: str, out_gz: str, args, bundle, labeler) -> dict:
    import pandas as pd
    from scapy.all import PcapReader
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
    from packet_visualizer import PacketVisualizer
    from model_registry import compute_anomaly_scores

    # 設定必須與訓練 / analyzer/tasks.py 推論時完全一致
    vis = PacketVisualizer("medium", apply_mask=True, skip_ethernet=True)
    cols = ["ts", "src", "dst", "sport", "dport", "proto", "label"]
    rows: list = []
    imgs: list = []
    st = {"first": True, "n_scored": 0}
    n_read = n_fail = 0
    t0 = time.time()
    tmp = out_gz + ".tmp"
    linktype = None

    def flush(fh):
        if not imgs:
            return
        res = compute_anomaly_scores(bundle, np.stack(imgs).astype(np.float32))
        df = pd.DataFrame(rows, columns=cols)
        df["score"] = res["score"].astype(np.float32)
        df["is_anomaly"] = res["is_anomaly"].astype(bool)
        if "status" in res:
            df["status"] = res["status"]
        df.to_csv(fh, header=st["first"], index=False)
        st["first"] = False
        st["n_scored"] += len(df)
        rows.clear()
        imgs.clear()

    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
        with PcapReader(pcap_path) as reader:
            linktype = getattr(reader, "linktype", None)
            for pkt in reader:
                n_read += 1
                if args.max_packets_per_file and n_read > args.max_packets_per_file:
                    break
                if args.sample_every > 1 and n_read % args.sample_every:
                    continue
                try:
                    img = vis.bytes_to_image(bytes(pkt))
                except Exception:
                    n_fail += 1
                    continue
                sport = dport = 0
                if pkt.haslayer(IP):
                    ip = pkt[IP]
                    src, dst, proto = ip.src, ip.dst, int(ip.proto)
                elif pkt.haslayer(IPv6):
                    ip = pkt[IPv6]
                    src, dst, proto = ip.src, ip.dst, int(ip.nh)
                else:
                    src = dst = ""
                    proto = -1
                if pkt.haslayer(TCP):
                    sport, dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
                elif pkt.haslayer(UDP):
                    sport, dport = int(pkt[UDP].sport), int(pkt[UDP].dport)
                ts = float(pkt.time)
                lab = labeler(src, dst, ts, ts) if (labeler and src) else ("UNKNOWN" if labeler is None else "BENIGN")
                imgs.append(img)
                rows.append((ts, src, dst, sport, dport, proto, lab))
                if len(imgs) >= args.chunk:
                    flush(fh)
                if n_read % 200000 == 0:
                    rate = n_read / max(time.time() - t0, 1e-6)
                    print(f"    已讀 {n_read:,} 封包（{rate:,.0f} pkt/s）")
            flush(fh)
    os.replace(tmp, out_gz)
    return {"n_read": n_read, "n_scored": st["n_scored"], "n_fail": n_fail, "linktype": linktype}


# ============================================================
# 統計分析
# ============================================================
def load_scored(path: str):
    import pandas as pd
    try:
        df = pd.read_csv(path, dtype={"src": "category", "dst": "category", "label": "category",
                                      "sport": "int32", "dport": "int32", "proto": "int16",
                                      "score": "float32"})
    except pd.errors.EmptyDataError:
        return None
    if "is_anomaly" in df.columns and df["is_anomaly"].dtype != bool:
        df["is_anomaly"] = df["is_anomaly"].astype(str).str.lower().eq("true")
    return df if len(df) else None


def _ks(a: np.ndarray, b: np.ndarray, cap: int = 200000, seed: int = 0):
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return float("nan")
    rng = np.random.default_rng(seed)
    if len(a) > cap:
        a = rng.choice(a, cap, replace=False)
    if len(b) > cap:
        b = rng.choice(b, cap, replace=False)
    return float(ks_2samp(a, b).statistic)


def analyze(args, out_dir: str, scored: List[tuple], run_meta: dict) -> dict:
    """scored = [(tag, path)]；產生 CSV / JSON / 圖表 / report.md。"""
    import pandas as pd

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 參考分布（例如 Monday 良性流量），用來算各檔案的分布位移（KS）
    ref_scores = None
    if args.reference_csv and os.path.exists(args.reference_csv):
        d = load_scored(args.reference_csv)
        ref_scores = d["score"].to_numpy() if d is not None else None
    elif args.reference_substr:
        for tag, path in scored:
            if args.reference_substr.lower() in tag.lower():
                d = load_scored(path)
                ref_scores = d["score"].to_numpy() if d is not None else None
                break

    vocab: Dict[str, int] = {}
    S, L, P = [], [], []
    per_file, label_by_file, pp_parts, src_parts = [], [], [], []
    timelines: Dict[str, "pd.DataFrame"] = {}

    for tag, path in scored:
        df = load_scored(path)
        if df is None:
            print(f"  [略過] {tag}: 無資料")
            continue
        score = df["score"].to_numpy()
        pred = df["is_anomaly"].to_numpy() if args.threshold is None else (score > args.threshold)
        lab = df["label"].astype(str)
        attack = ~lab.isin(["BENIGN", "UNKNOWN"]).to_numpy()
        n, n_pred = len(df), int(pred.sum())
        lo, hi = wilson(n_pred, n)
        q = np.quantile(score, [0.5, 0.9, 0.95, 0.99])
        row = {
            "file": tag, "n": n, "n_anomaly": n_pred, "anomaly_rate": n_pred / n,
            "rate_ci_lo": lo, "rate_ci_hi": hi,
            "score_p50": q[0], "score_p90": q[1], "score_p95": q[2], "score_p99": q[3],
            "score_max": float(score.max()), "n_attack_label": int(attack.sum()),
            "n_benign_label": int((lab == "BENIGN").sum()),
            "ks_vs_ref": _ks(score, ref_scores) if ref_scores is not None else float("nan"),
        }
        per_file.append(row)

        g = pd.DataFrame({"lab": lab.to_numpy(), "p": pred}).groupby("lab").agg(n=("p", "size"), a=("p", "sum"))
        for lb, r in g.iterrows():
            label_by_file.append({"file": tag, "label": lb, "n": int(r["n"]), "n_anomaly": int(r["a"]),
                                  "rate": float(r["a"] / r["n"])})

        for u in np.unique(lab.to_numpy()):
            vocab.setdefault(u, len(vocab))
        S.append(score)
        L.append(pd.Series(lab.to_numpy()).map(vocab).to_numpy(np.int16))
        P.append(pred)

        # 時間軸（每分鐘）
        m = (df["ts"].to_numpy() // 60).astype(np.int64)
        timelines[tag] = pd.DataFrame({"m": m, "p": pred, "k": attack}).groupby("m").agg(
            n=("p", "size"), a=("p", "sum"), k=("k", "sum")).reset_index()

        # 協定 / 服務埠 / 來源 IP 分解
        svc = np.minimum(df["sport"].to_numpy(), df["dport"].to_numpy()).astype(np.int64)
        svc = np.where(svc > 10000, -1, svc)
        pp_parts.append(pd.DataFrame({"proto": df["proto"].to_numpy(), "svc": svc, "p": pred})
                        .groupby(["proto", "svc"]).agg(n=("p", "size"), a=("p", "sum")).reset_index())
        sg = pd.DataFrame({"src": df["src"].astype(str).to_numpy(), "p": pred}).groupby("src").agg(
            n=("p", "size"), a=("p", "sum")).reset_index().sort_values("a", ascending=False).head(300)
        src_parts.append(sg)
        del df
        print(f"  已分析 {tag}: n={n:,}  異常率={n_pred / n:.2%}")

    if not S:
        print("沒有可分析的資料。")
        return {}

    S = np.concatenate(S)
    L = np.concatenate(L)
    P = np.concatenate(P)
    inv = {v: k for k, v in vocab.items()}

    pf = pd.DataFrame(per_file).sort_values("anomaly_rate", ascending=False)
    pf.to_csv(os.path.join(out_dir, "per_file_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(label_by_file).to_csv(os.path.join(out_dir, "label_by_file.csv"), index=False, encoding="utf-8-sig")

    pp = pd.concat(pp_parts).groupby(["proto", "svc"]).sum().reset_index()
    pp["rate"] = pp["a"] / pp["n"]
    pp.sort_values("a", ascending=False).head(200).rename(columns={"a": "n_anomaly", "svc": "service_port(-1=>10000)"}) \
        .to_csv(os.path.join(out_dir, "breakdown_proto_port.csv"), index=False, encoding="utf-8-sig")
    sp = pd.concat(src_parts).groupby("src").sum().reset_index()
    sp["rate"] = sp["a"] / sp["n"]
    sp.sort_values("a", ascending=False).head(100).rename(columns={"a": "n_anomaly"}) \
        .to_csv(os.path.join(out_dir, "top_anomalous_sources.csv"), index=False, encoding="utf-8-sig")

    # ---------- 有標籤時的指標 ----------
    metrics: dict = {"overall": {
        "n": int(len(S)), "n_anomaly": int(P.sum()), "anomaly_rate": float(P.mean()),
        "score_quantiles": {k: float(v) for k, v in zip(["p50", "p90", "p95", "p99", "p99.9"],
                                                       np.quantile(S, [0.5, 0.9, 0.95, 0.99, 0.999]))}}}
    b_code, u_code = vocab.get("BENIGN"), vocab.get("UNKNOWN")
    valid = np.ones(len(L), bool) if u_code is None else (L != u_code)
    is_b = valid & (L == b_code) if b_code is not None else np.zeros(len(L), bool)
    is_a = valid & (~is_b)
    labeled = bool(is_b.any() and is_a.any())
    per_attack_rows, op_rows = [], []

    if labeled:
        from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
        vmask = is_a | is_b
        y, s, p = is_a[vmask].astype(int), S[vmask], P[vmask]
        tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
        fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        fpr_, tnr = fp / max(fp + tn, 1), tn / max(fp + tn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        mcc_den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics["at_model_threshold"] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": prec, "recall": rec, "f1": f1,
            "fpr": fpr_, "balanced_accuracy": (rec + tnr) / 2,
            "mcc": (tp * tn - fp * fn) / mcc_den if mcc_den > 0 else 0.0,
            "recall_ci95": wilson(tp, tp + fn), "fpr_ci95": wilson(fp, fp + tn),
            "roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
            "attack_prevalence": float(y.mean()),
        }
        # 固定 FPR 下的 Recall：門檻只由「良性」分數決定（不偷看攻擊），較誠實的比較方式
        sb, sa = s[y == 0], s[y == 1]
        for target in (0.001, 0.005, 0.01, 0.05):
            thr = float(np.quantile(sb, 1 - target))
            tpn, fpn = int((sa > thr).sum()), int((sb > thr).sum())
            pr = tpn / max(tpn + fpn, 1)
            rc = tpn / max(len(sa), 1)
            op_rows.append({"target_fpr": target, "threshold": thr, "actual_fpr": fpn / max(len(sb), 1),
                            "recall": rc, "precision": pr, "f1": 2 * pr * rc / max(pr + rc, 1e-12)})
        # 最佳 F1（偷看了標籤，屬樂觀值，只供參考）
        pp_, rr_, th_ = precision_recall_curve(y, s)
        f1s = 2 * pp_[:-1] * rr_[:-1] / np.maximum(pp_[:-1] + rr_[:-1], 1e-12)
        if len(f1s):
            bi = int(np.argmax(f1s))
            metrics["best_f1_oracle"] = {"threshold": float(th_[bi]), "f1": float(f1s[bi]),
                                          "precision": float(pp_[bi]), "recall": float(rr_[bi])}
        thr_fpr1 = float(np.quantile(sb, 0.99))
        for code, name in sorted(inv.items(), key=lambda kv: kv[1]):
            if name in ("BENIGN", "UNKNOWN"):
                continue
            m = L == code
            n_, d_ = int(m.sum()), int(P[m].sum())
            lo, hi = wilson(d_, n_)
            per_attack_rows.append({
                "attack_type": name, "n": n_, "detected": d_, "recall": d_ / max(n_, 1),
                "recall_ci_lo": lo, "recall_ci_hi": hi,
                "recall_at_fpr1pct": float((S[m] > thr_fpr1).mean()),
                "score_mean": float(S[m].mean()), "score_median": float(np.median(S[m]))})
        pd.DataFrame(per_attack_rows).sort_values("n", ascending=False).to_csv(
            os.path.join(out_dir, "per_attack_type.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(op_rows).to_csv(os.path.join(out_dir, "operating_points.csv"), index=False, encoding="utf-8-sig")
        metrics["operating_points"] = op_rows
        metrics["per_attack_type"] = per_attack_rows

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=_json_default)

    _make_plots(plots_dir, S, is_a, is_b, labeled, P, pf, per_attack_rows, timelines, metrics, args, run_meta)
    _write_report(out_dir, args, run_meta, metrics, pf, per_attack_rows, op_rows, pp, sp, labeled)
    return metrics


def _make_plots(plots_dir, S, is_a, is_b, labeled, P, pf, per_attack_rows, timelines, metrics, args, run_meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    rng = np.random.default_rng(0)
    thr = args.threshold if args.threshold is not None else run_meta.get("model_threshold")

    # 1) 分數分布
    idx = rng.choice(len(S), min(len(S), 500000), replace=False)
    ls = np.log10(np.maximum(S[idx], 1e-9))
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(np.percentile(ls, 0.1), np.percentile(ls, 99.9), 100)
    if labeled:
        ax.hist(ls[is_b[idx]], bins=bins, alpha=.6, density=True, color="#2ca02c", label="benign")
        ax.hist(ls[is_a[idx]], bins=bins, alpha=.6, density=True, color="#d62728", label="attack")
    else:
        ax.hist(ls, bins=bins, alpha=.8, density=True, color="#1f77b4", label="all (unlabeled)")
    if thr:
        ax.axvline(math.log10(max(thr, 1e-9)), color="k", ls="--", label=f"threshold={thr:.4g}")
    ax.set_xlabel("log10(anomaly score)")
    ax.set_ylabel("density")
    ax.set_title("Anomaly score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "score_distribution.png"), dpi=130)
    plt.close(fig)

    # 2) ROC / PR
    if labeled:
        from sklearn.metrics import roc_curve, precision_recall_curve
        vm = is_a | is_b
        y, s = is_a[vm].astype(int), S[vm]
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
        m_ = metrics["at_model_threshold"]
        a1.plot(fpr, tpr, label=f"AUC={m_['roc_auc']:.3f}")
        a1.plot([0, 1], [0, 1], "k--", lw=.8)
        a1.scatter([m_["fpr"]], [m_["recall"]], c="r", zorder=5, label="model threshold")
        a1.set_xlabel("FPR")
        a1.set_ylabel("TPR")
        a1.set_title("ROC")
        a1.legend()
        a2.plot(rc, pr, label=f"PR-AUC={m_['pr_auc']:.3f}")
        a2.axhline(m_["attack_prevalence"], color="gray", ls=":", label="prevalence (random)")
        a2.scatter([m_["recall"]], [m_["precision"]], c="r", zorder=5, label="model threshold")
        a2.set_xlabel("Recall")
        a2.set_ylabel("Precision")
        a2.set_title("Precision-Recall")
        a2.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "roc_pr.png"), dpi=130)
        plt.close(fig)

        # 3) 各攻擊類型偵測率
        if per_attack_rows:
            rows = sorted(per_attack_rows, key=lambda r: r["recall"])
            fig, ax = plt.subplots(figsize=(9, max(3, .45 * len(rows) + 1)))
            names = [f"{r['attack_type']} (n={r['n']:,})" for r in rows]
            xs = [r["recall"] for r in rows]
            err = [[r["recall"] - r["recall_ci_lo"] for r in rows], [r["recall_ci_hi"] - r["recall"] for r in rows]]
            ax.barh(names, xs, xerr=err, color="#d62728", alpha=.8, label="at model threshold")
            ax.scatter([r["recall_at_fpr1pct"] for r in rows], names, c="k", marker="|", s=200, label="recall @ FPR=1%")
            ax.set_xlim(0, 1.02)
            ax.set_xlabel("detection rate (recall)")
            ax.set_title("Detection rate per attack type")
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "per_attack_recall.png"), dpi=130)
            plt.close(fig)

    # 4) 各檔案異常率
    d = pf[pf["n"] >= 30].head(40).iloc[::-1]
    if len(d):
        fig, ax = plt.subplots(figsize=(10, max(3, .3 * len(d) + 1)))
        err = [(d["anomaly_rate"] - d["rate_ci_lo"]).to_numpy(), (d["rate_ci_hi"] - d["anomaly_rate"]).to_numpy()]
        ax.barh([f[-60:] for f in d["file"]], d["anomaly_rate"], xerr=err, color="#1f77b4")
        ax.set_xlabel("anomaly rate (95% Wilson CI)")
        ax.set_title("Anomaly rate per PCAP file (top 40, n>=30)")
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "per_file_anomaly_rate.png"), dpi=130)
        plt.close(fig)

    # 5) 時間軸（最大的 8 個檔案；有標籤時陰影 = 標籤為攻擊的分鐘）
    tz = timezone(timedelta(hours=args.plot_tz))
    big = sorted(timelines.items(), key=lambda kv: -kv[1]["n"].sum())[:8]
    for tag, t in big:
        if len(t) < 5:
            continue
        x = [datetime.fromtimestamp(int(m) * 60, tz=tz) for m in t["m"]]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
        a1.plot(x, t["a"] / t["n"], lw=1, color="#1f77b4", label="predicted anomaly fraction")
        if (t["k"] > 0).any():
            a1.fill_between(x, 0, 1, where=(t["k"] > 0).to_numpy(), color="#d62728", alpha=.15,
                            transform=a1.get_xaxis_transform(), label="label = attack")
        a1.set_ylabel("anomaly fraction / min")
        a1.set_ylim(0, 1.02)
        a1.legend(loc="upper right")
        a1.set_title(f"Timeline: {tag[-70:]}  (UTC{args.plot_tz:+g})")
        a2.semilogy(x, np.maximum(t["n"], 1), lw=1, color="gray")
        a2.set_ylabel("items / min")
        a2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=tz))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"timeline_{tag[-60:]}.png"), dpi=120)
        plt.close(fig)


def _write_report(out_dir, args, run_meta, metrics, pf, per_attack_rows, op_rows, pp, sp, labeled):
    o = metrics["overall"]
    L = ["# 批次 PCAP 測試報告", "", f"- 產生時間：{datetime.now():%Y-%m-%d %H:%M:%S}",
         f"- 模型：`{run_meta.get('model', '?')}`（{run_meta.get('model_type', '?')}）",
         f"- 推論模式：**{run_meta.get('mode', '?')}**（單位 = {'封包' if run_meta.get('mode') == 'packet' else 'Flow'}）",
         f"- 判定閾值：{args.threshold if args.threshold is not None else run_meta.get('model_threshold')}"
         f"{'（--threshold 覆寫，忽略 hybrid 分類器）' if args.threshold is not None else '（模型 checkpoint 內建）'}",
         f"- 檔案數：{len(pf)}　總筆數：{o['n']:,}　異常數：{o['n_anomaly']:,}　整體異常率：{o['anomaly_rate']:.2%}", ""]
    q = o["score_quantiles"]
    L += ["## 分數分位數", md_table(list(q.keys()), [list(q.values())]), ""]

    if labeled:
        m = metrics["at_model_threshold"]
        L += ["## 有標籤評估（模型內建閾值）",
              md_table(["Precision", "Recall", "F1", "FPR", "BalancedAcc", "MCC", "ROC-AUC", "PR-AUC", "攻擊佔比"],
                       [[m["precision"], m["recall"], m["f1"], m["fpr"], m["balanced_accuracy"], m["mcc"],
                         m["roc_auc"], m["pr_auc"], m["attack_prevalence"]]]),
              "", f"混淆矩陣：TP={m['TP']:,}　FP={m['FP']:,}　FN={m['FN']:,}　TN={m['TN']:,}",
              f"Recall 95% CI = [{m['recall_ci95'][0]:.4f}, {m['recall_ci95'][1]:.4f}]；"
              f"FPR 95% CI = [{m['fpr_ci95'][0]:.4f}, {m['fpr_ci95'][1]:.4f}]（Wilson，假設樣本獨立；"
              f"相鄰封包/Flow 高度相關，實際不確定性更大）", "",
              "## 固定 FPR 下的 Recall（閾值只由良性分數決定）",
              md_table(["目標 FPR", "閾值", "實際 FPR", "Recall", "Precision", "F1"],
                       [[r["target_fpr"], r["threshold"], r["actual_fpr"], r["recall"], r["precision"], r["f1"]] for r in op_rows]),
              ""]
        if "best_f1_oracle" in metrics:
            b = metrics["best_f1_oracle"]
            L += [f"> 事後最佳 F1（偷看標籤，僅供參考，屬樂觀值）：F1={b['f1']:.4f}，閾值={b['threshold']:.6g}，"
                  f"Precision={b['precision']:.4f}，Recall={b['recall']:.4f}", ""]
        L += ["## 各攻擊類型偵測率",
              md_table(["攻擊類型", "筆數", "偵測數", "Recall", "95% CI", "Recall@FPR1%", "分數中位數"],
                       [[r["attack_type"], r["n"], r["detected"], r["recall"],
                         f"[{r['recall_ci_lo']:.3f}, {r['recall_ci_hi']:.3f}]", r["recall_at_fpr1pct"], r["score_median"]]
                        for r in sorted(per_attack_rows, key=lambda r: -r["n"])]), ""]
    else:
        L += ["## 標籤", "本次沒有可用的良性/攻擊標籤（未提供 --labels 或檔名未匹配），只做**未標記健康檢查**："
              "異常率不是準確率，不可解讀為 Precision/Recall。", ""]

    show = pf.head(25)
    has_ks = show["ks_vs_ref"].notna().any()
    hdr = ["檔案", "筆數", "異常率", "95% CI", "p50", "p95", "p99"] + (["KS vs 參考"] if has_ks else [])
    rows = []
    for _, r in show.iterrows():
        row = [r["file"][-55:], int(r["n"]), float(r["anomaly_rate"]), f"[{r['rate_ci_lo']:.3f}, {r['rate_ci_hi']:.3f}]",
               float(r["score_p50"]), float(r["score_p95"]), float(r["score_p99"])]
        if has_ks:
            row.append(float(r["ks_vs_ref"]))
        rows.append(row)
    L += ["## 各檔案異常率（前 25，依異常率排序）", md_table(hdr, rows), "",
          "> n<30 的檔案異常率不穩定；KS 越大代表該檔案分數分布與參考（例如 Monday 良性）差越多。", ""]

    top = pp.sort_values("a", ascending=False).head(15)
    L += ["## 被判為異常最多的協定 / 服務埠",
          md_table(["proto", "service_port(-1=>10000)", "筆數", "異常數", "異常率"],
                   [[int(r["proto"]), int(r["svc"]), int(r["n"]), int(r["a"]), float(r["rate"])] for _, r in top.iterrows()]),
          "", "（proto: 6=TCP 17=UDP 1=ICMP -1=非 IP；service_port = min(src,dst port)）", ""]
    ts_ = sp.sort_values("a", ascending=False).head(15)
    L += ["## 貢獻最多異常的來源 IP（近似，每檔僅保留前 300 名）",
          md_table(["src", "筆數", "異常數", "異常率"],
                   [[r["src"], int(r["n"]), int(r["a"]), float(r["rate"])] for _, r in ts_.iterrows()]), ""]

    L += ["## 解讀注意事項",
          "- 有標籤時的 Recall 受 IP+時間窗標記粗糙度限制（攻擊時段內含無 payload 的 Attempted 流量）。",
          "- 固定 FPR 的表格比「模型閾值」的表格更適合比較不同模型/資料集。",
          "- 圖表在 `plots/`：score_distribution、roc_pr、per_attack_recall、per_file_anomaly_rate、timeline_*。",
          "- 若時間軸圖中異常率高峰與陰影（攻擊時段）整體平移固定小時數，代表時區設定有誤，請調整 labels JSON 的 timezone_offset_hours。"]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ============================================================
# 主流程
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="大量 PCAP 批次測試與統計分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
範例（於專案根目錄執行；Windows 路徑用引號）：
  # 0) 先看看有哪些檔案、時間、標記方式（不推論）
  python core/batch_pcap_evaluate.py --pcap-dir "PCAPs\cicids2017-PCAPs" --labels core/cicids2017_labels.json --inspect

  # 1) CICIDS2017 + CSV 訓練的模型（flow 模式），先小量試跑
  python core/batch_pcap_evaluate.py --model media/model/best_vae_cicids2017.pt --mode flow ^
      --scaler output/model_cicids2017/cicids2017_preprocessor.pkl ^
      --pcap-dir "PCAPs\cicids2017-PCAPs" --labels core/cicids2017_labels.json ^
      --include monday --max-packets-per-file 500000 --output output/batch_eval/cicids_smoke

  # 2) 封包位元組模型（packet 模式）跑 Malcolm-PCAP（無標籤 → 健康檢查）
  python core/batch_pcap_evaluate.py --model media/model/semi_cicids2017.pt --mode packet ^
      --pcap-dir "PCAPs\Malcolm-PCAP-main" --max-packets-per-file 200000 ^
      --output output/batch_eval/malcolm

  # 3) 不重跑推論，只換閾值重新統計
  python core/batch_pcap_evaluate.py --analyze-only --output output/batch_eval/cicids --threshold 0.0123
""")
    p.add_argument("--pcap-dir", nargs="+", help="PCAP 根目錄（可多個，遞迴掃描）")
    p.add_argument("--include", nargs="*", help="只處理檔名含這些子字串的檔案")
    p.add_argument("--exclude", nargs="*", help="排除檔名含這些子字串的檔案")
    p.add_argument("--model", help="已部署/已訓練的 .pt checkpoint")
    p.add_argument("--mode", choices=["auto", "packet", "flow"], default="auto")
    p.add_argument("--scaler", help="flow 模式：訓練期前處理器 .pkl（scaler + feature_columns）")
    p.add_argument("--labels", help="標籤對照表 JSON（見 core/cicids2017_labels.json）；省略 = 全部 UNKNOWN")
    p.add_argument("--output", default="output/batch_eval")
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--threshold", type=float, default=None, help="覆寫判定閾值（重新統計用）")
    # 規模控制
    p.add_argument("--max-packets-per-file", type=int, default=0, help="每檔最多讀取封包數（0=全部）")
    p.add_argument("--sample-every", type=int, default=1, help="packet 模式：每 N 個封包取 1 個")
    p.add_argument("--chunk", type=int, default=4096, help="packet 模式：每批推論封包數")
    p.add_argument("--workers", type=int, default=1, help="flow 模式：PCAP→特徵 平行行程數（每行程各佔一份記憶體）")
    p.add_argument("--flow-timeout", type=float, default=120.0)
    p.add_argument("--idle-threshold", type=float, default=1.0)
    p.add_argument("--bulk-threshold", type=float, default=1.0)
    # 快取
    p.add_argument("--force", action="store_true", help="忽略 scored/ 快取，全部重算")
    p.add_argument("--force-convert", action="store_true", help="flow 模式：重新轉換 PCAP→特徵 CSV")
    # 分析
    p.add_argument("--reference-substr", default="monday", help="以檔名含此字串的檔案作為分布位移(KS)的參考")
    p.add_argument("--reference-csv", help="改用指定的 scored csv.gz 作為參考分布（例如上次 Monday 的結果）")
    p.add_argument("--plot-tz", type=float, default=None, help="時間軸圖的時區偏移（預設沿用 labels 的設定，否則 0）")
    p.add_argument("--inspect", action="store_true", help="只列出檔案資訊與標記方式，不推論")
    p.add_argument("--analyze-only", action="store_true", help="跳過轉換/推論，只重新統計 <output>/scored/")
    return p.parse_args()


def inspect_files(files, schedule):
    from scapy.all import PcapReader
    print(f"共 {len(files)} 個檔案")
    tz = timezone(timedelta(hours=schedule.tz if schedule else 0))
    for _, path in files:
        size = os.path.getsize(path) / 1e9
        info = ""
        try:
            with PcapReader(path) as r:
                lt = getattr(r, "linktype", None)
                first = next(iter(r), None)
                if first is not None:
                    t = datetime.fromtimestamp(float(first.time), tz=timezone.utc)
                    info = f"linktype={lt}  首包 UTC {t:%Y-%m-%d %H:%M:%S} / 本地 {t.astimezone(tz):%H:%M:%S}"
        except Exception as e:  # noqa: BLE001
            info = f"[讀取失敗] {type(e).__name__}: {e}"
        lab = schedule.describe(os.path.basename(path)) if schedule else "UNKNOWN（未提供 --labels）"
        print(f"- {os.path.basename(path)}  {size:.2f} GB  {info}\n    標記: {lab}")
    print("\n[提示] linktype 應為 1（Ethernet）；packet 模式會假設封包前 14 bytes 是 Ethernet 表頭。")


def main():
    _setup_stdout()
    args = parse_args()
    out_dir = args.output
    scored_dir = os.path.join(out_dir, "scored")
    flows_dir = os.path.join(out_dir, "flows")
    os.makedirs(scored_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "run_meta.json")

    schedule = None
    spec = None
    if args.labels:
        with open(args.labels, "r", encoding="utf-8") as f:
            spec = json.load(f)
        schedule = LabelSchedule(spec)
    if args.plot_tz is None:
        args.plot_tz = schedule.tz if schedule else 0.0

    # ---------- 只重新統計 ----------
    if args.analyze_only:
        run_meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
        scored = [(f[:-7], os.path.join(scored_dir, f)) for f in sorted(os.listdir(scored_dir)) if f.endswith(".csv.gz")]
        print(f"[analyze-only] 找到 {len(scored)} 個已計分檔案")
        analyze(args, out_dir, scored, run_meta)
        print(f"\n完成 → {os.path.abspath(out_dir)}\\report.md")
        return

    if not args.pcap_dir:
        sys.exit("請指定 --pcap-dir")
    files = find_pcaps(args.pcap_dir, args.include, args.exclude)
    if not files:
        sys.exit(f"在 {args.pcap_dir} 找不到任何 {PCAP_EXTS} 檔案（或被 --include/--exclude 濾掉）")
    if args.inspect:
        inspect_files(files, schedule)
        return
    if not args.model:
        sys.exit("請指定 --model")

    # ---------- 載入模型並決定模式 ----------
    import torch
    from model_registry import load_anomaly_model
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    bundle = load_anomaly_model(args.model, device=device, default_latent_dim=args.latent_dim)
    rep = bundle.input_representation
    mode = args.mode
    if mode == "auto":
        if rep == "packet_bytes_visualizer":
            mode = "packet"
        elif rep.startswith("csv_features"):
            mode = "flow"
        else:
            sys.exit(f"無法由 representation={rep!r} 判斷模式，請明確指定 --mode packet|flow")
        print(f"[模式] auto → {mode}（checkpoint representation={rep!r}）")
        if rep == "csv_features_legacy":
            print("  [注意] 'csv_features_legacy' 是「checkpoint 沒有標記 representation」時的預設值。\n"
                  "         train_pcap_native.py / train_packet_native_vae.py 訓練的模型不會寫入此欄位，\n"
                  "         若你的模型是它們訓練的（封包位元組），請改用 --mode packet。")
    else:
        expect = "packet" if rep == "packet_bytes_visualizer" else "flow"
        if rep != "csv_features_legacy" and mode != expect:
            print(f"  [警告] 指定 --mode {mode}，但 checkpoint 標記 representation={rep!r}（應為 {expect}），分數可能無意義")
    thr_model = float(bundle.threshold)
    print(f"[模型] type={bundle.model_type}  threshold={thr_model:.6g}  device={device}  檔案數={len(files)}")

    payload = None
    if mode == "flow":
        if not args.scaler:
            sys.exit("flow 模式需要 --scaler（訓練期前處理器 .pkl；run_training_cicids2017_vae_fixed.py 會輸出 "
                     "<output>/cicids2017_preprocessor.pkl，舊流程請用 fit_reference_scaler.py 產生）")
        with open(args.scaler, "rb") as f:
            payload = pickle.load(f)   # 只載入你自己產生的檔案
        os.makedirs(flows_dir, exist_ok=True)
        print(f"[前處理器] 特徵欄位 {len(payload['feature_columns'])} 個，normalization={payload.get('normalization', 'none')}")

    run_meta = {"model": args.model, "model_type": bundle.model_type, "mode": mode,
                "model_threshold": thr_model, "representation": rep, "labels": args.labels,
                "started": datetime.now().isoformat(timespec="seconds")}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    tags = [(make_tag(r, p), p) for r, p in files]

    # ---------- flow 模式 stage 1：轉換（可平行） ----------
    log: List[dict] = []
    if mode == "flow":
        params = {"flow_timeout": args.flow_timeout, "idle_threshold": args.idle_threshold,
                  "bulk_threshold": args.bulk_threshold, "max_packets": args.max_packets_per_file}
        todo = []
        for tag, path in tags:
            csv_path = os.path.join(flows_dir, tag + ".csv")
            if os.path.exists(csv_path) and not args.force_convert and not args.force:
                print(f"[快取] 已有特徵 CSV：{tag}")
            else:
                todo.append((tag, path, csv_path))
        if todo:
            print(f"\n[Stage 1] PCAP → 流量特徵：{len(todo)} 個檔案，workers={args.workers}")
            if args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futs = {ex.submit(convert_worker, p, c, spec, params): (t, p) for t, p, c in todo}
                    for fu in as_completed(futs):
                        t, p = futs[fu]
                        res = fu.result()
                        print(f"  {'完成' if res['ok'] else '失敗'} {t}  {res.get('seconds')}s  {res.get('error', '')}")
                        log.append({"tag": t, "stage": "convert", **res})
            else:
                for t, p, c in todo:
                    print(f"\n[轉換] {os.path.basename(p)} ({os.path.getsize(p) / 1e9:.2f} GB)")
                    res = convert_worker(p, c, spec, params)
                    print(f"  {'完成' if res['ok'] else '失敗'}  {res.get('seconds')}s  {res.get('error', '')}")
                    log.append({"tag": t, "stage": "convert", **res})

    # ---------- stage 2：計分 ----------
    print("\n[Stage 2] 計分")
    scored = []
    for tag, path in tags:
        out_gz = os.path.join(scored_dir, tag + ".csv.gz")
        if os.path.exists(out_gz) and not args.force:
            print(f"[快取] 已有計分結果：{tag}")
            scored.append((tag, out_gz))
            continue
        t0 = time.time()
        try:
            print(f"\n[計分] {os.path.basename(path)}")
            if mode == "flow":
                csv_path = os.path.join(flows_dir, tag + ".csv")
                if not os.path.exists(csv_path):
                    raise FileNotFoundError("特徵 CSV 不存在（stage 1 失敗？）")
                info = score_flow_csv(csv_path, out_gz, payload, bundle)
            else:
                labeler = schedule.make_labeler(os.path.basename(path)) if schedule else None
                info = score_packet_file(path, out_gz, args, bundle, labeler)
                if info.get("linktype") not in (None, 1):
                    print(f"  [警告] linktype={info['linktype']}（非 Ethernet），封包影像化的偏移量會錯位，分數不可信")
            sec = time.time() - t0
            print(f"  完成：{info.get('n_scored', 0):,} 筆，{sec:.0f}s")
            log.append({"tag": tag, "stage": "score", "ok": True, "seconds": round(sec, 1), **info})
            scored.append((tag, out_gz))
        except Exception as e:  # noqa: BLE001
            print(f"  [失敗] {type(e).__name__}: {e}")
            log.append({"tag": tag, "stage": "score", "ok": False, "error": f"{type(e).__name__}: {e}"})
            if os.path.exists(out_gz + ".tmp"):
                os.remove(out_gz + ".tmp")

    with open(os.path.join(out_dir, "run_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False, default=_json_default)

    # ---------- stage 3：統計 ----------
    print("\n[Stage 3] 統計分析")
    analyze(args, out_dir, scored, run_meta)
    print(f"\n完成 → {os.path.abspath(os.path.join(out_dir, 'report.md'))}")


if __name__ == "__main__":
    main()
