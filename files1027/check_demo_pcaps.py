# -*- coding: utf-8 -*-
"""
check_demo_pcaps.py —— Demo PCAP 的「模型前」健檢（不需載入任何模型）

在跑 E2E 之前先確認：PCAP 轉出的 Flow 數量/特徵是否合理、規則引擎告警是否正確歸因。
用法（在專案 core 目錄下）：
    python check_demo_pcaps.py --pcap-dir test_pcaps/cicids2017_demo_v2 [--detector original|extended]
選配：--csv-dir data/cicids2017  會把每個 Flow 特徵拿去和 CICIDS2017 訓練 CSV 的 Benign / 對應攻擊分布比對
      （每個特徵是否落在 P1~P99 內），這是量化「合成資料偏離訓練分布」的指標。
"""
import argparse, contextlib, csv, glob, io, os, statistics as st, sys
sys.path.insert(0, os.getcwd())
from scapy.all import PcapReader
from pcap_flow_converter import PcapFlowConverter
from parser import PacketParser

ATTACKER, VICTIM = "172.16.0.1", "192.168.10.50"
# name 前綴 -> (最少 flow 數, 期望告警子字串 or None, CICIDS2017 Label)
EXPECT = {
    "normal_01": (8, None, "BENIGN"), "normal_02": (20, None, "BENIGN"), "normal_03": (3, None, "BENIGN"),
    "normal_04": (4, None, "BENIGN"), "normal_05": (8, None, "BENIGN"),
    "attack_01": (100, "SYN Flood", "DDoS"), "attack_02": (80, "Port Scan", "PortScan"),
    "attack_03": (20, "SSH Brute Force", "SSH-Patator"),
    "attack_04": (8, "Web Login Brute Force", "Web Attack"), "attack_05": (25, "HTTP Flood", "DoS Hulk"),
}


def med(rows, k):
    v = [float(r[k]) for r in rows]
    return st.median(v) if v else 0.0


def convert(pcap):
    csvp = pcap + ".flows.csv"
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        PcapFlowConverter().convert(pcap, csvp, default_label="UNKNOWN")
    rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
    reasons = [l for l in buf.getvalue().splitlines() if "結束原因" in l]
    return rows, (reasons[0].split(":", 1)[1].strip() if reasons else "")


def alerts_for(pcap, which):
    if which == "extended":
        from anomaly_detector_extra import ExtendedAnomalyDetector as D
    else:
        from anomaly_detector import AnomalyDetector as D
    det, p, out = D(on_alert=lambda a: None), PacketParser(), []
    with contextlib.redirect_stdout(io.StringIO()):
        for pkt in PcapReader(pcap):
            out += det.inspect(pkt, p.parse(pkt), float(pkt.time))
    return out


def csv_ranges(csv_dir, labels):
    import pandas as pd
    cols = None; frames = []
    for f in glob.glob(os.path.join(csv_dir, "*.csv")):
        df = pd.read_csv(f, skipinitialspace=True, low_memory=False); df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df


def in_range_ratio(rows, ref, label_filter):
    """每個數值特徵：我們的 flow 值是否落在參考(類別)分布的 P1~P99。回傳落在範圍內的比例。"""
    import pandas as pd
    from pcap_flow_converter import FEATURE_COLUMNS
    sub = ref[ref["Label"].astype(str).str.contains(label_filter, case=False, regex=False)]
    if len(sub) < 50 or not rows:
        return None
    ok = tot = 0
    for c in FEATURE_COLUMNS:
        if c == "Destination Port" or c not in sub.columns:
            continue
        col = pd.to_numeric(sub[c], errors="coerce").replace([float("inf"), float("-inf")], float("nan")).dropna()
        if col.empty:
            continue
        lo, hi = col.quantile(0.01), col.quantile(0.99)
        for r in rows:
            try:
                v = float(r[c])
            except (ValueError, KeyError):
                continue
            tot += 1
            ok += (lo <= v <= hi)
    return ok / tot if tot else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap-dir", required=True)
    ap.add_argument("--detector", choices=["original", "extended"], default="extended")
    ap.add_argument("--csv-dir")
    a = ap.parse_args()
    ref = csv_ranges(a.csv_dir, None) if a.csv_dir else None
    bad = 0
    print(f"{'PCAP':26s}{'KB':>6s}{'pkts':>6s}{'flows':>6s}{'fwdP':>5s}{'bwdP':>5s}{'dur(s)':>8s}{'pps':>9s}  告警")
    for pcap in sorted(glob.glob(os.path.join(a.pcap_dir, "*.pcap"))):
        name = os.path.basename(pcap)[:-5]
        n_pk = sum(1 for _ in PcapReader(pcap))
        rows, reasons = convert(pcap)
        al = alerts_for(pcap, a.detector)
        kinds = sorted({(x["attack_type"], x["src_ip"]) for x in al})
        print(f"{name:26s}{os.path.getsize(pcap)/1024:6.1f}{n_pk:6d}{len(rows):6d}"
              f"{med(rows,'Total Fwd Packets'):5.0f}{med(rows,'Total Backward Packets'):5.0f}"
              f"{med(rows,'Flow Duration')/1e6:8.3f}{med(rows,'Flow Packets/s'):9.1f}  "
              + ("; ".join(f"{t}@{s}" for t, s in kinds) or "(無)"))
        print(f"{'':26s}結束原因: {reasons}")
        min_flows, want, label = next(v for k, v in EXPECT.items() if name.startswith(k))
        problems = []
        if len(rows) < min_flows: problems.append(f"flow 數 {len(rows)} < {min_flows}")
        if want and not any(want in t for t, _ in kinds): problems.append(f"缺少告警「{want}」")
        if not want and kinds: problems.append("正常流量不應有告警")
        if any(s == VICTIM for _, s in kinds): problems.append("告警來源是受害者（歸因錯誤）")
        if any(s not in (ATTACKER, VICTIM) for _, s in kinds) and name.startswith("attack"): problems.append("告警來源非攻擊者")
        if ref is not None:
            r = in_range_ratio(rows, ref, label)
            rb = in_range_ratio(rows, ref, "BENIGN")
            print(f"{'':26s}特徵落在 CICIDS2017[{label}] P1~P99 比例: {r if r is None else f'{r:.0%}'}  |  [BENIGN]: {rb if rb is None else f'{rb:.0%}'}")
        for p in problems:
            print(f"{'':26s}❌ {p}"); bad += 1
    print("\n結果：", "全部通過 ✅" if not bad else f"{bad} 項問題 ❌")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
