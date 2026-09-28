#!/usr/bin/env python3
# ============================================================
# core/convert_test_pcap.py
# CLI 入口：把單一 PCAP 轉換為 CICFlowMeter 相容的流量特徵 CSV
#
# 使用情境：CICDDoS2019 / CICIDS2017 / Malware Traffic Analysis（MTA）
#           的原始 PCAP（僅用於測試，不用於訓練）。
#
# 範例：
#   # CICIDS2017：已知官方文件公布的攻擊者 IP 與攻擊時段，可精確標記
#   python core/convert_test_pcap.py \
#       --pcap data/cicids2017_pcap/Wednesday-WorkingHours.pcap \
#       --dataset cicids2017 \
#       --output output/test_flows/wed_cicids2017.csv \
#       --attacker-ip 172.16.0.1 \
#       --time-start 2017-07-05T09:20:00 --time-end 2017-07-05T10:00:00
#
#   # CICIDS2017：已知整份 PCAP 皆為良性流量（例如官方 Monday 檔案）
#   python core/convert_test_pcap.py \
#       --pcap data/cicids2017_pcap/Monday-WorkingHours.pcap \
#       --dataset cicids2017 \
#       --output output/test_flows/monday_cicids2017.csv \
#       --label BENIGN
#
#   # CICDDoS2019：同理
#   python core/convert_test_pcap.py \
#       --pcap data/cicddos2019_pcap/DrDoS_DNS.pcap \
#       --dataset cicddos2019 \
#       --output output/test_flows/drdos_dns.csv \
#       --attacker-ip 172.16.0.5 --label ATTACK
#
#   # Malware Traffic Analysis：通常沒有官方 IP／時段對照表，
#   # 不指定 --label / --attacker-ip 時一律標記為 UNKNOWN，
#   # 僅供模型推論觀察，不建議用於計算 Precision/Recall 等準確率指標。
#   python core/convert_test_pcap.py \
#       --pcap data/mta/2026-01-15-traffic-analysis-exercise.pcap \
#       --dataset mta \
#       --output output/test_flows/mta_20260115.csv
# ============================================================

import os
import sys
import argparse
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from pcap_flow_converter import PcapFlowConverter, make_static_labeler, make_ip_time_labeler


def _parse_time(value):
    """接受 Unix Epoch（數字字串）或 ISO 8601（如 2017-07-05T09:20:00）。"""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return datetime.fromisoformat(value).timestamp()


def main():
    parser = argparse.ArgumentParser(
        description="將 PCAP 轉換為 CICFlowMeter 相容的流量特徵 CSV（測試資料專用，不影響既有 CSV 訓練流程）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pcap", required=True, help="輸入 PCAP 檔案路徑")
    parser.add_argument("--output", required=True, help="輸出 CSV 路徑")
    parser.add_argument("--dataset", default="generic",
                        choices=["cicids2017", "cicddos2019", "mta", "generic"],
                        help="僅影響提示訊息，不影響特徵計算邏輯（三者共用同一套流量特徵化規則）")

    parser.add_argument("--flow-timeout", type=float, default=120.0,
                        help="Flow 逾時秒數，超過此值視為新 Flow（預設 120）")
    parser.add_argument("--idle-threshold", type=float, default=1.0,
                        help="Active/Idle、Subflow 特徵使用的閒置門檻秒數（預設 1.0）")
    parser.add_argument("--bulk-threshold", type=float, default=1.0,
                        help="Bulk 傳輸偵測使用的封包間隔秒數門檻（預設 1.0）")

    parser.add_argument("--label", default=None,
                        help="整份 PCAP 固定標記為此標籤（如 BENIGN / ATTACK）。"
                             "若同時提供 --attacker-ip，則以 --attacker-ip 的逐流量判斷為準")
    parser.add_argument("--attacker-ip", action="append", default=None,
                        help="攻擊者／受害者 IP（可重複指定多個）。"
                             "命中則標記為 ATTACK，否則標記為 BENIGN")
    parser.add_argument("--time-start", default=None,
                        help="搭配 --attacker-ip 使用：攻擊時段起點（Unix Epoch 或 ISO 8601）")
    parser.add_argument("--time-end", default=None,
                        help="搭配 --attacker-ip 使用：攻擊時段終點（Unix Epoch 或 ISO 8601）")

    args = parser.parse_args()

    if args.attacker_ip:
        labeler = make_ip_time_labeler(
            args.attacker_ip,
            time_start=_parse_time(args.time_start),
            time_end=_parse_time(args.time_end),
        )
        default_label = "UNKNOWN"
        print(f"  [標記模式] 依攻擊者 IP {args.attacker_ip} "
              f"{'與時段 ' + str(args.time_start) + ' ~ ' + str(args.time_end) if args.time_start else ''}"
              f"逐 Flow 判斷 BENIGN / ATTACK")
    elif args.label:
        labeler = make_static_labeler(args.label)
        default_label = args.label
        print(f"  [標記模式] 整份 PCAP 固定標記為: {args.label}")
    else:
        labeler = None
        default_label = "UNKNOWN"
        print("  [標記模式] 未提供標籤資訊，所有 Flow 標記為 UNKNOWN。")
        if args.dataset == "mta":
            print("  [提示] MTA 資料通常沒有官方 IP／時段對照表，"
                  "UNKNOWN 標籤僅供模型推論觀察用，不建議用於計算準確率指標。")
        else:
            print("  [提示] 若知道整份檔案的性質，建議加上 --label BENIGN/ATTACK；"
                  "若知道攻擊者 IP 與時段，建議加上 --attacker-ip / --time-start / --time-end "
                  "以取得逐 Flow 的精確標記。")

    converter = PcapFlowConverter(
        flow_timeout=args.flow_timeout,
        idle_threshold=args.idle_threshold,
        bulk_threshold=args.bulk_threshold,
    )
    converter.convert(args.pcap, args.output, label_fn=labeler, default_label=default_label)


if __name__ == "__main__":
    main()
