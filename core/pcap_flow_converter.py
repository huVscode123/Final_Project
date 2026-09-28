# ============================================================
# core/pcap_flow_converter.py
# PCAP → CICFlowMeter 相容流量特徵 CSV 轉換器（測試資料專用）
#
# 目的：
#   讓 CICDDoS2019 / CICIDS2017 / Malware Traffic Analysis（MTA）等
#   「只有原始 PCAP、沒有現成 CSV」的測試資料，能被轉換成與既有
#   core/dataset_loader.py（CICIDSLoader / CICDDoS2019Loader）訓練時
#   所使用的 CICFlowMeter 統計特徵格式一致的 CSV 檔案，因此可以直接
#   餵給「維持原樣、以 CSV 特徵訓練」的既有模型做測試/推論。
#
#   本檔案不依賴、也不修改任何既有的訓練程式碼
#   （core/dataset_loader.py、run_training.py 等完全不變）。
#
# 與 core/packet_visualizer.py 的差異（重要）：
#   packet_visualizer.py 是把「封包原始位元組」轉成 32×32 影像，
#   對應的是另一套「封包位元組原生」模型（core/train_packet_native_vae.py
#   那條路線）。
#   本檔案做的是完全相反的事：把封包「還原成 CICFlowMeter 那種
#   逐 Flow 統計特徵」，對應的是「維持原樣、以 CSV 訓練」的既有
#   CICIDSLoader / CICDDoS2019Loader 那條路線。兩者不可混用，
#   詳見隨附的說明文件。
#
# 特徵定義與已知近似值（務必先讀過再使用於嚴謹的評估）：
#   多數欄位（封包數、位元組數、IAT、Flag 計數、Header 長度等）為
#   精確計算；少數欄位（Bulk 統計、Active/Idle 切分、
#   min_seg_size_forward、Subflow 統計）因缺乏官方 CICFlowMeter
#   原始碼可對照，採用文獻中常見的簡化近似實作，數值可能與官方
#   CICFlowMeter 工具的輸出有小幅落差，但統計意義與相對大小關係
#   一致，足以作為「測試資料是否落在模型認得的分布內」的參考依據。
# ============================================================

from __future__ import annotations

import csv
import os
from typing import Callable, Dict, List, Optional, Tuple

from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6


# ──────────────────────────────────────────────────────────
# 欄位定義（與 core/Semi-supervised model and trainer/
# semi_supervised_cicids2017.py::_CICIDS2017_FEATURES 完全一致，
# 為 CICIDS2017／CICDDoS2019 兩份特徵清單的超集合，兩邊的
# 具名特徵比對都找得到對應欄位）
# ──────────────────────────────────────────────────────────
METADATA_COLUMNS: List[str] = [
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port",
    "Protocol", "Timestamp",
]

FEATURE_COLUMNS: List[str] = [
    "Destination Port", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

OUTPUT_COLUMNS: List[str] = METADATA_COLUMNS + FEATURE_COLUMNS + ["Label"]

# 型別別名：canon flow key = ((ip,port), (ip,port), protocol)
FlowKey = Tuple[Tuple[str, int], Tuple[str, int], int]
LabelFn = Callable[[FlowKey, float, float], str]


# ──────────────────────────────────────────────────────────
# 統計小工具（純 Python 實作，避免對 numpy 的逐流量呼叫開銷）
# ──────────────────────────────────────────────────────────
def _stats(values: List[float]) -> Tuple[float, float, float, float]:
    """回傳 (mean, std, max, min)；空清單回傳全 0。"""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in values) / n
        std = var ** 0.5
    else:
        std = 0.0
    return mean, std, max(values), min(values)


def _iat_stats(sorted_times: List[float]) -> Tuple[float, float, float, float, float]:
    """回傳連續時間差（微秒）的 (total, mean, std, max, min)。"""
    if len(sorted_times) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    diffs = [
        (sorted_times[i + 1] - sorted_times[i]) * 1e6
        for i in range(len(sorted_times) - 1)
    ]
    total = sum(diffs)
    mean, std, mx, mn = _stats(diffs)
    return total, mean, std, mx, mn


def _count_active_segments(sorted_times: List[float], idle_threshold: float) -> int:
    """依 idle_threshold 切分 Flow 為幾段連續活躍區間（供 Subflow 特徵使用）。"""
    if not sorted_times:
        return 1
    segments = 1
    for i in range(1, len(sorted_times)):
        if sorted_times[i] - sorted_times[i - 1] > idle_threshold:
            segments += 1
    return max(1, segments)


def _active_idle_stats(
    sorted_times: List[float], idle_threshold: float
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    """
    將 Flow 內的封包時間序列依 idle_threshold 切分為「活躍區間」與
    「閒置區間」，回傳 (Active 統計, Idle 統計)，各為 (mean,std,max,min)，
    單位微秒。少於 2 個封包時回傳全 0（沒有足夠資訊判斷活躍/閒置）。
    """
    if len(sorted_times) < 2:
        return (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)

    active_durations: List[float] = []
    idle_durations: List[float] = []
    seg_start = sorted_times[0]
    prev = sorted_times[0]

    for t in sorted_times[1:]:
        gap = t - prev
        if gap > idle_threshold:
            active_durations.append((prev - seg_start) * 1e6)
            idle_durations.append(gap * 1e6)
            seg_start = t
        prev = t
    active_durations.append((prev - seg_start) * 1e6)

    a_mean, a_std, a_max, a_min = _stats(active_durations)
    if idle_durations:
        i_mean, i_std, i_max, i_min = _stats(idle_durations)
    else:
        i_mean = i_std = i_max = i_min = 0.0
    return (a_mean, a_std, a_max, a_min), (i_mean, i_std, i_max, i_min)


def _bulk_stats(
    times: List[float], lens: List[int], bulk_threshold: float, min_bulk_size: int = 4
) -> Tuple[float, float, float]:
    """
    簡化版 Bulk（連續大量傳輸）偵測：同方向封包間隔 <= bulk_threshold
    且連續封包數 >= min_bulk_size 視為一個 bulk。
    回傳 (平均每 bulk 位元組數, 平均每 bulk 封包數, 平均 bulk 傳輸速率)。
    無 bulk 時回傳 (0, 0, 0)。這是官方 CICFlowMeter Bulk 特徵的簡化近似。
    """
    if len(times) < min_bulk_size:
        return 0.0, 0.0, 0.0

    paired = sorted(zip(times, lens), key=lambda p: p[0])
    bulks: List[Tuple[int, int, float]] = []  # (bytes, count, duration)
    cur_times = [paired[0][0]]
    cur_bytes = paired[0][1]
    cur_count = 1

    for t, l in paired[1:]:
        if t - cur_times[-1] <= bulk_threshold:
            cur_times.append(t)
            cur_bytes += l
            cur_count += 1
        else:
            if cur_count >= min_bulk_size:
                bulks.append((cur_bytes, cur_count, cur_times[-1] - cur_times[0]))
            cur_times = [t]
            cur_bytes = l
            cur_count = 1
    if cur_count >= min_bulk_size:
        bulks.append((cur_bytes, cur_count, cur_times[-1] - cur_times[0]))

    if not bulks:
        return 0.0, 0.0, 0.0

    avg_bytes = sum(b[0] for b in bulks) / len(bulks)
    avg_pkts = sum(b[1] for b in bulks) / len(bulks)
    rates = [(b[0] / b[2]) if b[2] > 0 else 0.0 for b in bulks]
    avg_rate = sum(rates) / len(rates)
    return avg_bytes, avg_pkts, avg_rate


# ──────────────────────────────────────────────────────────
# 封包欄位擷取（獨立於 core/parser.py，避免耦合，只取本模組需要的欄位）
# ──────────────────────────────────────────────────────────
def _extract_packet_info(pkt) -> Optional[dict]:
    """
    從單一 Scapy 封包擷取流量特徵化所需的欄位。
    非 IP/IPv6 封包（如純 ARP）回傳 None，予以略過。

    total_len / header_len 皆以「IP 層起始」為基準計算（排除 Ethernet
    等 L2 表頭），與 CICFlowMeter 的計算基準一致。
    """
    if pkt.haslayer(IP):
        ip = pkt[IP]
        src_ip, dst_ip = ip.src, ip.dst
        total_len = len(ip)
        header_len = (ip.ihl or 5) * 4
        proto_num = int(ip.proto)
    elif pkt.haslayer(IPv6):
        ip = pkt[IPv6]
        src_ip, dst_ip = ip.src, ip.dst
        total_len = len(ip)
        header_len = 40
        proto_num = int(ip.nh)
    else:
        return None

    src_port = dst_port = 0
    tcp_flags: Optional[str] = None
    tcp_window: Optional[int] = None

    if pkt.haslayer(TCP):
        tcp = pkt[TCP]
        src_port, dst_port = int(tcp.sport), int(tcp.dport)
        header_len += (tcp.dataofs or 5) * 4
        tcp_flags = str(tcp.flags)
        tcp_window = int(tcp.window) if tcp.window is not None else 0
    elif pkt.haslayer(UDP):
        udp = pkt[UDP]
        src_port, dst_port = int(udp.sport), int(udp.dport)
        header_len += 8

    payload_len = max(0, total_len - header_len)

    return {
        "time": float(pkt.time),
        "src_ip": src_ip, "dst_ip": dst_ip,
        "src_port": src_port, "dst_port": dst_port,
        "protocol": proto_num,
        "total_len": total_len,
        "header_len": header_len,
        "payload_len": payload_len,
        "tcp_flags": tcp_flags,
        "tcp_window": tcp_window,
    }


# ──────────────────────────────────────────────────────────
# 單一 Flow 的狀態累積器
# ──────────────────────────────────────────────────────────
class _FlowRecord:
    def __init__(self, fwd_ip: str, fwd_port: int, bwd_ip: str, bwd_port: int,
                 protocol: int, first_time: float):
        self.fwd_ip, self.fwd_port = fwd_ip, fwd_port
        self.bwd_ip, self.bwd_port = bwd_ip, bwd_port
        self.protocol = protocol
        self.first_time = first_time
        self.last_time = first_time

        self.fwd_lens: List[int] = []
        self.bwd_lens: List[int] = []
        self.fwd_times: List[float] = []
        self.bwd_times: List[float] = []
        self.all_times: List[float] = []

        self.fwd_header_bytes = 0
        self.bwd_header_bytes = 0

        # scapy TCP flags 字母對應：F=FIN S=SYN R=RST P=PSH A=ACK U=URG E=ECE C=CWR
        self.flag_counts: Dict[str, int] = {k: 0 for k in "FSRPAUEC"}
        self.fwd_psh = 0
        self.bwd_psh = 0
        self.fwd_urg = 0
        self.bwd_urg = 0

        self.init_win_fwd: Optional[int] = None
        self.init_win_bwd: Optional[int] = None
        self.act_data_pkt_fwd = 0
        self.min_seg_size_fwd: Optional[int] = None

    def add_packet(self, direction: str, ts: float, total_len: int, header_len: int,
                    payload_len: int, tcp_flags: Optional[str], tcp_window: Optional[int]):
        self.last_time = max(self.last_time, ts)
        self.all_times.append(ts)

        if direction == "fwd":
            self.fwd_lens.append(total_len)
            self.fwd_times.append(ts)
            self.fwd_header_bytes += header_len
            if payload_len > 0:
                self.act_data_pkt_fwd += 1
            if self.min_seg_size_fwd is None or header_len < self.min_seg_size_fwd:
                self.min_seg_size_fwd = header_len
            if self.init_win_fwd is None and tcp_window is not None:
                self.init_win_fwd = tcp_window
            if tcp_flags:
                if "P" in tcp_flags:
                    self.fwd_psh += 1
                if "U" in tcp_flags:
                    self.fwd_urg += 1
        else:
            self.bwd_lens.append(total_len)
            self.bwd_times.append(ts)
            self.bwd_header_bytes += header_len
            if self.init_win_bwd is None and tcp_window is not None:
                self.init_win_bwd = tcp_window
            if tcp_flags:
                if "P" in tcp_flags:
                    self.bwd_psh += 1
                if "U" in tcp_flags:
                    self.bwd_urg += 1

        if tcp_flags:
            for ch in tcp_flags:
                if ch in self.flag_counts:
                    self.flag_counts[ch] += 1

    def finalize(self, idle_threshold: float, bulk_threshold: float) -> Dict[str, object]:
        duration_s = max(self.last_time - self.first_time, 0.0)
        duration_us = duration_s * 1e6

        total_fwd = len(self.fwd_lens)
        total_bwd = len(self.bwd_lens)
        total_fwd_bytes = sum(self.fwd_lens)
        total_bwd_bytes = sum(self.bwd_lens)

        f_mean, f_std, f_max, f_min = _stats(self.fwd_lens)
        b_mean, b_std, b_max, b_min = _stats(self.bwd_lens)
        all_lens = self.fwd_lens + self.bwd_lens
        p_mean, p_std, p_max, p_min = _stats(all_lens)

        flow_bytes_s = (total_fwd_bytes + total_bwd_bytes) / duration_s if duration_s > 0 else 0.0
        flow_pkts_s = (total_fwd + total_bwd) / duration_s if duration_s > 0 else 0.0
        fwd_pkts_s = total_fwd / duration_s if duration_s > 0 else 0.0
        bwd_pkts_s = total_bwd / duration_s if duration_s > 0 else 0.0

        _, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = \
            _iat_stats(sorted(self.all_times))
        fwd_iat_total, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = \
            _iat_stats(sorted(self.fwd_times))
        bwd_iat_total, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = \
            _iat_stats(sorted(self.bwd_times))

        down_up = (total_bwd / total_fwd) if total_fwd > 0 else 0.0

        fwd_bulk_bytes, fwd_bulk_pkts, fwd_bulk_rate = \
            _bulk_stats(self.fwd_times, self.fwd_lens, bulk_threshold)
        bwd_bulk_bytes, bwd_bulk_pkts, bwd_bulk_rate = \
            _bulk_stats(self.bwd_times, self.bwd_lens, bulk_threshold)

        n_sub = _count_active_segments(sorted(self.all_times), idle_threshold)
        (a_mean, a_std, a_max, a_min), (i_mean, i_std, i_max, i_min) = \
            _active_idle_stats(sorted(self.all_times), idle_threshold)

        row: Dict[str, object] = {
            "Flow ID": f"{self.fwd_ip}:{self.fwd_port}-{self.bwd_ip}:{self.bwd_port}-{self.protocol}",
            "Src IP": self.fwd_ip, "Src Port": self.fwd_port,
            "Dst IP": self.bwd_ip, "Dst Port": self.bwd_port,
            "Protocol": self.protocol,
            "Timestamp": self.first_time,

            "Destination Port": self.bwd_port,
            "Flow Duration": duration_us,
            "Total Fwd Packets": total_fwd,
            "Total Backward Packets": total_bwd,
            "Total Length of Fwd Packets": total_fwd_bytes,
            "Total Length of Bwd Packets": total_bwd_bytes,
            "Fwd Packet Length Max": f_max, "Fwd Packet Length Min": f_min,
            "Fwd Packet Length Mean": f_mean, "Fwd Packet Length Std": f_std,
            "Bwd Packet Length Max": b_max, "Bwd Packet Length Min": b_min,
            "Bwd Packet Length Mean": b_mean, "Bwd Packet Length Std": b_std,
            "Flow Bytes/s": flow_bytes_s, "Flow Packets/s": flow_pkts_s,
            "Flow IAT Mean": flow_iat_mean, "Flow IAT Std": flow_iat_std,
            "Flow IAT Max": flow_iat_max, "Flow IAT Min": flow_iat_min,
            "Fwd IAT Total": fwd_iat_total, "Fwd IAT Mean": fwd_iat_mean,
            "Fwd IAT Std": fwd_iat_std, "Fwd IAT Max": fwd_iat_max, "Fwd IAT Min": fwd_iat_min,
            "Bwd IAT Total": bwd_iat_total, "Bwd IAT Mean": bwd_iat_mean,
            "Bwd IAT Std": bwd_iat_std, "Bwd IAT Max": bwd_iat_max, "Bwd IAT Min": bwd_iat_min,
            "Fwd PSH Flags": self.fwd_psh, "Bwd PSH Flags": self.bwd_psh,
            "Fwd URG Flags": self.fwd_urg, "Bwd URG Flags": self.bwd_urg,
            "Fwd Header Length": self.fwd_header_bytes, "Bwd Header Length": self.bwd_header_bytes,
            "Fwd Packets/s": fwd_pkts_s, "Bwd Packets/s": bwd_pkts_s,
            "Min Packet Length": p_min, "Max Packet Length": p_max,
            "Packet Length Mean": p_mean, "Packet Length Std": p_std,
            "Packet Length Variance": p_std ** 2,
            "FIN Flag Count": self.flag_counts["F"], "SYN Flag Count": self.flag_counts["S"],
            "RST Flag Count": self.flag_counts["R"], "PSH Flag Count": self.flag_counts["P"],
            "ACK Flag Count": self.flag_counts["A"], "URG Flag Count": self.flag_counts["U"],
            "CWE Flag Count": self.flag_counts["C"], "ECE Flag Count": self.flag_counts["E"],
            "Down/Up Ratio": down_up, "Average Packet Size": p_mean,
            "Avg Fwd Segment Size": f_mean, "Avg Bwd Segment Size": b_mean,
            "Fwd Header Length.1": self.fwd_header_bytes,
            "Fwd Avg Bytes/Bulk": fwd_bulk_bytes, "Fwd Avg Packets/Bulk": fwd_bulk_pkts,
            "Fwd Avg Bulk Rate": fwd_bulk_rate,
            "Bwd Avg Bytes/Bulk": bwd_bulk_bytes, "Bwd Avg Packets/Bulk": bwd_bulk_pkts,
            "Bwd Avg Bulk Rate": bwd_bulk_rate,
            "Subflow Fwd Packets": total_fwd / n_sub, "Subflow Fwd Bytes": total_fwd_bytes / n_sub,
            "Subflow Bwd Packets": total_bwd / n_sub, "Subflow Bwd Bytes": total_bwd_bytes / n_sub,
            "Init_Win_bytes_forward": self.init_win_fwd or 0,
            "Init_Win_bytes_backward": self.init_win_bwd or 0,
            "act_data_pkt_fwd": self.act_data_pkt_fwd,
            "min_seg_size_forward": self.min_seg_size_fwd or 0,
            "Active Mean": a_mean, "Active Std": a_std, "Active Max": a_max, "Active Min": a_min,
            "Idle Mean": i_mean, "Idle Std": i_std, "Idle Max": i_max, "Idle Min": i_min,
        }
        return row


# ──────────────────────────────────────────────────────────
# 標籤產生器（可選）
# ──────────────────────────────────────────────────────────
def make_static_labeler(label: str) -> LabelFn:
    """整個 PCAP 的所有流量都標記為同一個標籤（例如已知整份檔案皆為良性流量）。"""
    return lambda key, t0, t1: label


def make_ip_time_labeler(
    attacker_ips: List[str],
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
    benign_label: str = "BENIGN",
    attack_label: str = "ATTACK",
) -> LabelFn:
    """
    依官方資料集文件公布的攻擊者 IP／攻擊時段區間標記流量，
    比照隨附說明文件、以及「異常封包模擬檢測問題與解決方案.md」
    第 4.4 節建議的篩選方式：IP 命中且時間重疊才視為攻擊流量。
    time_start/time_end 皆為 None 時，只依 IP 判斷。
    """
    attacker_set = set(attacker_ips or [])

    def _labeler(key: FlowKey, t0: float, t1: float) -> str:
        (ip_a, _port_a), (ip_b, _port_b), _proto = key
        ip_match = (ip_a in attacker_set) or (ip_b in attacker_set)
        if time_start is not None and time_end is not None:
            time_match = not (t1 < time_start or t0 > time_end)
        else:
            time_match = True
        return attack_label if (ip_match and time_match) else benign_label

    return _labeler


# ──────────────────────────────────────────────────────────
# 主轉換器
# ──────────────────────────────────────────────────────────
class PcapFlowConverter:
    """
    將 PCAP 轉換為 CICFlowMeter 相容的逐 Flow 統計特徵 CSV。

    Args:
        flow_timeout    : 同一組五元組超過此秒數沒有新封包，視為舊 Flow
                           結束、新封包起算新 Flow（CICFlowMeter 預設概念
                           相同，實際數值可依需求調整；預設 120 秒）。
        idle_threshold   : 判定「閒置區間」的秒數門檻，供 Active/Idle、
                           Subflow 系列特徵使用（預設 1 秒）。
        bulk_threshold   : 判定「Bulk 傳輸」的封包間隔秒數門檻（預設 1 秒）。
        expiry_check_every: 每處理多少個封包，掃描一次是否有逾時的舊
                           Flow 需要寫出並釋放記憶體（大型 PCAP 用）。
    """

    def __init__(self, flow_timeout: float = 120.0, idle_threshold: float = 1.0,
                 bulk_threshold: float = 1.0, expiry_check_every: int = 20000):
        self.flow_timeout = flow_timeout
        self.idle_threshold = idle_threshold
        self.bulk_threshold = bulk_threshold
        self.expiry_check_every = expiry_check_every

    def convert(self, pcap_path: str, output_csv: str,
                label_fn: Optional[LabelFn] = None,
                default_label: str = "UNKNOWN",
                progress_every: int = 200000) -> str:
        label_fn = label_fn or (lambda key, t0, t1: default_label)
        flows: Dict[FlowKey, _FlowRecord] = {}
        n_pkts = 0
        n_flows_written = 0
        n_skipped_non_ip = 0

        out_dir = os.path.dirname(os.path.abspath(output_csv))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()

            def _flush_flow(key: FlowKey, rec: _FlowRecord):
                nonlocal n_flows_written
                row = rec.finalize(self.idle_threshold, self.bulk_threshold)
                row["Label"] = label_fn(key, rec.first_time, rec.last_time)
                writer.writerow(row)
                n_flows_written += 1

            print(f"[PcapFlowConverter] 開始讀取: {pcap_path}")
            with PcapReader(pcap_path) as reader:
                for pkt in reader:
                    n_pkts += 1
                    info = _extract_packet_info(pkt)
                    if info is None:
                        n_skipped_non_ip += 1
                        continue

                    canon: FlowKey = tuple(sorted([
                        (info["src_ip"], info["src_port"]),
                        (info["dst_ip"], info["dst_port"]),
                    ])) + (info["protocol"],)  # type: ignore[assignment]

                    rec = flows.get(canon)
                    if rec is not None and info["time"] - rec.last_time > self.flow_timeout:
                        _flush_flow(canon, rec)
                        rec = None
                    if rec is None:
                        rec = _FlowRecord(
                            fwd_ip=info["src_ip"], fwd_port=info["src_port"],
                            bwd_ip=info["dst_ip"], bwd_port=info["dst_port"],
                            protocol=info["protocol"], first_time=info["time"],
                        )
                        flows[canon] = rec

                    direction = ("fwd" if (info["src_ip"], info["src_port"])
                                 == (rec.fwd_ip, rec.fwd_port) else "bwd")
                    rec.add_packet(direction, info["time"], info["total_len"],
                                    info["header_len"], info["payload_len"],
                                    info["tcp_flags"], info["tcp_window"])

                    if n_pkts % self.expiry_check_every == 0:
                        now = info["time"]
                        expired = [k for k, r in flows.items()
                                   if now - r.last_time > self.flow_timeout]
                        for k in expired:
                            _flush_flow(k, flows.pop(k))

                    if progress_every and n_pkts % progress_every == 0:
                        print(f"  已處理 {n_pkts:,} 個封包，目前活躍 Flow "
                              f"{len(flows):,} 條，已輸出 {n_flows_written:,} 條")

            for k, r in list(flows.items()):
                _flush_flow(k, r)

        print(f"[PcapFlowConverter] 完成: {pcap_path}")
        print(f"  總封包數        : {n_pkts:,}")
        print(f"  略過（非 IP 層）: {n_skipped_non_ip:,}")
        print(f"  輸出 Flow 記錄數: {n_flows_written:,}")
        print(f"  輸出檔案        : {output_csv}")
        return output_csv
