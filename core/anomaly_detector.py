# ============================================================
# anomaly_detector.py - 獨立異常偵測模組（v2.0.0 新增）
#
# 支援偵測類型：
#   1. SYN Flood          - 大量 SYN 封包攻擊
#   2. Port Scan          - 端口掃描（橫向 / 縱向）
#   3. ICMP Flood         - 大量 ICMP 封包攻擊（Ping Flood）
#   4. UDP Flood          - 大量 UDP 封包攻擊
#   5. ARP Spoofing       - ARP 欺騙（同一 IP 出現多個 MAC）
#   6. DNS Amplification  - DNS 放大攻擊（回應遠大於請求）
# ============================================================

import time
from collections import defaultdict
from datetime import datetime
from colorama import Fore, Style, init

from config import (
    ALERT_THRESHOLD_SYN,
    ALERT_THRESHOLD_PORTS,
    ALERT_THRESHOLD_ICMP,
    ALERT_THRESHOLD_UDP,
    ALERT_THRESHOLD_DNS_AMP,
    ALERT_ARP_SPOOF_WINDOW,
)

init(autoreset=True)


class AnomalyDetector:
    """
    統一異常偵測器

    使用方法：
        detector = AnomalyDetector()
        alerts = detector.inspect(pkt, record)
        for alert in alerts:
            print(alert)
    """

    # 告警嚴重程度
    SEVERITY_LOW    = "LOW"
    SEVERITY_MEDIUM = "MEDIUM"
    SEVERITY_HIGH   = "HIGH"
    SEVERITY_CRITICAL = "CRITICAL"

    def __init__(self,
                 on_alert=None,
                 threshold_syn=ALERT_THRESHOLD_SYN,
                 threshold_ports=ALERT_THRESHOLD_PORTS,
                 threshold_icmp=ALERT_THRESHOLD_ICMP,
                 threshold_udp=ALERT_THRESHOLD_UDP):
        """
        Args:
            on_alert      : 告警回調函數 fn(alert_dict)，None 則自動印出
            threshold_syn : SYN Flood 閾值
            threshold_ports: Port Scan 閾值
            threshold_icmp: ICMP Flood 閾值
            threshold_udp : UDP Flood 閾值
        """
        self.on_alert       = on_alert or self._default_print_alert
        self.thr_syn        = threshold_syn
        self.thr_ports      = threshold_ports
        self.thr_icmp       = threshold_icmp
        self.thr_udp        = threshold_udp

        # 計數器
        self.syn_count   = defaultdict(int)        # {src_ip: count}
        self.icmp_count  = defaultdict(int)        # {src_ip: count}
        self.udp_count   = defaultdict(int)        # {src_ip: count}
        self.port_scan   = defaultdict(set)        # {src_ip: {port, ...}}
        self.arp_map     = defaultdict(dict)       # {ip: {mac: first_seen}}
        self.dns_req     = defaultdict(int)        # {src_ip: dns_request_count}
        self.dns_resp    = defaultdict(int)        # {src_ip: dns_response_count}

        # 已告警集合（防重複告警）
        self.alerted     = set()

        # 所有告警歷史記錄
        self.alert_history = []

    # ── 主要入口 ──────────────────────────────────────────
    def inspect(self, pkt, record: dict) -> list:
        """
        檢查單一封包，回傳本次觸發的告警列表

        Args:
            pkt    : Scapy Packet 物件
            record : PacketParser.parse() 的輸出

        Returns:
            list[dict]: 本次觸發的告警（可能為空列表）
        """
        alerts = []
        alerts += self._check_syn_flood(pkt, record)
        alerts += self._check_port_scan(record)
        alerts += self._check_icmp_flood(pkt, record)
        alerts += self._check_udp_flood(pkt, record)
        alerts += self._check_arp_spoof(record)
        alerts += self._check_dns_amplification(pkt, record)

        for a in alerts:
            self.alert_history.append(a)
            self.on_alert(a)

        return alerts

    # ── 1. SYN Flood ──────────────────────────────────────
    def _check_syn_flood(self, pkt, record) -> list:
        from scapy.layers.inet import TCP
        if not pkt.haslayer(TCP):
            return []
        if pkt[TCP].flags != "S":   # 只計純 SYN（無 ACK）
            return []

        src_ip = record.get("src_ip", "")
        self.syn_count[src_ip] += 1

        key = f"SYN_FLOOD_{src_ip}"
        if self.syn_count[src_ip] > self.thr_syn and key not in self.alerted:
            self.alerted.add(key)
            return [self._make_alert(
                attack_type="SYN Flood",
                severity=self.SEVERITY_HIGH,
                src_ip=src_ip,
                detail=f"SYN 封包數: {self.syn_count[src_ip]} (閾值: {self.thr_syn})",
                suggestion="封鎖來源 IP 或啟用 SYN Cookie 防護",
            )]
        return []

    # ── 2. Port Scan ──────────────────────────────────────
    def _check_port_scan(self, record) -> list:
        src_ip   = record.get("src_ip", "")
        dst_port = record.get("dst_port")

        if not src_ip or not isinstance(dst_port, int):
            return []

        self.port_scan[src_ip].add(dst_port)
        unique_ports = len(self.port_scan[src_ip])

        key = f"PORT_SCAN_{src_ip}"
        if unique_ports > self.thr_ports and key not in self.alerted:
            self.alerted.add(key)
            # 判斷掃描類型
            scan_type = self._classify_port_scan(record)
            return [self._make_alert(
                attack_type=f"Port Scan ({scan_type})",
                severity=self.SEVERITY_MEDIUM,
                src_ip=src_ip,
                detail=(f"已掃描 {unique_ports} 個 Port (閾值: {self.thr_ports})\n"
                        f"  掃描的 Port: {sorted(list(self.port_scan[src_ip]))[:20]}..."),
                suggestion="封鎖來源 IP；檢查是否為授權掃描",
            )]
        return []

    def _classify_port_scan(self, record) -> str:
        """依 TCP Flags 判斷掃描類型"""
        flags = record.get("flags", "")
        if flags == "SYN":
            return "SYN Scan"
        elif flags == "NONE":
            return "NULL Scan"
        elif flags and "FIN" in flags and "PSH" in flags and "URG" in flags:
            return "XMAS Scan"
        elif flags and "FIN" in flags:
            return "FIN Scan"
        elif flags and "ACK" in flags and "SYN" not in flags:
            return "ACK Scan"
        else:
            return "TCP Scan"

    # ── 3. ICMP Flood ─────────────────────────────────────
    def _check_icmp_flood(self, pkt, record) -> list:
        from scapy.layers.inet import ICMP
        if not pkt.haslayer(ICMP):
            return []
        # 只計 Echo Request（type=8）
        if pkt[ICMP].type != 8:
            return []

        src_ip = record.get("src_ip", "")
        self.icmp_count[src_ip] += 1

        key = f"ICMP_FLOOD_{src_ip}"
        if self.icmp_count[src_ip] > self.thr_icmp and key not in self.alerted:
            self.alerted.add(key)
            return [self._make_alert(
                attack_type="ICMP Flood (Ping Flood)",
                severity=self.SEVERITY_MEDIUM,
                src_ip=src_ip,
                detail=f"ICMP Echo Request 數: {self.icmp_count[src_ip]} (閾值: {self.thr_icmp})",
                suggestion="在防火牆封鎖 ICMP Echo Request 或限速",
            )]
        return []

    # ── 4. UDP Flood ──────────────────────────────────────
    def _check_udp_flood(self, pkt, record) -> list:
        from scapy.layers.inet import UDP
        if not pkt.haslayer(UDP):
            return []

        src_ip = record.get("src_ip", "")
        self.udp_count[src_ip] += 1

        key = f"UDP_FLOOD_{src_ip}"
        if self.udp_count[src_ip] > self.thr_udp and key not in self.alerted:
            self.alerted.add(key)
            return [self._make_alert(
                attack_type="UDP Flood",
                severity=self.SEVERITY_HIGH,
                src_ip=src_ip,
                detail=f"UDP 封包數: {self.udp_count[src_ip]} (閾值: {self.thr_udp})",
                suggestion="封鎖來源 IP；啟用 UDP 速率限制",
            )]
        return []

    # ── 5. ARP Spoofing ───────────────────────────────────
    def _check_arp_spoof(self, record) -> list:
        if record.get("protocol") != "ARP":
            return []

        src_ip  = record.get("arp_src_ip", "")
        src_mac = record.get("arp_src_mac", "")

        if not src_ip or not src_mac:
            return []

        now = time.time()
        ip_macs = self.arp_map[src_ip]

        if src_mac not in ip_macs:
            ip_macs[src_mac] = now
            # 若同一 IP 出現第二個（或以上）MAC，觸發告警
            if len(ip_macs) >= 2:
                key = f"ARP_SPOOF_{src_ip}"
                if key not in self.alerted:
                    self.alerted.add(key)
                    macs_str = ", ".join(ip_macs.keys())
                    return [self._make_alert(
                        attack_type="ARP Spoofing",
                        severity=self.SEVERITY_CRITICAL,
                        src_ip=src_ip,
                        detail=(f"IP {src_ip} 對應到多個 MAC 地址:\n"
                                f"  {macs_str}"),
                        suggestion="確認哪個 MAC 為合法主機；啟用動態 ARP 檢測（DAI）",
                    )]
        return []

    # ── 6. DNS Amplification ──────────────────────────────
    def _check_dns_amplification(self, pkt, record) -> list:
        """
        偵測 DNS 放大攻擊：
        攻擊者偽造受害者 IP 發送 ANY 查詢 -> DNS 伺服器回覆大量資料給受害者
        特徵：某 IP 收到大量 DNS 回應（qr=1）但幾乎沒有發出查詢（qr=0）
        """
        from scapy.layers.dns import DNS
        if not pkt.haslayer(DNS):
            return []

        dns = pkt[DNS]
        src_ip = record.get("src_ip", "")

        if dns.qr == 0:   # 查詢
            self.dns_req[src_ip] += 1
        else:              # 回應
            self.dns_resp[src_ip] += 1

        # 回應/請求 比例過高 -> 放大攻擊跡象
        req   = self.dns_req.get(src_ip, 0)
        resp  = self.dns_resp.get(src_ip, 0)
        ratio = resp / max(req, 1)

        key = f"DNS_AMP_{src_ip}"
        if (resp > 20 and ratio > ALERT_THRESHOLD_DNS_AMP
                and key not in self.alerted):
            self.alerted.add(key)
            return [self._make_alert(
                attack_type="DNS Amplification",
                severity=self.SEVERITY_HIGH,
                src_ip=src_ip,
                detail=(f"DNS 回應數: {resp}，查詢數: {req}，"
                        f"回應/查詢比: {ratio:.1f}x (閾值: {ALERT_THRESHOLD_DNS_AMP}x)"),
                suggestion="在 DNS 伺服器停用 ANY 查詢；限制 DNS 回應速率",
            )]
        return []

    # ── 重置計數器 ────────────────────────────────────────
    def reset(self):
        """重置所有計數器（保留 alert_history）"""
        self.syn_count.clear()
        self.icmp_count.clear()
        self.udp_count.clear()
        self.port_scan.clear()
        self.arp_map.clear()
        self.dns_req.clear()
        self.dns_resp.clear()
        self.alerted.clear()

    # ── 取得統計摘要 ──────────────────────────────────────
    def get_summary(self) -> dict:
        """回傳各類偵測計數摘要"""
        return {
            "total_alerts":  len(self.alert_history),
            "syn_top_ip":    self._top_n(self.syn_count),
            "icmp_top_ip":   self._top_n(self.icmp_count),
            "udp_top_ip":    self._top_n(self.udp_count),
            "port_scan_ips": {ip: len(ports)
                              for ip, ports in self.port_scan.items()},
            "arp_spoof_ips": {ip: list(macs.keys())
                              for ip, macs in self.arp_map.items()
                              if len(macs) >= 2},
        }

    @staticmethod
    def _top_n(counter: dict, n=5) -> dict:
        return dict(sorted(counter.items(), key=lambda x: x[1], reverse=True)[:n])

    # ── 建立告警 dict ─────────────────────────────────────
    @staticmethod
    def _make_alert(attack_type, severity, src_ip, detail, suggestion="") -> dict:
        return {
            "time":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attack_type": attack_type,
            "severity":    severity,
            "src_ip":      src_ip,
            "detail":      detail,
            "suggestion":  suggestion,
        }

    # ── 預設告警輸出 ──────────────────────────────────────
    @staticmethod
    def _default_print_alert(alert: dict):
        sev = alert["severity"]
        color = {
            "LOW":      Fore.YELLOW,
            "MEDIUM":   Fore.LIGHTYELLOW_EX,
            "HIGH":     Fore.RED,
            "CRITICAL": Fore.LIGHTRED_EX,
        }.get(sev, Fore.RED)

        print(f"\n{color}{'!'*65}")
        print(f"  !! [{sev}] {alert['attack_type']}")
        print(f"  時間    : {alert['time']}")
        print(f"  來源 IP : {alert['src_ip']}")
        print(f"  詳細    : {alert['detail']}")
        if alert.get("suggestion"):
            print(f"  建議    : {alert['suggestion']}")
        print(f"{'!'*65}{Style.RESET_ALL}\n")

    # ── 列印所有告警歷史 ──────────────────────────────────
    def print_history(self):
        if not self.alert_history:
            print(f"{Fore.GREEN}  無異常告警記錄{Style.RESET_ALL}")
            return
        print(f"\n{Fore.CYAN}{'='*65}\n  告警歷史 (共 {len(self.alert_history)} 筆)\n{'='*65}{Style.RESET_ALL}")
        for i, a in enumerate(self.alert_history, 1):
            print(f"  [{i:03d}] {a['time']}  [{a['severity']}]  {a['attack_type']}"
                  f"  src={a['src_ip']}")
