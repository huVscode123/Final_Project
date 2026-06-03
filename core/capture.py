# ============================================================
# capture.py - 即時監聽網路介面（NIC）v2.0.1 修正版
#
# 修正清單：
#   [Bug 7] _packet_callback：packet_count / total_bytes 在
#           parser.parse() 拋出例外時仍會遞增，導致統計數字
#           比 parsed_records 大。現在將統計遞增移到 parse 成功
#           之後，並加入 try/except 確保例外封包以 fallback
#           record 繼續計入，不影響後續流程。
# ============================================================

import time
import threading
from collections import defaultdict, Counter
from datetime import datetime

from scapy.all import sniff, get_if_list, conf, wrpcap
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import Ether, ARP

from colorama import Fore, Style, init
from tabulate import tabulate

from config import OUTPUT_PCAP, PORT_SERVICE_MAP
from parser import PacketParser
from storage import PacketStorage
from anomaly_detector import AnomalyDetector

init(autoreset=True)


class LiveCapture:
    """
    即時網路封包擷取類別（v2.0.1 修正版）

    功能：
      - 監聽指定網路介面（NIC）
      - 即時解析並顯示封包資訊（含協議顏色）
      - 整合 AnomalyDetector：SYN Flood / Port Scan / ICMP Flood /
        UDP Flood / ARP Spoofing / DNS 放大
      - 每 stats_interval 秒自動列印即時流量統計
      - 自動儲存為 PCAP、CSV、JSON（可選 SQLite）
    """

    def __init__(self,
                 interface=None,
                 bpf_filter="",
                 count=0,
                 save_pcap=True,
                 stats_interval=10,
                 save_db=False,
                 custom_alert_callback=None,
                 session_dir=None):
        self.interface       = interface or conf.iface
        self.bpf_filter      = bpf_filter
        self.count           = count
        self.save_pcap       = save_pcap
        self.stats_interval  = stats_interval
        self.save_db         = save_db

        self.packets         = []
        self.parsed_records  = []
        self.packet_count    = 0
        self.total_bytes     = 0
        self.start_time      = None

        self.session_dir = session_dir
        self.parser   = PacketParser()
        self.storage  = PacketStorage(session_dir=session_dir)
        self.detector = AnomalyDetector(on_alert=custom_alert_callback)

        # 統計計數器
        self.proto_counter    = Counter()
        self.src_ip_counter   = Counter()
        self.dst_port_counter = Counter()
        self.size_buckets     = Counter()

        self._stop_event   = threading.Event()
        self._stats_thread = None
        self._lock         = threading.Lock()

    # ── 列出所有網路介面 ─────────────────────────────────
    @staticmethod
    def list_interfaces():
        print(f"\n{Fore.CYAN}{'='*65}")
        print("  可用的網路介面")
        print(f"{'='*65}{Style.RESET_ALL}")

        try:
            from scapy.arch.windows import get_windows_if_list
            ifaces = get_windows_if_list()
            for idx, iface in enumerate(ifaces):
                name = iface["name"]
                desc = iface.get("description", "")[:40]
                mark = (f"{Fore.GREEN}*{Style.RESET_ALL}"
                        if name == conf.iface else " ")
                short = name[-45:] if len(name) > 45 else name
                print(f"  {mark} [{idx}] {desc:<42} {Fore.LIGHTBLACK_EX}{short}{Style.RESET_ALL}")
        except Exception:
            for idx, iface in enumerate(get_if_list()):
                mark = (f"{Fore.GREEN}*{Style.RESET_ALL}"
                        if iface == conf.iface else " ")
                print(f"  {mark} [{idx}] {iface}")

        print(f"\n{Fore.YELLOW}  * = 目前預設介面{Style.RESET_ALL}")
        print(f'  使用方式: python main.py live -i "介面名稱" -c 50')
        print(f"{Fore.CYAN}{'='*65}{Style.RESET_ALL}\n")

    # ── 封包回調 ─────────────────────────────────────────
    def _packet_callback(self, pkt):
        """
        每次擷取到封包時呼叫。

        [Bug 7 修正] 原版在 parser.parse() 呼叫之前就已
        遞增 packet_count / total_bytes，若 parse 拋出例外
        則 parsed_records 不會新增，造成兩者計數不一致。

        修正策略：
          1. 先嘗試 parse；失敗時使用 fallback record。
          2. 在同一個 lock 區塊中，一次性完成
             packet_count、total_bytes、packets、parsed_records
             與統計計數器的更新，確保原子性一致。
        """
        # ① 先解析（在 lock 外執行，避免長時間持有鎖）
        try:
            record = self.parser.parse(pkt)
        except Exception:
            # 解析失敗時使用最小化 fallback，仍計入統計
            record = {
                "protocol": "UNKNOWN",
                "length":   len(pkt),
                "src_ip":   None,
                "dst_port": None,
                "flags":    "",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.000"),
            }

        # ② 在 lock 內一次性更新所有計數器（原子性）
        with self._lock:
            self.packet_count += 1
            self.total_bytes  += len(pkt)
            self.packets.append(pkt)
            self.parsed_records.append(record)

            # 更新統計計數器
            proto = record.get("protocol", "UNKNOWN")
            self.proto_counter[proto] += 1
            if record.get("src_ip"):
                self.src_ip_counter[record["src_ip"]] += 1
            if isinstance(record.get("dst_port"), int):
                self.dst_port_counter[record["dst_port"]] += 1
            bucket = self._size_bucket(len(pkt))
            self.size_buckets[bucket] += 1

        self._display_packet(record)
        self.detector.inspect(pkt, record)

    @staticmethod
    def _size_bucket(size: int) -> str:
        if size < 64:    return "<64B"
        if size < 256:   return "64-255B"
        if size < 512:   return "256-511B"
        if size < 1024:  return "512-1023B"
        if size < 1500:  return "1024-1499B"
        return ">=1500B"

    # ── 即時顯示 ─────────────────────────────────────────
    def _display_packet(self, record):
        proto    = record.get("protocol", "UNK")
        src_ip   = record.get("src_ip",   "N/A")
        dst_ip   = record.get("dst_ip",   "N/A")
        src_port = record.get("src_port", "-")
        dst_port = record.get("dst_port", "-")
        length   = record.get("length",   0)
        flags    = record.get("flags",    "")
        ts       = record.get("timestamp","")

        tls_info = ""
        if proto == "TLS":
            tls_info = (f" [{record.get('tls_version','')} "
                        f"{record.get('tls_handshake') or record.get('tls_type','')}]")

        dns_info = ""
        if proto == "DNS" and record.get("dns_query"):
            dns_info = f" [{record['dns_query']}]"

        color_map = {
            "TCP":    Fore.GREEN,
            "UDP":    Fore.CYAN,
            "ICMP":   Fore.YELLOW,
            "ICMPv6": Fore.LIGHTYELLOW_EX,
            "ARP":    Fore.MAGENTA,
            "DNS":    Fore.BLUE,
            "TLS":    Fore.LIGHTCYAN_EX,
            "DHCP":   Fore.LIGHTBLUE_EX,
        }
        color   = color_map.get(proto, Fore.WHITE)
        svc     = PORT_SERVICE_MAP.get(dst_port, "")
        svc_str = f"({svc})" if svc else ""

        print(
            f"{Fore.WHITE}[{self.packet_count:05d}] "
            f"{Fore.LIGHTBLACK_EX}{ts}  "
            f"{color}{proto:<8}{Style.RESET_ALL}  "
            f"{src_ip}:{src_port}  ->  "
            f"{dst_ip}:{dst_port} {svc_str}  "
            f"{Fore.LIGHTBLACK_EX}{length}B  "
            f"{Fore.YELLOW}{flags}"
            f"{Fore.LIGHTCYAN_EX}{tls_info}{dns_info}"
        )

    # ── 即時統計儀表板 ────────────────────────────────────
    def _stats_loop(self):
        while not self._stop_event.wait(timeout=self.stats_interval):
            self._print_stats_dashboard()

    def _print_stats_dashboard(self):
        with self._lock:
            elapsed    = time.time() - (self.start_time or time.time())
            pkt_count  = self.packet_count
            tot_bytes  = self.total_bytes
            proto_snap = dict(self.proto_counter.most_common(8))
            src_snap   = dict(self.src_ip_counter.most_common(5))
            size_snap  = dict(self.size_buckets)

        pps = pkt_count / max(elapsed, 1)
        bps = tot_bytes / max(elapsed, 1)

        print(f"\n{Fore.CYAN}{'─'*65}")
        print(f"  📊 即時統計  |  時間: {elapsed:.0f}s  "
              f"|  封包: {pkt_count:,}  "
              f"|  速率: {pps:.1f} pps  "
              f"|  流量: {bps/1024:.1f} KB/s")
        print(f"{'─'*65}{Style.RESET_ALL}")

        if proto_snap:
            proto_rows = [[p, c, f"{c/max(pkt_count,1)*100:.1f}%"]
                          for p, c in proto_snap.items()]
            print(tabulate(proto_rows, headers=["協議", "封包數", "佔比"],
                           tablefmt="simple"))

        if src_snap:
            print(f"\n  Top 來源 IP:")
            for ip, cnt in src_snap.items():
                print(f"    {ip:<20} {cnt:>6} 封包")

        alert_count = len(self.detector.alert_history)
        if alert_count:
            print(f"\n  {Fore.RED}⚠  累計告警: {alert_count} 筆{Style.RESET_ALL}")

        print(f"{Fore.CYAN}{'─'*65}{Style.RESET_ALL}\n")

    # ── 開始 / 停止 ──────────────────────────────────────
    def start(self):
        print(f"\n{Fore.GREEN}{'='*65}")
        print(f"  開始即時封包擷取")
        print(f"  介面    : {self.interface}")
        print(f"  過濾器  : {self.bpf_filter or '(無，擷取全部)'}")
        print(f"  數量限制: {self.count or '無限制'}")
        print(f"  統計間隔: {self.stats_interval}s")
        print(f"  異常偵測: ✓ SYN Flood / Port Scan / ICMP Flood / "
              f"UDP Flood / ARP Spoof / DNS Amp")
        print(f"  按 Ctrl+C 停止")
        print(f"{'='*65}{Style.RESET_ALL}\n")

        self.start_time = time.time()

        if self.stats_interval > 0:
            self._stats_thread = threading.Thread(
                target=self._stats_loop, daemon=True)
            self._stats_thread.start()

        try:
            sniff(
                iface=self.interface,
                filter=self.bpf_filter,
                prn=self._packet_callback,
                count=self.count,
                store=False,
                stop_filter=lambda x: self._stop_event.is_set()
            )
        except KeyboardInterrupt:
            pass
        finally:
            self._on_stop()

    def stop(self):
        self._stop_event.set()

    def _on_stop(self):
        self._stop_event.set()
        elapsed = time.time() - self.start_time

        print(f"\n{Fore.CYAN}{'='*65}")
        print(f"  擷取結束")
        print(f"  總封包: {self.packet_count:,}")
        print(f"  總流量: {self.total_bytes:,} Bytes ({self.total_bytes/1024:.1f} KB)")
        print(f"  時間:   {elapsed:.1f} 秒")
        print(f"  速率:   {self.packet_count / max(elapsed, 1):.1f} pps  |  "
              f"{self.total_bytes / max(elapsed, 1) / 1024:.1f} KB/s")
        print(f"{'='*65}{Style.RESET_ALL}\n")

        self._print_stats_dashboard()
        self.detector.print_history()

        if self.parsed_records:
            self.storage.save_csv(self.parsed_records)
            self.storage.save_json(self.parsed_records)
            if self.save_db:
                self.storage.save_sqlite(self.parsed_records)

        if self.save_pcap and self.packets:
            wrpcap(OUTPUT_PCAP, self.packets)
            print(f"{Fore.GREEN}  PCAP 已儲存: {OUTPUT_PCAP}{Style.RESET_ALL}")