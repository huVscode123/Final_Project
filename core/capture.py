# ============================================================
# capture.py - 即時監聽網路介面（NIC）v3.0.0 優化版
#
# 優化清單：
#   [修正] _packet_callback：將封包序號快照傳入 _display_packet，
#          避免在 lock 外讀取 self.packet_count 的競態條件
#   [新增] 記憶體保護：max_memory_packets 限制記憶體中的封包數量，
#          超過時自動丟棄最舊的封包，統計計數器不受影響
#   [新增] PCAP 分段儲存：每 N 個封包或 N 秒自動儲存一次 PCAP，
#          避免長時間擷取因程式異常中斷而遺失所有資料
#   [新增] 擷取效能指標：峰值速率、平均封包大小等
# ============================================================

import os
import time
import threading
from collections import Counter
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

# 嘗試匯入新增設定，若不存在則使用預設值（向下相容）
try:
    from config import MAX_MEMORY_PACKETS, PCAP_SEGMENT_SIZE, PCAP_SEGMENT_SECONDS
except ImportError:
    MAX_MEMORY_PACKETS = 100000
    PCAP_SEGMENT_SIZE = 50000
    PCAP_SEGMENT_SECONDS = 300

init(autoreset=True)


class LiveCapture:
    """
    即時網路封包擷取類別（v3.0.0 優化版）

    功能：
      - 監聽指定網路介面（NIC）
      - 即時解析並顯示封包資訊（含協議顏色）
      - 整合 AnomalyDetector：SYN Flood / Port Scan / ICMP Flood /
        UDP Flood / ARP Spoofing / DNS 放大 / DNS Tunneling
      - 每 stats_interval 秒自動列印即時流量統計
      - 自動儲存為 PCAP、CSV、JSON（可選 SQLite）
      - 記憶體保護：超過上限自動清除最舊封包
      - PCAP 分段儲存：定期自動備份
    """

    def __init__(self,
                 interface=None,
                 bpf_filter="",
                 count=0,
                 save_pcap=True,
                 stats_interval=10,
                 save_db=False,
                 custom_alert_callback=None,
                 session_dir=None,
                 max_memory_packets=MAX_MEMORY_PACKETS):
        """初始化即時擷取器

        Args:
            interface          : 網路介面名稱（None 為自動偵測）
            bpf_filter         : BPF 過濾語法
            count              : 擷取封包數量上限（0 為無限制）
            save_pcap          : 是否儲存 PCAP 檔案
            stats_interval     : 統計報告間隔（秒）
            save_db            : 是否儲存至 SQLite
            custom_alert_callback: 自訂告警回調函數
            session_dir        : Session 資料夾路徑
            max_memory_packets : 記憶體中保留的最大封包數量
        """
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

        # [新增] 記憶體保護設定
        self.max_memory_packets = max_memory_packets

        self.session_dir = session_dir
        self.parser   = PacketParser()
        self.storage  = PacketStorage(session_dir=session_dir)
        self.detector = AnomalyDetector(on_alert=custom_alert_callback)

        # 統計計數器
        self.proto_counter    = Counter()
        self.src_ip_counter   = Counter()
        self.dst_port_counter = Counter()
        self.size_buckets     = Counter()

        # [新增] 效能指標
        self._peak_pps       = 0.0      # 峰值封包速率（packets/second）
        self._last_pps_time  = 0.0      # 上次計算 PPS 的時間
        self._last_pps_count = 0        # 上次計算 PPS 時的封包數
        self._min_pkt_size   = float('inf')   # 最小封包大小
        self._max_pkt_size   = 0        # 最大封包大小

        # [新增] PCAP 分段儲存
        self._segment_counter     = 0   # 當前段的封包計數
        self._segment_index       = 0   # 分段檔案索引
        self._last_segment_time   = 0.0 # 上次分段儲存時間

        self._stop_event   = threading.Event()
        self._stats_thread = None
        self._lock         = threading.Lock()

    # ── 列出所有網路介面 ─────────────────────────────────
    @staticmethod
    def list_interfaces():
        """列出系統上所有可用的網路介面"""
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

        [競態修正] 在 lock 內取得封包序號快照，傳給 _display_packet，
        避免在 lock 外讀取 self.packet_count 導致的序號不一致。

        [記憶體保護] 當記憶體中的封包數超過 max_memory_packets 時，
        自動移除最舊的封包和解析記錄，統計計數器不受影響。
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

        pkt_len = len(pkt)

        # ② 在 lock 內一次性更新所有計數器（原子性）
        with self._lock:
            self.packet_count += 1
            self.total_bytes  += pkt_len
            self.packets.append(pkt)
            self.parsed_records.append(record)

            # [修正] 取得序號快照供 _display_packet 使用
            pkt_seq = self.packet_count

            # 更新統計計數器
            proto = record.get("protocol", "UNKNOWN")
            self.proto_counter[proto] += 1
            if record.get("src_ip"):
                self.src_ip_counter[record["src_ip"]] += 1
            if isinstance(record.get("dst_port"), int):
                self.dst_port_counter[record["dst_port"]] += 1
            bucket = self._size_bucket(pkt_len)
            self.size_buckets[bucket] += 1

            # [新增] 更新效能指標
            if pkt_len < self._min_pkt_size:
                self._min_pkt_size = pkt_len
            if pkt_len > self._max_pkt_size:
                self._max_pkt_size = pkt_len

            # [新增] 記憶體保護：超過上限時移除最舊的封包
            if len(self.packets) > self.max_memory_packets:
                overflow = len(self.packets) - self.max_memory_packets
                del self.packets[:overflow]
                del self.parsed_records[:overflow]

            # [新增] PCAP 分段儲存計數
            self._segment_counter += 1

        # ③ 在 lock 外顯示封包（傳入序號快照）
        self._display_packet(record, pkt_seq)

        # ④ 異常偵測
        self.detector.inspect(pkt, record)

        # ⑤ [新增] 檢查是否需要分段儲存 PCAP
        self._check_segment_save()

    @staticmethod
    def _size_bucket(size: int) -> str:
        """將封包大小分類到預定義的區間"""
        if size < 64:    return "<64B"
        if size < 256:   return "64-255B"
        if size < 512:   return "256-511B"
        if size < 1024:  return "512-1023B"
        if size < 1500:  return "1024-1499B"
        return ">=1500B"

    # ── 即時顯示 ─────────────────────────────────────────
    def _display_packet(self, record, pkt_seq: int):
        """
        在終端機顯示封包資訊（含協議顏色編碼）

        Args:
            record  : 封包解析記錄
            pkt_seq : 封包序號（在 lock 內取得的快照值）
        """
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
            "QUIC":   Fore.LIGHTGREEN_EX,
        }
        color   = color_map.get(proto, Fore.WHITE)
        svc     = PORT_SERVICE_MAP.get(dst_port, "")
        svc_str = f"({svc})" if svc else ""

        # [修正] 使用傳入的 pkt_seq 而非 self.packet_count
        print(
            f"{Fore.WHITE}[{pkt_seq:05d}] "
            f"{Fore.LIGHTBLACK_EX}{ts}  "
            f"{color}{proto:<8}{Style.RESET_ALL}  "
            f"{src_ip}:{src_port}  ->  "
            f"{dst_ip}:{dst_port} {svc_str}  "
            f"{Fore.LIGHTBLACK_EX}{length}B  "
            f"{Fore.YELLOW}{flags}"
            f"{Fore.LIGHTCYAN_EX}{tls_info}{dns_info}"
        )

    # ── PCAP 分段儲存 ─────────────────────────────────────
    def _check_segment_save(self):
        """
        檢查是否需要進行 PCAP 分段儲存。

        觸發條件（滿足任一即觸發）：
          1. 自上次儲存後封包數達到 PCAP_SEGMENT_SIZE
          2. 自上次儲存後經過 PCAP_SEGMENT_SECONDS 秒
        """
        if not self.save_pcap:
            return

        now = time.time()
        should_save = False
        packets_copy = None

        with self._lock:
            # 條件一：封包數達到分段上限
            if self._segment_counter >= PCAP_SEGMENT_SIZE:
                should_save = True
            # 條件二：時間達到分段間隔
            elif (self._last_segment_time > 0 and
                  now - self._last_segment_time >= PCAP_SEGMENT_SECONDS):
                should_save = True

            if should_save and self.packets:
                self._segment_index += 1
                segment_file = self._get_segment_filename()
                packets_copy = list(self.packets)
                self._segment_counter = 0
                self._last_segment_time = now

        if should_save and packets_copy:
            try:
                wrpcap(segment_file, packets_copy)
                print(f"\n{Fore.GREEN}  📁 PCAP 分段已儲存: {segment_file} "
                      f"({len(packets_copy):,} 封包){Style.RESET_ALL}")
            except Exception as e:
                print(f"\n{Fore.RED}  PCAP 分段儲存失敗: {e}{Style.RESET_ALL}")

    def _get_segment_filename(self) -> str:
        """產生分段 PCAP 檔案名稱"""
        base, ext = os.path.splitext(OUTPUT_PCAP)
        return f"{base}_seg{self._segment_index:04d}{ext}"

    # ── 即時統計儀表板 ────────────────────────────────────
    def _stats_loop(self):
        """定期列印統計報告的背景執行緒"""
        while not self._stop_event.wait(timeout=self.stats_interval):
            self._print_stats_dashboard()

    def _print_stats_dashboard(self):
        """列印即時流量統計儀表板"""
        with self._lock:
            elapsed    = time.time() - (self.start_time or time.time())
            pkt_count  = self.packet_count
            tot_bytes  = self.total_bytes
            proto_snap = dict(self.proto_counter.most_common(8))
            src_snap   = dict(self.src_ip_counter.most_common(5))
            mem_pkts   = len(self.packets)

            # [新增] 計算峰值 PPS
            now = time.time()
            if self._last_pps_time > 0:
                dt = now - self._last_pps_time
                if dt > 0:
                    current_pps = (pkt_count - self._last_pps_count) / dt
                    if current_pps > self._peak_pps:
                        self._peak_pps = current_pps
            self._last_pps_time  = now
            self._last_pps_count = pkt_count

            peak_pps = self._peak_pps
            min_size = self._min_pkt_size if self._min_pkt_size != float('inf') else 0
            max_size = self._max_pkt_size
            avg_size = tot_bytes / max(pkt_count, 1)

        pps = pkt_count / max(elapsed, 1)
        bps = tot_bytes / max(elapsed, 1)

        print(f"\n{Fore.CYAN}{'─'*65}")
        print(f"  📊 即時統計  |  時間: {elapsed:.0f}s  "
              f"|  封包: {pkt_count:,}  "
              f"|  速率: {pps:.1f} pps  "
              f"|  流量: {bps/1024:.1f} KB/s")
        print(f"  📈 峰值: {peak_pps:.1f} pps  "
              f"|  封包大小: {min_size}~{max_size}B (平均 {avg_size:.0f}B)  "
              f"|  記憶體封包: {mem_pkts:,}/{self.max_memory_packets:,}")
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
        """開始即時封包擷取"""
        print(f"\n{Fore.GREEN}{'='*65}")
        print(f"  開始即時封包擷取")
        print(f"  介面    : {self.interface}")
        print(f"  過濾器  : {self.bpf_filter or '(無，擷取全部)'}")
        print(f"  數量限制: {self.count or '無限制'}")
        print(f"  統計間隔: {self.stats_interval}s")
        print(f"  記憶體上限: {self.max_memory_packets:,} 封包")
        print(f"  異常偵測: ✓ SYN Flood / Port Scan / ICMP Flood / "
              f"UDP Flood / ARP Spoof / DNS Amp / DNS Tunnel")
        print(f"  按 Ctrl+C 停止")
        print(f"{'='*65}{Style.RESET_ALL}\n")

        self.start_time = time.time()
        self._last_segment_time = self.start_time

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
        """停止擷取"""
        self._stop_event.set()

    def _on_stop(self):
        """擷取結束後的清理與儲存工作"""
        self._stop_event.set()
        elapsed = time.time() - self.start_time

        avg_size = self.total_bytes / max(self.packet_count, 1)

        print(f"\n{Fore.CYAN}{'='*65}")
        print(f"  擷取結束")
        print(f"  總封包: {self.packet_count:,}")
        print(f"  總流量: {self.total_bytes:,} Bytes ({self.total_bytes/1024:.1f} KB)")
        print(f"  時間:   {elapsed:.1f} 秒")
        print(f"  速率:   {self.packet_count / max(elapsed, 1):.1f} pps  |  "
              f"{self.total_bytes / max(elapsed, 1) / 1024:.1f} KB/s")
        print(f"  峰值:   {self._peak_pps:.1f} pps")
        print(f"  封包大小: {self._min_pkt_size if self._min_pkt_size != float('inf') else 0}"
              f"~{self._max_pkt_size}B (平均 {avg_size:.0f}B)")
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