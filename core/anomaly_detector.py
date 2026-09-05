# ============================================================
# anomaly_detector.py - 異常偵測模組（v3.0.0 優化版）
#
# 優化清單：
#   [效能] 將 scapy import 移至模組頂層，消除每次 inspect 的查找開銷
#   [修正] 導入時間窗口滑動計數，使用 ANOMALY_WINDOW_SECONDS 設定，
#          避免長時間擷取累計誤報
#   [新增] 告警冷卻機制：同一 IP 的同類型告警在冷卻時間內不重複觸發
#   [新增] DNS Tunneling 偵測：偵測異常長度的 DNS 查詢名稱
#
# 支援偵測類型：
#   1. SYN Flood          - 大量 SYN 封包攻擊
#   2. Port Scan          - 端口掃描（橫向 / 縱向）
#   3. ICMP Flood         - 大量 ICMP 封包攻擊（Ping Flood）
#   4. UDP Flood          - 大量 UDP 封包攻擊
#   5. ARP Spoofing       - ARP 欺騙（同一 IP 出現多個 MAC）
#   6. DNS Amplification  - DNS 放大攻擊（回應遠大於請求）
#   7. DNS Tunneling      - DNS 隧道（異常長查詢名稱）
# ============================================================

import time
from collections import defaultdict, deque
from datetime import datetime
from colorama import Fore, Style, init

# [效能優化] 在模組頂層匯入 scapy 層級，避免每次 inspect() 呼叫時重複查找
from scapy.layers.inet import TCP, UDP, ICMP
from scapy.layers.dns import DNS, DNSQR

from config import (
    ALERT_THRESHOLD_SYN,
    ALERT_THRESHOLD_PORTS,
    ALERT_THRESHOLD_ICMP,
    ALERT_THRESHOLD_UDP,
    ALERT_THRESHOLD_DNS_AMP,
    ALERT_ARP_SPOOF_WINDOW,
    ANOMALY_WINDOW_SECONDS,
)

# 嘗試匯入新增設定，若不存在則使用預設值（向下相容）
try:
    from config import (
        ALERT_COOLDOWN_SECONDS,
        ALERT_THRESHOLD_DNS_TUNNEL_LEN,
        ALERT_THRESHOLD_DNS_TUNNEL_COUNT,
    )
except ImportError:
    ALERT_COOLDOWN_SECONDS = 60
    ALERT_THRESHOLD_DNS_TUNNEL_LEN = 50
    ALERT_THRESHOLD_DNS_TUNNEL_COUNT = 30

init(autoreset=True)


class _SlidingWindowCounter:
    """
    滑動時間窗口計數器

    在指定的時間窗口內計算事件次數，過期的事件自動被清除。
    使用 deque 實作，時間複雜度：
      - 新增事件：O(1) 攤銷
      - 查詢計數：O(k)，k 為過期事件數

    用途：
      取代原本的 defaultdict(int) 累計計數器，避免長時間擷取
      因為正常流量累積而產生誤報。

    範例：
      counter = _SlidingWindowCounter(window_seconds=10)
      counter.add("192.168.1.1")
      print(counter.count("192.168.1.1"))  # 窗口內的事件數
    """

    def __init__(self, window_seconds: float):
        """初始化滑動窗口計數器

        Args:
            window_seconds: 時間窗口大小（秒），只統計此秒數內的事件
        """
        self.window = window_seconds
        # {key: deque([timestamp, ...])} — 每個 key（如 src_ip）各自維護時間戳佇列
        self._data = defaultdict(deque)
        self._last_evict_time = {}

    def add(self, key: str, timestamp: float = None):
        """新增一筆事件

        Args:
            key: 事件鍵值（通常是來源 IP）
            timestamp: 事件時間戳（預設為當前時間）
        """
        now = timestamp or time.time()
        self._data[key].append(now)
        # 清除過期事件（從左端彈出）
        self._evict(key, now)

    def count(self, key: str, timestamp: float = None) -> int:
        """查詢指定 key 在時間窗口內的事件數

        Args:
            key: 事件鍵值
            timestamp: 參考時間（預設為當前時間）

        Returns:
            窗口內的事件計數
        """
        now = timestamp or time.time()
        self._evict(key, now)
        return len(self._data[key])

    def _evict(self, key: str, now: float):
        """清除指定 key 中超出時間窗口的過期事件"""
        # ── [修正] 改用 dict 取代 setattr，避免動態屬性膨脹 ──
        if now - self._last_evict_time.get(key, 0) < 1.0:
            return
        self._last_evict_time[key] = now
        cutoff = now - self.window
        dq = self._data.get(key)
        if dq:
            while dq and dq[0] < cutoff:
                dq.popleft()

    def clear(self):
        """重置所有計數器"""
        self._data.clear()
        self._last_evict_time.clear()


class AnomalyDetector:
    """
    統一異常偵測器（v3.0.0 優化版）

    核心改進：
      - 時間窗口滑動計數：只統計最近 ANOMALY_WINDOW_SECONDS 秒的事件，
        避免長時間擷取因正常流量累積而誤報
      - 告警冷卻機制：同一 IP 的同類型告警在冷卻時間內不重複觸發
      - DNS Tunneling 偵測：偵測異常長度的 DNS 查詢名稱

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
                 threshold_udp=ALERT_THRESHOLD_UDP,
                 window_seconds=ANOMALY_WINDOW_SECONDS,
                 cooldown_seconds=ALERT_COOLDOWN_SECONDS):
        """
        初始化異常偵測器

        Args:
            on_alert        : 告警回調函數 fn(alert_dict)，None 則自動印出
            threshold_syn   : SYN Flood 閾值（窗口內封包數）
            threshold_ports : Port Scan 閾值（不重複 Port 數）
            threshold_icmp  : ICMP Flood 閾值（窗口內封包數）
            threshold_udp   : UDP Flood 閾值（窗口內封包數）
            window_seconds  : 滑動時間窗口大小（秒）
            cooldown_seconds: 告警冷卻時間（秒）
        """
        self.on_alert       = on_alert or self._default_print_alert
        self.thr_syn        = threshold_syn
        self.thr_ports      = threshold_ports
        self.thr_icmp       = threshold_icmp
        self.thr_udp        = threshold_udp
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds

        # [優化] 使用滑動窗口計數器取代累計式 defaultdict(int)
        self.syn_counter  = _SlidingWindowCounter(window_seconds)
        self.icmp_counter = _SlidingWindowCounter(window_seconds)
        self.udp_counter  = _SlidingWindowCounter(window_seconds)

        # Port Scan 仍需記錄不重複 Port 集合
        self.port_scan     = defaultdict(set)

        # ARP / DNS 計數器
        # [Bug 5 修正 — 2026] 舊版註解宣稱 ARP Spoofing 需全程追蹤、不使用
        # 滑動窗口，但 config.py 明明定義了 ALERT_ARP_SPOOF_WINDOW 且從未
        # 被使用（形成 dead import）。改為使用 ALERT_ARP_SPOOF_WINDOW 秒的
        # 滑動窗口：{ip: {mac: last_seen}}，last_seen 超過窗口的 MAC 會被
        # 視為過期並剔除，避免 (a) 字典無限增長 (b) 合法 MAC 變更被永久誤判。
        self.arp_map   = defaultdict(dict)       # {ip: {mac: last_seen}}
        self.dns_req   = defaultdict(int)         # {src_ip: dns_request_count}
        self.dns_resp  = defaultdict(int)         # {dst_ip(受害者): dns_response_count}

        # [新增] DNS Tunneling 計數器
        self.dns_tunnel_counter = _SlidingWindowCounter(window_seconds)

        # [新增] 告警冷卻追蹤：{alert_key: last_alert_time}
        self._alert_cooldown = {}

        # 所有告警歷史記錄
        self.alert_history = []

        # 定期清理設定（避免 port_scan / dns 字典無限增長）
        self._cleanup_interval = window_seconds * 2  # 兩個窗口週期清一次
        self._last_cleanup = time.time()

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
        # ── [修正] 定期清理累計計數器，防止長時間運行誤報 ──
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self._last_cleanup = now
            self.port_scan.clear()
            self.dns_req.clear()
            self.dns_resp.clear()
            # [Bug 5 修正 — 2026] 舊版遺漏了 arp_map 的定期清理，
            # 導致長時間執行時 IP→MAC 對應表無限增長。
            self._cleanup_arp_map(now)

        alerts = []
        alerts += self._check_syn_flood(pkt, record)
        alerts += self._check_port_scan(record)
        alerts += self._check_icmp_flood(pkt, record)
        alerts += self._check_udp_flood(pkt, record)
        alerts += self._check_arp_spoof(record)
        alerts += self._check_dns_amplification(pkt, record)
        alerts += self._check_dns_tunneling(pkt, record)

        for a in alerts:
            self.alert_history.append(a)
            self.on_alert(a)

        return alerts

    # ── 告警冷卻檢查 ──────────────────────────────────────
    def _is_in_cooldown(self, key: str) -> bool:
        """
        檢查指定告警鍵是否仍在冷卻期內

        Args:
            key: 告警鍵值（如 "SYN_FLOOD_10.0.0.1"）

        Returns:
            True 表示仍在冷卻期，不應重複觸發
        """
        last_time = self._alert_cooldown.get(key)
        if last_time is None:
            return False
        return (time.time() - last_time) < self.cooldown_seconds

    def _mark_alerted(self, key: str):
        """記錄告警觸發時間，啟動冷卻計時"""
        self._alert_cooldown[key] = time.time()

    # ── 1. SYN Flood ──────────────────────────────────────
    def _check_syn_flood(self, pkt, record) -> list:
        """偵測 SYN Flood 攻擊

        使用滑動時間窗口計算指定秒數內的 SYN 封包數量，
        超過閾值且不在冷卻期時觸發告警。
        """
        if not pkt.haslayer(TCP):
            return []
        if pkt[TCP].flags != "S":   # 只計純 SYN（無 ACK）
            return []

        src_ip = record.get("src_ip", "")
        now = time.time()
        self.syn_counter.add(src_ip, now)
        current_count = self.syn_counter.count(src_ip, now)

        key = f"SYN_FLOOD_{src_ip}"
        if current_count > self.thr_syn and not self._is_in_cooldown(key):
            self._mark_alerted(key)
            return [self._make_alert(
                attack_type="SYN Flood",
                severity=self.SEVERITY_HIGH,
                src_ip=src_ip,
                detail=(f"SYN 封包數: {current_count} "
                        f"(閾值: {self.thr_syn}，窗口: {self.window_seconds}s)"),
                suggestion="封鎖來源 IP 或啟用 SYN Cookie 防護",
            )]
        return []

    # ── 2. Port Scan ──────────────────────────────────────
    def _check_port_scan(self, record) -> list:
        """偵測端口掃描攻擊

        追蹤每個來源 IP 存取的不重複 Port 數量，
        超過閾值時根據 TCP Flags 分類掃描類型。
        """
        src_ip   = record.get("src_ip", "")
        dst_port = record.get("dst_port")

        if not src_ip or not isinstance(dst_port, int):
            return []

        self.port_scan[src_ip].add(dst_port)
        unique_ports = len(self.port_scan[src_ip])

        key = f"PORT_SCAN_{src_ip}"
        if unique_ports > self.thr_ports and not self._is_in_cooldown(key):
            self._mark_alerted(key)
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
        """偵測 ICMP Flood（Ping Flood）攻擊

        使用滑動時間窗口計算 Echo Request 封包數量。
        """
        if not pkt.haslayer(ICMP):
            return []
        # 只計 Echo Request（type=8）
        if pkt[ICMP].type != 8:
            return []

        src_ip = record.get("src_ip", "")
        now = time.time()
        self.icmp_counter.add(src_ip, now)
        current_count = self.icmp_counter.count(src_ip, now)

        key = f"ICMP_FLOOD_{src_ip}"
        if current_count > self.thr_icmp and not self._is_in_cooldown(key):
            self._mark_alerted(key)
            return [self._make_alert(
                attack_type="ICMP Flood (Ping Flood)",
                severity=self.SEVERITY_MEDIUM,
                src_ip=src_ip,
                detail=(f"ICMP Echo Request 數: {current_count} "
                        f"(閾值: {self.thr_icmp}，窗口: {self.window_seconds}s)"),
                suggestion="在防火牆封鎖 ICMP Echo Request 或限速",
            )]
        return []

    # ── 4. UDP Flood ──────────────────────────────────────
    def _check_udp_flood(self, pkt, record) -> list:
        """偵測 UDP Flood 攻擊

        使用滑動時間窗口計算 UDP 封包數量。
        """
        if not pkt.haslayer(UDP):
            return []

        src_ip = record.get("src_ip", "")
        now = time.time()
        self.udp_counter.add(src_ip, now)
        current_count = self.udp_counter.count(src_ip, now)

        key = f"UDP_FLOOD_{src_ip}"
        if current_count > self.thr_udp and not self._is_in_cooldown(key):
            self._mark_alerted(key)
            return [self._make_alert(
                attack_type="UDP Flood",
                severity=self.SEVERITY_HIGH,
                src_ip=src_ip,
                detail=(f"UDP 封包數: {current_count} "
                        f"(閾值: {self.thr_udp}，窗口: {self.window_seconds}s)"),
                suggestion="封鎖來源 IP；啟用 UDP 速率限制",
            )]
        return []

    # ── 5. ARP Spoofing ───────────────────────────────────
    def _check_arp_spoof(self, record) -> list:
        """偵測 ARP Spoofing（ARP 欺騙）

        同一 IP 在 ALERT_ARP_SPOOF_WINDOW 秒內出現多個 MAC 地址時觸發告警。

        [Bug 5 修正 — 2026]
        舊版註解宣稱「不使用滑動窗口」，但完全沒有任何機制清除
        self.arp_map 內過期的 MAC 記錄，實際造成兩個問題：
          1) IP→MAC 對應表無限增長，長時間執行有記憶體洩漏風險；
          2) 合法的 MAC 變更（DHCP 續約、裝置漫遊、VM 遷移等，
             只要間隔超過設定的時間窗口）理論上不該被視為攻擊，
             但舊 MAC 永遠不會過期，導致該 IP 一旦被標記過，
             之後永久被視為「多重 MAC」而持續誤報。
        修正：比照 _SlidingWindowCounter 的精神，只保留窗口內
        （ALERT_ARP_SPOOF_WINDOW 秒）的 MAC 紀錄，超過窗口的舊紀錄
        會在每次檢查時被剔除，同一 MAC 持續出現則視為活躍並更新
        last_seen（不會被誤判為過期）。
        """
        if record.get("protocol") != "ARP":
            return []

        src_ip  = record.get("arp_src_ip", "")
        src_mac = record.get("arp_src_mac", "")

        if not src_ip or not src_mac:
            return []

        now = time.time()
        ip_macs = self.arp_map[src_ip]

        # 先剔除超出時間窗口、已不活躍的舊 MAC 紀錄
        expired_macs = [
            mac for mac, last_seen in ip_macs.items()
            if now - last_seen > ALERT_ARP_SPOOF_WINDOW
        ]
        for mac in expired_macs:
            del ip_macs[mac]

        is_new_mac = src_mac not in ip_macs
        ip_macs[src_mac] = now   # 記錄/更新最後出現時間

        if is_new_mac and len(ip_macs) >= 2:
            # 窗口內同一 IP 出現第二個（或以上）MAC，觸發告警
            key = f"ARP_SPOOF_{src_ip}"
            if not self._is_in_cooldown(key):
                self._mark_alerted(key)
                macs_str = ", ".join(ip_macs.keys())
                return [self._make_alert(
                    attack_type="ARP Spoofing",
                    severity=self.SEVERITY_CRITICAL,
                    src_ip=src_ip,
                    detail=(f"IP {src_ip} 在 {ALERT_ARP_SPOOF_WINDOW} 秒內"
                            f"對應到多個 MAC 地址:\n  {macs_str}"),
                    suggestion="確認哪個 MAC 為合法主機；啟用動態 ARP 檢測（DAI）",
                )]

        # 若清理後該 IP 已無任何 MAC 紀錄，移除空字典節省記憶體
        if not self.arp_map[src_ip]:
            del self.arp_map[src_ip]

        return []

    # [Bug 5 修正 — 2026 新增] 定期清理 arp_map 中所有 IP 的過期 MAC 紀錄
    def _cleanup_arp_map(self, now: float = None):
        """
        清除所有 IP 對應表中，超過 ALERT_ARP_SPOOF_WINDOW 秒未出現的 MAC 紀錄。

        由 inspect() 的定期清理排程呼叫，確保即使某個 IP 之後不再送出
        任何 ARP 封包，其過期紀錄仍會被回收，避免長時間執行時的記憶體洩漏。
        """
        now = now if now is not None else time.time()
        stale_ips = []
        for ip, macs in self.arp_map.items():
            expired = [m for m, last_seen in macs.items()
                       if now - last_seen > ALERT_ARP_SPOOF_WINDOW]
            for m in expired:
                del macs[m]
            if not macs:
                stale_ips.append(ip)
        for ip in stale_ips:
            del self.arp_map[ip]

    # ── 6. DNS Amplification ──────────────────────────────
    def _check_dns_amplification(self, pkt, record) -> list:
        """
        偵測 DNS 放大攻擊

        攻擊者偽造受害者 IP 發送 ANY 查詢 -> DNS 伺服器回覆大量資料給受害者。
        特徵：某 IP 收到大量 DNS 回應（qr=1）但幾乎沒有發出查詢（qr=0）。
        """
        if not pkt.haslayer(DNS):
            return []

        dns = pkt[DNS]
        src_ip = record.get("src_ip", "")
        dst_ip = record.get("dst_ip", "")

        if dns.qr == 0:   # 查詢
            self.dns_req[src_ip] += 1
            check_ip = src_ip
        else:              # 回應
            # ── [P1-4 修正] DNS 回應統計受害者（dst_ip） ──
            if dst_ip:
                self.dns_resp[dst_ip] += 1
            check_ip = dst_ip or src_ip

        # 從受害者視角檢查：收到大量回應卻幾乎沒發出查詢 → 放大攻擊
        req   = self.dns_req.get(check_ip, 0)
        resp  = self.dns_resp.get(check_ip, 0)
        ratio = resp / max(req, 1)

        key = f"DNS_AMP_{check_ip}"
        if (resp > 20 and ratio > ALERT_THRESHOLD_DNS_AMP
                and not self._is_in_cooldown(key)):
            self._mark_alerted(key)
            return [self._make_alert(
                attack_type="DNS Amplification",
                severity=self.SEVERITY_HIGH,
                src_ip=check_ip,
                detail=(f"DNS 回應數: {resp}，查詢數: {req}，"
                        f"回應/查詢比: {ratio:.1f}x (閾值: {ALERT_THRESHOLD_DNS_AMP}x)"),
                suggestion="在 DNS 伺服器停用 ANY 查詢；限制 DNS 回應速率",
            )]
        return []

    # ── 7. DNS Tunneling（新增）────────────────────────────
    def _check_dns_tunneling(self, pkt, record) -> list:
        """
        偵測 DNS Tunneling（DNS 隧道攻擊）

        攻擊者將資料編碼在 DNS 查詢名稱中（通常是 Base64 或 Hex），
        特徵為異常長的子域名。正常 DNS 查詢名稱通常 < 30 字元，
        DNS Tunneling 的查詢名稱常 > 50 字元。

        偵測邏輯：
          1. 檢查 DNS 查詢名稱長度是否超過閾值
          2. 使用滑動窗口計算同一 IP 的長名稱查詢次數
          3. 超過計數閾值時觸發告警
        """
        if not pkt.haslayer(DNS) or not pkt.haslayer(DNSQR):
            return []

        dns = pkt[DNS]
        # 只檢查查詢封包（qr=0）
        if dns.qr != 0:
            return []

        try:
            qname = pkt[DNSQR].qname.decode(errors="replace").rstrip(".")
        except Exception:
            return []

        # 查詢名稱長度超過閾值才計入
        if len(qname) < ALERT_THRESHOLD_DNS_TUNNEL_LEN:
            return []

        src_ip = record.get("src_ip", "")
        now = time.time()
        self.dns_tunnel_counter.add(src_ip, now)
        current_count = self.dns_tunnel_counter.count(src_ip, now)

        key = f"DNS_TUNNEL_{src_ip}"
        if (current_count > ALERT_THRESHOLD_DNS_TUNNEL_COUNT
                and not self._is_in_cooldown(key)):
            self._mark_alerted(key)
            return [self._make_alert(
                attack_type="DNS Tunneling",
                severity=self.SEVERITY_HIGH,
                src_ip=src_ip,
                detail=(f"異常長 DNS 查詢數: {current_count} "
                        f"(閾值: {ALERT_THRESHOLD_DNS_TUNNEL_COUNT}，"
                        f"名稱長度閾值: {ALERT_THRESHOLD_DNS_TUNNEL_LEN})\n"
                        f"  最近查詢: {qname[:80]}..."),
                suggestion="檢查是否有 DNS 隧道工具（如 iodine, dnscat2）；"
                           "限制 DNS 查詢長度或封鎖可疑子域名",
            )]
        return []

    # ── 重置計數器 ────────────────────────────────────────
    def reset(self):
        """重置所有計數器與冷卻狀態（保留 alert_history）"""
        self.syn_counter.clear()
        self.icmp_counter.clear()
        self.udp_counter.clear()
        self.port_scan.clear()
        self.arp_map.clear()
        self.dns_req.clear()
        self.dns_resp.clear()
        self.dns_tunnel_counter.clear()
        self._alert_cooldown.clear()

    # ── 取得統計摘要 ──────────────────────────────────────
    def get_summary(self) -> dict:
        """回傳各類偵測計數摘要"""
        return {
            "total_alerts":  len(self.alert_history),
            "syn_top_ip":    self._top_n_window(self.syn_counter),
            "icmp_top_ip":   self._top_n_window(self.icmp_counter),
            "udp_top_ip":    self._top_n_window(self.udp_counter),
            "port_scan_ips": {ip: len(ports)
                              for ip, ports in self.port_scan.items()},
            "arp_spoof_ips": {ip: list(macs.keys())
                              for ip, macs in self.arp_map.items()
                              if len(macs) >= 2},
        }

    @staticmethod
    def _top_n(counter: dict, n=5) -> dict:
        """從普通 dict 計數器取得前 N 名"""
        return dict(sorted(counter.items(), key=lambda x: x[1], reverse=True)[:n])

    @staticmethod
    def _top_n_window(window_counter, n=5) -> dict:
        """從滑動窗口計數器取得前 N 名的當前計數"""
        now = time.time()
        counts = {}
        for key in list(window_counter._data.keys()):
            c = window_counter.count(key, now)
            if c > 0:
                counts[key] = c
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n])

    # ── 建立告警 dict ─────────────────────────────────────
    @staticmethod
    def _make_alert(attack_type, severity, src_ip, detail, suggestion="") -> dict:
        """建構標準化告警字典"""
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
        """預設的終端機告警輸出格式"""
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
        """列印所有告警歷史記錄"""
        if not self.alert_history:
            print(f"{Fore.GREEN}  無異常告警記錄{Style.RESET_ALL}")
            return
        print(f"\n{Fore.CYAN}{'='*65}\n  告警歷史 (共 {len(self.alert_history)} 筆)\n{'='*65}{Style.RESET_ALL}")
        for i, a in enumerate(self.alert_history, 1):
            print(f"  [{i:03d}] {a['time']}  [{a['severity']}]  {a['attack_type']}"
                  f"  src={a['src_ip']}")
