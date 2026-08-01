# ============================================================
# pcap_analyzer.py - PCAP 離線分析模組（v2.0.1 修正版）
#
# 修正清單：
#   [Bug 2] detect_attacks()：原版回傳 detector.get_summary()
#           （dict），但 analyzer/views.py 的 upload_pcap 把
#           回傳值當作告警列表（list[dict]）迭代，導致
#           AttributeError: 'str' object has no attribute 'get'。
#
#           修正：detect_attacks() 改為回傳
#               {"alerts": list[dict], "summary": dict}
#           的複合 dict，views.py 透過 result["alerts"] 取得
#           可迭代的告警列表，summary 統計同時保留。
#
#           對應 views.py 的使用方式：
#               result    = analyzer.detect_attacks()
#               alert_list= result.get("alerts", [])
#               summary   = result.get("summary", {})
#               for a in alert_list:
#                   Alert.objects.create(
#                       attack_type = a.get("attack_type", ""),
#                       ...
#                   )
# ============================================================

import os
from collections import Counter, defaultdict
from datetime import datetime

from scapy.all import rdpcap, PcapReader
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.dns import DNS, DNSQR
from tabulate import tabulate
from colorama import Fore, Style

from config import PORT_SERVICE_MAP
from parser import PacketParser
from storage import PacketStorage
from anomaly_detector import AnomalyDetector


class PcapAnalyzer:
    """
    PCAP 離線分析類別（v2.0.1 修正版）
    """

    def __init__(self, pcap_path: str, save_db=False, session_dir=None):
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"找不到 PCAP 檔案: {pcap_path}")
        self.pcap_path   = pcap_path
        self.save_db     = save_db
        self.session_dir = session_dir
        self.parser      = PacketParser()
        self.storage     = PacketStorage(session_dir=session_dir)
        self.detector    = AnomalyDetector()
        self.records     = []
        self.packets     = []
        self.total_bytes = 0   # [新增] 快取總流量，避免重複計算

    # ── 讀取 PCAP ────────────────────────────────────────
    def load(self, use_streaming=False):
        """載入 PCAP 檔案

        Args:
            use_streaming: 是否使用串流模式逐封包讀取（適用於大型 PCAP）

        Returns:
            self（支援鏈式呼叫）
        """
        print(f"\n{Fore.CYAN}載入 PCAP: {self.pcap_path}{Style.RESET_ALL}")

        if use_streaming:
            with PcapReader(self.pcap_path) as reader:
                for pkt in reader:
                    self.packets.append(pkt)
                    self.records.append(self.parser.parse(pkt))
        else:
            self.packets = rdpcap(self.pcap_path)
            self.records = [self.parser.parse(p) for p in self.packets]

        # [優化] 快取 total_bytes，避免後續 summary() 重複計算
        self.total_bytes = sum(r.get("length", 0) for r in self.records)
        print(f"{Fore.GREEN}  共載入 {len(self.packets):,} 個封包  "
              f"({self.total_bytes/1024:.1f} KB){Style.RESET_ALL}")
        return self

    # ── 基本統計摘要 ──────────────────────────────────────
    def summary(self):
        self._require_loaded()

        protocols  = Counter(r["protocol"] for r in self.records)
        src_ips    = Counter(r["src_ip"] for r in self.records if r["src_ip"])
        dst_ports  = Counter(
            r["dst_port"] for r in self.records
            if isinstance(r.get("dst_port"), int)
        )
        # [優化] 使用 load() 階段已快取的 total_bytes，避免重複迭代
        total_bytes  = self.total_bytes
        timestamps   = [r["timestamp"] for r in self.records if r.get("timestamp")]
        ip_versions  = Counter(r.get("ip_version") for r in self.records
                               if r.get("ip_version"))

        size_buckets = Counter()
        for r in self.records:
            size = r.get("length", 0)
            if size < 64:    size_buckets["<64B"] += 1
            elif size < 256: size_buckets["64-255B"] += 1
            elif size < 512: size_buckets["256-511B"] += 1
            elif size < 1024:size_buckets["512-1023B"] += 1
            elif size < 1500:size_buckets["1024-1499B"] += 1
            else:            size_buckets[">=1500B"] += 1

        print(f"\n{Fore.CYAN}{'='*65}\n  PCAP 分析摘要\n{'='*65}{Style.RESET_ALL}")
        print(f"  封包總數  : {len(self.records):,}")
        print(f"  總流量    : {total_bytes:,} Bytes ({total_bytes/1024:.1f} KB)")
        if timestamps:
            print(f"  開始時間  : {timestamps[0]}")
            print(f"  結束時間  : {timestamps[-1]}")
        print(f"  IP 版本   : " +
              " / ".join(f"IPv{v}: {c}" for v, c in ip_versions.items()))

        print(f"\n{Fore.YELLOW}  [協議分布 Top 10]{Style.RESET_ALL}")
        print(tabulate(
            [[p, c, f"{c/len(self.records)*100:.1f}%"]
             for p, c in protocols.most_common(10)],
            headers=["協議", "封包數", "佔比"],
            tablefmt="rounded_outline"
        ))

        print(f"\n{Fore.YELLOW}  [Top 10 來源 IP]{Style.RESET_ALL}")
        print(tabulate(
            list(src_ips.most_common(10)),
            headers=["來源 IP", "封包數"],
            tablefmt="rounded_outline"
        ))

        print(f"\n{Fore.YELLOW}  [Top 10 目標 Port]{Style.RESET_ALL}")
        print(tabulate(
            [[p, PORT_SERVICE_MAP.get(p, "Unknown"), c]
             for p, c in dst_ports.most_common(10)],
            headers=["Port", "服務", "封包數"],
            tablefmt="rounded_outline"
        ))

        print(f"\n{Fore.YELLOW}  [封包大小分布]{Style.RESET_ALL}")
        for bucket in ["<64B","64-255B","256-511B","512-1023B","1024-1499B",">=1500B"]:
            cnt = size_buckets.get(bucket, 0)
            if cnt:
                bar = "█" * int(cnt / max(size_buckets.values()) * 30)
                print(f"  {bucket:<12}  {bar:<30}  {cnt}")

        return self

    # ── 提取 DNS 查詢 ─────────────────────────────────────
    def extract_dns(self):
        """提取並分析所有 DNS 查詢與回應記錄"""
        self._require_loaded()
        dns_records = []
        for pkt in self.packets:
            if not pkt.haslayer(DNS):
                continue
            dns    = pkt[DNS]
            src_ip = pkt[IP].src if pkt.haslayer(IP) else "N/A"
            ts     = self.parser._get_timestamp(pkt)

            if pkt.haslayer(DNSQR) and dns.qr == 0:
                dns_records.append({
                    "type":      "Query",
                    "timestamp": ts,
                    "src_ip":    src_ip,
                    "name":      pkt[DNSQR].qname.decode(errors="replace").rstrip("."),
                    "qtype":     pkt[DNSQR].qtype,
                })
            elif dns.qr == 1:
                from scapy.layers.dns import DNSRR
                if pkt.haslayer(DNSRR):
                    try:
                        rr = pkt[DNSRR]
                        dns_records.append({
                            "type":      "Response",
                            "timestamp": ts,
                            "src_ip":    src_ip,
                            "name":      rr.rrname.decode(errors="replace").rstrip("."),
                            "rdata":     str(rr.rdata),
                        })
                    except Exception:
                        pass

        if dns_records:
            queries   = [r for r in dns_records if r["type"] == "Query"]
            responses = [r for r in dns_records if r["type"] == "Response"]
            print(f"\n{Fore.YELLOW}  [DNS 記錄] 查詢 {len(queries)} 筆 / "
                  f"回應 {len(responses)} 筆{Style.RESET_ALL}")
            if queries:
                print(tabulate(
                    [[q["timestamp"], q["src_ip"], q["name"]]
                     for q in queries[:20]],
                    headers=["時間", "來源 IP", "查詢域名"],
                    tablefmt="rounded_outline"
                ))
        return dns_records

    # ── 提取 HTTP 請求 ────────────────────────────────────
    def extract_http(self):
        """提取並分析所有 HTTP 請求"""
        self._require_loaded()
        http_requests = []
        for pkt in self.packets:
            if not (pkt.haslayer(TCP) and pkt[TCP].dport == 80
                    and pkt.haslayer("Raw")):
                continue
            payload = bytes(pkt["Raw"].load)
            if not payload.startswith((b"GET", b"POST", b"HEAD", b"PUT",
                                       b"DELETE", b"OPTIONS", b"PATCH")):
                continue
            lines = payload.decode(errors="replace").split("\r\n")
            host  = next(
                (l.split(": ", 1)[1] for l in lines if l.startswith("Host:")),
                "Unknown"
            )
            src_ip = pkt[IP].src if pkt.haslayer(IP) else "N/A"
            http_requests.append({
                "src_ip":    src_ip,
                "host":      host,
                "method":    lines[0][:80],
                "timestamp": self.parser._get_timestamp(pkt),
            })

        if http_requests:
            print(f"\n{Fore.YELLOW}  [HTTP 請求] 共 {len(http_requests)} 筆{Style.RESET_ALL}")
            print(tabulate(
                [[r["timestamp"], r["src_ip"], r["host"], r["method"]]
                 for r in http_requests[:20]],
                headers=["時間", "來源 IP", "Host", "請求"],
                tablefmt="rounded_outline"
            ))
        return http_requests

    # ── TLS/SSL 分析 ──────────────────────────────────────
    def analyze_tls(self):
        """分析所有 TLS/SSL 封包"""
        self._require_loaded()
        tls_records = [r for r in self.records if r.get("protocol") == "TLS"]
        if not tls_records:
            print(f"\n{Fore.YELLOW}  [TLS] 未找到 TLS 封包{Style.RESET_ALL}")
            return []

        version_counter = Counter(r.get("tls_version") for r in tls_records
                                  if r.get("tls_version"))
        hs_counter      = Counter(r.get("tls_handshake") for r in tls_records
                                  if r.get("tls_handshake"))
        type_counter    = Counter(r.get("tls_type") for r in tls_records
                                  if r.get("tls_type"))
        src_ips         = Counter(r.get("src_ip") for r in tls_records
                                  if r.get("src_ip"))

        print(f"\n{Fore.YELLOW}  [TLS/SSL 分析] 共 {len(tls_records)} 個 TLS 封包{Style.RESET_ALL}")
        if version_counter:
            print(tabulate(list(version_counter.most_common()),
                           headers=["版本", "封包數"], tablefmt="rounded_outline"))
        if hs_counter:
            print(tabulate(list(hs_counter.most_common()),
                           headers=["Handshake 類型", "封包數"], tablefmt="rounded_outline"))
        return tls_records

    # ── ARP 分析 ──────────────────────────────────────────
    def analyze_arp(self):
        """分析所有 ARP 封包並偵測疑似 ARP Spoofing"""
        self._require_loaded()
        arp_records = [r for r in self.records if r.get("protocol") == "ARP"]
        if not arp_records:
            print(f"\n{Fore.YELLOW}  [ARP] 未找到 ARP 封包{Style.RESET_ALL}")
            return {}

        op_counter = Counter(r.get("arp_op") for r in arp_records)
        ip_mac_map = defaultdict(set)
        for r in arp_records:
            ip  = r.get("arp_src_ip", "")
            mac = r.get("arp_src_mac", "")
            if ip and mac:
                ip_mac_map[ip].add(mac)

        spoof_candidates = {ip: macs for ip, macs in ip_mac_map.items()
                            if len(macs) >= 2}

        print(f"\n{Fore.YELLOW}  [ARP 分析] 共 {len(arp_records)} 個 ARP 封包{Style.RESET_ALL}")
        print(tabulate([[op, cnt] for op, cnt in op_counter.most_common()],
                       headers=["操作", "封包數"], tablefmt="rounded_outline"))

        if spoof_candidates:
            print(f"\n{Fore.RED}  ⚠  偵測到 ARP Spoofing 疑似目標:{Style.RESET_ALL}")
            for ip, macs in spoof_candidates.items():
                print(f"    IP: {ip}  ->  MAC: {', '.join(macs)}")

        return ip_mac_map

    # ── 時間軸分析 ────────────────────────────────────────
    def analyze_timeline(self, granularity="second"):
        """按時間軸分析封包分布

        Args:
            granularity: 時間粒度，'second' 或 'minute'
        """
        self._require_loaded()
        time_counter = Counter()
        for r in self.records:
            ts = r.get("timestamp", "")
            if not ts:
                continue
            try:
                key = ts[:19] if granularity == "second" else ts[:16]
                time_counter[key] += 1
            except Exception:
                pass

        if not time_counter:
            return {}

        sorted_times = sorted(time_counter.items())
        max_count    = max(v for _, v in sorted_times)

        print(f"\n{Fore.YELLOW}  [時間軸分析] ({granularity}){Style.RESET_ALL}")
        display = sorted_times[-30:] if len(sorted_times) > 30 else sorted_times
        for ts, cnt in display:
            bar = "█" * int(cnt / max_count * 40)
            print(f"  {ts}  {bar:<40}  {cnt:>5}")

        return dict(sorted_times)

    # ── 離線異常偵測 ──────────────────────────────────────
    def detect_attacks(self) -> dict:
        """
        對已載入的 PCAP 進行完整攻擊特徵偵測。

        [Bug 2 修正] 原版直接回傳 detector.get_summary()（dict）。
        views.py 的 upload_pcap 把回傳值當成告警列表（list[dict]）
        迭代，導致「for a in alerts_raw」只迭代 dict 的鍵（字串），
        進而觸發 AttributeError: 'str' has no attribute 'get'。

        修正後回傳結構：
            {
                "alerts":  list[dict],   ← 可直接迭代的告警列表
                "summary": dict          ← 原有的統計摘要
            }

        views.py 對應修改（只需一行）：
            result     = analyzer.detect_attacks()
            alert_list = result.get("alerts", [])
            for a in alert_list:
                Alert.objects.create(attack_type=a.get("attack_type", ""), ...)
        """
        print(f"\n{Fore.CYAN}  開始離線攻擊偵測...{Style.RESET_ALL}")
        self.detector.reset()

        for pkt, record in zip(self.packets, self.records):
            self.detector.inspect(pkt, record)

        print(f"\n  偵測完成")
        self.detector.print_history()

        summary     = self.detector.get_summary()
        alert_list  = list(self.detector.alert_history)  # list[dict]

        # [修正] 將 total_packets 加入摘要，供 tasks.py 正確取得封包數量
        # 原本 tasks.py 呼叫 summary_info.get('total_packets') 會得到 None，
        # 因為 detector.get_summary() 只包含 total_alerts 而無 total_packets。
        summary["total_packets"] = len(self.records)
        summary["total_bytes"]   = self.total_bytes

        if summary.get("total_alerts", 0) == 0:
            print(f"{Fore.GREEN}  ✓ 未發現攻擊特徵{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}  ⚠  發現 {summary['total_alerts']} 個告警{Style.RESET_ALL}")

        # 回傳複合 dict，告警列表與摘要分開，避免 views.py 迭代 dict 鍵
        return {
            "alerts":  alert_list,
            "summary": summary,
        }

    # ── 重建 TCP 連線流 ───────────────────────────────────
    def rebuild_tcp_streams(self):
        """
        重建 TCP 連線流。
        [Bug 5 修正] 原版使用 (src, sport, dst, dport) 作為 key，
        導致雙向流量被拆分為兩條流。現在將 (src_ip, sport) 與 (dst_ip, dport)
        排序後組合成 key，以合併雙向流。
        """
        streams = defaultdict(list)
        for pkt in self.packets:
            if pkt.haslayer(TCP) and pkt.haslayer(IP):
                addr1 = (pkt[IP].src, pkt[TCP].sport)
                addr2 = (pkt[IP].dst, pkt[TCP].dport)
                # 排序以確保 (A, B) 與 (B, A) 產生相同的 key
                key = tuple(sorted([addr1, addr2]))
                streams[key].append(pkt)

        print(f"\n{Fore.YELLOW}  [TCP 連線流] 共 {len(streams)} 條{Style.RESET_ALL}")
        table = []
        # 注意：key 現在是 ((ip1, port1), (ip2, port2))
        for (addr_a, addr_b), pkts in list(streams.items())[:15]:
            total_bytes = sum(len(p) for p in pkts)
            # 嘗試找出服務（通常在 dst_port，但現在是雙向，檢查兩端）
            svc = PORT_SERVICE_MAP.get(addr_a[1]) or PORT_SERVICE_MAP.get(addr_b[1]) or ""
            table.append([
                f"{addr_a[0]}:{addr_a[1]}",
                f"{addr_b[0]}:{addr_b[1]}",
                svc, len(pkts),
                f"{total_bytes:,} B"
            ])
        print(tabulate(table,
                       headers=["端點 A", "端點 B", "服務", "封包數", "流量"],
                       tablefmt="rounded_outline"))
        return streams

    # ── 連線五元組統計 ──────────────────────────────────────
    def analyze_connections(self):
        """分析連線五元組（src_ip, src_port, dst_ip, dst_port, protocol）

        統計每個唯一連線的封包數與總流量，用於識別
        高流量連線和可疑的持續性連線。

        Returns:
            list[dict]: 依封包數降序排列的連線統計
        """
        self._require_loaded()

        connections = defaultdict(lambda: {
            "count": 0, "bytes": 0,
            "first_seen": None, "last_seen": None
        })

        for r in self.records:
            src_ip   = r.get("src_ip", "N/A")
            dst_ip   = r.get("dst_ip", "N/A")
            src_port = r.get("src_port", 0)
            dst_port = r.get("dst_port", 0)
            proto    = r.get("protocol", "UNKNOWN")
            ts       = r.get("timestamp", "")

            # 使用排序後的端點組合，確保雙向流量合併
            ep1 = (src_ip, src_port)
            ep2 = (dst_ip, dst_port)
            key = (tuple(sorted([ep1, ep2])), proto)

            conn = connections[key]
            conn["count"] += 1
            conn["bytes"] += r.get("length", 0)
            if conn["first_seen"] is None or ts < conn["first_seen"]:
                conn["first_seen"] = ts
            if conn["last_seen"] is None or ts > conn["last_seen"]:
                conn["last_seen"] = ts

        # 轉換為列表並排序
        result = []
        for (endpoints, proto), stats in connections.items():
            (ip1, port1), (ip2, port2) = endpoints
            svc = PORT_SERVICE_MAP.get(port1) or PORT_SERVICE_MAP.get(port2) or ""
            result.append({
                "endpoint_a": f"{ip1}:{port1}",
                "endpoint_b": f"{ip2}:{port2}",
                "protocol":   proto,
                "service":    svc,
                "packets":    stats["count"],
                "bytes":      stats["bytes"],
                "first_seen": stats["first_seen"],
                "last_seen":  stats["last_seen"],
            })

        result.sort(key=lambda x: x["packets"], reverse=True)

        # 列印前 15 名
        print(f"\n{Fore.YELLOW}  [連線五元組分析] 共 {len(result)} 條連線{Style.RESET_ALL}")
        if result:
            table = [
                [c["endpoint_a"], c["endpoint_b"], c["protocol"],
                 c["service"], c["packets"], f"{c['bytes']:,} B"]
                for c in result[:15]
            ]
            print(tabulate(table,
                           headers=["端點 A", "端點 B", "協議", "服務", "封包數", "流量"],
                           tablefmt="rounded_outline"))

        return result

    # ── 儲存結果 ──────────────────────────────────────────
    def save_results(self):
        if self.records:
            self.storage.save_csv(self.records, filename="pcap_analysis.csv")
            self.storage.save_json(self.records, filename="pcap_analysis.json")
            if self.save_db:
                self.storage.save_sqlite(self.records)
        return self

    # ── 一鍵完整分析 ──────────────────────────────────────
    def full_analysis(self):
        """一鍵執行完整 PCAP 分析流程"""
        self.load()
        self.summary()
        self.extract_dns()
        self.extract_http()
        self.analyze_tls()
        self.analyze_arp()
        self.analyze_timeline()
        self.detect_attacks()
        self.rebuild_tcp_streams()
        self.analyze_connections()
        self.save_results()
        return self

    # ── 輔助 ──────────────────────────────────────────────
    def _require_loaded(self):
        if not self.records:
            raise RuntimeError("請先執行 load() 載入 PCAP 檔案")