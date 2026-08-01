# ============================================================
# parser.py - 封包欄位解析工具（v2.0.1 修正版）
#
# 修正清單：
#   [Bug 1] _parse_application：TLS 判斷條件的 or/and 優先級錯誤，
#           已加入括號確保 src_port 判斷同樣受到 haslayer 保護。
#   [Bug 9] _parse_tcp：TCP dport=53 現在必須同時有 DNS layer
#           才設定 protocol = "DNS"，避免 SYN/FIN 封包被誤判。
# ============================================================

from datetime import datetime
from scapy.all import Packet
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.inet6 import IPv6, ICMPv6EchoRequest, ICMPv6EchoReply
from scapy.layers.l2 import Ether, ARP
from scapy.layers.dns import DNS, DNSQR, DNSRR
from config import (
    PROTOCOL_MAP, PORT_SERVICE_MAP, MAX_PAYLOAD_DISPLAY,
    ICMP_TYPE_MAP, ICMP_UNREACH_CODE_MAP,
    TLS_VERSION_MAP, TLS_CONTENT_TYPE_MAP, TLS_HANDSHAKE_TYPE_MAP,
)


class PacketParser:
    """
    封包解析器（v2.0.1 修正版）
    輸入：Scapy Packet 物件
    輸出：標準化 dict，包含所有關鍵欄位
    """

    def parse(self, pkt: Packet) -> dict:
        record = {
            "timestamp":     self._get_timestamp(pkt),
            "length":        len(pkt),
            "src_mac":       None, "dst_mac":      None, "ethertype":    None,
            "src_ip":        None, "dst_ip":       None, "ip_version":   None,
            "protocol":      "UNKNOWN", "protocol_num": None,
            "ttl":           None, "ip_flags":     None, "ip_fragment":  None,
            "src_port":      None, "dst_port":     None, "service":      None,
            "flags":         None, "seq":          None, "ack":          None,
            "window":        None, "checksum":     None,
            "ip_checksum":   None, "transport_checksum": None,
            "icmp_type":     None, "icmp_code":    None, "icmp_desc":    None,
            "arp_op":        None, "arp_src_ip":   None, "arp_dst_ip":   None,
            "arp_src_mac":   None,
            "dns_query":     None, "dns_response": None,
            "tls_version":   None, "tls_type":     None, "tls_handshake": None,
            "payload_hex":   None, "payload_text": None, "payload_len":  None,
        }

        self._parse_ethernet(pkt, record)

        if pkt.haslayer(IP):
            self._parse_ipv4(pkt, record)
        elif pkt.haslayer(IPv6):
            self._parse_ipv6(pkt, record)

        self._parse_transport(pkt, record)
        self._parse_application(pkt, record)
        self._parse_payload(pkt, record)

        if record["dst_port"]:
            record["service"] = PORT_SERVICE_MAP.get(record["dst_port"])

        return record

    # ── 第二層：Ethernet / ARP ───────────────────────────
    def _parse_ethernet(self, pkt, record):
        if pkt.haslayer(Ether):
            eth = pkt[Ether]
            record["src_mac"]   = eth.src
            record["dst_mac"]   = eth.dst
            record["ethertype"] = hex(eth.type)

        if pkt.haslayer(ARP):
            arp = pkt[ARP]
            record["protocol"]   = "ARP"
            record["arp_op"]     = {1: "Request", 2: "Reply"}.get(arp.op, str(arp.op))
            record["arp_src_ip"] = arp.psrc
            record["arp_dst_ip"] = arp.pdst
            record["arp_src_mac"]= arp.hwsrc
            record["src_ip"]     = arp.psrc
            record["dst_ip"]     = arp.pdst

    # ── 第三層：IPv4 ─────────────────────────────────────
    def _parse_ipv4(self, pkt, record):
        ip = pkt[IP]
        record["ip_version"]   = 4
        record["src_ip"]       = ip.src
        record["dst_ip"]       = ip.dst
        record["ttl"]          = ip.ttl
        record["protocol_num"] = ip.proto
        record["protocol"]     = PROTOCOL_MAP.get(ip.proto, str(ip.proto))
        record["ip_flags"]     = str(ip.flags)
        record["ip_fragment"]  = ip.frag
        record["ip_checksum"]  = hex(ip.chksum) if ip.chksum is not None else None

    # ── 第三層：IPv6 ─────────────────────────────────────
    def _parse_ipv6(self, pkt, record):
        ip6 = pkt[IPv6]
        record["ip_version"]   = 6
        record["src_ip"]       = ip6.src
        record["dst_ip"]       = ip6.dst
        record["ttl"]          = ip6.hlim
        record["protocol_num"] = ip6.nh
        record["protocol"]     = PROTOCOL_MAP.get(ip6.nh, str(ip6.nh))

        if pkt.haslayer(ICMPv6EchoRequest):
            record["protocol"]  = "ICMPv6"
            record["icmp_type"] = 128
            record["icmp_desc"] = "Echo Request"
            record["flags"]     = "Echo Request"
        elif pkt.haslayer(ICMPv6EchoReply):
            record["protocol"]  = "ICMPv6"
            record["icmp_type"] = 129
            record["icmp_desc"] = "Echo Reply"
            record["flags"]     = "Echo Reply"

    # ── 第四層：TCP / UDP / ICMP ─────────────────────────
    def _parse_transport(self, pkt, record):
        if pkt.haslayer(TCP):
            self._parse_tcp(pkt, record)
        elif pkt.haslayer(UDP):
            self._parse_udp(pkt, record)
        elif pkt.haslayer(ICMP):
            self._parse_icmp(pkt, record)

    def _parse_tcp(self, pkt, record):
        tcp = pkt[TCP]
        record["src_port"] = tcp.sport
        record["dst_port"] = tcp.dport
        record["seq"]      = tcp.seq
        record["ack"]      = tcp.ack
        record["window"]   = tcp.window
        record["transport_checksum"] = hex(tcp.chksum) if tcp.chksum is not None else None
        record["checksum"] = record["transport_checksum"]
        record["flags"]    = self._parse_tcp_flags(tcp.flags)

        # [Bug 9 修正] 必須同時有 DNS layer 才判定為 DNS，
        # 避免 SYN/FIN/RST 等控制封包因 port=53 被誤標為 DNS。
        if tcp.dport in (53, 5353) and pkt.haslayer(DNS):
            record["protocol"] = "DNS"
        elif tcp.sport in (53, 5353) and pkt.haslayer(DNS):
            record["protocol"] = "DNS"

    def _parse_udp(self, pkt, record):
        udp = pkt[UDP]
        record["src_port"] = udp.sport
        record["dst_port"] = udp.dport
        record["transport_checksum"] = hex(udp.chksum) if udp.chksum is not None else None
        record["checksum"] = record["transport_checksum"]

        # [修正] dport 也必須同時有 DNS layer 才判定為 DNS，
        # 避免普通 UDP 封包因 port=53 被誤標為 DNS。
        # 與 _parse_tcp() 的 Bug 9 修正保持一致。
        if udp.dport in (53, 5353) and pkt.haslayer(DNS):
            record["protocol"] = "DNS"
        elif udp.sport in (53, 5353) and pkt.haslayer(DNS):
            record["protocol"] = "DNS"
        elif udp.dport in (67, 68):
            record["protocol"] = "DHCP"
        elif udp.dport == 5355:
            record["protocol"] = "LLMNR"
        elif udp.dport == 1900:
            record["protocol"] = "SSDP"
        elif udp.dport == 123:
            record["protocol"] = "NTP"
        elif udp.dport == 443 or udp.sport == 443:
            # QUIC / HTTP3 使用 UDP 443 埠
            record["protocol"] = "QUIC"

    def _parse_icmp(self, pkt, record):
        icmp = pkt[ICMP]
        itype = icmp.type
        icode = icmp.code
        record["icmp_type"] = itype
        record["icmp_code"] = icode

        type_str = ICMP_TYPE_MAP.get(itype, f"Type {itype}")

        if itype == 3:
            code_str = ICMP_UNREACH_CODE_MAP.get(icode, f"Code {icode}")
            record["icmp_desc"] = f"{type_str} ({code_str})"
        elif itype == 11:
            code_str = "TTL Exceeded" if icode == 0 else "Fragment Reassembly"
            record["icmp_desc"] = f"{type_str} ({code_str})"
        else:
            record["icmp_desc"] = type_str

        record["flags"] = record["icmp_desc"]

    # ── 應用層：DNS / TLS ─────────────────────────────────
    def _parse_application(self, pkt, record):
        if pkt.haslayer(DNS):
            self._parse_dns(pkt, record)

        # [Bug 1 修正] 原本 or 的右側（src_port 判斷）缺少 haslayer 保護，
        # 導致非 TCP 封包也可能進入 _parse_tls。
        # 修正：將 dst_port / src_port 判斷同時括在 haslayer 條件內。
        TLS_PORTS = (443, 8443, 993, 995, 465, 587)
        if (pkt.haslayer(TCP) and pkt.haslayer("Raw")
                and (record.get("dst_port") in TLS_PORTS
                     or record.get("src_port") in TLS_PORTS)):
            self._parse_tls(pkt, record)

    def _parse_dns(self, pkt, record):
        dns = pkt[DNS]
        if pkt.haslayer(DNSQR):
            try:
                record["dns_query"] = pkt[DNSQR].qname.decode(
                    errors="replace").rstrip(".")
            except Exception:
                pass
        if dns.qr == 1 and pkt.haslayer(DNSRR):
            try:
                rr = pkt[DNSRR]
                record["dns_response"] = (
                    f"{rr.rrname.decode(errors='replace').rstrip('.')} -> {rr.rdata}"
                )
            except Exception:
                pass

    def _parse_tls(self, pkt, record):
        try:
            raw = bytes(pkt["Raw"].load)
            if len(raw) < 5:
                return

            content_type  = raw[0]
            version_bytes = raw[1:3]

            ct_str  = TLS_CONTENT_TYPE_MAP.get(content_type)
            ver_str = TLS_VERSION_MAP.get(version_bytes)

            if ct_str is None or ver_str is None:
                return

            record["protocol"]    = "TLS"
            record["tls_type"]    = ct_str
            record["tls_version"] = ver_str

            if content_type == 22 and len(raw) >= 6:
                hs_type = raw[5]
                record["tls_handshake"] = TLS_HANDSHAKE_TYPE_MAP.get(
                    hs_type, f"Type {hs_type}"
                )
        except Exception:
            pass

    # ── Payload ───────────────────────────────────────────
    def _parse_payload(self, pkt, record):
        if not pkt.haslayer("Raw"):
            return
        raw = bytes(pkt["Raw"].load)
        if not raw:
            return
        record["payload_len"]  = len(raw)
        display = raw[:MAX_PAYLOAD_DISPLAY]
        record["payload_hex"]  = display.hex()
        record["payload_text"] = "".join(
            chr(b) if 32 <= b < 127 else "." for b in display
        )

    # ── 靜態工具 ──────────────────────────────────────────
    @staticmethod
    def _parse_tcp_flags(flags) -> str:
        """解析 TCP 旗標位元，包含 NS (Nonce Sum) 支援"""
        flag_map = {
            0x001: "FIN", 0x002: "SYN", 0x004: "RST", 0x008: "PSH",
            0x010: "ACK", 0x020: "URG", 0x040: "ECE", 0x080: "CWR",
            0x100: "NS",
        }
        active = [name for bit, name in flag_map.items() if int(flags) & bit]
        return "+".join(active) if active else "NONE"

    @staticmethod
    def _get_timestamp(pkt) -> str:
        try:
            return datetime.fromtimestamp(float(pkt.time)).strftime(
                "%Y-%m-%d %H:%M:%S.%f")[:-3]
        except Exception:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S.000")