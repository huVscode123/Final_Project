# ============================================================
# core/tests/test_parser.py - PacketParser 單元測試（v2.0.0）
#
# 執行方式:
#   pytest core/tests/test_parser.py -v
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.inet6 import IPv6, ICMPv6EchoRequest, ICMPv6EchoReply
from scapy.layers.l2 import Ether, ARP
from scapy.layers.dns import DNS, DNSQR, DNSRR

from parser import PacketParser


@pytest.fixture
def parser():
    return PacketParser()


# ────────────────────────────────────────────────────────────
# TCP 封包解析
# ────────────────────────────────────────────────────────────
class TestTCPPacket:

    def test_basic_fields(self, parser):
        """TCP 封包基本欄位解析"""
        pkt = (Ether() / IP(src="192.168.1.100", dst="10.0.0.1", ttl=64)
               / TCP(sport=54321, dport=80, flags="S"))
        r = parser.parse(pkt)
        assert r["src_ip"]    == "192.168.1.100"
        assert r["dst_ip"]    == "10.0.0.1"
        assert r["src_port"]  == 54321
        assert r["dst_port"]  == 80
        assert r["protocol"]  == "TCP"
        assert r["ttl"]       == 64
        assert "SYN" in r["flags"]

    def test_syn_ack_flags(self, parser):
        """SYN+ACK 旗標解析"""
        pkt = IP(src="10.0.0.1", dst="192.168.1.1") / TCP(flags="SA")
        r = parser.parse(pkt)
        assert "SYN" in r["flags"] and "ACK" in r["flags"]

    def test_psh_ack_flags(self, parser):
        """PSH+ACK 旗標解析"""
        pkt = IP(src="1.1.1.1", dst="2.2.2.2") / TCP(dport=80, flags="PA")
        r = parser.parse(pkt)
        assert "PSH" in r["flags"] and "ACK" in r["flags"]

    def test_http_service_detection(self, parser):
        """HTTP 服務識別（Port 80）"""
        pkt = (IP(src="1.1.1.1", dst="2.2.2.2")
               / TCP(sport=12345, dport=80, flags="PA")
               / b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
        r = parser.parse(pkt)
        assert r["service"] == "HTTP"
        assert "GET" in r["payload_text"]

    def test_ssh_service_detection(self, parser):
        """SSH 服務識別（Port 22）"""
        pkt = IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=55000, dport=22)
        r = parser.parse(pkt)
        assert r["service"] == "SSH"

    def test_https_service_detection(self, parser):
        """HTTPS 服務識別（Port 443）"""
        pkt = IP(src="1.1.1.1", dst="2.2.2.2") / TCP(dport=443)
        r = parser.parse(pkt)
        assert r["service"] == "HTTPS"

    def test_seq_ack_window_fields(self, parser):
        """TCP seq、ack、window 欄位"""
        pkt = IP(src="1.1.1.1", dst="2.2.2.2") / TCP(
            dport=80, flags="SA", seq=12345, ack=67890, window=65535)
        r = parser.parse(pkt)
        assert r["seq"]    == 12345
        assert r["ack"]    == 67890
        assert r["window"] == 65535

    def test_ip_version_ipv4(self, parser):
        """IPv4 封包的 ip_version 欄位應為 4"""
        pkt = IP(src="1.1.1.1", dst="2.2.2.2") / TCP(dport=80, flags="S")
        r = parser.parse(pkt)
        assert r["ip_version"] == 4


# ────────────────────────────────────────────────────────────
# UDP 封包解析
# ────────────────────────────────────────────────────────────
class TestUDPPacket:

    def test_dns_over_udp(self, parser):
        """DNS over UDP（Port 53）識別 — 必須同時含有 DNS layer"""
        pkt = (IP(src="192.168.1.1", dst="8.8.8.8")
               / UDP(sport=55000, dport=53)
               / DNS(rd=1, qd=DNSQR(qname="example.com")))
        r = parser.parse(pkt)
        assert r["protocol"] == "DNS"

    def test_udp_port53_without_dns_layer(self, parser):
        """純 UDP dport=53 但無 DNS layer 不應被標為 DNS——這是 Bug 修正驗證"""
        pkt = IP(src="192.168.1.1", dst="8.8.8.8") / UDP(sport=55000, dport=53)
        r = parser.parse(pkt)
        assert r["protocol"] == "UDP"

    def test_dhcp_detection(self, parser):
        """DHCP（Port 67/68）識別"""
        pkt = IP(src="0.0.0.0", dst="255.255.255.255") / UDP(sport=68, dport=67)
        r = parser.parse(pkt)
        assert r["protocol"] == "DHCP"

    def test_ntp_detection(self, parser):
        """NTP（Port 123）識別"""
        pkt = IP(src="192.168.1.1", dst="pool.ntp.org") / UDP(dport=123)
        r = parser.parse(pkt)
        assert r["protocol"] == "NTP"

    def test_ssdp_detection(self, parser):
        """SSDP（Port 1900）識別"""
        pkt = IP(src="192.168.1.1", dst="239.255.255.250") / UDP(dport=1900)
        r = parser.parse(pkt)
        assert r["protocol"] == "SSDP"


# ────────────────────────────────────────────────────────────
# ICMP 封包解析（加強版）
# ────────────────────────────────────────────────────────────
class TestICMPPacket:

    def test_echo_request(self, parser):
        """ICMP Echo Request（ping）"""
        pkt = IP(src="192.168.1.1", dst="8.8.8.8") / ICMP(type=8, code=0)
        r = parser.parse(pkt)
        assert r["protocol"]  == "ICMP"
        assert r["icmp_type"] == 8
        assert "Echo Request" in r["flags"]

    def test_echo_reply(self, parser):
        """ICMP Echo Reply"""
        pkt = IP(src="8.8.8.8", dst="192.168.1.1") / ICMP(type=0, code=0)
        r = parser.parse(pkt)
        assert r["icmp_type"] == 0
        assert "Echo Reply" in r["flags"]

    def test_destination_unreachable_port(self, parser):
        """ICMP Destination Unreachable - Port Unreachable (type=3, code=3)"""
        pkt = IP(src="192.168.1.1", dst="10.0.0.1") / ICMP(type=3, code=3)
        r = parser.parse(pkt)
        assert r["icmp_type"] == 3
        assert r["icmp_code"] == 3
        assert "Port Unreachable" in r["icmp_desc"]

    def test_destination_unreachable_host(self, parser):
        """ICMP Destination Unreachable - Host Unreachable (type=3, code=1)"""
        pkt = IP(src="192.168.1.1", dst="10.0.0.1") / ICMP(type=3, code=1)
        r = parser.parse(pkt)
        assert "Host Unreachable" in r["icmp_desc"]

    def test_time_exceeded_ttl(self, parser):
        """ICMP Time Exceeded - TTL Exceeded (type=11, code=0)"""
        pkt = IP(src="10.0.0.1", dst="192.168.1.1") / ICMP(type=11, code=0)
        r = parser.parse(pkt)
        assert r["icmp_type"] == 11
        assert "TTL" in r["icmp_desc"]

    def test_time_exceeded_fragment(self, parser):
        """ICMP Time Exceeded - Fragment Reassembly (type=11, code=1)"""
        pkt = IP(src="10.0.0.1", dst="192.168.1.1") / ICMP(type=11, code=1)
        r = parser.parse(pkt)
        assert "Fragment" in r["icmp_desc"]

    def test_icmp_desc_not_none(self, parser):
        """所有 ICMP 封包都應有非空的 icmp_desc"""
        pkt = IP(src="1.1.1.1", dst="2.2.2.2") / ICMP(type=8)
        r = parser.parse(pkt)
        assert r["icmp_desc"] is not None and len(r["icmp_desc"]) > 0


# ────────────────────────────────────────────────────────────
# ARP 封包解析
# ────────────────────────────────────────────────────────────
class TestARPPacket:

    def test_arp_request(self, parser):
        """ARP Request 解析"""
        pkt = (Ether(src="aa:bb:cc:dd:ee:ff", dst="ff:ff:ff:ff:ff:ff")
               / ARP(op=1, psrc="192.168.1.100", pdst="192.168.1.1",
                     hwsrc="aa:bb:cc:dd:ee:ff"))
        r = parser.parse(pkt)
        assert r["protocol"]    == "ARP"
        assert r["arp_op"]      == "Request"
        assert r["arp_src_ip"]  == "192.168.1.100"
        assert r["arp_src_mac"] == "aa:bb:cc:dd:ee:ff"  # 新增欄位

    def test_arp_reply(self, parser):
        """ARP Reply 解析"""
        pkt = Ether() / ARP(op=2, psrc="192.168.1.1", pdst="192.168.1.100")
        r = parser.parse(pkt)
        assert r["arp_op"] == "Reply"

    def test_arp_src_mac_populated(self, parser):
        """ARP 解析應填入 arp_src_mac 欄位"""
        mac = "11:22:33:44:55:66"
        pkt = Ether(src=mac) / ARP(op=1, psrc="10.0.0.1", pdst="10.0.0.2",
                                   hwsrc=mac)
        r = parser.parse(pkt)
        assert r["arp_src_mac"] == mac


# ────────────────────────────────────────────────────────────
# IPv6 封包解析（新增）
# ────────────────────────────────────────────────────────────
class TestIPv6Packet:

    def test_ipv6_basic_fields(self, parser):
        """IPv6 封包基本欄位"""
        pkt = (IPv6(src="2001:db8::1", dst="2001:db8::2", hlim=64)
               / TCP(dport=80, flags="S"))
        r = parser.parse(pkt)
        assert r["src_ip"]     == "2001:db8::1"
        assert r["dst_ip"]     == "2001:db8::2"
        assert r["ip_version"] == 6
        assert r["ttl"]        == 64   # Hop Limit

    def test_icmpv6_echo_request(self, parser):
        """ICMPv6 Echo Request 解析"""
        pkt = IPv6(src="::1", dst="::2") / ICMPv6EchoRequest()
        r = parser.parse(pkt)
        assert r["protocol"]  == "ICMPv6"
        assert r["icmp_type"] == 128
        assert "Echo Request" in r["flags"]

    def test_icmpv6_echo_reply(self, parser):
        """ICMPv6 Echo Reply 解析"""
        pkt = IPv6(src="::2", dst="::1") / ICMPv6EchoReply()
        r = parser.parse(pkt)
        assert r["protocol"]  == "ICMPv6"
        assert r["icmp_type"] == 129
        assert "Echo Reply" in r["flags"]


# ────────────────────────────────────────────────────────────
# TCP Flags
# ────────────────────────────────────────────────────────────
class TestTCPFlags:

    def test_syn(self, parser):
        assert parser._parse_tcp_flags(0x002) == "SYN"

    def test_ack(self, parser):
        assert parser._parse_tcp_flags(0x010) == "ACK"

    def test_syn_ack(self, parser):
        f = parser._parse_tcp_flags(0x012)
        assert "SYN" in f and "ACK" in f

    def test_psh_ack(self, parser):
        f = parser._parse_tcp_flags(0x018)
        assert "PSH" in f and "ACK" in f

    def test_fin(self, parser):
        assert "FIN" in parser._parse_tcp_flags(0x001)

    def test_rst(self, parser):
        assert "RST" in parser._parse_tcp_flags(0x004)

    def test_no_flags(self, parser):
        assert parser._parse_tcp_flags(0x000) == "NONE"

    def test_all_flags(self, parser):
        f = parser._parse_tcp_flags(0x03F)
        for flag in ["FIN", "SYN", "RST", "PSH", "ACK", "URG"]:
            assert flag in f

    def test_ece_cwr_flags(self, parser):
        """ECE 與 CWR（ECN 使用）"""
        f = parser._parse_tcp_flags(0x0C0)
        assert "ECE" in f or "CWR" in f


# ────────────────────────────────────────────────────────────
# Payload 提取
# ────────────────────────────────────────────────────────────
class TestPayload:

    def test_http_payload(self, parser):
        """HTTP Payload 提取與 ASCII 顯示"""
        pkt = (IP(src="1.1.1.1", dst="2.2.2.2")
               / TCP(dport=80)
               / b"GET /index.html HTTP/1.1\r\n")
        r = parser.parse(pkt)
        assert r["payload_hex"]  is not None
        assert r["payload_text"] is not None
        assert "GET" in r["payload_text"]

    def test_binary_payload_dots(self, parser):
        """二進位 Payload 中不可見字符應顯示為 '.'"""
        pkt = (IP(src="1.1.1.1", dst="2.2.2.2")
               / UDP(dport=9999)
               / bytes(range(32)))
        r = parser.parse(pkt)
        if r["payload_text"]:
            assert "." in r["payload_text"]

    def test_no_payload(self, parser):
        """無 Payload 的封包（SYN）欄位應為 None"""
        pkt = IP(src="1.1.1.1", dst="2.2.2.2") / TCP(flags="S")
        r = parser.parse(pkt)
        assert r["payload_hex"]  is None
        assert r["payload_text"] is None

    def test_payload_len_field(self, parser):
        """payload_len 欄位應等於實際 payload 長度"""
        payload = b"Hello World!"
        pkt = (IP(src="1.1.1.1", dst="2.2.2.2")
               / TCP(dport=8080, flags="PA")
               / payload)
        r = parser.parse(pkt)
        assert r["payload_len"] == len(payload)


# ────────────────────────────────────────────────────────────
# DNS 解析（新增）
# ────────────────────────────────────────────────────────────
class TestDNSParsing:

    def test_dns_query_name(self, parser):
        """DNS 查詢域名應填入 dns_query 欄位"""
        pkt = (IP(src="192.168.1.1", dst="8.8.8.8")
               / UDP(sport=54321, dport=53)
               / DNS(rd=1, qd=DNSQR(qname="example.com")))
        r = parser.parse(pkt)
        assert r["protocol"]  == "DNS"
        assert r["dns_query"] == "example.com"

    def test_dns_response_rdata(self, parser):
        """DNS 回應應填入 dns_response 欄位"""
        pkt = (IP(src="8.8.8.8", dst="192.168.1.1")
               / UDP(sport=53, dport=54321)
               / DNS(qr=1,
                     qd=DNSQR(qname="example.com"),
                     an=DNSRR(rrname="example.com", rdata="93.184.216.34")))
        r = parser.parse(pkt)
        assert r["dns_response"] is not None
        assert "example.com" in r["dns_response"]


# ────────────────────────────────────────────────────────────
# TLS 解析（新增）
# ────────────────────────────────────────────────────────────
class TestTLSParsing:

    def test_tls12_client_hello(self, parser):
        """TLS 1.2 ClientHello 識別"""
        # Content Type=22(Handshake), Version=0x0303(TLS1.2), HS Type=1(ClientHello)
        tls_payload = b"\x16\x03\x03\x00\x05\x01\x00\x00\x01\x00"
        pkt = (IP(src="192.168.1.1", dst="93.184.216.34")
               / TCP(sport=50000, dport=443, flags="PA")
               / tls_payload)
        r = parser.parse(pkt)
        assert r["protocol"]      == "TLS"
        assert r["tls_version"]   == "TLS 1.2"
        assert r["tls_type"]      == "Handshake"
        assert r["tls_handshake"] == "ClientHello"

    def test_tls13_client_hello(self, parser):
        """TLS 1.3 ClientHello 識別"""
        tls_payload = b"\x16\x03\x04\x00\x05\x01\x00\x00\x01\x00"
        pkt = (IP(src="192.168.1.1", dst="1.1.1.1")
               / TCP(sport=50001, dport=443, flags="PA")
               / tls_payload)
        r = parser.parse(pkt)
        assert r["protocol"]    == "TLS"
        assert r["tls_version"] == "TLS 1.3"

    def test_tls12_server_hello(self, parser):
        """TLS 1.2 ServerHello 識別"""
        tls_payload = b"\x16\x03\x03\x00\x05\x02\x00\x00\x01\x00"
        pkt = (IP(src="93.184.216.34", dst="192.168.1.1")
               / TCP(sport=443, dport=50000, flags="PA")
               / tls_payload)
        r = parser.parse(pkt)
        assert r["tls_handshake"] == "ServerHello"

    def test_tls_application_data(self, parser):
        """TLS ApplicationData（加密資料）識別"""
        tls_payload = b"\x17\x03\x03\x00\x10" + b"\x00" * 16
        pkt = (IP(src="192.168.1.1", dst="93.184.216.34")
               / TCP(sport=50000, dport=443, flags="PA")
               / tls_payload)
        r = parser.parse(pkt)
        assert r["protocol"]  == "TLS"
        assert r["tls_type"]  == "ApplicationData"

    def test_non_tls_payload_no_tls_fields(self, parser):
        """非 TLS Port 的連線不應被識別為 TLS"""
        pkt = (IP(src="1.1.1.1", dst="2.2.2.2")
               / TCP(dport=8080, flags="PA")
               / b"GET / HTTP/1.1\r\n\r\n")
        r = parser.parse(pkt)
        assert r.get("tls_version") is None
        assert r.get("tls_type")    is None
