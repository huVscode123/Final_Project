# ============================================================
# tests/generate_attack_pcap.py - 攻擊測試封包生成腳本
#
# 功能：生成包含各種攻擊特徵的測試 PCAP 檔案
#
# 執行方式:
#   cd network_capture
#   python tests/generate_attack_pcap.py
#
# 輸出檔案:
#   tests/sample.pcap          - 基本正常流量
#   tests/syn_flood.pcap       - SYN Flood 攻擊
#   tests/port_scan.pcap       - Port Scan 攻擊
#   tests/icmp_flood.pcap      - ICMP Flood 攻擊
#   tests/udp_flood.pcap       - UDP Flood 攻擊
#   tests/arp_spoof.pcap       - ARP Spoofing
#   tests/mixed_attacks.pcap   - 混合攻擊（完整測試用）
#   tests/tls_traffic.pcap     - TLS/HTTPS 流量
# ============================================================

import os
import sys

# 確保在 network_capture 目錄執行
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

from scapy.all import wrpcap
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import Ether, ARP
from scapy.layers.dns import DNS, DNSQR, DNSRR

TESTS_DIR = "tests"
os.makedirs(TESTS_DIR, exist_ok=True)


def ok(name, count):
    print(f"  ✓  {name}  ({count} 封包)")


# ─── 1. 基本正常流量 sample.pcap ─────────────────────────
def gen_sample():
    pkts = [
        # TCP 三次握手 + HTTP
        IP(src="192.168.1.100", dst="93.184.216.34") / TCP(dport=80, flags="S"),
        IP(src="93.184.216.34", dst="192.168.1.100") / TCP(sport=80, dport=54321, flags="SA"),
        IP(src="192.168.1.100", dst="93.184.216.34") / TCP(dport=80, flags="PA")
            / b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n",
        IP(src="93.184.216.34", dst="192.168.1.100") / TCP(sport=80, dport=54321, flags="PA")
            / b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nHello World",
        IP(src="192.168.1.100", dst="93.184.216.34") / TCP(dport=80, flags="FA"),

        # DNS 查詢
        IP(src="192.168.1.100", dst="8.8.8.8") / UDP(sport=55000, dport=53)
            / DNS(rd=1, qd=DNSQR(qname="google.com")),
        IP(src="8.8.8.8", dst="192.168.1.100") / UDP(sport=53, dport=55000)
            / DNS(qr=1, qd=DNSQR(qname="google.com"),
                  an=DNSRR(rrname="google.com", rdata="142.250.185.78")),

        # ICMP Ping
        IP(src="192.168.1.100", dst="8.8.8.8") / ICMP(type=8, code=0),
        IP(src="8.8.8.8", dst="192.168.1.100") / ICMP(type=0, code=0),

        # SSH
        IP(src="192.168.1.100", dst="10.0.0.5") / TCP(dport=22, flags="S"),
        IP(src="10.0.0.5", dst="192.168.1.100") / TCP(sport=22, dport=50001, flags="SA"),

        # ARP 正常
        Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(op=1, psrc="192.168.1.100",
                                              pdst="192.168.1.1",
                                              hwsrc="aa:bb:cc:dd:ee:ff"),
        Ether(src="00:11:22:33:44:55") / ARP(op=2, psrc="192.168.1.1",
                                              pdst="192.168.1.100",
                                              hwsrc="00:11:22:33:44:55"),
    ] * 5

    wrpcap(f"{TESTS_DIR}/sample.pcap", pkts)
    ok("sample.pcap", len(pkts))


# ─── 2. SYN Flood ─────────────────────────────────────────
def gen_syn_flood():
    attacker = "10.10.10.10"
    victim   = "192.168.1.1"
    pkts = []

    # 模擬 200 個不同來源 Port 的 SYN（偽造 IP）
    for i in range(200):
        spoofed_src = f"172.16.{i // 256}.{i % 256}"
        pkts.append(
            IP(src=spoofed_src, dst=victim)
            / TCP(sport=10000+i, dport=80, flags="S", seq=1000+i)
        )

    # 加入少量正常流量
    for i in range(5):
        pkts.append(
            IP(src="192.168.1.100", dst=victim)
            / TCP(dport=80, flags="PA")
            / b"GET / HTTP/1.1\r\n\r\n"
        )

    wrpcap(f"{TESTS_DIR}/syn_flood.pcap", pkts)
    ok("syn_flood.pcap", len(pkts))


# ─── 3. Port Scan（SYN Scan） ───────────────────────────
def gen_port_scan():
    attacker = "10.20.20.20"
    target   = "192.168.1.50"
    pkts     = []

    # 掃描常見 Port
    common_ports = [
        21, 22, 23, 25, 53, 80, 110, 135, 139, 143,
        443, 445, 1433, 1521, 3306, 3389, 5432, 5900, 8080, 8443,
        27017, 6379, 9200, 11211, 1900
    ]

    for port in common_ports:
        # SYN
        pkts.append(IP(src=attacker, dst=target) / TCP(dport=port, flags="S"))
        # 目標回應 RST（port closed）
        pkts.append(IP(src=target, dst=attacker) / TCP(sport=port, flags="R"))

    # XMAS Scan（FIN+PSH+URG）
    for port in [22, 80, 443]:
        pkts.append(
            IP(src=attacker, dst=target)
            / TCP(dport=port, flags=0x029)  # FIN+PSH+URG
        )

    # NULL Scan（no flags）
    for port in [22, 80]:
        pkts.append(
            IP(src=attacker, dst=target)
            / TCP(dport=port, flags=0x000)
        )

    wrpcap(f"{TESTS_DIR}/port_scan.pcap", pkts)
    ok("port_scan.pcap", len(pkts))


# ─── 4. ICMP Flood ────────────────────────────────────────
def gen_icmp_flood():
    attacker = "10.30.30.30"
    victim   = "192.168.1.100"
    pkts     = []

    for i in range(200):
        pkts.append(
            IP(src=attacker, dst=victim, id=i)
            / ICMP(type=8, code=0, id=1, seq=i)
            / (b"X" * 56)   # 56 bytes payload（標準 ping 大小）
        )

    # 加入一些 ICMP Destination Unreachable（type=3）
    for code in range(5):
        pkts.append(
            IP(src="192.168.1.1", dst="192.168.1.100")
            / ICMP(type=3, code=code)
        )

    # TTL Exceeded（Traceroute 特徵）
    for ttl in range(1, 6):
        pkts.append(
            IP(src="192.168.1.100", dst="8.8.8.8", ttl=ttl)
            / ICMP(type=8)
        )
        pkts.append(
            IP(src=f"10.0.{ttl}.1", dst="192.168.1.100")
            / ICMP(type=11, code=0)
        )

    wrpcap(f"{TESTS_DIR}/icmp_flood.pcap", pkts)
    ok("icmp_flood.pcap", len(pkts))


# ─── 5. UDP Flood ─────────────────────────────────────────
def gen_udp_flood():
    attacker = "10.40.40.40"
    victim   = "192.168.1.100"
    pkts     = []

    for i in range(300):
        pkts.append(
            IP(src=attacker, dst=victim)
            / UDP(sport=10000+i, dport=80+i)
            / (b"\x00" * 512)   # 512 bytes payload
        )

    wrpcap(f"{TESTS_DIR}/udp_flood.pcap", pkts)
    ok("udp_flood.pcap", len(pkts))


# ─── 6. ARP Spoofing ──────────────────────────────────────
def gen_arp_spoof():
    pkts = []

    # 正常 ARP Table
    normal_pairs = [
        ("192.168.1.1",   "00:11:22:33:44:55"),  # 閘道
        ("192.168.1.100", "aa:bb:cc:dd:ee:ff"),  # 工作站 A
        ("192.168.1.101", "11:22:33:44:55:66"),  # 工作站 B
    ]

    for ip, mac in normal_pairs:
        pkts.append(
            Ether(src=mac, dst="ff:ff:ff:ff:ff:ff")
            / ARP(op=1, psrc=ip, pdst="192.168.1.1", hwsrc=mac)
        )
        pkts.append(
            Ether(src="00:11:22:33:44:55")
            / ARP(op=2, psrc="192.168.1.1", pdst=ip,
                  hwsrc="00:11:22:33:44:55", hwdst=mac)
        )

    # ARP Spoofing：攻擊者使用假 MAC 廣播
    attacker_mac = "ff:ee:dd:cc:bb:aa"
    # 偽裝成閘道（告訴工作站 A：192.168.1.1 的 MAC 是攻擊者的）
    for _ in range(10):
        pkts.append(
            Ether(src=attacker_mac, dst="aa:bb:cc:dd:ee:ff")
            / ARP(op=2, psrc="192.168.1.1", pdst="192.168.1.100",
                  hwsrc=attacker_mac, hwdst="aa:bb:cc:dd:ee:ff")
        )
        # 同時偽裝成工作站 A（告訴閘道）
        pkts.append(
            Ether(src=attacker_mac, dst="00:11:22:33:44:55")
            / ARP(op=2, psrc="192.168.1.100", pdst="192.168.1.1",
                  hwsrc=attacker_mac, hwdst="00:11:22:33:44:55")
        )

    wrpcap(f"{TESTS_DIR}/arp_spoof.pcap", pkts)
    ok("arp_spoof.pcap", len(pkts))


# ─── 7. TLS/HTTPS 流量 ────────────────────────────────────
def gen_tls_traffic():
    pkts = []

    # TLS 1.2 ClientHello（手動構造 TLS Record Layer bytes）
    # Content Type=22(Handshake) + Version=0x0303(TLS1.2) + Length + HS Type=1(ClientHello)
    tls_client_hello = (
        b"\x16\x03\x03\x00\x05"    # Handshake, TLS 1.2, length=5
        b"\x01\x00\x00\x01\x00"    # ClientHello
    )
    tls_server_hello = (
        b"\x16\x03\x03\x00\x05"
        b"\x02\x00\x00\x01\x00"    # ServerHello
    )
    tls_app_data = (
        b"\x17\x03\x03\x00\x10"    # ApplicationData, TLS 1.2, length=16
        + b"\x00" * 16
    )
    tls_1_3_client_hello = (
        b"\x16\x03\x04\x00\x05"    # Handshake, TLS 1.3
        b"\x01\x00\x00\x01\x00"
    )

    for i in range(10):
        # Client -> Server: ClientHello
        pkts.append(
            IP(src=f"192.168.1.{100+i}", dst="93.184.216.34")
            / TCP(sport=50000+i, dport=443, flags="PA")
            / tls_client_hello
        )
        # Server -> Client: ServerHello
        pkts.append(
            IP(src="93.184.216.34", dst=f"192.168.1.{100+i}")
            / TCP(sport=443, dport=50000+i, flags="PA")
            / tls_server_hello
        )
        # Application Data
        pkts.append(
            IP(src=f"192.168.1.{100+i}", dst="93.184.216.34")
            / TCP(sport=50000+i, dport=443, flags="PA")
            / tls_app_data
        )

    # TLS 1.3 Client Hello
    for i in range(5):
        pkts.append(
            IP(src=f"10.0.0.{i+1}", dst="1.1.1.1")
            / TCP(sport=60000+i, dport=443, flags="PA")
            / tls_1_3_client_hello
        )

    wrpcap(f"{TESTS_DIR}/tls_traffic.pcap", pkts)
    ok("tls_traffic.pcap", len(pkts))


# ─── 8. 混合攻擊（完整測試用）────────────────────────────
def gen_mixed_attacks():
    pkts = []

    # 正常 HTTP 流量
    for i in range(10):
        pkts.append(
            IP(src=f"192.168.1.{10+i}", dst="93.184.216.34")
            / TCP(dport=80, flags="PA")
            / b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
        )

    # SYN Flood（來自 attacker_1）
    for i in range(150):
        pkts.append(
            IP(src="172.16.0.1", dst="192.168.1.100")
            / TCP(sport=10000+i, dport=80, flags="S")
        )

    # Port Scan（來自 attacker_2）
    for port in range(1, 50):
        pkts.append(
            IP(src="172.16.0.2", dst="192.168.1.100")
            / TCP(dport=port, flags="S")
        )

    # ICMP Flood（來自 attacker_3）
    for i in range(60):
        pkts.append(
            IP(src="172.16.0.3", dst="192.168.1.100")
            / ICMP(type=8, seq=i)
        )

    # ARP Spoofing
    pkts.append(
        Ether(src="aa:aa:aa:aa:aa:aa")
        / ARP(op=1, psrc="192.168.1.1", pdst="192.168.1.100",
              hwsrc="aa:aa:aa:aa:aa:aa")
    )
    pkts.append(
        Ether(src="bb:bb:bb:bb:bb:bb")
        / ARP(op=2, psrc="192.168.1.1", pdst="192.168.1.100",
              hwsrc="bb:bb:bb:bb:bb:bb")
    )

    # DNS 查詢
    for i in range(5):
        pkts.append(
            IP(src=f"192.168.1.{20+i}", dst="8.8.8.8")
            / UDP(sport=55000+i, dport=53)
            / DNS(rd=1, qd=DNSQR(qname=f"site{i}.com"))
        )

    wrpcap(f"{TESTS_DIR}/mixed_attacks.pcap", pkts)
    ok("mixed_attacks.pcap", len(pkts))


# ─── 主程式 ───────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  測試 PCAP 生成工具")
    print("  輸出目錄:", os.path.abspath(TESTS_DIR))
    print("="*55)

    generators = [
        ("基本正常流量",      gen_sample),
        ("SYN Flood",         gen_syn_flood),
        ("Port Scan",         gen_port_scan),
        ("ICMP Flood",        gen_icmp_flood),
        ("UDP Flood",         gen_udp_flood),
        ("ARP Spoofing",      gen_arp_spoof),
        ("TLS/HTTPS 流量",    gen_tls_traffic),
        ("混合攻擊",          gen_mixed_attacks),
    ]

    for name, fn in generators:
        try:
            fn()
        except Exception as e:
            print(f"  ✗  {name}: {e}")

    print("\n  所有 PCAP 生成完成！")
    print("\n  測試指令：")
    print("    python main.py pcap -f tests/sample.pcap --full")
    print("    python main.py pcap -f tests/syn_flood.pcap --detect")
    print("    python main.py pcap -f tests/port_scan.pcap --detect --tcp")
    print("    python main.py pcap -f tests/arp_spoof.pcap --arp --detect")
    print("    python main.py pcap -f tests/tls_traffic.pcap --tls")
    print("    python main.py pcap -f tests/mixed_attacks.pcap --full")
    print()
