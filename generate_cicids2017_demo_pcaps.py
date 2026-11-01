#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_cicids2017_demo_pcaps.py

為 CICIDS2017 非監督式與半監督式模型產生 Demo 用測試 PCAP：
  - 5 種正常流量封包（各自獨立 PCAP）
  - 5 種攻擊流量封包（對應 CICIDS2017 標籤：DDoS / PortScan / SSH-Patator / Web Attack / DoS Hulk）

CICIDS2017 使用 Flow 層級 78 維 CICFlowMeter 統計特徵。
攻擊封包的特徵設計重點：產生與正常流量統計特性（流持續時間、封包數、位元組數、
IAT 分佈、旗標計數等）顯著不同的 Flow，讓模型在 Flow 特徵空間中容易區分。

用法：python generate_cicids2017_demo_pcaps.py
"""

import os
import sys
import time
import random

from scapy.all import (
    Ether, IP, TCP, UDP, ICMP, DNS, DNSQR, Raw, wrpcap, RandShort,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "test_pcaps", "cicids2017_demo")


def _ts():
    """回傳遞增基底時戳。"""
    return time.time()


# =====================================================================
# 正常流量
# =====================================================================

def gen_normal_web_browsing():
    """正常網頁瀏覽：HTTP GET 請求 + 回應，多個不同網站，合理間隔。"""
    pkts = []
    sites = [
        ("93.184.216.34", "example.com"),
        ("142.250.185.206", "www.google.com"),
        ("151.101.1.69", "reddit.com"),
        ("104.16.132.229", "cloudflare.com"),
        ("185.199.108.153", "github.com"),
    ]
    src_ip = "192.168.1.100"
    t = _ts()
    for dst_ip, host in sites:
        sport = random.randint(49152, 65535)
        # TCP 三向交握
        syn = IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=80, flags="S", seq=1000)
        syn.time = t
        pkts.append(syn)
        t += 0.001
        sa = IP(src=dst_ip, dst=src_ip) / TCP(sport=80, dport=sport, flags="SA", seq=2000, ack=1001)
        sa.time = t
        pkts.append(sa)
        t += 0.001
        ack = IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=80, flags="A", seq=1001, ack=2001)
        ack.time = t
        pkts.append(ack)
        t += 0.005
        # HTTP GET
        req = (IP(src=src_ip, dst=dst_ip) /
               TCP(sport=sport, dport=80, flags="PA", seq=1001, ack=2001) /
               Raw(load=f"GET / HTTP/1.1\r\nHost: {host}\r\nUser-Agent: Mozilla/5.0\r\n\r\n".encode()))
        req.time = t
        pkts.append(req)
        t += 0.05
        # HTTP Response
        body = f"<html><head><title>{host}</title></head><body>Welcome</body></html>"
        resp = (IP(src=dst_ip, dst=src_ip) /
                TCP(sport=80, dport=sport, flags="PA", seq=2001, ack=1200) /
                Raw(load=f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode()))
        resp.time = t
        pkts.append(resp)
        t += 0.3 + random.uniform(0.5, 2.0)  # 正常用戶間隔
        # FIN
        fin = IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=80, flags="FA", seq=1200, ack=2500)
        fin.time = t
        pkts.append(fin)
        t += 0.001
        fin_ack = IP(src=dst_ip, dst=src_ip) / TCP(sport=80, dport=sport, flags="FA", seq=2500, ack=1201)
        fin_ack.time = t
        pkts.append(fin_ack)
        t += 1.0
    return pkts


def gen_normal_dns_queries():
    """正常 DNS 查詢：查詢多個常見域名，合理間隔。"""
    pkts = []
    domains = ["www.google.com", "www.yahoo.com", "www.github.com",
               "mail.google.com", "docs.python.org", "en.wikipedia.org",
               "www.microsoft.com", "cdn.jsdelivr.net"]
    src_ip = "192.168.1.100"
    dns_srv = "8.8.8.8"
    t = _ts()
    for dom in domains:
        sport = random.randint(49152, 65535)
        qr = (IP(src=src_ip, dst=dns_srv) /
              UDP(sport=sport, dport=53) /
              DNS(rd=1, qd=DNSQR(qname=dom)))
        qr.time = t
        pkts.append(qr)
        t += 0.002
        # DNS response (simplified)
        resp_data = (IP(src=dns_srv, dst=src_ip) /
                     UDP(sport=53, dport=sport) /
                     DNS(qr=1, aa=1, qd=DNSQR(qname=dom)) /
                     Raw(load=b"\xc0\x0c\x00\x01\x00\x01\x00\x00\x00\x3c\x00\x04\x01\x02\x03\x04"))
        resp_data.time = t
        pkts.append(resp_data)
        t += random.uniform(0.5, 3.0)
    return pkts


def gen_normal_ssh_session():
    """正常 SSH 連線：完成三向交握後傳輸少量加密資料。"""
    pkts = []
    src_ip = "192.168.1.100"
    dst_ip = "192.168.1.10"
    sport = random.randint(49152, 65535)
    t = _ts()
    # 三向交握
    pkts.append(IP(src=src_ip, dst=dst_ip, id=1) / TCP(sport=sport, dport=22, flags="S", seq=100))
    pkts[-1].time = t; t += 0.001
    pkts.append(IP(src=dst_ip, dst=src_ip, id=2) / TCP(sport=22, dport=sport, flags="SA", seq=200, ack=101))
    pkts[-1].time = t; t += 0.001
    pkts.append(IP(src=src_ip, dst=dst_ip, id=3) / TCP(sport=sport, dport=22, flags="A", seq=101, ack=201))
    pkts[-1].time = t; t += 0.01
    # SSH banner
    pkts.append(IP(src=dst_ip, dst=src_ip) / TCP(sport=22, dport=sport, flags="PA", seq=201, ack=101) /
                Raw(load=b"SSH-2.0-OpenSSH_8.9\r\n"))
    pkts[-1].time = t; t += 0.05
    pkts.append(IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=22, flags="PA", seq=101, ack=230) /
                Raw(load=b"SSH-2.0-PuTTY_0.78\r\n"))
    pkts[-1].time = t; t += 0.1
    # 模擬加密資料交換（少量，合理大小）
    for i in range(5):
        data = os.urandom(random.randint(64, 256))
        pkts.append(IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=22, flags="PA") / Raw(load=data))
        pkts[-1].time = t; t += random.uniform(0.5, 3.0)
        resp = os.urandom(random.randint(64, 512))
        pkts.append(IP(src=dst_ip, dst=src_ip) / TCP(sport=22, dport=sport, flags="PA") / Raw(load=resp))
        pkts[-1].time = t; t += random.uniform(0.3, 2.0)
    return pkts


def gen_normal_email_imap():
    """正常 IMAP 郵件收取：模擬 IMAP 連線。"""
    pkts = []
    src_ip = "192.168.1.100"
    dst_ip = "74.125.200.108"  # Gmail IMAP
    sport = random.randint(49152, 65535)
    t = _ts()
    # 三向交握
    pkts.append(IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=993, flags="S"))
    pkts[-1].time = t; t += 0.02
    pkts.append(IP(src=dst_ip, dst=src_ip) / TCP(sport=993, dport=sport, flags="SA"))
    pkts[-1].time = t; t += 0.001
    pkts.append(IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=993, flags="A"))
    pkts[-1].time = t; t += 0.05
    # TLS handshake (模擬)
    pkts.append(IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=993, flags="PA") /
                Raw(load=b"\x16\x03\x01\x00\x05\x01\x00\x01\x00\x00"))  # ClientHello stub
    pkts[-1].time = t; t += 0.03
    pkts.append(IP(src=dst_ip, dst=src_ip) / TCP(sport=993, dport=sport, flags="PA") /
                Raw(load=b"\x16\x03\x03\x00\x05\x02\x00\x01\x00\x00"))  # ServerHello stub
    pkts[-1].time = t; t += 0.05
    # 加密資料
    for i in range(4):
        pkts.append(IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=993, flags="PA") /
                    Raw(load=os.urandom(random.randint(32, 128))))
        pkts[-1].time = t; t += random.uniform(1.0, 5.0)
        pkts.append(IP(src=dst_ip, dst=src_ip) / TCP(sport=993, dport=sport, flags="PA") /
                    Raw(load=os.urandom(random.randint(200, 1000))))
        pkts[-1].time = t; t += random.uniform(0.5, 2.0)
    return pkts


def gen_normal_icmp_ping():
    """正常 ICMP Ping：對幾個主機做標準 ping。"""
    pkts = []
    src_ip = "192.168.1.100"
    targets = ["192.168.1.1", "8.8.8.8", "1.1.1.1"]
    t = _ts()
    for dst in targets:
        for seq in range(4):
            req = IP(src=src_ip, dst=dst) / ICMP(type=8, code=0, id=0x1234, seq=seq) / Raw(load=b"A" * 32)
            req.time = t
            pkts.append(req)
            t += 0.01
            rep = IP(src=dst, dst=src_ip) / ICMP(type=0, code=0, id=0x1234, seq=seq) / Raw(load=b"A" * 32)
            rep.time = t
            pkts.append(rep)
            t += 1.0  # 正常 1 秒間隔
    return pkts


# =====================================================================
# 攻擊流量 — 對應 CICIDS2017 標籤
# =====================================================================

def gen_attack_ddos():
    """DDoS 攻擊：大量來源 IP 對同一目標發起 HTTP Flood（多 connection）。

    Flow 設計：
    - 每條 connection 發 SYN -> SYN-ACK -> ACK -> GET (flood) -> RST
    - 極短 IAT、大量 SYN/ACK/PSH flag、遠高於正常的 Fwd Packets / s
    - 每條 connection = 1 個 flow，確保 PcapFlowConverter 能產出 flow CSV
    """
    pkts = []
    victim = "192.168.10.50"
    t = _ts()

    # 40 個偽造來源 IP 各建立一條 HTTP flood connection
    for i in range(40):
        src = f"10.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,i + 1)}"
        sport = random.randint(49152, 65534)
        seq = random.randint(10000, 999999)

        # SYN
        syn = (IP(src=src, dst=victim) /
               TCP(sport=sport, dport=80, flags="S", seq=seq))
        syn.time = t
        pkts.append(syn)
        t += 0.0002

        # SYN-ACK (victim response)
        sa = (IP(src=victim, dst=src) /
              TCP(sport=80, dport=sport, flags="SA", seq=100000, ack=seq + 1))
        sa.time = t
        pkts.append(sa)
        t += 0.0001

        # ACK
        ack = (IP(src=src, dst=victim) /
               TCP(sport=sport, dport=80, flags="A", seq=seq + 1, ack=100001))
        ack.time = t
        pkts.append(ack)
        t += 0.0001

        # HTTP GET flood (3 requests per connection)
        for j in range(3):
            payload = (b"GET /?flood=" + str(random.randint(1, 99999)).encode() +
                       b" HTTP/1.1\r\nHost: target\r\nConnection: keep-alive\r\n\r\n")
            req = (IP(src=src, dst=victim) /
                   TCP(sport=sport, dport=80, flags="PA", seq=seq + 1 + j * 100, ack=100001) /
                   Raw(load=payload))
            req.time = t
            pkts.append(req)
            t += 0.0003

        # RST (攻擊者強制結束，讓 flow converter 能結案)
        rst = (IP(src=src, dst=victim) /
               TCP(sport=sport, dport=80, flags="R", seq=seq + 401))
        rst.time = t
        pkts.append(rst)
        t += 0.0005  # 極短間隔 0.5ms

    return pkts


def gen_attack_portscan():
    """PortScan 攻擊：對目標掃描大量連接埠。
    Flow 特徵：大量短連線、多目的埠號、幾乎全是 SYN/RST、極低資料傳輸。"""
    pkts = []
    attacker = "205.174.165.73"
    victim = "192.168.10.50"
    t = _ts()
    # 掃描 100 個連接埠
    ports = list(range(1, 101))
    for p in ports:
        sport = random.randint(49152, 65535)
        syn = IP(src=attacker, dst=victim) / TCP(sport=sport, dport=p, flags="S", seq=1000)
        syn.time = t
        pkts.append(syn)
        t += 0.0003
        # 大部分埠回 RST（關閉），少數回 SYN-ACK（開放）
        if p in (22, 80, 443, 3306):
            resp = IP(src=victim, dst=attacker) / TCP(sport=p, dport=sport, flags="SA", seq=2000, ack=1001)
            resp.time = t
            pkts.append(resp)
            t += 0.0002
            rst = IP(src=attacker, dst=victim) / TCP(sport=sport, dport=p, flags="R", seq=1001)
            rst.time = t
            pkts.append(rst)
        else:
            resp = IP(src=victim, dst=attacker) / TCP(sport=p, dport=sport, flags="RA", seq=0, ack=1001)
            resp.time = t
            pkts.append(resp)
        t += 0.001
    return pkts


def gen_attack_ssh_bruteforce():
    """SSH-Patator (SSH Brute Force)：對 SSH 埠大量嘗試登入。
    Flow 特徵：多次短 TCP 連線到 port 22、大量 PSH/ACK、高連線頻率。"""
    pkts = []
    attacker = "205.174.165.73"
    victim = "192.168.10.50"
    t = _ts()
    for attempt in range(30):
        sport = random.randint(49152, 65535)
        # 三向交握
        pkts.append(IP(src=attacker, dst=victim) / TCP(sport=sport, dport=22, flags="S", seq=100))
        pkts[-1].time = t; t += 0.001
        pkts.append(IP(src=victim, dst=attacker) / TCP(sport=22, dport=sport, flags="SA", seq=200, ack=101))
        pkts[-1].time = t; t += 0.001
        pkts.append(IP(src=attacker, dst=victim) / TCP(sport=sport, dport=22, flags="A", seq=101, ack=201))
        pkts[-1].time = t; t += 0.005
        # SSH banner
        pkts.append(IP(src=victim, dst=attacker) / TCP(sport=22, dport=sport, flags="PA") /
                    Raw(load=b"SSH-2.0-OpenSSH_7.6\r\n"))
        pkts[-1].time = t; t += 0.002
        # Client banner + auth attempt
        user = random.choice(["root", "admin", "test", "user", "ubuntu"])
        pwd = random.choice(["123456", "password", "admin", "root", "qwerty"])
        pkts.append(IP(src=attacker, dst=victim) / TCP(sport=sport, dport=22, flags="PA") /
                    Raw(load=f"SSH-2.0-libssh2\r\nuser={user}&pass={pwd}".encode()))
        pkts[-1].time = t; t += 0.01
        # Auth failure + RST
        pkts.append(IP(src=victim, dst=attacker) / TCP(sport=22, dport=sport, flags="PA") /
                    Raw(load=b"Permission denied\r\n"))
        pkts[-1].time = t; t += 0.001
        pkts.append(IP(src=victim, dst=attacker) / TCP(sport=22, dport=sport, flags="RA"))
        pkts[-1].time = t; t += 0.05  # 快速重連
    return pkts


def gen_attack_web_bruteforce():
    """Web Attack - Brute Force：對 HTTP 登入頁面暴力嘗試。
    Flow 特徵：大量 POST 請求到同一路徑、高頻率、固定大小封包。"""
    pkts = []
    attacker = "205.174.165.73"
    victim = "192.168.10.50"
    t = _ts()
    for attempt in range(40):
        sport = random.randint(49152, 65535)
        # 三向交握
        pkts.append(IP(src=attacker, dst=victim) / TCP(sport=sport, dport=80, flags="S"))
        pkts[-1].time = t; t += 0.001
        pkts.append(IP(src=victim, dst=attacker) / TCP(sport=80, dport=sport, flags="SA"))
        pkts[-1].time = t; t += 0.001
        pkts.append(IP(src=attacker, dst=victim) / TCP(sport=sport, dport=80, flags="A"))
        pkts[-1].time = t; t += 0.002
        # HTTP POST 暴力破解
        user = random.choice(["admin", "root", "test", "manager"])
        pwd = f"pass{random.randint(1, 99999):05d}"
        body = f"username={user}&password={pwd}"
        req = (IP(src=attacker, dst=victim) /
               TCP(sport=sport, dport=80, flags="PA") /
               Raw(load=f"POST /login HTTP/1.1\r\nHost: target\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode()))
        req.time = t
        pkts.append(req)
        t += 0.01
        # 401 回應
        resp_body = "Login Failed"
        resp = (IP(src=victim, dst=attacker) /
                TCP(sport=80, dport=sport, flags="PA") /
                Raw(load=f"HTTP/1.1 401 Unauthorized\r\nContent-Length: {len(resp_body)}\r\n\r\n{resp_body}".encode()))
        resp.time = t
        pkts.append(resp)
        t += 0.001
        # FIN
        pkts.append(IP(src=attacker, dst=victim) / TCP(sport=sport, dport=80, flags="FA"))
        pkts[-1].time = t; t += 0.001
        pkts.append(IP(src=victim, dst=attacker) / TCP(sport=80, dport=sport, flags="FA"))
        pkts[-1].time = t; t += 0.05
    return pkts


def gen_attack_dos_hulk():
    """DoS Hulk 攻擊：對目標發起高速、隨機化 HTTP GET 請求。
    Flow 特徵：單一來源持續高速連線、大量 GET 不同路徑、高 fwd 封包速率。"""
    pkts = []
    attacker = "205.174.165.73"
    victim = "192.168.10.50"
    t = _ts()
    for i in range(60):
        sport = random.randint(49152, 65535)
        # 快速三向交握
        pkts.append(IP(src=attacker, dst=victim) / TCP(sport=sport, dport=80, flags="S"))
        pkts[-1].time = t; t += 0.0005
        pkts.append(IP(src=victim, dst=attacker) / TCP(sport=80, dport=sport, flags="SA"))
        pkts[-1].time = t; t += 0.0005
        pkts.append(IP(src=attacker, dst=victim) / TCP(sport=sport, dport=80, flags="A"))
        pkts[-1].time = t; t += 0.001
        # Hulk 特徵：隨機化 URL 參數、User-Agent、Referer
        path = f"/?q={random.randint(10000, 99999)}&p={random.randint(1, 999)}"
        ua = random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (X11; Linux x86_64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        ])
        req = (IP(src=attacker, dst=victim) /
               TCP(sport=sport, dport=80, flags="PA") /
               Raw(load=f"GET {path} HTTP/1.1\r\nHost: target\r\nUser-Agent: {ua}\r\nKeep-Alive: 300\r\n\r\n".encode()))
        req.time = t
        pkts.append(req)
        t += 0.002  # 極短間隔
    return pkts


# =====================================================================
# 主程式
# =====================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 正常流量（5 個 PCAP）──
    normal_generators = [
        ("normal_01_web_browsing.pcap", gen_normal_web_browsing, "正常網頁瀏覽"),
        ("normal_02_dns_queries.pcap", gen_normal_dns_queries, "正常 DNS 查詢"),
        ("normal_03_ssh_session.pcap", gen_normal_ssh_session, "正常 SSH 連線"),
        ("normal_04_email_imap.pcap", gen_normal_email_imap, "正常 IMAP 郵件收取"),
        ("normal_05_icmp_ping.pcap", gen_normal_icmp_ping, "正常 ICMP Ping"),
    ]

    # ── 攻擊流量（5 個 PCAP，對應 CICIDS2017 標籤）──
    attack_generators = [
        ("attack_01_ddos.pcap", gen_attack_ddos, "DDoS (SYN Flood + HTTP Flood)"),
        ("attack_02_portscan.pcap", gen_attack_portscan, "PortScan (100 ports)"),
        ("attack_03_ssh_bruteforce.pcap", gen_attack_ssh_bruteforce, "SSH-Patator (Brute Force)"),
        ("attack_04_web_bruteforce.pcap", gen_attack_web_bruteforce, "Web Attack - Brute Force"),
        ("attack_05_dos_hulk.pcap", gen_attack_dos_hulk, "DoS Hulk (HTTP Flood)"),
    ]

    all_gen = normal_generators + attack_generators

    print("=" * 60)
    print("  CICIDS2017 Demo PCAP Generator")
    print(f"  Output: {OUT_DIR}")
    print("=" * 60)

    for fname, gen_func, desc in all_gen:
        pkts = gen_func()
        path = os.path.join(OUT_DIR, fname)
        wrpcap(path, pkts)
        size_kb = os.path.getsize(path) / 1024
        print(f"  [{desc}] {fname}")
        print(f"    Packets: {len(pkts)}, Size: {size_kb:.1f} KB")

    print("\n" + "=" * 60)
    print(f"  Total: {len(all_gen)} PCAP files")
    print("=" * 60)


if __name__ == "__main__":
    main()
