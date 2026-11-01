#!/usr/bin/env python3
# ============================================================
# generate_test_pcaps.py
# 為四個模型產生輕量級測試用 PCAP (正常 + 異常)
# ============================================================
"""
使用方式：
    python generate_test_pcaps.py

產生結構：
    test_pcaps/
        cicids2017_unsupervised/
            normal_http_dns.pcap        正常 HTTP + DNS
            attack_portscan_bruteforce.pcap  Port Scan + SSH Brute Force
        cicids2017_semi/
            normal_http_dns.pcap
            attack_portscan_bruteforce.pcap
        mta_unsupervised/
            normal_http_dns.pcap        正常日常流量
            attack_malware_c2.pcap      惡意 C2 回連 + 異常 DNS
        mta_semi/
            normal_http_dns.pcap
            attack_malware_c2.pcap

同時從 PCAPs/ 資料夾擷取真實攻擊封包，合併到異常 PCAP 中。
"""
import os
import sys
import shutil
import random

# 確保 core/ 在 path 中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
CORE_DIR = os.path.join(PROJECT_ROOT, 'core')
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from scapy.all import (
    Ether, IP, TCP, UDP, DNS, DNSQR, DNSRR, Raw, ICMP,
    ARP, RandShort, wrpcap, rdpcap, conf
)
conf.verb = 0

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'test_pcaps')

# ── Malcolm-PCAP 中可用的真實攻擊封包 ──────────────────────
MALCOLM_BASE = os.path.join(PROJECT_ROOT, 'PCAPs', 'Malcolm-PCAP-main', 'pcaps')

# 攻擊類型 PCAP (Port Scan / Exploit)
REAL_ATTACK_PCAPS = [
    os.path.join(MALCOLM_BASE, 'plugins', 'bro-simple-scan', 'port_scan.pcap'),
    os.path.join(MALCOLM_BASE, 'plugins', 'bro-simple-scan', 'multiport_scan.pcap'),
    os.path.join(MALCOLM_BASE, 'plugins', 'CVE-2021-41773', 'apache_exploit_success.pcap'),
]

# 惡意軟體類型 PCAP (C2 / Backdoor / Malware)
REAL_MALWARE_PCAPS = [
    os.path.join(MALCOLM_BASE, 'plugins', 'zeek-EternalSafety', 'doublepulsar-backdoor-connect-win7.pcap'),
    os.path.join(MALCOLM_BASE, 'plugins', 'zeek-strrat-detector', 'strrat-4423258f-59bc-4a88-bfec-d8ac08c88538.pcap'),
    os.path.join(MALCOLM_BASE, 'plugins', 'cve-2021-44228', 'log4j-attack.pcap'),
]

# 正常協定 PCAP
REAL_NORMAL_PCAPS = [
    os.path.join(MALCOLM_BASE, 'protocols', 'DHCP.pcap'),
    os.path.join(MALCOLM_BASE, 'protocols', 'MQTT.pcap'),
]


def _rand_ip():
    return f'192.168.{random.randint(1,10)}.{random.randint(2,250)}'


def _rand_public_ip():
    return f'{random.randint(40,200)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}'


def generate_normal_http_dns(n_packets=80):
    """產生正常的 HTTP GET/Response + DNS Query/Response 混合封包。"""
    pkts = []
    client = '192.168.1.100'
    dns_server = '8.8.8.8'
    web_servers = ['93.184.216.34', '151.101.1.69', '172.217.14.206']
    domains = ['www.example.com', 'www.google.com', 'www.github.com',
               'api.example.org', 'cdn.jsdelivr.net']

    for i in range(n_packets):
        if i % 4 == 0:
            # DNS Query
            domain = random.choice(domains)
            pkt = (IP(src=client, dst=dns_server) /
                   UDP(sport=RandShort(), dport=53) /
                   DNS(rd=1, qd=DNSQR(qname=domain)))
            pkts.append(pkt)
        elif i % 4 == 1:
            # DNS Response
            domain = random.choice(domains)
            ws = random.choice(web_servers)
            pkt = (IP(src=dns_server, dst=client) /
                   UDP(sport=53, dport=RandShort()) /
                   DNS(qr=1, aa=1, qd=DNSQR(qname=domain),
                       an=DNSRR(rrname=domain, rdata=ws, ttl=300)))
            pkts.append(pkt)
        elif i % 4 == 2:
            # HTTP GET
            ws = random.choice(web_servers)
            sport = random.randint(49152, 65535)
            paths = ['/', '/index.html', '/api/v1/status', '/assets/style.css', '/favicon.ico']
            payload = f'GET {random.choice(paths)} HTTP/1.1\r\nHost: {random.choice(domains)}\r\nUser-Agent: Mozilla/5.0\r\nAccept: text/html\r\n\r\n'
            pkt = (IP(src=client, dst=ws) /
                   TCP(sport=sport, dport=80, flags='PA', seq=1000+i, ack=2000+i) /
                   Raw(load=payload.encode()))
            pkts.append(pkt)
        else:
            # HTTP Response 200 OK
            ws = random.choice(web_servers)
            body = '<html><body>OK</body></html>'
            payload = f'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {len(body)}\r\n\r\n{body}'
            pkt = (IP(src=ws, dst=client) /
                   TCP(sport=80, dport=random.randint(49152, 65535), flags='PA', seq=2000+i, ack=1000+i) /
                   Raw(load=payload.encode()))
            pkts.append(pkt)

    # 加入少量 ICMP ping (正常行為)
    for _ in range(5):
        pkts.append(IP(src=client, dst=random.choice(web_servers)) / ICMP())
        pkts.append(IP(src=random.choice(web_servers), dst=client) / ICMP(type=0))

    random.shuffle(pkts)
    return pkts


def generate_attack_portscan_bruteforce(n_scan=60, n_brute=30):
    """產生 Port Scan + SSH Brute Force 攻擊封包 (CICIDS2017 常見攻擊類型)。"""
    pkts = []
    attacker = '10.0.0.50'
    victim = '192.168.1.200'

    # Port Scan: SYN 掃描大量連續 port
    for port in range(20, 20 + n_scan):
        # SYN
        pkt = (IP(src=attacker, dst=victim) /
               TCP(sport=RandShort(), dport=port, flags='S', seq=random.randint(100000, 999999)))
        pkts.append(pkt)
        # RST 回應 (port closed)
        if random.random() > 0.15:
            pkt = (IP(src=victim, dst=attacker) /
                   TCP(sport=port, dport=pkts[-1][TCP].sport, flags='RA', seq=0, ack=pkts[-1][TCP].seq+1))
            pkts.append(pkt)

    # SSH Brute Force: 對 port 22 快速嘗試
    for i in range(n_brute):
        sport = random.randint(49152, 65535)
        # SYN
        pkts.append(IP(src=attacker, dst=victim) /
                    TCP(sport=sport, dport=22, flags='S'))
        # SYN-ACK
        pkts.append(IP(src=victim, dst=attacker) /
                    TCP(sport=22, dport=sport, flags='SA'))
        # ACK + 短 payload (模擬 SSH 協議交換後立即斷線)
        pkts.append(IP(src=attacker, dst=victim) /
                    TCP(sport=sport, dport=22, flags='PA') /
                    Raw(load=b'SSH-2.0-OpenSSH_8.2\r\n'))
        # RST (被伺服器踢掉)
        pkts.append(IP(src=victim, dst=attacker) /
                    TCP(sport=22, dport=sport, flags='R'))

    return pkts


def generate_attack_malware_c2(n_c2=40, n_dns_tunnel=20):
    """產生惡意軟體 C2 回連 + DNS Tunneling 封包 (MTA 常見特徵)。"""
    pkts = []
    infected_host = '192.168.1.150'
    c2_servers = ['185.234.72.11', '91.215.85.142', '45.33.32.156']
    tunnel_domain = 'data.evil-c2-server.xyz'

    # C2 Beaconing: 定期向 C2 Server 回連
    for i in range(n_c2):
        c2 = random.choice(c2_servers)
        sport = random.randint(49152, 65535)
        c2_ports = [443, 8443, 4444, 8080]
        dport = random.choice(c2_ports)

        # SYN
        pkts.append(IP(src=infected_host, dst=c2) /
                    TCP(sport=sport, dport=dport, flags='S'))
        # SYN-ACK
        pkts.append(IP(src=c2, dst=infected_host) /
                    TCP(sport=dport, dport=sport, flags='SA'))
        # 加密 C2 payload (隨機 bytes 模擬加密通訊)
        payload_len = random.randint(64, 512)
        pkts.append(IP(src=infected_host, dst=c2) /
                    TCP(sport=sport, dport=dport, flags='PA') /
                    Raw(load=os.urandom(payload_len)))
        # C2 指令回傳
        pkts.append(IP(src=c2, dst=infected_host) /
                    TCP(sport=dport, dport=sport, flags='PA') /
                    Raw(load=os.urandom(random.randint(32, 256))))

    # DNS Tunneling: 超長子網域名稱 (Base64 資料外洩)
    import base64
    for i in range(n_dns_tunnel):
        # 模擬用 DNS 偷傳資料
        exfil_data = base64.b64encode(os.urandom(random.randint(20, 60))).decode().replace('=', '')
        query_name = f'{exfil_data[:60]}.{tunnel_domain}'
        pkt = (IP(src=infected_host, dst='8.8.8.8') /
               UDP(sport=RandShort(), dport=53) /
               DNS(rd=1, qd=DNSQR(qname=query_name, qtype='TXT')))
        pkts.append(pkt)

    # ARP Spoofing (MTA 常見的中間人攻擊)
    gateway_ip = '192.168.1.1'
    for _ in range(8):
        pkts.append(
            Ether(src='aa:bb:cc:dd:ee:ff', dst='ff:ff:ff:ff:ff:ff') /
            ARP(op=2, psrc=gateway_ip, hwsrc='aa:bb:cc:dd:ee:ff',
                pdst=infected_host, hwdst='00:11:22:33:44:55')
        )

    return pkts


def extract_packets(pcap_path, max_packets=30):
    """從真實 PCAP 擷取前 N 個封包。"""
    if not os.path.exists(pcap_path):
        print(f'  [SKIP] 找不到: {pcap_path}')
        return []
    try:
        pkts = rdpcap(pcap_path, count=max_packets)
        print(f'  [OK] 從 {os.path.basename(pcap_path)} 擷取 {len(pkts)} 個封包')
        return list(pkts)
    except Exception as e:
        print(f'  [WARN] 讀取失敗 {pcap_path}: {e}')
        return []


def save_pcap(pkts, filepath):
    """存檔並印出大小。"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    wrpcap(filepath, pkts)
    size_kb = os.path.getsize(filepath) / 1024
    print(f'  -> {os.path.relpath(filepath, PROJECT_ROOT)} ({len(pkts)} pkts, {size_kb:.1f} KB)')


def main():
    print('=' * 60)
    print('  產生測試用 PCAP 封包')
    print('  正常流量: Scapy 合成 + Malcolm 協定封包')
    print('  異常流量: Scapy 合成攻擊 + Malcolm 真實攻擊封包')
    print('=' * 60)

    # ── 1. CICIDS2017 非監督式 ──────────────────────────────
    print('\n[1/4] CICIDS2017 非監督式')
    subdir = os.path.join(OUTPUT_DIR, 'cicids2017_unsupervised')

    normal_pkts = generate_normal_http_dns(80)
    # 加入少量真實正常協定封包
    for p in REAL_NORMAL_PCAPS[:1]:
        normal_pkts += extract_packets(p, max_packets=15)
    save_pcap(normal_pkts, os.path.join(subdir, 'normal_http_dns.pcap'))

    attack_pkts = generate_attack_portscan_bruteforce(60, 30)
    # 加入真實 Port Scan 封包
    for p in REAL_ATTACK_PCAPS:
        attack_pkts += extract_packets(p, max_packets=15)
    save_pcap(attack_pkts, os.path.join(subdir, 'attack_portscan_bruteforce.pcap'))

    # ── 2. CICIDS2017 半監督式 ──────────────────────────────
    print('\n[2/4] CICIDS2017 半監督式')
    subdir = os.path.join(OUTPUT_DIR, 'cicids2017_semi')

    normal_pkts = generate_normal_http_dns(80)
    for p in REAL_NORMAL_PCAPS[:1]:
        normal_pkts += extract_packets(p, max_packets=15)
    save_pcap(normal_pkts, os.path.join(subdir, 'normal_http_dns.pcap'))

    attack_pkts = generate_attack_portscan_bruteforce(50, 25)
    for p in REAL_ATTACK_PCAPS:
        attack_pkts += extract_packets(p, max_packets=20)
    save_pcap(attack_pkts, os.path.join(subdir, 'attack_portscan_bruteforce.pcap'))

    # ── 3. MTA 非監督式 ────────────────────────────────────
    print('\n[3/4] MTA 非監督式 (真實惡意軟體偵測)')
    subdir = os.path.join(OUTPUT_DIR, 'mta_unsupervised')

    normal_pkts = generate_normal_http_dns(80)
    for p in REAL_NORMAL_PCAPS:
        normal_pkts += extract_packets(p, max_packets=10)
    save_pcap(normal_pkts, os.path.join(subdir, 'normal_http_dns.pcap'))

    attack_pkts = generate_attack_malware_c2(40, 20)
    # 加入真實惡意軟體封包
    for p in REAL_MALWARE_PCAPS:
        attack_pkts += extract_packets(p, max_packets=20)
    save_pcap(attack_pkts, os.path.join(subdir, 'attack_malware_c2.pcap'))

    # ── 4. MTA 半監督式 ────────────────────────────────────
    print('\n[4/4] MTA 半監督式 (真實惡意軟體偵測)')
    subdir = os.path.join(OUTPUT_DIR, 'mta_semi')

    normal_pkts = generate_normal_http_dns(80)
    for p in REAL_NORMAL_PCAPS:
        normal_pkts += extract_packets(p, max_packets=10)
    save_pcap(normal_pkts, os.path.join(subdir, 'normal_http_dns.pcap'))

    attack_pkts = generate_attack_malware_c2(35, 15)
    for p in REAL_MALWARE_PCAPS:
        attack_pkts += extract_packets(p, max_packets=25)
    save_pcap(attack_pkts, os.path.join(subdir, 'attack_malware_c2.pcap'))

    # ── 摘要 ───────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  產生完成! 檔案位於: test_pcaps/')
    print('=' * 60)
    total_size = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            total_size += sz
            rel = os.path.relpath(fp, PROJECT_ROOT)
            print(f'  {rel:<60} {sz/1024:.1f} KB')
    print(f'\n  Total: {total_size/1024:.1f} KB')
    print()
    print('  使用方式:')
    print('    正常流量 -> 上傳到對應模型，預期結果: 0 或極少異常')
    print('    異常流量 -> 上傳到對應模型，預期結果: 多數封包被標記異常')


if __name__ == '__main__':
    main()
