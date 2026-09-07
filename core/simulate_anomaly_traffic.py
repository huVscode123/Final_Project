# ============================================================
# simulate_anomaly_traffic.py - 隨機異常封包模擬產生器 (v1.0.0)
#
# 功能：
#   以 config.py 內既有的告警閾值 (ALERT_THRESHOLD_*) 為基準，
#   隨機產生六種異常流量（SYN Flood / Port Scan / ICMP Flood /
#   UDP Flood / ARP Spoofing / DNS Amplification），每次執行的
#   來源 IP、目標 Port、封包數量、封包間隔時間、Payload 內容都會
#   隨機變化（可用 --seed 固定亂數以利重現）。
#
#   產生完成後可選擇立即透過 PacketParser + AnomalyDetector
#   進行「自我驗證」，確認產生的封包確實會觸發對應的告警規則，
#   避免產生出偵測器根本抓不到的無效測試資料。
#
# 執行方式（於「專案根目錄」或「core 目錄」皆可執行）：
#   python core/simulate_anomaly_traffic.py --verify
#   python core/simulate_anomaly_traffic.py --types syn_flood,port_scan --seed 42
#   python core/simulate_anomaly_traffic.py --types all --scale 2.0 --verify
#
# 輸出：
#   output/sessions/<時間戳記>_simulate/<attack>.pcap   （個別攻擊）
#   output/sessions/<時間戳記>_simulate/mixed_attacks.pcap （混合流量）
# ============================================================

import os
import sys
import time
import random
import argparse

# ── [路徑修正] ────────────────────────────────────────────
# 這個檔案放在 core/ 內，但使用者可能從專案根目錄、core/ 目錄，
# 甚至其他工作目錄執行本腳本。若不處理路徑，會發生兩種常見錯誤：
#   1. ModuleNotFoundError: No module named 'config'
#      （因為 config.py / parser.py / anomaly_detector.py 都是
#       用「同層扁平 import」，必須確保 core/ 在 sys.path 裡）
#   2. 輸出的 output/sessions/... 資料夾跑到奇怪的地方
#      （因為 config.py 內的路徑如 "./output" 是相對路徑，
#       會受目前工作目錄影響）
#
# 解法：無論從哪裡執行，一律將「本檔案所在目錄」加入 sys.path，
# 並把目前工作目錄切換到「core 的上一層」（專案根目錄），
# 這樣不管使用者在哪個目錄下 `python .../simulate_anomaly_traffic.py`
# 都能正確 import，也能與 capture.py / storage.py / session_manager.py
# 產生的 output/ 資料夾維持在同一個地方。
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CORE_DIR)

if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)



from scapy.all import wrpcap                                    # noqa: E402
from scapy.layers.inet import IP, TCP, UDP, ICMP                # noqa: E402
from scapy.layers.l2 import Ether, ARP                          # noqa: E402
from scapy.layers.dns import DNS, DNSQR, DNSRR                  # noqa: E402
from colorama import Fore, Style, init                          # noqa: E402

init(autoreset=True)

from config import (                                            # noqa: E402
    ALERT_THRESHOLD_SYN,
    ALERT_THRESHOLD_PORTS,
    ALERT_THRESHOLD_ICMP,
    ALERT_THRESHOLD_UDP,
)
from parser import PacketParser                                 # noqa: E402
from anomaly_detector import AnomalyDetector                     # noqa: E402

try:
    from session_manager import SessionManager
    HAS_SESSION_MANAGER = True
except ImportError:
    HAS_SESSION_MANAGER = False


# ============================================================
#  隨機資源池
# ============================================================
# RFC 5737 文件／測試保留網段（192.0.2.0/24, 198.51.100.0/24,
# 203.0.113.0/24）：這些網段官方保留給文件與測試使用，不會
# 真實路由到網路上，很適合拿來當作模擬的「外部攻擊來源」，
# 不會誤植成真實存在的公司或個人 IP。
ATTACKER_POOL = ["192.0.2.", "198.51.100.", "203.0.113."]
VICTIM_POOL   = ["192.168.1.", "192.168.0.", "10.0.0."]
DNS_SERVER_POOL = ["8.8.8.", "1.1.1.", "9.9.9."]

WELL_KNOWN_PORTS = [
    21, 22, 23, 25, 53, 80, 110, 135, 139, 143,
    443, 445, 1433, 1521, 3306, 3389, 5432, 5900, 8080, 8443,
    27017, 6379, 9200, 11211, 1900,
]

SCAN_STYLES = [
    ("S",   "SYN Scan"),
    (0x000, "NULL Scan"),
    (0x029, "XMAS Scan"),   # FIN + PSH + URG
    ("A",   "ACK Scan"),
    ("F",   "FIN Scan"),
]


def rand_ip(pool):
    return f"{random.choice(pool)}{random.randint(1, 254)}"


def rand_mac():
    # 第一個位元組使用「本地管理位元」開頭 (02/06/0a/0e)，
    # 避免與真實廠商 OUI 混淆。
    first = random.choice([0x02, 0x06, 0x0a, 0x0e])
    rest = [random.randint(0, 255) for _ in range(5)]
    return ":".join(f"{b:02x}" for b in [first] + rest)


def eth_wrap(pkt):
    """
    在 IP 層封包外面包一層隨機 Ethernet Header。

    原因：ARP 封包本身是 Ether()/ARP() 建構（Link Layer），而
    IP()/TCP()/UDP()/ICMP() 若不加 Ether()，scapy 預設會把它們
    視為「裸 IP」封包（Raw IP Link Layer）。把兩種 linktype 混在
    同一個 pcap 檔內，wrpcap 會印出
    "WARNING: Inconsistent linktypes detected!" 且部分分析工具
    重新讀取後可能誤判封包內容。
    這裡統一幫所有 IP 層封包加上 Ethernet Header，維持全檔案
    linktype 一致，也讓 parser.py 的 _parse_ethernet()
    能正確填入 src_mac / dst_mac 欄位，更貼近真實擷取的封包。
    """
    return Ether(src=rand_mac(), dst=rand_mac()) / pkt


def bulk_scale(threshold, extra_min, extra_max, scale):
    """
    計算「保證超過偵測閾值」的隨機封包數量。

    重要：core/anomaly_detector.py 使用的 ALERT_THRESHOLD_* 是
    config.py 內的固定值，不會因為本腳本的 --scale 參數而改變。
    如果直接把 threshold 也乘上 scale（例如 scale=0.3 時閾值變成
    30），當 scale < 1 時算出來的封包數就可能低於「真正」的閾值
    （還是 100），導致產生出來的流量反而不會觸發告警。

    正確做法：threshold 保持原值不變，只把「超出閾值的緩衝空間」
    乘上 scale，scale 越大代表流量強度越高（緩衝越多），
    scale 越小則緩衝越少，但仍保證 count > threshold。
    """
    lo = threshold + max(1, int(extra_min * max(scale, 0.1)))
    hi = threshold + max(lo - threshold + 1, int(extra_max * max(scale, 0.1)))
    return random.randint(lo, hi)


def jitter_times(packets, mean_gap=0.01, jitter=0.5, start=None):
    """
    為封包序列指定隨機遞增的時間戳。

    Scapy 若未指定 pkt.time，wrpcap 會用「目前系統時間」寫入每個封包，
    導致整批封包時間幾乎一致，不像真實流量。這裡用常態分佈亂數模擬
    封包間隔（mean_gap 為平均間隔秒數，jitter 為相對抖動比例），
    讓輸出的 pcap 時間序列更接近真實擷取狀況。
    """
    t = start if start is not None else time.time()
    for p in packets:
        gap = max(0.0002, random.gauss(mean_gap, mean_gap * jitter))
        t += gap
        p.time = t
    return packets


# ============================================================
#  六種異常流量產生器
#  每個函式回傳 (packets: list, meta: dict)
# ============================================================
def gen_syn_flood(scale=1.0):
    """
    SYN Flood。

    注意：core/anomaly_detector.py 的 SYN Flood 規則是以
    「單一來源 IP 的 SYN 封包數」計數（syn_count[src_ip]）。
    因此若每個封包都隨機換一個來源 IP（完全偽造），detector 永遠
    不會針對單一 IP 累積超過閾值，規則就不會觸發。
    這裡改用「少量（1~3 個）隨機攻擊來源，各自發送大量 SYN」的方式，
    IP／Port／數量仍是隨機的，但符合 detector 實際偵測的邏輯。
    """
    victim = rand_ip(VICTIM_POOL)
    attackers = [rand_ip(ATTACKER_POOL) for _ in range(random.randint(1, 3))]

    pkts = []
    for atk in attackers:
        count = bulk_scale(ALERT_THRESHOLD_SYN, 10, 80, scale)
        for _ in range(count):
            pkts.append(eth_wrap(
                IP(src=atk, dst=victim)
                / TCP(sport=random.randint(1024, 65535), dport=80,
                      flags="S", seq=random.randint(0, 2**32 - 1))
            ))

    # 混入少量正常流量，增加真實感（不影響 SYN 計數，因為 flags 不是純 S）
    for _ in range(random.randint(3, 8)):
        legit = rand_ip(VICTIM_POOL)
        pkts.append(eth_wrap(
            IP(src=legit, dst=victim) / TCP(dport=80, flags="PA")
            / b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
        ))

    random.shuffle(pkts)
    jitter_times(pkts, mean_gap=0.003, jitter=0.6)
    return pkts, {"attackers": attackers, "victim": victim, "packet_count": len(pkts)}


def gen_port_scan(scale=1.0):
    victim = rand_ip(VICTIM_POOL)
    attacker = rand_ip(ATTACKER_POOL)

    n_ports = bulk_scale(ALERT_THRESHOLD_PORTS, 5, 40, scale)

    n_known = min(len(WELL_KNOWN_PORTS), n_ports // 2)
    ports = random.sample(WELL_KNOWN_PORTS, k=n_known)
    remaining = n_ports - len(ports)
    if remaining > 0:
        # 補上隨機高位 Port，避免只掃常見服務
        ports_set = set(ports)
        candidates = [p for p in range(1, 65536) if p not in ports_set]
        ports += random.sample(candidates, k=min(remaining, len(candidates)))
    random.shuffle(ports)

    style_flags, style_name = random.choice(SCAN_STYLES)

    pkts = []
    for port in ports:
        pkts.append(eth_wrap(IP(src=attacker, dst=victim) / TCP(dport=port, flags=style_flags)))
        if random.random() < 0.5:   # 隨機加入目標的 RST 回應（Port Closed）
            pkts.append(eth_wrap(IP(src=victim, dst=attacker) / TCP(sport=port, flags="R")))

    jitter_times(pkts, mean_gap=0.01, jitter=0.5)
    return pkts, {
        "attacker": attacker, "victim": victim,
        "ports_scanned": len(ports), "scan_style": style_name,
    }


def gen_icmp_flood(scale=1.0):
    victim = rand_ip(VICTIM_POOL)
    attacker = rand_ip(ATTACKER_POOL)
    count = bulk_scale(ALERT_THRESHOLD_ICMP, 10, 100, scale)

    pkts = []
    for i in range(count):
        payload_len = random.choice([16, 32, 56, 64, 128])
        pkts.append(eth_wrap(
            IP(src=attacker, dst=victim, id=random.randint(0, 65535))
            / ICMP(type=8, code=0, id=random.randint(0, 65535), seq=i)
            / os.urandom(payload_len)
        ))

    jitter_times(pkts, mean_gap=0.005, jitter=0.5)
    return pkts, {"attacker": attacker, "victim": victim, "packet_count": count}


def gen_udp_flood(scale=1.0):
    victim = rand_ip(VICTIM_POOL)
    attacker = rand_ip(ATTACKER_POOL)
    count = bulk_scale(ALERT_THRESHOLD_UDP, 20, 200, scale)

    pkts = []
    for _ in range(count):
        payload_len = random.choice([64, 128, 256, 512, 1024])
        pkts.append(eth_wrap(
            IP(src=attacker, dst=victim)
            / UDP(sport=random.randint(1024, 65535), dport=random.randint(1, 65535))
            / os.urandom(payload_len)
        ))

    jitter_times(pkts, mean_gap=0.001, jitter=0.5)
    return pkts, {"attacker": attacker, "victim": victim, "packet_count": count}


def gen_arp_spoof(scale=1.0):
    gateway_ip = "192.168.1.1"
    victim_ip = rand_ip(["192.168.1."])
    legit_mac = rand_mac()
    attacker_mac = rand_mac()
    n_spoof = random.randint(2, 8)

    pkts = [
        # 合法的閘道 ARP 公告
        Ether(src=legit_mac, dst="ff:ff:ff:ff:ff:ff")
        / ARP(op=1, psrc=gateway_ip, pdst=victim_ip, hwsrc=legit_mac)
    ]
    for _ in range(n_spoof):
        # 攻擊者偽造「我才是閘道」的 ARP 回應
        pkts.append(
            Ether(src=attacker_mac, dst="ff:ff:ff:ff:ff:ff")
            / ARP(op=2, psrc=gateway_ip, pdst=victim_ip, hwsrc=attacker_mac)
        )

    tail = pkts[1:]
    random.shuffle(tail)
    pkts = pkts[:1] + tail
    jitter_times(pkts, mean_gap=0.05, jitter=0.6)
    return pkts, {
        "gateway_ip": gateway_ip, "victim_ip": victim_ip,
        "legit_mac": legit_mac, "attacker_mac": attacker_mac,
    }


def gen_dns_amplification(scale=1.0):
    """
    DNS 放大攻擊模擬。

    偵測邏輯：core/anomaly_detector.py 從受害者視角檢查——
    當某個 IP 收到大量 DNS 回應（dns_resp[dst_ip]）但幾乎沒有
    發出查詢（dns_req[dst_ip]），且回應數 > 20，就會觸發告警。
    因此關鍵是讓 victim_ip 收到足量的 DNS 回應封包。
    """
    victim_ip = rand_ip(VICTIM_POOL)     # 被偽造來源的受害者
    dns_server = rand_ip(DNS_SERVER_POOL)

    req_count = random.randint(1, 5)
    resp_count = random.randint(max(21, req_count * 11), max(21, req_count * 11) + 80)

    pkts = []
    for i in range(req_count):
        pkts.append(eth_wrap(
            IP(src=victim_ip, dst=dns_server)
            / UDP(sport=random.randint(1024, 65535), dport=53)
            / DNS(rd=1, qd=DNSQR(qname=f"amplify{i}.example.com", qtype=255))  # 255 = ANY
        ))
    for i in range(resp_count):
        qname = f"amplify{i % max(req_count, 1)}.example.com"
        pkts.append(eth_wrap(
            IP(src=dns_server, dst=victim_ip)
            / UDP(sport=53, dport=random.randint(1024, 65535))
            / DNS(qr=1, qd=DNSQR(qname=qname),
                  an=DNSRR(rrname=qname, rdata=rand_ip(VICTIM_POOL)))
        ))

    random.shuffle(pkts)
    jitter_times(pkts, mean_gap=0.01, jitter=0.5)
    return pkts, {
        "victim_ip": victim_ip, "dns_server": dns_server,
        "request_count": req_count, "response_count": resp_count,
    }



def gen_normal_traffic(scale=1.0):
    """
    產生隨機正常流量（HTTP / DNS 查詢 / ICMP Echo / TLS 混合）。

    與其他六個 gen_* 函式風格一致：來源 IP 隨機取自私有／文件保留
    網段、封包間隔加入抖動、回傳 (packets, meta)。刻意「不」呼叫
    bulk_scale()，因為正常流量不需要保證超過任何門檻 —— 它的設計
    目的正好相反：不管 scale 怎麼調，都不應該讓 AnomalyDetector
    觸發任何告警。

    scale 在這裡單純控制「產生幾個封包」。
    """
    count = max(5, int(30 * max(scale, 0.1)))
    pkts = []
    for i in range(count):
        variant = i % 4
        src = rand_ip(VICTIM_POOL)
        if variant == 0:
            pkts.append(eth_wrap(
                IP(src=src, dst="93.184.216.34")
                / TCP(sport=random.randint(1024, 65535), dport=80, flags="PA")
                / b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"
            ))
        elif variant == 1:
            pkts.append(eth_wrap(
                IP(src=src, dst=rand_ip(DNS_SERVER_POOL))
                / UDP(sport=random.randint(1024, 65535), dport=53)
                / DNS(rd=1, qd=DNSQR(qname="example.com"))
            ))
        elif variant == 2:
            pkts.append(eth_wrap(
                IP(src=src, dst=rand_ip(DNS_SERVER_POOL))
                / ICMP(type=8, code=0, id=random.randint(0, 65535), seq=i)
            ))
        else:
            pkts.append(eth_wrap(
                IP(src=src, dst="93.184.216.34")
                / TCP(sport=random.randint(1024, 65535), dport=443, flags="PA")
                / b"\x16\x03\x03\x00\x05\x01\x00\x00\x01\x00"
            ))

    jitter_times(pkts, mean_gap=0.02, jitter=0.6)
    return pkts, {"packet_count": len(pkts)}


GENERATORS = {
    "syn_flood":         ("SYN Flood",         gen_syn_flood),
    "port_scan":         ("Port Scan",         gen_port_scan),
    "icmp_flood":        ("ICMP Flood",        gen_icmp_flood),
    "udp_flood":         ("UDP Flood",         gen_udp_flood),
    "arp_spoof":         ("ARP Spoofing",      gen_arp_spoof),
    "dns_amplification": ("DNS Amplification", gen_dns_amplification),
    "normal_traffic":    ("Normal Traffic",    gen_normal_traffic),
}


# ============================================================
#  自我驗證：把產生的封包餵給 PacketParser + AnomalyDetector，
#  確認確實會觸發對應的告警（避免產生出偵測不到的無效測資）
# ============================================================
def verify_packets(key, pkts):
    """把產生的封包餵給 PacketParser + AnomalyDetector，確認：
        - 六種攻擊類型：確實觸發對應告警。
        - normal_traffic：確實「不」觸發任何告警（規則引擎的誠實
          基準線，也是 simulation_api() 拿來算 CNN 分數基準線時
          期待的輸入型態）。
    """
    parser = PacketParser()
    detector = AnomalyDetector(on_alert=lambda a: None)  # 靜音，只看回傳結果

    triggered_types = set()
    for pkt in pkts:
        record = parser.parse(pkt)
        alerts = detector.inspect(pkt, record)
        for a in alerts:
            triggered_types.add(a["attack_type"])

    if key == "normal_traffic":
        matched = (len(triggered_types) == 0)
        return matched, triggered_types

    expect_substr = {
        "syn_flood": "SYN Flood",
        "port_scan": "Port Scan",
        "icmp_flood": "ICMP",
        "udp_flood": "UDP Flood",
        "arp_spoof": "ARP Spoofing",
        "dns_amplification": "DNS Amplification",
    }[key]

    matched = any(expect_substr in t for t in triggered_types)
    return matched, triggered_types


if __name__ == "__main__":
    main()
