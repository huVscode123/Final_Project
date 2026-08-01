# ============================================================
# core/tests/test_anomaly.py - AnomalyDetector 單元測試
#
# 執行方式:
#   pytest core/tests/test_anomaly.py -v
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import Ether, ARP
from scapy.layers.dns import DNS, DNSQR, DNSRR

from anomaly_detector import AnomalyDetector
from parser import PacketParser


@pytest.fixture
def detector():
    """每個測試建立獨立的 AnomalyDetector（較低閾值便於測試）"""
    return AnomalyDetector(
        on_alert=lambda a: None,   # 靜音，不印出
        threshold_syn=5,
        threshold_ports=3,
        threshold_icmp=5,
        threshold_udp=10,
    )


@pytest.fixture
def parser():
    return PacketParser()


# ────────────────────────────────────────────────────────────
# 1. SYN Flood
# ────────────────────────────────────────────────────────────
class TestSynFlood:

    def test_syn_flood_triggered(self, detector, parser):
        """連續 SYN 封包超過閾值 -> 觸發 SYN Flood 告警"""
        triggered = []

        def capture_alert(a):
            triggered.append(a)

        detector.on_alert = capture_alert
        src = "10.0.0.1"

        for _ in range(6):   # 閾值 5
            pkt = IP(src=src, dst="192.168.1.1") / TCP(dport=80, flags="S")
            record = parser.parse(pkt)
            detector.inspect(pkt, record)

        assert len(triggered) == 1
        assert triggered[0]["attack_type"] == "SYN Flood"
        assert triggered[0]["src_ip"] == src

    def test_syn_ack_not_counted(self, detector, parser):
        """SYN+ACK（握手回應）不應計入 SYN Flood"""
        for _ in range(10):
            pkt = IP(src="10.0.0.1", dst="192.168.1.1") / TCP(flags="SA")
            record = parser.parse(pkt)
            detector.inspect(pkt, record)
        assert len(detector.alert_history) == 0

    def test_no_duplicate_alert(self, detector, parser):
        """同一 IP 只觸發一次 SYN Flood 告警（防重複）"""
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)

        for _ in range(20):
            pkt = IP(src="10.0.0.2", dst="192.168.1.1") / TCP(dport=80, flags="S")
            record = parser.parse(pkt)
            detector.inspect(pkt, record)

        assert len(triggered) == 1

    def test_different_ips_separate_alerts(self, detector, parser):
        """不同 IP 各自計算，各觸發各自的告警"""
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)

        for ip in ["1.1.1.1", "2.2.2.2"]:
            for _ in range(6):
                pkt = IP(src=ip, dst="192.168.1.1") / TCP(dport=80, flags="S")
                record = parser.parse(pkt)
                detector.inspect(pkt, record)

        syn_alerts = [a for a in triggered if a["attack_type"] == "SYN Flood"]
        assert len(syn_alerts) == 2


# ────────────────────────────────────────────────────────────
# 2. Port Scan
# ────────────────────────────────────────────────────────────
class TestPortScan:

    def test_port_scan_triggered(self, detector, parser):
        """掃描超過閾值數量的 Port -> 觸發 Port Scan 告警"""
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)

        for port in [22, 23, 80, 443]:   # 閾值 3，第 4 個觸發
            pkt = (IP(src="10.0.0.3", dst="192.168.1.1")
                   / TCP(dport=port, flags="S"))
            record = parser.parse(pkt)
            detector.inspect(pkt, record)

        scan_alerts = [a for a in triggered if "Port Scan" in a["attack_type"]]
        assert len(scan_alerts) == 1

    def test_port_scan_details_contain_ports(self, detector, parser):
        """告警詳細資訊應包含掃描的 Port"""
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)

        for port in [21, 22, 23, 25]:
            pkt = (IP(src="10.0.0.4", dst="192.168.1.1")
                   / TCP(dport=port, flags="S"))
            record = parser.parse(pkt)
            detector.inspect(pkt, record)

        assert len(triggered) >= 1
        assert "21" in triggered[0]["detail"] or "port" in triggered[0]["detail"].lower()

    def test_null_scan_classification(self, detector, parser):
        """TCP flags=NONE -> 識別為 NULL Scan"""
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)

        for port in [21, 22, 23, 80]:
            pkt = (IP(src="10.0.0.5", dst="192.168.1.1")
                   / TCP(dport=port, flags=0x000))
            record = parser.parse(pkt)
            detector.inspect(pkt, record)

        scan_alerts = [a for a in triggered if "Port Scan" in a["attack_type"]]
        assert any("NULL" in a["attack_type"] for a in scan_alerts)


# ────────────────────────────────────────────────────────────
# 3. ICMP Flood
# ────────────────────────────────────────────────────────────
class TestIcmpFlood:

    def test_icmp_flood_triggered(self, detector, parser):
        """大量 ICMP Echo Request -> 觸發 ICMP Flood"""
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)

        for _ in range(6):
            pkt = IP(src="10.1.1.1", dst="192.168.1.1") / ICMP(type=8)
            record = parser.parse(pkt)
            detector.inspect(pkt, record)

        icmp_alerts = [a for a in triggered if "ICMP" in a["attack_type"]]
        assert len(icmp_alerts) == 1
        assert icmp_alerts[0]["severity"] == "MEDIUM"

    def test_icmp_echo_reply_not_counted(self, detector, parser):
        """ICMP Echo Reply（type=0）不計入 Flood"""
        for _ in range(10):
            pkt = IP(src="10.1.1.2", dst="192.168.1.1") / ICMP(type=0)
            record = parser.parse(pkt)
            detector.inspect(pkt, record)

        icmp_alerts = [a for a in detector.alert_history if "ICMP" in a["attack_type"]]
        assert len(icmp_alerts) == 0


# ────────────────────────────────────────────────────────────
# 4. UDP Flood
# ────────────────────────────────────────────────────────────
class TestUdpFlood:

    def test_udp_flood_triggered(self, detector, parser):
        """大量 UDP 封包 -> 觸發 UDP Flood"""
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)

        for port in range(1, 12):   # 閾值 10
            pkt = IP(src="10.2.2.2", dst="192.168.1.1") / UDP(dport=port)
            record = parser.parse(pkt)
            detector.inspect(pkt, record)

        udp_alerts = [a for a in triggered if "UDP" in a["attack_type"]]
        assert len(udp_alerts) == 1
        assert udp_alerts[0]["severity"] == "HIGH"


# ────────────────────────────────────────────────────────────
# 5. ARP Spoofing
# ────────────────────────────────────────────────────────────
class TestArpSpoofing:

    def test_arp_spoof_triggered_on_second_mac(self, detector, parser):
        """同一 IP 出現第二個 MAC -> 觸發 ARP Spoofing 告警"""
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)

        # 正常 ARP：IP=192.168.1.100, MAC=aa:bb:cc
        pkt1 = (Ether(src="aa:bb:cc:dd:ee:ff")
                / ARP(op=1, psrc="192.168.1.100", pdst="192.168.1.1",
                      hwsrc="aa:bb:cc:dd:ee:ff"))
        r1 = parser.parse(pkt1)
        detector.inspect(pkt1, r1)
        assert len(triggered) == 0

        # 欺騙 ARP：同一 IP，不同 MAC
        pkt2 = (Ether(src="11:22:33:44:55:66")
                / ARP(op=2, psrc="192.168.1.100", pdst="192.168.1.1",
                      hwsrc="11:22:33:44:55:66"))
        r2 = parser.parse(pkt2)
        detector.inspect(pkt2, r2)

        spoof_alerts = [a for a in triggered if "ARP" in a["attack_type"]]
        assert len(spoof_alerts) == 1
        assert spoof_alerts[0]["severity"] == "CRITICAL"
        assert "192.168.1.100" in spoof_alerts[0]["src_ip"]

    def test_same_mac_no_alert(self, detector, parser):
        """同一 IP 同一 MAC 多次 -> 不觸發告警"""
        for _ in range(5):
            pkt = (Ether(src="aa:bb:cc:dd:ee:ff")
                   / ARP(op=1, psrc="192.168.1.200", pdst="192.168.1.1",
                         hwsrc="aa:bb:cc:dd:ee:ff"))
            r = parser.parse(pkt)
            detector.inspect(pkt, r)

        assert len(detector.alert_history) == 0


# ────────────────────────────────────────────────────────────
# 6. DNS Amplification
# ────────────────────────────────────────────────────────────
class TestDnsAmplification:

    def test_dns_amp_triggered_on_high_ratio(self, detector, parser):
        """DNS 回應遠多於查詢 -> 觸發 DNS 放大攻擊告警"""
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)

        src = "8.8.8.8"  # 模擬攻擊者偽造來源

        # 1 筆查詢
        pkt_q = (IP(src=src, dst="192.168.1.1")
                 / UDP(sport=12345, dport=53)
                 / DNS(rd=1, qd=DNSQR(qname="example.com")))
        r_q = parser.parse(pkt_q)
        detector.inspect(pkt_q, r_q)

        # 25 筆回應（高比例）
        for _ in range(25):
            pkt_r = (IP(src=src, dst="192.168.1.1")
                     / UDP(sport=53, dport=12345)
                     / DNS(qr=1, qd=DNSQR(qname="example.com"),
                           an=DNSRR(rrname="example.com", rdata="93.184.216.34")))
            r_r = parser.parse(pkt_r)
            detector.inspect(pkt_r, r_r)

        dns_alerts = [a for a in triggered if "DNS" in a["attack_type"]]
        assert len(dns_alerts) >= 1

    def test_normal_dns_no_alert(self, detector, parser):
        """正常 DNS 查詢/回應 比例 -> 不觸發"""
        for i in range(5):
            pkt_q = (IP(src="192.168.1.2", dst="8.8.8.8")
                     / UDP(sport=50000+i, dport=53)
                     / DNS(rd=1, qd=DNSQR(qname=f"test{i}.com")))
            r = parser.parse(pkt_q)
            detector.inspect(pkt_q, r)

            pkt_r = (IP(src="8.8.8.8", dst="192.168.1.2")
                     / UDP(sport=53, dport=50000+i)
                     / DNS(qr=1, qd=DNSQR(qname=f"test{i}.com"),
                           an=DNSRR(rrname=f"test{i}.com", rdata="1.2.3.4")))
            r2 = parser.parse(pkt_r)
            detector.inspect(pkt_r, r2)

        dns_alerts = [a for a in detector.alert_history if "DNS" in a["attack_type"]]
        assert len(dns_alerts) == 0


# ────────────────────────────────────────────────────────────
# 7. Reset & Summary
# ────────────────────────────────────────────────────────────
class TestDetectorUtility:

    def test_reset_clears_counters(self, detector, parser):
        """reset() 後計數器歸零，冷卻狀態清空"""
        for _ in range(6):
            pkt = IP(src="1.1.1.1", dst="2.2.2.2") / TCP(dport=80, flags="S")
            detector.inspect(pkt, parser.parse(pkt))

        assert len(detector.alert_history) > 0
        detector.reset()

        # [更新] 使用滑動窗口計數器的 count() 方法驗證歸零
        assert detector.syn_counter.count("1.1.1.1") == 0
        assert len(detector._alert_cooldown) == 0
        # alert_history 保留（不清空）
        assert len(detector.alert_history) > 0

    def test_get_summary_structure(self, detector, parser):
        """get_summary() 回傳的 dict 包含所有預期 key"""
        pkt = IP(src="1.2.3.4", dst="5.6.7.8") / TCP(dport=80, flags="S")
        detector.inspect(pkt, parser.parse(pkt))

        summary = detector.get_summary()
        for key in ["total_alerts", "syn_top_ip", "icmp_top_ip",
                    "udp_top_ip", "port_scan_ips", "arp_spoof_ips"]:
            assert key in summary

    def test_alert_severity_levels(self, detector, parser):
        """不同攻擊類型應有對應的嚴重程度"""
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)

        # SYN Flood -> HIGH
        for _ in range(6):
            pkt = IP(src="1.1.1.1", dst="2.2.2.2") / TCP(dport=80, flags="S")
            detector.inspect(pkt, parser.parse(pkt))

        # ARP Spoof -> CRITICAL
        pkt1 = Ether(src="aa:aa:aa:aa:aa:aa") / ARP(
            op=1, psrc="10.0.0.1", pdst="10.0.0.2", hwsrc="aa:aa:aa:aa:aa:aa")
        detector.inspect(pkt1, parser.parse(pkt1))
        pkt2 = Ether(src="bb:bb:bb:bb:bb:bb") / ARP(
            op=2, psrc="10.0.0.1", pdst="10.0.0.2", hwsrc="bb:bb:bb:bb:bb:bb")
        detector.inspect(pkt2, parser.parse(pkt2))

        severities = {a["attack_type"]: a["severity"] for a in triggered}
        assert severities.get("SYN Flood") == "HIGH"
        assert severities.get("ARP Spoofing") == "CRITICAL"
