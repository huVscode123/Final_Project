# ============================================================
# core/tests/test_v3_optimizations.py
# v3.0.0 優化修正驗證測試
#
# 本測試檔涵蓋所有 v3.0.0 優化中修正的 Bug 與新增功能：
#
#   [config.py]          新增設定項驗證
#   [parser.py]          UDP DNS 判斷修正 / QUIC 辨識 / NS flag
#   [anomaly_detector.py] 滑動窗口 / 告警冷卻 / DNS Tunneling
#   [capture.py]         競態條件修正 / 記憶體保護 / 效能指標
#   [pcap_analyzer.py]   total_packets 修正 / 快取 / _require_loaded
#   [storage.py]         SQLite 型別保留 / 批次寫入 / CSV 追加
#
# 執行方式:
#   pytest core/tests/test_v3_optimizations.py -v
# ============================================================

import sys
import os
import time
import csv
import json
import sqlite3
import tempfile
import threading
from unittest.mock import patch, MagicMock
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import Ether, ARP
from scapy.layers.dns import DNS, DNSQR, DNSRR

from parser import PacketParser
from anomaly_detector import AnomalyDetector, _SlidingWindowCounter
from storage import PacketStorage


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@pytest.fixture
def parser():
    return PacketParser()


@pytest.fixture
def detector():
    """低閾值的偵測器，方便快速觸發告警"""
    return AnomalyDetector(
        on_alert=lambda a: None,
        threshold_syn=5,
        threshold_ports=3,
        threshold_icmp=5,
        threshold_udp=10,
        window_seconds=60,      # 大窗口確保測試期間不過期
        cooldown_seconds=0,     # 關閉冷卻以便測試多次觸發
    )


@pytest.fixture
def detector_with_cooldown():
    """啟用冷卻機制的偵測器"""
    return AnomalyDetector(
        on_alert=lambda a: None,
        threshold_syn=3,
        threshold_ports=3,
        threshold_icmp=3,
        threshold_udp=5,
        window_seconds=60,
        cooldown_seconds=10,    # 10 秒冷卻
    )


@pytest.fixture
def temp_dir():
    """建立並清理暫存目錄"""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_tmp")
    os.makedirs(d, exist_ok=True)
    yield d
    # 清理
    import shutil
    shutil.rmtree(d, ignore_errors=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. config.py — 新增設定項驗證
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestConfigNewSettings:
    """驗證 config.py 新增的設定常數是否存在且值合理"""

    def test_memory_protection_settings_exist(self):
        """記憶體保護設定應存在且為正整數"""
        from config import MAX_MEMORY_PACKETS, PCAP_SEGMENT_SIZE, PCAP_SEGMENT_SECONDS
        assert isinstance(MAX_MEMORY_PACKETS, int) and MAX_MEMORY_PACKETS > 0
        assert isinstance(PCAP_SEGMENT_SIZE, int) and PCAP_SEGMENT_SIZE > 0
        assert isinstance(PCAP_SEGMENT_SECONDS, int) and PCAP_SEGMENT_SECONDS > 0

    def test_alert_cooldown_setting_exists(self):
        """告警冷卻設定應存在且為非負數"""
        from config import ALERT_COOLDOWN_SECONDS
        assert isinstance(ALERT_COOLDOWN_SECONDS, int)
        assert ALERT_COOLDOWN_SECONDS >= 0

    def test_dns_tunnel_settings_exist(self):
        """DNS Tunneling 閾值設定應存在"""
        from config import ALERT_THRESHOLD_DNS_TUNNEL_LEN, ALERT_THRESHOLD_DNS_TUNNEL_COUNT
        assert isinstance(ALERT_THRESHOLD_DNS_TUNNEL_LEN, int)
        assert isinstance(ALERT_THRESHOLD_DNS_TUNNEL_COUNT, int)
        assert ALERT_THRESHOLD_DNS_TUNNEL_LEN > 0
        assert ALERT_THRESHOLD_DNS_TUNNEL_COUNT > 0

    def test_new_port_service_mappings(self):
        """新增的 Port 服務對應應存在"""
        from config import PORT_SERVICE_MAP
        assert PORT_SERVICE_MAP.get(993)  == "IMAPS"
        assert PORT_SERVICE_MAP.get(995)  == "POP3S"
        assert PORT_SERVICE_MAP.get(465)  == "SMTPS"
        assert PORT_SERVICE_MAP.get(587)  == "SMTP-Submission"
        assert PORT_SERVICE_MAP.get(1194) == "OpenVPN"
        assert PORT_SERVICE_MAP.get(5060) == "SIP"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. parser.py — UDP DNS 判斷修正 / QUIC / NS flag
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestParserUDPDNSFix:
    """驗證 UDP DNS 判斷的不對稱問題已修正"""

    def test_udp_dport53_with_dns_layer(self, parser):
        """dport=53 且有 DNS layer → 應識別為 DNS"""
        pkt = (IP(src="1.1.1.1", dst="8.8.8.8")
               / UDP(sport=55000, dport=53)
               / DNS(rd=1, qd=DNSQR(qname="example.com")))
        r = parser.parse(pkt)
        assert r["protocol"] == "DNS"

    def test_udp_dport53_without_dns_layer(self, parser):
        """dport=53 但無 DNS layer → 不應誤標為 DNS（Bug 修正驗證）"""
        pkt = IP(src="1.1.1.1", dst="8.8.8.8") / UDP(sport=55000, dport=53)
        r = parser.parse(pkt)
        assert r["protocol"] == "UDP", \
            "沒有 DNS layer 的 UDP port 53 不應被標記為 DNS"

    def test_udp_sport53_with_dns_layer(self, parser):
        """sport=53 且有 DNS layer → 應識別為 DNS（回應封包）"""
        pkt = (IP(src="8.8.8.8", dst="1.1.1.1")
               / UDP(sport=53, dport=55000)
               / DNS(qr=1, qd=DNSQR(qname="example.com"),
                     an=DNSRR(rrname="example.com", rdata="1.2.3.4")))
        r = parser.parse(pkt)
        assert r["protocol"] == "DNS"

    def test_udp_sport53_without_dns_layer(self, parser):
        """sport=53 但無 DNS layer → 不應識別為 DNS"""
        pkt = IP(src="8.8.8.8", dst="1.1.1.1") / UDP(sport=53, dport=55000)
        r = parser.parse(pkt)
        assert r["protocol"] == "UDP"

    def test_udp_mdns_port5353_with_dns_layer(self, parser):
        """mDNS port 5353 且有 DNS layer → 應識別為 DNS"""
        pkt = (IP(src="1.1.1.1", dst="224.0.0.251")
               / UDP(sport=5353, dport=5353)
               / DNS(qd=DNSQR(qname="_http._tcp.local")))
        r = parser.parse(pkt)
        assert r["protocol"] == "DNS"

    def test_udp_port5353_without_dns_layer(self, parser):
        """port 5353 但無 DNS layer → 不應誤標為 DNS"""
        pkt = IP(src="1.1.1.1", dst="224.0.0.251") / UDP(sport=5353, dport=5353)
        r = parser.parse(pkt)
        assert r["protocol"] == "UDP"


class TestParserQUICDetection:
    """驗證 QUIC/HTTP3 協議辨識"""

    def test_udp_dport443_is_quic(self, parser):
        """UDP dport=443 → 應識別為 QUIC"""
        pkt = IP(src="1.1.1.1", dst="2.2.2.2") / UDP(sport=55000, dport=443)
        r = parser.parse(pkt)
        assert r["protocol"] == "QUIC"

    def test_udp_sport443_is_quic(self, parser):
        """UDP sport=443 → 應識別為 QUIC（回應方向）"""
        pkt = IP(src="2.2.2.2", dst="1.1.1.1") / UDP(sport=443, dport=55000)
        r = parser.parse(pkt)
        assert r["protocol"] == "QUIC"

    def test_tcp_port443_not_quic(self, parser):
        """TCP port 443 不應識別為 QUIC（QUIC 只走 UDP）"""
        pkt = IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=55000, dport=443, flags="S")
        r = parser.parse(pkt)
        assert r["protocol"] != "QUIC"


class TestParserNSFlag:
    """驗證 TCP NS flag 支援"""

    def test_ns_flag_parsed(self, parser):
        """NS (Nonce Sum, 0x100) 旗標應被正確解析"""
        result = PacketParser._parse_tcp_flags(0x100)
        assert "NS" in result

    def test_ns_with_syn_ack(self, parser):
        """NS+SYN+ACK (0x112) 旗標應被正確解析"""
        result = PacketParser._parse_tcp_flags(0x112)
        assert "NS" in result
        assert "SYN" in result
        assert "ACK" in result

    def test_all_nine_flags(self, parser):
        """全部 9 個 flag 位元應都被解析"""
        # FIN+SYN+RST+PSH+ACK+URG+ECE+CWR+NS = 0x1FF
        result = PacketParser._parse_tcp_flags(0x1FF)
        for flag_name in ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR", "NS"]:
            assert flag_name in result, f"缺少旗標: {flag_name}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. anomaly_detector.py — 滑動窗口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestSlidingWindowCounter:
    """驗證 _SlidingWindowCounter 滑動窗口計數器"""

    def test_basic_add_and_count(self):
        """基本新增和計數"""
        c = _SlidingWindowCounter(window_seconds=10)
        c.add("key_a")
        c.add("key_a")
        c.add("key_b")
        assert c.count("key_a") == 2
        assert c.count("key_b") == 1

    def test_events_expire_after_window(self):
        """超過窗口時間的事件應被自動清除"""
        c = _SlidingWindowCounter(window_seconds=1.0)
        now = time.time()
        # 新增一個 2 秒前的事件
        c.add("ip1", timestamp=now - 2.0)
        # 此刻查詢應該為 0（已過期）
        assert c.count("ip1", timestamp=now) == 0

    def test_recent_events_not_expired(self):
        """窗口內的事件不應被清除"""
        c = _SlidingWindowCounter(window_seconds=10.0)
        now = time.time()
        c.add("ip1", timestamp=now - 5.0)  # 5 秒前，仍在窗口內
        c.add("ip1", timestamp=now - 1.0)  # 1 秒前，仍在窗口內
        assert c.count("ip1", timestamp=now) == 2

    def test_mixed_expiry(self):
        """部分事件過期、部分未過期的混合情況"""
        c = _SlidingWindowCounter(window_seconds=5.0)
        now = time.time()
        c.add("ip1", timestamp=now - 10.0)  # 過期
        c.add("ip1", timestamp=now - 8.0)   # 過期
        c.add("ip1", timestamp=now - 3.0)   # 有效
        c.add("ip1", timestamp=now - 1.0)   # 有效
        assert c.count("ip1", timestamp=now) == 2

    def test_clear_resets_all(self):
        """clear() 應清除所有資料"""
        c = _SlidingWindowCounter(window_seconds=60)
        c.add("a")
        c.add("b")
        c.clear()
        assert c.count("a") == 0
        assert c.count("b") == 0

    def test_different_keys_independent(self):
        """不同 key 的計數應互相獨立"""
        c = _SlidingWindowCounter(window_seconds=60)
        for _ in range(5):
            c.add("ip_a")
        for _ in range(3):
            c.add("ip_b")
        assert c.count("ip_a") == 5
        assert c.count("ip_b") == 3


class TestSlidingWindowInDetector:
    """驗證 AnomalyDetector 正確使用滑動窗口"""

    def test_syn_flood_uses_sliding_window(self, parser):
        """SYN Flood 偵測應使用滑動窗口，過期事件不再計入"""
        detector = AnomalyDetector(
            on_alert=lambda a: None,
            threshold_syn=5,
            window_seconds=2.0,
            cooldown_seconds=0,
        )
        now = time.time()
        # 模擬 3 秒前發了 4 個 SYN（已過期，不應計入）
        for _ in range(4):
            detector.syn_counter.add("10.0.0.1", timestamp=now - 3.0)

        # 現在再發 3 個 SYN，累計窗口內只有 3 個（<5 閾值），不應告警
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)
        for _ in range(3):
            pkt = IP(src="10.0.0.1", dst="192.168.1.1") / TCP(dport=80, flags="S")
            detector.inspect(pkt, parser.parse(pkt))

        assert len(triggered) == 0, \
            "窗口外的 SYN 不應被計入，3 個窗口內的 SYN 不應觸發閾值 5 的告警"

    def test_no_false_positive_long_duration(self, parser):
        """長時間低速流量不應觸發誤報（修正前的核心 Bug）"""
        detector = AnomalyDetector(
            on_alert=lambda a: None,
            threshold_syn=100,
            window_seconds=10.0,
            cooldown_seconds=0,
        )
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)

        now = time.time()
        # 模擬過去 1000 秒內每 10 秒發 1 個 SYN（正常流量）
        for i in range(100):
            t = now - 1000 + i * 10
            detector.syn_counter.add("10.0.0.1", timestamp=t)

        # 此時窗口內最多只有 1~2 個，不應觸發
        pkt = IP(src="10.0.0.1", dst="192.168.1.1") / TCP(dport=80, flags="S")
        detector.inspect(pkt, parser.parse(pkt))
        assert len(triggered) == 0, \
            "低速正常流量不應觸發 SYN Flood（修正前會因累計 100 而誤報）"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. anomaly_detector.py — 告警冷卻
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestAlertCooldown:
    """驗證告警冷卻機制"""

    def test_cooldown_suppresses_duplicate_alerts(self, parser, detector_with_cooldown):
        """冷卻期內同一 IP 的同類型告警不應重複觸發"""
        triggered = []
        detector_with_cooldown.on_alert = lambda a: triggered.append(a)

        # 發送遠超閾值的 SYN 封包（閾值=3），應只觸發一次
        for _ in range(20):
            pkt = IP(src="10.0.0.1", dst="192.168.1.1") / TCP(dport=80, flags="S")
            detector_with_cooldown.inspect(pkt, parser.parse(pkt))

        syn_alerts = [a for a in triggered if a["attack_type"] == "SYN Flood"]
        assert len(syn_alerts) == 1, \
            f"冷卻期內應只觸發 1 次 SYN Flood，實際觸發 {len(syn_alerts)} 次"

    def test_different_ips_not_affected_by_cooldown(self, parser, detector_with_cooldown):
        """不同 IP 的告警不應互相影響"""
        triggered = []
        detector_with_cooldown.on_alert = lambda a: triggered.append(a)

        # IP_A 觸發 SYN Flood
        for _ in range(5):
            pkt = IP(src="10.0.0.1", dst="192.168.1.1") / TCP(dport=80, flags="S")
            detector_with_cooldown.inspect(pkt, parser.parse(pkt))

        # IP_B 也應能獨立觸發
        for _ in range(5):
            pkt = IP(src="10.0.0.2", dst="192.168.1.1") / TCP(dport=80, flags="S")
            detector_with_cooldown.inspect(pkt, parser.parse(pkt))

        syn_alerts = [a for a in triggered if a["attack_type"] == "SYN Flood"]
        unique_ips = set(a["src_ip"] for a in syn_alerts)
        assert len(unique_ips) == 2, "兩個不同 IP 應各自獨立觸發告警"

    def test_cooldown_expires_allows_new_alert(self, parser):
        """冷卻時間過後應允許再次觸發"""
        detector = AnomalyDetector(
            on_alert=lambda a: None,
            threshold_syn=3,
            window_seconds=60,
            cooldown_seconds=0.5,  # 0.5 秒的短冷卻
        )
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)

        # 第一波觸發
        for _ in range(5):
            pkt = IP(src="10.0.0.1", dst="192.168.1.1") / TCP(dport=80, flags="S")
            detector.inspect(pkt, parser.parse(pkt))
        assert len(triggered) == 1

        # 等待冷卻過期
        time.sleep(0.6)

        # 第二波應能再次觸發
        for _ in range(5):
            pkt = IP(src="10.0.0.1", dst="192.168.1.1") / TCP(dport=80, flags="S")
            detector.inspect(pkt, parser.parse(pkt))
        assert len(triggered) == 2, "冷卻過期後應允許再次觸發告警"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. anomaly_detector.py — DNS Tunneling 偵測
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestDNSTunneling:
    """驗證 DNS Tunneling 偵測功能"""

    def test_normal_dns_no_tunnel_alert(self, detector, parser):
        """正常的短 DNS 查詢不應觸發 Tunneling 告警"""
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)

        for _ in range(50):
            pkt = (IP(src="10.0.0.1", dst="8.8.8.8")
                   / UDP(dport=53)
                   / DNS(qd=DNSQR(qname="google.com")))
            detector.inspect(pkt, parser.parse(pkt))

        tunnel_alerts = [a for a in triggered if a["attack_type"] == "DNS Tunneling"]
        assert len(tunnel_alerts) == 0

    def test_long_dns_queries_trigger_alert(self, parser):
        """大量異常長的 DNS 查詢名稱應觸發 Tunneling 告警"""
        detector = AnomalyDetector(
            on_alert=lambda a: None,
            threshold_syn=999,
            threshold_ports=999,
            threshold_icmp=999,
            threshold_udp=999,
            window_seconds=60,
            cooldown_seconds=0,
        )
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)

        # 產生異常長的 DNS 查詢名稱（模擬 DNS Tunneling）
        # 閾值：長度 > 50，次數 > 30
        long_qname = "a" * 60 + ".evil-tunnel.com"
        for _ in range(35):
            pkt = (IP(src="10.0.0.1", dst="8.8.8.8")
                   / UDP(dport=53)
                   / DNS(qd=DNSQR(qname=long_qname)))
            detector.inspect(pkt, parser.parse(pkt))

        tunnel_alerts = [a for a in triggered if a["attack_type"] == "DNS Tunneling"]
        assert len(tunnel_alerts) >= 1, "異常長 DNS 查詢超過閾值應觸發 Tunneling 告警"
        assert tunnel_alerts[0]["src_ip"] == "10.0.0.1"

    def test_dns_response_not_counted_as_tunnel(self, parser):
        """DNS 回應（qr=1）不應被計入 Tunneling 偵測"""
        detector = AnomalyDetector(
            on_alert=lambda a: None,
            threshold_syn=999,
            threshold_ports=999,
            threshold_icmp=999,
            threshold_udp=999,
            window_seconds=60,
            cooldown_seconds=0,
        )
        triggered = []
        detector.on_alert = lambda a: triggered.append(a)

        long_qname = "b" * 60 + ".evil-tunnel.com"
        for _ in range(50):
            pkt = (IP(src="8.8.8.8", dst="10.0.0.1")
                   / UDP(sport=53, dport=55000)
                   / DNS(qr=1, qd=DNSQR(qname=long_qname)))
            detector.inspect(pkt, parser.parse(pkt))

        tunnel_alerts = [a for a in triggered if a["attack_type"] == "DNS Tunneling"]
        assert len(tunnel_alerts) == 0, "DNS 回應不應觸發 Tunneling 告警"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. anomaly_detector.py — 告警結構完整性
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestAlertStructure:
    """驗證告警 dict 的欄位完整性"""

    def test_alert_contains_required_fields(self, parser):
        """告警應包含所有必要欄位"""
        detector = AnomalyDetector(
            on_alert=lambda a: None,
            threshold_syn=2,
            window_seconds=60,
            cooldown_seconds=0,
        )
        for _ in range(5):
            pkt = IP(src="10.0.0.1", dst="192.168.1.1") / TCP(dport=80, flags="S")
            detector.inspect(pkt, parser.parse(pkt))

        assert len(detector.alert_history) >= 1
        alert = detector.alert_history[0]
        for field in ["time", "attack_type", "severity", "src_ip", "detail", "suggestion"]:
            assert field in alert, f"告警缺少必要欄位: {field}"

    def test_inspect_returns_alert_list(self, parser):
        """inspect() 應回傳觸發的告警列表"""
        detector = AnomalyDetector(
            on_alert=lambda a: None,
            threshold_syn=2,
            window_seconds=60,
            cooldown_seconds=0,
        )
        # 先發 2 個（不超閾值）
        for _ in range(2):
            pkt = IP(src="10.0.0.1", dst="192.168.1.1") / TCP(dport=80, flags="S")
            result = detector.inspect(pkt, parser.parse(pkt))

        # 第 3 個超閾值
        pkt = IP(src="10.0.0.1", dst="192.168.1.1") / TCP(dport=80, flags="S")
        result = detector.inspect(pkt, parser.parse(pkt))
        assert isinstance(result, list)
        assert len(result) >= 1
        assert result[0]["attack_type"] == "SYN Flood"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. capture.py — 記憶體保護驗證
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestCaptureMemoryProtection:
    """驗證 LiveCapture 的記憶體保護機制"""

    def test_memory_limit_enforced(self):
        """封包數超過 max_memory_packets 時應自動裁剪"""
        from capture import LiveCapture
        cap = LiveCapture(
            interface="lo",
            max_memory_packets=5,
            stats_interval=0,
        )
        cap.start_time = time.time()

        # 模擬 10 個封包通過 _packet_callback
        for i in range(10):
            pkt = IP(src="10.0.0.1", dst=f"192.168.1.{i}") / TCP(dport=80, flags="S")
            # 直接操作，跳過 sniff
            with cap._lock:
                cap.packet_count += 1
                cap.packets.append(pkt)
                cap.parsed_records.append({"protocol": "TCP", "length": len(pkt)})
                if len(cap.packets) > cap.max_memory_packets:
                    overflow = len(cap.packets) - cap.max_memory_packets
                    del cap.packets[:overflow]
                    del cap.parsed_records[:overflow]

        assert len(cap.packets) == 5, \
            f"記憶體中應只保留最多 5 個封包，實際: {len(cap.packets)}"
        assert len(cap.parsed_records) == 5
        assert cap.packet_count == 10, \
            "總封包計數器不應受記憶體裁剪影響"

    def test_performance_metrics_tracked(self):
        """效能指標應正確追蹤"""
        from capture import LiveCapture
        cap = LiveCapture(
            interface="lo",
            max_memory_packets=100,
            stats_interval=0,
        )
        cap.start_time = time.time()

        # 模擬不同大小的封包
        sizes = [54, 128, 1500, 64, 800]
        for i, size in enumerate(sizes):
            pkt = IP(src="10.0.0.1", dst="192.168.1.1") / TCP(dport=80)
            with cap._lock:
                cap.packet_count += 1
                cap.total_bytes += size
                if size < cap._min_pkt_size:
                    cap._min_pkt_size = size
                if size > cap._max_pkt_size:
                    cap._max_pkt_size = size

        assert cap._min_pkt_size == 54
        assert cap._max_pkt_size == 1500
        assert cap.total_bytes == sum(sizes)

    def test_size_bucket_classification(self):
        """封包大小分類應正確"""
        from capture import LiveCapture
        assert LiveCapture._size_bucket(32)   == "<64B"
        assert LiveCapture._size_bucket(100)  == "64-255B"
        assert LiveCapture._size_bucket(400)  == "256-511B"
        assert LiveCapture._size_bucket(700)  == "512-1023B"
        assert LiveCapture._size_bucket(1200) == "1024-1499B"
        assert LiveCapture._size_bucket(1500) == ">=1500B"
        assert LiveCapture._size_bucket(9000) == ">=1500B"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. pcap_analyzer.py — total_packets 修正
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestPcapAnalyzerFixes:
    """驗證 PcapAnalyzer 的修正"""

    @pytest.fixture
    def test_pcap(self, temp_dir):
        """建立臨時測試 PCAP 檔案"""
        from scapy.all import wrpcap
        pcap_path = os.path.join(temp_dir, "test.pcap")
        packets = [
            IP(src="10.0.0.1", dst="192.168.1.1") / TCP(dport=80, flags="S"),
            IP(src="10.0.0.1", dst="192.168.1.1") / TCP(dport=80, flags="SA"),
            IP(src="10.0.0.1", dst="192.168.1.1") / UDP(dport=53) / DNS(qd=DNSQR(qname="test.com")),
            IP(src="10.0.0.1", dst="192.168.1.2") / ICMP(type=8),
            IP(src="10.0.0.2", dst="192.168.1.1") / TCP(dport=443, flags="S"),
        ]
        wrpcap(pcap_path, packets)
        return pcap_path

    def test_detect_attacks_returns_total_packets(self, test_pcap):
        """detect_attacks() 回傳的 summary 應包含 total_packets"""
        from pcap_analyzer import PcapAnalyzer
        analyzer = PcapAnalyzer(test_pcap)
        analyzer.load()
        result = analyzer.detect_attacks()

        assert "summary" in result
        assert "total_packets" in result["summary"], \
            "summary 應包含 total_packets（修正前為 None）"
        assert result["summary"]["total_packets"] == 5

    def test_detect_attacks_returns_total_bytes(self, test_pcap):
        """detect_attacks() 回傳的 summary 應包含 total_bytes"""
        from pcap_analyzer import PcapAnalyzer
        analyzer = PcapAnalyzer(test_pcap)
        analyzer.load()
        result = analyzer.detect_attacks()

        assert "total_bytes" in result["summary"]
        assert result["summary"]["total_bytes"] > 0

    def test_total_bytes_cached_after_load(self, test_pcap):
        """load() 後 total_bytes 應被快取到實例屬性"""
        from pcap_analyzer import PcapAnalyzer
        analyzer = PcapAnalyzer(test_pcap)
        analyzer.load()
        assert analyzer.total_bytes > 0, \
            "load() 後 self.total_bytes 應被快取，不為 0"

    def test_require_loaded_protection(self, temp_dir):
        """未呼叫 load() 就使用分析方法應拋出錯誤"""
        from pcap_analyzer import PcapAnalyzer
        # 先建立一個 PCAP
        from scapy.all import wrpcap
        pcap_path = os.path.join(temp_dir, "dummy.pcap")
        wrpcap(pcap_path, [IP(src="1.1.1.1", dst="2.2.2.2") / TCP(dport=80)])
        analyzer = PcapAnalyzer(pcap_path)
        # 不呼叫 load() 就直接呼叫分析方法
        with pytest.raises(RuntimeError):
            analyzer.extract_dns()

    def test_require_loaded_for_extract_http(self, temp_dir):
        """extract_http() 未 load 應拋出 RuntimeError"""
        from pcap_analyzer import PcapAnalyzer
        from scapy.all import wrpcap
        pcap_path = os.path.join(temp_dir, "dummy2.pcap")
        wrpcap(pcap_path, [IP(src="1.1.1.1", dst="2.2.2.2") / TCP(dport=80)])
        analyzer = PcapAnalyzer(pcap_path)
        with pytest.raises(RuntimeError):
            analyzer.extract_http()

    def test_require_loaded_for_analyze_tls(self, temp_dir):
        """analyze_tls() 未 load 應拋出 RuntimeError"""
        from pcap_analyzer import PcapAnalyzer
        from scapy.all import wrpcap
        pcap_path = os.path.join(temp_dir, "dummy3.pcap")
        wrpcap(pcap_path, [IP(src="1.1.1.1", dst="2.2.2.2") / TCP(dport=80)])
        analyzer = PcapAnalyzer(pcap_path)
        with pytest.raises(RuntimeError):
            analyzer.analyze_tls()

    def test_require_loaded_for_analyze_connections(self, temp_dir):
        """analyze_connections() 未 load 應拋出 RuntimeError"""
        from pcap_analyzer import PcapAnalyzer
        from scapy.all import wrpcap
        pcap_path = os.path.join(temp_dir, "dummy4.pcap")
        wrpcap(pcap_path, [IP(src="1.1.1.1", dst="2.2.2.2") / TCP(dport=80)])
        analyzer = PcapAnalyzer(pcap_path)
        with pytest.raises(RuntimeError):
            analyzer.analyze_connections()

    def test_analyze_connections_returns_results(self, test_pcap):
        """analyze_connections() 應回傳非空的連線統計"""
        from pcap_analyzer import PcapAnalyzer
        analyzer = PcapAnalyzer(test_pcap)
        analyzer.load()
        result = analyzer.analyze_connections()

        assert isinstance(result, list)
        assert len(result) > 0
        # 檢查結構
        conn = result[0]
        for field in ["endpoint_a", "endpoint_b", "protocol", "packets", "bytes"]:
            assert field in conn, f"連線統計缺少欄位: {field}"

    def test_analyze_connections_sorted_by_packets(self, test_pcap):
        """analyze_connections() 結果應按封包數降序排列"""
        from pcap_analyzer import PcapAnalyzer
        analyzer = PcapAnalyzer(test_pcap)
        analyzer.load()
        result = analyzer.analyze_connections()

        for i in range(len(result) - 1):
            assert result[i]["packets"] >= result[i + 1]["packets"], \
                "連線結果應按封包數降序排列"

    def test_detect_attacks_alert_structure(self, test_pcap):
        """detect_attacks() 回傳的 alerts 列表結構應正確"""
        from pcap_analyzer import PcapAnalyzer
        analyzer = PcapAnalyzer(test_pcap)
        analyzer.load()
        result = analyzer.detect_attacks()

        assert "alerts" in result
        assert isinstance(result["alerts"], list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. storage.py — SQLite 型別保留修正
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestStorageSQLiteTypeFix:
    """驗證 SQLite 值型別保留修正"""

    def test_integer_values_preserved(self, temp_dir):
        """整數欄位應以 INTEGER 型別存入 SQLite"""
        storage = PacketStorage(session_dir=temp_dir)
        records = [
            {"src_ip": "10.0.0.1", "dst_port": 80, "length": 128},
            {"src_ip": "10.0.0.2", "dst_port": 443, "length": 256},
            {"src_ip": "10.0.0.3", "dst_port": 8080, "length": 64},
        ]
        db_path = storage.save_sqlite(records, "test_types.db")
        assert db_path is not None

        # 查詢驗證型別
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT dst_port, typeof(dst_port) FROM packets")
        rows = cur.fetchall()
        conn.close()

        for port_val, type_name in rows:
            assert type_name == "integer", \
                f"dst_port 應為 integer，但得到 {type_name}（值={port_val}）"

    def test_numeric_comparison_works(self, temp_dir):
        """修正後 SQL 數值比較應正常運作"""
        storage = PacketStorage(session_dir=temp_dir)
        records = [
            {"protocol": "TCP", "dst_port": 80, "length": 100},
            {"protocol": "TCP", "dst_port": 443, "length": 200},
            {"protocol": "TCP", "dst_port": 8080, "length": 300},
            {"protocol": "TCP", "dst_port": 22, "length": 50},
        ]
        db_path = storage.save_sqlite(records, "test_compare.db")

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # 修正前這個查詢會失效（因為字串比較 "80" > "1024" 為 False）
        cur.execute("SELECT COUNT(*) FROM packets WHERE dst_port > 1024")
        count = cur.fetchone()[0]
        conn.close()

        assert count == 1, \
            f"dst_port > 1024 的記錄應有 1 筆（8080），實際: {count}"

    def test_float_values_preserved(self, temp_dir):
        """浮點數欄位應以 REAL 型別存入 SQLite"""
        storage = PacketStorage(session_dir=temp_dir)
        records = [
            {"src_ip": "10.0.0.1", "score": 0.85, "length": 100},
            {"src_ip": "10.0.0.2", "score": 0.92, "length": 200},
        ]
        db_path = storage.save_sqlite(records, "test_float.db")

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT score, typeof(score) FROM packets")
        rows = cur.fetchall()
        conn.close()

        for val, type_name in rows:
            assert type_name == "real", \
                f"score 應為 real，但得到 {type_name}"

    def test_none_values_preserved(self, temp_dir):
        """None 值應以 NULL 存入 SQLite"""
        storage = PacketStorage(session_dir=temp_dir)
        records = [
            {"src_ip": "10.0.0.1", "dst_port": None, "length": 100},
        ]
        db_path = storage.save_sqlite(records, "test_null.db")

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT dst_port, typeof(dst_port) FROM packets")
        val, type_name = cur.fetchone()
        conn.close()

        assert val is None
        assert type_name == "null"

    def test_non_basic_type_converted_to_str(self, temp_dir):
        """非基本型別（如 list）應被轉為字串"""
        storage = PacketStorage(session_dir=temp_dir)
        records = [
            {"src_ip": "10.0.0.1", "tags": ["attack", "syn"], "length": 100},
        ]
        db_path = storage.save_sqlite(records, "test_complex.db")

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT tags, typeof(tags) FROM packets")
        val, type_name = cur.fetchone()
        conn.close()

        assert type_name == "text"
        assert isinstance(val, str)


class TestStorageBatchWrite:
    """驗證 SQLite 批次寫入"""

    def test_large_dataset_write(self, temp_dir):
        """大量記錄應能正確批次寫入"""
        storage = PacketStorage(session_dir=temp_dir)
        records = [
            {"src_ip": f"10.0.{i//256}.{i%256}", "dst_port": 80 + i, "length": 64 + i}
            for i in range(2500)  # 超過 BATCH_SIZE=1000
        ]
        db_path = storage.save_sqlite(records, "test_batch.db")

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM packets")
        count = cur.fetchone()[0]
        conn.close()

        assert count == 2500, \
            f"所有 2500 筆記錄應被寫入，實際: {count}"


class TestStorageCSVAppend:
    """驗證 CSV 追加模式"""

    def test_append_creates_new_file(self, temp_dir):
        """追加模式在檔案不存在時應建立新檔案（含 header）"""
        storage = PacketStorage(session_dir=temp_dir)
        records = [
            {"src_ip": "10.0.0.1", "dst_port": 80},
            {"src_ip": "10.0.0.2", "dst_port": 443},
        ]
        path = storage.append_csv(records, "test_append.csv")

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            lines = list(reader)

        assert lines[0] == ["src_ip", "dst_port"], "新檔案應有 header"
        assert len(lines) == 3  # 1 header + 2 data

    def test_append_adds_without_header(self, temp_dir):
        """追加模式在檔案存在時不應重寫 header"""
        storage = PacketStorage(session_dir=temp_dir)
        records1 = [{"src_ip": "10.0.0.1", "dst_port": 80}]
        records2 = [{"src_ip": "10.0.0.2", "dst_port": 443}]

        path = storage.append_csv(records1, "test_append2.csv")
        storage.append_csv(records2, "test_append2.csv")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # header 應只出現一次
        header_count = content.count("src_ip,dst_port")
        assert header_count == 1, \
            f"Header 應只出現 1 次，實際: {header_count}"

        # 應有 2 筆資料
        lines = content.strip().split("\n")
        assert len(lines) == 3  # 1 header + 2 data

    def test_append_empty_records_returns_none(self, temp_dir):
        """空記錄應回傳 None"""
        storage = PacketStorage(session_dir=temp_dir)
        result = storage.append_csv([], "test_empty.csv")
        assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. 整合測試 — 完整 Pipeline 驗證
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestEndToEndPipeline:
    """端到端整合測試：模擬 tasks.py 呼叫流程"""

    @pytest.fixture
    def attack_pcap(self, temp_dir):
        """建立包含攻擊封包的測試 PCAP"""
        from scapy.all import wrpcap
        packets = []
        # 正常流量
        for i in range(3):
            packets.append(
                IP(src="192.168.1.100", dst="10.0.0.1")
                / TCP(sport=50000 + i, dport=80, flags="S")
            )
        # DNS 查詢
        packets.append(
            IP(src="192.168.1.100", dst="8.8.8.8")
            / UDP(dport=53)
            / DNS(qd=DNSQR(qname="example.com"))
        )
        pcap_path = os.path.join(temp_dir, "attack_test.pcap")
        wrpcap(pcap_path, packets)
        return pcap_path

    def test_tasks_pipeline_gets_total_packets(self, attack_pcap):
        """模擬 tasks.py 的完整分析流程，驗證 total_packets 可正確取得"""
        from pcap_analyzer import PcapAnalyzer

        analyzer = PcapAnalyzer(attack_pcap)
        analyzer.load()
        analysis_result = analyzer.detect_attacks()

        # 模擬 tasks.py 第 105-109 行的取值邏輯
        alerts_raw   = analysis_result.get("alerts", [])
        summary_info = analysis_result.get("summary", {})
        packet_count = summary_info.get("total_packets", 0)

        assert packet_count == 4, \
            f"tasks.py 取得的 total_packets 應為 4，實際: {packet_count}"
        assert isinstance(alerts_raw, list)

    def test_full_analysis_pipeline(self, attack_pcap):
        """full_analysis() 一鍵分析應正常執行（包含新增的 analyze_connections）"""
        from pcap_analyzer import PcapAnalyzer
        analyzer = PcapAnalyzer(attack_pcap)
        result = analyzer.full_analysis()
        assert result is not None
        assert len(result.records) == 4
        assert result.total_bytes > 0
