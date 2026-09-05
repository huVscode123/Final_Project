# ============================================================
# analyzer/tests.py — 配合模擬檢測修正的測試更新
#
# 使用方式：
#   用下面 SimulationFusionTests 類別的完整內容，取代原
#   analyzer/tests.py 中同名的類別（檔案開頭的 import、
#   setUpTestData、setUp 維持不變，只是這裡為求完整而一併列出）。
#
# ── 為什麼原本的測試需要修改 ─────────────────────────────────
# 原本的測試是針對舊版「黑箱」行為寫的，例如：
#   - 斷言 threshold == count*50%（demo 門檻）
#   - 斷言 ARP Spoofing／DNS Amplification 一定會被「強制」判定為攻擊
# 這些正是這次要修掉的問題本身，所以測試也要跟著更新，才能驗證
# 修正後的行為是正確的，而不是繼續驗證舊的錯誤行為。
# ============================================================

import os
import json
from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from django.conf import settings

import sys

core_path = os.path.join(settings.BASE_DIR, 'core')
if core_path not in sys.path:
    sys.path.insert(0, core_path)

from analyzer.views import _detect_simulation_behavior
from generate_attack_pcap import generate_attack_packets
from config import (
    ALERT_THRESHOLD_SYN, ALERT_THRESHOLD_PORTS,
    ALERT_THRESHOLD_ICMP, ALERT_THRESHOLD_UDP,
)


class SimulationFusionTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        User = get_user_model()
        cls.user = User.objects.create_superuser('testuser', 'test@test.com', 'testpass123')

    def setUp(self):
        self.client = Client()
        self.client.login(username='testuser', password='testpass123')

    # ────────────────────────────────────────────────────────
    # _detect_simulation_behavior 單元測試
    # ────────────────────────────────────────────────────────

    def test_behavior_detector_icmp_flood(self):
        """ICMP Flood 封包數超過正式門檻（ALERT_THRESHOLD_ICMP）時應被正確辨識"""
        packet_count = ALERT_THRESHOLD_ICMP + 10   # 60
        scapy_packets = generate_attack_packets('icmp_flood', packet_count)
        result = _detect_simulation_behavior(scapy_packets, 'icmp_flood', packet_count)

        self.assertTrue(result['is_attack'])
        self.assertEqual(result['attack_type'], 'ICMP Flood (Ping Flood)')
        self.assertGreater(len(result['alerts']), 0)

    def test_behavior_detector_syn_flood(self):
        """SYN Flood 封包數超過正式門檻（ALERT_THRESHOLD_SYN）時應被正確辨識"""
        packet_count = ALERT_THRESHOLD_SYN + 10   # 110
        scapy_packets = generate_attack_packets('syn_flood', packet_count)
        result = _detect_simulation_behavior(scapy_packets, 'syn_flood', packet_count)

        self.assertTrue(result['is_attack'])
        self.assertGreater(len(result['alerts']), 0)

    def test_behavior_detector_normal_traffic(self):
        """正常流量絕對不會被判定為攻擊"""
        packet_count = 15
        scapy_packets = generate_attack_packets('normal_traffic', packet_count)
        result = _detect_simulation_behavior(scapy_packets, 'normal_traffic', packet_count)

        self.assertFalse(result['is_attack'])

    def test_behavior_detector_low_count_honestly_not_triggered(self):
        """
        [根因修正驗證] 封包數不足以達到正式門檻時，應誠實回報「未觸發」，
        不再有任何「保底強制判定」把結果偷偷改成攻擊。
        """
        packet_count = 2
        scapy_packets = generate_attack_packets('icmp_flood', packet_count)
        result = _detect_simulation_behavior(scapy_packets, 'icmp_flood', packet_count)

        self.assertFalse(result['is_attack'],
            "封包數遠低於正式門檻時不應被判定為攻擊，不該有保底強制判定")
        self.assertIsNone(result['triggered_index'])

    # ────────────────────────────────────────────────────────
    # [新增] 透明化欄位單元測試
    # ────────────────────────────────────────────────────────

    def test_behavior_detector_has_triggered_index(self):
        """真實觸發攻擊時應包含 triggered_index 欄位"""
        packet_count = ALERT_THRESHOLD_ICMP + 10
        scapy_packets = generate_attack_packets('icmp_flood', packet_count)
        result = _detect_simulation_behavior(scapy_packets, 'icmp_flood', packet_count)

        self.assertIn('triggered_index', result)
        self.assertIsNotNone(result['triggered_index'])
        self.assertIsInstance(result['triggered_index'], int)

    def test_behavior_detector_has_thresholds_dict(self):
        """
        [根因修正驗證] thresholds 應等於 config.py 的正式系統門檻，
        不再隨封包數量縮放（不再是「封包數 * 50%」的示範門檻）。
        """
        for packet_count in (20, 110, 250):
            scapy_packets = generate_attack_packets('syn_flood', packet_count)
            result = _detect_simulation_behavior(scapy_packets, 'syn_flood', packet_count)

            self.assertIn('thresholds', result)
            self.assertEqual(result['thresholds']['syn'], ALERT_THRESHOLD_SYN)
            self.assertEqual(result['thresholds']['ports'], ALERT_THRESHOLD_PORTS)
            self.assertEqual(result['thresholds']['icmp'], ALERT_THRESHOLD_ICMP)
            self.assertEqual(result['thresholds']['udp'], ALERT_THRESHOLD_UDP)

    def test_behavior_detector_arp_spoof_not_forced(self):
        """
        [根因修正驗證] ARP Spoofing 不再需要任何「保底強制判定」，
        forced_by_demo_shortcut 恆為 False；同一 IP 出現第二個 MAC
        時，規則引擎本身就會真實觸發，不需要黑箱機制介入。
        """
        packet_count = 5
        scapy_packets = generate_attack_packets('arp_spoof', packet_count)
        result = _detect_simulation_behavior(scapy_packets, 'arp_spoof', packet_count)

        self.assertFalse(result['forced_by_demo_shortcut'])
        self.assertTrue(result['is_attack'],
            "ARP Spoofing 的偵測邏輯本身應能在合理封包數下真實觸發")

    def test_behavior_detector_dns_amplification_genuinely_triggers(self):
        """
        [根因修正驗證] 修正封包產生器的查詢/回應比例後，DNS
        Amplification 應能在封包數足夠時（>=22）真實觸發規則引擎，
        不再需要黑箱保底判定。
        """
        packet_count = 30
        scapy_packets = generate_attack_packets('dns_amplification', packet_count)
        result = _detect_simulation_behavior(scapy_packets, 'dns_amplification', packet_count)

        self.assertFalse(result['forced_by_demo_shortcut'])
        self.assertTrue(result['is_attack'],
            "封包數足夠時，DNS Amplification 應能真實觸發規則引擎，"
            "不應再需要保底判定")

    def test_behavior_detector_normal_no_triggered_index(self):
        """正常流量不應有 triggered_index"""
        packet_count = 10
        scapy_packets = generate_attack_packets('normal_traffic', packet_count)
        result = _detect_simulation_behavior(scapy_packets, 'normal_traffic', packet_count)

        self.assertIn('triggered_index', result)
        self.assertIsNone(result['triggered_index'])

    # ────────────────────────────────────────────────────────
    # simulation_api 端對端整合測試
    # ────────────────────────────────────────────────────────

    def test_api_normal_traffic_not_all_anomalous(self):
        """正常流量不應被判定為攻擊（規則式偵測為唯一判定依據）"""
        response = self.client.post(
            '/analyzer/simulation/api/',
            data=json.dumps({'attack_type': 'normal_traffic', 'packet_count': 20}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data['ok'])
        self.assertFalse(data['rule_based_anomaly_detected'])
        self.assertEqual(data['anomaly_count'], 0,
            "正常流量的規則式判定為 False 時，anomaly_count 應為 0")

    def test_api_is_anomaly_ignores_cnn_score(self):
        """
        [根因修正驗證] CNN 分數不應該能單獨把 is_anomaly 判定為 True。
        對正常流量送出模擬請求，即使某些封包的 cnn_anomaly 為 True，
        只要 behavior_anomaly 為 False，is_anomaly 就必須是 False。
        """
        response = self.client.post(
            '/analyzer/simulation/api/',
            data=json.dumps({'attack_type': 'normal_traffic', 'packet_count': 20}),
            content_type='application/json'
        )
        data = response.json()
        self.assertTrue(data['ok'])
        for r in data['results']:
            if not r['behavior_anomaly']:
                self.assertFalse(
                    r['is_anomaly'],
                    "is_anomaly 不應該因為 cnn_anomaly 而單獨變成 True"
                )

    def test_api_fusion_icmp_flood(self):
        """ICMP Flood 封包數超過正式門檻時 → 行為偵測器應攔截"""
        packet_count = ALERT_THRESHOLD_ICMP + 10
        response = self.client.post(
            '/analyzer/simulation/api/',
            data=json.dumps({'attack_type': 'icmp_flood', 'packet_count': packet_count}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data['ok'])
        self.assertTrue(data['rule_based_anomaly_detected'])

        for entry in data['results']:
            self.assertTrue(entry['behavior_anomaly'])
            self.assertTrue(entry['is_anomaly'])

    def test_api_fusion_arp_spoof(self):
        """ARP Spoof 少量封包也應被真實判定為攻擊（不靠保底）"""
        response = self.client.post(
            '/analyzer/simulation/api/',
            data=json.dumps({'attack_type': 'arp_spoof', 'packet_count': 5}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data['ok'])
        self.assertTrue(data['rule_based_anomaly_detected'])
        self.assertFalse(data['behavior_detection']['forced_by_demo_shortcut'])
        for entry in data['results']:
            self.assertTrue(entry['is_anomaly'])

    def test_api_udp_flood_supported(self):
        """UDP Flood 應在白名單內"""
        response = self.client.post(
            '/analyzer/simulation/api/',
            data=json.dumps({'attack_type': 'udp_flood', 'packet_count': 10}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['ok'])

    def test_api_effective_threshold_differs_from_model(self):
        """
        effective_threshold 應來自 baseline 99th percentile，
        而非 bundle.threshold（模型訓練閾值）。
        """
        response = self.client.post(
            '/analyzer/simulation/api/',
            data=json.dumps({'attack_type': 'syn_flood', 'packet_count': 5}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        eff = data.get('effective_threshold')
        model_thr = data.get('threshold')

        if eff and model_thr and model_thr > 0:
            self.assertGreater(
                eff, model_thr * 2,
                f"effective_threshold ({eff:.6f}) 應遠大於 model threshold ({model_thr:.6f})"
            )

    # ────────────────────────────────────────────────────────
    # [新增] 透明化 API 欄位整合測試
    # ────────────────────────────────────────────────────────

    def test_api_has_cnn_reliability_note(self):
        """API 回傳應包含 cnn_reliability_note 與 final_verdict_note 欄位"""
        response = self.client.post(
            '/analyzer/simulation/api/',
            data=json.dumps({'attack_type': 'syn_flood', 'packet_count': 5}),
            content_type='application/json'
        )
        data = response.json()
        self.assertIn('cnn_reliability_note', data)
        self.assertGreater(len(data['cnn_reliability_note']), 10)
        self.assertIn('final_verdict_note', data)
        self.assertGreater(len(data['final_verdict_note']), 10)

    def test_api_has_behavior_detection_thresholds(self):
        """API 回傳 behavior_detection 應包含 thresholds、triggered_index、
        forced_by_demo_shortcut（且 forced_by_demo_shortcut 恆為 False）"""
        packet_count = ALERT_THRESHOLD_ICMP + 10
        response = self.client.post(
            '/analyzer/simulation/api/',
            data=json.dumps({'attack_type': 'icmp_flood', 'packet_count': packet_count}),
            content_type='application/json'
        )
        data = response.json()
        bd = data.get('behavior_detection', {})

        self.assertIn('thresholds', bd)
        self.assertIn('triggered_index', bd)
        self.assertIn('forced_by_demo_shortcut', bd)
        self.assertFalse(bd['forced_by_demo_shortcut'])
        self.assertEqual(bd['thresholds']['icmp'], ALERT_THRESHOLD_ICMP)

    def test_api_results_have_rule_detail(self):
        """每筆結果應包含 rule_triggered_here 和 rule_detail 欄位"""
        packet_count = ALERT_THRESHOLD_SYN + 10
        response = self.client.post(
            '/analyzer/simulation/api/',
            data=json.dumps({'attack_type': 'syn_flood', 'packet_count': packet_count}),
            content_type='application/json'
        )
        data = response.json()

        for r in data['results']:
            self.assertIn('rule_triggered_here', r)
            self.assertIn('rule_detail', r)
            self.assertIsInstance(r['rule_detail'], list)

        # 封包數已超過正式門檻，至少有一個封包是直接觸發規則的
        triggered_count = sum(1 for r in data['results'] if r['rule_triggered_here'])
        self.assertGreater(triggered_count, 0, "SYN Flood 應有至少一個封包直接觸發規則")

    def test_api_results_have_header_field(self):
        """
        [新增/透明化驗證] 每筆結果應包含解析後的封包標頭資訊
        （header：protocol/src_ip/dst_ip/port/flags 等），避免黑箱。
        """
        response = self.client.post(
            '/analyzer/simulation/api/',
            data=json.dumps({'attack_type': 'syn_flood', 'packet_count': 10}),
            content_type='application/json'
        )
        data = response.json()
        self.assertTrue(data['ok'])
        for r in data['results']:
            self.assertIn('header', r)
            self.assertIsInstance(r['header'], dict)
            self.assertIn('protocol', r['header'])
