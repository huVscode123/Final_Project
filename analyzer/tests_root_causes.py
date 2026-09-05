# -*- coding: utf-8 -*-
"""
analyzer/tests_root_causes.py

針對 fix0904 修正的六大根因所撰寫的完整單元測試。
每個測試類別對應一個根因，並附有中文說明說明「驗證了什麼問題被解決了」。

執行方式：
    python manage.py test analyzer.tests_root_causes -v 2
"""
import json
import os
import sys

from django.test import TestCase, Client, RequestFactory
from django.contrib.auth import get_user_model
from django.conf import settings

core_path = os.path.join(settings.BASE_DIR, 'core')
if core_path not in sys.path:
    sys.path.insert(0, core_path)

from config import (
    ALERT_THRESHOLD_SYN, ALERT_THRESHOLD_PORTS,
    ALERT_THRESHOLD_ICMP, ALERT_THRESHOLD_UDP,
)
from analyzer.views import _detect_simulation_behavior, simulation


# ════════════════════════════════════════════════════════════════════════════
# 共用基底類別
# ════════════════════════════════════════════════════════════════════════════
class _BaseSimTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        User = get_user_model()
        cls.user = User.objects.create_superuser(
            'testroot', 'root@test.com', 'rootpass123'
        )

    def setUp(self):
        self.client = Client()
        self.client.login(username='testroot', password='rootpass123')
        self.factory = RequestFactory()

    def _post_api(self, attack_type, intensity=1.0, packet_count=None):
        """統一的 API 呼叫介面，同時相容新版 intensity 與舊版 packet_count。"""
        body = {'attack_type': attack_type, 'model_key': 'unsupervised_vae'}
        if packet_count is not None:
            body['packet_count'] = packet_count
        else:
            body['intensity'] = intensity
        return self.client.post(
            '/analyzer/simulation/api/',
            data=json.dumps(body),
            content_type='application/json',
        )


# ════════════════════════════════════════════════════════════════════════════
# 根因 A：封包產生器改用 simulate_anomaly_traffic::GENERATORS
#         保證封包量超過規則引擎門檻
# ════════════════════════════════════════════════════════════════════════════
class RootCauseA_PacketGeneratorThreshold(_BaseSimTest):
    """
    根因 A 驗證：舊版用 generate_attack_packets(type, count) 配合 60~250 的
    滑桿，SYN Flood（門檻100）和 UDP Flood（門檻200）在預設值下永遠無法觸發。
    修正後改用 GENERATORS 的 bulk_scale()，封包量保證超過正式門檻。
    """

    def _assert_triggers(self, attack_type, label):
        """通用斷言：攻擊場景一定觸發規則引擎，且 rule_based_anomaly_detected=True。"""
        resp = self._post_api(attack_type, intensity=1.0)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data['ok'], f"{label} API 應成功回傳")
        self.assertTrue(
            data['rule_based_anomaly_detected'],
            f"{label}（intensity=1.0）應觸發規則引擎，"
            f"但 rule_based_anomaly_detected={data.get('rule_based_anomaly_detected')}"
        )
        bd = data['behavior_detection']
        self.assertIsNotNone(
            bd['triggered_index'],
            f"{label} 應有至少一個封包真正觸發規則（triggered_index 不應為 None）"
        )
        return data

    def test_A1_syn_flood_triggers_at_default_intensity(self):
        """SYN Flood (門檻100) intensity=1.0 時應觸發，修正前預設封包數=60 永遠不夠。"""
        data = self._assert_triggers('syn_flood', 'SYN Flood')
        total = data['packet_count']
        self.assertGreater(
            total, ALERT_THRESHOLD_SYN,
            f"SYN Flood 封包數 {total} 應 > 門檻 {ALERT_THRESHOLD_SYN}"
        )

    def test_A2_udp_flood_triggers_at_default_intensity(self):
        """
        UDP Flood (門檻200) intensity=1.0 時應觸發。
        修正前舊版滑桿最大值250但後端限制250，幾乎永遠不夠（剛好壓線且是確定性）。
        """
        data = self._assert_triggers('udp_flood', 'UDP Flood')
        total = data['packet_count']
        self.assertGreater(
            total, ALERT_THRESHOLD_UDP,
            f"UDP Flood 封包數 {total} 應 > 門檻 {ALERT_THRESHOLD_UDP}"
        )

    def test_A3_port_scan_triggers(self):
        """Port Scan (門檻20) 應觸發。"""
        self._assert_triggers('port_scan', 'Port Scan')

    def test_A4_icmp_flood_triggers(self):
        """ICMP Flood (門檻50) 應觸發。"""
        self._assert_triggers('icmp_flood', 'ICMP Flood')

    def test_A5_arp_spoof_triggers(self):
        """ARP Spoofing 應觸發（依型樣偵測，第2個封包出現不同MAC即觸發）。"""
        self._assert_triggers('arp_spoof', 'ARP Spoofing')

    def test_A6_dns_amplification_triggers(self):
        """DNS 放大攻擊 應觸發（回應/查詢比 > 10x）。"""
        self._assert_triggers('dns_amplification', 'DNS 放大攻擊')

    def test_A7_normal_traffic_never_triggers(self):
        """正常流量絕對不應觸發任何攻擊告警。"""
        resp = self._post_api('normal_traffic', intensity=1.0)
        data = resp.json()
        self.assertTrue(data['ok'])
        self.assertFalse(
            data['rule_based_anomaly_detected'],
            "正常流量不應觸發規則引擎"
        )
        self.assertEqual(data['anomaly_count'], 0)

    def test_A8_high_intensity_still_triggers(self):
        """intensity=3.0（最高強度）仍應正確觸發，且封包數在安全上限 2000 以內。"""
        resp = self._post_api('syn_flood', intensity=3.0)
        data = resp.json()
        self.assertTrue(data['ok'])
        self.assertTrue(data['rule_based_anomaly_detected'])
        self.assertLessEqual(data['packet_count'], 2000, "封包數應在安全上限 2000 以內")

    def test_A9_low_intensity_still_triggers(self):
        """intensity=0.3（最低強度）仍應觸發，bulk_scale() 保證封包量超過門檻。"""
        resp = self._post_api('icmp_flood', intensity=0.3)
        data = resp.json()
        self.assertTrue(data['ok'])
        self.assertTrue(data['rule_based_anomaly_detected'])

    def test_A10_generation_meta_returned(self):
        """API 應回傳封包產生的元資料 generation_meta（由 GENERATORS 回傳）。"""
        resp = self._post_api('dns_amplification', intensity=1.0)
        data = resp.json()
        self.assertIn('generation_meta', data)
        meta = data['generation_meta']
        self.assertIsInstance(meta, dict)
        # DNS 放大應有 victim_ip, dns_server, request_count, response_count
        self.assertIn('victim_ip', meta)
        self.assertIn('response_count', meta)
        self.assertGreater(
            meta['response_count'], meta.get('request_count', 0),
            "DNS 放大攻擊的回應數應遠多於查詢數"
        )


# ════════════════════════════════════════════════════════════════════════════
# 根因 B：CNN train/serve skew — is_anomaly 最終判定只依規則引擎
# ════════════════════════════════════════════════════════════════════════════
class RootCauseB_CNNTrainServeSkew(_BaseSimTest):
    """
    根因 B 驗證：模型以 CSV 統計特徵訓練，推論時卻送入封包位元組，
    重建誤差不具鑑別力。修正後 is_anomaly 完全不受 CNN 分數影響，
    最終判定只依規則引擎（rule_based_anomaly_detected）。
    """

    def test_B1_normal_traffic_is_anomaly_all_false(self):
        """正常流量：即使 cnn_anomaly 某些封包為 True，is_anomaly 必須全為 False。"""
        resp = self._post_api('normal_traffic', intensity=1.0)
        data = resp.json()
        self.assertTrue(data['ok'])
        for r in data['results']:
            self.assertFalse(
                r['is_anomaly'],
                f"正常流量封包 #{r['index']} 的 is_anomaly 應為 False，"
                f"不應因 cnn_anomaly={r['cnn_anomaly']} 而被改為 True"
            )

    def test_B2_cnn_cannot_alone_determine_anomaly(self):
        """
        關鍵驗證：behavior_anomaly=False 時，is_anomaly 必須也為 False，
        無論 cnn_anomaly 是 True 還是 False。
        """
        resp = self._post_api('normal_traffic', intensity=1.0)
        data = resp.json()
        for r in data['results']:
            if not r['behavior_anomaly']:
                self.assertFalse(
                    r['is_anomaly'],
                    "CNN 分數不應單獨決定 is_anomaly，behavior_anomaly=False 時"
                    f" is_anomaly 應為 False（封包 #{r['index']} cnn_anomaly={r['cnn_anomaly']}）"
                )

    def test_B3_attack_is_anomaly_all_true_when_rule_triggers(self):
        """攻擊場景：規則引擎觸發時，所有封包的 is_anomaly 應為 True。"""
        resp = self._post_api('icmp_flood', intensity=1.0)
        data = resp.json()
        self.assertTrue(data['rule_based_anomaly_detected'])
        for r in data['results']:
            self.assertTrue(
                r['is_anomaly'],
                f"規則引擎判定為攻擊時，封包 #{r['index']} 的 is_anomaly 應為 True"
            )

    def test_B4_cnn_reliability_note_present(self):
        """API 應回傳 cnn_reliability_note，說明 CNN 分數的可信度狀態。"""
        resp = self._post_api('syn_flood', intensity=1.0)
        data = resp.json()
        self.assertIn('cnn_reliability_note', data)
        self.assertGreater(len(data['cnn_reliability_note']), 20)

    def test_B5_final_verdict_note_present(self):
        """API 應回傳 final_verdict_note，明確說明最終判定依據為規則引擎。"""
        resp = self._post_api('syn_flood', intensity=1.0)
        data = resp.json()
        self.assertIn('final_verdict_note', data)
        note = data['final_verdict_note']
        self.assertGreater(len(note), 10)
        # 說明文字應提到規則式偵測
        self.assertIn('規則', note)

    def test_B6_model_representation_field_present(self):
        """API 應回傳 model_representation，讓前端知道模型訓練時的資料表示法。"""
        resp = self._post_api('syn_flood', intensity=1.0)
        data = resp.json()
        self.assertIn('model_representation', data)
        # 可為 csv_features_legacy 或 packet_bytes_visualizer
        self.assertIn(
            data['model_representation'],
            ('csv_features_legacy', 'packet_bytes_visualizer', None),
        )

    def test_B7_anomaly_count_matches_rule_verdict(self):
        """
        anomaly_count 的計算依據應為規則引擎判定：
        - 正常流量 → 0
        - 攻擊 → 等於 packet_count
        """
        # 正常流量
        resp_n = self._post_api('normal_traffic', intensity=1.0)
        d_n = resp_n.json()
        self.assertEqual(d_n['anomaly_count'], 0)

        # 攻擊場景
        resp_a = self._post_api('icmp_flood', intensity=1.0)
        d_a = resp_a.json()
        self.assertTrue(d_a['rule_based_anomaly_detected'])
        self.assertEqual(
            d_a['anomaly_count'], d_a['packet_count'],
            "規則引擎判定為攻擊時，anomaly_count 應等於 packet_count"
        )


# ════════════════════════════════════════════════════════════════════════════
# 根因 C：第一次模擬慢（matplotlib 字型掃描）— 預熱後應快
# ════════════════════════════════════════════════════════════════════════════
class RootCauseC_WarmupAndPerformance(_BaseSimTest):
    """
    根因 C 驗證：apps.py 預熱執行緒完成後，模型應已快取，
    API 回應應包含 server_warmup_complete 與 model_was_cached 等透明欄位，
    讓使用者能看到「為什麼這次比較慢/快」。
    """

    def test_C1_server_warmup_complete_field_present(self):
        """API 應回傳 server_warmup_complete 欄位（布林值）。"""
        resp = self._post_api('syn_flood', intensity=1.0)
        data = resp.json()
        self.assertIn('server_warmup_complete', data)
        self.assertIsInstance(data['server_warmup_complete'], bool)

    def test_C2_timing_ms_field_present_with_all_stages(self):
        """API 應回傳 timing_ms，包含各階段耗時（封包產生/規則引擎/模型載入/CNN推論）。"""
        resp = self._post_api('icmp_flood', intensity=1.0)
        data = resp.json()
        self.assertIn('timing_ms', data)
        tm = data['timing_ms']
        for key in ('packet_generation_ms', 'rule_engine_ms', 'total_ms'):
            self.assertIn(key, tm, f"timing_ms 應包含 {key}")
            self.assertGreaterEqual(tm[key], 0)

    def test_C3_model_was_cached_field_present(self):
        """timing_ms 應包含 model_was_cached 欄位（True=命中快取）。"""
        resp = self._post_api('syn_flood', intensity=1.0)
        data = resp.json()
        tm = data.get('timing_ms', {})
        self.assertIn('model_was_cached', tm)

    def test_C4_model_load_ms_is_low_when_cached(self):
        """
        模型已快取後，model_load_ms 應接近 0（< 50ms）。
        測試先呼叫一次讓模型進入快取，第二次呼叫才驗證耗時。
        """
        # 第一次：讓模型載入進快取
        self._post_api('normal_traffic', intensity=1.0)
        # 第二次：驗證 model_load_ms
        resp = self._post_api('normal_traffic', intensity=1.0)
        data = resp.json()
        tm = data.get('timing_ms', {})
        if tm.get('model_was_cached'):
            self.assertLess(
                tm.get('model_load_ms', 999), 50,
                f"快取命中時 model_load_ms 應 < 50ms，實際 = {tm.get('model_load_ms')}"
            )

    def test_C5_total_ms_is_reasonable(self):
        """
        API 總耗時應在合理範圍（< 30000ms）。
        這個測試主要用來偵測因為缺少預熱而卡在 matplotlib 字型掃描的極端情況。
        """
        resp = self._post_api('icmp_flood', intensity=1.0)
        data = resp.json()
        tm = data.get('timing_ms', {})
        self.assertLess(
            tm.get('total_ms', 99999), 30000,
            "API 總耗時不應超過 30 秒（若超過代表可能有預熱未完成或效能問題）"
        )


# ════════════════════════════════════════════════════════════════════════════
# 根因 D：模型快取透明化 — is_model_cached() 函式正確性
# ════════════════════════════════════════════════════════════════════════════
class RootCauseD_ModelCacheTransparency(_BaseSimTest):
    """
    根因 D 驗證：model_registry.py 新增的 is_model_cached() 函式
    能正確反映快取狀態，不會觸發載入動作。
    """

    def test_D1_is_model_cached_function_exists(self):
        """model_registry 應匯出 is_model_cached 函式。"""
        import model_registry
        self.assertTrue(
            hasattr(model_registry, 'is_model_cached'),
            "model_registry 應有 is_model_cached 函式"
        )
        self.assertTrue(callable(model_registry.is_model_cached))

    def test_D2_is_model_cached_returns_bool(self):
        """is_model_cached 對任意路徑都應回傳布林值，不拋例外。"""
        import torch
        from model_registry import is_model_cached
        result = is_model_cached('/nonexistent/path.pt', torch.device('cpu'))
        self.assertIsInstance(result, bool)
        self.assertFalse(result, "不存在的路徑應回傳 False")

    def test_D3_cache_hit_after_load(self):
        """load_anomaly_model 後，is_model_cached 應回傳 True。"""
        import torch
        from model_registry import load_anomaly_model, is_model_cached, clear_model_cache

        model_cfg = settings.ANOMALY_MODELS.get('unsupervised_vae', {})
        model_path = str(model_cfg.get('path', ''))
        if not model_path or not os.path.exists(model_path):
            self.skipTest("unsupervised_vae 模型檔案不存在，跳過此測試")

        device = torch.device('cpu')
        # 清除快取確保乾淨狀態
        clear_model_cache(model_path)
        before = is_model_cached(model_path, device)

        load_anomaly_model(model_path, device=device,
                           default_latent_dim=settings.CNN_LATENT_DIM)
        after = is_model_cached(model_path, device)

        self.assertFalse(before, "載入前快取應為 False")
        self.assertTrue(after, "載入後快取應為 True")

    def test_D4_clear_model_cache_works(self):
        """clear_model_cache 後，is_model_cached 應回傳 False。"""
        import torch
        from model_registry import load_anomaly_model, is_model_cached, clear_model_cache

        model_cfg = settings.ANOMALY_MODELS.get('unsupervised_vae', {})
        model_path = str(model_cfg.get('path', ''))
        if not model_path or not os.path.exists(model_path):
            self.skipTest("unsupervised_vae 模型檔案不存在，跳過此測試")

        device = torch.device('cpu')
        load_anomaly_model(model_path, device=device,
                           default_latent_dim=settings.CNN_LATENT_DIM)
        self.assertTrue(is_model_cached(model_path, device))

        clear_model_cache(model_path)
        self.assertFalse(is_model_cached(model_path, device),
                         "clear_model_cache 後快取應為空")

    def test_D5_input_representation_attribute(self):
        """AnomalyModelBundle 應有 input_representation 屬性，回傳表示法字串。"""
        import torch
        from model_registry import load_anomaly_model

        model_cfg = settings.ANOMALY_MODELS.get('unsupervised_vae', {})
        model_path = str(model_cfg.get('path', ''))
        if not model_path or not os.path.exists(model_path):
            self.skipTest("unsupervised_vae 模型檔案不存在，跳過此測試")

        bundle = load_anomaly_model(model_path, device=torch.device('cpu'),
                                    default_latent_dim=settings.CNN_LATENT_DIM)
        self.assertTrue(hasattr(bundle, 'input_representation'))
        self.assertIn(
            bundle.input_representation,
            ('csv_features_legacy', 'packet_bytes_visualizer'),
        )


# ════════════════════════════════════════════════════════════════════════════
# 根因 E：模型下拉選單空白 — simulation() view 必須回傳 models
# ════════════════════════════════════════════════════════════════════════════
class RootCauseE_SimulationViewContext(_BaseSimTest):
    """
    根因 E 驗證：舊版 simulation() view 沒有把 models 放進 context，
    導致樣板的 {% for model in models %} 一直迭代空清單，
    選單永遠沒有任何選項。修正後必須包含 models 清單。
    """

    def test_E1_simulation_page_returns_200(self):
        """模擬檢測頁面應正常回傳 200。"""
        resp = self.client.get('/analyzer/simulation/')
        self.assertEqual(resp.status_code, 200)

    def test_E2_context_contains_models_list(self):
        """simulation() view 的 context 應包含非空的 models 清單。"""
        resp = self.client.get('/analyzer/simulation/')
        self.assertIn('models', resp.context,
                      "simulation() view context 應包含 'models' 鍵，"
                      "否則樣板的模型下拉選單永遠是空的")
        models = resp.context['models']
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0,
                           "models 清單不應為空，settings.ANOMALY_MODELS 至少有一個模型")

    def test_E3_each_model_has_required_keys(self):
        """models 清單中每個元素應有 key、label、ready 三個欄位。"""
        resp = self.client.get('/analyzer/simulation/')
        for model in resp.context['models']:
            self.assertIn('key', model)
            self.assertIn('label', model)
            self.assertIn('ready', model)
            self.assertIsInstance(model['ready'], bool)

    def test_E4_model_ready_reflects_actual_file(self):
        """model['ready'] 應與實際檔案是否存在一致。"""
        resp = self.client.get('/analyzer/simulation/')
        for model in resp.context['models']:
            cfg = settings.ANOMALY_MODELS.get(model['key'], {})
            path = str(cfg.get('path', ''))
            expected_ready = bool(path and os.path.exists(path))
            self.assertEqual(
                model['ready'], expected_ready,
                f"模型 {model['key']} 的 ready={model['ready']} "
                f"與實際檔案存在狀態 {expected_ready} 不符"
            )

    def test_E5_context_model_ready_flag(self):
        """model_ready 應為布林值，反映至少一個模型檔案存在。"""
        resp = self.client.get('/analyzer/simulation/')
        self.assertIn('model_ready', resp.context)
        self.assertIsInstance(resp.context['model_ready'], bool)

    def test_E6_context_has_cnn_threshold_and_latent_dim(self):
        """context 應包含 cnn_threshold 與 cnn_latent_dim（頁面 KPI 卡片需要）。"""
        resp = self.client.get('/analyzer/simulation/')
        self.assertIn('cnn_threshold', resp.context)
        self.assertIn('cnn_latent_dim', resp.context)


# ════════════════════════════════════════════════════════════════════════════
# 根因 F：黑箱問題 — 每筆結果需包含透明化欄位
# ════════════════════════════════════════════════════════════════════════════
class RootCauseF_Transparency(_BaseSimTest):
    """
    根因 F 驗證：舊版 API 每筆結果只有 index/score/is_anomaly，
    使用者完全不知道哪個封包真正觸發規則、規則看到了什麼。
    修正後每筆結果應包含 header、rule_triggered_here、rule_detail，
    以及整體回應中的 sample_visual、timing_ms、comparison_summary。
    """

    def test_F1_result_has_header_field(self):
        """每筆封包結果應包含 header（解析後的標頭欄位字典）。"""
        resp = self._post_api('syn_flood', intensity=1.0)
        data = resp.json()
        self.assertTrue(data['ok'])
        for r in data['results']:
            self.assertIn('header', r,
                          f"封包 #{r['index']} 缺少 header 欄位")
            self.assertIsInstance(r['header'], dict)
            self.assertIn('protocol', r['header'],
                          f"封包 #{r['index']} header 缺少 protocol")

    def test_F2_header_contains_expected_fields(self):
        """header 應包含 src_ip、dst_ip、protocol 等標準欄位（可為 None 但必須有鍵）。"""
        resp = self._post_api('icmp_flood', intensity=1.0)
        data = resp.json()
        required_keys = ('protocol', 'src_ip', 'dst_ip')
        for r in data['results'][:5]:  # 只檢查前5筆
            for key in required_keys:
                self.assertIn(key, r['header'],
                              f"封包 #{r['index']} header 缺少 '{key}' 欄位")

    def test_F3_rule_triggered_here_field(self):
        """每筆結果應有 rule_triggered_here（布林值）。"""
        resp = self._post_api('syn_flood', intensity=1.0)
        data = resp.json()
        for r in data['results']:
            self.assertIn('rule_triggered_here', r)
            self.assertIsInstance(r['rule_triggered_here'], bool)

    def test_F4_at_least_one_packet_directly_triggers_rule(self):
        """攻擊場景下應至少有一個封包直接觸發規則（rule_triggered_here=True）。"""
        for attack_type in ('syn_flood', 'icmp_flood', 'arp_spoof'):
            with self.subTest(attack_type=attack_type):
                resp = self._post_api(attack_type, intensity=1.0)
                data = resp.json()
                if not data['rule_based_anomaly_detected']:
                    continue
                triggered = [r for r in data['results'] if r.get('rule_triggered_here')]
                self.assertGreater(
                    len(triggered), 0,
                    f"{attack_type}：應有至少一個封包的 rule_triggered_here=True"
                )

    def test_F5_rule_detail_is_list(self):
        """每筆結果的 rule_detail 應為 list（可為空列表）。"""
        resp = self._post_api('syn_flood', intensity=1.0)
        data = resp.json()
        for r in data['results']:
            self.assertIn('rule_detail', r)
            self.assertIsInstance(r['rule_detail'], list)

    def test_F6_rule_detail_contains_attack_type_when_triggered(self):
        """直接觸發規則的封包，其 rule_detail 應包含 attack_type、severity、detail。"""
        resp = self._post_api('icmp_flood', intensity=1.0)
        data = resp.json()
        triggered_pkts = [r for r in data['results'] if r.get('rule_triggered_here')]
        for r in triggered_pkts[:3]:
            for item in r['rule_detail']:
                self.assertIn('attack_type', item)
                self.assertIn('severity', item)
                self.assertIn('detail', item)

    def test_F7_sample_visual_field_present(self):
        """API 應回傳 sample_visual（封包影像視覺化），用於比對黑箱問題。"""
        resp = self._post_api('syn_flood', intensity=1.0)
        data = resp.json()
        if not data.get('model_used'):
            self.skipTest("模型未載入，跳過 sample_visual 測試")
        self.assertIn('sample_visual', data)
        sv = data['sample_visual']
        self.assertIsNotNone(sv)
        self.assertIn('most_anomalous_packet_index', sv)
        self.assertIn('most_anomalous_packet_score', sv)

    def test_F8_sample_visual_contains_base64_images(self):
        """sample_visual 中的影像應為 base64 PNG data URL。"""
        resp = self._post_api('icmp_flood', intensity=1.0)
        data = resp.json()
        if not data.get('model_used'):
            self.skipTest("模型未載入，跳過 sample_visual 影像測試")
        sv = data.get('sample_visual', {})
        for key in ('most_anomalous_image', 'baseline_average_image'):
            self.assertIn(key, sv)
            img_str = sv.get(key, '')
            if img_str:
                self.assertTrue(
                    img_str.startswith('data:image/png;base64,'),
                    f"{key} 應為 base64 PNG data URL"
                )

    def test_F9_comparison_summary_present(self):
        """API 應回傳 comparison_summary（規則引擎 vs CNN 比對摘要）。"""
        resp = self._post_api('syn_flood', intensity=1.0)
        data = resp.json()
        self.assertIn('comparison_summary', data)
        cs = data['comparison_summary']
        required = ('rule_engine_verdict', 'rule_engine_alert_count',
                    'cnn_reference_flagged_count_in_sample',
                    'agree', 'authoritative_source')
        for key in required:
            self.assertIn(key, cs, f"comparison_summary 缺少 '{key}' 欄位")

    def test_F10_authoritative_source_is_rule_engine(self):
        """最終判定依據應永遠是 rule_engine，不應是 cnn。"""
        resp = self._post_api('icmp_flood', intensity=1.0)
        data = resp.json()
        cs = data.get('comparison_summary', {})
        self.assertEqual(
            cs.get('authoritative_source'), 'rule_engine',
            "最終判定依據應為 rule_engine，不應為 cnn"
        )

    def test_F11_forced_by_demo_shortcut_always_false(self):
        """
        修正後所有攻擊類型的 forced_by_demo_shortcut 應永遠為 False，
        不再有強制保底的黑箱邏輯。
        """
        for attack_type in ('syn_flood', 'icmp_flood', 'arp_spoof',
                            'dns_amplification', 'udp_flood', 'port_scan'):
            with self.subTest(attack_type=attack_type):
                resp = self._post_api(attack_type, intensity=1.0)
                data = resp.json()
                bd = data.get('behavior_detection', {})
                self.assertFalse(
                    bd.get('forced_by_demo_shortcut'),
                    f"{attack_type} 的 forced_by_demo_shortcut 應為 False，"
                    f"不應有強制保底的黑箱邏輯"
                )

    def test_F12_behavior_detection_threshold_matches_attack_type(self):
        """
        根因 B/F 合併驗證：behavior_detection['threshold'] 應依攻擊類型
        回傳正確的正式門檻，不再永遠回傳 SYN 的門檻（100）。
        """
        expected = {
            'syn_flood':  ALERT_THRESHOLD_SYN,
            'port_scan':  ALERT_THRESHOLD_PORTS,
            'icmp_flood': ALERT_THRESHOLD_ICMP,
            'udp_flood':  ALERT_THRESHOLD_UDP,
        }
        for attack_type, expected_threshold in expected.items():
            with self.subTest(attack_type=attack_type):
                resp = self._post_api(attack_type, intensity=1.0)
                data = resp.json()
                bd = data.get('behavior_detection', {})
                self.assertEqual(
                    bd.get('threshold'), expected_threshold,
                    f"{attack_type} 的 threshold 應為 {expected_threshold}，"
                    f"實際為 {bd.get('threshold')}"
                )

    def test_F13_packet_count_displayed_leq_total(self):
        """顯示的封包數應 <= 總封包數（大量封包時做等距抽樣）。"""
        resp = self._post_api('syn_flood', intensity=3.0)
        data = resp.json()
        self.assertLessEqual(
            data.get('packet_count_displayed', 0),
            data.get('packet_count', 0),
            "顯示封包數不應超過總封包數"
        )
        self.assertLessEqual(
            data.get('packet_count_displayed', 0), 400,
            "顯示封包數上限應為 400"
        )


# ════════════════════════════════════════════════════════════════════════════
# 整合測試：backward compatibility（向下相容舊版 packet_count 參數）
# ════════════════════════════════════════════════════════════════════════════
class BackwardCompatibility(_BaseSimTest):
    """
    修正後的 API 應相容舊版的 packet_count 請求參數，
    確保既有的呼叫端（測試、前端）不需要同步修改。
    """

    def test_compat_old_packet_count_param(self):
        """舊版 packet_count=60 應被換算為 intensity，API 正常回傳。"""
        resp = self._post_api('normal_traffic', packet_count=60)
        data = resp.json()
        self.assertTrue(data['ok'])
        self.assertIn('intensity', data)

    def test_compat_packet_count_to_intensity_ratio(self):
        """packet_count=60 應換算為 intensity=1.0（60/60=1.0）。"""
        resp = self._post_api('normal_traffic', packet_count=60)
        data = resp.json()
        self.assertAlmostEqual(data.get('intensity', 0), 1.0, places=1)

    def test_compat_intensity_overrides_packet_count(self):
        """同時提供 intensity 與 packet_count 時，intensity 應優先。"""
        body = json.dumps({
            'attack_type': 'normal_traffic',
            'model_key': 'unsupervised_vae',
            'intensity': 2.0,
            'packet_count': 10,
        })
        resp = self.client.post(
            '/analyzer/simulation/api/',
            data=body, content_type='application/json'
        )
        data = resp.json()
        self.assertAlmostEqual(data.get('intensity', 0), 2.0, places=1)

    def test_compat_all_original_response_fields_present(self):
        """舊版 API 回傳的所有欄位應仍然存在（向下相容）。"""
        resp = self._post_api('icmp_flood', packet_count=60)
        data = resp.json()
        original_fields = (
            'ok', 'attack_type', 'packet_count', 'anomaly_count',
            'avg_score', 'results', 'rule_based_anomaly_detected',
            'behavior_detection', 'baseline', 'pcap_filename',
        )
        for field in original_fields:
            self.assertIn(field, data, f"向下相容：舊版欄位 '{field}' 不應消失")
