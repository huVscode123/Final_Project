# ============================================================
# core/tests/test_pcap_flow_converter.py
# 單元測試：PCAP → CICFlowMeter 特徵轉換管線
#
# 測試範圍：
#   1. pcap_flow_converter.py — 統計小工具、封包擷取、Flow 重組、特徵輸出
#   2. convert_test_pcap.py   — CLI 入口（argparse 驗證）
#   3. fit_reference_scaler.py — Scaler 匯出（旁路前處理邏輯）
#   4. score_converted_pcap.py — 計分管線（特徵對齊 + 模型推論）
#
# 所有測試皆使用合成資料（Scapy 建構封包 / 模擬 CSV），不需要任何
# 外部資料集或 PCAP 檔案即可執行。
# ============================================================

import csv
import os
import sys
import tempfile
import unittest
import pickle

import numpy as np

# 確保 core/ 目錄在 sys.path 中
_CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)


# ──────────────────────────────────────────────────────────
# 1. 統計小工具測試
# ──────────────────────────────────────────────────────────
class TestStatsUtils(unittest.TestCase):
    """測試 pcap_flow_converter 的統計輔助函式。"""

    def setUp(self):
        from pcap_flow_converter import _stats, _iat_stats, _count_active_segments
        self._stats = _stats
        self._iat_stats = _iat_stats
        self._count_active_segments = _count_active_segments

    def test_stats_empty(self):
        """空清單應回傳全 0。"""
        mean, std, mx, mn = self._stats([])
        self.assertEqual((mean, std, mx, mn), (0.0, 0.0, 0.0, 0.0))

    def test_stats_single(self):
        """單一元素：mean = 該值，std = 0。"""
        mean, std, mx, mn = self._stats([5.0])
        self.assertAlmostEqual(mean, 5.0)
        self.assertAlmostEqual(std, 0.0)
        self.assertAlmostEqual(mx, 5.0)
        self.assertAlmostEqual(mn, 5.0)

    def test_stats_multiple(self):
        """多個元素的正確性。"""
        vals = [2.0, 4.0, 6.0, 8.0]
        mean, std, mx, mn = self._stats(vals)
        self.assertAlmostEqual(mean, 5.0)
        self.assertAlmostEqual(mx, 8.0)
        self.assertAlmostEqual(mn, 2.0)
        self.assertGreater(std, 0.0)

    def test_iat_stats_less_than_2(self):
        """少於 2 個時間點，無法計算 IAT，應回傳全 0。"""
        total, mean, std, mx, mn = self._iat_stats([1.0])
        self.assertEqual((total, mean, std, mx, mn), (0.0, 0.0, 0.0, 0.0, 0.0))

    def test_iat_stats_two_points(self):
        """2 個時間點的 IAT 計算。"""
        total, mean, std, mx, mn = self._iat_stats([0.0, 0.001])  # 1ms
        self.assertAlmostEqual(total, 1000.0, places=1)  # 1000 微秒
        self.assertAlmostEqual(mean, 1000.0, places=1)

    def test_iat_stats_three_points(self):
        """3 個時間點，2 個間隔。"""
        total, mean, std, mx, mn = self._iat_stats([0.0, 0.001, 0.003])
        # 間隔: 1000us, 2000us → total=3000, mean=1500
        self.assertAlmostEqual(total, 3000.0, places=1)
        self.assertAlmostEqual(mean, 1500.0, places=1)
        self.assertAlmostEqual(mx, 2000.0, places=1)
        self.assertAlmostEqual(mn, 1000.0, places=1)

    def test_count_active_segments_no_gap(self):
        """所有封包間隔都在 idle_threshold 以內，應為 1 個活躍區間。"""
        times = [0.0, 0.1, 0.2, 0.3]
        self.assertEqual(self._count_active_segments(times, 1.0), 1)

    def test_count_active_segments_with_gap(self):
        """有超過 idle_threshold 的間隔，應切分為多段。"""
        times = [0.0, 0.1, 0.2, 5.0, 5.1]  # 0.2→5.0 gap=4.8s > 1.0s
        self.assertEqual(self._count_active_segments(times, 1.0), 2)

    def test_count_active_segments_empty(self):
        """空清單至少回傳 1。"""
        self.assertEqual(self._count_active_segments([], 1.0), 1)


# ──────────────────────────────────────────────────────────
# 2. 標籤產生器測試
# ──────────────────────────────────────────────────────────
class TestLabelers(unittest.TestCase):
    """測試 make_static_labeler 和 make_ip_time_labeler。"""

    def setUp(self):
        from pcap_flow_converter import make_static_labeler, make_ip_time_labeler
        self.make_static = make_static_labeler
        self.make_ip_time = make_ip_time_labeler

    def test_static_labeler(self):
        """固定標記器對任何輸入都回傳相同標籤。"""
        fn = self.make_static("BENIGN")
        key = (("1.1.1.1", 80), ("2.2.2.2", 12345), 6)
        self.assertEqual(fn(key, 0.0, 1.0), "BENIGN")

    def test_ip_time_labeler_ip_match(self):
        """只依 IP 判斷（不指定時段）。"""
        fn = self.make_ip_time(["10.0.0.1"])
        key_match = (("10.0.0.1", 80), ("192.168.1.1", 12345), 6)
        key_no_match = (("192.168.1.2", 80), ("192.168.1.1", 12345), 6)
        self.assertEqual(fn(key_match, 0.0, 1.0), "ATTACK")
        self.assertEqual(fn(key_no_match, 0.0, 1.0), "BENIGN")

    def test_ip_time_labeler_with_time_window(self):
        """IP 命中但時間不在範圍內，應標記為 BENIGN。"""
        fn = self.make_ip_time(
            ["10.0.0.1"],
            time_start=100.0,
            time_end=200.0,
        )
        key = (("10.0.0.1", 80), ("192.168.1.1", 12345), 6)
        # Flow 時間在範圍內
        self.assertEqual(fn(key, 120.0, 150.0), "ATTACK")
        # Flow 時間完全在範圍外
        self.assertEqual(fn(key, 50.0, 80.0), "BENIGN")
        # Flow 時間部分重疊
        self.assertEqual(fn(key, 190.0, 210.0), "ATTACK")

    def test_ip_time_labeler_dst_match(self):
        """攻擊者 IP 出現在目的端也應命中。"""
        fn = self.make_ip_time(["10.0.0.1"])
        key = (("192.168.1.1", 12345), ("10.0.0.1", 80), 6)
        self.assertEqual(fn(key, 0.0, 1.0), "ATTACK")


# ──────────────────────────────────────────────────────────
# 3. 封包擷取函式測試
# ──────────────────────────────────────────────────────────
class TestExtractPacketInfo(unittest.TestCase):
    """測試 _extract_packet_info 封包欄位擷取。"""

    def setUp(self):
        from pcap_flow_converter import _extract_packet_info
        self._extract = _extract_packet_info

    def test_tcp_packet(self):
        """TCP 封包應正確擷取 IP、Port、Flag、Window。"""
        from scapy.all import Ether, IP, TCP
        pkt = Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=12345, dport=80, flags="S", window=65535)
        pkt.time = 1000.0
        info = self._extract(pkt)
        self.assertIsNotNone(info)
        self.assertEqual(info["src_ip"], "1.1.1.1")
        self.assertEqual(info["dst_ip"], "2.2.2.2")
        self.assertEqual(info["src_port"], 12345)
        self.assertEqual(info["dst_port"], 80)
        self.assertEqual(info["protocol"], 6)  # TCP
        self.assertIn("S", info["tcp_flags"])
        self.assertEqual(info["tcp_window"], 65535)

    def test_udp_packet(self):
        """UDP 封包應正確擷取且 tcp_flags 為 None。"""
        from scapy.all import Ether, IP, UDP
        pkt = Ether() / IP(src="3.3.3.3", dst="4.4.4.4") / UDP(sport=5353, dport=53)
        pkt.time = 2000.0
        info = self._extract(pkt)
        self.assertIsNotNone(info)
        self.assertEqual(info["src_port"], 5353)
        self.assertEqual(info["dst_port"], 53)
        self.assertEqual(info["protocol"], 17)  # UDP
        self.assertIsNone(info["tcp_flags"])

    def test_arp_packet_returns_none(self):
        """純 ARP 封包（非 IP 層）應回傳 None。"""
        from scapy.all import Ether, ARP
        pkt = Ether() / ARP()
        info = self._extract(pkt)
        self.assertIsNone(info)

    def test_payload_length(self):
        """payload_len 應 = total_len - header_len。"""
        from scapy.all import Ether, IP, TCP, Raw
        payload_data = b"Hello, World!"
        pkt = Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1111, dport=2222) / Raw(payload_data)
        pkt.time = 3000.0
        info = self._extract(pkt)
        self.assertIsNotNone(info)
        self.assertEqual(info["payload_len"], max(0, info["total_len"] - info["header_len"]))
        self.assertGreater(info["payload_len"], 0)


# ──────────────────────────────────────────────────────────
# 4. FlowRecord 測試
# ──────────────────────────────────────────────────────────
class TestFlowRecord(unittest.TestCase):
    """測試 _FlowRecord 的封包累加與特徵化。"""

    def setUp(self):
        from pcap_flow_converter import _FlowRecord
        self._FlowRecord = _FlowRecord

    def test_single_fwd_packet(self):
        """只有 1 個 Forward 封包的 Flow。"""
        rec = self._FlowRecord("1.1.1.1", 80, "2.2.2.2", 12345, 6, 100.0)
        rec.add_packet("fwd", 100.0, 60, 40, 20, "S", 65535)
        row = rec.finalize(idle_threshold=1.0, bulk_threshold=1.0)

        self.assertEqual(row["Total Fwd Packets"], 1)
        self.assertEqual(row["Total Backward Packets"], 0)
        self.assertEqual(row["Total Length of Fwd Packets"], 60)
        self.assertEqual(row["Flow Duration"], 0.0)  # 只有 1 個封包
        self.assertEqual(row["SYN Flag Count"], 1)
        self.assertEqual(row["Init_Win_bytes_forward"], 65535)
        self.assertEqual(row["Init_Win_bytes_backward"], 0)

    def test_bidirectional_flow(self):
        """雙向 Flow（TCP 三向交握）。"""
        rec = self._FlowRecord("1.1.1.1", 80, "2.2.2.2", 12345, 6, 100.0)
        # SYN
        rec.add_packet("fwd", 100.0, 60, 40, 0, "S", 65535)
        # SYN-ACK
        rec.add_packet("bwd", 100.001, 60, 40, 0, "SA", 32768)
        # ACK
        rec.add_packet("fwd", 100.002, 52, 40, 0, "A", 65535)

        row = rec.finalize(1.0, 1.0)
        self.assertEqual(row["Total Fwd Packets"], 2)
        self.assertEqual(row["Total Backward Packets"], 1)
        self.assertEqual(row["SYN Flag Count"], 2)  # S + SA
        self.assertEqual(row["ACK Flag Count"], 2)  # SA + A
        self.assertEqual(row["Init_Win_bytes_forward"], 65535)
        self.assertEqual(row["Init_Win_bytes_backward"], 32768)
        self.assertGreater(row["Flow Duration"], 0.0)


# ──────────────────────────────────────────────────────────
# 5. PcapFlowConverter 端對端測試（合成 PCAP）
# ──────────────────────────────────────────────────────────
class TestPcapFlowConverterEndToEnd(unittest.TestCase):
    """使用 Scapy 合成封包產生 PCAP，測試完整轉換流程。"""

    def _create_test_pcap(self, pcap_path):
        """建立一個含有 TCP 三向交握 + 資料傳輸 + FIN 的測試 PCAP。"""
        from scapy.all import Ether, IP, TCP, Raw, wrpcap

        base_time = 1000.0
        pkts = []

        # TCP 三向交握
        syn = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=40000, dport=80, flags="S", window=65535)
        syn.time = base_time
        pkts.append(syn)

        syn_ack = Ether() / IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=80, dport=40000, flags="SA", window=32768)
        syn_ack.time = base_time + 0.001
        pkts.append(syn_ack)

        ack = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=40000, dport=80, flags="A", window=65535)
        ack.time = base_time + 0.002
        pkts.append(ack)

        # 資料傳輸（Forward）
        data = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=40000, dport=80, flags="PA", window=65535) / Raw(b"GET / HTTP/1.1\r\n")
        data.time = base_time + 0.010
        pkts.append(data)

        # 資料回應（Backward）
        resp = Ether() / IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=80, dport=40000, flags="PA", window=32768) / Raw(b"HTTP/1.1 200 OK\r\n")
        resp.time = base_time + 0.020
        pkts.append(resp)

        # FIN
        fin = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=40000, dport=80, flags="FA", window=65535)
        fin.time = base_time + 0.030
        pkts.append(fin)

        wrpcap(pcap_path, pkts)
        return pkts

    def test_convert_produces_valid_csv(self):
        """轉換後的 CSV 應包含正確的欄位數與至少 1 條 Flow。"""
        from pcap_flow_converter import PcapFlowConverter, FEATURE_COLUMNS, OUTPUT_COLUMNS

        with tempfile.TemporaryDirectory() as tmpdir:
            pcap_path = os.path.join(tmpdir, "test.pcap")
            csv_path = os.path.join(tmpdir, "test_flows.csv")
            self._create_test_pcap(pcap_path)

            converter = PcapFlowConverter(flow_timeout=120.0)
            converter.convert(pcap_path, csv_path)

            # 讀取並驗證 CSV
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertGreaterEqual(len(rows), 1, "至少應產生 1 條 Flow")
            # 驗證所有輸出欄位都存在
            for col in OUTPUT_COLUMNS:
                self.assertIn(col, rows[0], f"缺少欄位: {col}")

    def test_convert_feature_values(self):
        """驗證特徵值的合理性。"""
        from pcap_flow_converter import PcapFlowConverter

        with tempfile.TemporaryDirectory() as tmpdir:
            pcap_path = os.path.join(tmpdir, "test.pcap")
            csv_path = os.path.join(tmpdir, "test_flows.csv")
            self._create_test_pcap(pcap_path)

            converter = PcapFlowConverter(flow_timeout=120.0)
            converter.convert(pcap_path, csv_path)

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            row = rows[0]

            # 基本方向統計
            total_fwd = int(row["Total Fwd Packets"])
            total_bwd = int(row["Total Backward Packets"])
            self.assertGreater(total_fwd, 0, "Forward 封包數應 > 0")
            self.assertGreater(total_bwd, 0, "Backward 封包數應 > 0")
            self.assertEqual(total_fwd + total_bwd, 6, "總封包數應為 6")

            # 時間持續度 > 0（因為多個封包有時間差）
            duration = float(row["Flow Duration"])
            self.assertGreater(duration, 0.0, "Flow Duration 應 > 0")

            # Flag 計數
            syn_count = int(row["SYN Flag Count"])
            fin_count = int(row["FIN Flag Count"])
            self.assertGreaterEqual(syn_count, 1, "至少要有 SYN")
            self.assertGreaterEqual(fin_count, 1, "至少要有 FIN")

            # Init Window
            init_win_fwd = int(row["Init_Win_bytes_forward"])
            init_win_bwd = int(row["Init_Win_bytes_backward"])
            self.assertGreater(init_win_fwd, 0)
            self.assertGreater(init_win_bwd, 0)

    def test_convert_with_static_label(self):
        """使用固定標記器時，所有 Flow 標籤應相同。"""
        from pcap_flow_converter import PcapFlowConverter, make_static_labeler

        with tempfile.TemporaryDirectory() as tmpdir:
            pcap_path = os.path.join(tmpdir, "test.pcap")
            csv_path = os.path.join(tmpdir, "test_flows.csv")
            self._create_test_pcap(pcap_path)

            converter = PcapFlowConverter(flow_timeout=120.0)
            converter.convert(pcap_path, csv_path, label_fn=make_static_labeler("ATTACK"))

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            for row in rows:
                self.assertEqual(row["Label"], "ATTACK")

    def test_convert_with_ip_labeler(self):
        """使用 IP 標記器時，含攻擊者 IP 的 Flow 應標記為 ATTACK。"""
        from pcap_flow_converter import PcapFlowConverter, make_ip_time_labeler

        with tempfile.TemporaryDirectory() as tmpdir:
            pcap_path = os.path.join(tmpdir, "test.pcap")
            csv_path = os.path.join(tmpdir, "test_flows.csv")
            self._create_test_pcap(pcap_path)

            labeler = make_ip_time_labeler(["10.0.0.1"])
            converter = PcapFlowConverter(flow_timeout=120.0)
            converter.convert(pcap_path, csv_path, label_fn=labeler)

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # 10.0.0.1 有參與的 Flow 應標記為 ATTACK
            for row in rows:
                self.assertEqual(row["Label"], "ATTACK")

    def test_flow_timeout_splits_flows(self):
        """超過 flow_timeout 的封包應被分割為不同的 Flow。"""
        from scapy.all import Ether, IP, TCP, wrpcap
        from pcap_flow_converter import PcapFlowConverter

        with tempfile.TemporaryDirectory() as tmpdir:
            pcap_path = os.path.join(tmpdir, "timeout.pcap")
            csv_path = os.path.join(tmpdir, "timeout_flows.csv")

            pkts = []
            # Flow 1: t=0
            pkt1 = Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1111, dport=80, flags="S")
            pkt1.time = 0.0
            pkts.append(pkt1)

            # Flow 2: t=200（超過 flow_timeout=120）
            pkt2 = Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1111, dport=80, flags="S")
            pkt2.time = 200.0
            pkts.append(pkt2)

            wrpcap(pcap_path, pkts)

            converter = PcapFlowConverter(flow_timeout=120.0)
            converter.convert(pcap_path, csv_path)

            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 2, "超過 timeout 應分割為 2 條 Flow")


# ──────────────────────────────────────────────────────────
# 6. Scaler 匯出與重現測試
# ──────────────────────────────────────────────────────────
class TestFitReferenceScaler(unittest.TestCase):
    """測試 fit_reference_scaler.py 的旁路 scaler 匯出邏輯。"""

    def _create_fake_csv(self, tmpdir, n_rows=100):
        """建立模擬的 CICFlowMeter CSV（含 Label 欄位）。"""
        csv_path = os.path.join(tmpdir, "fake_traffic.csv")
        rng = np.random.default_rng(42)

        columns = ["Feature1", "Feature2", "Feature3", "Label"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for i in range(n_rows):
                label = "BENIGN" if i < n_rows * 0.7 else "DoS"
                writer.writerow([
                    rng.uniform(0, 100),
                    rng.uniform(0, 1000),
                    rng.uniform(-10, 10),
                    label,
                ])
        return csv_path

    def test_scaler_output_contains_expected_keys(self):
        """匯出的 pickle 應包含 scaler、feature_columns、image_size。"""
        from fit_reference_scaler import _load_normal_matrix
        from sklearn.preprocessing import MinMaxScaler

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_fake_csv(tmpdir)
            X_normal, feature_cols = _load_normal_matrix(tmpdir, recursive=False)

            self.assertGreater(X_normal.shape[0], 0, "應有正常流量筆數")
            self.assertGreater(len(feature_cols), 0, "應有特徵欄位")

            scaler = MinMaxScaler()
            scaler.fit(X_normal)

            pkl_path = os.path.join(tmpdir, "scaler.pkl")
            payload = {
                "dataset": "test",
                "scaler": scaler,
                "feature_columns": feature_cols,
                "image_size": 32,
            }
            with open(pkl_path, "wb") as f:
                pickle.dump(payload, f)

            with open(pkl_path, "rb") as f:
                loaded = pickle.load(f)

            self.assertIn("scaler", loaded)
            self.assertIn("feature_columns", loaded)
            self.assertIn("image_size", loaded)
            self.assertEqual(loaded["image_size"], 32)
            self.assertEqual(len(loaded["feature_columns"]), len(feature_cols))

    def test_scaler_transform_consistency(self):
        """用匯出的 scaler transform 應與原始 fit_transform 結果一致。"""
        from fit_reference_scaler import _load_normal_matrix
        from sklearn.preprocessing import MinMaxScaler

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_fake_csv(tmpdir)
            X_normal, _ = _load_normal_matrix(tmpdir, recursive=False)

            scaler = MinMaxScaler()
            X_transformed = scaler.fit_transform(X_normal)

            # 序列化再反序列化
            pkl_path = os.path.join(tmpdir, "scaler.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump({"scaler": scaler}, f)
            with open(pkl_path, "rb") as f:
                loaded_scaler = pickle.load(f)["scaler"]

            X_re_transformed = loaded_scaler.transform(X_normal)
            np.testing.assert_array_almost_equal(X_transformed, X_re_transformed)


# ──────────────────────────────────────────────────────────
# 7. 特徵欄位完整性測試
# ──────────────────────────────────────────────────────────
class TestFeatureColumnsCompleteness(unittest.TestCase):
    """確認轉換器輸出的 78 個特徵欄位與 dataset_loader 相容。"""

    def test_feature_count(self):
        """應有 78 個特徵欄位。"""
        from pcap_flow_converter import FEATURE_COLUMNS
        self.assertEqual(len(FEATURE_COLUMNS), 78)

    def test_output_columns_order(self):
        """OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + [Label]。"""
        from pcap_flow_converter import METADATA_COLUMNS, FEATURE_COLUMNS, OUTPUT_COLUMNS
        expected = METADATA_COLUMNS + FEATURE_COLUMNS + ["Label"]
        self.assertEqual(OUTPUT_COLUMNS, expected)

    def test_key_features_present(self):
        """確認關鍵特徵都在清單中。"""
        from pcap_flow_converter import FEATURE_COLUMNS
        key_features = [
            "Destination Port", "Flow Duration",
            "Total Fwd Packets", "Total Backward Packets",
            "Flow Bytes/s", "Flow Packets/s",
            "SYN Flag Count", "FIN Flag Count", "ACK Flag Count",
            "Init_Win_bytes_forward", "Init_Win_bytes_backward",
            "Active Mean", "Idle Mean",
        ]
        for feat in key_features:
            self.assertIn(feat, FEATURE_COLUMNS, f"缺少關鍵特徵: {feat}")

    def test_fwd_header_length_dot_1_present(self):
        """CICFlowMeter 的重複欄位 Fwd Header Length.1 必須存在。"""
        from pcap_flow_converter import FEATURE_COLUMNS
        self.assertIn("Fwd Header Length.1", FEATURE_COLUMNS)


# ──────────────────────────────────────────────────────────
# 8. features_to_image 整合測試
# ──────────────────────────────────────────────────────────
class TestFeaturesImageIntegration(unittest.TestCase):
    """確認轉換後的特徵能正確透過 features_to_image 轉為模型輸入。"""

    def test_features_to_image_shape(self):
        """78 維特徵經 pad → 32x32 影像。"""
        from cnn_autoencoder import features_to_image
        X = np.random.rand(5, 78).astype(np.float32)
        img = features_to_image(X, image_size=32)
        self.assertEqual(img.shape, (5, 1, 32, 32))

    def test_features_to_image_preserves_values(self):
        """前 78 維的值應被保留（非全零）。"""
        from cnn_autoencoder import features_to_image
        X = np.ones((1, 78), dtype=np.float32)
        img = features_to_image(X, image_size=32)
        flat = img.reshape(1, -1)
        # 前 78 個值應為 1.0
        np.testing.assert_array_almost_equal(flat[0, :78], np.ones(78))
        # 第 79 個值起應為 0.0（padding）
        np.testing.assert_array_almost_equal(flat[0, 78:], np.zeros(1024 - 78))


# ──────────────────────────────────────────────────────────
# 9. Active/Idle 與 Bulk 統計測試
# ──────────────────────────────────────────────────────────
class TestActiveIdleAndBulk(unittest.TestCase):
    """測試 Active/Idle 切分與 Bulk 偵測的邊界情境。"""

    def setUp(self):
        from pcap_flow_converter import _active_idle_stats, _bulk_stats
        self._active_idle = _active_idle_stats
        self._bulk_stats = _bulk_stats

    def test_active_idle_single_packet(self):
        """只有 1 個封包無法計算，回傳全 0。"""
        active, idle = self._active_idle([1.0], idle_threshold=1.0)
        self.assertEqual(active, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(idle, (0.0, 0.0, 0.0, 0.0))

    def test_active_idle_continuous_flow(self):
        """連續 Flow（無閒置）：只有 Active，Idle 全 0。"""
        times = [0.0, 0.1, 0.2, 0.3, 0.4]
        active, idle = self._active_idle(times, idle_threshold=1.0)
        self.assertGreater(active[0], 0.0)  # active mean > 0
        self.assertEqual(idle, (0.0, 0.0, 0.0, 0.0))

    def test_bulk_stats_too_few_packets(self):
        """不足 4 個封包時不構成 bulk。"""
        result = self._bulk_stats([0.0, 0.1, 0.2], [100, 100, 100], bulk_threshold=1.0)
        self.assertEqual(result, (0.0, 0.0, 0.0))

    def test_bulk_stats_with_valid_bulk(self):
        """4+ 連續封包且間隔 < bulk_threshold 應偵測到 bulk。"""
        times = [0.0, 0.1, 0.2, 0.3, 0.4]
        lens = [100, 200, 150, 180, 120]
        avg_bytes, avg_pkts, avg_rate = self._bulk_stats(times, lens, bulk_threshold=1.0)
        self.assertGreater(avg_bytes, 0.0)
        self.assertGreaterEqual(avg_pkts, 4.0)


# ──────────────────────────────────────────────────────────
# 10. 多協定混合 PCAP 測試
# ──────────────────────────────────────────────────────────
class TestMixedProtocolPcap(unittest.TestCase):
    """TCP + UDP + ARP 混合封包的轉換測試。"""

    def test_mixed_protocols(self):
        """轉換器應正確處理混合協定，ARP 被跳過。"""
        from scapy.all import Ether, IP, TCP, UDP, ARP, wrpcap
        from pcap_flow_converter import PcapFlowConverter

        with tempfile.TemporaryDirectory() as tmpdir:
            pcap_path = os.path.join(tmpdir, "mixed.pcap")
            csv_path = os.path.join(tmpdir, "mixed_flows.csv")

            pkts = []
            # TCP Flow
            tcp_pkt = Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=111, dport=80, flags="S")
            tcp_pkt.time = 0.0
            pkts.append(tcp_pkt)

            # UDP Flow
            udp_pkt = Ether() / IP(src="3.3.3.3", dst="4.4.4.4") / UDP(sport=5353, dport=53)
            udp_pkt.time = 0.001
            pkts.append(udp_pkt)

            # ARP（應被忽略）
            arp_pkt = Ether() / ARP()
            arp_pkt.time = 0.002
            pkts.append(arp_pkt)

            wrpcap(pcap_path, pkts)

            converter = PcapFlowConverter(flow_timeout=120.0)
            converter.convert(pcap_path, csv_path)

            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            # 應有 2 條 Flow（TCP + UDP），ARP 被跳過
            self.assertEqual(len(rows), 2, "應有 TCP + UDP 兩條 Flow，ARP 被跳過")

            protocols = {int(row["Protocol"]) for row in rows}
            self.assertIn(6, protocols, "應包含 TCP (protocol=6)")
            self.assertIn(17, protocols, "應包含 UDP (protocol=17)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
