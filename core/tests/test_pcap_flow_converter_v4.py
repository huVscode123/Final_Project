# ============================================================
# core/tests/test_pcap_flow_converter_v4.py
# pcap_flow_converter.py（v4.1）的單元測試：TCP Flow 結束規則、Bulk 模式、
# 公式修正（Down/Up Ratio、Packet Length N+1、Average Packet Size）。
#
# 不需要 scapy 或任何 PCAP：直接餵「封包資訊 dict」給 PcapFlowConverter._ingest()；
# 若環境沒有 scapy，會先放入最小替身模組，讓 import 成功。
# 端對端的 convert() 測試以假的 PcapReader 取代 scapy。
# ============================================================
import csv
import os
import sys
import tempfile
import types
import unittest

_CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

try:  # pragma: no cover - 取決於環境
    import scapy  # noqa: F401
except ImportError:  # pragma: no cover
    _all = types.ModuleType("scapy.all")
    _all.PcapReader = object
    _inet = types.ModuleType("scapy.layers.inet")
    _inet.IP = _inet.TCP = _inet.UDP = object
    _inet6 = types.ModuleType("scapy.layers.inet6")
    _inet6.IPv6 = object
    _layers = types.ModuleType("scapy.layers")
    _scapy = types.ModuleType("scapy")
    sys.modules.update({
        "scapy": _scapy, "scapy.all": _all, "scapy.layers": _layers,
        "scapy.layers.inet": _inet, "scapy.layers.inet6": _inet6,
    })

import pcap_flow_converter as pfc  # noqa: E402

A, B = "10.0.0.1", "10.0.0.2"
APORT, BPORT = 40000, 80


def pkt(t, src, flags="A", payload=0, hdr=20, win=1000, proto=6):
    """建立 _extract_packet_info() 格式的封包資訊；src 為 'A' 或 'B'。"""
    if src == "A":
        s_ip, s_port, d_ip, d_port = A, APORT, B, BPORT
    else:
        s_ip, s_port, d_ip, d_port = B, BPORT, A, APORT
    return {"time": t, "src_ip": s_ip, "dst_ip": d_ip, "src_port": s_port,
            "dst_port": d_port, "protocol": proto, "payload_bytes": payload,
            "header_len": hdr, "tcp_flags": flags, "tcp_window": win}


def run(conv, packets):
    flows, expired, flushed = {}, {}, []
    for p in packets:
        conv._ingest(p, flows, expired, lambda k, r, why: flushed.append((why, r)))
    return flows, flushed


# 完整連線：握手 → 資料 → A 先 FIN → B 回 FIN → A 最後 ACK
TEARDOWN = [
    pkt(0.000, "A", "S"), pkt(0.001, "B", "SA"), pkt(0.002, "A", "A"),
    pkt(0.010, "A", "PA", payload=16), pkt(0.020, "B", "PA", payload=17),
    pkt(0.030, "A", "FA"),   # 第一個 FIN
    pkt(0.031, "B", "FA"),   # 對方 FIN
    pkt(0.032, "A", "A"),    # 最後 ACK
]


class TestOriginalTermination(unittest.TestCase):
    def setUp(self):
        self.conv = pfc.PcapFlowConverter()  # 預設應為原版語意

    def test_default_mode_is_original(self):
        self.assertEqual(self.conv.tcp_termination, pfc.TCP_TERMINATION_ORIGINAL)
        self.assertEqual(self.conv.bulk_mode, pfc.BULK_MODE_CICIDS2017_ZERO)

    def test_first_fin_terminates_and_is_counted(self):
        flows, flushed = run(self.conv, TEARDOWN)
        self.assertEqual([w for w, _ in flushed], ["fin"])
        rec = flushed[0][1]
        self.assertEqual(rec.packet_count, 6)          # 含第一個 FIN
        self.assertEqual(rec.flag_counts["F"], 1)

    def test_tcp_appendix_flow_has_flipped_direction(self):
        flows, flushed = run(self.conv, TEARDOWN)
        # FIN 之後的 FIN-ACK / ACK 在表中另起新 Flow，首包（B 的 FIN）決定方向。
        self.assertEqual(len(flows), 1)
        appendix = next(iter(flows.values()))
        self.assertEqual((appendix.fwd_ip, appendix.fwd_port), (B, BPORT))
        self.assertEqual(appendix.packet_count, 2)
        self.assertEqual(appendix.first_packet_direction, "fwd")

    def test_rst_terminates_existing_flow(self):
        packets = [pkt(0.0, "A", "S"), pkt(0.1, "B", "RA"), pkt(0.2, "A", "A"),
                   pkt(0.3, "A", "PA", payload=5)]
        flows, flushed = run(self.conv, packets)
        self.assertEqual([w for w, _ in flushed], ["rst"])
        self.assertEqual(flushed[0][1].packet_count, 2)
        # 後續同五元組封包另起新的 Flow。
        self.assertEqual(len(flows), 1)
        self.assertEqual(next(iter(flows.values())).packet_count, 2)

    def test_fin_as_first_packet_does_not_terminate(self):
        flows, flushed = run(self.conv, [pkt(0.0, "A", "FA")])
        self.assertEqual(flushed, [])
        self.assertEqual(len(flows), 1)

    def test_fin_first_packet_after_timeout_does_not_terminate(self):
        packets = [pkt(0.0, "A", "S"), pkt(0.1, "B", "SA"),
                   pkt(200.0, "A", "FA")]   # 超過 120 秒：舊 Flow 結束，FIN 成為新段首包
        flows, flushed = run(self.conv, packets)
        self.assertEqual([w for w, _ in flushed], ["timeout"])
        self.assertEqual(len(flows), 1)
        self.assertEqual(next(iter(flows.values())).packet_count, 1)

    def test_no_half_close_dropping(self):
        # 原版沒有「某方向送 FIN 後丟棄同方向封包」；FIN 已結束 Flow，之後的封包都進新 Flow。
        packets = [pkt(0.0, "A", "S"), pkt(0.1, "B", "SA"), pkt(0.2, "A", "FA"),
                   pkt(0.3, "A", "PA", payload=7), pkt(0.4, "A", "PA", payload=9)]
        flows, flushed = run(self.conv, packets)
        self.assertEqual(len(flushed), 1)
        rec = next(iter(flows.values()))
        self.assertEqual(rec.fwd_lens, [7, 9])

    def test_reused_tuple_after_fin_takes_new_direction(self):
        packets = [pkt(0.0, "A", "S"), pkt(0.1, "B", "SA"), pkt(0.2, "A", "FA"),
                   pkt(500.0, "B", "S")]  # 同五元組、由 B 發起的新連線
        flows, _ = run(self.conv, packets)
        rec = next(iter(flows.values()))
        self.assertEqual((rec.fwd_ip, rec.fwd_port), (B, BPORT))


class TestMutualTermination(unittest.TestCase):
    """舊 v4（Engelen 修正版語意）仍可透過 tcp_termination 選用。"""

    def setUp(self):
        self.conv = pfc.PcapFlowConverter(tcp_termination=pfc.TCP_TERMINATION_MUTUAL)

    def test_terminates_after_both_fins(self):
        flows, flushed = run(self.conv, TEARDOWN)
        self.assertEqual([w for w, _ in flushed], ["fin"])
        self.assertEqual(flushed[0][1].packet_count, 7)
        # 最後 ACK 另起單封包 Flow（輸出時會被排除）
        self.assertEqual(next(iter(flows.values())).packet_count, 1)

    def test_rst_terminates(self):
        packets = [pkt(0.0, "A", "S"), pkt(0.1, "B", "RA"), pkt(0.2, "A", "A")]
        flows, flushed = run(self.conv, packets)
        self.assertEqual([w for w, _ in flushed], ["rst"])
        self.assertEqual(flushed[0][1].packet_count, 2)

    def test_half_close_drops_same_direction(self):
        packets = [pkt(0.0, "A", "S"), pkt(0.1, "B", "SA"), pkt(0.2, "A", "FA"),
                   pkt(0.3, "A", "PA", payload=7), pkt(0.4, "B", "PA", payload=9)]
        flows, flushed = run(self.conv, packets)
        rec = next(iter(flows.values()))
        self.assertEqual(rec.fwd_lens, [0, 0])          # A 的後續資料封包被丟棄
        self.assertEqual(rec.bwd_lens, [0, 9])


class TestCicidsLegacyFeatureSemantics(unittest.TestCase):
    def _make_record(self, packet_specs):
        rec = pfc._FlowRecord(A, APORT, B, BPORT, 6, packet_specs[0][0])
        for i, (ts, direction, flags) in enumerate(packet_specs):
            rec.add_packet(direction, ts, 10 if i else 0, 20, flags, 1000)
        return rec

    def test_flag_features_use_first_packet_only_in_legacy_mode(self):
        rec = self._make_record([
            (0.0, "fwd", "S"),
            (0.1, "bwd", "FA"),
            (0.2, "fwd", "PA"),
        ])
        row = rec.finalize(5.0, 1.0, 1.0,
                           flag_mode=pfc.FLAG_MODE_CICIDS2017_LEGACY)
        self.assertEqual(row["SYN Flag Count"], 1)
        self.assertEqual(row["FIN Flag Count"], 0)
        self.assertEqual(row["ACK Flag Count"], 0)
        self.assertEqual(row["Fwd PSH Flags"], 0)
        self.assertEqual(row["Bwd PSH Flags"], 0)

    def test_directional_psh_urg_legacy_values_come_from_first_packet(self):
        rec = self._make_record([
            (0.0, "fwd", "PAU"),
            (0.1, "bwd", "PAU"),
        ])
        row = rec.finalize(5.0, 1.0, 1.0,
                           flag_mode=pfc.FLAG_MODE_CICIDS2017_LEGACY)
        self.assertEqual(row["Fwd PSH Flags"], 1)
        self.assertEqual(row["Bwd PSH Flags"], 0)
        self.assertEqual(row["Fwd URG Flags"], 1)
        self.assertEqual(row["Bwd URG Flags"], 0)

    def test_missing_forward_window_uses_minus_one(self):
        rec = pfc._FlowRecord(A, APORT, B, BPORT, 17, 0.0)
        rec.add_packet("fwd", 0.0, 0, 8, None, None)
        rec.add_packet("bwd", 0.1, 0, 8, None, None)
        row = rec.finalize(5.0, 1.0, 1.0, missing_window_value=-1)
        self.assertEqual(row["Init_Win_bytes_forward"], -1)
        self.assertEqual(row["Init_Win_bytes_backward"], -1)


class TestModeValidation(unittest.TestCase):
    def test_bad_termination(self):
        with self.assertRaises(ValueError):
            pfc.PcapFlowConverter(tcp_termination="nope")

    def test_bad_bulk_mode(self):
        with self.assertRaises(ValueError):
            pfc.PcapFlowConverter(bulk_mode="nope")

    def test_bad_flag_mode(self):
        with self.assertRaises(ValueError):
            pfc.PcapFlowConverter(flag_mode="nope")


class TestFormulas(unittest.TestCase):
    def _record(self, specs):
        rec = pfc._FlowRecord(A, APORT, B, BPORT, 6, specs[0][1])
        for d, ts, payload in specs:
            rec.add_packet(d, ts, payload, 20, "A", 1000)
        return rec

    def test_down_up_ratio_is_integer_division(self):
        rec = self._record([("fwd", 0.0, 0)] + [("bwd", 0.1 * i, 0) for i in range(1, 6)])
        row = rec.finalize(5.0, 1.0, 1.0)
        self.assertEqual(row["Down/Up Ratio"], 5.0)
        rec = self._record([("fwd", 0.0, 0), ("fwd", 0.1, 0), ("fwd", 0.2, 0), ("bwd", 0.3, 0), ("bwd", 0.4, 0)])
        self.assertEqual(rec.finalize(5.0, 1.0, 1.0)["Down/Up Ratio"], 0.0)  # 2 // 3

    def test_packet_length_stats_duplicate_first_payload(self):
        rec = self._record([("fwd", 0.0, 10), ("bwd", 0.1, 20)])
        row = rec.finalize(5.0, 1.0, 1.0)
        # 樣本 = [10(首包重複), 10, 20]
        self.assertAlmostEqual(row["Packet Length Mean"], 40 / 3)
        # Average Packet Size：分子含重複首包 (40)，分母為實際封包數 2
        self.assertAlmostEqual(row["Average Packet Size"], 20.0)

    def test_act_data_pkt_fwd_excludes_first_packet(self):
        rec = self._record([("fwd", 0.0, 10), ("fwd", 0.1, 10), ("bwd", 0.2, 10)])
        self.assertEqual(rec.finalize(5.0, 1.0, 1.0)["act_data_pkt_fwd"], 1)

    def test_bulk_modes(self):
        specs = [("fwd", 0.01 * i, 100) for i in range(6)]
        zero = self._record(specs).finalize(5.0, 1.0, 1.0)  # 預設
        sm = self._record(specs).finalize(5.0, 1.0, 1.0, pfc.BULK_MODE_STATE_MACHINE)
        for col in ("Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
                    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"):
            self.assertEqual(zero[col], 0.0)
        self.assertGreater(sm["Fwd Avg Bytes/Bulk"], 0.0)
        self.assertEqual(sm["Bwd Avg Bytes/Bulk"], 0.0)


class TestConvertEndToEnd(unittest.TestCase):
    """以假的 PcapReader 驗證 convert()：單封包 Flow 排除、結束原因統計、CSV 內容。"""

    class _FakeReader:
        packets = []

        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            return iter(self.packets)

    def _convert(self, packets, **kw):
        orig_reader, orig_extract = pfc.PcapReader, pfc._extract_packet_info
        self._FakeReader.packets = packets
        pfc.PcapReader, pfc._extract_packet_info = self._FakeReader, (lambda p: p)
        try:
            with tempfile.TemporaryDirectory() as d:
                src, out = os.path.join(d, "x.pcap"), os.path.join(d, "x.csv")
                open(src, "wb").close()
                pfc.PcapFlowConverter(**kw).convert(src, out, default_label="BENIGN")
                with open(out, newline="", encoding="utf-8") as f:
                    return list(csv.DictReader(f))
        finally:
            pfc.PcapReader, pfc._extract_packet_info = orig_reader, orig_extract

    def test_original_mode_rows(self):
        rows = self._convert(TEARDOWN)
        # Flow1（6 封包，FIN 結束）+ appendix（2 封包，檔尾輸出）；皆 >1 封包
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["Total Fwd Packets"], "4")
        self.assertEqual(rows[0]["FIN Flag Count"], "1")
        self.assertEqual(rows[0]["Label"], "BENIGN")
        self.assertEqual(rows[1]["Src IP"], B)           # appendix 方向翻轉
        self.assertEqual(rows[1]["Destination Port"], str(APORT))

    def test_mutual_mode_rows_exclude_single_packet(self):
        rows = self._convert(TEARDOWN, tcp_termination=pfc.TCP_TERMINATION_MUTUAL)
        self.assertEqual(len(rows), 1)                   # 最後 ACK 的單封包 Flow 被排除
        self.assertEqual(rows[0]["FIN Flag Count"], "2")

    def test_single_packet_flows_never_written(self):
        rows = self._convert([pkt(0.0, "A", "S")])
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
