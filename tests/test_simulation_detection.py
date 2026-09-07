"""規則式模擬的時間與封包語意測試。"""
import sys
from pathlib import Path
from unittest import TestCase


CORE_DIR = Path(__file__).resolve().parents[1] / 'core'
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from anomaly_detector import AnomalyDetector
from scapy.layers.inet import IP, TCP


class SimulationTimestampDetectionTest(TestCase):
    def _record(self, src, port, flags='SYN'):
        return {
            'protocol': 'TCP', 'src_ip': src, 'dst_ip': '10.0.0.2',
            'dst_port': port, 'flags': flags,
        }

    def test_syn_packets_outside_window_do_not_be_combined(self):
        detector = AnomalyDetector(
            on_alert=lambda _alert: None,
            threshold_syn=2,
            threshold_ports=999,
            window_seconds=10,
            cooldown_seconds=0,
        )
        packet = IP(src='192.0.2.10', dst='10.0.0.2') / TCP(dport=80, flags='S')

        alerts = []
        for timestamp in (100.0, 111.0, 122.0):
            alerts.extend(detector.inspect(packet, self._record('192.0.2.10', 80), timestamp=timestamp))
        self.assertEqual(alerts, [])

        detector.reset()
        alerts = []
        for timestamp in (200.0, 201.0, 202.0):
            alerts.extend(detector.inspect(packet, self._record('192.0.2.10', 80), timestamp=timestamp))
        self.assertTrue(any(alert['attack_type'] == 'SYN Flood' for alert in alerts))

    def test_rst_responses_are_not_counted_as_reverse_port_scan(self):
        detector = AnomalyDetector(
            on_alert=lambda _alert: None,
            threshold_syn=999,
            threshold_ports=1,
            window_seconds=10,
            cooldown_seconds=0,
        )
        rst = IP(src='10.0.0.2', dst='192.0.2.10') / TCP(dport=80, flags='R')

        alerts = []
        for index, port in enumerate((80, 443, 8080)):
            alerts.extend(detector.inspect(
                rst, self._record('10.0.0.2', port, flags='RST'), timestamp=100.0 + index
            ))
        self.assertFalse(any('Port Scan' in alert['attack_type'] for alert in alerts))

        syn = IP(src='192.0.2.10', dst='10.0.0.2') / TCP(dport=80, flags='S')
        alerts = []
        for index, port in enumerate((80, 443)):
            alerts.extend(detector.inspect(
                syn, self._record('192.0.2.10', port), timestamp=200.0 + index
            ))
        self.assertTrue(any('Port Scan' in alert['attack_type'] for alert in alerts))
