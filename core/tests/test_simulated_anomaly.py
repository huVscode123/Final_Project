# core/tests/test_simulated_anomaly.py
# 驗證 simulate_anomaly_traffic.py 產生的封包能被 AnomalyDetector 正確偵測
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from simulate_anomaly_traffic import GENERATORS, verify_packets

@pytest.mark.parametrize("key", list(GENERATORS.keys()))
def test_generated_traffic_triggers_alert(key):
    """每種模擬出的異常流量都必須觸發對應的 AnomalyDetector 告警"""
    _, fn = GENERATORS[key]
    pkts, meta = fn(scale=1.0)
    ok, triggered = verify_packets(key, pkts)
    assert ok, f"{key} 未觸發預期告警，實際觸發: {triggered}"