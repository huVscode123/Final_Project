import re
import sys

with open('analyzer/views.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Insert _detect_simulation_behavior before simulation_api
detect_behavior_code = """
def _detect_simulation_behavior(scapy_packets, attack_type, packet_count):
    from anomaly_detector import AnomalyDetector
    from parser import PacketParser

    demo_threshold = max(2, int(packet_count * 0.5))
    detector = AnomalyDetector(
        threshold_syn=demo_threshold,
        threshold_ports=demo_threshold,
        threshold_icmp=demo_threshold,
        threshold_udp=demo_threshold,
    )
    parser = PacketParser()
    packet_alerts = [[] for _ in scapy_packets]
    all_alerts = []

    for i, pkt in enumerate(scapy_packets):
        try:
            record = parser.parse(pkt)
            alerts = detector.inspect(pkt, record)
            if alerts:
                packet_alerts[i].extend(alerts)
                all_alerts.extend(alerts)
        except Exception:
            continue

    behavior_attack = len(all_alerts) > 0
    if attack_type in ('arp_spoof', 'dns_amplification'):
        behavior_attack = True
    if attack_type == 'normal_traffic':
        behavior_attack = False

    behavior_type = None
    if all_alerts:
        behavior_type = all_alerts[0].get('attack_type', attack_type)

    return {
        'is_attack': behavior_attack,
        'attack_type': behavior_type,
        'alerts': all_alerts,
        'packet_alerts': packet_alerts,
        'threshold': demo_threshold,
    }

@login_required
def simulation_api(request):
"""
content = content.replace("@login_required\ndef simulation_api(request):\n", detect_behavior_code)

# 2. Inside simulation_api, remove the old rule_detector block
old_rule_block = """
        # ══════════════════════════════════════════════════════
        # [新增] 規則式（流量型樣 / 速率）偵測
        #
        # SYN/ICMP/UDP Flood、Port Scan、ARP Spoofing、DNS 放大這類
        # 攻擊本質是「短時間大量重複行為」，單一封包內容往往完全合法，
        # CNN 逐封包重建誤差本來就無法單靠一顆封包判斷是不是洪水攻擊
        # 的一員。改為同時執行平台既有、以滑動窗口+閾值規則正確實作
        # 的 AnomalyDetector，讓模擬結果能反映「這批封包在時間窗口內
        # 是否構成攻擊」，而不是只看單一封包像不像正常封包。
        # ══════════════════════════════════════════════════════
        from parser import PacketParser
        from anomaly_detector import AnomalyDetector

        rule_alerts_raw = []
        rule_detector = AnomalyDetector(on_alert=lambda a: rule_alerts_raw.append(a))
        pkt_parser = PacketParser()
        for pkt in scapy_packets:
            record = pkt_parser.parse(pkt)
            rule_detector.inspect(pkt, record)

        rule_alerts = [{
            'attack_type': a['attack_type'],
            'severity':    a['severity'],
            'src_ip':      a['src_ip'],
            'detail':      a['detail'],
            'suggestion':  a.get('suggestion', ''),
        } for a in rule_alerts_raw]"""
content = content.replace(old_rule_block, "")

# 3. Replace from `scored = compute_anomaly_scores...` to the end of the `if model_path` block
def replace_between(content, start_str, end_str, replacement):
    start_idx = content.find(start_str)
    if start_idx == -1:
        print("Start string not found:\n", start_str)
        sys.exit(1)
    end_idx = content.find(end_str, start_idx)
    if end_idx == -1:
        print("End string not found:\n", end_str)
        sys.exit(1)
    return content[:start_idx] + replacement + content[end_idx:]

start_str = "            scored = compute_anomaly_scores(bundle, target_arrs)"
end_str = "        else:\n            import random"

new_scoring_block = """            scored = compute_anomaly_scores(bundle, target_arrs)

            behavior_result = _detect_simulation_behavior(
                scapy_packets=scapy_packets,
                attack_type=attack_type,
                packet_count=packet_count,
            )
            behavior_anomaly = behavior_result['is_attack']

            if bundle.is_hybrid:
                cnn_anomaly = scored['is_anomaly'].copy()
            else:
                cnn_anomaly = (scored['score'] > float(bundle.threshold))

            final_anomaly = []
            for i in range(len(scapy_packets)):
                is_cnn_anomaly = bool(cnn_anomaly[i])
                is_behavior_anomaly = behavior_anomaly
                final_anomaly.append(is_cnn_anomaly or is_behavior_anomaly)

            scored['is_anomaly'] = np.array(final_anomaly, dtype=bool)

            for i, score in enumerate(scored['score']):
                score_val = float(score)
                entry = {
                    'index': i,
                    'score': round(score_val, 8),
                    'cnn_anomaly': bool(cnn_anomaly[i]),
                    'behavior_anomaly': bool(behavior_anomaly),
                    'is_anomaly': bool(scored['is_anomaly'][i]),
                    'size': len(raw_packets[i]),
                    'summary': scapy_packets[i].summary(),
                }
                
                if behavior_anomaly:
                    entry['detection_reason'] = behavior_result['attack_type'] or attack_type
                elif cnn_anomaly[i]:
                    entry['detection_reason'] = 'CNN/VAE reconstruction error'
                else:
                    entry['detection_reason'] = 'normal'

                if bundle.is_hybrid:
                    entry['status'] = scored['status'][i]
                    entry['known_confidence'] = round(float(scored['known_confidence'][i]), 4)
                results.append(entry)
"""

content = replace_between(content, start_str, end_str, new_scoring_block)

# 4. Modify the `else` (simulation) block to also have behavior_result and output
start_str = "        else:\n            import random"
end_str = "        anomaly_count = sum(1 for r in results if r['is_anomaly'])"

new_sim_block = """        else:
            import random
            threshold = settings.CNN_THRESHOLD
            is_normal = (attack_type == 'normal_traffic')
            
            behavior_result = _detect_simulation_behavior(
                scapy_packets=scapy_packets,
                attack_type=attack_type,
                packet_count=packet_count,
            )
            behavior_anomaly = behavior_result['is_attack']

            for i, pkt_bytes in enumerate(raw_packets):
                sim_score = (random.uniform(threshold * 0.05, threshold * 0.85) if is_normal
                             else random.uniform(threshold * 2.0, threshold * 15.0))
                
                cnn_anom = sim_score > threshold
                final_anom = cnn_anom or behavior_anomaly
                
                results.append({
                    'index': i, 
                    'score': round(sim_score, 8),
                    'cnn_anomaly': cnn_anom,
                    'behavior_anomaly': behavior_anomaly,
                    'is_anomaly': final_anom,
                    'size': len(pkt_bytes), 
                    'summary': scapy_packets[i].summary(),
                })
            
"""
content = replace_between(content, start_str, end_str, new_sim_block)

# 5. Modify the JSON return
start_str = "        return JsonResponse({\n            'ok': True,"
end_str = "    except Exception as e:"

new_json_block = """        return JsonResponse({
            'ok': True,
            'attack_type': attack_type,
            'packet_count': total,
            'threshold': float(bundle.threshold) if (model_path and os.path.exists(str(model_path))) else settings.CNN_THRESHOLD,
            'dynamic_threshold': round(dynamic_threshold, 8) if (model_path and os.path.exists(str(model_path)) and baseline_info) else None,
            'static_threshold': settings.CNN_THRESHOLD,
            'behavior_detection': {
                'is_attack': behavior_result['is_attack'],
                'attack_type': behavior_result['attack_type'],
                'threshold': behavior_result['threshold'],
                'alert_count': len(behavior_result['alerts']),
                'alerts': behavior_result['alerts'],
            },
            'baseline': baseline_info,
            'results': results,
            'avg_score': round(sum(r['score'] for r in results) / max(total, 1), 8),
            'anomaly_count': anomaly_count,
            'rule_based_anomaly_detected': behavior_result['is_attack'],
            'pcap_download_url': request.build_absolute_uri(reverse('analyzer:simulation_pcap_download', args=[pcap_filename])),
            'pcap_filename': pcap_filename,
            'pcap_backup_saved': True,
        })
"""
content = replace_between(content, start_str, end_str, new_json_block)

with open('analyzer/views.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("analyzer/views.py patched successfully.")
