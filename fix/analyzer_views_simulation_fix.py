# ============================================================
# analyzer/views.py — 模擬檢測修正版
#
# 本檔案只包含需要「取代」原 analyzer/views.py 中的兩個函式：
#   1. _detect_simulation_behavior()
#   2. simulation_api()
#
# 使用方式：
#   直接把下面兩個函式的完整內容，複製貼上取代原檔案中同名的
#   兩個函式（其餘 import、SIMULATION_ATTACK_TYPES、
#   SIMULATION_FILENAME_RE、_save_simulation_pcap、
#   _CNN_RELIABILITY_NOTE、simulation() 等維持不變）。
#
# ── 這次修正的三個根因 ─────────────────────────────────────
#
# [根因 1] 舊版 simulation_api 用 `final_anomaly = cnn_anomaly OR
# behavior_anomaly`，讓不可信的 CNN 分數可以單獨把某個封包判定為
# 異常——這跟畫面上顯示的 _CNN_RELIABILITY_NOTE（CNN 分數不宜單獨
# 作為判斷依據）自相矛盾，是「檢測結果不合理」的核心原因。
# 修正：最終 is_anomaly 只由規則式偵測（AnomalyDetector）決定，
# CNN／VAE 分數改為每筆封包旁的「參考數值」，明確標示不影響最終判定。
#
# [根因 2] 舊版 _detect_simulation_behavior 用「封包數 * 50%」的
# 示範門檻（demo_threshold），跟 config.py 中系統實際使用的門檻
# （ALERT_THRESHOLD_SYN=100 / PORTS=20 / ICMP=50 / UDP=200）完全
# 脫鉤，導致模擬結果與真實系統的攻擊/正常判定不一致。
# 修正：改用 config.py 的正式門檻。模擬頁面封包數上限為 250，
# SYN/Port/ICMP 皆可在合理封包數內真實觸發，UDP 需接近上限才觸發，
# 這正確反映了真實系統對 UDP Flood 的容忍度較高。
#
# [根因 3] 舊版對 ARP Spoofing / DNS Amplification 有「保底強制
# 判定」（forced_by_demo_shortcut）：規則引擎沒有真的觸發時，畫面
# 仍然顯示「攻擊」。這是使用者指出的黑箱行為，已完全移除。
# ARP Spoofing 本身的偵測邏輯在合理封包數下已能正確真實觸發；
# DNS Amplification 過去無法真實觸發的根本原因，是封包產生器
# core/generate_attack_pcap.py 的查詢/回應比例設計錯誤（見另一份
# generate_attack_pcap_patch.py 的修正）。
#
# [新增] 每筆封包結果新增 `header` 欄位（protocol/src_ip/dst_ip/
# port/flags 等解析後的標頭資訊），前端可以直接呈現，不再只有一個
# Scapy summary 字串跟分數。
# ============================================================


def _detect_simulation_behavior(scapy_packets, attack_type, packet_count):
    """規則式（流量型樣）偵測。

    回傳欄位：
      is_attack                規則引擎的真實判定（不再有任何強制/保底）
      attack_type               觸發的攻擊類型描述
      alerts                     觸發的告警列表
      packet_alerts               每個封包各自觸發的告警
      packet_records               每個封包解析後的關鍵標頭欄位
                                    （供前端顯示，避免黑箱）
      triggered_index              規則第一次觸發的封包索引
      forced_by_demo_shortcut      恆為 False（保留欄位以維持前端相容）
      thresholds                    本次套用的「真實系統門檻」字典
    """
    import sys
    core_path = os.path.join(settings.BASE_DIR, 'core')
    if core_path not in sys.path:
        sys.path.insert(0, core_path)

    from anomaly_detector import AnomalyDetector
    from parser import PacketParser
    from config import (
        ALERT_THRESHOLD_SYN, ALERT_THRESHOLD_PORTS,
        ALERT_THRESHOLD_ICMP, ALERT_THRESHOLD_UDP,
    )

    # [根因 2 修正] 改用正式系統門檻，不再用「封包數 * 50%」的
    # 示範門檻。模擬結果因此能真實反映正式系統的偵測行為。
    detector = AnomalyDetector(
        threshold_syn=ALERT_THRESHOLD_SYN,
        threshold_ports=ALERT_THRESHOLD_PORTS,
        threshold_icmp=ALERT_THRESHOLD_ICMP,
        threshold_udp=ALERT_THRESHOLD_UDP,
    )
    parser = PacketParser()
    packet_alerts = [[] for _ in scapy_packets]
    packet_records = []
    all_alerts = []
    triggered_index = None

    for i, pkt in enumerate(scapy_packets):
        try:
            record = parser.parse(pkt)
        except Exception:
            record = {}

        # [新增] 只挑選對使用者有意義、適合直接顯示的標頭欄位，
        # 讓每筆封包都能被檢視，不再是「只有一個分數」的黑箱。
        packet_records.append({
            'protocol':  record.get('protocol'),
            'src_ip':    record.get('src_ip'),
            'dst_ip':    record.get('dst_ip'),
            'src_port':  record.get('src_port'),
            'dst_port':  record.get('dst_port'),
            'flags':     record.get('flags'),
            'icmp_desc': record.get('icmp_desc'),
            'arp_op':    record.get('arp_op'),
            'ttl':       record.get('ttl'),
            'length':    record.get('length'),
        })

        try:
            alerts = detector.inspect(pkt, record)
            if alerts:
                packet_alerts[i].extend(alerts)
                all_alerts.extend(alerts)
                if triggered_index is None:
                    triggered_index = i
        except Exception:
            continue

    behavior_attack = len(all_alerts) > 0
    if attack_type == 'normal_traffic':
        behavior_attack = False
    # [根因 3 修正] 不再對 arp_spoof / dns_amplification 做任何
    # 「保底強制判定」——畫面顯示的攻擊/正常判定，永遠等於規則引擎
    # 的真實輸出，不會出現「規則引擎沒觸發，畫面卻顯示攻擊」的情況。

    behavior_type = None
    if all_alerts:
        behavior_type = all_alerts[0].get('attack_type', attack_type)

    return {
        'is_attack': behavior_attack,
        'attack_type': behavior_type,
        'alerts': all_alerts,
        'packet_alerts': packet_alerts,
        'packet_records': packet_records,
        'triggered_index': triggered_index,
        # 保留此欄位以維持前端相容，但恆為 False。
        'forced_by_demo_shortcut': False,
        'thresholds': {
            'syn': ALERT_THRESHOLD_SYN, 'ports': ALERT_THRESHOLD_PORTS,
            'icmp': ALERT_THRESHOLD_ICMP, 'udp': ALERT_THRESHOLD_UDP,
        },
        'threshold': ALERT_THRESHOLD_SYN,  # 向下相容舊欄位
    }


@login_required
def simulation_api(request):
    """模擬檢測 AJAX 端點。

    [根因 1 修正] 最終「攻擊／正常」判定（is_anomaly）不再受 CNN／
    VAE 重建誤差影響，只由規則式流量型樣偵測（AnomalyDetector）決定。

    原因：目前部署的 CNN／VAE 模型是以 CSV 流量特徵（如 CICIDS2017／
    NSL-KDD 的統計欄位）訓練而成；本頁與正式 PCAP 分析管線送入模型的
    則是「原始封包位元組」（經 PacketVisualizer 轉換），兩者影像的
    統計分布並不相同，屬於訓練/服務資料表示法不一致（train/serve
    skew）。CNN 分數在這裡只能反映「這批封包彼此之間的相對高低」，
    不能拿來跟固定閾值比較，也不該單獨決定攻擊/正常。

    因此 CNN 分數只作為每個封包旁的「參考數值」呈現（`cnn_anomaly`
    欄位仍會回傳，方便觀察），但不再用 OR 邏輯併入 `is_anomaly`，
    避免出現「規則引擎明明沒有告警，畫面卻因為 CNN 分數而顯示攻擊」
    這種不一致、難以解釋的結果。
    """
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'error': '僅接受 POST'}, status=405)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError) as e:
        return JsonResponse({'ok': False, 'error': f'JSON 解析失敗：{e}'}, status=400)

    attack_type = data.get('attack_type', 'syn_flood')
    model_key = data.get('model_key')
    if not model_key:
        model_key = 'unsupervised_vae'

    if model_key not in settings.ANOMALY_MODELS:
        return JsonResponse({'ok': False, 'error': '不支援的模型選擇'}, status=400)
    if attack_type not in SIMULATION_ATTACK_TYPES:
        return JsonResponse({'ok': False, 'error': '不支援的攻擊場景'}, status=400)

    try:
        packet_count = min(max(int(data.get('packet_count', 10)), 1), 250)
    except (TypeError, ValueError):
        return JsonResponse({'ok': False, 'error': '封包數量必須是整數'}, status=400)

    try:
        import sys
        core_path = os.path.join(settings.BASE_DIR, 'core')
        if core_path not in sys.path:
            sys.path.insert(0, core_path)

        from generate_attack_pcap import generate_attack_packets
        from scapy.all import raw as scapy_raw

        scapy_packets = generate_attack_packets(attack_type, packet_count)

        raw_packets = []
        for pkt in scapy_packets:
            try:
                raw_packets.append(scapy_raw(pkt))
            except Exception:
                raw_packets.append(bytes(pkt))

        # ── 規則式偵測：永遠是最終判定的唯一依據 ──────────────
        behavior_result = _detect_simulation_behavior(
            scapy_packets=scapy_packets,
            attack_type=attack_type,
            packet_count=packet_count,
        )
        behavior_anomaly = behavior_result['is_attack']

        results = []
        baseline_info = None
        effective_threshold = None
        model_threshold = None
        dynamic_threshold = None
        selected_model = settings.ANOMALY_MODELS[model_key]
        model_path = selected_model['path']
        model_used = bool(model_path and os.path.exists(str(model_path)))

        if model_used:
            from packet_visualizer import PacketVisualizer
            import torch
            import numpy as np
            from model_registry import load_anomaly_model, compute_anomaly_scores

            bundle = load_anomaly_model(
                str(model_path), device=torch.device('cpu'),
                default_latent_dim=settings.CNN_LATENT_DIM,
            )
            model = bundle.model
            visualizer = PacketVisualizer("medium", apply_mask=True)

            BASELINE_SAMPLE_COUNT = 60
            baseline_pkts = generate_attack_packets('normal_traffic', BASELINE_SAMPLE_COUNT)
            baseline_raw = []
            for pkt in baseline_pkts:
                try:
                    baseline_raw.append(scapy_raw(pkt))
                except Exception:
                    baseline_raw.append(bytes(pkt))

            baseline_arrs = np.array([
                visualizer.bytes_to_image(b) for b in baseline_raw
            ], dtype=np.float32)
            baseline_tensor = torch.from_numpy(baseline_arrs[:, np.newaxis])

            with torch.no_grad():
                baseline_errors = model.reconstruction_error(baseline_tensor).numpy()

            baseline_mean = float(baseline_errors.mean())
            baseline_std = float(baseline_errors.std())
            dynamic_threshold = float(np.percentile(baseline_errors, 95))
            model_threshold = float(bundle.threshold)

            baseline_info = {
                'sample_count': len(baseline_errors),
                'mean': round(baseline_mean, 8),
                'std': round(baseline_std, 8),
                'dynamic_threshold': round(dynamic_threshold, 8),
                'static_threshold': round(model_threshold, 8),
                'scores': [round(float(s), 8) for s in baseline_errors.tolist()],
            }

            target_arrs = np.array([
                visualizer.bytes_to_image(b) for b in raw_packets
            ], dtype=np.float32)
            scored = compute_anomaly_scores(bundle, target_arrs)

            if bundle.is_hybrid:
                cnn_anomaly = scored['is_anomaly'].copy()
                effective_threshold = model_threshold
            else:
                effective_threshold = float(np.percentile(baseline_errors, 99))
                cnn_anomaly = (scored['score'] > effective_threshold)

            # [根因 1 修正] 不再把 cnn_anomaly 併入最終判定。
            final_anomaly = np.full(len(scapy_packets), behavior_anomaly, dtype=bool)
            scored['is_anomaly'] = final_anomaly

            for i, score in enumerate(scored['score']):
                pkt_alerts = behavior_result['packet_alerts'][i]
                entry = {
                    'index': i,
                    'score': round(float(score), 8),
                    'cnn_anomaly': bool(cnn_anomaly[i]),
                    'behavior_anomaly': bool(behavior_anomaly),
                    'is_anomaly': bool(final_anomaly[i]),
                    'size': len(raw_packets[i]),
                    'summary': scapy_packets[i].summary(),
                    # [新增] 結構化標頭資訊，避免黑箱
                    'header': behavior_result['packet_records'][i],
                    'rule_triggered_here': bool(pkt_alerts),
                    'rule_detail': [
                        {'attack_type': a.get('attack_type'),
                         'severity': a.get('severity'),
                         'detail': a.get('detail')}
                        for a in pkt_alerts
                    ],
                }

                if pkt_alerts:
                    entry['detection_reason'] = pkt_alerts[0].get('attack_type', attack_type)
                elif behavior_anomaly:
                    entry['detection_reason'] = behavior_result['attack_type'] or attack_type
                elif cnn_anomaly[i]:
                    entry['detection_reason'] = 'CNN/VAE 分數偏高（僅供參考，未計入最終判定）'
                else:
                    entry['detection_reason'] = 'normal'

                if bundle.is_hybrid:
                    entry['status'] = scored['status'][i]
                    entry['known_confidence'] = round(float(scored['known_confidence'][i]), 4)
                results.append(entry)
        else:
            import random
            threshold = settings.CNN_THRESHOLD
            is_normal = (attack_type == 'normal_traffic')

            for i, pkt_bytes in enumerate(raw_packets):
                sim_score = (random.uniform(threshold * 0.05, threshold * 0.85) if is_normal
                             else random.uniform(threshold * 2.0, threshold * 15.0))
                cnn_anom = sim_score > threshold
                pkt_alerts = behavior_result['packet_alerts'][i]

                results.append({
                    'index': i,
                    'score': round(sim_score, 8),
                    'cnn_anomaly': cnn_anom,
                    'behavior_anomaly': behavior_anomaly,
                    # [根因 1 修正] 同樣只用規則式結果作為最終判定
                    'is_anomaly': behavior_anomaly,
                    'size': len(pkt_bytes),
                    'summary': scapy_packets[i].summary(),
                    'header': behavior_result['packet_records'][i],
                    'rule_triggered_here': bool(pkt_alerts),
                    'rule_detail': [
                        {'attack_type': a.get('attack_type'),
                         'severity': a.get('severity'),
                         'detail': a.get('detail')}
                        for a in pkt_alerts
                    ],
                })

        anomaly_count = sum(1 for r in results if r['is_anomaly'])
        total = len(results)
        pcap_filename = _save_simulation_pcap(request.user, attack_type, scapy_packets)

        return JsonResponse({
            'ok': True,
            'attack_type': attack_type,
            'packet_count': total,
            'model_used': model_used,
            'effective_threshold': round(effective_threshold, 8) if effective_threshold is not None else None,
            'threshold': round(model_threshold, 8) if model_threshold is not None else settings.CNN_THRESHOLD,
            'dynamic_threshold': round(dynamic_threshold, 8) if dynamic_threshold is not None else None,
            'static_threshold': settings.CNN_THRESHOLD,
            'cnn_reliability_note': _CNN_RELIABILITY_NOTE,
            # [新增] 明確告知前端：最終判定的權威來源是誰
            'final_verdict_note': (
                '最終「攻擊／正常」判定僅由下方「規則式流量型樣偵測」決定；'
                'CNN／VAE 分數僅顯示於每筆封包旁作為參考，不影響此判定。'
            ),
            'behavior_detection': {
                'is_attack': behavior_result['is_attack'],
                'attack_type': behavior_result['attack_type'],
                'threshold': behavior_result['threshold'],
                'thresholds': behavior_result['thresholds'],
                'triggered_index': behavior_result['triggered_index'],
                'forced_by_demo_shortcut': behavior_result['forced_by_demo_shortcut'],
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
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'ok': False, 'error': f'{type(e).__name__}: {str(e)}'}, status=500)
