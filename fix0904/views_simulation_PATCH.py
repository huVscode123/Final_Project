# ============================================================
# analyzer/views.py — 修補指示（非整檔取代）
#
# 請在 analyzer/views.py 中，用下面標示的區塊「整段取代」對應的
# 舊函式／新增對應的區塊。每一段開頭都標明「取代」或「新增」及其
# 在檔案中的相對位置，方便對照。
#
# ============================================================
# 【新增】檔案最上方 import 區塊
# 原本：
#     import os, json
#     from django.shortcuts import render, get_object_or_404, redirect
#     ...
# 請在 `import os, json` 後面，額外加入：
#     import time
# 並在檔案中任何一個 `from .models import ...` 之後（例如緊接在
# `from .models import AnalysisSession, Alert, CNNResult, GradCAMImage, AnalysisReport`
# 這行下面），加入：
#     from .apps import warmup_event
#
# ── 為什麼要加 time 與 warmup_event ──
#   time：simulation_api() 新增了逐階段計時（封包產生 / 規則引擎 /
#         模型載入 / CNN 推論），把「為什麼這次比較慢」的原因量化
#         回傳給前端，取代原本使用者只能憑感覺猜的黑箱狀態。
#   warmup_event：來自 analyzer/apps.py 修正版新增的
#         threading.Event()，simulation_api() 用它回報「伺服器背景
#         預熱是否已完成」，讓「第一次模擬比較慢」有明確、可驗證的
#         解釋，而不是不可預期的隨機現象。
# ============================================================


# ============================================================
# 【新增】在 SIMULATION_FILENAME_RE 定義之後、
#         def _save_simulation_pcap(...) 之前，新增以下常數與輔助函式
# ============================================================

# [根因 2 修正] 各攻擊類型對應「規則引擎實際會用來判定的門檻」。
# 舊版 _detect_simulation_behavior() 不論選擇哪種攻擊類型，
# behavior_result['threshold'] 永遠回傳 ALERT_THRESHOLD_SYN（100），
# 例如模擬 UDP Flood（正式門檻其實是 200）時，若前端或除錯過程中
# 讀取這個欄位，看到的會是不相關的 100，屬於誤導性的殘留欄位。
# 這裡改為依攻擊類型回傳真正相關的門檻。
def _relevant_threshold_for(attack_type: str) -> int:
    from config import (
        ALERT_THRESHOLD_SYN, ALERT_THRESHOLD_PORTS,
        ALERT_THRESHOLD_ICMP, ALERT_THRESHOLD_UDP,
    )
    return {
        'syn_flood':  ALERT_THRESHOLD_SYN,
        'port_scan':  ALERT_THRESHOLD_PORTS,
        'icmp_flood': ALERT_THRESHOLD_ICMP,
        'udp_flood':  ALERT_THRESHOLD_UDP,
    }.get(attack_type, ALERT_THRESHOLD_SYN)


def _array_to_data_url(arr_2d, upscale=4):
    """
    [黑箱修正] 把模型實際看到的 32x32 灰階封包影像，轉成可以直接
    塞進前端 <img src="..."> 的 base64 PNG。

    這是「需要顯示展示模擬的資料」這項需求的具體實作：過去使用者
    只能看到一串分數（例如 0.00081234），完全無法判斷這個數字合不
    合理；現在可以親眼看到「模型到底在看什麼」，並與正常流量的
    基準線影像並排比較。32x32 的影像很小，即使放大 4 倍編碼成
    base64 也只有幾 KB，不會對回應大小造成明顯負擔。
    """
    import io
    import base64
    import numpy as np
    from PIL import Image

    clipped = np.clip(arr_2d, 0.0, 1.0)
    img = Image.fromarray((clipped * 255).astype('uint8'), mode='L')
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def _select_display_indices(behavior_result, total, cap=400):
    """
    [效能] 強度滑桿拉高時（例如同時有 3 個來源 IP 各自灌 300+ 個
    SYN 封包），單次模擬可能產生超過千筆封包。逐筆全部塞進前端
    表格既沒必要也會拖慢渲染，這裡改為：
      - 一定保留「有直接觸發規則」的封包（讓使用者看到真正的觸發點）
      - 其餘封包等距抽樣湊到上限 cap
    統計數字（異常封包數、平均分數）仍然是用『全部』封包算出來的，
    只有表格顯示的明細列數被裁切，不影響任何判定或統計的正確性。
    """
    triggered = [i for i, alerts in enumerate(behavior_result['packet_alerts']) if alerts]
    if total <= cap:
        return list(range(total))

    triggered_set = set(triggered)
    remaining_budget = max(0, cap - len(triggered_set))
    others = [i for i in range(total) if i not in triggered_set]
    sampled = []
    if remaining_budget > 0 and others:
        step = max(1, len(others) // remaining_budget)
        sampled = others[::step][:remaining_budget]

    idx = sorted(triggered_set.union(sampled))
    return idx[:cap]


# [新增] CNN 分數可信度說明 —— 依模型的訓練資料表示法動態選用。
_CNN_RELIABILITY_NOTE_CSV_LEGACY = (
    '目前選用的 CNN／VAE 模型是以 CSV 流量特徵（如 CICIDS2017／NSL-KDD 的'
    '統計欄位）訓練而成；本頁與正式 PCAP 分析管線送入模型的則是「原始封包'
    '位元組」（經 PacketVisualizer 轉換），兩者影像的統計分布並不相同，屬於'
    '訓練/服務資料表示法不一致（train/serve skew）。因此 CNN 分數在這裡'
    '僅供參考，不宜單獨作為攻擊／正常的判斷依據，請優先參考下方「規則式'
    '流量型樣偵測」的結果。若要讓 CNN 分數變得有意義，請參考'
    'core/packet_dataset_builder.py 與 core/train_packet_native_vae.py，'
    '改用與推論一致的封包位元組表示法重新訓練模型。'
)

_CNN_RELIABILITY_NOTE_PACKET_NATIVE = (
    '此模型已使用與本頁、與正式 PCAP 分析管線一致的「封包位元組影像」'
    '表示法訓練（見 core/packet_dataset_builder.py），CNN／VAE 分數'
    '具備參考意義。惟為求判定結果一致、可解釋，系統目前仍以下方'
    '「規則式流量型樣偵測」作為最終判定的唯一依據，CNN 分數僅作輔助'
    '對照，尚未計入最終判定。'
)

# 保留舊名稱以維持向下相容（例如有其他程式碼曾經 import 這個常數）。
_CNN_RELIABILITY_NOTE = _CNN_RELIABILITY_NOTE_CSV_LEGACY


# ============================================================
# 【取代】def simulation(request): ...
# ============================================================
@login_required
def simulation(request):
    """異常封包模擬檢測頁面 — 支援預設攻擊場景和自訂封包。

    [根因修正] 舊版這裡完全沒有把 `models` 放進 context，但樣板
    simulation.html（以及 upload.html）都有：
        {% for model in models %}<option value="{{ model.key }}"...
    Django 樣板在找不到 `models` 這個 context 變數時，{% for %} 迴圈
    會直接視為空清單、不拋錯，於是「異常偵測模型」下拉選單永遠是
    空的 —— 使用者以為可以選擇 4 個模型之一，實際上選單完全沒有
    任何選項可選。這不是「讀註解」能發現的問題（程式碼裡沒有任何
    註解提到這件事），只有實際核對 view 回傳的 context 和樣板用到
    的變數名稱才會發現。

    修正：實際檢查 settings.ANOMALY_MODELS 中每個模型檔案是否存在，
    組成 `models` 清單傳給樣板；同時修正原本用來判斷「CNN 模型狀態」
    KPI 卡片的 model_ready，原本檢查的是 settings.CNN_MODEL_PATH
    （指向 media/model/best_model.pt），這是舊版單模型架構遺留、
    現在 4 選 1 的 ANOMALY_MODELS 系統裡完全用不到的路徑 —— 也就是說
    即使 unsupervised_vae（best_vae_model.pt）確實存在且能正常運作，
    這張 KPI 卡片先前也一律顯示「✗ 未載入」，是另一個與實際狀態
    脫鉤的黑箱來源。
    """
    models = []
    for key, cfg in settings.ANOMALY_MODELS.items():
        path = str(cfg.get('path', ''))
        models.append({
            'key':   key,
            'label': cfg.get('label', key),
            'ready': bool(path and os.path.exists(path)),
        })

    model_ready = any(m['ready'] for m in models)

    return render(request, 'analyzer/simulation.html', {
        'model_ready':    model_ready,
        'models':         models,
        'cnn_threshold':  settings.CNN_THRESHOLD,
        'cnn_latent_dim': settings.CNN_LATENT_DIM,
    })


# ============================================================
# 【取代】def _detect_simulation_behavior(scapy_packets, attack_type, packet_count): ...
#
# 與原本正式修正版相比，唯一差異是最後回傳值的 'threshold' 欄位，
# 改用 _relevant_threshold_for(attack_type) 取代寫死的 ALERT_THRESHOLD_SYN。
# 其餘邏輯（規則引擎判定、封包標頭記錄、triggered_index 等）完全不變，
# 這部分先前的判定邏輯經過實際追蹤確認是正確的，不是本次問題的成因。
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
      threshold                     [修正] 依 attack_type 回傳對應的
                                     正式門檻，不再永遠回傳 SYN 的門檻
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
        'forced_by_demo_shortcut': False,
        'thresholds': {
            'syn': ALERT_THRESHOLD_SYN, 'ports': ALERT_THRESHOLD_PORTS,
            'icmp': ALERT_THRESHOLD_ICMP, 'udp': ALERT_THRESHOLD_UDP,
        },
        # [修正] 依攻擊類型回傳相關門檻，不再永遠是 SYN 的 100。
        'threshold': _relevant_threshold_for(attack_type),
    }


# ============================================================
# 【取代】def simulation_api(request): ...
# ============================================================
@login_required
def simulation_api(request):
    """模擬檢測 AJAX 端點（v2）。

    與前一版相比的四項核心修正：

    [根因 1／已存在，保留] 最終「攻擊／正常」判定只由規則式流量
        型樣偵測（AnomalyDetector）決定，CNN／VAE 分數只作為每筆
        封包旁的參考數值。理由不變：目前多數已部署模型是以 CSV
        統計特徵訓練，套用在這裡送入的「原始封包位元組」影像上，
        重建誤差不具判斷力（train/serve skew）。

    [根因 2／本次新修正] 封包產生邏輯改用
        core/simulate_anomaly_traffic.py 的 GENERATORS（見同批修正
        的 simulate_anomaly_traffic_PATCH.py），其 bulk_scale() 會
        依「正式系統門檻 + 隨機安全margin」反推需要產生的封包數，
        保證只要強度落在允許範圍內，攻擊封包數量『一定』超過規則
        引擎的正式門檻。這直接修正了舊版「選 SYN Flood / UDP Flood
        但因為滑桿預設值/上限本來就低於正式門檻，導致模擬結果一律
        顯示正常」的問題 —— 該問題出在「模擬封包產生器給的資料量
        不足」，不是規則引擎判斷錯誤。

    [黑箱修正] 新增：
        - sample_visual：模型實際看到的封包影像（base64 PNG），
          與正常流量基準線平均影像並排，可直接比對。
        - timing_ms：逐階段耗時（封包產生 / 規則引擎 / 模型載入 /
          CNN 推論），並附上 model_was_cached 與
          server_warmup_complete，讓「第一次模擬比較慢」有明確、
          可驗證的解釋。
        - comparison_summary：規則引擎判定 vs CNN 參考判定的量化
          比對摘要。
        - model_representation：本次使用的模型是用「封包位元組」
          還是「CSV 特徵」訓練的，讓 CNN 分數的可信度說明能動態
          反映實際狀況，而不是永遠顯示同一句警語。

    [向下相容] 請求仍接受舊版的 packet_count 欄位（會換算成大致對應
        的 intensity），也接受新的 intensity 欄位（0.3~3.0，1.0為
        預設強度）；既有呼叫端（例如 analyzer/tests.py 的既有測試）
        不需要修改即可繼續運作。
    """
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'error': '僅接受 POST'}, status=405)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError) as e:
        return JsonResponse({'ok': False, 'error': f'JSON 解析失敗：{e}'}, status=400)

    t_request_start = time.perf_counter()
    timing = {}

    attack_type = data.get('attack_type', 'syn_flood')
    model_key = data.get('model_key') or 'unsupervised_vae'

    if model_key not in settings.ANOMALY_MODELS:
        return JsonResponse({'ok': False, 'error': '不支援的模型選擇'}, status=400)
    if attack_type not in SIMULATION_ATTACK_TYPES:
        return JsonResponse({'ok': False, 'error': '不支援的攻擊場景'}, status=400)

    # ── 強度（取代舊版直接輸入封包數量的欄位）──────────────────
    # 換算規則：若前端有帶新版 intensity 就直接用；否則若帶舊版
    # packet_count，除以舊版滑桿預設值 60 粗略換算成強度倍率，
    # 確保既有呼叫端不會壞掉；都沒帶則預設 1.0。
    try:
        if 'intensity' in data:
            intensity = float(data.get('intensity'))
        elif 'packet_count' in data:
            intensity = float(data.get('packet_count')) / 60.0
        else:
            intensity = 1.0
    except (TypeError, ValueError):
        intensity = 1.0
    intensity = min(max(intensity, 0.3), 3.0)

    try:
        import sys
        core_path = os.path.join(settings.BASE_DIR, 'core')
        if core_path not in sys.path:
            sys.path.insert(0, core_path)

        from simulate_anomaly_traffic import GENERATORS
        from scapy.all import raw as scapy_raw

        # ── Step 1：產生模擬封包 ──────────────────────────────
        t0 = time.perf_counter()
        _label, gen_fn = GENERATORS[attack_type]
        scapy_packets, gen_meta = gen_fn(scale=intensity)

        # 安全上限：intensity 拉到 3.0 時，多來源 SYN Flood 等場景
        # 可能產生上千筆封包；規則引擎逐筆處理仍然很快（純 dict/
        # 計數器操作），這裡的上限主要是避免極端情況下佔用過多記憶體。
        MAX_TOTAL_PACKETS = 2000
        packet_count_truncated = False
        if len(scapy_packets) > MAX_TOTAL_PACKETS:
            scapy_packets = scapy_packets[:MAX_TOTAL_PACKETS]
            packet_count_truncated = True

        raw_packets = []
        for pkt in scapy_packets:
            try:
                raw_packets.append(scapy_raw(pkt))
            except Exception:
                raw_packets.append(bytes(pkt))
        timing['packet_generation_ms'] = round((time.perf_counter() - t0) * 1000, 1)

        # ── Step 2：規則式偵測：永遠是最終判定的唯一依據 ────────
        t1 = time.perf_counter()
        behavior_result = _detect_simulation_behavior(
            scapy_packets=scapy_packets,
            attack_type=attack_type,
            packet_count=len(scapy_packets),
        )
        behavior_anomaly = behavior_result['is_attack']
        timing['rule_engine_ms'] = round((time.perf_counter() - t1) * 1000, 1)

        total_all = len(scapy_packets)
        display_indices = _select_display_indices(behavior_result, total_all, cap=400)

        results = []
        baseline_info = None
        effective_threshold = None
        model_threshold = None
        dynamic_threshold = None
        model_representation = None
        selected_model = settings.ANOMALY_MODELS[model_key]
        model_path = selected_model['path']
        model_used = bool(model_path and os.path.exists(str(model_path)))
        sample_visual = None
        anomaly_count = 0
        avg_score = 0.0

        if model_used:
            from packet_visualizer import PacketVisualizer
            import torch
            import numpy as np
            from model_registry import load_anomaly_model, compute_anomaly_scores, is_model_cached

            device = torch.device('cpu')
            was_cached = is_model_cached(str(model_path), device)

            t2 = time.perf_counter()
            bundle = load_anomaly_model(
                str(model_path), device=device,
                default_latent_dim=settings.CNN_LATENT_DIM,
            )
            timing['model_load_ms'] = round((time.perf_counter() - t2) * 1000, 1)
            timing['model_was_cached'] = was_cached

            model = bundle.model
            model_representation = bundle.input_representation
            visualizer = PacketVisualizer("medium", apply_mask=True)

            t3 = time.perf_counter()

            # ── 基準線：即時產生一批正常流量，計算動態閾值 ──────
            BASELINE_SAMPLE_COUNT = 60
            _, normal_gen = GENERATORS['normal_traffic']
            baseline_pkts, _bmeta = normal_gen(scale=BASELINE_SAMPLE_COUNT / 30.0)
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

            # [根因 1，維持] 最終判定只看規則引擎，不受 CNN 分數影響。
            final_anomaly = np.full(total_all, behavior_anomaly, dtype=bool)
            scored['is_anomaly'] = final_anomaly
            timing['cnn_inference_ms'] = round((time.perf_counter() - t3) * 1000, 1)

            anomaly_count = int(final_anomaly.sum())
            avg_score = float(scored['score'].mean()) if total_all else 0.0

            # ── [黑箱修正] 把模型實際看到的影像畫出來 ─────────────
            worst_idx = int(np.argmax(scored['score'])) if total_all else 0
            baseline_avg_img = baseline_arrs.mean(axis=0)
            sample_visual = {
                'most_anomalous_packet_index': worst_idx,
                'most_anomalous_packet_score': round(float(scored['score'][worst_idx]), 8) if total_all else None,
                'most_anomalous_image': _array_to_data_url(target_arrs[worst_idx]) if total_all else None,
                'baseline_average_image': _array_to_data_url(baseline_avg_img),
                'note': (
                    '左圖為本次模擬中 CNN／VAE 重建誤差最高的一張封包影像，'
                    '右圖為基準線（正常流量）平均影像；模型是否具備判斷力，'
                    '取決於兩張影像的統計結構差異是否確實對應真實的封包特徵，'
                    '而不是只看分數本身。'
                ),
            }

            for i in display_indices:
                score = scored['score'][i]
                pkt_alerts = behavior_result['packet_alerts'][i]
                entry = {
                    'index': i,
                    'score': round(float(score), 8),
                    'cnn_anomaly': bool(cnn_anomaly[i]),
                    'behavior_anomaly': bool(behavior_anomaly),
                    'is_anomaly': bool(final_anomaly[i]),
                    'size': len(raw_packets[i]),
                    'summary': scapy_packets[i].summary(),
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
            timing['model_load_ms'] = 0.0
            timing['cnn_inference_ms'] = 0.0
            timing['model_was_cached'] = None
            import random as _random
            threshold = settings.CNN_THRESHOLD
            is_normal = (attack_type == 'normal_traffic')

            all_scores = []
            for i in range(total_all):
                sim_score = (_random.uniform(threshold * 0.05, threshold * 0.85) if is_normal
                             else _random.uniform(threshold * 2.0, threshold * 15.0))
                all_scores.append(sim_score)
            anomaly_count = total_all if behavior_anomaly else 0
            avg_score = sum(all_scores) / max(total_all, 1)

            for i in display_indices:
                pkt_alerts = behavior_result['packet_alerts'][i]
                cnn_anom = all_scores[i] > threshold
                results.append({
                    'index': i,
                    'score': round(all_scores[i], 8),
                    'cnn_anomaly': cnn_anom,
                    'behavior_anomaly': behavior_anomaly,
                    'is_anomaly': behavior_anomaly,
                    'size': len(raw_packets[i]),
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

        pcap_filename = _save_simulation_pcap(request.user, attack_type, scapy_packets)
        timing['total_ms'] = round((time.perf_counter() - t_request_start) * 1000, 1)

        # ── [黑箱修正] 規則引擎 vs CNN 參考判定的量化比對摘要 ─────
        cnn_flagged = sum(1 for r in results if r.get('cnn_anomaly'))
        comparison_summary = {
            'rule_engine_verdict': bool(behavior_anomaly),
            'rule_engine_alert_count': len(behavior_result['alerts']),
            'cnn_reference_flagged_count_in_sample': cnn_flagged,
            'cnn_reference_sample_size': len(results),
            'agree': bool(behavior_anomaly) == bool(cnn_flagged > 0),
            'authoritative_source': 'rule_engine',
        }

        reliability_note = _CNN_RELIABILITY_NOTE_CSV_LEGACY
        if model_representation == 'packet_bytes_visualizer':
            reliability_note = _CNN_RELIABILITY_NOTE_PACKET_NATIVE

        return JsonResponse({
            'ok': True,
            'attack_type': attack_type,
            'intensity': round(intensity, 2),
            'packet_count': total_all,
            'packet_count_displayed': len(results),
            'packet_count_truncated': packet_count_truncated,
            'generation_meta': gen_meta,
            'model_used': model_used,
            'model_representation': model_representation,
            'effective_threshold': round(effective_threshold, 8) if effective_threshold is not None else None,
            'threshold': round(model_threshold, 8) if model_threshold is not None else settings.CNN_THRESHOLD,
            'dynamic_threshold': round(dynamic_threshold, 8) if dynamic_threshold is not None else None,
            'static_threshold': settings.CNN_THRESHOLD,
            'cnn_reliability_note': reliability_note,
            'final_verdict_note': (
                '最終「攻擊／正常」判定僅由下方「規則式流量型樣偵測」決定；'
                'CNN／VAE 分數僅顯示於每筆封包旁作為參考，不影響此判定。'
            ),
            'comparison_summary': comparison_summary,
            'sample_visual': sample_visual,
            'timing_ms': timing,
            'server_warmup_complete': bool(warmup_event.is_set()),
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
            'avg_score': round(avg_score, 8),
            'anomaly_count': anomaly_count,
            'rule_based_anomaly_detected': behavior_result['is_attack'],
            'pcap_download_url': request.build_absolute_uri(
                reverse('analyzer:simulation_pcap_download', args=[pcap_filename])),
            'pcap_filename': pcap_filename,
            'pcap_backup_saved': True,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'ok': False, 'error': f'{type(e).__name__}: {str(e)}'}, status=500)
