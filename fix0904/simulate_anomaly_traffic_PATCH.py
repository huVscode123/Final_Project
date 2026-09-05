# ============================================================
# core/simulate_anomaly_traffic.py — 修補指示（非整檔取代）
#
# ── 為什麼要動這個檔案 ────────────────────────────────────────
# 根因 2（封包數量與規則引擎門檻脫節）的證據：
#
#   analyzer/views.py::simulation_api() 目前呼叫的是
#   core/generate_attack_pcap.py::generate_attack_packets(type, count)，
#   這個函式對每種攻擊「原封不動」依照使用者從滑桿選的 count 產生
#   對應數量的封包，本身完全不知道 core/config.py 定義的正式門檻：
#       ALERT_THRESHOLD_SYN   = 100
#       ALERT_THRESHOLD_PORTS = 20
#       ALERT_THRESHOLD_ICMP  = 50
#       ALERT_THRESHOLD_UDP   = 200
#
#   而 simulation.html 的滑桿設定是：
#       <input type="range" id="packetCount" min="5" max="250" value="60" ...>
#
#   實際算過之後：
#     - 預設值 60：icmp_flood(門檻50) ✓會觸發／syn_flood(門檻100) ✗不會
#       觸發／udp_flood(門檻200) ✗不會觸發。
#     - 滑桿最大值 250：udp_flood 仍然 ✗不會觸發（250 未大於 200，且
#       250 剛好是後端 min(...,250) 的硬上限，永遠無法真正做出會觸發
#       UDP Flood 的模擬）。
#
#   這正是使用者回報「選了 SYN Flood / UDP Flood，結果卻顯示正常」
#   的直接原因 —— 不是規則引擎判斷錯，是「模擬封包產生器」給的
#   資料量本來就不夠讓正式門檻觸發，規則引擎誠實地回報「未觸發」
#   完全正確，只是不符合使用者「選了攻擊應該要被判定為攻擊」的
#   直覺預期。
#
#   本檔（simulate_anomaly_traffic.py）其實早就正確solve了這個問題：
#   它的 bulk_scale() 會用「正式門檻 + 隨機安全margin」反推封包數量，
#   保證產生出來的攻擊封包數量『一定』超過正式門檻（見下方驗算），
#   而且已經有自己的單元測試（tests/test_simulated_anomaly.py）逐一
#   驗證六種攻擊都能觸發 AnomalyDetector。問題是 analyzer/views.py
#   目前根本沒有呼叫這個模組的產生器，白白放著一套已經驗證正確的
#   邏輯沒用。
#
#   驗算（scale=1.0 為預設強度）：
#     syn_flood : bulk_scale(100,10,80,1.0)  → 每個攻擊來源 IP 110~180
#                 個封包（1~3 個來源），皆 > 100 ✓
#     port_scan : bulk_scale(20,5,40,1.0)    → 25~60 個不重複 Port，
#                 > 20 ✓
#     icmp_flood: bulk_scale(50,10,100,1.0)  → 60~150 個封包，> 50 ✓
#     udp_flood : bulk_scale(200,20,200,1.0) → 220~400 個封包，
#                 > 200 ✓（這是舊版滑桿設計『無論如何都無法觸發』的
#                 那個場景，新版可以正確觸發）
#     arp_spoof / dns_amplification：本身用的是「型樣式」判定
#                （第二個不同 MAC / 回應查詢比例過高），不依賴封包
#                數量，一定觸發。
#
# ── 本次修補內容 ─────────────────────────────────────────────
#   (A) 新增 gen_normal_traffic(scale=1.0) 函式，補齊 GENERATORS 缺少
#       的「正常流量」產生器（原檔六種攻擊都有，唯獨沒有正常流量，
#       因為原本這支腳本的定位是「產生攻擊測試資料」，不含正常流量）。
#   (B) 把 GENERATORS 字典多加一筆 "normal_traffic"。
#   (C) 修正 verify_packets()：原本的 expect_substr 字典沒有
#       "normal_traffic" 這個 key，若照舊在 CLI 用
#       --types normal_traffic --verify 會直接 KeyError；改為對
#       normal_traffic 做「不應觸發任何告警」的相反驗證。
#
# ── 如何套用 ─────────────────────────────────────────────────
#   1. 在 core/simulate_anomaly_traffic.py 現有的
#      def gen_dns_amplification(scale=1.0): ... 函式「之後」、
#      GENERATORS = {...} 定義「之前」，插入下方 (A) 的完整函式。
#   2. 把現有的
#          GENERATORS = {
#              "syn_flood":         ("SYN Flood",         gen_syn_flood),
#              "port_scan":         ("Port Scan",         gen_port_scan),
#              "icmp_flood":        ("ICMP Flood",        gen_icmp_flood),
#              "udp_flood":         ("UDP Flood",         gen_udp_flood),
#              "arp_spoof":         ("ARP Spoofing",      gen_arp_spoof),
#              "dns_amplification": ("DNS Amplification", gen_dns_amplification),
#          }
#      整段取代為下方 (B) 的版本。
#   3. 把現有的 def verify_packets(key, pkts): ... 整個函式取代為
#      下方 (C) 的版本。
#   不需要新增任何 import（DNS / DNSQR / random / rand_ip / jitter_times /
#   eth_wrap 等本檔開頭都已經有了）。
# ============================================================


# ────────────────────────────────────────────────────────────
# (A) 新增：正常流量產生器（插入在 gen_dns_amplification 之後）
# ────────────────────────────────────────────────────────────
def gen_normal_traffic(scale=1.0):
    """
    產生隨機正常流量（HTTP / DNS 查詢 / ICMP Echo / TLS 混合）。

    與其他六個 gen_* 函式風格一致：來源 IP 隨機取自私有／文件保留
    網段、封包間隔加入抖動、回傳 (packets, meta)。刻意「不」呼叫
    bulk_scale()，因為正常流量不需要保證超過任何門檻 —— 它的設計
    目的正好相反：不管 scale 怎麼調，都不應該讓 AnomalyDetector
    觸發任何告警（見下方 verify_packets 的對應驗證）。

    scale 在這裡單純控制「產生幾個封包」（用於模擬頁面的『強度』
    滑桿仍能對正常流量產生視覺上的數量差異），而非攻擊強度。
    """
    count = max(5, int(30 * max(scale, 0.1)))
    pkts = []
    for i in range(count):
        variant = i % 4
        src = rand_ip(VICTIM_POOL)
        if variant == 0:
            pkts.append(eth_wrap(
                IP(src=src, dst="93.184.216.34")
                / TCP(sport=random.randint(1024, 65535), dport=80, flags="PA")
                / b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"
            ))
        elif variant == 1:
            pkts.append(eth_wrap(
                IP(src=src, dst=rand_ip(DNS_SERVER_POOL))
                / UDP(sport=random.randint(1024, 65535), dport=53)
                / DNS(rd=1, qd=DNSQR(qname="example.com"))
            ))
        elif variant == 2:
            pkts.append(eth_wrap(
                IP(src=src, dst=rand_ip(DNS_SERVER_POOL))
                / ICMP(type=8, code=0, id=random.randint(0, 65535), seq=i)
            ))
        else:
            pkts.append(eth_wrap(
                IP(src=src, dst="93.184.216.34")
                / TCP(sport=random.randint(1024, 65535), dport=443, flags="PA")
                / b"\x16\x03\x03\x00\x05\x01\x00\x00\x01\x00"
            ))

    jitter_times(pkts, mean_gap=0.02, jitter=0.6)
    return pkts, {"packet_count": len(pkts)}


# ────────────────────────────────────────────────────────────
# (B) 取代：GENERATORS 字典（加入 normal_traffic）
# ────────────────────────────────────────────────────────────
GENERATORS = {
    "syn_flood":         ("SYN Flood",         gen_syn_flood),
    "port_scan":         ("Port Scan",         gen_port_scan),
    "icmp_flood":        ("ICMP Flood",        gen_icmp_flood),
    "udp_flood":         ("UDP Flood",         gen_udp_flood),
    "arp_spoof":         ("ARP Spoofing",      gen_arp_spoof),
    "dns_amplification": ("DNS Amplification", gen_dns_amplification),
    "normal_traffic":    ("Normal Traffic",    gen_normal_traffic),
}


# ────────────────────────────────────────────────────────────
# (C) 取代：verify_packets()（原版沒有處理 normal_traffic，
#     缺少對應 key 會在 --verify 時直接 KeyError）
# ────────────────────────────────────────────────────────────
def verify_packets(key, pkts):
    """把產生的封包餵給 PacketParser + AnomalyDetector，確認：
        - 六種攻擊類型：確實觸發對應告警。
        - normal_traffic：確實『不』觸發任何告警（規則引擎的誠實
          基準線，也是 simulation_api() 拿來算 CNN 分數基準線時
          期待的輸入型態）。
    """
    parser = PacketParser()
    detector = AnomalyDetector(on_alert=lambda a: None)  # 靜音，只看回傳結果

    triggered_types = set()
    for pkt in pkts:
        record = parser.parse(pkt)
        alerts = detector.inspect(pkt, record)
        for a in alerts:
            triggered_types.add(a["attack_type"])

    if key == "normal_traffic":
        matched = (len(triggered_types) == 0)
        return matched, triggered_types

    expect_substr = {
        "syn_flood": "SYN Flood",
        "port_scan": "Port Scan",
        "icmp_flood": "ICMP",
        "udp_flood": "UDP Flood",
        "arp_spoof": "ARP Spoofing",
        "dns_amplification": "DNS Amplification",
    }[key]

    matched = any(expect_substr in t for t in triggered_types)
    return matched, triggered_types


# ────────────────────────────────────────────────────────────
# 補充：main() 裡的 --types all 邏輯不需要修改
# ────────────────────────────────────────────────────────────
# main() 中原本就是：
#     if args.types.strip().lower() == "all":
#         selected = list(GENERATORS.keys())
# 一旦 GENERATORS 多了 "normal_traffic"，--types all 會自動連正常
# 流量的 PCAP 也一併輸出（sim_normal_traffic_....pcap），這是預期
# 中、對後續建立訓練資料集（見 core/packet_dataset_builder.py）
# 也有幫助的附帶效果，不需要額外處理。
