# ============================================================
# core/generate_attack_pcap.py — DNS Amplification 封包產生器修正
#
# 使用方式：
#   在 generate_attack_packets() 函式內，找到
#       elif attack_type == 'dns_amplification':
#   這一整個分支，用下面的內容完整取代它（其餘攻擊類型分支、
#   函式簽名、import 等都維持原樣不動）。
#
# ── 根因說明 ─────────────────────────────────────────────────
# 舊版：
#     n_queries = max(1, count // 10)
#     n_responses = count - n_queries
# 查詢:回應固定約 1:9 的比例。但 core/anomaly_detector.py 的
# _check_dns_amplification() 判定條件是：
#     resp > 20  且  ratio(=resp/req) > ALERT_THRESHOLD_DNS_AMP(=10)
# 1:9 的比例算出來的 ratio 最高只會逼近 9，永遠達不到「大於 10」
# 的門檻——也就是說，這個攻擊類型無論怎麼調整封包數量，規則引擎
# 都「真的」不會觸發，過去只能靠 analyzer/views.py 裡的「保底強制
# 判定」硬把畫面改成攻擊，這正是黑箱行為的根本原因。
#
# 修正：固定只送 1 筆查詢，其餘全部作為回應。回應/查詢比會等於
# 回應封包數本身，隨著封包數量線性成長且遠高於 10 倍門檻。
# 只要封包數 >= 22（回應數 = count - 1 > 20，同時滿足 resp>20 與
# ratio>10 兩個條件）即可在規則引擎中真實觸發，不再需要任何保底。
# ============================================================

    elif attack_type == 'dns_amplification':
        dns_server = "8.8.8.8"
        n_queries = 1
        n_responses = max(0, count - n_queries)
        for i in range(n_queries):
            pkts.append(
                Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02")
                / IP(src=victim, dst=dns_server)
                / UDP(sport=10000 + i, dport=53)
                / DNS(rd=1, qd=DNSQR(qname=f"amplify{i}.com", qtype=255))
            )
        for i in range(n_responses):
            pkts.append(
                Ether(src="02:00:00:00:00:02", dst="02:00:00:00:00:01")
                / IP(src=dns_server, dst=victim)
                / UDP(sport=53, dport=10000 + (i % n_queries))
                / DNS(qr=1, qd=DNSQR(qname=f"amplify{i % n_queries}.com"),
                      an=DNSRR(rrname=f"amplify{i % n_queries}.com", rdata="1.2.3.4"))
            )
