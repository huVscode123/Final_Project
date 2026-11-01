# -*- coding: utf-8 -*-
"""
anomaly_detector_extra.py —— 以「子類別」方式擴充 core/anomaly_detector.py，不改動原檔。

修正 / 新增：
  [Fix] Port Scan 只計算「探測封包」(純 SYN / NULL / FIN / XMAS，且不含 ACK、RST)。
        原版只排除 flags == "RST"，但 parser 輸出的是 "RST+ACK"、"SYN+ACK"、"PSH+ACK"…，
        所以受害主機的回應封包（dst_port = 攻擊者的臨時 port）會被當成「反向掃描」，
        產生 src_ip=受害者的誤報（例如 DDoS/SSH/Web/Hulk 全被標成 Port Scan (TCP Scan)）。
  [New] Brute Force：同一來源在 30 秒內對同一目標的 21/22/23/3389 建立 >=15 條新連線。
  [New] Web Login Brute Force：同一來源 30 秒內 >=20 次 GET/POST 帶 password= 參數。
  [New] HTTP Flood (Hulk-like)：同一來源 10 秒內 >=60 個 GET 且 >=50 個不同 URL。
"""
from collections import defaultdict
from scapy.layers.inet import TCP
from scapy.packet import Raw

from anomaly_detector import AnomalyDetector, _SlidingWindowCounter

AUTH_PORTS = {21: "FTP", 22: "SSH", 23: "Telnet", 3389: "RDP"}


class ExtendedAnomalyDetector(AnomalyDetector):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.bf_counter = _SlidingWindowCounter(30)
        self.web_bf_counter = _SlidingWindowCounter(30)
        self.http_counter = _SlidingWindowCounter(10)
        self.http_urls = defaultdict(dict)          # src -> {url: last_ts}

    # ── Fix：只計探測封包 ─────────────────────────────────
    def _check_port_scan(self, record, now):
        flags = record.get("flags") or ""
        if "ACK" in flags or "RST" in flags:        # 回應 / 已建立連線的封包，不是探測
            return []
        return super()._check_port_scan(record, now)

    # ── 擴充規則 ──────────────────────────────────────────
    def inspect(self, pkt, record, timestamp=None):
        alerts = super().inspect(pkt, record, timestamp)
        now = float(timestamp) if timestamp is not None else float(pkt.time)
        extra = []
        if pkt.haslayer(TCP):
            extra += self._check_bruteforce(pkt, record, now)
            extra += self._check_http(pkt, record, now)
        for a in extra:
            self.alert_history.append(a)
            self.on_alert(a)
        return alerts + extra

    def _check_bruteforce(self, pkt, record, now):
        tcp = pkt[TCP]
        if tcp.flags != "S" or int(tcp.dport) not in AUTH_PORTS:
            return []
        key = f"{record['src_ip']}>{record['dst_ip']}:{tcp.dport}"
        self.bf_counter.add(key, now)
        n = self.bf_counter.count(key, now)
        ck = f"BF_{key}"
        if n >= 15 and not self._is_in_cooldown(ck, now):
            self._mark_alerted(ck, now)
            svc = AUTH_PORTS[int(tcp.dport)]
            return [self._make_alert(
                attack_type=f"{svc} Brute Force", severity=self.SEVERITY_HIGH, src_ip=record["src_ip"],
                detail=f"30 秒內對 {record['dst_ip']}:{tcp.dport} 建立 {n} 條新連線（閾值 15）",
                suggestion="封鎖來源 IP；啟用 fail2ban / 登入失敗鎖定")]
        return []

    def _check_http(self, pkt, record, now):
        if not pkt.haslayer(Raw):
            return []
        data = bytes(pkt[Raw].load[:600])
        if not (data.startswith(b"GET ") or data.startswith(b"POST ")):
            return []
        src = record["src_ip"]
        try:
            url = data.split(b" ", 2)[1].decode("latin1")
        except Exception:
            return []
        alerts = []
        if b"password=" in data.split(b"\r\n", 1)[0].lower() or b"password=" in data[-200:].lower():
            self.web_bf_counter.add(src, now)
            n = self.web_bf_counter.count(src, now)
            if n >= 20 and not self._is_in_cooldown(f"WBF_{src}", now):
                self._mark_alerted(f"WBF_{src}", now)
                alerts.append(self._make_alert(
                    attack_type="Web Login Brute Force", severity=self.SEVERITY_HIGH, src_ip=src,
                    detail=f"30 秒內 {n} 次帶 password= 參數的請求（閾值 20）",
                    suggestion="啟用登入失敗鎖定 / CAPTCHA；封鎖來源 IP"))
        self.http_counter.add(src, now)
        urls = self.http_urls[src]
        urls[url] = now
        for u in [u for u, t in urls.items() if t < now - 10]:
            del urls[u]
        n = self.http_counter.count(src, now)
        if n >= 60 and len(urls) >= 50 and not self._is_in_cooldown(f"HF_{src}", now):
            self._mark_alerted(f"HF_{src}", now)
            alerts.append(self._make_alert(
                attack_type="HTTP Flood (Hulk-like)", severity=self.SEVERITY_HIGH, src_ip=src,
                detail=f"10 秒內 {n} 個 HTTP 請求、{len(urls)} 個不同 URL（閾值 60 / 50）",
                suggestion="啟用 rate limit / WAF；封鎖來源 IP"))
        return alerts
