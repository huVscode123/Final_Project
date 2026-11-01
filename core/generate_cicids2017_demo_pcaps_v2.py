# -*- coding: utf-8 -*-
"""
generate_cicids2017_demo_pcaps_v2.py
CICIDS2017 Demo PCAP 產生器 v2（輕量、可重現、特徵貼近 CICFlowMeter 訓練分布）

與 v1 的主要差異（原因見 DEMO_PCAP_REVIEW.md）：
  1. 每個 PCAP 都有「足夠多的 Flow」(正常 10~40 條、攻擊 20~150 條)，
     不再出現 1-flow 的 PCAP 讓異常率只能是 0% 或 100%。
  2. 所有 TCP Flow 至少 2 個封包（PcapFlowConverter 會丟棄單封包 Flow）。
  3. 使用 CICIDS2017 實際拓樸：攻擊者 172.16.0.1（NAT 後唯一來源）、
     受害者 192.168.10.50（Ubuntu Web Server）、內網正常用戶 192.168.10.x。
  4. 模擬真實 OS 的 TCP 行為：SYN 視窗 / 後續視窗 (已縮放值)、TCP options
     (MSS/WS/SACK/Timestamp) → Header Length、Init_Win_bytes_* 不再是 Scapy 預設的 8192。
  5. 時間間隔加入隨機抖動（RTT、思考時間、服務時間），不再是固定間隔。
  6. 使用 Ethernet link-layer (DLT=1)，所有工具都能開啟。
  7. 每個攻擊只放「一眼可辨」的特徵：
       DDoS          → 單一來源 >100 個 SYN / 10 秒（觸發 SYN Flood 規則）
       PortScan      → 單一來源掃描 90 個不同 dst port（觸發 Port Scan 規則）
       SSH Brute     → 同一 dst:22 的重複短連線 + SSH banner / KEXINIT
       Web Brute     → 重複 GET /dvwa/vulnerabilities/brute/?username=..&password=..
       DoS Hulk      → 隨機 query string 的 keep-alive GET 轟炸 + Hulk 的 UA/Keep-Alive 標頭
  8. 每個檔案大小預設上限：正常 60 KB、攻擊 100 KB，超過會 assert 失敗。

用法：
    python generate_cicids2017_demo_pcaps_v2.py --out test_pcaps/cicids2017_demo_v2
"""
from __future__ import annotations

import argparse
import os
import random
import string
from typing import List

from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap
from scapy.layers.dns import DNS, DNSQR, DNSRR

SEED = 2017
ATTACKER = "172.16.0.1"          # CICIDS2017：所有攻擊流量經 NAT 後都來自此 IP
VICTIM = "192.168.10.50"         # Ubuntu16 Web Server
DNS_SRV = "192.168.10.3"
MAC = {"c": "00:50:56:aa:00:01", "s": "00:50:56:bb:00:02"}
MSS = 1448

# 各 OS 的 TCP 行為：(SYN 視窗, 後續封包的「已縮放」視窗, SYN options, 一般封包 options)
OS_PROFILE = {
    "linux": dict(
        syn_win=29200, win=229,
        syn_opts=lambda ts: [("MSS", 1460), ("SAckOK", b""), ("Timestamp", (ts, 0)),
                             ("NOP", None), ("WScale", 7)],
        opts=lambda ts, ecr: [("NOP", None), ("NOP", None), ("Timestamp", (ts, ecr))]),
    "linux_srv": dict(
        syn_win=28960, win=235,
        syn_opts=lambda ts: [("MSS", 1460), ("SAckOK", b""), ("Timestamp", (ts, 0)),
                             ("NOP", None), ("WScale", 7)],
        opts=lambda ts, ecr: [("NOP", None), ("NOP", None), ("Timestamp", (ts, ecr))]),
    "windows": dict(
        syn_win=8192, win=256,
        syn_opts=lambda ts: [("MSS", 1460), ("NOP", None), ("WScale", 8),
                             ("NOP", None), ("NOP", None), ("SAckOK", b"")],
        opts=lambda ts, ecr: []),
    "nmap": dict(  # nmap -sS：視窗 1024、僅 MSS option
        syn_win=1024, win=1024,
        syn_opts=lambda ts: [("MSS", 1460)],
        opts=lambda ts, ecr: []),
}


class Gen:
    """一個 PCAP 的封包收集器。"""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.pkts: List = []
        self.keep_tail = False      # True：保留 FIN 後的最後 ACK（會多出一條 2 封包的「appendix flow」）

    def add(self, pkt, t: float):
        pkt.time = t
        self.pkts.append(pkt)

    def save(self, path: str):
        self.pkts.sort(key=lambda p: float(p.time))
        wrpcap(path, self.pkts)


class TcpConn:
    """含序號追蹤的 TCP 連線；client/server 各自套用 OS profile。"""

    def __init__(self, g: Gen, t0: float, cip: str, cport: int, sip: str, sport: int,
                 c_os="windows", s_os="linux_srv", rtt=0.0005):
        self.g, self.t = g, t0
        self.cip, self.cport, self.sip, self.sport = cip, cport, sip, sport
        self.cp, self.sp = OS_PROFILE[c_os], OS_PROFILE[s_os]
        self.rtt = rtt
        self.seq = {"c": g.rng.randrange(1, 2**31), "s": g.rng.randrange(1, 2**31)}
        self.ts = {"c": g.rng.randrange(10**6, 10**8), "s": g.rng.randrange(10**6, 10**8)}
        self.syn_sent = False
        self.last_flags_fin = False

    # -- 低階：送出一個封包 ------------------------------------------------
    def _emit(self, side: str, flags: str, payload: bytes = b"", dt: float = 0.0):
        self.t += max(dt, 0.0)
        prof = self.cp if side == "c" else self.sp
        src, dst = (self.cip, self.sip) if side == "c" else (self.sip, self.cip)
        sp, dp = (self.cport, self.sport) if side == "c" else (self.sport, self.cport)
        other = "s" if side == "c" else "c"
        self.ts[side] += int(max(dt, 0.001) * 1000)
        is_syn = flags == "S" or flags == "SA"
        win = prof["syn_win"] if is_syn else max(1, prof["win"] + self.g.rng.randint(-2, 2))
        opts = prof["syn_opts"](self.ts[side]) if is_syn else prof["opts"](self.ts[side], self.ts[other])
        ack = 0 if flags == "S" else self.seq[other]
        pkt = (Ether(src=MAC[side], dst=MAC[other]) / IP(src=src, dst=dst, flags="DF",
               ttl=128 if (side == "c" and prof is OS_PROFILE["windows"]) else 64)
               / TCP(sport=sp, dport=dp, flags=flags, seq=self.seq[side], ack=ack,
                     window=win, options=opts))
        if payload:
            pkt = pkt / Raw(payload)
        self.g.add(pkt, self.t)
        self.seq[side] += len(payload) + (1 if flags in ("S", "SA", "F", "FA", "FPA") else 0)

    def _j(self, base: float) -> float:        # 抖動
        return base * self.g.rng.uniform(0.7, 1.5)

    # -- 高階動作 ---------------------------------------------------------
    def handshake(self):
        self._emit("c", "S")
        self._emit("s", "SA", dt=self._j(self.rtt))
        self._emit("c", "A", dt=self._j(self.rtt))

    def send(self, side: str, payload: bytes, dt: float, flags="PA"):
        """依 MSS 切段送出 payload。"""
        for i in range(0, max(len(payload), 1), MSS):
            chunk = payload[i:i + MSS]
            last = i + MSS >= len(payload)
            self._emit(side, flags if last else "A", chunk, dt=dt if i == 0 else self._j(0.00005))
        return self

    def ack(self, side: str, dt: float):
        self._emit(side, "A", dt=dt)

    def request(self, req: bytes, resp: bytes, think: float, service: float):
        """client 送 req → server 回 resp → client ACK。"""
        self.send("c", req, think)
        self.send("s", resp, self._j(service))
        self.ack("c", self._j(self.rtt))

    def close_fin(self, initiator="c", dt=0.01):
        o = "s" if initiator == "c" else "c"
        self._emit(initiator, "FA", dt=dt)
        self._emit(o, "FA", dt=self._j(self.rtt))
        if self.g.keep_tail:
            self._emit(initiator, "A", dt=self._j(self.rtt))

    def close_rst(self, side="c", dt=0.005):
        self._emit(side, "R", dt=dt)


# ───────────────────────── 內容產生器 ─────────────────────────
def rand_bytes(g: Gen, n: int) -> bytes:
    return bytes(g.rng.getrandbits(8) for _ in range(n))


def rand_word(g: Gen, n: int) -> str:
    return "".join(g.rng.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def http_req(path: str, host: str, ua: str, extra: str = "", method="GET", body: bytes = b"") -> bytes:
    h = f"{method} {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: {ua}\r\nAccept: text/html,*/*;q=0.8\r\n{extra}"
    if body:
        h += f"Content-Length: {len(body)}\r\n"
    return (h + "\r\n").encode() + body


def http_resp(g: Gen, body_len: int, status="200 OK", ctype="text/html") -> bytes:
    head = (f"HTTP/1.1 {status}\r\nServer: Apache/2.4.18 (Ubuntu)\r\nContent-Type: {ctype}\r\n"
            f"Content-Length: {body_len}\r\n\r\n").encode()
    body = (b"<html><body>" + (b"x" * max(body_len - 26, 0)) + b"</body></html>")[:body_len]
    return head + body


def tls_record(g: Gen, kind: int, n: int) -> bytes:
    return bytes([kind, 3, 3]) + n.to_bytes(2, "big") + rand_bytes(g, max(n, 1))


UA_WIN = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0 Safari/537.36"


# ───────────────────────── 正常流量（5 個）─────────────────────────
def gen_normal_web(g: Gen):
    """HTTP 瀏覽：4 位內網用戶 → 3 台外部 Web，每條連線 1~4 個 request，大小與思考時間皆有變化。"""
    t = 1_700_000_000.0
    sites = [("93.184.216.34", "example.com"), ("151.101.1.69", "news.example.org"),
             ("104.16.20.10", "static.example.net")]
    for i in range(12):
        sip, host = g.rng.choice(sites)
        cip = f"192.168.10.{g.rng.choice([14, 15, 16, 17])}"
        c = TcpConn(g, t, cip, g.rng.randint(49152, 65000), sip, 80, "windows", "linux_srv",
                    rtt=g.rng.uniform(0.01, 0.05))
        c.handshake()
        for _ in range(g.rng.randint(1, 4)):
            path = g.rng.choice(["/", "/index.html", "/style.css", "/app.js", "/img/logo.png", "/news/" + rand_word(g, 5)])
            body = int(g.rng.lognormvariate(6.2, 0.9))           # 中位數 ≈ 500 B
            c.request(http_req(path, host, UA_WIN), http_resp(g, min(body, 2500)),
                      think=g.rng.uniform(0.05, 1.2), service=c.rtt * 1.5)
        c.close_fin("c" if g.rng.random() < .6 else "s", dt=g.rng.uniform(0.2, 2.0))
        t += g.rng.uniform(0.3, 2.5)


def gen_normal_dns(g: Gen):
    """30 組 DNS 查詢/回應（UDP，每條 flow 2 封包）。"""
    t = 1_700_000_100.0
    names = ["www.google.com", "mail.example.com", "update.microsoft.com", "cdn.example.net",
             "api.github.com", "ntp.ubuntu.com", "login.live.com", "static.example.org"]
    for _ in range(30):
        cip, sport = f"192.168.10.{g.rng.randint(10, 20)}", g.rng.randint(49152, 65000)
        q = g.rng.choice(names)
        qid = g.rng.randrange(65536)
        g.add(Ether(src=MAC["c"], dst=MAC["s"]) / IP(src=cip, dst=DNS_SRV, ttl=128) / UDP(sport=sport, dport=53)
              / DNS(id=qid, rd=1, qd=DNSQR(qname=q)), t)
        rr = [DNSRR(rrname=q, ttl=300, rdata=f"10.{g.rng.randint(0, 255)}.{g.rng.randint(0, 255)}.{g.rng.randint(1, 254)}")
              for _ in range(g.rng.randint(1, 3))]
        g.add(Ether(src=MAC["s"], dst=MAC["c"]) / IP(src=DNS_SRV, dst=cip, ttl=64) / UDP(sport=53, dport=sport)
              / DNS(id=qid, qr=1, rd=1, ra=1, qd=DNSQR(qname=q), an=rr[0] if len(rr) == 1 else rr[0] / rr[1]),
              t + g.rng.uniform(0.0005, 0.02))
        t += g.rng.uniform(0.05, 1.5)


def gen_normal_ssh(g: Gen):
    """5 段互動式 SSH：小封包（按鍵/回顯）、思考停頓（>5 秒會產生 Idle 特徵）、FIN 結束。"""
    t = 1_700_000_200.0
    for sess in range(5):
        c = TcpConn(g, t, "192.168.10.8", g.rng.randint(49152, 65000), "192.168.10.51", 22,
                    "linux", "linux_srv", rtt=0.0004)
        c.handshake()
        c.send("c", b"SSH-2.0-OpenSSH_7.4\r\n", 0.01)
        c.send("s", b"SSH-2.0-OpenSSH_7.2p2 Ubuntu-4ubuntu2.2\r\n", 0.003)
        c.send("c", rand_bytes(g, 1040), 0.02)        # KEXINIT
        c.send("s", rand_bytes(g, 1080), 0.01)
        c.send("c", rand_bytes(g, 96), 0.02)
        c.send("s", rand_bytes(g, 340), 0.02)
        for k in range(g.rng.randint(9, 15)):          # 互動 shell
            gap = g.rng.choice([0.15, 0.3, 0.6, 1.2, 6.5]) if k % 5 == 4 else g.rng.uniform(0.05, 0.4)
            c.send("c", rand_bytes(g, g.rng.choice([36, 36, 52, 68])), gap)
            c.send("s", rand_bytes(g, g.rng.choice([36, 36, 100, 164, 420])), 0.004)
            if g.rng.random() < .5:
                c.ack("c", 0.001)
        c.close_fin("c", dt=g.rng.uniform(0.5, 3))
        t += g.rng.uniform(3, 8)


def gen_normal_imap(g: Gen):
    """5 段 IMAP (143) 收信：LOGIN → SELECT → FETCH → LOGOUT。"""
    t = 1_700_000_300.0
    for _ in range(5):
        c = TcpConn(g, t, f"192.168.10.{g.rng.randint(14, 18)}", g.rng.randint(49152, 65000),
                    "192.168.10.52", 143, "windows", "linux_srv", rtt=0.0006)
        c.handshake()
        c.send("s", b"* OK [CAPABILITY IMAP4rev1] Dovecot ready.\r\n", 0.004)
        for cmd, resp in [(b"a1 LOGIN user pass\r\n", b"a1 OK Logged in\r\n"),
                          (b"a2 SELECT INBOX\r\n", b"* 12 EXISTS\r\n* 0 RECENT\r\na2 OK [READ-WRITE] Select completed\r\n"),
                          (b"a3 FETCH 1:3 (FLAGS BODY[HEADER])\r\n",
                           b"* 1 FETCH (FLAGS (\\Seen) BODY[HEADER] {" + b"%d" % g.rng.randint(300, 900) + b"}\r\n" + b"H" * g.rng.randint(300, 900) + b")\r\na3 OK Fetch completed\r\n"),
                          (b"a4 LOGOUT\r\n", b"* BYE\r\na4 OK Logout completed\r\n")]:
            c.request(cmd, resp, think=g.rng.uniform(0.05, 0.6), service=0.003)
        c.close_fin("c", dt=0.02)
        t += g.rng.uniform(1, 5)


def gen_normal_https(g: Gen):
    """HTTPS (443)：TLS 握手 + 1~3 個 Application Data 來回；10 條連線。"""
    t = 1_700_000_400.0
    for _ in range(9):
        c = TcpConn(g, t, f"192.168.10.{g.rng.randint(14, 19)}", g.rng.randint(49152, 65000),
                    g.rng.choice(["172.217.3.110", "13.107.42.14", "151.101.65.69"]), 443,
                    "windows", "linux_srv", rtt=g.rng.uniform(0.01, 0.04))
        c.handshake()
        c.send("c", tls_record(g, 0x16, 512), 0.01)                  # ClientHello
        c.send("s", tls_record(g, 0x16, 1400), 0.02)          # ServerHello+Cert
        c.send("c", tls_record(g, 0x16, 126) + tls_record(g, 0x14, 1) + tls_record(g, 0x16, 40), 0.02)
        c.send("s", tls_record(g, 0x14, 1) + tls_record(g, 0x16, 40), 0.02)
        for _ in range(g.rng.randint(1, 3)):
            c.request(tls_record(g, 0x17, g.rng.randint(120, 400)),
                      tls_record(g, 0x17, min(int(g.rng.lognormvariate(6.2, 0.8)), 1800)),
                      think=g.rng.uniform(0.05, 1.5), service=c.rtt * 1.5)
        c.close_fin("c", dt=g.rng.uniform(0.2, 3))
        t += g.rng.uniform(0.3, 2)


# ───────────────────────── 攻擊（5 個）─────────────────────────
def gen_attack_ddos(g: Gen):
    """DDoS (LOIC 式 HTTP Flood)：110 條短連線 / ~0.6 秒 → 單一來源 >100 SYN（觸發 SYN Flood）。"""
    t = 1_700_001_000.0
    for i in range(110):
        c = TcpConn(g, t, ATTACKER, 1025 + i * 3 + g.rng.randint(0, 2), VICTIM, 80, "linux", "linux_srv",
                    rtt=0.0003)
        c.handshake()
        c.send("c", http_req("/", VICTIM, "Mozilla/5.0 (compatible; LOIC)"), 0.0005)
        c.send("s", http_resp(g, 120), 0.0004)
        c.close_rst("c", dt=0.0008)          # LOIC 收到回應後直接 RST，不等 FIN 四次揮手
        t += g.rng.uniform(0.003, 0.008)


def gen_attack_portscan(g: Gen):
    """PortScan (nmap -sS)：固定 sport、視窗 1024、90 個隨機順序 dst port（<100，避免同時觸發 SYN Flood 規則）；closed→RST/ACK，open→SYN/ACK 後 RST。"""
    t = 1_700_002_000.0
    open_ports = {22, 80, 443}
    ports = list(set([21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 445, 993, 995, 1723, 3306,
                      3389, 5900, 8080, 8443] + g.rng.sample(range(1, 10000), 140)))[:90]
    g.rng.shuffle(ports)
    sport = g.rng.randint(35000, 60000)
    for dp in ports:
        c = TcpConn(g, t, ATTACKER, sport, VICTIM, dp, "nmap", "linux_srv", rtt=0.0002)
        c._emit("c", "S")
        if dp in open_ports:
            c._emit("s", "SA", dt=c._j(0.0002))
            c._emit("c", "R", dt=c._j(0.0002))
        else:
            c._emit("s", "RA", dt=c._j(0.0002))
        t += g.rng.uniform(0.0008, 0.003)


def gen_attack_ssh(g: Gen):
    """SSH-Patator：22 次嘗試，每次一條完整 SSH 連線（banner→KEXINIT→ECDH→NEWKEYS→2 次認證→斷線）。"""
    t = 1_700_003_000.0
    for i in range(22):
        c = TcpConn(g, t, ATTACKER, 40000 + i * 7 + g.rng.randint(0, 5), VICTIM, 22, "linux", "linux_srv", rtt=0.0004)
        c.handshake()
        c.send("c", b"SSH-2.0-paramiko_2.0.0\r\n", 0.003)
        c.send("s", b"SSH-2.0-OpenSSH_7.2p2 Ubuntu-4ubuntu2.2\r\n", 0.002)
        c.send("c", rand_bytes(g, 560), 0.002)
        c.send("s", rand_bytes(g, 680), 0.003)
        c.send("c", rand_bytes(g, 100), 0.002)
        c.send("s", rand_bytes(g, 380), 0.004)
        c.send("c", rand_bytes(g, 16), 0.001)
        c.send("s", rand_bytes(g, 16), 0.001)
        for _ in range(2):                                  # 2 次密碼嘗試 → 失敗
            c.request(rand_bytes(g, 64), rand_bytes(g, 48), think=g.rng.uniform(0.02, 0.08),
                      service=g.rng.uniform(0.8, 2.0))   # OpenSSH 認證失敗會延遲 1~2 秒才回應
        c.close_fin("s", dt=0.01)
        t += g.rng.uniform(0.25, 0.6)          # patator 多執行緒：連線彼此重疊


def gen_attack_web_bruteforce(g: Gen):
    """Web Attack – Brute Force (DVWA)：10 條 keep-alive 連線 × 4 次 GET，password 參數逐一嘗試。"""
    t = 1_700_004_000.0
    ua = "Mozilla/5.0 (X11; Linux x86_64; rv:45.0) Gecko/20100101 Firefox/45.0"
    words = ["123456", "password", "admin", "letmein", "qwerty", "dragon", "monkey", "abc123", "iloveyou", "root"]
    n = 0
    for _ in range(10):
        c = TcpConn(g, t, ATTACKER, g.rng.randint(35000, 60000), VICTIM, 80, "linux", "linux_srv", rtt=0.0004)
        c.handshake()
        for _ in range(4):
            pw = words[n % len(words)] + (str(n // len(words)) if n >= len(words) else "")
            n += 1
            path = f"/dvwa/vulnerabilities/brute/?username=admin&password={pw}&Login=Login"
            c.request(http_req(path, VICTIM, ua, "Cookie: PHPSESSID=" + rand_word(g, 26) + "; security=low\r\nReferer: http://192.168.10.50/dvwa/vulnerabilities/brute/\r\n"),
                      http_resp(g, 900), think=g.rng.uniform(0.15, 0.45), service=0.01)
        c.close_fin("c", dt=0.05)
        t += g.rng.uniform(0.05, 0.2)


def gen_attack_dos_hulk(g: Gen):
    """DoS Hulk：30 條連線 × 3 個 GET，每個 URL 都帶隨機 query、隨機 Referer、Keep-Alive: 110~120；
    受害者過載 → 回 503 小封包。"""
    t = 1_700_005_000.0
    uas = ["Mozilla/5.0 (X11; U; Linux x86_64; en-US; rv:1.9.1.3) Gecko/20090913 Firefox/3.5.3",
           "Mozilla/5.0 (Windows; U; Windows NT 6.1; en; rv:1.9.1.3) Gecko/20090824 Firefox/3.5.3 (.NET CLR 3.5.30729)",
           "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10.5; en-US; rv:1.9.1.3) Gecko/20090824 Firefox/3.5.3"]
    for _ in range(30):
        c = TcpConn(g, t, ATTACKER, g.rng.randint(1025, 65000), VICTIM, 80, "linux", "linux_srv", rtt=0.0003)
        c.handshake()
        for _ in range(3):
            q = f"/?{rand_word(g, 6)}={rand_word(g, 8)}"
            extra = (f"Cache-Control: no-cache\r\nAccept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7\r\n"
                     f"Referer: http://www.{rand_word(g, 8)}.com/{rand_word(g, 5)}\r\n"
                     f"Keep-Alive: {g.rng.randint(110, 120)}\r\nConnection: keep-alive\r\n")
            c.request(http_req(q, VICTIM, g.rng.choice(uas), extra),
                      http_resp(g, 120, status="503 Service Unavailable"),
                      think=g.rng.uniform(0.002, 0.02), service=0.02)
        c.close_fin("c", dt=0.05)
        t += g.rng.uniform(0.02, 0.09)


NORMAL = [("normal_01_web_browsing", gen_normal_web), ("normal_02_dns_queries", gen_normal_dns),
          ("normal_03_ssh_session", gen_normal_ssh), ("normal_04_email_imap", gen_normal_imap),
          ("normal_05_https_browsing", gen_normal_https)]
ATTACK = [("attack_01_ddos", gen_attack_ddos), ("attack_02_portscan", gen_attack_portscan),
          ("attack_03_ssh_bruteforce", gen_attack_ssh), ("attack_04_web_bruteforce", gen_attack_web_bruteforce),
          ("attack_05_dos_hulk", gen_attack_dos_hulk)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_pcaps/cicids2017_demo_v2")
    ap.add_argument("--fin-tail", choices=["drop", "keep"], default="drop",
                    help="keep：保留 FIN 後最後一個 ACK，轉 flow 時會多出 2 封包的 appendix flow（與原版 CICFlowMeter 行為一致，"
                         "但正常/攻擊兩邊都會多出幾乎相同的小 flow，會稀釋異常率）；drop（預設）：每條連線剛好 1 個 flow")
    ap.add_argument("--max-normal-kb", type=int, default=60)
    ap.add_argument("--max-attack-kb", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for idx, (name, fn) in enumerate(NORMAL + ATTACK):
        g = Gen(SEED + idx)
        g.keep_tail = (a.fin_tail == "keep")
        fn(g)
        path = os.path.join(a.out, name + ".pcap")
        g.save(path)
        kb = os.path.getsize(path) / 1024
        limit = a.max_normal_kb if name.startswith("normal") else a.max_attack_kb
        print(f"  {name:28s} {len(g.pkts):5d} pkts  {kb:6.1f} KB  (上限 {limit} KB)")
        if kb > limit:
            print(f"    !! {name} 超過大小上限 {limit} KB")


if __name__ == "__main__":
    main()
