# ============================================================
# config.py - 系統設定檔（v2.0.0 加強版）
# ============================================================

import os

# ── 網路介面設定 ──────────────────────────────────────────
DEFAULT_INTERFACE = None          # None = 自動偵測預設介面
CAPTURE_COUNT     = 0             # 0 = 無限制

# BPF 過濾語法（Berkeley Packet Filter）
# "tcp"              只擷取 TCP
# "port 80"          只擷取 port 80
# "host 192.168.1.1" 只擷取特定 IP
# ""                 擷取全部
DEFAULT_FILTER = ""

# ── 輸出設定 ──────────────────────────────────────────────
OUTPUT_DIR   = "./output"
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "packets.csv")
OUTPUT_JSON  = os.path.join(OUTPUT_DIR, "packets.json")
OUTPUT_PCAP  = os.path.join(OUTPUT_DIR, "captured.pcap")
OUTPUT_DB    = os.path.join(OUTPUT_DIR, "packets.db")   # SQLite 資料庫

# ── 顯示設定 ──────────────────────────────────────────────
MAX_PAYLOAD_DISPLAY = 64   # 終端機顯示的最大 Payload 長度（bytes）
SHOW_HEX_PAYLOAD    = True # 是否顯示原始 Hex Payload

# ── 異常偵測閾值 ──────────────────────────────────────────
ALERT_THRESHOLD_SYN     = 100  # 同一 IP SYN 封包超過此數   -> SYN Flood
ALERT_THRESHOLD_PORTS   = 20   # 同一 IP 掃描 Port 超過此數 -> Port Scan
ALERT_THRESHOLD_ICMP    = 50   # 同一 IP ICMP 封包超過此數  -> ICMP Flood
ALERT_THRESHOLD_UDP     = 200  # 同一 IP UDP 封包超過此數   -> UDP Flood
ALERT_THRESHOLD_DNS_AMP = 10   # 同一 IP DNS 回應/請求比例  -> DNS 放大攻擊
ALERT_ARP_SPOOF_WINDOW  = 5    # 同一 IP 在此秒數內出現多個 MAC -> ARP Spoofing

# ── 時間窗口設定（秒）────────────────────────────────────
ANOMALY_WINDOW_SECONDS = 10    # 滑動窗口分析時間範圍

# ── 協議號對應表 ──────────────────────────────────────────
PROTOCOL_MAP = {
    1:   "ICMP",
    2:   "IGMP",
    6:   "TCP",
    17:  "UDP",
    41:  "IPv6-in-IPv4",
    47:  "GRE",
    50:  "ESP",
    51:  "AH",
    58:  "ICMPv6",
    89:  "OSPF",
    132: "SCTP",
}

# ── 常見 Port 服務對應表 ──────────────────────────────────
PORT_SERVICE_MAP = {
    20:    "FTP-Data",
    21:    "FTP",
    22:    "SSH",
    23:    "Telnet",
    25:    "SMTP",
    53:    "DNS",
    67:    "DHCP-Server",
    68:    "DHCP-Client",
    80:    "HTTP",
    110:   "POP3",
    143:   "IMAP",
    443:   "HTTPS",
    445:   "SMB",
    1433:  "MSSQL",
    3306:  "MySQL",
    3389:  "RDP",
    5432:  "PostgreSQL",
    5900:  "VNC",
    6379:  "Redis",
    8080:  "HTTP-Alt",
    8443:  "HTTPS-Alt",
    9200:  "Elasticsearch",
    11211: "Memcached",
    27017: "MongoDB",
}

# ── ICMP 類型對應表（加強版）────────────────────────────
ICMP_TYPE_MAP = {
    0:  "Echo Reply",
    3:  "Destination Unreachable",
    4:  "Source Quench",
    5:  "Redirect",
    8:  "Echo Request",
    9:  "Router Advertisement",
    10: "Router Solicitation",
    11: "Time Exceeded",
    12: "Parameter Problem",
    13: "Timestamp",
    14: "Timestamp Reply",
    30: "Traceroute",
}

# ── ICMP Destination Unreachable Code 對應表 ─────────────
ICMP_UNREACH_CODE_MAP = {
    0:  "Net Unreachable",
    1:  "Host Unreachable",
    2:  "Protocol Unreachable",
    3:  "Port Unreachable",
    4:  "Fragmentation Needed",
    5:  "Source Route Failed",
    9:  "Net Admin Prohibited",
    10: "Host Admin Prohibited",
    13: "Communication Admin Prohibited",
}

# ── TLS/SSL 版本對應表 ────────────────────────────────────
TLS_VERSION_MAP = {
    b"\x03\x00": "SSL 3.0",
    b"\x03\x01": "TLS 1.0",
    b"\x03\x02": "TLS 1.1",
    b"\x03\x03": "TLS 1.2",
    b"\x03\x04": "TLS 1.3",
}

# ── TLS Content Type 對應表 ───────────────────────────────
TLS_CONTENT_TYPE_MAP = {
    20: "ChangeCipherSpec",
    21: "Alert",
    22: "Handshake",
    23: "ApplicationData",
    24: "Heartbeat",
}

# ── TLS Handshake Type 對應表 ─────────────────────────────
TLS_HANDSHAKE_TYPE_MAP = {
    1:  "ClientHello",
    2:  "ServerHello",
    4:  "NewSessionTicket",
    8:  "EncryptedExtensions",
    11: "Certificate",
    12: "ServerKeyExchange",
    13: "CertificateRequest",
    14: "ServerHelloDone",
    15: "CertificateVerify",
    16: "ClientKeyExchange",
    20: "Finished",
}
