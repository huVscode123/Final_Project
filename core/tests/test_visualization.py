# ============================================================
# core/tests/test_visualization.py
# 封包影像化完整測試與 Demo 生成腳本
# ============================================================
# 執行方式：
#   pytest core/tests/test_visualization.py -v
# ============================================================

import sys
import os
import struct
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from packet_visualizer import PacketVisualizer
from dataset_builder import DatasetBuilder

OUTPUT = "output/visualization_demo"
os.makedirs(OUTPUT, exist_ok=True)

# ── 手工構造模擬封包（不依賴 scapy）────────────────────────
def make_tcp_syn(src_ip=(192,168,1,100), dst_ip=(8,8,8,8),
                 sport=54321, dport=80):
    """模擬 TCP SYN 封包（Ethernet+IPv4+TCP）"""
    eth = bytes([
        0xff,0xff,0xff,0xff,0xff,0xff,  # dst MAC
        0xaa,0xbb,0xcc,0xdd,0xee,0xff,  # src MAC
        0x08,0x00,                       # EtherType IPv4
    ])
    ip = bytes([
        0x45, 0x00, 0x00, 0x28,          # Ver/IHL/TOS/TotalLen
        0x00, 0x01, 0x40, 0x00,          # ID/Flags/Fragment
        0x40, 0x06, 0x00, 0x00,          # TTL=64/Proto=TCP/Checksum
        *src_ip, *dst_ip,                # Src/Dst IP
    ])
    tcp = bytes([
        (sport >> 8) & 0xFF, sport & 0xFF,
        (dport >> 8) & 0xFF, dport & 0xFF,
        0x00, 0x00, 0x00, 0x01,          # seq
        0x00, 0x00, 0x00, 0x00,          # ack
        0x50, 0x02,                       # offset/flags: SYN
        0xFF, 0xFF, 0x00, 0x00,          # window/checksum
        0x00, 0x00,                       # urgent
    ])
    return eth + ip + tcp

def make_http_request(src_ip=(192,168,1,100), dst_ip=(93,184,216,34)):
    """模擬 HTTP GET 請求封包"""
    payload = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\nUser-Agent: Mozilla/5.0\r\n\r\n"
    eth = bytes([0xff]*6 + [0xaa,0xbb,0xcc,0xdd,0xee,0xff] + [0x08,0x00])
    ip  = bytes([0x45,0x00,0x00,0x28,0x00,0x02,0x40,0x00,0x40,0x06,0x00,0x00,*src_ip,*dst_ip])
    tcp = bytes([0xd4,0x31,0x00,0x50, 0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x01,
                 0x50,0x18,0xFF,0xFF,0x00,0x00,0x00,0x00])
    return eth + ip + tcp + payload

def make_icmp_flood(src_ip=(10,30,30,30), dst_ip=(192,168,1,1), seq=0):
    """模擬 ICMP Echo Request（Ping Flood）"""
    payload = bytes([0x58] * 56)
    eth = bytes([0xff]*6 + [0xaa,0xbb,0xcc,0xdd,0xee,0xff] + [0x08,0x00])
    ip  = bytes([0x45,0x00,0x00,0x54,0x00,0x03,0x40,0x00,0x40,0x01,0x00,0x00,*src_ip,*dst_ip])
    icmp = bytes([0x08,0x00,0x00,0x00,0x00,0x01,
                  (seq>>8)&0xFF, seq&0xFF])
    return eth + ip + icmp + payload

def make_arp_spoof(src_ip=(192,168,1,1), fake_mac=(0xff,0xee,0xdd,0xcc,0xbb,0xaa)):
    """模擬 ARP Spoofing 封包"""
    eth = bytes([0xff]*6 + list(fake_mac) + [0x08,0x06])
    arp = bytes([
        0x00,0x01, 0x08,0x00, 0x06,0x04, 0x00,0x02,
        *fake_mac, *src_ip,
        0xff,0xff,0xff,0xff,0xff,0xff, 192,168,1,100,
    ])
    return eth + arp

def make_dns_query(src_ip=(192,168,1,100), dst_ip=(8,8,8,8)):
    """模擬 DNS 查詢封包"""
    dns = bytes([
        0x12,0x34, 0x01,0x00, 0x00,0x01, 0x00,0x00,
        0x00,0x00, 0x00,0x00,
        0x06,b"g"[0],b"o"[0],b"o"[0],b"g"[0],b"l"[0],b"e"[0],
        0x03,b"c"[0],b"o"[0],b"m"[0],
        0x00, 0x00,0x01, 0x00,0x01,
    ])
    eth = bytes([0xff]*6 + [0xaa,0xbb,0xcc,0xdd,0xee,0xff] + [0x08,0x00])
    ip  = bytes([0x45,0x00,0x00,0x28,0x00,0x04,0x40,0x00,0x40,0x11,0x00,0x00,*src_ip,*dst_ip])
    udp = bytes([0xd6,0xb0,0x00,0x35, 0x00,0x1d,0x00,0x00])
    return eth + ip + udp + dns

def make_tls_client_hello(src_ip=(192,168,1,100), dst_ip=(93,184,216,34)):
    """模擬 TLS 1.2 ClientHello"""
    tls = bytes([0x16,0x03,0x03,0x00,0x28,
                 0x01,0x00,0x00,0x24,0x03,0x03] + [0xAB]*36 +
                [0x00,0x02,0xC0,0x2B,0x01,0x00])
    eth = bytes([0xff]*6 + [0xaa,0xbb,0xcc,0xdd,0xee,0xff] + [0x08,0x00])
    ip  = bytes([0x45,0x00,0x00,0x28,0x00,0x05,0x40,0x00,0x40,0x06,0x00,0x00,*src_ip,*dst_ip])
    tcp = bytes([0xd0,0x30,0x01,0xBB,0x00,0x00,0x00,0x03,0x00,0x00,0x00,0x02,
                 0x50,0x18,0xFF,0xFF,0x00,0x00,0x00,0x00])
    return eth + ip + tcp + tls

def write_pcap(packets, path):
    """寫出 PCAP 檔案"""
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))
        for pkt in packets:
            f.write(struct.pack("<IIII", 0, 0, len(pkt), len(pkt)))
            f.write(pkt)


# ====================================================================
print("\n" + "="*60)
print("  封包影像化模組測試與 Demo")
print("="*60)

# ── 準備各類封包 ─────────────────────────────────────────
packets = {
    "TCP SYN":         make_tcp_syn(),
    "HTTP GET":        make_http_request(),
    "ICMP Flood":      make_icmp_flood(),
    "ARP Spoof":       make_arp_spoof(),
    "DNS Query":       make_dns_query(),
    "TLS ClientHello": make_tls_client_hello(),
}

# ── 測試 1：三種尺寸的影像化 ─────────────────────────────
print("\n[測試 1] 三種尺寸影像化")
for size_name in ["small", "medium", "large"]:
    vis = PacketVisualizer(image_size=size_name, apply_mask=True)
    arr = vis.bytes_to_image(packets["HTTP GET"])
    assert arr.shape == (vis.H, vis.W), f"尺寸不符: {arr.shape}"
    assert arr.dtype == np.float32, f"dtype 錯誤: {arr.dtype}"
    assert 0.0 <= arr.min() and arr.max() <= 1.0, "數值超出 [0,1]"
    print(f"  ✓  {size_name}: shape={arr.shape}, max={arr.max():.3f}, mean={arr.mean():.3f}")

# ── 測試 2：欄位遮罩效果 ─────────────────────────────────
print("\n[測試 2] 欄位遮罩效果")
vis_no_mask  = PacketVisualizer(apply_mask=False)
vis_with_mask = PacketVisualizer(apply_mask=True)
arr_orig = vis_no_mask.bytes_to_image(packets["TCP SYN"])
arr_mask = vis_with_mask.bytes_to_image(packets["TCP SYN"])
diff = np.abs(arr_orig - arr_mask).sum()
assert diff > 0, "遮罩後應有差異"
print(f"  ✓  原始 mean={arr_orig.mean():.4f}, 遮罩後 mean={arr_mask.mean():.4f}, 差異={diff:.4f}")
# 驗證遮罩區域確實為 0（IP 欄位，跳過 Ethernet(14)+IP_offset 後的 bytes 12~19）
# 在影像中對應特定像素位置（IP layer start=0 since skip_ethernet=True）
# 位元組 12~19 在 32×32 時對應 row=0 col=12~19
H, W = 32, 32
masked_region = arr_mask.flatten()[12:20]
print(f"  ✓  遮罩 IP 區域（bytes 12~19）= {masked_region}（預期全為 0.0）")

# ── 測試 3：補零與截斷 ────────────────────────────────────
print("\n[測試 3] 截斷與補零（Padding/Truncation）")
vis = PacketVisualizer("medium")
# 超短封包（少於 MAX_BYTES）
short_pkt = bytes([0xFF, 0x00, 0xAB] * 10)   # 30 bytes
arr_short = vis.bytes_to_image(short_pkt)
assert arr_short.shape == (32, 32)
print(f"  ✓  短封包 (30B → 1024B 補零): shape={arr_short.shape}")
# 超長封包（超過 MAX_BYTES）
long_pkt = bytes(range(256)) * 10             # 2560 bytes
arr_long = vis.bytes_to_image(long_pkt)
assert arr_long.shape == (32, 32)
print(f"  ✓  長封包 (2560B → 截斷至 1024B): shape={arr_long.shape}")

# ── 測試 4：影像統計資訊 ─────────────────────────────────
print("\n[測試 4] 影像統計資訊")
vis = PacketVisualizer("medium")
for name, pkt_bytes in packets.items():
    arr  = vis.bytes_to_image(pkt_bytes)
    stat = vis.get_stats(arr)
    print(f"  {name:<20} 非零比={stat['nonzero_ratio']:.2f}  "
          f"entropy={stat['entropy']:.2f}  mean={stat['mean']:.3f}")

# ── 測試 5：生成對比圖 ────────────────────────────────────
print("\n[測試 5] 生成對比圖（原始 vs 遮罩後）")
vis = PacketVisualizer("medium", apply_mask=True)
for name, pkt_bytes in packets.items():
    safe_name = name.replace(" ", "_").lower()
    fig = vis.visualize_comparison(
        pkt_bytes,
        label=name,
        save_path=f"{OUTPUT}/compare_{safe_name}.png"
    )
    plt.close(fig)
    print(f"  ✓  {name}")

# ── 測試 6：Hex 熱力圖 ────────────────────────────────────
print("\n[測試 6] 生成 Hex 位元組熱力圖")
vis = PacketVisualizer("medium")
for name, pkt_bytes in packets.items():
    safe_name = name.replace(" ", "_").lower()
    vis.visualize_byte_heatmap(
        pkt_bytes,
        label=name,
        save_path=f"{OUTPUT}/heatmap_{safe_name}.png"
    )
    print(f"  ✓  {name}")

# ── 測試 7：批次轉換 ──────────────────────────────────────
print("\n[測試 7] 批次轉換（所有封包一次轉為矩陣）")
all_pkts = list(packets.values())
vis = PacketVisualizer("medium")
batch = vis.batch_convert(all_pkts)
assert batch.shape == (len(all_pkts), 32, 32)
assert batch.dtype == np.float32
print(f"  ✓  batch shape={batch.shape}, dtype={batch.dtype}")

# ── 測試 8：Grad-CAM 疊加介面 ────────────────────────────
print("\n[測試 8] Grad-CAM 熱力圖疊加介面（模擬重建誤差圖）")
vis = PacketVisualizer("medium")
http_img = vis.bytes_to_image(packets["HTTP GET"])
# 模擬重建誤差圖（Payload 區域誤差高）
fake_cam = np.zeros((32, 32), dtype=np.float32)
fake_cam[15:, 10:] = 0.9   # 模擬 Payload 區域高異常分數
fake_cam[5:8, :] = 0.5     # 模擬 TCP Header 區域中等異常
overlay = vis.create_heatmap_overlay(
    http_img, fake_cam,
    save_path=f"{OUTPUT}/gradcam_demo.png"
)
assert overlay.shape == (32, 32, 3)
print(f"  ✓  Grad-CAM overlay shape={overlay.shape}")

# ── 測試 9：DatasetBuilder ────────────────────────────────
print("\n[測試 9] DatasetBuilder 資料集建構")
builder = DatasetBuilder(
    output_dir=f"{OUTPUT}/dataset",
    image_size="medium",
    apply_mask=True
)

# 用自製的模擬封包各建一個類別
normal_pkts  = [make_http_request(), make_dns_query(), make_tls_client_hello()] * 20
attack_pkts  = [make_tcp_syn()] * 15 + [make_icmp_flood(seq=i) for i in range(15)]
arp_pkts     = [make_arp_spoof()] * 10

n1 = builder.build_from_bytes_list(normal_pkts,  label="normal",    save_png=True)
n2 = builder.build_from_bytes_list(attack_pkts,  label="syn_flood", save_png=True)
n3 = builder.build_from_bytes_list(arp_pkts,     label="arp_spoof", save_png=True)

print(f"  ✓  normal: {n1}, syn_flood: {n2}, arp_spoof: {n3}")

# 儲存 numpy 矩陣
saved = builder.save_numpy_arrays()
for k, v in saved.items():
    arr = np.load(v)
    print(f"  ✓  {k}: {v}  shape={arr.shape}")

# 儲存索引
builder.save_index_csv()

# ── 測試 10：PCAP 讀取（純 Python 解析器） ───────────────
print("\n[測試 10] 純 Python PCAP 讀取與影像化")
pcap_path = f"{OUTPUT}/demo.pcap"
test_pkts = [make_tcp_syn(), make_http_request(), make_icmp_flood(),
             make_dns_query(), make_arp_spoof(), make_tls_client_hello()]
write_pcap(test_pkts, pcap_path)

raw_pkts = DatasetBuilder._read_pcap_raw(pcap_path)
assert len(raw_pkts) == len(test_pkts), f"預期 {len(test_pkts)}, 讀到 {len(raw_pkts)}"
vis = PacketVisualizer("medium")
for i, raw in enumerate(raw_pkts):
    arr = vis.bytes_to_image(raw)
    assert arr.shape == (32, 32)
print(f"  ✓  從 PCAP 讀取並影像化 {len(raw_pkts)} 個封包")

# ── 統計報告 ──────────────────────────────────────────────
print("\n[報告] 生成資料集統計報告")
builder.generate_report(save_path=f"{OUTPUT}/dataset/dataset_report.png")

# ── 綜合展示圖 ────────────────────────────────────────────
print("\n[報告] 生成六種封包影像綜合展示圖")
vis = PacketVisualizer("medium", apply_mask=True)
fig, axes = plt.subplots(2, 6, figsize=(18, 7))
fig.patch.set_facecolor("#0d1117")

packet_list = list(packets.items())
for idx, (name, pkt_bytes) in enumerate(packet_list):
    # 上排：遮罩後
    arr_mask = vis.bytes_to_image(pkt_bytes)
    axes[0][idx].imshow(arr_mask, cmap="hot", vmin=0, vmax=1, aspect="equal")
    axes[0][idx].set_title(name, color="white", fontsize=8, pad=4)
    axes[0][idx].axis("off")

    # 下排：原始（無遮罩）
    vis_raw = PacketVisualizer("medium", apply_mask=False)
    arr_raw = vis_raw.bytes_to_image(pkt_bytes)
    axes[1][idx].imshow(arr_raw, cmap="gray", vmin=0, vmax=1, aspect="equal")
    axes[1][idx].set_title("原始", color="#888", fontsize=7, pad=2)
    axes[1][idx].axis("off")

axes[0][0].set_ylabel("欄位遮罩後", color="#00e5ff", fontsize=9)
axes[1][0].set_ylabel("原始影像", color="#aaa", fontsize=9)
axes[0][0].yaxis.set_visible(True)
axes[1][0].yaxis.set_visible(True)

fig.suptitle(
    "視覺化網路攻擊自動化分析平台  |  封包影像化展示  |  32×32 灰階影像",
    color="white", fontsize=13
)
fig.tight_layout()
fig.savefig(f"{OUTPUT}/overview_all_types.png", dpi=130,
            bbox_inches="tight", facecolor="#0d1117")
plt.close(fig)
print(f"  ✓  綜合展示圖: {OUTPUT}/overview_all_types.png")

print("\n" + "="*60)
print("  所有測試通過！")
print(f"  輸出目錄：{os.path.abspath(OUTPUT)}")
print("="*60 + "\n")
