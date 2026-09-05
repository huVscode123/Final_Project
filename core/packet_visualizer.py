# ============================================================
# packet_visualizer.py - 封包特徵影像化模組（v2.0.1 修正版）
#
# 修正清單：
#   [Bug 3] save_image：dtype 判斷條件由 "or arr.max() <= 1.0"
#           改為僅依 dtype 判斷，避免極暗 uint8 影像被誤乘以 255。
#   [Bug 4] _apply_field_mask：讀取 IPv4 IHL 欄位動態計算
#           transport header 偏移，修正 IP Options 存在時的遮罩偏差。
#   [Bug 5] _entropy：分辨 float / uint8 輸入型別，
#           避免 uint8 影像被重複乘以 255 導致直方圖失真。
# ============================================================

import os
import struct
import hashlib
from typing import Optional, Tuple, List, Union

import warnings
warnings.filterwarnings("ignore", message="Glyph.*missing from font", category=UserWarning)

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as _fm

_CJK_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang TC",
    "Heiti TC",
    "WenQuanYi Zen Hei",
    "Noto Sans CJK TC",
]

def _find_cjk_font():
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    return None

_cjk_font = _find_cjk_font()
if _cjk_font:
    matplotlib.rcParams["font.family"] = [_cjk_font, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


# ── 影像尺寸預設值 ────────────────────────────────────────
IMAGE_SIZES = {
    "small":  (28, 28, 784),
    "medium": (32, 32, 1024),
    "large":  (40, 40, 1600),
}

# ── IPv4 Header 欄位偏移（相對於 IP 層起始） ──────────────
ETH_HEADER_LEN  = 14
IPV4_SRC_OFFSET = ETH_HEADER_LEN + 12
IPV4_DST_OFFSET = ETH_HEADER_LEN + 16
IPV4_IHL_OFFSET = ETH_HEADER_LEN + 0
TCP_SPORT_OFFSET_BASE = ETH_HEADER_LEN + 20
TCP_DPORT_OFFSET_BASE = ETH_HEADER_LEN + 22


class PacketVisualizer:
    """
    封包影像化類別（v2.0.1 修正版）

    核心方法：
        bytes_to_image(raw_bytes)  -> np.ndarray (H, W) 灰階矩陣
        packet_to_image(pkt)       -> np.ndarray (需要 Scapy)
        save_image(arr, path)      -> 儲存 PNG
        visualize_comparison(...)  -> 生成對比圖（原始 vs 遮罩後）
        create_heatmap_overlay(...)-> Grad-CAM 熱力圖疊加
    """

    def __init__(self,
                 image_size: str = "medium",
                 apply_mask: bool = True,
                 normalize: bool = True,
                 skip_ethernet: bool = True):
        if image_size not in IMAGE_SIZES:
            raise ValueError(f"image_size 必須為 {list(IMAGE_SIZES.keys())} 之一")

        self.H, self.W, self.MAX_BYTES = IMAGE_SIZES[image_size]
        self.image_size    = image_size
        self.apply_mask    = apply_mask
        self.normalize     = normalize
        self.skip_ethernet = skip_ethernet

        if apply_mask and not skip_ethernet:
            raise ValueError(
                "apply_mask=True 目前僅支援搭配 skip_ethernet=True，"
                "因為欄位遮罩的位移量是以『已跳過 Ethernet Header』為前提計算。"
            )

    # ──────────────────────────────────────────────────────
    # 核心轉換：原始 bytes → 2D 灰階矩陣
    # ──────────────────────────────────────────────────────
    def bytes_to_image(self, raw_bytes: bytes,
                       packet_type: str = "unknown") -> np.ndarray:
        data = bytearray(raw_bytes)

        # ① 跳過 Ethernet Header
        if self.skip_ethernet and len(data) > ETH_HEADER_LEN:
            data = data[ETH_HEADER_LEN:]

        # ② 欄位遮罩
        if self.apply_mask:
            data = self._apply_field_mask(data)

        # ③ 截斷
        data = data[:self.MAX_BYTES]

        # ④ 補零（Padding）
        if len(data) < self.MAX_BYTES:
            data = data + bytearray(self.MAX_BYTES - len(data))

        # ⑤ Reshape 為 2D
        arr = np.frombuffer(bytes(data), dtype=np.uint8).reshape(self.H, self.W)

        # ⑥ 正規化
        if self.normalize:
            arr = arr.astype(np.float32) / 255.0

        return arr

    # ──────────────────────────────────────────────────────
    # 欄位遮罩策略
    # ──────────────────────────────────────────────────────
    def _apply_field_mask(self, data: bytearray) -> bytearray:
        """
        遮蔽封包中的識別型欄位（相對於 IP 層起始，已跳過 Ethernet）。

        [Bug 4 修正] 讀取 IPv4 IHL 以精確定位 transport header。
        [優化] 新增遮蔽 Checksum 欄位，確保相同行為不同 IP 的封包影像一致。

        遮蔽欄位：
            IPv4 Checksum    : bytes 10~11
            IPv4 src/dst IP  : bytes 12~19
            Transport Port   : bytes ihl ~ ihl+3
            TCP seq/ack      : bytes ihl+4 ~ ihl+11
            Transport Cksum  : bytes ihl+16~17 (TCP) / ihl+6~7 (UDP) / ihl+2~3 (ICMP)
        """
        if len(data) < 20:
            return data

        masked = bytearray(data)
        # 1. IPv4 Header 遮罩
        # [新增修正] ID（bytes 4~5）－ Scapy 對未指定的欄位預設使用 RandShort()，
        # 若不遮罩，同一種語意的流量（例如兩次呼叫
        # generate_attack_packets('normal_traffic', ...)）會因為隨機 IP ID
        # 產生不同的封包影像，導致「模擬用的正常流量基準線」與
        # 「同一種流量的檢測樣本」重建誤差無法穩定對齊，造成分數飄動。
        if len(masked) > 5:
            masked[4:6] = b"\x00" * 2

        # Checksum（bytes 10~11）
        if len(masked) > 11:
            masked[10:12] = b"\x00" * 2
        # src/dst IP（bytes 12~19）
        if len(masked) > 19:
            masked[12:20] = b"\x00" * 8

        # 2. 動態計算 Transport Header 位置
        # ── [P0-2 修正] IPv6 固定表頭長度 ──
        version = masked[0] >> 4
        if version == 6:
            ihl = 40  # IPv6 固定表頭長度
            # IPv6 來源/目標位址遮罩（offset 8-39）
            if len(masked) > 8:
                end = min(40, len(masked))
                masked[8:end] = b"\x00" * (end - 8)
        else:
            ihl = (masked[0] & 0x0F) * 4
        if ihl < 20: ihl = 20

        # 3. Transport Layer 遮罩（依協議）
        # ── [修正] IPv6 Next Header 在 offset 6 ──
        if version == 6:
            proto = masked[6] if len(masked) > 6 else 0
        else:
            proto = masked[9] if len(masked) > 9 else 0

        # src/dst Port (TCP=6, UDP=17)
        if proto in (6, 17):
            if len(masked) >= ihl + 4:
                masked[ihl : ihl+4] = b"\x00" * 4

        if proto == 6:   # TCP
            # seq/ack (ihl+4 ~ ihl+11)
            if len(masked) >= ihl + 12:
                masked[ihl+4 : ihl+12] = b"\x00" * 8
            # checksum (ihl+16 ~ ihl+17)
            if len(masked) >= ihl + 18:
                masked[ihl+16 : ihl+18] = b"\x00" * 2
        elif proto == 17: # UDP
            # checksum (ihl+6 ~ ihl+7)
            if len(masked) >= ihl + 8:
                masked[ihl+6 : ihl+8] = b"\x00" * 2
        elif proto == 1:  # ICMP
            # checksum (ihl+2 ~ ihl+3)
            if len(masked) >= ihl + 4:
                masked[ihl+2 : ihl+4] = b"\x00" * 2

        return masked

    # ──────────────────────────────────────────────────────
    # 從 Scapy Packet 物件轉換（整合用）
    # ──────────────────────────────────────────────────────
    def packet_to_image(self, pkt) -> np.ndarray:
        raw = bytes(pkt)
        return self.bytes_to_image(raw)

    # ──────────────────────────────────────────────────────
    # 儲存與讀取
    # ──────────────────────────────────────────────────────
    def save_image(self, arr: np.ndarray, path: str,
                   colormap: str = "gray") -> str:
        """
        儲存影像矩陣為 PNG 檔案。

        [Bug 3 修正] 原版使用 "or arr.max() <= 1.0" 判斷是否為 float 影像，
        導致全黑或極暗的 uint8 矩陣（max 值為 0 或 1）被誤乘以 255。
        現在改為僅依 dtype 判斷，不依賴值域。
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # [Bug 3 修正] 僅依 dtype 判斷，不依 max() 值
        if arr.dtype in (np.float32, np.float64):
            pixel = (arr * 255).astype(np.uint8)
        else:
            pixel = arr.astype(np.uint8)

        if colormap == "gray":
            img = Image.fromarray(pixel, mode="L")
        else:
            cmap = plt.get_cmap(colormap)
            colored = (cmap(pixel / 255.0) * 255).astype(np.uint8)[:, :, :3]
            img = Image.fromarray(colored, mode="RGB")

        img.save(path)
        return path

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        img = Image.open(path).convert("L")
        arr = np.array(img).astype(np.float32) / 255.0
        return arr

    # ──────────────────────────────────────────────────────
    # 視覺化：原始 vs 遮罩後 對比圖
    # ──────────────────────────────────────────────────────
    def visualize_comparison(self,
                              raw_bytes: bytes,
                              label: str = "",
                              save_path: Optional[str] = None,
                              show: bool = False) -> plt.Figure:
        orig_vis = PacketVisualizer(self.image_size, apply_mask=False, normalize=True)
        arr_orig = orig_vis.bytes_to_image(raw_bytes)
        arr_mask = self.bytes_to_image(raw_bytes)
        diff = np.abs(arr_orig.astype(float) - arr_mask.astype(float))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        fig.patch.set_facecolor("#0d1117")

        titles = ["原始封包影像", "欄位遮罩後影像", "差異（遮罩位置）"]
        arrays = [arr_orig, arr_mask, diff]
        cmaps  = ["gray", "gray", "hot"]

        for ax, title, arr, cmap in zip(axes, titles, arrays, cmaps):
            im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1 if cmap != "hot" else None,
                           aspect="equal", interpolation="nearest")
            ax.set_title(title, color="white", fontsize=12, pad=8)
            ax.set_xlabel(f"{self.W} pixels", color="#888", fontsize=9)
            ax.set_ylabel(f"{self.H} pixels", color="#888", fontsize=9)
            ax.tick_params(colors="#555")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(
                color="white", labelcolor="white")

        n_nonzero_orig = np.count_nonzero(arr_orig)
        n_nonzero_mask = np.count_nonzero(arr_mask)
        pkt_len = len(raw_bytes)
        info = (f"封包長度: {pkt_len} bytes  |  "
                f"有效像素(原始): {n_nonzero_orig}  |  "
                f"有效像素(遮罩後): {n_nonzero_mask}  |  "
                f"尺寸: {self.H}×{self.W} = {self.MAX_BYTES} bytes")

        fig.suptitle(
            f"封包影像化視覺化  {'— ' + label if label else ''}\n{info}",
            color="white", fontsize=11, y=1.01
        )
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
            print(f"  [影像化] 對比圖已儲存: {save_path}")

        if show:
            plt.show()

        return fig

    # ──────────────────────────────────────────────────────
    # 視覺化：Hex 位元組熱力圖
    # ──────────────────────────────────────────────────────
    def visualize_byte_heatmap(self,
                                raw_bytes: bytes,
                                label: str = "",
                                save_path: Optional[str] = None) -> plt.Figure:
        data = np.frombuffer(raw_bytes[:256], dtype=np.uint8)
        rows = (len(data) + 15) // 16
        padded = np.zeros(rows * 16, dtype=np.uint8)
        padded[:len(data)] = data
        grid = padded.reshape(rows, 16)

        fig, (ax_main, ax_hex) = plt.subplots(
            1, 2, figsize=(14, max(rows * 0.45 + 1.5, 6)),
            gridspec_kw={"width_ratios": [2, 1]}
        )
        fig.patch.set_facecolor("#0d1117")

        custom_cmap = LinearSegmentedColormap.from_list(
            "pkt", ["#0d1117", "#1a3a6b", "#2979ff", "#00e5ff", "#ffffff"]
        )
        im = ax_main.imshow(grid, cmap=custom_cmap, aspect="auto",
                            vmin=0, vmax=255, interpolation="nearest")
        ax_main.set_title(f"封包位元組熱力圖  {'— ' + label if label else ''}",
                          color="white", fontsize=12, pad=10)
        ax_main.set_xlabel("位元組偏移（列）", color="#aaa", fontsize=9)
        ax_main.set_ylabel("行（×16 bytes）",  color="#aaa", fontsize=9)
        ax_main.set_xticks(range(16))
        ax_main.set_xticklabels([f"+{i:X}" for i in range(16)], color="#aaa", fontsize=8)
        ax_main.tick_params(colors="#555")

        boundaries = {
            14: ("Ethernet", "#ff6b6b"),
            34: ("IP Header", "#ffd93d"),
            54: ("TCP Header", "#6bcb77"),
        }
        for byte_offset, (name, color) in boundaries.items():
            row = byte_offset // 16
            if row < rows:
                ax_main.axhline(y=row - 0.5, color=color,
                                linewidth=1.5, linestyle="--", alpha=0.7)
                ax_main.text(15.5, row - 0.5, name, color=color,
                             fontsize=7, va="center", ha="right")

        plt.colorbar(im, ax=ax_main, fraction=0.025, pad=0.02,
                     label="位元組值 (0x00~0xFF)").ax.yaxis.set_tick_params(
            color="white", labelcolor="white")

        ax_hex.set_facecolor("#0d1117")
        ax_hex.axis("off")
        hex_lines = []
        for r in range(min(rows, 16)):
            row_bytes = grid[r]
            hex_str  = " ".join(f"{b:02X}" for b in row_bytes[:8])
            hex_str2 = " ".join(f"{b:02X}" for b in row_bytes[8:])
            ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in row_bytes)
            hex_lines.append(f"{r*16:04X}  {hex_str}  {hex_str2}  {ascii_str}")

        ax_hex.text(0.02, 0.98, "\n".join(hex_lines),
                    transform=ax_hex.transAxes,
                    fontfamily="monospace", fontsize=7.5,
                    color="#00e5ff", va="top", ha="left",
                    linespacing=1.6)
        ax_hex.set_title("Hex Dump", color="white", fontsize=11, pad=10)

        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
            print(f"  [影像化] 熱力圖已儲存: {save_path}")

        plt.close(fig)
        return fig

    # ──────────────────────────────────────────────────────
    # Grad-CAM 熱力圖疊加
    # ──────────────────────────────────────────────────────
    def create_heatmap_overlay(self,
                                packet_image: np.ndarray,
                                cam_map: np.ndarray,
                                alpha: float = 0.6,
                                save_path: Optional[str] = None) -> np.ndarray:
        H, W = packet_image.shape

        if cam_map.shape != (H, W):
            cam_pil = Image.fromarray((cam_map * 255).astype(np.uint8), mode="L")
            cam_pil = cam_pil.resize((W, H), Image.BILINEAR)
            cam_map = np.array(cam_pil).astype(np.float32) / 255.0

        cmap = plt.get_cmap("jet")
        heat_rgb = (cmap(cam_map)[:, :, :3] * 255).astype(np.uint8)
        orig_rgb = np.stack(
            [(packet_image * 255).astype(np.uint8)] * 3, axis=-1)
        overlay = (alpha * heat_rgb + (1 - alpha) * orig_rgb).astype(np.uint8)

        if save_path:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.patch.set_facecolor("#0d1117")
            panels = [
                (orig_rgb, "原始封包影像", "gray"),
                ((cam_map * 255).astype(np.uint8), "Grad-CAM / 重建誤差圖", "jet"),
                (overlay, "疊加視覺化", None),
            ]
            for ax, (img, title, cmap_str) in zip(axes, panels):
                ax.imshow(img, cmap=cmap_str, interpolation="nearest")
                ax.set_title(title, color="white", fontsize=11)
                ax.axis("off")
            fig.suptitle("封包異常特徵 Grad-CAM 熱力圖", color="white", fontsize=13)
            fig.tight_layout()
            fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
            print(f"  [影像化] Grad-CAM 疊加圖已儲存: {save_path}")
            plt.close(fig)

        return overlay

    # ──────────────────────────────────────────────────────
    # 批次處理
    # ──────────────────────────────────────────────────────
    def batch_convert(self,
                      packets_bytes: List[bytes],
                      labels: Optional[List[str]] = None) -> np.ndarray:
        n = len(packets_bytes)
        result = np.zeros((n, self.H, self.W), dtype=np.float32)

        for i, raw in enumerate(packets_bytes):
            result[i] = self.bytes_to_image(raw)
            if (i + 1) % 500 == 0 or (i + 1) == n:
                label_str = labels[i] if labels else ""
                print(f"  [影像化] 進度: {i+1}/{n}  {label_str}")

        return result

    # ──────────────────────────────────────────────────────
    # 統計資訊
    # ──────────────────────────────────────────────────────
    def get_stats(self, arr: np.ndarray) -> dict:
        return {
            "shape":    arr.shape,
            "min":      float(arr.min()),
            "max":      float(arr.max()),
            "mean":     float(arr.mean()),
            "std":      float(arr.std()),
            "nonzero_ratio": float(np.count_nonzero(arr) / arr.size),
            "entropy":  float(self._entropy(arr)),
        }

    @staticmethod
    def _entropy(arr: np.ndarray) -> float:
        """
        計算影像的資訊熵（衡量封包複雜度）。

        [Bug 5 修正] 原版無條件執行 (arr * 255).astype(np.uint8)，
        若 arr 已是 uint8 則發生溢位回繞（255*2=254），直方圖完全失真。
        現在根據 dtype 分支處理。
        """
        if arr.dtype in (np.float32, np.float64):
            pixel_int = (arr * 255).astype(np.uint8).flatten()
        else:
            pixel_int = arr.astype(np.uint8).flatten()

        hist, _ = np.histogram(pixel_int, bins=256, range=(0, 255))
        hist = hist[hist > 0].astype(float)
        prob = hist / hist.sum()
        return float(-np.sum(prob * np.log2(prob)))