# ============================================================
# dataset_builder.py - 封包影像資料集建構模組
#
# 功能：
#   從 PCAP 檔案讀取封包，透過 PacketVisualizer 轉換為影像，
#   建構 CNN Autoencoder 所需的訓練/測試資料集。
#
# 支援格式：
#   - 單 PCAP 輸出目錄（PNG 影像 + 索引 CSV）
#   - NumPy .npy 批次矩陣（shape: N×H×W）
#   - 資料集統計報告（影像特徵分布）
# ============================================================

import os
import csv
import json
from datetime import datetime
from collections import defaultdict
from typing import Optional, List, Dict, Tuple

import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore", message="Glyph.*missing from font", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as _fm

# 自動偵測中文字型（與 packet_visualizer.py 相同邏輯）
_CJK_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "PingFang TC",
    "Heiti TC", "WenQuanYi Zen Hei", "Noto Sans CJK TC",
]
_available_fonts = {f.name for f in _fm.fontManager.ttflist}
for _f in _CJK_CANDIDATES:
    if _f in _available_fonts:
        matplotlib.rcParams["font.family"] = [_f, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break

from packet_visualizer import PacketVisualizer


class DatasetBuilder:
    """
    封包影像資料集建構器

    使用方式：
        builder = DatasetBuilder(output_dir="output/dataset")
        builder.build_from_pcap("tests/sample.pcap", label="normal")
        builder.build_from_pcap("tests/syn_flood.pcap", label="attack")
        builder.save_numpy_arrays()
        builder.generate_report()
    """

    def __init__(self,
                 output_dir: str = "output/dataset",
                 image_size: str = "medium",
                 apply_mask: bool = True,
                 max_packets_per_file: int = 0):
        """
        Args:
            output_dir            : 輸出目錄根路徑
            image_size            : 影像尺寸（small/medium/large）
            apply_mask            : 是否套用欄位遮罩
            max_packets_per_file  : 每個 PCAP 最多處理幾個封包（0=全部）
        """
        self.output_dir   = output_dir
        self.visualizer   = PacketVisualizer(image_size, apply_mask)
        self.max_packets  = max_packets_per_file

        os.makedirs(output_dir, exist_ok=True)

        # 資料集記錄
        self.index: List[Dict] = []        # {filename, label, pcap_src, pkt_no, length}
        self.arrays: Dict[str, List] = defaultdict(list)   # label -> [np.ndarray]
        self.stats:  Dict[str, Dict] = {}

    # ──────────────────────────────────────────────────────
    # 從 PCAP 建構資料集
    # ──────────────────────────────────────────────────────
    def build_from_pcap(self,
                         pcap_path: str,
                         label: str = "unknown",
                         save_png: bool = True) -> int:
        """
        讀取 PCAP 檔案並轉換每個封包為影像

        Args:
            pcap_path : PCAP 檔案路徑
            label     : 流量標籤（normal / syn_flood / port_scan / arp_spoof ...）
            save_png  : 是否儲存個別 PNG 檔案

        Returns:
            int: 成功轉換的封包數量
        """
        # 嘗試使用 scapy 讀取，若無法讀取則使用純 pcap 解析
        try:
            from scapy.all import PcapReader
            packets = self._read_pcap_scapy(pcap_path)
        except ImportError:
            packets = self._read_pcap_raw(pcap_path)

        if not packets:
            print(f"  [資料集] 警告：{pcap_path} 無法讀取或為空")
            return 0

        if self.max_packets > 0:
            packets = packets[:self.max_packets]

        # 建立輸出子目錄
        label_dir = os.path.join(self.output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        pcap_name = os.path.basename(pcap_path).replace(".pcap", "")
        count = 0

        print(f"\n  [資料集] 處理 {pcap_path}（{len(packets)} 個封包, label={label}）")

        for i, raw_bytes in enumerate(packets):
            try:
                # 轉換為影像矩陣
                arr = self.visualizer.bytes_to_image(raw_bytes)

                # 記錄到 arrays
                self.arrays[label].append(arr)

                # 儲存 PNG
                if save_png:
                    filename = f"{pcap_name}_{i:05d}.png"
                    png_path = os.path.join(label_dir, filename)
                    self.visualizer.save_image(arr, png_path)

                    self.index.append({
                        "filename":  os.path.join(label, filename),
                        "label":     label,
                        "pcap_src":  os.path.basename(pcap_path),
                        "pkt_no":    i,
                        "length":    len(raw_bytes),
                    })

                count += 1
                if (count) % 100 == 0:
                    print(f"    進度: {count}/{len(packets)}")

            except Exception as e:
                pass   # 跳過損壞的封包

        print(f"  [資料集] 完成：{count} 個影像 → {label_dir}")
        return count

    # ──────────────────────────────────────────────────────
    # 從 bytes 列表建構（不依賴 PCAP）
    # ──────────────────────────────────────────────────────
    def build_from_bytes_list(self,
                               packets: List[bytes],
                               label: str = "unknown",
                               save_png: bool = True) -> int:
        """直接從 bytes 列表建構資料集（不依賴 PCAP 檔案）"""
        label_dir = os.path.join(self.output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        count = 0

        for i, raw_bytes in enumerate(packets):
            try:
                arr = self.visualizer.bytes_to_image(raw_bytes)
                self.arrays[label].append(arr)

                if save_png:
                    filename = f"pkt_{i:05d}.png"
                    png_path = os.path.join(label_dir, filename)
                    self.visualizer.save_image(arr, png_path)
                    self.index.append({
                        "filename": os.path.join(label, filename),
                        "label":    label,
                        "pcap_src": "bytes_list",
                        "pkt_no":   i,
                        "length":   len(raw_bytes),
                    })
                count += 1
            except Exception:
                pass

        return count

    # ──────────────────────────────────────────────────────
    # 儲存 NumPy 矩陣
    # ──────────────────────────────────────────────────────
    def save_numpy_arrays(self) -> Dict[str, str]:
        """
        將所有影像矩陣儲存為 .npy 檔案

        輸出：
            output/dataset/X_normal.npy    shape=(N, H, W)
            output/dataset/X_attack.npy    shape=(M, H, W)
            output/dataset/X_all.npy       shape=(N+M, H, W) 混合
            output/dataset/y_all.npy       shape=(N+M,) 標籤（0=normal, 1=attack）
        """
        saved = {}
        all_arrays = []
        all_labels = []

        label_to_int = {}
        label_counter = 0

        for label, arrs in self.arrays.items():
            if not arrs:
                continue

            stack = np.stack(arrs, axis=0)    # (N, H, W)
            path  = os.path.join(self.output_dir, f"X_{label}.npy")
            np.save(path, stack)
            saved[label] = path
            print(f"  [資料集] {label} → {path}  shape={stack.shape}")

            if label not in label_to_int:
                label_to_int[label] = label_counter
                label_counter += 1

            all_arrays.append(stack)
            all_labels.extend([label_to_int[label]] * len(arrs))

        # 合併全部
        if all_arrays:
            X_all = np.concatenate(all_arrays, axis=0)
            y_all = np.array(all_labels, dtype=np.int32)

            x_path = os.path.join(self.output_dir, "X_all.npy")
            y_path = os.path.join(self.output_dir, "y_all.npy")
            np.save(x_path, X_all)
            np.save(y_path, y_all)

            # 儲存標籤對應
            label_map_path = os.path.join(self.output_dir, "label_map.json")
            with open(label_map_path, "w", encoding="utf-8") as f:
                json.dump(label_to_int, f, ensure_ascii=False, indent=2)

            saved["X_all"] = x_path
            saved["y_all"] = y_path
            print(f"  [資料集] 合併 → X_all.npy shape={X_all.shape}")
            print(f"  [資料集] 標籤 → y_all.npy shape={y_all.shape}")
            print(f"  [資料集] 對應 → {label_map_path}")

        return saved

    # ──────────────────────────────────────────────────────
    # 儲存索引 CSV
    # ──────────────────────────────────────────────────────
    def save_index_csv(self) -> str:
        """儲存影像索引 CSV（filename, label, pcap_src, pkt_no, length）"""
        path = os.path.join(self.output_dir, "index.csv")
        if self.index:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.index[0].keys())
                writer.writeheader()
                writer.writerows(self.index)
            print(f"  [資料集] 索引 → {path}  ({len(self.index)} 筆)")
        return path

    # ──────────────────────────────────────────────────────
    # 資料集統計報告
    # ──────────────────────────────────────────────────────
    def generate_report(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        生成資料集統計報告圖：
          - 各類別封包數量長條圖
          - 封包長度分布直方圖
          - 各類別影像平均值（Average Image）
          - 像素值分布
        """
        if not self.arrays:
            print("  [資料集] 尚無資料，無法生成報告")
            return None

        labels   = list(self.arrays.keys())
        counts   = [len(self.arrays[l]) for l in labels]
        H, W     = self.visualizer.H, self.visualizer.W
        n_labels = len(labels)

        fig = plt.figure(figsize=(15, 4 + n_labels * 1.5))
        fig.patch.set_facecolor("#0d1117")
        gs = gridspec.GridSpec(2 + (n_labels + 3) // 4, 4, figure=fig,
                               hspace=0.55, wspace=0.4)

        # ① 封包數量
        ax1 = fig.add_subplot(gs[0, :2])
        colors = plt.cm.Set2(np.linspace(0, 1, n_labels))
        bars = ax1.bar(labels, counts, color=colors, edgecolor="#333", linewidth=0.8)
        ax1.set_title("各類別封包數量", color="white", fontsize=11)
        ax1.set_ylabel("封包數", color="#aaa")
        ax1.tick_params(colors="#aaa")
        ax1.set_facecolor("#161b22")
        for bar, cnt in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(cnt), ha="center", va="bottom", color="white", fontsize=9)

        # ② 封包長度分布
        ax2 = fig.add_subplot(gs[0, 2:])
        for label, color in zip(labels, colors):
            lengths = [r["length"] for r in self.index if r["label"] == label]
            if lengths:
                ax2.hist(lengths, bins=30, alpha=0.65, color=color, label=label,
                         edgecolor="#333")
        ax2.set_title("封包長度分布", color="white", fontsize=11)
        ax2.set_xlabel("封包長度（bytes）", color="#aaa")
        ax2.set_ylabel("頻率", color="#aaa")
        ax2.tick_params(colors="#aaa")
        ax2.set_facecolor("#161b22")
        ax2.legend(facecolor="#161b22", labelcolor="white", fontsize=8)

        # ③ 各類別 Average Image
        for idx, (label, color) in enumerate(zip(labels, colors)):
            arrs = self.arrays[label]
            if not arrs:
                continue
            avg = np.mean(arrs, axis=0)
            col = idx % 4
            row = 1 + idx // 4
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(avg, cmap="hot", vmin=0, vmax=1, aspect="equal",
                      interpolation="nearest")
            ax.set_title(f"平均影像\n({label})", color="white", fontsize=9)
            ax.axis("off")

        # ④ 像素值分布（最後一行）
        ax_pix = fig.add_subplot(gs[-1, :])
        for label, color in zip(labels, colors):
            arrs = self.arrays[label]
            if arrs:
                all_pix = np.concatenate([a.flatten() for a in arrs[:50]])
                ax_pix.hist(all_pix, bins=64, alpha=0.5, color=color,
                            label=label, density=True)
        ax_pix.set_title("像素值分布（前50張，各類別）", color="white", fontsize=11)
        ax_pix.set_xlabel("像素值（0~1）", color="#aaa")
        ax_pix.tick_params(colors="#aaa")
        ax_pix.set_facecolor("#161b22")
        ax_pix.legend(facecolor="#161b22", labelcolor="white", fontsize=8)

        total = sum(counts)
        fig.suptitle(
            f"封包影像資料集統計報告  |  "
            f"總封包: {total}  |  影像尺寸: {H}×{W}  |  "
            f"欄位遮罩: {'✓' if self.visualizer.apply_mask else '✗'}  |  "
            f"產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            color="white", fontsize=11, y=1.01
        )

        if save_path is None:
            save_path = os.path.join(self.output_dir, "dataset_report.png")

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        print(f"  [資料集] 統計報告已儲存: {save_path}")
        plt.close(fig)

        return fig

    # ──────────────────────────────────────────────────────
    # PCAP 讀取工具
    # ──────────────────────────────────────────────────────
    @staticmethod
    def _read_pcap_scapy(pcap_path: str) -> List[bytes]:
        """使用 Scapy 讀取 PCAP 封包位元組"""
        from scapy.all import PcapReader
        pkts = []
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                pkts.append(bytes(pkt))
        return pkts

    @staticmethod
    def _read_pcap_raw(pcap_path: str) -> List[bytes]:
        """
        純 Python 解析 PCAP（無 Scapy 依賴）
        支援 pcap（magic: 0xa1b2c3d4）格式
        """
        import struct
        packets = []
        try:
            with open(pcap_path, "rb") as f:
                magic = f.read(4)
                if magic not in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4"):
                    return []
                little_endian = (magic == b"\xd4\xc3\xb2\xa1")
                endian = "<" if little_endian else ">"
                f.read(20)   # skip rest of global header

                while True:
                    rec_hdr = f.read(16)
                    if len(rec_hdr) < 16:
                        break
                    ts_sec, ts_usec, incl_len, orig_len = struct.unpack(
                        endian + "IIII", rec_hdr)
                    data = f.read(incl_len)
                    if len(data) < incl_len:
                        break
                    packets.append(data)
        except Exception as e:
            print(f"  [PCAP] 讀取錯誤: {e}")
        return packets
