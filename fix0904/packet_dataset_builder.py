# ============================================================
# core/packet_dataset_builder.py（新增檔案）
#
# ── 這個腳本要解決的根本問題 ──────────────────────────────────
# 這是本次追查到「最深層」的問題，也是「每個攻擊場景結果都不合理」
# 「閾值不合理」這兩個症狀真正的根因，比封包產生器與規則引擎門檻
# 脫節（已在 simulate_anomaly_traffic_PATCH.py 修正）更根本。
#
# 證據（皆為實際追程式碼、非讀註解得出）：
#
#   1) core/run_training.py／core/run_semi_supervised.py 訓練模型時，
#      呼叫的是 core/dataset_loader.py::DatasetFactory.load(...)。
#      該模組（NSLKDDLoader / CICIDSLoader / CICDDoS2019Loader）讀的
#      是 CICFlowMeter 算出的「CSV 統計流量特徵」（duration、
#      byte 數、封包數、IAT 統計、Flag 計數...等 41～78 個欄位），
#      經 MinMaxScaler 正規化後，補零 reshape 成 32x32 影像。
#      也就是說：訓練時，影像裡的每個「像素」對應的是一個統計特徵。
#
#   2) 但正式 PCAP 分析（analyzer/tasks.py::_run_cnn_analysis）與
#      模擬檢測（analyzer/views.py::simulation_api）在「推論」時，
#      呼叫的是 core/packet_visualizer.py::PacketVisualizer
#      .bytes_to_image()，直接把「封包的原始位元組」（遮罩掉
#      IP/Port/Checksum 等欄位後）當成像素值 reshape 成 32x32 影像。
#      也就是說：推論時，影像裡的每個「像素」對應的是封包某個
#      位元組偏移量的數值。
#
#   3) (1) 和 (2) 是完全不同的資料分布 —— 同一個座標 (0,0) 在訓練
#      資料裡代表的是「某個流量統計特徵」，在推論資料裡代表的是
#      「封包 Ethernet Header 之後第 0 個位元組」。模型從來沒有
#      學過如何重建「封包原始位元組排列出來的影像」，重建誤差在
#      這種輸入下基本上不具備「這是不是異常流量」的判斷力，分數
#      可能對任何輸入都偏高、偏低，或忽高忽低，這就是「閾值看起來
#      不合理」的真正原因，並非閾值百分位數算錯，而是這個分數
#      從一開始就不是在講同一件事。
#
#   4) 這個問題並非模擬頁面獨有：analyzer/tasks.py 的正式 PCAP
#      分析管線一樣呼叫 PacketVisualizer，代表 Session 詳情頁、
#      儀表板 KPI 上顯示的 CNN 偵測率／異常數，同樣建立在這個
#      表示法不一致之上。目前系統能維持「大致堪用」，完全是因為
#      analyzer/views.py::simulation_api 已經把最終判定改成完全
#      依賴規則式偵測（AnomalyDetector），CNN 分數被降級為僅供
#      參考 —— 這是正確、必要的安全網，但終究是繞過問題，不是
#      解決問題。
#
#   5) core/dataset_builder.py::DatasetBuilder 其實「已經」正確地
#      使用 PacketVisualizer 處理真實 PCAP（build_from_pcap()），
#      與推論時的表示法完全一致；但檢查 run_training.py /
#      run_semi_supervised.py / run_threshold_tuning.py /
#      run_full_pipeline.py 這幾支訓練入口腳本後，沒有任何一支
#      import 或呼叫 DatasetBuilder —— 它是一段正確但從未被使用
#      過的程式碼。
#
#   6) 「目前所有模型只訓練 50 次」不是這個問題的根因，而是症狀
#      被誤判後開的錯藥方：即使把 epoch 從 50 加到 500，模型依然
#      是在「CSV 統計特徵影像」上學習重建，拿去解讀「封包位元組
#      影像」一樣沒有意義。epoch 數量只有在『訓練資料表示法已經
#      修正』之後才有意義去調。
#
# ── 這個腳本怎麼修正 ─────────────────────────────────────────
# 改用 core/simulate_anomaly_traffic.py 的封包產生器（已在同批修正
# 中補上 gen_normal_traffic 並涵蓋六種攻擊；這些產生器本身已有
# bulk_scale() 的門檻感知隨機化邏輯與 tests/test_simulated_anomaly.py
# 的通過測試作為品質保證），大量、多樣化地產生「正常」與「六種
# 攻擊」封包，透過與推論時完全相同的 PacketVisualizer 轉成 32x32
# 影像，輸出成 .npy，可直接餵給既有的 VAETrainer /
# HybridSemiSupervisedTrainer（不需要修改那些類別本身）。
#
# ── 使用方式 ────────────────────────────────────────────────
#   python core/packet_dataset_builder.py \
#       --output output/dataset_packet_native \
#       --n-normal-batches 400 \
#       --n-attack-batches 80 \
#       --seed 42
#
# ── 輸出 ────────────────────────────────────────────────────
#   output/dataset_packet_native/X_normal.npy       (N, 32, 32) float32
#   output/dataset_packet_native/X_attack.npy       (M, 32, 32) float32
#   output/dataset_packet_native/y_attack_type.npy  (M,) 物件陣列，
#                                                     記錄每筆攻擊樣本
#                                                     原始的攻擊類型
#   output/dataset_packet_native/build_manifest.json 產生設定與統計摘要
# ============================================================

import os
import sys
import json
import random
import argparse
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _packets_to_images(scapy_packets, visualizer):
    """把一批 Scapy 封包轉成 PacketVisualizer 影像陣列（list of (H,W) ndarray）。
    與 analyzer/tasks.py::_run_cnn_analysis()、
    analyzer/views.py::simulation_api() 使用完全相同的轉換方式，
    確保訓練資料與推論資料在同一個表示法下。
    """
    from scapy.all import raw as scapy_raw
    images = []
    for pkt in scapy_packets:
        try:
            b = scapy_raw(pkt)
        except Exception:
            b = bytes(pkt)
        try:
            images.append(visualizer.bytes_to_image(b))
        except Exception:
            continue
    return images


def build_dataset(output_dir,
                   n_normal_batches=400,
                   n_attack_batches_per_type=80,
                   max_per_attack_type=3000,
                   scale_range=(0.4, 2.5),
                   image_size="medium",
                   seed=42):
    from packet_visualizer import PacketVisualizer
    from simulate_anomaly_traffic import GENERATORS  # 需已套用同批修正

    if "normal_traffic" not in GENERATORS:
        raise RuntimeError(
            "core/simulate_anomaly_traffic.py 尚未套用 "
            "simulate_anomaly_traffic_PATCH.py 的修正（缺少 "
            "gen_normal_traffic / GENERATORS['normal_traffic']）。"
            "請先套用該修正再執行本腳本。"
        )

    random.seed(seed)
    np.random.seed(seed)

    # apply_mask=True、skip_ethernet=True 需與 analyzer/tasks.py、
    # analyzer/views.py 使用的設定完全一致，否則同樣會造成表示法
    # 不一致（只是換一種方式重蹈覆轍）。
    visualizer = PacketVisualizer(image_size, apply_mask=True, skip_ethernet=True)

    # ── 正常流量 ──────────────────────────────────────────────
    normal_images = []
    print(f"[packet_dataset_builder] 產生 {n_normal_batches} 批正常流量...")
    _, normal_gen = GENERATORS["normal_traffic"]
    for b in range(n_normal_batches):
        scale = random.uniform(*scale_range)
        pkts, _meta = normal_gen(scale=scale)
        normal_images.extend(_packets_to_images(pkts, visualizer))
        if (b + 1) % 100 == 0:
            print(f"  正常流量批次 {b + 1}/{n_normal_batches}，累積影像 {len(normal_images)}")

    if not normal_images:
        raise RuntimeError("未能產生任何正常流量影像，請檢查 gen_normal_traffic() 是否正常運作。")

    # ── 六種攻擊流量 ──────────────────────────────────────────
    attack_images = []
    attack_labels = []
    attack_types = [k for k in GENERATORS.keys() if k != "normal_traffic"]
    print(f"[packet_dataset_builder] 產生攻擊流量，類型：{attack_types}")

    for atk_type in attack_types:
        _, gen_fn = GENERATORS[atk_type]
        type_count_before = len(attack_labels)
        for b in range(n_attack_batches_per_type):
            scale = random.uniform(*scale_range)
            pkts, _meta = gen_fn(scale=scale)
            imgs = _packets_to_images(pkts, visualizer)
            attack_images.extend(imgs)
            attack_labels.extend([atk_type] * len(imgs))
        produced = len(attack_labels) - type_count_before
        print(f"  {atk_type}: 產生 {produced} 張影像")

    X_normal = np.stack(normal_images).astype(np.float32)
    attack_images_arr = np.stack(attack_images).astype(np.float32)
    attack_labels_arr = np.array(attack_labels, dtype=object)

    # ── 各攻擊類型平衡抽樣 ────────────────────────────────────
    # 不同攻擊類型天生封包數量差很多（例如 ARP Spoofing 每批只有
    # 個位數封包，SYN Flood 每批卻有上百個），若不處理，半監督
    # 微調（HybridSemiSupervisedTrainer）會被樣本數多的攻擊類型
    # 主導。這裡對每個類型做上限抽樣，讓資料集更均衡。
    rng = np.random.default_rng(seed)
    keep_idx_parts = []
    for t in attack_types:
        idx_t = np.where(attack_labels_arr == t)[0]
        if len(idx_t) == 0:
            print(f"  [警告] {t} 沒有產生任何影像，請檢查對應的 gen_* 函式")
            continue
        if len(idx_t) > max_per_attack_type:
            idx_t = rng.choice(idx_t, max_per_attack_type, replace=False)
        keep_idx_parts.append(idx_t)

    keep_idx = np.concatenate(keep_idx_parts) if keep_idx_parts else np.array([], dtype=int)
    rng.shuffle(keep_idx)

    X_attack = attack_images_arr[keep_idx] if len(keep_idx) else attack_images_arr[:0]
    y_attack_type = attack_labels_arr[keep_idx] if len(keep_idx) else attack_labels_arr[:0]

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_normal.npy"), X_normal)
    np.save(os.path.join(output_dir, "X_attack.npy"), X_attack)
    np.save(os.path.join(output_dir, "y_attack_type.npy"), y_attack_type)

    manifest = {
        "representation": "packet_bytes_visualizer",
        "note": (
            "本資料集使用與 analyzer/tasks.py、analyzer/views.py 推論時"
            "完全相同的 PacketVisualizer 封包位元組影像表示法，"
            "修正原本 CSV 統計特徵訓練 / 封包位元組推論的 train/serve "
            "資料表示法不一致問題。"
        ),
        "image_size": image_size,
        "apply_mask": True,
        "skip_ethernet": True,
        "n_normal": int(len(X_normal)),
        "n_attack": int(len(X_attack)),
        "attack_type_counts": {
            t: int((y_attack_type == t).sum()) for t in attack_types
        },
        "scale_range": list(scale_range),
        "max_per_attack_type": max_per_attack_type,
        "seed": seed,
    }
    with open(os.path.join(output_dir, "build_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[packet_dataset_builder] 完成：正常 {len(X_normal)} 筆，攻擊 {len(X_attack)} 筆")
    print(f"  各攻擊類型分布：{manifest['attack_type_counts']}")
    print(f"  已輸出至 {os.path.abspath(output_dir)}")
    return X_normal, X_attack, y_attack_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="建立與推論時一致（PacketVisualizer 封包位元組影像）的訓練資料集，"
                     "修正 CNN/VAE 模型 train/serve 資料表示法不一致的根本問題。"
    )
    parser.add_argument("--output", default="output/dataset_packet_native")
    parser.add_argument("--n-normal-batches", type=int, default=400)
    parser.add_argument("--n-attack-batches", type=int, default=80,
                         help="每種攻擊類型各自的批次數")
    parser.add_argument("--max-per-attack-type", type=int, default=3000,
                         help="每種攻擊類型最多保留的影像數（用於平衡資料集）")
    parser.add_argument("--scale-min", type=float, default=0.4)
    parser.add_argument("--scale-max", type=float, default=2.5)
    parser.add_argument("--image-size", default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_dataset(
        output_dir=args.output,
        n_normal_batches=args.n_normal_batches,
        n_attack_batches_per_type=args.n_attack_batches,
        max_per_attack_type=args.max_per_attack_type,
        scale_range=(args.scale_min, args.scale_max),
        image_size=args.image_size,
        seed=args.seed,
    )
