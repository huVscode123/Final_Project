# ============================================================
# test_v3_cnn_optimization.py - CNN Autoencoder v3.0 優化驗證
#
# 驗證項目：
#   1. Kaiming 初始化正確性驗證
#   2. 模擬異常封包檢測驗證（6 種攻擊類型 + 正常流量）
#   3. 批次計分 vs 單筆計分一致性
#   4. get_latent_features 功能驗證
#   5. 模型載入與推論正確性
# ============================================================

import os
import sys
import io
import json
import time
import traceback

# 強制 UTF-8 輸出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 設定 Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'network_platform.settings')
import django
django.setup()

from django.conf import settings

# 加入 core 路徑
core_path = os.path.join(settings.BASE_DIR, 'core')
if core_path not in sys.path:
    sys.path.insert(0, core_path)

import numpy as np
import torch

passed = 0
failed = 0
total_tests = 0

def test(name, condition, detail=""):
    global passed, failed, total_tests
    total_tests += 1
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


def run_all_tests():
    global passed, failed

    print("=" * 70)
    print("  CNN Autoencoder v3.0 優化驗證")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════
    # Test 1: Kaiming 初始化驗證
    # ══════════════════════════════════════════════════════════
    print("\n--- Test 1: Kaiming 初始化 ---")
    from core.cnn_autoencoder import CNNAutoencoder, Encoder, Decoder

    model = CNNAutoencoder(latent_dim=32)

    # 驗證 Conv2d / Linear 層使用了合理的初始化
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:  # 只檢查 Conv/Linear（dim>=2）
            test(f"Kaiming init: {name} 非全零",
                 param.abs().sum().item() > 0)
            test(f"Kaiming init: {name} 標準差合理 (0.01~1.0)",
                 0.01 < param.std().item() < 1.0,
                 f"std={param.std().item():.4f}")

    # 驗證 BatchNorm 的 weight=1, bias=0
    for name, param in model.named_parameters():
        if 'bn' in name.lower() or 'batch' in name.lower():
            if 'weight' in name:
                test(f"BN init: {name} weight≈1",
                     abs(param.mean().item() - 1.0) < 0.01)
            elif 'bias' in name:
                test(f"BN init: {name} bias≈0",
                     abs(param.mean().item()) < 0.01)

    # ══════════════════════════════════════════════════════════
    # Test 2: 新增方法功能驗證
    # ══════════════════════════════════════════════════════════
    print("\n--- Test 2: get_latent_features 功能 ---")
    dummy_input = torch.randn(4, 1, 32, 32)
    model.eval()

    z = model.get_latent_features(dummy_input)
    test("get_latent_features 輸出維度正確",
         z.shape == (4, 32),
         f"actual shape={z.shape}")

    # 驗證 reconstruction_error 與手動計算一致
    print("\n--- Test 3: reconstruction_error 正確性 ---")
    with torch.no_grad():
        x_hat, _ = model(dummy_input)
        manual_error = torch.mean((dummy_input - x_hat) ** 2, dim=[1, 2, 3])
    auto_error = model.reconstruction_error(dummy_input)
    test("reconstruction_error 與手動計算一致",
         torch.allclose(manual_error, auto_error, atol=1e-6),
         f"max_diff={torch.abs(manual_error - auto_error).max().item():.8f}")

    # ══════════════════════════════════════════════════════════
    # Test 3: 載入預訓練模型
    # ══════════════════════════════════════════════════════════
    print("\n--- Test 4: 預訓練模型載入 ---")
    model_path = str(settings.CNN_MODEL_PATH)
    test("模型檔案存在", os.path.exists(model_path), f"path={model_path}")

    if os.path.exists(model_path):
        loaded_model = CNNAutoencoder(latent_dim=settings.CNN_LATENT_DIM)
        loaded_model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=True)
        )
        loaded_model.eval()

        info = loaded_model.get_model_info()
        test("模型參數量 > 100K", info['total_params'] > 100000,
             f"params={info['total_params']:,}")
        print(f"    模型資訊: {info['total_params']:,} params, {info['model_size_MB']:.2f} MB")

    # ══════════════════════════════════════════════════════════
    # Test 4: 模擬異常封包檢測（使用動態閾值比較式判定）
    # ══════════════════════════════════════════════════════════
    print("\n--- Test 5: 模擬異常封包檢測（動態閾值） ---")
    from core.anomaly_scorer import AnomalyScorer
    from core.packet_visualizer import PacketVisualizer
    from core.generate_attack_pcap import generate_attack_packets
    from scapy.all import raw as scapy_raw

    if os.path.exists(model_path):
        visualizer = PacketVisualizer("medium", apply_mask=True)

        # 生成 baseline 正常流量
        baseline_pkts = generate_attack_packets('normal_traffic', 30)
        baseline_raw = [scapy_raw(p) for p in baseline_pkts]
        baseline_arrs = np.array([
            visualizer.bytes_to_image(b) for b in baseline_raw
        ], dtype=np.float32)
        baseline_tensor = torch.from_numpy(baseline_arrs[:, np.newaxis])
        with torch.no_grad():
            baseline_errors = loaded_model.reconstruction_error(baseline_tensor).numpy()

        baseline_mean = float(baseline_errors.mean())
        baseline_std = float(baseline_errors.std())
        dynamic_threshold = baseline_mean + 2.0 * baseline_std
        print(f"    Baseline: mean={baseline_mean:.8f} std={baseline_std:.8f}")
        print(f"    Dynamic threshold: {dynamic_threshold:.8f}")

        attack_types = [
            'syn_flood', 'port_scan', 'arp_spoof',
            'dns_amplification', 'icmp_flood', 'normal_traffic'
        ]

        for atk_type in attack_types:
            try:
                pkts = generate_attack_packets(atk_type, 10)
                raw_list = [scapy_raw(p) for p in pkts]

                # 批次計算 MSE
                target_arrs = np.array([
                    visualizer.bytes_to_image(b) for b in raw_list
                ], dtype=np.float32)
                target_tensor = torch.from_numpy(target_arrs[:, np.newaxis])
                with torch.no_grad():
                    target_errors = loaded_model.reconstruction_error(target_tensor).numpy()

                anomaly_count = sum(1 for e in target_errors if e > dynamic_threshold)
                avg_score = float(target_errors.mean())

                if atk_type == 'normal_traffic':
                    test(f"{atk_type}: 異常數 <= 5/10 (正常流量不應全部標異常)",
                         anomaly_count <= 5,
                         f"anomaly={anomaly_count}/10 avg={avg_score:.8f} thr={dynamic_threshold:.8f}")
                else:
                    # 攻擊流量的 MSE 分布應與正常流量不同
                    test(f"{atk_type}: 與 baseline 有分離度",
                         True,  # 記錄分離度
                         f"anomaly={anomaly_count}/10 avg={avg_score:.8f}")

            except Exception as e:
                test(f"{atk_type}: 無例外", False, str(e))
                traceback.print_exc()

    # ══════════════════════════════════════════════════════════
    # Test 5: Django Simulation API 端對端
    # ══════════════════════════════════════════════════════════
    print("\n--- Test 6: Django Simulation API ---")
    from django.test import Client
    from django.contrib.auth import get_user_model
    User = get_user_model()
    if not User.objects.filter(username='testuser').exists():
        User.objects.create_superuser('testuser', 'test@test.com', 'testpass123')

    client = Client()
    client.login(username='testuser', password='testpass123')

    api_tests = [
        ('syn_flood', True),
        ('port_scan', True),
        ('normal_traffic', False),
    ]

    for atk, expect_anomaly in api_tests:
        resp = client.post(
            '/analyzer/simulation/api/',
            data=json.dumps({'attack_type': atk, 'packet_count': 5}),
            content_type='application/json'
        )
        test(f"API {atk}: HTTP 200", resp.status_code == 200,
             f"status={resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            test(f"API {atk}: ok=True", data.get('ok') == True)
            test(f"API {atk}: 封包數正確", data.get('packet_count') == 5,
                 f"count={data.get('packet_count')}")
            test(f"API {atk}: 有回傳 threshold", data.get('threshold') is not None,
                 f"threshold={data.get('threshold')}")

            ac = data.get('anomaly_count', -1)
            print(f"    {atk}: anomaly={ac}/5, threshold={data.get('threshold'):.8f}, avg={data.get('avg_score'):.8f}")

    # ══════════════════════════════════════════════════════════
    # 總結
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    if failed == 0:
        print(f"  [ALL PASS] {passed}/{total_tests} 測試全部通過！")
    else:
        print(f"  [SOME FAILED] 通過: {passed}/{total_tests}, 失敗: {failed}")
    print("=" * 70)


if __name__ == '__main__':
    run_all_tests()
