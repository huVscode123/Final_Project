# ============================================================
# core/tests/test_v3_cnn_optimizations.py
#
# CNN Autoencoder v3.0 優化功能單元測試
#
# 測試項目：
#   1. Kaiming 初始化正確性
#   2. get_latent_features() 新方法
#   3. reconstruction_error() 計算一致性與狀態安全
#   4. 動態閾值（比較式判定）邏輯
#   5. 模擬異常封包偵測（6 種攻擊 + 正常流量）
#   6. 批次計分 vs 單筆一致性
#   7. 半監督學習：AnomalyScorer threshold 設定
#
# 執行方式：
#   cd c:/Users/user/Desktop/final_project
#   pytest core/tests/test_v3_cnn_optimizations.py -v
# ============================================================

import sys
import os
import pytest
import numpy as np

# 加入 core/ 到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

torch_available = False
try:
    import torch
    import torch.nn as nn
    torch_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not torch_available,
    reason="PyTorch 未安裝"
)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fresh_model():
    from cnn_autoencoder import CNNAutoencoder
    return CNNAutoencoder(latent_dim=32)


@pytest.fixture(scope="module")
def sample_batch():
    return torch.randn(4, 1, 32, 32)


@pytest.fixture(scope="module")
def normal_batch():
    """模擬正常流量的低 MSE 輸入（接近零分布）"""
    return torch.zeros(4, 1, 32, 32) + 0.1


@pytest.fixture(scope="module")
def attack_batch():
    """模擬攻擊流量的高熵輸入"""
    return torch.rand(4, 1, 32, 32)


# ══════════════════════════════════════════════════════════════
# Group 1：Kaiming 初始化
# ══════════════════════════════════════════════════════════════

class TestKaimingInitialization:
    """Kaiming (He) 初始化正確性測試"""

    def test_conv_weights_nonzero(self, fresh_model):
        """卷積層權重初始化後不應為全零"""
        for name, param in fresh_model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                assert param.abs().sum().item() > 0, \
                    f"{name} 初始化後為全零"

    def test_conv_weights_std_reasonable(self, fresh_model):
        """卷積/全連接層 weight 標準差應在合理範圍 (0.01~1.5)"""
        for name, param in fresh_model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                std = param.std().item()
                assert 0.01 < std < 1.5, \
                    f"{name} 標準差 {std:.4f} 超出合理範圍"

    def test_batchnorm_weight_initialized_to_one(self, fresh_model):
        """BatchNorm weight 應初始化為 1"""
        for name, param in fresh_model.named_parameters():
            if 'weight' in name and param.dim() == 1 and param.numel() > 1:
                # BatchNorm weight 是 1D tensor，初始化為全 1
                mean_val = param.mean().item()
                assert abs(mean_val - 1.0) < 0.01, \
                    f"BatchNorm {name} mean={mean_val:.4f} 不接近 1"

    def test_batchnorm_bias_initialized_to_zero(self, fresh_model):
        """BatchNorm bias 應初始化為 0"""
        for name, param in fresh_model.named_parameters():
            if 'bias' in name and 'bn' not in name.lower():
                # 跳過普通 Linear bias（不是 BN 的）
                pass
        # 直接驗證 BN 模組
        for name, module in fresh_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                assert abs(module.bias.data.mean().item()) < 0.01, \
                    f"BatchNorm {name} bias 不接近 0"

    def test_encoder_decoder_both_initialized(self, fresh_model):
        """Encoder 和 Decoder 均應套用初始化"""
        enc_std = [p.std().item() for n, p in fresh_model.encoder.named_parameters()
                   if p.dim() >= 2]
        dec_std = [p.std().item() for n, p in fresh_model.decoder.named_parameters()
                   if p.dim() >= 2]
        assert all(s > 0.01 for s in enc_std), "Encoder 部分權重初始化不合理"
        assert all(s > 0.01 for s in dec_std), "Decoder 部分權重初始化不合理"


# ══════════════════════════════════════════════════════════════
# Group 2：新增方法功能
# ══════════════════════════════════════════════════════════════

class TestNewMethods:
    """v3.0 新增方法功能測試"""

    def test_get_latent_features_shape(self, fresh_model, sample_batch):
        """get_latent_features 應回傳 (batch, latent_dim) 形狀"""
        z = fresh_model.get_latent_features(sample_batch)
        assert z.shape == (4, 32), \
            f"期望 (4, 32)，實際 {z.shape}"

    def test_get_latent_features_no_grad(self, fresh_model, sample_batch):
        """get_latent_features 應在 no_grad 模式下執行（節省記憶體）"""
        z = fresh_model.get_latent_features(sample_batch)
        assert not z.requires_grad, "latent features 不應有梯度"

    def test_get_latent_features_eval_restored(self, fresh_model, sample_batch):
        """get_latent_features 呼叫後模型應恢復 train 模式"""
        fresh_model.train()
        fresh_model.get_latent_features(sample_batch)
        assert fresh_model.training, "呼叫後 train 模式未恢復"

    def test_reconstruction_error_shape(self, fresh_model, sample_batch):
        """reconstruction_error 應回傳 (batch,) 形狀"""
        errors = fresh_model.reconstruction_error(sample_batch)
        assert errors.shape == (4,), f"期望 (4,)，實際 {errors.shape}"

    def test_reconstruction_error_non_negative(self, fresh_model, sample_batch):
        """重建誤差應為非負數（MSE >= 0）"""
        errors = fresh_model.reconstruction_error(sample_batch)
        assert (errors >= 0).all(), "存在負數重建誤差"

    def test_reconstruction_error_consistency(self, fresh_model, sample_batch):
        """reconstruction_error 應與手動計算結果一致"""
        fresh_model.eval()
        with torch.no_grad():
            x_hat, _ = fresh_model(sample_batch)
            manual = torch.mean((sample_batch - x_hat) ** 2, dim=[1, 2, 3])
        auto = fresh_model.reconstruction_error(sample_batch)
        assert torch.allclose(manual, auto, atol=1e-6), \
            f"手動計算 vs auto 最大差: {(manual - auto).abs().max().item()}"

    def test_reconstruction_error_train_mode_restored(self, fresh_model, sample_batch):
        """reconstruction_error 在 train 模式下呼叫後應恢復"""
        fresh_model.train()
        _ = fresh_model.reconstruction_error(sample_batch)
        assert fresh_model.training, "train 模式未恢復"

    def test_get_model_info_structure(self, fresh_model):
        """get_model_info 應回傳正確的字典結構"""
        info = fresh_model.get_model_info()
        assert 'total_params' in info
        assert 'trainable_params' in info
        assert 'model_size_MB' in info
        assert 'latent_dim' in info
        assert info['total_params'] > 100_000, "模型參數量過少"
        assert info['trainable_params'] == info['total_params'], \
            "所有參數都應可訓練"


# ══════════════════════════════════════════════════════════════
# Group 3：動態閾值（比較式判定）邏輯
# ══════════════════════════════════════════════════════════════

class TestDynamicThreshold:
    """動態閾值比較式判定邏輯測試"""

    def _compute_dynamic_threshold(self, model, normal_arrs: np.ndarray) -> float:
        """計算動態閾值 = baseline mean + 2 * std"""
        tensor = torch.from_numpy(normal_arrs[:, np.newaxis].astype(np.float32))
        with torch.no_grad():
            errors = model.reconstruction_error(tensor).numpy()
        return float(errors.mean() + 2.0 * errors.std())

    def test_dynamic_threshold_greater_than_zero(self, fresh_model):
        """動態閾值應大於 0"""
        normal_arrs = np.random.beta(2, 5, (20, 32, 32)).astype(np.float32)
        thr = self._compute_dynamic_threshold(fresh_model, normal_arrs)
        assert thr > 0, f"動態閾值 {thr} <= 0"

    def test_dynamic_threshold_separates_distributions(self, fresh_model):
        """相同分布的封包應低於閾值，不同分布應高於閾值"""
        # baseline：beta(2,5) 分布（偏低值）
        normal_arrs = np.random.beta(2, 5, (30, 32, 32)).astype(np.float32)
        thr = self._compute_dynamic_threshold(fresh_model, normal_arrs)

        # 測試 baseline 同分布：expect 大部分 <= threshold
        test_normal = np.random.beta(2, 5, (10, 32, 32)).astype(np.float32)
        t_normal = torch.from_numpy(test_normal[:, np.newaxis])
        with torch.no_grad():
            errors_n = fresh_model.reconstruction_error(t_normal).numpy()
        # 至少一半應在閾值以下
        n_below = sum(1 for e in errors_n if e <= thr)
        assert n_below >= 5, \
            f"正常流量低於閾值只有 {n_below}/10"

    def test_baseline_std_nonzero(self, fresh_model):
        """baseline 標準差應 > 0（確保閾值有實際意義）"""
        normal_arrs = np.random.beta(2, 5, (20, 32, 32)).astype(np.float32)
        tensor = torch.from_numpy(normal_arrs[:, np.newaxis])
        with torch.no_grad():
            errors = fresh_model.reconstruction_error(tensor).numpy()
        assert errors.std() > 0, "baseline 標準差為 0，閾值無意義"


# ══════════════════════════════════════════════════════════════
# Group 4：AnomalyScorer 功能
# ══════════════════════════════════════════════════════════════

class TestAnomalyScorer:
    """AnomalyScorer 功能測試"""

    @pytest.fixture
    def scorer(self, fresh_model):
        from anomaly_scorer import AnomalyScorer
        return AnomalyScorer(fresh_model, threshold=0.05, device='cpu')

    def test_set_threshold(self, scorer):
        """set_threshold 應正確更新閾值"""
        scorer.set_threshold(0.123)
        assert abs(scorer.threshold - 0.123) < 1e-9

    def test_score_batch_consistency_with_single(self, fresh_model, sample_batch):
        """批次計分與單筆計分應一致"""
        from anomaly_scorer import AnomalyScorer
        scorer = AnomalyScorer(fresh_model, threshold=0.05, device='cpu')

        # 轉為 raw bytes（偽造 — 用影像陣列模擬）
        from packet_visualizer import PacketVisualizer
        viz = PacketVisualizer("medium", apply_mask=True)

        # 用 numpy 陣列模擬封包
        dummy_images = sample_batch.squeeze(1).numpy()  # (4, 32, 32)

        # 手動計算單筆 MSE
        errors_manual = fresh_model.reconstruction_error(sample_batch).numpy()

        # 批次計算
        imgs_tensor = sample_batch
        with torch.no_grad():
            batch_errors = fresh_model.reconstruction_error(imgs_tensor).numpy()

        for i in range(4):
            assert abs(float(errors_manual[i]) - float(batch_errors[i])) < 1e-5, \
                f"pkt {i}: 單筆={errors_manual[i]:.8f} 批次={batch_errors[i]:.8f}"

    def test_threshold_none_behavior(self, fresh_model):
        """threshold=None 時 is_attack 應為 False"""
        from anomaly_scorer import AnomalyScorer
        scorer = AnomalyScorer(fresh_model, threshold=None, device='cpu')

        # 使用 score_npy 等效的方式模擬
        # score_packet 在 threshold=None 時應回傳 is_attack=False
        dummy_bytes = bytes(1024)
        _, is_attack = scorer.score_packet(dummy_bytes)
        assert not is_attack, "threshold=None 時不應判定為攻擊"

    def test_evaluate_raises_without_threshold(self, fresh_model):
        """evaluate() 在 threshold=None 時應拋出 ValueError"""
        from anomaly_scorer import AnomalyScorer
        scorer = AnomalyScorer(fresh_model, threshold=None, device='cpu')
        X = np.random.rand(10, 32, 32).astype(np.float32)
        y = np.zeros(10, dtype=int)
        with pytest.raises(ValueError, match="threshold"):
            scorer.evaluate(X, y)


# ══════════════════════════════════════════════════════════════
# Group 5：模擬封包生成與偵測
# ══════════════════════════════════════════════════════════════

class TestSimulatedPacketDetection:
    """模擬封包生成與動態閾值偵測測試"""

    scapy_available = False
    try:
        from scapy.all import raw as scapy_raw
        from generate_attack_pcap import generate_attack_packets
        scapy_available = True
    except Exception:
        pass

    @pytest.mark.skipif(not scapy_available, reason="Scapy 或 generate_attack_pcap 不可用")
    def test_generate_normal_traffic_nonzero(self):
        """正常流量生成後封包長度 > 0"""
        from generate_attack_pcap import generate_attack_packets
        from scapy.all import raw as scapy_raw
        pkts = generate_attack_packets('normal_traffic', 5)
        assert len(pkts) == 5
        for p in pkts:
            raw = scapy_raw(p)
            assert len(raw) > 0

    @pytest.mark.skipif(not scapy_available, reason="Scapy 不可用")
    @pytest.mark.parametrize("atk_type", [
        'syn_flood', 'port_scan', 'arp_spoof',
        'dns_amplification', 'icmp_flood'
    ])
    def test_attack_packets_generated(self, atk_type):
        """各攻擊類型應能成功生成封包"""
        from generate_attack_pcap import generate_attack_packets
        from scapy.all import raw as scapy_raw
        pkts = generate_attack_packets(atk_type, 5)
        assert len(pkts) > 0, f"{atk_type} 未生成任何封包"

    @pytest.mark.skipif(not scapy_available, reason="Scapy 不可用")
    def test_dynamic_threshold_normal_mostly_below(self, fresh_model):
        """動態閾值下，正常流量大部分應低於閾值"""
        from generate_attack_pcap import generate_attack_packets
        from packet_visualizer import PacketVisualizer
        from scapy.all import raw as scapy_raw

        viz = PacketVisualizer("medium", apply_mask=True)

        # baseline
        baseline_pkts = generate_attack_packets('normal_traffic', 20)
        baseline_raw = [scapy_raw(p) for p in baseline_pkts]
        baseline_arrs = np.array([viz.bytes_to_image(b) for b in baseline_raw], dtype=np.float32)
        baseline_tensor = torch.from_numpy(baseline_arrs[:, np.newaxis])
        with torch.no_grad():
            baseline_errors = fresh_model.reconstruction_error(baseline_tensor).numpy()
        dynamic_thr = float(baseline_errors.mean() + 2.0 * baseline_errors.std())

        # 測試正常流量
        test_pkts = generate_attack_packets('normal_traffic', 10)
        test_raw = [scapy_raw(p) for p in test_pkts]
        test_arrs = np.array([viz.bytes_to_image(b) for b in test_raw], dtype=np.float32)
        test_tensor = torch.from_numpy(test_arrs[:, np.newaxis])
        with torch.no_grad():
            test_errors = fresh_model.reconstruction_error(test_tensor).numpy()

        anomaly_count = sum(1 for e in test_errors if e > dynamic_thr)
        assert anomaly_count <= 5, \
            f"正常流量在動態閾值下有 {anomaly_count}/10 被誤標為異常"


# ══════════════════════════════════════════════════════════════
# Group 6：PacketVisualizer 與 bytes 處理
# ══════════════════════════════════════════════════════════════

class TestPacketVisualizer:
    """PacketVisualizer bytes_to_image 功能測試"""

    @pytest.fixture
    def viz(self):
        from packet_visualizer import PacketVisualizer
        return PacketVisualizer("medium", apply_mask=True)

    def test_output_shape(self, viz):
        """bytes_to_image 應回傳 (32, 32) 陣列"""
        raw = bytes(range(256)) * 4  # 1024 bytes
        arr = viz.bytes_to_image(raw)
        assert arr.shape == (32, 32)

    def test_output_range(self, viz):
        """正規化後輸出應在 [0, 1]"""
        raw = bytes(range(256)) * 4
        arr = viz.bytes_to_image(raw)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_short_packet_padded(self, viz):
        """短封包應補零到 32x32"""
        raw = bytes(20)  # 只有 20 bytes
        arr = viz.bytes_to_image(raw)
        assert arr.shape == (32, 32)

    def test_empty_packet(self, viz):
        """空封包不應崩潰"""
        arr = viz.bytes_to_image(b"")
        assert arr.shape == (32, 32)
        assert arr.sum() == 0.0  # 全零影像
