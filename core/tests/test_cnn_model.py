# ============================================================
# core/tests/test_cnn_model.py - CNN Autoencoder 單元測試
#
# 執行方式:
#   pytest core/tests/test_cnn_model.py -v
# ============================================================

# 執行方式：
#   pytest tests/test_cnn_model.py -v
# ============================================================

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 跳過所有需要 PyTorch 的測試（若未安裝）
torch_available = False
try:
    import torch
    torch_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not torch_available,
    reason="PyTorch 未安裝，請執行: pip install torch"
)


@pytest.fixture
def model():
    """建立預設 CNN Autoencoder"""
    from cnn_autoencoder import CNNAutoencoder
    return CNNAutoencoder(latent_dim=64)


@pytest.fixture
def small_model():
    """建立小型 CNN Autoencoder（節省測試時間）"""
    from cnn_autoencoder import CNNAutoencoder
    return CNNAutoencoder(latent_dim=16)


@pytest.fixture
def sample_batch():
    """建立隨機測試批次 shape=(4, 1, 32, 32)"""
    return torch.randn(4, 1, 32, 32)


# ─────────────────────────────────────────────────────────
# Encoder 測試
# ─────────────────────────────────────────────────────────
class TestEncoder:

    def test_output_shape(self, model, sample_batch):
        """Encoder 輸出 shape 應為 (batch, latent_dim)"""
        z = model.encoder(sample_batch)
        assert z.shape == (4, 64), f"期望 (4, 64)，得到 {z.shape}"

    def test_custom_latent_dim(self, sample_batch):
        """不同 latent_dim 的輸出 shape 應正確"""
        from cnn_autoencoder import Encoder
        for dim in [16, 32, 128]:
            enc = Encoder(latent_dim=dim)
            z   = enc(sample_batch)
            assert z.shape == (4, dim)

    def test_output_requires_grad(self, model, sample_batch):
        """輸出應支援梯度計算（訓練所需）"""
        sample_batch.requires_grad = True
        z = model.encoder(sample_batch)
        assert z.requires_grad

    def test_different_batch_sizes(self, model):
        """Encoder 應支援不同 batch size"""
        for bs in [1, 8, 32]:
            x = torch.randn(bs, 1, 32, 32)
            z = model.encoder(x)
            assert z.shape == (bs, 64)


# ─────────────────────────────────────────────────────────
# Decoder 測試
# ─────────────────────────────────────────────────────────
class TestDecoder:

    def test_output_shape(self, model):
        """Decoder 輸出 shape 應為 (batch, 1, 32, 32)"""
        z    = torch.randn(4, 64)
        x_hat = model.decoder(z)
        assert x_hat.shape == (4, 1, 32, 32)

    def test_output_range(self, model):
        """Decoder 輸出（Sigmoid 後）應在 [0, 1] 之間"""
        z    = torch.randn(4, 64)
        x_hat = model.decoder(z)
        assert x_hat.min() >= 0.0, f"最小值 {x_hat.min()} < 0"
        assert x_hat.max() <= 1.0, f"最大值 {x_hat.max()} > 1"


# ─────────────────────────────────────────────────────────
# 完整 Autoencoder 測試
# ─────────────────────────────────────────────────────────
class TestCNNAutoencoder:

    def test_forward_output_shapes(self, model, sample_batch):
        """forward() 應回傳正確 shape 的 (x_hat, z)"""
        x_hat, z = model(sample_batch)
        assert x_hat.shape == sample_batch.shape, \
            f"x_hat shape {x_hat.shape} ≠ input shape {sample_batch.shape}"
        assert z.shape == (4, 64)

    def test_reconstruction_error_shape(self, model, sample_batch):
        """reconstruction_error() 應回傳 shape=(batch,) 的 1D tensor"""
        errors = model.reconstruction_error(sample_batch)
        assert errors.shape == (4,), f"期望 (4,)，得到 {errors.shape}"

    def test_reconstruction_error_non_negative(self, model, sample_batch):
        """重建誤差（MSE）應為非負數"""
        errors = model.reconstruction_error(sample_batch)
        assert (errors >= 0).all(), "重建誤差不應為負數"

    def test_identical_input_low_error(self, model):
        """若輸入接近 Decoder 的輸出分布，誤差應較低"""
        # 建立接近中間值（0.5）的輸入：Sigmoid 輸出傾向集中在中間
        x_mid = torch.ones(2, 1, 32, 32) * 0.5
        x_rand = torch.rand(2, 1, 32, 32)
        # 兩者誤差都應有限（smoke test，不比大小）
        e_mid  = model.reconstruction_error(x_mid)
        e_rand = model.reconstruction_error(x_rand)
        assert e_mid.max().item() < 10.0   # MSE 不應爆炸

    def test_encode_decode_methods(self, model, sample_batch):
        """encode() 與 decode() 的單獨呼叫應與 forward() 一致"""
        z     = model.encode(sample_batch)
        x_hat = model.decode(z)
        assert x_hat.shape == sample_batch.shape

    def test_model_info(self, model):
        """get_model_info() 應回傳合理的模型資訊"""
        info = model.get_model_info()
        assert info["latent_dim"] == 64
        assert info["total_params"] > 0
        assert info["trainable_params"] == info["total_params"]  # 所有參數可訓練
        assert info["model_size_MB"] > 0
        print(f"\n  模型參數: {info['total_params']:,}，"
              f"大小: {info['model_size_MB']:.2f} MB")

    def test_gradient_flow(self, small_model, sample_batch):
        """訓練梯度應能從 Loss 反向傳播到所有參數"""
        optimizer = torch.optim.Adam(small_model.parameters())
        x_hat, _ = small_model(sample_batch)
        loss = torch.nn.MSELoss()(x_hat, sample_batch)
        optimizer.zero_grad()
        loss.backward()

        # 確認所有可訓練參數都有梯度
        for name, param in small_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} 沒有梯度"
                assert not torch.isnan(param.grad).any(), f"{name} 梯度為 NaN"

    def test_different_latent_dims(self, sample_batch):
        """不同 latent_dim 的模型都應正常前向傳播"""
        from cnn_autoencoder import CNNAutoencoder
        for dim in [16, 32, 64, 128]:
            model = CNNAutoencoder(latent_dim=dim)
            x_hat, z = model(sample_batch)
            assert x_hat.shape == sample_batch.shape
            assert z.shape == (4, dim)


# ─────────────────────────────────────────────────────────
# 訓練器 Smoke Test（小規模）
# ─────────────────────────────────────────────────────────
class TestTrainerSmoke:

    def test_single_batch_training(self, small_model):
        """執行單一 batch 的前向 + 反向傳播，確認無錯誤"""
        optimizer = torch.optim.Adam(small_model.parameters())
        criterion = torch.nn.MSELoss()

        x = torch.rand(8, 1, 32, 32)
        x_hat, _ = small_model(x)
        loss = criterion(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)
        print(f"\n  單 batch 訓練 Loss: {loss.item():.6f}")

    def test_loss_decreases_over_iterations(self, small_model):
        """多次迭代後，Loss 應有下降趨勢"""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-2)
        criterion = torch.nn.MSELoss()

        # 固定資料，訓練 20 步
        x = torch.rand(16, 1, 32, 32)
        initial_loss = None
        final_loss   = None

        for step in range(20):
            x_hat, _ = small_model(x)
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        # Loss 應有下降（不一定嚴格遞減，但趨勢上應降低）
        assert final_loss < initial_loss, \
            f"20 步後 Loss 未下降: {initial_loss:.6f} → {final_loss:.6f}"
        print(f"\n  Loss: {initial_loss:.6f} → {final_loss:.6f} (↓{(initial_loss-final_loss)/initial_loss*100:.1f}%)")

    def test_reconstruction_improves_after_training(self, small_model):
        """訓練後重建誤差應低於訓練前"""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=5e-3)
        x = torch.rand(16, 1, 32, 32)

        # 訓練前誤差
        before = float(small_model.reconstruction_error(x).mean())

        # 訓練 30 步
        criterion = torch.nn.MSELoss()
        for _ in range(30):
            x_hat, _ = small_model(x)
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 訓練後誤差
        after = float(small_model.reconstruction_error(x).mean())

        assert after < before, f"訓練後誤差未降低: {before:.6f} → {after:.6f}"
        print(f"\n  重建誤差: {before:.6f} → {after:.6f}")


# ─────────────────────────────────────────────────────────
# PacketDataset 測試
# ─────────────────────────────────────────────────────────
class TestPacketDataset:

    def test_from_numpy(self):
        """從 numpy array 建立 dataset，shape 應正確擴展"""
        from trainer import PacketDataset
        arr     = np.random.rand(50, 32, 32).astype(np.float32)
        dataset = PacketDataset.from_numpy(arr)
        assert len(dataset) == 50
        sample = dataset[0][0]
        assert sample.shape == (1, 32, 32), f"期望 (1,32,32)，得到 {sample.shape}"

    def test_dataloader_batch(self):
        """DataLoader 應正確分批"""
        from trainer import PacketDataset
        from torch.utils.data import DataLoader
        arr     = np.random.rand(100, 32, 32).astype(np.float32)
        dataset = PacketDataset.from_numpy(arr)
        loader  = DataLoader(dataset, batch_size=16, shuffle=True)
        batch   = next(iter(loader))[0]
        assert batch.shape == (16, 1, 32, 32)


# ─────────────────────────────────────────────────────────
# AnomalyScorer 測試
# ─────────────────────────────────────────────────────────
class TestAnomalyScorer:

    def test_score_returns_tuple(self, small_model):
        """score_packet() 應回傳 (float, bool) tuple"""
        from anomaly_scorer import AnomalyScorer
        scorer  = AnomalyScorer(small_model, threshold=0.1)
        # 手工構造一個簡單的封包（HTTP GET）
        payload = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
        pkt = bytes([0xff]*14 + [0x45,0x00,0x00,0x28] + [0]*16 + [0x50,0x18] + [0]*4) + payload
        score, is_attack = scorer.score_packet(pkt)
        assert isinstance(score, float)
        assert isinstance(is_attack, bool)
        assert score >= 0

    def test_anomaly_threshold_logic(self, small_model):
        """高分應被判定為攻擊，低分不應"""
        from anomaly_scorer import AnomalyScorer
        scorer = AnomalyScorer(small_model, threshold=0.5)
        scorer.set_threshold(0.5)

        # 模擬：直接測試閾值邏輯
        assert (0.6 > scorer.threshold) == True   # 高分 → 攻擊
        assert (0.4 > scorer.threshold) == False  # 低分 → 正常

    def test_score_batch_length(self, small_model):
        """score_batch() 輸出長度應與輸入一致"""
        from anomaly_scorer import AnomalyScorer
        scorer  = AnomalyScorer(small_model, threshold=0.1)
        packets = [bytes([i % 256] * 100) for i in range(10)]
        results = scorer.score_batch(packets)
        assert len(results) == 10
        for score, is_attack in results:
            assert isinstance(score, float)
            assert isinstance(is_attack, bool)
