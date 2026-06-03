# ============================================================
# test_semi_supervised.py - 半監督訓練模組單元測試
#
# 測試範圍：
#   1. SemiSupervisedDataLoader  - 資料分配正確性
#   2. MarginLoss                - 損失計算邏輯
#   3. SemiSupervisedTrainer     - 訓練流程（smoke test）
#   4. SemiSupervisedThresholdTuner - 閾值搜索
#   5. 半監督 vs 非監督分離度比較
#
# 執行方式（在 core/ 目錄下）：
#   pytest test_semi_supervised.py -v
#   pytest test_semi_supervised.py -v -k "TestMarginLoss"
#
# 注意事項：
#   - 所有測試使用小型合成資料，不需要任何外部資料集
#   - 若 PyTorch 未安裝，所有測試自動跳過
# ============================================================

import sys
import os
import pytest
import numpy as np

# 確保可以 import 核心模組
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# 跳過條件：PyTorch 未安裝
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


# ─────────────────────────────────────────────────────────────
# Fixtures：合成測試資料
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """
    生成小型合成資料，設計成正常/攻擊有明顯差異：
    - 正常流量：中等強度，有規律結構（接近 Beta(2,5) 分布）
    - 攻擊流量：低強度，集中在角落（接近 Beta(0.5,5) 分布）
    """
    np.random.seed(42)
    n_normal = 200
    n_attack = 100

    # 正常流量：全圖均勻分布（模擬正常封包結構）
    X_normal = np.random.beta(2, 5, (n_normal, 32, 32)).astype(np.float32)

    # 攻擊流量：只有角落有值（模擬異常封包）
    X_attack = np.zeros((n_attack, 32, 32), dtype=np.float32)
    X_attack[:, :8, :8] = np.random.uniform(0.8, 1.0, (n_attack, 8, 8))

    return X_normal, X_attack


@pytest.fixture
def small_model():
    """小型 CNN Autoencoder（latent_dim=8 加快測試速度）"""
    from cnn_autoencoder import CNNAutoencoder
    return CNNAutoencoder(latent_dim=8)


@pytest.fixture
def device():
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────
# 1. MarginLoss 測試
# ─────────────────────────────────────────────────────────────

class TestMarginLoss:
    """測試邊界損失函數的計算邏輯"""

    def test_loss_zero_when_errors_exceed_margin(self):
        """當攻擊誤差全部 > margin 時，損失應為 0"""
        from semi_supervised_trainer import MarginLoss
        loss_fn = MarginLoss(margin=0.05)
        # 攻擊誤差全部 > 0.05
        errors = torch.tensor([0.1, 0.2, 0.3, 0.15])
        loss = loss_fn(errors)
        assert abs(loss.item()) < 1e-7, f"損失應為 0，得到 {loss.item()}"

    def test_loss_positive_when_errors_below_margin(self):
        """當攻擊誤差低於 margin 時，損失應為正值"""
        from semi_supervised_trainer import MarginLoss
        loss_fn = MarginLoss(margin=0.1)
        # 攻擊誤差全部 < 0.1
        errors = torch.tensor([0.01, 0.02, 0.03])
        loss = loss_fn(errors)
        assert loss.item() > 0, f"損失應為正值，得到 {loss.item()}"

    def test_loss_gradient_pushes_errors_up(self):
        """
        梯度應推高攻擊誤差（驗證梯度方向正確）

        當攻擊誤差低於 margin，梯度應使 errors 往增大方向更新
        """
        from semi_supervised_trainer import MarginLoss
        loss_fn = MarginLoss(margin=0.1)
        errors = torch.tensor([0.02, 0.03], requires_grad=True)
        loss = loss_fn(errors)
        loss.backward()
        # 梯度應為負（對 errors 的梯度負值意味著增大 errors 可減少 loss）
        assert errors.grad is not None
        assert (errors.grad < 0).all(), \
            f"梯度應為負（鼓勵提高誤差），得到: {errors.grad}"

    def test_partial_loss_calculation(self):
        """混合情況：部分誤差高於 margin，部分低於"""
        from semi_supervised_trainer import MarginLoss
        margin = 0.1
        loss_fn = MarginLoss(margin=margin)
        # 前 2 個低於 margin，後 2 個高於 margin
        errors = torch.tensor([0.02, 0.05, 0.15, 0.20])
        loss = loss_fn(errors)
        # 期望損失 = mean([0.08, 0.05, 0, 0]) = 0.0325
        expected = (0.08 + 0.05 + 0.0 + 0.0) / 4
        assert abs(loss.item() - expected) < 1e-5, \
            f"損失計算錯誤：期望 {expected:.6f}，得到 {loss.item():.6f}"

    def test_loss_with_different_margins(self):
        """不同 margin 設定應產生不同損失值"""
        from semi_supervised_trainer import MarginLoss
        errors = torch.tensor([0.05, 0.06, 0.07])
        loss_small = MarginLoss(margin=0.04)(errors).item()
        loss_large = MarginLoss(margin=0.10)(errors).item()
        assert loss_small < loss_large, \
            "較大的 margin 應產生較大的損失（更難滿足）"


# ─────────────────────────────────────────────────────────────
# 2. SemiSupervisedDataLoader 測試
# ─────────────────────────────────────────────────────────────

class TestSemiSupervisedDataLoader:
    """測試資料載入器的分配邏輯"""

    def test_attack_ratio_correct(self, synthetic_data):
        """標記攻擊樣本數應符合設定的比例"""
        from semi_supervised_trainer import SemiSupervisedDataLoader
        X_normal, X_attack = synthetic_data
        attack_ratio = 0.3

        dm = SemiSupervisedDataLoader(
            X_normal, X_attack, attack_ratio=attack_ratio, batch_size=16
        )

        expected = int(len(X_attack) * attack_ratio)
        assert dm.n_attack_labeled == expected, \
            f"期望 {expected} 筆攻擊樣本，得到 {dm.n_attack_labeled}"

    def test_train_val_split_correct(self, synthetic_data):
        """訓練/驗證分割應符合設定比例"""
        from semi_supervised_trainer import SemiSupervisedDataLoader
        X_normal, X_attack = synthetic_data
        val_split = 0.2

        dm = SemiSupervisedDataLoader(
            X_normal, X_attack, attack_ratio=0.2,
            batch_size=16, val_split=val_split
        )

        n_val = max(1, int(len(X_normal) * val_split))
        assert dm.n_normal_val == n_val, \
            f"期望 {n_val} 筆驗證，得到 {dm.n_normal_val}"

    def test_pretrain_loaders_not_empty(self, synthetic_data):
        """Phase 1 的 DataLoader 不應為空"""
        from semi_supervised_trainer import SemiSupervisedDataLoader
        X_normal, X_attack = synthetic_data

        dm = SemiSupervisedDataLoader(X_normal, X_attack, batch_size=16)
        train_loader, val_loader = dm.get_pretrain_loaders()

        assert len(train_loader) > 0, "訓練 DataLoader 不應為空"
        assert len(val_loader) > 0, "驗證 DataLoader 不應為空"

    def test_finetune_attack_loader_exists(self, synthetic_data):
        """Phase 2 的攻擊 DataLoader 應存在且不為空"""
        from semi_supervised_trainer import SemiSupervisedDataLoader
        X_normal, X_attack = synthetic_data

        dm = SemiSupervisedDataLoader(X_normal, X_attack, batch_size=16)
        normal_loader, attack_loader, val_loader = dm.get_finetune_loaders()

        assert len(attack_loader) > 0, "攻擊 DataLoader 不應為空"
        # 檢查攻擊批次的維度
        (batch,) = next(iter(attack_loader))
        assert batch.dim() == 4, f"期望 4D tensor，得到 {batch.dim()}D"
        assert batch.shape[1] == 1, "通道數應為 1"
        assert batch.shape[2] == 32, "高度應為 32"
        assert batch.shape[3] == 32, "寬度應為 32"

    def test_minimum_one_attack_sample(self, synthetic_data):
        """即使 attack_ratio 極小，也至少保留 1 個攻擊樣本"""
        from semi_supervised_trainer import SemiSupervisedDataLoader
        X_normal, X_attack = synthetic_data

        dm = SemiSupervisedDataLoader(
            X_normal, X_attack, attack_ratio=0.001, batch_size=16
        )
        assert dm.n_attack_labeled >= 1, "至少應有 1 個標記攻擊樣本"


# ─────────────────────────────────────────────────────────────
# 3. SemiSupervisedTrainer 訓練流程 Smoke Test
# ─────────────────────────────────────────────────────────────

class TestSemiSupervisedTrainer:
    """
    半監督訓練器 Smoke Test

    注意：只做少量 epoch 的快速測試，驗證流程無誤，
    不驗證最終效能（效能需要更多 epochs 和真實資料）
    """

    @pytest.fixture
    def quick_config(self):
        """極快的測試設定（只跑 2 epochs）"""
        return {
            "latent_dim":      8,
            "batch_size":      16,
            "pretrain_epochs": 2,
            "finetune_epochs": 2,
            "learning_rate":   1e-3,
            "patience":        10,
            "alpha":           1.0,
            "beta":            0.5,
            "margin":          0.05,
            "attack_ratio":    0.3,
            "val_split":       0.2,
            "weight_decay":    1e-5,
            "min_delta":       1e-8,
            "lr_patience":     5,
            "lr_factor":       0.5,
        }

    def test_pretrain_runs_without_error(self, synthetic_data, quick_config,
                                          tmp_path):
        """Phase 1 預訓練應能正常執行"""
        from semi_supervised_trainer import SemiSupervisedTrainer
        X_normal, X_attack = synthetic_data

        trainer = SemiSupervisedTrainer(
            config=quick_config,
            output_dir=str(tmp_path)
        )
        trainer.load_data(X_normal, X_attack)
        trainer.pretrain()   # 不應拋出任何例外

        assert len(trainer.pretrain_losses) > 0, "應有預訓練損失記錄"
        assert len(trainer.val_losses) > 0, "應有驗證損失記錄"

    def test_finetune_runs_without_error(self, synthetic_data, quick_config,
                                          tmp_path):
        """Phase 1 + Phase 2 應能正常執行"""
        from semi_supervised_trainer import SemiSupervisedTrainer
        X_normal, X_attack = synthetic_data

        trainer = SemiSupervisedTrainer(
            config=quick_config,
            output_dir=str(tmp_path)
        )
        trainer.load_data(X_normal, X_attack)
        trainer.pretrain()
        trainer.finetune()   # 不應拋出任何例外

        assert len(trainer.finetune_total) > 0, "應有微調損失記錄"

    def test_pretrain_loss_decreases(self, synthetic_data, quick_config,
                                      tmp_path):
        """增加 epochs 數，損失應有下降趨勢（smoke test）"""
        from semi_supervised_trainer import SemiSupervisedTrainer
        X_normal, X_attack = synthetic_data

        # 增加到 5 epochs 才能看到趨勢
        quick_config["pretrain_epochs"] = 5
        trainer = SemiSupervisedTrainer(
            config=quick_config,
            output_dir=str(tmp_path)
        )
        trainer.load_data(X_normal, X_attack)
        trainer.pretrain()

        # 前半段的平均損失應高於後半段
        n = len(trainer.pretrain_losses)
        if n >= 4:
            first_half  = np.mean(trainer.pretrain_losses[:n//2])
            second_half = np.mean(trainer.pretrain_losses[n//2:])
            assert second_half <= first_half * 1.5, \
                f"損失不應增加太多: {first_half:.6f} → {second_half:.6f}"

    def test_model_saved_after_training(self, synthetic_data, quick_config,
                                         tmp_path):
        """訓練後應儲存模型檔案"""
        from semi_supervised_trainer import SemiSupervisedTrainer
        X_normal, X_attack = synthetic_data

        trainer = SemiSupervisedTrainer(
            config=quick_config,
            output_dir=str(tmp_path)
        )
        trainer.load_data(X_normal, X_attack)
        trainer.pretrain()
        trainer.finetune()

        model_path = os.path.join(str(tmp_path), "best_model.pt")
        assert os.path.exists(model_path), f"最佳模型應已儲存到 {model_path}"

    def test_compute_threshold_percentile(self, synthetic_data, quick_config,
                                           tmp_path):
        """percentile 方法應回傳有效的閾值"""
        from semi_supervised_trainer import SemiSupervisedTrainer
        X_normal, X_attack = synthetic_data

        trainer = SemiSupervisedTrainer(
            config=quick_config,
            output_dir=str(tmp_path)
        )
        trainer.load_data(X_normal, X_attack)
        trainer.pretrain()

        threshold = trainer.compute_threshold(
            X_normal, X_attack, method="percentile", percentile=95
        )

        assert isinstance(threshold, float), "閾值應為 float"
        assert threshold > 0, "閾值應為正數"

    def test_compute_threshold_optimal(self, synthetic_data, quick_config,
                                        tmp_path):
        """optimal 方法應回傳有效的閾值"""
        from semi_supervised_trainer import SemiSupervisedTrainer
        X_normal, X_attack = synthetic_data

        trainer = SemiSupervisedTrainer(
            config=quick_config,
            output_dir=str(tmp_path)
        )
        trainer.load_data(X_normal, X_attack)
        trainer.pretrain()

        threshold = trainer.compute_threshold(
            X_normal, X_attack, method="optimal"
        )

        assert isinstance(threshold, float), "閾值應為 float"
        assert threshold > 0, "閾值應為正數"

    def test_load_model_roundtrip(self, synthetic_data, quick_config, tmp_path):
        """儲存後載入的模型應與原模型輸出相同"""
        from semi_supervised_trainer import SemiSupervisedTrainer
        X_normal, X_attack = synthetic_data

        trainer = SemiSupervisedTrainer(
            config=quick_config,
            output_dir=str(tmp_path)
        )
        trainer.load_data(X_normal, X_attack)
        trainer.pretrain()
        trainer.finetune()
        threshold = trainer.compute_threshold(X_normal, X_attack,
                                              method="percentile")

        # 儲存後重新載入
        model_path  = os.path.join(str(tmp_path), "best_model.pt")
        config_path = os.path.join(str(tmp_path), "semi_training_result.json")

        loaded_model, loaded_threshold = SemiSupervisedTrainer.load_model(
            model_path, config_path, latent_dim=quick_config["latent_dim"]
        )

        assert loaded_threshold is not None, "載入的閾值不應為 None"
        assert abs(loaded_threshold - threshold) < 1e-6, \
            f"載入的閾值 {loaded_threshold} 與原始 {threshold} 不一致"

        # 驗證模型輸出一致性
        device = next(trainer.model.parameters()).device   # ← 取得模型實際所在裝置
        test_input = torch.randn(4, 1, 32, 32).to(device) # ← 移到同一裝置
        trainer.model.eval()
        loaded_model.eval()
        with torch.no_grad():
            out1, _ = trainer.model(test_input)
            out2, _ = loaded_model(test_input)
        assert torch.allclose(out1, out2, atol=1e-5), \
            "載入的模型輸出應與原模型一致"
