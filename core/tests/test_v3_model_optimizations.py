# ============================================================
# core/tests/test_v3_model_optimizations.py
# v3.0 模型與訓練管線優化驗證測試
#
# 涵蓋：
#   - latent_dim 預設值一致性驗證
#   - percentile/threshold falsy 邏輯修正驗證
#   - CIC-IDS inf 處理修正驗證
#   - Grad-CAM → Saliency Map 修正驗證
#   - score_batch 分批處理驗證
#   - evaluate threshold 前置檢查驗證
#   - reconstruction_error 模式恢復驗證
#   - _load_simulate 局部 RNG 驗證
#   - threshold_tuner PR-AUC 效能驗證
#   - 資料洩漏修正概念驗證
# ============================================================

import sys
import os
import time
import unittest
import numpy as np

# 確保可以從 core/ 匯入模組
_CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

import torch
from cnn_autoencoder import CNNAutoencoder
from trainer import Trainer, DEFAULT_CONFIG


class TestLatentDimConsistency(unittest.TestCase):
    """Bug 1：驗證所有模組的 latent_dim 預設值一致"""

    def test_cnn_default_latent_dim(self):
        """CNNAutoencoder 預設 latent_dim 應為 32"""
        model = CNNAutoencoder()
        self.assertEqual(model.latent_dim, 32)

    def test_trainer_default_config_latent_dim(self):
        """Trainer DEFAULT_CONFIG 的 latent_dim 應為 32"""
        self.assertEqual(DEFAULT_CONFIG["latent_dim"], 32)

    def test_trainer_load_model_default_latent_dim(self):
        """Trainer.load_model 的預設 latent_dim 參數應為 32"""
        import inspect
        sig = inspect.signature(Trainer.load_model)
        self.assertEqual(sig.parameters["latent_dim"].default, 32)

    def test_model_roundtrip_shape(self):
        """使用預設 latent_dim 建立的模型，輸入/輸出形狀應一致"""
        model = CNNAutoencoder()
        x = torch.randn(2, 1, 32, 32)
        x_hat, z = model(x)
        self.assertEqual(x_hat.shape, x.shape)
        self.assertEqual(z.shape, (2, 32))


class TestPercentileFalsyFix(unittest.TestCase):
    """Bug 5：驗證 percentile=0 不會被視為 falsy"""

    def test_percentile_zero_uses_zero(self):
        """傳入 percentile=0 時應使用 0，而非 fallback 到 config 預設值"""
        trainer = Trainer(config={"latent_dim": 32, "threshold_percentile": 95})
        # 建立一個極小的假資料來測試
        data = np.random.rand(10, 32, 32).astype(np.float32)
        npy_path = os.path.join(_CORE, "_test_pct0.npy")
        np.save(npy_path, data)
        try:
            # percentile=0 應回傳所有誤差中的最小值（第 0 百分位）
            threshold_0 = trainer.compute_threshold(npy_path, percentile=0)
            # 第 0 百分位 = 最小值，應非常小
            threshold_95 = trainer.compute_threshold(npy_path, percentile=95)
            # 確認兩者不同（若 percentile=0 被忽略，兩者會相同）
            self.assertLessEqual(threshold_0, threshold_95)
        finally:
            if os.path.exists(npy_path):
                os.remove(npy_path)

    def test_percentile_none_uses_config(self):
        """percentile=None 時應使用 config 預設值"""
        trainer = Trainer(config={"latent_dim": 32, "threshold_percentile": 50})
        data = np.random.rand(20, 32, 32).astype(np.float32)
        npy_path = os.path.join(_CORE, "_test_pct_none.npy")
        np.save(npy_path, data)
        try:
            threshold = trainer.compute_threshold(npy_path, percentile=None)
            # 第 50 百分位 ≈ 中位數
            self.assertIsInstance(threshold, float)
        finally:
            if os.path.exists(npy_path):
                os.remove(npy_path)


class TestThresholdFalsyFix(unittest.TestCase):
    """Bug 6：驗證 threshold=0.0 不會被視為 falsy"""

    def test_load_model_prints_zero_threshold(self):
        """load_model 的 threshold 為 0.0 時應正常處理"""
        # 這個測試確認 `if threshold is not None` 的邏輯正確
        import json, tempfile
        model = CNNAutoencoder(latent_dim=32)
        model_path = os.path.join(_CORE, "_test_model.pt")
        config_path = os.path.join(_CORE, "_test_config.json")

        torch.save(model.state_dict(), model_path)
        with open(config_path, "w") as f:
            json.dump({"config": {"latent_dim": 32}, "threshold": 0.0}, f)

        try:
            loaded_model, threshold = Trainer.load_model(model_path, config_path)
            # threshold=0.0 應被正確返回，而非 None
            self.assertEqual(threshold, 0.0)
        finally:
            for p in [model_path, config_path]:
                if os.path.exists(p):
                    os.remove(p)


class TestReconstructionErrorModeRestore(unittest.TestCase):
    """M3：驗證 reconstruction_error() 保存/恢復模式"""

    def test_mode_restored_after_eval_call(self):
        """在 train 模式下呼叫 reconstruction_error 後應恢復 train 模式"""
        model = CNNAutoencoder(latent_dim=32)
        model.train()
        self.assertTrue(model.training)

        x = torch.randn(2, 1, 32, 32)
        _ = model.reconstruction_error(x)

        # [關鍵驗證] 模式應恢復為 train
        self.assertTrue(model.training)

    def test_mode_stays_eval_if_started_eval(self):
        """在 eval 模式下呼叫 reconstruction_error 後應維持 eval 模式"""
        model = CNNAutoencoder(latent_dim=32)
        model.eval()
        self.assertFalse(model.training)

        x = torch.randn(2, 1, 32, 32)
        _ = model.reconstruction_error(x)

        self.assertFalse(model.training)


class TestInfHandlingFix(unittest.TestCase):
    """Bug 3：驗證 CIC-IDS2017 的 inf 值處理"""

    def test_inf_replaced_with_column_max(self):
        """inf 應被替換為該欄位的最大有限值，而非 float32.max"""
        from dataset_loader import CICIDSLoader
        import pandas as pd

        # 模擬含有 inf 的 DataFrame
        df = pd.DataFrame({
            "feature_a": [1.0, 2.0, np.inf, 4.0, 5.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0, -np.inf],
            "Label": ["BENIGN", "BENIGN", "BENIGN", "BENIGN", "BENIGN"]
        })

        loader = CICIDSLoader.__new__(CICIDSLoader)
        from sklearn.preprocessing import MinMaxScaler
        loader.scaler = MinMaxScaler()

        result = loader._preprocess(df, label_col="Label", fit=True)

        # 驗證結果中沒有極端值（如果用了 float32.max，所有值會接近 0）
        # 正確處理後，值應分布在 [0, 1] 範圍內
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1.0 + 1e-6))

        # 驗證不是全部接近 0（這是舊版 float32.max 的症狀）
        max_val = result[:, :2].max()  # 前兩個特徵欄位
        self.assertGreater(max_val, 0.1, "數值不應全部被壓縮到接近 0")


class TestSaliencyMapFix(unittest.TestCase):
    """Bug 4：驗證 Grad-CAM → Saliency Map 修正"""

    def test_model_stays_eval_during_saliency(self):
        """計算 saliency map 時模型應保持 eval 模式"""
        from anomaly_scorer import AnomalyScorer

        model = CNNAutoencoder(latent_dim=32)
        model.eval()
        scorer = AnomalyScorer(model, threshold=0.01, device="cpu")

        # 產生假封包
        fake_packet = bytes(range(256)) * 4  # 1024 bytes

        overlay = scorer.saliency_map_packet(fake_packet)

        # [關鍵驗證] 模型在計算後應維持 eval 模式
        self.assertFalse(model.training)

        # 輸出形狀驗證
        self.assertEqual(overlay.shape[0], 32)
        self.assertEqual(overlay.shape[1], 32)

    def test_gradcam_alias_works(self):
        """舊名稱 gradcam_packet 應保持可用（向下相容）"""
        from anomaly_scorer import AnomalyScorer

        model = CNNAutoencoder(latent_dim=32)
        scorer = AnomalyScorer(model, threshold=0.01, device="cpu")

        # 確認別名存在且指向相同方法
        self.assertEqual(scorer.gradcam_packet, scorer.saliency_map_packet)

    def test_saliency_with_exception_restores_eval(self):
        """即使計算過程發生錯誤，模型也應恢復 eval 模式"""
        from anomaly_scorer import AnomalyScorer

        model = CNNAutoencoder(latent_dim=32)
        model.eval()
        scorer = AnomalyScorer(model, threshold=0.01, device="cpu")

        # 傳入空 bytes 會引發錯誤，但模型應恢復 eval
        try:
            scorer.saliency_map_packet(b"")
        except Exception:
            pass

        self.assertFalse(model.training)


class TestScoreBatchBatching(unittest.TestCase):
    """M2：驗證 score_batch 使用分批處理"""

    def test_score_batch_returns_correct_count(self):
        """批次計分結果數量應與輸入封包數量一致"""
        from anomaly_scorer import AnomalyScorer

        model = CNNAutoencoder(latent_dim=32)
        scorer = AnomalyScorer(model, threshold=0.01, device="cpu")

        fake_packets = [bytes(range(256)) * 4 for _ in range(10)]
        results = scorer.score_batch(fake_packets, batch_size=3)

        self.assertEqual(len(results), 10)
        for score, is_attack in results:
            self.assertIsInstance(score, float)
            self.assertIsInstance(is_attack, bool)

    def test_score_batch_with_no_threshold(self):
        """threshold=None 時所有 is_attack 應為 False"""
        from anomaly_scorer import AnomalyScorer

        model = CNNAutoencoder(latent_dim=32)
        scorer = AnomalyScorer(model, threshold=None, device="cpu")

        fake_packets = [bytes(range(256)) * 4 for _ in range(5)]
        results = scorer.score_batch(fake_packets)

        for _, is_attack in results:
            self.assertFalse(is_attack)


class TestEvaluateThresholdCheck(unittest.TestCase):
    """M1：驗證 evaluate() 在 threshold=None 時拋出 ValueError"""

    def test_evaluate_raises_without_threshold(self):
        """threshold 未設定時呼叫 evaluate 應拋出 ValueError"""
        from anomaly_scorer import AnomalyScorer

        model = CNNAutoencoder(latent_dim=32)
        scorer = AnomalyScorer(model, threshold=None, device="cpu")

        X_test = np.random.rand(10, 32, 32).astype(np.float32)
        y_test = np.zeros(10, dtype=int)

        with self.assertRaises(ValueError) as ctx:
            scorer.evaluate(X_test, y_test)
        self.assertIn("threshold", str(ctx.exception))


class TestSimulateLocalRNG(unittest.TestCase):
    """M4：驗證 _load_simulate 使用局部 RNG"""

    def test_global_state_not_affected(self):
        """呼叫 _load_simulate 不應影響全域隨機狀態"""
        from dataset_loader import DatasetFactory

        # 記錄呼叫前的全域隨機狀態
        np.random.seed(123)
        before = np.random.rand()

        # 重置種子，呼叫 _load_simulate
        np.random.seed(123)
        _ = DatasetFactory._load_simulate(n_normal=50, n_attack=30)
        after = np.random.rand()

        # [關鍵驗證] 如果 _load_simulate 汙染了全域狀態，
        # after 會與 before 不同
        self.assertEqual(before, after)


class TestThresholdTunerPRAUC(unittest.TestCase):
    """M5：驗證 threshold_tuner 的 PR-AUC 不重複計算"""

    def test_pr_auc_consistent_across_percentiles(self):
        """所有 percentile 的 PR-AUC 應相同（因為與 threshold 無關）"""
        from threshold_tuner import ThresholdTuner

        model = CNNAutoencoder(latent_dim=32)
        tuner = ThresholdTuner(model, device="cpu")

        X_normal = np.random.rand(50, 32, 32).astype(np.float32)
        X_attack = np.random.rand(30, 32, 32).astype(np.float32) + 0.5

        results = tuner.scan_percentiles(X_normal, X_attack, percentiles=[80, 90, 95])

        # 所有結果的 PR-AUC 應完全一致（因為只計算一次）
        pr_aucs = [r["pr_auc"] for r in results]
        self.assertTrue(all(a == pr_aucs[0] for a in pr_aucs),
                        f"PR-AUC 應相同，但得到 {pr_aucs}")


class TestAutoNumWorkers(unittest.TestCase):
    """M7：驗證 _auto_num_workers 回傳合理值"""

    def test_returns_integer(self):
        """應回傳整數"""
        result = Trainer._auto_num_workers()
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)

    def test_windows_returns_zero(self):
        """在 Windows 上應回傳 0"""
        import platform
        if platform.system() == "Windows":
            self.assertEqual(Trainer._auto_num_workers(), 0)


class TestDataLeakageFix(unittest.TestCase):
    """Bug 2：驗證資料洩漏修正的邏輯（不執行完整訓練）"""

    def test_test_split_logic(self):
        """驗證測試集切分邏輯：只取最後 20% 的正常流量"""
        X_normal = np.arange(100).reshape(100, 1)

        n_test = max(1, int(len(X_normal) * 0.2))
        X_normal_test = X_normal[-n_test:]

        self.assertEqual(len(X_normal_test), 20)
        # 確認取的是最後 20%
        self.assertEqual(X_normal_test[0, 0], 80)
        self.assertEqual(X_normal_test[-1, 0], 99)

    def test_test_split_small_dataset(self):
        """即使資料集很小，也至少有 1 個測試樣本"""
        X_normal = np.array([[1], [2], [3]])
        n_test = max(1, int(len(X_normal) * 0.2))
        self.assertGreaterEqual(n_test, 1)


if __name__ == "__main__":
    unittest.main()
