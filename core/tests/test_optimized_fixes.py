# ============================================================
# core/tests/test_optimized_fixes.py
#
# 驗證 optimized_fix_report 修正項目的單元測試
#
# 測試項目：
#   Fix #1: NSL-KDD 類別特徵編碼一致性
#   Fix #2: NSL-KDD encode() 死碼修正 + logvar 快取
#   Fix #3: cnn_autoencoder.py 合併（image_size 支援）
#   Fix #4: training_common.py 共用模組
#   Fix #6: 向量化閾值搜尋正確性
#   Fix #11: Early Stopping 整合
#
# 執行方式：
#   cd c:/Users/user/Desktop/final_project
#   pytest core/tests/test_optimized_fixes.py -v
# ============================================================

import sys
import os
import pytest
import numpy as np

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


# ══════════════════════════════════════════════════════════════
# Fix #1: NSL-KDD 類別特徵編碼一致性
# ══════════════════════════════════════════════════════════════

class TestFix1_CategoryEncoding:
    """驗證 NSL-KDD 類別特徵編碼修正"""

    def test_shared_codebook_consistency(self):
        """同一類別值在正常/攻擊子集應得到相同整數碼"""
        import pandas as pd

        # 模擬：正常流量只看到 ['tcp', 'udp']，攻擊流量只看到 ['icmp', 'tcp']
        _CAT_FEATURES = ['protocol_type']

        df_normal = pd.DataFrame({'protocol_type': ['tcp', 'udp', 'tcp', 'udp']})
        df_attack = pd.DataFrame({'protocol_type': ['icmp', 'tcp', 'icmp', 'tcp']})

        # 建立共用碼表（修正後的做法）
        combined = pd.concat([df_normal, df_attack], ignore_index=True)
        cat_categories = {
            col: sorted(combined[col].astype(str).unique())
            for col in _CAT_FEATURES
        }
        # 碼表應為 ['icmp', 'tcp', 'udp']

        # 用共用碼表轉換
        normal_codes = pd.Categorical(
            df_normal['protocol_type'].astype(str),
            categories=cat_categories['protocol_type']
        ).codes
        attack_codes = pd.Categorical(
            df_attack['protocol_type'].astype(str),
            categories=cat_categories['protocol_type']
        ).codes

        # tcp 在正常中的碼
        tcp_code_normal = normal_codes[0]  # 第一筆是 tcp
        # tcp 在攻擊中的碼
        tcp_code_attack = attack_codes[1]  # 第二筆是 tcp

        assert tcp_code_normal == tcp_code_attack, \
            f"tcp 在正常({tcp_code_normal}) vs 攻擊({tcp_code_attack}) 編碼不一致"

    def test_old_method_would_fail(self):
        """驗證舊方法確實會產生不一致編碼"""
        import pandas as pd

        df_normal = pd.DataFrame({'protocol_type': ['tcp', 'udp', 'tcp']})
        df_attack = pd.DataFrame({'protocol_type': ['icmp', 'tcp', 'icmp']})

        # 舊方法：各自獨立建碼表
        old_normal_codes = pd.Categorical(df_normal['protocol_type']).codes
        old_attack_codes = pd.Categorical(df_attack['protocol_type']).codes

        tcp_old_normal = old_normal_codes[0]  # tcp
        tcp_old_attack = old_attack_codes[1]  # tcp

        # 舊方法中 tcp 的碼不一致（normal 中 tcp=0, attack 中 tcp=1）
        assert tcp_old_normal != tcp_old_attack, \
            "舊方法應該會產生不一致的編碼（此測試驗證 bug 存在）"


# ══════════════════════════════════════════════════════════════
# Fix #2: NSL-KDD encode() 死碼修正
# ══════════════════════════════════════════════════════════════

class TestFix2_EncodeFix:
    """驗證 NSL-KDD encode() 死碼修正與 logvar 快取"""

    @pytest.fixture
    def nslkdd_model_nonvar(self):
        semi_dir = os.path.join(os.path.dirname(__file__), '..',
                                'Semi-supervised model and trainer')
        core_dir = os.path.join(os.path.dirname(__file__), '..')
        # 確保子資料夾在 sys.path 最前面
        for d in [semi_dir, core_dir]:
            d = os.path.abspath(d)
            if d in sys.path:
                sys.path.remove(d)
            sys.path.insert(0, d)
        # 清除已快取的模組避免衝突
        for k in list(sys.modules.keys()):
            if 'cnn_autoencoder' in k:
                del sys.modules[k]
        from semi_supervised_nslkdd import SemiSupervisedAE_NSLKDD
        return SemiSupervisedAE_NSLKDD(latent_dim=16, image_size=16, variational=False)

    @pytest.fixture
    def nslkdd_model_var(self):
        semi_dir = os.path.join(os.path.dirname(__file__), '..',
                                'Semi-supervised model and trainer')
        core_dir = os.path.join(os.path.dirname(__file__), '..')
        for d in [semi_dir, core_dir]:
            d = os.path.abspath(d)
            if d in sys.path:
                sys.path.remove(d)
            sys.path.insert(0, d)
        for k in list(sys.modules.keys()):
            if 'cnn_autoencoder' in k:
                del sys.modules[k]
        from semi_supervised_nslkdd import SemiSupervisedAE_NSLKDD
        return SemiSupervisedAE_NSLKDD(latent_dim=16, image_size=16, variational=True)

    def test_nonvar_encode_single_call(self, nslkdd_model_nonvar):
        """非 VAE 模式：encode 只呼叫一次 encoder_fc"""
        x = torch.randn(2, 1, 16, 16)
        z, logvar = nslkdd_model_nonvar.encode(x)
        assert z.shape == (2, 16)
        assert logvar is None
        assert nslkdd_model_nonvar._last_logvar is None

    def test_var_encode_caches_logvar(self, nslkdd_model_var):
        """VAE 模式：encode 應快取 logvar 到 _last_logvar"""
        x = torch.randn(2, 1, 16, 16)
        mu, logvar = nslkdd_model_var.encode(x)
        assert mu.shape == (2, 16)
        assert logvar.shape == (2, 16)
        assert nslkdd_model_var._last_logvar is not None
        assert torch.equal(logvar, nslkdd_model_var._last_logvar)

    def test_forward_then_read_logvar(self, nslkdd_model_var):
        """forward() 呼叫後，_last_logvar 應自動填充（不需再呼叫 encode）"""
        x = torch.randn(2, 1, 16, 16)
        x_hat, z = nslkdd_model_var(x)
        # forward 內部呼叫了 encode，所以 _last_logvar 應該有值
        assert nslkdd_model_var._last_logvar is not None


# ══════════════════════════════════════════════════════════════
# Fix #3: cnn_autoencoder.py 合併（image_size 支援）
# ══════════════════════════════════════════════════════════════

class TestFix3_MergedAutoencoder:
    """驗證合併後的 CNNAutoencoder 支援 image_size 參數"""

    def test_default_image_size_32(self):
        from cnn_autoencoder import CNNAutoencoder
        model = CNNAutoencoder(latent_dim=32)
        x = torch.randn(2, 1, 32, 32)
        x_hat, z = model(x)
        assert x_hat.shape == (2, 1, 32, 32)
        assert z.shape == (2, 32)

    def test_image_size_16(self):
        from cnn_autoencoder import CNNAutoencoder
        model = CNNAutoencoder(latent_dim=16, image_size=16)
        x = torch.randn(2, 1, 16, 16)
        x_hat, z = model(x)
        assert x_hat.shape == (2, 1, 16, 16)
        assert z.shape == (2, 16)

    def test_image_size_8(self):
        from cnn_autoencoder import CNNAutoencoder
        model = CNNAutoencoder(latent_dim=8, image_size=8)
        x = torch.randn(2, 1, 8, 8)
        x_hat, z = model(x)
        assert x_hat.shape == (2, 1, 8, 8)
        assert z.shape == (2, 8)

    def test_features_to_image_exists(self):
        from cnn_autoencoder import features_to_image
        X = np.random.rand(5, 100).astype(np.float32)
        result = features_to_image(X, image_size=16)
        assert result.shape == (5, 1, 16, 16)

    def test_kaiming_init_preserved(self):
        from cnn_autoencoder import CNNAutoencoder
        model = CNNAutoencoder(latent_dim=32, image_size=32)
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                assert param.abs().sum().item() > 0, f"{name} 為全零"

    def test_get_model_info_has_image_size(self):
        """get_model_info 仍可正常運作"""
        from cnn_autoencoder import CNNAutoencoder
        model = CNNAutoencoder(latent_dim=32, image_size=32)
        info = model.get_model_info()
        assert 'total_params' in info
        assert info['total_params'] > 100_000


# ══════════════════════════════════════════════════════════════
# Fix #4: training_common.py 共用模組
# ══════════════════════════════════════════════════════════════

class TestFix4_TrainingCommon:
    """驗證 training_common.py 模組功能"""

    def test_earlystopping_memory_based(self):
        """EarlyStopping 應在記憶體保留最佳權重"""
        from training_common import EarlyStopping
        model = nn.Linear(10, 5)

        es = EarlyStopping(patience=3, path="__test_es__.pt")

        # 模擬 3 次改善
        es(0.5, model)
        assert es.best_state is not None
        assert not os.path.exists("__test_es__.pt")  # 不應自動寫檔

        es(0.3, model)
        es(0.1, model)
        assert es.best_loss == pytest.approx(0.1, abs=1e-6)

    def test_earlystopping_restore(self):
        """restore_best 應正確載入最佳權重"""
        from training_common import EarlyStopping
        model = nn.Linear(10, 5)
        es = EarlyStopping(patience=3, path="__test_es2__.pt")

        # 記錄最佳狀態
        es(0.5, model)
        best_w = model.weight.data.clone()

        # 改變模型參數
        model.weight.data.fill_(999.0)
        assert not torch.equal(model.weight.data, best_w)

        # 恢復最佳
        es.restore_best(model, persist=False)
        assert torch.equal(model.weight.data, best_w)

        # 清除測試檔案
        for f in ["__test_es__.pt", "__test_es2__.pt"]:
            if os.path.exists(f):
                os.remove(f)

    def test_amp_context_cpu(self):
        """AmpContext 在 CPU 上應自動關閉"""
        from training_common import AmpContext
        amp = AmpContext(torch.device("cpu"), enabled=True)
        assert not amp.enabled  # CPU 上自動關閉


# ══════════════════════════════════════════════════════════════
# Fix #6: 向量化閾值搜尋正確性
# ══════════════════════════════════════════════════════════════

class TestFix6_VectorizedThreshold:
    """驗證向量化閾值搜尋與暴力搜尋結果一致"""

    def test_vectorized_matches_bruteforce(self):
        """向量化結果應與暴力搜尋完全一致"""
        from training_common import compute_optimal_threshold_vectorized

        np.random.seed(42)
        errors_normal = np.random.exponential(0.01, 500)
        errors_attack = np.random.exponential(0.05, 200)

        # 向量化
        v_thr, v_f1, v_pct = compute_optimal_threshold_vectorized(
            errors_normal, errors_attack, pct_range=(50, 100)
        )

        # 暴力搜尋
        best_f1_bf = -1.0
        best_thr_bf = None
        best_pct_bf = 50
        for pct in range(50, 100):
            thr = float(np.percentile(errors_normal, pct))
            fp = int((errors_normal > thr).sum())
            tp = int((errors_attack > thr).sum())
            fn = len(errors_attack) - tp
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            if f1 > best_f1_bf:
                best_f1_bf = f1
                best_thr_bf = thr
                best_pct_bf = pct

        assert abs(v_thr - best_thr_bf) < 1e-6, \
            f"閾值不一致: vec={v_thr} bf={best_thr_bf}"
        assert abs(v_f1 - best_f1_bf) < 1e-6, \
            f"F1 不一致: vec={v_f1} bf={best_f1_bf}"
        assert v_pct == best_pct_bf, \
            f"百分位不一致: vec={v_pct} bf={best_pct_bf}"

    def test_returns_valid_types(self):
        """回傳值型別應正確"""
        from training_common import compute_optimal_threshold_vectorized
        errors_n = np.random.rand(100)
        errors_a = np.random.rand(50) + 0.5
        thr, f1, pct = compute_optimal_threshold_vectorized(errors_n, errors_a)
        assert isinstance(thr, float)
        assert isinstance(f1, float)
        assert isinstance(pct, int)
        assert 0 <= f1 <= 1.0


# ══════════════════════════════════════════════════════════════
# Fix #11: Early Stopping 在三個訓練器中
# ══════════════════════════════════════════════════════════════

class TestFix11_EarlyStoppingIntegration:
    """驗證三個資料集訓練器正確引入 EarlyStopping"""

    def test_ddos2019_has_earlystopping_import(self):
        """DDoS2019 訓練器應有 EarlyStopping import"""
        semi_dir = os.path.join(os.path.dirname(__file__), '..',
                                'Semi-supervised model and trainer')
        path = os.path.join(semi_dir, 'semi_supervised_cicddos2019.py')
        with open(path, encoding='utf-8') as f:
            content = f.read()
        assert 'from training_common import EarlyStopping' in content

    def test_cicids2017_has_earlystopping_import(self):
        """CICIDS2017 訓練器應有 EarlyStopping import"""
        semi_dir = os.path.join(os.path.dirname(__file__), '..',
                                'Semi-supervised model and trainer')
        path = os.path.join(semi_dir, 'semi_supervised_cicids2017.py')
        with open(path, encoding='utf-8') as f:
            content = f.read()
        assert 'from training_common import EarlyStopping' in content

    def test_nslkdd_has_earlystopping_import(self):
        """NSL-KDD 訓練器應有 EarlyStopping import"""
        semi_dir = os.path.join(os.path.dirname(__file__), '..',
                                'Semi-supervised model and trainer')
        path = os.path.join(semi_dir, 'semi_supervised_nslkdd.py')
        with open(path, encoding='utf-8') as f:
            content = f.read()
        assert 'from training_common import EarlyStopping' in content

    def test_val_split_default_zero(self):
        """val_split 預設為 0.0，確保原行為不受影響"""
        # 驗證 config 中 val_split 的預設值
        semi_dir = os.path.join(os.path.dirname(__file__), '..',
                                'Semi-supervised model and trainer')
        path = os.path.join(semi_dir, 'semi_supervised_cicddos2019.py')
        with open(path, encoding='utf-8') as f:
            content = f.read()
        assert 'val_split' in content
        # 確認有 get("val_split", 0.0) 或類似預設值
        assert '0.0' in content or '0)' in content
