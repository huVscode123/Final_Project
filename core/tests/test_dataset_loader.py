# ============================================================
# core/tests/test_dataset_loader.py - DatasetLoader 單元測試
#
# 測試範圍：
#   - NSLKDDLoader：CSV 解析、類別特徵編碼、正規化、影像轉換
#   - CICIDSLoader：標籤欄位偵測、無限值處理、特徵選取
#   - CICDDoS2019Loader：繼承邏輯與遞迴 CSV 掃描
#   - DatasetFactory：統一介面、模擬資料輸出格式
#   - 特徵轉影像：shape 正確性、值域、補零/截斷邊界
#
# 全部測試使用 mock 資料（小型 CSV），不需要下載任何真實資料集。
#
# 執行方式：
#   cd network_capture
#   pytest tests/test_dataset_loader.py -v
# ============================================================

import sys
import os
import tempfile
import shutil

import pytest
import numpy as np
import pandas as pd

# 確保可以 import 上層模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_loader import (
    NSLKDDLoader, CICIDSLoader, CICDDoS2019Loader,
    DatasetFactory, IMAGE_SIZE, FEATURE_DIM
)


# ============================================================
# Fixtures：建立臨時測試資料
# ============================================================

@pytest.fixture
def tmp_dir():
    """建立臨時目錄，測試結束後自動清理"""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def nslkdd_dir(tmp_dir):
    """
    建立模擬 NSL-KDD 資料目錄

    包含 20 筆正常 + 15 筆攻擊，共 35 筆
    欄位格式依官方：41 特徵 + label + difficulty（無標題列）
    """
    rows = []
    # 正常流量（label = normal）
    for i in range(20):
        row = [
            i % 60,          # duration
            "tcp",           # protocol_type
            "http",          # service
            "SF",            # flag
            1000 + i * 100,  # src_bytes
            500 + i * 50,    # dst_bytes
        ] + [0] * 35 + ["normal", 10]   # 35 個填 0，label, difficulty
        rows.append(row)

    # 攻擊流量（label = neptune）
    for i in range(15):
        row = [
            0,               # duration = 0 (SYN Flood 特徵)
            "tcp",
            "private",
            "S0",
            0, 0,
        ] + [0] * 35 + ["neptune", 5]
        rows.append(row)

    from dataset_loader import NSLKDDLoader
    cols = NSLKDDLoader.FEATURE_NAMES
    df = pd.DataFrame(rows, columns=cols)
    path = os.path.join(tmp_dir, "KDDTrain+.txt")
    df.to_csv(path, header=False, index=False)
    return tmp_dir


@pytest.fixture
def cicids_dir(tmp_dir):
    """
    建立模擬 CIC-IDS2017 資料目錄

    包含 25 筆正常（BENIGN）+ 15 筆攻擊（DoS Hulk）
    欄位模擬 CICFlowMeter 輸出（含前置空格的 " Label" 欄位名）
    """
    n_features = 78
    feature_names = [f"Feature{i}" for i in range(n_features)]

    normal_data = np.random.rand(25, n_features).astype(np.float32)
    attack_data = np.random.rand(15, n_features).astype(np.float32) * 0.1

    normal_df = pd.DataFrame(normal_data, columns=feature_names)
    normal_df[" Label"] = "BENIGN"

    attack_df = pd.DataFrame(attack_data, columns=feature_names)
    attack_df[" Label"] = "DoS Hulk"

    df = pd.concat([normal_df, attack_df], ignore_index=True)
    path = os.path.join(tmp_dir, "Monday-WorkingHours.pcap_ISCX.csv")
    df.to_csv(path, index=False)
    return tmp_dir


@pytest.fixture
def cicddos_dir(tmp_dir):
    """
    建立模擬 CIC-DDoS2019 資料目錄（含子目錄）
    """
    sub_dir = os.path.join(tmp_dir, "03-11")
    os.makedirs(sub_dir)

    n_features = 78
    feature_names = [f"Feature{i}" for i in range(n_features)]
    data = np.random.rand(20, n_features).astype(np.float32)

    df = pd.DataFrame(data, columns=feature_names)
    # 放 10 筆正常、10 筆攻擊
    df[" Label"] = ["BENIGN"] * 10 + ["UDP"] * 10

    path = os.path.join(sub_dir, "DrDoS_UDP.csv")
    df.to_csv(path, index=False)
    return tmp_dir


# ============================================================
# 特徵轉影像（共用邏輯）
# ============================================================
class TestFeaturesToImages:

    def test_output_shape_41_features(self):
        """NSL-KDD 的 41 個特徵應轉換為 (N, 32, 32) 影像"""
        feat = np.random.rand(10, 41).astype(np.float32)
        images = NSLKDDLoader._features_to_images(feat)
        assert images.shape == (10, IMAGE_SIZE, IMAGE_SIZE)

    def test_output_shape_78_features(self):
        """CIC 的 78 個特徵同樣轉換為 (N, 32, 32)"""
        feat = np.random.rand(10, 78).astype(np.float32)
        images = NSLKDDLoader._features_to_images(feat)
        assert images.shape == (10, IMAGE_SIZE, IMAGE_SIZE)

    def test_padding_correctness(self):
        """補零後，超出特徵數的位置應為 0"""
        feat = np.ones((5, 41), dtype=np.float32)
        images = NSLKDDLoader._features_to_images(feat)
        # 影像攤平後，前 41 個像素應為正規化後的值，後面應為 0
        flat = images[0].flatten()
        assert flat[41:].sum() == 0.0, "補零區域應全為 0"

    def test_truncation_over_1024(self):
        """特徵數超過 1024 時應截斷，不報錯"""
        feat = np.random.rand(3, 2000).astype(np.float32)
        images = NSLKDDLoader._features_to_images(feat)
        assert images.shape == (3, 32, 32)

    def test_output_dtype(self):
        """輸出應為 float32"""
        feat = np.random.rand(4, 41).astype(np.float64)  # 輸入 float64
        images = NSLKDDLoader._features_to_images(feat.astype(np.float32))
        assert images.dtype == np.float32

    def test_single_sample(self):
        """單一樣本也應正常轉換"""
        feat = np.random.rand(1, 41).astype(np.float32)
        images = NSLKDDLoader._features_to_images(feat)
        assert images.shape == (1, 32, 32)


# ============================================================
# NSLKDDLoader 測試
# ============================================================
class TestNSLKDDLoader:

    def test_load_returns_correct_shapes(self, nslkdd_dir):
        """load() 應返回正確 shape 的三個陣列"""
        loader = NSLKDDLoader(nslkdd_dir)
        X_normal, X_attack, y = loader.load()

        assert X_normal.ndim == 3
        assert X_attack.ndim == 3
        assert X_normal.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE)
        assert X_attack.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE)

    def test_load_correct_counts(self, nslkdd_dir):
        """load() 應正確分離 20 筆正常與 15 筆攻擊"""
        loader = NSLKDDLoader(nslkdd_dir)
        X_normal, X_attack, y = loader.load()

        assert len(X_normal) == 20
        assert len(X_attack) == 15

    def test_label_vector_consistency(self, nslkdd_dir):
        """y 的標籤分布應與 X_normal / X_attack 數量一致"""
        loader = NSLKDDLoader(nslkdd_dir)
        X_normal, X_attack, y = loader.load()

        assert len(y) == len(X_normal) + len(X_attack)
        assert (y == 0).sum() == len(X_normal)
        assert (y == 1).sum() == len(X_attack)

    def test_output_value_range(self, nslkdd_dir):
        """正規化後所有值應在 [0, 1]"""
        loader = NSLKDDLoader(nslkdd_dir)
        X_normal, X_attack, y = loader.load()

        assert X_normal.min() >= 0.0, f"最小值 {X_normal.min()} < 0"
        assert X_normal.max() <= 1.0, f"最大值 {X_normal.max()} > 1"

    def test_output_dtype(self, nslkdd_dir):
        """輸出應為 float32"""
        loader = NSLKDDLoader(nslkdd_dir)
        X_normal, _, _ = loader.load()
        assert X_normal.dtype == np.float32

    def test_file_not_found_raises(self, tmp_dir):
        """目錄不存在時應拋出 FileNotFoundError"""
        loader = NSLKDDLoader(os.path.join(tmp_dir, "nonexistent"))
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_categorical_encoding(self, nslkdd_dir):
        """類別特徵（protocol_type/service/flag）應被正確編碼為數值"""
        loader = NSLKDDLoader(nslkdd_dir)
        loader.load()
        # 確認 encoders 已建立
        for col in NSLKDDLoader.CATEGORICAL_FEATURES:
            assert col in loader.encoders, f"{col} encoder 未建立"

    def test_y_dtype(self, nslkdd_dir):
        """標籤應為整數型別"""
        loader = NSLKDDLoader(nslkdd_dir)
        _, _, y = loader.load()
        assert y.dtype == np.int32


# ============================================================
# CICIDSLoader 測試
# ============================================================
class TestCICIDSLoader:

    def test_load_returns_correct_shapes(self, cicids_dir):
        """load() 應返回 (N, 32, 32) shape"""
        loader = CICIDSLoader(cicids_dir)
        X_normal, X_attack, y = loader.load()

        assert X_normal.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE)
        assert X_attack.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE)

    def test_load_correct_counts(self, cicids_dir):
        """應正確分離 25 筆正常與 15 筆攻擊"""
        loader = CICIDSLoader(cicids_dir)
        X_normal, X_attack, y = loader.load()

        assert len(X_normal) == 25
        assert len(X_attack) == 15

    def test_label_column_with_leading_space(self, cicids_dir):
        """應正確處理前置空格的 ' Label' 欄位名稱"""
        loader = CICIDSLoader(cicids_dir)
        X_normal, X_attack, _ = loader.load()
        # 若標籤欄位找不到，X_normal 會是 0
        assert len(X_normal) > 0, "無法找到 ' Label' 欄位（空格問題）"

    def test_value_range(self, cicids_dir):
        """正規化後 X_normal 應在 [0, 1]；X_attack 可超出（預期行為）"""
        loader = CICIDSLoader(cicids_dir)
        X_normal, X_attack, _ = loader.load()

        # Scaler 只 fit 在正常資料上，正常資料 clip 到 [0,1]
        assert X_normal.min() >= 0.0, f"X_normal min {X_normal.min()} < 0"
        assert X_normal.max() <= 1.0, f"X_normal max {X_normal.max()} > 1"
        # X_attack 不要求在 [0,1]，超出範圍有助於 CNN 偵測異常

    def test_no_nan_in_output(self, cicids_dir):
        """輸出不應包含 NaN"""
        loader = CICIDSLoader(cicids_dir)
        X_normal, X_attack, _ = loader.load()

        assert not np.isnan(X_normal).any(), "X_normal 含有 NaN"
        assert not np.isnan(X_attack).any(), "X_attack 含有 NaN"

    def test_infinite_value_handling(self, tmp_dir):
        """含有 inf / -inf 的資料應被正確處理（不出現在輸出中）"""
        feature_names = [f"F{i}" for i in range(10)]
        data = np.random.rand(10, 10).astype(np.float32)
        # 插入無限值
        data[0, 0] = np.inf
        data[1, 1] = -np.inf
        data[2, 2] = np.nan

        df = pd.DataFrame(data, columns=feature_names)
        df[" Label"] = ["BENIGN"] * 5 + ["Attack"] * 5
        path = os.path.join(tmp_dir, "test_inf.csv")
        df.to_csv(path, index=False)

        loader = CICIDSLoader(tmp_dir)
        X_normal, X_attack, _ = loader.load(csv_files=[path])

        assert not np.isnan(X_normal).any()
        assert not np.isinf(X_normal).any()

    def test_file_not_found_raises(self, tmp_dir):
        """目錄為空時應拋出 FileNotFoundError"""
        empty_dir = os.path.join(tmp_dir, "empty")
        os.makedirs(empty_dir)
        loader = CICIDSLoader(empty_dir)
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_max_sample_limit(self, cicids_dir):
        """max_normal / max_attack 參數應正確限制樣本數"""
        loader = CICIDSLoader(cicids_dir)
        X_normal, X_attack, _ = loader.load(max_normal=10, max_attack=5)

        assert len(X_normal) <= 10
        assert len(X_attack) <= 5

    def test_label_vector_length(self, cicids_dir):
        """y 的長度應等於 X_normal + X_attack"""
        loader = CICIDSLoader(cicids_dir)
        X_normal, X_attack, y = loader.load()
        assert len(y) == len(X_normal) + len(X_attack)


# ============================================================
# CICDDoS2019Loader 測試
# ============================================================
class TestCICDDoS2019Loader:

    def test_recursive_csv_scan(self, cicddos_dir):
        """應能遞迴掃描子目錄中的 CSV 檔案"""
        loader = CICDDoS2019Loader(cicddos_dir)
        X_normal, X_attack, y = loader.load()

        # 測試資料有 10 正常 + 10 攻擊
        assert len(X_normal) == 10
        assert len(X_attack) == 10

    def test_output_shape(self, cicddos_dir):
        """輸出 shape 應與 CIC-IDS2017 一致（32x32）"""
        loader = CICDDoS2019Loader(cicddos_dir)
        X_normal, _, _ = loader.load()
        assert X_normal.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE)

    def test_inherits_cicids_preprocessing(self, cicddos_dir):
        """DDoS2019 繼承 CICIDSLoader，應通過相同的前處理邏輯"""
        loader = CICDDoS2019Loader(cicddos_dir)
        X_normal, X_attack, _ = loader.load()

        # X_normal 無 NaN、值域 [0,1]（已 clip）
        assert not np.isnan(X_normal).any()
        assert X_normal.min() >= 0.0, f"X_normal min={X_normal.min()} < 0"
        assert X_normal.max() <= 1.0, f"X_normal max={X_normal.max()} > 1"


# ============================================================
# DatasetFactory 測試
# ============================================================
class TestDatasetFactory:

    def test_simulate_returns_correct_shapes(self):
        """模擬資料應返回 (N, 32, 32) 的影像矩陣"""
        X_normal, X_attack, y = DatasetFactory.load("simulate",
                                                     n_normal=50, n_attack=30)
        assert X_normal.shape == (50, IMAGE_SIZE, IMAGE_SIZE)
        assert X_attack.shape == (30, IMAGE_SIZE, IMAGE_SIZE)

    def test_simulate_label_counts(self):
        """模擬資料的標籤分布應與樣本數一致"""
        X_normal, X_attack, y = DatasetFactory.load("simulate",
                                                     n_normal=100, n_attack=60)
        assert (y == 0).sum() == 100
        assert (y == 1).sum() == 60

    def test_simulate_value_range(self):
        """模擬資料值應在 [0, 1]"""
        X_normal, X_attack, y = DatasetFactory.load("simulate")
        assert X_normal.min() >= 0.0
        assert X_normal.max() <= 1.0

    def test_simulate_dtype(self):
        """模擬資料應為 float32"""
        X_normal, _, _ = DatasetFactory.load("simulate")
        assert X_normal.dtype == np.float32

    def test_simulate_no_nan(self):
        """模擬資料不應含 NaN"""
        X_normal, X_attack, _ = DatasetFactory.load("simulate")
        assert not np.isnan(X_normal).any()
        assert not np.isnan(X_attack).any()

    def test_invalid_dataset_name_raises(self):
        """不支援的資料集名稱應拋出 ValueError"""
        with pytest.raises(ValueError):
            DatasetFactory.load("unknown_dataset")

    def test_case_insensitive_name(self):
        """資料集名稱應不分大小寫"""
        X_normal, _, _ = DatasetFactory.load("SIMULATE")
        assert X_normal.ndim == 3

    def test_nslkdd_via_factory(self, nslkdd_dir):
        """透過 DatasetFactory 載入 NSL-KDD 應正常運作"""
        X_normal, X_attack, y = DatasetFactory.load(
            "nslkdd", data_dir=nslkdd_dir
        )
        assert X_normal.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE)
        assert len(X_normal) == 20
        assert len(X_attack) == 15

    def test_cicids2017_via_factory(self, cicids_dir):
        """透過 DatasetFactory 載入 CIC-IDS2017 應正常運作"""
        X_normal, X_attack, y = DatasetFactory.load(
            "cicids2017", data_dir=cicids_dir
        )
        assert X_normal.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE)

    def test_cicddos2019_via_factory(self, cicddos_dir):
        """透過 DatasetFactory 載入 CIC-DDoS2019 應正常運作"""
        X_normal, X_attack, y = DatasetFactory.load(
            "cicddos2019", data_dir=cicddos_dir
        )
        assert X_normal.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE)

    def test_save_as_npy(self, tmp_dir):
        """save_as_npy 應正確儲存四個 .npy 檔案"""
        X_normal = np.random.rand(20, 32, 32).astype(np.float32)
        X_attack = np.random.rand(10, 32, 32).astype(np.float32)
        y = np.concatenate([np.zeros(20), np.ones(10)]).astype(np.int32)

        paths = DatasetFactory.save_as_npy(X_normal, X_attack, y, tmp_dir)

        for key in ("normal", "attack", "X_all", "y_all"):
            assert key in paths
            assert os.path.exists(paths[key])

        # 驗證 X_all 的 shape
        X_all = np.load(paths["X_all"])
        assert X_all.shape == (30, 32, 32)

        y_loaded = np.load(paths["y_all"])
        assert len(y_loaded) == 30
