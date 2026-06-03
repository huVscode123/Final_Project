# ============================================================
# data_augmentor.py - 封包影像資料擴增模組
#
# 問題背景：
#   CNN Autoencoder 使用非監督式學習，訓練資料越多樣化，
#   模型學到的「正常流量特徵」越豐富，異常偵測效果越好。
#   當錄製的封包數量不足（< 1000 筆），可用資料擴增補充。
#
# 擴增策略（只對正常流量做擴增）：
#   1. 高斯雜訊（Gaussian Noise）
#      模擬網路傳輸中輕微的位元抖動，增加模型對雜訊的容忍度
#   2. 隨機遮罩（Random Mask）
#      遮蔽部分像素，讓模型學習從部分資訊重建完整封包
#   3. 亮度縮放（Brightness Scale）
#      模擬不同大小封包（payload 長短）的影像差異
#   4. 混合（Mixup）
#      將兩筆正常封包線性插值，生成新的正常樣本
#
# 注意：攻擊封包不做擴增（只用於閾值評估，不參與訓練）
#
# 使用方式：
#   from data_augmentor import DataAugmentor
#   aug = DataAugmentor(noise_std=0.02, mask_ratio=0.1)
#   X_aug = aug.augment(X_normal, multiplier=3)
#   # X_normal 1000 筆 → X_aug 3000 筆
# ============================================================

import numpy as np


class DataAugmentor:
    """
    封包影像資料擴增器

    Args:
        noise_std  : 高斯雜訊標準差（預設 0.02，值域 [0,1] 的 2%）
        mask_ratio : 隨機遮罩比例（預設 0.1，遮蔽 10% 的像素）
        scale_range: 亮度縮放範圍（預設 0.9~1.1）
        use_mixup  : 是否啟用 Mixup 擴增（預設 True）
        mixup_alpha: Mixup 的 Beta 分布參數（預設 0.2）
        seed       : 隨機種子（確保可重現）
    """

    def __init__(self,
                 noise_std:   float = 0.02,
                 mask_ratio:  float = 0.10,
                 scale_range: tuple = (0.90, 1.10),
                 use_mixup:   bool  = True,
                 mixup_alpha: float = 0.20,
                 seed:        int   = 42):
        self.noise_std   = noise_std
        self.mask_ratio  = mask_ratio
        self.scale_range = scale_range
        self.use_mixup   = use_mixup
        self.mixup_alpha = mixup_alpha
        self.rng         = np.random.default_rng(seed)

    # ── 主要擴增入口 ──────────────────────────────────────
    def augment(self, X: np.ndarray, multiplier: int = 3) -> np.ndarray:
        """
        對正常流量影像矩陣進行擴增

        Args:
            X         : 原始影像矩陣 shape=(N, 32, 32) float32，值域 [0,1]
            multiplier: 擴增倍數（3 表示最終資料量為原始的 3 倍）

        Returns:
            np.ndarray shape=(N * multiplier, 32, 32) float32
            包含原始資料 + 各種擴增資料
        """
        assert X.ndim == 3, "X 應為 (N, H, W) 格式"
        assert X.dtype == np.float32, "X 應為 float32"
        assert 0.0 <= X.min() and X.max() <= 1.0, "X 值域應在 [0, 1]"

        n = len(X)
        print(f"  [DataAugmentor] 原始資料: {n} 筆 → 擴增至 {n * multiplier} 筆")

        all_data = [X.copy()]   # 保留原始資料

        for i in range(multiplier - 1):
            # 每一輪隨機選擇一種或多種擴增策略組合
            method = i % 4

            if method == 0:
                aug = self.add_gaussian_noise(X)
                print(f"    第 {i+1} 輪：高斯雜訊（std={self.noise_std}）")
            elif method == 1:
                aug = self.random_mask(X)
                print(f"    第 {i+1} 輪：隨機遮罩（ratio={self.mask_ratio}）")
            elif method == 2:
                aug = self.brightness_scale(X)
                print(f"    第 {i+1} 輪：亮度縮放（range={self.scale_range}）")
            else:
                if self.use_mixup:
                    aug = self.mixup(X)
                    print(f"    第 {i+1} 輪：Mixup（alpha={self.mixup_alpha}）")
                else:
                    aug = self.add_gaussian_noise(X)
                    print(f"    第 {i+1} 輪：高斯雜訊（替代 Mixup）")

            all_data.append(aug)

        result = np.concatenate(all_data, axis=0)
        # 打亂順序，避免訓練時批次內全是同種擴增
        idx = self.rng.permutation(len(result))
        result = result[idx]

        print(f"  [DataAugmentor] 擴增完成：{len(result)} 筆")
        return result

    # ── 擴增策略 1：高斯雜訊 ─────────────────────────────
    def add_gaussian_noise(self, X: np.ndarray) -> np.ndarray:
        """
        對每個像素加上小量高斯雜訊

        模擬原理：
            真實網路環境中同類型的封包，每次擷取的位元組可能有
            輕微差異（如時間戳、Checksum 計算差異）。
            加入雜訊讓模型對這些小波動具備容忍度。

        數學：
            x_aug = clip(x + N(0, noise_std), 0, 1)
        """
        noise = self.rng.normal(0, self.noise_std, X.shape).astype(np.float32)
        return np.clip(X + noise, 0.0, 1.0)

    # ── 擴增策略 2：隨機遮罩 ─────────────────────────────
    def random_mask(self, X: np.ndarray) -> np.ndarray:
        """
        隨機遮蔽部分像素（設為 0）

        模擬原理：
            封包被截斷或部分欄位為空的情況。
            讓模型學習即使資訊不完整，仍能識別正常流量的結構。

        數學：
            mask ~ Bernoulli(mask_ratio)
            x_aug = x * (1 - mask)
        """
        X_aug = X.copy()
        n, h, w = X.shape
        n_mask = int(h * w * self.mask_ratio)

        for i in range(n):
            # 隨機選擇要遮蔽的像素位置
            flat_idx = self.rng.choice(h * w, size=n_mask, replace=False)
            rows, cols = np.unravel_index(flat_idx, (h, w))
            X_aug[i, rows, cols] = 0.0

        return X_aug

    # ── 擴增策略 3：亮度縮放 ─────────────────────────────
    def brightness_scale(self, X: np.ndarray) -> np.ndarray:
        """
        對每筆樣本隨機縮放像素亮度

        模擬原理：
            不同長度的封包（Payload 多寡）會影響影像的整體亮度。
            縮放模擬正常流量中封包大小的自然變化。

        數學：
            scale ~ Uniform(scale_range[0], scale_range[1])
            x_aug = clip(x * scale, 0, 1)
        """
        lo, hi = self.scale_range
        scales = self.rng.uniform(lo, hi, size=(len(X), 1, 1)).astype(np.float32)
        return np.clip(X * scales, 0.0, 1.0)

    # ── 擴增策略 4：Mixup ────────────────────────────────
    def mixup(self, X: np.ndarray) -> np.ndarray:
        """
        將兩筆正常封包線性插值，生成新的正常樣本

        模擬原理：
            兩筆都是正常流量，它們的加權平均在語義上仍是正常流量。
            Mixup 是圖像分類中廣泛使用的資料擴增技術。

        數學：
            lam ~ Beta(alpha, alpha)
            x_aug = lam * x_i + (1 - lam) * x_j
            其中 i, j 是隨機配對的兩筆不同樣本

        參考：
            Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
        """
        n = len(X)
        # 隨機配對
        idx = self.rng.permutation(n)
        X_shuffled = X[idx]

        # 從 Beta 分布取得插值係數
        lam = self.rng.beta(self.mixup_alpha, self.mixup_alpha,
                            size=(n, 1, 1)).astype(np.float32)

        return lam * X + (1 - lam) * X_shuffled

    # ── 合併多個資料集 ────────────────────────────────────
    @staticmethod
    def merge_datasets(*npy_paths: str,
                       output_path: str = None) -> np.ndarray:
        """
        合併多個 .npy 資料集（例如：合併 CIC-IDS2017 + NSL-KDD 的正常流量）

        Args:
            *npy_paths  : 多個 .npy 檔案路徑
            output_path : 合併後儲存路徑（None 則只回傳不儲存）

        Returns:
            合併後的影像矩陣 shape=(N_total, 32, 32)

        使用範例：
            X_merged = DataAugmentor.merge_datasets(
                "output/dataset_cicids2017/X_normal.npy",
                "output/dataset_nslkdd/X_normal.npy",
                output_path="output/dataset_merged/X_normal.npy"
            )
        """
        arrays = []
        for path in npy_paths:
            arr = np.load(path).astype(np.float32)
            print(f"  載入: {path}  shape={arr.shape}")
            arrays.append(arr)

        merged = np.concatenate(arrays, axis=0)
        print(f"  合併後: shape={merged.shape}")

        if output_path:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, merged)
            print(f"  已儲存: {output_path}")

        return merged