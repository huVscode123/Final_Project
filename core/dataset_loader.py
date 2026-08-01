# ============================================================
# dataset_loader.py - 真實資料集載入模組
#
# 支援三種公開資料集：
#   1. NSL-KDD    : CSV 格式，41 個流量特徵 + 標籤
#   2. CIC-IDS2017: CSV 格式（CICFlowMeter 提取）+ 可選 PCAP
#   3. CIC-DDoS2019: CSV 格式（與 CIC-IDS2017 格式相同）
#
# 核心功能：
#   - 讀取並清洗各資料集的 CSV 檔案
#   - 分離正常（Benign）與攻擊（Attack）流量
#   - 將流量特徵向量轉換為 32x32 影像矩陣（供 CNN Autoencoder 使用）
#   - 提供統一的 DatasetFactory 介面，用一行程式切換資料集
#
# 轉換原理（特徵向量 -> 影像）：
#   各資料集的特徵向量長度不同（NSL-KDD: 41, CIC: 78），
#   統一流程為：
#     ① 選取數值型特徵，移除標籤欄位
#     ② 處理無限值（inf -> 欄位最大值）與缺失值（NaN -> 0）
#     ③ MinMaxScaler 正規化至 [0, 1]
#     ④ 補零至 1024 維，reshape 成 32x32 float32 矩陣
#   前段非零區域含有實際特徵，後段補零區域為黑色，
#   CNN 可從特徵的空間排列與數值分布學習正常流量的模式。
#
# 資料集下載連結：
#   NSL-KDD:     https://www.unb.ca/cic/datasets/nsl.html
#   CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
#   CIC-DDoS2019:https://www.unb.ca/cic/datasets/ddos-2019.html
#
# 參考 GitHub：
#   - https://github.com/jmnwong/NSL-KDD-Network-Intrusion-Detection-System
#   - https://github.com/aws-samples/amazon-sagemaker-network-intrusion-detection
#   - https://github.com/ylchan87/CIC-IDS-2017-EDA
# ============================================================

import os
import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 忽略 pandas 的 dtype 警告（大型 CSV 常見）
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


# ── 影像轉換常數 ──────────────────────────────────────────
IMAGE_SIZE    = 32          # 32x32 像素，對應 1024 維特徵向量
FEATURE_DIM   = IMAGE_SIZE * IMAGE_SIZE   # 1024


# ============================================================
# NSL-KDD 資料集載入器
# ============================================================
class NSLKDDLoader:
    """
    NSL-KDD 資料集載入器

    資料集說明：
      KDD Cup 1999 的改良版，去除重複樣本，平衡正常與攻擊比例。
      格式：CSV，無標題列，41 個特徵欄位 + 攻擊類型標籤 + 難度分數。

    特徵類型：
      - 數值特徵：duration, src_bytes, dst_bytes 等 38 個
      - 類別特徵：protocol_type (tcp/udp/icmp)、service (http/ftp/...)、
                  flag (SF/S0/REJ...) 共 3 個 -> 使用 LabelEncoder 轉數值

    標籤對應：
      'normal' -> 正常流量（label = 0）
      其他所有值（neptune, smurf, back...）-> 攻擊流量（label = 1）

    檔案結構（下載後）：
      data/nslkdd/
        KDDTrain+.txt    訓練集（帶難度分數）
        KDDTest+.txt     測試集（帶難度分數）
        KDDTrain+_20Percent.txt   訓練集 20% 子集

    下載連結：https://www.unb.ca/cic/datasets/nsl.html
    """

    # NSL-KDD 的 41 個特徵名稱（依官方文件順序）
    FEATURE_NAMES = [
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "label",          # 攻擊類型（normal / neptune / smurf ...）
        "difficulty",     # 難度分數（1~21，KDDTrain+/Test+ 才有）
    ]

    # 三個類別型特徵，需要 LabelEncoder 轉數值
    CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

    def __init__(self, data_dir: str = "data/nslkdd"):
        """
        Args:
            data_dir: NSL-KDD 資料集目錄，應包含 KDDTrain+.txt 等檔案
        """
        self.data_dir = data_dir
        self.scaler   = MinMaxScaler()
        self.encoders = {}   # {feature_name: LabelEncoder}

    def load(self, use_train: bool = True, filepath: str = None) -> tuple:
        """
        載入 NSL-KDD 資料集（支援 .txt 與 .arff）

        Args:
            use_train: True = KDDTrain+.txt，False = KDDTest+.txt
            filepath:  手動指定檔案路徑

        Returns:
            X_normal, X_attack, y
        """
        if filepath is None:
            filename = "KDDTrain+.txt" if use_train else "KDDTest+.txt"
            filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            # 嘗試找 .arff 版本
            arff_path = filepath.replace(".txt", ".arff")
            if os.path.exists(arff_path):
                filepath = arff_path
            else:
                raise FileNotFoundError(f"找不到 {filepath} 或其 .arff 版本")

        print(f"  [NSL-KDD] 載入 {filepath}")

        if filepath.endswith(".arff"):
            return self._load_arff(filepath)
        else:
            return self._load_txt(filepath)

    def _load_txt(self, path: str) -> tuple:
        """載入原始 TXT 格式 (KDD Cup 格式)"""
        n_cols = len(self.FEATURE_NAMES)
        df = pd.read_csv(path, header=None, names=self.FEATURE_NAMES[:n_cols], low_memory=False)
        return self._process_df(df)

    def _load_arff(self, path: str) -> tuple:
        """載入 WEKA ARFF 格式"""
        from scipy.io import arff
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        # ARFF 的字串欄位通常是 bytes，需解碼
        for col in df.select_dtypes([object]).columns:
            df[col] = df[col].str.decode("utf-8")
        return self._process_df(df)

    def _process_df(self, df: pd.DataFrame) -> tuple:
        """統一處理 DataFrame 的編碼、過濾與轉換"""
        print(f"  [NSL-KDD] 處理樣本數: {len(df):,}")

        # 分離正常與攻擊
        normal_mask = df["label"].astype(str).str.lower() == "normal"
        normal_df = df[normal_mask].copy()
        attack_df = df[~normal_mask].copy()

        # 特徵前處理
        X_normal_feat = self._preprocess(normal_df, fit=True)
        X_attack_feat = self._preprocess(attack_df, fit=False)

        # 轉換為影像矩陣
        X_normal = self._features_to_images(X_normal_feat)
        X_attack = self._features_to_images(X_attack_feat)

        y = np.concatenate([
            np.zeros(len(X_normal), dtype=np.int32),
            np.ones(len(X_attack),  dtype=np.int32)
        ])
        return X_normal, X_attack, y

    def _preprocess(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        前處理流程：
          ① 類別特徵 LabelEncoder 轉數值
          ② 移除標籤欄位（label, difficulty）
          ③ MinMaxScaler 正規化至 [0, 1]

        Args:
            df : 原始 DataFrame
            fit: True = 用此資料 fit scaler（訓練集），False = 只 transform（測試集）
        Returns:
            np.ndarray shape=(N, 41)，正規化後的特徵矩陣
        """
        df = df.copy()

        # 類別特徵編碼
        for col in self.CATEGORICAL_FEATURES:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                # 先 fit 一個包含所有已知類別的集合（避免 transform 時出現 unseen label）
                all_vals = ["tcp", "udp", "icmp",
                            "http", "ftp", "smtp", "ssh", "dns",
                            "SF", "S0", "REJ", "RSTO", "SH", "S1", "S2", "S3",
                            "RSTOS0", "RSTRH", "OTH"]
                self.encoders[col].fit(all_vals + list(df[col].unique()))
            # 處理未知類別（設為 0）
            known = set(self.encoders[col].classes_)
            df[col] = df[col].map(lambda x: x if x in known else "SF")
            df[col] = self.encoders[col].transform(df[col])

        # 移除標籤欄位
        drop_cols = [c for c in ["label", "difficulty"] if c in df.columns]
        df = df.drop(columns=drop_cols)

        # 轉 float，填補 NaN
        X = df.values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 正規化
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X.astype(np.float32)

    @staticmethod
    def _features_to_images(features: np.ndarray) -> np.ndarray:
        """
        將特徵向量矩陣轉換為影像矩陣

        流程：
          (N, F) -> 補零至 (N, 1024) -> reshape (N, 32, 32)

        Args:
            features: shape=(N, F)，F <= 1024
        Returns:
            images: shape=(N, 32, 32) float32
        """
        n, f = features.shape
        if f > FEATURE_DIM:
            features = features[:, :FEATURE_DIM]   # 截斷
            f = FEATURE_DIM

        # 補零至 1024
        if f < FEATURE_DIM:
            pad = np.zeros((n, FEATURE_DIM - f), dtype=np.float32)
            features = np.concatenate([features, pad], axis=1)

        return features.reshape(n, IMAGE_SIZE, IMAGE_SIZE)


# ============================================================
# CIC-IDS2017 資料集載入器
# ============================================================
class CICIDSLoader:
    """
    CIC-IDS2017 資料集載入器

    資料集說明：
      加拿大網路安全研究所（CICIDS）2017 年發布，包含五天的真實網路流量，
      涵蓋正常流量與 DoS、DDoS、PortScan、Brute Force、Web Attack、Botnet 等攻擊。
      由 CICFlowMeter 工具從 PCAP 提取 78 個流量統計特徵。

    標籤欄位：
      'BENIGN'  -> 正常流量（label = 0）
      其他所有值 -> 攻擊流量（label = 1）

    主要攻擊類型（標籤值）：
      DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest
      DDoS, PortScan, FTP-Patator, SSH-Patator
      Web Attack - Brute Force, Web Attack - XSS, Web Attack - Sql Injection
      Infiltration, Bot, Heartbleed

    檔案結構（下載後）：
      data/cicids2017/
        Monday-WorkingHours.pcap_ISCX.csv      正常流量
        Tuesday-WorkingHours.pcap_ISCX.csv     Brute Force
        Wednesday-WorkingHours.pcap_ISCX.csv   DoS / Heartbleed
        Thursday-WorkingHours.pcap_ISCX.csv    Web Attack / Infiltration
        Friday-WorkingHours.pcap_ISCX.csv      PortScan / Botnet / DDoS

    下載連結：https://www.unb.ca/cic/datasets/ids-2017.html
    """

    # CICFlowMeter 輸出的標籤欄位名稱（不同版本可能有空格差異）
    LABEL_COLUMNS = [" Label", "Label", "label"]

    # 正常流量的標籤值
    NORMAL_LABEL = "BENIGN"

    def __init__(self, data_dir: str = "data/cicids2017"):
        """
        Args:
            data_dir: CIC-IDS2017 CSV 檔案目錄
        """
        self.data_dir = data_dir
        self.scaler   = MinMaxScaler()

    def load(self, csv_files: list = None,
             max_normal: int = 50000,
             max_attack: int = 30000) -> tuple:
        """
        載入 CIC-IDS2017 資料集

        Args:
            csv_files  : 指定要載入的 CSV 路徑列表（None = 自動掃描 data_dir）
            max_normal : 最大正常樣本數（避免記憶體不足，預設 50000）
            max_attack : 最大攻擊樣本數

        Returns:
            X_normal: shape=(N_normal, 32, 32)
            X_attack: shape=(N_attack, 32, 32)
            y:        shape=(N,)
        """
        if csv_files is None:
            csv_files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))

        if not csv_files:
            raise FileNotFoundError(
                f"在 {self.data_dir} 找不到 CSV 檔案\n"
                f"請下載 CIC-IDS2017：https://www.unb.ca/cic/datasets/ids-2017.html\n"
                f"解壓後放到 {self.data_dir}/"
            )

        print(f"  [CIC-IDS2017] 載入 {len(csv_files)} 個 CSV 檔案")

        all_normal = []
        all_attack = []

        for csv_path in csv_files:
            print(f"    讀取: {os.path.basename(csv_path)}")
            try:
                df = pd.read_csv(csv_path, low_memory=False)
            except Exception as e:
                print(f"    警告：無法讀取 {csv_path}（{e}）")
                continue

            # 找到標籤欄位（名稱可能有前置空格）
            label_col = self._find_label_column(df)
            if label_col is None:
                print(f"    警告：找不到標籤欄位，跳過")
                continue

            # 統一清理欄位名稱（移除前後空格）
            df.columns = df.columns.str.strip()
            label_col = label_col.strip()

            # 彈性判定正常流量標籤（不分大小寫，支援 BENIGN, NORMAL, NORMAL TRAFFIC）
            normal_names = ["BENIGN", "NORMAL", "NORMAL TRAFFIC"]
            normal_mask = df[label_col].astype(str).str.strip().str.upper().isin(normal_names)

            normal = df[normal_mask].copy()
            attack = df[~normal_mask].copy()

            print(f"      正常: {len(normal):,}  攻擊: {len(attack):,}")
            all_normal.append(normal)
            all_attack.append(attack)

        if not all_normal or sum(len(d) for d in all_normal) == 0:
            raise ValueError("沒有成功載入任何正常流量資料")

        # 合併所有 CSV
        df_normal = pd.concat(all_normal, ignore_index=True)
        df_attack = pd.concat(all_attack, ignore_index=True) if all_attack else pd.DataFrame()

        # 限制樣本數（避免記憶體不足）
        if len(df_normal) > max_normal:
            df_normal = df_normal.sample(max_normal, random_state=42)
        if len(df_attack) > max_attack:
            df_attack = df_attack.sample(max_attack, random_state=42)

        print(f"  [CIC-IDS2017] 使用: 正常 {len(df_normal):,}  攻擊 {len(df_attack):,}")

        # 特徵前處理
        label_col = label_col.strip()
        X_normal_feat = self._preprocess(df_normal, label_col, fit=True)
        X_attack_feat = (self._preprocess(df_attack, label_col, fit=False)
                         if len(df_attack) > 0
                         else np.zeros((0, FEATURE_DIM), dtype=np.float32))

        # 轉換為影像
        X_normal = X_normal_feat.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
        X_attack = X_attack_feat.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)

        y = np.concatenate([
            np.zeros(len(X_normal), dtype=np.int32),
            np.ones(len(X_attack),  dtype=np.int32)
        ])

        print(f"  [CIC-IDS2017] 影像尺寸: {X_normal.shape[1:]}")
        return X_normal, X_attack, y

    def _preprocess(self, df: pd.DataFrame, label_col: str,
                    fit: bool = True) -> np.ndarray:
        """
        前處理流程：
          ① 移除標籤欄位
          ② 只保留數值型欄位
          ③ 處理無限值（inf -> 欄位最大有限值）與 NaN（-> 0）
          ④ MinMaxScaler 正規化
          ⑤ 補零或截斷至 1024 維
        """
        df = df.copy()

        # 移除標籤欄位
        drop_cols = [c for c in df.columns
                     if c.strip() in ["Label", "label"] or c == label_col]
        df = df.drop(columns=drop_cols, errors="ignore")

        # 只保留數值型欄位
        df = df.select_dtypes(include=[np.number])

        X = df.values.astype(np.float64)

        # 處理無限值：用該欄位的最大有限值取代 inf，再填 0 處理 NaN
        # [修正] 原版使用 np.finfo(float32).max (≈3.4×10³⁸) 取代 inf，
        #        導致 MinMaxScaler 後所有正常值都被壓縮到接近 0，
        #        影像幾乎全黑，CNN 無法學到有效特徵。
        #        現在改用每個欄位的最大有限值，確保 scaling 不被極端值扭曲。
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            inf_mask = np.isinf(col)
            if inf_mask.any():
                finite_vals = col[np.isfinite(col)]
                col_max = np.max(finite_vals) if len(finite_vals) > 0 else 0.0
                col_min = np.min(finite_vals) if len(finite_vals) > 0 else 0.0
                X[np.isposinf(X[:, col_idx]), col_idx] = col_max
                X[np.isneginf(X[:, col_idx]), col_idx] = col_min
        X = np.nan_to_num(X, nan=0.0)
        X = X.astype(np.float32)

        # 正規化
        # fit=True（正常資料）：fit 並 transform，clip 至 [0,1] 消除浮點誤差
        # fit=False（攻擊資料）：只 transform，不 clip，攻擊資料的特徵值若超出
        #   正常資料的分布範圍，會得到 <0 或 >1 的值，這是預期行為，
        #   使 CNN 對攻擊封包的重建誤差更大，有助於區分正常與攻擊。
        if fit:
            X = np.clip(
                self.scaler.fit_transform(X).astype(np.float32), 0.0, 1.0
            )
        else:
            X = self.scaler.transform(X).astype(np.float32)

        # 補零或截斷至 FEATURE_DIM
        n, f = X.shape
        if f >= FEATURE_DIM:
            X = X[:, :FEATURE_DIM]
        else:
            pad = np.zeros((n, FEATURE_DIM - f), dtype=np.float32)
            X = np.concatenate([X, pad], axis=1)

        return X

    def _find_label_column(self, df: pd.DataFrame):
        """在 DataFrame 中尋找標籤欄位（處理前後空格差異與不同命名）"""
        # 轉小寫，涵蓋常見的 Kaggle 清理版標籤名稱與帶空格的名稱
        possible_names = ["label", "class", "attack", "attack_type", "attack type", "target"]
        
        for col in df.columns:
            # 清除所有不可見字元、空白，並轉小寫
            clean_col = str(col).strip().lower()
            if clean_col in possible_names:
                return col
        return None


# ============================================================
# CIC-DDoS2019 資料集載入器
# ============================================================
class CICDDoS2019Loader(CICIDSLoader):
    """
    CIC-DDoS2019 資料集載入器

    資料集說明：
      CIC-IDS2017 的 DDoS 擴充版，包含 27 種 DDoS 攻擊類型，
      如 UDP、UDP-Lag、SYN、NetBIOS、MSSQL、LDAP、Portmap 等。
      CSV 格式與 CIC-IDS2017 完全相同（CICFlowMeter 輸出），
      因此繼承 CICIDSLoader 並只修改資料目錄預設值。

    檔案結構（下載後）：
      data/cicddos2019/
        03-11/                 各日期的 CSV 目錄
          DrDoS_DNS.csv
          DrDoS_LDAP.csv
          DrDoS_MSSQL.csv
          ...
        CSV-03-11-Traffic-For-ML-Detection/
          ...

    下載連結：https://www.unb.ca/cic/datasets/ddos-2019.html
    """

    def __init__(self, data_dir: str = "data/cicddos2019"):
        super().__init__(data_dir=data_dir)

    def load(self, csv_files: list = None,
             max_normal: int = 50000,
             max_attack: int = 30000) -> tuple:
        """
        覆寫 load()：DDoS2019 的正常標籤同樣為 'BENIGN'，
        掃描邏輯與 CIC-IDS2017 完全相同。
        """
        if csv_files is None:
            # DDoS2019 的 CSV 可能在子目錄中，遞迴搜尋
            csv_files = sorted(
                glob.glob(os.path.join(self.data_dir, "**", "*.csv"),
                          recursive=True)
            )

        print(f"  [CIC-DDoS2019] 掃描到 {len(csv_files)} 個 CSV 檔案")
        return super().load(csv_files=csv_files,
                            max_normal=max_normal,
                            max_attack=max_attack)


# ============================================================
# 統一介面：DatasetFactory
# ============================================================
class DatasetFactory:
    """
    資料集工廠，提供統一的載入介面

    使用方式：
        # 自動模擬資料（開發測試用）
        X_normal, X_attack, y = DatasetFactory.load("simulate")

        # 真實資料集
        X_normal, X_attack, y = DatasetFactory.load(
            "nslkdd", data_dir="data/nslkdd"
        )
        X_normal, X_attack, y = DatasetFactory.load(
            "cicids2017", data_dir="data/cicids2017"
        )
        X_normal, X_attack, y = DatasetFactory.load(
            "cicddos2019", data_dir="data/cicddos2019"
        )

    支援的 dataset_name：
        "simulate"    使用模擬封包（預設，無需下載）
        "nslkdd"      NSL-KDD
        "cicids2017"  CIC-IDS2017
        "cicddos2019" CIC-DDoS2019
    """

    SUPPORTED = ["simulate", "nslkdd", "cicids2017", "cicddos2019"]

    @staticmethod
    def load(dataset_name: str,
             data_dir: str = None,
             **kwargs) -> tuple:
        """
        載入指定資料集（支援模糊名稱匹配）

        Args:
            dataset_name: 資料集名稱或包含關鍵字的字串
            data_dir    : 資料集目錄
            **kwargs    : 額外參數

        Returns:
            (X_normal, X_attack, y)
        """
        p = dataset_name.lower().strip()

        if "simulate" in p:
            return DatasetFactory._load_simulate(**kwargs)

        elif "nsl" in p or "kdd" in p:
            loader = NSLKDDLoader(data_dir or "data/nslkdd")
            return loader.load(**kwargs)

        elif "ddos" in p:
            loader = CICDDoS2019Loader(data_dir or "data/cicddos2019")
            return loader.load(**kwargs)

        elif "cic" in p or "ids" in p:
            loader = CICIDSLoader(data_dir or "data/cicids2017")
            return loader.load(**kwargs)

        else:
            raise ValueError(
                f"無法辨識的資料集名稱：{dataset_name}。 "
                f"請確保名稱中包含 'nsl', 'kdd', 'cic', 'ids', 'ddos' 或 'simulate'。"
            )

    @staticmethod
    def _load_simulate(n_normal: int = 500, n_attack: int = 300,
                       **kwargs) -> tuple:
        """
        生成模擬資料（不需要下載任何資料集，供開發測試使用）

        正常流量特徵：模擬 HTTP/DNS/SSH 連線的典型統計值（高 entropy，多樣化）
        攻擊流量特徵：模擬 SYN Flood/Port Scan（低 entropy，均勻分布）
        """
        print(f"  [DatasetFactory] 使用模擬資料（{n_normal} 正常 + {n_attack} 攻擊）")
        # [修正] 使用局部 Generator，避免 np.random.seed() 污染全域隨機狀態
        rng = np.random.default_rng(42)

        # 正常流量：41 維模擬特徵，中等數值、有變化
        normal_feat = rng.beta(2, 5, size=(n_normal, 41)).astype(np.float32)
        # 加入一些高數值特徵（模擬 src_bytes, dst_bytes）
        normal_feat[:, 4:6] = rng.beta(5, 2, size=(n_normal, 2)).astype(np.float32)

        # 攻擊流量：集中在低數值（SYN Flood = 小封包，Port Scan = 短連線）
        attack_feat = rng.beta(0.5, 5, size=(n_attack, 41)).astype(np.float32)
        attack_feat[:, 0] = 0.0    # duration = 0（SYN 無連線時間）
        attack_feat[:, 4] = 0.0    # src_bytes = 0（無 Payload）

        # 補零至 1024，reshape 32x32
        def to_images(feat):
            n, f = feat.shape
            pad = np.zeros((n, FEATURE_DIM - f), dtype=np.float32)
            return np.concatenate([feat, pad], axis=1).reshape(n, 32, 32)

        X_normal = to_images(normal_feat)
        X_attack = to_images(attack_feat)
        y = np.concatenate([
            np.zeros(n_normal, dtype=np.int32),
            np.ones(n_attack,  dtype=np.int32)
        ])

        return X_normal, X_attack, y

    @staticmethod
    def save_as_npy(X_normal: np.ndarray, X_attack: np.ndarray,
                    y: np.ndarray, output_dir: str = "output/dataset"):
        """
        將載入的資料集儲存為 .npy 檔案，供 Trainer 使用

        Args:
            X_normal  : 正常流量影像矩陣
            X_attack  : 攻擊流量影像矩陣
            y         : 標籤向量
            output_dir: 輸出目錄

        Returns:
            dict: {"normal": path, "attack": path, "X_all": path, "y_all": path}
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        normal_path = os.path.join(output_dir, "X_normal.npy")
        attack_path = os.path.join(output_dir, "X_attack.npy")
        x_all_path  = os.path.join(output_dir, "X_all.npy")
        y_all_path  = os.path.join(output_dir, "y_all.npy")

        np.save(normal_path, X_normal)
        np.save(attack_path, X_attack)

        X_all = np.concatenate([X_normal, X_attack], axis=0)
        np.save(x_all_path, X_all)
        np.save(y_all_path, y)

        paths = {
            "normal": normal_path,
            "attack": attack_path,
            "X_all":  x_all_path,
            "y_all":  y_all_path,
        }

        for k, v in paths.items():
            arr = np.load(v)
            print(f"  [DatasetFactory] {k}: {v}  shape={arr.shape}")

        return paths
