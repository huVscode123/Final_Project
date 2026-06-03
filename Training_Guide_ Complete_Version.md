# 模型訓練完整技術指南
## 視覺化網路攻擊自動化分析平台 — CNN Autoencoder 訓練流程深度解析

---

## 目錄

1. [整體訓練架構概覽](#1-整體訓練架構概覽)
2. [環境準備與套件安裝](#2-環境準備與套件安裝)
3. [資料準備階段](#3-資料準備階段)
   - 3.1 資料集下載
   - 3.2 DatasetFactory 統一載入介面
   - 3.3 各資料集載入器詳解
   - 3.4 特徵向量轉影像機制
   - 3.5 資料儲存為 .npy
4. [路徑 A：純非監督訓練（Unsupervised）](#4-路徑-a純非監督訓練)
   - 4.1 Trainer 類別詳解
   - 4.2 PacketDataset 與 DataLoader
   - 4.3 EarlyStopping 機制
   - 4.4 訓練主迴圈
   - 4.5 完整執行指令
5. [路徑 B：半監督訓練（通用版）](#5-路徑-b半監督訓練通用版)
   - 5.1 MarginLoss 設計
   - 5.2 SemiSupervisedDataLoader 資料分配
   - 5.3 Phase 1：無監督預訓練
   - 5.4 Phase 2：半監督微調
   - 5.5 完整執行指令
6. [路徑 C：資料集專用半監督腳本](#6-路徑-c資料集專用半監督腳本)
   - 6.1 CIC-DDoS2019 專用訓練
   - 6.2 CICIDS2017 專用訓練（含 Latent 分離策略）
   - 6.3 NSL-KDD 專用訓練（含 VAE 風格）
7. [資料擴增（選用）](#7-資料擴增選用)
8. [閾值計算與調校](#8-閾值計算與調校)
9. [模型評估與輸出檔案](#9-模型評估與輸出檔案)
10. [訓練流程一覽與快速指令](#10-訓練流程一覽與快速指令)
11. [常見問題與疑難排解](#11-常見問題與疑難排解)

---

## 1. 整體訓練架構概覽

本系統提供三條訓練路徑，對應不同的資料可用性場景：

```
原始資料
    │
    ├── PCAP 封包流量 ──→ DatasetBuilder ──────────────┐
    │                    (packet_visualizer.py)        │
    └── CSV 特徵資料 ───→ DatasetFactory ──────────────┤
         (NSL-KDD /                                   ▼
          CIC-IDS2017 /                     .npy 影像矩陣
          CIC-DDoS2019)                    (N × 32 × 32)
                                                      │
                          ┌───────────────────────────┤
                          │                           │
                          ▼                           ▼
              路徑 A：純非監督              路徑 B/C：半監督
              run_training.py             run_semi_supervised.py
              Trainer                     SemiSupervisedTrainer
                │                                     │
                │  Phase 1: MSE Loss                  │  Phase 1: MSE (正常)
                │  (只用正常流量)                      │  Phase 2: MSE + Margin
                │                                     │          (正常 + 攻擊)
                ▼                                     ▼
           best_model.pt                        best_model.pt
                │                                     │
                └──────────────┬──────────────────────┘
                               ▼
                    ThresholdTuner / 百分位數法
                               │
                               ▼
                        threshold.json
                               │
                               ▼
                    AnomalyDetector / AnomalyScorer
                    (eval_result.json, ROC curve...)
```

---

## 2. 環境準備與套件安裝

### 2.1 Python 環境建立

```bash
# 建立虛擬環境（建議）
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 升級 pip
pip install --upgrade pip
```

### 2.2 安裝核心依賴

```bash
# CPU 版本（無 GPU）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU 版本（CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# GPU 版本（CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

```bash
# 其他依賴
pip install numpy pandas scikit-learn matplotlib Pillow scapy scipy
```

### 2.3 驗證環境

```bash
python -c "
import torch
import numpy as np
import sklearn
print(f'PyTorch  : {torch.__version__}')
print(f'CUDA 可用 : {torch.cuda.is_available()}')
print(f'NumPy    : {np.__version__}')
print(f'Sklearn  : {sklearn.__version__}')
"
```

---

## 3. 資料準備階段

### 3.1 資料集下載

系統支援三種公開資料集，下載後放置在對應目錄：

| 資料集 | 下載網址 | 放置路徑 | 格式 |
|---|---|---|---|
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html | `data/nslkdd/` | `.txt` / `.arff` |
| CIC-IDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html | `data/cicids2017/` | `.csv` |
| CIC-DDoS2019 | https://www.unb.ca/cic/datasets/ddos-2019.html | `data/cicddos2019/` | `.csv`（子目錄） |

**NSL-KDD 目錄結構：**
```
data/nslkdd/
    KDDTrain+.txt       ← 主要訓練集（125,973 筆）
    KDDTest+.txt        ← 測試集（22,544 筆）
    KDDTrain+_20Percent.txt  ← 20% 子集（快速測試用）
```

**CIC-IDS2017 目錄結構：**
```
data/cicids2017/
    Monday-WorkingHours.pcap_ISCX.csv     ← 正常流量（529,918 筆）
    Tuesday-WorkingHours.pcap_ISCX.csv    ← Brute Force
    Wednesday-WorkingHours.pcap_ISCX.csv  ← DoS / Heartbleed
    Thursday-WorkingHours.pcap_ISCX.csv   ← Web Attack
    Friday-WorkingHours.pcap_ISCX.csv     ← PortScan / DDoS
```

**CIC-DDoS2019 目錄結構（遞迴掃描）：**
```
data/cicddos2019/
    03-11/
        DrDoS_DNS.csv
        DrDoS_LDAP.csv
        DrDoS_MSSQL.csv
        ...（共 27 種 DDoS 類型）
```

---

### 3.2 DatasetFactory 統一載入介面

`dataset_loader.py` 提供 `DatasetFactory` 作為統一入口，用一行程式切換三種資料集：

```python
# dataset_loader.py - DatasetFactory 靜態方法
@staticmethod
def load(dataset_name: str, data_dir: str = None, **kwargs) -> tuple:
    p = dataset_name.lower().strip()

    if "simulate" in p:
        return DatasetFactory._load_simulate(**kwargs)   # 模擬資料，無需下載
    elif "nsl" in p or "kdd" in p:
        loader = NSLKDDLoader(data_dir or "data/nslkdd")
        return loader.load(**kwargs)
    elif "ddos" in p:
        loader = CICDDoS2019Loader(data_dir or "data/cicddos2019")
        return loader.load(**kwargs)
    elif "cic" in p or "ids" in p:
        loader = CICIDSLoader(data_dir or "data/cicids2017")
        return loader.load(**kwargs)
```

所有 loader 的回傳格式統一為：
```python
(X_normal, X_attack, y)
# X_normal: shape=(N, 32, 32) float32  ← 正常流量影像矩陣
# X_attack: shape=(M, 32, 32) float32  ← 攻擊流量影像矩陣
# y:        shape=(N+M,) int32         ← 標籤（0=正常，1=攻擊）
```

---

### 3.3 各資料集載入器詳解

#### NSLKDDLoader

NSL-KDD 的 41 個特徵中含有 3 個**類別型欄位**（`protocol_type`、`service`、`flag`），需先用 `LabelEncoder` 轉換：

```python
# dataset_loader.py - NSLKDDLoader._preprocess()
def _preprocess(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
    # ① 類別特徵 LabelEncoder 轉數值
    for col in self.CATEGORICAL_FEATURES:  # ["protocol_type", "service", "flag"]
        if col not in self.encoders:
            self.encoders[col] = LabelEncoder()
            # 預先 fit 已知類別集合，避免 transform 時遇到 unseen label
            known_vals = ["tcp", "udp", "icmp", "http", "SF", "S0", "REJ", ...]
            self.encoders[col].fit(known_vals + list(df[col].unique()))
        # 未知類別設為 "SF"（最常見的正常 Flag）
        df[col] = df[col].map(lambda x: x if x in known else "SF")
        df[col] = self.encoders[col].transform(df[col])

    # ② 移除標籤欄位（label, difficulty）
    df = df.drop(columns=["label", "difficulty"], errors="ignore")

    # ③ 處理 NaN / inf，轉為 float32
    X = df.values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ④ MinMaxScaler 正規化（fit 只對正常流量執行）
    if fit:
        X = self.scaler.fit_transform(X)   # 用正常流量定義 [0,1] 空間
    else:
        X = self.scaler.transform(X)       # 攻擊資料直接 transform（可能超出 [0,1]）
    return X.astype(np.float32)
```

> **關鍵設計**：正常流量 `fit` Scaler，攻擊流量只 `transform` 不 `clip`。
> 攻擊流量的特徵若超出正常流量的分布範圍，會得到 < 0 或 > 1 的值，這是**刻意**的設計，讓 CNN 對攻擊封包產生更大的重建誤差。

#### CICIDSLoader（亦為 CICDDoS2019Loader 的父類別）

CIC 系列資料集的標籤欄位名稱在不同版本中有**前後空格差異**，載入器使用容錯機制處理：

```python
# dataset_loader.py - CICIDSLoader._find_label_column()
def _find_label_column(self, df: pd.DataFrame):
    possible_names = ["label", "class", "attack", "attack_type", "target"]
    for col in df.columns:
        clean_col = str(col).strip().lower()  # 去除空格並轉小寫
        if clean_col in possible_names:
            return col
    return None
```

正常標籤的判斷也採用容錯匹配（涵蓋 Kaggle 清理版）：
```python
# 支援 BENIGN / NORMAL / NORMAL TRAFFIC 等多種命名
normal_names = ["BENIGN", "NORMAL", "NORMAL TRAFFIC"]
normal_mask = df[label_col].astype(str).str.strip().str.upper().isin(normal_names)
```

CICDDoS2019Loader 繼承 CICIDSLoader，只覆寫 `load()` 中的目錄掃描方式（改為遞迴搜尋子目錄）：
```python
# 遞迴掃描子目錄中的所有 CSV
csv_files = sorted(
    glob.glob(os.path.join(self.data_dir, "**", "*.csv"), recursive=True)
)
```

---

### 3.4 特徵向量轉影像機制

所有 CSV 資料集共用同一套「特徵向量 → 32×32 影像」轉換流程：

```
原始特徵向量 (N, F)
        │
        ├── F > 1024 → 截斷至前 1024 維
        ├── F < 1024 → 在右方補零至 1024 維
        └── F = 1024 → 不變
        │
        ▼
(N, 1024) float32
        │
        ▼  reshape
(N, 32, 32) float32      ← 每個 32×32 影像即一筆流量樣本
```

```python
# dataset_loader.py - NSLKDDLoader._features_to_images()
@staticmethod
def _features_to_images(features: np.ndarray) -> np.ndarray:
    n, f = features.shape
    IMAGE_SIZE  = 32        # 影像邊長
    FEATURE_DIM = 32 * 32   # 1024

    if f > FEATURE_DIM:
        features = features[:, :FEATURE_DIM]   # 截斷
    elif f < FEATURE_DIM:
        pad = np.zeros((n, FEATURE_DIM - f), dtype=np.float32)
        features = np.concatenate([features, pad], axis=1)  # 補零

    return features.reshape(n, IMAGE_SIZE, IMAGE_SIZE)
```

> **為何轉成影像？**
> NSL-KDD 有 41 個特徵、CIC 系列有 78 個特徵，補零後都成為 1024 維。
> 前段（非零區域）包含實際特徵，後段（補零區域）為純黑色。
> CNN 能夠從特徵的**空間排列**與**數值分布**學習正常流量的統計模式，
> 而非逐特徵線性比對，對高維稀疏資料的效果優於傳統 SVM/RF。

---

### 3.5 資料儲存為 .npy

載入完成後，`DatasetFactory.save_as_npy()` 將資料序列化至磁碟：

```python
# dataset_loader.py - DatasetFactory.save_as_npy()
@staticmethod
def save_as_npy(X_normal, X_attack, y, output_dir="output/dataset"):
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/X_normal.npy", X_normal)   # (N, 32, 32) 正常
    np.save(f"{output_dir}/X_attack.npy", X_attack)   # (M, 32, 32) 攻擊
    np.save(f"{output_dir}/X_all.npy",
            np.concatenate([X_normal, X_attack], axis=0))
    np.save(f"{output_dir}/y_all.npy", y)
```

執行後在 `output/dataset_<name>/` 目錄下生成：
```
output/dataset_cicids2017/
    X_normal.npy     ← 正常流量影像（訓練用）
    X_attack.npy     ← 攻擊流量影像（閾值評估用）
    X_all.npy        ← 全部樣本（評估用）
    y_all.npy        ← 標籤向量
```

---

## 4. 路徑 A：純非監督訓練

適用場景：**沒有任何已標記的攻擊樣本**，完全依賴正常流量學習。

入口腳本：`core/run_training.py` → 呼叫 `core/trainer.py` 的 `Trainer` 類別。

### 4.1 Trainer 類別詳解

`Trainer.__init__()` 完成以下初始化：

```python
# trainer.py - Trainer.__init__()
DEFAULT_CONFIG = {
    "latent_dim":    64,      # Encoder 壓縮後的潛在向量維度
    "batch_size":    32,      # 每批次樣本數
    "epochs":        100,     # 最大訓練輪數（搭配 EarlyStopping 可提早停止）
    "learning_rate": 1e-3,    # Adam 初始學習率
    "weight_decay":  1e-5,    # L2 正則化係數（防止過擬合）
    "val_split":     0.2,     # 20% 資料作為驗證集
    "patience":      15,      # EarlyStopping 等待輪數
    "min_delta":     1e-6,    # 最小改善量，低於此值視為未改善
    "lr_patience":   7,       # 學習率調度器等待輪數
    "lr_factor":     0.5,     # 學習率降低倍率（新 LR = 舊 LR × 0.5）
    "threshold_percentile": 95,  # 閾值百分位數
}

# 損失函數：MSE（均方誤差）
# 選 MSE 而非 MAE 的原因：MSE 對大誤差懲罰更重，
# 異常封包通常有較大偏差，使用 MSE 使誤差差距更顯著
self.criterion = nn.MSELoss()

# 優化器：Adam（自適應學習率，對稀疏梯度效果好）
self.optimizer = optim.Adam(
    self.model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)

# 學習率調度器：當驗證損失連續 lr_patience 輪不改善，LR × lr_factor
self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer, mode="min",
    factor=config["lr_factor"],
    patience=config["lr_patience"],
)
```

---

### 4.2 PacketDataset 與 DataLoader

`PacketDataset` 將 `.npy` 矩陣包裝為 PyTorch 資料集，同時處理維度擴展：

```python
# trainer.py - PacketDataset.from_npy()
@classmethod
def from_npy(cls, npy_path: str) -> "PacketDataset":
    data = np.load(npy_path).astype(np.float32)  # (N, 32, 32)
    data = data[:, np.newaxis, :, :]              # (N, 1, 32, 32) ← 加入 Channel 維度
    tensor = torch.from_numpy(data)               # 轉為 PyTorch Tensor
    return cls(tensor)
```

> **為何需要加 Channel 維度？**
> PyTorch 的 `Conv2d` 輸入格式為 `(Batch, Channel, Height, Width)`。
> 封包影像為灰階（單通道），因此 Channel=1，需手動加入 `np.newaxis`。

`load_data()` 使用固定隨機種子切分訓練/驗證集，確保可重現性：

```python
# trainer.py - Trainer.load_data()
def load_data(self, npy_path: str):
    dataset = PacketDataset.from_npy(npy_path)
    n_total = len(dataset)
    n_val   = int(n_total * self.config["val_split"])   # 20%
    n_train = n_total - n_val                           # 80%

    self.train_set, self.val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)     # 固定種子 → 可重現
    )

    self.train_loader = DataLoader(
        self.train_set,
        batch_size=self.config["batch_size"],
        shuffle=True,       # 訓練時打亂，避免模型記憶資料順序
        num_workers=0,      # Windows 需設 0；Linux 可設 2~4 加速
        pin_memory=(self.device.type == "cuda"),  # GPU 訓練時加速資料傳輸
    )
```

---

### 4.3 EarlyStopping 機制

`EarlyStopping` 在每個 epoch 結束後監控驗證損失，若連續 `patience` 輪沒有改善，自動停止訓練並**恢復最佳模型權重**：

```python
# trainer.py - EarlyStopping.__call__()
def __call__(self, val_loss: float, model: nn.Module) -> bool:
    if val_loss < self.best_loss - self.min_delta:
        # ✅ 有改善 → 儲存當前最佳模型，計數器歸零
        self.best_loss = val_loss
        self.counter   = 0
        torch.save(model.state_dict(), self.path)  # 儲存最佳權重
    else:
        # ❌ 無改善 → 計數器 +1
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True  # 觸發停止
    return self.early_stop
```

---

### 4.4 訓練主迴圈

`Trainer.train()` 的每個 epoch 執行以下步驟：

```
每個 Epoch：
  ① 訓練階段（model.train()）
      for batch in train_loader:
          x = batch[0].to(device)     # 移至 GPU/CPU
          x_hat, z = model(x)         # 前向傳播：輸入 → 重建
          loss = MSE(x_hat, x)        # 計算重建誤差
          optimizer.zero_grad()       # 清除上一步梯度
          loss.backward()             # 反向傳播：計算梯度
          clip_grad_norm_(model, 1.0) # 梯度裁剪（防止梯度爆炸）
          optimizer.step()            # Adam 更新權重
  
  ② 驗證階段（model.eval() + torch.no_grad()）
      for batch in val_loader:
          x_hat, _ = model(x)
          val_loss += MSE(x_hat, x)
  
  ③ 學習率調度：scheduler.step(val_loss)
     若連續 7 輪 val_loss 未改善 → LR × 0.5
  
  ④ EarlyStopping 檢查：
     若連續 15 輪 val_loss 未改善 → 停止訓練，載回最佳模型
```

---

### 4.5 完整執行指令（路徑 A）

**Step 1：載入資料並儲存 .npy（只需執行一次）**

```bash
# 使用模擬資料（快速測試，無需下載資料集）
python core/run_training.py --dataset simulate

# 使用 NSL-KDD
python core/run_training.py \
    --dataset nslkdd \
    --data-dir data/nslkdd

# 使用 CIC-IDS2017
python core/run_training.py \
    --dataset cicids2017 \
    --data-dir data/cicids2017 \
    --max-normal 50000 \
    --max-attack 30000

# 使用 CIC-DDoS2019
python core/run_training.py \
    --dataset cicddos2019 \
    --data-dir data/cicddos2019 \
    --max-normal 60000 \
    --max-attack 30000
```

**Step 2：自訂超參數訓練**

```bash
python core/run_training.py \
    --dataset cicids2017 \
    --data-dir data/cicids2017 \
    --epochs 150 \        # 最大訓練輪數（EarlyStopping 可提早停止）
    --latent 32 \         # 潛在向量維度（越小 → 壓縮越強 → 重建誤差越大）
    --batch 64 \          # 批次大小（記憶體不足時縮小）
    --lr 0.0005 \         # 初始學習率
    --patience 20 \       # EarlyStopping 等待輪數
    --pct 95 \            # 閾值百分位數
    --output output/model_cicids2017_custom
```

**Step 3：只做評估（跳過訓練，載入已訓練模型）**

```bash
python core/run_training.py \
    --dataset cicids2017 \
    --data-dir data/cicids2017 \
    --eval-only \
    --model output/model_cicids2017/best_model_cicids2017.pt \
    --latent 32
```

**`run_training.py` 內部執行的完整步驟（Source Code 對應）：**

```python
# run_training.py - main() 各 Step 對應

# [Step 1] DatasetFactory.load() → 載入 CSV → 特徵正規化 → 轉 32×32 影像
X_normal, X_attack, y = DatasetFactory.load(
    args.dataset, data_dir=args.data_dir, **load_kwargs
)

# [Step 2] DatasetFactory.save_as_npy() → 儲存 .npy 至磁碟
paths = DatasetFactory.save_as_npy(X_normal, X_attack, y, dataset_dir)

# [Step 3] Trainer() → 載入 .npy → train()（含 EarlyStopping）
trainer = Trainer(config=config, output_dir=args.output, model_name=model_name)
trainer.load_data(normal_npy)
trainer.train()

# [Step 4] compute_threshold() → 計算第 95 百分位閾值
threshold = trainer.compute_threshold(normal_npy, args.pct)

# [Step 5] plot_reconstruction_samples() → 繪製重建對比圖
trainer.plot_reconstruction_samples(normal_npy)

# [Step 6] AnomalyScorer.evaluate() → 計算 F1/AUC/混淆矩陣
scorer = AnomalyScorer(trainer.model, threshold=threshold)
results = scorer.evaluate(X_test, y_test, output_dir=args.output)
scorer.plot_roc_curve(errors_normal, errors_attack, args.output)
scorer.plot_score_distribution(errors_normal, errors_attack, args.output)
```

---

## 5. 路徑 B：半監督訓練（通用版）

適用場景：**有少量已標記的攻擊樣本**（即使只有全部攻擊資料的 10~20%），可顯著提升偵測效能。

入口腳本：`core/run_semi_supervised.py` → 呼叫 `core/semi_supervised_trainer.py` 的 `SemiSupervisedTrainer`。

### 5.1 MarginLoss 設計

`MarginLoss` 是半監督訓練的核心損失函數，設計目標是**推高攻擊封包的重建誤差**：

```python
# semi_supervised_trainer.py - MarginLoss.forward()
class MarginLoss(nn.Module):
    """
    數學式：L_margin = mean( max(0, margin - reconstruction_error_attack) )

    直觀解釋：
        若攻擊樣本誤差 < margin → 產生正梯度，推動模型「重建得更差」
        若攻擊樣本誤差 ≥ margin → 損失為 0，停止更新（已達目標）
    """
    def __init__(self, margin: float = 0.05):
        super().__init__()
        self.margin = margin

    def forward(self, errors_attack: torch.Tensor) -> torch.Tensor:
        # errors_attack: shape=(N,)，每個攻擊樣本的重建 MSE
        loss = torch.clamp(self.margin - errors_attack, min=0.0)
        return loss.mean()
```

**Phase 2 組合損失函數：**
```
L_total = α × L_recon_normal + β × L_margin_attack

其中：
  L_recon_normal  = MSE(x_normal, x̂_normal)
                  → 正常流量要重建得準（誤差低）

  L_margin_attack = mean(max(0, margin - MSE(x_attack, x̂_attack)))
                  → 攻擊流量要重建得差（誤差高於 margin）

  α = 1.0（正常損失權重，通常固定）
  β = 0.3 ~ 0.8（邊界損失權重，β 過大會損壞重建能力）
  margin = 0.05（攻擊誤差目標值，通常設為正常誤差均值的 2~5 倍）
```

---

### 5.2 SemiSupervisedDataLoader 資料分配

```python
# semi_supervised_trainer.py - SemiSupervisedDataLoader.__init__()

# 正常流量：80% 訓練，20% 驗證
n_val   = max(1, int(len(X_normal) * val_split))   # 20%
n_train = len(X_normal) - n_val                    # 80%
normal_ds = TensorDataset(normal_tensor)
self.train_normal_ds, self.val_normal_ds = random_split(normal_ds, [n_train, n_val])

# 攻擊流量：只取 attack_ratio（預設 20%）用於微調
# 模擬真實場景中「標記資料稀缺」的情況
n_attack_labeled = max(1, int(len(X_attack) * attack_ratio))
attack_indices   = rng.choice(len(X_attack), n_attack_labeled, replace=False)
self.attack_ds   = TensorDataset(attack_tensor[attack_indices])
```

**Phase 2 微調時的批次策略（攻擊樣本少 → 循環使用）：**
```python
# semi_supervised_trainer.py - _train_one_epoch_semi()
# 攻擊 DataLoader 的 batch_size = batch_size // 4（避免攻擊批次佔比過大）
attack_loader = DataLoader(attack_ds, batch_size=max(1, batch_size // 4), shuffle=True)

# 每個正常批次匹配一個攻擊批次，攻擊資料用完後循環重複
attack_iter = iter(attack_loader)
for (x_normal,) in normal_loader:
    try:
        (x_attack,) = next(attack_iter)
    except StopIteration:
        attack_iter = iter(attack_loader)   # 循環
        (x_attack,) = next(attack_iter)
```

---

### 5.3 Phase 1：無監督預訓練

```python
# semi_supervised_trainer.py - SemiSupervisedTrainer.pretrain()

# 優化器：Adam，使用正常學習率
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# 學習率調度：ReduceLROnPlateau（與 Trainer 相同策略）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
    factor=config["lr_factor"], patience=config["lr_patience"])

# EarlyStopping 監控驗證損失，最佳模型儲存為 pretrained_model.pt
early_stop = EarlyStopping(patience=config["patience"], path="pretrained_model.pt")

for epoch in range(1, config["pretrain_epochs"] + 1):
    # 訓練步驟（只用正常流量）
    train_loss = _train_one_epoch_unsupervised(train_loader, optimizer)
    val_loss   = _validate_epoch(val_loader)

    scheduler.step(val_loss)
    if early_stop(val_loss, model):
        print(f"EarlyStopping at epoch {epoch}")
        break

# 訓練結束後，載回最佳預訓練模型權重
model.load_state_dict(torch.load("pretrained_model.pt"))
```

---

### 5.4 Phase 2：半監督微調

Phase 2 **使用更小的學習率**（Phase 1 的 1/10），避免破壞預訓練學到的正常流量知識：

```python
# semi_supervised_trainer.py - SemiSupervisedTrainer.finetune()

# ⚠️ Phase 2 使用較小學習率
optimizer = optim.Adam(
    model.parameters(),
    lr=config["learning_rate"] * 0.1,   # Phase 1 的 1/10
    weight_decay=config["weight_decay"],
)

# Phase 2 使用 CosineAnnealingLR（更平滑的衰減曲線）
# Phase 1 用 ReduceLROnPlateau（基於驗證損失的觸發式降低）
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["finetune_epochs"],
    eta_min=1e-6,
)

for epoch in range(1, config["finetune_epochs"] + 1):
    # 組合損失訓練
    loss_n, loss_a, loss_total = _train_one_epoch_semi(
        normal_loader, attack_loader, optimizer
    )

    # 每個 epoch 完整計算：
    # loss_n     = L_recon_normal（正常流量重建誤差，越低越好）
    # loss_a     = L_margin_attack（邊界損失，攻擊誤差推高進度）
    # loss_total = α×loss_n + β×loss_a

    scheduler.step()     # CosineAnnealingLR 不需要 val_loss 參數

    # 僅根據驗證集正常流量的重建損失判斷是否保存最佳模型
    if val_loss < best_val - min_delta:
        torch.save(model.state_dict(), "best_model.pt")
```

---

### 5.5 完整執行指令（路徑 B）

**快速測試（模擬資料，無需下載）：**

```bash
python core/run_semi_supervised.py
# 預設：simulate 資料集，50 pretrain + 30 finetune epochs
```

**CIC-IDS2017 完整訓練：**

```bash
python core/run_semi_supervised.py \
    --dataset cicids2017 \
    --data-dir data/cicids2017 \
    --pretrain-epochs 70 \
    --finetune-epochs 30 \
    --batch 32 \
    --latent 32 \
    --alpha 1.0 \
    --beta 0.5 \
    --margin 0.05 \
    --attack-ratio 0.1 \          # 10% 攻擊樣本用於微調
    --threshold-method optimal \  # 用標記攻擊樣本找最佳 F1 閾值
    --output output/model_semi_cicids2017
```

**含無監督基準對比（加 `--compare-unsupervised`）：**

```bash
python core/run_semi_supervised.py \
    --dataset cicids2017 \
    --data-dir data/cicids2017 \
    --pretrain-epochs 70 \
    --finetune-epochs 30 \
    --compare-unsupervised    # 同時訓練純非監督版本，生成對比圖
```

**含資料擴增（加 `--augment`）：**

```bash
python core/run_semi_supervised.py \
    --dataset nslkdd \
    --data-dir data/nslkdd \
    --augment \               # 啟用資料擴增，正常訓練集擴增 2 倍
    --pretrain-epochs 70 \
    --finetune-epochs 30
```

**`run_semi_supervised.py` 內部 7 個步驟（Source Code 對應）：**

```python
# run_semi_supervised.py - main() 各 Step

# [Step 1] DatasetFactory.load() → 載入資料
X_normal, X_attack, _ = DatasetFactory.load(args.dataset, data_dir=args.data_dir)

# 手動切分訓練/測試集（80% 正常 + 50% 攻擊作訓練，其餘測試）
X_train_normal, X_test_normal = split_80_20(X_normal)
X_train_attack_pool, X_test_attack = split_50_50(X_attack)

# 從攻擊池中取 attack_ratio 比例放入訓練集
X_train_attack = X_train_attack_pool[:int(len(X_train_normal) * attack_ratio)]

# [Step 2] DataAugmentor.augment_batch() → 資料擴增（可選）
if args.augment:
    X_train_normal = DataAugmentor().augment_batch(X_train_normal, multiplier=2)

# [Step 3/4] SemiSupervisedTrainer.train_full() → Phase1 + Phase2 + 閾值
config = {...}
semi_trainer = SemiSupervisedTrainer(config, output_dir=args.output)
threshold = semi_trainer.train_full(
    X_normal=X_train_normal,
    X_attack=X_train_attack,
    threshold_method=args.threshold_method   # "percentile" 或 "optimal"
)

# [Step 5] evaluate_model() → 計算 Precision/Recall/F1/AUC
results = evaluate_model(semi_trainer.model, X_test_normal, X_test_attack, threshold)

# [Step 6] 對比實驗（可選，加 --compare-unsupervised）
if args.compare_unsupervised:
    unsup_trainer = Trainer(...)
    unsup_trainer.from_numpy(X_train_normal)
    unsup_trainer.train()
    # ... 比較並繪圖

# [Step 7] 儲存評估報告
with open("semi_evaluation_report.json", "w") as f:
    json.dump(results, f, indent=4)
```

---

## 6. 路徑 C：資料集專用半監督腳本

除通用版外，系統另提供三個針對各資料集特性精細調整的**資料集專用腳本**。

### 6.1 CIC-DDoS2019 專用訓練

腳本：`core/semi_supervised_cicddos2019.py`

```bash
python core/semi_supervised_cicddos2019.py \
    --data-dir data/cic-ddos2019 \
    --output output/model_cicddos2019 \
    --pretrain-epochs 70 \
    --finetune-epochs 30 \
    --latent 32 \
    --batch 32 \
    --margin 0.05 \
    --alpha 1.0 \
    --beta 0.5 \
    --max-normal 60000 \
    --max-attack 30000
```

輸出：`output/model_cicddos2019/semi_supervised_cicddos2019.pt`

---

### 6.2 CICIDS2017 專用訓練（含 Latent Space 分離策略）

腳本：`core/semi_supervised_cicids2017.py`

```bash
python core/semi_supervised_cicids2017.py \
    --data-dir data/cicids2017 \
    --output output/model_cicids2017 \
    --pretrain-epochs 70 \
    --finetune-epochs 30 \
    --latent 48 \          # 較大的 latent_dim 配合 Latent 分離策略
    --batch 32 \
    --margin 0.05 \
    --alpha 1.0 \
    --beta 0.6 \           # 較高的 β（攻擊 Latent 與正常 Latent 拉開距離）
    --gamma 0.1            # 額外的 Latent 分離損失權重（CICIDS2017 專用）
```

**CICIDS2017 特有的 gamma 參數：** 在 Phase 2 的組合損失中加入「潛在空間分離損失」：
```
L_total = α×L_recon_normal + β×L_margin_attack + γ×L_latent_separation

L_latent_separation = -dist(z_normal_mean, z_attack_mean)
# 最大化正常與攻擊樣本在 Latent Space 的距離
```

輸出：`output/model_cicids2017/semi_supervised_cicids2017.pt`

---

### 6.3 NSL-KDD 專用訓練（含 VAE 風格）

腳本：`core/semi_supervised_nslkdd.py`

```bash
python core/semi_supervised_nslkdd.py \
    --train data/nsl-kdd/KDDTrain+.txt \
    --test  data/nsl-kdd/KDDTest+.txt \
    --output output/model_nslkdd \
    --image-size 16 \      # NSL-KDD 使用 16×16（41 特徵較少，不需要 32×32）
    --latent 16 \          # 搭配 16×16 影像的較小潛在維度
    --pretrain-epochs 70 \
    --finetune-epochs 30 \
    --batch 64 \           # NSL-KDD 樣本數多，可用較大批次
    --alpha 1.0 \
    --beta 0.8 \           # 較高的 β（NSL-KDD 攻擊/正常邊界更明顯）
    --gamma 0.01           # VAE 風格的 KL Divergence 正則化係數
```

**NSL-KDD 特有的 gamma 參數（KL Divergence）：**
```
L_total = α×L_recon + β×L_margin + γ×KL(q(z|x) || p(z))

KL 項使 Latent Space 接近標準正態分布，
增強模型對未見過攻擊類型的泛化能力（VAE-like regularization）
```

輸出：`output/model_nslkdd/semi_supervised_nslkdd.pt`

---

## 7. 資料擴增（選用）

當訓練資料量不足（< 1000 筆正常樣本）時，`DataAugmentor` 提供 4 種擴增策略。
**注意：只對正常流量擴增，攻擊流量不擴增。**

```python
# data_augmentor.py - 4 種擴增策略

# 策略 1：高斯雜訊 - 模擬網路傳輸中的輕微位元抖動
x_aug = np.clip(x + N(0, noise_std=0.02), 0, 1)

# 策略 2：隨機遮罩 - 遮蔽 10% 像素，學習從部分資訊重建
mask_positions = random_choice(H×W, size=H*W*mask_ratio)
x_aug[mask_positions] = 0.0

# 策略 3：亮度縮放 - 模擬不同大小封包的影像差異
scale ~ Uniform(0.9, 1.1)
x_aug = clip(x × scale, 0, 1)

# 策略 4：Mixup - 兩筆正常封包線性插值（Zhang et al., ICLR 2018）
lam ~ Beta(0.2, 0.2)
x_aug = lam × x_i + (1 - lam) × x_j
```

**Python 呼叫方式：**
```python
from core.data_augmentor import DataAugmentor

aug = DataAugmentor(noise_std=0.02, mask_ratio=0.1, use_mixup=True)

# multiplier=3：原始 1000 筆 → 擴增至 3000 筆
X_normal_aug = aug.augment(X_normal, multiplier=3)

# 合併多個資料集的正常流量
X_merged = DataAugmentor.merge_datasets(
    "output/dataset_cicids2017/X_normal.npy",
    "output/dataset_nslkdd/X_normal.npy",
    output_path="output/dataset_merged/X_normal.npy"
)
```

---

## 8. 閾值計算與調校

### 8.1 基本百分位數法

訓練完成後，`compute_threshold()` 取正常流量重建誤差的第 N 個百分位數：

```python
# semi_supervised_trainer.py - compute_threshold()
errors_normal = compute_errors(X_normal)   # 計算所有正常樣本的重建 MSE

# method="percentile"：直接取百分位
threshold = float(np.percentile(errors_normal, percentile=95))
# → 只有 5% 的正常流量會被誤判為異常（假陽性率 ≈ 5%）
```

### 8.2 最佳閾值自動搜尋

`method="optimal"` 利用已標記攻擊樣本掃描所有百分位，選出 F1 最大的閾值：

```python
# semi_supervised_trainer.py - compute_threshold() method="optimal"
best_f1, best_thr = -1.0, None
for pct in range(50, 100):             # 掃描第 50 ~ 99 百分位
    thr = float(np.percentile(errors_normal, pct))
    fp  = int((errors_normal > thr).sum())
    tp  = int((errors_attack > thr).sum())
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    if f1 > best_f1:
        best_f1, best_thr, best_pct = f1, thr, pct
```

### 8.3 閾值調校工具（ThresholdTuner）

獨立的一鍵閾值掃描指令（掃描 [80, 85, 90, 92, 95, 97, 99, 99.5] 等百分位並輸出詳細報告）：

```bash
python core/run_threshold_tuning.py \
    --model output/model_cicddos2019/semi_supervised_cicddos2019.pt \
    --data-dir data/cic-ddos2019 \
    --output output/threshold_analysis
```

輸出：
```
output/threshold_analysis/
    threshold_scan_results.json      ← 各百分位的 P/R/F1/FPR/PR-AUC
    pr_f1_curve.png                  ← Precision/Recall/F1 vs 閾值曲線圖
    threshold.json                   ← 推薦閾值（High Recall / Balanced / High Precision）
```

---

## 9. 模型評估與輸出檔案

### 9.1 評估指標

| 指標 | 計算方式 | 意義 |
|---|---|---|
| **Precision** | TP / (TP + FP) | 告警中真實攻擊的比例（低 FP 率） |
| **Recall** | TP / (TP + FN) | 所有攻擊中被偵測到的比例（偵測率） |
| **F1-Score** | 2×P×R / (P+R) | Precision 與 Recall 的調和平均 |
| **AUC-ROC** | ROC 曲線下面積 | 整體判別能力（與閾值無關） |
| **PR-AUC** | PR 曲線下面積 | 不平衡資料下的效能（比 AUC-ROC 更嚴苛） |
| **MCC** | (TP×TN−FP×FN)/√… | Matthews 相關係數（對不平衡資料更公平） |
| **SepRatio** | mean(errors_attack) / mean(errors_normal) | 攻擊/正常誤差分離比，越大越易設閾值 |

### 9.2 訓練輸出檔案清單

純非監督訓練（`run_training.py`）輸出：
```
output/model_<name>/
    best_model_<name>.pt          ← 最佳模型權重（PyTorch state_dict）
    training_result_<name>.json   ← 訓練設定與結果（latent, threshold, F1...）
    training_curve.png            ← 訓練/驗證損失曲線
    reconstruction_samples.png    ← 重建對比圖（輸入 vs 重建）
    roc_curve.png                 ← ROC 曲線
    score_distribution.png        ← 正常/攻擊重建誤差分布圖
```

半監督訓練（`run_semi_supervised.py` / 資料集專用腳本）額外輸出：
```
output/model_semi_<name>/
    pretrained_model_<name>.pt    ← Phase 1 預訓練模型（中間產物）
    best_model_<name>.pt          ← Phase 2 最終最佳模型
    semi_training_result.json     ← 完整訓練設定 + 最終閾值
    semi_training_curve.png       ← Phase1 + Phase2 雙欄損失曲線
    semi_error_distribution.png   ← 正常/攻擊誤差分布（密度圖 + ECDF）
    semi_evaluation_report.json   ← 最終評估報告（P/R/F1/AUC + 混淆矩陣數值）
    comparison_report.png         ← （加 --compare-unsupervised）對比長條圖
```

---

## 10. 訓練流程一覽與快速指令

### 10.1 完整流程圖

```
【第一步】環境安裝
pip install torch numpy pandas scikit-learn matplotlib scapy

【第二步】資料準備
下載資料集 → 放入對應目錄
data/cicids2017/ 或 data/nslkdd/ 或 data/cicddos2019/

【第三步】選擇訓練路徑
│
├── 無攻擊標籤 ──→ 路徑 A（run_training.py）
│                  純非監督 CNN AE，只用正常流量
│
├── 少量攻擊標籤 → 路徑 B（run_semi_supervised.py）
│                  通用版半監督，Phase1 + Phase2
│
└── 特定資料集 ──→ 路徑 C（semi_supervised_cicddos2019.py 等）
                   資料集專用，含 Latent/VAE 策略

【第四步】閾值調校
run_threshold_tuning.py → 掃描多百分位，輸出推薦值

【第五步】評估
AnomalyScorer → F1/AUC/ROC 曲線 → eval_result.json
```

### 10.2 所有訓練指令速查表

```bash
# ──────────────── 路徑 A：純非監督 ────────────────

# 快速測試（模擬資料）
python core/run_training.py

# NSL-KDD
python core/run_training.py --dataset nslkdd --data-dir data/nslkdd

# CIC-IDS2017（限制樣本數）
python core/run_training.py \
    --dataset cicids2017 --data-dir data/cicids2017 \
    --max-normal 50000 --max-attack 30000 \
    --epochs 100 --latent 32 --batch 32 --pct 95

# CIC-DDoS2019
python core/run_training.py \
    --dataset cicddos2019 --data-dir data/cicddos2019 \
    --max-normal 60000 --max-attack 30000


# ──────────────── 路徑 B：半監督通用版 ────────────────

# 快速測試
python core/run_semi_supervised.py

# CIC-IDS2017（含閾值自動搜尋）
python core/run_semi_supervised.py \
    --dataset cicids2017 --data-dir data/cicids2017 \
    --pretrain-epochs 70 --finetune-epochs 30 \
    --alpha 1.0 --beta 0.5 --margin 0.05 --attack-ratio 0.1 \
    --threshold-method optimal

# NSL-KDD（含資料擴增 + 比較對比）
python core/run_semi_supervised.py \
    --dataset nslkdd --data-dir data/nslkdd \
    --pretrain-epochs 70 --finetune-epochs 30 \
    --augment --compare-unsupervised


# ──────────────── 路徑 C：資料集專用腳本 ────────────────

# CIC-DDoS2019
python core/semi_supervised_cicddos2019.py \
    --data-dir data/cic-ddos2019 --output output/model_cicddos2019 \
    --pretrain-epochs 70 --finetune-epochs 30 \
    --latent 32 --batch 32 --margin 0.05 --alpha 1.0 --beta 0.5 \
    --max-normal 60000 --max-attack 30000

# CIC-IDS2017（含 Latent 分離策略）
python core/semi_supervised_cicids2017.py \
    --data-dir data/cicids2017 --output output/model_cicids2017 \
    --pretrain-epochs 70 --finetune-epochs 30 \
    --latent 48 --batch 32 --margin 0.05 --alpha 1.0 --beta 0.6 --gamma 0.1

# NSL-KDD（16×16 影像 + VAE KL 正則化）
python core/semi_supervised_nslkdd.py \
    --train data/nsl-kdd/KDDTrain+.txt --test data/nsl-kdd/KDDTest+.txt \
    --output output/model_nslkdd \
    --image-size 16 --latent 16 \
    --pretrain-epochs 70 --finetune-epochs 30 \
    --batch 64 --alpha 1.0 --beta 0.8 --gamma 0.01


# ──────────────── 閾值調校 ────────────────

python core/run_threshold_tuning.py \
    --model output/model_cicddos2019/semi_supervised_cicddos2019.pt \
    --data-dir data/cic-ddos2019 \
    --output output/threshold_analysis


# ──────────────── 僅評估（不訓練）────────────────

python core/run_training.py \
    --dataset cicids2017 --data-dir data/cicids2017 \
    --eval-only \
    --model output/model_cicids2017/best_model_cicids2017.pt \
    --latent 32
```

---

## 11. 常見問題與疑難排解

### Q1：找不到 CSV 標籤欄位（`找不到標籤欄位，跳過`）

CIC 系列 CSV 的標籤欄位名稱可能有前後空格。`CICIDSLoader._find_label_column()` 已處理此問題。
若仍出錯，可手動確認欄位名稱：

```python
import pandas as pd
df = pd.read_csv("data/cicids2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=3)
print(df.columns.tolist())
# 若輸出 [' Label', ' Flow Duration', ...] 有前置空格，屬正常情況，程式會自動處理
```

### Q2：訓練時 CUDA out of memory

縮小 `batch_size` 或 `latent_dim`：

```bash
python core/run_semi_supervised.py \
    --batch 16 \      # 從 32 縮小至 16
    --latent 16       # 從 32 縮小至 16
```

### Q3：半監督訓練後 F1 沒有提升

可能原因：
- `beta` 值過大，破壞了正常流量重建能力 → 嘗試 `--beta 0.3`
- `margin` 設定太高（超過大多數攻擊樣本的實際誤差）→ 先執行路徑 A，觀察誤差分布後再設定

```bash
# 先用純非監督觀察誤差分布
python core/run_training.py --dataset cicids2017 --data-dir data/cicids2017
# 查看 output/model_cicids2017/score_distribution.png 了解正常/攻擊誤差的分布範圍
# 再根據圖表調整 margin 值（建議設在兩個分布的中間值）
```

### Q4：NSL-KDD 出現 `found input variables with inconsistent numbers of samples`

NSL-KDD 的 `KDDTest+.txt` 測試集與 `KDDTrain+.txt` 訓練集的 Scaler 需分開 fit。程式中已透過 `fit=True/False` 參數控制，確保 Scaler 只在訓練集上 fit。

### Q5：模型評估時 Recall 很低（大量漏報）

嘗試降低閾值百分位數，或切換至 `optimal` 方法：

```bash
# 將百分位從 95 降至 90
python core/run_training.py --dataset cicids2017 --pct 90

# 或使用最佳 F1 搜尋
python core/run_semi_supervised.py \
    --dataset cicids2017 \
    --threshold-method optimal
```

---

