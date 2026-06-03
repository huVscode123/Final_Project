# 網路攻擊偵測模型：完整訓練手冊 (真實資料集版)

本手冊說明如何針對四種不同場景的真實網路流量資料集進行模型訓練。所有指令均已優化，預設執行 **100 次 (Epochs)** 以確保模型在複雜流量中的收斂效果。

---

## 訓練流程 (Workflow)

請依序執行以下三個步驟：

### 第一步：準備環境與資料
1. **安裝依賴**：
   ```bash
   pip install torch numpy pandas scikit-learn
   ```
2. **放置資料集**：請確保 CSV 檔案已下載並放置於以下路徑：
   - **CICIDS2017**: `data/cicids2017/` (支援多個 CSV 自動合併)
   - **DDoS2019**: `data/cic-ddos2019/` (支援多個 CSV 自動合併)
   - **NSL-KDD**: `data/nsl-kdd/KDDTrain+.txt`

### 第二步：執行模型訓練
根據您的偵測目標，選擇對應的指令執行。指令中的路徑均以專案根目錄為基準。

| 目標資料集 | 訓練策略 | 總訓練次數 | 快速執行指令 |
| :--- | :--- | :--- | :--- |
| **自定義流量** | 純非監督 (CAE) | 100 Epochs | `python core/"Semi-supervised model and trainer"/cnn_autoencoder.py --normal data/normal_traffic.npy --epochs 100` |
| **DDoS2019** | 半監督 (Margin) | 70 + 30 | `python core/"Semi-supervised model and trainer"/semi_supervised_cicddos2019.py --data-dir data/cic-ddos2019 --pretrain-epochs 70 --finetune-epochs 30` |
| **CICIDS2017** | 半監督 (Latent) | 70 + 30 | `python core/"Semi-supervised model and trainer"/semi_supervised_cicids2017.py --data-dir data/cicids2017 --pretrain-epochs 70 --finetune-epochs 30` |
| **NSL-KDD** | 半監督 (VAE/Cat) | 70 + 30 | `python core/"Semi-supervised model and trainer"/semi_supervised_nslkdd.py --train data/nsl-kdd/KDDTrain+.txt --pretrain-epochs 70 --finetune-epochs 30` |

### 第三步：驗證訓練結果
訓練完成後，請檢查對應的 `output/model_xxxx/` 目錄，確認包含以下檔案：
1. `xxxx.pt`: 模型權重檔案（可用於推論）。
2. `threshold.json`: 自動計算的異常判定閾值。
3. `eval_result.json`: 訓練後的模型效能報告（含 Precision, Recall, F1）。

---

## 快速複製指令集 (單行格式)

> **注意**：請確保在專案根目錄下執行。

#### [1] 非監督 CAE
```bash
python core/"Semi-supervised model and trainer"/cnn_autoencoder.py --normal data/normal_traffic.npy --attack data/attack_traffic.npy --output output/model_unsupervised --epochs 100 --latent 32 --batch 32 --pct 95.0
```

#### [2] 半監督 CIC-DDoS2019 (100 Epochs)
```bash
python core/"Semi-supervised model and trainer"/semi_supervised_cicddos2019.py --data-dir data/cic-ddos2019 --output output/model_cicddos2019 --pretrain-epochs 70 --finetune-epochs 30 --latent 32 --batch 32 --margin 0.05 --alpha 1.0 --beta 0.5 --max-normal 60000 --max-attack 30000
```

#### [3] 半監督 CICIDS2017 (100 Epochs)
```bash
python core/"Semi-supervised model and trainer"/semi_supervised_cicids2017.py --data-dir data/cicids2017 --output output/model_cicids2017 --pretrain-epochs 70 --finetune-epochs 30 --latent 48 --batch 32 --margin 0.05 --alpha 1.0 --beta 0.6 --gamma 0.1 --max-normal 60000 --max-attack 30000
```

#### [4] 半監督 NSL-KDD (100 Epochs)
```bash
python core/"Semi-supervised model and trainer"/semi_supervised_nslkdd.py --train data/nsl-kdd/KDDTrain+.txt --test data/nsl-kdd/KDDTest+.txt --output output/model_nslkdd --image-size 16 --latent 16 --pretrain-epochs 70 --finetune-epochs 30 --batch 64 --alpha 1.0 --beta 0.8 --gamma 0.01
```

---

## 調參建議 (Tips)

### 關於訓練次數 (Epochs)
- **100 次 (70+30)** 是針對真實資料集的建議平衡點。
- 若發現 **正常流量的 MSE 仍然很高**：請增加 Pretrain Epochs (例如調至 100)。
- 若發現 **攻擊流量偵測不到 (Recall 低)**：請增加 Finetune Epochs 或調大 `--margin` 參數。

### 記憶體優化 (OOM)
若在訓練時遇到顯存不足：
1. 加入 `--batch 16` 減小批次大小。
2. 加入 `--max-normal 40000` 減少載入的樣本總數。

---

## 🔍 結果解讀
在 `eval_result.json` 中，您應該關注：
- **F1-Score**: 模型整體的偵測平衡力，建議 > 0.90。
- **Recall (召回率)**: 攻擊漏報率，若此值太低，代表模型太過放鬆，需增加 `margin`。
- **Precision (精準率)**: 誤報率，若此值太低，代表模型太過敏感。
