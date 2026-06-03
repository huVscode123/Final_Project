# 實驗執行指南 (Experiment Execution Guide)

本文件彙整了「視覺化網路攻擊自動化分析平台」中所有核心實驗的執行指令、參數說明及預期的產出報告。

---

## 1. 消融實驗 (Ablation Study)
**目的**：系統性地驗證模型各個組件（如隱藏層維度、影像尺寸、遮罩策略）對效能的貢獻。

### 執行指令
*   **完整六大維度實驗**：
    ```bash
    python core/run_threshold_tuning.py --dataset cicids2017 --data-dir data/cicids2017 --ablation
    ```
*   **指定特定維度實驗**：
    ```bash
    # 可選維度：latent, arch, image_size, field_mask, data_size, activation
    python core/run_threshold_tuning.py --dataset simulate --ablation --ablation-exp latent arch
    ```

### 實驗維度說明
1.  **Latent Dimension**: 測試瓶頸層大小（8, 16, 32, 64）對壓縮與重構的影響。
2.  **Architecture Scale**: 比較輕量版 (Light) vs 標準版 (Standard) vs 寬層版 (Wide) 模型。
3.  **Image Size**: 測試封包影像解析度（16x16, 32x32, 64x64）。
4.  **Field Masking**: 驗證遮蔽 IP/Port 資訊是否能提升模型對「攻擊行為」的泛化能力。
5.  **Data Size**: 測試不同訓練樣本量（25%, 50%, 75%, 100%）下的表現。
6.  **Activation**: 比較 ReLU, GELU, LeakyReLU 的訓練效果。

### 產出路徑
*   圖表：`output/<dataset>/ablation/ablation_*.png`
*   報告：`output/<dataset>/ablation/ablation_report.json`

---

## 2. 策略基準測試 (Comparison Benchmark)
**目的**：比較「純非監督學習」與「半監督邊界微調 (Margin Fine-tuning)」在相同條件下的效能差異。

### 執行指令
```bash
python core/comparison_benchmark.py --dataset cicids2017 --data-dir data/cicids2017
```

### 評估指標
*   **Precision/Recall/F1/AUC**: 標準效能指標。
*   **Separability Ratio**: 攻擊誤差與正常誤差的均值比值（越高代表區分度越好）。
*   **Inference Latency**: 比較兩者的推論速度。

### 產出路徑
*   對比圖：`output/<dataset>/comparison/comparison_metrics_bar.png`
*   雷達圖：`output/<dataset>/comparison/comparison_radar.png`
*   誤差分佈：`output/<dataset>/comparison/comparison_error_dist.png`

---

## 3. 模型壓縮實驗 (Model Compression)
**目的**：在維持高效能的前提下，減小模型體積並提升推論速度，適合部署於邊緣運算裝置。

### 執行指令
```bash
python core/run_threshold_tuning.py --dataset cicids2017 --compress
```

### 測試技術
1.  **Pruning (剪枝)**：移除不重要的權重（支援結構化與非結構化剪枝）。
2.  **Quantization (量化)**：將 FP32 轉換為 INT8，大幅降低模型體積。
3.  **Distillation (知識蒸餾)**：用大型模型指導輕量級模型。

### 產出路徑
*   權重檔：`output/<dataset>/compression/*.pt`
*   分析圖：`output/<dataset>/compression/compression_tradeoff.png`
*   報告：`output/<dataset>/compression/compression_report.json`

---

## 4. 可解釋性實驗 (Grad-CAM XAI)
**目的**：視覺化深度學習模型的決策依據，指出封包中觸發異常的位元組區域。

### 執行指令
```bash
python core/cnn_gradcam/run_gradcam.py
```

### 產出路徑
*   視覺化圖：`output/<dataset>/gradcam/gradcam_packet_*.png`
*   分析報告：`output/<dataset>/gradcam/gradcam_analysis.json`

---

## 5. 閾值調校實驗 (Threshold Tuning)
**目的**：尋找最佳的百分位數閾值，平衡誤報 (FP) 與漏報 (FN)。

### 執行指令
```bash
python core/run_threshold_tuning.py --dataset cicids2017 --data-dir data/cicids2017
```

### 關鍵參數
*   `--eval-only`: 僅評估現有模型，不重新訓練。
*   `--model`: 指定評估的路徑，例如 `output/model/best_model.pt`。

### 產出路徑
*   曲線圖：`output/<dataset>/model/threshold_curve.png`
*   JSON 報告：`output/<dataset>/model/threshold_report.json`

---

## 快速速查表

| 實驗項目 | 執行指令 (範例) |
| :--- | :--- |
| **消融實驗** | `python core/run_threshold_tuning.py --dataset <ds> --ablation` |
| **策略對比** | `python core/comparison_benchmark.py --dataset <ds>` |
| **模型壓縮** | `python core/run_threshold_tuning.py --dataset <ds> --compress` |
| **可解釋性** | `python core/cnn_gradcam/run_gradcam.py --dataset <ds>` |
| **自動調校** | `python core/run_threshold_tuning.py --dataset <ds>` |

> **提示**：若硬體資源有限或僅需測試流程，請將 `--dataset` 設為 `simulate` 以使用模擬資料快速執行。
