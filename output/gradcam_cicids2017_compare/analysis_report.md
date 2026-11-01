# CICIDS2017 Grad-CAM 解釋比較分析報告

> **歷史版本注意**：本文件僅描述舊版 `encoder_conv.11` 的輸出，部分結論過於武斷，且全零熱圖的 Top-10 Jaccard 不具實質解釋意義。請以 [fixed 版更新分析報告](../gradcam_cicids2017_compare_fixed/analysis_report.md) 為準；其中包含更正及新舊版本比較。

- 資料夾：`output/gradcam_cicids2017_compare/`
- 模型：CNN-VAE（異常偵測），checkpoint 為 `output/threshold_study/unsup_cicids2017/best_vae_cicids2017.pt`
- 判定規則：`anomaly_score > threshold => ATTACK`，threshold = `0.00081994108`
- 異常分數：`reconstruction_mse + 0.001 × KL divergence`
- 解釋方法：Grad-CAM、Grad-CAM++、Score-CAM，目標層皆為 `encoder_conv.11`
- 樣本：8 筆（TN×2、FP×2、TP×2、FN×2），特徵數 78，輸入以 32×32 影像排列（前 78 個位置對應特徵，其餘為 padding）

## 1. 資料夾內容盤點

| 項目 | 內容 |
|---|---|
| `run_metadata.json` | 執行設定、checkpoint、threshold、8 筆樣本資訊 |
| `selected_samples.csv` / `sample_scores_and_runtime.csv` | 樣本來源、row index、重算分數、判定結果 |
| `scaled_feature_inputs.csv` | 8 筆樣本的縮放後 78 維特徵值 |
| `feature_heatmap_values.csv` | 每方法×每樣本×每特徵的熱度值（1,872 列） |
| `heatmap_pixels.csv` | 每方法×每樣本×每像素的熱度值（24,576 列） |
| `variant_summary.csv` / `variant_comparison.csv` | 依群組彙整的 top 特徵、方法間一致性 |
| `variant_comparison_top10.png` | Top-10 比較圖 |
| `gradcam/`, `gradcampp/`, `scorecam/` | 各方法 32×32 heatmap（`.npy`，共 24 檔） |
| `images/<method>/` | 每樣本的 heatmap、overlay、triptych 圖（各方法 24 張），加上 all_methods/ 8 張，共 80 張 PNG |
| `upload_package/` | 上述主要 CSV、JSON、PNG 的複本（內容相同） |

`upload_package/` 與主目錄的關鍵檔案經比對完全一致，屬於重複內容。

## 2. 樣本與模型判定

| ID | 預期群組 | 來源 | 預期標籤 | 重算分數 | 判定 | 實際群組 | 與原報告一致 |
|---|---|---|---|---|---|---|---|
| 001 | TN | Monday | BENIGN | 0.00082004 | ATTACK | **FP** | ✗ |
| 002 | TN | Monday | BENIGN | 0.00081972 | NORMAL | TN | ✓ |
| 003 | FP | Monday | BENIGN | 0.00082006 | ATTACK | FP | ✓ |
| 004 | FP | Friday DDoS | BENIGN | 0.00082019 | ATTACK | FP | ✓ |
| 005 | TP | Friday DDoS | DDoS | 0.00081987 | NORMAL | **FN** | ✗ |
| 006 | TP | Friday DDoS | DDoS | 0.00082644 | ATTACK | TP | ✓ |
| 007 | FN | Friday DDoS | DDoS | 0.00079094 | NORMAL | FN | ✓ |
| 008 | FN | Friday DDoS | DDoS | 0.00079085 | NORMAL | FN | ✓ |

**觀察**

- **樣本 001、005 的群組標記與重算結果不一致。** 兩者的分數都非常接近 threshold（相差約 1e-7 至 5e-7，約 0.02%–0.06%），重算時跨越了門檻，所以實際群組變成 FP（001）與 FN（005）。
- 原本 `sample_scores_and_runtime.csv` 中的 `reported_score_ch4` 與重算值差異很小，但足以改變判定。這代表結論對數值精度非常敏感，建議以重算值為準，並重新標記群組。
- 其餘 6 筆與預期一致。

## 3. 各方法的 heatmap 表現

### 3.1 逐樣本結果（以 `feature_heatmap_values.csv` 計算）

| 方法 | 樣本 | 群組（實際） | 非零特徵數 | 熱度總和 | Top-1 特徵 |
|---|---|---|---|---|---|
| Grad-CAM | 001 | TN→FP | 0 | 0.00 | 全部為 0 |
| Grad-CAM | 002 | TN | 0 | 0.00 | 全部為 0 |
| Grad-CAM | 003 | FP | 0 | 0.00 | 全部為 0 |
| Grad-CAM | 004 | FP | 0 | 0.00 | 全部為 0 |
| Grad-CAM | 005 | TP→FN | 78 | 30.77 | ECE Flag Count (0.67) |
| Grad-CAM | 006 | TP | 56 | 12.59 | Flow Bytes/s (1.00) |
| Grad-CAM | 007 | FN | 78 | 30.86 | Destination Port (1.00) |
| Grad-CAM | 008 | FN | 78 | 30.85 | Destination Port (1.00) |
| Grad++ | 001–006 | 全部 | 0 | 0.00 | 全部為 0 |
| Grad++ | 007 | FN | 58 | 16.54 | Bwd Packet Length Std (1.00) |
| Grad++ | 008 | FN | 58 | 16.55 | Bwd Packet Length Std (1.00) |
| Score-CAM | 001 | TN→FP | 78 | 43.37 | Bwd Packet Length Max (1.00) |
| Score-CAM | 002 | TN | 78 | 51.94 | Fwd Packet Length Min (1.00) |
| Score-CAM | 003 | FP | 78 | 30.12 | Flow IAT Max (1.00) |
| Score-CAM | 004 | FP | 78 | 41.87 | Fwd Packet Length Std (1.00) |
| Score-CAM | 005 | TP→FN | 78 | 45.53 | Bwd Packet Length Mean (1.00) |
| Score-CAM | 006 | TP | 78 | 45.01 | Total Length of Bwd Packets (1.00) |
| Score-CAM | 007 | FN | 78 | 27.90 | Total Length of Bwd Packets (1.00) |
| Score-CAM | 008 | FN | 78 | 27.90 | Total Length of Bwd Packets (1.00) |

### 3.2 方法層級的問題

1. **Grad-CAM 與 Grad-CAM++ 大量輸出全零。**
   - Grad-CAM 在 TN/FP 四筆全為零，Grad-CAM++ 在 001–006 全為零。
   - 全零通常表示 ReLU 後沒有正向貢獻，或梯度在目標層被截斷。這些樣本無法用這兩種方法解釋。
   - 這不代表「模型沒有依據」，而是方法在此模型與目標層上失效。
2. **Score-CAM 不會輸出全零，但產生的 heatmap 幾乎都是滿的。**
   - 8 筆樣本中有 78 個特徵都有非零值，且 Top-5 幾乎都接近 1.0。
   - 這表示它的對比度低，難以分辨真正重要的特徵。
3. **三種方法在同一筆樣本上的結論常常互相矛盾。**
   - 例如樣本 005：Grad-CAM 的 Top-1 是 ECE Flag Count，Score-CAM 的 Top-1 是 Bwd Packet Length Mean，兩者的 Spearman 相關為 **−0.71**。
   - 樣本 006 同樣是負相關（−0.41）。

## 4. 方法間一致性

| 比較 | Top-10 Jaccard（全樣本平均） | Spearman ρ（全樣本平均） |
|---|---|---|
| Grad-CAM vs Grad-CAM++ | 0.51 | 0.75（只有 007、008 可算） |
| Grad-CAM vs Score-CAM | 0.05 | −0.13 |
| Grad-CAM++ vs Score-CAM | 0.06 | 0.12 |

- Grad-CAM 與 Grad-CAM++ 在 FN 樣本（007、008）高度一致（ρ ≈ 0.75），但這兩者在其他樣本上有很多全零情況，因此一致性是在有限樣本上成立。
- Score-CAM 與另外兩者幾乎沒有共識，Top-10 重疊僅約 5%。

## 5. 特徵層級的重要發現（依群組）

以 `variant_summary.csv` 為準（每組 2 筆樣本）：

- **TP（005、006）**
  - Grad-CAM：Flow Bytes/s、PSH Flag Count、Flow Packets/s、ACK Flag Count 等。
  - Score-CAM：Fwd Packet Length Max、Total Length of Bwd Packets、Bwd Packets/s 等。
  - 兩者有關鍵的「流量速率、封包長度」類特徵，但實際 Top 特徵並不一致。
- **FN（007、008）**
  - Grad-CAM：Destination Port、Fwd URG Flags、Flow Duration。
  - Score-CAM：Total Length of Bwd Packets、Fwd Packet Length Max、Bwd Packets/s。
  - 007 與 008 的結果幾乎相同，因為兩者的特徵值非常接近（來自同一 Friday DDoS 檔案，兩筆可能是相似流量）。
- **TN / FP**
  - Grad-CAM 與 Grad-CAM++ 無有效輸出。
  - Score-CAM 的 Top 特徵多為 Packet Length、IAT 與 Flag 類特徵，但缺乏一致的主題。

## 6. 結構性限制（解讀前必須注意）

1. **表格特徵被人為排成 2D 影像。**
   - 78 個特徵依照欄位順序填入 32×32 格子，之後才做 CNN 解釋。
   - 相鄰像素沒有真實的語意關係，因此「熱區的位置」與「空間相鄰」沒有意義。只能把它當作逐特徵的分數。
   - 不同的特徵排列會得到不同的解釋結果，這一點沒有驗證。
2. **Heatmap 逐樣本正規化到 [0,1]。** 不同樣本之間的熱度絕對值不可比較，只能比較排序。
3. **樣本數極少。** 每群組僅 2 筆，無法推論整體模型行為。
4. **Score-CAM 權重使用標準化分數差。** 因為 VAE 分數本身很小（約 1e-4），實作上做了標準化。這讓 Score-CAM 的數值尺度與其他方法不同。
5. **Grad-CAM 的 heatmap 與 Score-CAM 的 heatmap 不在同一尺度。** 直接比較數值沒有意義。

## 7. 執行時間

| 方法 | 平均（秒） | 中位數（秒） | 備註 |
|---|---|---|---|
| Grad-CAM | 0.0568 | 0.0077 | 樣本 001 首次執行 0.40 s（含暖機），拉高平均值 |
| Grad-CAM++ | 0.0070 | 0.0063 | 穩定 |
| Score-CAM | 0.0201 | 0.0062 | 樣本 001 為 0.118 s，其餘約 6 ms |

去除暖機離群值後，三種方法都在約 6–8 ms 量級，差異不大。Score-CAM 理論上需要多次前向傳遞，但此模型很小，成本不明顯。

## 8. 結論與建議

**結論**

1. 在這個 CNN-VAE 上，**Grad-CAM 與 Grad-CAM++ 不適合作為解釋工具**：大多數樣本輸出全零，無法提供資訊。
2. **Score-CAM 有輸出，但區分度低**，且與其他方法的結論常相反，不能單獨信任。
3. 目前的特徵熱度結果**無法可靠地指出攻擊的關鍵特徵**，任何對 Top 特徵的解讀都需要更多樣本與驗證。
4. 樣本 001 與 005 的判定在重算後改變，表示結果本身對 threshold 非常敏感。

**建議的下一步**

1. **修正群組標記**：把 001 改為 FP、005 改為 FN，或改用同一版本的分數重新產生結果，確保 `reported_score_ch4` 與實際分數一致。
2. **排除 Grad-CAM 全零問題**：檢查 `encoder_conv.11` 之後的 ReLU 與梯度流，或改用較淺的卷積層，確認全零不是實作錯誤。
3. **用非空間方法驗證**：例如對每個特徵做 permutation importance、Integrated Gradients 或 SHAP，與 heatmap 結果交叉比對。
4. **擴大樣本數**：每群組至少 30 筆，才能評估方法的穩定性。
5. **測試不同特徵排列**：例如依特徵群組（流量、封包長度、旗標）排序，看解釋結果是否穩定。

## 9. 本次檢查範圍

- 已讀取：全部 CSV、`run_metadata.json`，並用 Python 重算各方法的 Top-5 特徵、非零數量與方法間 Spearman 相關。
- 已驗證：`upload_package/` 與主目錄的關鍵檔案內容相同；`.npy` heatmap 的形狀（32×32）與數值範圍（0–1）。
- **未檢查**：`images/` 下的 PNG 與 `variant_comparison_top10.png` 的視覺內容（未逐張開啟或用視覺模型確認）。
- **未重新執行**模型或重算分數，所有數值皆取自既有輸出檔。
