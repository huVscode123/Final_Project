# CICIDS2017 Grad-CAM 比較分析報告（fixed 版）

> 本報告分析 `output/gradcam_cicids2017_compare_fixed/` 的既有輸出，並與 `output/gradcam_cicids2017_compare/` 舊版比較。未重新執行模型；結果是 8 個個案的描述，不代表整個測試集的效能。

## 一、重點結論

1. **全零熱圖問題在這批 fixed 輸出中消失**：三種方法各 8 張熱圖皆非全零；舊版 Grad-CAM 有 4 張全零、Grad-CAM++ 有 6 張全零。目標層由 `encoder_conv.11` 改成 `encoder_conv.13`；但僅憑輸出無法確認所有改動或建立單一因果關係。非零不等於解釋正確。
2. **群組彙整已依實際判定而非預期群組**：001 是 FP、005 是 FN，因此實際樣本組成為 TN 1、FP 3、TP 1、FN 3；不再是每組兩筆。兩筆跨閾值的分數差異來源尚未查明。
3. **方法間一致性因個案而異**：Grad-CAM 與 Grad-CAM++ 的 8 筆平均 Spearman ρ 為 0.521，Grad-CAM 與 Score-CAM 為 0.139；平均值掩蓋了個案差異與 007、008 近似重複的影響。
4. **重要性仍需驗證**：熱圖經插值及逐張正規化，不能直接視為特徵因果影響。甚至有輸入縮放值為 0 的特徵列在前幾名。不能只憑排名斷言攻擊原因。

## 二、實驗設定與檔案盤點

- 資料集：CICIDS2017；模型：CNN-VAE；checkpoint 與 preprocessor 路徑記於 `run_metadata.json`。
- 異常分數：`reconstruction_mse + 0.001 × KL divergence`；大於 `0.0008199410804081708` 判定為 ATTACK，否則 NORMAL。
- 方法：Grad-CAM、Grad-CAM++、Score-CAM，三者目標層皆為 `encoder_conv.13`。Score-CAM 使用 channel mask，並對 masked anomaly score 與基準分數的差值標準化後做 softmax（temperature 1.0、batch size 32）。
- 8 筆固定樣本、78 個特徵；輸入 32×32 共 1,024 像素，前 78 格依 row-major 順序對應特徵，其餘 946 格為 padding。

| 檔案／目錄 | 數量及用途 |
|---|---|
| `run_metadata.json` | 設定、樣本、目標層與注意事項 |
| `selected_samples.csv` | 8 筆樣本來源、實際分數與分類 |
| `scaled_feature_inputs.csv` | 624 筆：8 × 78 的縮放輸入 |
| `feature_heatmap_values.csv` | 1,872 筆：3 × 8 × 78 特徵熱度 |
| `heatmap_pixels.csv` | 24,576 筆：3 × 8 × 1,024 像素熱度 |
| `sample_scores_and_runtime.csv` | 24 筆：每方法／樣本的分數、時間、梯度或 Score-CAM 權重診斷 |
| `variant_summary.csv` | 各方法依實際群組彙整 |
| `variant_comparison.csv` | 逐樣本方法配對的 Spearman ρ、Top-10 Jaccard 與平均 |
| `gradcam/`、`gradcampp/`、`scorecam/` | 共 24 個 32×32 `.npy` 熱圖 |
| `images/` | 80 張圖：24 張 heatmap、24 張 overlay、24 張 triptych、8 張 all-methods |
| `variant_comparison_top10.png` | 1 張方法比較圖 |

分析前共 113 個來源檔案（7 CSV、1 JSON、24 NPY、81 PNG）。fixed 目錄沒有舊版的 `upload_package/`。

## 三、樣本及實際分類

TN／FP 表示良性分別判為正常／攻擊；TP／FN 表示攻擊分別判為攻擊／正常。以下分數為固定輸出中記載的值，非本次重跑。

| ID | 來源與 row index | 實際標籤 | 預期 → 實際群組 | 異常分數 | 分數 − 閾值 |
|---|---|---|---|---:|---:|
| 001 | Monday / 41093 | BENIGN | TN → **FP** | 0.000820044428 | +1.0335×10⁻⁷ |
| 002 | Monday / 3024 | BENIGN | TN → TN | 0.000819722714 | −2.1837×10⁻⁷ |
| 003 | Monday / 18563 | BENIGN | FP → FP | 0.000820058922 | +1.1784×10⁻⁷ |
| 004 | Friday DDoS / 12194 | BENIGN | FP → FP | 0.000820191810 | +2.5073×10⁻⁷ |
| 005 | Friday DDoS / 22571 | DDoS | TP → **FN** | 0.000819870271 | −7.0810×10⁻⁸ |
| 006 | Friday DDoS / 27472 | DDoS | TP → TP | 0.000826444360 | +6.5033×10⁻⁶ |
| 007 | Friday DDoS / 24061 | DDoS | FN → FN | 0.000790935534 | −2.9006×10⁻⁵ |
| 008 | Friday DDoS / 46589 | DDoS | FN → FN | 0.000790854741 | −2.9086×10⁻⁵ |

001 與 005 的實際分數分別距閾值約 +0.0126% 與 −0.0086%；相較 `reported_score_ch4`，差異分別為 +1.3413×10⁻⁷ 與 −5.0813×10⁻⁷，足以跨越閾值。這顯示邊界個案判定敏感，**不能據此直接認定原因是浮點精度**；需檢查推論環境、資料處理與 VAE 推論的決定性。`selected_samples.csv` 保留原預期群組供追溯，fixed 彙整使用 `observed_group`。

## 四、逐樣本熱圖結果

下表 Top-1 由 `feature_heatmap_values.csv` 的 78 個特徵位置取最大值；如有並列，列出其中一個。非零數以特徵區域計，並非整張圖；每張圖各自正規化，**不能把不同樣本或方法的熱度大小直接解讀為貢獻強弱**。

| ID／實際群組 | Grad-CAM：Top-1（非零數） | Grad-CAM++：Top-1（非零數） | Score-CAM：Top-1（非零數） |
|---|---|---|---|
| 001 / FP | Destination Port（78） | Destination Port（78） | Bwd Packet Length Max（78） |
| 002 / TN | Total Length of Bwd Packets（78） | Flow IAT Max（78） | Fwd Packet Length Min（78） |
| 003 / FP | Flow IAT Std（78） | Flow Packets/s（78） | Flow IAT Std（78） |
| 004 / FP | Total Length of Fwd Packets（77） | Total Length of Bwd Packets（78） | Fwd Packet Length Mean（78） |
| 005 / FN | ECE Flag Count（54） | Flow IAT Std（78） | Fwd Packet Length Std（78） |
| 006 / TP | Flow IAT Std（78） | Flow IAT Std（78） | Flow IAT Max（78） |
| 007 / FN | Destination Port（77） | Total Length of Bwd Packets（78） | Total Length of Bwd Packets（78） |
| 008 / FN | Destination Port（77） | Total Length of Bwd Packets（78） | Total Length of Bwd Packets（78） |

### 群組層級觀察

- **TN（002，一筆）**：Grad-CAM 偏向封包總長度，Grad-CAM++ 偏向 Flow IAT；Score-CAM 偏向封包長度。只有一筆，不能視為 TN 群體模式。
- **FP（001、003、004）**：三法依群組平均的 Top-1 分別是 Destination Port、Total Length of Fwd Packets、Fwd Packet Length Std。這是個案內正規化後的平均排名，不是群體因果結論。
- **TP（006，一筆）**：三法 Top-1 落在 Flow IAT Std／Max；對這一筆有共同的 IAT 訊號，但沒有跨個案驗證。
- **FN（005、007、008）**：Grad-CAM 群組 Top-1 為 Destination Port；Grad-CAM++ 與 Score-CAM 為 Total Length of Bwd Packets。007 與 008 的縮放特徵最大差約 0.00011346，兩者各方法熱圖最大差均小於 0.00017，可能使這組平均結果重複反映相近樣本；不能假定兩筆彼此獨立。

### 梯度與 Score-CAM 診斷

`sample_scores_and_runtime.csv` 顯示 8 筆 Grad-CAM 梯度絕對值平均皆非零。由產生程式可見，`positive_evidence_fraction`／`negative_evidence_fraction` 是**上採樣前 signed CAM 空間位置**中正／負值比例，並非特徵貢獻比例。例如 005 的 Grad-CAM 約 10.94% 為正、89.06% 為負；顯示圖中保留的正向區域不涵蓋大部分 signed CAM，不能說負值特徵對分類造成多少影響。

Score-CAM 的 softmax 權重有效通道數為 **32.24–38.16 / 64**（平均 35.79，entropy 指數），權重並非只集中於極少通道。8 張圖的 78 個特徵位置皆非零，但「非零」本身不證明無區分度或無效；應用遮罩測試及穩定性指標評估。

## 五、方法間一致性

下表取自 `variant_comparison.csv` 的 `ALL_SAMPLES_MEAN`：每組配對皆有 8 個有效樣本；Top-10 Jaccard 是各自前 10 特徵集合的交集／聯集，**不是重疊特徵所占前 10 的百分比**。

| 方法配對 | 平均 Spearman ρ | 平均 Top-10 Jaccard |
|---|---:|---:|
| Grad-CAM / Grad-CAM++ | 0.521 | 0.526 |
| Grad-CAM / Score-CAM | 0.139 | 0.151 |
| Grad-CAM++ / Score-CAM | 0.466 | 0.310 |

例子：001 的 Grad-CAM／Grad-CAM++ ρ=0.945、Jaccard=1.000，但 Grad-CAM／Score-CAM ρ=−0.479、Jaccard=0；003 的 Grad-CAM++／Score-CAM ρ=0.924、Jaccard=0.667；007 與 008 的 Grad-CAM／Grad-CAM++ ρ 都約 −0.069、Jaccard 都為 0。排名相關與 Top-10 重疊衡量不同層面，不必一致；一張圖中的並列值和排序規則也會影響 Top-10。方法有分歧不等於能從這些資料判定誰正確。

## 六、與舊版比較

| 項目 | 舊版 `compare` | fixed 版 |
|---|---|---|
| 目標層 | `encoder_conv.11` | `encoder_conv.13` |
| Grad-CAM 全零圖 | 4 / 8 | 0 / 8 |
| Grad-CAM++ 全零圖 | 6 / 8 | 0 / 8 |
| Score-CAM 全零圖 | 0 / 8 | 0 / 8 |
| 分組彙整 | 依預期群組 TN/FP/TP/FN 各 2 | 依實際群組 TN 1、FP 3、TP 1、FN 3 |
| 診斷 | 時間與熱圖統計 | 額外記錄 activation、gradient、signed CAM 與 Score-CAM 權重診斷 |

兩版的 8 筆樣本、記載之實際分數及 624 個縮放特徵數值相同；fixed 多了 `observed_group` 欄位。兩版熱圖受目標層及其他可能的實作變更影響，不可只根據本比較斷定特定改動是唯一原因。

**舊報告更正**：舊版全零圖的 Top-10 排名沒有實際特徵辨識意義；兩張全零圖可能因平手排序而得到 Jaccard 1，因此舊版 Grad-CAM／Grad-CAM++ 的 0.514 平均 Jaccard 不代表有效解釋高度一致。舊版兩者的平均 Spearman ρ=0.747 只由 007、008 兩筆有效值計算；不能直接與 fixed 的 8 筆均值比較。舊報告將方法整體判為不適用、僅因非零就稱 Score-CAM 區分度低，也都過於武斷；應以固定層、具體樣本與忠實度驗證描述。

## 七、時間與限制

| 方法 | 平均秒／樣本 | 中位數秒／樣本 | 排除 001 後平均秒／樣本 |
|---|---:|---:|---:|
| Grad-CAM | 0.02047 | 0.00749 | 0.00728 |
| Grad-CAM++ | 0.00832 | 0.00736 | 0.00836 |
| Score-CAM | 0.01707 | 0.00934 | 0.00991 |

Grad-CAM 的 001 耗時 0.11275 秒、Score-CAM 的 001 耗時 0.06717 秒，明顯高於各自其他樣本。可能受初始化或環境影響，但沒有獨立暖機紀錄，不可確定原因。單次、8 筆的測量不足以作為嚴格效能評測。

重要限制：

1. **2D 排列與插值**：表格欄位相鄰不必然有語意關聯。CAM 在目標層空間圖上插值到輸入大小；鄰近特徵可能共享相似熱度。改變欄位順序可能改變解釋，尚未測試。
2. **padding 熱度**：每張圖 946 / 1,024 格是 padding；像素 CSV 顯示 padding 區也可能有非零熱度，不應將其當作原始特徵。24 張全圖的最大值均為 1，但正規化依整張熱圖，特徵區最大值不一定為 1。
3. **歸一化與正負方向**：各張熱圖各自正規化，跨樣本絕對熱度與平均熱度不應視為可比較的貢獻大小；Grad-CAM 相關圖主要保留提高異常分數的正向證據，不是完整的正負貢獻。
4. **零值亦可能高熱**：例如 001 的 Grad-CAM Top-1 `Destination Port` 的縮放輸入為 0，而熱度為 1；003 的 `CWE Flag Count` 縮放值為 0、熱度約 0.999。這不必然是錯誤，但提醒 CAM 熱度不是輸入數值或單特徵介入效果。
5. **樣本與分類**：只有 8 個刻意挑選的個案，TN、TP 各 1 筆，不能估計整體準確率或各群組的穩定特徵，也不宜視為代表性抽樣。

## 八、建議後續驗證

1. 釐清 001／005 與 `reported_score_ch4` 的差異：固定 checkpoint、預處理、隨機種子與推論模式，多次重算並記錄結果；不要只改原始預期標記。
2. 使用真實 `observed_group` 擴大樣本，檢查來源重複與相近流量；以適當的群組數與信賴區間評估穩定性，而不是只看少量個案。
3. 對 Top-k 特徵做合適基線的遮罩／置換與對照，量測異常分數變化；比較隨機特徵及低排名特徵，同時考量特徵間相依與分布外輸入。
4. 與 Integrated Gradients 等輸入層方法交叉驗證，檢查正負方向、不同目標層、特徵排列與重跑的穩定性；不要把一致性當成忠實度的替代品。
5. 若需比較效能，固定硬體及同步方式，先暖機、重複多次並報告分布。

## 九、檢查範圍

- 讀取並分析 JSON、全部 7 個 CSV，依特徵 CSV 重算逐樣本排名、非零數與 007／008 差異，並檢查原有一致性與執行時間統計。
- 逐一載入 24 個 NPY，確認形狀為 32×32、數值有限且位於 [0,1]；逐列比對 24,576 筆像素 CSV 與 1,872 筆特徵 CSV 對應的 NPY 值，最大差皆為 0。比較兩版 624 個縮放輸入值，最大差為 0。
- 逐一以影像解碼器驗證 81 個 PNG 可讀；**未逐張檢視視覺內容，也未驗證圖表標註的正確性**。
- 未重新執行模型、檢查原始資料集或驗證解釋忠實度；報告中的分數和分類取自既有輸出。
