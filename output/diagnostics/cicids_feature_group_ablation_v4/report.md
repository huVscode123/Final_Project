# CICIDS2017 特徵群組反事實診斷

- 產生時間：2026-10-19T15:49:51
- 模型：`output\threshold_study\unsup_cicids2017\best_vae_cicids2017.pt`，類型 `cnn_vae`，representation `csv_features_cicids2017`
- 閾值：`0.000819941080408`
- 訓練 CSV 抽樣：50,000；PCAP 候選樣本：50,000

> 重要：這是診斷用反事實試驗。將 PCAP 的某組特徵換成訓練資料的值，只用來判斷該特徵群是否與分數差異有關，不是正確的轉換方式，也不可用來宣稱模型效能改善。

## 基準

| 資料 | 異常率 | Score p50 | Score p95 | Score p99 |
|---|---:|---:|---:|---:|
| TRAIN CSV BENIGN | 2.88% | 0.00034530071 | 0.00070532498 | 0.001431215 |
| PCAP candidate baseline | 44.65% | 0.00047014951 | 0.0079674351 | 0.0095993366 |

## 單一群組替換結果

異常率越接近訓練 CSV 不代表可以直接套用這項修正；需回到轉換器與 CICFlowMeter v3.0 定義查證。

| 情境 | 替換欄位數 | 異常率 | 相對基準變化 (百分點) | Score p95 | Score p99 |
|---|---:|---:|---:|---:|---:|
| `replace_min_seg_size_forward_with_train_values` | 1 | 45.69% | +1.03 | 0.0081353248 | 0.0097472295 |
| `replace_all_candidate_groups_with_train_values` | 67 | 55.92% | +11.27 | 0.0048455044 | 0.0061606774 |
| `replace_act_data_pkt_fwd_with_train_values` | 1 | 60.13% | +15.47 | 0.0086553648 | 0.010365523 |
| `replace_initial_TCP_window_with_train_values` | 2 | 67.78% | +23.13 | 0.0085502166 | 0.011841303 |
| `replace_subflow_with_train_values` | 4 | 67.93% | +23.28 | 0.011647889 | 0.013360807 |
| `replace_packet_counts_and_total_bytes_with_train_values` | 4 | 68.07% | +23.41 | 0.010881598 | 0.012939965 |
| `replace_flags_with_train_values` | 12 | 68.81% | +24.16 | 0.0048674651 | 0.0064502906 |
| `replace_rates_duration_active_idle_with_train_values` | 13 | 72.54% | +27.89 | 0.01188722 | 0.014972503 |
| `replace_IAT_with_train_values` | 14 | 75.97% | +31.31 | 0.0191123 | 0.023367465 |
| `replace_packet_length_and_segment_size_with_train_values` | 18 | 82.36% | +37.71 | 0.011923259 | 0.01465566 |

## 判讀

- 若替換某一群組使異常率與 p95 大幅下降，該群組值得優先追查其 PCAP 特徵公式／版本差異。
- 若只有把所有群組一起替換才大幅下降，表示有多組差異共同影響模型，應對照原始 CICFlowMeter 特徵生成器。
- 若全部替換後分數仍顯著偏高，除列出的群組外還存在其他資料表示法／Flow 組成差異。
- 這個試驗不能告訴你訓練資料的數值一定正確，只表示模型目前對哪組輸入偏移較敏感。

## 檔案

- `feature_group_ablation.csv`：完整群組替換結果。
- `summary.json`：基準資訊、分組欄位與摘要。
