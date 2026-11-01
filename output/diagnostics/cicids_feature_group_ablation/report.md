# CICIDS2017 特徵群組反事實診斷

- 產生時間：2026-10-19T13:27:56
- 模型：`output\threshold_study\unsup_cicids2017\best_vae_cicids2017.pt`，類型 `cnn_vae`，representation `csv_features_cicids2017`
- 閾值：`0.000819941080408`
- 訓練 CSV 抽樣：50,000；PCAP 候選樣本：50,000

> 重要：這是診斷用反事實試驗。將 PCAP 的某組特徵換成訓練資料的值，只用來判斷該特徵群是否與分數差異有關，不是正確的轉換方式，也不可用來宣稱模型效能改善。

## 基準

| 資料 | 異常率 | Score p50 | Score p95 | Score p99 |
|---|---:|---:|---:|---:|
| TRAIN CSV BENIGN | 2.88% | 0.00034530071 | 0.00070532498 | 0.001431215 |
| PCAP candidate baseline | 45.43% | 0.00068665668 | 0.0077642415 | 0.010176362 |

## 單一群組替換結果

異常率越接近訓練 CSV 不代表可以直接套用這項修正；需回到轉換器與 CICFlowMeter v3.0 定義查證。

| 情境 | 替換欄位數 | 異常率 | 相對基準變化 (百分點) | Score p95 | Score p99 |
|---|---:|---:|---:|---:|---:|
| `replace_min_seg_size_forward_with_train_values` | 1 | 47.76% | +2.32 | 0.0078766237 | 0.010280892 |
| `replace_all_candidate_groups_with_train_values` | 67 | 56.32% | +10.88 | 0.0048190478 | 0.0061103203 |
| `replace_act_data_pkt_fwd_with_train_values` | 1 | 59.90% | +14.46 | 0.0083054584 | 0.010629793 |
| `replace_flags_with_train_values` | 12 | 68.12% | +22.69 | 0.004834417 | 0.0066976294 |
| `replace_initial_TCP_window_with_train_values` | 2 | 68.69% | +23.25 | 0.0086682876 | 0.012406587 |
| `replace_packet_counts_and_total_bytes_with_train_values` | 4 | 71.50% | +26.07 | 0.010677004 | 0.012587718 |
| `replace_subflow_with_train_values` | 4 | 71.90% | +26.47 | 0.011302871 | 0.012949511 |
| `replace_rates_duration_active_idle_with_train_values` | 13 | 74.87% | +29.44 | 0.011792446 | 0.014800869 |
| `replace_IAT_with_train_values` | 14 | 78.85% | +33.42 | 0.018720872 | 0.022544468 |
| `replace_packet_length_and_segment_size_with_train_values` | 18 | 85.81% | +40.38 | 0.011583354 | 0.014730933 |

## 判讀

- 若替換某一群組使異常率與 p95 大幅下降，該群組值得優先追查其 PCAP 特徵公式／版本差異。
- 若只有把所有群組一起替換才大幅下降，表示有多組差異共同影響模型，應對照原始 CICFlowMeter 特徵生成器。
- 若全部替換後分數仍顯著偏高，除列出的群組外還存在其他資料表示法／Flow 組成差異。
- 這個試驗不能告訴你訓練資料的數值一定正確，只表示模型目前對哪組輸入偏移較敏感。

## 檔案

- `feature_group_ablation.csv`：完整群組替換結果。
- `summary.json`：基準資訊、分組欄位與摘要。
