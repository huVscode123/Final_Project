# CICIDS2017 特徵公式一致性稽核

- Sample rows: 50000 (0=all)
- Seed: 42
- 只讀取 CSV；不會修改輸入、模型、scaler 或閾值。
- 這份稽核檢查特徵之間的逐列數學關係；不代表已證明 PCAP 與 CSV 來源完全相同。

## TRAIN CSV

- 檔案：`data/cicids2017/Monday-WorkingHours.pcap_ISCX.csv`
- 原始列數：529,918
- 標籤篩選後列數：529,918
- 實際稽核列數：50,000

| 公式／檢查 | 可用 | 樣本數 | 符合率／摘要 | 中位絕對誤差 | P95 絕對誤差 |
|---|---:|---:|---:|---:|---:|
| Fwd Packet Length Mean == total_fwd_bytes / total_fwd_packets | 是 | 50,000 | 100.0% | 0 | 3.33387e-11 |
| Bwd Packet Length Mean == total_bwd_bytes / total_bwd_packets | 是 | 42,685 | 100.0% | 0 | 3.34239e-11 |
| Packet Length Mean == (total_fwd_bytes + total_bwd_bytes) / total_packets | 是 | 50,000 | 24.774% | 4.8 | 23.6 |
| Average Packet Size == (total_fwd_bytes + total_bwd_bytes) / total_packets | 是 | 50,000 | 37.188% | 3 | 26.5 |
| Packet Length Mean == Average Packet Size | 是 | 50,000 | 10.848% | 11.3333 | 39.6667 |
| Packet Length Variance == Packet Length Std ** 2 | 是 | 50,000 | 100.0% | 1.08651e-09 | 3.88842e-08 |
| Avg Fwd Segment Size == total_fwd_bytes / total_fwd_packets | 是 | 50,000 | 100.0% | 0 | 3.33387e-11 |
| Avg Bwd Segment Size == total_bwd_bytes / total_bwd_packets | 是 | 42,685 | 100.0% | 0 | 3.34239e-11 |
| Avg Fwd Segment Size == Fwd Packet Length Mean | 是 | 50,000 | 100.0% | 0 | 0 |
| Avg Bwd Segment Size == Bwd Packet Length Mean | 是 | 50,000 | 100.0% | 0 | 0 |
| Flow Bytes/s == count_or_bytes / duration_seconds | 是 | 49,966 | 100.0% | 2.74172e-11 | 3.72529e-09 |
| Flow Packets/s == count_or_bytes / duration_seconds | 是 | 49,966 | 100.0% | 2.54659e-11 | 5.63887e-11 |
| Fwd Packets/s == count_or_bytes / duration_seconds | 是 | 49,966 | 100.0% | 2.40306e-11 | 5.45697e-11 |
| Bwd Packets/s == count_or_bytes / duration_seconds | 是 | 49,966 | 100.0% | 1.92157e-11 | 4.78679e-11 |
| Down/Up Ratio == floating Bwd/Fwd ratio | 是 | 50,000 | 70.114% | 0 | 0.814815 |
| Down/Up Ratio == floor(Bwd/Fwd) (legacy integer division candidate) | 是 | 50,000 | 100.0% | 0 | 0 |
| Fwd Avg Bytes/Bulk: rows nonzero | 是 | 50,000 | 非零 0 (0.0%) | - | - |
| Fwd Avg Packets/Bulk: rows nonzero | 是 | 50,000 | 非零 0 (0.0%) | - | - |
| Fwd Avg Bulk Rate: rows nonzero | 是 | 50,000 | 非零 0 (0.0%) | - | - |
| Bwd Avg Bytes/Bulk: rows nonzero | 是 | 50,000 | 非零 0 (0.0%) | - | - |
| Bwd Avg Packets/Bulk: rows nonzero | 是 | 50,000 | 非零 0 (0.0%) | - | - |
| Bwd Avg Bulk Rate: rows nonzero | 是 | 50,000 | 非零 0 (0.0%) | - | - |
| Fwd Header Length.1 == Fwd Header Length | 是 | 50,000 | 100.0% | - | - |
| act_data_pkt_fwd bounds | 是 | 50,000 | negative=0; >Fwd=0; >=Fwd=0 | - | - |
| Init_Win_bytes_backward when Total Backward Packets == 0 | 是 | 7,315 | -1=100.0%; 0=0.0% | - | - |
| Short-flow Subflow Fwd Bytes == Total Length of Fwd Packets | 是 | 33,736 | 100.0% | - | - |

## PCAP CANDIDATE

- 檔案：`output/diagnostics/cicids_compat_candidate/Monday-WorkingHours.compat.csv`
- 原始列數：391,671
- 標籤篩選後列數：391,671
- 實際稽核列數：50,000

| 公式／檢查 | 可用 | 樣本數 | 符合率／摘要 | 中位絕對誤差 | P95 絕對誤差 |
|---|---:|---:|---:|---:|---:|
| Fwd Packet Length Mean == total_fwd_bytes / total_fwd_packets | 是 | 50,000 | 100.0% | 0 | 0 |
| Bwd Packet Length Mean == total_bwd_bytes / total_bwd_packets | 是 | 49,034 | 100.0% | 0 | 0 |
| Packet Length Mean == (total_fwd_bytes + total_bwd_bytes) / total_packets | 是 | 50,000 | 100.0% | 0 | 0 |
| Average Packet Size == (total_fwd_bytes + total_bwd_bytes) / total_packets | 是 | 50,000 | 100.0% | 0 | 0 |
| Packet Length Mean == Average Packet Size | 是 | 50,000 | 100.0% | 0 | 0 |
| Packet Length Variance == Packet Length Std ** 2 | 是 | 50,000 | 100.0% | 0 | 2.91038e-11 |
| Avg Fwd Segment Size == total_fwd_bytes / total_fwd_packets | 是 | 50,000 | 100.0% | 0 | 0 |
| Avg Bwd Segment Size == total_bwd_bytes / total_bwd_packets | 是 | 49,034 | 100.0% | 0 | 0 |
| Avg Fwd Segment Size == Fwd Packet Length Mean | 是 | 50,000 | 100.0% | 0 | 0 |
| Avg Bwd Segment Size == Bwd Packet Length Mean | 是 | 50,000 | 100.0% | 0 | 0 |
| Flow Bytes/s == count_or_bytes / duration_seconds | 是 | 49,991 | 100.0% | 0 | 1.16415e-10 |
| Flow Packets/s == count_or_bytes / duration_seconds | 是 | 49,991 | 100.0% | 0 | 3.63798e-12 |
| Fwd Packets/s == count_or_bytes / duration_seconds | 是 | 49,991 | 100.0% | 0 | 1.81899e-12 |
| Bwd Packets/s == count_or_bytes / duration_seconds | 是 | 49,991 | 100.0% | 0 | 1.81899e-12 |
| Down/Up Ratio == floating Bwd/Fwd ratio | 是 | 50,000 | 100.0% | 0 | 0 |
| Down/Up Ratio == floor(Bwd/Fwd) (legacy integer division candidate) | 是 | 50,000 | 65.446% | 0 | 0.878788 |
| Fwd Avg Bytes/Bulk: rows nonzero | 是 | 50,000 | 非零 0 (0.0%) | - | - |
| Fwd Avg Packets/Bulk: rows nonzero | 是 | 50,000 | 非零 0 (0.0%) | - | - |
| Fwd Avg Bulk Rate: rows nonzero | 是 | 50,000 | 非零 0 (0.0%) | - | - |
| Bwd Avg Bytes/Bulk: rows nonzero | 是 | 50,000 | 非零 0 (0.0%) | - | - |
| Bwd Avg Packets/Bulk: rows nonzero | 是 | 50,000 | 非零 0 (0.0%) | - | - |
| Bwd Avg Bulk Rate: rows nonzero | 是 | 50,000 | 非零 0 (0.0%) | - | - |
| Fwd Header Length.1 == Fwd Header Length | 是 | 50,000 | 100.0% | - | - |
| act_data_pkt_fwd bounds | 是 | 50,000 | negative=0; >Fwd=0; >=Fwd=29578 | - | - |
| Init_Win_bytes_backward when Total Backward Packets == 0 | 是 | 966 | -1=100.0%; 0=0.0% | - | - |
| Short-flow Subflow Fwd Bytes == Total Length of Fwd Packets | 是 | 33,879 | 100.0% | - | - |

## 如何使用這份稽核

1. `Down/Up Ratio` 若只符合 floor(Bwd/Fwd)，候選轉換器才應採用整數除法；若更符合浮點比率，維持浮點算法。
2. 若訓練 CSV 的 `Packet Length Mean`、`Average Packet Size` 都不等於總 payload／總封包數，且兩者彼此不相等，需查核目標 CICFlowMeter 版本是否對第一個封包重複累計，再決定是否模仿該歷史行為。
3. 若 `Packet Length Variance` 不等於 `Packet Length Std ** 2`，先確認是統計分母（樣本／母體）差異或 CSV 產生器版本差異。
4. `Fwd Packet Length Mean` 與 `Avg Fwd Segment Size`、`Bwd Packet Length Mean` 與 `Avg Bwd Segment Size` 若不一致，也需先查版本／公式，不要直接強制相等。
5. 修正後仍需以相同 checkpoint、scaler、特徵順序、clipping 和計分函式重新比較分數。不要直接調高閾值。

JSON 詳細輸出：`formula_audit.json`
