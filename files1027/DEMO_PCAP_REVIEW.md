# CICIDS2017 Demo PCAP：問題診斷與解法

## 一、先釐清：舊報告有哪些結論站不住腳（以你的原始碼實測）

| # | 舊報告說法 | 實測 / 讀碼結果 |
|---|---|---|
| 1 | DDoS 失敗是因為「SYN-only 連線要等 120 秒 timeout」 | 錯。`pcap_flow_converter.py` 檔尾會 flush 所有 Flow（`eof`）。真正原因是 **`packet_count <= 1` 的 Flow 被丟棄**：SYN flood 每個 sport 只有 1 個封包 → 0 條 Flow。實測 200 個 unique-sport SYN → `flows=0`。 |
| 2 | DDoS / SSH / Web BF / Hulk 的 `Port Scan (TCP Scan)` 告警「合理」 | 錯，是**歸因錯誤**。實測告警 `src_ip = 192.168.10.50`（受害者）。原因：`_check_port_scan` 只排除 `flags == "RST"`，但 parser 輸出 `"RST+ACK"`、`"SYN+ACK"`、`"PSH+ACK"`，受害者回給攻擊者各個臨時 port 的封包被當成「反向掃描」。 |
| 3 | PortScan 的 `ACK Scan ×2` 告警「正確」 | 高度可疑（依讀碼推斷，舊 PCAP 不在手上無法重跑）：`RST+ACK` 回應被計入並被 `_classify_port_scan` 歸為 ACK Scan。 |
| 4 | semi 模型「已知/未知 = 0/40」暗示分類器沒作用 | 不是 bug：`hybrid_decision` 設計是「重建誤差超過閾值 → 一律 unknown，優先於分類」。且分類器 `num_classes=2`（攻擊/正常），**本來就無法說出攻擊類型**。 |
| 5 | 正常流量誤報「在真實環境應接近 3.84%/4.79%」 | 本次測試無法支持這句話；那是 CSV 評估的 FPR，PCAP→Flow 這條路徑的 FPR 需另外量。 |
| 6 | 1~3 條 Flow 的 PCAP 判 PASS/WARN | 統計上無意義：1 條 Flow 的異常率只能是 0% 或 100%。 |
| 7 | Grad-CAM「每 session 6 張以上」 | 張數與 Flow 數不成比例（1 flow→3、5→6、8→12、100→6 或 12），需確認選取邏輯（見第四節）。 |

規則引擎只有 7 條規則（SYN Flood / Port Scan / ICMP Flood / UDP Flood / ARP / DNS Amp / DNS Tunnel），
**根本沒有 SSH 暴力破解、Web 暴力破解、HTTP Flood 的規則**，所以舊 PCAP 不可能「容易看出是哪種攻擊」。

## 二、新版解法

1. `generate_cicids2017_demo_pcaps_v2.py`：重新產生 10 個 PCAP（總計約 420 KB）
   - 拓樸沿用 CICIDS2017：攻擊者 172.16.0.1 → 受害者 192.168.10.50；正常為 192.168.10.x。
   - 每個 PCAP 都有足夠 Flow 數；每條 TCP Flow ≥ 2 封包；時間有抖動；
     OS 視窗值 / TCP options 貼近真實（Init_Win、Header Length 不再是 Scapy 預設）。
   - Ethernet link-layer；預設 `--fin-tail drop`：每條連線剛好 1 個 Flow
     （保留 FIN 尾 ACK 會多出 2 封包的 appendix flow，正常與攻擊兩邊都有，稀釋異常率）。
   - ICMP 換成 HTTPS(443)：ICMP 不是 CICIDS2017 的典型流量，且我無法從匯出的程式確認訓練集含 ICMP Flow。要換回請自行加回。
2. `anomaly_detector_extra.py`：**子類別**擴充，不改原檔
   - 修正 Port Scan 只計探測封包（排除含 ACK / RST）。
   - 新增 SSH/FTP/RDP Brute Force、Web Login Brute Force、HTTP Flood (Hulk-like) 三條規則。
3. `check_demo_pcaps.py`：跑模型**之前**的健檢（Flow 數、告警與歸因、可選的與 CICIDS2017 CSV 分布比對）。

| PCAP | 大小 | 封包 | Flow | 擴充規則引擎的告警 |
|---|---|---|---|---|
| attack_01_ddos | 90 KB | 660 | 110 | SYN Flood @172.16.0.1 |
| attack_02_portscan | 14 KB | 182 | 90 | Port Scan (SYN Scan) @172.16.0.1 |
| attack_03_ssh_bruteforce | 78 KB | 418 | 22 | SSH Brute Force @172.16.0.1 |
| attack_04_web_bruteforce | 66 KB | 170 | 10 | Web Login Brute Force @172.16.0.1 |
| attack_05_dos_hulk | 85 KB | 420 | 30 | HTTP Flood (Hulk-like) @172.16.0.1 |
| normal_01_web_browsing | 40 KB | 150 | 12 | 無 |
| normal_02_dns_queries | 7 KB | 60 | 30 | 無 |
| normal_03_ssh_session | 40 KB | 207 | 5 | 無 |
| normal_04_email_imap | 11 KB | 90 | 5 | 無 |
| normal_05_https_browsing | 47 KB | 142 | 9 | 無 |

以上是在沙箱用你匯出的 `pcap_flow_converter.py`、`anomaly_detector.py`、`parser.py` 實跑的結果。
**未驗證**：模型推論、Grad-CAM、報告/圖表（沒有 checkpoint）。

## 三、你本機要做的步驟

```powershell
cd core
python generate_cicids2017_demo_pcaps_v2.py --out ..\test_pcaps\cicids2017_demo_v2
python check_demo_pcaps.py --pcap-dir ..\test_pcaps\cicids2017_demo_v2 --detector extended --csv-dir ..\data\cicids2017
python check_demo_pcaps.py --pcap-dir ..\test_pcaps\cicids2017_demo_v2 --detector original   # 對照：會看到受害者誤報
```
再把 `test_cicids2017_demo_e2e.py` 的 PCAP 目錄指向 v2，並把 `AnomalyDetector` 換成 `ExtendedAnomalyDetector`。

`--csv-dir` 的「落在 P1~P99 比例」是關鍵指標：若某個 PCAP 對自己的類別比例很低（例如 < 70%），
就用 CSV 的類別中位數回頭調那個攻擊的封包數 / 間隔 / 載荷，而不是調模型。
（我沒有你的 CSV，攻擊特徵目前是依公開描述與常識設計，需要用這個指標校正。）

## 四、檢查模型輸出是否「正確、合理」的清單

先做一個控制實驗，把「模型/前處理有問題」和「合成資料分布偏移」分開：
用 `compare_cicids_csv_vs_pcap_inference.py` 對**真實 CICIDS2017 PCAP 片段**（官方有發布）跑同一條路徑。
真實 PCAP 的 FPR 正常、合成 PCAP 偏高 → 是合成資料問題；兩者都高 → 是 PCAP→Flow 或前處理問題。

| 項目 | 合理性檢查 |
|---|---|
| 異常率 | `異常 Flow 數 / Flow 數` 與 JSON 一致；另外記錄 `avg_error / threshold`（倍數），只報百分比看不出離閾值多遠 |
| 閾值 | 報告顯示的閾值 = checkpoint 內的閾值（unsup 0.00072870、semi 0.00221278） |
| 告警歸因 | `src_ip` 必須是攻擊者；攻擊類型要與 PCAP 對得上，不能只檢查「有告警」 |
| semi 的已知/未知 | 異常 Flow 全部算 unknown 是設計；要驗分類器，需挑重建誤差 < 閾值的 Flow 看分類結果 |
| Grad-CAM | 確認張數的選取規則（top-k 異常？）；攻擊與正常的熱力圖應有可辨差異（例如攻擊集中在封包數/速率相關的格子），且 3 種 variant 與同一 Flow 對得上 |
| 圖表 | 每張圖的分母（Flow 數、封包數）與 CSV/JSON 一致；PortScan 這種 2 封包 Flow 不該出現在「長連線」統計 |
| 通過判定 | 攻擊 PASS 條件加上「期望告警類型 + 歸因正確」；正常 PASS 條件加上「Flow 數 ≥ 5」 |

## 五、已知限制
- 合成資料一定有分布偏移；v2 只是縮小它，不保證兩個模型都 0 誤報。
- PortScan 在 semi 模型上偏弱（2 封包小 Flow 與短連線重疊）是模型/閾值問題，PCAP 不該為此作弊。這個案例本身就是很好的 demo：AI 分支漏掉，規則引擎補上。
- 新增的三條規則閾值是為 demo 規模（幾十~一百個請求）設定，上線前要用真實流量調整。
