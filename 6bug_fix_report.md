# 6 Bug 問題說明與解決方法報告

> 本報告詳細說明 6 個已修正的 bug，涵蓋「封包擷取與分析」、「CNN 異常偵測」、「熱力圖產生」、「報告與圖表產生」四大功能區。

---

## 目錄

1. [Bug A — PCAP 分段儲存重疊](#bug-a--pcap-分段儲存重疊)
2. [Bug B — packet_count 賦值位置錯誤](#bug-b--packet_count-賦值位置錯誤)
3. [Bug C — Grad-CAM 半監督模型判定邏輯不完整](#bug-c--grad-cam-半監督模型判定邏輯不完整)
4. [Bug D — 重建誤差分布圖使用假高斯曲線](#bug-d--重建誤差分布圖使用假高斯曲線)
5. [Bug E — heatmap_analysis.html 引用不存在的欄位](#bug-e--heatmap_analysishtml-引用不存在的欄位)
6. [Bug F — gradcam_gallery.html 引用不存在的欄位](#bug-f--gradcam_galleryhtml-引用不存在的欄位)
7. [熱力圖完整性檢查結果](#熱力圖完整性檢查結果)
8. [單元測試驗證](#單元測試驗證)

---

## Bug A — PCAP 分段儲存重疊

**影響範圍：** 封包擷取與分析
**嚴重度：** HIGH
**修正檔案：** core/capture.py（第 312-325 行）

### 問題描述

LiveCapture 類別的 _check_segment_save() 方法負責在即時擷取過程中，每隔一段封包數或時間就將記憶體中的封包寫出為一個 PCAP 分段檔。

問題出在第 315 行：執行 packets_copy = list(self.packets) 複製了整個封包列表後，並未清空 self.packets。這導致了兩個嚴重後果：

1. **分段檔重疊**：第 1 個分段包含封包 1~1000，第 2 個分段包含封包 1~2000（而非預期的 1001~2000），每個分段檔都會越來越大，浪費磁碟空間且資料冗餘。

2. **封包永久丟失**：因為 self.packets 不清空，列表會持續成長。當達到 max_memory_packets 上限（預設 100,000）時，記憶體保護機制會從列表開頭刪除最舊的封包（第 206-209 行），但這些封包可能尚未被任何分段檔寫出——就這樣永久消失了。

### 修正前程式碼

```python
# capture.py L312-317（修正前）
if should_save and self.packets:
    self._segment_index += 1
    segment_file = self._get_segment_filename()
    packets_copy = list(self.packets)    # ← 複製後未清空！
    self._segment_counter = 0
    self._last_segment_time = now
```

### 修正後程式碼

```python
# capture.py L312-325（修正後）
if should_save and self.packets:
    self._segment_index += 1
    segment_file = self._get_segment_filename()
    packets_copy = list(self.packets)
    # [Bug A 修正] 分段寫出後清空緩衝區
    self.packets.clear()
    self.parsed_records.clear()
    self._segment_counter = 0
    self._last_segment_time = now
```

### 為何這樣修正是安全的

- **統計計數器不受影響**：packet_count、total_bytes、proto_counter 等累計統計在 _packet_callback() 中已即時更新（第 181-197 行），與 self.packets 列表的內容無關。
- **清空在鎖內執行**：self.packets.clear() 在 with self._lock: 保護下執行，不會與 _packet_callback() 的 append 產生競爭條件。

---

## Bug B — packet_count 賦值位置錯誤

**影響範圍：** 封包擷取與分析
**嚴重度：** MEDIUM
**修正檔案：** analyzer/tasks.py（第 100-125 行）

### 問題描述

run_pcap_analysis() 任務的「規則式偵測」區段中，session.packet_count 的賦值原本寫在 detect_attacks() 之後（內層 try 區塊的最後一行）。

如果 detect_attacks() 拋出例外（例如某個攻擊偵測器遇到格式異常的封包），程式碼會跳到 except 分支，session.packet_count 的賦值永遠不會執行。最終資料庫中的 packet_count 會是預設值 0，即使 PCAP 檔案確實有數百個封包、CNN 也可能正常產出結果。

### 修正前程式碼

```python
# tasks.py（修正前）
try:
    analyzer = PcapAnalyzer(pcap_path)
    analyzer.load()
    analysis_result = analyzer.detect_attacks()  # ← 這裡可能拋例外
    alerts_raw = analysis_result.get("alerts", [])
    summary_info = analysis_result.get("summary", {})
    session.packet_count = summary_info.get('total_packets', 0)  # ← 永遠不到
except Exception as e:
    alerts_raw = []  # packet_count 維持 0
```

### 修正後程式碼

```python
# tasks.py（修正後）
try:
    analyzer = PcapAnalyzer(pcap_path)
    analyzer.load()
    # [Bug B 修正] 先記錄封包數量，再執行偵測
    session.packet_count = len(analyzer.packets) if hasattr(analyzer, 'packets') else 0
    analysis_result = analyzer.detect_attacks()
    alerts_raw = analysis_result.get("alerts", [])
    summary_info = analysis_result.get("summary", {})
    if summary_info.get('total_packets', 0) > 0:
        session.packet_count = summary_info['total_packets']
except Exception as e:
    alerts_raw = []  # packet_count 已在上方設定，不會是 0
```

### 修正邏輯

1. 在 analyzer.load() 成功後（封包已載入記憶體），立即從 len(analyzer.packets) 取得封包數。
2. 如果後續 detect_attacks() 的 summary_info 提供了更精確的數字，再覆蓋。
3. 即使 detect_attacks() 失敗，封包數量仍然會被正確記錄。

---

## Bug C — Grad-CAM 半監督模型判定邏輯不完整

**影響範圍：** CNN、熱力圖產生
**嚴重度：** MEDIUM
**修正檔案：** analyzer/tasks.py（第 313-330 行）

### 問題描述

run_gradcam() 任務負責對每個封包產生 Grad-CAM 熱力圖，並在 GradCAMImage 記錄中標記 is_anomaly。

原本的異常判定寫死為 is_anomaly = err > threshold（第 310 行），只看 VAE 的重建誤差是否超過閾值。對於半監督模型（hybrid_semi），這完全忽略了 CNN-LSTM 分類器的「已知攻擊」判定。

這會導致：
- 分類器明確判定為攻擊的封包，如果 VAE 重建誤差恰好低於閾值，就會被標記為「正常」。
- GradCAMImage.is_anomaly 與 CNNResult 中 known_attack_count 的統計互相矛盾。
- 前端 Grad-CAM 圖庫的「異常/正常」分類與 Session 詳情頁的攻擊統計不一致。

### 修正後程式碼

```python
# tasks.py L313-328（修正後）
with torch.no_grad():
    err = float(model.reconstruction_error(x_tensor).cpu().numpy()[0])

# [Bug C 修正] 使用與 compute_anomaly_scores() 一致的三態邏輯
if bundle.is_hybrid:
    with torch.no_grad():
        probs = model.classify_known(x_tensor).cpu()
        _, label = probs.max(dim=1)
        classifier_says_attack = (label.item() == 1)
    is_anomaly = classifier_says_attack or (err > threshold)
else:
    is_anomaly = err > threshold
```

### 三態判定邏輯對照

| 狀態 | 分類器 | VAE 重建誤差 | is_anomaly |
|---|---|---|---|
| **已知攻擊** (known_attack) | 判定為攻擊 | 不論 | True |
| **未知/新型攻擊** (unknown_attack) | 判定為正常 | 超過閾值 | True |
| **正常** (normal) | 判定為正常 | 未超過閾值 | False |

---

## Bug D — 重建誤差分布圖使用假高斯曲線

**影響範圍：** 報告與圖表產生
**嚴重度：** LOW
**修正檔案：** templates/analyzer/session_detail.html（第 326-388 行）

### 問題描述

Session 詳情頁面中的「CNN 重建誤差分布圖」，原本是用資料庫中儲存的 avg_normal_error 和 avg_attack_error 兩個平均值，套用高斯公式 Math.exp(-0.5 * Math.pow(...)) 反推出一條假的鐘型曲線。

這不是真正的分布資料，而是一個完全虛構的圖表。它會給使用者造成誤導，以為系統真的量測到了某種分布，但實際上根本沒有儲存每個封包的個別誤差值。

### 解決方法

改為水平 bar chart，直接對比三個真實數值：
- 正常封包平均誤差（綠色）
- 異常封包平均誤差（紅色）
- 判定閾值（橘色）

圖表標題明確標註「實際量測之重建誤差對比（平均值）」，不偽造任何分布資料。

---

## Bug E — heatmap_analysis.html 引用不存在的欄位

**影響範圍：** 報告與圖表產生
**嚴重度：** HIGH
**修正檔案：** templates/analyzer/heatmap_analysis.html（第 48, 52 行）

### 問題描述

熱力圖深度分析頁面的「CNN 模型分析摘要」卡片中引用了 CNNResult 模型不存在的欄位：

- 第 48 行：cnn_result.avg_reconstruction_error（不存在）
- 第 52 行：cnn_result.threshold_used（不存在）

Django 模板在引用不存在的屬性時不會報錯，而是靜默輸出空字串。結果就是這兩個 KPI 數值卡片永遠顯示為空白。

### 修正對照

| 位置 | 修正前（不存在） | 修正後（正確） | 標籤文字 |
|---|---|---|---|
| L48 | cnn_result.avg_reconstruction_error | cnn_result.avg_attack_error | 「異常封包平均誤差」 |
| L52 | cnn_result.threshold_used | cnn_result.threshold | 「使用閾值」 |

---

## Bug F — gradcam_gallery.html 引用不存在的欄位

**影響範圍：** 熱力圖產生
**嚴重度：** CRITICAL（本次最嚴重的 bug）
**修正檔案：** templates/analyzer/gradcam_gallery.html, templates/analyzer/heatmap_analysis.html

### 問題描述

GradCAMImage 模型的影像欄位定義為 original_image、heatmap_image、comparison_image，分數欄位為 recon_error。但兩個模板中大量引用了不存在的欄位名稱：

- img.image.url（不存在 image 欄位）→ 導致所有圖片都是破圖
- img.anomaly_score（不存在 anomaly_score 欄位）→ 導致所有分數都不顯示

此外，gradcam_gallery.html 的 Grad-CAM++ 篩選按鈕使用 variant=gradcam_pp，但模型中定義的值是 gradcam++，導致篩選永遠無結果。

### 完整修正清單

| 檔案 | 修正前 | 修正後 | 修正處數 |
|---|---|---|---|
| gradcam_gallery.html | img.image.url | img.comparison_image.url | 1 處 |
| gradcam_gallery.html | img.anomaly_score | img.recon_error | 2 處 |
| gradcam_gallery.html | variant=gradcam_pp | variant=gradcam%2B%2B | 1 處 |
| heatmap_analysis.html | img.image.url | img.comparison_image.url | 4 處 |
| heatmap_analysis.html | img.anomaly_score | img.recon_error | 5 處 |

### 為何選擇 comparison_image 而非 heatmap_image

comparison_image 是原始封包影像和 Grad-CAM 熱力圖的並排對比圖，在圖庫瀏覽中資訊量最大（可同時看到原始封包結構和模型關注的區域）。

---

## 熱力圖完整性檢查結果

修正完成後，對熱力圖相關的完整程式碼路徑進行了逐行檢查，涵蓋 6 個檔案：

| 檢查目標 | 審查範圍 | 結果 |
|---|---|---|
| analyzer/views.py | 5 個 View 函式、Context 變數、Query 語法 | ✅ 全部正確 |
| analyzer/urls.py | gradcam_gallery, heatmap_analysis, trigger_gradcam | ✅ 全部正確 |
| analyzer/models.py | GradCAMImage 與 CNNResult 欄位定義 | ✅ 全部正確 |
| gradcam_gallery.html | 所有 {{ }} 與 {% %} 標籤 | ✅ 全部正確 |
| heatmap_analysis.html | 所有 {{ }} 與 {% %} 標籤 | ✅ 全部正確 |
| analyzer/tasks.py | run_gradcam() 儲存欄位與三態判定 | ✅ 全部正確 |

### 關鍵檢查點

- ✅ views.py 中所有 context 變數名稱與模板引用一致
- ✅ GradCAMImage.objects.update_or_create() 使用正確欄位 is_anomaly 和 recon_error
- ✅ 影像儲存使用 original_image / heatmap_image / comparison_image
- ✅ 模板中不再有任何 img.image、img.anomaly_score、avg_reconstruction_error、threshold_used
- ✅ variant=gradcam%2B%2B 正確 URL 編碼
- ✅ hybrid 模型的三態判定邏輯與 model_registry.compute_anomaly_scores() 完全對齊

---

## 單元測試驗證

測試檔案：tests/test_6bug_fixes.py

```
tests/test_6bug_fixes.py  16 passed in 2.05s
```

| 測試類別 | 測試數 | 驗證內容 |
|---|---|---|
| TestBugA | 3 | 分段儲存後 packets 被清空、未觸發時不清空、index 遞增 |
| TestBugB | 1 | packet_count 賦值行號在 detect_attacks 之前 |
| TestBugC | 2 | hybrid 模型呼叫 classify_known、非 hybrid 使用簡單閾值 |
| TestBugD | 2 | 無假高斯公式、標註「實際量測」 |
| TestBugE | 2 | 無不存在欄位、有正確欄位 |
| TestBugF | 4 | gallery/heatmap 無不存在欄位、有正確欄位、URL 編碼正確 |
| TestModelFieldConsistency | 2 | models.py 欄位定義與模板引用交叉驗證 |
