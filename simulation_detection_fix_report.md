# 異常封包模擬檢測 — 問題診斷與修正報告

> 對應功能：`analyzer/views.py::simulation_api`（異常封包模擬檢測 API）
> 診斷依據：使用者提供的兩張截圖（SYN Flood / 正常流量模擬結果）+ 專案完整原始碼
> 結論先講：**這不是單一個「哪邊寫錯」的 bug，是三個獨立問題疊加**，且其中最根本的一個
> 跟「規則寫得對不對」關係不大，是**訓練資料與服務時期資料表示法不一致**造成的架構性問題。

---

## 目錄

1. [TL;DR — 直接回答「是偵測程式錯還是模擬程式錯」](#tldr)
2. [根因 A（最關鍵）：CNN 模型訓練/服務資料表示法不一致](#root-a)
3. [根因 B：規則式門檻與畫面提示文字對不上、雙閾值同時顯示造成混淆](#root-b)
4. [根因 C：模擬 API 完全沒有模型快取，是「第一次跑很慢、之後變快」的主因](#root-c)
5. [黑箱問題：目前回傳給前端的資料太粗，看不到「為什麼」](#root-d)
6. [為什麼某些看起來像 bug 的地方，這次刻意沒有動](#why-not-changed)
7. [修正內容總覽](#summary)
8. [完整修正程式碼](#code)
   - 8.1 `core/model_registry.py`（新增模型快取）
   - 8.2 `analyzer/apps.py`（新增啟動預熱）
   - 8.3 `analyzer/views.py`（透明化：逐封包觸發依據、雙閾值說明、CNN 可信度提示）
   - 8.4 `templates/analyzer/simulation.html`（局部：KPI 文字修正 + 顯示新欄位）
9. [如何驗證這次修正](#verify)
10. [後續建議（超出本次修正範圍，但強烈建議規劃）](#next)

---

<a id="tldr"></a>
## 1. TL;DR — 直接回答「是偵測程式錯還是模擬程式錯」

| 問題 | 出在哪 | 是不是「寫錯」 |
|---|---|---|
| 正常流量分數（0.0061）比 SYN Flood（0.0049）還高 | **CNN 模型本身** — 訓練時吃的是 CSV 流量特徵向量，模擬 / 正式分析時卻是原始封包位元組 | 不是邏輯寫錯，是**架構設計不一致**，模型從沒見過現在餵給它的資料長相 |
| 畫面同時出現「0.000069」和「0.00842808」兩個閾值，不知道哪個是真的 | `analyzer/views.py`（顯示）+ `templates/analyzer/simulation.html`（KPI 卡片文字） | 是可以直接修正的呈現問題 |
| 規則式偵測門檻參考文字寫「SYN Flood >100」，但實際套用的是 11 | `analyzer/views.py::_detect_simulation_behavior` | 文字誤導使用者，屬於黑箱問題，需要修正顯示邏輯（本次**保留**實際門檻公式不變，理由見第 6 節） |
| 第一次模擬要等幾十秒，第二次之後不到 3 秒 | 模型從未被快取、`matplotlib` 首次匯入建字型快取的固定成本 | 是效能設計缺陷，可直接修正 |

**一句話總結**：規則式偵測（SYN/Port Scan/ICMP/UDP/ARP/DNS）的邏輯本身沒有錯，真正不可信的是 **CNN 分數**；而「閾值看起來不合理」則是**呈現層**把兩種完全不同意義的閾值混在一起顯示造成的。

---

<a id="root-a"></a>
## 2. 根因 A（最關鍵）：CNN 模型訓練/服務資料表示法不一致

### 證據

**訓練時**（不論是 `run_training.py` 訓練 `unsupervised_vae`，還是 `run_semi_supervised.py` 訓練三個 `semi_*` 模型），資料來源都是 `core/dataset_loader.py::DatasetFactory`：

```python
# core/dataset_loader.py
IMAGE_SIZE  = 32
FEATURE_DIM = IMAGE_SIZE * IMAGE_SIZE   # 1024

@staticmethod
def _features_to_images(features: np.ndarray) -> np.ndarray:
    n, f = features.shape
    ...
    if f < FEATURE_DIM:
        pad = np.zeros((n, FEATURE_DIM - f), dtype=np.float32)
        features = np.concatenate([features, pad], axis=1)
    return features.reshape(n, IMAGE_SIZE, IMAGE_SIZE)
```

不管選 `nslkdd` / `cicids2017` / `cicddos2019`，甚至連 `--dataset simulate` 都一樣——都是把 **CSV 流量統計特徵**（duration、src_bytes、封包計數比率…41～78 維數值向量）攤平、補零、reshape 成 32×32。影像裡真正有內容的只有前面 1～3 列，其餘全是 0。

**服務時（模擬頁面 + 正式 PCAP 分析）**，資料來源卻是 `core/packet_visualizer.py::bytes_to_image`：

```python
# core/packet_visualizer.py
def bytes_to_image(self, raw_bytes: bytes, packet_type: str = "unknown") -> np.ndarray:
    data = bytearray(raw_bytes)
    if self.skip_ethernet and len(data) > ETH_HEADER_LEN:
        data = data[ETH_HEADER_LEN:]
    if self.apply_mask:
        data = self._apply_field_mask(data)
    data = data[:self.MAX_BYTES]
    ...
    arr = np.frombuffer(bytes(data), dtype=np.uint8).reshape(self.H, self.W)
    if self.normalize:
        arr = arr.astype(np.float32) / 255.0
    return arr
```

這裡吃的是**封包本身的原始位元組**（IP/TCP/ICMP header + payload，遮罩掉 IP/Port/Checksum 等欄位後）直接 reshape 成 32×32。

而且這不是模擬頁面獨有的問題——`analyzer/tasks.py::_run_cnn_analysis`（正式上傳 PCAP 分析用的那條路）**也是呼叫同一個 `PacketVisualizer.bytes_to_image`**：

```python
# analyzer/tasks.py
from packet_visualizer import PacketVisualizer
...
vis = PacketVisualizer('medium', apply_mask=True, skip_ethernet=True)
images = [vis.bytes_to_image(bytes(pkt)) for pkt in packets]
```

### 這代表什麼

模型在訓練時學到的是「CSV 統計特徵向量的重建規則」，但不管是模擬頁面還是正式分析，餵進去的都是「原始封包位元組」。這兩種 32×32 影像在像素分佈、資訊密度、空間結構上完全是不同的東西——對模型來說，**兩者都是它從沒見過的分佈**。

這正好解釋了你截圖裡「正常流量分數比攻擊還高」的反直覺結果：CNN 重建誤差在這裡本質上接近雜訊，誰高誰低不具有意義。專案 `views.py` 裡其實已經有前人留下的診斷註解承認了一半（模型是用 CSV 特徵訓練、送進來的是 Scapy 封包位元組），但目前的因應方式（用「即時取樣的 baseline 99th percentile」當作動態閾值）**只是讓 CNN 很少誤報，並沒有讓 CNN 的分數變得有鑑別力**——它沒辦法保證攻擊封包的分數會比正常封包高，這件事在你的截圖裡被直接證實了。

### 建議修復方向

不是這次能立即改完的（需要重新訓練），完整方案見第 10 節，但重點是：**專案裡其實已經有現成、且和服務時期表示法完全匹配的工具沒被用上** —— `core/dataset_builder.py::DatasetBuilder.build_from_pcap()`，它就是用同一個 `PacketVisualizer` 把 PCAP 轉成訓練集。目前所有訓練腳本（`run_training.py` / `run_semi_supervised.py` / `run_threshold_tuning.py`）都只呼叫 `DatasetFactory`（CSV 路徑），完全沒有任何地方用到 `DatasetBuilder`（封包位元組路徑）。這是全系統性的落差，不是模擬頁面獨有。

---

<a id="root-b"></a>
## 3. 根因 B：規則式門檻與畫面提示文字對不上、雙閾值同時顯示造成混淆

### 證據 1：畫面同時顯示兩個「閾值」，意義完全不同

上方 KPI 卡片：

```html
<!-- templates/analyzer/simulation.html -->
<div class="kpi-value mono" style="font-size:1.1rem;">{{ cnn_threshold }}</div>
<div class="kpi-label">異常閾值 (Threshold)</div>
```

這裡的 `cnn_threshold` 來自 `settings.CNN_THRESHOLD = float(os.getenv('CNN_THRESHOLD', '0.000069'))`——是**寫死在 Django 設定檔裡、給舊版非監督模型參考用的靜態數字**，跟模擬 API 實際使用的閾值毫無關係。

但下方結果面板顯示的 `0.008428`，是**每次請求都即時重新計算**的「baseline 99th percentile」動態閾值（`analyzer/views.py::simulation_api` 裡的 `effective_threshold`）。

兩個數字同時出現在同一頁、字面上都叫「閾值」，卻是完全不同的東西——這是使用者第一時間會覺得「閾值不合理」的直接原因。

### 證據 2：規則式偵測門檻的提示文字與實際套用值不一致

```html
<div class="text-muted" style="font-size:.66rem;margin-top:4px;">
  規則式偵測門檻參考：Port Scan >20 · ICMP Flood >50 · SYN Flood >100 · UDP Flood >200
</div>
```

這四個數字其實是抄自 `core/config.py` 的**正式環境門檻**（`ALERT_THRESHOLD_SYN=100` 等），但 `_detect_simulation_behavior` 實際套用的門檻是：

```python
# analyzer/views.py（修正前）
demo_threshold = max(1, int(packet_count * 0.5))
detector = AnomalyDetector(
    threshold_syn=demo_threshold, threshold_ports=demo_threshold,
    threshold_icmp=demo_threshold, threshold_udp=demo_threshold,
)
```

22 個封包時 `demo_threshold = 11`，四種規則全部共用同一個數字——跟畫面寫的「SYN>100」完全對不上，你截圖裡看到的告警文字「SYN 封包數: 12（閾值: 11…）」正是證據。

---

<a id="root-c"></a>
## 4. 根因 C：模擬 API 完全沒有模型快取

`core/model_registry.py::load_anomaly_model` 每次呼叫都會重新對磁碟做 `torch.load()`、重新 `Encoder`/`Decoder`/`Classifier` 建構、重新搬到 device 上——**模擬 API 是同步 view（沒有用 Celery），每一次請求都完整付一次這個成本**。

而「第一次要等幾十秒、之後不到 3 秒」這種落差幅度，單靠重複建構模型解釋不了（CPU 推論一顆幾百 K 參數的小模型，重建構通常是百毫秒級）。真正的大頭很可能是：

`core/packet_visualizer.py` 檔案最上面就 `import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt`——**matplotlib 第一次被匯入時，如果本機還沒有字型快取，會觸發字型掃描與快取建置**，這個動作在乾淨環境（例如新部署的容器/伺服器）上花上 10～30 秒是常見現象，且**快取建好之後，同一個行程後續的呼叫幾乎瞬間完成**——這跟你觀察到的「第一次慢、後面快」的行為模式完全吻合。而 `simulation_api` 內部 `from packet_visualizer import PacketVisualizer` 是寫在 view 函式**內部**（延遲匯入），所以這個成本會落在「第一個打進來的使用者請求」身上，而不是伺服器啟動的時候。

---

<a id="root-d"></a>
## 5. 黑箱問題：目前回傳給前端的資料太粗，看不到「為什麼」

目前 `simulation_api` 回傳的每筆結果只有：

```python
entry = {
    'index': i, 'score': ..., 'cnn_anomaly': ..., 'behavior_anomaly': ...,
    'is_anomaly': ..., 'size': ..., 'summary': scapy_packets[i].summary(),
}
```

`behavior_anomaly` 是**整批共用同一個布林值**——只要規則引擎在批次中任何一個封包觸發告警，`behavior_result['is_attack']` 就變成 `True`，然後這個值被套用到**每一筆**結果的 `behavior_anomaly` 欄位上。使用者完全看不出「這 22 個封包裡，究竟是哪一個封包真的觸發了規則、哪些是被整批判定波及的」，也看不到規則引擎當下實際比對到的欄位內容——這正是「黑箱」感受的來源。

---

<a id="why-not-changed"></a>
## 6. 為什麼某些看起來像 bug 的地方，這次刻意沒有動

在準備修正前，我先核對了 `analyzer/tests.py` 目前對行為的既有斷言，發現兩個地方雖然表面上看起來可疑，但**已經被測試明確鎖定為預期行為**，貿然改掉會直接讓測試變紅：

1. **ARP Spoof / DNS Amplification 的強制 `True` 判定**
   ```python
   if attack_type in ('arp_spoof', 'dns_amplification'):
       behavior_attack = True
   ```
   `test_api_fusion_arp_spoof` 明確斷言 5 個封包的 ARP Spoof 場景中，**回傳的每一筆結果都要是 `is_anomaly=True`**。經過推演確認：DNS 放大攻擊規則需要 **>20 筆偽造回應封包**才會真正觸發（`resp > 20`），但模擬頁面產生的攻擊封包數上限雖有 250，一般示範用的封包量很難剛好配置出 20+ 筆回應；若拿掉這個保底邏輯，示範會經常「假裝在展示 DNS 放大攻擊，但規則引擎其實從頭到尾沒觸發」。這次做法是**保留判定結果，但新增 `forced_by_demo_shortcut` 欄位誠實告知前端「這是示範保底，不是規則引擎即時觸發」**（ARP Spoof 經實測其實通常第 2 個封包規則引擎就會真的觸發，這個保底邏輯多數時候是安全網、不會真正發揮作用）。

2. **SYN/Port Scan/ICMP/UDP 四類別共用同一個 `packet_count * 0.5` 門檻**
   `test_behavior_detector_small_count` 明確斷言 `packet_count=2` 時 `result['threshold'] == 1`，與現行公式完全對應。這次**保留公式不變**，只是新增 `thresholds` 欄位把「四個類別目前套用的是同一個數字」這件事誠實攤開給前端顯示，並修正畫面上原本誤導的靜態提示文字。

換句話說：**這次修正的原則是「不改變任何既有測試鎖定的判定結果，只把黑箱攤開、把效能問題解決」**。如果你們團隊確認要動判定邏輯本身（例如讓 SYN/ICMP/UDP 改成逐封包判定、門檻改成與正式環境等比例縮放），那會需要同步更新 `analyzer/tests.py` 裡對應的斷言，我可以另外處理，但建議先確認產品意圖（例如「規則一旦觸發，整批視為受影響」是不是刻意的資安產品設計選擇，而非疏漏）。

---

<a id="summary"></a>
## 7. 修正內容總覽

| # | 檔案 | 修正內容 | 是否影響既有測試 |
|---|---|---|---|
| 1 | `core/model_registry.py` | 新增行程內模型快取（依路徑+mtime+size 作為快取鍵） | 否，純效能優化 |
| 2 | `analyzer/apps.py` | 新增啟動時背景預熱（matplotlib 字型快取 + 模型預載） | 否 |
| 3 | `analyzer/views.py` | `_detect_simulation_behavior` / `simulation_api` 新增透明化欄位（`thresholds`、`triggered_index`、`forced_by_demo_shortcut`、逐封包 `rule_triggered_here`/`rule_detail`、`cnn_reliability_note`） | 否，全部是新增欄位，既有欄位數值不變 |
| 4 | `templates/analyzer/simulation.html` | 修正 KPI 卡片文字、顯示新欄位、動態渲染規則式門檻說明 | 否 |

---

<a id="code"></a>
## 8. 完整修正程式碼

### 8.1 `core/model_registry.py` — 新增模型快取

在檔案開頭 `import` 區塊之後，新增快取機制；`load_anomaly_model` 只需要在最前面加入查快取、最後面加入寫快取兩段：

```python
# core/model_registry.py
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch

KNOWN_MODEL_TYPES = ("cnn_vae", "hybrid_semi", "legacy_cnn_ae")

# ────────────────────────────────────────────────────────────
# [新增] 行程內模型快取
#
# 修正前：每次呼叫 load_anomaly_model() 都會重新對磁碟 torch.load()、
# 重新建構模型物件並搬到 device 上。模擬檢測 API 是同步 view（未走
# Celery），代表「每一次模擬請求」都要重付一次這個成本，也是使用者
# 觀察到「延遲不穩定」的原因之一。
#
# 修正後：以 (檔案路徑, mtime, size, device) 作為快取鍵值。同一顆
# 模型檔案只會在「第一次用到」或「檔案內容已變更（重新訓練部署後
# mtime/size 改變）」時才重新讀取與建構；其餘請求直接複用記憶體中
# 已 eval() 好的模型物件。由於推論全程使用 torch.no_grad()、模型
# 停留在 eval 模式、不會被訓練或原地修改，多個請求安全共用同一份
# 物件是標準的模型服務作法。
# ────────────────────────────────────────────────────────────
_MODEL_CACHE: Dict[tuple, "AnomalyModelBundle"] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _cache_key(checkpoint_path: str, device: torch.device) -> tuple:
    checkpoint_path = str(checkpoint_path)
    try:
        stat = os.stat(checkpoint_path)
        file_sig = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        file_sig = (None, None)
    return (checkpoint_path, *file_sig, str(device))


def clear_model_cache(checkpoint_path: Optional[str] = None) -> None:
    """
    手動清空模型快取。

    通常不需要手動呼叫（檔案 mtime/size 一變就會自動判定快取失效並
    重新載入），但部署腳本若想在「覆蓋模型檔案後立刻強制重新載入」
    （例如檔案覆蓋方式導致 mtime 沒變化的極端情況），可以呼叫本函式。

    Args:
        checkpoint_path: 只清除此路徑對應的快取；None 則清空全部。
    """
    with _MODEL_CACHE_LOCK:
        if checkpoint_path is None:
            _MODEL_CACHE.clear()
            return
        checkpoint_path = str(checkpoint_path)
        for key in [k for k in _MODEL_CACHE if k[0] == checkpoint_path]:
            del _MODEL_CACHE[key]


@dataclass
class AnomalyModelBundle:
    model: torch.nn.Module
    model_type: str
    threshold: float
    device: torch.device
    config: Dict[str, Any] = field(default_factory=dict)
    gradcam_target_layer: Optional[str] = None

    @property
    def is_hybrid(self) -> bool:
        return self.model_type == "hybrid_semi"


def load_anomaly_model(checkpoint_path, device: Optional[torch.device] = None,
                        default_latent_dim: int = 32) -> AnomalyModelBundle:
    """依 checkpoint 內儲存的 model_type 動態建構正確的模型類別並載入權重。

    [效能/一致性修正] 見檔案開頭「行程內模型快取」說明。
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = str(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型檔案: {checkpoint_path}")

    # [新增] 查快取
    key = _cache_key(checkpoint_path, device)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_type = ckpt.get("model_type") if isinstance(ckpt, dict) else None
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    if model_type == "hybrid_semi":
        from hybrid_semi_supervised import HybridSemiSupervisedDetector
        model = HybridSemiSupervisedDetector.from_checkpoint(ckpt).to(device)
        gradcam_layer = "vae.encoder_conv"
        threshold = float(ckpt.get("threshold", getattr(model, "threshold", 0.0)))

    elif model_type == "cnn_vae":
        from variational_autoencoder import CNNVariationalAutoencoder
        model = CNNVariationalAutoencoder(
            latent_dim=cfg.get("latent_dim", default_latent_dim),
            kl_weight=cfg.get("kl_weight", 1e-3),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        gradcam_layer = None
        threshold = float(ckpt.get("threshold", 0.0))

    else:
        from cnn_autoencoder import CNNAutoencoder, load_compatible_state_dict
        model = CNNAutoencoder(latent_dim=cfg.get("latent_dim", default_latent_dim)).to(device)
        load_compatible_state_dict(model, ckpt)
        model_type = "legacy_cnn_ae"
        gradcam_layer = None
        threshold = float(ckpt.get("threshold", 0.0)) if isinstance(ckpt, dict) else 0.0

    model.eval()
    bundle = AnomalyModelBundle(
        model=model, model_type=model_type, threshold=threshold,
        device=device, config=cfg, gradcam_target_layer=gradcam_layer,
    )

    # [新增] 寫入快取
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[key] = bundle

    return bundle


def compute_anomaly_scores(bundle: AnomalyModelBundle, X: np.ndarray,
                            batch_size: int = 256) -> Dict[str, np.ndarray]:
    """（本函式內容與修正前完全相同，未變動，故省略重複貼上；
    請保留你專案內現有的實作。）"""
    raise NotImplementedError("請保留原檔案中 compute_anomaly_scores 的既有實作，僅套用上方 load_anomaly_model 的變更")
```

> **套用方式**：`compute_anomaly_scores` 完全不用改，只要把 `load_anomaly_model` 換成上面的版本、並在檔案開頭加入快取相關程式碼即可。

---

### 8.2 `analyzer/apps.py` — 新增啟動時背景預熱

```python
# analyzer/apps.py
from django.apps import AppConfig


class AnalyzerConfig(AppConfig):
    name = 'analyzer'
    verbose_name = '封包分析'

    def ready(self):
        """
        [新增] 伺服器啟動時，在背景執行緒預熱一次性的重量級成本，
        避免這些成本轉嫁到第一位使用者的請求上：

          1. matplotlib 首次匯入時的字型快取建置
             （core/packet_visualizer.py 等多個模組在 import 階段就會
             載入 matplotlib，首次執行常見耗時 10~30 秒，之後同一行程
             內幾乎瞬間完成 —— 這正是「第一次模擬跑很久、之後很快」
             的主要成因之一）。
          2. 依 settings.ANOMALY_MODELS 把目前已存在的模型檔案全部
             預先載入進 core/model_registry.py 的行程內快取，讓第一位
             使用者送出模擬請求時就能直接命中快取。

        使用背景執行緒 + 全面 try/except 是為了避免任何預熱步驟失敗
        （例如模型檔案還沒訓練好、GPU 驅動異常）連帶讓 Django 應用程式
        啟動失敗；預熱只是「錦上添花」，不是必要條件。
        """
        import threading

        def _warmup():
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt  # noqa: F401  # 觸發字型快取建置
            except Exception:
                pass

            try:
                import sys
                import os
                from django.conf import settings

                core_dir = str(settings.BASE_DIR / 'core')
                if core_dir not in sys.path:
                    sys.path.insert(0, core_dir)

                import torch
                from model_registry import load_anomaly_model

                device = torch.device('cpu')  # 與 simulation_api 使用的 device 一致
                for _key, cfg in settings.ANOMALY_MODELS.items():
                    model_path = str(cfg.get('path', ''))
                    if model_path and os.path.exists(model_path):
                        try:
                            load_anomaly_model(
                                model_path, device=device,
                                default_latent_dim=settings.CNN_LATENT_DIM,
                            )
                        except Exception:
                            # 單一模型載入失敗不應影響其他模型的預熱
                            continue
            except Exception:
                pass

        threading.Thread(target=_warmup, daemon=True, name='vnaap-warmup').start()
```

---

### 8.3 `analyzer/views.py` — 透明化修正

以下是 `_detect_simulation_behavior` 與 `simulation_api` 的**完整修正版本**（可直接取代原本兩個函式；其餘 `views.py` 內容不變）：

```python
# analyzer/views.py（局部：取代 _detect_simulation_behavior 與 simulation_api）

def _detect_simulation_behavior(scapy_packets, attack_type, packet_count):
    """
    規則式（流量型樣）偵測。

    [透明化修正] 新增回傳欄位，讓前端能呈現「規則引擎實際看到了什麼、
    在哪個封包上真正觸發」，取代先前只給一個整批適用的黑箱結論：

      - packet_alerts            每個封包各自觸發了哪些告警（原本就有
                                  計算，但先前沒有回傳給前端使用）
      - triggered_index          規則第一次於本批次觸發告警的封包索引
                                  （None 代表規則引擎全程未觸發，此時
                                  is_attack=True 只可能是下面
                                  forced_by_demo_shortcut 造成）
      - forced_by_demo_shortcut  True 代表這個攻擊類型在目前示範封包量
                                  下，即使規則引擎沒有真的觸發，也會固定
                                  顯示為「偵測到攻擊」（目前只有
                                  dns_amplification 真正需要這個保底，
                                  因為該規則需要 >20 筆偽造回應封包才會
                                  觸發，多數示範封包量下規則引擎本身就
                                  無法達標；arp_spoof 理論上第 2 個封包
                                  規則引擎就會真的觸發，這裡的保底邏輯
                                  多數情況下不會真正發揮作用，純粹是
                                  異常封包產生器改版時的保險）
      - thresholds                本次套用在 SYN / Port Scan / ICMP /
                                  UDP 四個類別的實際判定門檻

    已知限制（本次修正刻意保留，避免破壞既有測試 analyzer/tests.py
    對這兩點行為的斷言，理由詳見修正報告「為什麼某些地方沒有動」）：
      1. SYN / Port Scan / ICMP / UDP 四個類別目前仍共用同一個
         max(1, int(packet_count * 0.5)) 門檻，而非各自獨立、貼近
         core/config.py 正式門檻等比例縮放的數值。
      2. 一旦規則引擎在批次中任一封包觸發告警，is_attack／
         behavior_anomaly 仍會套用到「整批」封包，而非只有真正觸發
         規則那一刻起的封包（ARP Spoof 情境下，此為既有測試明確要求
         的行為，非本次修正範圍）。
    """
    from anomaly_detector import AnomalyDetector
    from parser import PacketParser

    demo_threshold = max(1, int(packet_count * 0.5))
    detector = AnomalyDetector(
        threshold_syn=demo_threshold,
        threshold_ports=demo_threshold,
        threshold_icmp=demo_threshold,
        threshold_udp=demo_threshold,
    )
    parser = PacketParser()
    packet_alerts = [[] for _ in scapy_packets]
    all_alerts = []
    triggered_index = None

    for i, pkt in enumerate(scapy_packets):
        try:
            record = parser.parse(pkt)
            alerts = detector.inspect(pkt, record)
            if alerts:
                packet_alerts[i].extend(alerts)
                all_alerts.extend(alerts)
                if triggered_index is None:
                    triggered_index = i
        except Exception:
            continue

    behavior_attack = len(all_alerts) > 0
    forced_by_demo_shortcut = False

    if attack_type in ('arp_spoof', 'dns_amplification'):
        if not behavior_attack:
            forced_by_demo_shortcut = True
        behavior_attack = True
    if attack_type == 'normal_traffic':
        behavior_attack = False

    behavior_type = None
    if all_alerts:
        behavior_type = all_alerts[0].get('attack_type', attack_type)

    return {
        'is_attack': behavior_attack,
        'attack_type': behavior_type,
        'alerts': all_alerts,
        'packet_alerts': packet_alerts,
        'triggered_index': triggered_index,
        'forced_by_demo_shortcut': forced_by_demo_shortcut,
        'thresholds': {
            'syn': demo_threshold, 'ports': demo_threshold,
            'icmp': demo_threshold, 'udp': demo_threshold,
        },
        'threshold': demo_threshold,  # 向下相容既有欄位
    }


# [新增] CNN 分數可信度說明。集中成常數方便日後模型換成
# packet-level 訓練後統一調整/移除此提示。
_CNN_RELIABILITY_NOTE = (
    '目前部署的 CNN／VAE 模型是以 CSV 流量特徵（如 CICIDS2017／NSL-KDD 的'
    '統計欄位）訓練而成；本頁與正式 PCAP 分析管線送入模型的則是「原始封包'
    '位元組」（經 PacketVisualizer 轉換），兩者影像的統計分布並不相同，屬於'
    '訓練/服務資料表示法不一致。因此 CNN 分數在這裡僅供參考，不宜單獨作為'
    '攻擊／正常的判斷依據，請優先參考下方「規則式流量型樣偵測」的結果。'
)


@login_required
def simulation_api(request):
    """模擬檢測 AJAX 端點 — 回傳 CNN 逐封包分數 + 規則式（流量型樣）偵測結果。"""
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'error': '僅接受 POST'}, status=405)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError) as e:
        return JsonResponse({'ok': False, 'error': f'JSON 解析失敗：{e}'}, status=400)

    attack_type = data.get('attack_type', 'syn_flood')
    model_key = data.get('model_key')
    if not model_key:
        model_key = 'unsupervised_vae'

    if model_key not in settings.ANOMALY_MODELS:
        return JsonResponse({'ok': False, 'error': '不支援的模型選擇'}, status=400)
    if attack_type not in SIMULATION_ATTACK_TYPES:
        return JsonResponse({'ok': False, 'error': '不支援的攻擊場景'}, status=400)

    try:
        packet_count = min(max(int(data.get('packet_count', 10)), 1), 250)
    except (TypeError, ValueError):
        return JsonResponse({'ok': False, 'error': '封包數量必須是整數'}, status=400)

    try:
        import sys
        core_path = os.path.join(settings.BASE_DIR, 'core')
        if core_path not in sys.path:
            sys.path.insert(0, core_path)

        from generate_attack_pcap import generate_attack_packets
        from scapy.all import raw as scapy_raw

        scapy_packets = generate_attack_packets(attack_type, packet_count)

        raw_packets = []
        for pkt in scapy_packets:
            try:
                raw_packets.append(scapy_raw(pkt))
            except Exception:
                raw_packets.append(bytes(pkt))

        results = []
        baseline_info = None
        selected_model = settings.ANOMALY_MODELS[model_key]
        model_path = selected_model['path']

        if model_path and os.path.exists(str(model_path)):
            from packet_visualizer import PacketVisualizer
            import torch
            import numpy as np
            from model_registry import load_anomaly_model, compute_anomaly_scores

            bundle = load_anomaly_model(
                str(model_path), device=torch.device('cpu'),
                default_latent_dim=settings.CNN_LATENT_DIM,
            )
            model = bundle.model
            visualizer = PacketVisualizer("medium", apply_mask=True)

            BASELINE_SAMPLE_COUNT = 60
            baseline_pkts = generate_attack_packets('normal_traffic', BASELINE_SAMPLE_COUNT)
            baseline_raw = []
            for pkt in baseline_pkts:
                try:
                    baseline_raw.append(scapy_raw(pkt))
                except Exception:
                    baseline_raw.append(bytes(pkt))

            baseline_arrs = np.array([
                visualizer.bytes_to_image(b) for b in baseline_raw
            ], dtype=np.float32)
            baseline_tensor = torch.from_numpy(baseline_arrs[:, np.newaxis])

            with torch.no_grad():
                baseline_errors = model.reconstruction_error(baseline_tensor).numpy()

            baseline_mean = float(baseline_errors.mean())
            baseline_std = float(baseline_errors.std())
            dynamic_threshold = float(np.percentile(baseline_errors, 95))

            baseline_info = {
                'sample_count': len(baseline_errors),
                'mean': round(baseline_mean, 8),
                'std': round(baseline_std, 8),
                'dynamic_threshold': round(dynamic_threshold, 8),
                'static_threshold': round(float(bundle.threshold), 8),
                'scores': [round(float(s), 8) for s in baseline_errors.tolist()],
            }

            target_arrs = np.array([
                visualizer.bytes_to_image(b) for b in raw_packets
            ], dtype=np.float32)
            scored = compute_anomaly_scores(bundle, target_arrs)

            behavior_result = _detect_simulation_behavior(
                scapy_packets=scapy_packets,
                attack_type=attack_type,
                packet_count=packet_count,
            )
            behavior_anomaly = behavior_result['is_attack']

            if bundle.is_hybrid:
                cnn_anomaly = scored['is_anomaly'].copy()
                effective_threshold = float(bundle.threshold)
            else:
                effective_threshold = float(np.percentile(baseline_errors, 99))
                cnn_anomaly = (scored['score'] > effective_threshold)

            final_anomaly = []
            for i in range(len(scapy_packets)):
                is_cnn_anomaly = bool(cnn_anomaly[i])
                is_behavior_anomaly = behavior_anomaly
                final_anomaly.append(is_cnn_anomaly or is_behavior_anomaly)

            scored['is_anomaly'] = np.array(final_anomaly, dtype=bool)

            for i, score in enumerate(scored['score']):
                score_val = float(score)
                pkt_specific_alerts = behavior_result['packet_alerts'][i]
                entry = {
                    'index': i,
                    'score': round(score_val, 8),
                    'cnn_anomaly': bool(cnn_anomaly[i]),
                    'behavior_anomaly': bool(behavior_anomaly),
                    'is_anomaly': bool(scored['is_anomaly'][i]),
                    'size': len(raw_packets[i]),
                    'summary': scapy_packets[i].summary(),
                    # [新增] 透明化：這個封包自己是否直接觸發了規則
                    # （而非因為同批次其他封包觸發、被整批套用判定）
                    'rule_triggered_here': bool(pkt_specific_alerts),
                    'rule_detail': [
                        {
                            'attack_type': a.get('attack_type'),
                            'severity': a.get('severity'),
                            'detail': a.get('detail'),
                        }
                        for a in pkt_specific_alerts
                    ],
                }

                if pkt_specific_alerts:
                    entry['detection_reason'] = pkt_specific_alerts[0].get('attack_type', attack_type)
                elif behavior_anomaly:
                    entry['detection_reason'] = behavior_result['attack_type'] or attack_type
                elif cnn_anomaly[i]:
                    entry['detection_reason'] = 'CNN/VAE reconstruction error'
                else:
                    entry['detection_reason'] = 'normal'

                if bundle.is_hybrid:
                    entry['status'] = scored['status'][i]
                    entry['known_confidence'] = round(float(scored['known_confidence'][i]), 4)
                results.append(entry)
        else:
            import random
            threshold = settings.CNN_THRESHOLD
            is_normal = (attack_type == 'normal_traffic')

            behavior_result = _detect_simulation_behavior(
                scapy_packets=scapy_packets,
                attack_type=attack_type,
                packet_count=packet_count,
            )
            behavior_anomaly = behavior_result['is_attack']

            for i, pkt_bytes in enumerate(raw_packets):
                sim_score = (random.uniform(threshold * 0.05, threshold * 0.85) if is_normal
                             else random.uniform(threshold * 2.0, threshold * 15.0))

                cnn_anom = sim_score > threshold
                final_anom = cnn_anom or behavior_anomaly
                pkt_specific_alerts = behavior_result['packet_alerts'][i]

                results.append({
                    'index': i,
                    'score': round(sim_score, 8),
                    'cnn_anomaly': cnn_anom,
                    'behavior_anomaly': behavior_anomaly,
                    'is_anomaly': final_anom,
                    'size': len(pkt_bytes),
                    'summary': scapy_packets[i].summary(),
                    'rule_triggered_here': bool(pkt_specific_alerts),
                    'rule_detail': [
                        {'attack_type': a.get('attack_type'), 'severity': a.get('severity'), 'detail': a.get('detail')}
                        for a in pkt_specific_alerts
                    ],
                })

        anomaly_count = sum(1 for r in results if r['is_anomaly'])
        total = len(results)
        pcap_filename = _save_simulation_pcap(request.user, attack_type, scapy_packets)

        return JsonResponse({
            'ok': True,
            'attack_type': attack_type,
            'packet_count': total,
            'effective_threshold': round(effective_threshold, 8) if (model_path and os.path.exists(str(model_path))) else settings.CNN_THRESHOLD,
            'threshold': float(bundle.threshold) if (model_path and os.path.exists(str(model_path))) else settings.CNN_THRESHOLD,
            'dynamic_threshold': round(dynamic_threshold, 8) if (model_path and os.path.exists(str(model_path)) and baseline_info) else None,
            'static_threshold': settings.CNN_THRESHOLD,
            # [新增] 明確標示 static_threshold 只是「模型訓練時期的參考值」，
            # 不是本頁實際判定所使用的門檻，避免與 effective_threshold 混淆。
            'static_threshold_note': '此為模型訓練時的參考閾值，本頁實際判定請以 effective_threshold 為準',
            'cnn_reliability_note': _CNN_RELIABILITY_NOTE,
            'behavior_detection': {
                'is_attack': behavior_result['is_attack'],
                'attack_type': behavior_result['attack_type'],
                'threshold': behavior_result['threshold'],
                'thresholds': behavior_result['thresholds'],
                'triggered_index': behavior_result['triggered_index'],
                'forced_by_demo_shortcut': behavior_result['forced_by_demo_shortcut'],
                'alert_count': len(behavior_result['alerts']),
                'alerts': behavior_result['alerts'],
            },
            'baseline': baseline_info,
            'results': results,
            'avg_score': round(sum(r['score'] for r in results) / max(total, 1), 8),
            'anomaly_count': anomaly_count,
            'rule_based_anomaly_detected': behavior_result['is_attack'],
            'pcap_download_url': request.build_absolute_uri(reverse('analyzer:simulation_pcap_download', args=[pcap_filename])),
            'pcap_filename': pcap_filename,
            'pcap_backup_saved': True,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'ok': False, 'error': f'{type(e).__name__}: {str(e)}'}, status=500)
```

> **注意**：這版本除了新增欄位之外，`is_anomaly` / `behavior_anomaly` / `cnn_anomaly` 等既有欄位的**數值計算方式完全沒有變**，`analyzer/tests.py` 目前所有斷言都應該維持通過。

---

### 8.4 `templates/analyzer/simulation.html` — 局部修正

**(a) 修正 KPI 卡片文字**（原本容易被誤認為是本頁實際判定門檻）：

```html
<!-- 修正前 -->
<div class="kpi-card kpi-accent">
  <div class="kpi-value mono" style="font-size:1.1rem;">{{ cnn_threshold }}</div>
  <div class="kpi-label">異常閾值 (Threshold)</div>
</div>

<!-- 修正後 -->
<div class="kpi-card kpi-accent">
  <div class="kpi-value mono" style="font-size:1.1rem;">{{ cnn_threshold }}</div>
  <div class="kpi-label">模型訓練閾值（參考）</div>
  <div class="text-muted" style="font-size:.62rem;margin-top:2px;">
    非本頁判定門檻，實際門檻見右側結果面板
  </div>
</div>
```

**(b) 在結果面板固定顯示 CNN 可信度提示**（不是條件式顯示，因為只要模型是 CSV 訓練出來的，這個限制就一直存在）：

```html
<!-- 在 baselineInfo 區塊上方新增 -->
<div id="cnnReliabilityNote" style="display:none;margin-bottom:12px;padding:10px 12px;
     background:rgba(234,179,8,.08);border:1px solid rgba(234,179,8,.25);border-radius:8px;">
  <div style="font-size:.72rem;color:var(--yellow);line-height:1.6;">
    ⚠️ <span id="cnnReliabilityText"></span>
  </div>
</div>
```

**(c) JS：渲染新欄位**（加在 `runSimulation()` 內、原本 `baselineInfo` 渲染邏輯之後）：

```javascript
// CNN 可信度提示（固定顯示，不論攻擊類型）
if (res.cnn_reliability_note) {
  document.getElementById('cnnReliabilityText').textContent = res.cnn_reliability_note;
  document.getElementById('cnnReliabilityNote').style.display = 'block';
}

// 規則式門檻：改成動態渲染實際套用的四個類別門檻，不再用寫死文字
const bd = res.behavior_detection || {};
if (bd.thresholds) {
  const t = bd.thresholds;
  let shortcutNote = '';
  if (bd.forced_by_demo_shortcut) {
    shortcutNote = '<br><span style="color:var(--yellow)">⚠ 此結果為示範保底判定，規則引擎於本批次未實際達標觸發（詳見報告第 6 節）</span>';
  }
  document.getElementById('ruleBasedContent').insertAdjacentHTML('beforeend', `
    <div style="margin-top:6px;font-size:.7rem;color:var(--text-muted);">
      本次套用門檻（示範用，非正式環境門檻）：
      SYN=${t.syn} · Port Scan=${t.ports} · ICMP=${t.icmp} · UDP=${t.udp}
      ${shortcutNote}
    </div>`);
}

// 封包結果表格：新增「觸發依據」欄位
res.results.forEach((r, i) => {
  const trigger = r.rule_triggered_here
    ? `★ 本封包直接觸發：${(r.rule_detail[0] || {}).attack_type || ''}`
    : (r.behavior_anomaly ? '同批次判定（規則引擎於本批次其他封包觸發）' : '');
  // 併入既有表格列渲染邏輯的 tr.innerHTML 中，新增一個 <td>${trigger}</td>
});
```

---

<a id="verify"></a>
## 9. 如何驗證這次修正

1. **既有測試維持綠燈**：
   ```bash
   python manage.py test analyzer.tests
   ```
   `SimulationFusionTests` 全部案例（含 `test_api_fusion_arp_spoof`、`test_behavior_detector_small_count` 等）應維持通過，因為判定邏輯數值完全沒變。

2. **效能改善驗證**：連續打兩次 `/analyzer/simulation/api/`（同一模型），第一次與第二次的回應時間應該接近，不再有「第一次幾十秒、第二次幾秒」的明顯落差（伺服器重啟後第一次仍會有一次性的預熱成本，但已從「使用者請求」轉移到「伺服器啟動」）。

3. **黑箱透明化驗證**：檢查回傳 JSON 是否包含 `cnn_reliability_note`、`behavior_detection.thresholds`、每筆 `results[i].rule_triggered_here` / `rule_detail`。

---

<a id="next"></a>
## 10. 後續建議（超出本次修正範圍，但強烈建議規劃）

1. **用封包位元組重新訓練模型**，讓訓練與服務資料表示法一致：
   - 用 `core/generate_attack_pcap.py`（或真實擷取的 PCAP）產生正常/攻擊封包
   - 用 `core/dataset_builder.py::DatasetBuilder.build_from_pcap()`（**現成、目前無人使用**）轉成與 `PacketVisualizer.bytes_to_image()` 完全匹配的 32×32 訓練集
   - 用既有的 `Trainer` / `VAETrainer`（已內建 patience-based Early Stopping）重新訓練，訓練時把 `epochs` 上限拉高（目前固定 50），讓 Early Stopping 自然決定收斂點，而不是被人為砍在 50
   - 建議以新的 `model_key`（例如 `unsupervised_vae_packetlevel`）新增進 `settings.ANOMALY_MODELS`，與現有 CSV 訓練模型並列，方便直接比較兩者在模擬頁面上的分數分佈差異

2. **在模型 checkpoint 中記錄訓練 metadata**（實際訓練 epoch 數、資料集來源、訓練時間），並在管理員面板 / 模擬頁面顯示，讓「這顆模型到底是怎麼來的」不再是黑箱。

3. **正式環境門檻與示範門檻分離**：若確定要讓示範門檻更貼近正式環境（`core/config.py` 的 `ALERT_THRESHOLD_*`），需要同步更新 `analyzer/tests.py` 中依賴目前 `packet_count * 0.5` 公式的斷言，建議另開一張票規劃，不建議跟這次的透明化/效能修正混在一起送出。
