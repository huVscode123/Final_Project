# VNAAP / NetGuard 程式碼審查報告

**審查範圍**：`core_source_export.txt`（60 個核心模組）、`django_backend_system.txt`、`django_frontend_templates.txt`、`django_logic.txt`
**審查日期**：2026-08-01
**說明**：`django_logic.txt` 與 `django_backend_system.txt` 內容經逐檔比對後**完全相同**（後者僅多了 `manage.py` 與 `requirements.txt`），因此以下只需以 `backend_system` 為準，不存在「新舊版本落差」的問題。

---

## 目錄

1. [🔴 P0 — 資料洩漏（Data Leakage）疑似回歸](#p0-1)
2. [🔴 P0 — Django REST API 系統性 IDOR 越權漏洞](#p0-2)
3. [🟠 P1 — Grad-CAM 執行腳本 sys.path 設定錯誤](#p1-1)
4. [🟠 P1 — 閾值調校與 Django 推論斷連](#p1-2)
5. [🟠 P1 — Celery CNN 推論未分批，OOM 風險](#p1-3)
6. [🟠 P1 — 封包上傳無檔案格式驗證](#p1-4)
7. [🟡 P2 — 消融實驗 / 半監督閾值調校 效能未優化](#p2-1)
8. [🟡 P2 — cnn_autoencoder.py 雙重定義（importlib 動態載入）](#p2-2)
9. [🟡 P2 — 封包解析欄位覆寫問題](#p2-3)
10. [🟡 P2 — Django settings.py 正式環境設定](#p2-4)
11. [🟡 P2 — 異常封包模擬器：`os.chdir()` 於 import 時期執行，汙染測試環境](#p2-5)
12. [🟢 P3 — 其他次要問題與程式碼品質建議](#p3-1)
13. [✅ 已確認做得好的部分](#good)
14. [修正優先順序總表](#priority-table)

---

<a id="p0-1"></a>
## 1. 🔴 P0 — 資料洩漏（Data Leakage）疑似回歸

### 問題

您先前已修復過「Phase 2 微調用的攻擊樣本與最終評估樣本重複」的資料洩漏問題，並以 `run_semi_supervised.py` 作為正確切分邏輯的參考版本。但**這次匯出的程式碼中，除了 `run_semi_supervised.py` 本身，其餘所有訓練腳本都退回到未切分的狀態**：

| 檔案 | 位置 | 狀態 |
|---|---|---|
| `run_semi_supervised.py` | 第190-207行 | ✅ 正確：`X_normal` 80/20、`X_attack` 50/50 切分 |
| `Semi-supervised model and trainer/semi_supervised_nslkdd.py` | 第759-762行（`__main__`） | ❌ 洩漏 |
| `Semi-supervised model and trainer/semi_supervised_cicids2017.py` | 第664-666行（`__main__`） | ❌ 洩漏 |
| `Semi-supervised model and trainer/semi_supervised_cicddos2019.py` | 第570-573行（`__main__`） | ❌ 洩漏 |
| `Semi-supervised model and trainer/train_all_models.py` | 全部 4 個訓練函式 | ❌ 洩漏 |
| `comparison_benchmark.py` | 第392-393行 | ❌ 洩漏（且變數命名會誤導審查者） |

`ablation_study.py` **不受影響**（純非監督式訓練、從不使用攻擊樣本訓練，已有明確的 80/20 holdout 註解），可以放心。

### 證據：`comparison_benchmark.py` 第 392-393 行（最隱蔽的一處）

```python
self.X_attack_train = X_attack.astype(np.float32)   # Phase 2 微調抽樣用
self.X_attack_test  = X_attack.astype(np.float32)   # 評估時全部使用
```

變數刻意取名為 `_train` / `_test`，讓人誤以為已經切分，但兩者其實是**同一份資料的兩次複製**——沒有任何 index 切分、shuffle 或 `train_test_split`。

### 證據：`train_all_models.py`（原本應是「統一入口」）

```python
# train_nslkdd() / train_cicids2017() / train_ddos2019() / train_unsupervised() 皆同樣模式
trainer.train_full(X_normal, X_attack, attack_cats=attack_cats)   # 微調用了 X_attack
result = evaluate(trainer.model, X_normal, X_attack,               # 評估又用同一份 X_attack
                   trainer.threshold, device=trainer.device)
```

### 影響

除 `run_semi_supervised.py` 外，其餘管線產出的 F1 / Precision / Recall / AUC 等指標都可能**虛高、無效**，因為模型在微調階段已經「看過」評估時要判斷的攻擊樣本。

### ✅ 修正方式（統一套用 `run_semi_supervised.py` 的切分邏輯）

以 `semi_supervised_nslkdd.py` 為例，修改 `__main__` 區塊：

```python
if __name__ == "__main__":
    ...
    args = parser.parse_args()

    loader = NSLKDDLoader(
        train_path          = args.train,
        test_path            = args.test,
        image_size            = args.image_size,
        use_test_as_attack  = args.use_test_attack,
    )
    X_normal, X_attack, attack_cats = loader.load()

    # ── [修正] Holdout 切分，避免 Phase 2 微調與最終評估使用相同攻擊樣本 ──
    rng = np.random.default_rng(42)

    idx_n = rng.permutation(len(X_normal))
    split_n = int(len(X_normal) * 0.8)
    X_train_normal = X_normal[idx_n[:split_n]]
    X_test_normal  = X_normal[idx_n[split_n:]]

    idx_a = rng.permutation(len(X_attack))
    split_a = int(len(X_attack) * 0.5)
    X_finetune_attack = X_attack[idx_a[:split_a]]   # 只給 Phase 2 微調使用
    X_test_attack     = X_attack[idx_a[split_a:]]   # 只給最終評估使用

    if attack_cats is not None:
        attack_cats_finetune = [attack_cats[i] for i in idx_a[:split_a]]
    else:
        attack_cats_finetune = None

    config = { ... }  # 不變
    trainer = SemiSupervisedTrainer_NSLKDD(config, output_dir=args.output)

    # 訓練只用切出來的訓練集
    trainer.train_full(X_train_normal, X_finetune_attack,
                        attack_cats=attack_cats_finetune)

    # 評估只用切出來、模型從未見過的測試集
    result = evaluate(trainer.model, X_test_normal, X_test_attack,
                      trainer.threshold, device=trainer.device)
    ...
```

`comparison_benchmark.py` 的修正重點只需把第 392-393 行改成：

```python
# ── [修正] 實際切分，而非複製同一份資料 ──
idx_a = np.random.default_rng(42).permutation(len(X_attack))
split_a = int(len(X_attack) * 0.5)
self.X_attack_train = X_attack[idx_a[:split_a]].astype(np.float32)   # Phase 2 微調抽樣用
self.X_attack_test  = X_attack[idx_a[split_a:]].astype(np.float32)   # 評估時使用（未參與訓練）
```

`train_all_models.py` 的四個訓練函式，全部比照 `run_semi_supervised.py` 的切分方式重寫。

**建議**：這次落差很可能是快照/剪貼過程中不小心用了較舊版本的檔案。修正後強烈建議在 `tests/test_semi_supervised.py` 中補一個「fine-tune 樣本與 eval 樣本無交集」的斷言（例如比對 array 的 hash 或 index 集合），避免未來再度回歸。

```python
def test_no_attack_leakage(self):
    """確保微調用的攻擊樣本與評估用的攻擊樣本無交集"""
    idx_finetune = set(map(tuple, X_finetune_attack.reshape(len(X_finetune_attack), -1)))
    idx_eval     = set(map(tuple, X_test_attack.reshape(len(X_test_attack), -1)))
    assert idx_finetune.isdisjoint(idx_eval), "偵測到資料洩漏：微調集與評估集有重疊樣本"
```

---

<a id="p0-2"></a>
## 2. 🔴 P0 — Django REST API 系統性 IDOR 越權漏洞

### 問題

`analyzer/views.py`（HTML 頁面版）**每一個**存取 `AnalysisSession` 的 view 都有呼叫 `_check_session_access(request, session)` 做擁有者/專案成員檢查。但 `api/views.py`（REST API 版）中，除了 `SessionDetailAPI`（有自己的 `_get_session` 檢查）之外，**幾乎所有「Session 子資源」端點都完全沒有做這個檢查**，只檢查了「角色權限」（如 `can_run_ai_analysis`），卻沒檢查「這個 session 是不是你的」。

| API 端點 | 檢查了角色權限？ | 檢查了 Session 擁有權？ | 風險 |
|---|:---:|:---:|---|
| `SessionStatusAPI.get` | — | ❌ | 任何登入者可輪詢任意 Session 的任務狀態、封包數、告警數、錯誤訊息 |
| `CNNResultAPI.get` | — | ❌ | 任何登入者可讀取任意 Session 的 CNN 異常偵測結果 |
| `CNNRunAPI.post` | ✅ | ❌ | 任何有該角色的使用者可對**別人的** Session 觸發重新分析 |
| `GradCAMRunAPI.post` | ✅ | ❌ | 同上，可對別人的 Session 觸發 Grad-CAM 產生（消耗運算資源） |
| `GradCAMListAPI.get` | — | ❌ | 任何登入者可讀取任意 Session 的封包熱力圖影像 |
| `AlertListAPI.get` | — | ❌ | 任何登入者可讀取任意 Session 的攻擊告警（含來源/目的 IP） |
| `ReportExportAPI.post` | ✅ | ❌ | 任何有該角色的使用者可為**別人的** Session 產生匯出報告 |
| `ReportListAPI.get` | — | ❌ | 任何登入者可列出任意 Session 已產生的報告 |

只要是登入的使用者，把網址列的 `pk` 改成別人的 Session ID，就能讀取或操作別人專案裡的封包分析資料——完全繞過了 `analyzer/views.py` 那一層精心設計的權限模型。

### 證據

```python
# api/views.py 第 309-325 行
class SessionStatusAPI(APIView):
    def get(self, request, pk):
        try:
            s = AnalysisSession.objects.get(pk=pk)   # ← 沒有檢查 s.created_by 或 project 成員
        except AnalysisSession.DoesNotExist:
            return Response({'error': 'Session 不存在'}, status=404)
        return Response({...})   # 直接把別人的資料回傳
```

### ✅ 修正方式

把 `SessionDetailAPI._get_session` 抽成共用的 mixin/工具函式，讓所有 Session 子資源端點都重複使用：

```python
# api/permissions.py（新建檔案）
from rest_framework.response import Response
from analyzer.models import AnalysisSession


def get_session_or_403(request, pk):
    """
    共用的 Session 存取檢查（比照 analyzer/views.py 的 _check_session_access）。
    回傳 (session, None) 或 (None, error_response)。
    """
    try:
        s = AnalysisSession.objects.get(pk=pk)
    except AnalysisSession.DoesNotExist:
        return None, Response({'error': 'Session 不存在'}, status=404)

    user = request.user
    has_access = (
        user.profile.is_admin
        or s.created_by == user
        or (s.project and (
            s.project.owner == user
            or s.project.members.filter(pk=user.pk).exists()
        ))
    )
    if not has_access:
        return None, Response({'error': '權限不足'}, status=403)
    return s, None
```

然後每個端點只需要加兩行：

```python
class SessionStatusAPI(APIView):
    def get(self, request, pk):
        from .permissions import get_session_or_403
        s, err = get_session_or_403(request, pk)
        if err:
            return err
        return Response({...})


class CNNResultAPI(APIView):
    def get(self, request, pk):
        from .permissions import get_session_or_403
        session, err = get_session_or_403(request, pk)
        if err:
            return err
        try:
            result = CNNResult.objects.get(session_id=pk)
        except CNNResult.DoesNotExist:
            return Response({'error': 'CNN 結果不存在'}, status=404)
        return Response(CNNResultSerializer(result, context={'request': request}).data)


class CNNRunAPI(APIView):
    def post(self, request, pk):
        from .permissions import get_session_or_403
        session, err = get_session_or_403(request, pk)
        if err:
            return err
        if not request.user.profile.can_run_ai_analysis:
            return Response({'error': '權限不足'}, status=403)
        ...

# GradCAMRunAPI / GradCAMListAPI / AlertListAPI / ReportExportAPI / ReportListAPI
# 皆比照上述模式，在最前面加入 get_session_or_403 檢查
```

**建議優先修正順序**：`AlertListAPI`、`CNNResultAPI`、`GradCAMListAPI`（資訊洩漏，被動即可觸發）> `ReportExportAPI`、`CNNRunAPI`、`GradCAMRunAPI`（需額外角色權限，但仍可對他人資源動作）> `SessionStatusAPI`（洩漏程度最低）。

---

<a id="p1-1"></a>
## 3. 🟠 P1 — `cnn_gradcam/run_gradcam.py` sys.path 設定錯誤

### 問題

檔案 docstring 假設自己放在 `core/run_gradcam.py`（與 `run_training.py` 同層），但實際上依照匯出目錄結構，它位於 `core/cnn_gradcam/run_gradcam.py`（比 `core/` 深一層）。

```python
# core/cnn_gradcam/run_gradcam.py 第 45-47 行
# 確保 core 目錄在 sys.path 中（與其他 run_*.py 一致）
_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
```

`_BASE` 實際上會解析成 `core/cnn_gradcam/`，而非 `core/`。因此後續：

```python
from ablation_study import CNNAutoencoderFlex   # 位於 core/，不在 core/cnn_gradcam/
...
from cnn_autoencoder import CNNAutoencoder       # 同樣位於 core/
```

在目前的 `_BASE` 設定下都會 `ModuleNotFoundError`。第一個有 `try/except ImportError` 包住會 fallback，但 fallback 目標（`cnn_autoencoder`）**沒有**再包一層 try/except，會直接讓整支腳本崩潰。

### ✅ 修正方式

```python
# ── [修正] 往上一層才是 core/ ──
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
```

或者更穩健的做法，把 `core/` 的路徑用「已知固定錨點」推導，避免未來檔案再搬動時又壞掉：

```python
# core/cnn_gradcam/run_gradcam.py 位於 core/cnn_gradcam/，core/ 是其父目錄的父目錄
_THIS_FILE = os.path.abspath(__file__)
_CORE_DIR  = os.path.dirname(os.path.dirname(_THIS_FILE))   # .../core
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)
```

---

<a id="p1-2"></a>
## 4. 🟠 P1 — 閾值調校與 Django 推論斷連

### 問題

`network_platform/settings.py`：

```python
CNN_THRESHOLD  = 0.000069   # 寫死的固定值
```

`analyzer/tasks.py` 直接讀取這個寫死的值來判斷異常：

```python
threshold    = float(settings.CNN_THRESHOLD)
anomaly_mask = errors > threshold
```

但 `run_threshold_tuning.py` 的調校結果是輸出到 `output/<dir>/threshold_report.json`，兩者之間**沒有任何自動同步機制**。每次重新訓練模型或重新調校閾值後，都必須有人手動把新的閾值抄進 `settings.py`，非常容易忘記，導致部署的模型與實際使用的閾值不一致。

### ✅ 修正方式

**方案 A（推薦，改動最小）**：讓模型檔本身攜帶閾值，訓練完成後把 threshold 和 state_dict 一起存，Django 端載入模型時一併讀出：

```python
# trainer.py / semi_supervised_trainer.py 儲存模型時 ── [修正]
torch.save({
    "model_state": self.model.state_dict(),
    "threshold":   self.threshold,
    "config":      self.config,
}, model_path)
```

```python
# analyzer/tasks.py ── [修正] _run_cnn_analysis / run_gradcam
ckpt = torch.load(model_path, map_location=device)
if isinstance(ckpt, dict) and "model_state" in ckpt:
    model = CNNAutoencoder(latent_dim=settings.CNN_LATENT_DIM)
    model.load_state_dict(ckpt["model_state"])
    threshold = float(ckpt.get("threshold", settings.CNN_THRESHOLD))  # fallback 保底
else:
    # 相容目前純 state_dict 格式
    model = CNNAutoencoder(latent_dim=settings.CNN_LATENT_DIM)
    model.load_state_dict(ckpt)
    threshold = float(settings.CNN_THRESHOLD)
model.eval()
```

**方案 B（不改儲存格式）**：在 Django 啟動或部署腳本中，讀取 `threshold_report.json` 並寫回 `settings.py`（或改用環境變數 `CNN_THRESHOLD` 讀取，讓 CI/CD 腳本負責同步）：

```python
# settings.py
CNN_THRESHOLD = float(os.getenv("CNN_THRESHOLD", "0.000069"))
```

兩種方案擇一即可；方案 A 更根本，建議優先採用。

---

<a id="p1-3"></a>
## 5. 🟠 P1 — Celery CNN 推論未分批，OOM 風險

### 問題

`analyzer/tasks.py` 的 `_run_cnn_analysis` 把整個 pcap 檔案的所有封包一次性轉圖、一次性送進模型做推論：

```python
images = []
for pkt in packets:
    img = vis.bytes_to_image(bytes(pkt))
    images.append(img)
...
X_tensor = torch.from_numpy(X[:, np.newaxis, :, :]).to(device)
with torch.no_grad():
    errors = model.reconstruction_error(X_tensor).cpu().numpy()   # 一次性全部丟進去
```

這與 core 訓練腳本中處處可見的「分批處理避免 GPU OOM」原則（在您的 8GB VRAM RTX 4060 環境下特別重要）不一致。大型 pcap（數萬〜數十萬封包）會一次性佔用大量顯存，很容易 OOM。

### ✅ 修正方式

```python
def _compute_errors_batched(model, X, device, batch_size=256):
    """[修正] 分批推論，避免大型 pcap 造成 GPU OOM"""
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i+batch_size, np.newaxis, :, :]).to(device)
            e = model.reconstruction_error(batch)
            errs.append(e.cpu().numpy())
    return np.concatenate(errs)


def _run_cnn_analysis(session, pcap_path):
    ...
    X = np.array(images, dtype=np.float32)
    errors = _compute_errors_batched(model, X, device, batch_size=256)   # 取代原本一次性推論
    ...
```

---

<a id="p1-4"></a>
## 6. 🟠 P1 — 封包上傳無檔案格式驗證

### 問題

`projects/forms.py` 的 `PacketFileUploadForm` 完全沒有對上傳的檔案做副檔名或內容驗證：

```python
class PacketFileUploadForm(forms.ModelForm):
    class Meta:
        model  = PacketFile
        fields = ('file', 'description', 'network_tag', 'capture_time')
```

任何已登入且有上傳權限的使用者，可以上傳**任意類型、任意大小（上限只受 `DATA_UPLOAD_MAX_MEMORY_SIZE = 500MB` 限制）**的檔案，標籤成「PCAP 檔案」。下游 `process_packet_file` 任務用 `scapy.rdpcap()` 讀取非法檔案時雖然有 `except Exception` 包住不會崩潰，但仍會佔用儲存空間、造成使用者困惑（誤以為成功但實際上封包數是 0）。

### ✅ 修正方式

```python
# projects/forms.py ── [修正] 加入副檔名與大小驗證
from django.core.exceptions import ValidationError

ALLOWED_EXTENSIONS = ('.pcap', '.pcapng', '.cap')
MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200MB，可依需求調整

class PacketFileUploadForm(forms.ModelForm):
    class Meta:
        model  = PacketFile
        fields = ('file', 'description', 'network_tag', 'capture_time')
        ...

    def clean_file(self):
        f = self.cleaned_data['file']
        name_lower = f.name.lower()
        if not name_lower.endswith(ALLOWED_EXTENSIONS):
            raise ValidationError(
                f'不支援的檔案格式，僅接受 {", ".join(ALLOWED_EXTENSIONS)}。')
        if f.size > MAX_UPLOAD_SIZE:
            raise ValidationError(
                f'檔案過大（{f.size/1024/1024:.1f}MB），上限為 '
                f'{MAX_UPLOAD_SIZE/1024/1024:.0f}MB。')
        # 進一步檢查 magic bytes（pcap: 0xA1B2C3D4 / 0xD4C3B2A1；pcapng: 0x0A0D0D0A）
        head = f.read(4)
        f.seek(0)
        valid_magic = head in (
            b'\xa1\xb2\xc3\xd4', b'\xd4\xc3\xb2\xa1',   # pcap (大小端)
            b'\x0a\x0d\x0d\x0a',                          # pcapng
        )
        if not valid_magic:
            raise ValidationError('檔案內容不像是合法的 PCAP/PCAPNG 檔案。')
        return f
```

（`api/views.py` 的 `PacketFileUploadAPI` 若也是直接存檔而未走這個表單，需要另外補上相同檢查。）

---

<a id="p2-1"></a>
## 7. 🟡 P2 — 消融實驗 / 半監督閾值調校 效能未優化

### 問題

`threshold_tuner.py` 的 `scan_percentiles()` 已經正確做到「PR-AUC 只計算一次，因為它與 threshold 無關」：

```python
# threshold_tuner.py 第 80-91 行（已優化，做得很好）
pr_pre, pr_rec, _ = precision_recall_curve(all_labels, all_errors)
pr_auc = float(sk_auc(pr_rec, pr_pre))
for pct in percentiles:
    thr = float(np.percentile(en, pct))
    res = self._evaluate_at_threshold(en, ea, thr, pr_auc=pr_auc)   # 傳入預先算好的值
```

但同一份精神**沒有**套用到另外兩處：

**(a) `ablation_study.py` 的 `_evaluate_threshold_auto`**（第214-228行）：

```python
def _evaluate_threshold_auto(errors_normal, errors_attack, pct_range=(80, 99.9), n_steps=40):
    best = {"f1": -1}
    for pct in np.linspace(*pct_range, n_steps):
        metrics = _evaluate_threshold(errors_normal, errors_attack, pct)   # 每次都重算 AUC/PR-AUC！
        ...
```

`_evaluate_threshold`（第231-281行）內部每次呼叫都：
1. 用**純 Python for 迴圈**（非向量化）手刻計算 AUC-ROC（第251-264行）
2. 呼叫 sklearn `precision_recall_curve` 計算 PR-AUC

這兩個指標其實與 percentile／threshold 完全無關，只取決於 `errors_normal`、`errors_attack` 本身，40 步掃描等於重複算了 40 次一模一樣的東西，而且 AUC-ROC 那段還是未向量化的 Python 迴圈，是最慢的部分。

**(b) `threshold_tuner_semi.py` 的 `calibrate_with_labels`**（第 109-121 行）情況更嚴重，掃描 `n_thresholds=200`（是 ablation 的 5 倍）個候選閾值：

```python
for thr in candidates:      # 200 次迴圈
    metrics = self._evaluate_at_threshold(
        errors_normal, errors_attack, float(thr)   # ← 沒有傳入 pr_auc，導致每次都重算
    )
```

它繼承自 `ThresholdTuner`，而父類別的 `_evaluate_at_threshold` 明明已經支援 `pr_auc=` 參數來跳過重算，這裡卻沒有使用。

### ✅ 修正方式

**`ablation_study.py`**：拆成「算一次的全域指標」+「逐 threshold 算的指標」：

```python
def _global_metrics(errors_normal, errors_attack):
    """[修正] AUC-ROC / PR-AUC 與 threshold 無關，只需計算一次"""
    all_errors = np.concatenate([errors_normal, errors_attack])
    all_labels = np.concatenate([np.zeros(len(errors_normal)), np.ones(len(errors_attack))])

    # 改用 sklearn 向量化實作取代手刻 Python 迴圈
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(all_labels, all_errors))

    pr_pre, pr_rec, _ = precision_recall_curve(all_labels, all_errors)
    pr_auc = float(sk_auc(pr_rec, pr_pre))
    return auc, pr_auc


def _evaluate_threshold(errors_normal, errors_attack, pct=95.0, auc=None, pr_auc=None):
    """[修正] auc / pr_auc 可由外部傳入，避免重算"""
    threshold = float(np.percentile(errors_normal, pct))
    fp = int((errors_normal > threshold).sum()); tn = len(errors_normal) - fp
    tp = int((errors_attack > threshold).sum()); fn = len(errors_attack) - tp
    precision = tp / (tp + fp + 1e-9); recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / (len(errors_normal) + len(errors_attack))
    fpr = fp / (fp + tn + 1e-9)

    if auc is None or pr_auc is None:
        auc, pr_auc = _global_metrics(errors_normal, errors_attack)

    return {"threshold": threshold, "percentile": pct, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
            "accuracy": accuracy, "auc": auc, "pr_auc": pr_auc}


def _evaluate_threshold_auto(errors_normal, errors_attack, pct_range=(80, 99.9), n_steps=40):
    """[修正] 全域指標算一次，迴圈內只重算 threshold 相依的部分"""
    auc, pr_auc = _global_metrics(errors_normal, errors_attack)   # ← 只算一次
    best = {"f1": -1}
    for pct in np.linspace(*pct_range, n_steps):
        metrics = _evaluate_threshold(errors_normal, errors_attack, pct, auc=auc, pr_auc=pr_auc)
        if metrics["f1"] > best["f1"]:
            best = metrics
            best["percentile"] = pct
    return best
```

**`threshold_tuner_semi.py`**：只需要在呼叫前補算一次、並傳進去：

```python
def calibrate_with_labels(self, X_normal, X_attack, n_thresholds: int = 200) -> list:
    ...
    errors_normal = self._compute_errors(X_normal)
    errors_attack = self._compute_errors(X_attack)

    # ── [修正] PR-AUC 與 threshold 無關，只需算一次 ──
    all_errors = np.concatenate([errors_normal, errors_attack])
    all_labels = np.concatenate([np.zeros(len(errors_normal)), np.ones(len(errors_attack))])
    pr_pre, pr_rec, _ = precision_recall_curve(all_labels, all_errors)
    pr_auc = float(sk_auc(pr_rec, pr_pre))

    ...
    for thr in candidates:
        metrics = self._evaluate_at_threshold(
            errors_normal, errors_attack, float(thr), pr_auc=pr_auc   # ← 補上
        )
        ...
```

這個修正對 `threshold_tuner_semi.py` 的效益最明顯：200 次 sklearn `precision_recall_curve` 呼叫降為 1 次。

---

<a id="p2-2"></a>
## 8. 🟡 P2 — `cnn_autoencoder.py` 雙重定義（importlib 動態載入）

### 問題

專案中有兩個檔案都叫 `cnn_autoencoder.py`：

- `core/cnn_autoencoder.py`：真正的模型定義（`Encoder`、`Decoder`、`CNNAutoencoder`），v3.0 優化版
- `core/Semi-supervised model and trainer/cnn_autoencoder.py`：一個「轉接層」，用 `importlib.util.spec_from_file_location` **動態載入**根目錄版本，避免命名衝突造成的循環 import：

```python
# core/Semi-supervised model and trainer/cnn_autoencoder.py 第 31-44 行
_root_cnn_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cnn_autoencoder.py"
)
_spec = _ilu.spec_from_file_location("cnn_autoencoder_root", _root_cnn_path)
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Encoder = _mod.Encoder
...
```

**風險**：這個模組被登記在 `sys.modules` 中的名稱是 `"cnn_autoencoder_root"`，而如果同一個 Python process 中，其他程式（如 `ablation_study.py`、`trainer.py`）用一般的 `from cnn_autoencoder import CNNAutoencoder`（在 `core/` 目錄下執行時會解析成 `sys.modules["cnn_autoencoder"]`），兩邊拿到的其實是**兩個獨立載入、互不相同的 class 物件**。任何用到 `isinstance(model, CNNAutoencoder)` 的地方，只要比較的物件分別來自這兩條載入路徑，判斷都會失敗（即便看起來是「同一個類別」）。此外，`os.path.dirname(os.path.dirname(...))` 這種寫死的相對層級關係，只要檔案搬動位置就會失效（類似前面 P1-1 的問題）。

### ✅ 修正方式

最根本的解法是把 `core/` 變成一個正常的 Python package（加上 `__init__.py`），讓所有子目錄用相對 import：

```python
# core/__init__.py（新建，可留空）
# core/Semi-supervised model and trainer/__init__.py（新建，可留空）
```

```python
# core/Semi-supervised model and trainer/cnn_autoencoder.py ── [修正]
# 改用相對 import，取代 importlib 動態載入
from ..cnn_autoencoder import (
    Encoder, Decoder, CNNAutoencoder, features_to_image, evaluate,
)
```

若暫時不想大改目錄結構（例如已有很多 `python xxx.py` 直接執行的使用慣例，不想全部改成 `python -m`），至少要讓 `import cnn_autoencoder` 全專案都走同一份，方法是在每支腳本開頭統一插入 `core/` 到 `sys.path`最前面，而不要讓「腳本自己所在目錄」被優先解析到：

```python
import os, sys
_CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)   # 確保 core/ 優先於腳本自己的目錄
```

並且刪除 `Semi-supervised model and trainer/` 資料夾內重複的 `cnn_autoencoder.py`，全部改成從 `core/cnn_autoencoder.py` 直接 import（`UnsupervisedTrainer` / CLI 入口可以另外拆成獨立檔名，例如 `unsupervised_trainer.py`，避免檔名撞在一起）。

---

<a id="p2-3"></a>
## 9. 🟡 P2 — 封包解析欄位覆寫問題

### 問題

`parser.py` 的 `record` dict 只有**一個** `"checksum"` 欄位，但 IP 層與傳輸層（TCP/UDP）都會寫入這個欄位：

```python
# _parse_ipv4()（第 85-95 行）
record["checksum"] = hex(ip.chksum) if ip.chksum else None

# _parse_tcp()（第 127-134 行，稍後才執行）
record["checksum"] = hex(tcp.chksum) if tcp.chksum else None   # 覆蓋掉上面 IP 的值
```

因為 `parse()` 方法會依序呼叫 `_parse_ipv4()` 再呼叫 `_parse_transport()`（進而呼叫 `_parse_tcp`/`_parse_udp`），對於任何 TCP/UDP 封包，**IP 層的 checksum 會被無聲地丟棄**，最終 `record["checksum"]` 只反映傳輸層的值。如果任何報告/分析功能依賴這個欄位判斷 IP 層完整性，結果會是錯的。

另外，`hex(ip.chksum) if ip.chksum else None` 這種寫法對 `chksum == 0`（合法但少見的數值，在模擬/構造封包時更容易出現）也會被誤判成「沒有值」而回傳 `None`。

### ✅ 修正方式

```python
# parser.py ── [修正] 拆成獨立欄位，並用 is not None 取代真值判斷
record = {
    ...
    "ip_checksum":        None,
    "transport_checksum": None,
    # 移除共用的 "checksum" 欄位，或保留但不要被覆寫
    ...
}

def _parse_ipv4(self, pkt, record):
    ip = pkt[IP]
    ...
    record["ip_checksum"] = hex(ip.chksum) if ip.chksum is not None else None

def _parse_tcp(self, pkt, record):
    tcp = pkt[TCP]
    ...
    record["transport_checksum"] = hex(tcp.chksum) if tcp.chksum is not None else None

def _parse_udp(self, pkt, record):
    udp = pkt[UDP]
    ...
    record["transport_checksum"] = hex(udp.chksum) if udp.chksum is not None else None
```

（若下游資料庫 schema 或前端模板已經綁定 `checksum` 這個欄位名稱，可以保留 `checksum` 作為 `transport_checksum` 的別名，同時新增 `ip_checksum`，將改動範圍降到最小。）

---

<a id="p2-4"></a>
## 10. 🟡 P2 — Django `settings.py` 正式環境設定

```python
SECRET_KEY = 'django-dev-key-請在正式環境替換成隨機字串'   # 已自我註記，但仍是明碼
DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '*']
AUTH_PASSWORD_VALIDATORS = []
CORS_ALLOW_ALL_ORIGINS = True
```

這組合對「正式部署」是不安全的：`DEBUG=True` 會在錯誤頁面洩漏完整 traceback 與環境變數；`ALLOWED_HOSTS` 含 `'*'` 等於停用 Host header 驗證；`AUTH_PASSWORD_VALIDATORS=[]` 允許任意弱密碼；`CORS_ALLOW_ALL_ORIGINS=True` 讓任何網域都能發 CORS 請求。若目前僅用於本機開發/展示，這些可以先不動，但**上線或公開展示前**建議至少做到：

```python
# settings.py ── [修正] 用環境變數區分開發/正式環境
import os

DEBUG = os.getenv('DJANGO_DEBUG', 'false').lower() == 'true'
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY')
if not SECRET_KEY:
    if DEBUG:
        SECRET_KEY = 'django-dev-key-僅供本機開發使用'
    else:
        raise RuntimeError('正式環境必須設定 DJANGO_SECRET_KEY 環境變數')

ALLOWED_HOSTS = os.getenv('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
     'OPTIONS': {'min_length': 8}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

CORS_ALLOW_ALL_ORIGINS = DEBUG   # 正式環境改用 CORS_ALLOWED_ORIGINS 白名單
```

---

<a id="p2-5"></a>
## 11. 🟡 P2 — 異常封包模擬器：`os.chdir()` 於 import 時期執行，汙染測試環境

### 說明

`simulate_anomaly_traffic.py` 的**六種攻擊產生器本身邏輯設計得很扎實**——`gen_syn_flood()`、`gen_dns_amplification()` 都有明確註解說明「為什麼要這樣寫才能符合 `anomaly_detector.py` 實際的偵測邏輯」（例如 SYN Flood 改用「少量固定來源、各自發送大量封包」而非每個封包都隨機換來源 IP，避免偵測器的 `syn_count[src_ip]` 永遠不會累積超過閾值），`bulk_scale()` 對邊界值的處理也考慮得很周全，本次審查**沒有在攻擊產生邏輯本身找到 bug**。

但深入複查 `main()` 與模組載入邏輯後，找到一個實際會影響測試穩定性的問題。

### 問題

```python
# simulate_anomaly_traffic.py 第 46-52 行
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CORE_DIR)

if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

os.chdir(PROJECT_ROOT)   # ← 模組層級的程式碼，沒有被 `if __name__ == "__main__":` 保護
```

`os.chdir()` 是**模組層級**的敘述，不在任何函式內，代表只要這個檔案被 **import**（而不只是被當作腳本直接執行），這行就會立刻執行一次，把整個 Python 行程的工作目錄切換掉。

問題在於 `tests/test_simulated_anomaly.py` 正是用 import 的方式使用這個模組：

```python
# core/tests/test_simulated_anomaly.py 第 7 行
from simulate_anomaly_traffic import GENERATORS, verify_packets
```

只要執行到這行 `import`，pytest 行程的工作目錄就會被切換到 `PROJECT_ROOT`。如果同一個 pytest session 裡還有其他測試依賴「原本的工作目錄」讀取相對路徑檔案（例如其他測試模組用相對路徑載入 fixture 或設定檔），就會因為測試檔案的收集/執行順序不同而出現不穩定、難以重現的失敗——這類「import 產生副作用」的寫法是比較危險的模式，效果類似全域變數，但更隱蔽。

### ✅ 修正方式

把 `os.chdir()` 移到 `main()` 內部，只有直接執行腳本時才切換工作目錄；模組被 import 時只做 `sys.path` 設定，不應有任何全域副作用：

```python
# ── [修正] import 階段只設定 sys.path，不做 chdir ──
import os
import sys

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CORE_DIR)

if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

# os.chdir(PROJECT_ROOT)   ← 刪除這裡的呼叫

...

def main():
    os.chdir(PROJECT_ROOT)   # ← [修正] 只有直接執行 python simulate_anomaly_traffic.py 時才切換
    parser = argparse.ArgumentParser(
        description="隨機模擬異常網路封包，輸出 PCAP 並可自我驗證是否觸發 AnomalyDetector"
    )
    ...


if __name__ == "__main__":
    main()
```

更根本的解法，是讓 `config.py` 內的輸出路徑（`output/sessions/...` 等）一開始就用**絕對路徑**（以 `config.py` 自身檔案位置為錨點推算），這樣無論呼叫者當下的工作目錄是什麼，路徑都會正確，完全不需要靠 `os.chdir()` 這種全域副作用來保證：

```python
# config.py 建議寫法（若尚未這樣做）── [修正]
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")   # 絕對路徑，不受呼叫時的 cwd 影響
```

採用這個修正後，`simulate_anomaly_traffic.py` 無論是被直接執行、被 `pytest` import，或未來被 Django（例如 `analyzer/tasks.py` 若想串接「一鍵產生測試流量」功能）import，都不會再意外改動呼叫者的工作目錄。

### 附帶次要優化建議（非 bug）

`gen_port_scan()` 第 221 行：

```python
candidates = [p for p in range(1, 65536) if p not in ports]
```

`ports` 是 list，`p not in ports` 是線性搜尋，對 6.5 萬個候選 port 各做一次會有點浪費（雖然以目前 threshold 規模不會真的造成效能問題，優先度低於上面的 `os.chdir()` 問題）：

```python
# ── [優化] 用 set 取代 list 做成員檢查，O(1) 取代 O(n) ──
ports_set = set(ports)
candidates = [p for p in range(1, 65536) if p not in ports_set]
```

---

<a id="p3-1"></a>
## 12. 🟢 P3 — 其他次要問題與程式碼品質建議

### (a) `gradcam_core.py` — `_scorecam` 多做了一次不必要的前向傳播

```python
# 第 184-190 行
with torch.no_grad():
    x_hat_base, _ = self.model(x)          # 第一次 forward（順便觸發 hook）
    baseline_score = F.mse_loss(...)

with torch.no_grad():
    _, _ = self.model(x)                    # ← 多餘的第二次 forward，只為了「重新」觸發 hook
activation = self._hook_manager.get_activation(self._resolved_layer).detach()
```

其實第一次 `self.model(x)` 已經觸發了 forward hook，`activation` 這時就已經可用，不需要再呼叫一次。修正：

```python
with torch.no_grad():
    x_hat_base, _ = self.model(x)
    baseline_score = F.mse_loss(x_hat_base, x, reduction="none").mean(dim=[1, 2, 3])
activation = self._hook_manager.get_activation(self._resolved_layer).detach()   # 直接重用，刪除多餘的第二次 forward
```

### (b) `gradcam_core.py` — `_gaussian_smooth` 用純 Python 巢狀迴圈做卷積

第 246-273 行的高斯平滑是用兩層 Python `for` 迴圈（逐像素）實作可分離卷積，對 32×32 影像也要跑 1024×2 次 Python 層級運算，在批次分析（Django 一次處理 20 張、ablation/gradcam 分析多個樣本）時會是明顯瓶頸。建議改用 `scipy.ndimage.gaussian_filter1d`：

```python
from scipy.ndimage import gaussian_filter1d

@staticmethod
def _gaussian_smooth(cam: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    # 對每張圖的 H、W 兩個軸分別做 1D 高斯濾波（等效於原本的可分離卷積）
    smoothed = gaussian_filter1d(cam, sigma=sigma, axis=1, mode="reflect")
    smoothed = gaussian_filter1d(smoothed, sigma=sigma, axis=2, mode="reflect")
    return smoothed
```

若不想引入 scipy 依賴，也可以用 `numpy` 的 `np.convolve` 沿軸向量化，同樣能避免逐像素 Python 迴圈。

### (c) `packet_visualizer.py` — `_apply_field_mask` 隱性假設 `skip_ethernet=True`

`_apply_field_mask()` 內部用的位移量（byte 0 = IP 版本/IHL、byte 9 = protocol、byte 12-19 = src/dst IP）都是「已經跳過 Ethernet Header」為前提。但建構子允許 `apply_mask=True` 搭配 `skip_ethernet=False`，這種組合下遮罩會套用到錯誤的位移（誤把 MAC 位址、EtherType 當成 IP 欄位），產生的封包影像會是壞資料。目前程式碼沒有防呆機制：

```python
# packet_visualizer.py ── [修正] 加入防呆或明確禁止不合理組合
def __init__(self, image_size="medium", apply_mask=True, normalize=True, skip_ethernet=True):
    if apply_mask and not skip_ethernet:
        raise ValueError(
            "apply_mask=True 目前僅支援搭配 skip_ethernet=True，"
            "因為欄位遮罩的位移量是以「已跳過 Ethernet Header」為前提計算。"
        )
    ...
```

另外，模組頂部定義的 `IPV4_SRC_OFFSET`、`IPV4_DST_OFFSET`、`IPV4_IHL_OFFSET`、`TCP_SPORT_OFFSET_BASE`、`TCP_DPORT_OFFSET_BASE` 這幾個常數實際上完全沒被 `_apply_field_mask()` 使用（該函式改用內部字面值 `10`、`12`、`ihl` 等），屬於死碼，建議清除或改成實際被引用。

### (d) `projects/views.py` — `member_add` 缺少 `@require_POST`

`member_add`、`member_remove` 這類有 side effect 的 view，其他類似操作（如 `session_delete`）都有加上 `@require_POST`，但這兩個沒有。雖然目前不構成可利用的漏洞（GET 請求不會帶入 `request.POST` 資料，形同無效呼叫），但為了程式碼一致性與意圖明確，建議補上：

```python
from django.views.decorators.http import require_POST

@login_required
@require_POST
def member_add(request, pk):
    ...

@login_required
@require_POST
def member_remove(request, pk, user_pk):
    ...
```

### (e) `analyzer/views.py` — `session_delete` 權限模型與 `_check_session_access` 不一致

`_check_session_access`（其他 view 使用）允許：建立者、admin、專案擁有者、專案成員。但 `session_delete` 自己另外寫了一套更嚴格的檢查（只允許建立者或 admin）：

```python
def session_delete(request, pk):
    session = get_object_or_404(AnalysisSession, pk=pk)
    if session.created_by != request.user and not request.user.profile.is_admin:
        ...
```

這不是安全漏洞（更嚴格而非更寬鬆），但屬於權限模型不一致：專案成員可以檢視、觸發 Grad-CAM、匯出報告，卻不能刪除 Session。如果這是刻意設計（例如「刪除」被視為更敏感的操作），建議在程式碼註解中說明；如果不是刻意的，建議統一改用 `_check_session_access` 或明確定義新的角色權限旗標（如 `can_delete_session`）。

### (f) `simulate_anomaly_traffic.py` — 六種攻擊產生器邏輯本身經複查無誤

`bulk_scale`、`eth_wrap`、`jitter_times`、`gen_syn_flood`、`gen_dns_amplification` 等都有詳細註解說明設計理由（例如刻意讓 SYN Flood 用少量固定來源集中發送，以符合 `anomaly_detector.py` 以單一來源 IP 計數的偵測邏輯），顯示先前已仔細除錯過，本次複查未在攻擊產生邏輯本身找到問題。**但模組載入階段有一個會影響測試穩定性的問題，詳見 [11. 🟡 P2 — 異常封包模擬器：`os.chdir()` 於 import 時期執行](#p2-5)。**

### (g) `pcap_analyzer.py` / `parser.py` 已記錄的既有修正

這兩個檔案內部都留有清楚的 `[Bug N 修正]` 註解（TLS 判斷優先權、DNS port 誤判、TCP 雙向流合併等），修正邏輯經檢查皆正確無誤，值得肯定。

---

<a id="good"></a>
## 13. ✅ 已確認做得好的部分

- **`threshold_tuner.py`**：`scan_percentiles()` 正確地把「與 threshold 無關的 PR-AUC」抽出來只算一次，是本次審查中「效能優化」的正確參考範例（可惜沒有被套用到 `ablation_study.py` 和 `threshold_tuner_semi.py`）。
- **`ablation_study.py`**：`AblationStudy.__init__` 有正確做 80/20 holdout 切分並附註解「實作 Holdout 分割以避免資料外洩」；統計顯著性檢定（Welch t-test、95% CI）設計嚴謹。
- **`gradcam_hooks.py`**：`HookManager` 的 context manager 設計（`__enter__`/`__exit__` 自動 `remove_all()`）正確避免了 PyTorch hook 常見的記憶體洩漏問題。
- **`gradcam_core.py`**：已修正 Grad-CAM++ 缺少 `model.zero_grad()` 導致梯度污染的問題，並移除不必要的 `create_graph=True`。
- **`packet_visualizer.py`**：v2.0.1 已修正三個具體 bug（dtype 誤判、IHL 動態計算、entropy 溢位），修正邏輯正確。
- **`accounts/forms.py`**：`RegisterForm` 已明確排除 `admin` 角色自選，防止註冊時的權限提升漏洞，安全意識良好。
- **`projects/views.py` / `analyzer/views.py`（HTML 版）**：物件層級權限檢查（owner / member / admin）覆蓋完整、一致。
- **`reports/generators.py`**：匯出檔名以 `session.pk`（伺服器端整數）組成，無路徑穿越風險。
- **`pcap_analyzer.py`**：`detect_attacks()` 回傳格式、`rebuild_tcp_streams()` 雙向流合併等問題都有清楚的修正記錄且邏輯正確。
- **`simulate_anomaly_traffic.py`**：六種攻擊產生器（`gen_syn_flood` 等）皆有明確註解說明如何對齊 `anomaly_detector.py` 的實際偵測邏輯（例如避免每個封包都隨機換來源 IP 導致偵測器計數器打散），設計思路嚴謹。

---

<a id="priority-table"></a>
## 14. 修正優先順序總表

| 優先級 | 問題 | 檔案 | 影響範圍 |
|---|---|---|---|
| 🔴 P0 | 資料洩漏疑似回歸 | 5 個訓練/評估腳本 | **研究結果有效性**（最優先） |
| 🔴 P0 | REST API IDOR 越權 | `api/views.py`（8 個端點） | **資訊安全**（跨使用者資料外洩） |
| 🟠 P1 | Grad-CAM sys.path 錯誤 | `cnn_gradcam/run_gradcam.py` | 功能無法執行 |
| 🟠 P1 | 閾值與訓練管線脫節 | `settings.py` + `tasks.py` | 部署後偵測結果失準 |
| 🟠 P1 | CNN 推論未分批 | `analyzer/tasks.py` | GPU OOM 風險 |
| 🟠 P1 | 封包上傳無格式驗證 | `projects/forms.py` | 任意檔案上傳 |
| 🟡 P2 | 消融/閾值調校效能 | `ablation_study.py`、`threshold_tuner_semi.py` | 訓練/實驗耗時 |
| 🟡 P2 | cnn_autoencoder 雙重定義 | 兩份 `cnn_autoencoder.py` | 潛在 isinstance 錯誤 |
| 🟡 P2 | checksum 欄位覆寫 | `parser.py` | 封包分析資料正確性 |
| 🟡 P2 | 正式環境設定 | `settings.py` | 上線前必須處理 |
| 🟡 P2 | `os.chdir()` 於 import 時期執行 | `simulate_anomaly_traffic.py` | 測試結果不穩定（pytest 順序相依） |
| 🟢 P3 | 其餘程式碼品質建議 | 多檔案 | 可視時間彈性處理 |

---

*本報告基於 2026-07-27 匯出的程式碼快照。若您已在其他分支/工作副本中修正了部分問題（尤其是第 1 節提到的資料洩漏疑似回歸），建議先確認匯出當下的檔案版本是否為最新。*
