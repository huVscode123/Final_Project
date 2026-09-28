# VNAAP AI 客服代理人（Gemini 直連版）— 安裝與使用說明

這份文件說明如何在 VNAAP（視覺化網路攻擊自動化分析平台）中，把原本
「Django → n8n（通常用 Docker 跑）→ Gemini」的 AI 客服架構，換成
**「Django → Gemini」直連架構**，全程不需要 Docker、不需要另外啟動
任何服務，只依賴專案原本就有的 `requests` 套件與 Google Gemini 的
公開 HTTPS API。

---

## 1. 這個功能做什麼

在 `/analyzer/ai-chat/` 頁面裡，AI 客服「小網」支援五種對話模式：

| 模式 | 對應前端按鈕 | 做什麼 |
|---|---|---|
| `general` | 💬 一般對話 | 閒聊、平台簡介、不確定分類的問題 |
| `customer_service` | 🎧 客服對話 | 帳號、專案協作、權限、報告匯出等行政面問題 |
| `web_guide` | 🧭 網頁使用引導 | 手把手教操作，附上實際頁面路徑與步驟 |
| `security_knowledge` | 🛡️ 資安知識 | 用白話文科普攻擊手法與防禦觀念（純教育，不提供攻擊步驟） |
| `explain_result` | 📊 白話文解讀結果 | 把選定 Session 的告警／CNN 結果翻成白話文說明 |

每個模式都有各自的系統提示詞（`prompts.json`），並會依使用者輸入的
關鍵字，從內建知識庫（`knowledge_base.json`）撈出相關的操作步驟／
資安知識／FAQ 片段，組進提示詞後才送給 Gemini，讓回答更貼近平台實際
的功能與設計（例如「最終判定只看規則引擎、CNN 分數僅供參考」這個
VNAAP 的核心設計原則，AI 在 `explain_result` 模式會確實遵守）。

---

## 2. 架構說明：為什麼不需要 Docker

**舊架構**：

```
瀏覽器 → Django (AIChatAPI) → HTTP POST → n8n (通常跑在 Docker 容器裡)
                                              → Gemini API
```

**這一版**：

```
瀏覽器 → Django (AIChatAPI) → analyzer.ai_agent.agent.generate_reply()
                                  → Gemini API（HTTPS 直連公開端點）
```

n8n 本身是一個需要獨立部署（常見做法是用 Docker）的工作流引擎；這裡
把「組系統提示詞 → 呼叫 Gemini → 回傳文字」整段邏輯直接寫成 Python
函式，跑在 Django 的同一個行程裡。少了 n8n 這一層之後：

- 不需要 Docker / docker-compose。
- 不需要維護、啟動、監控另一個服務。
- 唯一的外部相依是 `https://generativelanguage.googleapis.com`（HTTPS
  直連），只要能上網就能用，跟 n8n 是否存活無關。
- 所有邏輯（模式提示詞、知識庫）都是版本控制得到的檔案，不是存在
  n8n 介面上的視覺化工作流，比較方便 code review 與教學展示。

---

## 3. 檔案清單與放置位置

請把以下檔案複製到你 VNAAP 專案對應的路徑（路徑已經對齊實際專案結構，
直接照樣貼上即可）：

```
analyzer/
├── ai_agent/
│   ├── __init__.py              ← 新增
│   ├── gemini_client.py         ← 新增：Gemini REST API 最小化封裝
│   ├── agent.py                 ← 新增：組提示詞 + 關鍵字檢索 + 呼叫 Gemini
│   ├── prompts.json             ← 新增：五種模式的系統提示詞
│   └── knowledge_base.json      ← 新增：操作引導 / 資安知識 / FAQ 知識庫
└── management/
    ├── __init__.py               ← 新增（若已存在則不用重複建立）
    └── commands/
        ├── __init__.py           ← 新增
        └── list_gemini_models.py ← 新增：查詢目前 API Key 可用的模型

api/
└── views.py                      ← 修改：套用 views_ai_chat_patch.py 的內容
                                     （patch 檔本身不需要放進專案，只是
                                     用來對照、複製貼上的參考檔）

network_platform/
└── settings.py                   ← 修改：加入第 4 節的設定值
```

> `analyzer` app 已經在 `INSTALLED_APPS` 裡，所以新增
> `analyzer/ai_agent/`、`analyzer/management/` 不需要額外註冊任何
> Django app，也不需要新增資料庫 migration（這個功能完全沒有用到
> ORM model）。

---

## 4. 安裝步驟

### 4.1 複製檔案

依照第 3 節的對照表，把檔案放到專案裡對應的位置。

### 4.2 設定 `GEMINI_API_KEY`（你自己手動輸入）

先到 [Google AI Studio](https://aistudio.google.com/apikey) 申請一組
免費的 Gemini API Key，接著擇一設定：

**方式 A：環境變數（建議）**

```bash
# Linux / macOS
export GEMINI_API_KEY="貼上你的金鑰"

# Windows PowerShell
$env:GEMINI_API_KEY = "貼上你的金鑰"
```

**方式 B：直接寫進 `settings.py`**（個人／教學用途最快，但**不要**把
含有金鑰的 `settings.py` 推上公開的 git repo）

```python
# network_platform/settings.py
GEMINI_API_KEY = "貼上你的金鑰"
```

### 4.3 在 `settings.py` 加入設定

在 `network_platform/settings.py` 裡（放在原本 `N8N_WEBHOOK_URL` 那幾
行附近即可）加入：

```python
# ── Gemini API（直連，不需要 Docker / n8n）─────────────────────
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')          # 見 4.2，二選一設定
GEMINI_MODEL   = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
GEMINI_API_TIMEOUT      = int(os.getenv('GEMINI_API_TIMEOUT', '30'))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv('GEMINI_MAX_OUTPUT_TOKENS', '1024'))
```

`GEMINI_MODEL` 的預設值只是「先求能動」的合理猜測——Gemini 的模型
名稱會隨時間更新，請務必做下一步確認。原本的 `N8N_WEBHOOK_URL` /
`AI_CHAT_TIMEOUT` 這兩個設定套用本功能後不會再被用到，留著或刪除都
不影響運作。

### 4.4 套用 `api/views.py` 補丁

打開 `api/views_ai_chat_patch.py`，依照檔案最上方的說明，把
`api/views.py` 裡原本呼叫 n8n webhook 的 `AI_CHAT_MODES` /
`_build_ai_context` / `AIChatAPI` 三個區塊，換成補丁檔裡的版本，並在
檔案頂端加上：

```python
from analyzer.ai_agent.agent import generate_reply
from analyzer.ai_agent.gemini_client import GeminiConfigError, GeminiAPIError
```

### 4.5 確認依賴套件

`requirements.txt` 裡已經有 `requests>=2.31`，不需要新增任何套件、
也不需要安裝 `google-generativeai` 這類 SDK——本功能只用標準的 HTTP
POST 呼叫 Gemini 的 REST API。

### 4.6 確認金鑰與模型可用

```bash
python manage.py list_gemini_models
```

會列出目前這把金鑰實際可用的模型，前面標 `[✓]` 的代表支援
`generateContent`（也就是聊天用的那個功能），選一個填回
`GEMINI_MODEL`（環境變數或 `settings.py` 都可以）。

### 4.7 啟動並測試

```bash
python manage.py runserver
```

瀏覽器打開 `/analyzer/ai-chat/`，任選一個模式輸入訊息測試。也可以直
接用 curl 打 API（`YOUR_SESSION_COOKIE` 換成登入後瀏覽器的 session
cookie，或改用 Token 認證）：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ai/chat/ \
  -H "Content-Type: application/json" \
  -H "Cookie: sessionid=YOUR_SESSION_COOKIE" \
  -H "X-CSRFToken: YOUR_CSRF_TOKEN" \
  -d '{"message": "什麼是 SYN Flood？", "mode": "security_knowledge"}'
```

---

## 5. 如何擴充知識庫

`prompts.json` 和 `knowledge_base.json` 都是純資料檔，改完存檔即可，
不需要重啟服務──若使用的是開發伺服器（`runserver`）本來就會偵測檔案
變動自動重載；若是正式部署（gunicorn 等常駐行程），改完後呼叫一次：

```python
from analyzer.ai_agent.agent import reload_knowledge
reload_knowledge()
```

即可清除行程內快取、立即生效（或直接重啟服務也可以）。

**新增一筆操作引導**（`web_guide_steps` 陣列裡加一個物件）：

```json
{
  "topic": "功能名稱",
  "keywords": ["使用者可能會打的關鍵字", "同義詞"],
  "url": "/analyzer/xxx/",
  "steps": ["第一步...", "第二步...", "第三步..."]
}
```

**新增一筆資安知識**（`security_topics` 陣列）：

```json
{
  "topic": "攻擊或觀念名稱",
  "keywords": ["關鍵字1", "關鍵字2"],
  "summary": "一段白話說明",
  "tips": ["防禦建議1", "防禦建議2"]
}
```

**新增一筆 FAQ**（`faq` 陣列）：

```json
{ "question": "常見問題", "answer": "標準答案", "keywords": ["關鍵字"] }
```

知識片段的挑選方式是**純關鍵字比對**（`agent.py::retrieve_snippets`）：
把使用者輸入斷詞後，和每個條目的 `keywords` 取交集計分，取分數最高
的前 `top_k`（預設 2）筆組進提示詞。這個做法刻意不引入向量資料庫或
額外服務，符合「不使用 Docker」的限制；條目數量不多時，關鍵字比對已
經夠準，如果之後條目數暴增、比對開始不夠準，再考慮升級成向量檢索也
不遲。

---

## 6. 疑難排解

| 現象 | 可能原因 / 處理方式 |
|---|---|
| 回覆是「⚠ AI 服務尚未設定完成」 | 還沒設定 `GEMINI_API_KEY`，見第 4.2 節 |
| Gemini API 回傳 400 | 通常是 `GEMINI_MODEL` 字串打錯或該模型不支援 `generateContent`，執行 `python manage.py list_gemini_models` 確認 |
| Gemini API 回傳 403 | API Key 錯誤，或該金鑰尚未啟用 Generative Language API 的存取權限 |
| Gemini API 回傳 404 | 模型名稱不存在，或這把金鑰／帳號無權使用該模型 |
| Gemini API 回傳 429 | 已達免費額度或速率上限，稍等一下再試，或參考 Google AI Studio 的配額頁面 |
| 回覆內容跟平台實際功能對不上 | 檢查 `knowledge_base.json` 是否有對應條目、`keywords` 是否涵蓋使用者常用的講法 |
| 回覆很籠統、沒有引用知識庫內容 | 表示關鍵字比對沒有命中，補充該主題在 `knowledge_base.json` 的 `keywords` |

---

## 7. 安全性與隱私提醒

- **不要把含有真實 API Key 的 `settings.py` 或 `.env` 推上公開的 git
  repo**；優先使用環境變數，或把金鑰所在的檔案加進 `.gitignore`。
- `explain_result` 模式會把選定 Session 的告警內容、CNN 統計數字送到
  Gemini 的伺服器進行推論；若未來要處理更敏感的內部資料，記得評估
  這一層的資料外流風險（例如改用地端／私有部署的模型）。
- `security_knowledge` 模式的系統提示詞已經明確要求模型「只做觀念科
  普與防禦建議，不提供實際攻擊步驟」，但這是提示詞層級的約束，不是
  100% 保證；如果要更嚴謹，可以額外加一層輸出過濾或人工審核機制。
- 目前設計是「無狀態」：每一輪對話都是獨立呼叫，不會記住前幾輪講過
  什麼（除了同一輪裡附上的 Session context）。若要做到「同一次對話
  記得前面講過什麼」，可以之後再加一個 `ChatMessage` model 儲存歷史
  紀錄，並在 `agent.generate_reply()` 裡把最近幾輪對話一併組進
  `contents`，這屬於後續可以再擴充的方向，這次先聚焦在「不用 Docker
  也能動」這個目標。

---

## 8. 後續可能的強化方向（非必要，先列給有興趣的人）

- 串流回覆（`streamGenerateContent`），讓前端能一個字一個字顯示，體感
  更快。
- 加上 `ChatMessage` model，讓同一位使用者的對話有記憶、可回顧歷史。
- 針對 `security_knowledge` / `explain_result` 模式加上簡單的關鍵字
  黑名單，在送出請求前就攔截明顯的惡意提問（提示詞防禦之外再加一層）。
- 依使用者或 IP 做簡單的呼叫頻率限制，避免免費額度被單一使用者用光。
