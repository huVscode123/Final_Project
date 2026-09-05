# VNAAP AI Agent — n8n 部署與配置指南

## 前置需求

| 項目 | 版本 / 說明 |
|---|---|
| n8n | ≥ 1.40（需支援 LangChain 節點） |
| Google Gemini API Key | 透過 [Google AI Studio](https://aistudio.google.com/app/apikey) 取得 |
| Django 後端 | VNAAP 平台已啟動（`python manage.py runserver`） |

---

## 快速部署步驟

### Step 1：啟動 n8n

```bash
# Docker 方式（推薦）
docker run -it --rm --name n8n -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  n8nio/n8n

# 或 npm 安裝方式
npx n8n start
```

### Step 2：配置 Gemini API 憑證

1. 開啟 n8n 介面 → 左側選單 → **Credentials**
2. 點擊 **Add Credential** → 搜尋 **Google Gemini (PaLM) API**
3. 輸入你的 API Key → 儲存
4. **⚠️ 安全提醒：絕對不要在任何文件或聊天中分享 API Key**

### Step 3：導入 Workflow

1. 開啟 n8n 介面 → 左上角 **+** → **Import from File**
2. 選擇 `files/vnaap_ai_agent_workflow.json`
3. 導入後，打開 **Gemini 2.0 Flash** 節點
4. 在 Credential 欄位選擇 Step 2 建立的 Gemini API 憑證
5. 儲存 Workflow

### Step 4：測試 Workflow

1. 點擊 n8n 介面上方的 **Test workflow** 按鈕
2. 使用以下 cURL 指令測試：

```bash
curl -X POST http://localhost:5678/webhook-test/ai-chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好，請問 VNAAP 是什麼？",
    "mode": "general",
    "context": {},
    "user": {"username": "test_user", "is_admin": false},
    "chat_session_id": "user-1"
  }'
```

預期回應：
```json
{
  "reply": "你好！VNAAP 是視覺化網路攻擊自動化分析平台...",
  "mode": "general",
  "ok": true
}
```

### Step 5：啟用 Workflow

確認測試通過後：
1. 點擊右上角的 **Inactive** 開關 → 切換為 **Active**
2. 此時 Webhook URL 從 `/webhook-test/ai-chat` 變為 `/webhook/ai-chat`
3. Django 後端的 `N8N_WEBHOOK_URL` 設定為 `http://localhost:5678/webhook/ai-chat`

---

## Workflow 節點說明

```
[AI Chat Webhook] → [Build Prompt] → [AI Agent] → [Format Response] → [Respond to Webhook]
   POST接收             模式分流         Gemini        格式化回應         回傳 JSON
                      系統提示詞       + 對話記憶
```

### 節點詳情

| 節點名稱 | 類型 | 職責 |
|---|---|---|
| **AI Chat Webhook** | Webhook | 接收 Django 的 POST 請求 |
| **Build Prompt** | Code | 根據 mode 選擇系統提示詞，explain_result 模式附加分析數據 |
| **AI Agent** | LangChain Agent | 使用 Gemini 2.0 Flash 生成回應 |
| **Gemini 2.0 Flash** | Chat Model | Google Gemini API 模型（temperature: 0.7） |
| **Chat Memory** | Window Buffer | 記住最近 10 輪對話（按 user-{pk} 分隔） |
| **Format Response** | Code | 組裝 `{reply, mode, ok}` 回應格式 |
| **Respond to Webhook** | Response | 回傳 JSON 給 Django |

---

## 五種對話模式

| 模式 | 功能 | 系統提示詞重點 |
|---|---|---|
| `general` | 一般問答 | 平台介紹、引導切換其他模式 |
| `customer_service` | 客服對話 | 帳號/專案/上傳/報告的操作問答 |
| `web_guide` | 網頁引導 | 步驟式教學各功能頁面操作 |
| `security_knowledge` | 資安知識 | 白話文 + 生活比喻解釋攻擊手法 |
| `explain_result` | 結果解讀 | 讀取 Session Context 用白話文翻譯 |

---

## 故障排除

### 問題：AI Chat 顯示「無法連線至 AI 服務」

1. 確認 n8n 正在運行：`curl http://localhost:5678/healthz`
2. 確認 Workflow 已啟用（Active 狀態）
3. 確認 Django 的 `N8N_WEBHOOK_URL` 設定正確
4. 查看 n8n 的 Execution Log 是否有錯誤

### 問題：AI 回應為空或錯誤

1. 打開 n8n → Executions → 查看最近一次執行
2. 檢查 **Build Prompt** 節點的 output 是否正確
3. 檢查 **Gemini 2.0 Flash** 節點是否有 API 錯誤
4. 確認 Gemini API Key 額度未耗盡

### 問題：對話沒有記憶

1. 確認 **Chat Memory** 節點的 Session Key 表達式正確
2. 確認每次請求的 `chat_session_id` 格式一致（`user-{pk}`）

---

## 安全注意事項

- ⛔ 絕對不要在 Workflow 中硬編碼 API Key
- ⛔ 不要將 Webhook 設為公開可存取（僅限 localhost）
- ⛔ 不要在 System Prompt 中包含敏感的系統架構資訊
- ✅ Django API 已實作 IDOR 防護，確保用戶只能看到自己的資料
- ✅ 對話記憶按使用者 ID 隔離，不會跨用戶洩漏
