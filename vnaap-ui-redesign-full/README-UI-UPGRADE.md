# VNAAP UI/UX 改版說明（完整版）

這次交付的是**整套系統**的模板與樣式，不只是首頁與登入頁——共 21 個模板 +
1 份設計系統 CSS + 1 份共用 JS。可以直接整包覆蓋到 Django 專案的
`templates/` 與 `static/` 目錄。

## 目錄結構

```
static/css/main.css                     全站設計系統（深/淺主題 token + 所有元件樣式）
static/js/main.js                       主題引擎 + 共用互動邏輯（含圖表配色同步）

templates/base/base.html                共用版面（Header／Sidebar／主題切換鈕）
templates/accounts/login.html           登入頁（左右分割版型）
templates/accounts/register.html        註冊頁
templates/accounts/profile.html         個人設定
templates/accounts/admin_panel.html     管理員面板
templates/accounts/activity_logs.html   操作稽核日誌

templates/analyzer/dashboard.html       儀表板
templates/analyzer/upload.html          上傳 PCAP
templates/analyzer/sessions.html        分析任務列表
templates/analyzer/session_detail.html  任務詳情（含 CNN／Grad-CAM／報告 Tab）
templates/analyzer/gradcam_gallery.html Grad-CAM 圖庫
templates/analyzer/heatmap_analysis.html 熱力圖深度分析
templates/analyzer/simulation.html      異常模擬檢測
templates/analyzer/ablation.html        消融實驗分析
templates/analyzer/live.html            即時監控
templates/analyzer/ai_chat.html         AI 助手對話

templates/projects/project_list.html    專案列表
templates/projects/project_detail.html  專案詳情
templates/projects/project_form.html    新增／編輯專案
templates/projects/packet_upload.html   專案附件上傳

templates/reports/report_list.html      報告列表
```

> 註：原始碼中的 `analyzer/session_list.html` 沒有被任何 view 使用（實際列表頁
> 是 `sessions.html`），屬於未串接的死檔案，因此本次未納入改版範圍。

直接覆蓋這些檔案即可，**不需要修改任何 `.py` 檔案**——所有 view / URL /
context 變數名稱都維持原樣，只換了樣式與模板結構。

## 設計方式：一套 Design Token，兩份主題

`main.css` 最上方定義共用的間距、圓角、字體、動畫時間等 token，接著用
`html[data-theme="dark"]` 與 `html[data-theme="light"]` 各自定義一組色彩變數
（背景、文字、邊框、品牌綠、五種語意色）。全站元件一律只使用 `var(--xxx)`，
不寫死色碼，因此新增頁面或調整配色都只需要改一個地方。

### 主題切換怎麼運作
- 右上角有一顆膠囊型切換鈕，點擊呼叫 `main.js` 的 `vnaapToggleTheme()`，
  把 `<html data-theme="...">` 換掉並寫入 `localStorage['vnaap-theme']`。
- 每個頁面 `<head>` 最上方有一段內嵌 script，在畫面繪製前就讀取
  `localStorage` 套用主題，避免「先深後淺」的畫面閃爍（FOUC）。
- 若使用者從未手動選過主題，以瀏覽器的 `prefers-color-scheme` 作為預設值。

### Chart.js 圖表的主題同步
`dashboard.html`、`admin_panel.html`、`session_detail.html` 建立圖表時，
座標軸格線／刻度文字一律透過 `vnaapThemeColors()`（`main.js` 提供）取得目前
主題對應的顏色，而不是寫死深色系的色碼；圓餅圖／雷達圖的分隔邊框顏色也改為
`_vc.cardBg`，讓它在深/淺卡片背景下都能正確融合。此外 `main.js` 內建
`vnaapRecolorCharts()`，會在**主題切換當下**與**頁面載入完成後**都執行一次，
確保無論使用者是切換主題、或直接以淺色主題重新整理頁面，圖表都不會出現
「淺色背景配深色格線」看不清楚的狀況。長條圖／環圈圖本身的資料顏色
（半透明的綠、紅、藍…）維持原設計，因為飽和度足夠，兩種主題下都清楚可辨。

## 本次修正的實際問題（淺色模式下曾經「壞掉」的地方）

在盤點所有頁面後，找到幾處**寫死深色色碼、完全沒有理由跟著主題切換**的地方，
這些屬於真正的缺陷而非設計選擇，已全部修正為 `var(--bg-elevated)` 等變數：

| 頁面 | 問題元件 | 修正前 | 修正後 |
|---|---|---|---|
| `ai_chat.html` | AI 對話泡泡、輸入框 | `background:#161b22` / `#0d1117` / `#010409` | `var(--bg-elevated)` / `var(--bg-input)` |
| `session_detail.html` | Grad-CAM 卡片、CNN 誤差圖表容器 | `background:#0d1117` | `var(--bg-elevated)` |
| `live.html` | 任務迷你統計方塊 `.stat-mini` | `background:#0a0f1a` | `var(--bg-elevated)` |
| `ablation.html` | 實驗結果卡片 `.result-card` | `background:#161b22;border:#30363d` | `var(--bg-elevated)` / `var(--border2)` |
| `dashboard.html`／`admin_panel.html` | KPI／功能圖示的 `stroke="#22c55e"` 等寫死色碼 | 固定十六進位色碼 | `stroke="var(--green)"` 等 |

修正前，若使用者切到淺色主題，上述元件會變成「白色頁面上突兀的深色方塊」，
看起來像沒套用到主題；修正後這些元件會跟著主題自然轉換背景與邊框深淺。

## 刻意保留、不隨主題切換的地方（設計選擇，非缺陷）

- **`ablation.html` 的終端機日誌區塊**（`.terminal-container`）維持固定深色，
  如同程式碼／終端機主控台，這是常見且合理的設計慣例（許多 IDE、CI 介面的
  日誌區塊也不會跟著淺色主題變白）。
- **登入／註冊頁左側品牌面板**固定使用深色漸層，作為品牌識別區塊，
  不隨系統主題切換（如同多數 SaaS 產品的行銷面板），只有右側表單面板跟隨
  系統主題。
- 各頁面內少數半透明色（如 `rgba(34,197,94,.2)` 邊框光暈、嚴重度徽章的
  `rgba(239,68,68,.15)` 底色）刻意保留，因為半透明疊加色在深/淺兩種背景上
  都清晰可辨，不需要為了理論一致性而增加不必要的改動風險。

## 登入頁設計說明

採用業界常見、但執行得乾淨的「左右分割」版型，而非灑滿動態粒子的駭客風格：

- **左側品牌面板**：固定深色，以極簡網格底紋 + 一組低調的節點連線 SVG
  動畫代表「封包在網路中流動」。動畫只有：連線上緩慢流動的虛線、三個節點的
  呼吸光暈、一顆沿路徑移動的小光點，速度刻意放慢，不追求視覺衝擊。文案聚焦
  平台實際能力（規則式＋CNN 雙軌偵測、Grad-CAM 可解釋性、報告匯出）。
- **右側表單面板**：跟隨系統主題，維持原有欄位與後端邏輯（`username`/
  `password`、CSRF、Django messages 與表單錯誤顯示皆保留）。
- 手機寬度（< 920px）隱藏品牌面板，只顯示表單，確保小螢幕可用性。

`register.html` 沿用相同版面骨架與角色選擇卡片，讓註冊流程與登入頁視覺一致。

## 檔案完整性檢查

- `main.css`：花括號配對已驗證平衡（250 組）。
- `main.js`：已用 Node.js 驗證語法正確、花括號／括號配對平衡（67 組）。
- 21 個模板均已確認可對應原始 view 所需的 context 變數與 URL name，
  未變更任何後端邏輯。
