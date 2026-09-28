# ============================================================
# core/ai_agent.py
# Gemini AI Agent — 直接呼叫 Google Gemini API（不需 n8n / Docker）
#
# 功能：
#   1. 依對話模式動態組裝 System Prompt
#   2. 從 ai_knowledge_base.md 檢索相關知識段落注入 prompt
#   3. 接收 _build_ai_context() 的結構化 context
#   4. 行程內 per-user 對話記憶（最近 10 輪）
#   5. 完善的錯誤處理
#
# 使用方式：
#   from ai_agent import GeminiAgent
#   agent = GeminiAgent(api_key="YOUR_KEY")
#   reply = agent.chat(message, mode, context, user_id)
# ============================================================

from __future__ import annotations

import os
import re
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────
# 知識庫載入與檢索
# ──────────────────────────────────────────────────────────
_KNOWLEDGE_BASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ai_knowledge_base.md"
)


class KnowledgeBase:
    """從 ai_knowledge_base.md 載入段落並依關鍵字比對檢索。"""

    def __init__(self, path: str = _KNOWLEDGE_BASE_PATH):
        self.sections: List[Tuple[str, List[str], str]] = []  # (title, tags, content)
        self._load(path)

    def _load(self, path: str):
        """解析 markdown，依 ## 標題切分段落，擷取 [tags: ...] 行。"""
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        # 依 ## 切分
        sections_raw = re.split(r"\n(?=## )", raw)
        for sec in sections_raw:
            sec = sec.strip()
            if not sec.startswith("## "):
                continue
            lines = sec.split("\n")
            title = lines[0].lstrip("# ").strip()

            # 擷取 tags
            tags: List[str] = []
            content_lines: List[str] = []
            for line in lines[1:]:
                tag_match = re.match(r"\[tags?:\s*(.+?)\]", line.strip())
                if tag_match:
                    tags = [t.strip().lower() for t in tag_match.group(1).split(",")]
                else:
                    content_lines.append(line)

            content = "\n".join(content_lines).strip()
            if content:
                self.sections.append((title, tags, content))

    def search(self, query: str, max_results: int = 3) -> List[Tuple[str, str]]:
        """依查詢字串的關鍵字比對，回傳最相關的段落 (title, content)。"""
        query_lower = query.lower()
        scored: List[Tuple[int, str, str]] = []

        for title, tags, content in self.sections:
            score = 0
            # 標題比對
            if any(word in title.lower() for word in query_lower.split() if len(word) > 1):
                score += 3
            # 標籤比對
            for tag in tags:
                if tag in query_lower:
                    score += 5
                elif any(word in tag for word in query_lower.split() if len(word) > 1):
                    score += 2
            # 內容比對
            for word in query_lower.split():
                if len(word) > 1 and word in content.lower():
                    score += 1

            if score > 0:
                scored.append((score, title, content))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(title, content) for _, title, content in scored[:max_results]]


# ──────────────────────────────────────────────────────────
# 各模式的 System Prompt
# ──────────────────────────────────────────────────────────
_BASE_SYSTEM_PROMPT = """你是「小網」，VNAAP（視覺化網路攻擊自動化分析平台）的 AI 資安助手。

基本規則：
- 一律使用繁體中文回答
- 用白話文、口語化的方式說明，避免過多專業術語（必要時附上簡單解釋）
- 回答要有條理，適當使用項目符號或編號
- 語氣友善、專業但不生硬
- 如果不確定答案，誠實說「這部分我不太確定，建議您...」
- 不要編造不存在的功能或數據
"""

_MODE_PROMPTS: Dict[str, str] = {
    "general": _BASE_SYSTEM_PROMPT + """
你目前處於「一般對話」模式。
可以回答關於平台功能、資安知識、操作方式等各種問題。
如果使用者問的問題比較適合其他模式（如白話文解讀結果），可以建議他切換模式。
""",

    "customer_service": _BASE_SYSTEM_PROMPT + """
你目前處於「客服對話」模式。
主要協助使用者解決平台操作問題，包括：
- 帳號與登入問題
- 專案管理（建立、邀請成員、權限）
- 檔案上傳與分析流程
- 報告匯出
- 任何使用上遇到的困難

回答風格：像一位耐心的客服人員，一步一步引導使用者解決問題。
""",

    "web_guide": _BASE_SYSTEM_PROMPT + """
你目前處於「網頁使用引導」模式。
手把手教使用者如何操作 VNAAP 平台的各項功能。

回答風格：
- 使用「Step 1 → Step 2 → ...」的步驟式說明
- 明確指出要點擊哪個按鈕、在哪個頁面
- 如果有注意事項，用「⚠ 提醒」標示
""",

    "security_knowledge": _BASE_SYSTEM_PROMPT + """
你目前處於「資安知識」模式。
用白話文向使用者科普網路安全知識。

回答風格：
- 把專業概念翻譯成日常生活的比喻（例如：把防火牆比喻成大門的門鎖）
- 解釋「是什麼→為什麼危險→怎麼防禦」的三段式結構
- 適合完全沒有資安背景的人閱讀
""",

    "explain_result": _BASE_SYSTEM_PROMPT + """
你目前處於「白話文解讀分析結果」模式。
使用者會提供一次分析 Session 的資料（告警、CNN 偵測結果等），
你的任務是把這些技術數據翻譯成一般人看得懂的說明。

回答風格：
- 先給出一句話總結（例如：「這次分析發現了一些可疑的網路活動」）
- 逐項解釋每個告警代表什麼意思
- 用嚴重程度分類（🔴 高風險 / 🟡 中風險 / 🟢 低風險）
- 最後給出具體的建議行動
- 如果沒有提供 Session 資料，提醒使用者先在右側選擇一個 Session
""",
}


# ──────────────────────────────────────────────────────────
# Context 格式化
# ──────────────────────────────────────────────────────────
def _format_context(context: Optional[Dict]) -> str:
    """將 _build_ai_context() 產生的 dict 格式化為易讀的文字。"""
    if not context:
        return ""

    parts: List[str] = []

    session = context.get("session")
    if session:
        parts.append(
            f"【分析 Session #{session['id']}】\n"
            f"  名稱：{session.get('label', 'N/A')}\n"
            f"  狀態：{session.get('status', 'N/A')}\n"
            f"  封包數：{session.get('packet_count', 0):,}\n"
            f"  告警數：{session.get('alert_count', 0)}"
        )

    alerts = context.get("alerts")
    if alerts:
        parts.append("【告警詳情】")
        for i, a in enumerate(alerts, 1):
            parts.append(
                f"  {i}. 攻擊類型：{a.get('attack_type', 'N/A')}\n"
                f"     嚴重等級：{a.get('severity', 'N/A')}\n"
                f"     來源 IP：{a.get('src_ip', 'N/A')} → 目的 IP：{a.get('dst_ip', 'N/A')}\n"
                f"     建議：{a.get('suggestion', 'N/A')}"
            )

    cnn = context.get("cnn_result")
    if cnn:
        parts.append(
            f"【CNN 異常偵測結果】\n"
            f"  判定閾值：{cnn.get('threshold', 0)}\n"
            f"  正常封包：{cnn.get('normal_count', 0):,} 個\n"
            f"  異常封包：{cnn.get('anomaly_count', 0):,} 個\n"
            f"  偵測率：{cnn.get('detection_rate', 0)}%\n"
            f"  正常封包平均重建誤差：{cnn.get('avg_normal_error', 0)}\n"
            f"  異常封包平均重建誤差：{cnn.get('avg_attack_error', 0)}"
        )

    gradcam_count = context.get("gradcam_anomaly_image_count")
    if gradcam_count is not None:
        parts.append(f"【Grad-CAM】已產生 {gradcam_count} 張異常封包熱力圖")

    alerts_summary = context.get("alerts_summary")
    if alerts_summary:
        parts.append(f"【告警類型摘要】{', '.join(alerts_summary)}")

    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────
# Gemini Agent
# ──────────────────────────────────────────────────────────
class GeminiAgent:
    """直接呼叫 Google Gemini API 的 AI Agent。

    Args:
        api_key: Google Gemini API Key
        model_name: Gemini 模型名稱（預設 gemini-2.0-flash，免費方案）
        max_history: 每位使用者保留的最大對話歷史輪數
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        max_history: int = 10,
    ):
        if not api_key:
            raise ValueError("Gemini API Key 未設定")

        import google.generativeai as genai
        genai.configure(api_key=api_key)

        self._genai = genai
        self._model_name = model_name
        self._max_history = max_history
        self._knowledge = KnowledgeBase()

        # per-user 對話記憶（行程內，重啟後清空）
        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self._lock = threading.Lock()

    def chat(
        self,
        message: str,
        mode: str = "general",
        context: Optional[Dict] = None,
        user_id: str = "anonymous",
    ) -> str:
        """送出一則訊息並取得 AI 回覆。

        Args:
            message: 使用者輸入的文字
            mode: 對話模式（general / customer_service / web_guide /
                  security_knowledge / explain_result）
            context: _build_ai_context() 產生的結構化資料
            user_id: 使用者識別（用於對話記憶隔離）

        Returns:
            AI 回覆的文字
        """
        # 1. 選取 system prompt
        system_prompt = _MODE_PROMPTS.get(mode, _MODE_PROMPTS["general"])

        # 2. 檢索知識庫
        kb_results = self._knowledge.search(message, max_results=2)
        if kb_results:
            kb_text = "\n\n---\n\n".join(
                f"【參考知識：{title}】\n{content}" for title, content in kb_results
            )
            system_prompt += f"\n\n以下是與使用者問題相關的知識庫內容，請參考回答：\n\n{kb_text}"

        # 3. 注入 Session context
        context_text = _format_context(context)
        if context_text:
            system_prompt += f"\n\n以下是使用者目前選擇的分析 Session 資料：\n\n{context_text}"

        # 4. 建構對話歷史
        with self._lock:
            history_list = list(self._history[user_id])

        # 5. 組裝 Gemini contents
        contents = []
        for role, text in history_list:
            contents.append({"role": role, "parts": [text]})
        contents.append({"role": "user", "parts": [message]})

        # 6. 呼叫 Gemini API
        try:
            model = self._genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=system_prompt,
            )
            response = model.generate_content(contents)
            reply = response.text.strip()
        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg.upper() or "PERMISSION" in error_msg.upper():
                return "⚠ Gemini API Key 無效或權限不足，請確認 settings.py 中的 GEMINI_API_KEY 設定。"
            if "QUOTA" in error_msg.upper() or "429" in error_msg:
                return "⚠ Gemini API 呼叫次數已達上限，請稍候再試（免費方案限制：15 次/分鐘）。"
            if "SAFETY" in error_msg.upper():
                return "⚠ 由於安全過濾機制，AI 無法回覆此問題。請換個方式提問。"
            return f"⚠ AI 服務發生錯誤：{error_msg}"

        # 7. 更新對話記憶
        with self._lock:
            self._history[user_id].append(("user", message))
            self._history[user_id].append(("model", reply))

        return reply

    def clear_history(self, user_id: str):
        """清除指定使用者的對話記憶。"""
        with self._lock:
            self._history.pop(user_id, None)


# ──────────────────────────────────────────────────────────
# 全域單例（Django 行程內共用）
# ──────────────────────────────────────────────────────────
_AGENT_INSTANCE: Optional[GeminiAgent] = None
_AGENT_LOCK = threading.Lock()


def get_agent(api_key: str, model_name: str = "gemini-2.0-flash") -> GeminiAgent:
    """取得或建立全域 GeminiAgent 單例。

    若 api_key 改變（使用者在 settings.py 更新了 key），會自動重建。
    """
    global _AGENT_INSTANCE
    with _AGENT_LOCK:
        if _AGENT_INSTANCE is None or getattr(_AGENT_INSTANCE, "_api_key", None) != api_key:
            _AGENT_INSTANCE = GeminiAgent(api_key=api_key, model_name=model_name)
            _AGENT_INSTANCE._api_key = api_key  # type: ignore[attr-defined]
        return _AGENT_INSTANCE
