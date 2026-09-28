# -*- coding: utf-8 -*-
"""
analyzer/ai_agent/agent.py

AI 客服代理人主邏輯：
  1. 依對話模式（mode）從 prompts.json 挑選系統提示詞。
  2. 依使用者輸入的關鍵字，從 knowledge_base.json 找出相關的「操作引導
     步驟」「資安知識條目」「FAQ」，組進系統提示詞──純關鍵字比對，
     不需要向量資料庫、不需要額外的本機服務，符合「不使用 Docker」的
     需求，同時也不會把不相關的內容硬塞給模型。
  3. 呼叫 gemini_client.generate() 取得回覆。

這裡取代原本 api/views.py::AIChatAPI 呼叫 n8n webhook 的作法：整個 AI
客服流程現在完全在 Django process 內完成，唯一的外部相依是 Google
Gemini API（HTTPS 直連公開端點，不是本機服務，因此不需要 Docker）。
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Dict, List, Optional

from . import gemini_client
from .gemini_client import GeminiAPIError, GeminiConfigError  # noqa: F401  (re-export)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROMPTS_PATH = os.path.join(_BASE_DIR, "prompts.json")
_KB_PATH = os.path.join(_BASE_DIR, "knowledge_base.json")

_lock = threading.Lock()
_cache: Dict[str, dict] = {}


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get(name: str, path: str) -> dict:
    """行程內快取 JSON 內容，避免每次請求都重讀磁碟。"""
    with _lock:
        if name not in _cache:
            _cache[name] = _load_json(path)
        return _cache[name]


def _prompts() -> dict:
    return _get("prompts", _PROMPTS_PATH)


def _knowledge_base() -> dict:
    return _get("kb", _KB_PATH)


def reload_knowledge() -> None:
    """開發時若修改了 prompts.json / knowledge_base.json，呼叫這個函式即可
    立即生效，不需要重啟 Django（例如可以掛在一個限管理員使用的小工具上）。
    """
    with _lock:
        _cache.pop("prompts", None)
        _cache.pop("kb", None)


_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _score_entry(query_tokens: List[str], keywords: List[str]) -> int:
    kw_tokens = set()
    for kw in keywords:
        kw_tokens.update(_tokenize(kw))
    return sum(1 for t in query_tokens if t in kw_tokens)


def retrieve_snippets(message: str, mode: str, top_k: int = 2) -> List[str]:
    """從 knowledge_base.json 依關鍵字比對，挑出最相關的條目（供組進系統提示詞）。

    這是刻意簡化過的「關鍵字比對」版本，不是向量檢索：VNAAP 的知識庫
    條目數量不大，關鍵字比對已經夠用，而且完全不需要额外的 embedding
    服務或資料庫，符合這次「不使用 Docker」的限制。
    """
    kb = _knowledge_base()
    query_tokens = _tokenize(message)
    if not query_tokens:
        return []

    pools = []
    if mode == "web_guide":
        pools.append(("web_guide_steps", kb.get("web_guide_steps", [])))
    elif mode == "security_knowledge":
        pools.append(("security_topics", kb.get("security_topics", [])))
    else:
        # 一般對話 / 客服對話也常常混雜操作問題或資安問題，兩個知識庫都放入候選。
        pools.append(("web_guide_steps", kb.get("web_guide_steps", [])))
        pools.append(("security_topics", kb.get("security_topics", [])))
    pools.append(("faq", kb.get("faq", [])))

    scored = []
    for pool_name, entries in pools:
        for entry in entries:
            label = entry.get("topic") or entry.get("question") or ""
            score = _score_entry(query_tokens, entry.get("keywords", []) + [label])
            if score > 0:
                scored.append((score, pool_name, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [_format_entry(pool_name, entry) for _, pool_name, entry in scored[:top_k]]


def _format_entry(pool_name: str, entry: dict) -> str:
    if pool_name == "faq":
        return f"[FAQ] Q: {entry.get('question', '')}\nA: {entry.get('answer', '')}"
    if pool_name == "web_guide_steps":
        steps = entry.get("steps", [])
        steps_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(steps))
        url = entry.get("url", "")
        return f"[操作引導：{entry.get('topic', '')}]\n{steps_text}\n(相關頁面: {url})"
    if pool_name == "security_topics":
        tips = entry.get("tips", [])
        tips_text = "\n".join(f"  - {t}" for t in tips)
        return (
            f"[資安知識：{entry.get('topic', '')}]\n{entry.get('summary', '')}\n"
            f"防禦建議：\n{tips_text}"
        )
    return json.dumps(entry, ensure_ascii=False)


def _mode_prompt(mode: str) -> str:
    modes = _prompts().get("modes", {})
    cfg = modes.get(mode) or modes.get("general", {})
    return cfg.get("system_prompt", "")


def _compose_system_instruction(
    mode: str, message: str, context: Optional[dict], username: Optional[str]
) -> str:
    parts = [_prompts().get("base_persona", "").strip()]

    mode_prompt = _mode_prompt(mode).strip()
    if mode_prompt:
        parts.append(mode_prompt)

    snippets = retrieve_snippets(message, mode)
    if snippets:
        parts.append(
            "以下是可能相關的內部知識，請優先參考（若與問題無關可忽略，"
            "請用自己的話重新組織，不要逐字照抄）：\n" + "\n\n".join(snippets)
        )

    if context and context.get("session"):
        parts.append(
            "以下是目前選定的分析任務（Session）資料，僅在使用者的問題與此相關時"
            "才使用，不要在無關的問題中提及：\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
        )

    if username:
        parts.append(f"目前與你對話的使用者帳號是「{username}」，可視情況稱呼，但不要每句都提。")

    parts.append(
        "請一律使用繁體中文回覆，語氣友善、清楚、避免不必要的專業術語堆疊；"
        "遇到你不確定、或系統沒有提供的資訊，請誠實告知使用者你不確定，"
        "絕對不要編造平台不存在的功能、頁面或數據。"
    )

    return "\n\n---\n\n".join(p for p in parts if p)


def generate_reply(
    message: str,
    mode: str,
    context: Optional[dict] = None,
    username: Optional[str] = None,
) -> str:
    """組出系統提示詞並呼叫 Gemini，回傳純文字回覆。

    Raises:
        GeminiConfigError / GeminiAPIError（見 gemini_client.py）
        requests 相關例外（逾時、連線失敗等）
    """
    system_instruction = _compose_system_instruction(mode, message, context, username)
    return gemini_client.generate(system_instruction, message)
