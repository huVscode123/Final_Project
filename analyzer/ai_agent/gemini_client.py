# -*- coding: utf-8 -*-
"""
analyzer/ai_agent/gemini_client.py

Gemini API 的最小化直接整合（不透過 n8n、不需要 Docker）。

只依賴專案既有的 `requests` 套件（見 requirements.txt），以 REST 方式
直接呼叫 Google Gemini API：

    POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    Header: x-goog-api-key: <GEMINI_API_KEY>

API Key 設定方式（擇一，詳見同目錄的 AI_AGENT_SETUP.md）：
  1. 環境變數： export GEMINI_API_KEY="AIza...."
  2. 直接寫在 network_platform/settings.py 的 GEMINI_API_KEY 變數裡
     （僅建議個人 / 教學用途，且該檔案不要推上公開的 git repo）。

Gemini 的可用模型名稱會隨時間調整，若不確定目前這把金鑰可以用哪個
模型，執行：
    python manage.py list_gemini_models
會直接向 API 查詢並列出來，比查文件更準確。
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# 沒有在 settings.py 覆寫時使用的預設值。
# Gemini 的模型型號會持續更新／汰換，這裡只是一個「先能動」的合理預設，
# 正式使用前請務必執行 `python manage.py list_gemini_models` 確認。
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.4


class GeminiConfigError(RuntimeError):
    """尚未設定 GEMINI_API_KEY，或設定內容明顯無效時拋出。"""


class GeminiAPIError(RuntimeError):
    """Gemini API 回傳非預期狀態碼，或回應格式不符預期時拋出。"""


def _get_api_key() -> str:
    key = (getattr(settings, "GEMINI_API_KEY", "") or "").strip()
    if not key:
        raise GeminiConfigError(
            "尚未設定 GEMINI_API_KEY。請於環境變數，或 "
            "network_platform/settings.py 中設定你自己的 Gemini API Key "
            "（詳見 AI_AGENT_SETUP.md 的設定說明）。"
        )
    return key


def _get_model() -> str:
    return getattr(settings, "GEMINI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


def _get_timeout() -> int:
    return int(getattr(settings, "GEMINI_API_TIMEOUT", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT)


def _get_max_output_tokens() -> int:
    return int(
        getattr(settings, "GEMINI_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
        or DEFAULT_MAX_OUTPUT_TOKENS
    )


def list_models(api_key: Optional[str] = None) -> List[dict]:
    """列出目前這把 API Key 實際可用的模型（用來確認 GEMINI_MODEL 該填什麼）。

    使用方式：
        python manage.py list_gemini_models
    或在程式中：
        from analyzer.ai_agent.gemini_client import list_models
        for m in list_models():
            print(m["name"])
    """
    key = api_key or _get_api_key()
    url = f"{GEMINI_API_BASE}/models"
    resp = requests.get(url, headers={"x-goog-api-key": key}, timeout=_get_timeout())
    resp.raise_for_status()
    data = resp.json()
    return data.get("models", [])


def generate(
    system_instruction: str,
    user_message: str,
    *,
    model: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: Optional[int] = None,
) -> str:
    """呼叫 Gemini generateContent，回傳純文字回覆。

    Args:
        system_instruction: 系統提示詞（人設 + 模式規則 + 知識片段 + 對話上下文
                             都在呼叫端組好，這裡只負責送出去）
        user_message      : 使用者這一輪輸入的訊息
        model             : 覆寫預設模型（否則讀取 settings.GEMINI_MODEL）
        temperature       : 生成溫度，越高越發散
        max_output_tokens : 最大輸出 token 數（否則讀取 settings.GEMINI_MAX_OUTPUT_TOKENS）

    Returns:
        Gemini 回覆的純文字內容。

    Raises:
        GeminiConfigError: 未設定 API Key。
        GeminiAPIError   : API 回傳非 2xx，或回應格式不符預期（例如被安全政策擋下）。
        requests.exceptions.Timeout / ConnectionError:
            網路逾時或連線失敗，交由呼叫端（api/views.py）處理成給使用者的友善訊息。
    """
    api_key = _get_api_key()
    model_name = model or _get_model()
    url = f"{GEMINI_API_BASE}/models/{model_name}:generateContent"

    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [
            {"role": "user", "parts": [{"text": user_message}]},
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens or _get_max_output_tokens(),
        },
    }

    resp = requests.post(
        url,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=_get_timeout(),
    )

    # 針對常見錯誤狀態碼給出可直接行動的中文說明，而不是讓使用者面對原始 HTTP 錯誤。
    if resp.status_code == 400:
        raise GeminiAPIError(
            f"Gemini API 回傳 400（通常代表模型名稱錯誤，或請求格式有問題）："
            f"{resp.text[:500]}"
        )
    if resp.status_code == 403:
        raise GeminiAPIError(
            "Gemini API 回傳 403，請確認 API Key 是否正確、是否已在 Google AI "
            "Studio / Google Cloud 啟用 Generative Language API 的存取權限。"
        )
    if resp.status_code == 404:
        raise GeminiAPIError(
            f"Gemini API 回傳 404，模型「{model_name}」可能不存在，或你的帳號無權"
            f"使用。可執行 `python manage.py list_gemini_models` 查詢實際可用的模型。"
        )
    if resp.status_code == 429:
        raise GeminiAPIError("Gemini API 回傳 429（已達速率或配額上限），請稍後再試。")
    resp.raise_for_status()

    data = resp.json()
    try:
        candidates = data["candidates"]
        if not candidates:
            raise KeyError("candidates 為空")
        parts = candidates[0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
        if not text:
            raise KeyError("parts 內沒有文字內容")
        return text.strip()
    except (KeyError, IndexError, TypeError) as e:
        finish_reason = None
        try:
            finish_reason = data["candidates"][0].get("finishReason")
        except Exception:
            pass
        logger.error("Gemini 回應格式不符預期: %s / finishReason=%s", data, finish_reason)
        if finish_reason == "SAFETY":
            raise GeminiAPIError("Gemini 因安全政策拒絕回答此請求，請換個方式提問。") from e
        raise GeminiAPIError(f"無法解析 Gemini 回應內容（finishReason={finish_reason}）。") from e
