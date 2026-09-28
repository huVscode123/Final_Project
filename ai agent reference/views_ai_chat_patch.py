# -*- coding: utf-8 -*-
"""
api/views_ai_chat_patch.py

★ 這不是一個要被 import 的模組，而是「補丁片段」★

用途：取代 api/views.py 裡原本呼叫 n8n webhook 的 AI 客服區塊，改成直接
呼叫 Gemini（不需要 Docker、不需要另外跑 n8n 服務）。

套用方式：
  1. 打開 api/views.py。
  2. 找到並「整段刪除」以下三個東西（在原始檔案中彼此相鄰）：
       - AI_CHAT_MODES = {...}
       - def _build_ai_context(request, mode, session_id): ...
       - class AIChatAPI(APIView): ...
  3. 把本檔案「AI_CHAT_MODES 到 AIChatAPI 結尾」整段程式碼貼到原本的位置。
  4. 在檔案最上方的 import 區塊，加入：
       from analyzer.ai_agent.agent import generate_reply
       from analyzer.ai_agent.gemini_client import GeminiConfigError, GeminiAPIError
     以及本檔案用到的 `import requests` 已不再需要（原本只有 n8n 那段
     用到），若 api/views.py 其他地方沒有再使用 requests，可以順手移除
     檔案最上方的 `import requests`（保留也不會出錯，只是多一個沒用到
     的匯入）。
  5. `_build_ai_context()` 的邏輯完全沒有變（沿用原本的 IDOR 防護、
     explain_result 模式的 session 摘要組裝方式），這裡照抄一份只是為
     了讓這個補丁檔案可以整段貼上、不用再回頭比對差異。

同場說明：settings.py 裡原本的 N8N_WEBHOOK_URL / AI_CHAT_TIMEOUT 這兩個
設定，套用本補丁後就不會再被使用，可以留著（無副作用）也可以刪除；
真正要新增的設定請見 AI_AGENT_SETUP.md。
"""

# ─────────────────────────────────────────────
# AI Chat API（改為直接呼叫 Gemini，不再透過 n8n）
# ─────────────────────────────────────────────
AI_CHAT_MODES = {
    'customer_service':   '客服對話',
    'web_guide':           '網頁使用引導',
    'security_knowledge':  '資安知識',
    'explain_result':      '白話文解讀偵測結果',
    'general':              '一般對話',
}


def _build_ai_context(request, mode, session_id):
    """
    依據 mode 組出要送給 AI 的結構化 context。

    回傳 (context: dict, error_response: Response|None)。
    error_response 不為 None 時，view 必須直接把它回傳給前端，
    不可以繼續把任何資料送去 Gemini（避免 IDOR）。
    """
    context = {'mode': mode}
    if not session_id:
        return context, None

    session, err = get_session_or_403(request, session_id)
    if err:
        # session 不存在或使用者無權限時，直接擋下，不把任何 session 資料
        # 送出去，避免把別人 session 的告警／CNN 結果摘要洩露給不相關的使用者。
        return context, err

    context['session'] = {
        'id':            session.pk,
        'label':         session.label,
        'status':        session.task_status,
        'packet_count':  session.packet_count,
        'alert_count':   session.alert_count,
    }

    if mode == 'explain_result':
        # 白話文解讀模式：帶完整一點的資料，讓 AI 有足夠依據解釋
        alerts = session.alerts.all()[:8]
        context['alerts'] = [{
            'attack_type': a.attack_type,
            'severity':    a.severity,
            'src_ip':      a.src_ip,
            'dst_ip':      a.dst_ip,
            'suggestion':  a.suggestion,
        } for a in alerts]

        cnn = getattr(session, 'cnn_result', None)
        if cnn:
            context['cnn_result'] = {
                'threshold':        cnn.threshold,
                'normal_count':     cnn.normal_count,
                'anomaly_count':    cnn.anomaly_count,
                'detection_rate':   round(cnn.detection_rate * 100, 1),
                'avg_normal_error': cnn.avg_normal_error,
                'avg_attack_error': cnn.avg_attack_error,
            }

        context['gradcam_anomaly_image_count'] = (
            session.gradcam_images.filter(is_anomaly=True).count()
        )
    else:
        # 其他模式只帶精簡摘要，避免把不必要的細節（IP、建議內容等）
        # 傳給跟這次任務無關的對話模式
        alerts = session.alerts.all()[:5]
        context['alerts_summary'] = [a.attack_type for a in alerts]

    return context, None


class AIChatAPI(APIView):
    """
    POST /api/v1/ai/chat/

    Request body:
        {
            "message":    "使用者輸入的文字",
            "mode":       "customer_service | web_guide | security_knowledge | explain_result | general",
            "session_id": 123   // 選填，explain_result 模式建議一定要帶
        }

    Response:
        { "reply": "AI 回覆內容", "mode": "..." }

    ── 與舊版的差異 ──
    舊版把 message + context 打包成 JSON，POST 給 n8n 的 webhook，
    再由 n8n 串接 Gemini（因此需要另外跑一個 n8n 服務，常見做法是用
    Docker 起一個容器）。

    這一版把「組提示詞 → 呼叫 Gemini → 回傳文字」整段邏輯搬進
    analyzer/ai_agent/ 套件，Django 這裡直接呼叫 Python 函式
    `generate_reply()`，不再需要 n8n、也不需要 Docker；唯一的外部相依
    是 HTTPS 直連 Google 的 Gemini API。
    """

    def post(self, request):
        message    = request.data.get('message', '').strip()
        mode       = request.data.get('mode', 'general')
        session_id = request.data.get('session_id')

        if not message:
            return Response({'error': '請輸入訊息'}, status=400)
        if mode not in AI_CHAT_MODES:
            mode = 'general'

        context, err = _build_ai_context(request, mode, session_id)
        if err:
            return err

        try:
            reply = generate_reply(
                message=message,
                mode=mode,
                context=context,
                username=request.user.username,
            )
        except GeminiConfigError as e:
            # 尚未設定 API Key：用 200 + 友善訊息回覆，前端聊天視窗會直接
            # 顯示這句話，而不是彈出一般錯誤，方便使用者知道要去哪裡設定。
            return Response({'reply': f'⚠ AI 服務尚未設定完成：{e}', 'mode': mode})
        except GeminiAPIError as e:
            return Response({'reply': f'⚠ AI 服務發生錯誤：{e}', 'mode': mode})
        except Exception as e:
            # 連線逾時／DNS 失敗等網路層例外，統一在這裡攔截，訊息維持
            # 與舊版一致的語氣，方便前端既有的顯示邏輯直接沿用。
            return Response({'reply': 'AI 服務暫時無法連線，請確認網路狀況或稍後再試。'})

        # 稽核日誌（與舊版行為相同）
        try:
            UserActivityLog.objects.create(
                user=request.user, action='ai_chat',
                detail=f'[{AI_CHAT_MODES[mode]}] {message[:80]}',
                ip_address=request.META.get('REMOTE_ADDR'))
        except Exception:
            pass

        return Response({'reply': reply, 'mode': mode})
