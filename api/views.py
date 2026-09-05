# ============================================================
# api/views.py
# REST API：Token 認證、專案、Session、CNN、Grad-CAM、AI Chat
# ============================================================
import os
import requests
from django.conf import settings
from django.contrib.auth.models import User
from django.db.models import Q
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny, IsAdminUser
from rest_framework.authtoken.models import Token
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from accounts.models import UserActivityLog
from projects.models import Project, PacketFile
from analyzer.models import AnalysisSession, Alert, CNNResult, GradCAMImage

from .serializers import (
    UserSerializer, ProjectListSerializer, ProjectDetailSerializer,
    PacketFileSerializer, AnalysisSessionListSerializer,
    AnalysisSessionDetailSerializer, AlertSerializer,
    CNNResultSerializer, GradCAMImageSerializer, AnalysisReportSerializer,
)
from .permissions import get_session_or_403



# ─────────────────────────────────────────────────────────────
# 認證 API
# ─────────────────────────────────────────────────────────────
class LoginAPI(APIView):
    """POST /api/v1/auth/login/ → Token"""
    permission_classes = [AllowAny]

    def post(self, request):
        from django.contrib.auth import authenticate
        username = request.data.get('username', '').strip()
        password = request.data.get('password', '')
        user = authenticate(username=username, password=password)
        if not user:
            return Response({'error': '帳號或密碼錯誤'}, status=400)
        token, _ = Token.objects.get_or_create(user=user)
        # 稽核日誌
        ip = request.META.get('REMOTE_ADDR', '')
        UserActivityLog.objects.create(
            user=user, action='login',
            detail='API 登入', ip_address=ip or None)
        return Response({
            'token':    token.key,
            'user_id':  user.pk,
            'username': user.username,
            'role':     user.profile.role,
        })


class LogoutAPI(APIView):
    """POST /api/v1/auth/logout/ → 銷毀 Token"""

    def post(self, request):
        request.user.auth_token.delete()
        UserActivityLog.objects.create(
            user=request.user, action='logout', detail='API 登出')
        return Response({'message': '已登出'})


class MeAPI(APIView):
    """GET /api/v1/auth/me/ → 當前使用者資訊"""

    def get(self, request):
        return Response(UserSerializer(request.user, context={'request': request}).data)


class RegisterAPI(APIView):
    """POST /api/v1/auth/register/ → 建立帳號並回傳 Token"""
    permission_classes = [AllowAny]

    def post(self, request):
        from accounts.forms import RegisterForm
        form = RegisterForm(request.data)
        if form.is_valid():
            user = form.save()
            token, _ = Token.objects.get_or_create(user=user)
            return Response({
                'token':    token.key,
                'user_id':  user.pk,
                'username': user.username,
            }, status=201)
        return Response(form.errors, status=400)


# ─────────────────────────────────────────────────────────────
# 專案 API
# ─────────────────────────────────────────────────────────────
class ProjectListAPI(APIView):
    """GET /api/v1/projects/  POST /api/v1/projects/"""

    def get(self, request):
        qs = Project.objects.filter(
            Q(owner=request.user) | Q(members=request.user)
        ).exclude(status='deleted').distinct()
        return Response(ProjectListSerializer(qs, many=True).data)

    def post(self, request):
        if not request.user.profile.can_manage_projects:
            return Response({'error': '權限不足'}, status=403)
        serializer = ProjectListSerializer(data=request.data)
        if serializer.is_valid():
            from projects.models import ProjectMembership
            project = serializer.save(owner=request.user)
            ProjectMembership.objects.create(
                project=project, user=request.user, role='owner')
            return Response(ProjectDetailSerializer(
                project, context={'request': request}).data, status=201)
        return Response(serializer.errors, status=400)


class ProjectDetailAPI(APIView):
    """GET/PUT/DELETE /api/v1/projects/<pk>/"""

    def _get_project(self, request, pk):
        try:
            p = Project.objects.get(pk=pk)
        except Project.DoesNotExist:
            return None, Response({'error': '專案不存在'}, status=404)
        if p.owner != request.user and \
           not p.members.filter(pk=request.user.pk).exists():
            return None, Response({'error': '權限不足'}, status=403)
        return p, None

    def get(self, request, pk):
        p, err = self._get_project(request, pk)
        if err:
            return err
        return Response(ProjectDetailSerializer(p, context={'request': request}).data)

    def put(self, request, pk):
        p, err = self._get_project(request, pk)
        if err:
            return err
        if p.owner != request.user:
            return Response({'error': '只有擁有者可編輯'}, status=403)
        serializer = ProjectListSerializer(p, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    def delete(self, request, pk):
        p, err = self._get_project(request, pk)
        if err:
            return err
        if p.owner != request.user and not request.user.profile.is_admin:
            return Response({'error': '只有擁有者可刪除'}, status=403)
        p.status = 'deleted'
        p.save()
        return Response(status=204)


# ─────────────────────────────────────────────────────────────
# 封包上傳 API
# ─────────────────────────────────────────────────────────────
class PacketFileUploadAPI(APIView):
    """POST /api/v1/projects/<pk>/upload/ — 非同步封包上傳"""
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, pk):
        if not request.user.profile.can_upload_pcap:
            return Response({'error': '您無上傳權限'}, status=403)
        try:
            project = Project.objects.get(pk=pk)
        except Project.DoesNotExist:
            return Response({'error': '專案不存在'}, status=404)

        if project.owner != request.user and not project.members.filter(pk=request.user.pk).exists():
            return Response({'error': '權限不足'}, status=403)


        file = request.FILES.get('file')
        if not file:
            return Response({'error': '請提供 PCAP 檔案'}, status=400)

        pf = PacketFile.objects.create(
            project       = project,
            uploaded_by   = request.user,
            file          = file,
            original_name = file.name,
            file_size     = file.size,
            description   = request.data.get('description', ''),
            network_tag   = request.data.get('network_tag', ''),
        )

        # 非同步前處理
        try:
            from analyzer.tasks import process_packet_file
            task = process_packet_file.delay(pf.pk)
        except Exception:
            pass

        UserActivityLog.objects.create(
            user=request.user, action='upload_pcap',
            detail=f'API 上傳: {pf.original_name}',
            ip_address=request.META.get('REMOTE_ADDR'))

        return Response(PacketFileSerializer(pf, context={'request': request}).data,
                        status=201)


# ─────────────────────────────────────────────────────────────
# 分析 Session API
# ─────────────────────────────────────────────────────────────
class SessionListAPI(APIView):
    """GET /api/v1/sessions/   POST /api/v1/sessions/（建立分析任務）"""
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def get(self, request):
        user = request.user
        if user.profile.is_admin:
            qs = AnalysisSession.objects.all()
        else:
            qs = AnalysisSession.objects.filter(
                Q(created_by=user) |
                Q(project__owner=user) |
                Q(project__members=user)
            ).distinct()

        project_id = request.query_params.get('project')
        status_f   = request.query_params.get('status')
        if project_id:
            qs = qs.filter(project_id=project_id)
        if status_f:
            qs = qs.filter(task_status=status_f)

        return Response(AnalysisSessionListSerializer(
            qs.order_by('-created_at'), many=True).data)

    def post(self, request):
        if not request.user.profile.can_upload_pcap:
            return Response({'error': '您無上傳權限'}, status=403)

        pcap_file  = request.FILES.get('pcap_file')
        project_id = request.data.get('project_id')
        label      = request.data.get('label', '')

        if not pcap_file:
            return Response({'error': '請提供 pcap_file'}, status=400)

        project = None
        if project_id:
            try:
                project = Project.objects.get(pk=project_id)
            except Project.DoesNotExist:
                return Response({'error': '專案不存在'}, status=404)

            # ── [P1-7 修正] 檢查使用者是否為專案成員 ──
            if project.owner != request.user and not project.members.filter(pk=request.user.pk).exists():
                return Response({'error': '您不是此專案的成員，無法建立分析任務'}, status=403)

        session = AnalysisSession.objects.create(
            mode       = 'pcap',
            label      = label or pcap_file.name,
            pcap_file  = pcap_file,
            project    = project,
            created_by = request.user,
            task_status = 'pending',
        )

        try:
            from analyzer.tasks import run_pcap_analysis
            task = run_pcap_analysis.delay(session.pk)
            session.celery_task_id = task.id
            session.save(update_fields=['celery_task_id'])
        except Exception:
            pass

        UserActivityLog.objects.create(
            user=request.user, action='run_analysis',
            detail=f'API 建立 Session {session.pk}',
            ip_address=request.META.get('REMOTE_ADDR'))

        return Response(AnalysisSessionDetailSerializer(
            session, context={'request': request}).data, status=201)


class SessionDetailAPI(APIView):
    """GET/DELETE /api/v1/sessions/<pk>/"""

    def _get_session(self, request, pk):
        try:
            s = AnalysisSession.objects.get(pk=pk)
        except AnalysisSession.DoesNotExist:
            return None, Response({'error': 'Session 不存在'}, status=404)
        user = request.user
        if not (user.profile.is_admin or s.created_by == user or
                (s.project and (s.project.owner == user or
                                s.project.members.filter(pk=user.pk).exists()))):
            return None, Response({'error': '權限不足'}, status=403)
        return s, None

    def get(self, request, pk):
        s, err = self._get_session(request, pk)
        if err:
            return err
        return Response(AnalysisSessionDetailSerializer(
            s, context={'request': request}).data)

    def delete(self, request, pk):
        s, err = self._get_session(request, pk)
        if err:
            return err
        if s.created_by != request.user and not request.user.profile.is_admin:
            return Response({'error': '只有建立者或管理員可刪除'}, status=403)
        s.delete()
        return Response(status=204)


class SessionStatusAPI(APIView):
    """GET /api/v1/sessions/<pk>/status/ — 輪詢任務進度"""

    def get(self, request, pk):
        s, err = get_session_or_403(request, pk)
        if err:
            return err

        return Response({
            'task_status':   s.task_status,
            'packet_count':  s.packet_count,
            'alert_count':   s.alert_count,
            'error_message': s.error_message,
            'finished_at':   s.finished_at,
            'has_cnn':       hasattr(s, 'cnn_result'),
            'gradcam_count': s.gradcam_images.count(),
        })


# ─────────────────────────────────────────────────────────────
# CNN 結果 API
# ─────────────────────────────────────────────────────────────
class CNNResultAPI(APIView):
    """GET /api/v1/sessions/<pk>/cnn/"""

    def get(self, request, pk):
        s, err = get_session_or_403(request, pk)
        if err:
            return err
        try:
            result = CNNResult.objects.get(session_id=pk)
        except CNNResult.DoesNotExist:
            return Response({'error': 'CNN 結果不存在'}, status=404)
        return Response(CNNResultSerializer(result, context={'request': request}).data)


class CNNRunAPI(APIView):
    """POST /api/v1/sessions/<pk>/cnn/run/ — 重新執行 CNN 分析"""

    def post(self, request, pk):
        if not request.user.profile.can_run_ai_analysis:
            return Response({'error': '權限不足'}, status=403)
        session, err = get_session_or_403(request, pk)
        if err:
            return err
        try:
            from analyzer.tasks import run_pcap_analysis
            task = run_pcap_analysis.delay(session.pk)
            session.task_status    = 'pending'
            session.celery_task_id = task.id
            session.save(update_fields=['task_status', 'celery_task_id'])
            return Response({'task_id': task.id, 'status': 'queued'})
        except Exception as e:
            return Response({'error': str(e)}, status=500)


# ─────────────────────────────────────────────────────────────
# Grad-CAM API
# ─────────────────────────────────────────────────────────────
class GradCAMRunAPI(APIView):
    """POST /api/v1/sessions/<pk>/gradcam/run/ — 派發 Grad-CAM 任務"""

    def post(self, request, pk):
        if not request.user.profile.can_run_ai_analysis:
            return Response({'error': '您無執行 AI 分析的權限'}, status=403)
        session, err = get_session_or_403(request, pk)
        if err:
            return err

        variant      = request.data.get('variant', 'gradcam')
        try:
            max_images = int(request.data.get('max_images', 20))
        except (ValueError, TypeError):
            return Response({'error': 'max_images 必須為整數'}, status=400)
        anomaly_only = bool(request.data.get('anomaly_only', True))

        try:
            from analyzer.tasks import run_gradcam
            task = run_gradcam.delay(session.pk,
                                     max_images=max_images,
                                     variant=variant,
                                     anomaly_only=anomaly_only)
            UserActivityLog.objects.create(
                user=request.user, action='run_gradcam',
                detail=f'Session {pk} variant={variant}',
                ip_address=request.META.get('REMOTE_ADDR'))
            return Response({'task_id': task.id, 'status': 'queued',
                             'variant': variant, 'max_images': max_images})
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class GradCAMListAPI(APIView):
    """GET /api/v1/sessions/<pk>/gradcam/ — 取得 Grad-CAM 影像列表"""

    def get(self, request, pk):
        s, err = get_session_or_403(request, pk)
        if err:
            return err
        qs = GradCAMImage.objects.filter(session_id=pk)
        variant = request.query_params.get('variant')
        anomaly = request.query_params.get('anomaly_only')
        if variant:
            qs = qs.filter(variant=variant)
        if anomaly == '1':
            qs = qs.filter(is_anomaly=True)
        return Response(GradCAMImageSerializer(
            qs.order_by('packet_index'), many=True,
            context={'request': request}).data)


# ─────────────────────────────────────────────────────────────
# 告警 API
# ─────────────────────────────────────────────────────────────
class AlertListAPI(APIView):
    """GET /api/v1/sessions/<pk>/alerts/"""

    def get(self, request, pk):
        s, err = get_session_or_403(request, pk)
        if err:
            return err
        qs = Alert.objects.filter(session_id=pk).order_by('-timestamp')
        severity = request.query_params.get('severity')
        if severity:
            qs = qs.filter(severity=severity.upper())
        return Response(AlertSerializer(qs, many=True).data)


# ─────────────────────────────────────────────────────────────
# 報告匯出 API
# ─────────────────────────────────────────────────────────────
class ReportExportAPI(APIView):
    """POST /api/v1/sessions/<pk>/reports/export/"""

    def post(self, request, pk):
        if not request.user.profile.can_export_report:
            return Response({'error': '您無匯出報告的權限'}, status=403)
        session, err = get_session_or_403(request, pk)
        if err:
            return err

        fmt = request.data.get('format', 'pdf').lower()
        if fmt not in ('pdf', 'json', 'csv'):
            return Response({'error': 'format 需為 pdf / json / csv'}, status=400)

        try:
            from reports.generators import ReportGenerator
            gen    = ReportGenerator(session, request.user)
            report = gen.generate(fmt)
            UserActivityLog.objects.create(
                user=request.user, action='export_report',
                detail=f'Session {pk} format={fmt}',
                ip_address=request.META.get('REMOTE_ADDR'))
            return Response(AnalysisReportSerializer(
                report, context={'request': request}).data, status=201)
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class ReportListAPI(APIView):
    """GET /api/v1/sessions/<pk>/reports/"""

    def get(self, request, pk):
        s, err = get_session_or_403(request, pk)
        if err:
            return err
        qs = __import__('analyzer.models', fromlist=['AnalysisReport']
                         ).AnalysisReport.objects.filter(session_id=pk)
        return Response(AnalysisReportSerializer(
            qs, many=True, context={'request': request}).data)


# ─────────────────────────────────────────────
# AI Chat API（轉發至 n8n / Gemini）
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
    依據 mode 組出要送給 n8n AI Agent 的結構化 context。

    回傳 (context: dict, error_response: Response|None)。
    error_response 不為 None 時，view 必須直接把它回傳給前端，
    不可以繼續把任何資料送去 n8n（避免 IDOR）。
    """
    context = {'mode': mode}
    if not session_id:
        return context, None

    session, err = get_session_or_403(request, session_id)
    if err:
        # [安全修正] session 不存在或使用者無權限時，直接擋下，
        # 不再像舊版一樣「找不到就默默略過」，避免误把別人 session
        # 的告警/CNN結果摘要送給不相關的使用者。
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

        payload = {
            'message': message,
            'mode':    mode,
            'context': context,
            'user': {
                'username': request.user.username,
                'is_admin': request.user.profile.is_admin,
            },
            # 供 n8n 的 Window Buffer Memory 節點使用，
            # 讓「同一位使用者」在對話過程中維持記憶（而不是每輪都失憶）。
            'chat_session_id': f'user-{request.user.pk}',
        }

        webhook_url = getattr(
            settings, 'N8N_WEBHOOK_URL', 'http://localhost:5678/webhook/ai-chat')
        timeout = getattr(settings, 'AI_CHAT_TIMEOUT', 30)

        try:
            resp = requests.post(webhook_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data  = resp.json()
            reply = data.get('reply', data.get('text', ''))
        except requests.exceptions.ConnectionError:
            return Response({'reply': 'AI 服務暫時無法連線，請確認 n8n 是否已啟動。'})
        except requests.exceptions.Timeout:
            return Response({'reply': 'AI 服務回應逾時，請稍後再試一次。'})
        except Exception as e:
            return Response({'error': str(e)}, status=500)

        # 稽核日誌
        try:
            UserActivityLog.objects.create(
                user=request.user, action='ai_chat',
                detail=f'[{AI_CHAT_MODES[mode]}] {message[:80]}',
                ip_address=request.META.get('REMOTE_ADDR'))
        except Exception:
            pass

        return Response({'reply': reply, 'mode': mode})



# ─────────────────────────────────────────────────────────────
# 系統統計 API（管理員專用）
# ─────────────────────────────────────────────────────────────
class SystemStatsAPI(APIView):
    """GET /api/v1/admin/stats/"""
    permission_classes = [IsAdminUser]

    def get(self, request):
        from accounts.models import UserActivityLog
        return Response({
            'users':    User.objects.count(),
            'projects': Project.objects.exclude(status='deleted').count(),
            'sessions': AnalysisSession.objects.count(),
            'alerts':   Alert.objects.count(),
            'recent_actions': [
                {'user': l.user.username, 'action': l.action,
                 'timestamp': l.timestamp}
                for l in UserActivityLog.objects.order_by('-timestamp')[:20]
            ],
        })
