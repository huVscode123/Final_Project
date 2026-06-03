# ============================================================
# analyzer/tasks.py
# Celery 非同步任務：封包前處理、規則偵測、CNN 分析、Grad-CAM
# ============================================================
import os
import sys
import logging
from datetime import datetime

from celery import shared_task
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

# 確保 core/ 在路徑中
_CORE = str(settings.BASE_DIR / 'core')
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)


# ── 工具函式 ─────────────────────────────────────────────────
def _get_session(session_id):
    from analyzer.models import AnalysisSession
    return AnalysisSession.objects.get(pk=session_id)


def _mark_failed(session, error):
    session.task_status  = 'failed'
    session.error_message = str(error)[:2000]
    session.finished_at  = timezone.now()
    session.save(update_fields=['task_status', 'error_message', 'finished_at'])
    logger.error(f'Session {session.pk} 失敗: {error}')


# ── Task 1：封包檔案前處理（SHA-256 + 封包數量計算）──────────
@shared_task(bind=True, max_retries=2, default_retry_delay=10)
def process_packet_file(self, packet_file_pk):
    """
    非同步計算 PacketFile 的 SHA-256 與封包數量。
    任務完成後自動觸發 run_pcap_analysis。
    """
    from projects.models import PacketFile
    try:
        pf = PacketFile.objects.get(pk=packet_file_pk)
        pf.status = 'processing'
        pf.save(update_fields=['status'])

        # SHA-256
        pf.compute_sha256()

        # 封包數（使用 scapy）
        try:
            from scapy.all import rdpcap
            pkts = rdpcap(pf.file.path)
            pf.packet_count = len(pkts)
        except Exception as e:
            logger.warning(f'scapy 讀取失敗: {e}')

        pf.status = 'done'
        pf.save(update_fields=['status', 'packet_count'])
        logger.info(f'PacketFile {packet_file_pk} 處理完成，共 {pf.packet_count} 封包')

    except Exception as exc:
        logger.error(f'process_packet_file 失敗: {exc}')
        try:
            pf.status = 'failed'
            pf.save(update_fields=['status'])
        except Exception:
            pass
        raise self.retry(exc=exc)


# ── Task 2：完整 PCAP 分析（規則偵測 + CNN）─────────────────
@shared_task(bind=True, max_retries=1, default_retry_delay=5,
             time_limit=600, soft_time_limit=540)
def run_pcap_analysis(self, session_id):
    """
    主要分析流程：
      1. 規則式攻擊偵測（PcapAnalyzer）
      2. CNN Autoencoder 異常偵測
      3. 儲存結果至資料庫
    """
    from analyzer.models import AnalysisSession, Alert, CNNResult

    try:
        session = _get_session(session_id)
        session.task_status   = 'running'
        session.celery_task_id = self.request.id or ''
        session.save(update_fields=['task_status', 'celery_task_id'])

        pcap_path = str(settings.MEDIA_ROOT / session.pcap_file.name)
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f'PCAP 不存在: {pcap_path}')

        # ── Step 1：規則式偵測 ────────────────────────────────
        logger.info(f'[Session {session_id}] 執行規則式偵測...')
        try:
            from pcap_analyzer import PcapAnalyzer
            analyzer = PcapAnalyzer(pcap_path)
            analyzer.load()
            
            # [Bug 2 修正] detect_attacks 現在回傳複合 dict {"alerts": [...], "summary": ...}
            analysis_result = analyzer.detect_attacks()
            alerts_raw      = analysis_result.get("alerts", [])
            summary_info    = analysis_result.get("summary", {})
            
            # 從摘要中取得總封包數
            session.packet_count = summary_info.get('total_packets', 0)
        except ImportError:
            logger.warning('PcapAnalyzer 未安裝，跳過規則式偵測')
            alerts_raw  = []
        except Exception as e:
            logger.error(f'規則式偵測過程發生錯誤: {e}')
            alerts_raw = []

        # 清除舊告警再重寫
        Alert.objects.filter(session=session).delete()
        for a in alerts_raw:
            Alert.objects.create(
                session     = session,
                attack_type = a.get('attack_type', 'Unknown'),
                severity    = a.get('severity', 'LOW'),
                src_ip      = a.get('src_ip', ''),
                dst_ip      = a.get('dst_ip', ''),
                dst_port    = a.get('dst_port'),
                detail      = a.get('detail', ''),
                suggestion  = a.get('suggestion', ''),
            )
        session.alert_count = session.alerts.count()

        # ── Step 2：CNN Autoencoder 偵測 ─────────────────────
        logger.info(f'[Session {session_id}] 執行 CNN 異常偵測...')
        cnn_metrics = _run_cnn_analysis(session, pcap_path)
        if cnn_metrics:
            CNNResult.objects.update_or_create(
                session=session,
                defaults=cnn_metrics,
            )

        session.task_status = 'done'
        session.finished_at = timezone.now()
        session.save(update_fields=[
            'task_status', 'finished_at', 'packet_count', 'alert_count'])
        logger.info(f'[Session {session_id}] 分析完成')

    except Exception as exc:
        session = _get_session(session_id)
        _mark_failed(session, exc)
        raise self.retry(exc=exc)


def _run_cnn_analysis(session, pcap_path):
    """CNN Autoencoder 推論，回傳指標 dict（失敗回傳 None）。"""
    import numpy as np
    import torch

    model_path = str(settings.CNN_MODEL_PATH)
    if not os.path.exists(model_path):
        logger.warning(f'CNN 模型不存在: {model_path}，跳過 CNN 分析')
        return None

    try:
        from cnn_autoencoder import CNNAutoencoder
        from packet_visualizer import PacketVisualizer
        from scapy.all import rdpcap

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = CNNAutoencoder(latent_dim=settings.CNN_LATENT_DIM)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        packets = rdpcap(pcap_path)
        vis     = PacketVisualizer('medium', apply_mask=True, skip_ethernet=True)
        images  = []
        for pkt in packets:
            try:
                img = vis.bytes_to_image(bytes(pkt))
                images.append(img)
            except Exception:
                continue

        if not images:
            return None

        X        = np.array(images, dtype=np.float32)
        X_tensor = torch.from_numpy(X[:, np.newaxis, :, :]).to(device)

        with torch.no_grad():
            errors = model.reconstruction_error(X_tensor).cpu().numpy()

        threshold    = float(settings.CNN_THRESHOLD)
        anomaly_mask = errors > threshold
        n_normal     = int((~anomaly_mask).sum())
        n_anomaly    = int(anomaly_mask.sum())

        return {
            'threshold':        threshold,
            'normal_count':     n_normal,
            'anomaly_count':    n_anomaly,
            'detection_rate':   float(n_anomaly / (len(errors) + 1e-9)),
            'avg_normal_error': float(errors[~anomaly_mask].mean()) if n_normal  > 0 else 0.0,
            'avg_attack_error': float(errors[anomaly_mask].mean())  if n_anomaly > 0 else 0.0,
        }

    except Exception as e:
        logger.error(f'CNN 分析失敗: {e}')
        return None


# ── Task 3：Grad-CAM 視覺化（非同步掛載 AI 核心）─────────────
@shared_task(bind=True, max_retries=1, time_limit=300, soft_time_limit=270)
def run_gradcam(self, session_id, max_images=20,
                variant='gradcam', anomaly_only=True):
    """
    對 Session 中的封包執行 Grad-CAM，並將熱力圖儲存到 GradCAMImage。

    Args:
        session_id  : AnalysisSession.pk
        max_images  : 最多處理幾張影像（預設 20）
        variant     : gradcam / gradcam++ / scorecam
        anomaly_only: 是否只對異常封包產生熱力圖
    """
    import io
    import numpy as np
    import torch
    from PIL import Image
    from django.core.files.base import ContentFile

    from analyzer.models import AnalysisSession, GradCAMImage, CNNResult

    try:
        session = _get_session(session_id)

        model_path = str(settings.CNN_MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'CNN 模型不存在: {model_path}')

        # ── 載入 CNN 模型 ─────────────────────────────────────
        from cnn_autoencoder import CNNAutoencoder
        from packet_visualizer import PacketVisualizer
        from scapy.all import rdpcap

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = CNNAutoencoder(latent_dim=settings.CNN_LATENT_DIM)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # ── 建立 GradCAM 實例 ─────────────────────────────────
        from cnn_gradcam import GradCAM, GradCAMVisualizer
        cam = GradCAM(model, target_layer='encoder_conv', variant=variant)
        viz = GradCAMVisualizer()

        # ── 讀取封包並產生影像 ────────────────────────────────
        pcap_path = str(settings.MEDIA_ROOT / session.pcap_file.name)
        packets   = rdpcap(pcap_path)
        vis_tool  = PacketVisualizer('medium', apply_mask=True, skip_ethernet=True)

        threshold  = float(settings.CNN_THRESHOLD)
        processed  = 0

        for idx, pkt in enumerate(packets):
            if processed >= max_images:
                break
            try:
                img = vis_tool.bytes_to_image(bytes(pkt))
            except Exception:
                continue

            x_np     = np.array(img, dtype=np.float32)
            x_tensor = torch.from_numpy(x_np[np.newaxis, np.newaxis, :, :]).to(device)

            # 取得重建誤差
            with torch.no_grad():
                err = float(model.reconstruction_error(x_tensor).cpu().numpy()[0])

            is_anomaly = err > threshold
            if anomaly_only and not is_anomaly:
                continue

            # 產生熱力圖
            heatmap = cam.generate(x_tensor)  # (1, H, W)

            # 轉換為 PNG bytes
            orig_png = _array_to_png(x_np)
            heat_png = _heatmap_to_png(viz, x_tensor, heatmap)
            comp_png = _comparison_png(viz, x_tensor, heatmap)

            # 儲存至資料庫
            gcam_obj, _ = GradCAMImage.objects.update_or_create(
                session=session, packet_index=idx, variant=variant,
                defaults={
                    'is_anomaly':  is_anomaly,
                    'recon_error': err,
                },
            )
            gcam_obj.original_image.save(
                f'orig_{idx}.png', ContentFile(orig_png), save=False)
            gcam_obj.heatmap_image.save(
                f'heat_{idx}.png', ContentFile(heat_png), save=False)
            gcam_obj.comparison_image.save(
                f'comp_{idx}.png', ContentFile(comp_png), save=False)
            gcam_obj.save()

            processed += 1

        logger.info(f'[Session {session_id}] Grad-CAM 完成，共產生 {processed} 張影像')
        return {'session_id': session_id, 'images_generated': processed}

    except Exception as exc:
        logger.error(f'run_gradcam 失敗: {exc}')
        raise self.retry(exc=exc)


# ── 影像轉換輔助函式 ──────────────────────────────────────────
def _array_to_png(arr_2d):
    """灰階 ndarray → PNG bytes。"""
    import io
    import numpy as np
    from PIL import Image
    img = Image.fromarray((arr_2d * 255).astype('uint8'), mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def _heatmap_to_png(viz, x_tensor, heatmap):
    """熱力圖疊圖 → PNG bytes。"""
    import io
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=72)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    orig = x_tensor.cpu().numpy()[0, 0]
    heat = heatmap[0]

    ax.imshow(orig, cmap='gray', vmin=0, vmax=1)
    ax.imshow(heat, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


def _comparison_png(viz, x_tensor, heatmap):
    """原圖 + 熱力圖並排 → PNG bytes。"""
    import io
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=72)
    fig.patch.set_facecolor('#0d1117')

    orig = x_tensor.cpu().numpy()[0, 0]
    heat = heatmap[0]

    for ax in (ax1, ax2):
        ax.set_facecolor('#0d1117')
        ax.axis('off')

    ax1.imshow(orig, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('原始封包', color='white', fontsize=8)

    ax2.imshow(orig, cmap='gray', vmin=0, vmax=1)
    ax2.imshow(heat, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    ax2.set_title('Grad-CAM', color='white', fontsize=8)

    plt.tight_layout(pad=0.2)
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0.1,
                facecolor='#0d1117')
    plt.close(fig)
    return buf.getvalue()
