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
from django.core.files.base import ContentFile

logger = logging.getLogger(__name__)

# 確保 core/ 在路徑中
_CORE = str(settings.BASE_DIR / 'core')
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)


# ── 工具函式 ─────────────────────────────────────────────────
def _get_session(session_id):
    from analyzer.models import AnalysisSession
    return AnalysisSession.objects.get(pk=session_id)


def _set_progress(session, percent, stage, detail='', processed=None, total=None):
    """Persist a concrete stage/sample progress update for AJAX polling."""
    session.progress_percent = max(0, min(100, int(percent)))
    session.progress_stage = str(stage)[:120]
    session.progress_detail = str(detail)[:240]
    fields = ['progress_percent', 'progress_stage', 'progress_detail']
    if processed is not None:
        session.processed_samples = max(0, int(processed))
        fields.append('processed_samples')
    if total is not None:
        session.total_samples = max(0, int(total))
        fields.append('total_samples')
    session.save(update_fields=fields)


def _mark_failed(session, error):
    session.task_status  = 'failed'
    session.error_message = str(error)[:2000]
    session.progress_stage = '分析失敗'
    session.progress_detail = str(error)[:240]
    session.finished_at  = timezone.now()
    session.save(update_fields=['task_status', 'error_message', 'finished_at',
                                'progress_stage', 'progress_detail'])
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
            # ── [P1-5 修正] 串流計數避免 OOM ──
            from scapy.utils import PcapReader
            count = 0
            with PcapReader(pf.file.path) as reader:
                for _ in reader:
                    count += 1
            pf.packet_count = count
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


# ── 模型輸入表示與前處理 ─────────────────────────────────────
def _resolve_model_config(session):
    """回傳 session 對應的 (settings model config, checkpoint path)。"""
    selected = settings.ANOMALY_MODELS.get(session.model_key)
    if selected is None:
        raise KeyError(
            f'未知模型 key={session.model_key!r}。請重新建立分析任務並選擇目前支援的模型；'
            '系統不會自動以其他 checkpoint 代替。'
        )
    if not selected.get('enabled', True):
        raise RuntimeError(f'模型 {session.model_key} 已停用')
    model_path = str(selected.get('path', ''))
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f'找不到模型檔案: {model_path}')
    return selected, model_path


def _resolve_flow_preprocessor(bundle, selected, session):
    """解析 flow 模型需要的「訓練期前處理器」。

    優先順序：checkpoint metadata -> settings.ANOMALY_PREPROCESSORS ->
    VNAAP_* 環境變數。找不到時寧可明確失敗，也不把原始 packet bytes
    直接餵給 CSV/flow 模型，避免再次產生 train/serve skew。
    """
    import os as _os

    candidates = []
    for cfg in (bundle.config, selected):
        if isinstance(cfg, dict):
            value = cfg.get('scaler_path') or cfg.get('preprocessor_path')
            if value:
                candidates.append(str(value))

    mapping = getattr(settings, 'ANOMALY_PREPROCESSORS', {})
    mapped = mapping.get(session.model_key)
    if isinstance(mapped, (list, tuple)):
        candidates.extend(str(x) for x in mapped)
    elif mapped:
        candidates.append(str(mapped))

    env_key = 'VNAAP_PREPROCESSOR_' + ''.join(
        c if c.isalnum() else '_' for c in session.model_key.upper()
    )
    if _os.getenv(env_key):
        candidates.insert(0, _os.getenv(env_key))

    # 去重但保留優先順序。
    seen = set()
    for raw in candidates:
        raw = str(raw)
        if raw in seen:
            continue
        seen.add(raw)
        paths = [raw]
        if not os.path.isabs(raw):
            paths.append(str(settings.BASE_DIR / raw))
        for path in paths:
            if os.path.exists(path):
                return path

    raise FileNotFoundError(
        '此模型是 flow/CSV 特徵模型，但找不到訓練期前處理器。'
        f' model_key={session.model_key}, representation={bundle.input_representation}. '
        '請在 checkpoint metadata 寫入 scaler_path，或設定 '
        f'{env_key}=<訓練期 .pkl>。禁止改用原始 PCAP bytes 直接推論。'
    )


def _build_flow_images(pcap_path, bundle, selected, session, progress_callback=None):
    """PCAP -> CICFlow-like features -> training preprocessor -> 32x32 image。"""
    import pickle
    import tempfile
    import numpy as np
    import pandas as pd
    from pcap_flow_converter import PcapFlowConverter
    from cnn_autoencoder import features_to_image

    class _NumpyCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core', 1)
            return super().find_class(module, name)

    preprocessor_path = _resolve_flow_preprocessor(bundle, selected, session)
    with open(preprocessor_path, 'rb') as fh:
        try:
            payload = _NumpyCompatUnpickler(fh).load()
        except Exception:
            fh.seek(0)
            payload = pickle.load(fh)

    scaler = payload['scaler']
    feature_cols = payload['feature_columns']
    image_size = int(payload.get('image_size', bundle.config.get('image_size', 32)))
    normalization = payload.get('normalization', 'none')

    with tempfile.NamedTemporaryFile(prefix='vnaap_flow_', suffix='.csv', delete=False) as tf:
        flow_csv = tf.name

    try:
        PcapFlowConverter().convert(
            pcap_path,
            flow_csv,
            default_label='UNKNOWN',
            progress_callback=progress_callback,
        )
        df = pd.read_csv(flow_csv)
        if df.empty:
            raise RuntimeError('PCAP 未轉換出任何 Flow，無法進行 CNN 分析。')

        missing = [c for c in feature_cols if c not in df.columns]
        for col in missing:
            df[col] = 0.0

        X_raw = (df[feature_cols]
                 .apply(pd.to_numeric, errors='coerce')
                 .fillna(0.0)
                 .to_numpy(dtype=np.float64))
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = scaler.transform(X_raw).astype(np.float32)
        if normalization == 'clip_0_1':
            X_scaled = np.clip(X_scaled, 0.0, 1.0)
        X_img = features_to_image(X_scaled, image_size).squeeze(1)

        meta = {
            'unit': 'flow',
            'count': int(len(df)),
            'feature_columns': list(feature_cols),
            'preprocessor_path': preprocessor_path,
            'image_size': image_size,
            'normalization': normalization,
            'dataframe': df,
        }
        return X_img, meta
    finally:
        try:
            os.remove(flow_csv)
        except OSError:
            pass


def _load_packet_images(pcap_path):
    """串流讀取 PCAP，避免整個 500MB 上傳檔一次性 rdpcap 佔滿 RAM。"""
    import numpy as np
    from scapy.utils import PcapReader
    from packet_visualizer import PacketVisualizer

    vis = PacketVisualizer('medium', apply_mask=True, skip_ethernet=True)
    batch_x, batch_idx = [], []
    with PcapReader(pcap_path) as reader:
        for idx, pkt in enumerate(reader):
            try:
                batch_x.append(vis.bytes_to_image(bytes(pkt)))
                batch_idx.append(idx)
            except Exception as exc:
                logger.warning(f'封包 {idx} 影像化失敗，略過：{exc}')
            if len(batch_x) >= 256:
                yield batch_idx, np.asarray(batch_x, dtype=np.float32), {'unit': 'packet'}
                batch_x, batch_idx = [], []
    if batch_x:
        yield batch_idx, np.asarray(batch_x, dtype=np.float32), {'unit': 'packet'}


def _iter_model_batches(pcap_path, bundle, selected, session, progress_callback=None):
    """依模型 representation 產生同一套推論 batch；CNN 與 Grad-CAM 共用。"""
    import numpy as np
    if bundle.is_packet_native:
        yield from _load_packet_images(pcap_path)
        return

    X_img, meta = _build_flow_images(pcap_path, bundle, selected, session, progress_callback=progress_callback)
    for start in range(0, len(X_img), 256):
        end = min(start + 256, len(X_img))
        yield list(range(start, end)), X_img[start:end], meta


def _make_score_histogram_png(scores, threshold):
    """產生真實推論分數分布圖；不再用前端模擬常態分布。"""
    import io
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    values = np.asarray(scores, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=150)
    if len(values):
        bins = min(40, max(10, int(np.sqrt(len(values)))))
        ax.hist(values, bins=bins, alpha=0.78, edgecolor='white', linewidth=0.35)
    ax.axvline(float(threshold), linestyle='--', linewidth=2.0,
               label=f'Threshold = {float(threshold):.6g}')
    ax.set_title('CNN Anomaly Score Distribution')
    ax.set_xlabel('Reconstruction / anomaly score')
    ax.set_ylabel('Sample count')
    ax.grid(alpha=0.18, axis='y')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()


def _rank_evidence(evidence, sample_limit=6):
    """同時保留高分異常與低分正常，讓報告有對照組。"""
    anomalies = sorted(
        (x for x in evidence if x['is_anomaly']),
        key=lambda x: x['score'], reverse=True,
    )[:max(1, sample_limit // 2)]
    normals = sorted(
        (x for x in evidence if not x['is_anomaly']),
        key=lambda x: x['score'],
    )[:max(1, sample_limit - len(anomalies))]
    selected = anomalies + normals
    selected.sort(key=lambda x: (not x['is_anomaly'], -x['score'] if x['is_anomaly'] else x['score']))
    return selected[:sample_limit]


# ── Task 2：完整 PCAP 分析（規則偵測 + CNN + 自動證據）─────────
def execute_pcap_analysis(session_id, celery_task_id='', auto_gradcam=True):
    """Celery 與同步 fallback 共用的唯一完整分析流程。"""
    from analyzer.models import AnalysisSession, Alert, CNNResult

    session = _get_session(session_id)
    session.task_status = 'running'
    if celery_task_id:
        session.celery_task_id = celery_task_id
    session.error_message = ''
    session.progress_percent = 2
    session.progress_stage = '檢查輸入檔案'
    session.progress_detail = '確認 PCAP 檔案路徑與模型設定。'
    session.processed_samples = 0
    session.total_samples = 0
    session.save(update_fields=['task_status', 'celery_task_id', 'error_message',
                                'progress_percent', 'progress_stage', 'progress_detail',
                                'processed_samples', 'total_samples'])

    pcap_path = str(settings.MEDIA_ROOT / session.pcap_file.name)
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f'PCAP 不存在: {pcap_path}')

    _set_progress(session, 7, '規則式分析', '解析封包並比對目前規則引擎。')
    # ① 規則式偵測
    alerts_raw = []
    try:
        from pcap_analyzer import PcapAnalyzer
        analyzer = PcapAnalyzer(pcap_path)
        def _rule_progress(stage, percent, processed, total, detail):
            _set_progress(session, percent, stage, detail,
                          processed=processed, total=total or processed)
        analyzer.load(progress_callback=_rule_progress)
        session.packet_count = len(analyzer.packets) if hasattr(analyzer, 'packets') else 0
        analysis_result = analyzer.detect_attacks(progress_callback=_rule_progress)
        alerts_raw = analysis_result.get('alerts', [])
        summary_info = analysis_result.get('summary', {})
        if summary_info.get('total_packets', 0) > 0:
            session.packet_count = summary_info['total_packets']
    except ImportError:
        logger.warning('PcapAnalyzer 未安裝，跳過規則式偵測')
    except Exception as exc:
        # 規則引擎故障不應吃掉整份 session；CNN 仍可繼續。
        logger.exception(f'規則式偵測失敗: {exc}')

    Alert.objects.filter(session=session).delete()
    for a in alerts_raw:
        Alert.objects.create(
            session=session,
            attack_type=a.get('attack_type', 'Unknown'),
            severity=a.get('severity', 'LOW'),
            src_ip=a.get('src_ip', ''),
            dst_ip=a.get('dst_ip', ''),
            dst_port=a.get('dst_port'),
            detail=a.get('detail', ''),
            suggestion=a.get('suggestion', ''),
        )
    session.alert_count = session.alerts.count()
    session.save(update_fields=['packet_count', 'alert_count'])
    _set_progress(session, 20, 'CNN 模型推論', '依所選 checkpoint 的輸入表示法執行推論。',
                  processed=0, total=max(int(session.packet_count or 0), 0))

    # ② CNN
    def _cnn_progress(processed, total, stage='模型推論', exact_percent=None):
        denominator = max(int(total or session.packet_count or processed or 1), 1)
        fraction = min(max(float(processed) / denominator, 0.0), 1.0)
        if exact_percent is not None:
            pct = int(exact_percent)  # Flow 轉換階段依實際檔案讀取位置估算
        elif stage == 'Flow 特徵模型推論':
            pct = 50 + int(25 * fraction)
        else:
            pct = 20 + int(55 * fraction)
        # 避免流量數量估計差異令畫面百分比倒退。
        pct = max(int(session.progress_percent or 0), min(75, pct))
        _set_progress(session, pct, stage,
                      f'已處理 {int(processed):,} / 約 {denominator:,} 個樣本。',
                      processed=processed, total=denominator)
    cnn_metrics = _run_cnn_analysis(session, pcap_path, progress_callback=_cnn_progress)
    if cnn_metrics:
        metrics = dict(cnn_metrics)
        histogram_png = metrics.pop('_histogram_png', None)
        CNNResult.objects.filter(session=session).delete()
        cnn_obj = CNNResult.objects.create(session=session, **metrics)
        if histogram_png:
            cnn_obj.error_histogram.save(
                f'session_{session.pk}_score_distribution.png',
                ContentFile(histogram_png), save=True,
            )

        # ③ 自動產生一組「異常 + 正常」Grad-CAM 證據。
        if auto_gradcam:
            for cam_idx, variant in enumerate(('gradcam', 'gradcam++', 'scorecam')):
                _set_progress(session, 78 + cam_idx * 6, f'產生 {variant}',
                              f'針對代表樣本計算 {variant} 解釋圖。')
                try:
                    base_progress = 78 + cam_idx * 6
                    def _cam_progress(stage, inner_pct, processed, total, detail):
                        pct = base_progress + int(5 * min(max(inner_pct, 0), 100) / 100)
                        _set_progress(session, pct, f'{variant}：{stage}', detail,
                                      processed=processed, total=total)
                    _generate_gradcam_images(
                        session_id=session.pk, max_images=4,
                        variant=variant, anomaly_only=False,
                        progress_callback=_cam_progress,
                    )
                except Exception as exc:
                    # 某種 CAM 失敗不能抹掉已完成的偵測，但將錯誤寫入日誌。
                    logger.exception(f'Session {session.pk} {variant} 產生失敗: {exc}')

    session.task_status = 'done'
    session.progress_percent = 100
    session.progress_stage = '分析完成'
    session.progress_detail = '規則偵測、模型推論與可用的解釋圖已完成。'
    session.finished_at = timezone.now()
    session.save(update_fields=[
        'task_status', 'finished_at', 'packet_count', 'alert_count', 'error_message',
        'progress_percent', 'progress_stage', 'progress_detail'])
    logger.info(f'[Session {session.pk}] 完整分析完成')
    return session.pk


@shared_task(bind=True, max_retries=1, default_retry_delay=5,
             time_limit=7200, soft_time_limit=6900)
def run_pcap_analysis(self, session_id):
    try:
        return execute_pcap_analysis(
            session_id,
            celery_task_id=self.request.id or '',
            auto_gradcam=True,
        )
    except Exception as exc:
        session = _get_session(session_id)
        _mark_failed(session, exc)
        raise self.retry(exc=exc)


def _run_cnn_analysis(session, pcap_path, progress_callback=None):
    """唯一 CNN 推論入口：依 checkpoint representation 選擇正確輸入格式。"""
    import numpy as np
    from model_registry import load_anomaly_model, compute_anomaly_scores
    from analyzer.models import CNNResult

    selected, model_path = _resolve_model_config(session)
    try:
        bundle = load_anomaly_model(
            model_path,
            default_latent_dim=settings.CNN_LATENT_DIM,
        )
        logger.info(
            f'[Session {session.pk}] CNN model={bundle.model_type}, '
            f'representation={bundle.input_representation}, threshold={bundle.threshold:.8g}'
        )

        scores_list = []
        status_list = []
        total_seen = 0
        def _flow_convert_progress(stage, percent, processed, total, detail):
            if progress_callback:
                progress_callback(processed, total, stage, percent)
        for indices, X, meta in _iter_model_batches(
            pcap_path, bundle, selected, session,
            progress_callback=_flow_convert_progress,
        ):
            scored = compute_anomaly_scores(bundle, X)
            scores = np.asarray(scored['score'], dtype=np.float32)
            scores_list.append(scores)
            total_seen += len(scores)
            if progress_callback:
                progress_callback(total_seen, max(int(session.packet_count or total_seen), total_seen),
                                  '封包模型推論' if bundle.is_packet_native else 'Flow 特徵模型推論')
            if bundle.is_hybrid:
                status_list.append(np.asarray(scored['status'], dtype=object))

        if not scores_list:
            raise RuntimeError('PCAP 未產生任何可供 CNN 推論的樣本。')

        errors = np.concatenate(scores_list)
        if bundle.is_hybrid:
            statuses = np.concatenate(status_list)
            anomaly_mask = statuses != 'normal'
            known_count = int((statuses == 'known_attack').sum())
            unknown_count = int((statuses == 'unknown_attack').sum())
        else:
            anomaly_mask = errors > float(bundle.threshold)
            known_count = 0
            unknown_count = 0

        n_anomaly = int(anomaly_mask.sum())
        n_normal = int((~anomaly_mask).sum())
        threshold = float(bundle.threshold)
        metrics = {
            'threshold': threshold,
            'normal_count': n_normal,
            'anomaly_count': n_anomaly,
            'detection_rate': float(n_anomaly / max(len(errors), 1)),
            'avg_normal_error': float(errors[~anomaly_mask].mean()) if n_normal else 0.0,
            'avg_attack_error': float(errors[anomaly_mask].mean()) if n_anomaly else 0.0,
            'known_attack_count': known_count,
            'unknown_attack_count': unknown_count,
            '_histogram_png': _make_score_histogram_png(errors, threshold),
        }
        logger.info(
            f'[Session {session.pk}] CNN samples={total_seen}, '
            f'normal={n_normal}, anomaly={n_anomaly}, representation={bundle.input_representation}'
        )
        return metrics
    except Exception:
        logger.exception(f'CNN 分析失敗 [Session {session.pk}]')
        raise


# ── Task 3：Grad-CAM 視覺化 ──────────────────────────────────
def _generate_gradcam_images(session_id, max_images=6,
                             variant='gradcam', anomaly_only=False,
                             progress_callback=None):
    """依實際 checkpoint 表示法產生 CAM，並只保留少量代表樣本避免整批張量佔滿記憶體。

    progress_callback(stage, percent, processed, total, detail) 可選；percent 是此 CAM
    工作內部的 0–100 進度。代表樣本仍由實際異常分數／模型閾值排序，不會合成分數。
    """
    import heapq
    import numpy as np
    import torch
    from django.core.files.base import ContentFile
    from analyzer.models import GradCAMImage

    session = _get_session(session_id)
    selected_cfg, model_path = _resolve_model_config(session)
    from model_registry import load_anomaly_model, compute_anomaly_scores
    bundle = load_anomaly_model(model_path, default_latent_dim=settings.CNN_LATENT_DIM)
    from cnn_gradcam import GradCAM

    cam = GradCAM(bundle.model, target_layer=bundle.gradcam_target_layer, variant=variant)
    # 避免把所有 batch 的影像張量存起來。只保留異常分數最高的少量樣本，
    # 以及分數最低的正常參考樣本，降低大型 PCAP 的 RAM 使用量。
    anomaly_keep = max(1, int(max_images) if anomaly_only else int(max_images) // 2)
    normal_keep = max(1, int(max_images))
    anomaly_heap = []  # (score, index, x, meta)，min-heap 保留最高分異常
    normal_heap = []   # (-score, index, x, meta)，保留最低分正常
    seen = 0
    pcap_path = str(settings.MEDIA_ROOT / session.pcap_file.name)
    total_hint = max(int(session.packet_count or 0), 1)
    last_inner_percent = 0

    def _emit_progress(stage, percent, processed, total, detail):
        nonlocal last_inner_percent
        current = min(100, max(last_inner_percent, int(percent)))
        last_inner_percent = current
        if progress_callback:
            progress_callback(stage, current, processed, total, detail)

    def _cam_flow_progress(stage, percent, processed, total, detail):
        # Flow converter 的百分比是外層任務百分比 20–50；轉成本 CAM 工作的 0–55。
        inner_percent = int(55 * min(max((float(percent) - 20.0) / 30.0, 0.0), 1.0))
        _emit_progress(stage, inner_percent, processed, total, detail)
    for indices, X, meta in _iter_model_batches(
        pcap_path, bundle, selected_cfg, session,
        progress_callback=_cam_flow_progress,
    ):
        scored = compute_anomaly_scores(bundle, X)
        scores = np.asarray(scored['score'], dtype=np.float32).reshape(-1)
        if bundle.is_hybrid:
            flags = np.asarray(scored['status'], dtype=object) != 'normal'
        else:
            flags = scores > float(bundle.threshold)
        meta_copy = dict(meta) if isinstance(meta, dict) else {'unit': 'packet'}
        for local_idx, global_idx in enumerate(indices):
            score = float(scores[local_idx])
            x = np.asarray(X[local_idx], dtype=np.float32).copy()
            idx = int(global_idx)
            if bool(flags[local_idx]):
                item = (score, idx, x, meta_copy)
                if len(anomaly_heap) < anomaly_keep:
                    heapq.heappush(anomaly_heap, item)
                elif score > anomaly_heap[0][0]:
                    heapq.heapreplace(anomaly_heap, item)
            else:
                item = (-score, idx, x, meta_copy)
                if len(normal_heap) < normal_keep:
                    heapq.heappush(normal_heap, item)
                elif -score > normal_heap[0][0]:
                    heapq.heapreplace(normal_heap, item)
        seen += len(scores)
        # 樣本數可能是 flow 而非 packet；total_hint 僅為進度估計，完成後仍會到 100%。
        pct = min(55, int(55 * seen / max(total_hint, seen)))
        _emit_progress('挑選代表樣本', pct, seen, max(total_hint, seen),
                       f'已評估 {seen:,} 個樣本並更新代表樣本候選。')

    anomalies = [
        {'index': idx, 'score': score, 'is_anomaly': True, '_x': x, '_meta': meta}
        for score, idx, x, meta in anomaly_heap
    ]
    normals = [
        {'index': idx, 'score': -neg_score, 'is_anomaly': False, '_x': x, '_meta': meta}
        for neg_score, idx, x, meta in normal_heap
    ]
    anomalies.sort(key=lambda x: x['score'], reverse=True)
    normals.sort(key=lambda x: x['score'])
    if anomaly_only:
        candidates = anomalies[:max_images]
    else:
        candidates = anomalies[:anomaly_keep] + normals[:max(1, max_images - min(len(anomalies), anomaly_keep))]
        candidates.sort(key=lambda x: (not x['is_anomaly'], -x['score'] if x['is_anomaly'] else x['score']))
        candidates = candidates[:max_images]

    if not candidates:
        _emit_progress('沒有可產生 CAM 的樣本', 100, 0, 0,
                       '沒有取得可用的樣本影像。')
        return 0

    processed = 0
    for item in candidates:
        idx = int(item['index'])
        x_np = item['_x']
        meta = item['_meta']
        x_tensor = torch.from_numpy(x_np[None, None, :, :]).to(bundle.device)
        heatmap = cam.generate(x_tensor)
        heat_np = np.asarray(heatmap[0], dtype=np.float32)
        heat_np = np.nan_to_num(heat_np, nan=0.0, posinf=1.0, neginf=0.0)
        heat_np = np.clip(heat_np, 0.0, 1.0)

        orig_png = _array_to_png(x_np)
        heat_png = _heatmap_to_png(x_tensor, heat_np)
        comp_png = _comparison_png(
            x_tensor, heat_np,
            unit=meta.get('unit', 'packet'),
            feature_columns=meta.get('feature_columns'),
        )
        gcam_obj, _ = GradCAMImage.objects.update_or_create(
            session=session, packet_index=idx, variant=variant,
            defaults={'is_anomaly': bool(item['is_anomaly']),
                      'recon_error': float(item['score'])},
        )
        gcam_obj.original_image.save(f'orig_{idx}.png', ContentFile(orig_png), save=False)
        gcam_obj.heatmap_image.save(f'heat_{idx}.png', ContentFile(heat_png), save=False)
        gcam_obj.comparison_image.save(f'comp_{idx}.png', ContentFile(comp_png), save=False)
        gcam_obj.save()
        processed += 1
        pct = 55 + int(45 * processed / max(len(candidates), 1))
        _emit_progress('產生熱力圖', pct, processed, len(candidates),
                       f'已產生 {processed}/{len(candidates)} 張 {variant} 解釋圖。')

    logger.info(f'[Session {session.pk}] {variant} 完成，共產生 {processed} 張影像')
    return processed


@shared_task(bind=True, max_retries=1, time_limit=300, soft_time_limit=270)
def run_gradcam(self, session_id, max_images=20,
                variant='gradcam', anomaly_only=True):
    """對與 CNN 完全相同的模型輸入表示產生 Grad-CAM。"""
    allowed = {'gradcam', 'gradcam++', 'scorecam'}
    if variant not in allowed:
        raise ValueError(f'不支援的 Grad-CAM variant: {variant}')
    max_images = max(1, min(int(max_images), 100))
    try:
        count = _generate_gradcam_images(
            session_id=session_id,
            max_images=max_images,
            variant=variant,
            anomaly_only=bool(anomaly_only),
        )
        return {'session_id': session_id, 'images_generated': count}
    except Exception as exc:
        logger.exception(f'run_gradcam 失敗: {exc}')
        raise self.retry(exc=exc)


# ── 影像轉換輔助函式 ──────────────────────────────────────────
def _array_to_png(arr_2d):
    """灰階 ndarray -> PNG bytes。"""
    import io
    import numpy as np
    from PIL import Image
    arr = np.clip(np.asarray(arr_2d, dtype=np.float32), 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype('uint8'), mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def _heatmap_to_png(x_tensor, heatmap):
    """輸出獨立 Grad-CAM 疊圖。"""
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    orig = x_tensor.detach().cpu().numpy()[0, 0]
    heat = heatmap[0] if getattr(heatmap, 'ndim', 0) == 3 else heatmap
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.imshow(orig, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    im = ax.imshow(heat, cmap='jet', alpha=0.55, vmin=0, vmax=1, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Grad-CAM', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    return buf.getvalue()


def _comparison_png(x_tensor, heatmap, unit='packet', feature_columns=None):
    """輸出原始輸入 + Grad-CAM 對照圖，並附上熱點證據。"""
    import io
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    orig = x_tensor.detach().cpu().numpy()[0, 0]
    heat = heatmap[0] if getattr(heatmap, 'ndim', 0) == 3 else heatmap
    heat = np.asarray(heat, dtype=np.float32)

    # 取前 8 個不重複熱點，將 CNN 的空間位置轉成可閱讀的證據。
    flat_order = np.argsort(heat.reshape(-1))[::-1]
    top_ids = []
    for flat_idx in flat_order:
        if heat.reshape(-1)[flat_idx] <= 0:
            break
        top_ids.append(int(flat_idx))
        if len(top_ids) >= 8:
            break

    if unit == 'flow' and feature_columns:
        labels = []
        for flat_idx in top_ids:
            if flat_idx < len(feature_columns):
                labels.append(str(feature_columns[flat_idx]))
        evidence_text = 'Top sensitive features: ' + ', '.join(labels[:6]) if labels else 'Top sensitive features: none'
    elif unit == 'packet':
        offsets = ', '.join(str(i) for i in top_ids[:8])
        evidence_text = 'Top sensitive byte offsets: ' + (offsets if offsets else 'none')
    else:
        evidence_text = 'Top sensitive image positions: ' + ', '.join(str(i) for i in top_ids[:8])

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), dpi=120)
    axes[0].imshow(orig, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_title(f'輸入影像（{unit}）', fontsize=9)
    axes[1].imshow(orig, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    im = axes[1].imshow(heat, cmap='jet', alpha=0.55, vmin=0, vmax=1, interpolation='bilinear')
    axes[1].set_title('Grad-CAM 熱力圖疊加', fontsize=9)
    for ax in axes:
        ax.axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.text(0.5, 0.018, evidence_text, ha='center', va='bottom', fontsize=7.5, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1), pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    return buf.getvalue()
