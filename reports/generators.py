# ============================================================
# reports/generators.py
# 分析報告匯出系統：PDF（ReportLab）/ JSON / CSV
# ============================================================
import os
import io
import json
import csv
import html
import sys
from datetime import datetime

from django.conf import settings
from django.core.files.base import ContentFile

from analyzer.models import AnalysisSession, AnalysisReport


class ReportGenerator:
    """產生具有可追溯證據的 PDF / JSON / CSV 報告。"""

    def __init__(self, session: AnalysisSession, user):
        self.session = session
        self.user = user
        self.alerts = list(session.alerts.order_by('-timestamp'))
        self.cnn = getattr(session, 'cnn_result', None)
        # 不只取異常；同時保留正常參考樣本，讓報告可以比較。
        # 各變體至少保留一張最具代表性的證據；若有正常參考樣本，也盡量一併納入。
        all_cams = list(session.gradcam_images.order_by('-is_anomaly', '-recon_error'))
        chosen = []
        for variant in ('gradcam', 'gradcam++', 'scorecam'):
            rows = [x for x in all_cams if x.variant == variant]
            if rows:
                chosen.append(rows[0])
                normal = next((x for x in rows if not x.is_anomaly), None)
                if normal and normal.pk != rows[0].pk:
                    chosen.append(normal)
        self.gcam_imgs = chosen[:6] or all_cams[:6]

    def generate(self, fmt: str) -> AnalysisReport:
        fmt = fmt.lower()
        if fmt == 'pdf':
            return self._generate_pdf()
        if fmt == 'json':
            return self._generate_json()
        if fmt == 'csv':
            return self._generate_csv()
        raise ValueError(f'不支援的格式: {fmt}')

    # ── 共用模型 metadata ────────────────────────────────────
    def _model_info(self):
        core = str(settings.BASE_DIR / 'core')
        if core not in sys.path:
            sys.path.insert(0, core)
        try:
            from model_registry import load_anomaly_model
            selected = settings.ANOMALY_MODELS.get(self.session.model_key)
            if selected is None:
                return {
                    'model_key': self.session.model_key, 'label': self.session.model_key,
                    'model_type': 'unknown', 'representation': 'unknown', 'unit': 'packet',
                    'preprocessor': None, 'error': '此任務使用的模型 key 已不在目前模型清單中。',
                }
            path = str(selected.get('path', ''))
            if not os.path.exists(path):
                return {
                    'model_key': self.session.model_key,
                    'label': selected.get('label', self.session.model_key),
                    'model_type': selected.get('type', 'unknown'),
                    'representation': 'unknown',
                    'unit': 'packet',
                    'preprocessor': None,
                }
            bundle = load_anomaly_model(
                path,
                default_latent_dim=settings.CNN_LATENT_DIM,
            )
            rep = bundle.input_representation
            packet_native = bundle.is_packet_native
            preprocessor = None
            if not packet_native:
                preprocessor = bundle.config.get('scaler_path') or bundle.config.get('preprocessor_path')
                if preprocessor and not os.path.isabs(str(preprocessor)):
                    preprocessor = str(settings.BASE_DIR / str(preprocessor))
            classifier_threshold = None
            # Hybrid checkpoint 的分類器門檻通常儲存在 checkpoint metadata；
            # 和 VAE reconstruction threshold 分開呈現，避免混為同一條判定規則。
            if bundle.is_hybrid:
                try:
                    import torch
                    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                    if isinstance(checkpoint, dict):
                        value = checkpoint.get('classifier_threshold')
                        if value is not None:
                            classifier_threshold = float(value)
                except Exception:
                    # 報告仍可產生，但會把分類器門檻標為未能讀取，不臆測數值。
                    classifier_threshold = None
            return {
                'model_key': self.session.model_key,
                'label': selected.get('label', self.session.model_key),
                'model_type': bundle.model_type,
                'representation': rep,
                'unit': 'packet' if packet_native else 'flow',
                'preprocessor': preprocessor,
                'threshold': float(bundle.threshold),
                'classifier_threshold': classifier_threshold,
            }
        except Exception as exc:
            return {
                'model_key': self.session.model_key,
                'label': self.session.model_key,
                'model_type': 'unknown',
                'representation': 'unknown',
                'unit': 'packet',
                'preprocessor': None,
                'error': str(exc),
            }

    def _evidence_summary(self, model_info):
        if not self.cnn:
            return {
                'status': '未產生 CNN 結果',
                'reasons': ['本 Session 沒有可用的 CNN 結果。'],
            }
        total = self.cnn.normal_count + self.cnn.anomaly_count
        reasons = []
        is_hybrid = model_info.get('model_type') == 'hybrid_semi'
        if is_hybrid:
            reasons.append(
                '半監督式模型使用兩條判定路徑：分類器辨識已知攻擊；VAE 分數超過異常閾值且分類器未判為已知攻擊時，列為未知／新型攻擊。'
            )
            reasons.append(
                f'半監督分類結果：已知攻擊 {self.cnn.known_attack_count:,}、'
                f'未知／新型攻擊 {self.cnn.unknown_attack_count:,}、'
                f'正常預測 {self.cnn.normal_count:,}。VAE 異常閾值為 {self.cnn.threshold:.6g}。'
            )
            if model_info.get('classifier_threshold') is not None:
                reasons.append(f'已知攻擊分類器門檻：{model_info["classifier_threshold"]:.6g}（取自 checkpoint metadata）。')
            else:
                reasons.append('已知攻擊分類器門檻未能從 checkpoint metadata 讀取；本報告不推測該值。')
        elif self.cnn.anomaly_count:
            reasons.append(
                f'CNN/VAE：{self.cnn.anomaly_count:,}/{total:,} 個樣本的異常分數高於訓練時校準閾值 '
                f'{self.cnn.threshold:.6g}；這是此非監督模型的直接異常判定條件。'
            )
        else:
            reasons.append(
                f'CNN/VAE：{total:,} 個樣本均未超過異常閾值 {self.cnn.threshold:.6g}。'
            )
        if self.alerts:
            attack_types = sorted({a.attack_type for a in self.alerts if a.attack_type})
            reasons.append(
                f'規則式偵測：產生 {len(self.alerts):,} 筆告警。'
                + (f' 攻擊類型包含：{", ".join(attack_types[:8])}。' if attack_types else '')
            )
        else:
            reasons.append('規則式偵測：本次沒有產生告警；這只代表目前規則未觸發，不代表所有流量皆安全。')
        if self.gcam_imgs:
            variants = sorted({x.variant for x in self.gcam_imgs})
            labels = {'gradcam': 'Grad-CAM', 'gradcam++': 'Grad-CAM++', 'scorecam': 'Score-CAM'}
            reasons.append(
                f'可解釋性證據：已產生 {len(self.gcam_imgs)} 張樣本圖；本報告包含的變體為 '
                f'{", ".join(labels.get(v, v) for v in variants)}。熱區表示模型敏感區域，不是單獨的攻擊證明。'
            )
        status = '偵測到異常證據' if (self.alerts or self.cnn.anomaly_count or self.cnn.known_attack_count or self.cnn.unknown_attack_count) else '本次未偵測到異常證據（不代表絕對安全）'
        return {'status': status, 'reasons': reasons}

    @staticmethod
    def _register_cjk_fonts():
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont

        regular_name = 'VNAAP-CJK'
        bold_name = 'VNAAP-CJK-Bold'
        regular_candidates = [
            r'C:\Windows\Fonts\msyh.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf',
            '/System/Library/Fonts/PingFang.ttc',
        ]
        bold_candidates = [
            r'C:\Windows\Fonts\msyhbd.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJKtc-Bold.otf',
            '/System/Library/Fonts/PingFang.ttc',
        ]

        def _register(name, candidates):
            if name in pdfmetrics.getRegisteredFontNames():
                return name
            for path in candidates:
                if not os.path.exists(path):
                    continue
                try:
                    if path.lower().endswith('.ttc'):
                        pdfmetrics.registerFont(TTFont(name, path, subfontIndex=0))
                    else:
                        pdfmetrics.registerFont(TTFont(name, path))
                    return name
                except Exception:
                    continue
            return None

        regular = _register(regular_name, regular_candidates)
        bold = _register(bold_name, bold_candidates)
        if not regular:
            # ReportLab 的 CID fallback 不依賴本機中文字型檔，可避免中文空白。
            regular = 'STSong-Light'
            pdfmetrics.registerFont(UnicodeCIDFont(regular))
            bold = regular
        elif not bold:
            bold = regular
        return regular, bold

    @staticmethod
    def _safe_text(value):
        return html.escape('' if value is None else str(value))

    @staticmethod
    def _plot_severity(alerts):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        counts = [sum(1 for a in alerts if a.severity == x) for x in order]
        fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=150)
        ax.bar(order, counts)
        ax.set_title('Rule-based Alert Severity')
        ax.set_ylabel('Alert count')
        ax.grid(axis='y', alpha=0.18)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        return buf.getvalue()

    def _pdf_image(self, raw_or_path, width, height=None):
        from reportlab.platypus import Image as RLImage
        if isinstance(raw_or_path, (bytes, bytearray)):
            obj = RLImage(io.BytesIO(raw_or_path))
        else:
            obj = RLImage(raw_or_path)
        obj.drawWidth = width
        if height is not None:
            obj.drawHeight = height
        return obj

    # ── PDF ──────────────────────────────────────────────────
    def _generate_pdf(self) -> AnalysisReport:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                Image as RLImage, HRFlowable, PageBreak, KeepTogether,
            )
        except ImportError:
            raise ImportError('請安裝 reportlab：pip install reportlab')

        font_regular, font_bold = self._register_cjk_fonts()
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=1.55*cm,
            rightMargin=1.55*cm,
            topMargin=1.55*cm,
            bottomMargin=1.55*cm,
            title=f'VNAAP Session {self.session.pk} Analysis Report',
            author=self.user.get_full_name() or self.user.username,
        )

        styles = getSampleStyleSheet()
        title = ParagraphStyle('vTitle', parent=styles['Title'], fontName=font_bold,
                               fontSize=19, leading=24, textColor=colors.HexColor('#1456a0'),
                               spaceAfter=8)
        subtitle = ParagraphStyle('vSub', parent=styles['Normal'], fontName=font_regular,
                                  fontSize=9.5, leading=14, textColor=colors.HexColor('#4b5563'))
        h1 = ParagraphStyle('vH1', parent=styles['Heading1'], fontName=font_bold,
                            fontSize=13, leading=17, spaceBefore=8, spaceAfter=8,
                            textColor=colors.HexColor('#0f172a'),
                            backColor=colors.HexColor('#eaf2ff'), borderPadding=5)
        h2 = ParagraphStyle('vH2', parent=styles['Heading2'], fontName=font_bold,
                            fontSize=11, leading=15, spaceBefore=6, spaceAfter=6,
                            textColor=colors.HexColor('#1456a0'))
        body = ParagraphStyle('vBody', parent=styles['BodyText'], fontName=font_regular,
                              fontSize=9.2, leading=14, textColor=colors.HexColor('#111827'))
        small = ParagraphStyle('vSmall', parent=body, fontSize=7.6, leading=10)
        note = ParagraphStyle('vNote', parent=body, fontSize=8.1, leading=12,
                              textColor=colors.HexColor('#374151'), backColor=colors.HexColor('#f8fafc'),
                              borderPadding=6)
        center = ParagraphStyle('vCenter', parent=body, alignment=TA_CENTER)

        def P(text, style=body):
            return Paragraph(self._safe_text(text), style)

        def PT(text, style=body):
            # 已包含少量 ReportLab inline tags 時使用。
            return Paragraph(text, style)

        model_info = self._model_info()
        evidence = self._evidence_summary(model_info)
        story = []

        # Cover / executive summary
        story += [Spacer(1, 0.25*cm), PT('視覺化網路攻擊自動化分析平台 — PCAP 分析報告', title)]
        story += [PT('Network Attack Automated Analysis Platform — PCAP Analysis Report', subtitle),
                  Spacer(1, 0.15*cm),
                  HRFlowable(width='100%', thickness=2, color=colors.HexColor('#1f6feb')),
                  Spacer(1, 0.35*cm)]
        status_color = '#b91c1c' if evidence['status'] == '偵測到異常證據' else '#166534'
        story.append(PT(f'<font color="{status_color}"><b>本次分析摘要：{self._safe_text(evidence["status"])}</b></font>', body))
        story.append(Spacer(1, 0.18*cm))
        for reason in evidence['reasons']:
            story.append(PT(f'• {self._safe_text(reason)}', body))
        story.append(Spacer(1, 0.25*cm))

        meta = [
            ['Session', str(self.session.pk)],
            ['標籤', self.session.label],
            ['來源模式', self.session.get_mode_display()],
            ['模型', f'{model_info.get("label", self.session.model_key)} ({model_info.get("model_type", "unknown")})'],
            ['模型輸入表示法', model_info.get('representation', 'unknown')],
            ['樣本單位', 'Packet' if model_info.get('unit') == 'packet' else 'Flow'],
            ['封包總數', f'{self.session.packet_count:,}'],
            ['規則式告警', f'{self.session.alert_count:,}'],
            ['產生時間', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ]
        if model_info.get('preprocessor'):
            meta.append(['訓練期前處理器', str(model_info['preprocessor'])])
        if model_info.get('classifier_threshold') is not None:
            meta.append(['已知攻擊分類器門檻', f'{model_info["classifier_threshold"]:.8g}'])
        if self.session.project:
            meta.insert(2, ['專案', self.session.project.name])
        t = Table([[P(k, small), P(v, small)] for k, v in meta], colWidths=[4.6*cm, 12.0*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#eff6ff')),
            ('FONTNAME', (0,0), (0,-1), font_bold),
            ('GRID', (0,0), (-1,-1), 0.35, colors.HexColor('#d1d5db')),
            ('ROWBACKGROUNDS', (1,0), (1,-1), [colors.white, colors.HexColor('#f8fafc')]),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        story += [t, Spacer(1, 0.45*cm)]

        # 1. Decision basis
        story.append(PT('1. 判定依據與分析流程', h1))
        if model_info.get('unit') == 'packet':
            pipeline = ('PCAP → Scapy/PacketVisualizer → 欄位遮罩與固定 32×32 影像 → '
                        'CNN/VAE 推論 → 重建誤差與模型閾值比較 → Grad-CAM')
        else:
            pipeline = ('PCAP → Flow Converter → 訓練期 feature schema → 訓練期 scaler → '
                        '32×32 特徵影像 → CNN/VAE/Hybrid 推論 → 閾值/分類器判定 → Grad-CAM')
        story.append(PT(f'<b>輸入一致性：</b>{self._safe_text(pipeline)}', body))
        story.append(Spacer(1, 0.12*cm))
        story.append(PT(
            '<b>重要限制：</b>Grad-CAM 是局部解釋工具，圖中的高值區域代表模型對該輸入區域的梯度敏感度，'
            '不等同於「這個區域本身就是攻擊」；規則式告警則是已知流量型樣/門檻的獨立證據。', note
        ))
        story.append(Spacer(1, 0.28*cm))

        # 2. CNN results and real chart
        if self.cnn:
            story.append(PT('2. CNN / VAE 異常偵測結果', h1))
            story.append(PT(
                '本次 PCAP 為待分析輸入，除非另行提供可對應到每個樣本的人工標註，系統無法僅憑推論結果證明其真實標籤。'
                '因此本節的「正常／異常數」是依模型規則產生的預測分組，不是經人工確認的真實正常數／真實攻擊數；'
                '偵測結果應與規則告警、流量背景及熱力圖共同研判。', note
            ))
            cnn_rows = [
                ['指標', '結果'],
                ['異常閾值', f'{self.cnn.threshold:.8g}'],
                ['樣本總數', f'{self.cnn.normal_count + self.cnn.anomaly_count:,}'],
                ['正常樣本數', f'{self.cnn.normal_count:,}'],
                ['異常樣本數', f'{self.cnn.anomaly_count:,}'],
                ['正常預測組平均分數', f'{self.cnn.avg_normal_error:.8g}'],
                ['異常預測組平均分數', f'{self.cnn.avg_attack_error:.8g}'],
                ['異常比例', f'{self.cnn.detection_rate*100:.2f}%'],
            ]
            if self.cnn.known_attack_count or self.cnn.unknown_attack_count:
                cnn_rows += [
                    ['已知攻擊（分類器）', f'{self.cnn.known_attack_count:,}'],
                    ['未知/新型攻擊（VAE）', f'{self.cnn.unknown_attack_count:,}'],
                ]
            ct = Table([[P(x, small) for x in row] for row in cnn_rows], colWidths=[7*cm, 9.6*cm])
            ct.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f6feb')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), font_bold),
                ('GRID', (0,0), (-1,-1), 0.35, colors.HexColor('#d1d5db')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('PADDING', (0,0), (-1,-1), 5),
            ]))
            story += [ct, Spacer(1, 0.25*cm)]
            if model_info.get('model_type') == 'hybrid_semi':
                hybrid_reason = (
                    '<b>半監督式模型的判定方式：</b>模型先以分類器判定訓練期間已知的攻擊類別；若未被分類器辨識為已知攻擊，但 VAE 異常分數高於異常閾值，則列為「未知／新型攻擊」；兩條判定路徑都未觸發時才列為正常預測。'
                    f'本次已知攻擊 {self.cnn.known_attack_count:,} 個、未知／新型攻擊 {self.cnn.unknown_attack_count:,} 個、正常預測 {self.cnn.normal_count:,} 個。'
                    f' VAE 閾值={self.cnn.threshold:.6g}。分類器門檻=' +
                    (f'{model_info["classifier_threshold"]:.6g}。' if model_info.get('classifier_threshold') is not None else '未能由 checkpoint metadata 讀取，故未列數值。')
                )
                story.append(PT(hybrid_reason, body))
            else:
                story.append(PT(
                    f'<b>非監督式模型的判定方式：</b>每個樣本的重建／異常分數高於訓練時校準的閾值 {self.cnn.threshold:.6g}，即標記為模型預測異常；未超過閾值則標記為正常預測。異常數量依逐樣本分數與閾值比較統計，而不是由平均值推估。', body
                ))
            story.append(Spacer(1, 0.16*cm))

            if self.cnn.error_histogram:
                try:
                    story.append(PT('實際分數分布圖（分析階段產生）', h2))
                    story.append(self._pdf_image(self.cnn.error_histogram.path, 16.4*cm, 7.8*cm))
                    story.append(Spacer(1, 0.18*cm))
                except Exception:
                    pass

        # 3. Rule alerts
        story.append(PT('3. 規則式告警與已知型樣證據', h1))
        if self.alerts:
            attack_types = sorted({a.attack_type for a in self.alerts if a.attack_type})
            story.append(PT(
                f'本次規則式引擎共產生 <b>{len(self.alerts):,}</b> 筆告警。'
                + (f' 攻擊類型：{self._safe_text(", ".join(attack_types))}。' if attack_types else ''), body))
            try:
                story.append(self._pdf_image(self._plot_severity(self.alerts), 14.5*cm, 5.6*cm))
            except Exception:
                pass
            story.append(Spacer(1, 0.15*cm))
            rows = [['#','攻擊類型','嚴重度','來源 IP','目的 IP/Port','說明']]
            for i, a in enumerate(self.alerts[:30], 1):
                dst = f'{a.dst_ip}:{a.dst_port}' if a.dst_port else a.dst_ip
                rows.append([str(i), a.attack_type, a.severity, a.src_ip, dst, (a.detail or '')[:80]])
            at = Table([[P(x, small) for x in row] for row in rows],
                       colWidths=[0.7*cm, 2.8*cm, 1.9*cm, 2.9*cm, 3.3*cm, 4.8*cm])
            at.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f6feb')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), font_bold),
                ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#d1d5db')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('PADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(at)
        else:
            story.append(PT('本次規則式偵測沒有產生告警；這只表示未觸發目前規則引擎的已知型樣，不等同於對所有未知威脅的絕對保證。', body))

        # 4. Grad-CAM evidence
        story.append(PageBreak())
        story.append(PT('4. Grad-CAM 可解釋證據', h1))
        if self.gcam_imgs:
            story.append(PT(
                '本平台會嘗試產生 Grad-CAM、Grad-CAM++、Score-CAM 三種解釋圖。'
                '下列只列出本次實際生成並存檔的圖像；紅／黃區域表示模型對輸入局部區域較敏感，'
                '不等於該區域必然是惡意內容，必須與分數、訓練閾值及規則式告警共同解讀。', body
            ))
            present_variants = {g.variant for g in self.gcam_imgs}
            variant_names = {'gradcam': 'Grad-CAM', 'gradcam++': 'Grad-CAM++', 'scorecam': 'Score-CAM'}
            missing_variants = [variant_names[k] for k in ('gradcam', 'gradcam++', 'scorecam') if k not in present_variants]
            if missing_variants:
                story.append(PT(
                    '<b>未產生的解釋方法：</b>' + self._safe_text('、'.join(missing_variants)) +
                    '。這些方法本次沒有可附加的實際影像；請檢查 Celery／Django 任務日誌後重新執行。'
                    '本報告不會以其他熱力圖複製或偽造缺失方法的結果。', note
                ))
            for g in self.gcam_imgs[:6]:
                label = '異常樣本' if g.is_anomaly else '正常參考'
                variant_label = {'gradcam': 'Grad-CAM', 'gradcam++': 'Grad-CAM++', 'scorecam': 'Score-CAM'}.get(g.variant, g.variant)
                head = PT(
                    f'<b>{label}</b> · 樣本 #{g.packet_index} · 異常分數={g.recon_error:.8g} · 方法={variant_label}',
                    h2)
                story.append(head)
                img_file = None
                try:
                    if g.comparison_image:
                        img_file = g.comparison_image.path
                except Exception:
                    img_file = None
                if img_file and os.path.exists(img_file):
                    story.append(self._pdf_image(img_file, 15.8*cm, 7.0*cm))
                story.append(Spacer(1, 0.12*cm))
        else:
            story.append(PT('本 Session 尚未產生 Grad-CAM 影像，因此目前沒有 XAI 視覺證據可附於報告。請重新執行對應變體並再次匯出。', body))

        story.append(PT('Grad-CAM 解讀', h2))
        if model_info.get('unit') == 'packet':
            story.append(PT(
                '此模型使用封包位元組影像；熱力圖高值位置是影像中的敏感區域。由於 PacketVisualizer 會遮罩部分識別欄位，'
                '因此不應直接把座標解讀成原始 IP/Port 等欄位，而應視為模型對輸入影像局部結構的反應。', note
            ))
        else:
            story.append(PT(
                '此模型使用 Flow 統計特徵影像；熱力圖對應的是特徵影像座標，而不是原始 PCAP 位元組。'
                '因此它應與相同 checkpoint 的 feature schema / scaler 一起解讀。', note
            ))

        # 5. Final factual breakdown
        story.append(PT('5. 最終證據拆解', h1))
        conclusion_rows = [['證據來源', '本次觀察', '如何影響判定']]
        if self.cnn:
            if model_info.get('model_type') == 'hybrid_semi':
                conclusion_rows.append([
                    '半監督模型',
                    f'已知攻擊 {self.cnn.known_attack_count:,}；未知攻擊 {self.cnn.unknown_attack_count:,}；正常預測 {self.cnn.normal_count:,}',
                    '分類器偵測已知類別；VAE 閾值補捉分類器未辨識但重建分數異常的樣本',
                ])
            else:
                conclusion_rows.append([
                    'CNN/VAE',
                    f'{self.cnn.anomaly_count:,} 個樣本超過 threshold {self.cnn.threshold:.6g}',
                    '逐樣本分數高於訓練閾值 → 模型預測異常',
                ])
        conclusion_rows.append([
            '規則式引擎',
            f'{len(self.alerts):,} 筆告警',
            '若觸發已知攻擊型樣，提供獨立證據與攻擊類型線索',
        ])
        conclusion_rows.append([
            'Grad-CAM',
            f'{len(self.gcam_imgs):,} 張代表影像',
            '說明模型對輸入影像哪些局部區域較敏感，不直接提供攻擊名稱',
        ])
        final_table = Table([[P(x, small) for x in row] for row in conclusion_rows],
                            colWidths=[3.0*cm, 6.4*cm, 7.0*cm])
        final_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f6feb')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), font_bold),
            ('GRID', (0,0), (-1,-1), 0.35, colors.HexColor('#d1d5db')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(final_table)
        story.append(Spacer(1, 0.3*cm))
        story.append(PT(
            f'報告結論標籤：<b>{self._safe_text(evidence["status"])}</b>。此標籤是對本次規則式與 CNN 結果的摘要，'
            '不是對整個網路環境「絕對安全/絕對遭受攻擊」的保證。', note
        ))
        story.append(Spacer(1, 0.35*cm))
        story.append(PT(
            f'本報告由 VNAAP 自動產生｜Session #{self.session.pk}｜{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            ParagraphStyle('footer', parent=small, alignment=TA_CENTER, textColor=colors.HexColor('#6b7280'))
        ))

        doc.build(story)
        pdf_bytes = buf.getvalue()
        report = AnalysisReport(
            session=self.session,
            generated_by=self.user,
            format='pdf',
            file_size=len(pdf_bytes),
        )
        filename = f'report_{self.session.pk}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        report.file.save(filename, ContentFile(pdf_bytes), save=False)
        report.save()
        return report

    # ── JSON ─────────────────────────────────────────────────
    def _generate_json(self) -> AnalysisReport:
        model_info = self._model_info()
        evidence = self._evidence_summary(model_info)
        data = {
            'meta': {
                'session_id': self.session.pk,
                'label': self.session.label,
                'mode': self.session.mode,
                'task_status': self.session.task_status,
                'packet_count': self.session.packet_count,
                'alert_count': self.session.alert_count,
                'model_key': self.session.model_key,
                'model': model_info,
                'created_at': self.session.created_at.isoformat(),
                'finished_at': self.session.finished_at.isoformat() if self.session.finished_at else None,
                'generated_at': datetime.now().isoformat(),
                'generated_by': self.user.username,
            },
            'decision_summary': evidence,
            'cnn_result': None,
            'alerts': [],
            'gradcam': [],
        }
        if self.cnn:
            data['cnn_result'] = {
                'threshold': self.cnn.threshold,
                'normal_count': self.cnn.normal_count,
                'anomaly_count': self.cnn.anomaly_count,
                'avg_normal_error': self.cnn.avg_normal_error,
                'avg_attack_error': self.cnn.avg_attack_error,
                'detection_rate': self.cnn.detection_rate,
                'known_attack_count': self.cnn.known_attack_count,
                'unknown_attack_count': self.cnn.unknown_attack_count,
                'histogram_url': self.cnn.error_histogram.url if self.cnn.error_histogram else None,
            }
        for a in self.alerts:
            data['alerts'].append({
                'attack_type': a.attack_type,
                'severity': a.severity,
                'src_ip': a.src_ip,
                'dst_ip': a.dst_ip,
                'dst_port': a.dst_port,
                'detail': a.detail,
                'suggestion': a.suggestion,
                'timestamp': a.timestamp.isoformat(),
            })
        for g in self.gcam_imgs:
            data['gradcam'].append({
                'sample_index': g.packet_index,
                'variant': g.variant,
                'is_anomaly': g.is_anomaly,
                'recon_error': g.recon_error,
                'original_image': g.original_image.url if g.original_image else None,
                'heatmap_image': g.heatmap_image.url if g.heatmap_image else None,
                'comparison_image': g.comparison_image.url if g.comparison_image else None,
            })
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
        report = AnalysisReport(session=self.session, generated_by=self.user,
                                format='json', file_size=len(payload))
        report.file.save(
            f'report_{self.session.pk}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            ContentFile(payload), save=False,
        )
        report.save()
        return report

    # ── CSV ──────────────────────────────────────────────────
    def _generate_csv(self) -> AnalysisReport:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(['=== Session 摘要 ==='])
        writer.writerow(['Session ID', self.session.pk])
        writer.writerow(['標籤', self.session.label])
        writer.writerow(['模式', self.session.get_mode_display()])
        writer.writerow(['模型', self.session.model_key])
        writer.writerow(['封包數', self.session.packet_count])
        writer.writerow(['告警數', self.session.alert_count])
        writer.writerow(['產生時間', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow([])
        if self.cnn:
            writer.writerow(['=== CNN 分析結果 ==='])
            writer.writerow([
                '閾值','正常預測樣本','異常預測樣本','正常預測組平均分數','異常預測組平均分數',
                '偵測率','已知攻擊','未知/新型攻擊'
            ])
            writer.writerow([
                self.cnn.threshold, self.cnn.normal_count, self.cnn.anomaly_count,
                self.cnn.avg_normal_error, self.cnn.avg_attack_error,
                f'{self.cnn.detection_rate*100:.2f}%',
                self.cnn.known_attack_count, self.cnn.unknown_attack_count,
            ])
            writer.writerow([])
        writer.writerow(['=== 告警清單 ==='])
        writer.writerow(['#','攻擊類型','嚴重程度','來源 IP','目標 IP','目標 Port','說明','建議','時間'])
        for i, a in enumerate(self.alerts, 1):
            writer.writerow([
                i, a.attack_type, a.severity, a.src_ip, a.dst_ip,
                a.dst_port or '', a.detail, a.suggestion,
                a.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            ])
        payload = buf.getvalue().encode('utf-8-sig')
        report = AnalysisReport(session=self.session, generated_by=self.user,
                                format='csv', file_size=len(payload))
        report.file.save(
            f'report_{self.session.pk}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            ContentFile(payload), save=False,
        )
        report.save()
        return report
