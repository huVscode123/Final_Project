# ============================================================
# reports/generators.py
# 分析報告匯出系統：PDF（ReportLab）/ JSON / CSV
# ============================================================
import os
import io
import json
import csv
from datetime import datetime

from django.conf import settings
from django.utils import timezone
from django.core.files.base import ContentFile

from analyzer.models import AnalysisSession, Alert, CNNResult, AnalysisReport


class ReportGenerator:
    """
    統一報告生成介面。

    使用方式：
        gen    = ReportGenerator(session, user)
        report = gen.generate('pdf')   # 回傳 AnalysisReport 物件
        report = gen.generate('json')
        report = gen.generate('csv')
    """

    def __init__(self, session: AnalysisSession, user):
        self.session = session
        self.user    = user
        self.alerts  = list(session.alerts.order_by('-timestamp'))
        self.cnn     = getattr(session, 'cnn_result', None)
        self.gcam_imgs = list(session.gradcam_images.filter(is_anomaly=True)[:5])

    def generate(self, fmt: str) -> AnalysisReport:
        fmt = fmt.lower()
        if fmt == 'pdf':
            return self._generate_pdf()
        elif fmt == 'json':
            return self._generate_json()
        elif fmt == 'csv':
            return self._generate_csv()
        else:
            raise ValueError(f'不支援的格式: {fmt}')

    # ── PDF ──────────────────────────────────────────────────
    def _generate_pdf(self) -> AnalysisReport:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.units import cm
            from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                            Table, TableStyle, Image as RLImage,
                                            HRFlowable, PageBreak)
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            raise ImportError('請安裝 reportlab：pip install reportlab')

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )

        styles = getSampleStyleSheet()
        # 自定義樣式
        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Title'],
            fontSize=20, spaceAfter=12, textColor=colors.HexColor('#1f6feb'),
        )
        h1_style = ParagraphStyle(
            'H1', parent=styles['Heading1'],
            fontSize=14, textColor=colors.HexColor('#0d1117'),
            borderPad=4, backColor=colors.HexColor('#e6f0ff'),
            leftIndent=6, spaceAfter=8,
        )
        h2_style = ParagraphStyle(
            'H2', parent=styles['Heading2'],
            fontSize=12, textColor=colors.HexColor('#1f6feb'),
        )
        body_style = styles['Normal']
        body_style.fontSize = 10

        severity_colors = {
            'CRITICAL': colors.HexColor('#f85149'),
            'HIGH':     colors.HexColor('#ff7b72'),
            'MEDIUM':   colors.HexColor('#d29922'),
            'LOW':      colors.HexColor('#3fb950'),
        }

        story = []

        # ── 封面 ──────────────────────────────────────────────
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph('網路攻擊分析報告', title_style))
        story.append(Paragraph(
            f'<font color="#8b949e">Network Security Analysis Report</font>',
            ParagraphStyle('sub', parent=styles['Normal'],
                           fontSize=12, alignment=TA_LEFT)))
        story.append(HRFlowable(width='100%', thickness=2,
                                color=colors.HexColor('#1f6feb')))
        story.append(Spacer(1, 0.5*cm))

        meta_data = [
            ['Session 名稱', self.session.label],
            ['分析模式',    self.session.get_mode_display()],
            ['分析狀態',    self.session.get_task_status_display()],
            ['封包數量',    str(self.session.packet_count)],
            ['告警總數',    str(self.session.alert_count)],
            ['產生時間',    datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['產生者',      self.user.get_full_name() or self.user.username],
        ]
        if self.session.project:
            meta_data.insert(1, ['所屬專案', self.session.project.name])

        meta_table = Table(meta_data, colWidths=[4*cm, 12*cm])
        meta_table.setStyle(TableStyle([
            ('FONTSIZE',  (0, 0), (-1, -1), 10),
            ('BACKGROUND',(0, 0), (0, -1), colors.HexColor('#f0f6ff')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1f6feb')),
            ('FONTNAME',  (0, 0), (0, -1), 'Helvetica-Bold'),
            ('GRID',      (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d7de')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1),
             [colors.white, colors.HexColor('#f6f8fa')]),
            ('PADDING',   (0, 0), (-1, -1), 6),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 0.8*cm))

        # ── CNN 分析結果 ──────────────────────────────────────
        if self.cnn:
            story.append(Paragraph('1. CNN Autoencoder 異常偵測結果', h1_style))
            cnn_data = [
                ['指標', '數值'],
                ['重建誤差閾值',   f'{self.cnn.threshold:.6f}'],
                ['正常封包數量',   str(self.cnn.normal_count)],
                ['異常封包數量',   str(self.cnn.anomaly_count)],
                ['平均正常誤差',   f'{self.cnn.avg_normal_error:.6f}'],
                ['平均異常誤差',   f'{self.cnn.avg_attack_error:.6f}'],
                ['偵測率',         f'{self.cnn.detection_rate*100:.2f}%'],
            ]
            cnn_table = Table(cnn_data, colWidths=[6*cm, 10*cm])
            cnn_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
                ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
                ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE',   (0, 0), (-1, -1), 10),
                ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d7de')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.white, colors.HexColor('#f6f8fa')]),
                ('PADDING',    (0, 0), (-1, -1), 6),
            ]))
            story.append(cnn_table)
            story.append(Spacer(1, 0.6*cm))

            # Grad-CAM 影像（最多 3 張）
            if self.gcam_imgs:
                story.append(Paragraph('Grad-CAM 熱力圖（異常封包）', h2_style))
                img_row = []
                for gcam in self.gcam_imgs[:3]:
                    if gcam.comparison_image:
                        try:
                            img_path = gcam.comparison_image.path
                            rl_img = RLImage(img_path, width=5*cm, height=2.5*cm)
                            img_row.append(rl_img)
                        except Exception:
                            pass
                if img_row:
                    while len(img_row) < 3:
                        img_row.append('')
                    img_table = Table([img_row], colWidths=[5.5*cm]*3)
                    img_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    story.append(img_table)
                story.append(Spacer(1, 0.6*cm))

        # ── 告警摘要 ──────────────────────────────────────────
        story.append(Paragraph('2. 告警偵測摘要', h1_style))

        # 嚴重程度統計
        sev_counts = {}
        for a in self.alerts:
            sev_counts[a.severity] = sev_counts.get(a.severity, 0) + 1

        if sev_counts:
            sev_data = [['嚴重程度', '數量']] + [
                [k, str(v)] for k, v in sorted(
                    sev_counts.items(),
                    key=lambda x: ['CRITICAL','HIGH','MEDIUM','LOW'].index(x[0])
                    if x[0] in ['CRITICAL','HIGH','MEDIUM','LOW'] else 99
                )
            ]
            sev_table = Table(sev_data, colWidths=[4*cm, 4*cm])
            sev_ts = [
                ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
                ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
                ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d7de')),
                ('PADDING',    (0, 0), (-1, -1), 6),
            ]
            for i, row in enumerate(sev_data[1:], 1):
                sev = row[0]
                c   = severity_colors.get(sev, colors.gray)
                sev_ts.append(('TEXTCOLOR', (0, i), (0, i), c))
                sev_ts.append(('FONTNAME',  (0, i), (0, i), 'Helvetica-Bold'))
            sev_table.setStyle(TableStyle(sev_ts))
            story.append(sev_table)
            story.append(Spacer(1, 0.4*cm))

        # 詳細告警清單（前 30 筆）
        if self.alerts:
            story.append(Paragraph('告警詳情（前 30 筆）', h2_style))
            alert_data = [['#', '類型', '嚴重程度', '來源 IP', '說明']]
            for i, a in enumerate(self.alerts[:30], 1):
                detail_short = (a.detail[:60] + '...') if len(a.detail) > 60 else a.detail
                alert_data.append([
                    str(i), a.attack_type, a.severity,
                    a.src_ip, detail_short,
                ])
            alert_table = Table(alert_data,
                                colWidths=[0.8*cm, 3.5*cm, 2.5*cm, 3*cm, 6.2*cm])
            a_ts = [
                ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
                ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
                ('FONTSIZE',   (0, 0), (-1, -1), 8),
                ('GRID',       (0, 0), (-1, -1), 0.3, colors.HexColor('#d0d7de')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.white, colors.HexColor('#f6f8fa')]),
                ('PADDING',    (0, 0), (-1, -1), 4),
                ('VALIGN',     (0, 0), (-1, -1), 'TOP'),
            ]
            for i, a in enumerate(self.alerts[:30], 1):
                c = severity_colors.get(a.severity, colors.gray)
                a_ts.append(('TEXTCOLOR', (2, i), (2, i), c))
                a_ts.append(('FONTNAME',  (2, i), (2, i), 'Helvetica-Bold'))
            alert_table.setStyle(TableStyle(a_ts))
            story.append(alert_table)

        # ── 頁尾 ──────────────────────────────────────────────
        story.append(PageBreak())
        story.append(Spacer(1, 5*cm))
        story.append(HRFlowable(width='100%', thickness=1,
                                color=colors.HexColor('#d0d7de')))
        story.append(Paragraph(
            f'本報告由網路攻擊分析平台自動產生｜{datetime.now().strftime("%Y-%m-%d")}',
            ParagraphStyle('footer', parent=styles['Normal'],
                           fontSize=8, textColor=colors.gray,
                           alignment=TA_CENTER)))

        doc.build(story)
        pdf_bytes = buf.getvalue()

        report = AnalysisReport(
            session=self.session, generated_by=self.user, format='pdf',
            file_size=len(pdf_bytes),
        )
        filename = (f'report_{self.session.pk}_'
                    f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
        report.file.save(filename, ContentFile(pdf_bytes), save=False)
        report.save()
        return report

    # ── JSON ─────────────────────────────────────────────────
    def _generate_json(self) -> AnalysisReport:
        data = {
            'meta': {
                'session_id':    self.session.pk,
                'label':         self.session.label,
                'mode':          self.session.mode,
                'task_status':   self.session.task_status,
                'packet_count':  self.session.packet_count,
                'alert_count':   self.session.alert_count,
                'project':       self.session.project.name if self.session.project else None,
                'created_at':    self.session.created_at.isoformat(),
                'finished_at':   self.session.finished_at.isoformat() if self.session.finished_at else None,
                'generated_at':  datetime.now().isoformat(),
                'generated_by':  self.user.username,
            },
            'cnn_result': None,
            'alerts':     [],
            'gradcam_summary': {
                'total_images':   self.session.gradcam_images.count(),
                'anomaly_images': self.session.gradcam_images.filter(is_anomaly=True).count(),
            },
        }

        if self.cnn:
            data['cnn_result'] = {
                'threshold':        self.cnn.threshold,
                'normal_count':     self.cnn.normal_count,
                'anomaly_count':    self.cnn.anomaly_count,
                'avg_normal_error': self.cnn.avg_normal_error,
                'avg_attack_error': self.cnn.avg_attack_error,
                'detection_rate':   self.cnn.detection_rate,
            }

        for a in self.alerts:
            data['alerts'].append({
                'attack_type': a.attack_type,
                'severity':    a.severity,
                'src_ip':      a.src_ip,
                'dst_ip':      a.dst_ip,
                'dst_port':    a.dst_port,
                'detail':      a.detail,
                'suggestion':  a.suggestion,
                'timestamp':   a.timestamp.isoformat(),
            })

        json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
        report = AnalysisReport(
            session=self.session, generated_by=self.user, format='json',
            file_size=len(json_bytes),
        )
        filename = (f'report_{self.session.pk}_'
                    f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        report.file.save(filename, ContentFile(json_bytes), save=False)
        report.save()
        return report

    # ── CSV ──────────────────────────────────────────────────
    def _generate_csv(self) -> AnalysisReport:
        buf = io.StringIO()
        writer = csv.writer(buf)

        # Session 摘要區塊
        writer.writerow(['=== Session 摘要 ==='])
        writer.writerow(['Session ID', self.session.pk])
        writer.writerow(['標籤',        self.session.label])
        writer.writerow(['模式',        self.session.get_mode_display()])
        writer.writerow(['封包數',      self.session.packet_count])
        writer.writerow(['告警數',      self.session.alert_count])
        writer.writerow(['產生時間',    datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow([])

        # CNN 結果
        if self.cnn:
            writer.writerow(['=== CNN 分析結果 ==='])
            writer.writerow(['閾值', '正常封包', '異常封包',
                             '平均正常誤差', '平均異常誤差', '偵測率'])
            writer.writerow([
                self.cnn.threshold,
                self.cnn.normal_count,
                self.cnn.anomaly_count,
                self.cnn.avg_normal_error,
                self.cnn.avg_attack_error,
                f'{self.cnn.detection_rate*100:.2f}%',
            ])
            writer.writerow([])

        # 告警清單
        writer.writerow(['=== 告警清單 ==='])
        writer.writerow(['#', '攻擊類型', '嚴重程度', '來源 IP',
                         '目標 IP', '目標 Port', '說明', '建議', '時間'])
        for i, a in enumerate(self.alerts, 1):
            writer.writerow([
                i, a.attack_type, a.severity,
                a.src_ip, a.dst_ip, a.dst_port or '',
                a.detail, a.suggestion,
                a.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            ])

        csv_bytes = buf.getvalue().encode('utf-8-sig')  # BOM for Excel
        report = AnalysisReport(
            session=self.session, generated_by=self.user, format='csv',
            file_size=len(csv_bytes),
        )
        filename = (f'report_{self.session.pk}_'
                    f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        report.file.save(filename, ContentFile(csv_bytes), save=False)
        report.save()
        return report
