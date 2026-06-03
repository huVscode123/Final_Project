# ============================================================
# core/gradcam/gradcam_reporter.py  - JSON / HTML 報告匯出模組
# ============================================================
"""
GradCAMReporter：將 Grad-CAM 分析結果匯出為結構化報告。

輸出格式：
  - gradcam_report.json  完整 JSON（統計摘要 + 每樣本資料）
  - gradcam_report.html  互動式 HTML（內嵌圖表 + 樣本排序表格）

設計目標：
  - JSON 格式可直接供 api/views.py 的 /api/cnn/analyze/ 端點讀取
  - HTML 不依賴 Jinja2（純 f-string），可直接在瀏覽器開啟
  - 圖表以 base64 內嵌，HTML 為單一自含檔案
"""

from __future__ import annotations

import os
import json
import base64
import datetime
from typing import Dict, List, Optional, Any

import numpy as np

# 使用相對 import（同套件內）
from .gradcam_analyzer import AnalysisReport, SampleResult


class GradCAMReporter:
    """
    Grad-CAM 分析報告匯出器。

    Args:
        output_dir   (str): 報告輸出目錄（建議: output/gradcam/<variant>/）
        model_name   (str): 模型名稱（顯示於報告標題）
        dataset_name (str): 資料集名稱

    使用範例::

        from core.gradcam import GradCAMReporter
        reporter = GradCAMReporter(output_dir="output/gradcam/gradcam")
        reporter.export_all(report, summary, image_paths)
    """

    def __init__(
        self,
        output_dir:   str = "output/gradcam",
        model_name:   str = "CNNAutoencoder",
        dataset_name: str = "simulate",
    ):
        self.output_dir   = output_dir
        self.model_name   = model_name
        self.dataset_name = dataset_name
        os.makedirs(output_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # JSON 報告
    # ══════════════════════════════════════════════════════════

    def save_json(
        self,
        report:       AnalysisReport,
        summary:      Dict,
        variant:      str = "gradcam",
        target_layer: str = "encoder_conv",
    ) -> str:
        """
        儲存完整 JSON 報告。

        JSON 結構：
          meta            → 模型 / 日期 / 參數
          summary         → 統計摘要
          feature_importance → 特徵重要性排序
          normal_samples  → 正常樣本資料列表
          attack_samples  → 攻擊樣本資料列表（依重建誤差降序排列）
        """
        def _to_dict(r: SampleResult) -> dict:
            return {
                "sample_idx":   r.sample_idx,
                "label":        r.label,
                "recon_score":  round(r.recon_score, 8),
                "cam_mean":     round(r.cam_mean, 6),
                "cam_max":      round(r.cam_max, 6),
                "top_k_pixels": [[int(rc[0]), int(rc[1])] for rc in r.top_k_pixels],
            }

        data = {
            "meta": {
                "model_name":   self.model_name,
                "dataset":      self.dataset_name,
                "variant":      variant,
                "target_layer": target_layer,
                "generated_at": datetime.datetime.now().isoformat(),
                "n_normal":     len(report.normal_results),
                "n_attack":     len(report.attack_results),
            },
            "summary": summary,
            "feature_importance": (
                [
                    {
                        "name": (report.feature_names[i]
                                 if report.feature_names and i < len(report.feature_names)
                                 else f"F{i}"),
                        "importance": round(float(v), 6),
                    }
                    for i, v in enumerate(report.feature_importance)
                ]
                if report.feature_importance is not None else []
            ),
            "normal_samples": [_to_dict(r) for r in report.normal_results],
            "attack_samples": sorted(
                [_to_dict(r) for r in report.attack_results],
                key=lambda x: x["recon_score"], reverse=True
            ),
        }

        path = os.path.join(self.output_dir, "gradcam_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[GradCAMReporter] JSON 報告已儲存至: {path}")
        return path

    # ══════════════════════════════════════════════════════════
    # HTML 報告
    # ══════════════════════════════════════════════════════════

    def save_html(
        self,
        report:      AnalysisReport,
        summary:     Dict,
        image_paths: Optional[Dict[str, str]] = None,
        variant:     str = "gradcam",
    ) -> str:
        """
        儲存互動式 HTML 報告（純 f-string，無外部模板依賴）。

        Args:
            report      : AnalysisReport
            summary     : 摘要字典（由 GradCAMAnalyzer.summarize() 產生）
            image_paths : { 圖表 key: 本地路徑 }（自動轉 base64 內嵌）
            variant     : Grad-CAM 變體名稱
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── 圖表 base64 嵌入 ──────────────────────────────────
        embedded = {}
        if image_paths:
            for key, path in image_paths.items():
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        embedded[key] = "data:image/png;base64," + base64.b64encode(f.read()).decode()

        # ── 摘要指標卡片 ──────────────────────────────────────
        def _card(label, value, color):
            return f"""<div class="metric-card">
                <div class="metric-value" style="color:{color}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>"""

        ns = summary.get("normal", {})
        as_ = summary.get("attack", {})
        cards = (
            _card("正常樣本數",  ns.get("count", "N/A"),                              "#3fb950")
          + _card("攻擊樣本數",  as_.get("count", "N/A"),                             "#ff6b6b")
          + _card("正常平均誤差", f"{ns.get('recon_score_mean', 0):.5f}",             "#58a6ff")
          + _card("攻擊平均誤差", f"{as_.get('recon_score_mean', 0):.5f}",            "#ff6b6b")
          + _card("正常 CAM 強度", f"{ns.get('cam_intensity_mean', 0):.4f}",          "#3fb950")
          + _card("攻擊 CAM 強度", f"{as_.get('cam_intensity_mean', 0):.4f}",         "#ff6b6b")
        )

        # ── 特徵重要性表格 ────────────────────────────────────
        feat_rows = ""
        max_imp = max((abs(f.get("importance", 0)) for f in summary.get("top5_important_features", [])), default=1e-9)
        max_imp = max_imp if max_imp > 1e-9 else 1e-9
        for rank, feat in enumerate(summary.get("top5_important_features", []), 1):
            imp   = feat.get("importance", 0)
            bar_w = min(100, abs(imp) / max_imp * 100)
            color = "#ff6b6b" if imp > 0 else "#58a6ff"
            feat_rows += f"""<tr>
                <td>{rank}</td><td><strong>{feat.get("name","")}</strong></td>
                <td><div class="bar-bg"><div class="bar-fill" style="width:{bar_w:.1f}%;background:{color}"></div></div></td>
                <td style="color:{color}">{imp:+.5f}</td>
            </tr>"""

        # ── 攻擊樣本表格（Top 50） ───────────────────────────
        attack_rows = ""
        top_attacks = sorted(report.attack_results, key=lambda r: r.recon_score, reverse=True)[:50]
        for r in top_attacks:
            bar_w = min(100, r.recon_score * 1000)
            attack_rows += f"""<tr>
                <td>{r.sample_idx}</td>
                <td style="color:#ff6b6b">Attack</td>
                <td><div class="bar-bg"><div class="bar-fill" style="width:{bar_w:.1f}%;background:#ff6b6b"></div></div>{r.recon_score:.6f}</td>
                <td>{r.cam_mean:.4f}</td><td>{r.cam_max:.4f}</td>
            </tr>"""

        # ── 圖表區塊 ──────────────────────────────────────────
        img_sections = ""
        img_title_map = {
            "mean_cam_comparison":        "平均 CAM 對比（正常 vs 攻擊）",
            "feature_importance":         "特徵重要性分布",
            "cam_intensity_distribution": "CAM 強度分布",
            "score_vs_cam_intensity":     "重建誤差 vs CAM 強度散點圖",
            "gradcam_summary_grid":       "Grad-CAM 摘要網格",
        }
        for key, title in img_title_map.items():
            if key in embedded:
                img_sections += f"""<div class="section">
                    <h2>📊 {title}</h2>
                    <img src="{embedded[key]}" style="max-width:100%;border-radius:8px;">
                </div>"""

        html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Grad-CAM 分析報告 — {self.model_name}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,"PingFang TC","Microsoft YaHei",sans-serif;padding:32px}}
  h1{{color:#58a6ff;font-size:1.8rem;margin-bottom:8px}}
  h2{{color:#8b949e;font-size:1.1rem;margin:24px 0 12px}}
  .meta{{color:#8b949e;font-size:.85rem;margin-bottom:24px}}
  .metrics-row{{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:32px}}
  .metric-card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px 28px;min-width:160px;text-align:center}}
  .metric-value{{font-size:1.8rem;font-weight:700}}
  .metric-label{{color:#8b949e;font-size:.8rem;margin-top:6px}}
  .section{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:24px;margin-bottom:24px}}
  table{{width:100%;border-collapse:collapse;font-size:.85rem}}
  th,td{{padding:10px 14px;text-align:left;border-bottom:1px solid #21262d}}
  th{{color:#8b949e;background:#0d1117;position:sticky;top:0}}
  tr:hover{{background:#1c2128}}
  .bar-bg{{background:#21262d;border-radius:4px;height:8px;width:140px;display:inline-block;vertical-align:middle}}
  .bar-fill{{height:100%;border-radius:4px}}
  .scrollable{{max-height:420px;overflow-y:auto}}
</style>
</head>
<body>
<h1>🔍 Grad-CAM 視覺化解釋報告</h1>
<div class="meta">
  模型：<strong>{self.model_name}</strong> ｜
  資料集：<strong>{self.dataset_name}</strong> ｜
  變體：<strong>{variant}</strong> ｜
  生成時間：{timestamp}
</div>
<div class="metrics-row">{cards}</div>
{img_sections}
<div class="section">
  <h2>🏆 Top-5 重要特徵（攻擊 vs 正常 Grad-CAM 差異）</h2>
  <table><thead><tr><th>排名</th><th>特徵名稱</th><th>重要性</th><th>數值</th></tr></thead>
  <tbody>{feat_rows}</tbody></table>
</div>
<div class="section">
  <h2>⚠️ 高異常分數攻擊樣本 Top-50（依重建誤差降序）</h2>
  <div class="scrollable">
  <table><thead><tr><th>樣本索引</th><th>標籤</th><th>重建誤差</th><th>CAM 均值</th><th>CAM 最大值</th></tr></thead>
  <tbody>{attack_rows}</tbody></table>
  </div>
</div>
</body></html>"""

        path = os.path.join(self.output_dir, "gradcam_report.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[GradCAMReporter] HTML 報告已儲存至: {path}")
        return path

    # ══════════════════════════════════════════════════════════
    # 一鍵匯出全部
    # ══════════════════════════════════════════════════════════

    def export_all(
        self,
        report:       AnalysisReport,
        summary:      Dict,
        image_paths:  Optional[Dict[str, str]] = None,
        variant:      str = "gradcam",
        target_layer: str = "encoder_conv",
    ) -> Dict[str, str]:
        """
        一鍵匯出 JSON + HTML。

        Returns:
            {"json": path, "html": path}
        """
        return {
            "json": self.save_json(report, summary, variant, target_layer),
            "html": self.save_html(report, summary, image_paths, variant),
        }
