# core/report_generator.py

import json
import os
from datetime import datetime


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_markdown(output_dir="output"):
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append("# 視覺化網路攻擊自動化分析平台 — 分析報告")
    lines.append(f"> 產生時間：{now}\n")

    # ── 1. 閾值調校報告 ──────────────────────────────────────────
    threshold_path = os.path.join(output_dir, "model", "threshold_report.json")
    if os.path.exists(threshold_path):
        data = load_json(threshold_path)
        lines.append("## 一、閾值調校結果\n")
        lines.append("| 百分位 | 閾值 | Precision | Recall | F1 | FPR | Accuracy |")
        lines.append("|--------|------|-----------|--------|----|-----|----------|")
        for row in data:
            lines.append(
                f"| {row['percentile']}% "
                f"| {row['threshold']:.6f} "
                f"| {row['precision']:.4f} "
                f"| {row['recall']:.4f} "
                f"| {row['f1']:.4f} "
                f"| {row['fpr']:.4f} "
                f"| {row['accuracy']:.4f} |"
            )
        best = max(data, key=lambda x: x["f1"])
        lines.append(
            f"\n **最佳閾值（F1 最高）**：百分位 {best['percentile']}%，"
            f"閾值 = `{best['threshold']:.6f}`，"
            f"F1 = `{best['f1']:.4f}`，"
            f"Precision = `{best['precision']:.4f}`，"
            f"Recall = `{best['recall']:.4f}`\n"
        )
    else:
        print(f"[Report] 找不到閾值報告：{threshold_path}，略過此區段")

    # ── 2. 消融實驗報告 ──────────────────────────────────────────
    ablation_path = os.path.join(output_dir, "ablation", "ablation_report.json")
    if os.path.exists(ablation_path):
        raw = load_json(ablation_path)
        # 相容兩種格式：{ "latent_dim": [...], ... } 或直接是 list
        if isinstance(raw, list):
            data = {"latent_dim": raw}
        else:
            data = raw

        lines.append("## 二、消融實驗結果\n")
        exp_labels = {
            "latent_dim":    "實驗 1：潛在空間維度（Latent Dim）",
            "arch_scale":    "實驗 2：架構寬度（Channel Scale）",
            "image_size":    "實驗 3：影像尺寸（Image Size）",
            "field_masking": "實驗 4：欄位遮罩（Field Masking）",
            "data_size":     "實驗 5：訓練資料量",
            "activation":    "實驗 6：激活函數",
        }
        for exp_key, exp_title in exp_labels.items():
            if exp_key not in data:
                continue
            rows = data[exp_key]
            if not rows:
                continue

            lines.append(f"### {exp_title}\n")

            # 動態偵測是否有 auc 欄位
            has_auc    = "auc"    in rows[0]
            has_params = "params" in rows[0]

            header = "| 設定 | Precision | Recall | F1 |"
            sep    = "|------|-----------|--------|----|"
            if has_auc:
                header += " AUC |"
                sep    += "-----|"
            if has_params:
                header += " 參數量 |"
                sep    += "--------|"
            lines.append(header)
            lines.append(sep)

            best_f1 = max(rows, key=lambda x: float(x.get("f1", 0)))
            for row in rows:
                marker = " " if row.get("name") == best_f1.get("name") else ""
                line = (
                    f"| {row.get('name', 'N/A')}{marker} "
                    f"| {float(row.get('precision', 0)):.4f} "
                    f"| {float(row.get('recall', 0)):.4f} "
                    f"| {float(row.get('f1', 0)):.4f} |"
                )
                if has_auc:
                    line += f" {float(row.get('auc', 0)):.4f} |"
                if has_params:
                    line += f" {int(row.get('params', 0)):,} |"
                lines.append(line)
            lines.append("")
    else:
        print(f"[Report] 找不到消融實驗報告：{ablation_path}，略過此區段")

    # ── 3. 模型壓縮報告 ──────────────────────────────────────────
    compress_path = os.path.join(output_dir, "compression", "compression_report.json")
    if os.path.exists(compress_path):
        raw     = load_json(compress_path)
        meta    = raw.get("meta", {})
        results = raw.get("results", raw)  # 相容有無 meta 包裝

        lines.append("## 三、模型壓縮評估結果\n")

        if meta:
            lines.append(f"- Baseline 參數量：`{int(meta.get('baseline_params', 0)):,}`")
            lines.append(f"- Baseline 大小：`{float(meta.get('baseline_size_mb', 0)):.2f} MB`")
            lines.append(f"- Baseline F1：`{float(meta.get('baseline_f1', 0)):.4f}`")
            if "baseline_auc" in meta:
                lines.append(f"- Baseline AUC：`{float(meta.get('baseline_auc', 0)):.4f}`")
            lines.append("")

        # 動態偵測 baseline 是否有 auc
        b = results.get("baseline", {})
        has_auc = "auc" in b

        header = "| 方法 | 壓縮率 | F1 |"
        sep    = "|------|--------|----|"
        if has_auc:
            header += " AUC |"
            sep    += "-----|"
        header += " 推論延遲 (ms) | 大小 (MB) |"
        sep    += "--------------|----------|"
        lines.append(header)
        lines.append(sep)

        # Baseline 列
        if b:
            line = (
                f"| Baseline（原始） | 1.0x "
                f"| {float(b.get('f1', 0)):.4f} |"
            )
            if has_auc:
                line += f" {float(b.get('auc', 0)):.4f} |"
            line += (
                f" {float(b.get('latency_ms', 0)):.2f} "
                f"| {float(b.get('size_mb', 0)):.2f} |"
            )
            lines.append(line)

        method_map = {
            "unstructured": "非結構化剪枝",
            "structured":   "結構化剪枝",
            "distillation": "知識蒸餾",
        }
        for key, label in method_map.items():
            for row in results.get(key, []):
                cr   = row.get("compression_ratio", "N/A")
                name = row.get("name", "").split()[-1]
                line = (
                    f"| {label} {name} "
                    f"| {cr}x "
                    f"| {float(row.get('f1', 0)):.4f} |"
                )
                if has_auc:
                    line += f" {float(row.get('auc', 0)):.4f} |"
                line += (
                    f" {float(row.get('latency_ms', 0)):.2f} "
                    f"| {float(row.get('size_mb', 0)):.2f} |"
                )
                lines.append(line)

        ptq = results.get("ptq")
        if ptq:
            cr   = ptq.get("compression_ratio", "N/A")
            line = f"| PTQ INT8 | {cr}x | {float(ptq.get('f1', 0)):.4f} |"
            if has_auc:
                line += f" {float(ptq.get('auc', 0)):.4f} |"
            line += (
                f" {float(ptq.get('latency_ms', 0)):.2f} "
                f"| {float(ptq.get('size_mb', 0)):.2f} |"
            )
            lines.append(line)
        lines.append("")
    else:
        print(f"[Report] 找不到壓縮評估報告：{compress_path}，略過此區段")

    # ── 4. 訓練策略比較報告 ──────────────────────────────────────
    benchmark_path = os.path.join(output_dir, "benchmark", "benchmark_report.json")
    if os.path.exists(benchmark_path):
        data = load_json(benchmark_path)

        # 動態偵測欄位
        sample = data.get("unsupervised", data.get("semi_supervised", {}))
        has_auc     = "auc"       in sample
        has_sep     = "sep_ratio" in sample
        has_time    = "train_time" in sample

        lines.append("## 四、訓練策略比較（非監督 vs 半監督）\n")

        header = "| 策略 | Precision | Recall | F1 |"
        sep    = "|------|-----------|--------|----|"
        if has_auc:
            header += " AUC |"
            sep    += "-----|"
        if has_sep:
            header += " 分離比 |"
            sep    += "--------|"
        if has_time:
            header += " 訓練時間(s) |"
            sep    += "------------|"
        lines.append(header)
        lines.append(sep)

        for key in ["unsupervised", "semi_supervised"]:
            r = data.get(key)
            if not r:
                continue
            line = (
                f"| {r.get('name', key)} "
                f"| {float(r.get('precision', 0)):.4f} "
                f"| {float(r.get('recall', 0)):.4f} "
                f"| {float(r.get('f1', 0)):.4f} |"
            )
            if has_auc:
                line += f" {float(r.get('auc', 0)):.4f} |"
            if has_sep:
                line += f" {float(r.get('sep_ratio', 0)):.2f} |"
            if has_time:
                line += f" {float(r.get('train_time', 0)):.1f} |"
            lines.append(line)
        lines.append("")
    else:
        print(f"[Report] 找不到訓練策略比較報告：{benchmark_path}，略過此區段")

    # ── 輸出 Markdown 檔案 ───────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "analysis_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Report] Markdown 報告已產生：{out_path}")
    return out_path


def generate_pdf_report(task_id, results, output_dir="media/reports"):
    """
    為 Django Task 生成 PDF 報告的存根 (Stub)。
    目前僅生成一個文字檔作為佔位符。
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"report_task_{task_id}.pdf")
    
    # 這裡應該使用 reportlab 或 fpdf 生成真正的 PDF
    # 目前先寫入簡單內容模擬生成成功
    with open(pdf_path, "w", encoding="utf-8") as f:
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Total Packets: {results.get('total_packets', 0)}\n")
        f.write(f"Anomalies: {len(results.get('anomalies', []))}\n")
    
    print(f"[Report] PDF 報告(存根)已產生：{pdf_path}")
    return pdf_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="將 JSON 分析結果轉換為 Markdown 報告")
    parser.add_argument("--output", default="output", help="output 目錄路徑（預設: output）")
    args = parser.parse_args()
    generate_markdown(args.output)