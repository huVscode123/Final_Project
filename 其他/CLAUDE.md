# CLAUDE.md — Network Attack Automated Analysis Platform
# 視覺化網路攻擊自動化分析平台

> **Place this file at the project root** so Claude Code automatically loads it as context.

---

## Language & Interaction Guidelines

- **CRITICAL DIRECTIVE:** You **MUST ALWAYS** reply in **Traditional Chinese (繁體中文)**, regardless of the language used by the user in the prompt.
- **NO EMOJIS:** 禁止在程式碼或回覆中加入 emoji 和圖案（例如："💡"）。
- Keep explanations concise, technical, and directly applicable to the codebase.
- Provide targeted code modifications rather than outputting entire unmodified files.

---

## Project Overview

This project is a **Visualized Network Attack Automated Analysis Platform** built with:

| Layer | Technology |
|---|---|
| Web framework | Django 4.2 + Django REST Framework |
| Packet capture | Scapy 2.5 |
| Deep learning | PyTorch 2.0 (CPU/GPU) — CNN Autoencoder |
| Visualization | Matplotlib, Plotly, Chart.js |
| Datasets | NSL-KDD, CIC-IDS2017, CIC-DDoS2019 |
| Frontend | Tailwind CSS, Chart.js |
| AI Agent | n8n + LLM APIs |

### Source File Map

```text
network_platform/          ← Django project root
├── network_platform/      ← Project settings
│   ├── settings.py        ← CNN_MODEL_PATH, CNN_THRESHOLD, N8N_WEBHOOK_URL
│   └── urls.py
├── analyzer/              ← Frontend pages App
│   ├── models.py          ← AnalysisSession / Alert / CNNResult (DB models)
│   ├── views.py           ← Dashboard / Upload / Live / AI-chat views
│   └── urls.py
├── api/                   ← REST API App
│   ├── views.py           ← /api/sessions/ /api/cnn/ /api/ai/
│   └── urls.py
└── core/                  ← Core engine (Automated Analysis & AI)
    ├── config.py              Global thresholds & protocol maps
    ├── parser.py              PacketParser — Ethernet→IP→TCP/UDP/ICMP
    ├── anomaly_detector.py    AnomalyDetector — Rule-based (SYN Flood, etc.)
    ├── capture.py             LiveCapture — Multi-thread real-time capture
    ├── pcap_analyzer.py       PcapAnalyzer — Offline PCAP analysis
    ├── packet_visualizer.py   PacketVisualizer — bytes → 32×32 grayscale matrix
    ├── dataset_builder.py     DatasetBuilder — batch PCAP → .npy
    ├── dataset_loader.py      DatasetFactory — NSL-KDD / CICIDS2017 / DDoS2019
    ├── cnn_autoencoder.py     CNNAutoencoder — Encoder/Decoder architecture
    ├── trainer.py             Trainer — Unsupervised MSE training
    ├── semi_supervised_trainer.py  SemiSupervisedTrainer — Normal + Attack labels
    ├── anomaly_scorer.py      AnomalyScorer — scoring & evaluation
    ├── threshold_tuner.py     ThresholdTuner — percentile scanning
    ├── threshold_tuner_semi.py ThresholdTunerSemi — for semi-supervised models
    ├── ablation_study.py      AblationStudy — 6-dimension latent space testing
    ├── model_compression.py   ModelCompression — Pruning, PTQ, Distillation
    ├── comparison_benchmark.py Benchmark — Semi vs Unsupervised comparison
    ├── report_generator.py    ReportGenerator — Automated PDF/Markdown reports
    ├── run_full_pipeline.py   FullPipeline — Training to threshold tuning
    ├── run_training.py        Training script (Unsupervised)
    ├── run_semi_supervised.py Training script (Semi-supervised)
    ├── run_threshold_tuning.py Integration script (Ablation & Compression)
    ├── storage.py             CSV / JSON / SQLite storage logic
    ├── session_manager.py     Session directory & cleanup management
    ├── cnn_gradcam/           Grad-CAM Visualization Package
    │   ├── gradcam_core.py    Algorithms: mse/pixel/channel + variants
    │   ├── gradcam_hooks.py   Forward/Backward hook management
    │   ├── gradcam_analyzer.py Batch analysis & feature ranking
    │   ├── gradcam_visualizer.py Heatmap generation & overlay
    │   ├── gradcam_reporter.py  JSON/HTML/PDF report export
    │   └── run_gradcam.py     Execution entry point for Grad-CAM
    └── tests/                 Unit & Integration Tests
        ├── test_parser.py / test_anomaly.py
        ├── test_cnn_model.py / test_semi_supervised.py
        ├── test_gradcam_core.py / test_gradcam_integration.py
        └── run_gradcam_tests.py
```

---

## Key Architecture Decisions

### 1. Convolutional Autoencoder (CAE)
- **Unsupervised Learning:** Trains exclusively on normal traffic to establish a baseline, enabling the detection of **zero-day attacks** without requiring prior attack samples.
- **Spatial Feature Extraction:** Convolutional layers capture spatial byte-distribution patterns in the converted 32×32 packet image (e.g., rigid TCP header structures, payload entropy).
- **Semi-supervised Expansion:** Employs a small subset of labeled data to fine-tune detection boundaries and minimize false positives.

### 2. Model Optimization & Robustness
- **Ablation Study:** Systematically evaluates latent dimensions (8, 16, 32, 64) to identify the optimal tradeoff between compression ratio and reconstruction fidelity.
- **Edge Deployment Optimization:** Implements L1/L2 Pruning, Post-Training Quantization (PTQ), and Knowledge Distillation to reduce model footprint.

### 3. Grad-CAM for Explainable AI (XAI)
Provides transparency by backpropagating reconstruction error gradients to the original input pixels. This generates a gradient magnitude heatmap over the packet image, highlighting the specific byte regions responsible for triggering an anomaly.
- **MSE CAM:** Visualizes the overall error distribution.
- **Pixel CAM:** Isolates and highlights pixel-level importance.
- **Channel CAM:** Analyzes the respective contributions of different feature maps.

---

## Common Execution Commands

```bash
# 1. Full Pipeline (Train + Tune + Ablation)
python core/run_threshold_tuning.py --dataset cicids2017 --ablation --compress

# 2. Semi-Supervised Training
python core/run_semi_supervised.py --dataset cicids2017 --epochs 50

# 3. Grad-CAM Analysis
python core/cnn_gradcam/run_gradcam.py --model media/model/model.pth --input data/sample.pcap

# 4. Model Compression Evaluation
python core/model_compression.py --model media/model/model.pth --method quantize

# 5. Run All Tests
pytest core/tests/ -v
```

---

## Source Code Export Utility

To provide complete context to Claude when encountering complex architectural issues, use the `export_source.py` script. This bundles all relevant Python source files into a single text file (`network_platform_source_export.txt`).

Save it as `export_source.py` at the project root and run:

```bash
python export_source.py
```

```python
# export_source.py
# ============================================================
# Source Code Export Tool
# Collects all project Python source files into one TXT file.
#
# Excluded (environment / setup / boilerplate):
#   - requirements.txt       (package list, not source logic)
#   - manage.py              (Django boilerplate entry point)
#   - wsgi.py / asgi.py      (deployment interface only)
#   - __init__.py            (empty namespace markers)
#   - migrations/            (auto-generated DB migration files)
#
# Output: network_platform_source_export.txt
#
# Usage:
#   python export_source.py                  # export to default path
#   python export_source.py --out my.txt     # custom output path
#   python export_source.py --include-tests  # also include test_*.py
# ============================================================

import os
import argparse
from datetime import datetime

SKIP_FILENAMES = {
    "manage.py",       
    "wsgi.py",         
    "asgi.py",         
    "__init__.py",     
}

SKIP_DIR_PARTS = {
    "migrations",      
    "__pycache__",     
    ".git",            
    "venv", ".venv",   
    "node_modules",    
    "staticfiles",     
}

def should_skip_file(rel_path: str, include_tests: bool) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    filename = parts[-1]

    if filename in SKIP_FILENAMES:
        return True

    for part in parts[:-1]:
        if part in SKIP_DIR_PARTS:
            return True

    if not include_tests and filename.startswith("test_"):
        return True

    return False

def collect_python_files(root: str, include_tests: bool) -> list[tuple[str, str]]:
    collected = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames) if d not in SKIP_DIR_PARTS]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root)
            if should_skip_file(rel_path, include_tests):
                continue
            collected.append((rel_path, abs_path))
    return sorted(collected, key=lambda x: x[0])

def build_header(rel_path: str, total: int, index: int) -> str:
    bar = "=" * 72
    return f"\n{bar}\n  [{index:02d}/{total:02d}]  {rel_path}\n{bar}\n\n"

def export(root: str, output_path: str, include_tests: bool) -> None:
    files = collect_python_files(root, include_tests)
    if not files:
        print("[export_source] No Python source files found.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(files)

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("=" * 72 + "\n")
        out.write("  視覺化網路攻擊自動化分析平台\n")
        out.write("  Network Attack Automated Analysis Platform\n")
        out.write(f"  Source Export — {timestamp}\n")
        out.write(f"  Total files : {total}\n")
        out.write(f"  Project root: {root}\n")
        out.write("=" * 72 + "\n\n[Table of Contents]\n")
        
        for i, (rel, _) in enumerate(files, 1):
            out.write(f"  {i:02d}. {rel}\n")
        out.write("\n")

        for i, (rel_path, abs_path) in enumerate(files, 1):
            out.write(build_header(rel_path, total, i))
            try:
                with open(abs_path, "r", encoding="utf-8") as src:
                    content = src.read()
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
            except UnicodeDecodeError:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as src:
                    content = src.read()
                out.write(content)
                out.write("\n# [WARNING] This file contained non-UTF-8 bytes.\n")

    print(f"\n[export_source] Export complete! Files: {total}")

def main():
    parser = argparse.ArgumentParser(description="Export all Python source files to a single TXT file.")
    parser.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--out", default="network_platform_source_export.txt")
    parser.add_argument("--include-tests", action="store_true")
    args = parser.parse_args()
    export(os.path.abspath(args.root), args.out, args.include_tests)

if __name__ == "__main__":
    main()
```

---

## References & Inspiration

| Module | Reference Source |
|---|---|
| `cnn_autoencoder.py` | [pytorch-beginner](https://github.com/L1aoXingyu/pytorch-beginner) |
| `trainer.py` (EarlyStopping) | [early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch) |
| `trainer.py` (General Loop) | [pytorch/examples mnist](https://github.com/pytorch/examples/blob/main/mnist/main.py) |
| `trainer.py` (MedZoo EarlyStopping) | [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch) |
| `anomaly_scorer.py` (Grad-CAM) | [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) |
| `anomaly_scorer.py` (Anomaly Pattern)| [keras timeseries anomaly](https://github.com/keras-team/keras/blob/master/examples/timeseries_anomaly_detection.py) |
| `anomaly_scorer.py` (Tutorial Base) | [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) |
| Datasets | [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) · [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) · [CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html) |
| UI/UX Visual Design | *Refactoring UI* — Adam Wathan & Steve Schoger |

---

## Prompting Tips for Claude Code

When interacting with Claude regarding this codebase, use these patterns for the best context integration:

```text
# Ask about a specific module's internal logic
請解釋 threshold_tuner.py 的 scan_percentiles() 是如何運作的？

# Request code improvements with formatting constraints
請在不改變原本輸入輸出格式的情況下，為 anomaly_scorer.py 的 gradcam_packet() 加入 docstring。

# Debugging specific errors
以下錯誤發生在 dataset_loader.py，請找出原因並提供修正建議：
<paste only the relevant function + traceback>

# Request code reviews for potential issues
請審閱以下 trainer.py 的 compute_threshold() 方法，並指出效能或邏輯上的潛在問題：
<paste only that method>
```