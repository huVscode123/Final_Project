# GEMINI.md — Network Attack Automated Analysis Platform
# 視覺化網路攻擊自動化分析平台

> **Place this file at the project root** so Gemini CLI automatically loads it as context.

---

## Language & Model Settings

- **Always reply in Traditional Chinese (繁體中文)**, regardless of the language used in the prompt.
- Use **`gemini-2.0-flash`** (free tier) to conserve quota. Only escalate to `gemini-2.5-pro` for heavy reasoning tasks if needed.
- Run with quota-safe flags:

```bash
# Recommended startup — free tier, Traditional Chinese output
gemini --model gemini-2.0-flash

# If you hit rate limits (429), wait 60 s then retry
# Free tier limits: 15 RPM / 1,500 req/day / 1M TPM (gemini-2.0-flash)
```

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
| AI Agent | n8n + Google Gemini API (free tier) |

### Source File Map

```
network_platform/          ← Django project root
├── network_platform/      ← Project settings
│   ├── settings.py        ← CNN_MODEL_PATH, CNN_THRESHOLD configured here
│   └── urls.py
├── analyzer/              ← Frontend pages App
│   ├── models.py          ← AnalysisSession / Alert / CNNResult (DB models)
│   ├── views.py           ← Dashboard / Upload / Live / AI-chat views
│   └── urls.py
├── api/                   ← REST API App
│   ├── views.py           ← /api/sessions/ /api/cnn/ /api/ai/
│   └── urls.py
└── core/                  ← Core engine (fully migrated from CLI version)
    ├── config.py              Global thresholds & protocol maps
    ├── parser.py              PacketParser — Ethernet→IP→TCP/UDP/ICMP→DNS/TLS
    ├── anomaly_detector.py    AnomalyDetector — SYN Flood / Port Scan / ARP Spoof …
    ├── capture.py             LiveCapture — multi-thread real-time capture
    ├── pcap_analyzer.py       PcapAnalyzer — offline PCAP analysis
    ├── packet_visualizer.py   PacketVisualizer — bytes → 32×32 grayscale matrix
    ├── dataset_builder.py     DatasetBuilder — batch PCAP → .npy
    ├── dataset_loader.py      DatasetFactory — NSL-KDD / CICIDS2017 / DDoS2019
    ├── cnn_autoencoder.py     CNNAutoencoder — Encoder (4×Conv) + Decoder (4×TransposeConv)
    ├── trainer.py             Trainer — MSE loss, EarlyStopping, LR scheduler
    ├── anomaly_scorer.py      AnomalyScorer — score_packet / evaluate / Grad-CAM
    ├── threshold_tuner.py     ThresholdTuner — percentile scan + ablation study
    ├── run_training.py        End-to-end training script (--dataset flag)
    ├── run_threshold_tuning.py  Training + threshold tuning + ablation
    ├───storage.py             CSV / JSON / SQLite storage
    ├───session_manager.py     Session directory management
    ├───cleaner.py             Output cleanup tool
    ├───generate_attack_pcap.py  Test PCAP generator (SYN Flood / ARP Spoof …)
    └───tests/                 Unit tests
        ├───test_parser.py         45 unit tests — PacketParser
        ├───test_anomaly.py        17 unit tests — AnomalyDetector
        ├───test_visualization.py  PacketVisualizer tests
        ├───test_cnn_model.py      22 unit tests — CNNAutoencoder / Trainer / AnomalyScorer
        └───test_dataset_loader.py 37 unit tests — DatasetFactory / Loaders
```

---

## Key Architecture Decisions (for Gemini context)

### 1. Why Convolutional Autoencoder (CAE)?
- **Unsupervised learning**: only normal traffic is needed for training; attack samples are not required upfront — enabling **zero-day attack detection**.
- **Convolutional layers** capture spatial byte-distribution patterns in the 32×32 packet image (e.g., fixed TCP header structure, payload entropy).
- Reference: [pytorch-beginner](https://github.com/L1aoXingyu/pytorch-beginner)

### 2. Packet Visualization Pipeline
```
raw bytes → mask IP/Port/SeqNo → pad/truncate to 1024 bytes → reshape 32×32 → normalize [0,1]
```
Masking removes identity fields so the model learns **behavioral patterns**, not communication identities.

### 3. Threshold Strategy (ThresholdTuner)
| Environment | Metric target | Recommended percentile |
|---|---|---|
| High-risk (gov / finance) | Recall first | 80–85th |
| General monitoring | F1 balanced | 90–97th |
| Low false-positive | Precision first | 99th |

### 4. Grad-CAM for XAI
Backpropagates reconstruction error gradients to input pixels → gradient magnitude heatmap overlaid on packet image → shows **which byte region** triggered the anomaly flag.
Reference: [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

---

## Common Commands

```bash
# Train with real dataset
python core/run_training.py --dataset cicids2017 \
  --data-dir data/cicids2017 --epochs 100 --latent 32

# Threshold tuning + ablation study
python core/run_threshold_tuning.py --dataset cicids2017 \
  --data-dir data/cicids2017 --ablation

# Run all unit tests
pytest core/test_parser.py core/test_anomaly.py \
       core/test_cnn_model.py core/test_dataset_loader.py -v

# Start Django server
python manage.py runserver

# Generate test PCAPs
python core/generate_attack_pcap.py
```

---

## Quota Management Tips

> These rules help you stay within the **free tier** (15 RPM, 1 M TPM/day).

1. **Batch your questions** — ask multiple things in one prompt instead of sending rapid successive messages.
2. **Paste only relevant code** — copy a single module at a time; do not paste the entire codebase.
3. **Use `--model gemini-2.0-flash`** at CLI startup; do not switch to `gemini-2.5-pro` unless strictly necessary.
4. **Cache repeated context** — if you ask about the same module repeatedly, keep Gemini CLI open in the same session to reuse conversation context instead of re-loading.
5. **If you hit 429 (rate limit)**: wait 60 seconds and retry. The free tier resets per minute.

---

## Export All Python Source Code to TXT

The script below collects **all project Python source files** (excluding environment/setup files such as `requirements.txt`, `manage.py`, `wsgi.py`, `asgi.py`, `__init__.py`, and `migrations/`) and writes them into a single `network_platform_source_export.txt`.

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

# ── Files and directories to always skip ──────────────────
# References:
#   Django project structure: https://docs.djangoproject.com/en/4.2/intro/tutorial01/
SKIP_FILENAMES = {
    "manage.py",       # Django CLI boilerplate — not project logic
    "wsgi.py",         # WSGI deployment interface
    "asgi.py",         # ASGI deployment interface
    "__init__.py",     # Empty namespace marker
}

SKIP_DIR_PARTS = {
    "migrations",      # Auto-generated by Django ORM
    "__pycache__",     # Compiled bytecode cache
    ".git",            # Version control internals
    "venv", ".venv",   # Virtual environment
    "node_modules",    # JS dependencies (if any)
    "staticfiles",     # Collected static files
}


def should_skip_file(rel_path: str, include_tests: bool) -> bool:
    """
    Determine whether a file should be excluded from the export.

    Args:
        rel_path     : Path relative to project root (using '/' separator)
        include_tests: If False, skip files matching 'test_*.py'

    Returns:
        True  → skip this file
        False → include this file
    """
    parts = rel_path.replace("\\", "/").split("/")
    filename = parts[-1]

    # Skip by filename (setup / environment files)
    if filename in SKIP_FILENAMES:
        return True

    # Skip any file inside a blacklisted directory
    for part in parts[:-1]:
        if part in SKIP_DIR_PARTS:
            return True

    # Optionally skip test files
    if not include_tests and filename.startswith("test_"):
        return True

    return False


def collect_python_files(root: str, include_tests: bool) -> list[tuple[str, str]]:
    """
    Walk the project tree and collect all .py source files.

    Args:
        root         : Absolute path to project root directory
        include_tests: Whether to include test_*.py files

    Returns:
        List of (relative_path, absolute_path) tuples, sorted alphabetically
    """
    collected = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune blacklisted directories in-place (prevents os.walk from descending)
        # This is more efficient than checking every file path individually
        dirnames[:] = [
            d for d in sorted(dirnames)
            if d not in SKIP_DIR_PARTS
        ]

        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue  # Only collect Python source files

            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root)

            if should_skip_file(rel_path, include_tests):
                continue

            collected.append((rel_path, abs_path))

    return sorted(collected, key=lambda x: x[0])


def build_header(rel_path: str, total: int, index: int) -> str:
    """
    Build a separator header block for each file section in the TXT output.

    Args:
        rel_path : Relative file path shown in the header
        total    : Total number of files being exported
        index    : 1-based file index

    Returns:
        Multi-line string used as a visual separator
    """
    bar = "=" * 72
    return (
        f"\n{bar}\n"
        f"  [{index:02d}/{total:02d}]  {rel_path}\n"
        f"{bar}\n\n"
    )


def export(root: str, output_path: str, include_tests: bool) -> None:
    """
    Main export function: collects files and writes them to a single TXT.

    Args:
        root         : Project root directory
        output_path  : Destination .txt file path
        include_tests: Whether to include test_*.py files
    """
    files = collect_python_files(root, include_tests)

    if not files:
        print("[export_source] No Python source files found. Check project root.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(files)

    # ── Write output TXT ──────────────────────────────────
    # encoding="utf-8" is required because source files contain CJK comments
    with open(output_path, "w", encoding="utf-8") as out:

        # Document header
        out.write("=" * 72 + "\n")
        out.write("  視覺化網路攻擊自動化分析平台\n")
        out.write("  Network Attack Automated Analysis Platform\n")
        out.write(f"  Source Export — {timestamp}\n")
        out.write(f"  Total files : {total}\n")
        out.write(f"  Project root: {root}\n")
        out.write("=" * 72 + "\n")

        # Table of contents
        out.write("\n[Table of Contents]\n")
        for i, (rel, _) in enumerate(files, 1):
            out.write(f"  {i:02d}. {rel}\n")
        out.write("\n")

        # File contents
        for i, (rel_path, abs_path) in enumerate(files, 1):
            out.write(build_header(rel_path, total, i))
            try:
                with open(abs_path, "r", encoding="utf-8") as src:
                    content = src.read()
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
            except UnicodeDecodeError:
                # Fallback for files with unexpected encodings
                with open(abs_path, "r", encoding="utf-8", errors="replace") as src:
                    content = src.read()
                out.write(content)
                out.write("\n# [WARNING] This file contained non-UTF-8 bytes; "
                          "some characters may be replaced.\n")

    # ── Print summary ─────────────────────────────────────
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n[export_source] Export complete!")
    print(f"  Files    : {total}")
    print(f"  Output   : {os.path.abspath(output_path)}")
    print(f"  Size     : {size_kb:.1f} KB")
    print(f"\n  Files included:")
    for rel, _ in files:
        print(f"    {rel}")


def main():
    # ── Argument parser ───────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Export all Python source files to a single TXT file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_source.py
  python export_source.py --out my_export.txt
  python export_source.py --include-tests
  python export_source.py --root /path/to/project --out /tmp/export.txt
        """
    )
    parser.add_argument(
        "--root",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Project root directory (default: directory of this script)"
    )
    parser.add_argument(
        "--out",
        default="network_platform_source_export.txt",
        help="Output TXT file path (default: network_platform_source_export.txt)"
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Also include test_*.py files in the export"
    )
    args = parser.parse_args()

    export(
        root=os.path.abspath(args.root),
        output_path=args.out,
        include_tests=args.include_tests
    )


if __name__ == "__main__":
    main()
```

### What is excluded and why

| File / Pattern | Reason excluded |
|---|---|
| `requirements.txt` | Package list, not source logic |
| `manage.py` | Django-generated boilerplate entry point |
| `wsgi.py`, `asgi.py` | Deployment interface, not business logic |
| `__init__.py` | Empty namespace markers (no logic) |
| `migrations/` | Auto-generated by `python manage.py makemigrations` |
| `__pycache__/` | Compiled bytecode cache |
| `venv/`, `.venv/` | Virtual environment |
| `test_*.py` | Excluded by default; add `--include-tests` to keep them |

---

## References & GitHub

| Module | Reference |
|---|---|
| `cnn_autoencoder.py` | [pytorch-beginner](https://github.com/L1aoXingyu/pytorch-beginner) |
| `trainer.py` EarlyStopping | [early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch) |
| `trainer.py` general training loop | [pytorch/examples mnist](https://github.com/pytorch/examples/blob/main/mnist/main.py) |
| `trainer.py` MedicalZoo EarlyStopping design | [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch) |
| `anomaly_scorer.py` Grad-CAM | [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) |
| `anomaly_scorer.py` anomaly detection pattern | [keras timeseries anomaly](https://github.com/keras-team/keras/blob/master/examples/timeseries_anomaly_detection.py) |
| `anomaly_scorer.py` tutorial base | [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) |
| Datasets | [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) · [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) · [CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html) |
| Threshold UX design | *Information Dashboard Design* — Stephen Few |
| UI/UX visual design | *Refactoring UI* — Adam Wathan & Steve Schoger |

---

## Prompting Tips for This Project

When asking Gemini CLI about this codebase, use these patterns to stay within quota:

```
# Ask about a specific module
請解釋 threshold_tuner.py 的 scan_percentiles() 是如何工作的？

# Request code improvement with constraints
請在不增加太多 Token 的情況下，為 anomaly_scorer.py 的 gradcam_packet() 加入 docstring。

# Debug a specific error
以下錯誤發生在 dataset_loader.py，請找出原因並修正：
<paste only the relevant function + traceback>

# Ask for a code review
請審閱以下 trainer.py 的 compute_threshold() 方法，指出潛在問題：
<paste only that method>
```

> **Quota tip**: always paste the **smallest relevant snippet**, not entire files.
