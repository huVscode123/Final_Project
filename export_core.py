import os
import argparse
from datetime import datetime

SKIP_FILENAMES = {
    "__init__.py",
}

SKIP_DIR_PARTS = {
    "__pycache__",
    ".pytest_cache",
}

def should_skip_file(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    filename = parts[-1]
    if filename in SKIP_FILENAMES:
        return True
    for part in parts:
        if part in SKIP_DIR_PARTS:
            return True
    return False

def collect_core_files(project_root: str) -> list[tuple[str, str]]:
    core_dir = os.path.join(project_root, "core")
    collected = []
    if not os.path.exists(core_dir):
        return []
        
    for dirpath, dirnames, filenames in os.walk(core_dir):
        dirnames[:] = [d for d in sorted(dirnames) if d not in SKIP_DIR_PARTS]
        for filename in sorted(filenames):
            if not (filename.endswith(".py") or filename.endswith(".md")):
                continue
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, project_root)
            if should_skip_file(rel_path):
                continue
            collected.append((rel_path, abs_path))
    return sorted(collected, key=lambda x: x[0])

def build_header(rel_path: str, total: int, index: int) -> str:
    bar = "=" * 72
    return f"\n{bar}\n  [{index:02d}/{total:02d}]  {rel_path}\n{bar}\n\n"

def export(project_root: str, output_path: str) -> None:
    files = collect_core_files(project_root)
    if not files:
        print("[export_core] No core Python source files found.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(files)

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("=" * 72 + "\n")
        out.write("  視覺化網路攻擊自動化分析平台 - 核心模組匯出\n")
        out.write("  Network Attack Automated Analysis Platform - Core Modules\n")
        out.write(f"  Source Export — {timestamp}\n")
        out.write(f"  Total Core files : {total}\n")
        out.write("=" * 72 + "\n")
        out.write("\n[Table of Contents]\n")
        for i, (rel, _) in enumerate(files, 1):
            out.write(f"  {i:02d}. {rel}\n")
        out.write("\n")

        for i, (rel_path, abs_path) in enumerate(files, 1):
            out.write(build_header(rel_path, total, i))
            try:
                with open(abs_path, "r", encoding="utf-8") as src:
                    out.write(src.read())
                    out.write("\n")
            except Exception as e:
                out.write(f"# [ERROR] Could not read file {rel_path}: {e}\n")

    print(f"[export_core] Export complete: {output_path} (Total: {total} files)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="core_source_export.txt")
    args = parser.parse_args()
    export(os.path.abspath(args.root), args.out)
