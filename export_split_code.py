import os
from datetime import datetime

# ── 配置 ─────────────────────────────────────────────────
BACKEND_DIRS = ['analyzer', 'api', 'network_platform']
BACKEND_FILES = ['manage.py', 'requirements.txt']
BACKEND_OUTPUT = 'django_backend_system.txt'

FRONTEND_DIRS = ['templates']
FRONTEND_OUTPUT = 'django_frontend_templates.txt'

def collect_backend_files():
    collected = []
    for d in BACKEND_DIRS:
        if not os.path.exists(d): continue
        for root, dirs, files in os.walk(d):
            if 'migrations' in root or '__pycache__' in root: continue
            for f in files:
                if f.endswith('.py') and f != '__init__.py':
                    collected.append(os.path.join(root, f))
    for f in BACKEND_FILES:
        if os.path.exists(f): collected.append(f)
    return sorted(collected)

def collect_frontend_files():
    collected = []
    for d in FRONTEND_DIRS:
        if not os.path.exists(d): continue
        for root, dirs, files in os.walk(d):
            if '__pycache__' in root: continue
            for f in files:
                if f.endswith('.html'):
                    collected.append(os.path.join(root, f))
    return sorted(collected)

def export_to_file(file_list, output_path, title):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write("=" * 72 + "\n")
        out.write(f"  {title}\n")
        out.write(f"  匯出時間 : {timestamp}\n")
        out.write(f"  總檔案數 : {len(file_list)}\n")
        out.write("=" * 72 + "\n\n")
        
        out.write("[檔案目錄]\n")
        for i, f in enumerate(file_list, 1):
            out.write(f"  {i:02d}. {f}\n")
        out.write("\n" + "=" * 72 + "\n\n")
        
        for f in file_list:
            out.write(f"// FILE: {f}\n")
            out.write("-" * 40 + "\n")
            try:
                with open(f, 'r', encoding='utf-8') as src:
                    out.write(src.read())
            except Exception as e:
                out.write(f"  [讀取錯誤]: {e}\n")
            out.write("\n\n" + "=" * 72 + "\n\n")
    print(f"成功匯出: {output_path}")

if __name__ == "__main__":
    # 執行後端匯出
    backend_files = collect_backend_files()
    export_to_file(backend_files, BACKEND_OUTPUT, "Django 後端系統程式碼彙整 (Python/Logic)")
    
    # 執行前端匯出
    frontend_files = collect_frontend_files()
    export_to_file(frontend_files, FRONTEND_OUTPUT, "Django 前端模板與佈局彙整 (HTML/Templates)")
