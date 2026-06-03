# ============================================================
# cleaner.py - 輸出結果清空 / 初始化模組
#
# 功能：
#   清空或初始化 output/ 目錄下的各類輸出結果，
#   支援細粒度選擇（只清某一類）或一鍵全部清空。
#
#   可清除的類別：
#     sessions  - output/sessions/（Session 擷取紀錄）
#     dataset   - output/dataset*/（封包影像 .npy 資料集）
#     model     - output/model*/（CNN 模型權重與訓練圖表）
#     pcap      - output/captured.pcap 等擷取封包
#     csv       - output/*.csv、output/*.json、output/*.db
#     viz       - output/visualization_demo/（影像化展示圖）
#     all       - 以上全部
#
# 使用方式（CLI）：
#   python main.py clean --list              列出所有可清除項目與大小
#   python main.py clean sessions            清空 Session 紀錄
#   python main.py clean dataset             清空資料集 .npy 矩陣
#   python main.py clean model               清空 CNN 模型與訓練圖表
#   python main.py clean all                 清空全部（需確認）
#   python main.py clean all --yes           略過確認直接清空
#   python main.py clean sessions model --yes  同時清空多個類別
# ============================================================

import os
import shutil
import glob
from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)

# output/ 目錄的根路徑
OUTPUT_ROOT = "output"


class OutputCleaner:
    """
    輸出結果清理器

    使用方式：
        cleaner = OutputCleaner()
        cleaner.show_status()               # 列出所有可清除項目
        cleaner.clean("sessions")           # 清空 Session
        cleaner.clean("all", confirm=True)  # 清空全部（略過確認）
    """

    # 各類別的目錄/檔案掃描規則
    # 格式：{類別名: {"label": 顯示名稱, "patterns": [glob pattern]}}
    CATEGORIES = {
        "sessions": {
            "label":    "Session 擷取紀錄",
            "patterns": [os.path.join(OUTPUT_ROOT, "sessions")],
            "type":     "dir",
        },
        "dataset": {
            "label":    "封包影像資料集（.npy 矩陣）",
            "patterns": [
                os.path.join(OUTPUT_ROOT, "dataset*"),
            ],
            "type":     "glob_dirs",
        },
        "model": {
            "label":    "CNN 模型權重與訓練圖表",
            "patterns": [
                os.path.join(OUTPUT_ROOT, "model*"),
            ],
            "type":     "glob_dirs",
        },
        "pcap": {
            "label":    "擷取的 PCAP 封包檔案",
            "patterns": [
                os.path.join(OUTPUT_ROOT, "*.pcap"),
            ],
            "type":     "glob_files",
        },
        "csv": {
            "label":    "CSV / JSON / SQLite 分析結果",
            "patterns": [
                os.path.join(OUTPUT_ROOT, "*.csv"),
                os.path.join(OUTPUT_ROOT, "*.json"),
                os.path.join(OUTPUT_ROOT, "*.db"),
            ],
            "type":     "glob_files",
        },
        "viz": {
            "label":    "影像化展示圖（visualization_demo）",
            "patterns": [os.path.join(OUTPUT_ROOT, "visualization_demo")],
            "type":     "dir",
        },
    }

    def __init__(self, output_root: str = OUTPUT_ROOT):
        self.output_root = output_root

    # ── 列出狀態 ──────────────────────────────────────────
    def show_status(self):
        """
        列出所有可清除類別的目前狀態（大小、檔案數），
        讓使用者在清空前了解每個類別包含什麼。
        """
        print(f"\n{Fore.CYAN}{'='*65}")
        print(f"  輸出目錄狀態：{os.path.abspath(self.output_root)}")
        print(f"{'='*65}{Style.RESET_ALL}")

        total_size  = 0
        total_files = 0

        for cat_name, cat_info in self.CATEGORIES.items():
            items   = self._resolve_paths(cat_info)
            size    = sum(self._get_size(p) for p in items)
            n_files = sum(self._count_files(p) for p in items)

            total_size  += size
            total_files += n_files

            status = (f"{Fore.YELLOW}空{Style.RESET_ALL}"
                      if n_files == 0
                      else f"{n_files} 個檔案，{self._fmt_size(size)}")

            print(f"  {cat_name:<12} {cat_info['label']:<30} {status}")

        print(f"\n  {'合計':<12} {'所有類別':<30} "
              f"{total_files} 個檔案，{self._fmt_size(total_size)}")
        print(f"{Fore.CYAN}{'='*65}{Style.RESET_ALL}")
        print(f"\n  使用方式：")
        print(f"    python main.py clean --list              列出狀態")
        print(f"    python main.py clean sessions            清空 Session")
        print(f"    python main.py clean dataset model       清空多個類別")
        print(f"    python main.py clean all                 清空全部")
        print(f"    python main.py clean all --yes           略過確認直接清空\n")

    # ── 清空指定類別 ──────────────────────────────────────
    def clean(self, target: str, confirm: bool = False) -> bool:
        """
        清空指定類別的輸出結果

        Args:
            target : 類別名稱（sessions / dataset / model / pcap / csv / viz / all）
            confirm: True = 略過確認提示（批次操作時使用）
        Returns:
            bool: 是否執行清空
        """
        if target == "all":
            return self._clean_all(confirm)

        if target not in self.CATEGORIES:
            print(f"{Fore.RED}  未知類別：{target}")
            print(f"  可用類別：{', '.join(list(self.CATEGORIES.keys()) + ['all'])}"
                  f"{Style.RESET_ALL}")
            return False

        cat_info = self.CATEGORIES[target]
        items    = self._resolve_paths(cat_info)
        size     = sum(self._get_size(p) for p in items)
        n_files  = sum(self._count_files(p) for p in items)

        if n_files == 0:
            print(f"{Fore.YELLOW}  [{target}] 目前沒有任何內容，無需清空{Style.RESET_ALL}")
            return False

        # 確認提示
        if not confirm:
            print(f"\n{Fore.YELLOW}  即將清空：{cat_info['label']}")
            print(f"  共 {n_files} 個檔案，{self._fmt_size(size)}")
            ans = input(f"  確認清空 [{target}]？(輸入 yes 確認) > {Style.RESET_ALL}").strip().lower()
            if ans != "yes":
                print(f"{Fore.CYAN}  已取消{Style.RESET_ALL}")
                return False

        # 執行刪除
        deleted = 0
        for path in items:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    # 重新建立空目錄（保持目錄結構）
                    if target in ("sessions", "viz"):
                        os.makedirs(path, exist_ok=True)
                    deleted += 1
                elif os.path.isfile(path):
                    os.remove(path)
                    deleted += 1
            except Exception as e:
                print(f"{Fore.RED}  無法刪除 {path}：{e}{Style.RESET_ALL}")

        print(f"{Fore.GREEN}  [{target}] 已清空（{deleted} 個項目，釋放 {self._fmt_size(size)}）"
              f"{Style.RESET_ALL}")
        return True

    # ── 清空全部 ──────────────────────────────────────────
    def _clean_all(self, confirm: bool = False) -> bool:
        """清空所有類別"""
        # 先計算總量
        total_size  = sum(
            self._get_size(p)
            for cat in self.CATEGORIES.values()
            for p in self._resolve_paths(cat)
        )
        total_files = sum(
            self._count_files(p)
            for cat in self.CATEGORIES.values()
            for p in self._resolve_paths(cat)
        )

        if total_files == 0:
            print(f"{Fore.YELLOW}  output/ 目錄目前沒有任何內容{Style.RESET_ALL}")
            return False

        if not confirm:
            print(f"\n{Fore.YELLOW}  即將清空所有輸出結果")
            print(f"  共 {total_files} 個檔案，{self._fmt_size(total_size)}")
            ans = input(f"  確認清空全部？(輸入 yes 確認) > {Style.RESET_ALL}").strip().lower()
            if ans != "yes":
                print(f"{Fore.CYAN}  已取消{Style.RESET_ALL}")
                return False

        # 依序清空每個類別
        total_freed = 0
        for cat_name, cat_info in self.CATEGORIES.items():
            items = self._resolve_paths(cat_info)
            size  = sum(self._get_size(p) for p in items)
            for path in items:
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        if cat_name in ("sessions", "viz"):
                            os.makedirs(path, exist_ok=True)
                    elif os.path.isfile(path):
                        os.remove(path)
                except Exception as e:
                    print(f"{Fore.RED}  無法刪除 {path}：{e}{Style.RESET_ALL}")
            if size > 0:
                print(f"  [{cat_name}] 已清空 {self._fmt_size(size)}")
            total_freed += size

        print(f"\n{Fore.GREEN}  全部清空完成，共釋放 {self._fmt_size(total_freed)}{Style.RESET_ALL}")
        return True

    # ── 清空多個類別 ──────────────────────────────────────
    def clean_multiple(self, targets: list, confirm: bool = False) -> int:
        """
        同時清空多個類別

        Args:
            targets: 類別名稱列表（例：["sessions", "model"]）
            confirm: 略過確認
        Returns:
            int: 成功清空的類別數
        """
        # 先彙總要清空的內容再確認一次
        valid_targets = [t for t in targets if t in self.CATEGORIES]
        invalid = [t for t in targets if t not in self.CATEGORIES and t != "all"]
        if invalid:
            print(f"{Fore.YELLOW}  忽略未知類別：{', '.join(invalid)}{Style.RESET_ALL}")

        if not valid_targets:
            return 0

        # 計算彙總
        total_size  = sum(
            self._get_size(p)
            for t in valid_targets
            for p in self._resolve_paths(self.CATEGORIES[t])
        )
        total_files = sum(
            self._count_files(p)
            for t in valid_targets
            for p in self._resolve_paths(self.CATEGORIES[t])
        )

        if total_files == 0:
            print(f"{Fore.YELLOW}  選擇的類別目前沒有任何內容{Style.RESET_ALL}")
            return 0

        if not confirm:
            labels = "、".join(self.CATEGORIES[t]["label"] for t in valid_targets)
            print(f"\n{Fore.YELLOW}  即將清空：{labels}")
            print(f"  共 {total_files} 個檔案，{self._fmt_size(total_size)}")
            ans = input(f"  確認清空？(輸入 yes 確認) > {Style.RESET_ALL}").strip().lower()
            if ans != "yes":
                print(f"{Fore.CYAN}  已取消{Style.RESET_ALL}")
                return 0

        count = 0
        for t in valid_targets:
            if self.clean(t, confirm=True):
                count += 1
        return count

    # ── 路徑解析輔助 ─────────────────────────────────────
    def _resolve_paths(self, cat_info: dict) -> list:
        """將類別設定解析為實際存在的路徑清單"""
        paths = []
        cat_type = cat_info["type"]
        for pattern in cat_info["patterns"]:
            if cat_type == "dir":
                if os.path.isdir(pattern):
                    paths.append(pattern)
            elif cat_type == "glob_dirs":
                paths.extend(
                    p for p in glob.glob(pattern) if os.path.isdir(p)
                )
            elif cat_type == "glob_files":
                paths.extend(
                    p for p in glob.glob(pattern) if os.path.isfile(p)
                )
        return paths

    @staticmethod
    def _get_size(path: str) -> int:
        """取得目錄或檔案的總大小（bytes）"""
        if os.path.isfile(path):
            return os.path.getsize(path)
        if os.path.isdir(path):
            total = 0
            for root, _, files in os.walk(path):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            return total
        return 0

    @staticmethod
    def _count_files(path: str) -> int:
        """計算目錄或檔案的數量"""
        if os.path.isfile(path):
            return 1
        if os.path.isdir(path):
            count = 0
            for _, _, files in os.walk(path):
                count += len(files)
            return count
        return 0

    @staticmethod
    def _fmt_size(size_bytes: int) -> str:
        """位元組轉人類可讀格式"""
        if size_bytes == 0:
            return "0 B"
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
