# ============================================================
# session_manager.py - Session 管理模組（v2.1.0 新增）
#
# 功能：
#   - 每次執行自動建立獨立的時間戳記資料夾
#     格式：output/sessions/20260309_143022_live/
#   - 列出所有歷史 Session（時間、類型、檔案數、大小）
#   - 刪除指定 Session 或全部清空
#   - 設定當前 Session 路徑（供 storage.py 使用）
# ============================================================

import os
import shutil
from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)

# Session 根目錄
SESSIONS_DIR = os.path.join("output", "sessions")


class SessionManager:
    """
    Session 管理器

    使用方式：
        # 建立新 Session（每次測試開始前呼叫）
        sm = SessionManager()
        session_dir = sm.create("live")   # → output/sessions/20260309_143022_live/

        # 列出所有 Session
        sm.list_sessions()

        # 刪除指定 Session
        sm.delete("20260309_143022_live")

        # 清空所有 Session
        sm.clear_all()
    """

    def __init__(self):
        os.makedirs(SESSIONS_DIR, exist_ok=True)

    # ── 建立新 Session ────────────────────────────────────
    def create(self, mode: str = "session", label: str = "") -> str:
        """
        建立新 Session 資料夾

        Args:
            mode  : 模式名稱（live / pcap / custom）
            label : 自訂標籤，附加在資料夾名稱後（可選）

        Returns:
            str: Session 資料夾的完整路徑

        範例輸出路徑：
            output/sessions/20260309_143022_live/
            output/sessions/20260309_143022_pcap_syn_flood/
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [ts, mode]
        if label:
            # 將標籤中的特殊字元替換成底線
            safe_label = "".join(c if c.isalnum() or c in "-_" else "_"
                                 for c in label)[:30]
            parts.append(safe_label)

        session_name = "_".join(parts)
        session_dir  = os.path.join(SESSIONS_DIR, session_name)
        os.makedirs(session_dir, exist_ok=True)

        # 建立 session.info 檔案記錄 metadata
        info_path = os.path.join(session_dir, "session.info")
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"session_name={session_name}\n")
            f.write(f"mode={mode}\n")
            f.write(f"label={label}\n")
            f.write(f"created_at={datetime.now().isoformat()}\n")

        print(f"{Fore.CYAN}  [Session] 新 Session 已建立: {session_dir}{Style.RESET_ALL}")
        return session_dir

    # ── 列出所有 Session ──────────────────────────────────
    def list_sessions(self) -> list:
        """
        列出所有歷史 Session，顯示時間、類型、包含的檔案與大小

        Returns:
            list[dict]: Session 資訊列表
        """
        if not os.path.exists(SESSIONS_DIR):
            print(f"{Fore.YELLOW}  尚無任何 Session 紀錄{Style.RESET_ALL}")
            return []

        sessions = []
        entries  = sorted(os.listdir(SESSIONS_DIR), reverse=True)  # 最新優先

        if not entries:
            print(f"{Fore.YELLOW}  尚無任何 Session 紀錄{Style.RESET_ALL}")
            return []

        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"  歷史 Session 列表（共 {len(entries)} 筆）")
        print(f"{'='*70}{Style.RESET_ALL}")
        print(f"  {'#':<4} {'Session 名稱':<40} {'大小':>8}  {'檔案數':>6}")
        print(f"  {'─'*4} {'─'*40} {'─'*8}  {'─'*6}")

        for idx, name in enumerate(entries, 1):
            full_path = os.path.join(SESSIONS_DIR, name)
            if not os.path.isdir(full_path):
                continue

            # 計算大小與檔案數
            total_size  = 0
            total_files = 0
            for root, dirs, files in os.walk(full_path):
                for fname in files:
                    if fname == "session.info":
                        continue
                    fpath = os.path.join(root, fname)
                    total_size  += os.path.getsize(fpath)
                    total_files += 1

            size_str = self._format_size(total_size)
            info     = self._read_info(full_path)
            mode     = info.get("mode", "?")
            label    = info.get("label", "")
            created  = info.get("created_at", "")[:16].replace("T", " ")

            display_name = name[:40]
            print(f"  {idx:<4} {display_name:<40} {size_str:>8}  {total_files:>6} 個檔案")
            if label:
                print(f"       標籤: {label}  |  建立: {created}  |  模式: {mode}")

            sessions.append({
                "name":        name,
                "path":        full_path,
                "mode":        mode,
                "label":       label,
                "created_at":  created,
                "size_bytes":  total_size,
                "file_count":  total_files,
            })

        total_all = sum(s["size_bytes"] for s in sessions)
        print(f"\n  總計: {len(sessions)} 個 Session，"
              f"占用 {self._format_size(total_all)}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")

        return sessions

    # ── 刪除指定 Session ──────────────────────────────────
    def delete(self, session_name: str) -> bool:
        """
        刪除指定名稱的 Session 資料夾

        Args:
            session_name: Session 名稱（不含路徑）

        Returns:
            bool: 是否成功刪除
        """
        target = os.path.join(SESSIONS_DIR, session_name)

        if not os.path.exists(target):
            print(f"{Fore.RED}  找不到 Session: {session_name}{Style.RESET_ALL}")
            return False

        # 計算刪除前的大小
        size = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(target)
            for f in files
        )

        shutil.rmtree(target)
        print(f"{Fore.GREEN}  已刪除 Session: {session_name} "
              f"（釋放 {self._format_size(size)}）{Style.RESET_ALL}")
        return True

    # ── 刪除多個 Session（依序號）────────────────────────
    def delete_by_index(self, indices: list) -> int:
        """
        依列表中的序號（從 list_sessions 取得）刪除 Session

        Args:
            indices: 序號列表（1-based，例如 [1, 3, 5]）

        Returns:
            int: 成功刪除的數量
        """
        sessions = self._get_session_list()
        to_delete = []

        for idx in indices:
            if 1 <= idx <= len(sessions):
                to_delete.append(sessions[idx - 1])
            else:
                print(f"{Fore.YELLOW}  序號 {idx} 超出範圍（共 {len(sessions)} 個）{Style.RESET_ALL}")

        count = 0
        for name in to_delete:
            if self.delete(name):
                count += 1

        return count

    # ── 清空所有 Session ──────────────────────────────────
    def clear_all(self, confirm: bool = False) -> bool:
        """
        清空所有 Session 紀錄

        Args:
            confirm: 若為 True 則不再詢問確認（適合程式內部呼叫）

        Returns:
            bool: 是否執行清空
        """
        sessions = self._get_session_list()
        if not sessions:
            print(f"{Fore.YELLOW}  目前沒有任何 Session 可清空{Style.RESET_ALL}")
            return False

        total_size = sum(
            os.path.getsize(os.path.join(r, f))
            for name in sessions
            for r, _, files in os.walk(os.path.join(SESSIONS_DIR, name))
            for f in files
        )

        if not confirm:
            print(f"\n{Fore.YELLOW}  ⚠  即將刪除全部 {len(sessions)} 個 Session")
            print(f"     共 {self._format_size(total_size)} 的資料將被永久刪除")
            ans = input(f"  確認清空？(輸入 yes 確認) > {Style.RESET_ALL}").strip().lower()
            if ans != "yes":
                print(f"{Fore.CYAN}  已取消{Style.RESET_ALL}")
                return False

        shutil.rmtree(SESSIONS_DIR)
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        print(f"{Fore.GREEN}  已清空所有 Session（釋放 {self._format_size(total_size)}）{Style.RESET_ALL}")
        return True

    # ── 顯示 Session 詳細內容 ─────────────────────────────
    def show(self, session_name: str):
        """顯示指定 Session 的詳細檔案列表"""
        target = os.path.join(SESSIONS_DIR, session_name)
        if not os.path.exists(target):
            print(f"{Fore.RED}  找不到 Session: {session_name}{Style.RESET_ALL}")
            return

        info = self._read_info(target)
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"  Session: {session_name}")
        print(f"  模式: {info.get('mode','?')}  |  建立: {info.get('created_at','')[:16]}")
        if info.get("label"):
            print(f"  標籤: {info['label']}")
        print(f"{'='*60}{Style.RESET_ALL}")

        for fname in sorted(os.listdir(target)):
            if fname == "session.info":
                continue
            fpath = os.path.join(target, fname)
            size  = os.path.getsize(fpath)
            print(f"  📄 {fname:<35} {self._format_size(size):>8}")

        print()

    # ── 輔助方法 ──────────────────────────────────────────
    def _get_session_list(self) -> list:
        """回傳所有 Session 名稱列表（排序後）"""
        if not os.path.exists(SESSIONS_DIR):
            return []
        return sorted(
            [d for d in os.listdir(SESSIONS_DIR)
             if os.path.isdir(os.path.join(SESSIONS_DIR, d))],
            reverse=True
        )

    @staticmethod
    def _read_info(session_dir: str) -> dict:
        """讀取 session.info 檔案"""
        info_path = os.path.join(session_dir, "session.info")
        info = {}
        if os.path.exists(info_path):
            with open(info_path, encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        info[k] = v
        return info

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """位元組轉人類可讀格式"""
        if size_bytes == 0:
            return "0 B"
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
