# ============================================================
# storage.py - 資料儲存模組（v2.1.0 加強版）
#
# 加強功能：
#   - Session 支援：每次儲存可指定 session_dir 獨立資料夾
#   - CSV / JSON / SQLite 全部支援 session 路徑
#   - 自動建立輸出目錄
# ============================================================

import os
import json
import csv
import sqlite3
from datetime import datetime
from colorama import Fore, Style
from config import OUTPUT_DIR, OUTPUT_CSV, OUTPUT_JSON, OUTPUT_DB


class PacketStorage:
    """
    封包資料儲存類別，支援 CSV、JSON 與 SQLite 格式

    Session 使用方式：
        storage = PacketStorage(session_dir="output/sessions/20260309_143022_live")
        storage.save_csv(records)    # 儲存到 session 資料夾內
    """

    def __init__(self, session_dir: str = None):
        """
        Args:
            session_dir: Session 資料夾路徑（None 則使用預設 output/ 目錄）
        """
        self.session_dir = session_dir
        self.base_dir    = session_dir if session_dir else OUTPUT_DIR
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _resolve_path(self, filename: str, default_path: str) -> str:
        if self.session_dir:
            return os.path.join(self.session_dir, filename)
        return default_path

    # ── CSV ───────────────────────────────────────────────
    def save_csv(self, records: list, filename: str = "packets.csv") -> str:
        if not records:
            return None
        path = self._resolve_path(filename, OUTPUT_CSV)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"{Fore.GREEN}  CSV 已儲存: {path} ({len(records):,} 筆){Style.RESET_ALL}")
        return path

    def append_csv(self, records: list, filename: str = "packets.csv") -> str:
        """追加模式儲存 CSV

        若檔案已存在，追加記錄（不重寫 header）；
        若檔案不存在，建立新檔案（含 header）。

        Args:
            records  : 封包記錄列表
            filename : CSV 檔案名稱

        Returns:
            儲存的檔案路徑
        """
        if not records:
            return None
        path = self._resolve_path(filename, OUTPUT_CSV)
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerows(records)
        print(f"{Fore.GREEN}  CSV 已追加: {path} (+{len(records):,} 筆){Style.RESET_ALL}")
        return path

    # ── JSON ──────────────────────────────────────────────
    def save_json(self, records: list, filename: str = "packets.json") -> str:
        if not records:
            return None
        path = self._resolve_path(filename, OUTPUT_JSON)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at":  datetime.now().isoformat(),
                "session_dir":   self.session_dir or OUTPUT_DIR,
                "total_packets": len(records),
                "records":       records
            }, f, ensure_ascii=False, indent=2, default=str)
        print(f"{Fore.GREEN}  JSON 已儲存: {path} ({len(records):,} 筆){Style.RESET_ALL}")
        return path

    # ── SQLite ────────────────────────────────────────────
    def save_sqlite(self, records: list, filename: str = "packets.db") -> str:
        if not records:
            return None
        path = self._resolve_path(filename, OUTPUT_DB)
        columns = list(records[0].keys())

        def infer_type(val):
            if isinstance(val, int):   return "INTEGER"
            if isinstance(val, float): return "REAL"
            return "TEXT"

        sample   = records[0]
        col_defs = ", ".join(
            f'"{col}" {infer_type(sample.get(col))}' for col in columns
        )
        conn = sqlite3.connect(path)
        cur  = conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS packets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {col_defs}
            )
        """)
        for col in ["src_ip","dst_ip","protocol","src_port","dst_port","timestamp"]:
            if col in columns:
                try:
                    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{col} ON packets("{col}")')
                except sqlite3.OperationalError:
                    pass
        # [修正] 保留原始型別（int, float, str），只對非基本型別做 str 轉換
        # 原版將所有值轉為 str，導致 SQL 數值比較失效（如 WHERE dst_port > 1024）
        def _to_sql_value(val):
            """將 Python 值轉換為 SQLite 相容的型別"""
            if val is None:
                return None
            if isinstance(val, (int, float, str)):
                return val
            if isinstance(val, bool):
                return int(val)
            return str(val)

        rows = [
            tuple(_to_sql_value(r.get(col)) for col in columns)
            for r in records
        ]
        # [優化] 批次寫入：每 1000 筆 commit 一次，避免大量記錄時 transaction 過大
        BATCH_SIZE = 1000
        placeholders = ", ".join("?" for _ in columns)
        col_names    = ", ".join(f'"{c}"' for c in columns)
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            cur.executemany(
                f'INSERT INTO packets ({col_names}) VALUES ({placeholders})',
                batch
            )
            conn.commit()
        conn.close()
        print(f"{Fore.GREEN}  SQLite 已儲存: {path} ({len(records):,} 筆){Style.RESET_ALL}")
        return path

    # ── SQLite 查詢輔助 ───────────────────────────────────
    @staticmethod
    def query_sqlite(sql: str, db_path: str = None, params: tuple = ()) -> list:
        path = db_path or OUTPUT_DB
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到資料庫: {path}")
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
