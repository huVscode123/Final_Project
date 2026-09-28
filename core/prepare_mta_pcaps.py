#!/usr/bin/env python3
# ============================================================
# core/prepare_mta_pcaps.py（新增檔案）
#
# ── 為什麼需要這一步 ──────────────────────────────────────
# https://github.com/alcthomp/malware_traffic_analysis 這個 repo
# 是把 malware-traffic-analysis.net 訓練用範例「原樣」鏡射下來的
# 扁平 zip 集合，一個 commit、沒有 normal/ 或 malware/ 這種子目錄。
# 每個案例通常會有好幾個 zip，例如：
#   2016-06-03-traffic-analysis-exercise.pcap.zip          ← 真正的封包
#   2016-06-03-traffic-analysis-exercise-answers.pdf.zip   ← 解答文件（非封包）
#   2016-06-03-artifacts-from-infected-host.zip            ← 惡意軟體本身/樣本（非封包）
#   2016-06-03-traffic-analysis-exercise-suspicious-emails.zip ← 可疑郵件（非封包）
#
# 而且依照 malware-traffic-analysis.net 長年的慣例，含有真正封包
# 或惡意樣本的 zip 幾乎都會用密碼保護（避免信件/防毒掃描器誤刪），
# 慣用密碼固定是 "infected"。
#
# 本腳本做的事：
#   1. 遞迴掃描 --src 目錄下所有 .zip
#   2. 依檔名跳過明顯不是封包的 zip（-answers / -artifacts-from-infected-host）
#   3. 對其餘 zip 嘗試用候選密碼（預設含 "infected"）解壓
#   4. 只取出其中的 .pcap / .pcapng / .cap 檔案，攤平搬到 --out 目錄，
#      並用「來源 zip 檔名」當前綴命名，避免不同案例的檔名互相覆蓋
#   5. 印出成功/略過清單，方便回頭手動處理密碼失敗的少數案例
#
# 產出的 --out 目錄可以直接餵給 train_mta_malware.py 的 --malware-dir。
#
# 使用方式：
#   git clone https://github.com/alcthomp/malware_traffic_analysis.git
#   python core/prepare_mta_pcaps.py \
#       --src malware_traffic_analysis \
#       --out data/mta_pcaps
#
#   # 若還有個別 zip 用了其他密碼，可疊加候選密碼再跑一次：
#   python core/prepare_mta_pcaps.py \
#       --src malware_traffic_analysis --out data/mta_pcaps \
#       --password infected --password malware
# ============================================================

import argparse
import os
import shutil
import tempfile
import zipfile

# malware-traffic-analysis.net 長年慣用密碼 "infected"；
# "" 代表「先試試看不需要密碼」。
DEFAULT_PASSWORDS = ["infected", ""]
PCAP_EXTS = (".pcap", ".pcapng", ".cap")

# 檔名含這些關鍵字的 zip，內容通常是文件/樣本而非封包，直接跳過可省下大量時間，
# 也避免不小心把惡意軟體「本體」解壓到磁碟上。
_SKIP_NAME_HINTS = (
    "answers",
    "artifacts-from-infected-host",
    "suspicious-emails",
)


def _try_extract(zip_path: str, dest_dir: str, passwords: list):
    """嘗試用候選密碼清單解壓 zip 內的 pcap 成員，回傳 (取出的成員列表, 錯誤訊息或 None)。"""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        pcap_members = [n for n in names if n.lower().endswith(PCAP_EXTS)]
        if not pcap_members:
            return [], "zip 內找不到 .pcap/.pcapng/.cap"

        last_err = None
        for pwd in passwords:
            try:
                pwd_bytes = pwd.encode() if pwd else None
                for member in pcap_members:
                    zf.extract(member, dest_dir, pwd=pwd_bytes)
                return pcap_members, None
            except (RuntimeError, zipfile.BadZipFile, NotImplementedError) as e:
                last_err = e
                continue
        return [], f"所有候選密碼皆失敗（最後錯誤：{last_err}）"


def prepare(src_dir: str, out_dir: str, extra_passwords=None):
    os.makedirs(out_dir, exist_ok=True)
    # 保留順序去重：使用者指定的密碼優先於預設密碼
    passwords = list(dict.fromkeys((extra_passwords or []) + DEFAULT_PASSWORDS))

    zip_paths = []
    for root, _, files in os.walk(src_dir):
        for fn in files:
            if fn.lower().endswith(".zip"):
                zip_paths.append(os.path.join(root, fn))

    if not zip_paths:
        print(f"[prepare_mta_pcaps] 在 {src_dir} 找不到任何 .zip 檔案，"
              f"請確認 --src 指向 git clone 下來的 malware_traffic_analysis 目錄。")
        return

    print(f"[prepare_mta_pcaps] 找到 {len(zip_paths)} 個 zip，開始處理...")
    print(f"[prepare_mta_pcaps] 候選密碼：{[p or '(無密碼)' for p in passwords]}")

    n_ok, n_skip, n_pcaps = 0, 0, 0
    skipped = []

    with tempfile.TemporaryDirectory() as tmp:
        for zp in sorted(zip_paths):
            base = os.path.splitext(os.path.basename(zp))[0]

            if any(hint in base.lower() for hint in _SKIP_NAME_HINTS):
                skipped.append((base, "檔名判斷為非封包內容（writeup/artifacts/emails）"))
                n_skip += 1
                continue

            work = os.path.join(tmp, base)
            os.makedirs(work, exist_ok=True)

            try:
                members, err = _try_extract(zp, work, passwords)
            except zipfile.BadZipFile:
                skipped.append((base, "zip 檔本身已損毀，無法讀取"))
                n_skip += 1
                continue

            if err:
                skipped.append((base, err))
                n_skip += 1
                continue

            for root, _, files in os.walk(work):
                for fn in files:
                    if fn.lower().endswith(PCAP_EXTS):
                        src_path = os.path.join(root, fn)
                        dst_name = f"{base}__{fn}"
                        dst_path = os.path.join(out_dir, dst_name)
                        shutil.move(src_path, dst_path)
                        n_pcaps += 1
            n_ok += 1

    print(f"\n[prepare_mta_pcaps] 完成")
    print(f"  成功解壓縮的 zip ：{n_ok}")
    print(f"  略過的 zip       ：{n_skip}")
    print(f"  取出的 PCAP 檔案 ：{n_pcaps}")
    print(f"  輸出目錄         ：{os.path.abspath(out_dir)}")

    if skipped:
        print(f"\n  略過清單（最多顯示 20 筆）：")
        for name, reason in skipped[:20]:
            print(f"    - {name}: {reason}")
        if any("密碼皆失敗" in r for _, r in skipped):
            print(f"\n  [提示] 仍有因密碼錯誤被略過的 zip，"
                  f"可到該案例在 malware-traffic-analysis.net 的網頁確認密碼後，"
                  f"用 --password 加入候選再重跑一次。")

    if n_pcaps == 0:
        print(f"\n  [警告] 沒有取出任何 PCAP，train_mta_malware.py 的 "
              f"--malware-dir 將會是空的，請先排除上方的略過原因。")


def main():
    parser = argparse.ArgumentParser(
        description="從 malware_traffic_analysis repo 的扁平 zip 集合中取出 PCAP 檔案，"
                     "供 train_mta_malware.py 的 --malware-dir 使用。"
    )
    parser.add_argument("--src", required=True,
                        help="git clone 下來的 malware_traffic_analysis 目錄")
    parser.add_argument("--out", required=True,
                        help="輸出的扁平 PCAP 目錄")
    parser.add_argument("--password", action="append", default=None,
                        help="額外嘗試的 zip 密碼（可重複指定多次）；"
                             "預設已包含慣用密碼 'infected'")
    args = parser.parse_args()
    prepare(args.src, args.out, extra_passwords=args.password)


if __name__ == "__main__":
    main()
