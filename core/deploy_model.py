# ============================================================
# core/deploy_model.py - 訓練成果部署小工具（新增檔案）
#
# 背景：
#   run_training.py 訓練出的 VAE 會存到
#     output/model_<dataset>/best_vae_<dataset>.pt
#   run_semi_supervised.py 訓練出的 Hybrid 半監督模型會存到
#     output/model_semi_<dataset>/semi_<dataset>.pt
#
#   但 Django（network_platform/settings.py::ANOMALY_MODELS）實際讀取的
#   路徑固定是：
#     media/model/best_vae_model.pt      (unsupervised_vae)
#     media/model/semi_cicddos2019.pt    (semi_cicddos2019)
#     media/model/semi_cicids2017.pt     (semi_cicids2017)
#     media/model/semi_nslkdd.pt         (semi_nslkdd)
#
#   兩邊路徑與檔名並不會自動對齊，需要手動複製＋改檔名，容易複製錯
#   檔案或忘記重新命名，導致「4 選 1」介面顯示模型已就緒，實際上
#   Celery worker 讀到的其實是上一次訓練的舊檔案。
#
#   本工具將「訓練輸出 -> Django media 目錄」的對應關係集中管理，
#   一行指令即可正確部署，避免上述人為疏失。
#
# 使用方式（於專案根目錄執行）：
#   python core/deploy_model.py unsupervised_vae output/model_cicids2017/best_vae_cicids2017.pt
#   python core/deploy_model.py semi_cicddos2019 output/model_semi_cicddos2019/semi_cicddos2019.pt
#   python core/deploy_model.py --list                # 列出目前 4 個模型的部署狀態
# ============================================================
from __future__ import annotations

import argparse
import os
import shutil
import sys

# 與 network_platform/settings.py::ANOMALY_MODELS 保持一致；
# 若日後在 settings.py 調整了路徑，請同步更新這裡（或改為直接
# import Django settings，但本工具刻意設計成不依賴 Django，
# 方便在還沒建好資料庫的訓練環境中也能執行）。
DEPLOY_TARGETS = {
    "unsupervised_vae": "media/model/best_vae_model.pt",
    "semi_cicddos2019":  "media/model/semi_cicddos2019.pt",
    "semi_cicids2017":   "media/model/semi_cicids2017.pt",
    "semi_nslkdd":        "media/model/semi_nslkdd.pt",
}


def deploy(model_key: str, src_path: str, project_root: str = "."):
    if model_key not in DEPLOY_TARGETS:
        raise ValueError(
            f"未知的 model_key: {model_key!r}，可用值: {list(DEPLOY_TARGETS)}"
        )
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"找不到來源檔案: {src_path}")

    dst_path = os.path.join(project_root, DEPLOY_TARGETS[model_key])
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    print(f"[deploy_model] {model_key}: {src_path}  ->  {dst_path}")
    return dst_path


def list_status(project_root: str = "."):
    print(f"{'model_key':<20} {'狀態':<8} 路徑")
    for key, rel_path in DEPLOY_TARGETS.items():
        path = os.path.join(project_root, rel_path)
        ready = os.path.exists(path)
        mark = "✅ 就緒" if ready else "❌ 未部署"
        print(f"{key:<20} {mark:<8} {path}")


def main():
    parser = argparse.ArgumentParser(description="將訓練輸出的模型部署到 Django media/model/")
    parser.add_argument("model_key", nargs="?", choices=list(DEPLOY_TARGETS),
                        help="unsupervised_vae / semi_cicddos2019 / semi_cicids2017 / semi_nslkdd")
    parser.add_argument("src_path", nargs="?", help="訓練輸出的 .pt 檔案路徑")
    parser.add_argument("--project-root", default=".", help="Django 專案根目錄（含 media/）")
    parser.add_argument("--list", action="store_true", help="列出 4 個模型目前的部署狀態")
    args = parser.parse_args()

    if args.list:
        list_status(args.project_root)
        return

    if not args.model_key or not args.src_path:
        parser.error("請提供 model_key 與 src_path，或使用 --list 查看目前狀態")

    deploy(args.model_key, args.src_path, args.project_root)


if __name__ == "__main__":
    main()
