# ============================================================
# analyzer/apps.py（修正版）
#
# ── 這次修正什麼、為什麼 ─────────────────────────────────────
# 症狀：「異常封包模擬檢測」只有第一次執行要等數十秒，之後每次
#       都不到 3 秒。
#
# 追查（不是看註解，是實際追程式碼呼叫鏈）：
#   simulation_api() 在 model_used 分支會
#       from packet_visualizer import PacketVisualizer
#   而 core/packet_visualizer.py 檔案「最上層」（模組被 import 的
#   當下、不是在任何函式裡）就會執行：
#       import matplotlib
#       import matplotlib.font_manager as _fm
#       ...
#       _cjk_font = _find_cjk_font()   # 內部呼叫 _fm.fontManager.ttflist
#   fontManager.ttflist 在該行程「第一次」被存取時，matplotlib 會去
#   掃描系統上所有字型並建立快取（這一步是公認會耗時 10~30 秒的
#   操作，尤其是 ttflist cache 檔案不存在或環境是全新容器時）。
#
#   而舊版 AnalyzerConfig.ready() 的預熱執行緒只做了：
#       import matplotlib
#       import matplotlib.pyplot as plt
#   完全沒有 import packet_visualizer，也沒有主動觸碰
#   matplotlib.font_manager.fontManager.ttflist，所以「模擬檢測」
#   頁面實際會用到的那段慢邏輯，從來沒有被真正預熱過 —— 它是在
#   使用者送出「第一次」模擬請求時，於請求執行緒內第一次觸發，
#   因此使用者感受到的是「第一次模擬」變慢，而不是「伺服器啟動」
#   變慢。第二次之後之所以只要 <3 秒，是因為：
#     (1) packet_visualizer 模組已經被 import 過（Python 模組快取），
#         不會重複掃字型。
#     (2) core/model_registry.py 的 _MODEL_CACHE 已經快取了模型。
#     (3) core/generate_attack_pcap.py 的封包產生邏輯是決定性
#         （deterministic）的，本身運算就很快。
#   這是「行程內快取」正常運作的結果，不是回傳了錯誤/過期的結果，
#   但呈現方式對使用者來說完全是黑箱（無法判斷「這次比較慢」是
#   正常現象還是系統異常）。
#
# 修正：
#   1. 預熱執行緒改為明確 import packet_visualizer，讓「模擬檢測」
#      真正用到的那段初始化成本，在伺服器啟動的背景執行緒裡付掉，
#      而不是留給第一位使用者。
#   2. 新增模組層級的 warmup_event（threading.Event）。
#      analyzer/views.py 的 simulation_api() 會把
#      warmup_event.is_set() 一併放進 API 回應（'server_warmup_complete'
#      欄位），前端可以據此顯示「伺服器正在背景預熱，本次較慢屬於
#      正常現象」的提示，把原本無法解釋的忽快忽慢，變成看得懂、
#      可驗證的資訊。
# ============================================================
import threading

from django.apps import AppConfig

# 伺服器啟動後，預熱執行緒完成時會 set() 這個旗標。
# analyzer/views.py 會在 simulation_api() 的回應中附上這個狀態。
warmup_event = threading.Event()


class AnalyzerConfig(AppConfig):
    name = 'analyzer'
    verbose_name = '封包分析'

    def ready(self):
        """
        伺服器啟動時，在背景執行緒預熱一次性的重量級成本：
          1. matplotlib 字型快取建置，特別是 core/packet_visualizer.py
             模組最上層觸發的字型掃描（異常封包模擬檢測頁面第一次
             請求變慢的實際成因）。
          2. 預載所有 settings.ANOMALY_MODELS 模型到
             core/model_registry.py 的行程內快取。
        避免這些成本轉嫁到第一位使用者的請求上。

        注意：這是「盡力預熱」，不保證在伺服器開始接受請求前完成
        （Django ready() 本身不應該被寫成同步阻塞，那會拖慢部署/
        健康檢查）。若有使用者的請求恰好在預熱完成前送達，該請求
        仍會付出等同的初始化成本 —— 這是背景執行緒設計的已知取捨，
        API 回應中的 'server_warmup_complete' 欄位讓這個情況變得
        可見、可解釋，而不是繼續當黑箱。
        """

        def _warmup():
            # ── 1. matplotlib 字型快取 + packet_visualizer ──
            try:
                import sys
                import os
                from django.conf import settings

                core_dir = str(settings.BASE_DIR / 'core')
                if core_dir not in sys.path:
                    sys.path.insert(0, core_dir)

                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt  # noqa: F401

                # [本次修正的核心] 明確 import packet_visualizer。
                # simulation_api() 與 analyzer/tasks.py::_run_cnn_analysis()
                # 實際使用的就是這個模組，它在模組最上層就會做字型
                # 掃描；只 import matplotlib.pyplot 並不保證觸發同一段
                # 掃描邏輯，因此先前的預熱對「模擬檢測」頁面沒有真正
                # 生效過。
                import packet_visualizer  # noqa: F401
            except Exception:
                pass

            # ── 2. 預載模型 ──
            try:
                import sys
                import os
                from django.conf import settings

                core_dir = str(settings.BASE_DIR / 'core')
                if core_dir not in sys.path:
                    sys.path.insert(0, core_dir)

                import torch
                from model_registry import load_anomaly_model

                device = torch.device('cpu')
                for _key, cfg in settings.ANOMALY_MODELS.items():
                    model_path = str(cfg.get('path', ''))
                    if model_path and os.path.exists(model_path):
                        try:
                            load_anomaly_model(
                                model_path, device=device,
                                default_latent_dim=settings.CNN_LATENT_DIM,
                            )
                        except Exception:
                            continue
            except Exception:
                pass

            # ── 3. 標記預熱完成 ──
            warmup_event.set()

        threading.Thread(target=_warmup, daemon=True, name='vnaap-warmup').start()
