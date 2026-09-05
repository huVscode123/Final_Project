# ============================================================
# core/model_registry.py - 異常偵測模型註冊表（修正版）
#
# ── 本次異動 ──────────────────────────────────────────────
# 新增 is_model_cached()：純粹的「查詢」函式，不會觸發載入，讓
# analyzer/views.py::simulation_api() 可以在「呼叫 load_anomaly_model()
# 之前」先問一句「這個模型現在是不是已經在快取裡」，藉此把
# 「第一次模擬很慢、之後很快」的效能落差，從使用者猜不透的黑箱，
# 變成 API 回應裡一個誠實的 model_was_cached: true/false 欄位。
#
# 其餘內容（load_anomaly_model / compute_anomaly_scores /
# AnomalyModelBundle / clear_model_cache 等）與原檔完全相同，
# 未變更任何既有行為。
# ============================================================
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
import threading

# ── 行程內模型快取 ─────────────────────────────────────────
# 以 (檔案路徑, mtime, size, device) 作為快取鍵。
# 同一模型檔只在「首次用到」或「檔案變更」時才重新讀取。
_MODEL_CACHE: Dict[tuple, 'AnomalyModelBundle'] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _cache_key(checkpoint_path: str, device: torch.device) -> tuple:
    checkpoint_path = str(checkpoint_path)
    try:
        stat = os.stat(checkpoint_path)
        file_sig = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        file_sig = (None, None)
    return (checkpoint_path, *file_sig, str(device))


def is_model_cached(checkpoint_path: str, device: torch.device) -> bool:
    """回傳指定模型檔案「目前」是否已在行程內快取中。

    這個函式本身不會觸發任何載入動作，純粹查詢快取狀態，
    設計目的是讓呼叫端（例如 simulation_api）能在真正呼叫
    load_anomaly_model() 之前，先知道這次呼叫會不會付出
    「從磁碟讀取 + 建構模型 + 載入權重」的成本，藉此把
    「第一次模擬很慢、之後很快」這個純粹是快取生效的正常現象，
    誠實地回報給使用者，而不是讓它看起來像隨機發生的黑箱行為。
    """
    key = _cache_key(checkpoint_path, device)
    with _MODEL_CACHE_LOCK:
        return key in _MODEL_CACHE


def clear_model_cache(checkpoint_path: Optional[str] = None) -> None:
    """手動清空模型快取。通常不需要（mtime/size 一變就會自動失效）。

    [部署提示] 若你透過 core/train_packet_native_vae.py 重新訓練並
    覆蓋了 media/model/best_vae_model.pt，但 Django 行程是長駐執行
    （例如 gunicorn 未設定自動重啟），行程內的 _MODEL_CACHE 仍會是
    舊模型物件（雖然檔案的 mtime/size 通常會變，理論上快取鍵會
    自動失效，但若檔案覆蓋方式沒有更新 mtime，或想要保險起見立即
    生效），可在 Django shell 中呼叫：
        python manage.py shell
        >>> import sys; sys.path.insert(0, 'core')
        >>> from model_registry import clear_model_cache
        >>> clear_model_cache()
    或直接重新啟動 Django / Celery worker 行程。
    """
    with _MODEL_CACHE_LOCK:
        if checkpoint_path is None:
            _MODEL_CACHE.clear()
            return
        checkpoint_path = str(checkpoint_path)
        for key in [k for k in _MODEL_CACHE if k[0] == checkpoint_path]:
            del _MODEL_CACHE[key]

# 目前支援的 4 種模型（對應 Django settings.ANOMALY_MODELS 的 4 個 key）
#   unsupervised_vae : model_type == 'cnn_vae'      -> CNNVariationalAutoencoder
#   semi_cicddos2019  \
#   semi_cicids2017    }  model_type == 'hybrid_semi' -> HybridSemiSupervisedDetector
#   semi_nslkdd       /
KNOWN_MODEL_TYPES = ("cnn_vae", "hybrid_semi", "legacy_cnn_ae")


@dataclass
class AnomalyModelBundle:
    """已載入、可直接推論的異常偵測模型，以及還原時一併決定好的
    Grad-CAM 目標層與判定閾值。"""

    model: torch.nn.Module
    model_type: str                 # 'cnn_vae' | 'hybrid_semi' | 'legacy_cnn_ae'
    threshold: float
    device: torch.device
    config: Dict[str, Any] = field(default_factory=dict)
    # None 代表交給 GradCAM 的自動偵測（_auto_detect_target_layer）；
    # hybrid_semi 必須明確指定，理由見檔案開頭說明。
    gradcam_target_layer: Optional[str] = None

    @property
    def is_hybrid(self) -> bool:
        return self.model_type == "hybrid_semi"

    @property
    def input_representation(self) -> str:
        """本模型訓練時使用的輸入資料表示法。

        'packet_bytes_visualizer' : 訓練資料是透過
            core/packet_visualizer.py::PacketVisualizer.bytes_to_image()
            將『原始封包位元組』轉成影像 —— 與 analyzer/tasks.py 的正式
            PCAP 分析管線、以及 analyzer/views.py::simulation_api() 在
            推論時餵給模型的資料完全一致（見
            core/packet_dataset_builder.py + core/train_packet_native_vae.py）。
            這種模型的 CNN／VAE 重建誤差分數具備參考意義。

        'csv_features_legacy'（預設，涵蓋所有沒有標記
            representation 的舊 checkpoint）: 訓練資料來自
            core/dataset_loader.py，是 CICIDS2017／NSL-KDD 等資料集的
            『CSV 統計流量特徵』（經 CICFlowMeter 或類似工具算出的
            duration / byte 數 / IAT 統計等）reshape 成的影像，
            與封包原始位元組是完全不同的資料分布（train/serve skew）。
            這種模型在封包位元組影像上的重建誤差原則上不具判斷力，
            只能當參考數值。
        """
        return str(self.config.get("representation", "csv_features_legacy"))


def load_anomaly_model(checkpoint_path, device: Optional[torch.device] = None,
                        default_latent_dim: int = 32) -> AnomalyModelBundle:
    """依 checkpoint 內儲存的 model_type 動態建構正確的模型類別並載入權重。

    這是整個平台唯一應該存在「4 選 1 模型還原成哪個類別」判斷邏輯的地方。
    analyzer/tasks.py 與 analyzer/views.py 都應呼叫本函式取代自行 if/elif。
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = str(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型檔案: {checkpoint_path}")

    # 查快取
    key = _cache_key(checkpoint_path, device)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_type = ckpt.get("model_type") if isinstance(ckpt, dict) else None
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    if model_type == "hybrid_semi":
        from hybrid_semi_supervised import HybridSemiSupervisedDetector
        model = HybridSemiSupervisedDetector.from_checkpoint(ckpt).to(device)
        # 不可交給自動偵測：'vae.encoder_conv' 是唯一同時「名稱含 encoder」
        # 且「確實會被 forward() 執行到」的卷積路徑。
        gradcam_layer = "vae.encoder_conv"
        threshold = float(ckpt.get("threshold", getattr(model, "threshold", 0.0)))

    elif model_type == "cnn_vae":
        from variational_autoencoder import CNNVariationalAutoencoder
        model = CNNVariationalAutoencoder(
            latent_dim=cfg.get("latent_dim", default_latent_dim),
            kl_weight=cfg.get("kl_weight", 1e-3),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        gradcam_layer = None  # 自動偵測會正確找到 encoder_conv.11
        threshold = float(ckpt.get("threshold", 0.0))

    else:
        # 沒有 model_type（例如舊版 checkpoint）→ 視為原始 CNNAutoencoder
        from cnn_autoencoder import CNNAutoencoder, load_compatible_state_dict
        model = CNNAutoencoder(latent_dim=cfg.get("latent_dim", default_latent_dim)).to(device)
        load_compatible_state_dict(model, ckpt)
        model_type = "legacy_cnn_ae"
        gradcam_layer = None  # 自動偵測會正確找到 encoder.conv_layers.11
        threshold = float(ckpt.get("threshold", 0.0)) if isinstance(ckpt, dict) else 0.0

    model.eval()
    bundle = AnomalyModelBundle(
        model=model, model_type=model_type, threshold=threshold,
        device=device, config=cfg, gradcam_target_layer=gradcam_layer,
    )

    # 寫入快取
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[key] = bundle

    return bundle


def compute_anomaly_scores(bundle: AnomalyModelBundle, X: np.ndarray,
                            batch_size: int = 256) -> Dict[str, np.ndarray]:
    """對整批封包影像計算異常分數。

    - 任何模型皆回傳 'score'（重建誤差，含 KL 項則已在模型內部加總）
      與 'is_anomaly'（score > threshold）。
    - hybrid_semi 模型另外呼叫 classify_known() 取得 CNN-LSTM 分類器的
      「已知攻擊」信心分數，並依 hybrid_semi_supervised.hybrid_decision()
      同樣的邏輯，回傳三態 'status'：
        known_attack   : 分類器判定為攻擊（不論 VAE 重建誤差是否也超標）
        unknown_attack : 分類器判定為正常，但 VAE 重建誤差超過閾值
                         （半監督模型的核心價值：抓到微調時沒看過的新型攻擊）
        normal         : 兩者皆判定正常
    """
    model = bundle.model
    device = bundle.device
    scores = []
    known_conf, known_label = ([], []) if bundle.is_hybrid else (None, None)

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(
                X[i:i + batch_size, np.newaxis, :, :].astype(np.float32)
            ).to(device)
            scores.append(model.reconstruction_error(batch).cpu().numpy())
            if bundle.is_hybrid:
                probs = model.classify_known(batch).cpu()
                conf, label = probs.max(dim=1)
                known_conf.append(conf.numpy())
                known_label.append(label.numpy())

    result: Dict[str, np.ndarray] = {"score": np.concatenate(scores)}
    threshold = bundle.threshold or 0.0

    if bundle.is_hybrid:
        result["known_confidence"] = np.concatenate(known_conf)
        result["known_label"] = np.concatenate(known_label)          # 1=分類器判定為攻擊
        unknown_gate = result["score"] > threshold

        status = np.full(len(result["score"]), "normal", dtype=object)
        status[unknown_gate & (result["known_label"] == 0)] = "unknown_attack"
        status[result["known_label"] == 1] = "known_attack"
        result["status"] = status
        result["is_anomaly"] = status != "normal"
    else:
        result["is_anomaly"] = result["score"] > threshold

    return result
