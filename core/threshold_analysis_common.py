# -*- coding: utf-8 -*-
"""共同閾值分析工具：三種方法（Percentile 95、F1 最大化、Youden）共用一致的定義與指標計算。

支援：
- 非監督 VAE：只根據 anomaly score 選 threshold。
- 半監督 Hybrid：固定分類器門檻，針對 VAE unknown-anomaly gate 校準 threshold，
  並以完整 Hybrid 最終輸出計算 F1 / Youden J。

threshold 的判定規則統一為 score > threshold。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np

METHODS = ("percentile", "f1_max", "youden")

METHOD_LABELS = {
    "percentile": "Percentile 95",
    "f1_max": "F1 最大化",
    "youden": "Youden",
}



def split_four_way(X, train_ratio=0.60, validation_ratio=0.20, seed=42):
    """固定切成 train / early-stop-validation / threshold-calibration / test。

    validation_ratio 是兩份驗證資料的總比例，平均切成兩半：
    預設為 60% train / 10% early-stop / 10% calibration / 20% test。
    CICIDS2017 非監督與半監督程式共用此函式及 seed，確保同類資料索引一致。
    """
    x = np.asarray(X)
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError("train_ratio 必須介於 0 與 1")
    if not 0.0 < float(validation_ratio) < 1.0:
        raise ValueError("validation_ratio 必須介於 0 與 1")
    if float(train_ratio) + float(validation_ratio) >= 1.0:
        raise ValueError("train_ratio + validation_ratio 必須小於 1")
    n = len(x)
    if n < 10:
        raise ValueError(f"樣本數 {n} 過少，無法切分四個子集")

    idx = np.random.default_rng(seed).permutation(n)
    n_train = max(1, int(n * train_ratio))
    n_val_pool = max(2, int(n * validation_ratio))
    n_cal = max(1, n_val_pool // 2)
    n_early = n_val_pool - n_cal
    n_test_start = n_train + n_val_pool
    if n_test_start >= n:
        raise ValueError("資料量不足以保留最終測試集")

    return (
        x[idx[:n_train]],
        x[idx[n_train:n_train + n_early]],
        x[idx[n_train + n_early:n_test_start]],
        x[idx[n_test_start:]],
    )


def _finite(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    return a[np.isfinite(a)]


def _finite_pair(first, second):
    """成對清除非有限值，避免兩組對應分數因獨立過濾而錯位。"""
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if len(a) != len(b):
        raise ValueError("成對分數長度不一致")
    valid = np.isfinite(a) & np.isfinite(b)
    return a[valid], b[valid]


def _metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if len(y) != len(p):
        raise ValueError("y_true 與 y_pred 長度不一致")
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "fpr": float(fpr),
        "youden_j": float(recall - fpr),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "n_normal": int(np.sum(y == 0)), "n_attack": int(np.sum(y == 1)),
    }


def _threshold_candidates(normal_scores, attack_scores):
    """產生能表示所有不同預測切分的閾值候選值。"""
    n = _finite(normal_scores)
    a = _finite(attack_scores)
    if len(n) == 0 or len(a) == 0:
        raise ValueError("f1_max/youden 需要正常與攻擊校準分數")
    u = np.unique(np.concatenate([n, a]))
    if len(u) == 1:
        return np.asarray([
            np.nextafter(u[0], -np.inf),
            np.nextafter(u[0], np.inf),
        ], dtype=np.float64)
    mids = u[:-1] + (u[1:] - u[:-1]) / 2.0
    candidates = np.concatenate((
        [np.nextafter(u[0], -np.inf)], mids,
        [np.nextafter(u[-1], np.inf)],
    ))
    return np.unique(candidates)


def _scores_metrics(normal_scores, attack_scores, threshold):
    n = _finite(normal_scores)
    a = _finite(attack_scores)
    y = np.concatenate([np.zeros(len(n), dtype=np.int64), np.ones(len(a), dtype=np.int64)])
    pred = np.concatenate([n > threshold, a > threshold]).astype(np.int64)
    return _metrics(y, pred)


def select_scalar_thresholds(cal_normal, cal_attack, percentile=95.0,
                             youden_max_fpr: Optional[float] = None):
    """在校準集選擇 Percentile、F1 最大化與 Youden 三種 VAE 閾值；最終測試集不參與選閾值。"""
    n = _finite(cal_normal)
    a = _finite(cal_attack)
    if len(n) == 0:
        raise ValueError("正常校準分數為空")
    if not 0.0 < float(percentile) < 100.0:
        raise ValueError("percentile 必須介於 0 與 100")

    thresholds = {
        "percentile": float(np.percentile(n, percentile)),
    }
    infos: Dict[str, Dict[str, Any]] = {
        "percentile": {
            "objective": "normal_calibration_percentile",
            "percentile": float(percentile),
            "n_normal": int(len(n)), "n_attack": int(len(a)),
        }
    }
    if len(a) == 0:
        return thresholds, infos

    candidates = _threshold_candidates(n, a)
    cal_metrics = [_scores_metrics(n, a, float(t)) for t in candidates]
    f1s = np.asarray([m["f1"] for m in cal_metrics])
    fprs = np.asarray([m["fpr"] for m in cal_metrics])
    js = np.asarray([m["youden_j"] for m in cal_metrics])

    # 候選 threshold 由低到高，平手時選比較高的 threshold（通常 FPR 較低）。
    best_idx = int(np.flatnonzero(f1s == np.max(f1s))[-1])
    thresholds["f1_max"] = float(candidates[best_idx])
    infos["f1_max"] = {
        "objective": "maximize_f1", **cal_metrics[best_idx],
        "n_candidates": int(len(candidates)),
    }

    valid = np.ones(len(candidates), dtype=bool)
    if youden_max_fpr is not None:
        max_fpr = float(youden_max_fpr)
        if not 0.0 <= max_fpr <= 1.0:
            raise ValueError("youden_max_fpr 必須介於 0 與 1 之間")
        valid &= (fprs <= max_fpr)

    if not valid.any():
        # With a valid FPR bound in [0, 1], the all-normal boundary candidate
        # should be feasible. Raise rather than falsely label percentile as Youden.
        raise ValueError("沒有符合 FPR 條件的 Youden 候選閾值")

    best_j = np.max(js[valid])
    best_y = int(np.flatnonzero(valid & (js == best_j))[-1])
    thresholds["youden"] = float(candidates[best_y])
    infos["youden"] = {
        "objective": "maximize_youden_j",
        "max_fpr": youden_max_fpr,
        **cal_metrics[best_y],
        "n_candidates": int(len(candidates)),
        "fallback": False,
    }
    return thresholds, infos


def analyze_scalar_thresholds(model_key, cal_normal, cal_attack, test_normal,
                              test_attack, percentile=95.0,
                              youden_max_fpr: Optional[float] = None):
    """選出三種閾值並在同一測試集評估，回傳 JSON-safe 結果與分數快取。"""
    cal_n, cal_a = _finite(cal_normal), _finite(cal_attack)
    test_n, test_a = _finite(test_normal), _finite(test_attack)
    if len(test_n) == 0 or len(test_a) == 0:
        raise ValueError("最終測試集必須同時有正常與攻擊樣本")
    thresholds, infos = select_scalar_thresholds(cal_n, cal_a, percentile, youden_max_fpr)
    out: Dict[str, Any] = {
        "model_key": model_key,
        "threshold_policy": "higher anomaly score means more anomalous; score > threshold is attack",
        "percentile": float(percentile),
        "youden_max_fpr": youden_max_fpr,
        "split": {
            "cal_normal": int(len(cal_n)), "cal_attack": int(len(cal_a)),
            "test_normal": int(len(test_n)), "test_attack": int(len(test_a)),
        },
        "methods": {},
    }
    for method in METHODS:
        if method not in thresholds:
            out["methods"][method] = {"available": False, "reason": "缺少攻擊校準樣本"}
            continue
        t = thresholds[method]
        out["methods"][method] = {
            "available": True,
            "threshold": float(t),
            "info": infos[method],
            "calibration": _scores_metrics(cal_n, cal_a, t) if len(cal_a) else None,
            "test": _scores_metrics(test_n, test_a, t),
        }
    from sklearn.metrics import roc_auc_score, average_precision_score
    if len(cal_n) > 0 and len(cal_a) > 0:
        try:
            y_cal = np.r_[np.zeros(len(cal_n), dtype=np.int64),
                          np.ones(len(cal_a), dtype=np.int64)]
            cal_score = np.r_[cal_n, cal_a]
            out["calibration_auc"] = {
                "roc_auc": float(roc_auc_score(y_cal, cal_score)),
                "pr_auc": float(average_precision_score(y_cal, cal_score)),
            }
        except Exception as exc:
            out["calibration_auc"] = {"warning": str(exc)}
    else:
        out["calibration_auc"] = {
            "warning": "缺少正常或攻擊校準樣本，無法計算校準 AUC"
        }

    try:
        y_test = np.r_[np.zeros(len(test_n), dtype=np.int64),
                       np.ones(len(test_a), dtype=np.int64)]
        test_score = np.r_[test_n, test_a]
        out["test_auc"] = {
            "roc_auc": float(roc_auc_score(y_test, test_score)),
            "pr_auc": float(average_precision_score(y_test, test_score)),
        }
    except Exception as exc:
        out["test_auc"] = {"warning": str(exc)}
    return out, {
        "cal_normal": cal_n, "cal_attack": cal_a,
        "test_normal": test_n, "test_attack": test_a,
    }


def _hybrid_pred(vae_scores, attack_prob, vae_threshold,
                 classifier_threshold=0.5):
    """計算 Hybrid 預測；若有非有限值，成對剔除以維持分數索引對齊。"""
    s = np.asarray(vae_scores, dtype=np.float64).reshape(-1)
    p = np.asarray(attack_prob, dtype=np.float64).reshape(-1)
    if len(s) != len(p):
        raise ValueError("VAE 分數和分類器機率長度不一致")
    valid = np.isfinite(s) & np.isfinite(p)
    s = s[valid]
    p = p[valid]
    known_attack = p >= float(classifier_threshold)
    unknown_attack = (~known_attack) & (s > float(vae_threshold))
    return (known_attack | unknown_attack).astype(np.int64)


def _hybrid_metrics(normal_vae, normal_prob, attack_vae, attack_prob,
                    threshold, classifier_threshold=0.5):
    n_pred = _hybrid_pred(normal_vae, normal_prob, threshold, classifier_threshold)
    a_pred = _hybrid_pred(attack_vae, attack_prob, threshold, classifier_threshold)
    y = np.concatenate([
        np.zeros(len(n_pred), dtype=np.int64),
        np.ones(len(a_pred), dtype=np.int64),
    ])
    return _metrics(y, np.concatenate([n_pred, a_pred]))


def select_hybrid_thresholds(cal_normal_vae, cal_normal_prob,
                              cal_attack_vae, cal_attack_prob,
                              percentile=95.0, classifier_threshold=0.5,
                              youden_max_fpr: Optional[float] = None):
    """針對完整 Hybrid 最終輸出選擇 VAE gate 閾值；分類器門檻固定。"""
    sn, pn = _finite_pair(cal_normal_vae, cal_normal_prob)
    sa, pa = _finite_pair(cal_attack_vae, cal_attack_prob)
    if len(sn) == 0:
        raise ValueError("Hybrid 正常校準集為空")
    if not 0.0 < float(percentile) < 100.0:
        raise ValueError("percentile 必須介於 0 與 100")

    percentile_threshold = float(np.percentile(sn, percentile))
    thresholds = {"percentile": percentile_threshold}
    infos: Dict[str, Dict[str, Any]] = {
        "percentile": {
            "objective": "normal_calibration_percentile",
            "percentile": float(percentile),
            "classifier_threshold_fixed": float(classifier_threshold),
        }
    }
    if len(sa) == 0:
        # 沒有攻擊校準樣本，就無法根據標籤計算 F1 / Youden。
        # 只回傳 percentile，讓報告把其他兩種方法標記為 unavailable，
        # 不把 percentile 結果冒充為 F1 最大化或 Youden。
        return thresholds, infos

    candidates = _threshold_candidates(sn, sa)
    metrics_list = [
        _hybrid_metrics(sn, pn, sa, pa, float(t), classifier_threshold)
        for t in candidates
    ]
    f1s = np.asarray([m["f1"] for m in metrics_list])
    fprs = np.asarray([m["fpr"] for m in metrics_list])
    js = np.asarray([m["youden_j"] for m in metrics_list])

    best_idx = int(np.flatnonzero(f1s == np.max(f1s))[-1])
    thresholds["f1_max"] = float(candidates[best_idx])
    infos["f1_max"] = {
        "objective": "maximize_hybrid_f1_with_fixed_classifier_threshold",
        "classifier_threshold_fixed": float(classifier_threshold),
        **metrics_list[best_idx], "n_candidates": int(len(candidates)),
    }

    valid = np.ones(len(candidates), dtype=bool)
    if youden_max_fpr is not None:
        max_fpr = float(youden_max_fpr)
        if not 0.0 <= max_fpr <= 1.0:
            raise ValueError("youden_max_fpr 必須介於 0 與 1 之間")
        valid &= (fprs <= max_fpr)

    if not valid.any():
        raise ValueError("沒有符合 FPR 條件的 Youden 候選閾值")

    best_j = np.max(js[valid])
    best_y = int(np.flatnonzero(valid & (js == best_j))[-1])
    thresholds["youden"] = float(candidates[best_y])
    infos["youden"] = {
        "objective": "maximize_hybrid_youden_j_with_fixed_classifier_threshold",
        "classifier_threshold_fixed": float(classifier_threshold),
        "max_fpr": youden_max_fpr,
        **metrics_list[best_y],
        "n_candidates": int(len(candidates)),
        "fallback": False,
    }
    return thresholds, infos


def analyze_hybrid_thresholds(model_key,
                              cal_normal_vae, cal_normal_prob,
                              cal_attack_vae, cal_attack_prob,
                              test_normal_vae, test_normal_prob,
                              test_attack_vae, test_attack_prob,
                              percentile=95.0, classifier_threshold=0.5,
                              youden_max_fpr: Optional[float] = None):
    """完整 Hybrid 三種閾值分析；所有方法在同一校準集選門檻、同一測試集報告。"""
    cnv, cnp = _finite_pair(cal_normal_vae, cal_normal_prob)
    cav, cap = _finite_pair(cal_attack_vae, cal_attack_prob)
    tnv, tnp = _finite_pair(test_normal_vae, test_normal_prob)
    tav, tap = _finite_pair(test_attack_vae, test_attack_prob)
    if len(tnv) == 0 or len(tav) == 0:
        raise ValueError("最終測試集必須同時有正常與攻擊樣本")

    thresholds, infos = select_hybrid_thresholds(
        cnv, cnp, cav, cap, percentile, classifier_threshold, youden_max_fpr)
    out: Dict[str, Any] = {
        "model_key": model_key,
        "model_type": "hybrid_semi",
        "threshold_policy": "final attack = classifier_attack_probability >= fixed threshold OR VAE score > VAE threshold",
        "classifier_threshold_fixed": float(classifier_threshold),
        "percentile": float(percentile), "youden_max_fpr": youden_max_fpr,
        "split": {
            "cal_normal": int(len(cnv)), "cal_attack": int(len(cav)),
            "test_normal": int(len(tnv)), "test_attack": int(len(tav)),
        },
        "metrics_note": "閾值式指標以完整 Hybrid 最終輸出計算；AUC 另以分類器機率或 VAE score 分開呈現。",
        "methods": {},
    }
    for method in METHODS:
        if method not in thresholds:
            out["methods"][method] = {
                "available": False,
                "reason": "缺少攻擊校準樣本，無法依標籤最大化 F1 或 Youden",
            }
            continue
        t = thresholds[method]
        out["methods"][method] = {
            "available": True,
            "threshold": float(t),
            "classifier_threshold": float(classifier_threshold),
            "info": infos[method],
            "calibration": (
                _hybrid_metrics(cnv, cnp, cav, cap, t, classifier_threshold)
                if len(cav) > 0 else None
            ),
            "test": _hybrid_metrics(tnv, tnp, tav, tap, t, classifier_threshold),
        }
    # AUC is independent of VAE gate threshold; report it separately to avoid mixing metrics.
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        y_test = np.concatenate([np.zeros(len(tnp), dtype=np.int64), np.ones(len(tap), dtype=np.int64)])
        prob = np.concatenate([tnp, tap])
        vae = np.concatenate([tnv, tav])
        out["test_auc"] = {
            "classifier_roc_auc": float(roc_auc_score(y_test, prob)),
            "classifier_pr_auc": float(average_precision_score(y_test, prob)),
            "vae_score_roc_auc": float(roc_auc_score(y_test, vae)),
            "vae_score_pr_auc": float(average_precision_score(y_test, vae)),
        }
    except Exception as exc:
        out["test_auc"] = {"warning": str(exc)}

    cache = {
        "cal_normal_vae": cnv, "cal_normal_attack_probability": cnp,
        "cal_attack_vae": cav, "cal_attack_probability": cap,
        "test_normal_vae": tnv, "test_normal_attack_probability": tnp,
        "test_attack_vae": tav, "test_attack_probability": tap,
    }
    return out, cache


def write_analysis(analysis, cache, output_dir, stem="threshold_analysis"):
    """將 analysis 寫為 JSON、分數寫為 NPZ。"""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{stem}.json")
    npz_path = os.path.join(output_dir, "scores_for_threshold_analysis.npz")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2, default=_json_default)
    np.savez_compressed(npz_path, **{k: np.asarray(v) for k, v in cache.items()})
    return json_path, npz_path


def _json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.bool_,)): return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
