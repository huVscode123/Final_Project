#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""從與半監督 CICIDS2017 訓練相同的 DatasetLoader 流程匯出 scaler/schema。

此工具的重點不是另外擬合一個新的 scaler，而是重跑 CICIDSLoader 相同的
CSV 檔案排序、取樣參數、欄位選取與 MinMaxScaler fit 流程，供舊版 checkpoint
在部署時使用。只有在 CSV 檔案、版本與訓練參數和當初相同時，才可能重建相同的
前處理器；若原訓練用過自訂 CSV、seed 或 max-normal/max-attack，請傳入原參數。
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = BASE_DIR / 'core'
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


def main():
    parser = argparse.ArgumentParser(description='匯出半監督 CICIDS2017 CSV/Flow 前處理器')
    parser.add_argument('--data-dir', default=str(BASE_DIR / 'data/cicids2017'))
    parser.add_argument('--max-normal', type=int, default=60000)
    parser.add_argument('--max-attack', type=int, default=30000)
    parser.add_argument('--output', default=str(BASE_DIR / 'output/model_semi_cicids2017/cicids2017_preprocessor.pkl'))
    args = parser.parse_args()
    if args.max_normal < 1 or args.max_attack < 1:
        parser.error('--max-normal 和 --max-attack 必須大於 0')

    from dataset_loader import CICIDSLoader, FEATURE_DIM, IMAGE_SIZE
    original_preprocess = CICIDSLoader._preprocess
    captured = {}

    def _capture_preprocess(self, df, label_col, fit=True):
        # 與原版 _preprocess 中的 drop/select 操作相同，只記錄 scaler 的輸入欄位順序。
        probe = df.copy()
        drop_cols = [c for c in probe.columns
                     if str(c).strip() in ('Label', 'label') or c == label_col]
        probe = probe.drop(columns=drop_cols, errors='ignore')
        probe = probe.select_dtypes(include=[np.number])
        if fit:
            captured['feature_columns'] = [str(c) for c in probe.columns]
        return original_preprocess(self, df, label_col, fit=fit)

    CICIDSLoader._preprocess = _capture_preprocess
    try:
        loader = CICIDSLoader(data_dir=args.data_dir, name='CICIDS2017 preprocessor export')
        # 會依訓練資料流程讀 CSV、固定 random_state=42 取樣並 fit 正常流量 scaler。
        loader.load(max_normal=args.max_normal, max_attack=args.max_attack)
    finally:
        CICIDSLoader._preprocess = original_preprocess

    columns = captured.get('feature_columns')
    if loader.scaler is None or not columns:
        raise RuntimeError('未能從 CICIDSLoader 取得已 fit scaler 或 feature_columns。')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'dataset': 'cicids2017',
        'scaler': loader.scaler,
        'feature_columns': columns,
        'image_size': IMAGE_SIZE,
        'feature_dim': FEATURE_DIM,
        'scaler_type': type(loader.scaler).__name__,
        'normalization': 'none',  # 原 DatasetLoader：正常樣本 fit/transform 後 clip；攻擊／推論 transform 不 clip。
        'model_type': 'hybrid_semi',
        'representation': 'csv_features_legacy',
        'training_data_dir': str(Path(args.data_dir).resolve()),
        'max_normal': int(args.max_normal),
        'max_attack': int(args.max_attack),
        'sample_random_state': 42,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'warning': ('只在此 CSV 版本、欄位與取樣參數和訓練時一致時，才可視為相同 scaler。'
                    '若無法確認訓練配置，請重新訓練並直接將 scaler_path 寫入 checkpoint。'),
    }
    with out.open('wb') as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'已輸出：{out}')
    print(f'scaler={payload["scaler_type"]}, image_size={IMAGE_SIZE}, feature_dim={FEATURE_DIM}')
    print(f'欄位數={len(columns)}; 取樣參數 normal={args.max_normal}, attack={args.max_attack}')
    print('重要：請確認 data-dir、CSV 版本與訓練時一致；此工具不會修改或重訓 .pt。')


if __name__ == '__main__':
    main()
