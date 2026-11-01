#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""檢查四個部署 checkpoint 能否由目前 core/model_registry.py 載入。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / 'core'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

MODEL_PATHS = {
    'unsup_cicids2017': ROOT / 'media/model/best_vae_cicids2017.pt',
    'semi_cicids2017': ROOT / 'media/model/semi_cicids2017.pt',
    'mta_malware': ROOT / 'media/model/mta_malware.pt',
    'semi_mta_malware': ROOT / 'media/model/semi_mta_malware.pt',
}


def main():
    try:
        import torch
        from model_registry import load_anomaly_model
    except Exception as exc:
        print(f'無法載入 PyTorch / model_registry：{type(exc).__name__}: {exc}')
        return 2

    device = torch.device('cpu')
    failed = False
    for key, path in MODEL_PATHS.items():
        print(f'\n[{key}] {path}')
        if not path.is_file():
            print('  ERROR: checkpoint 不存在')
            failed = True
            continue
        try:
            bundle = load_anomaly_model(str(path), device=device)
            print(f'  model_type           : {bundle.model_type}')
            print(f'  input_representation : {bundle.input_representation}')
            print(f'  packet_native        : {bundle.is_packet_native}')
            print(f'  hybrid               : {bundle.is_hybrid}')
            print(f'  threshold            : {bundle.threshold}')
            config = bundle.config if isinstance(bundle.config, dict) else {}
            scaler = config.get('scaler_path') or config.get('preprocessor_path')
            print(f'  checkpoint preproc   : {scaler or "未記錄"}')
            if not bundle.is_packet_native and key in ('unsup_cicids2017', 'semi_cicids2017'):
                pkl = ROOT / 'output/model_cicids2017/cicids2017_preprocessor.pkl'
                print(f'  default scaler exists: {pkl.is_file()} ({pkl})')
            if key in ('mta_malware', 'semi_mta_malware') and not bundle.is_packet_native:
                print('  WARNING: MTA 目前預期使用 packet-native checkpoint；請核對此模型訓練時的表示法。')
        except Exception as exc:
            failed = True
            print(f'  ERROR: checkpoint 無法由現有模型登錄器載入：{type(exc).__name__}: {exc}')
    print('\n完成。ERROR 表示權重不存在或與目前模型架構不相容。')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
