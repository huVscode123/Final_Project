#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""固定 checkpoint 的輸入資訊消融；不重新訓練模型架構。

輸出：output/<model-key>/ablation/ablation_report.json、metrics.png、ablation.log。
提醒：若資料陣列是訓練資料，結果只能用於方法敏感度分析，不可當成獨立測試集的泛化成績。
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR / 'core') not in sys.path:
    sys.path.insert(0, str(BASE_DIR / 'core'))

MODEL_PATHS = {
    'unsup_cicids2017': BASE_DIR / 'media/model/best_vae_cicids2017.pt',
    'semi_cicids2017': BASE_DIR / 'media/model/semi_cicids2017.pt',
    'mta_malware': BASE_DIR / 'media/model/mta_malware.pt',
    'semi_mta_malware': BASE_DIR / 'media/model/semi_mta_malware.pt',
}
DATA_PATHS = {
    # MTA packet-native 訓練流程直接輸出的影像矩陣；由訓練腳本產生。
    'mta_malware': (BASE_DIR / 'output/model_mta_malware/dataset/X_normal_synthetic.npy',
                    BASE_DIR / 'output/model_mta_malware/dataset/X_attack.npy'),
}
CICIDS_DATA_DIR_CANDIDATES = [BASE_DIR / 'data/cicids2017', BASE_DIR / 'data/CICIDS2017']
EXPERIMENTS = ('baseline', 'mask10', 'mask25', 'shuffle_rows', 'sample50')


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def as_images(arr, path):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 2:
        n = int(round(arr.shape[1] ** 0.5))
        if n * n != arr.shape[1]:
            raise ValueError(f'{path} 是二維特徵矩陣，欄位數 {arr.shape[1]} 不能 reshape 成方形影像。')
        arr = arr.reshape(-1, n, n)
    if arr.ndim != 3:
        raise ValueError(f'{path} 維度應為 (N,H,W)，實際為 {arr.shape}。')
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr


def _find_label_col(columns):
    normalized = {str(c).strip().lower(): c for c in columns}
    for candidate in ('label', 'attack type', 'attack_type', 'class', 'target', 'attack'):
        if candidate in normalized:
            return normalized[candidate]
    for key, original in normalized.items():
        if 'label' in key or 'attack' in key:
            return original
    raise ValueError(f'找不到 CICIDS2017 標籤欄位。現有欄位：{list(columns)[:12]}')


def load_cicids_flow_data(bundle, model_key, max_per_class=5000, seed=20261010):
    """從原始 CICIDS2017 CSV 取樣，並用該模型訓練期 scaler 轉成同表示法。

    不讀取舊版通用 X_normal.npy，因其可能由其他 scaler 或 packet-byte builder 產生，
    僅靠 (N,32,32) shape 無法證明語意一致。
    """
    import pickle
    import pandas as pd
    from cnn_autoencoder import features_to_image

    if bundle.is_packet_native:
        raise RuntimeError(
            '這個半監督式 CICIDS2017 checkpoint 是 packet-native（原始封包位元組影像）表示法，'
            '不能拿 CICIDS2017 CSV/Flow 樣本做消融。請先以該 checkpoint 相同的 PacketVisualizer '
            '處理帶有可信標籤的 CICIDS2017 PCAP，建立專用的正常／攻擊測試矩陣；目前不會拿模擬封包'
            '冒充 CICIDS2017 資料。'
        )

    preproc_candidates = []
    cfg = bundle.config if isinstance(bundle.config, dict) else {}
    # checkpoint metadata 擁有最高優先權；舊 checkpoint 再選取該模型專屬 .pkl。
    for key in ('scaler_path', 'preprocessor_path'):
        if cfg.get(key):
            raw = Path(str(cfg[key]))
            preproc_candidates.append(raw if raw.is_absolute() else BASE_DIR / raw)
    model_specific_preprocessor = {
        'unsup_cicids2017': BASE_DIR / 'output/model_cicids2017/cicids2017_preprocessor.pkl',
        'semi_cicids2017': BASE_DIR / 'output/model_semi_cicids2017/cicids2017_preprocessor.pkl',
    }.get(model_key)
    if model_specific_preprocessor is not None:
        preproc_candidates.append(model_specific_preprocessor)
    # 不允許半監督模型靜默沿用另一個模型的 scaler，避免 train/serve skew。
    preprocessor_path = next((p for p in preproc_candidates if p.is_file()), None)
    if preprocessor_path is None:
        raise FileNotFoundError(
            '找不到此 CICIDS2017 Flow 模型的訓練期 .pkl 前處理器。候選路徑：\n' +
            '\n'.join(str(p) for p in preproc_candidates) +
            '\n請使用該 checkpoint 訓練時產生的 scaler 與 feature_columns。'
        )
    with preprocessor_path.open('rb') as fh:
        prep = pickle.load(fh)
    scaler = prep.get('scaler')
    feature_cols = list(prep.get('feature_columns') or [])
    if scaler is None or not feature_cols:
        raise ValueError(f'前處理器缺少 scaler 或 feature_columns：{preprocessor_path}')
    image_size = int(prep.get('image_size', cfg.get('image_size', 32)))
    normalization = prep.get('normalization', 'none')

    data_dir = next((p for p in CICIDS_DATA_DIR_CANDIDATES if p.is_dir()), None)
    if data_dir is None:
        raise FileNotFoundError(
            '找不到 CICIDS2017 CSV 資料夾。預期路徑之一：\n' +
            '\n'.join(str(p) for p in CICIDS_DATA_DIR_CANDIDATES)
        )
    csv_files = sorted(data_dir.rglob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'在 {data_dir} 找不到 CICIDS2017 CSV。')

    def chunks():
        for csv_path in csv_files:
            try:
                for chunk in pd.read_csv(csv_path, chunksize=100_000, low_memory=False):
                    chunk.columns = chunk.columns.astype(str).str.strip()
                    label_col = _find_label_col(chunk.columns)
                    labels = chunk[label_col].astype(str).str.strip().str.upper()
                    normal_mask = labels.isin({'BENIGN', 'NORMAL', 'NORMAL TRAFFIC', '0'})
                    yield csv_path, chunk, label_col, normal_mask.to_numpy(dtype=bool)
            except Exception as exc:
                raise RuntimeError(f'讀取 CICIDS2017 CSV 失敗：{csv_path}: {exc}') from exc

    # 第一遍只統計每個 chunk 的類別大小；第二遍按比例取樣，避免將數百萬列全載入記憶體。
    total_n = total_a = 0
    for _path, chunk, _label_col, normal_mask in chunks():
        total_n += int(normal_mask.sum())
        total_a += int((~normal_mask).sum())
    if total_n == 0 or total_a == 0:
        raise ValueError(f'CICIDS2017 CSV 未同時找到正常與攻擊樣本（normal={total_n}, attack={total_a}）。')

    rng = np.random.default_rng(seed)
    normal_parts, attack_parts = [], []
    normal_taken = attack_taken = 0
    for _path, chunk, label_col, normal_mask in chunks():
        normal_positions = np.flatnonzero(normal_mask)
        attack_positions = np.flatnonzero(~normal_mask)
        nq = min(len(normal_positions), max(0, int(round(max_per_class * len(normal_positions) / total_n))))
        aq = min(len(attack_positions), max(0, int(round(max_per_class * len(attack_positions) / total_a))))
        if nq:
            chosen = rng.choice(normal_positions, size=nq, replace=False)
            normal_parts.append(chunk.iloc[chosen].copy())
            normal_taken += nq
        if aq:
            chosen = rng.choice(attack_positions, size=aq, replace=False)
            attack_parts.append(chunk.iloc[chosen].copy())
            attack_taken += aq

    if not normal_parts or not attack_parts:
        raise ValueError('取樣後沒有同時取得正常與攻擊樣本。')
    normal_df = pd.concat(normal_parts, ignore_index=True)
    attack_df = pd.concat(attack_parts, ignore_index=True)
    # chunk 配額四捨五入可能略超過上限；統一打散並裁切至指定數量。
    if len(normal_df) > max_per_class:
        normal_df = normal_df.sample(max_per_class, random_state=seed)
    if len(attack_df) > max_per_class:
        attack_df = attack_df.sample(max_per_class, random_state=seed + 1)

    def to_model_images(df):
        xdf = df.drop(columns=[c for c in df.columns if str(c).strip().lower() == str(label_col).strip().lower()], errors='ignore')
        # 正式推論流程對缺少欄位補 0；此處也沿用同一規則和 scaler。
        for col in feature_cols:
            if col not in xdf.columns:
                xdf[col] = 0.0
        raw = xdf[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = scaler.transform(raw).astype(np.float32)
        if normalization == 'clip_0_1':
            scaled = np.clip(scaled, 0.0, 1.0)
        images = features_to_image(scaled, image_size).squeeze(1)
        return images.astype(np.float32, copy=False)

    Xn = to_model_images(normal_df)
    Xa = to_model_images(attack_df)
    X = np.concatenate([Xn, Xa], axis=0)
    y = np.concatenate([np.zeros(len(Xn), dtype=np.int64), np.ones(len(Xa), dtype=np.int64)])
    order = rng.permutation(len(y))
    source = f'{data_dir} CSV; scaler={preprocessor_path}; sample_seed={seed}'
    return X[order], y[order], source, source, str(preprocessor_path)


def load_cicids_packet_data(max_per_class=5000, seed=20261010):
    """從 CICIDS2017 PCAP + labels JSON 建立 PacketVisualizer 同表示法的抽樣集。

    標籤由來源端點與時間窗推導，屬於資料集標記規則，不是逐封包人工標籤；
    未匹配的 PCAP 會略過，避免把 UNKNOWN 當作 BENIGN。
    """
    import json
    from datetime import datetime, timedelta, timezone
    from scapy.utils import PcapReader
    from scapy.layers.inet import IP
    try:
        from scapy.layers.inet6 import IPv6
    except Exception:
        IPv6 = None
    from packet_visualizer import PacketVisualizer

    labels_path = Path(os.getenv('VNAAP_CICIDS_LABELS', str(BASE_DIR / 'core/cicids2017_labels.json')))
    if not labels_path.is_file():
        raise FileNotFoundError(
            f'找不到 CICIDS2017 PCAP 標籤規則：{labels_path}。'
            '請放置 core/cicids2017_labels.json 或設定 VNAAP_CICIDS_LABELS。'
        )
    spec = json.loads(labels_path.read_text(encoding='utf-8'))
    tz = float(spec.get('timezone_offset_hours', 0))
    tzinfo = timezone(timedelta(hours=tz))

    def to_epoch(date, stamp):
        fmt = '%Y-%m-%d %H:%M:%S' if str(stamp).count(':') == 2 else '%Y-%m-%d %H:%M'
        return datetime.strptime(f'{date} {stamp}', fmt).replace(tzinfo=tzinfo).timestamp()

    benign_tokens = [str(x).lower() for x in spec.get('benign_files', [])]
    entries = []
    for item in spec.get('entries', []):
        windows = item.get('windows') or [[item['start'], item['end']]]
        entries.append({
            'file_contains': str(item.get('file_contains', '')).lower(),
            'label': str(item.get('label', 'ATTACK')),
            'attackers': set(map(str, item.get('attacker_ips', []))),
            'victims': set(map(str, item.get('victim_ips', []))),
            'windows': [(to_epoch(item['date'], a), to_epoch(item['date'], b) + 60.0) for a, b in windows],
        })

    roots = [
        BASE_DIR / 'PCAPs/cicids2017-PCAPs',
        BASE_DIR / 'PCAPS/cicids2017-PCAPs',
        BASE_DIR / 'PCAPs/CICIDS2017-PCAPs',
        BASE_DIR / 'data/cicids2017_pcaps',
    ]
    pcap_root = next((p for p in roots if p.is_dir()), None)
    if pcap_root is None:
        raise FileNotFoundError('找不到 CICIDS2017 PCAP 目錄。已檢查：\n' + '\n'.join(map(str, roots)))
    pcap_files = sorted(p for p in pcap_root.rglob('*') if p.is_file() and p.suffix.lower() in {'.pcap', '.pcapng', '.cap'})
    if not pcap_files:
        raise FileNotFoundError(f'目錄中沒有 PCAP/PCAPNG 檔案：{pcap_root}')

    def classify(filename, src, dst, ts):
        low_name = filename.lower()
        if any(token in low_name for token in benign_tokens):
            return 0
        applicable = [e for e in entries if e['file_contains'] and e['file_contains'] in low_name]
        if not applicable:
            return None
        for e in applicable:
            endpoint_hit = ((src in e['attackers'] and (not e['victims'] or dst in e['victims'])) or
                            (dst in e['attackers'] and (not e['victims'] or src in e['victims'])))
            if endpoint_hit and any(ts >= start and ts <= end for start, end in e['windows']):
                return 1
        # 檔名對應已知攻擊 capture，但此封包不在來源端點/時間窗內，依 label schedule 設為良性。
        return 0

    rng = np.random.default_rng(seed)
    reservoir = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    visualizer = PacketVisualizer('medium', apply_mask=True, skip_ethernet=True)
    processed = skipped_unknown = 0
    for file_index, pcap_path in enumerate(pcap_files, 1):
        log(f'CICIDS2017 PCAP 標記與影像化 ({file_index}/{len(pcap_files)}): {pcap_path.name}')
        try:
            with PcapReader(str(pcap_path)) as reader:
                for packet in reader:
                    try:
                        src = dst = ''
                        if packet.haslayer(IP):
                            src, dst = str(packet[IP].src), str(packet[IP].dst)
                        elif IPv6 is not None and packet.haslayer(IPv6):
                            src, dst = str(packet[IPv6].src), str(packet[IPv6].dst)
                        timestamp = float(packet.time)
                        label = classify(pcap_path.name, src, dst, timestamp)
                        if label is None:
                            skipped_unknown += 1
                            continue
                        arr = np.asarray(visualizer.bytes_to_image(bytes(packet)), dtype=np.float32)
                        if arr.ndim != 2:
                            continue
                        seen[label] += 1
                        bucket = reservoir[label]
                        if len(bucket) < max_per_class:
                            bucket.append(arr.copy())
                        else:
                            replace_idx = int(rng.integers(0, seen[label]))
                            if replace_idx < max_per_class:
                                bucket[replace_idx] = arr.copy()
                        processed += 1
                    except Exception:
                        continue
        except Exception as exc:
            log(f'警告：略過無法讀取的 PCAP {pcap_path}: {exc}')

    if not reservoir[0] or not reservoir[1]:
        raise RuntimeError(
            f'CICIDS2017 PCAP 標籤後樣本不足。已處理={processed}，正常={seen[0]}，攻擊={seen[1]}，'
            f'UNKNOWN/未匹配封包={skipped_unknown}。請檢查 labels JSON 中的檔名、IP 與時區設定。'
        )
    Xn = np.stack(reservoir[0]).astype(np.float32)
    Xa = np.stack(reservoir[1]).astype(np.float32)
    X = np.concatenate([Xn, Xa], axis=0)
    y = np.concatenate([np.zeros(len(Xn), dtype=np.int64), np.ones(len(Xa), dtype=np.int64)])
    order = rng.permutation(len(y))
    source = f'{pcap_root} + {labels_path}; PacketVisualizer; sampled normal={len(Xn)}, attack={len(Xa)}, processed_labeled={processed}, skipped_unknown={skipped_unknown}'
    return X[order], y[order], source, source, None


def load_eval_data(model_key, bundle, max_per_class=5000):
    dataset = 'cicids2017' if 'cicids2017' in model_key else 'mta_malware'
    if dataset == 'cicids2017':
        if bundle.is_packet_native:
            return load_cicids_packet_data(max_per_class=max_per_class)
        return load_cicids_flow_data(bundle, model_key=model_key, max_per_class=max_per_class)

    normal_path, attack_path = DATA_PATHS[dataset]
    if not normal_path.is_file() or not attack_path.is_file():
        raise FileNotFoundError(
            f'缺少與 {dataset} 對應的資料陣列：\n  {normal_path}\n  {attack_path}\n'
            '請先執行 MTA 訓練／資料建構流程；本工具不會把其他資料集假裝成該資料集。'
        )
    if not bundle.is_packet_native:
        raise RuntimeError(
            'MTA 配套資料是 packet-native 影像，但這個 checkpoint 是 Flow/CSV 模型。'
            '請提供此 checkpoint 訓練時的 feature schema 與 scaler 所產生的 MTA Flow 陣列。'
        )
    Xn = as_images(np.load(normal_path, mmap_mode='r'), normal_path)
    Xa = as_images(np.load(attack_path, mmap_mode='r'), attack_path)
    rng = np.random.default_rng(20261010)
    ni = rng.choice(len(Xn), size=min(max_per_class, len(Xn)), replace=False)
    ai = rng.choice(len(Xa), size=min(max_per_class, len(Xa)), replace=False)
    X = np.concatenate([Xn[ni], Xa[ai]], axis=0).astype(np.float32, copy=False)
    y = np.concatenate([np.zeros(len(ni), dtype=np.int64), np.ones(len(ai), dtype=np.int64)])
    order = rng.permutation(len(y))
    return X[order], y[order], str(normal_path), str(attack_path), None


def transform(X, name, seed=42):
    Y = X.copy()
    if name == 'baseline':
        return Y
    if name in ('mask10', 'mask25'):
        fraction = 0.10 if name == 'mask10' else 0.25
        h, w = Y.shape[1:]
        side_h = max(1, int(round(h * fraction ** 0.5)))
        side_w = max(1, int(round(w * fraction ** 0.5)))
        y0, x0 = (h - side_h) // 2, (w - side_w) // 2
        Y[:, y0:y0 + side_h, x0:x0 + side_w] = 0.0
    elif name == 'shuffle_rows':
        rng = np.random.default_rng(seed)
        Y = Y[:, rng.permutation(Y.shape[1]), :]
    elif name == 'sample50':
        return Y[::2]
    else:
        raise ValueError(f'未知消融項目：{name}')
    return Y


def predict(bundle, X):
    from model_registry import compute_anomaly_scores
    scored = compute_anomaly_scores(bundle, X)
    if bundle.is_hybrid:
        pred = np.asarray(scored.get('is_anomaly', np.asarray(scored['status']) != 'normal'), dtype=bool)
    else:
        pred = np.asarray(scored['score'], dtype=np.float64) > float(bundle.threshold)
    return pred


def metrics(y, pred):
    y = np.asarray(y, dtype=bool); pred = np.asarray(pred, dtype=bool)
    tp = int(np.sum(y & pred)); tn = int(np.sum(~y & ~pred))
    fp = int(np.sum(~y & pred)); fn = int(np.sum(y & ~pred))
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        'sample_count': int(len(y)), 'true_positive': tp, 'true_negative': tn,
        'false_positive': fp, 'false_negative': fn,
        'accuracy': round((tp + tn) / max(len(y), 1), 6),
        'precision': round(precision, 6), 'recall': round(recall, 6),
        'f1': round(f1, 6), 'false_positive_rate': round(fp / max(fp + tn, 1), 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-key', choices=MODEL_PATHS, required=True)
    ap.add_argument('--experiments', nargs='+', default=list(EXPERIMENTS[1:]),
                    choices=('mask10', 'mask25', 'shuffle_rows', 'sample50', 'baseline'))
    ap.add_argument('--max-per-class', type=int, default=5000)
    args = ap.parse_args()
    out_dir = BASE_DIR / 'output' / args.model_key / 'ablation'
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / 'ablation_report.json'
    model_path = MODEL_PATHS[args.model_key]
    if not model_path.is_file():
        raise FileNotFoundError(f'找不到對應模型 checkpoint：{model_path}')

    from model_registry import load_anomaly_model
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'載入模型：{model_path}')
    bundle = load_anomaly_model(str(model_path), device=device)
    X, y, normal_path, attack_path, preprocessor_path = load_eval_data(args.model_key, bundle, args.max_per_class)
    log(f'資料表示法={bundle.input_representation}，normal/attack 樣本總數={len(y)}，threshold={bundle.threshold:.8g}')
    plan = ['baseline'] + [x for x in args.experiments if x != 'baseline']
    plan = list(dict.fromkeys(plan))
    results = {}
    for n, exp in enumerate(plan, 1):
        log(f'({n}/{len(plan)}) 執行 {exp}')
        Xexp = transform(X, exp)
        yexp = y if exp != 'sample50' else y[::2]
        pred = predict(bundle, Xexp)
        results[exp] = metrics(yexp, pred)
        results[exp]['description'] = {
            'baseline': '未修改模型輸入的基準結果',
            'mask10': '將中央約 10% 面積設為 0，觀察局部資訊缺失影響',
            'mask25': '將中央約 25% 面積設為 0，觀察較大局部資訊缺失影響',
            'shuffle_rows': '置換影像列順序，觀察空間排列對模型判斷的影響',
            'sample50': '以每隔一筆的方式取 50% 評估樣本，觀察樣本量與結果波動；非特徵消融',
        }[exp]
        log(f"F1={results[exp]['f1']:.4f}, Precision={results[exp]['precision']:.4f}, Recall={results[exp]['recall']:.4f}")

    report = {
        'model_key': args.model_key,
        'checkpoint': str(model_path),
        'dataset': 'CICIDS2017' if 'cicids2017' in args.model_key else 'Malware Traffic Analysis (MTA)',
        'learning_mode': 'unsupervised' if args.model_key.startswith('unsup_') or args.model_key == 'mta_malware' else 'semi_supervised',
        'input_representation': bundle.input_representation,
        'threshold': float(bundle.threshold),
        'normal_data_source': normal_path, 'attack_data_source': attack_path,
        'preprocessor_path': preprocessor_path,
        'sample_count_total': int(len(y)),
        'evaluation_warning': '若所用 .npy 包含模型訓練時使用的樣本，這些數字屬於輸入敏感度／消融觀察，不可當成獨立測試集泛化效能。正式效能報告應使用獨立且去重的 test split。',
        'experiment_type': '固定已訓練 checkpoint 的推論輸入消融；不重新訓練架構、latent dimension 或模型權重。',
        'results': results,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names = list(results.keys())
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
        x = np.arange(len(names)); w = 0.25
        ax.bar(x-w, [results[n]['precision'] for n in names], w, label='Precision')
        ax.bar(x, [results[n]['recall'] for n in names], w, label='Recall')
        ax.bar(x+w, [results[n]['f1'] for n in names], w, label='F1')
        ax.set_xticks(x, names, rotation=15, ha='right'); ax.set_ylim(0, 1)
        ax.set_ylabel('Metric'); ax.set_title(f'Input Ablation — {args.model_key}')
        ax.grid(axis='y', alpha=.2); ax.legend(); fig.tight_layout()
        fig.savefig(out_dir / 'ablation_input_metrics.png', bbox_inches='tight'); plt.close(fig)
    except Exception as exc:
        log(f'警告：無法產生繪圖（JSON 已儲存）：{exc}')
    log(f'完成。JSON：{report_path}')

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        log(f'ERROR: {type(exc).__name__}: {exc}')
        raise
