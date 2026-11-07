"""Train USAD on the clean UCI training period and score the injected test period.

Same settings as notebooks/Step3 (window 12, hidden 100, 100 epochs, seed 0).
Saves window scores, metrics, per-segment detections and the threshold.

Usage:
    python src/run_usad_uci.py --data data/uci --out results/uci
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(__file__))
from usad import UsadModel, training, testing          # official code, unmodified
from utils import get_default_device, to_device
from common import make_windows, window_labels, best_f1

WINDOW_SIZE, HIDDEN_SIZE, BATCH_SIZE, N_EPOCHS = 12, 100, 7919, 100


def main(data, out):
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    device = get_default_device()

    train_df = pd.read_csv(f'{data}/uci_constructed_train.csv', index_col=0, parse_dates=True)
    test_df = pd.read_csv(f'{data}/uci_constructed_test.csv', index_col=0, parse_dates=True)
    inj = pd.read_csv(f'{data}/injection_log.csv')
    CH = [c for c in train_df.columns if c != 'label']
    labels = test_df['label'].values

    scaler = MinMaxScaler().fit(train_df[CH].values)
    tr = scaler.transform(train_df[CH].values)
    te = np.clip(scaler.transform(test_df[CH].values), -5, 5)
    w_tr, w_te = make_windows(tr, WINDOW_SIZE), make_windows(te, WINDOW_SIZE)
    win_lab = window_labels(labels, WINDOW_SIZE)

    w_size, z_size = WINDOW_SIZE * len(CH), WINDOW_SIZE * HIDDEN_SIZE
    to_t = lambda a: torch.from_numpy(a).float().view(-1, w_size)
    split = int(0.8 * len(w_tr))
    dl = lambda a: data_utils.DataLoader(data_utils.TensorDataset(to_t(a)),
                                         batch_size=BATCH_SIZE, shuffle=False)
    model = to_device(UsadModel(w_size, z_size), device)
    t0 = time.time()
    history = training(N_EPOCHS, model, dl(w_tr[:split]), dl(w_tr[split:]))
    minutes = (time.time() - t0) / 60
    with torch.no_grad():
        score = torch.cat([r.flatten() for r in testing(model, dl(w_te))]).cpu().numpy()

    raw_r, pa_r = best_f1(score, win_lab, False), best_f1(score, win_lab, True)
    auc = roc_auc_score(win_lab, score)
    pd.DataFrame([
        {'setting': 'with point adjustment', 'Precision': pa_r[0], 'Recall': pa_r[1], 'F1': pa_r[2]},
        {'setting': 'without point adjustment', 'Precision': raw_r[0], 'Recall': raw_r[1], 'F1': raw_r[2]},
    ]).to_csv(f'{out}/uci_usad_results.csv', index=False)

    # Per-segment detection at the best raw-F1 threshold
    th = raw_r[3]
    flag = score >= th

    def detected(r):
        a, b = max(0, r.start_idx - WINDOW_SIZE + 1), min(len(flag), r.end_idx)
        return bool(flag[a:b].any())

    inj['detected'] = inj.apply(detected, axis=1)
    by_type = inj.groupby(['type', 'severity']).detected.mean().unstack().round(2)
    by_type['overall'] = inj.groupby('type').detected.mean().round(2)
    by_type.to_csv(f'{out}/uci_detection_by_type.csv')
    inj.to_csv(f'{out}/injection_log_with_detection.csv', index=False)

    np.save(f'{out}/uci_scores.npy', score)
    np.save(f'{out}/uci_window_labels.npy', win_lab)
    pd.DataFrame(history).to_csv(f'{out}/uci_history.csv', index=False)
    json.dump({'auc': auc, 'threshold_raw': float(th), 'threshold_pa': float(pa_r[3]),
               'train_minutes': minutes, 'device': str(device),
               'torch': torch.__version__}, open(f'{out}/uci_run_info.json', 'w'), indent=2)
    print(f'AUC {auc:.3f} | F1 raw {raw_r[2]:.3f} | F1 PA {pa_r[2]:.3f} | {minutes:.1f} min')
    print(by_type)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/uci')
    ap.add_argument('--out', default='results/uci')
    a = ap.parse_args()
    main(a.data, a.out)
