"""Build the self-constructed dataset: UCI Household Power + injected anomalies.

Script version of notebooks/Step3_Constructed_Dataset.ipynb (same code, same
seed), so the dataset can be rebuilt without Colab.

Usage:
    python src/build_constructed_dataset.py --raw data/household_power_consumption.txt --out data/uci
"""
import argparse
import os

import numpy as np
import pandas as pd

SEED = 42
FREQ = '10min'
N_PER_TYPE = 20
LENGTHS = {'spike': (1, 3), 'drift': (36, 144), 'dropout': (12, 72), 'decoupling': (24, 72)}
GAP = 144  # >= 1 day between injected segments


def main(raw_path, out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # 1. Collection
    raw = pd.read_csv(raw_path, sep=';', na_values='?', low_memory=False)
    raw['datetime'] = pd.to_datetime(raw['Date'] + ' ' + raw['Time'], format='%d/%m/%Y %H:%M:%S')
    raw = raw.drop(columns=['Date', 'Time']).set_index('datetime').astype(float)
    CH = list(raw.columns)
    print(f'raw: {len(raw):,} rows, {len(CH)} channels, '
          f'{100 * raw.isna().any(axis=1).mean():.2f}% rows missing')

    # 2. Preprocessing: 10-min means, fill short gaps, flag long outages, date split
    df = raw.resample(FREQ).mean()
    df = df.interpolate(method='time', limit=6)  # gaps up to 1 hour
    long_gap = df.isna().any(axis=1)            # still missing => long outage
    df = df.interpolate(method='time').bfill().ffill()
    imputed = long_gap.values

    train_df = df.loc['2007-01-01':'2009-06-30'].copy()
    test_clean = df.loc['2009-07-01':].copy()
    test_imputed = imputed[df.index >= pd.Timestamp('2009-07-01')]

    # 3. Annotation: inject anomalies of known type, timing and severity
    sigma = train_df.std()
    X = test_clean.values.copy()
    n = len(X)
    occupied = test_imputed.copy()
    labels = np.zeros(n, dtype=int)
    log = []
    ci = {c: i for i, c in enumerate(CH)}

    def find_slot(length):
        for _ in range(10000):
            s = int(rng.integers(GAP, n - length - GAP))
            if not occupied[s - GAP:s + length + GAP].any():
                occupied[s - GAP:s + length + GAP] = True
                return s
        raise RuntimeError('no free slot')

    types = [t for t in LENGTHS for _ in range(N_PER_TYPE)]
    rng.shuffle(types)
    for k, t in enumerate(types):
        L = int(rng.integers(LENGTHS[t][0], LENGTHS[t][1] + 1))
        s = find_slot(L)
        e = s + L
        level = rng.choice(['low', 'high'])
        if t == 'spike':
            ch = rng.choice(CH)
            mag = (3 if level == 'low' else 6) * sigma[ch]
            sign = 1 if ch != 'Voltage' else rng.choice([-1, 1])
            X[s:e, ci[ch]] += sign * mag
            chans = ch
        elif t == 'drift':
            ch = rng.choice(CH)
            mag = (2 if level == 'low' else 4) * sigma[ch]
            sign = 1 if ch != 'Voltage' else rng.choice([-1, 1])
            X[s:e, ci[ch]] += sign * np.linspace(0, mag, L)
            chans = ch
        elif t == 'dropout':
            if level == 'low':
                ch = rng.choice(CH)
                X[s:e, ci[ch]] = 0
                chans = ch
            else:
                X[s:e, :] = 0
                chans = 'all'
        else:  # decoupling: current no longer tracks power
            j = ci['Global_intensity']
            src = int(rng.integers(0, n - L))
            while abs(src - s) < 1008:  # replacement from >= 1 week away
                src = int(rng.integers(0, n - L))
            w = 0.5 if level == 'low' else 1.0
            X[s:e, j] = (1 - w) * X[s:e, j] + w * test_clean.values[src:src + L, j]
            chans = 'Global_intensity'
        X = np.clip(X, 0, None)
        labels[s:e] = 1
        log.append({'id': k, 'type': t, 'channel': chans, 'severity': level,
                    'start': test_clean.index[s], 'end': test_clean.index[e - 1],
                    'start_idx': s, 'end_idx': e, 'length_steps': L})

    test_df = pd.DataFrame(X, index=test_clean.index, columns=CH)
    test_df['label'] = labels
    inj = pd.DataFrame(log).sort_values('start_idx').reset_index(drop=True)

    train_df.to_csv(f'{out}/uci_constructed_train.csv')
    test_df.to_csv(f'{out}/uci_constructed_test.csv')
    test_clean.to_csv(f'{out}/uci_test_clean.csv')
    inj.to_csv(f'{out}/injection_log.csv', index=False)
    print(f'train {len(train_df):,} steps | test {len(test_df):,} steps | '
          f'{len(inj)} segments | anomaly ratio {100 * labels.mean():.2f}%')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='data/household_power_consumption.txt')
    ap.add_argument('--out', default='data/uci')
    a = ap.parse_args()
    main(a.raw, a.out)
