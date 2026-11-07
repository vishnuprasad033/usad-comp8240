"""Prepare LLM-as-judge items from USAD alerts on the constructed dataset.

Each USAD alert (a run of flagged windows at the best raw-F1 threshold, runs
less than 6 windows apart merged) becomes one item. The item shows the judge
the raw sensor values around the alert, plus reference values for the same
hour of day from the clean training period. Items are shuffled and given
random ids; no labels, dates or injection details are included, so the judge
is blind to the ground truth.

Outputs:
    results/llm_judge/items.jsonl         what the judge sees (prompt text per item)
    results/llm_judge/items_key.csv       hidden key: item id -> segment, ground truth
"""
import json
import os

import numpy as np
import pandas as pd

WINDOW = 12
CONTEXT = 12          # 2 hours each side (10-min steps)
MERGE_GAP = 6
MAX_POINTS = 20
SEED = 7

SYSTEM_PROMPT = """You are an expert in household electricity monitoring, acting as a judge of alerts raised by an automatic anomaly detector.

The data come from one house's smart meter, averaged over 10-minute steps. Channels:
- Global_active_power (kW): total active power
- Global_reactive_power (kW)
- Voltage (V): normally about 233-246 V
- Global_intensity (A): total current. Physically it tracks active power closely, roughly I = P*1000/V, so I/P is normally about 4.1-4.4
- Sub_metering_1 (Wh per minute): kitchen (dishwasher, oven, microwave)
- Sub_metering_2 (Wh per minute): laundry (washing machine, dryer, fridge, light)
- Sub_metering_3 (Wh per minute): electric water heater and air conditioner

Normal household behaviour is bursty: appliances switching on and off create large but physically consistent jumps. A genuine anomaly is something a real house would not normally produce, such as a meter fault, a sensor reading zero while power is drawn, a sustained offset that drifts away from normal, a single-step implausible spike, or channels that stop being physically consistent with each other.

For the alert you are shown, decide whether it is a GENUINE anomaly or a NORMAL (false alarm) pattern. If genuine, give the most likely type: spike (1-3 steps), drift (gradual sustained offset over hours), dropout (one or all channels read 0), decoupling (current no longer consistent with active power).

Answer with JSON only:
{"verdict": "anomaly" | "normal", "type": "spike" | "drift" | "dropout" | "decoupling" | "none", "confidence": 1-5, "rationale": "<one sentence>"}"""


def downsample(v, k=MAX_POINTS):
    if len(v) <= k:
        return v
    idx = np.linspace(0, len(v) - 1, k).round().astype(int)
    return v[idx]


def fmt(v):
    return ' '.join(f'{x:.2f}' if abs(x) < 100 else f'{x:.0f}' for x in v)


def main(data='data/uci', res='results/uci', out='results/llm_judge'):
    os.makedirs(out, exist_ok=True)
    test = pd.read_csv(f'{data}/uci_constructed_test.csv', index_col=0, parse_dates=True)
    train = pd.read_csv(f'{data}/uci_constructed_train.csv', index_col=0, parse_dates=True)
    inj = pd.read_csv(f'{data}/injection_log.csv')
    CH = [c for c in test.columns if c != 'label']
    score = np.load(f'{res}/uci_scores.npy')
    th = json.load(open(f'{res}/uci_run_info.json'))['threshold_raw']

    # alerts = merged runs of flagged windows, converted to time-step spans
    flag = (score >= th).astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, flag, 0]))
    runs = list(zip(edges[::2], edges[1::2]))
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    hourly_ref = train.groupby(train.index.hour)[CH].median()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(merged))
    items, key = [], []
    for k, j in enumerate(order):
        wa, wb = merged[j]
        s, e = int(wa), int(wb) + WINDOW - 1          # steps covered by the alert
        a, b = max(0, s - CONTEXT), min(len(test), e + CONTEXT)
        seg = test.iloc[a:b]
        t0 = test.index[s]
        lines = [f'Alert length: {e - s} steps ({(e - s) * 10} min), starting {t0:%A} {t0:%H:%M}.',
                 f'Shown: {CONTEXT} steps before, the alert, {CONTEXT} steps after; {len(seg)} steps '
                 f'downsampled to at most {MAX_POINTS} points. Alert covers positions '
                 f'{s - a}..{e - a - 1} of the {len(seg)} steps.']
        for c in CH:
            before = test[c].values[a:s]
            inside = test[c].values[s:e]
            ref = hourly_ref.loc[test.index[s:e].hour, c].median()
            lines.append(f'{c}: before mean {before.mean():.2f} | alert mean {inside.mean():.2f} '
                         f'min {inside.min():.2f} max {inside.max():.2f} | typical at this hour {ref:.2f}')
            lines.append(f'   series: {fmt(downsample(seg[c].values))}')
        p = seg['Global_active_power'].values
        i = seg['Global_intensity'].values
        ratio = np.where(p > 0.05, i / np.maximum(p, 1e-6), np.nan)
        lines.append('I/P ratio series (nan where P<0.05): ' +
                     ' '.join('nan' if np.isnan(x) else f'{x:.2f}' for x in downsample(ratio)))
        item_id = f'A{k + 1:03d}'
        items.append({'id': item_id, 'system': SYSTEM_PROMPT, 'user': '\n'.join(lines)})

        ov = inj[(inj.start_idx < e) & (inj.end_idx > s)]
        key.append({'id': item_id, 'start_step': s, 'end_step': e,
                    'is_anomaly': int(len(ov) > 0),
                    'true_type': ov.iloc[0].type if len(ov) else 'none',
                    'true_severity': ov.iloc[0].severity if len(ov) else 'none',
                    'injection_id': int(ov.iloc[0].id) if len(ov) else -1})

    with open(f'{out}/items.jsonl', 'w') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    pd.DataFrame(key).sort_values('id').to_csv(f'{out}/items_key.csv', index=False)
    print(f'{len(items)} items written; key holds {sum(k["is_anomaly"] for k in key)} true anomalies')


if __name__ == '__main__':
    main()
