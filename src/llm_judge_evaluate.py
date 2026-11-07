"""Score LLM-judge verdicts against the hidden injection key.

Usage:
    python src/llm_judge_evaluate.py --judgments results/llm_judge/judgments.jsonl
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def main(judgments, key, out):
    j = pd.DataFrame([json.loads(l) for l in open(judgments)])
    k = pd.read_csv(key)
    d = k.merge(j, on='id', how='left')
    d['judge_anomaly'] = (d.verdict == 'anomaly').astype(int)
    d.to_csv(f'{out}/judge_vs_truth.csv', index=False)

    y, p = d.is_anomaly.values, d.judge_anomaly.values
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    acc = (tp + tn) / len(d)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    kappa = cohen_kappa_score(y, p)

    # type agreement on alerts that are truly anomalous and judged anomalous
    both = d[(d.is_anomaly == 1) & (d.judge_anomaly == 1)]
    type_acc = (both.type == both.true_type).mean() if len(both) else np.nan

    # effect of using the judge as a filter on USAD alerts
    usad_prec = y.mean()
    filt_prec = prec
    kept_true_segments = d[(d.is_anomaly == 1) & (d.judge_anomaly == 1)].injection_id.nunique()
    all_true_segments = d[d.is_anomaly == 1].injection_id.nunique()

    # judge confidence calibration
    calib = d.groupby('confidence').apply(
        lambda g: pd.Series({'n': len(g), 'judge_correct': (g.judge_anomaly == g.is_anomaly).mean()}))

    by_type = d[d.is_anomaly == 1].groupby('true_type').agg(
        alerts=('id', 'count'), judged_anomaly=('judge_anomaly', 'mean'))

    summary = {
        'items': int(len(d)), 'true_anomalous_alerts': int(y.sum()), 'false_alarms': int((1 - y).sum()),
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': 2 * prec * rec / max(prec + rec, 1e-9), 'cohen_kappa': kappa,
        'type_accuracy_on_agreed_anomalies': type_acc,
        'usad_alert_precision': usad_prec, 'usad_plus_judge_precision': filt_prec,
        'true_segments_in_alerts': int(all_true_segments),
        'true_segments_kept_after_judge': int(kept_true_segments),
    }
    json.dump(summary, open(f'{out}/judge_summary.json', 'w'), indent=2)
    calib.to_csv(f'{out}/judge_confidence.csv')
    by_type.to_csv(f'{out}/judge_by_true_type.csv')
    print(json.dumps(summary, indent=2))
    print(calib)
    print(by_type)
    print(d[d.judge_anomaly != d.is_anomaly][['id', 'is_anomaly', 'true_type', 'true_severity',
                                              'verdict', 'type', 'confidence']])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--judgments', default='results/llm_judge/judgments.jsonl')
    ap.add_argument('--key', default='results/llm_judge/items_key.csv')
    ap.add_argument('--out', default='results/llm_judge')
    a = ap.parse_args()
    main(a.judgments, a.key, a.out)
