"""Shared helpers used by every experiment script.

The model itself (UsadModel, training, testing) is the unmodified official
implementation in usad.py / utils.py (github.com/manigalati/usad, BSD-3).
"""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def make_windows(x, w):
    """Sliding windows of length w over a (time, channels) array."""
    idx = np.arange(w)[None, :] + np.arange(x.shape[0] - w)[:, None]
    return x[idx]


def window_labels(labels, w):
    """A window is anomalous if any point inside it is anomalous."""
    return np.array([1 if labels[i:i + w].sum() > 0 else 0
                     for i in range(len(labels) - w)])


def point_adjust(pred, label):
    """If any point in a true anomaly segment is flagged, flag the whole segment
    (Xu et al., 2018; the protocol USAD uses for SMD)."""
    pred = pred.copy()
    i, n = 0, len(label)
    while i < n:
        if label[i] == 1:
            j = i
            while j < n and label[j] == 1:
                j += 1
            if pred[i:j].any():
                pred[i:j] = 1
            i = j
        else:
            i += 1
    return pred


def best_f1(score, label, adjust, q_low=0.5, n_th=300):
    """Search thresholds over score quantiles and keep the best F1
    (the 'best-F1' protocol of the OmniAnomaly benchmark).
    Returns (precision, recall, f1, threshold)."""
    best = (0, 0, 0, 0)
    for th in np.quantile(score, np.linspace(q_low, 0.9999, n_th)):
        pred = (score >= th).astype(int)
        if adjust:
            pred = point_adjust(pred, label)
        p, r, f, _ = precision_recall_fscore_support(
            label, pred, average='binary', zero_division=0)
        if f > best[2]:
            best = (p, r, f, th)
    return best
