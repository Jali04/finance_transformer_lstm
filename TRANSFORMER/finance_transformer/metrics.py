# src/metrics.py
import numpy as np
from math import sqrt

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def rmse(y_true, y_pred):
    return float(sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def mape(y_true, y_pred, eps=1e-12):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)

def directional_accuracy(y_true, y_pred):
    gt = (np.asarray(y_true) > 0).astype(int)
    pdv = (np.asarray(y_pred) > 0).astype(int)
    return float(np.mean(gt == pdv))

def matthews_corr(y_true, y_pred):
    gt = (np.asarray(y_true) > 0).astype(int)
    pdv = (np.asarray(y_pred) > 0).astype(int)
    tp = float(((pdv == 1) & (gt == 1)).sum())
    tn = float(((pdv == 0) & (gt == 0)).sum())
    fp = float(((pdv == 1) & (gt == 0)).sum())
    fn = float(((pdv == 0) & (gt == 1)).sum())
    denom = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    return float(((tp*tn) - (fp*fn)) / denom) if denom != 0 else 0.0