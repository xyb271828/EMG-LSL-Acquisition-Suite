"""Shared feature matrices and label-preserving personal score calibration."""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    f1_score,
)

LOADS = [0.2, 0.4, 0.6, 0.8]
SIGNAL = [
    f"{s}_ch{c}"
    for s in [
        "rms",
        "rms_squared",
        "mdf",
        "mnf",
        "low_ratio",
        "spectral_entropy",
        "envelope_cv",
    ]
    for c in [1, 2]
]


def context(df):
    load = df.condition_y.to_numpy()
    t = df.trial_in_condition.to_numpy() / 40
    onehot = np.column_stack([np.isclose(load, v).astype(float) for v in LOADS])
    return np.c_[onehot, onehot * t[:, None]]


def inputs(df, family):
    z = context(df)
    return (
        z
        if family == "context"
        else (
            df[SIGNAL].to_numpy()
            if family == "emg"
            else np.c_[z, df[SIGNAL].to_numpy()]
        )
    )


def metrics(y, mu):
    y = np.asarray(y)
    mu = np.clip(mu, 0, 10)
    ref = y >= 7
    alert = mu >= 7
    tn, fp, fn, tp = confusion_matrix(ref, alert, labels=[False, True]).ravel()
    return dict(
        mae=float(mean_absolute_error(y, mu)),
        rmse=float(np.sqrt(mean_squared_error(y, mu))),
        recall=float(tp / max(tp + fn, 1)),
        specificity=float(tn / max(tn + fp, 1)),
        balanced_accuracy=float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))),
        f1=float(f1_score(ref, alert, zero_division=0)),
        tp=int(tp),
        fn=int(fn),
        tn=int(tn),
        fp=int(fp),
    )


def calibrate_offsets(y, p, loads):
    residual = np.asarray(y) - np.asarray(p)
    global_shift = float(np.mean(residual))
    offsets = {}
    for load in LOADS:
        ix = np.isclose(loads, load)
        n = int(ix.sum())
        offsets[load] = global_shift + n / (n + 4) * (
            float(np.mean(residual[ix])) - global_shift
        )
    return offsets


def model(kind, param):
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=param) if kind == "ridge" else SVR(C=param, epsilon=0.25),
    )
