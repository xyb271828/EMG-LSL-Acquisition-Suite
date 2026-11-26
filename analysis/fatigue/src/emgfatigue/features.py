"""Causal EMG preprocessing and explicitly named signal variability statistics."""

import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, welch


class CausalEMGFilter:
    """Stateful bandpass and 50 Hz notch. Chunking preserves filter state."""

    def __init__(self, fs=1000.0, channels=2):
        if fs <= 900:
            raise ValueError("20–450 Hz protocol requires sampling rate > 900 Hz")
        b, a = iirnotch(50, 30, fs=fs)
        self.sos = np.vstack(
            [butter(4, [20, 450], btype="bandpass", fs=fs, output="sos"), tf2sos(b, a)]
        )
        self.state = np.zeros((len(self.sos), 2, channels))

    def process(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("Expected finite samples × channels")
        y, self.state = sosfilt(self.sos, x, axis=0, zi=self.state)
        return y


def trial_features(x, env, fs=1000.0):
    """Only samples at or before trial end are used. No MVC/future calibration."""
    if len(x) < int(0.2 * fs):
        raise ValueError("Trial shorter than one 200 ms window")
    win, hop = int(0.2 * fs), int(0.05 * fs)
    out = {"n_samples": len(x), "duration_s": len(x) / fs}
    for c in range(x.shape[1]):
        v, e = x[:, c], env[:, c]
        rms = np.sqrt(np.mean(v * v))
        # Zero-copy window views replace a Python loop; the feature definitions,
        # causality and 200ms/50ms window boundaries remain identical.
        envelope_windows = np.lib.stride_tricks.sliding_window_view(e, win)[::hop]
        residual_windows = np.lib.stride_tricks.sliding_window_view(v - e, win)[::hop]
        means = np.mean(envelope_windows, axis=1)
        residual_vars = np.var(residual_windows, axis=1)
        f, p = welch(v, fs, nperseg=min(1024, len(v)))
        m = (f >= 20) & (f <= 450)
        f, p = f[m], p[m]
        q = p / max(p.sum(), 1e-20)
        features = {
            "rms": rms,
            "rms_squared": rms * rms,
            "mav": np.mean(np.abs(v)),
            "mnf": np.sum(f * q),
            "mdf": np.interp(0.5, np.cumsum(q), f),
            "low_ratio": np.sum(q[f <= 60]),
            "spectral_entropy": -np.sum(q * np.log(q + 1e-20)) / np.log(len(q)),
            "envelope_mean_var": np.var(means),
            "residual_var": np.mean(residual_vars),
            "legacy_si": np.var(means) + np.mean(residual_vars),
            "envelope_cv": np.std(e) / (abs(np.mean(e)) + 1e-12),
            "waveform_length_per_sample": np.mean(np.abs(np.diff(v))),
        }
        out.update({f"{k}_ch{c+1}": float(val) for k, val in features.items()})
        out[f"n_windows_ch{c+1}"] = len(means)
    return out


def continuous_features(x, fs=1000.0):
    y = CausalEMGFilter(fs, x.shape[1]).process(x)
    env = sosfilt(butter(4, 5, fs=fs, output="sos"), np.abs(y), axis=0)
    return y, env
