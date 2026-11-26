"""Rebuild optional CNN reference weights from original training parents."""

from pathlib import Path
import numpy as np
import pandas as pd
from .splits import temporal_partitions


def fit_saved_cnns(root):
    import torch
    from .cnn import train_cnn

    root = Path(root)
    out = root / "results/direct_fatigue"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "results/rated_trial_features.csv")
    parts = temporal_partitions(df)
    tr = parts["train"]
    va = parts["validation"]
    fit = np.r_[tr, va]
    X = np.load(out / "local_signal_cache.npz")["signals"]
    y = df.perceived_fatigue.to_numpy()
    rms = np.log(np.maximum(np.sqrt(np.mean(X**2, axis=2)), 1e-12))
    oh = np.column_stack(
        [
            np.isclose(df.condition_y.to_numpy(), v).astype(float)
            for v in [0.2, 0.4, 0.6, 0.8]
        ]
    )
    rows = []
    for name, extra in [
        ("cnn_emg", rms),
        ("cnn_emg_context", np.c_[rms, oh, df.trial_in_condition.to_numpy() / 40]),
    ]:
        scores = []
        for seed in [42, 93, 173]:
            first = train_cnn(X, extra, y, tr, va, seed)
            epoch = first[-2]
            from .cnn import cnn_predict, seed_all

            seed_all(500 + seed)
            mu, _, _ = cnn_predict(first, X, extra, va, mc=32)
            scores.append(mu)
            final = train_cnn(X, extra, y, fit, None, seed, epochs=epoch)
            model, mean, std, scaler, *_ = final
            torch.save(
                dict(
                    state_dict=model.state_dict(),
                    extra_dim=extra.shape[1],
                    input_mean=mean.tolist(),
                    input_std=std.tolist(),
                    extra_mean=scaler.mean_.tolist(),
                    extra_scale=scaler.scale_.tolist(),
                    training_trial_uids=df.trial_uid.iloc[fit].tolist(),
                    threshold=7,
                    target="onsite perceived fatigue 0–10",
                    seed=seed,
                    epochs=epoch,
                ),
                out / f"{name}_{seed}.pt",
            )
        rows.append(
            dict(
                model=name,
                validation_mae=float(np.mean(abs(y[va] - np.mean(scores, axis=0)))),
            )
        )
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
