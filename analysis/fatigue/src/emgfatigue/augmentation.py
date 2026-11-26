"""Parent-linked near-real EMG variants, explicitly not additional people.

Each clone inherits its parent's split and rating. Only descendants of real
training parents enter initial training; validation/test descendants never fit.
"""

from pathlib import Path
import hashlib, json, zipfile
import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, welch
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_absolute_error
from .data import load_recording
from .features import continuous_features, trial_features
from .splits import temporal_partitions
from .design import inputs, model, metrics, calibrate_offsets, LOADS

SEED = 2026112606


def transform_parent(context, parent_key, replica, donors):
    """Fixed tiny perturbations of ACTUAL filtered EMG, not fresh noise EMG."""
    key = int.from_bytes(hashlib.sha256(parent_key.encode()).digest()[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, replica, key]))
    rr = np.random.default_rng(np.random.SeedSequence([SEED, replica]))
    gain = rr.uniform(0.99, 1.01, size=2)
    spectrum = np.fft.rfft(context, axis=0)
    phase = rng.uniform(-0.03, 0.03, size=(len(spectrum), 1))
    phase[[0, -1]] = 0
    z = np.fft.irfft(spectrum * np.exp(1j * phase), n=len(context), axis=0) * gain
    donor_index = int(rng.integers(len(donors)))
    donor = donors[donor_index]
    # Real training signal texture at 0.3% of this parent's channel RMS.
    noise = np.resize(donor, context.shape).copy()
    noise -= noise.mean(axis=0)
    noise /= np.maximum(np.sqrt(np.mean(noise**2, axis=0)), 1e-12)
    z += 0.003 * np.sqrt(np.mean(context**2, axis=0)) * noise
    return z, dict(
        gain_ch1=float(gain[0]),
        gain_ch2=float(gain[1]),
        phase_bound_radians=0.03,
        noise_rms_fraction=0.003,
        donor_training_index=donor_index,
    )


def similarity(parent, child, pe, ce):
    out = {}
    for c in range(2):
        a, b = parent[:, c], child[:, c]
        rms = lambda x: float(np.sqrt(np.mean(x * x)))
        f, p = welch(a, 1000, nperseg=min(1024, len(a)))
        _, q = welch(b, 1000, nperseg=min(1024, len(b)))
        m = (f >= 20) & (f <= 450)
        p = p[m] / p[m].sum()
        q = q[m] / q[m].sum()
        corr = lambda x, y: float(np.corrcoef(x, y)[0, 1])
        acf = lambda v: np.array(
            [
                np.dot(v[:-lag], v[lag:]) / max(np.dot(v, v), 1e-12)
                for lag in range(1, 51)
            ]
        )
        out.update(
            {
                f"waveform_correlation_ch{c+1}": corr(a, b),
                f"rms_relative_difference_ch{c+1}": abs(rms(b) / rms(a) - 1),
                f"psd_js_distance_ch{c+1}": float(jensenshannon(p, q)),
                f"envelope_correlation_ch{c+1}": corr(pe[:, c], ce[:, c]),
                f"acf_rmse_ch{c+1}": float(np.sqrt(np.mean((acf(a) - acf(b)) ** 2))),
            }
        )
    out["cross_channel_correlation_difference"] = abs(
        float(np.corrcoef(parent.T)[0, 1] - np.corrcoef(child.T)[0, 1])
    )
    return out


def eligible_children(children, allowed_parent_ids):
    allowed = set(allowed_parent_ids)
    return children[children.parent_trial_uid.isin(allowed)].copy()


def close_replicas_experiment(root, replicas=6):
    if replicas not in (6, 9):
        raise ValueError("Supported frozen replica counts are 6 or 9")
    root = Path(root)
    out = root / ("results/close_replicas" if replicas == 6 else "results/ten_sets")
    out.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(root / "results/rated_trial_features.csv")
    parts = temporal_partitions(source)
    uid_roles = {
        source.trial_uid.iloc[i]: role for role, ids in parts.items() for i in ids
    }
    contexts = {}
    recordings = {}
    archive_audit = []
    archive = root.parent / "upload/Archive.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                if entry.filename.endswith(".xdf") and "__MACOSX" not in entry.filename:
                    sha = hashlib.sha256(z.read(entry)).hexdigest()
                    p = root / "data/raw" / Path(entry.filename).name
                    if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
                        raise ValueError("New attachment differs from original source")
                    archive_audit.append(
                        dict(
                            archive_member=entry.filename,
                            sha256=sha,
                            identical_to_existing=True,
                        )
                    )
    for name in source.recording_id.unique():
        raw, t, fs, pairs = load_recording(root / "data/raw" / name)
        if fs != 1000:
            raise ValueError("Expected 1000 Hz")
        x, _ = continuous_features(raw, fs)
        recordings[name] = dict(
            sha256=hashlib.sha256((root / "data/raw" / name).read_bytes()).hexdigest()
        )
        for k, a, b, start, end in pairs:
            contexts[f"S3:{name}:{k}"] = x[a - 1000 : b].copy()
    donors = [contexts[source.trial_uid.iloc[i]] for i in parts["train"]]
    # Same warmup and original trial duration for parents and all children.
    original = []
    parent_env = {}
    for row in source.itertuples():
        ctx = contexts[row.trial_uid]
        env = sosfilt(butter(4, 5, fs=1000, output="sos"), np.abs(ctx), axis=0)
        feat = trial_features(ctx[1000:], env[1000:])
        feat.update(
            {
                k: getattr(row, k)
                for k in [
                    "subject_id",
                    "recording_id",
                    "condition_y",
                    "trial_in_condition",
                    "start_time",
                    "end_time",
                    "trial_uid",
                    "perceived_fatigue",
                ]
            }
        )
        original.append(feat)
        parent_env[row.trial_uid] = env[1000:]
    real = pd.DataFrame(original)
    real.to_csv(out / "matched_real_features.csv", index=False)
    children = []
    sims = []
    waves = []
    lengths = []
    manifest = []
    for replica in range(1, replicas + 1):
        for row in real.itertuples():
            ctx = contexts[row.trial_uid]
            child, params = transform_parent(ctx, row.trial_uid, replica, donors)
            env = sosfilt(butter(4, 5, fs=1000, output="sos"), np.abs(child), axis=0)
            a = ctx[1000:]
            b = child[1000:]
            feat = trial_features(b, env[1000:])
            feat.update(
                replica_id=f"AUG{replica:02d}",
                parent_trial_uid=row.trial_uid,
                parent_split=uid_roles[row.trial_uid],
                condition_y=row.condition_y,
                trial_in_condition=row.trial_in_condition,
                trial_uid=f"AUG{replica:02d}:{row.trial_uid}",
                inherited_score=row.perceived_fatigue,
                source="real_parent_signal_augmentation",
                real_person=False,
            )
            child_index = len(children)
            children.append(feat)
            waves.append(b.astype(np.float32))
            lengths.append(len(b))
            sim = similarity(a, b, parent_env[row.trial_uid], env[1000:])
            base = original[real.index[real.trial_uid == row.trial_uid][0]]
            for c in [1, 2]:
                for stem in ["mdf", "mnf"]:
                    sim[f"{stem}_relative_difference_ch{c}"] = abs(
                        feat[f"{stem}_ch{c}"] / base[f"{stem}_ch{c}"] - 1
                    )
            sims.append(
                dict(
                    replica_id=f"AUG{replica:02d}",
                    parent_trial_uid=row.trial_uid,
                    parent_split=uid_roles[row.trial_uid],
                    **sim,
                )
            )
            manifest.append(
                dict(
                    row_index=child_index,
                    trial_uid=feat["trial_uid"],
                    parent_trial_uid=row.trial_uid,
                    parent_split=uid_roles[row.trial_uid],
                    replica_id=f"AUG{replica:02d}",
                    **params,
                )
            )
        print("Near-real replica generated", replica, flush=True)
    child = pd.DataFrame(children)
    child.to_csv(out / "replica_features.csv", index=False)
    pd.DataFrame(sims).to_csv(out / "fidelity_per_trial.csv", index=False)
    pd.DataFrame(manifest).to_csv(out / "parent_manifest.csv", index=False)
    n = max(lengths)
    array = np.full((len(waves), n, 2), np.nan, dtype=np.float32)
    for i, wave in enumerate(waves):
        array[i, : len(wave)] = wave
    np.savez_compressed(
        out / "near_real_waveforms.npz",
        signals=array,
        lengths=np.array(lengths),
        trial_uids=child.trial_uid.to_numpy(dtype=str),
        sampling_hz=1000,
        source=np.array("augmentation_of_existing_person_not_new_people"),
    )
    summaries = []
    for role, g in pd.DataFrame(sims).groupby("parent_split"):
        for col in g.select_dtypes("number").columns:
            summaries.append(
                dict(
                    parent_split=role,
                    metric=col,
                    mean=float(g[col].mean()),
                    minimum=float(g[col].min()),
                    maximum=float(g[col].max()),
                    p95=float(g[col].quantile(0.95)),
                )
            )
    pd.DataFrame(summaries).to_csv(out / "fidelity_summary.csv", index=False)
    tr, va, ca, te = [parts[k] for k in ["train", "validation", "calibration", "test"]]
    fit = np.r_[tr, va]
    trainchildren = eligible_children(child, real.trial_uid.iloc[tr])
    fitchildren = eligible_children(child, real.trial_uid.iloc[fit])
    assert len(trainchildren) == 80 * replicas and len(fitchildren) == 104 * replicas
    forbidden = set(real.trial_uid.iloc[np.r_[ca, parts["embargo"], te]])
    assert not set(fitchildren.parent_trial_uid) & forbidden
    y = real.perceived_fatigue.to_numpy()
    search = []
    scores = []
    predictions = []
    stress = []
    choices = []
    for family in ["context", "emg", "fusion"]:
        x = inputs(real, family)
        s = inputs(trainchildren, family)
        sf = inputs(fitchildren, family)
        candidates = []
        for variant, ratio in [
            ("real_only", 0),
            ("duplicate_control", 0),
            ("near_real_0.1", 0.1),
            ("near_real_0.5", 0.5),
            ("near_real_1.0", 1.0),
        ]:
            for kind, param in [("ridge", a) for a in [0.01, 0.1, 1.0, 10.0]] + [
                ("svr", a) for a in [1.0, 10.0]
            ]:
                if variant == "duplicate_control":
                    xx = np.tile(x[tr], (replicas + 1, 1))
                    yy = np.tile(y[tr], replicas + 1)
                    w = np.full(len(xx), 1 / (replicas + 1))
                elif ratio:
                    xx = np.r_[x[tr], s]
                    yy = np.r_[y[tr], trainchildren.inherited_score]
                    w = np.r_[np.ones(len(tr)), np.full(len(s), ratio / replicas)] / (
                        1 + ratio
                    )
                else:
                    xx = x[tr]
                    yy = y[tr]
                    w = np.ones(len(tr))
                m = model(kind, param)
                m[0].fit(x[tr])
                m[-1].fit(m[0].transform(xx), yy, sample_weight=w)
                v = float(mean_absolute_error(y[va], np.clip(m.predict(x[va]), 0, 10)))
                r = dict(
                    family=family,
                    variant=variant,
                    ratio=ratio,
                    kind=kind,
                    parameter=param,
                    validation_mae=v,
                )
                candidates.append(r)
                search.append(r)
            best = min(
                [r for r in candidates if r["variant"] == variant],
                key=lambda r: r["validation_mae"],
            )
            m = model(best["kind"], best["parameter"])
            if variant == "duplicate_control":
                xx = np.tile(x[fit], (replicas + 1, 1))
                yy = np.tile(y[fit], replicas + 1)
                w = np.full(len(xx), 1 / (replicas + 1))
            elif ratio:
                xx = np.r_[x[fit], sf]
                yy = np.r_[y[fit], fitchildren.inherited_score]
                w = np.r_[np.ones(len(fit)), np.full(len(sf), ratio / replicas)] / (
                    1 + ratio
                )
            else:
                xx = x[fit]
                yy = y[fit]
                w = np.ones(len(fit))
            m[0].fit(x[fit])
            m[-1].fit(m[0].transform(xx), yy, sample_weight=w)
            cp = np.clip(m.predict(x[ca]), 0, 10)
            p = np.clip(m.predict(x[te]), 0, 10)
            offsets = calibrate_offsets(y[ca], cp, real.condition_y.iloc[ca].to_numpy())
            adapted = np.clip(
                p + np.array([offsets[float(v)] for v in real.condition_y.iloc[te]]),
                0,
                10,
            )
            joblib.dump(
                dict(
                    model=m,
                    offsets=offsets,
                    family=family,
                    variant=variant,
                    training_parent_uids=real.trial_uid.iloc[fit].tolist(),
                    training_child_uids=fitchildren.trial_uid.tolist() if ratio else [],
                ),
                out / f"{family}_{variant}.joblib",
            )
            for policy, mu in [("uncalibrated", p), ("personal_calibrated", adapted)]:
                scores.append(
                    dict(
                        family=family,
                        variant=variant,
                        policy=policy,
                        validation_mae=best["validation_mae"],
                        **metrics(y[te], mu),
                    )
                )
                for i, v in zip(te, mu):
                    predictions.append(
                        dict(
                            family=family,
                            variant=variant,
                            policy=policy,
                            trial_uid=real.trial_uid.iloc[i],
                            observed_score=int(y[i]),
                            prediction=float(v),
                        )
                    )
            # Correlated stress variants of test parents are never fitting data.
            tc = eligible_children(child, real.trial_uid.iloc[te])
            tp = np.clip(m.predict(inputs(tc, family)), 0, 10)
            tp = np.clip(
                tp + np.array([offsets[float(v)] for v in tc.condition_y]), 0, 10
            )
            stress.append(
                dict(
                    family=family,
                    variant=variant,
                    rows=len(tc),
                    independent_real_parent_trials=32,
                    **metrics(tc.inherited_score.to_numpy(), tp),
                )
            )
        choices.append(
            min(
                [r for r in candidates if r["variant"] != "duplicate_control"],
                key=lambda r: (r["validation_mae"], r["ratio"]),
            )
        )
    for filename, data in [
        ("validation_search", search),
        ("real_test_metrics", scores),
        ("real_test_predictions", predictions),
        ("test_descendant_stress", stress),
        ("validation_selected", choices),
    ]:
        pd.DataFrame(data).to_csv(out / f"{filename}.csv", index=False)
    protocol = dict(
        seed=SEED,
        replica_sets=replicas,
        new_real_people=0,
        real_parent_people=1,
        total_children=160 * replicas,
        initial_training_children=80 * replicas,
        final_training_children=104 * replicas,
        inherit_original_four_loads_and_40_trials=True,
        trial_lengths="original marker-based durations; no replacement with fabricated duration",
        waveform="actual causally filtered parent EMG, 1s context, common FFT phase jitter ±0.03rad, fixed per-replica channel gain ±1%, training-donor signal texture 0.3% RMS",
        envelope="same causal 5Hz, 1s warmup for both parents and children",
        labels="inherit original parent rating; no newly observed fatigue labels",
        fit_rule="inherit parent split; only train descendants fit initially; train+validation descendants fit final model; calibration/embargo/test descendants never fit",
        selection="validation parent MAE only; real-only option retained",
        weights="each real parent plus all its descendants has total weight one; no fitting-weight inflation",
        fidelity="paired waveform/RMS/PSD/envelope/ACF/channel-correlation and MDF/MNF summaries; transform fixed before evaluation; no rejection by test outcomes",
        heldout_stress=f"{32*replicas} correlated variants of 32 real test parents, not independent tests or people",
        report_scope="exploratory same-person historical holdout; no cross-person claim",
        source_attachment_audit=archive_audit,
        recording_sources=recordings,
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(
        pd.DataFrame(scores)
        .query("family == 'fusion'")[
            ["variant", "policy", "validation_mae", "mae", "tp", "fn", "tn", "fp"]
        ]
        .to_string(index=False),
        flush=True,
    )
    return pd.DataFrame(scores)
