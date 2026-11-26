import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pyxdf
from .features import continuous_features, trial_features


def load_recording(path):
    streams, _ = pyxdf.load_xdf(str(path))
    candidates = [s for s in streams if s["info"]["type"][0] == "EMG Signal"]
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous EMG stream in {path}")
    s = candidates[0]
    fs = float(s["info"]["nominal_srate"][0])
    x = np.asarray(s["time_series"], float)[:, :2]
    t = np.asarray(s["time_stamps"])
    if len(t) < 2 or np.any(np.diff(t) <= 0) or not np.isfinite(x).all():
        raise ValueError(f"Invalid samples or timestamps: {path}")
    markers = [s for s in streams if s["info"]["type"][0] == "Markers"]
    pairs = {}
    if markers:
        for lab, ts in zip(markers[0]["time_series"], markers[0]["time_stamps"]):
            m = re.fullmatch(r"GRIP_(START|END)_(\d+)", str(lab[0]))
            if m:
                key = int(m[2])
                kind = m[1]
                if kind in pairs.get(key, {}):
                    raise ValueError("Duplicate trial marker")
                pairs.setdefault(key, {})[kind] = float(ts)
    complete = []
    for k, v in sorted(pairs.items()):
        if set(v) != {"START", "END"}:
            raise ValueError(f"Unmatched marker for trial {k}")
        a, b = np.searchsorted(t, [v["START"], v["END"]])
        if a >= b or a < 1000 or b > len(x):
            raise ValueError(f"Invalid/warm-up-overlapping trial {k}")
        complete.append((k, a, b, v["START"], v["END"]))
    return x, t, fs, complete


def extract(root):
    root = Path(root)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    ratings = pd.read_csv(root / "data/ratings.csv")
    if ratings.duplicated(["condition_y", "trial_in_condition"]).any():
        raise ValueError("Duplicate rating keys")
    rows = []
    signals = []
    audit = []
    for load in [20, 40, 60, 80]:
        name = f"{load}MVC_left.xdf"
        path = root / "data/raw" / name
        x, t, fs, pairs = load_recording(path)
        if fs != 1000 or len(pairs) != 40:
            raise ValueError(
                "Expected 1000 Hz and forty complete contractions per load"
            )
        filtered, env = continuous_features(x, fs)
        for k, a, b, start, end in pairs:
            if b - a < 2000:
                raise ValueError(
                    "CNN cache needs at least 2000 samples per contraction"
                )
            feat = trial_features(filtered[a:b], env[a:b], fs)
            feat.update(
                subject_id="S3",
                recording_id=name,
                condition_y=load / 100,
                trial_in_condition=k,
                start_time=start,
                end_time=end,
                trial_uid=f"S3:{name}:{k}",
            )
            rows.append(feat)
            signals.append(filtered[a : a + 2000].T.astype(np.float32))
        audit.append(
            dict(
                recording=name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                sampling_hz=fs,
                trials=len(pairs),
            )
        )
    df = (
        pd.DataFrame(rows)
        .merge(
            ratings,
            on=["condition_y", "trial_in_condition"],
            validate="one_to_one",
            how="left",
        )
        .rename(columns={"Fatigue": "perceived_fatigue"})
    )
    if (
        len(df) != 160
        or df.perceived_fatigue.isna().any()
        or not df.perceived_fatigue.between(0, 10).all()
    ):
        raise ValueError("Ratings must match all 160 contractions and lie in 0–10")
    df.to_csv(out / "rated_trial_features.csv", index=False)
    cache = out / "direct_fatigue"
    cache.mkdir(exist_ok=True)
    np.savez_compressed(
        cache / "local_signal_cache.npz",
        signals=np.stack(signals),
        scores=df.perceived_fatigue.to_numpy(),
    )
    import json

    (out / "recording_sources.json").write_text(json.dumps(audit, indent=2))
    return df
