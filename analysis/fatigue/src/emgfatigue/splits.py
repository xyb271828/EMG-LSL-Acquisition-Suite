import numpy as np


def temporal_partitions(df):
    """Fixed retrospective within-recording future-trial diagnostic.

    train 1:20, validation 21:26, calibration 27:30, embargo 31:32, test 33:40.
    Different load recordings are NOT claimed to form a global prospective test.
    """
    t = df.trial_in_condition.to_numpy()
    parts = {
        "train": np.flatnonzero(t <= 20),
        "validation": np.flatnonzero((t >= 21) & (t <= 26)),
        "calibration": np.flatnonzero((t >= 27) & (t <= 30)),
        "embargo": np.flatnonzero((t >= 31) & (t <= 32)),
        "test": np.flatnonzero((t >= 33) & (t <= 40)),
    }
    if len(np.unique(np.concatenate(list(parts.values())))) != len(df):
        raise ValueError("Invalid temporal partition coverage")
    for r, group in df.groupby("recording_id"):
        for a, b in [
            ("train", "validation"),
            ("validation", "calibration"),
            ("calibration", "test"),
        ]:
            x = group.loc[group.index.intersection(parts[a])]
            y = group.loc[group.index.intersection(parts[b])]
            if x.empty or y.empty or x.end_time.max() >= y.start_time.min():
                raise ValueError(f"Temporal overlap or empty split: {r}")
    return parts
