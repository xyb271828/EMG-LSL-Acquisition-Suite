"""Small-data probabilistic fatigue models and parent-separated UQ evaluation.

Original data only fit the new models; nine near-duplicates are robustness probes,
not nine new people or independent GP observations. Selection never sees tests.
"""

import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
    DotProduct,
)
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    log_loss,
    cohen_kappa_score,
)
from .design import inputs, calibrate_offsets, metrics
from .splits import temporal_partitions


def ordinal_probabilities(mu, sd, maximum=10):
    """Rounded/censored Gaussian score probabilities; endpoint mass retained."""
    mu = np.asarray(mu)
    sd = np.maximum(np.asarray(sd), 0.05)
    cuts = np.arange(maximum) + 0.5
    cdf = norm.cdf((cuts[None, :] - mu[:, None]) / sd[:, None])
    p = np.diff(np.c_[np.zeros(len(mu)), cdf, np.ones(len(mu))], axis=1)
    p = np.maximum(p, 1e-12)
    return p / p.sum(axis=1, keepdims=True)


def proper_scores(y, p):
    y = np.asarray(y, int)
    maximum = p.shape[1] - 1
    obs = np.eye(maximum + 1)[y]
    return dict(
        categorical_nll=float(
            -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()
        ),
        categorical_brier=float(np.mean(np.sum((p - obs) ** 2, axis=1))),
        normalized_rps=float(
            np.mean(
                np.sum(
                    (np.cumsum(p, axis=1)[:, :-1] - np.cumsum(obs, axis=1)[:, :-1])
                    ** 2,
                    axis=1,
                )
                / maximum
            )
        ),
    )


def risk_coverage(errors, uncertainty):
    """Expected prefix risk within tied uncertainty groups, avoiding row-order bias."""
    errors = np.asarray(errors, float)
    uncertainty = np.asarray(uncertainty, float)
    n = len(errors)
    order = np.argsort(uncertainty, kind="stable")
    u = uncertainty[order]
    e = errors[order]
    # Round tiny numeric jitter only; a fixed-width method must not gain from order.
    key = np.round(u, 10)
    curve = []
    prior = 0.0
    start = 0
    while start < n:
        end = start + 1
        while end < n and key[end] == key[start]:
            end += 1
        mean = float(e[start:end].mean())
        for k in range(start + 1, end + 1):
            curve.append((prior + (k - start) * mean) / k)
        prior += float(e[start:end].sum())
        start = end
    return np.asarray(curve)


def uq_metrics(y, mu, sd, interval_quantile=None):
    p = ordinal_probabilities(mu, sd)
    prob = p[:, 7:].sum(1)
    ref = np.asarray(y) >= 7
    lower = np.maximum(0, mu - norm.ppf(0.95) * sd)
    upper = np.minimum(10, mu + norm.ppf(0.95) * sd)
    error = abs(y - mu)
    rc = risk_coverage(error, sd)
    entropy = -(
        prob * np.log(np.maximum(prob, 1e-12))
        + (1 - prob) * np.log(np.maximum(1 - prob, 1e-12))
    )
    binaryrc = risk_coverage(((mu >= 7) != ref).astype(float), entropy)
    result = dict(
        **proper_scores(y, p),
        binary_brier=float(np.mean((prob - ref) ** 2)),
        binary_nll=float(log_loss(ref, np.c_[1 - prob, prob], labels=[False, True])),
        average_precision=float(average_precision_score(ref, prob)),
        positive_prevalence=float(ref.mean()),
        probability_auroc=(
            float(roc_auc_score(ref, prob)) if np.unique(ref).size == 2 else np.nan
        ),
        interval_coverage_90=float(np.mean((y >= lower) & (y <= upper))),
        interval_mean_width=float(np.mean(upper - lower)),
        interval_score_90=float(
            np.mean(
                upper
                - lower
                + 20 * np.maximum(lower - y, 0)
                + 20 * np.maximum(y - upper, 0)
            )
        ),
        score_aurc=float(rc.mean()),
        random_rejection_mae=float(error.mean()),
        oracle_score_aurc=float(risk_coverage(error, error).mean()),
        binary_aurc=float(binaryrc.mean()),
        error_over_one_auroc=(
            float(roc_auc_score(error > 1, sd))
            if np.unique(error > 1).size == 2
            else np.nan
        ),
    )
    for coverage in [0.5, 0.8, 1.0]:
        k = int(np.ceil(len(y) * coverage))
        result[f"mae_at_{int(coverage*100)}pct_coverage"] = float(rc[k - 1])
        result[f"rows_at_{int(coverage*100)}pct_coverage"] = k
    if interval_quantile is not None:
        lo = np.maximum(0, mu - interval_quantile * sd)
        hi = np.minimum(10, mu + interval_quantile * sd)
        result.update(
            residual_adjusted_coverage_90=float(np.mean((y >= lo) & (y <= hi))),
            residual_adjusted_mean_width=float(np.mean(hi - lo)),
            residual_adjusted_interval_score_90=float(
                np.mean(
                    hi - lo + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
                )
            ),
        )
    return result, p, rc, binaryrc


class ProbabilisticScoreModel:
    def __init__(self, kind, family, param):
        self.kind = kind
        self.family = family
        self.param = param

    def fit(self, df, y):
        X = inputs(df, self.family)
        self.scaler = StandardScaler().fit(X)
        z = self.scaler.transform(X)
        param = self.param
        if self.kind == "ridge":
            self.estimator = Ridge(alpha=param["alpha"]).fit(z, y)
        elif self.kind == "bayesian":
            self.estimator = BayesianRidge().fit(z, y)
        elif self.kind in ["gp", "residual_gp"]:
            residual = np.asarray(y).copy()
            if self.kind == "residual_gp":
                self.mean_scaler = StandardScaler().fit(inputs(df, "context"))
                self.mean_model = Ridge(alpha=param.get("alpha", 0.01)).fit(
                    self.mean_scaler.transform(inputs(df, "context")), y
                )
                residual = y - self.mean_model.predict(
                    self.mean_scaler.transform(inputs(df, "context"))
                )
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
                param["length"], length_scale_bounds="fixed", nu=1.5
            )
            if param.get("linear", False):
                kernel += DotProduct(sigma_0=1.0, sigma_0_bounds="fixed")
            kernel += WhiteKernel(param["noise"], noise_level_bounds="fixed")
            self.estimator = GaussianProcessRegressor(
                kernel=kernel, optimizer=None, normalize_y=True, alpha=1e-6
            ).fit(z, residual)
        elif self.kind == "extra_trees":
            self.estimator = ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=param["leaf"],
                random_state=84,
                n_jobs=1,
            ).fit(z, y)
        elif self.kind == "quantile_gb":
            self.quantiles = [
                GradientBoostingRegressor(
                    loss="quantile",
                    alpha=q,
                    n_estimators=150,
                    learning_rate=0.03,
                    max_depth=param["depth"],
                    min_samples_leaf=param["leaf"],
                    random_state=84,
                ).fit(z, y)
                for q in [0.05, 0.5, 0.95]
            ]
            self.estimator = self.quantiles[1]
        self.noise_floor = max(
            float(np.sqrt(np.mean((y - self.predict(df)[0]) ** 2))), 0.25
        )
        return self

    def predict(self, df):
        z = self.scaler.transform(inputs(df, self.family))
        if self.kind in ["gp", "residual_gp", "bayesian"]:
            mu, sd = self.estimator.predict(z, return_std=True)
            if self.kind == "residual_gp":
                mu += self.mean_model.predict(
                    self.mean_scaler.transform(inputs(df, "context"))
                )
        elif self.kind == "extra_trees":
            members = np.array([t.predict(z) for t in self.estimator.estimators_])
            mu = members.mean(0)
            sd = np.maximum(members.std(0), 0.25)
        elif self.kind == "quantile_gb":
            lo, mu, hi = [e.predict(z) for e in self.quantiles]
            sd = np.maximum(
                (np.maximum(hi, lo) - np.minimum(hi, lo)) / (2 * norm.ppf(0.95)), 0.25
            )
        else:
            mu = self.estimator.predict(z)
            sd = np.full(len(z), getattr(self, "noise_floor", 0.25))
        return np.clip(mu, 0, 10), np.maximum(sd, 0.15)


def calibrate_personal(y, mu, sd, loads):
    y = np.asarray(y)
    mu = np.asarray(mu)
    sd = np.asarray(sd)
    loads = np.asarray(loads)
    loo = []
    for j in range(len(y)):
        ix = np.arange(len(y)) != j
        offsets = calibrate_offsets(y[ix], mu[ix], loads[ix])
        loo.append(float(np.clip(mu[j] + offsets[float(loads[j])], 0, 10)))
    loo = np.asarray(loo)
    scale = min(
        [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        key=lambda s: proper_scores(
            y, ordinal_probabilities(loo, np.maximum(sd * s, 0.15))
        )["categorical_nll"],
    )
    normalized = np.sort(abs(y - loo) / np.maximum(sd * scale, 0.15))
    rank = min(int(np.ceil((len(y) + 1) * 0.9)), len(y))
    return dict(
        offsets=calibrate_offsets(y, mu, loads),
        sd_scale=scale,
        loo_mae=float(mean_absolute_error(y, loo)),
        loo_prediction=loo,
        interval_quantile=float(normalized[rank - 1]),
        interval_scope="cross-fitted calibration residual diagnostic, not exchangeability-guaranteed conformal coverage",
    )


def apply_personal(mu, sd, loads, cal):
    return np.clip(
        mu + np.array([cal["offsets"][float(v)] for v in loads]), 0, 10
    ), np.maximum(sd * cal["sd_scale"], 0.15)


def candidates():
    grid = {}
    for family in ["context", "emg", "fusion"]:
        grid[f"ridge_{family}"] = [
            ("ridge", family, dict(alpha=a)) for a in [0.01, 0.1, 1.0, 10.0]
        ]
    grid["bayesian_fusion"] = [("bayesian", "fusion", {})]
    for family in ["emg", "fusion"]:
        grid[f"gpr_{family}"] = [
            ("gp", family, dict(length=l, noise=n, linear=linear))
            for l in [2.0, 8.0, 32.0]
            for n in [0.03, 0.3]
            for linear in [False, True]
        ]
    grid["residual_gpr_fusion"] = [
        ("residual_gp", "fusion", dict(length=l, noise=n, alpha=a))
        for l in [4.0, 16.0]
        for n in [0.1, 1.0]
        for a in [0.01, 1.0]
    ]
    grid["extra_trees_fusion"] = [
        ("extra_trees", "fusion", dict(leaf=l)) for l in [3, 8]
    ]
    grid["quantile_gb_fusion"] = [
        ("quantile_gb", "fusion", dict(depth=d, leaf=l))
        for d in [1, 2]
        for l in [5, 10]
    ]
    return grid


def cnn_bundles(root, name):
    import torch
    from .cnn import DirectFatigueNet

    bundles = []
    for seed in [42, 93, 173]:
        saved = torch.load(
            root / f"results/direct_fatigue/{name}_{seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        model = DirectFatigueNet(saved["extra_dim"])
        model.load_state_dict(saved["state_dict"])
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(saved["extra_mean"])
        scaler.scale_ = np.asarray(saved["extra_scale"])
        scaler.var_ = scaler.scale_**2
        scaler.n_features_in_ = len(scaler.mean_)
        bundles.append(
            (
                model,
                np.asarray(saved["input_mean"], np.float32),
                np.asarray(saved["input_std"], np.float32),
                scaler,
            )
        )
    return bundles


def cnn_score(bundles, signals, df, name):
    import torch
    from .cnn import cnn_predict, seed_all

    torch.set_num_threads(2)
    rms = np.log(np.maximum(np.sqrt(np.mean(signals**2, axis=2)), 1e-12))
    oh = np.column_stack(
        [
            np.isclose(df.condition_y.to_numpy(), v).astype(float)
            for v in [0.2, 0.4, 0.6, 0.8]
        ]
    )
    extra = (
        rms
        if name == "cnn_emg"
        else np.c_[rms, oh, df.trial_in_condition.to_numpy() / 40]
    )
    means = []
    var = []
    for seed, bundle in zip([42, 93, 173], bundles):
        seed_all(800 + seed)
        mu, prob, v = cnn_predict(bundle, signals, extra, np.arange(len(df)), mc=32)
        means.append(mu)
        var.append(v)
    return np.mean(means, 0), np.sqrt(np.mean(var, 0) + np.var(means, 0))


def advanced_experiment(root, include_cnn=True):
    root = Path(root)
    out = root / "results/advanced"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "results/rated_trial_features.csv")
    parts = temporal_partitions(df)
    tr = parts["train"]
    va = parts["validation"]
    ca = parts["calibration"]
    te = parts["test"]
    fit = np.r_[tr, va]
    y = df.perceived_fatigue.to_numpy()
    child = pd.read_csv(root / "results/ten_sets/replica_features.csv")
    child = child[child.parent_trial_uid.isin(df.trial_uid.iloc[te])].copy()
    # Fix both child and parent order for common-random-number neural robustness.
    children = [
        child[child.replica_id == rep]
        .set_index("parent_trial_uid")
        .loc[df.trial_uid.iloc[te]]
        .reset_index()
        for rep in sorted(child.replica_id.unique())
    ]
    protocol = dict(
        models=list(candidates())
        + (["cnn_emg", "cnn_emg_context"] if include_cnn else []),
        target="same-person onsite 0–10 rating; score >=7 alert",
        split={k: len(v) for k, v in parts.items()},
        selection="hyperparameters: 24 validation trials; family: 16 personal-calibration leave-one-out MAE",
        model_fit="104 original train+validation trials; never duplicate GP observations",
        personal_calibration="16 labels for load-shrunken offsets; scale selected by rounded/censored Gaussian NLL on leave-one-out adjusted calibration predictions",
        probability="11 ordinal categories, thresholds 0.5…9.5 and endpoint tail mass; P(score>=7) sums category mass 7…10",
        alert="fixed point score>=7, distinct from probability operating point; no test threshold tuning",
        uncertainty="approximate score distributions; no guaranteed cross-person coverage; adaptation and interval-scale selection share the 16 calibration labels",
        risk_coverage="score error ranked by predictive SD; binary errors ranked by probability entropy; ties randomized in expectation; coverage+retained counts always reported",
        robustness="9 variants of 32 test parents=288 correlated perturbations, not independent observations; common dropout draws for neural parent/child",
        scope="exploratory historical development data, one participant; no clinical or future-onset validation",
        candidate_grid=candidates(),
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    search = []
    packs = {}
    for name, specs in candidates().items():
        attempts = []
        for kind, family, param in specs:
            m = ProbabilisticScoreModel(kind, family, param).fit(df.iloc[tr], y[tr])
            mu, sd = m.predict(df.iloc[va])
            score = float(mean_absolute_error(y[va], mu))
            row = dict(
                model=name,
                kind=kind,
                family=family,
                parameters=json.dumps(param),
                validation_mae=score,
            )
            search.append(row)
            attempts.append((score, kind, family, param))
        best = min(attempts, key=lambda x: x[0])
        score, kind, family, param = best
        m = ProbabilisticScoreModel(kind, family, param).fit(df.iloc[fit], y[fit])
        cm, cs = m.predict(df.iloc[ca])
        tm, ts = m.predict(df.iloc[te])
        calibration = calibrate_personal(y[ca], cm, cs, df.condition_y.iloc[ca])
        bundle = dict(
            model=m,
            calibration=calibration,
            validation_mae=score,
            train_ids=df.trial_uid.iloc[fit].tolist(),
            calibration_ids=df.trial_uid.iloc[ca].tolist(),
            test_ids=df.trial_uid.iloc[te].tolist(),
            score_threshold=7,
        )
        joblib.dump(bundle, out / f"{name}.joblib")
        packs[name] = dict(
            calibration=calibration,
            validation_mae=score,
            raw_mu=tm,
            raw_sd=ts,
            children=[m.predict(c) for c in children],
            kind=kind,
        )
        print(
            "Fitted advanced model", name, "validation MAE", round(score, 4), flush=True
        )
    if include_cnn:
        # Frozen original CNN ensemble weights: no CNN retraining on historical tests.
        import torch

        signal = np.load(root / "results/direct_fatigue/local_signal_cache.npz")[
            "signals"
        ]
        allwaves = np.load(root / "results/ten_sets/near_real_waveforms.npz")["signals"]
        fullchild = pd.read_csv(root / "results/ten_sets/replica_features.csv")
        indices = dict(zip(fullchild.trial_uid, fullchild.index))
        oldmetrics = pd.read_csv(root / "results/direct_fatigue/metrics.csv").set_index(
            "model"
        )
        for name in ["cnn_emg", "cnn_emg_context"]:
            bundles = cnn_bundles(root, name)
            cm, cs = cnn_score(bundles, signal[ca], df.iloc[ca], name)
            tm, ts = cnn_score(bundles, signal[te], df.iloc[te], name)
            calibration = calibrate_personal(y[ca], cm, cs, df.condition_y.iloc[ca])
            cp = []
            for c in children:
                raw = np.stack([allwaves[indices[uid], :2000].T for uid in c.trial_uid])
                cp.append(cnn_score(bundles, raw, c, name))
            packs[name] = dict(
                calibration=calibration,
                validation_mae=float(oldmetrics.loc[name, "validation_mae"]),
                raw_mu=tm,
                raw_sd=ts,
                children=cp,
                kind="CNN ensemble + MC Dropout + predicted variance",
            )
            joblib.dump(
                dict(
                    calibration=calibration,
                    source_weights=[
                        f"results/direct_fatigue/{name}_{seed}.pt"
                        for seed in [42, 93, 173]
                    ],
                    validation_mae=float(oldmetrics.loc[name, "validation_mae"]),
                    inference_draws=32,
                ),
                out / f"{name}_calibration.joblib",
            )
            print("Scored frozen CNN and nine replica sets", name, flush=True)
    rows = []
    preds = []
    coverage = []
    robust = []
    childpred = []
    selections = []
    for name, p in packs.items():
        cal = p["calibration"]
        selections.append(
            dict(
                model=name,
                validation_mae=p["validation_mae"],
                personal_calibration_loo_mae=cal["loo_mae"],
                uncertainty_scale=cal["sd_scale"],
            )
        )
        for policy in ["before_personal_calibration", "after_personal_calibration"]:
            mu, sd = (
                (p["raw_mu"], p["raw_sd"])
                if policy == "before_personal_calibration"
                else apply_personal(
                    p["raw_mu"], p["raw_sd"], df.condition_y.iloc[te], cal
                )
            )
            extra, prob, rc, brc = uq_metrics(
                y[te],
                mu,
                sd,
                (
                    cal["interval_quantile"]
                    if policy == "after_personal_calibration"
                    else None
                ),
            )
            rows.append(
                dict(
                    model=name,
                    policy=policy,
                    validation_mae=p["validation_mae"],
                    calibration_loo_mae=cal["loo_mae"],
                    **metrics(y[te], mu),
                    **extra,
                )
            )
            for k, (r, b) in enumerate(zip(rc, brc), 1):
                coverage.append(
                    dict(
                        model=name,
                        policy=policy,
                        coverage=k / len(te),
                        retained=k,
                        score_mae=float(r),
                        binary_error=float(b),
                    )
                )
            for i, mean, ss, probs in zip(te, mu, sd, prob):
                preds.append(
                    dict(
                        model=name,
                        policy=policy,
                        trial_uid=df.trial_uid.iloc[i],
                        observed_score=int(y[i]),
                        mean=float(mean),
                        sd=float(ss),
                        **{f"p_{k}": float(v) for k, v in enumerate(probs)},
                    )
                )
        base_mu, base_sd = apply_personal(
            p["raw_mu"], p["raw_sd"], df.condition_y.iloc[te], cal
        )
        diffs = []
        flips = []
        near = []
        for c, (mu, sd) in zip(children, p["children"]):
            mu, sd = apply_personal(mu, sd, c.condition_y, cal)
            delta = abs(mu - base_mu)
            flip = (mu >= 7) != (base_mu >= 7)
            diffs.extend(delta)
            flips.extend(flip)
            near.extend(delta[(y[te] >= 6) & (y[te] <= 8)])
            for parent, uid, pred, d, fli in zip(
                c.parent_trial_uid, c.trial_uid, mu, delta, flip
            ):
                childpred.append(
                    dict(
                        model=name,
                        parent_trial_uid=parent,
                        child_trial_uid=uid,
                        prediction=float(pred),
                        absolute_change=float(d),
                        alert_flipped=bool(fli),
                    )
                )
        robust.append(
            dict(
                model=name,
                original_test_parents=32,
                correlated_children=288,
                mean_absolute_score_change=float(np.mean(diffs)),
                p95_absolute_score_change=float(np.quantile(diffs, 0.95)),
                max_absolute_score_change=float(np.max(diffs)),
                alert_flip_rate=float(np.mean(flips)),
                alert_flip_count=int(np.sum(flips)),
                near_threshold_reference_6_to_8_mean_change=float(np.mean(near)),
            )
        )
    for name, data in [
        ("metrics", rows),
        ("predictions", preds),
        ("risk_coverage", coverage),
        ("robustness", robust),
        ("replica_predictions", childpred),
        ("selection", selections),
        ("validation_search", search),
    ]:
        pd.DataFrame(data).to_csv(out / f"{name}.csv", index=False)
    selected = min(selections, key=lambda r: r["personal_calibration_loo_mae"])["model"]
    (out / "selected.json").write_text(
        json.dumps(
            dict(
                selected_model=selected,
                selection="personal calibration leave-one-out MAE only; no test outcome selection",
                original_baseline="retained as original reference; new candidate is exploratory",
                scope="one-person retrospective development benchmark",
            ),
            indent=2,
        )
    )
    if (out / f"{selected}.joblib").exists():
        saved = joblib.load(out / f"{selected}.joblib")
        saved.update(
            selected_model=selected,
            selection_rule="minimum personal calibration leave-one-out MAE; historical exploratory data",
            target="onsite perceived fatigue, one participant",
            scope="original 104 training labels plus 16 personal-calibration ratings; no force target or new-person validation",
        )
        joblib.dump(saved, out / "selected_advanced_predictor.joblib")
    print("Calibration-selected advanced model:", selected, flush=True)
    return pd.DataFrame(rows)


def public_probability_audit(root):
    root = Path(root)
    out = root / "results/advanced"
    out.mkdir(exist_ok=True, parents=True)
    p = pd.read_csv(root / "results/full_prediction/predictions.csv")
    rows = []
    curves = []
    for (name, person), g in p.groupby(["model", "test_person"]):
        y = g.observed.to_numpy(int)
        probs = g[["p_0", "p_1", "p_2"]].to_numpy()
        score = probs @ np.arange(3)
        fat = probs[:, 2]
        ref = y == 2
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
        rc = risk_coverage(abs(y - score), entropy)
        rows.append(
            dict(
                model=name,
                person=int(person),
                **proper_scores(y, probs),
                binary_brier=float(np.mean((fat - ref) ** 2)),
                binary_nll=float(
                    log_loss(ref, np.c_[1 - fat, fat], labels=[False, True])
                ),
                average_precision=float(average_precision_score(ref, fat)),
                positive_prevalence=float(ref.mean()),
                score_aurc=float(rc.mean()),
                random_rejection_mae=float(np.mean(abs(y - score))),
                mae_at_80pct_coverage=float(rc[int(np.ceil(0.8 * len(y))) - 1]),
                quadratic_weighted_kappa=float(
                    cohen_kappa_score(
                        y, probs.argmax(1), labels=[0, 1, 2], weights="quadratic"
                    )
                ),
            )
        )
        for cut in np.linspace(0, 1, 6)[:-1]:
            ix = (fat >= cut) & (fat < cut + 0.2 if cut < 0.8 else fat <= 1)
            if ix.any():
                curves.append(
                    dict(
                        model=name,
                        person=int(person),
                        bin_low=float(cut),
                        count=int(ix.sum()),
                        fraction_of_person=float(ix.mean()),
                        mean_probability=float(fat[ix].mean()),
                        observed_frequency=float(ref[ix].mean()),
                    )
                )
    f = pd.DataFrame(rows)
    f.to_csv(out / "public_probability_per_person.csv", index=False)
    f.groupby("model").mean(numeric_only=True).drop(columns="person").to_csv(
        out / "public_probability_summary.csv"
    )
    pd.DataFrame(curves).to_csv(out / "public_calibration_bins.csv", index=False)
    paired = []
    wide = f.pivot(index="person", columns="model")
    rng = np.random.default_rng(8128)
    for metric in ["normalized_rps", "binary_brier", "categorical_nll", "score_aurc"]:
        for baseline in ["signal_task", "time_task"]:
            d = wide[metric]["fusion"] - wide[metric][baseline]
            boot = rng.choice(d.to_numpy(), size=(3000, len(d)), replace=True).mean(1)
            paired.append(
                dict(
                    metric=metric,
                    model="fusion",
                    baseline=baseline,
                    mean_difference=float(d.mean()),
                    low95=float(np.quantile(boot, 0.025)),
                    high95=float(np.quantile(boot, 0.975)),
                    unit="13 real participants; exploratory pointwise bootstrap, no multiplicity correction",
                )
            )
    pd.DataFrame(paired).to_csv(
        out / "public_paired_probability_differences.csv", index=False
    )
    return f


def predict_advanced_features(frame, model_path):
    saved = joblib.load(model_path)
    mu, sd = saved["model"].predict(frame)
    cal = saved["calibration"]
    mu, sd = apply_personal(mu, sd, frame.condition_y, cal)
    p = ordinal_probabilities(mu, sd)
    return dict(
        score=mu,
        sd=sd,
        probability_fatigued=p[:, 7:].sum(1),
        ordinal_probabilities=p,
        alert=mu >= 7,
        empirical_interval_lower=np.maximum(0, mu - cal["interval_quantile"] * sd),
        empirical_interval_upper=np.minimum(10, mu + cal["interval_quantile"] * sd),
        scope=saved["scope"],
    )


def direct_alert_experiment(root):
    """Separate fatigue-state task; criterion differs from rating regression.

    Supplementary exploratory branch: select settings/family by validation
    balanced accuracy then Brier, never by the already-inspected test outcomes.
    """
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score
    from scipy.special import logit

    root = Path(root)
    out = root / "results/advanced"
    df = pd.read_csv(root / "results/rated_trial_features.csv")
    parts = temporal_partitions(df)
    tr = parts["train"]
    va = parts["validation"]
    ca = parts["calibration"]
    te = parts["test"]
    fit = np.r_[tr, va]
    y = (df.perceived_fatigue.to_numpy() >= 7).astype(int)
    child = pd.read_csv(root / "results/ten_sets/replica_features.csv")
    child = child[child.parent_trial_uid.isin(df.trial_uid.iloc[te])].copy()
    grid = {
        f"logistic_{family}": [
            ("logistic", family, dict(C=c)) for c in [0.1, 1.0, 10.0, 100.0]
        ]
        for family in ["context", "emg", "fusion"]
    }
    grid["gpc_fusion"] = [
        ("gpc", "fusion", dict(length=l, amplitude=a))
        for l in [2.0, 8.0, 32.0]
        for a in [1.0, 10.0]
    ]
    grid["boosted_classifier_fusion"] = [
        ("hgb", "fusion", dict(leaf=l, l2=q)) for l in [5, 10] for q in [1.0, 10.0]
    ]

    def construct(kind, params):
        if kind == "logistic":
            return LogisticRegression(C=params["C"], max_iter=1000, random_state=84)
        if kind == "gpc":
            return GaussianProcessClassifier(
                kernel=ConstantKernel(
                    params["amplitude"], constant_value_bounds="fixed"
                )
                * Matern(params["length"], length_scale_bounds="fixed", nu=1.5),
                optimizer=None,
                random_state=84,
                max_iter_predict=100,
            )
        return HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=7,
            min_samples_leaf=params["leaf"],
            l2_regularization=params["l2"],
            learning_rate=0.05,
            random_state=84,
        )

    (out / "direct_alert_protocol.json").write_text(
        json.dumps(
            dict(
                scope="supplementary direct binary classification on historical development data; one person",
                reference="onsite self-report>=7",
                fit="104 original trials",
                selection="80/24 validation balanced accuracy then lower Brier; alert probability threshold fixed 0.5",
                personal_calibration="Platt logistic calibration using only separate 16 ratings, C=1 fixed; raw and calibrated results both reported",
                supplementary="Added to distinguish lower regression error from fatigue-state recognition; not a new independent confirmatory test",
                grid=grid,
            ),
            indent=2,
        )
    )
    rows = []
    preds = []
    search = []
    selection = []
    stress = []
    curves = []
    for name, specs in grid.items():
        opts = []
        for kind, family, param in specs:
            scaler = StandardScaler().fit(inputs(df.iloc[tr], family))
            m = construct(kind, param).fit(
                scaler.transform(inputs(df.iloc[tr], family)), y[tr]
            )
            p = m.predict_proba(scaler.transform(inputs(df.iloc[va], family)))[:, 1]
            ba = float(balanced_accuracy_score(y[va], p >= 0.5))
            brier = float(np.mean((y[va] - p) ** 2))
            opts.append((ba, -brier, kind, family, param))
            search.append(
                dict(
                    model=name,
                    parameters=json.dumps(param),
                    validation_balanced_accuracy=ba,
                    validation_brier=brier,
                )
            )
        best = max(opts, key=lambda r: r[:2])
        ba, neg_brier, kind, family, param = best
        scaler = StandardScaler().fit(inputs(df.iloc[fit], family))
        m = construct(kind, param).fit(
            scaler.transform(inputs(df.iloc[fit], family)), y[fit]
        )
        raw = m.predict_proba(scaler.transform(inputs(df.iloc[ca], family)))[:, 1]
        cal = LogisticRegression(C=1.0, random_state=84).fit(
            logit(np.clip(raw, 1e-6, 1 - 1e-6)).reshape(-1, 1), y[ca]
        )
        p = m.predict_proba(scaler.transform(inputs(df.iloc[te], family)))[:, 1]
        selection.append(
            dict(
                model=name, validation_balanced_accuracy=ba, validation_brier=-neg_brier
            )
        )
        joblib.dump(
            dict(
                model=m,
                scaler=scaler,
                family=family,
                platt=cal,
                probability_threshold=0.5,
                reference="onsite score>=7",
                training_ids=df.trial_uid.iloc[fit].tolist(),
                calibration_ids=df.trial_uid.iloc[ca].tolist(),
                test_ids=df.trial_uid.iloc[te].tolist(),
            ),
            out / f"alert_{name}.joblib",
        )
        after = cal.predict_proba(logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1))[
            :, 1
        ]
        for policy, prob in [("raw", p), ("personal_platt", after)]:
            tn, fp, fn, tp = confusion_matrix(y[te], prob >= 0.5, labels=[0, 1]).ravel()
            entropy = -(
                prob * np.log(np.maximum(prob, 1e-12))
                + (1 - prob) * np.log(np.maximum(1 - prob, 1e-12))
            )
            rc = risk_coverage(((prob >= 0.5) != y[te]).astype(float), entropy)
            rows.append(
                dict(
                    model=name,
                    policy=policy,
                    balanced_accuracy=float(
                        balanced_accuracy_score(y[te], prob >= 0.5)
                    ),
                    recall=float(tp / (tp + fn)),
                    specificity=float(tn / (tn + fp)),
                    precision=float(tp / max(tp + fp, 1)),
                    binary_brier=float(np.mean((prob - y[te]) ** 2)),
                    binary_nll=float(
                        log_loss(y[te], np.c_[1 - prob, prob], labels=[0, 1])
                    ),
                    average_precision=float(average_precision_score(y[te], prob)),
                    auroc=float(roc_auc_score(y[te], prob)),
                    binary_aurc=float(rc.mean()),
                    retained_error_at_80pct=float(rc[int(np.ceil(0.8 * len(te))) - 1]),
                    tp=int(tp),
                    fn=int(fn),
                    tn=int(tn),
                    fp=int(fp),
                )
            )
            for uid, target, value in zip(df.trial_uid.iloc[te], y[te], prob):
                preds.append(
                    dict(
                        model=name,
                        policy=policy,
                        trial_uid=uid,
                        observed_fatigue=int(target),
                        probability=float(value),
                        alert=bool(value >= 0.5),
                    )
                )
            for k, risk in enumerate(rc, 1):
                curves.append(
                    dict(
                        model=name,
                        policy=policy,
                        coverage=k / len(te),
                        retained=k,
                        error=float(risk),
                    )
                )
        rawchild = m.predict_proba(scaler.transform(inputs(child, family)))[:, 1]
        pc = cal.predict_proba(logit(np.clip(rawchild, 1e-6, 1 - 1e-6)).reshape(-1, 1))[
            :, 1
        ]
        byparent = dict(zip(df.trial_uid.iloc[te], after))
        original = np.array([byparent[uid] for uid in child.parent_trial_uid])
        delta = abs(pc - original)
        stress.append(
            dict(
                model=name,
                correlated_children=288,
                real_parent_trials=32,
                mean_probability_change=float(delta.mean()),
                p95_probability_change=float(np.quantile(delta, 0.95)),
                maximum_probability_change=float(delta.max()),
                alert_flip_count=int(np.sum((pc >= 0.5) != (original >= 0.5))),
                alert_flip_rate=float(np.mean((pc >= 0.5) != (original >= 0.5))),
            )
        )
        print("Fitted direct alert classifier", name, flush=True)
    for name, data in [
        ("alert_metrics", rows),
        ("alert_predictions", preds),
        ("alert_validation_search", search),
        ("alert_selection", selection),
        ("alert_robustness", stress),
        ("alert_risk_coverage", curves),
    ]:
        pd.DataFrame(data).to_csv(out / f"{name}.csv", index=False)
    selected = max(
        selection,
        key=lambda r: (r["validation_balanced_accuracy"], -r["validation_brier"]),
    )["model"]
    (out / "selected_alert.json").write_text(
        json.dumps(
            dict(
                model=selected,
                rule="validation balanced accuracy, then Brier; fixed probability threshold 0.5; no test selection",
                scope="supplementary exploratory direct classifier",
            ),
            indent=2,
        )
    )
    return pd.DataFrame(rows)
