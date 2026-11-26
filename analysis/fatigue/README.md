# Perceived-fatigue modeling core

Offline, trial-end EMG analysis for the acquisition suite. This directory contains
core source and tests only. Recordings, onsite ratings, augmented waveforms,
trained models, cached predictions and experiment-history reports are distributed
separately and must remain outside Git.

## Retained experiment

- **Primary model:** regularized EMG + known load/progress regression, with personal
  score calibration. Family selection uses calibration leave-one-out MAE.
- **Candidate:** a load/progress mean model with a residual Gaussian process;
  retained for independent follow-up rather than selected by test performance.
- **Comparators:** Bayesian ridge, linear–Matérn GPR, quantile/tree methods,
  original three-seed CNN + MC Dropout, and direct fatigue-state classifiers.
- **Evaluation:** MAE/RMSE, ordinal RPS and NLL, Brier, interval coverage/width/score,
  tie-aware risk–coverage, probability reliability, and parent-linked alert flips.

The local reference is onsite **0–10 perceived fatigue**, with **score >=7** as the
operational fatigue alert. There is one participant with 160 matched contractions:
four nominal MVC conditions (20/40/60/80%), forty contractions each. Nine near-real
augmentation sets increase record capacity to 1,600; they do not add people or
independent fatigue observations. No force stream is imputed.

Each load uses trials 1–20 to train, 21–26 to validate, 27–30 for personal
calibration, 31–32 as a gap, and 33–40 to test. Final base fitting uses 104 original
train+validation records plus sixteen separate personal-calibration ratings.
This is retrospective within-recording later-trial evaluation, not a globally
chronological prospective study or cross-person validation. Historical tests have
been inspected during development. Probability intervals remain empirical
research diagnostics, not clinical or cross-person guarantees.

## Install

From this directory, using Python 3.11+:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

The frozen CNN comparison additionally requires Torch:

```bash
python -m pip install -r requirements-neural-cpu.txt
```

## Local private-data layout

Unzip the separately supplied private bundle outside this repository. Its root
contains `data/raw/{20,40,60,80}MVC_left.xdf`, `data/ratings.csv`, derived features,
nine parent-linked variant sets, and frozen reference CNN weights. The saved
public OOF predictions support a separate probability audit; native public labels
remain 0/1/2. New local GPR models are not claimed to have public cross-person
validation.

```bash
# Set this to the extracted private bundle's root, not the repository.
EMG_DATA_ROOT=/absolute/path/to/EMG_Private_Experiment_Data

# Optional: rebuild original features and the CNN signal cache from raw XDF.
emg-fatigue extract --root "$EMG_DATA_ROOT"

# Feature-model comparisons, with no Torch or CNN-weight dependency.
emg-fatigue benchmark --root "$EMG_DATA_ROOT" --skip-cnn

# Include the frozen original CNN reference when Torch is installed.
emg-fatigue benchmark --root "$EMG_DATA_ROOT"

# Optional: rebuild CNN reference weights, using only training/validation parents.
emg-fatigue cnn-fit --root "$EMG_DATA_ROOT"

# Optional: rebuild nine high-similarity augmentation sets and guarded comparisons.
emg-fatigue augment --root "$EMG_DATA_ROOT"

# Audit the original public model predictions, without retraining public GPRs.
emg-fatigue public-audit --root "$EMG_DATA_ROOT"

emg-fatigue predict --root "$EMG_DATA_ROOT"   --features /absolute/path/to/features.csv --output /absolute/path/to/predictions.csv
```

`data/ratings.csv` requires `condition_y`, `trial_in_condition`, `Fatigue`.
`condition_y` is 0.2/0.4/0.6/0.8, not 20/40/60/80. Input feature columns are
specified in `design.py`; `data.py` derives them from the matched source records.
Outputs are written only beneath the specified private root.

A parent and all its variants must stay in the same split. Augmented calibration,
gap and test descendants never fit a model. Near-duplicates are not treated as
independent GP observations. The optional augmentation command preserves a total
fitting weight of one per parent family.

## Interpret results

Model complexity is not a success criterion. Keep the validation/calibration
selection rule and report false positives as well as missed alerts. A direct
classifier that predicts fatigue for every trial has full recall but no
specificity. Do not promote a candidate only because it wins on an already-seen
test table. The CLI preserves all comparisons locally and retains selection
metadata in `results/advanced/selected.json`.

RPS is normalized within the target scale: local scores use eleven categories,
public scores three. Do not directly equate those tasks. Risk–coverage must be
reported together with the fraction of cases retained; rejected cases still
require review. Constant uncertainty uses tie-aware expected risk, avoiding an
artificial advantage from data-row order.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use explicit artificial fixtures, not private recordings. The original
acquisition implementation remains under the repository's existing acquisition
folder and is not modified by this analysis package.
