"""Local-only data paths: raw recordings never belong in this repository."""

import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "action",
        choices=[
            "extract",
            "cnn-fit",
            "benchmark",
            "augment",
            "public-audit",
            "predict",
        ],
    )
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Private data/work directory outside the repository",
    )
    p.add_argument(
        "--skip-cnn",
        action="store_true",
        help="Run feature models without Torch or saved CNN weights",
    )
    p.add_argument("--features", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.action == "extract":
        from .data import extract

        print(extract(args.root).shape)
    elif args.action == "augment":
        from .augmentation import close_replicas_experiment

        close_replicas_experiment(args.root, replicas=9)
    elif args.action == "benchmark":
        from .advanced import advanced_experiment, direct_alert_experiment

        print(
            advanced_experiment(args.root, include_cnn=not args.skip_cnn).to_string(
                index=False
            )
        )
        direct_alert_experiment(args.root)
    elif args.action == "public-audit":
        from .advanced import public_probability_audit

        public_probability_audit(args.root)
    elif args.action == "cnn-fit":
        from .train_cnn import fit_saved_cnns

        fit_saved_cnns(args.root)
    else:
        if args.features is None or args.output is None:
            p.error("predict requires --features and --output")
        import pandas as pd
        from .advanced import predict_advanced_features

        path = (
            args.model
            or args.root / "results/advanced/selected_advanced_predictor.joblib"
        )
        frame = pd.read_csv(args.features)
        r = predict_advanced_features(frame, path)
        result = pd.DataFrame(
            {
                k: r[k]
                for k in [
                    "score",
                    "sd",
                    "probability_fatigued",
                    "alert",
                    "empirical_interval_lower",
                    "empirical_interval_upper",
                ]
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
