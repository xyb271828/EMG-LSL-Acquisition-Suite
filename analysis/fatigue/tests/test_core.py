"""Artificial fixtures only: no private signal samples or observed scores."""

import unittest
import numpy as np
import pandas as pd
from emgfatigue.advanced import (
    ordinal_probabilities,
    proper_scores,
    risk_coverage,
    calibrate_personal,
    ProbabilisticScoreModel,
    apply_personal,
)
from emgfatigue.augmentation import transform_parent, eligible_children
from emgfatigue.design import SIGNAL
from emgfatigue.splits import temporal_partitions
from emgfatigue.features import CausalEMGFilter


class CoreTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(71)
        rows = []
        for load in [0.2, 0.4, 0.6, 0.8]:
            for k in range(1, 41):
                row = dict(
                    condition_y=load,
                    trial_in_condition=k,
                    recording_id=f"fixture-{load}",
                    trial_uid=f"fixture-{load}-{k}",
                    start_time=100 + 4 * k,
                    end_time=102 + 4 * k,
                )
                row.update(
                    {
                        c: float(v)
                        for c, v in zip(SIGNAL, rng.uniform(0.5, 2, size=len(SIGNAL)))
                    }
                )
                rows.append(row)
        return pd.DataFrame(rows)

    def test_endpoint_probability_mass_and_proper_score(self):
        p = ordinal_probabilities(np.array([0.0, 10.0, 7.0]), np.array([0.15] * 3))
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-12)
        self.assertGreater(p[0, 0], 0.999)
        self.assertGreater(p[1, 10], 0.999)
        self.assertEqual(
            proper_scores([0, 10, 7], np.eye(11)[[0, 10, 7]])["normalized_rps"], 0.0
        )

    def test_tied_uncertainty_does_not_exploit_order(self):
        e = np.array([0.0, 4.0, 2.0, 0.0])
        sd = np.ones(4)
        np.testing.assert_allclose(risk_coverage(e, sd), e.mean())
        np.testing.assert_allclose(risk_coverage(e[::-1], sd), risk_coverage(e, sd))

    def test_causal_filter_chunking_matches_continuous(self):
        x = np.random.default_rng(5).normal(size=(3000, 2))
        a = CausalEMGFilter().process(x)
        f = CausalEMGFilter()
        b = np.r_[f.process(x[:913]), f.process(x[913:])]
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_split_retains_temporal_gap(self):
        df = self.fixture()
        p = temporal_partitions(df)
        self.assertEqual(
            {k: len(v) for k, v in p.items()},
            dict(train=80, validation=24, calibration=16, embargo=8, test=32),
        )
        self.assertTrue((df.trial_in_condition.iloc[p["test"]] >= 33).all())

    def test_descendants_inherit_parent_partition(self):
        children = pd.DataFrame(
            dict(parent_trial_uid=["train", "test", "train", "calibration"])
        )
        result = eligible_children(children, ["train"])
        self.assertEqual(result.parent_trial_uid.tolist(), ["train", "train"])

    def test_near_real_augmentation_reproducible(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(3000, 2))
        donors = [rng.normal(size=(3020, 2))]
        a, _ = transform_parent(x, "explicit-artificial-fixture", 1, donors)
        b, _ = transform_parent(x, "explicit-artificial-fixture", 1, donors)
        np.testing.assert_array_equal(a, b)
        self.assertGreater(np.corrcoef(x.ravel(), a.ravel())[0, 1], 0.999)
        self.assertFalse(np.array_equal(a, x))

    def test_score_model_does_not_read_target_column(self):
        frame = self.fixture()
        p = temporal_partitions(frame)
        y = (
            2
            + frame.condition_y.to_numpy() * 4
            + frame.trial_in_condition.to_numpy() / 20
        )
        m = ProbabilisticScoreModel(
            "residual_gp", "fusion", dict(length=4.0, noise=0.1, alpha=0.01)
        ).fit(frame.iloc[p["train"]], y[p["train"]])
        rows = frame.iloc[p["test"]].copy()
        a = m.predict(rows)
        rows["perceived_fatigue"] = -999
        b = m.predict(rows)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_personal_calibration_has_no_test_input(self):
        y = np.array([2.0, 3.0, 4.0, 5.0] * 4)
        mu = y + 0.4
        load = np.repeat([0.2, 0.4, 0.6, 0.8], 4)
        cal = calibrate_personal(y, mu, np.ones(16), load)
        self.assertLess(cal["loo_mae"], 1e-10)
        mean, sd = apply_personal(
            np.array([6.4, 7.4]), np.ones(2), np.array([0.2, 0.8]), cal
        )
        np.testing.assert_allclose(mean, [6.0, 7.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
