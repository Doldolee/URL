"""Leakage and estimator-contract tests; no real model training is performed."""

import agent
import metric
import model
import util
import util
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import agent as behavior


class BehaviorTests(unittest.TestCase):
    def test_terminal_boundaries_and_subject_split(self):
        done = np.tile([0, 1], 12)
        groups = np.repeat(np.repeat(np.arange(6), 2), 2)
        result = util.grouped_partition(done, groups, random_seed=42)
        self.assertEqual(sum(map(len, result.values())), len(done))
        group_sets = {key: set(groups[indices]) for key, indices in result.items()}
        for left, right in [("fit", "calibration"), ("fit", "validation"), ("calibration", "validation")]:
            self.assertFalse(group_sets[left] & group_sets[right])
        self.assertEqual(util.episode_slices(done)[-1].stop, len(done))
        with self.assertRaises(ValueError):
            util.episode_slices([0, 1, 0])
        with self.assertRaises(ValueError):
            util.grouped_partition(done, np.arange(len(done)))

    def test_full_25_classes_preserves_model_class_order_and_zero_support(self):
        class Stub:
            classes_ = np.array([24, 0, 7])
            def predict_proba(self, state):
                return np.tile([.2, .5, .3], (len(state), 1))
        result = model.full_action_proba(Stub(), np.zeros((2, 43)))
        self.assertEqual(result.shape, (2, 25))
        np.testing.assert_array_equal(result[:, [24, 0, 7]], [[.2, .5, .3]] * 2)
        self.assertTrue((result[:, 1:7] == 0).all())
        np.testing.assert_allclose(result.sum(1), 1)

    def test_metrics_known_multiclass_values(self):
        probability = np.zeros((2, 25))
        probability[:, :2] = [[.8, .2], [.1, .9]]
        result = metric.probability_metrics([0, 1], probability, ece_bins=10)
        self.assertAlmostEqual(result["log_loss"], -np.log([.8, .9]).mean())
        self.assertAlmostEqual(result["brier_multiclass_sum"], .05)
        self.assertAlmostEqual(result["top_label_ece"], .15)
        self.assertEqual(result["confusion_true_row_predicted_column"][0][0], 1)
        zeros = probability.copy()
        zeros[0] = 0
        zeros[0, 1] = 1
        result = metric.probability_metrics([0, 1], zeros)
        self.assertEqual(result["observed_action_zero_probability_count"], 1)
        self.assertEqual(zeros[0, 0], 0)  # reporting floor never mutates probabilities

    def test_pipeline_fits_only_disjoint_train_partitions(self):
        # Every synthetic episode contains every action to exercise class checks.
        state = np.c_[np.arange(250), np.zeros((250, 42))].astype(np.float32)
        action = np.tile(np.arange(25), 10)
        done = np.tile(np.r_[np.zeros(24), 1], 10)

        class RF:
            def __init__(self, **params): self.params = params
            def fit(self, x, y):
                self.seen = x[:, 0].astype(int)
                self.classes_ = np.unique(y)
                return self
            def predict_proba(self, x):
                return np.full((len(x), 25), 1/25)
            def get_params(self): return self.params
        class Frozen:
            def __init__(self, estimator): self.estimator = estimator
        class Calibrated:
            def __init__(self, estimator, **kwargs): self.estimator = estimator
            def fit(self, x, y):
                self.seen = x[:, 0].astype(int)
                self.classes_ = self.estimator.estimator.classes_
                return self
            def predict_proba(self, x): return self.estimator.estimator.predict_proba(x)
        with patch("sklearn.ensemble.RandomForestClassifier", RF), patch("sklearn.frozen.FrozenEstimator", Frozen), patch("sklearn.calibration.CalibratedClassifierCV", Calibrated):
            raw, calibrated, report = agent.fit_behavior(state, action, done)
        partition = report["_partition_indices"]
        np.testing.assert_array_equal(raw.seen, partition["fit"])
        np.testing.assert_array_equal(calibrated.seen, partition["calibration"])
        self.assertFalse(set(raw.seen) & set(calibrated.seen))
        self.assertFalse(set(partition["validation"]) & (set(raw.seen) | set(calibrated.seen)))
        self.assertEqual(report["selected_behavior"], "raw")  # declared tie break
        self.assertFalse(report["final_refit"])
        self.assertFalse(report["policy_support_mask"])

    def test_exact_episode_link_excludes_overlapping_test_subject(self):
        header = ["traj", "step", "m:icustayid"] + sorted(util.OBS_COLUMNS) + sorted(util.DEM_COLUMNS) + ["a:action", "r:reward"]
        numeric_columns = [col for col in header if col in util.OBS_COLUMNS]
        numeric_columns += [col for col in header if col in util.DEM_COLUMNS]
        csv_rows = []
        datasets = []
        for ep in range(3):
            state = np.asarray([[ep * 100 + step + feature/100 for feature in range(43)] for step in range(3)], dtype=np.float32)
            for step in range(3):
                row = {"traj": ep, "step": step, "m:icustayid": 100 + ep,
                       "a:action": step, "r:reward": int(step == 2)}
                row.update(dict(zip(numeric_columns, state[step])))
                csv_rows.append(row)
            datasets.append({"state": state[:-1], "next_state": state[1:],
                             "action": np.array([0, 1]), "reward": np.array([0, 1]), "done": np.array([0, 1])})
        train = {key: np.concatenate([datasets[0][key], datasets[1][key]]) for key in datasets[0]}
        test = datasets[2]
        with tempfile.TemporaryDirectory() as directory:
            cohort = Path(directory) / "cohort.csv"
            demog = Path(directory) / "demog.csv"
            with cohort.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=header)
                writer.writeheader(); writer.writerows(csv_rows)
            demog.write_text("subject_id|icustay_id\n10|100\n20|101\n10|102\n")
            train_groups, test_groups, keep, report = util.recover_subject_groups(train, test, cohort, demog)
        np.testing.assert_array_equal(train_groups, [10, 10, 20, 20])
        np.testing.assert_array_equal(test_groups, [10, 10])
        np.testing.assert_array_equal(keep, [False, False, True, True])
        self.assertEqual(report["overlapping_subjects"], 1)
        self.assertEqual(report["excluded_train_episodes"], 1)
        self.assertEqual(report["matched_train_episodes"], 2)


if __name__ == "__main__":
    unittest.main()
