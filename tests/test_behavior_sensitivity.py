"""Train-only sensitivity models, probability alignment and serialization."""
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch

import agent
import model


def synthetic_data():
    rng = np.random.default_rng(8)
    # Every patient has common action 0 and five rare action labels.
    groups = np.repeat(np.arange(20), 12)
    action = np.tile(np.r_[np.zeros(7, dtype=int), np.arange(1, 6)], 20)
    state = rng.normal(size=(len(groups), 4))
    state[:, 0] += action*.4
    done = np.tile(np.r_[np.zeros(11), 1], 20)
    partition = {"fit": np.flatnonzero(groups < 12),
                 "calibration": np.flatnonzero((groups >= 12) & (groups < 16)),
                 "validation": np.flatnonzero(groups >= 16)}
    state[partition["calibration"], 1] += 2.
    state[partition["validation"], 1] -= 3.
    return state, action, done, groups, partition


class SensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state, cls.action, cls.done, cls.groups, cls.partition = synthetic_data()
        cls.models, cls.report = agent.fit_behavior_sensitivity_candidates(
            cls.state, cls.action, cls.done, cls.groups, 6, cls.partition,
            seed=15, n_jobs=1)

    def test_generic_action_probabilities_and_rare_class_smoothing(self):
        self.assertEqual(len(self.models), 15)
        expected_prior = np.bincount(self.action[self.partition["fit"]], minlength=6)
        expected_prior = expected_prior/expected_prior.sum()
        np.testing.assert_allclose(self.report["fit_action_prior"], expected_prior)
        for name, predictor in self.models.items():
            probability = predictor.predict_proba(self.state[:9])
            self.assertEqual(probability.shape, (9, 6), name)
            np.testing.assert_array_equal(predictor.classes_, np.arange(6))
            np.testing.assert_allclose(probability.sum(1), 1., atol=1e-12)
            self.assertTrue(np.isfinite(probability).all(), name)
            self.assertTrue((probability > 0).all(), name)
            logits = predictor.decision_function(self.state[:9])
            self.assertEqual(logits.shape, (9, 6), name)
        for name in ["knn_k100", "knn_k300", "knn_k1000"]:
            predictor = self.models[name]
            self.assertEqual(predictor.n_neighbors, min(int(name[5:]), len(self.partition["fit"])))
        # No neighbors of action 5 is still positive through the FIT prior.
        counts = np.zeros((1, 6)); counts[0, 0] = 100
        p = self.models["knn_k100"].probabilities_from_counts(counts)
        self.assertGreater(p[0, 5], 0.)
        self.assertIsNone(self.report["saved_probability_floor"])

    def test_json_report_and_joblib_round_trip(self):
        json.dumps(self.report, allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"predictors.joblib"
            joblib.dump(self.models, path)
            restored = joblib.load(path)
        for name in self.models:
            np.testing.assert_array_equal(restored[name].predict_proba(self.state[:7]),
                                          self.models[name].predict_proba(self.state[:7]))

    def test_scaler_and_smoothing_use_declared_partitions(self):
        np.testing.assert_allclose(self.report["scaler"]["mean"],
                                   self.state[self.partition["fit"]].mean(0))
        for name in ["knn_k100", "cluster_k50"]:
            predictor = self.models[name]
            np.testing.assert_allclose(predictor.scaler.mean_,
                                       self.state[self.partition["fit"]].mean(0))
            candidate = self.report["training"][name]["smoothing_calibration"]
            losses = candidate["calibration_log_losses"]
            grid = candidate["concentration_grid"]
            self.assertEqual(predictor.concentration, grid[int(np.argmin(losses))])
            self.assertAlmostEqual(min(losses), self.report["calibration"][name]["log_loss"])
        for name, rows in self.partition.items():
            self.assertEqual(self.report["split"][name]["rows"], len(rows))
            self.assertEqual(self.report["split"][name]["row_indices_sha256_int64"],
                             agent._index_hash(rows))
        for name, predictor in self.models.items():
            probability = predictor.predict_proba(self.state[self.partition["validation"]])
            observed = probability[np.arange(len(probability)),
                                   self.action[self.partition["validation"]]]
            self.assertAlmostEqual(float(-np.log(observed).mean()),
                                   self.report["validation"][name]["log_loss"])

    def test_validation_labels_do_not_fit_models_and_cpu_seed_is_deterministic(self):
        state = self.state.copy()
        action = self.action.copy()
        state[self.partition["validation"]] += 100.
        action[self.partition["validation"]] = 0
        rng_state = torch.get_rng_state().clone()
        threads = torch.get_num_threads()
        second, second_report = agent.fit_behavior_sensitivity_candidates(
            state, action, self.done, self.groups, 6, self.partition, seed=15, n_jobs=1)
        self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())
        for name in self.models:
            np.testing.assert_array_equal(second[name].predict_proba(self.state[:8]),
                                          self.models[name].predict_proba(self.state[:8]))
        self.assertNotEqual(second_report["validation"]["mlp_64x64_raw"]["log_loss"],
                            self.report["validation"]["mlp_64x64_raw"]["log_loss"])

    def test_partition_errors_raise_before_training(self):
        overlapping = {key: rows.copy() for key, rows in self.partition.items()}
        overlapping["validation"][0] = overlapping["fit"][0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            agent.fit_behavior_sensitivity_candidates(self.state, self.action, self.done,
                self.groups, 6, overlapping)
        incomplete = {key: rows.copy() for key, rows in self.partition.items()}
        incomplete["validation"] = incomplete["validation"][1:]
        with self.assertRaisesRegex(ValueError, "cover every"):
            agent.fit_behavior_sensitivity_candidates(self.state, self.action, self.done,
                self.groups, 6, incomplete)
        wrong_groups = self.groups.copy()
        wrong_groups[self.partition["validation"]] = 0
        with self.assertRaisesRegex(ValueError, "Patient groups overlap"):
            agent.fit_behavior_sensitivity_candidates(self.state, self.action, self.done,
                wrong_groups, 6, self.partition)
        missing_action = self.action.copy()
        missing_action[self.partition["fit"]] = 0
        with self.assertRaisesRegex(ValueError, "fit patients lack"):
            agent.fit_behavior_sensitivity_candidates(self.state, missing_action, self.done,
                self.groups, 6, self.partition)

    def test_generic_temperature_respects_reversed_class_columns(self):
        class ReversedScores:
            classes_ = np.arange(6)[::-1]
            def decision_function(self, state):
                return np.tile(np.arange(6, dtype=float), (len(state), 1))
        predictor = model.BehaviorLogitPredictor(ReversedScores(), 6, temperature=2.)
        probability = predictor.predict_proba(np.zeros((2, 1)))
        self.assertEqual(probability.argmax(1).tolist(), [0, 0])
        np.testing.assert_array_equal(predictor.decision_function(np.zeros((1, 1)))[0],
                                      np.arange(6)[::-1]/2.)
        calibrated, report = agent._fit_sensitivity_temperature(ReversedScores(),
            np.zeros((20, 1)), np.zeros(20, dtype=int), 6)
        self.assertLess(report["calibration_log_loss"], report["uncalibrated_calibration_log_loss"])
        self.assertGreater(calibrated.predict_proba(np.zeros((1, 1)))[0, 0],
                           model.BehaviorLogitPredictor(ReversedScores(), 6).predict_proba(np.zeros((1, 1)))[0, 0])

    def test_tied_neighbors_use_same_max_k_prefix_as_calibration(self):
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        state = np.zeros((1500, 2))
        actions = np.arange(len(state)) % 6
        scaler = StandardScaler().fit(state)
        neighbors = NearestNeighbors(algorithm="brute", n_jobs=1).fit(state)
        predictor = model.NeighborCountBehaviorPredictor(scaler, neighbors, actions,
            np.full(6, 1/6), n_neighbors=100, query_neighbors=1000)
        query = np.zeros((2, 2))
        index = neighbors.kneighbors(query, n_neighbors=1000, return_distance=False)[:, :100]
        expected = np.stack([np.bincount(actions[row], minlength=6) for row in index])
        np.testing.assert_array_equal(predictor.action_counts(query), expected)

    def test_generic_twenty_five_action_alignment(self):
        class Scores:
            classes_ = np.arange(25)[::-1]
            def decision_function(self, state):
                return np.tile(np.linspace(-2., 2., 25), (len(state), 1))
        predictor = model.BehaviorLogitPredictor(Scores(), num_actions=25)
        probability = predictor.predict_proba(np.zeros((3, 1)))
        self.assertEqual(probability.shape, (3, 25))
        self.assertEqual(probability.argmax(1).tolist(), [0, 0, 0])
        np.testing.assert_allclose(probability.sum(1), 1.)


if __name__ == "__main__":
    unittest.main()
