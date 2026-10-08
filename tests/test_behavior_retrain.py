"""Candidate class mapping, calibration, serialization, and patient leakage tests."""

import agent
import model
import util
import pickle
import unittest
from unittest.mock import patch

import numpy as np

import agent as behavior
import agent as retrain


class Scores:
    classes_ = np.arange(25)[::-1]

    def decision_function(self, state):
        return np.tile(np.linspace(-2., 2., 25), (len(state), 1))

    def predict_proba(self, state):
        x = self.decision_function(state)
        p = np.exp(x - x.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)


class BehaviorRetrainTests(unittest.TestCase):
    def test_explicit_class_mapping_and_temperature_serialization(self):
        x = np.zeros((3, 2))
        raw = model.FullActionPredictor(Scores())
        p = raw.predict_proba(x)
        np.testing.assert_allclose(p[:, ::-1], Scores().predict_proba(x))
        calibrated = model.TemperatureScaledPredictor(Scores(), 2.)
        q = calibrated.predict_proba(x)
        np.testing.assert_array_equal(calibrated.classes_, np.arange(25))
        np.testing.assert_allclose(q.sum(1), 1.)
        self.assertLess(q.max(), p.max())
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(calibrated)).predict_proba(x), q)

    def test_zero_probabilities_remain_zero_and_missing_classes_raise(self):
        class ZeroModel:
            classes_ = np.arange(25)[::-1]
            def predict_proba(self, x):
                p = np.zeros((len(x), 25)); p[:, 0] = 1.
                return p
        p = model.FullActionPredictor(ZeroModel()).predict_proba(np.zeros((2, 1)))
        self.assertTrue((p[:, :24] == 0).all())
        np.testing.assert_array_equal(p[:, 24], 1.)
        class Missing:
            classes_ = np.arange(24)
        with self.assertRaisesRegex(ValueError, "all 25"):
            model.FullActionPredictor(Missing())

    def test_temperature_optimizer_uses_labels_and_stable_logits(self):
        # Class 0 has the largest logit in this reversed class ordering.
        x = np.zeros((50, 1))
        predictor, report = agent._fit_temperature(Scores(), x, np.zeros(50, dtype=int))
        self.assertGreater(predictor.predict_proba(x)[0, 0], Scores().predict_proba(x)[0, -1])
        self.assertLess(report["calibration_log_loss"], report["uncalibrated_calibration_log_loss"])
        self.assertTrue(-2 <= report["log_temperature"] <= 2)

    def test_missing_fit_classes_fail_before_rf_training(self):
        x = np.zeros((40, 2)); action = np.zeros(40, dtype=int)
        done = np.tile([0, 1], 20); groups = np.repeat(np.arange(20), 2)
        with patch.object(retrain, "fit_behavior") as fit:
            with self.assertRaisesRegex(ValueError, "fit patients lack"):
                agent.fit_behavior_candidates(x, action, done, groups)
            fit.assert_not_called()

    def test_candidates_share_patient_split_and_fit_scaler_only_on_fit_rows(self):
        # Two complete episodes per patient, each containing every action.
        action = np.tile(np.arange(25), 20)
        state = np.c_[np.arange(len(action)), np.ones(len(action))].astype(float)
        done = np.tile(np.r_[np.zeros(24), 1], 20)
        groups = np.repeat(np.arange(10), 50)
        partition = util.grouped_partition(done, groups, random_seed=42)

        class Logistic:
            fits = []
            def __init__(self, **params):
                self.params = params
            def fit(self, x, y):
                self.classes_ = np.unique(y); self.n_iter_ = np.array([1]); self.n_features_in_ = x.shape[1]
                self.fits.append(np.asarray(x).copy()); return self
            def decision_function(self, x):
                return np.zeros((len(x), 25))
            def predict_proba(self, x):
                return np.full((len(x), 25), 1/25)
        class Uniform:
            classes_ = np.arange(25)
            def predict_proba(self, x): return np.full((len(x), 25), 1/25)
        def rf_fit(*args, **kwargs):
            np.testing.assert_array_equal(kwargs["groups"], groups)
            return Uniform(), Uniform(), {"_partition_indices": partition.copy()}

        with patch.object(retrain, "fit_behavior", side_effect=rf_fit), patch("sklearn.linear_model.LogisticRegression", Logistic):
            selected, report, candidates = agent.fit_behavior_candidates(state, action, done, groups)
        self.assertEqual(len(candidates), 8)
        self.assertEqual(report["selected_behavior"], "rf_raw")
        self.assertIs(selected, candidates["rf_raw"])
        group_sets = {name: set(groups[idx]) for name, idx in report["_partition_indices"].items()}
        for a, b in [("fit", "calibration"), ("fit", "validation"), ("calibration", "validation")]:
            self.assertFalse(group_sets[a] & group_sets[b])
        for c in [.1, 1., 10.]:
            predictor = candidates["logistic_C" + format(c, "g") + "_raw"]
            np.testing.assert_allclose(predictor.estimator[0].mean_, state[partition["fit"]].mean(axis=0))
        for array in Logistic.fits:
            self.assertEqual(len(array), len(partition["fit"]))
            np.testing.assert_allclose(array.mean(axis=0), 0., atol=1e-12)
        for name in partition:
            self.assertEqual(report["split"][name]["rows"], len(partition[name]))
        self.assertEqual(report["validation"]["rf_raw"]["per_action_recorded_action_strata"]["0"]["rows"], 4)
        np.testing.assert_allclose(selected.predict_proba(state[:3]).sum(1), 1.)
        self.assertIsNone(report["saved_probability_floor"])

    def test_fixed_logistic_refit_uses_selected_c_without_rf_or_reselection(self):
        action = np.tile(np.arange(25), 20)
        state = np.c_[np.arange(len(action)), np.ones(len(action))].astype(float)
        done = np.tile(np.r_[np.zeros(24), 1], 20)
        groups = np.repeat(np.arange(10), 50)
        with patch.object(retrain, "fit_behavior") as rf:
            predictor, report = agent.fit_selected_behavior(
                state, action, done, groups, "logistic_C0.1_temperature")
            rf.assert_not_called()
        self.assertEqual(predictor.params["C"], .1)
        self.assertEqual(predictor.params["calibration"], "temperature")
        self.assertEqual(report["selected_behavior"], "logistic_C0.1_temperature")
        expected = util.grouped_partition(done, groups, random_seed=42)
        for name in expected:
            np.testing.assert_array_equal(report["_partition_indices"][name], expected[name])
        np.testing.assert_allclose(predictor.estimator[0].mean_, state[expected["fit"]].mean(0))
        p = predictor.predict_proba(state[:4])
        self.assertEqual(p.shape, (4, 25))
        np.testing.assert_allclose(p.sum(1), 1.)
        self.assertTrue((p > 0).all())

    def test_fixed_rf_refit_ignores_internal_candidate_choice(self):
        action = np.tile(np.arange(25), 10)
        state = np.c_[np.arange(len(action)), np.ones(len(action))].astype(float)
        done = np.tile(np.r_[np.zeros(24), 1], 10)
        groups = np.repeat(np.arange(10), 25)
        class Model:
            classes_ = np.arange(25)
            def predict_proba(self, x): return np.full((len(x), 25), 1/25)
        raw, calibrated = Model(), Model()
        partition = util.grouped_partition(done, groups, random_seed=42)
        with patch.object(retrain, "fit_behavior", return_value=(raw, calibrated,
                {"_partition_indices": partition, "selected_behavior": "raw"})):
            predictor, report = agent.fit_selected_behavior(state, action, done, groups, "rf_sigmoid")
        self.assertIs(predictor.estimator, calibrated)
        self.assertEqual(report["selected_behavior"], "rf_sigmoid")


if __name__ == "__main__":
    unittest.main()
