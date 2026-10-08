"""Scientific invariants of the new actor, preprocessing, and patient split.

These tests use synthetic arrays only and do not fit a clinical policy.
"""
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from util import TrainScaler
from util import patient_split
from agent import constrained_probs, _torch_actor
from metric import policy_diagnostics


class ConstrainedActorTests(unittest.TestCase):
    def test_rare_mass_zero_support_and_density_bound(self):
        b = np.array([[.25, .25, .495, .005], [0., .4, .3, .3],
                      [.01, .01, .49, .49]])
        q = np.array([[1., 1., -1., 1.], [-1., 1., 0., -1.],
                      [1., -1., -.8, .8]])
        counts = np.array([100, 99, 100, 100])
        eligible = (b >= .01) & (counts[None, :] >= 100)
        for beta in [.05, .1, .25]:
            with self.subTest(beta=beta):
                p = constrained_probs(q, b, beta, counts)
                np.testing.assert_allclose(p.sum(1), 1., atol=1e-14, rtol=0)
                np.testing.assert_array_equal(p[~eligible], b[~eligible])
                np.testing.assert_array_equal(p[b == 0], 0.)
                ratio = np.divide(p, b, out=np.zeros_like(p), where=b > 0)
                self.assertLessEqual(ratio.max(), np.exp(2 * beta) + 1e-14)
                np.testing.assert_allclose((p * eligible).sum(1),
                                           (b * eligible).sum(1), atol=1e-14)

    def test_beta_zero_identity_and_inputs_unchanged(self):
        q = np.array([[.8, -.3, .2]])
        b = np.array([[.2, .3, .5]])
        counts = np.array([500, 800, 10])
        before = [x.copy() for x in [q, b, counts]]
        p0 = constrained_probs(q, b, 0., counts)
        np.testing.assert_array_equal(p0, b)
        self.assertFalse(np.shares_memory(p0, b))
        constrained_probs(q, b, .25, counts)
        for x, original in zip([q, b, counts], before):
            np.testing.assert_array_equal(x, original)

    def test_single_or_no_eligible_action_cannot_change_policy(self):
        b = np.array([[.2, .3, .5], [.2, .3, .5]])
        q = np.array([[1., 0., -1.], [-1., 1., .3]])
        for counts in [np.array([100, 0, 0]), np.zeros(3, dtype=int)]:
            p = constrained_probs(q, b, .25, counts)
            np.testing.assert_allclose(p, b, atol=1e-15, rtol=0)

    def test_offset_invariance_and_eligible_odds_tilt(self):
        b = np.array([[.2, .3, .49, .01]])
        q = np.array([[.5, -.5, .2, -.2]])
        counts = np.array([200, 200, 20, 200])
        beta = .25
        p = constrained_probs(q, b, beta, counts)
        shifted = constrained_probs(q + .2, b, beta, counts)
        np.testing.assert_allclose(p, shifted, atol=1e-15, rtol=0)
        actual_odds_change = (p[0, 0] / p[0, 1]) / (b[0, 0] / b[0, 1])
        self.assertAlmostEqual(actual_odds_change, np.exp(beta), places=14)
        self.assertEqual(p[0, 2], b[0, 2])

    def test_training_and_deployment_actor_agree(self):
        rng = np.random.default_rng(7)
        b = rng.dirichlet(np.ones(5), size=17)
        q = rng.uniform(-1., 1., size=b.shape)
        counts = np.array([100, 99, 1000, 1000, 0])
        for beta in [0., .05, .1, .25]:
            expected = constrained_probs(q, b, beta, counts)
            actual = _torch_actor(torch.from_numpy(q), torch.from_numpy(b), beta,
                                  torch.from_numpy(counts), .01, 100).numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
            actual32 = _torch_actor(torch.from_numpy(q.astype(np.float32)),
                torch.from_numpy(b.astype(np.float32)), beta,
                torch.from_numpy(counts), .01, 100).numpy()
            np.testing.assert_allclose(actual32, expected, atol=2e-7, rtol=0)

    def test_invalid_q_probabilities_and_beta_rejected(self):
        q, b, counts = np.array([[.3, -.2]]), np.array([[.4, .6]]), np.array([100, 100])
        for beta in [-.1, np.nan, np.inf]:
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                constrained_probs(q, b, beta, counts)
        with self.assertRaises(ValueError):
            constrained_probs(np.array([[1.01, 0.]]), b, .1, counts)
        with self.assertRaises(ValueError):
            constrained_probs(q, np.array([[.3, .3]]), .1, counts)
        with self.assertRaises(ValueError):
            constrained_probs(q, b, .1, np.array([100]))

    def test_diagnostics_separate_actor_distance_and_q_degeneracy(self):
        b = np.array([[.2, .3, .5], [.2, .3, .5]])
        q = np.array([[.995, .995, -1.], [.2, .1, -.2]])
        counts = np.array([100, 100, 0])
        p = constrained_probs(q, b, .1, counts)
        report = policy_diagnostics(q, p, b, counts)
        self.assertEqual(report['rare_action_probability_max_change'], 0.)
        self.assertEqual(report['target_positive_behavior_zero_count'], 0)
        self.assertEqual(report['q_near_tie_fraction'], .5)
        self.assertEqual(report['q_saturation_fraction'], .5)


class PreprocessingAndSplitTests(unittest.TestCase):
    def test_scaler_uses_fit_states_only_and_does_not_clip_holdout(self):
        train = np.array([[1., 7.], [3., 7.], [5., 7.]])
        holdout = np.array([[1000., -10.], [-1000., 50.]])
        before = train.copy()
        scaler = TrainScaler().fit(train)
        np.testing.assert_array_equal(scaler.mean, [3., 7.])
        self.assertEqual(scaler.scale[1], 1.)
        transformed = scaler.transform(holdout)
        self.assertEqual(transformed.dtype, np.float32)
        self.assertGreater(np.abs(transformed[:, 0]).max(), 100.)
        np.testing.assert_array_equal(scaler.mean, [3., 7.])
        np.testing.assert_array_equal(train, before)

    def test_split_keeps_repeat_patient_episodes_together_and_removes_test(self):
        episode_patients = np.array([10, 11, 10, 12, 13, 14, 15, 16, 17, 18, 19])
        groups = np.repeat(episode_patients, 2)
        done = np.tile([0, 1], len(episode_patients))
        reward = np.zeros(len(groups))
        reward[1::2] = [1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        train, val, excluded = patient_split(groups, np.array([11, 99]), reward,
                                             done, fraction=.25, seed=11)
        np.testing.assert_array_equal(excluded, np.flatnonzero(groups == 11))
        self.assertFalse(np.intersect1d(groups[train], groups[val]).size)
        np.testing.assert_array_equal(np.sort(np.r_[train, val, excluded]),
                                      np.arange(len(groups)))
        for patient in np.unique(groups):
            members = np.flatnonzero(groups == patient)
            self.assertEqual(sum(np.isin(members, subset).all()
                                 for subset in [train, val, excluded]), 1)
        train2, val2, excluded2 = patient_split(groups, np.array([11, 99]), reward,
                                                done, fraction=.25, seed=11)
        for first, second in zip([train, val, excluded], [train2, val2, excluded2]):
            np.testing.assert_array_equal(first, second)

    def test_corrupt_original_episode_not_sanitized_by_test_exclusion(self):
        # Removing the middle test patient would make the remaining bad episode
        # appear consistent; validation must inspect ORIGINAL episode groups.
        groups = np.array([1, 2, 1, 3, 3, 4, 4, 5, 5])
        done = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1])
        reward = np.array([0., 0., 1., 0., -1., 0., 1., 0., -1.])
        with self.assertRaises(ValueError):
            patient_split(groups, np.array([2]), reward, done, seed=13)


class RunnerFailureBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'scripts/retrain_sepsis_policy.py'
        spec = importlib.util.spec_from_file_location('policy_retrain_runner_test', path)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_zero_normalization_diagnostics_reject_without_crashing(self):
        undefined = dict(dr=None, wis=None, wdr=None, weights=dict(
            trajectory_ess_fraction=None, maximum_normalized_trajectory_weight=None))
        protocol = dict(selection=dict(primary_ess_fraction=.1, primary_max_weight=.05,
            alternate_ess_fraction=.05, alternate_max_weight=.1,
            maximum_propensity_DR_and_WIS_span=.1))
        report = self.runner.gates(undefined, undefined, protocol)
        self.assertFalse(report['passed'])
        self.assertFalse(any(report['checks'].values()))

    def test_undefined_propensity_does_not_erase_available_fqe(self):
        d = dict(state=np.zeros((4, 2)), next_state=np.zeros((4, 2)),
            action=np.array([0, 0, 1, 1]), reward=np.array([0., 1., 0., -1.]),
            done=np.array([0, 1, 0, 1]))
        groups = np.array([1, 1, 2, 2])
        target = np.full((4, 2), .5)
        behavior = target.copy()
        behavior[0] = [0., 1.]
        with patch.object(self.runner, 'data', return_value=(d, groups)), \
             patch.object(self.runner, 'actor', return_value=(target, np.zeros((4, 2)))), \
             patch.object(self.runner, 'predict_q', return_value=np.zeros((4, 2))), \
             patch.object(self.runner, 'bprobs', return_value=behavior):
            result, initial = self.runner.evaluate(dict(name='synthetic'), 'validation',
                                                   object(), bootstrap=0)
        self.assertEqual(result['status'], 'undefined_ope')
        self.assertIn('recorded action is zero', result['reason'])
        self.assertIsNone(result['dr'])
        self.assertIsNone(result['wis'])
        self.assertIsNone(result['wdr'])
        self.assertEqual(result['fqe']['value'], 0.)
        np.testing.assert_array_equal(initial, [0., 0.])


if __name__ == '__main__':
    unittest.main()
