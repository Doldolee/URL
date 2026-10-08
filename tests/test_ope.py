"""Numerical and estimator-boundary tests for independent whole-trajectory OPE."""
import json
import unittest

import numpy as np

from metric import anchored_policy_probs, evaluate_ope, policy_probs


class WholeTrajectoryOPETests(unittest.TestCase):
    def evaluate(self, action, reward, done, pi, behavior, q=None, **kwargs):
        pi = np.asarray(pi, dtype=float)
        if pi.ndim == 1:
            pi = np.tile(pi, (len(action), 1))
        behavior = np.asarray(behavior, dtype=float)
        if behavior.ndim == 1:
            behavior = np.tile(behavior, (len(action), 1))
        if q is None:
            q = np.zeros_like(pi)
        return evaluate_ope(action, reward, done, pi, behavior, q,
                            **dict({"n_bootstrap": 0}, **kwargs))

    def test_exact_stochastic_onpolicy_bandit(self):
        # Both sampled outcomes have DR=V=1 under the exact critic; WIS mean=1.
        result = self.evaluate([0, 1], [0, 2], [1, 1], [.5, .5], [.5, .5],
                               q=[[0, 2], [0, 2]], n_bootstrap=20)
        for name in ["dr", "wis", "wdr"]:
            self.assertAlmostEqual(result[name], 1)
        self.assertEqual(result["bootstrap"]["intervals"]["dr"]["low"], 1.)
        self.assertEqual(result["bootstrap"]["intervals"]["dr"]["high"], 1.)

    def test_exact_offpolicy_bandit(self):
        result = self.evaluate([0, 1], [3, -1], [1, 1], [1, 0], [.5, .5],
                               q=[[3, -1], [3, -1]], n_bootstrap=100)
        for name in ["dr", "wis", "wdr"]:
            self.assertAlmostEqual(result[name], 3)
        interval = result["bootstrap"]["intervals"]["wis"]
        self.assertGreater(interval["undefined_or_overflow_resamples"], 0)
        self.assertAlmostEqual(interval["low"], 3)
        self.assertEqual(interval["status"], "interval_conditional_on_defined_resamples")

    def test_variable_length_absorbing_padding_and_batch_invariance(self):
        # Short episode r=1; long r=[0,1]. Padding keeps both denominator weights.
        kwargs = dict(q=[[12, -3], [-8, 5], [4, -9]], gamma=.9)
        for batch in [None, 1, 2, 64]:
            result = self.evaluate([0, 0, 0], [1, 0, 1], [1, 0, 1],
                                   [1, 0], [1, 0], batch_size=batch, **kwargs)
            self.assertEqual(result["episodes_used"], 2)
            self.assertEqual(result["episodes_excluded"], 0)
            for name in ["dr", "wis", "wdr"]:
                self.assertAlmostEqual(result[name], .95)
            self.assertEqual(result["weights"]["per_decision_ess_with_absorbing_padding"], [2., 2.])

    def test_short_surviving_trajectory_stays_in_wdr_denominator(self):
        # Only short trajectory survives target; the long one gets ratio zero at t=1.
        result = self.evaluate([0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0], [.5, .5], gamma=1)
        self.assertAlmostEqual(result["dr"], 1)
        self.assertAlmostEqual(result["wis"], 1)
        self.assertAlmostEqual(result["wdr"], .5)
        self.assertEqual(result["numerical_status"]["undefined_wdr_time_steps"], [])

    def test_logspace_overflow_without_artificial_dr_cap(self):
        # Each trajectory weight is 1e1000, although all probability inputs are finite.
        length = 10
        result = self.evaluate([0] * (2 * length), ([0] * 9 + [1]) * 2,
                               ([0] * 9 + [1]) * 2, [1e-200, 1.], [1e-300, 1.], gamma=1)
        self.assertIsNone(result["dr"])
        self.assertEqual(result["numerical_status"]["dr"], "float64_overflow")
        self.assertGreater(result["numerical_status"]["dr_log_abs"], 2000)
        self.assertAlmostEqual(result["wis"], 1)
        self.assertAlmostEqual(result["wdr"], 1)
        self.assertAlmostEqual(result["weights"]["trajectory_ess"], 2)
        json.dumps(result, allow_nan=False)

    def test_logspace_underflow_preserves_wis_and_wdr(self):
        # Terminal weights ~1e-500 must not become an all-zero normalization.
        length = 100
        result = self.evaluate([0] * (2 * length), ([0] * 99 + [1]) * 2,
                               ([0] * 99 + [1]) * 2, [5e-6, 1 - 5e-6], [.5, .5], gamma=1)
        self.assertEqual(result["dr"], 0)
        self.assertEqual(result["numerical_status"]["dr"], "underflow_to_zero")
        self.assertAlmostEqual(result["wis"], 1)
        self.assertAlmostEqual(result["wdr"], 1)
        self.assertEqual(result["weights"]["exact_zero_trajectory_weight_count"], 0)
        self.assertEqual(result["weights"]["trajectory_weights_below_float64_smallest_subnormal"], 2)

    def test_genuine_zero_target_weights_are_undefined(self):
        result = self.evaluate([1, 1], [1, -1], [1, 1], [1, 0], [.5, .5], n_bootstrap=10)
        self.assertEqual(result["dr"], 0)
        self.assertIsNone(result["wis"])
        self.assertIsNone(result["wdr"])
        self.assertEqual(result["bootstrap"]["intervals"]["wis"]["defined_finite_fraction"], 0)
        self.assertEqual(result["numerical_status"]["undefined_wdr_time_steps"], [0])
        json.dumps(result, allow_nan=False)

    def test_terminal_validation_and_masked_next_critic(self):
        with self.assertRaises(ValueError):
            self.evaluate([0], [1], [0], [1, 0], [1, 0])
        with self.assertRaises(ValueError):
            self.evaluate([0], [1], [2], [1, 0], [1, 0])
        result = self.evaluate([0], [1], [1], [1, 0], [1, 0],
                               next_q_values=[[1e200, 1e200]], next_target_probs=[[0, 0]])
        self.assertAlmostEqual(result["dr"], 1)
        self.assertEqual(result["next_critic_diagnostics"]["td_residual"]["mean"], 1)

    def test_full_behavior_support_and_no_epsilon(self):
        result = self.evaluate([0], [1], [1], [.5, .5], [1, 0])
        self.assertTrue(result["support"]["support_violation_at_observed_states"])
        self.assertEqual(result["support"]["target_mass_at_behavior_zero"]["mean"], .5)
        with self.assertRaises(ValueError):
            self.evaluate([1], [1], [1], [.5, .5], [1, 0])
        with self.assertRaises(ValueError):
            self.evaluate([0], [1], [1], [.5, .5], [.4, .4])

    def test_explicit_cap_is_sensitivity_and_does_not_cap_output(self):
        result = self.evaluate([0], [10], [1], [1, 0], [.1, .9], ratio_cap=2)
        self.assertAlmostEqual(result["dr"], 20)
        self.assertEqual(result["analysis_kind"], "clipped_sensitivity")
        self.assertEqual(result["support"]["explicitly_capped_step_count"], 1)

    def test_cluster_bootstrap_keeps_subject_episodes_together(self):
        # One subject owns all episodes: every cluster draw retains the whole sample.
        result = self.evaluate([0, 0], [-1, 1], [1, 1], [1, 0], [1, 0],
                               episode_groups=[7, 7], n_bootstrap=10)
        self.assertEqual(result["bootstrap"]["unit"], "whole subject cluster")
        self.assertEqual(result["bootstrap"]["independent_resampling_units"], 1)
        for name in ["dr", "wis", "wdr"]:
            self.assertEqual(result["bootstrap"]["intervals"][name]["low"], 0)
            self.assertEqual(result["bootstrap"]["intervals"][name]["high"], 0)
        with self.assertRaises(ValueError):
            self.evaluate([0, 0], [-1, 1], [1, 1], [1, 0], [1, 0], episode_groups=[7])

    def test_policy_freezing_and_softmax_temperature(self):
        q = np.array([[1000., 999.], [-1000., -999.]])
        np.testing.assert_array_equal(policy_probs(q), [[1, 0], [0, 1]])
        p = policy_probs(q, "softmax", temperature=.5)
        np.testing.assert_allclose(p.sum(axis=1), 1)
        np.testing.assert_allclose(p[0, 0], 1 / (1 + np.exp(-2)))
        np.testing.assert_allclose(policy_probs(q + 123, "softmax", .5), p)
        with self.assertRaises(ValueError):
            policy_probs(q, "softmax", 0)

    def test_optional_bootstrap_samples_do_not_change_results(self):
        kwargs = dict(n_bootstrap=20, seed=17)
        hidden = self.evaluate([0, 1], [3, -1], [1, 1], [1, 0], [.5, .5], **kwargs)
        exposed = self.evaluate([0, 1], [3, -1], [1, 1], [1, 0], [.5, .5],
                                return_bootstrap_samples=True, **kwargs)
        self.assertNotIn("samples", hidden["bootstrap"])
        samples = exposed["bootstrap"].pop("samples")
        self.assertEqual(hidden, exposed)
        for name in ["dr", "wis", "wdr"]:
            self.assertEqual(len(samples[name]), 20)
        self.assertIn(None, samples["wis"])
        json.dumps(samples, allow_nan=False)
        empty = self.evaluate([0], [1], [1], [1, 0], [1, 0],
                              n_bootstrap=0, return_bootstrap_samples=True)
        self.assertEqual(empty["bootstrap"]["samples"], {"dr": [], "wis": [], "wdr": []})
        json.dumps(empty, allow_nan=False)

    def test_anchored_policy_normalization_support_and_density_bound(self):
        q = np.array([[9., 1., 0.], [-7., 8., 0.], [4., 4., 0.]])
        behavior = np.array([[.1, .9, 0.], [1., 0., 0.], [.25, .25, .5]])
        original_q, original_behavior = q.copy(), behavior.copy()
        result = anchored_policy_probs(q, behavior, strength=1.)
        np.testing.assert_allclose(result.sum(axis=1), 1., rtol=0, atol=1e-15)
        np.testing.assert_array_equal(result[behavior == 0], 0.)
        np.testing.assert_array_equal(result[1], behavior[1])  # preferred action has no behavior support
        self.assertTrue(np.all(result[behavior > 0] / behavior[behavior > 0] <= 2.))
        np.testing.assert_allclose(result[0], [.2 / 1.1, .9 / 1.1, 0.])
        np.testing.assert_array_equal(anchored_policy_probs(q, behavior, 0), behavior)
        np.testing.assert_array_equal(q, original_q)
        np.testing.assert_array_equal(behavior, original_behavior)
        with self.assertRaises(ValueError):
            anchored_policy_probs(q, behavior, -1)
        with self.assertRaises(ValueError):
            anchored_policy_probs(q, behavior, np.inf)


if __name__ == "__main__":
    unittest.main()
