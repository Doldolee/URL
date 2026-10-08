"""WDR equation checks with hand arithmetic and an independent Decimal oracle."""
import json
import unittest
from types import SimpleNamespace

import numpy as np

from metric import eval_wdr_ci, evaluate_wdr
from wdr_reference import reference_wdr


class WDRPaperTests(unittest.TestCase):
    def evaluate(self, action, reward, done, target, behavior, q=None, **kwargs):
        target = np.asarray(target, dtype=float)
        behavior = np.asarray(behavior, dtype=float)
        if target.ndim == 1:
            target = np.tile(target, (len(action), 1))
        if behavior.ndim == 1:
            behavior = np.tile(behavior, (len(action), 1))
        if q is None:
            q = np.zeros_like(target)
        return evaluate_wdr(action, reward, done, target, behavior, q,
                            **dict({"n_bootstrap": 0}, **kwargs))

    def test_hand_calculation_uses_previous_time_weights(self):
        # t0: w=(.6,.4), previous=(.5,.5), contribution=.25.
        # t1: w=(.5,.5), previous=(.6,.4), contribution=.9*.022.
        q = [[.2, .8], [.1, .5], [.7, .3], [-.2, .4]]
        result = self.evaluate([0, 1, 1, 0], [0, 1, 0, -1], [0, 1, 0, 1],
                               [.6, .4], [.5, .5], q=q, gamma=.9)
        self.assertAlmostEqual(result["value"], .2698, places=13)

    def test_absorbing_episode_remains_in_later_denominator(self):
        # rho(t0)=(1.6,.4); rho(t1)=(1.6,.64), not just (.64).
        result = self.evaluate([0, 1, 0], [1, 0, -1], [1, 0, 1],
                               [.8, .2], [.5, .5], gamma=.9)
        self.assertAlmostEqual(result["value"], 19 / 35, places=13)

    def test_onpolicy_nonzero_critic_control_variate_does_not_equal_return(self):
        result = self.evaluate([0, 0], [1, -1], [1, 1], [.5, .5], [.5, .5],
                               q=[[.2, .8], [.2, .8]])
        self.assertAlmostEqual(result["value"], .3)
        self.assertAlmostEqual(result["diagnostics"]["dr"], .3)
        self.assertAlmostEqual(result["diagnostics"]["wis"], 0)

    def test_greedy_cap1_ess_counts_only_fully_matching_paths(self):
        result = self.evaluate([0, 0, 0, 1], [0, 1, 0, -1], [0, 1, 0, 1],
                               [1, 0], [.5, .5], ratio_cap=1)
        self.assertAlmostEqual(result["value"], .98)
        self.assertEqual(result["diagnostics"]["weights"]["trajectory_ess"], 1)
        self.assertEqual(result["analysis_kind"], "clipped_sensitivity")

    def test_zero_normalization_is_undefined_not_uniform(self):
        result = self.evaluate([1, 1], [1, -1], [1, 1], [1, 0], [.5, .5],
                               n_bootstrap=10, return_bootstrap_samples=True)
        self.assertIsNone(result["value"])
        self.assertEqual(result["undefined_time_steps"], [0])
        self.assertEqual(result["interval"]["defined_finite_resamples"], 0)
        json.dumps(result, allow_nan=False)

    def test_positive_tiny_ratios_do_not_underflow_normalization(self):
        length = 100
        result = self.evaluate([0] * (2 * length), ([0] * 99 + [1]) * 2,
                               ([0] * 99 + [1]) * 2,
                               [5e-6, 1 - 5e-6], [.5, .5], gamma=1)
        self.assertAlmostEqual(result["value"], 1)
        self.assertEqual(result["numerical_status"], "finite")

    def test_large_ratios_do_not_overflow_normalized_wdr(self):
        result = self.evaluate([0] * 40, ([0] * 19 + [1]) * 2,
                               ([0] * 19 + [1]) * 2,
                               [1, 0], [1e-20, 1 - 1e-20], gamma=1)
        self.assertAlmostEqual(result["value"], 1)
        self.assertEqual(result["numerical_status"], "finite")

    def test_ragged_random_cases_match_independent_decimal_equation(self):
        rng = np.random.RandomState(76)
        for gamma in [0., .4, .98, 1.]:
            for cap in [None, 1., 5.]:
                lengths = rng.randint(1, 8, size=8)
                n = int(lengths.sum())
                action = rng.randint(0, 3, size=n)
                reward = rng.uniform(-1, 1, size=n)
                done = np.zeros(n, dtype=int)
                done[np.cumsum(lengths) - 1] = 1
                target = rng.dirichlet([1., 2., 1.], size=n)
                behavior = rng.dirichlet([2., 1., 2.], size=n)
                q = rng.uniform(-1, 1, size=(n, 3))
                result = evaluate_wdr(action, reward, done, target, behavior, q,
                                      gamma=gamma, ratio_cap=cap, n_bootstrap=0)
                reference = reference_wdr(action, reward, done, target, behavior, q,
                                           gamma=gamma, ratio_cap=cap)
                self.assertAlmostEqual(result["value"], reference["value"], places=11)
                np.testing.assert_allclose(
                    result["diagnostics"]["weights"]["per_decision_ess_with_absorbing_padding"],
                    [row["ess"] for row in reference["columns"]], rtol=1e-12, atol=1e-12)

    def test_patient_bootstrap_preserves_whole_episodes(self):
        result = self.evaluate([0, 0, 0], [1, 0, -1], [1, 0, 1], [1, 0], [1, 0],
                               episode_groups=[7, 7], n_bootstrap=20)
        self.assertAlmostEqual(result["value"], .01)
        self.assertAlmostEqual(result["low"], .01)
        self.assertAlmostEqual(result["high"], .01)
        self.assertEqual(result["diagnostics"]["bootstrap"]["independent_resampling_units"], 1)

    def test_metric_buffer_interface_freezes_actor_and_uses_independent_critic(self):
        import torch

        class TableQ(torch.nn.Module):
            def __init__(self, q):
                super().__init__()
                self.state_dim = 1
                self.values = torch.nn.Parameter(torch.tensor(q, dtype=torch.float32))

            def forward(self, states):
                return self.values[states[:, 0].long()]

        policy = SimpleNamespace(Q=TableQ([[2., 1.], [2., 1.], [2., 1.]]))
        critic = TableQ([[0., 0.]] * 3)
        buffer = SimpleNamespace(state=np.array([[0.], [1.], [2.]]),
                                 next_state=np.array([[0.], [2.], [2.]]),
                                 action=np.array([0, 0, 0]), reward=np.array([1, 0, -1]),
                                 done=np.array([1, 0, 1]), crt_size=3)
        original = policy.Q.values.detach().clone()
        for batch_size in [1, 2, 99]:
            result = eval_wdr_ci("CQL", policy, buffer, critic=critic,
                                  behavior_probs=[[.5, .5]] * 3,
                                  batch_size=batch_size, n_bootstrap=0, return_details=True)
            # t0 weights=(.5,.5); t1=(1/3,2/3).
            self.assertAlmostEqual(result["value"], .5 - .98 * 2 / 3)
            self.assertEqual(result["episodes_used"], 2)
            self.assertEqual(result["episodes_excluded"], 0)
            torch.testing.assert_close(policy.Q.values, original)
        with self.assertRaisesRegex(ValueError, "independently fitted"):
            eval_wdr_ci("CQL", policy, buffer, critic=None,
                        behavior_probs=[[.5, .5]] * 3, n_bootstrap=0)


if __name__ == "__main__":
    unittest.main()
