"""Small semantic tests; no clinical data or original policy checkpoints used."""

import unittest

import numpy as np
import torch

from model import FQECritic
from metric import _batch_targets, validate_transitions
from util import _trajectory_split
from metric import bootstrap_mean_ci
from metric import evaluate_initial_states, fit_fqe, policy_values


def toy_transitions(episodes=128):
    # s0 --zero reward--> s1 --one terminal +/-1 reward--> end.
    # At s1, action 0 earns +1 and action 1 earns -1. Fixed pi is [.75,.25],
    # hence V(s1)=.5 and Q(s0,action0)=gamma*.5, not softmax(Q)-weighted V.
    states = np.tile(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32), (episodes, 1))
    next_states = states.copy()
    next_states[::2] = states[1::2]
    actions = np.zeros(2 * episodes, dtype=np.int64)
    actions[1::2] = np.arange(episodes) % 2
    rewards = np.zeros(2 * episodes, dtype=np.float32)
    rewards[1::2] = 1 - 2 * actions[1::2]
    done = np.tile(np.array([0, 1], dtype=np.float32), episodes)
    pi_next = np.tile(np.array([0.75, 0.25], dtype=np.float32), (len(states), 1))
    return states, next_states, actions, rewards, done, pi_next


class FQESemanticsTests(unittest.TestCase):
    def test_terminal_mask_and_policy_are_fixed(self):
        class ConstantQ(torch.nn.Module):
            def forward(self, state):
                return torch.tensor([[1.0, -1.0]]).repeat(len(state), 1)

        pi = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        before = pi.clone()
        target = _batch_targets(
            ConstantQ(), torch.zeros(2, 1), torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 1.0]), pi, 0.9,
        )
        torch.testing.assert_close(target, torch.tensor([-0.9, 1.0]))
        torch.testing.assert_close(pi, before)

    def test_terminal_reward_and_continuity_validation(self):
        values = list(toy_transitions(4))
        validate_transitions(*values)
        values[3] = values[3].copy()
        values[3][0] = 1
        with self.assertRaisesRegex(ValueError, "zero nonterminal"):
            validate_transitions(*values)
        values = list(toy_transitions(4))
        values[4] = 1 - values[4]
        with self.assertRaisesRegex(ValueError, "final row"):
            validate_transitions(*values)
        values = list(toy_transitions(4))
        values[1] = values[1].copy()
        values[1][0] = [2, 2]
        with self.assertRaisesRegex(ValueError, "not aligned"):
            validate_transitions(*values)

    def test_patient_groups_do_not_cross_holdout(self):
        values = toy_transitions(16)
        # Two contiguous two-step episodes per patient, one positive/negative.
        groups = np.repeat(np.arange(8), 4)
        fit, val, val_episodes, count, val_groups = _trajectory_split(values[4], values[3], 0.25, 42, groups)
        self.assertFalse(set(groups[fit]) & set(groups[val]))
        self.assertEqual(count, 8)
        self.assertEqual(len(val_groups), 2)
        self.assertEqual(len(val_episodes), 4)
        malformed = groups.copy()
        malformed[0] = 999
        with self.assertRaisesRegex(ValueError, "exactly one split group"):
            _trajectory_split(values[4], values[3], 0.25, 42, malformed)

    def test_one_terminal_reward_propagates_under_frozen_policy(self):
        values = toy_transitions()
        fixed_before = values[-1].copy()
        model, report = fit_fqe(
            *values, gamma=0.9, hidden_dim=32, epochs=100, min_epochs=50,
            patience=20, batch_size=128, lr=0.01, min_improvement=1e-8,
            num_threads=2, seed=42,
        )
        np.testing.assert_array_equal(values[-1], fixed_before)
        with torch.no_grad():
            q = model(torch.tensor([[0.0, 1.0], [1.0, 0.0]])).numpy()
        self.assertTrue((np.abs(q) <= 1).all())
        np.testing.assert_allclose(q[1], [1.0, -1.0], atol=0.04)
        self.assertAlmostEqual(float(q[0, 0]), 0.45, delta=0.04)
        value = policy_values(model, np.array([[1.0, 0.0]]), [[0.75, 0.25]], num_threads=2)
        self.assertAlmostEqual(float(value[0]), 0.5, delta=0.04)
        initial_pi = np.tile([1.0, 0.0], (len(values[0]), 1))
        evaluation = evaluate_initial_states(model, values[0], values[4], initial_pi, num_threads=2)
        self.assertEqual(evaluation["episode_count"], 128)
        self.assertAlmostEqual(evaluation["mean"], 0.45, delta=0.04)
        ci = bootstrap_mean_ci(evaluation["initial_values"], n_bootstrap=100, seed=42)
        self.assertAlmostEqual(ci["ci_lower"], ci["ci_upper"], places=7)
        self.assertGreaterEqual(report["selected_epoch"], 50)
        self.assertEqual(report["data_validation"]["nonterminal_reward_count"], 0)

    def test_zero_initialized_bounded_auxiliary_critic(self):
        model = FQECritic(3, 2, 16)
        torch.testing.assert_close(model(torch.randn(20, 3)), torch.zeros(20, 2))


if __name__ == "__main__":
    unittest.main()
