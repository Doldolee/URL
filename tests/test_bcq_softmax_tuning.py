"""Selection, configuration and policy-definition safeguards for BCQ tuning."""
import unittest

import numpy as np
import torch

from agent import frozen_policy_arrays
from metric import policy_probs
from scripts.tune_heparin_mimic3_bcq_softmax import choose_setting, config


class BCQSoftmaxTuningTests(unittest.TestCase):
    def rows(self, lr, values, updates=1500):
        return [{'learning_rate': lr, 'bcq_threshold': .3, 'updates': updates,
                 'seed': seed, 'validation_fqe': value, 'candidate': str(seed)}
                for seed, value in zip((42, 43, 44), values)]

    def test_setting_selection_uses_paired_mean_and_complete_seeds(self):
        lucky = self.rows(1e-4, [9., -9., -9.])
        consistent = self.rows(1e-5, [1., 2., 3.])
        ranked, best = choose_setting(lucky + consistent)
        self.assertEqual(ranked[0]['learning_rate'], 1e-5)
        self.assertEqual(best['seed'], 44)
        for rows in [consistent[:-1], consistent + [consistent[0]],
                     self.rows(1e-5, [1., float('nan'), 3.])]:
            with self.assertRaises(ValueError):
                choose_setting(rows)

    def test_mask_threshold_cannot_remove_every_action(self):
        for threshold in [-.1, 1., float('nan')]:
            with self.assertRaises(ValueError):
                config(1e-5, threshold)
        for lr in [0., -1., float('inf')]:
            with self.assertRaises(ValueError):
                config(lr, .3)
        self.assertEqual(config(1e-5, .7)['bcq_threshold'], .7)

    def test_primary_softmax_is_unmasked_while_greedy_respects_imitation(self):
        class Table(torch.nn.Module):
            def forward(self, state):
                q = torch.tensor([[4., 1.]]).repeat(len(state), 1)
                imitation = torch.tensor([[.01, .99]]).log().repeat(len(state), 1)
                return q, imitation, torch.zeros_like(q)
        class Policy:
            Q = Table()
            threshold = .3
        states = np.zeros((2, 1), dtype=np.float32)
        policy = Policy()
        softmax = frozen_policy_arrays('BCQ', policy, states, mode='softmax', device='cpu')
        greedy = frozen_policy_arrays('BCQ', policy, states, mode='greedy', device='cpu')
        np.testing.assert_array_equal(softmax, policy_probs(np.tile([4., 1.], (2, 1)), mode='softmax'))
        np.testing.assert_array_equal(greedy.argmax(1), [1, 1])
        self.assertGreater(softmax[0, 0], softmax[0, 1])


if __name__ == '__main__':
    unittest.main()
