"""Independent checks for validation-only behavior ESS comparisons."""
import unittest

import numpy as np

from scripts.compare_behavior_policies import rank_candidates, trajectory_weight_diagnostics


class TrajectoryWeightDiagnosticsTests(unittest.TestCase):
    def test_known_terminal_ratios_and_unequal_lengths(self):
        # Episode 1 has ratios 2 * 3 = 6; episode 2 has ratio 2.
        actions = np.zeros(3, dtype=int)
        done = np.array([0, 1, 1])
        target = np.array([[.6, .4], [.6, .4], [.6, .4]])
        behavior = np.array([[.3, .7], [.2, .8], [.3, .7]])
        actual = trajectory_weight_diagnostics(actions, done, target, behavior)
        self.assertAlmostEqual(actual['trajectory_ess'], 1.6)
        self.assertAlmostEqual(actual['trajectory_ess_fraction'], .8)
        self.assertAlmostEqual(actual['maximum_normalized_trajectory_weight'], .75)

    def test_three_known_trajectory_weights(self):
        # Single-step ratios are 1, 2 and 3, giving ESS = 6^2 / 14.
        actions = np.zeros(3, dtype=int)
        done = np.ones(3, dtype=int)
        target = np.array([[.6, .4]] * 3)
        behavior = np.array([[.6, .4], [.3, .7], [.2, .8]])
        actual = trajectory_weight_diagnostics(actions, done, target, behavior)
        self.assertAlmostEqual(actual['trajectory_ess'], 36 / 14)
        self.assertAlmostEqual(actual['maximum_normalized_trajectory_weight'], .5)

    def test_equal_policies_have_one_unit_weight_per_episode(self):
        actions = np.array([0, 1, 0, 0, 1, 1])
        done = np.array([0, 1, 1, 0, 0, 1])
        target = np.array([[.3, .7], [.8, .2], [.4, .6],
                           [.7, .3], [.2, .8], [.5, .5]])
        actual = trajectory_weight_diagnostics(actions, done, target, target)
        self.assertAlmostEqual(actual['trajectory_ess'], 3.)
        self.assertAlmostEqual(actual['trajectory_ess_fraction'], 1.)

    def test_recorded_behavior_zero_is_explicitly_undefined(self):
        actual = trajectory_weight_diagnostics([0, 0], [1, 1],
            [[.5, .5], [.5, .5]], [[0., 1.], [.5, .5]])
        self.assertIsNone(actual['trajectory_ess'])
        self.assertEqual(actual['observed_behavior_zero_count'], 1)
        self.assertEqual(actual['status'], 'undefined_recorded_action_probability_zero')

    def test_all_target_paths_zero_are_undefined(self):
        actual = trajectory_weight_diagnostics([0, 0], [1, 1],
            [[0., 1.], [0., 1.]], [[.5, .5], [.5, .5]])
        self.assertIsNone(actual['trajectory_ess'])
        self.assertEqual(actual['status'], 'undefined_all_trajectory_weights_zero')

    def test_single_zero_target_path_keeps_remaining_path(self):
        actual = trajectory_weight_diagnostics([0, 0], [1, 1],
            [[0., 1.], [.5, .5]], [[.5, .5], [.5, .5]])
        self.assertAlmostEqual(actual['trajectory_ess'], 1.)
        self.assertAlmostEqual(actual['maximum_normalized_trajectory_weight'], 1.)

    def test_large_products_are_computed_in_log_space(self):
        # Products would overflow float64; equal log weights still imply ESS=2.
        n = 2000
        done = np.zeros(n, dtype=int)
        done[[999, 1999]] = 1
        target = np.tile([.5, .5], (n, 1))
        behavior = np.tile([.001, .999], (n, 1))
        actual = trajectory_weight_diagnostics(np.zeros(n, dtype=int), done, target, behavior)
        self.assertAlmostEqual(actual['trajectory_ess'], 2., places=10)


class CandidateSelectionTests(unittest.TestCase):
    @staticmethod
    def rows(candidate, ess):
        values = [ess] * 20 if np.isscalar(ess) else ess
        return [{'candidate': candidate, 'kind': 'matched_protocol',
                 'policy': f'policy_{i}', 'trajectory_ess': value}
                for i, value in enumerate(values)]

    def test_highest_mean_ess_is_selected(self):
        rows = self.rows('low', 3.) + self.rows('high', 5.)
        ranked = rank_candidates(rows)
        self.assertEqual(ranked[0]['candidate'], 'high')
        self.assertEqual(ranked[0]['mean_ess'], 5.)

    def test_undefined_candidate_is_ineligible_despite_high_defined_mean(self):
        rows = self.rows('partial', [100.] * 19 + [None]) + self.rows('complete', 2.)
        ranked = rank_candidates(rows)
        self.assertEqual(ranked[0]['candidate'], 'complete')
        self.assertFalse(ranked[1]['eligible'])
        self.assertEqual(ranked[1]['defined_policies'], 19)

    def test_validation_best_bcq_does_not_enter_shared_selection(self):
        rows = self.rows('low', 3.) + self.rows('high', 5.)
        rows.extend([{'candidate': 'low', 'kind': 'prior_validation_selected_best',
                      'policy': 'best_bcq', 'trajectory_ess': 10000.},
                     {'candidate': 'high', 'kind': 'prior_validation_selected_best',
                      'policy': 'best_bcq', 'trajectory_ess': 1.}])
        ranked = rank_candidates(rows)
        self.assertEqual(ranked[0]['candidate'], 'high')
        self.assertEqual(ranked[0]['policies'], 20)

    def test_exact_tie_keeps_declared_candidate_order(self):
        ranked = rank_candidates(self.rows('first', 3.) + self.rows('second', 3.))
        self.assertEqual(ranked[0]['candidate'], 'first')

    def test_missing_matched_policy_raises_instead_of_selective_average(self):
        with self.assertRaisesRegex(ValueError, '20 matched policies'):
            rank_candidates(self.rows('incomplete', 3.)[:-1])


if __name__ == '__main__':
    unittest.main()
