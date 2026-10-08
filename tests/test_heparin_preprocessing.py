import unittest

import numpy as np

from util import measured_aptt_reward, measured_bin_means, shift_rewards


class HeparinObservedRewardTests(unittest.TestCase):
    def test_therapeutic_window_and_missing(self):
        x = np.array([np.nan, 0, 60, 80, 100, 200, 1000], dtype=float)
        actual = measured_aptt_reward(x)
        # Independent sigmoid implementation at the therapeutic thresholds.
        expected = np.array([0, -1, 0, 2 / (1 + np.exp(-20)) - 2 / (1 + np.exp(20)) - 1, 0, -1, -1])
        np.testing.assert_allclose(actual, expected, atol=5e-16, rtol=0)
        with self.assertRaises(ValueError):
            measured_aptt_reward([np.inf])

    def test_inclusive_boundaries_missing_and_no_carry_forward(self):
        events = {(1, 0): 40, (1, 3600): 80, (1, 4000): 100, (2, 100): 70}
        means, counts = measured_bin_means(events, [1, 1, 1, 2], [0, 3600, 7200, 0])
        np.testing.assert_allclose(means, [60, 90, np.nan, 70], equal_nan=True)
        np.testing.assert_array_equal(counts, [2, 2, 0, 1])
        self.assertEqual(measured_aptt_reward(means)[2], 0)

    def test_timestamp_collisions_keep_last_value_not_duplicate_mean(self):
        events = {}
        for key, value in [((1, 100), 20), ((1, 100), 80), ((1, 200), 100)]:
            events[key] = value
        means, counts = measured_bin_means(events, [1], [0])
        np.testing.assert_array_equal(means, [90])
        np.testing.assert_array_equal(counts, [2])

    def test_reward_shift_skipped_hours_and_terminal(self):
        # Retained rows may be separated by multiple hours. The existing
        # transition is preserved and uses its next retained row's reward.
        np.testing.assert_array_equal(shift_rewards([0.4, 0, -1]), [[0], [-1], [0]])
        np.testing.assert_array_equal(shift_rewards([0.4]), [[0]])


if __name__ == "__main__":
    unittest.main()
