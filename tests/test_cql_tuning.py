"""Selection safeguards for the paired-seed Heparin CQL search."""
import unittest

from scripts.tune_heparin_mimic3_cql import choose_setting


class CQLTuningTests(unittest.TestCase):
    def rows(self, lr, values, updates=1500):
        return [{'learning_rate': lr, 'cql_temperature': 1., 'updates': updates,
                 'seed': seed, 'validation_fqe': value, 'candidate': str(seed)}
                for seed, value in zip((42, 43, 44), values)]

    def test_paired_mean_rejects_a_lucky_seed_as_setting_winner(self):
        # One exceptional seed must not beat a setting with the higher mean.
        lucky = self.rows(1e-4, [9., -9., -9.])
        consistent = self.rows(1e-5, [1., 2., 3.])
        ranked, selected = choose_setting(lucky + consistent)
        self.assertEqual(ranked[0]['learning_rate'], 1e-5)
        self.assertEqual(selected['seed'], 44)
        self.assertEqual(selected['validation_fqe'], 3.)

    def test_incomplete_duplicate_or_nonfinite_seeds_cannot_be_ranked(self):
        full = self.rows(1e-4, [1., 2., 3.])
        for rows in [full[:-1], full + [full[0]], self.rows(1e-4, [1., float('nan'), 3.])]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                choose_setting(rows)

    def test_ties_prefer_fewer_updates_then_lower_seed(self):
        ranked, selected = choose_setting(self.rows(1e-4, [1., 1., 1.], 5000) +
                                           self.rows(1e-4, [1., 1., 1.], 1500))
        self.assertEqual(ranked[0]['updates'], 1500)
        self.assertEqual(selected['seed'], 42)


if __name__ == '__main__':
    unittest.main()
