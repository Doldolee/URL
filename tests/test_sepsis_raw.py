"""Exact physical-state reconstruction tests; no training or real-data writes."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from util import DEM_COLUMNS, OBS_COLUMNS
from util import recover_raw_states


class RawStateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.normalized_path = Path(self.directory.name) / "normalized.csv"
        self.raw_path = Path(self.directory.name) / "raw.csv"
        self.columns = sorted(OBS_COLUMNS) + sorted(DEM_COLUMNS)
        self.header = ["traj", "step", "m:charttime", "m:icustayid", "m:clinical_note"]
        self.header += self.columns + ["a:action", "r:reward"]
        self.normalized = []
        self.raw = []
        self.episodes = []
        self.physical = []
        for ep in range(3):
            normalized = np.asarray([[ep + step / 10 + feature / 1000
                                      for feature in range(43)] for step in range(3)], dtype=np.float32)
            # Physical values deliberately cannot be recovered by identity.
            physical = np.asarray(normalized, dtype=np.float64) * np.arange(1, 44) + 100
            terminal_reward = -1 if ep == 1 else 1
            for step in range(3):
                metadata = {"traj": ep, "step": step, "m:charttime": ep * 100000 + step * 14400,
                            "m:icustayid": 100 + ep, "m:clinical_note": "discard-me-" + "x" * 140000,
                            "a:action": ep + step, "r:reward": terminal_reward if step == 2 else 0}
                self.normalized.append(dict(metadata, **dict(zip(self.columns, normalized[step]))))
                self.raw.append(dict(metadata, **dict(zip(self.columns, physical[step]))))
            self.episodes.append({"state": normalized[:-1].astype(np.float64),
                                  "next_state": normalized[1:].astype(np.float64),
                                  "action": np.asarray([ep, ep + 1]).reshape(-1, 1),
                                  "reward": np.asarray([0, terminal_reward]).reshape(-1, 1),
                                  "done": np.asarray([0, 1]).reshape(-1, 1)})
            self.physical.append(physical)
        # Saved train order differs from CSV source order.
        self.train = {field: np.concatenate([self.episodes[2][field], self.episodes[0][field]])
                      for field in self.episodes[0]}
        self.test = copy.deepcopy(self.episodes[1])
        self.write_sources()

    def write_csv(self, path, rows, header=None):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.header if header is None else header)
            writer.writeheader()
            writer.writerows(rows)

    def write_sources(self):
        self.write_csv(self.normalized_path, self.normalized)
        self.write_csv(self.raw_path, self.raw)

    def recover(self):
        return recover_raw_states(self.train, self.test, self.normalized_path, self.raw_path)

    def test_exact_join_preserves_saved_order_terminal_next_state_and_inputs(self):
        before = {name: {field: values.copy() for field, values in data.items()}
                  for name, data in [("train", self.train), ("test", self.test)]}
        states, report = self.recover()
        expected_train_state = np.concatenate([self.physical[2][:-1], self.physical[0][:-1]])
        expected_train_next = np.concatenate([self.physical[2][1:], self.physical[0][1:]])
        np.testing.assert_array_equal(states["train"]["state"], expected_train_state)
        np.testing.assert_array_equal(states["train"]["next_state"], expected_train_next)
        np.testing.assert_array_equal(states["test"]["state"], self.physical[1][:-1])
        np.testing.assert_array_equal(states["test"]["next_state"][-1], self.physical[1][-1])
        self.assertFalse(np.array_equal(states["test"]["state"][-1], states["test"]["next_state"][-1]))
        for name, data in [("train", self.train), ("test", self.test)]:
            for field, values in data.items():
                np.testing.assert_array_equal(values, before[name][field])
        self.assertEqual(report["feature_columns"], self.columns)
        self.assertEqual(report["matched_train_episodes"], 2)
        self.assertEqual(report["matched_test_episodes"], 1)
        self.assertEqual(report["matched_unique_source_transition_gaps_hours"], {"4.0": 6})
        self.assertFalse(report["scaler_fitted"])
        self.assertFalse(report["imputation_leakage_removed"])
        self.assertTrue(report["global_normalization_and_clipping_bypassed"])
        self.assertNotIn("discard-me", json.dumps(report))

    def test_join_uses_step_keys_and_feature_names_not_raw_row_or_header_order(self):
        # Keep trajectories contiguous but reverse steps inside each one.
        raw = [row for start in range(0, 9, 3) for row in self.raw[start:start+3][::-1]]
        self.write_csv(self.raw_path, raw, self.header[::-1])
        states, _ = self.recover()
        np.testing.assert_array_equal(states["test"]["next_state"], self.physical[1][1:])

    def test_raw_metadata_action_and_terminal_reward_mismatches_fail(self):
        for field, value in [("m:charttime", 1), ("m:icustayid", 999),
                             ("a:action", 24), ("r:reward", -1)]:
            with self.subTest(field=field):
                rows = copy.deepcopy(self.raw)
                # Last source action is absent from transitions but still must
                # agree between the two source CSVs.
                for row in rows[:3] if field == "m:icustayid" else [rows[2]]:
                    row[field] = value if field != "m:charttime" else row[field] + value
                self.write_csv(self.raw_path, rows)
                with self.assertRaisesRegex(ValueError, "alignment disagrees"):
                    self.recover()

    def test_missing_normalized_or_raw_episode_fails(self):
        self.write_csv(self.normalized_path, self.normalized[3:])
        with self.assertRaisesRegex(ValueError, "missing complete normalized"):
            self.recover()
        self.write_sources()
        self.write_csv(self.raw_path, self.raw[3:])
        with self.assertRaisesRegex(ValueError, "missing complete RAW"):
            self.recover()

    def test_duplicate_normalized_fingerprint_is_ambiguous_after_clipping(self):
        duplicate = copy.deepcopy(self.normalized[:3])
        for row in duplicate:
            row["traj"] = 99
            row["m:icustayid"] = 999
        self.write_csv(self.normalized_path, self.normalized + duplicate)
        with self.assertRaisesRegex(ValueError, "ambiguous normalized trajectory"):
            self.recover()

    def test_invalid_raw_values_steps_and_saved_reward_convention_fail(self):
        rows = copy.deepcopy(self.raw)
        rows[0][self.columns[0]] = float("nan")
        self.write_csv(self.raw_path, rows)
        with self.assertRaisesRegex(ValueError, "non-finite source"):
            self.recover()
        rows = copy.deepcopy(self.raw)
        rows[1]["step"] = 0
        self.write_csv(self.raw_path, rows)
        with self.assertRaisesRegex(ValueError, "unique and consecutive"):
            self.recover()
        self.write_sources()
        self.train["done"][0] = 1
        with self.assertRaisesRegex(ValueError, "terminal reward indicator"):
            self.recover()

    def test_full_five_field_matching_rejects_changed_saved_transition(self):
        self.train["next_state"][0, 0] += .25
        with self.assertRaisesRegex(ValueError, "missing complete normalized"):
            self.recover()


if __name__ == "__main__":
    unittest.main()
