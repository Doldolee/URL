#!/usr/bin/env python3
"""Independent full-data check using pre-SAH events and scalar reward math.

No import from the rebuilding implementation. Checks every generated reward,
next-row timing, split mapping and byte-preserved nonreward buffer file.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def artifact_path(output_root, relative, recorded_dataset, current_dataset):
    """Resolve a moved dataset without changing historical receipt hashes."""
    path = output_root / relative
    if path.exists():
        return path
    try:
        suffix = Path(relative).relative_to(recorded_dataset)
    except ValueError:
        return path
    return current_dataset / suffix


def reference_reward(x):
    if math.isnan(x):
        return 0.0
    def sigmoid(z):
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        e = math.exp(z)
        return e / (1 + e)
    return 2 * sigmoid(x - 60) - 2 * sigmoid(x - 100) - 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--run-name", default="heparin_mimic4_observed_aptt_20261007")
    p.add_argument("--dataset-name", default="heparin4_observed_aptt")
    a = p.parse_args()
    report = a.output_root / "outputs" / a.run_name
    final = report / "final_files"
    receipt = json.loads((report / "completion_receipt.json").read_text())
    events = defaultdict(list)
    checkpoint = np.load(a.source_root / "intermediate_files/reformat.npy")
    for stay, when, value in checkpoint[:, [1, 2, 54]]:
        if math.isfinite(value):
            events[int(stay)].append((when, value))
    with (final / "MIMICtable.csv").open() as f:
        table = [(int(float(x["icustayid"])), float(x["charttime"]),
                  float(x["reward_PTT"]) if x["reward_PTT"] else math.nan)
                 for x in csv.DictReader(f)]
    reference = []
    boundary_bins = 0
    for stay, start, observed in table:
        values = [v for t, v in events[stay] if start <= t <= start + 3600]
        expected = sum(values) / len(values) if values else math.nan
        if not (observed == expected or math.isnan(observed) and math.isnan(expected)):
            raise AssertionError("Raw-bin aPTT disagrees with independent direct event selection")
        boundary_bins += any(t == start or t == start + 3600 for t, v in events[stay])
        reference.append(reference_reward(expected))
    ref = np.asarray(reference)
    with (final / "heparin_final_data_withTimes.csv").open() as f:
        rows = [(int(x["traj"]), int(x["step"]), float(x["PTT_reward"]),
                 float(x["m:charttime"])) for x in csv.DictReader(f)]
    actual = np.asarray([x[2] for x in rows])
    error = float(np.max(np.abs(actual - ref)))
    if error > 1e-14:
        raise AssertionError("CSV reward formula disagrees with scalar sigmoid")
    verified = {}
    source_indices = {}
    for split in ["train", "val", "test"]:
        values = np.load(final / "tuples" / f"{split}_r.npy").reshape(-1)
        continuing = np.load(final / "tuples" / f"{split}_d.npy").reshape(-1)
        indexes = []
        with (report / f"{split}_row_mapping.csv").open() as f:
            for i, m in enumerate(csv.DictReader(f)):
                assert i == int(m["tuple_row"])
                row = int(m["source_csv_row"])
                terminal = bool(int(m["terminal"]))
                assert terminal == (continuing[i] == 0)
                assert rows[row][0] == int(m["traj"])
                nextrow = row if terminal else row + 1
                if not terminal:
                    assert rows[row][0] == rows[nextrow][0]
                    assert rows[nextrow][1] == rows[row][1] + 1
                    assert float(m["reward_bin_start"]) == rows[nextrow][3]
                expected = 0.0 if terminal else float(np.float32(ref[nextrow]))
                if values[i] != expected:
                    raise AssertionError("Transition reward does not equal independently calculated next-row reward")
                indexes.append(row)
        assert len(indexes) == len(values)
        source_indices[split] = indexes
        verified[split] = len(values)
    combined = source_indices["train"] + source_indices["val"] + source_indices["test"]
    assert len(combined) == len(rows) and len(set(combined)) == len(rows)
    dataset = a.output_root / "dataset" / a.dataset_name
    for split, parts in [("train", ["train", "val"]), ("test", ["test"])]:
        expected = np.concatenate([np.load(final / "tuples" / f"{k}_r.npy") for k in parts])
        np.testing.assert_array_equal(np.load(dataset / f"{split}_reward.npy"), expected)
    relocated = {}
    for relative, expected in receipt["artifact_sha256"].items():
        path = artifact_path(a.output_root, relative,
                             Path(receipt["dataset_relative_path"]), dataset)
        assert digest(path) == expected, relative
        if path != a.output_root / relative:
            relocated[relative] = str(path.relative_to(a.output_root))
    for source, expected in receipt["inputs_sha256"].items():
        assert digest(Path(source)) == expected, source
    result = {"status": "passed", "independent_raw_bin_rows": len(table),
              "bins_including_boundary_measurement": boundary_bins,
              "scalar_sigmoid_max_absolute_error": error,
              "all_source_buffer_rewards_verified": verified,
              "project_train_test_rewards_verified": True,
              "all_saved_artifact_hashes_verified": True,
              "dataset_artifacts_resolved_after_relocation": relocated,
              "all_read_input_hashes_preserved": True}
    dest = report / "independent_verification.json"
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
