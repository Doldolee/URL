#!/usr/bin/env python3
"""Create a new, observed-aPTT version of historical Heparin MIMIC-IV data.

Requires NumPy only. Does not rerun imputation, randomize splits, train policies,
or overwrite an existing output. Historical files remain read-only.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util import STATE_FEATURES, measured_aptt_reward, measured_bin_means, shift_rewards, transition_fingerprint as trajectory_fingerprint


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_columns(path, names):
    data = [[] for _ in names]
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        indexes = [header.index(k) for k in names]
        for row in reader:
            for values, j in zip(data, indexes):
                values.append(float(row[j]) if row[j] else np.nan)
    return [np.asarray(x, dtype=np.float64) for x in data]


def rewrite_column(source, destination, name, values):
    """Preserve every other original CSV field exactly as a string."""
    with source.open(newline="") as fi, destination.open("w", newline="") as fo:
        reader = csv.reader(fi)
        writer = csv.writer(fo, lineterminator="\n")
        header = next(reader)
        j = header.index(name)
        writer.writerow(header)
        count = 0
        for count, row in enumerate(reader, 1):
            row[j] = "" if np.isnan(values[count - 1]) else repr(float(values[count - 1]))
            writer.writerow(row)
    if count != len(values):
        raise ValueError("CSV row count changed")
    # Re-read both outputs to catch ordering, formatting or copy errors.
    with source.open(newline="") as fi, destination.open(newline="") as fo:
        a, b = csv.reader(fi), csv.reader(fo)
        if next(a) != next(b):
            raise ValueError("CSV header changed")
        for i, (old, new) in enumerate(zip(a, b)):
            if old[:j] + old[j + 1:] != new[:j] + new[j + 1:]:
                raise ValueError("Nonreward CSV field changed")
            expected = values[i]
            actual = float(new[j]) if new[j] else np.nan
            if not (actual == expected or np.isnan(actual) and np.isnan(expected)):
                raise ValueError("Incorrect CSV reward")


def original_lab_events(source, onset):
    # Match the notebook's max(reference row)+1 lookup, including duplicate IDs.
    mapping = {}
    with (source / "ReferenceFiles/Reflabs.tsv").open() as f:
        for i, row in enumerate(csv.reader(f, delimiter="\t")):
            for value in row:
                if value and value.lower() != "nan":
                    mapping[int(float(value))] = i + 1
    with (source / "ReferenceFiles/sample_and_hold.csv").open() as f:
        names = [x.strip("'") for x in next(csv.reader(f))]
    lab_offset = 28  # notebook stores labs after 28 vital feature columns
    mapped_ptt = names.index("PTT") - lab_offset + 1
    ids = {key for key, column in mapping.items() if column == mapped_ptt}
    events = {}
    stats = {"ptt_itemids": sorted(ids), "ptt_mapped_column": mapped_ptt,
             "files": {}, "timestamp_overwrites": 0}
    # Exactly the notebook concatenation order: CE first, then LE.
    for name in ["labs_ce.csv", "labs_le.csv"]:
        print("Reading original lab records:", name, flush=True)
        counts = {"rows_scanned": 0, "ptt_rows_selected": 0}
        with (source / "processed_files" / name).open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            si, ti, ii, vi = [header.index(k) for k in ["icustayid", "charttime", "itemid", "valuenum"]]
            for row in reader:
                counts["rows_scanned"] += 1
                if int(float(row[ii])) not in ids:
                    continue
                stay = int(float(row[si]))
                if stay not in onset:
                    continue
                when = float(row[ti])
                if not onset[stay] <= when <= onset[stay] + 48 * 3600:
                    continue
                key = (stay, when)
                stats["timestamp_overwrites"] += key in events
                events[key] = float(row[vi]) if row[vi] else np.nan
                counts["ptt_rows_selected"] += 1
        stats["files"][name] = counts
    stats["unique_timestamps"] = len(events)
    return events, stats


def stats(values):
    x = np.asarray(values).reshape(-1)
    return {"rows": len(x), "zero_rewards": int(np.sum(x == 0)),
            "positive_rewards": int(np.sum(x > 0)), "negative_rewards": int(np.sum(x < 0)),
            "min": float(np.min(x)), "max": float(np.max(x)), "mean": float(np.mean(x))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="preprocessing/mimic4")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="staging root or project root")
    parser.add_argument("--dataset-name", default="heparin4_observed_aptt")
    parser.add_argument("--run-name", default="heparin_mimic4_observed_aptt_20261007")
    args = parser.parse_args()
    src, project = args.source_root.resolve(), args.project_root.resolve()
    output = args.output_root.resolve() / "outputs" / args.run_name
    dataset = args.output_root.resolve() / "dataset" / args.dataset_name
    if output.exists() or dataset.exists():
        raise FileExistsError("Use new output paths; existing results are never overwritten")
    final = src / "final_files"
    original_dataset = project / "dataset/heparin4"
    iii = src.parent / "mimic3/final_result/1hourly_discrete/no_reward_SAH"
    paths = [final / name for name in ["MIMICtable.csv", "heparin_final_data_withTimes.csv", "heparin_final_data_RAW_withTimes.csv"]]
    paths += sorted((final / "tuples").glob("*.npy")) + sorted(original_dataset.glob("*.npy"))
    paths += [src / "processed_files" / k for k in ["labs_ce.csv", "labs_le.csv"]]
    paths += [src / "ReferenceFiles" / k for k in ["Reflabs.tsv", "sample_and_hold.csv"]]
    paths += [src / "intermediate_files/reformat.npy", src / "preprocess.ipynb",
              iii / "MIMICtable.csv", iii / "heparin_final_data_withTimes.csv",
              Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "util.py"]
    print("Hashing read-only inputs", flush=True)
    inputs = {str(p): sha(p) for p in paths}
    output.mkdir(parents=True)
    dataset.mkdir(parents=True)
    new_final = output / "final_files"
    new_tuples = new_final / "tuples"
    new_tuples.mkdir(parents=True)

    stays, times, onsets, old_raw, imputed, blocs = numeric_columns(final / "MIMICtable.csv",
        ["icustayid", "charttime", "presumed_onset", "reward_PTT", "PTT", "bloc"])
    onset = {}
    for stay, when in zip(stays, onsets):
        if int(stay) in onset and onset[int(stay)] != when:
            raise ValueError("Multiple onset windows in one ICU stay")
        onset[int(stay)] = when
    events, extraction = original_lab_events(src, onset)
    measured, counts = measured_bin_means(events, stays, times)
    reward = measured_aptt_reward(measured)
    print("Original-measurement bins:", int(np.sum(counts > 0)), "missing:", int(np.sum(counts == 0)), flush=True)
    # Independently saved pre-SAH timestamp table must reproduce every bin.
    checkpoint = np.load(src / "intermediate_files/reformat.npy", mmap_mode="r")
    column = 30 + extraction["ptt_mapped_column"]
    raw_events = {(int(s), float(t)): float(v) for s, t, v in checkpoint[:, [1, 2, column]] if int(s) in onset}
    check_means, check_counts = measured_bin_means(raw_events, stays, times)
    np.testing.assert_array_equal(counts, check_counts)
    np.testing.assert_allclose(measured, check_means, atol=0, rtol=0, equal_nan=True)
    extraction["pre_sah_checkpoint_bins_identical"] = True

    # Verify the III artifact, rather than assuming the notebook's current branch.
    iii_raw = numeric_columns(iii / "MIMICtable.csv", ["reward_PTT"])[0]
    iii_reward = numeric_columns(iii / "heparin_final_data_withTimes.csv", ["PTT_reward"])[0]
    iii_error = float(np.max(np.abs(measured_aptt_reward(iii_raw) - iii_reward)))
    if iii_error > 1e-14 or np.any(iii_reward[np.isnan(iii_raw)] != 0):
        raise ValueError("III reward convention did not match")

    names = ["traj", "step", "m:icustayid", "m:charttime", "a:action", "PTT_reward"] + list(STATE_FEATURES)
    columns = numeric_columns(final / "heparin_final_data_withTimes.csv", names)
    traj, step, csv_stays, csv_times, actions, old_reward = columns[:6]
    states = np.column_stack(columns[6:]).astype(np.float32)
    np.testing.assert_array_equal(stays, csv_stays)
    np.testing.assert_array_equal(times, csv_times)
    np.testing.assert_allclose(old_reward, measured_aptt_reward(imputed), atol=1e-14, rtol=0)
    rewrite_column(final / "MIMICtable.csv", new_final / "MIMICtable.csv", "reward_PTT", measured)
    for filename in ["heparin_final_data_withTimes.csv", "heparin_final_data_RAW_withTimes.csv"]:
        raw_stays, raw_times = numeric_columns(final / filename, ["m:icustayid", "m:charttime"])
        np.testing.assert_array_equal(stays, raw_stays)
        np.testing.assert_array_equal(times, raw_times)
        rewrite_column(final / filename, new_final / filename, "PTT_reward", reward)
    begin = np.flatnonzero(np.r_[True, traj[1:] != traj[:-1]])
    end = np.r_[begin[1:], len(traj)]
    lookup = defaultdict(list)
    for a, b in zip(begin, end):
        if not np.array_equal(step[a:b], np.arange(b - a)) or len(set(stays[a:b])) != 1:
            raise ValueError("Malformed CSV trajectory")
        lookup[trajectory_fingerprint(states[a:b], actions[a:b])].append((a, b))
    if any(len(v) != 1 for v in lookup.values()):
        raise ValueError("Ambiguous full-trajectory match")

    mapped = {}
    used = set()
    split_stats = {}
    for split in ["train", "val", "test"]:
        arrays = {k: np.load(final / "tuples" / f"{split}_{k}.npy") for k in ["s", "ns", "a", "r", "d"]}
        d = arrays["d"].reshape(-1)
        if not np.isin(d, [0, 1]).all() or d[-1] != 0:
            raise ValueError("Source tuple requires d=0 at trajectory end")
        finishes = np.flatnonzero(d == 0) + 1
        starts = np.r_[0, finishes[:-1]]
        row_index = np.full(len(d), -1, dtype=np.int64)
        new_reward = np.empty_like(arrays["r"])
        for lo, hi in zip(starts, finishes):
            key = trajectory_fingerprint(arrays["s"][lo:hi], arrays["a"][lo:hi])
            if len(lookup.get(key, [])) != 1:
                raise ValueError("Source tuple not uniquely matched to original CSV")
            a, b = lookup[key][0]
            if a in used:
                raise ValueError("Trajectory appears in multiple splits")
            used.add(a)
            np.testing.assert_array_equal(arrays["s"][lo:hi], states[a:b])
            np.testing.assert_array_equal(arrays["ns"][lo:hi], np.vstack([states[a + 1:b], states[b - 1:b]]))
            np.testing.assert_array_equal(arrays["a"][lo:hi].reshape(-1), actions[a:b])
            np.testing.assert_array_equal(arrays["r"][lo:hi], shift_rewards(old_reward[a:b]))
            new_reward[lo:hi] = shift_rewards(reward[a:b])
            row_index[lo:hi] = np.arange(a, b)
        if np.any(row_index < 0):
            raise ValueError("Unmapped tuple rows")
        for suffix in ["s", "ns", "a", "d"]:
            source_path = final / "tuples" / f"{split}_{suffix}.npy"
            target_path = new_tuples / source_path.name
            shutil.copy2(source_path, target_path)
            if sha(source_path) != sha(target_path):
                raise ValueError("Tuple copy changed")
        np.save(new_tuples / f"{split}_r.npy", new_reward)
        mapped[split] = (row_index, new_reward)
        split_stats[split] = {"trajectories": len(starts), "rewards": stats(new_reward),
            "changed_rewards": int(np.sum(new_reward != arrays["r"]))}
        with (output / f"{split}_row_mapping.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tuple_row", "source_csv_row", "traj", "icustayid", "state_bin_start", "next_state_bin_start", "reward_bin_start", "reward_measurement_count", "reward_aptt", "terminal", "old_reward", "new_reward"])
            for i, row in enumerate(row_index):
                terminal = d[i] == 0
                nr = row if terminal else row + 1
                w.writerow([i, row, int(traj[row]), int(stays[row]), times[row], times[nr],
                    "" if terminal else times[nr], 0 if terminal else counts[nr],
                    "" if terminal or np.isnan(measured[nr]) else measured[nr], int(terminal),
                    arrays["r"][i, 0], new_reward[i, 0]])
    if len(used) != len(begin):
        raise ValueError("Incomplete trajectory coverage")

    # Project train = original train + val. Project done is the inverse of the
    # source tuple's continuing mask. Preserve both file conventions exactly.
    replay_stats = {}
    for split in ["train", "test"]:
        source_splits = ["train", "val"] if split == "train" else ["test"]
        new_reward = np.concatenate([mapped[k][1] for k in source_splits])
        for name, suffix in [("state", "s"), ("next_state", "ns"), ("action", "a"), ("reward", "r"), ("done", "d")]:
            prior = np.load(original_dataset / f"{split}_{name}.npy")
            expected = np.concatenate([np.load(final / "tuples" / f"{k}_{suffix}.npy") for k in source_splits])
            if name == "done":
                expected = 1 - expected
            np.testing.assert_array_equal(prior, expected)
        for name in ["state", "next_state", "action", "done", "BC_prob"]:
            old_path = original_dataset / f"{split}_{name}.npy"
            new_path = dataset / old_path.name
            shutil.copy2(old_path, new_path)
            if sha(new_path) != inputs[str(old_path)]:
                raise ValueError("Replay buffer copy changed")
        np.save(dataset / f"{split}_reward.npy", new_reward)
        np.testing.assert_array_equal(np.load(dataset / f"{split}_reward.npy"), new_reward)
        old = np.load(original_dataset / f"{split}_reward.npy")
        replay_stats[split] = {"rewards": stats(new_reward), "changed_rewards": int(np.sum(new_reward != old)),
                              "trajectories": sum(split_stats[k]["trajectories"] for k in source_splits)}
    print("Checking input preservation", flush=True)
    if any(sha(p) != digest for p, digest in inputs.items()):
        raise ValueError("Read-only input changed during generation")
    receipt = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "logical_project_root": str(project), "execution_output_root": str(args.output_root.resolve()),
        "dataset_relative_path": f"dataset/{args.dataset_name}", "report_relative_path": f"outputs/{args.run_name}",
        "inputs_sha256": inputs, "all_read_inputs_preserved": True,
        "rule": {"reward_source": "original observed aPTT before SAH/interpolation/KNN", "missing_measurement_reward": 0,
            "bin_seconds": 3600, "bin_endpoints": "inclusive, preserved historical convention",
            "bin_aggregation": "mean of measured timestamp-level values, duplicate timestamp keeps last source value",
            "reward_formula": "2 sigmoid(aPTT-60) - 2 sigmoid(aPTT-100) - 1",
            "transition_reward": "next retained CSV row's reward; terminal row reward=0",
            "done_convention": {"source_tuples": "1=continuing, 0=terminal", "project_dataset": "1=terminal, 0=continuing"}},
        "cohort": {"rows": len(times), "trajectories": len(begin), "observed_aptt_bins": int(np.sum(counts > 0)),
            "missing_aptt_bins": int(np.sum(counts == 0)), "old_saved_reward_aptt_missing": int(np.sum(np.isnan(old_raw))),
            "changed_csv_rewards": int(np.sum(np.abs(reward - old_reward) > 1e-12)), "new_csv_rewards": stats(reward)},
        "extraction": extraction, "iii_formula_max_absolute_error": iii_error,
        "source_tuple_splits": split_stats, "project_dataset": replay_stats,
        "state_features": list(STATE_FEATURES),
        "verification": {"all_nonreward_csv_fields_identical": True, "all_nonreward_npy_files_identical": True,
            "all_trajectories_uniquely_mapped": True, "historical_splits_preserved": True,
            "missing_aptt_rewards_all_zero": bool(np.all(reward[counts == 0] == 0))},
        "scope": "reward-only data revision; no policy training/evaluation, imputation, bin grid or split change"}
    artifacts = [p for folder in [output, dataset] for p in folder.rglob("*") if p.is_file()]
    receipt["artifact_sha256"] = {str(p.relative_to(args.output_root.resolve())): sha(p) for p in artifacts}
    (output / "completion_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ["cohort", "project_dataset", "verification"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
