#!/usr/bin/env python3
"""Independently audit the BC-denominator FrozenLake comparison.

No behavior fitting, target selection, or online collection occurs here. The
full OPE reference multiplies importance ratios directly; the separate Decimal
reference checks WDR and ESS on complete episodes. Matching true policy value
or bootstrap coverage is deliberately not a correctness assertion.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gym_test.verify_results import (
    direct_reference, episodes_from_data, forward_value, load_npz, same, sha256,
    write_new_json,
)


def independent_features(data, horizon):
    """Rebuild the declared 16 state indicators plus normalized remaining time."""
    states = np.asarray(data["state"], dtype=np.int64)
    assert np.all((states >= 0) & (states < 16))
    steps = np.asarray(data["time_step"], dtype=np.int64)
    assert np.all((steps >= 0) & (steps < horizon))
    values = np.zeros((len(states), 17), dtype=np.float32)
    values[np.arange(len(states)), states] = 1
    values[:, 16] = (horizon - steps) / horizon
    return values


def verify_partition(data, partition, fit_report, seed, horizon):
    """Check all rows, whole episodes, and independent 60/20/20 membership."""
    starts, ends, _ = episodes_from_data(data, horizon=horizon)
    groups = np.asarray(data["episode"], dtype=np.int64)
    unique_groups = np.unique(groups)
    permutation = np.random.RandomState(seed).permutation(unique_groups)
    n_cal = int(np.ceil(.2 * len(unique_groups)))
    n_val = int(np.ceil(.2 * len(unique_groups)))
    expected_groups = {
        "calibration": permutation[:n_cal],
        "validation": permutation[n_cal:n_cal + n_val],
        "fit": permutation[n_cal + n_val:],
    }
    assert set(partition) == set(expected_groups)
    membership = np.full(len(groups), -1, dtype=np.int64)
    observed_sets = {}
    for number, name in enumerate(("fit", "calibration", "validation")):
        rows = np.asarray(partition[name])
        assert rows.ndim == 1 and np.issubdtype(rows.dtype, np.integer)
        assert len(rows) > 0 and np.all((rows >= 0) & (rows < len(groups)))
        assert len(np.unique(rows)) == len(rows)
        assert np.all(membership[rows] == -1), "Rows overlap across BC partitions"
        membership[rows] = number
        expected_rows = np.flatnonzero(np.isin(groups, expected_groups[name]))
        np.testing.assert_array_equal(rows, expected_rows,
                                      err_msg=f"{name}: independent group permutation")
        observed_sets[name] = set(groups[rows].tolist())
        report = fit_report["split"][name]
        assert report["rows"] == len(rows)
        assert report["groups"] == len(observed_sets[name])
        assert report["episodes"] == int(np.sum(membership[starts] == number))
        np.testing.assert_array_equal(
            report["action_counts"], np.bincount(data["action"][rows], minlength=4))
    assert np.all(membership >= 0), "BC partition does not cover every actor row"
    for begin, end in zip(starts, ends):
        assert np.unique(membership[begin:end + 1]).size == 1
    for left, right in (("fit", "calibration"), ("fit", "validation"),
                        ("calibration", "validation")):
        assert not observed_sets[left].intersection(observed_sets[right])
    assert fit_report["selected_behavior"] == "calibrated"
    assert fit_report["validation_used_for_selection"] is False
    assert fit_report["final_refit"] is False
    assert fit_report["policy_support_mask"] is False
    assert fit_report["saved_probability_floor"] is None
    assert fit_report["num_actions"] == 4
    np.testing.assert_array_equal(fit_report["rf_classes"], np.arange(4))
    assert fit_report["absent_fit_actions"] == []
    for name, fraction in (("fit", .6), ("calibration", .2), ("validation", .2)):
        same(fit_report["split_fraction_by_group"][name], fraction,
             name=f"{name}: BC episode fraction")
    required_parameters = {"n_estimators": 200, "max_depth": 20,
                           "min_samples_leaf": 4, "max_features": "sqrt",
                           "bootstrap": True, "random_state": seed}
    for name, expected in required_parameters.items():
        assert fit_report["rf_parameters"][name] == expected, name
    return {
        "status": "passed", "episodes": len(starts), "transitions": len(groups),
        "partition_group_counts": {name: len(values)
                                   for name, values in observed_sets.items()},
        "checks": ["independent group permutation", "all actor rows covered once",
                   "disjoint episode groups", "complete episodes kept together",
                   "fixed calibrated selection", "clinical RF parameters",
                   "no refit, support mask, or saved probability floor"],
    }


def verify_predictor(predictor, data, saved, horizon, name):
    assert predictor.n_features_in_ == 17
    np.testing.assert_array_equal(predictor.classes_, np.arange(4))
    probabilities = np.asarray(predictor.predict_proba(independent_features(data, horizon)))
    assert probabilities.shape == (len(data["state"]), 4)
    assert np.isfinite(probabilities).all() and np.all(probabilities >= 0)
    same(probabilities.sum(axis=1), 1, name=f"{name}: row normalization", atol=1e-12)
    same(saved, probabilities, name=f"{name}: saved prediction row alignment", atol=1e-12)
    return probabilities


def verify_frozen_rf(raw, calibrated, fit_rows):
    """Calibration must retain every already-fitted RF tree without a refit."""
    assert calibrated.method == "sigmoid"
    assert calibrated.ensemble is False
    assert len(calibrated.calibrated_classifiers_) == 1
    frozen = calibrated.calibrated_classifiers_[0].estimator
    if hasattr(frozen, "estimator"):
        frozen = frozen.estimator
    assert len(raw.estimators_) == len(frozen.estimators_) == 200
    for index, (left, right) in enumerate(zip(raw.estimators_, frozen.estimators_)):
        assert left.random_state == right.random_state
        for field in ("children_left", "children_right", "feature", "threshold",
                      "value", "n_node_samples", "weighted_n_node_samples"):
            np.testing.assert_array_equal(getattr(left.tree_, field),
                                          getattr(right.tree_, field),
                                          err_msg=f"RF tree {index}: {field} changed")
        same(left.tree_.weighted_n_node_samples[0], len(fit_rows),
             name=f"RF tree {index}: fit-row bootstrap sample count")


def verify_validation_metrics(raw, calibrated, data, partition, report, horizon):
    rows = partition["validation"]
    values = independent_features(data, horizon)[rows]
    actions = data["action"][rows]
    for name, predictor in (("raw", raw), ("calibrated", calibrated)):
        probabilities = predictor.predict_proba(values)
        chosen = probabilities[np.arange(len(actions)), actions]
        log_loss = float(-np.log(np.maximum(chosen, np.finfo(float).eps)).mean())
        indicators = np.zeros_like(probabilities)
        indicators[np.arange(len(actions)), actions] = 1
        brier = float(np.sum((probabilities - indicators) ** 2, axis=1).mean())
        same(report["validation"][name]["log_loss"], log_loss,
             name=f"{name}: independent validation log loss")
        same(report["validation"][name]["brier_multiclass_sum"], brier,
             name=f"{name}: independent validation Brier score")


def verify_ope(data, frozen, behavior, result, gamma, evaluate_ope, reference_wdr,
               name, metric_row=None):
    starts, ends, _ = episodes_from_data(data, horizon=len(frozen["target_table"]) - 1)
    assert behavior.shape == frozen["target_probs"].shape
    assert np.isfinite(behavior).all() and np.all(behavior >= 0)
    same(behavior.sum(axis=1), 1, name=f"{name}: behavior normalization", atol=1e-12)
    arrays = {**frozen, "behavior_probs": behavior}
    direct = direct_reference(data, arrays, gamma, starts, ends)
    assert result["ratio_cap"] is None
    assert result["episodes_used"] == len(starts)
    assert result["episodes_excluded"] == 0
    same(result["gamma"], gamma, name=f"{name}: gamma")
    for estimator in ("dr", "wis", "wdr"):
        same(result[estimator], direct[estimator], name=f"{name}: {estimator}")
        if metric_row is not None:
            value = float(metric_row[estimator]) if metric_row[estimator] else None
            same(value, direct[estimator], name=f"{name}: CSV {estimator}")
    weights = result["weights"]
    same(weights["trajectory_ess"], direct["terminal_ess"], name=f"{name}: terminal ESS")
    same(weights["maximum_normalized_trajectory_weight"],
         direct["maximum_normalized_trajectory_weight"], name=f"{name}: maximum weight")
    expected_steps = weights["per_decision_ess_with_absorbing_padding"]
    assert len(expected_steps) == len(direct["per_decision_ess"])
    for step, (actual, expected) in enumerate(zip(expected_steps, direct["per_decision_ess"])):
        same(actual, expected, name=f"{name}: ESS at time {step}")
    assert weights["exact_zero_trajectory_weight_count"] == len(starts) - direct["nonzero_terminal_paths"]
    assert result["numerical_status"]["undefined_wdr_time_steps"] == direct["undefined_time_steps"]
    if metric_row is not None:
        for key, reference_key in (("terminal_ess", "terminal_ess"),
                                   ("max_normalized_weight", "maximum_normalized_trajectory_weight")):
            value = float(metric_row[key]) if metric_row[key] else None
            same(value, direct[reference_key], name=f"{name}: CSV {key}")
        for key, value in (("first_step_ess", direct["per_decision_ess"][0]),
                           ("last_step_ess", direct["per_decision_ess"][-1])):
            actual = float(metric_row[key]) if metric_row[key] else None
            same(actual, value, name=f"{name}: CSV {key}")
        for key in ("nonzero_terminal_paths", "target_compatible_successes"):
            assert int(metric_row[key]) == direct[key], f"{name}: CSV {key}"
    cutoff = int(ends[min(len(ends), 30) - 1]) + 1
    subset_args = [data["action"][:cutoff], data["reward"][:cutoff], data["done"][:cutoff],
                   frozen["target_probs"][:cutoff], behavior[:cutoff], frozen["q_values"][:cutoff]]
    subset = evaluate_ope(*subset_args, gamma=gamma, n_bootstrap=0)
    decimal = reference_wdr(*subset_args, gamma=gamma)
    same(subset["wdr"], decimal["value"], name=f"{name}: independent Decimal WDR")
    for column in decimal["columns"]:
        step = column["time_step"]
        same(subset["weights"]["per_decision_ess_with_absorbing_padding"][step],
             column["ess"], name=f"{name}: Decimal ESS at time {step}")
    return {"direct_product_reference": direct,
            "decimal_subset": {"episodes": min(len(ends), 30), "transitions": cutoff,
                               "wdr": decimal["value"],
                               "undefined_time_step": decimal["undefined_time_step"]},
            "checks": ["full-data direct DR/WIS/WDR", "full-data terminal ESS",
                       "absorbing-padded ESS at every time", "exact support-zero counts",
                       "independent Decimal WDR/ESS on complete episodes"]}


def same_csv_field(actual, expected, name):
    if actual == expected:
        return
    if actual == "" or expected == "":
        raise AssertionError(f"{name}: blank/value mismatch: {actual!r} != {expected!r}")
    try:
        numeric_actual, numeric_expected = float(actual), float(expected)
    except ValueError:
        raise AssertionError(f"{name}: {actual!r} != {expected!r}")
    same(numeric_actual, numeric_expected, name=name, rtol=2e-12, atol=2e-12)


def verify_trace_csv(path, reports, selected_seed=None):
    expected = {}
    for seed_name, seed_report in reports.items():
        seed = int(seed_name.removeprefix("seed_"))
        if selected_seed is not None and seed != selected_seed:
            continue
        for policy, policy_report in seed_report["policies"].items():
            for behavior_model, behavior_report in policy_report.items():
                values = behavior_report["direct_product_reference"]["per_decision_ess"]
                for time_step, ess in enumerate(values, start=1):
                    expected[(seed, behavior_model, policy, time_step)] = ess
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["seed", "behavior_model", "policy", "time_step", "ess"]
        observed = set()
        for row in reader:
            key = (int(row["seed"]), row["behavior_model"], row["policy"], int(row["time_step"]))
            assert key in expected and key not in observed, f"Unexpected/duplicate ESS trace: {key}"
            observed.add(key)
            ess = float(row["ess"]) if row["ess"] else None
            same(ess, expected[key], name=f"{path.name}: {key}")
    assert observed == set(expected), f"Missing ESS trace rows: {len(expected) - len(observed)}"
    return {"status": "passed", "rows": len(observed), "time_step_indexing": "one-based"}


def verify_seed(run_dir, baseline, seed, gamma, horizon, metrics, baseline_metrics,
                evaluate_ope, reference_wdr):
    seed_name = f"seed_{seed}"
    source_dir = baseline / seed_name
    behavior_dir = run_dir / seed_name / "behavior"
    train = load_npz(source_dir / "datasets" / "actor_train.npz")
    test = load_npz(source_dir / "datasets" / "ope_test.npz")
    episodes_from_data(train, horizon=horizon)
    starts, ends, _ = episodes_from_data(test, horizon=horizon)
    model = load_npz(source_dir / "model.npz")
    oracle = load_npz(source_dir / "diagnostic_true_kernel.npz")
    fit_report = json.loads((behavior_dir / "fit.json").read_text())
    assert fit_report["random_seed"] == seed
    partition = load_npz(behavior_dir / "partition.npz")
    partition_report = verify_partition(train, partition, fit_report, seed, horizon)
    probabilities = load_npz(behavior_dir / "test_probabilities.npz")
    assert set(probabilities) == {"raw", "calibrated", "known"}
    raw = joblib.load(behavior_dir / "raw.joblib")
    calibrated = joblib.load(behavior_dir / "calibrated.joblib")
    verify_predictor(raw, test, probabilities["raw"], horizon, f"seed {seed}: raw RF")
    verify_predictor(calibrated, test, probabilities["calibrated"], horizon,
                     f"seed {seed}: calibrated RF")
    assert np.all(probabilities["calibrated"] > 0)
    np.testing.assert_array_equal(probabilities["known"],
                                  np.full((len(test["state"]), 4), .25))
    verify_frozen_rf(raw, calibrated, partition["fit"])
    verify_validation_metrics(raw, calibrated, train, partition, fit_report, horizon)
    reports = {}
    names = ("behavior_control", "softmax_T1", "softmax_T0p1", "softmax_T0p02", "greedy")
    for name in names:
        frozen = load_npz(source_dir / "policies" / f"{name}.npz")
        remaining = horizon - test["time_step"]
        np.testing.assert_array_equal(frozen["target_probs"],
                                      frozen["target_table"][remaining, test["state"]])
        np.testing.assert_array_equal(frozen["q_values"],
                                      frozen["critic_q"][remaining, test["state"]])
        np.testing.assert_array_equal(frozen["behavior_probs"], probabilities["known"])
        exact_value = forward_value(oracle["reward"], oracle["continuation"],
                                    frozen["target_table"], gamma)
        fqe_value = forward_value(model["critic_reward"], model["critic_kernel"],
                                  frozen["target_table"], gamma)
        if name == "behavior_control":
            np.testing.assert_array_equal(frozen["target_probs"], probabilities["known"])
        reports[name] = {}
        for behavior_name, array_key in (("known_uniform", "known"),
                                          ("bc_calibrated", "calibrated"),
                                          ("bc_raw", "raw")):
            row = metrics[(seed, behavior_name, name)]
            same(float(row["exact_value"]), exact_value,
                 name=f"{behavior_name}/{name}: frozen exact value")
            same(float(row["fqe"]), fqe_value,
                 name=f"{behavior_name}/{name}: frozen FQE value")
            if behavior_name == "known_uniform":
                for key, value in baseline_metrics[(seed, name)].items():
                    same_csv_field(row[key], value,
                                   name=f"Known-uniform CSV changed: {seed}/{name}/{key}")
                result_path = source_dir / name / "ope.json"
            else:
                result_path = run_dir / seed_name / behavior_name / name / "ope.json"
            result = json.loads(result_path.read_text())
            reports[name][behavior_name] = verify_ope(
                test, frozen, probabilities[array_key], result, gamma,
                evaluate_ope, reference_wdr, f"seed {seed}/{behavior_name}/{name}", row)
            if name == "behavior_control" and behavior_name == "known_uniform":
                same(result["weights"]["trajectory_ess"], len(starts),
                     name=f"seed {seed}: known pi=b terminal ESS")
                same(result["wis"], reports[name][behavior_name]["direct_product_reference"]["observed_return_mean"],
                     name=f"seed {seed}: known pi=b WIS")
    return {
        "bc_fit_partition": partition_report,
        "joblib_predictions_and_validation_metrics": "passed",
        "calibration_preserved_all_raw_rf_trees": "passed",
        "frozen_target_values": "passed",
        "policies": reports,
        "control_interpretation": "Uniform target is fixed; only known-uniform denominator has pi=b exactly.",
        "artifact_sha256": {
            str(path.relative_to(run_dir)): sha256(path)
            for path in sorted((run_dir / seed_name).rglob("*"))
            if path.is_file() and not path.name.startswith("._")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    protocol_path = run_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    baseline = Path(protocol["baseline_dir"]).resolve()
    assert baseline != run_dir
    gamma, horizon = float(protocol["gamma"]), int(protocol["horizon"])
    seeds = [int(seed) for seed in protocol["seeds"]]
    assert len(seeds) == len(set(seeds)) and seeds
    for name, expected in protocol["baseline_artifact_sha256"].items():
        assert sha256(baseline / name) == expected, f"Frozen baseline artifact changed: {name}"
    for name, expected in protocol["core_source_hashes_before"].items():
        assert sha256(ROOT / name) == expected, f"Clinical source changed: {name}"
    with (run_dir / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    metrics = {(int(row["seed"]), row["behavior_model"], row["policy"]): row for row in rows}
    assert len(rows) == len(metrics) == len(seeds) * 3 * 5
    with (baseline / "metrics.csv").open(newline="") as handle:
        baseline_metrics = {(int(row["seed"]), row["policy"]): row
                            for row in csv.DictReader(handle)}
    from metric import evaluate_ope
    reference_path = ROOT / "tests" / "wdr_reference.py"
    spec = importlib.util.spec_from_file_location("independent_wdr_reference", reference_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    report = {
        "status": "passed", "gamma": gamma, "horizon": horizon,
        "reference_arithmetic": "independent direct numpy.longdouble products and Decimal WDR",
        "longdouble_mantissa_bits": int(np.finfo(np.longdouble).nmant),
        "baseline_dir": str(baseline),
        "frozen_baseline_artifact_count": len(protocol["baseline_artifact_sha256"]),
        "frozen_baseline_and_clinical_sources_unchanged": True,
        "checks_do_not_assert": ["low-ESS estimator accuracy", "bootstrap coverage",
                                 "model-fitting independence from arrays alone"],
        "source_sha256": {
            "gym_test/verify_bc_results.py": sha256(Path(__file__)),
            "gym_test/verify_results.py": sha256(ROOT / "gym_test" / "verify_results.py"),
            "gym_test/run_bc_experiment.py": sha256(ROOT / "gym_test" / "run_bc_experiment.py"),
            "metric.py": sha256(ROOT / "metric.py"),
            "tests/wdr_reference.py": sha256(reference_path),
            "protocol.json": sha256(protocol_path),
        },
        "seeds": {},
    }
    for seed in seeds:
        report["seeds"][f"seed_{seed}"] = verify_seed(
            run_dir, baseline, seed, gamma, horizon, metrics, baseline_metrics,
            evaluate_ope, reference.reference_wdr)
        print(f"seed_{seed}: BC, frozen policies, and independent OPE/ESS checks passed", flush=True)
    report["ess_trace_csv"] = {
        "all_seeds": verify_trace_csv(run_dir / "per_step_ess.csv", report["seeds"]),
        **{f"seed_{seed}": verify_trace_csv(run_dir / f"seed_{seed}" / "per_step_ess.csv",
                                           report["seeds"], selected_seed=seed)
           for seed in seeds},
    }
    output = run_dir / "verification.json"
    write_new_json(output, report)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
