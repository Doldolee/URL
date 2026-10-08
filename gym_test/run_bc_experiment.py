"""Replace known FrozenLake behavior probabilities with the clinical BC fit.

Only the denominator model is trained. Previously frozen actor, target policies,
critic, test trajectories, and online/exact reference values remain unchanged.
The calibrated RF is chosen before looking at test OPE; raw RF is diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import fit_behavior
from metric import evaluate_ope, probability_metrics
from model import full_action_proba
from gym_test.run_experiment import POLICIES, core_hashes, sha256, summarize, write_csv, write_json

DENOMINATORS = ("known_uniform", "bc_calibrated", "bc_raw")


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def artifact_hashes(directory):
    return {str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and not path.name.startswith("._")}


def features(data, horizon):
    """Categorical current position and observable remaining time, not future data."""
    state = np.asarray(data["state"], dtype=np.int64)
    step = np.asarray(data["time_step"], dtype=np.int64)
    if not np.all((state >= 0) & (state < 16)) or not np.all((step >= 0) & (step < horizon)):
        raise ValueError("Unexpected FrozenLake state/time step")
    values = np.zeros((len(state), 17), dtype=np.float32)
    values[np.arange(len(state)), state] = 1
    values[:, 16] = (horizon - step) / horizon
    return values


def read_baseline_rows(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key in ("policy", "mode"):
                continue
            if not value:
                row[key] = None
            elif key in ("seed", "episodes", "nonzero_terminal_paths", "target_compatible_successes", "n_bootstrap") or key.endswith("_defined_bootstrap"):
                row[key] = int(value)
            else:
                row[key] = float(value)
        row["behavior_model"] = "known_uniform"
    return rows


def probability_diagnostic(action, probs):
    # True uniform probabilities are consulted only after the fit, for this audit.
    log_kl = np.mean(np.sum(.25 * np.log(.25 / probs), axis=1))
    logged = probs[np.arange(len(action)), action]
    return {**probability_metrics(action, probs, num_actions=4),
            "rmse_to_known_uniform": float(np.sqrt(np.mean((probs - .25)**2))),
            "mean_kl_known_uniform_to_estimate": float(log_kl),
            "minimum_probability": float(np.min(probs)),
            "maximum_probability": float(np.max(probs)),
            "logged_action_probability_mean": float(np.mean(logged)),
            "logged_action_probability_minimum": float(np.min(logged)),
            "logged_action_probability_maximum": float(np.max(logged))}


def evaluate_seed(baseline, output, baseline_protocol, baseline_rows, seed):
    started = time.perf_counter()
    directory = output / f"seed_{seed}"
    behavior_dir = directory / "behavior"
    behavior_dir.mkdir(parents=True, exist_ok=False)
    frozen_dir = baseline / f"seed_{seed}"
    train = load_npz(frozen_dir / "datasets/actor_train.npz")
    test = load_npz(frozen_dir / "datasets/ope_test.npz")
    gamma, horizon = baseline_protocol["gamma"], baseline_protocol["horizon"]
    train_state, test_state = features(train, horizon), features(test, horizon)
    raw, calibrated, report = fit_behavior(
        train_state, train["action"], train["done"], groups=train["episode"],
        random_seed=seed, n_jobs=4, n_estimators=200, num_actions=4,
        calibration_fraction=.2, validation_fraction=.2,
        selection="calibrated", require_all_actions=True)
    partition = report.pop("_partition_indices")
    np.savez_compressed(behavior_dir / "partition.npz", **partition)
    joblib.dump(raw, behavior_dir / "raw.joblib", compress=3)
    joblib.dump(calibrated, behavior_dir / "calibrated.joblib", compress=3)
    probabilities = {"raw": full_action_proba(raw, test_state, num_actions=4),
                     "calibrated": full_action_proba(calibrated, test_state, num_actions=4),
                     "known": np.full((len(test_state), 4), .25)}
    for name in ("raw", "calibrated"):
        reloaded = joblib.load(behavior_dir / f"{name}.joblib")
        np.testing.assert_allclose(full_action_proba(reloaded, test_state, num_actions=4),
                                   probabilities[name], rtol=1e-12, atol=1e-12)
    np.savez_compressed(behavior_dir / "test_probabilities.npz", **probabilities)
    write_json(behavior_dir / "fit.json", report)
    diagnostic = {
        "test": {name: probability_diagnostic(test["action"], probs)
                 for name, probs in probabilities.items()},
        "primary_behavior": "calibrated", "raw_is_diagnostic_only": True,
        "no_test_selection": True, "model_reload_predictions_match": True,
        "features": "16 one-hot current-state indicators + normalized remaining time",
        "fit_seconds": time.perf_counter() - started}
    write_json(behavior_dir / "diagnostics.json", diagnostic)
    print(f"seed={seed}: clinical BC fitted; test LL raw={diagnostic['test']['raw']['log_loss']:.6f}, "
          f"calibrated={diagnostic['test']['calibrated']['log_loss']:.6f}; seconds={time.perf_counter()-started:.1f}", flush=True)
    rows, traces = [], []
    for index, (policy, _, _) in enumerate(POLICIES):
        original = next(row for row in baseline_rows if row["seed"] == seed and row["policy"] == policy)
        frozen = load_npz(frozen_dir / "policies" / f"{policy}.npz")
        known_result = json.loads((frozen_dir / policy / "ope.json").read_text())
        traces.extend({"seed": seed, "behavior_model": "known_uniform", "policy": policy,
                       "time_step": step + 1, "ess": ess}
                      for step, ess in enumerate(known_result["weights"]["per_decision_ess_with_absorbing_padding"]))
        for denominator, prob_key in (("bc_calibrated", "calibrated"), ("bc_raw", "raw")):
            result = evaluate_ope(
                test["action"], test["reward"], test["done"], frozen["target_probs"],
                probabilities[prob_key], frozen["q_values"], gamma=gamma, ratio_cap=None,
                n_bootstrap=baseline_protocol["n_bootstrap"], seed=seed * 1009 + 100 + index,
                metadata={"environment": "FrozenLake-v1", "seed": seed, "policy": policy,
                          "behavior_model": denominator, "primary": denominator == "bc_calibrated",
                          "baseline_dir": str(baseline), "actor_and_critic_unchanged": True,
                          "behavior_fit_selection": "calibrated fixed before test",
                          "actual_collector": "uniform .25; estimated denominator in this experiment"})
            write_json(directory / denominator / policy / "ope.json", result)
            step_ess = result["weights"]["per_decision_ess_with_absorbing_padding"]
            weights = result["weights"]
            row = {**original, "behavior_model": denominator,
                   "dr": result["dr"], "wdr": result["wdr"], "wis": result["wis"],
                   "terminal_ess": weights["trajectory_ess"],
                   "first_step_ess": step_ess[0], "last_step_ess": step_ess[-1],
                   "max_normalized_weight": weights["maximum_normalized_trajectory_weight"],
                   "nonzero_terminal_paths": original["episodes"] - weights["exact_zero_trajectory_weight_count"]}
            for estimator in ("dr", "wdr", "wis"):
                interval = result["bootstrap"]["intervals"][estimator]
                row[f"{estimator}_ci_low"] = interval["low"]
                row[f"{estimator}_ci_high"] = interval["high"]
                row[f"{estimator}_defined_bootstrap"] = interval["defined_finite_resamples"]
            rows.append(row)
            traces.extend({"seed": seed, "behavior_model": denominator, "policy": policy,
                           "time_step": step + 1, "ess": ess}
                          for step, ess in enumerate(step_ess))
            print(f"seed={seed} {policy} {denominator}: WDR={result['wdr']}, "
                  f"WIS={result['wis']}, ESS={weights['trajectory_ess']}", flush=True)
    write_csv(directory / "per_step_ess.csv", traces)
    return rows, traces


def summarize_denominators(rows):
    return [{"behavior_model": denominator, **row}
            for denominator in DENOMINATORS
            for row in summarize([item for item in rows if item["behavior_model"] == denominator])]


def plot_comparison(directory, rows, traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, (policy, _, _) in enumerate(POLICIES[:-1]):
        for denominator, linestyle in (("known_uniform", "-"), ("bc_calibrated", "--")):
            curves = []
            for seed in sorted({row["seed"] for row in traces}):
                curve = [row["ess"] for row in traces if row["seed"] == seed and row["policy"] == policy
                         and row["behavior_model"] == denominator]
                curves.append(curve)
            length = max(map(len, curves))
            padded = np.asarray([curve + [curve[-1]] * (length - len(curve)) for curve in curves], dtype=float)
            finite = np.isfinite(padded)
            mean = np.full(length, np.nan)
            np.divide(np.where(finite, padded, 0).sum(axis=0), finite.sum(axis=0), out=mean,
                      where=finite.sum(axis=0) > 0)
            axes[0].plot(np.arange(1, length+1), mean, color=colors(index), linestyle=linestyle,
                         label=f"{policy}: {'known' if denominator == 'known_uniform' else 'BC'}")
    axes[0].set(xlabel="Time step (absorbing padding after termination)", ylabel="Cumulative-weight ESS",
                yscale="log", title="Frozen targets: known behavior vs clinical BC")
    axes[0].legend(fontsize=8)
    names = [name for name, _, _ in POLICIES]
    x = np.arange(len(names))
    for denominator, marker in (("known_uniform", "o"), ("bc_calibrated", "x")):
        for estimator, color in (("wdr", "tab:blue"), ("wis", "tab:orange")):
            values = []
            for policy in names:
                items = [row[estimator] for row in rows if row["policy"] == policy and
                         row["behavior_model"] == denominator and row[estimator] is not None]
                values.append(float(np.mean(items)) if items else np.nan)
            axes[1].plot(x + (.04 if denominator == "bc_calibrated" else -.04), values,
                         marker=marker, linestyle="none", color=color,
                         label=f"{estimator.upper()}: {'known' if denominator == 'known_uniform' else 'BC'}")
    truth = [np.mean([row["exact_value"] for row in rows if row["policy"] == policy and
                     row["behavior_model"] == "known_uniform"]) for policy in names]
    axes[1].plot(x, truth, "_", color="black", markersize=14, label="Exact value (unchanged)")
    axes[1].set(xticks=x, xticklabels=names, ylabel="Discounted return", title="OPE with estimated denominators")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(fontsize=8)
    fig.savefig(directory / "bc_comparison.png", dpi=180)
    plt.close(fig)


def write_report(directory, protocol, summary):
    def fmt(value):
        return "undefined" if value is None else f"{value:.6f}"
    lines = ["# FrozenLake: clinical behavior cloning versus known behavior", "",
             "The original FrozenLake logs, fitted actor, target policies, FQE critics, and exact/online values are frozen. "
             "Only the behavior-probability denominator is replaced. Data collection still used uniform probability 0.25.", "",
             "The primary behavior model calls the same agent.fit_behavior used for clinical per-step evaluation: "
             "RandomForestClassifier (200 trees, depth 20, min leaf 4, sqrt features), followed by sigmoid calibration "
             "of a frozen RF and multiclass normalization. Complete actor-training episodes are split 60% fit, "
             "20% calibration, 20% validation. Calibrated RF is fixed before test; raw RF is an additional diagnostic, "
             "never selected using test ESS or OPE.", "",
             "Features: 16 one-hot current-state indicators and normalized remaining time. Training uses observed actions "
             "as labels, without rewards, Q values, future states, true behavior probabilities, or test data. "
             "The validation cohort is held out from behavior fitting/calibration, but belongs to the original actor-training cohort.", "",
             f"Seeds={protocol['seeds']}; {protocol['sizes']['train_episodes']:,} actor-training episodes and "
             f"{protocol['sizes']['test_episodes']:,} independent OPE episodes per seed. Gamma={protocol['gamma']}; "
             f"horizon={protocol['horizon']}. No importance-ratio cap, probability floor, or output clipping.", "",
             "| Target | Behavior denominator | FQE | DR | WDR | WIS | Terminal ESS | First-step ESS |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        if row["behavior_model"] == "bc_raw":
            continue
        values = [row["policy"], row["behavior_model"]] + [fmt(row[f"{key}_mean"])
                 for key in ("fqe", "dr", "wdr", "wis", "terminal_ess", "first_step_ess")]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "Values above are 3-seed means. summary.csv includes ranges and defined-seed counts; metrics.csv "
              "includes every seed, raw-RF diagnostics, and whole-episode bootstrap intervals.", "",
              "![BC denominator comparison](bc_comparison.png)", "",
              "The behavior_control target remains the true uniform policy, not the fitted BC policy. Thus pi=b_hat "
              "does not hold for its BC-denominator estimate; its ESS need not equal N. Estimated probabilities introduce "
              "nuisance-model error even though the real collector has perfect uniform support.", "",
              "Greedy still has no fully compatible observed episode; replacing a strictly positive denominator "
              "cannot repair zero target probabilities. Undefined final-weight ESS/WIS/WDR remain undefined.", "",
              "WIS uses full-trajectory ratios. WDR uses prefix ratios with per-time normalization and absorbing padding. "
              "Terminal ESS describes final-weight concentration, not the effective sample size of the full WDR estimate. "
              f"Bootstrap uses {protocol['n_bootstrap']} episode resamples with fitted behavior, actor, and critic fixed; "
              "it does not include BC fitting uncertainty. Seed variation is a separate sensitivity diagnostic.", "",
              "Exact Gym values and online outcomes are diagnostic references. This is a denominator-estimation "
              "experiment on the existing tabular offline policy; clinical RL models are unchanged.", "",
              "Artifacts: fitted raw/calibrated joblib models, full group partitions, fit/validation reports, saved "
              "test probabilities, full OPE JSON, per-step ESS, immutable baseline hashes, independent verification "
              "and completion receipts."]
    (directory / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline, output = args.baseline_dir.resolve(), args.output_dir.resolve()
    prior = json.loads((baseline / "completion_receipt.json").read_text())
    if prior["status"] != "complete":
        raise ValueError("Baseline experiment must be complete and verified")
    baseline_protocol = json.loads((baseline / "protocol.json").read_text())
    baseline_hashes = artifact_hashes(baseline)
    before = core_hashes()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = {
        "environment": baseline_protocol["environment"], "baseline_dir": str(baseline),
        "seeds": baseline_protocol["seeds"], "gamma": baseline_protocol["gamma"],
        "horizon": baseline_protocol["horizon"], "sizes": baseline_protocol["sizes"],
        "n_bootstrap": baseline_protocol["n_bootstrap"], "ratio_cap": None, "probability_floor": None,
        "actual_collector": "uniform probability .25 for each of four actions",
        "primary_behavior": "bc_calibrated", "raw_behavior": "diagnostic only; no selection",
        "behavior_implementation": "agent.fit_behavior(selection='calibrated', num_actions=4, require_all_actions=True)",
        "features": [f"state_{state}" for state in range(16)] + ["remaining_time_fraction"],
        "behavior_fit_data": "original actor_train only; complete-episode 60/20/20 fit/calibration/validation",
        "target_critic_data_reference_unchanged": True,
        "bootstrap_scope": "episode sampling; fitted behavior, actor, critic fixed; same seeds as baseline",
        "core_source_hashes_before": before, "baseline_artifact_sha256": baseline_hashes,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__, "joblib": joblib.__version__},
        "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "protocol.json", protocol)
    baseline_rows = read_baseline_rows(baseline / "metrics.csv")
    rows, traces = list(baseline_rows), []
    for seed in protocol["seeds"]:
        seed_rows, seed_traces = evaluate_seed(baseline, output, baseline_protocol, baseline_rows, seed)
        rows.extend(seed_rows)
        traces.extend(seed_traces)
        write_csv(output / "metrics.csv", rows)
    summary = summarize_denominators(rows)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "per_step_ess.csv", traces)
    plot_comparison(output, rows, traces)
    write_report(output, protocol, summary)
    after = core_hashes()
    baseline_after = artifact_hashes(baseline)
    if before != after or baseline_hashes != baseline_after:
        raise RuntimeError("Core or baseline artifacts changed during BC experiment")
    write_json(output / "experiment_receipt.json", {
        "status": "experiment_finished_pending_independent_verification", "output_dir": str(output),
        "elapsed_seconds": time.perf_counter() - started, "rows": len(rows),
        "core_source_hashes_after": after, "core_source_unchanged": True, "baseline_artifacts_unchanged": True,
        "artifact_sha256": artifact_hashes(output), "finished_utc": datetime.now(timezone.utc).isoformat()})
    print(f"BC comparison finished: {output}", flush=True)


if __name__ == "__main__":
    main()
