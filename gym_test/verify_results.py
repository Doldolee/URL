#!/usr/bin/env python3
"""Independently verify the finite-horizon FrozenLake OPE experiment.

Usage: python gym_test/verify_results.py --run-dir gym_test/results/RUN

The full-data reference uses direct long-double importance-ratio products;
it does not call the production log-weight helpers. WDR is additionally
compared against the independent Decimal equation on 30 complete episodes.
Statistical agreement with true policy value is reported by the experiment,
not asserted here: very low ESS can coexist with a correct implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def number(value):
    return None if value is None else float(value)


def same(actual, expected, *, name, rtol=2e-10, atol=2e-11):
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise AssertionError(f"{name}: {actual!r} != {expected!r}")
        return
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol,
                               err_msg=name)


def find_named_value(value, name):
    if isinstance(value, dict):
        if name in value:
            return value[name]
        for child in value.values():
            found = find_named_value(child, name)
            if found is not None:
                return found
    return None


def episodes_from_data(data, *, horizon):
    required = ("state", "action", "reward", "next_state", "terminated",
                "truncated", "done", "time_step", "episode")
    missing = set(required).difference(data)
    if missing:
        raise AssertionError(f"Missing dataset arrays: {sorted(missing)}")
    rows = len(data["state"])
    assert rows > 0
    for name in required:
        assert np.asarray(data[name]).shape == (rows,), name
    for name in ("terminated", "truncated", "done"):
        assert np.isin(data[name], [0, 1]).all(), name
    np.testing.assert_array_equal(data["done"].astype(bool),
                                  data["terminated"] | data["truncated"])
    assert data["done"][-1], "Final episode is unfinished"
    ends = np.flatnonzero(data["done"])
    starts = np.r_[0, ends[:-1] + 1]
    lengths = ends - starts + 1
    assert lengths.max() <= horizon
    for begin, end, length in zip(starts, ends, lengths):
        sl = slice(begin, end + 1)
        np.testing.assert_array_equal(data["time_step"][sl], np.arange(length))
        assert np.unique(data["episode"][sl]).size == 1
        if length > 1:
            np.testing.assert_array_equal(data["next_state"][begin:end],
                                          data["state"][begin + 1:end + 1])
        if data["truncated"][end] and not data["terminated"][end]:
            assert length == horizon, "Early nonterminal time-limit cutoff"
    assert np.unique(data["episode"][starts]).size == len(starts)
    assert np.isfinite(data["reward"]).all()
    assert np.isin(data["reward"], [0, 1]).all()
    assert np.all((data["state"] >= 0) & (data["state"] < 16))
    assert np.all((data["next_state"] >= 0) & (data["next_state"] < 16))
    assert np.all((data["action"] >= 0) & (data["action"] < 4))
    return starts, ends, lengths


def direct_reference(data, arrays, gamma, starts, ends):
    """Direct products and arithmetic, independent of metric.py helpers."""
    dtype = np.longdouble
    pi = np.asarray(arrays["target_probs"], dtype=dtype)
    behavior = np.asarray(arrays["behavior_probs"], dtype=dtype)
    q = np.asarray(arrays["q_values"], dtype=dtype)
    action = data["action"].astype(int)
    reward = np.asarray(data["reward"], dtype=dtype)
    rows = np.arange(len(action))
    assert np.all(behavior[rows, action] > 0)
    ratios = pi[rows, action] / behavior[rows, action]
    v = (pi * q).sum(axis=1, dtype=dtype)
    qa = q[rows, action]
    n = len(starts)
    horizon = int((ends - starts + 1).max())
    padded = np.empty((n, horizon), dtype=dtype)
    padded_reward = np.zeros((n, horizon), dtype=dtype)
    padded_qa = np.zeros_like(padded_reward)
    padded_v = np.zeros_like(padded_reward)
    discount = np.power(dtype(gamma), np.arange(horizon, dtype=dtype))
    returns = np.empty(n, dtype=dtype)
    episode_dr = np.empty(n, dtype=dtype)
    success = np.empty(n, dtype=bool)
    compatible = np.empty(n, dtype=bool)
    for i, (begin, end) in enumerate(zip(starts, ends)):
        sl = slice(begin, end + 1)
        length = end - begin + 1
        weights = np.cumprod(ratios[sl], dtype=dtype)
        previous = np.r_[dtype(1), weights[:-1]]
        padded[i, :length] = weights
        padded[i, length:] = weights[-1]
        padded_reward[i, :length] = reward[sl]
        padded_qa[i, :length] = qa[sl]
        padded_v[i, :length] = v[sl]
        returns[i] = np.sum(discount[:length] * reward[sl], dtype=dtype)
        episode_dr[i] = np.sum(discount[:length] *
                               (weights * (reward[sl] - qa[sl]) + previous * v[sl]),
                               dtype=dtype)
        success[i] = np.any(reward[sl] > 0)
        # On macOS arm64 longdouble has float64 precision. Genuine support
        # zeros must remain distinct from arithmetic product underflow.
        compatible[i] = np.all(ratios[sl] > 0)
    assert np.isfinite(padded).all(), "Direct products overflowed"
    sums = padded.sum(axis=0, dtype=dtype)
    valid = sums > 0
    normalized = np.zeros_like(padded)
    normalized[:, valid] = padded[:, valid] / sums[valid]
    ess = [number(1 / np.sum(normalized[:, t] ** 2, dtype=dtype))
           if valid[t] else None for t in range(horizon)]
    terminal = normalized[:, -1]
    wis = number(np.dot(terminal, returns)) if valid[-1] else None
    wdr = None
    if valid.all():
        previous = np.column_stack((np.full(n, dtype(1) / n), normalized[:, :-1]))
        contributions = (np.sum(normalized * (padded_reward - padded_qa), axis=0,
                                dtype=dtype) +
                         np.sum(previous * padded_v, axis=0, dtype=dtype))
        wdr = number(np.sum(discount * contributions, dtype=dtype))
    return {
        "dr": number(episode_dr.mean(dtype=dtype)), "wis": wis, "wdr": wdr,
        "terminal_ess": ess[-1], "per_decision_ess": ess,
        "maximum_normalized_trajectory_weight": number(terminal.max())
        if valid[-1] else None,
        "nonzero_terminal_paths": int(compatible.sum()),
        "direct_product_terminal_underflows": int(np.sum(compatible & (padded[:, -1] == 0))),
        "target_compatible_successes": int(np.sum(compatible & success)),
        "observed_return_mean": number(returns.mean(dtype=dtype)),
        "undefined_time_steps": np.flatnonzero(~valid).tolist(),
    }


def forward_value(reward, continuation, target, gamma):
    """Forward live-state occupancy, independent of backward FQE recursion."""
    dtype = np.longdouble
    reward = np.asarray(reward, dtype=dtype)
    continuation = np.asarray(continuation, dtype=dtype)
    target = np.asarray(target, dtype=dtype)
    assert reward.shape == (16, 4)
    assert continuation.shape == (16, 4, 16)
    assert np.isfinite(reward).all() and np.isfinite(continuation).all()
    assert np.all(continuation >= 0)
    assert np.all(continuation.sum(axis=2) <= 1 + dtype(1e-12))
    horizon = len(target) - 1
    occupancy = np.zeros(16, dtype=dtype)
    occupancy[0] = 1
    discount, value = dtype(1), dtype(0)
    for time_step in range(horizon):
        joint = occupancy[:, None] * target[horizon - time_step]
        value += discount * np.sum(joint * reward, dtype=dtype)
        occupancy = np.sum(joint[:, :, None] * continuation, axis=(0, 1), dtype=dtype)
        discount *= dtype(gamma)
    return number(value)


def verify_logged_model(data, model, prefix, counts):
    """Reconstruct frozen empirical model statistics from its logged partition."""
    np.testing.assert_array_equal(model[f"{prefix}_counts"], counts)
    reward_sums = np.zeros((16, 4), dtype=np.longdouble)
    continuation_counts = np.zeros((16, 4, 16), dtype=np.int64)
    state, action = data["state"], data["action"]
    np.add.at(reward_sums, (state, action), data["reward"])
    live = ~data["terminated"].astype(bool)
    np.add.at(continuation_counts, (state[live], action[live], data["next_state"][live]), 1)
    denominator = np.maximum(counts, 1)
    same(model[f"{prefix}_reward"], reward_sums / denominator,
         name=f"{prefix}: logged reward sufficient statistics")
    same(model[f"{prefix}_kernel"], continuation_counts / denominator[:, :, None],
         name=f"{prefix}: logged natural-termination continuation statistics")


def verify_seed(seed_dir, gamma, evaluate_ope, reference_wdr, metrics):
    datasets = {}
    data_reports = {}
    actor = load_npz(seed_dir / "model.npz")
    actor_q = actor["actor_q"]
    assert actor_q.ndim == 3 and actor_q.shape[1:] == (16, 4), actor_q.shape
    assert actor_q.shape[0] > 1, "Horizon must be positive"
    assert np.isfinite(actor_q).all()
    np.testing.assert_array_equal(actor_q[0], 0)
    horizon = actor_q.shape[0] - 1
    live_states = np.array([0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14])
    true_path = seed_dir / "diagnostic_true_kernel.npz"
    true_model = load_npz(true_path)
    source_hashes = {
        str((seed_dir / "model.npz").relative_to(seed_dir.parent)):
            sha256(seed_dir / "model.npz"),
        str(true_path.relative_to(seed_dir.parent)): sha256(true_path),
    }
    for partition in ("actor_train", "critic_train", "ope_test"):
        path = seed_dir / "datasets" / f"{partition}.npz"
        data = load_npz(path)
        starts, ends, lengths = episodes_from_data(data, horizon=horizon)
        datasets[partition] = (data, starts, ends)
        counts = np.bincount(data["state"] * 4 + data["action"], minlength=64)
        counts = counts.reshape(16, 4)
        live_counts = counts[live_states]
        if partition in ("actor_train", "critic_train"):
            assert np.all(live_counts > 0), f"{partition}: unseen live state/action"
            verify_logged_model(data, actor, partition.removesuffix("_train"), counts)
        data_reports[partition] = {
            "episodes": len(starts), "transitions": len(data["state"]),
            "min_live_state_action_count": int(live_counts.min()),
            "unobserved_live_state_action_pairs": int(np.sum(live_counts == 0)),
            "state_action_counts": counts.tolist(), "max_length": int(lengths.max()),
            "pure_time_limit_truncations": int(np.sum(
                data["truncated"][ends] & ~data["terminated"][ends])),
        }
        source_hashes[str(path.relative_to(seed_dir.parent))] = sha256(path)
    data, starts, ends = datasets["ope_test"]
    remain = horizon - data["time_step"]
    policy_reports = {}
    for path in sorted((seed_dir / "policies").glob("*.npz")):
        if path.name.startswith("._"):
            continue  # macOS AppleDouble metadata are not NumPy datasets.
        name = path.stem
        arrays = load_npz(path)
        rows = len(data["state"])
        for key in ("target_probs", "behavior_probs", "q_values"):
            assert arrays[key].shape == (rows, 4), (name, key, arrays[key].shape)
            assert np.isfinite(arrays[key]).all(), (name, key)
        np.testing.assert_allclose(arrays["behavior_probs"], .25, rtol=0, atol=0)
        np.testing.assert_allclose(arrays["target_probs"].sum(axis=1), 1,
                                   rtol=0, atol=1e-12)
        assert np.all(arrays["target_probs"] >= 0)
        for key in ("target_table", "critic_q", "exact_q"):
            assert arrays[key].shape == actor_q.shape, (name, key)
            assert np.isfinite(arrays[key]).all(), (name, key)
        for key in ("critic_q", "exact_q"):
            np.testing.assert_array_equal(arrays[key][0], 0,
                                          err_msg=f"{name}: {key} Q_0 is not zero")
        np.testing.assert_allclose(arrays["target_probs"],
                                   arrays["target_table"][remain, data["state"]],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(arrays["q_values"],
                                   arrays["critic_q"][remain, data["state"]],
                                   rtol=0, atol=1e-12)
        result_path = seed_dir / name / "ope.json"
        result = json.loads(result_path.read_text())
        assert result["ratio_cap"] is None, "Primary ratios must be unclipped"
        assert result["episodes_used"] == len(starts)
        assert result["episodes_excluded"] == 0
        same(result["gamma"], gamma, name=f"{name}: gamma")
        oracle_forward = forward_value(true_model["reward"], true_model["continuation"],
                                       arrays["target_table"], gamma)
        empirical_forward = forward_value(actor["critic_reward"], actor["critic_kernel"],
                                          arrays["target_table"], gamma)
        exact_initial = number(np.dot(arrays["target_table"][horizon, 0],
                                      arrays["exact_q"][horizon, 0]))
        fqe_initial = number(np.dot(arrays["target_table"][horizon, 0],
                                    arrays["critic_q"][horizon, 0]))
        same(exact_initial, oracle_forward, name=f"{name}: forward exact Gym value")
        same(fqe_initial, empirical_forward, name=f"{name}: forward empirical FQE value")
        metric_row = metrics[(int(seed_dir.name.removeprefix("seed_")), name)]
        same(float(metric_row["exact_value"]), oracle_forward,
             name=f"{name}: CSV exact Gym value")
        same(float(metric_row["fqe"]), empirical_forward, name=f"{name}: CSV FQE value")
        direct = direct_reference(data, arrays, gamma, starts, ends)
        for estimator in ("dr", "wis", "wdr"):
            same(result[estimator], direct[estimator], name=f"{name}: {estimator}")
        weights = result["weights"]
        same(weights["trajectory_ess"], direct["terminal_ess"],
             name=f"{name}: terminal ESS")
        same(weights["maximum_normalized_trajectory_weight"],
             direct["maximum_normalized_trajectory_weight"], name=f"{name}: max weight")
        expected_steps = weights["per_decision_ess_with_absorbing_padding"]
        assert len(expected_steps) == len(direct["per_decision_ess"])
        for t, (actual, expected) in enumerate(zip(expected_steps, direct["per_decision_ess"])):
            same(actual, expected, name=f"{name}: ESS at t={t}")
        assert weights["exact_zero_trajectory_weight_count"] == (
            len(starts) - direct["nonzero_terminal_paths"])
        assert result["numerical_status"]["undefined_wdr_time_steps"] == direct["undefined_time_steps"]
        cutoff = int(ends[min(len(ends), 30) - 1]) + 1
        args = [data["action"][:cutoff], data["reward"][:cutoff], data["done"][:cutoff],
                arrays["target_probs"][:cutoff], arrays["behavior_probs"][:cutoff],
                arrays["q_values"][:cutoff]]
        subset = evaluate_ope(*args, gamma=gamma, n_bootstrap=0)
        decimal = reference_wdr(*args, gamma=gamma)
        same(subset["wdr"], decimal["value"], name=f"{name}: independent Decimal WDR")
        for column in decimal["columns"]:
            t = column["time_step"]
            same(subset["weights"]["per_decision_ess_with_absorbing_padding"][t],
                 column["ess"], name=f"{name}: Decimal subset ESS t={t}")
        if name == "behavior_control":
            np.testing.assert_allclose(arrays["target_probs"], arrays["behavior_probs"],
                                       rtol=0, atol=0)
            same(direct["terminal_ess"], len(starts), name="pi=b terminal ESS")
            same(direct["per_decision_ess"], [len(starts)] * len(expected_steps),
                 name="pi=b all-step ESS")
            same(result["wis"], direct["observed_return_mean"], name="pi=b WIS return")
        policy_reports[name] = {
            "direct_product_reference": direct,
            "decimal_subset": {"episodes": min(len(ends), 30), "transitions": cutoff,
                               "wdr": decimal["value"],
                               "undefined_time_step": decimal["undefined_time_step"]},
            "forward_occupancy_values": {"exact_gym": oracle_forward,
                                         "empirical_fqe": empirical_forward},
            "bootstrap_diagnostics": {
                estimator: {
                    **result["bootstrap"]["intervals"][estimator],
                    "oracle_inside_reported_interval": (
                        None if result["bootstrap"]["intervals"][estimator]["low"] is None
                        else result["bootstrap"]["intervals"][estimator]["low"] <= oracle_forward
                        <= result["bootstrap"]["intervals"][estimator]["high"]),
                    "accuracy_or_coverage_asserted": False,
                } for estimator in ("dr", "wis", "wdr")
            },
            "checks_passed": ["full DR/WIS/WDR direct products", "full terminal ESS",
                              "full absorbing-padded step ESS", "horizon-indexed arrays",
                              "Q_0 zero", "known uniform behavior", "Decimal subset WDR/ESS",
                              "forward exact Gym value", "forward empirical FQE value"],
        }
        source_hashes[str(path.relative_to(seed_dir.parent))] = sha256(path)
        source_hashes[str(result_path.relative_to(seed_dir.parent))] = sha256(result_path)
    assert set(policy_reports) == {"behavior_control", "softmax_T1", "softmax_T0p1",
                                  "softmax_T0p02", "greedy"}, set(policy_reports)
    # A real-dataset control with a zero critic isolates the estimator identities.
    behavior = np.full((len(data["state"]), 4), .25)
    zero_control = evaluate_ope(data["action"], data["reward"], data["done"],
                                behavior, behavior, np.zeros_like(behavior),
                                gamma=gamma, n_bootstrap=0)
    observed = policy_reports["behavior_control"]["direct_product_reference"]["observed_return_mean"]
    for estimator in ("dr", "wis", "wdr"):
        same(zero_control[estimator], observed, name=f"pi=b Q=0 {estimator}")
    return {"horizon": horizon, "dataset_checks": data_reports, "policies": policy_reports,
            "pi_equals_b_zero_critic_checks": "passed", "artifact_sha256": source_hashes}


def write_new_json(path, value):
    """Publish once; refuse to replace an existing verification artifact."""
    lock_path = path.with_name(f".{path.name}.lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = None
    try:
        if path.exists():
            raise FileExistsError(f"Verification already exists: {path}")
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".verification-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Exclusive writer lock plus rename works on exFAT (no hard links).
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    protocol_path = run_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    gamma = find_named_value(protocol, "gamma")
    gamma = .98 if gamma is None else float(gamma)
    from metric import evaluate_ope
    reference_path = ROOT / "tests" / "wdr_reference.py"
    spec = importlib.util.spec_from_file_location("independent_wdr_reference", reference_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    seed_dirs = sorted(path for path in run_dir.glob("seed_*") if path.is_dir())
    assert seed_dirs, "No seed directories found"
    with (run_dir / "metrics.csv").open(newline="") as handle:
        metrics = {(int(row["seed"]), row["policy"]): row for row in csv.DictReader(handle)}
    report = {
        "status": "passed", "gamma": gamma,
        "reference_arithmetic": "independent direct numpy.longdouble products and Decimal WDR",
        "longdouble_mantissa_bits": int(np.finfo(np.longdouble).nmant),
        "checks_do_not_assert": ["low-ESS estimator accuracy", "bootstrap coverage",
                                 "model-fitting independence from arrays alone"],
        "source_sha256": {"verifier": sha256(__file__),
                          "metric.py": sha256(ROOT / "metric.py"),
                          "tests/wdr_reference.py": sha256(reference_path),
                          "protocol.json": sha256(protocol_path)},
        "seeds": {},
    }
    for seed_dir in seed_dirs:
        report["seeds"][seed_dir.name] = verify_seed(seed_dir, gamma, evaluate_ope,
                                                   module.reference_wdr, metrics)
        print(f"{seed_dir.name}: all independent checks passed", flush=True)
    output = run_dir / "verification.json"
    write_new_json(output, report)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
