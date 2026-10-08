"""Learn from FrozenLake logs, evaluate with the clinical OPE engine, and check truth.

Training uses only logged transitions. Gym's transition model is read *after*
policy/critic fitting and is used only as a diagnostic ground-truth reference.
Both actor and FQE include remaining time: every estimate targets the same
finite-horizon discounted return, including a zero continuation at the time cap.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metric import evaluate_ope, policy_probs

POLICIES = (
    ("behavior_control", "behavior", None),
    ("softmax_T1", "softmax", 1.0),
    ("softmax_T0p1", "softmax", 0.1),
    ("softmax_T0p02", "softmax", 0.02),
    ("greedy", "greedy", None),
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def core_hashes():
    names = ["agent.py", "model.py", "metric.py", "util.py", "plot.py", "detector.py", "source_manifest.json"]
    return {name: sha256(ROOT / name) for name in names if (ROOT / name).is_file()}


def make_env(horizon):
    return gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True,
                    max_episode_steps=horizon)


def collect(episodes, seed, horizon, target_table=None):
    """Collect complete episodes; uniform behavior has exactly known p(a|s)=1/4."""
    env = make_env(horizon)
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ("state", "action", "reward", "next_state",
                                 "terminated", "truncated", "done", "time_step", "episode")}
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            for step in range(horizon):
                probs = np.full(4, .25) if target_table is None else target_table[horizon - step, state]
                action = int(rng.choice(4, p=probs))
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                row = (state, action, reward, next_state, terminated, truncated, done, step, episode)
                for key, value in zip(values, row):
                    values[key].append(value)
                state = next_state
                if done:
                    break
            else:
                raise RuntimeError("TimeLimit did not close the episode")
    finally:
        env.close()
    return {key: np.asarray(value, dtype=np.float64 if key == "reward" else np.int64)
            for key, value in values.items()}


def describe(data, gamma):
    ends = np.flatnonzero(data["done"])
    starts = np.r_[0, ends[:-1] + 1]
    lengths = ends - starts + 1
    returns = np.add.reduceat(data["reward"] * gamma ** data["time_step"], starts)
    return {
        "episodes": int(len(ends)), "transitions": int(len(data["state"])),
        "mean_length": float(lengths.mean()), "max_length": int(lengths.max()),
        "successes": int(np.sum(data["reward"][ends] == 1)),
        "time_limit_episodes": int(np.sum(data["truncated"][ends] != 0)),
        "discounted_return_mean": float(returns.mean()),
        "discounted_return_se": float(returns.std(ddof=1) / np.sqrt(len(ends))),
    }


def fit_logged_kernel(data, active_states):
    """Sufficient statistics of a time-homogeneous, tabular Bellman regression.

    A row is a logged transition, including TimeLimit transitions. Natural
    termination stops continuation. TimeLimit is represented instead by the
    remaining-time coordinate; Q[0]=0. No transition probabilities from Gym
    are consulted. Counts pooled across time exploit stationarity of FrozenLake.
    """
    pair = data["state"] * 4 + data["action"]
    counts = np.bincount(pair, minlength=64).reshape(16, 4)
    if np.any(counts[active_states] == 0):
        raise RuntimeError("Offline data are missing a nonterminal state/action; increase collection")
    denominator = np.maximum(counts, 1)
    reward = np.bincount(pair, weights=data["reward"], minlength=64).reshape(16, 4) / denominator
    live = data["terminated"] == 0
    triples = pair[live] * 16 + data["next_state"][live]
    continuation = np.bincount(triples, minlength=1024).reshape(16, 4, 16) / denominator[:, :, None]
    return reward, continuation, counts


def fit_actor(reward, continuation, horizon, gamma):
    """Finite-horizon tabular fitted Q iteration, fitted only from offline logs."""
    q = np.zeros((horizon + 1, 16, 4))
    for remaining in range(1, horizon + 1):
        next_value = q[remaining - 1].max(axis=1)
        q[remaining] = reward + gamma * np.einsum("san,n->sa", continuation, next_value)
    return q


def target_from_actor(actor_q, mode, temperature):
    if mode == "behavior":
        return np.full(actor_q.shape, .25)
    shape = actor_q.shape
    return policy_probs(actor_q.reshape(-1, 4), mode=mode,
                        temperature=1. if temperature is None else temperature).reshape(shape)


def fit_fqe(reward, continuation, target, horizon, gamma):
    """Exact solution of finite-horizon empirical tabular FQE Bellman regressions."""
    q = np.zeros((horizon + 1, 16, 4))
    for remaining in range(1, horizon + 1):
        next_value = np.sum(target[remaining - 1] * q[remaining - 1], axis=1)
        q[remaining] = reward + gamma * np.einsum("san,n->sa", continuation, next_value)
    return q


def diagnostic_true_kernel(horizon):
    """Read Gym P only for post-fitting oracle evaluation, never for training."""
    env = make_env(horizon)
    reward, continuation = np.zeros((16, 4)), np.zeros((16, 4, 16))
    try:
        for state, actions in env.unwrapped.P.items():
            for action, outcomes in actions.items():
                for probability, next_state, r, terminated in outcomes:
                    reward[state, action] += probability * r
                    if not terminated:
                        continuation[state, action, next_state] += probability
    finally:
        env.close()
    return reward, continuation


def evaluate_seed(args, run_dir, seed):
    directory = run_dir / f"seed_{seed}"
    (directory / "datasets").mkdir(parents=True)
    (directory / "policies").mkdir()
    env = make_env(args.horizon)
    active_states = np.flatnonzero(np.isin(env.unwrapped.desc.flatten(), [b"S", b"F"]))
    env.close()
    datasets, descriptions = {}, {}
    for offset, (name, episodes) in enumerate((
            ("actor_train", args.train_episodes), ("critic_train", args.critic_episodes),
            ("ope_test", args.test_episodes)), start=1):
        stream_seed = seed * 1009 + offset
        print(f"seed={seed} collecting {name}: {episodes} episodes", flush=True)
        data = collect(episodes, stream_seed, args.horizon)
        np.savez_compressed(directory / "datasets" / f"{name}.npz", **data)
        datasets[name] = data
        descriptions[name] = dict(describe(data, args.gamma), stream_seed=stream_seed)
    actor_reward, actor_kernel, actor_counts = fit_logged_kernel(datasets["actor_train"], active_states)
    critic_reward, critic_kernel, critic_counts = fit_logged_kernel(datasets["critic_train"], active_states)
    actor_q = fit_actor(actor_reward, actor_kernel, args.horizon, args.gamma)
    # Freeze all target policies and offline critics before consulting the oracle.
    targets, critics = {}, {}
    for name, mode, temperature in POLICIES:
        targets[name] = target_from_actor(actor_q, mode, temperature)
        critics[name] = fit_fqe(critic_reward, critic_kernel, targets[name], args.horizon, args.gamma)
    np.savez_compressed(directory / "model.npz", actor_q=actor_q,
                        actor_counts=actor_counts, critic_counts=critic_counts,
                        actor_reward=actor_reward, actor_kernel=actor_kernel,
                        critic_reward=critic_reward, critic_kernel=critic_kernel,
                        active_states=active_states)
    write_json(directory / "datasets.json", descriptions)
    true_reward, true_kernel = diagnostic_true_kernel(args.horizon)
    np.savez_compressed(directory / "diagnostic_true_kernel.npz", reward=true_reward, continuation=true_kernel)
    test = datasets["ope_test"]
    remaining = args.horizon - test["time_step"]
    behavior_probs = np.full((len(remaining), 4), .25)
    rows, traces = [], {}
    for index, (name, mode, temperature) in enumerate(POLICIES):
        started = time.perf_counter()
        target, critic = targets[name], critics[name]
        target_probs = target[remaining, test["state"]]
        q_values = critic[remaining, test["state"]]
        exact_q = fit_fqe(true_reward, true_kernel, target, args.horizon, args.gamma)
        exact_value = float(np.dot(target[args.horizon, 0], exact_q[args.horizon, 0]))
        fqe_value = float(np.dot(target[args.horizon, 0], critic[args.horizon, 0]))
        np.savez_compressed(directory / "policies" / f"{name}.npz", target_table=target,
                            critic_q=critic, exact_q=exact_q, target_probs=target_probs,
                            behavior_probs=behavior_probs, q_values=q_values)
        print(f"seed={seed} policy={name}: OPE / {args.bootstrap} bootstraps", flush=True)
        result = evaluate_ope(test["action"], test["reward"], test["done"],
                              target_probs, behavior_probs, q_values,
                              gamma=args.gamma, ratio_cap=None, n_bootstrap=args.bootstrap,
                              seed=seed * 1009 + 100 + index,
                              metadata={"environment": "FrozenLake-v1", "policy": name,
                                        "behavior": "known uniform .25", "seed": seed})
        write_json(directory / name / "ope.json", result)
        mc_seed = seed * 1009 + 1000 + index
        online = collect(args.mc_episodes, mc_seed, args.horizon, target)
        online_description = dict(describe(online, args.gamma), stream_seed=mc_seed)
        # Save online trajectories only for diagnostics; never re-fit/select from them.
        np.savez_compressed(directory / name / "online_mc.npz", **online)
        write_json(directory / name / "online_mc.json", online_description)
        ends = np.flatnonzero(test["done"])
        starts = np.r_[0, ends[:-1] + 1]
        chosen_target = target_probs[np.arange(len(remaining)), test["action"]]
        compatible = np.logical_and.reduceat(chosen_target > 0, starts)
        step_ess = result["weights"]["per_decision_ess_with_absorbing_padding"]
        row = {
            "seed": seed, "policy": name, "mode": mode, "temperature": temperature,
            "episodes": args.test_episodes, "exact_value": exact_value, "fqe": fqe_value,
            "dr": result["dr"], "wdr": result["wdr"], "wis": result["wis"],
            "terminal_ess": result["weights"]["trajectory_ess"],
            "first_step_ess": step_ess[0], "last_step_ess": step_ess[-1],
            "max_normalized_weight": result["weights"]["maximum_normalized_trajectory_weight"],
            "nonzero_terminal_paths": int(compatible.sum()),
            "target_compatible_successes": int(np.sum(compatible & (test["reward"][ends] == 1))),
            "mc_mean": online_description["discounted_return_mean"],
            "mc_se": online_description["discounted_return_se"],
            "mc_success_rate": online_description["successes"] / args.mc_episodes,
            "mc_mean_length": online_description["mean_length"], "n_bootstrap": args.bootstrap,
        }
        for estimator in ("dr", "wdr", "wis"):
            interval = result["bootstrap"]["intervals"][estimator]
            row[f"{estimator}_ci_low"], row[f"{estimator}_ci_high"] = interval["low"], interval["high"]
            row[f"{estimator}_defined_bootstrap"] = interval["defined_finite_resamples"]
        rows.append(row)
        traces[name] = step_ess
        print(f"seed={seed} {name}: truth={exact_value:.6f}, FQE={fqe_value:.6f}, "
              f"WDR={result['wdr']}, ESS={row['terminal_ess']}, "
              f"seconds={time.perf_counter()-started:.1f}", flush=True)
    trace_rows = [{"seed": seed, "policy": name, "time_step": step + 1, "ess": value}
                  for name, values in traces.items() for step, value in enumerate(values)]
    write_csv(directory / "per_step_ess.csv", trace_rows)
    return rows, traces


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    result = []
    for name, _, _ in POLICIES:
        group = [row for row in rows if row["policy"] == name]
        item = {"policy": name, "seeds": len(group), "episodes_per_seed": group[0]["episodes"]}
        for metric in ("exact_value", "fqe", "dr", "wdr", "wis", "terminal_ess", "first_step_ess",
                       "max_normalized_weight", "mc_mean", "mc_success_rate", "mc_mean_length"):
            values = [row[metric] for row in group if row[metric] is not None and np.isfinite(row[metric])]
            item[f"{metric}_mean"] = float(np.mean(values)) if values else None
            item[f"{metric}_min"] = float(np.min(values)) if values else None
            item[f"{metric}_max"] = float(np.max(values)) if values else None
            item[f"{metric}_defined_seeds"] = len(values)
        result.append(item)
    return result


def plot_results(run_dir, rows, traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for name, _, _ in POLICIES:
        curves = [trace[name] for trace in traces.values()]
        length = max(len(curve) for curve in curves)
        padded = np.array([curve + [curve[-1]] * (length - len(curve)) for curve in curves], dtype=float)
        # NaNs mark undefined normalization; never replace them with zero ESS.
        finite = np.isfinite(padded)
        mean = np.full(length, np.nan)
        np.divide(np.where(finite, padded, 0).sum(axis=0), finite.sum(axis=0),
                  out=mean, where=finite.sum(axis=0) > 0)
        axes[0].plot(np.arange(1, length + 1), mean, label=name)
    axes[0].set(xlabel="Time step (absorbing padding after termination)", ylabel="Cumulative-weight ESS",
                yscale="log", title="Same known behavior: uniform probability 0.25")
    axes[0].legend(fontsize=8)
    labels = [name for name, _, _ in POLICIES]
    x = np.arange(len(labels))
    for offset, (metric, label) in enumerate((("exact_value", "Exact Gym value"), ("fqe", "Offline FQE"),
                                             ("wdr", "WDR"), ("wis", "WIS"))):
        group_means = []
        for name in labels:
            values = [row[metric] for row in rows if row["policy"] == name and row[metric] is not None]
            group_means.append(float(np.mean(values)) if values else np.nan)
        axes[1].plot(x + (offset - 1.5) * .06, group_means, "o", label=label)
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Discounted return", title="Policy value: reference versus OPE")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(fontsize=8)
    fig.savefig(run_dir / "ope_and_ess.png", dpi=180)
    plt.close(fig)


def write_report(run_dir, args, summary):
    lines = ["# FrozenLake offline RL: OPE and ESS", "",
             "FrozenLake-v1 / default 4x4 map / slippery=True. Known behavior: each action probability 0.25.",
             f"Gamma={args.gamma}; finite horizon={args.horizon}. Seeds={args.seeds}.", "",
             f"Per seed: {args.train_episodes:,} actor-training episodes, {args.critic_episodes:,} independent critic-training episodes, "
             f"{args.test_episodes:,} held-out OPE episodes, and {args.mc_episodes:,} direct online episodes per frozen target.", "",
             "The offline actor is finite-horizon tabular fitted Q iteration. FQE solves fixed-policy tabular Bellman regressions "
             "on separate logged transitions. Time-homogeneous transition statistics are pooled across time; remaining time is "
             "included in actor and critic. Gym P is consulted only after fitting for diagnostic exact values. "
             "No neural-network/clinical checkpoint is trained or changed.", "",
             "Targets are the same offline Q table converted to three softmax temperatures and greedy, plus an on-policy "
             "behavior control. These are distinct policies, not estimator modifications or policy reselections.", "",
             "| Target | Exact value | Online MC | FQE | WDR | WIS | Terminal ESS | First-step ESS |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "undefined" if value is None else f"{value:.6f}"
    for row in summary:
        fields = [row["policy"]] + [fmt(row[f"{name}_mean"]) for name in (
            "exact_value", "mc_mean", "fqe", "wdr", "wis", "terminal_ess", "first_step_ess")]
        lines.append("| " + " | ".join(fields) + " |")
    lines += ["", "These are seed means; finite-only estimator means are conditional on defined seeds. "
              "See summary.csv for min/max and defined counts, metrics.csv for every seed and fixed-nuisance bootstrap interval.", "",
              "![OPE and ESS](ope_and_ess.png)", "",
              "WIS uses full-trajectory cumulative ratios. DR/WDR use ratios accumulated through each time step; WDR normalizes "
              "per time step. Ended trajectories remain in the denominator via absorbing padding. Gamma discounts rewards, "
              "not importance ratios. No ratio cap, probability floor, output clipping, or removal of zero-weight episodes.", "",
              f"Bootstrap: {args.bootstrap} whole-episode resamples, fixed actor/behavior/critic. Intervals cover sampling of OPE "
              "episodes, not actor or critic fitting uncertainty. FQE has a constant initial state, so no meaningless initial-state "
              "bootstrap is reported. Seed variation is a separate training/data sensitivity diagnostic.", "",
              "Exact Gym values and online outcomes are evaluation-only references. This experiment does not validate MIMIC "
              "behavior estimates, clinical policy support, or clinical causality. Final-weight ESS is a weight diagnostic, "
              "not the effective sample size of the entire WDR estimator. Greedy terminal weights can be supported only by short "
              "failed episodes: compatible path and success counts are saved.", "",
              "Artifacts: protocol.json, metrics.csv, summary.csv, per-seed NPZ datasets/policies, full OPE JSON, "
              "online trajectories, per-step ESS CSV, verification.json, and completion_receipt.json."]
    (run_dir / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--train-episodes", type=int, default=20000)
    parser.add_argument("--critic-episodes", type=int, default=10000)
    parser.add_argument("--test-episodes", type=int, default=3000)
    parser.add_argument("--mc-episodes", type=int, default=3000)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=.98)
    args = parser.parse_args()
    if not 0 <= args.gamma <= 1 or min(args.horizon, args.train_episodes, args.critic_episodes,
                                      args.test_episodes, args.mc_episodes) <= 0 or args.bootstrap < 0:
        parser.error("Invalid sizes, gamma, or bootstrap count")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    run_dir = args.output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    before = core_hashes()
    protocol = {"environment": "FrozenLake-v1", "map": ["SFFF", "FHFH", "FFFH", "HFFG"],
                "is_slippery": True, "gamma": args.gamma, "horizon": args.horizon,
                "behavior": "known uniform probability .25 for each of four actions", "seeds": args.seeds,
                "sizes": {key: getattr(args, key) for key in ("train_episodes", "critic_episodes", "test_episodes", "mc_episodes")},
                "n_bootstrap": args.bootstrap, "ratio_cap": None, "probability_floor": None,
                "actor": "finite-horizon offline tabular fitted Q iteration",
                "critic": "independent finite-horizon offline tabular FQE",
                "time_limit_objective": "capped discounted return; remaining time is part of state; Q[0]=0",
                "oracle_access": "Gym P only for diagnostic truth after all actor/critic fitting",
                "core_source_hashes_before": before,
                "versions": {"python": platform.python_version(), "gymnasium": gym.__version__, "numpy": np.__version__},
                "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(run_dir / "protocol.json", protocol)
    rows, traces = [], {}
    for seed in args.seeds:
        seed_rows, seed_traces = evaluate_seed(args, run_dir, seed)
        rows.extend(seed_rows)
        traces[seed] = seed_traces
        write_csv(run_dir / "metrics.csv", rows)
    summary = summarize(rows)
    write_csv(run_dir / "summary.csv", summary)
    plot_results(run_dir, rows, traces)
    write_report(run_dir, args, summary)
    after = core_hashes()
    if before != after:
        raise RuntimeError("Clinical core source changed during isolated experiment")
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and not path.name.startswith("._"))
    write_json(run_dir / "experiment_receipt.json", {
        "status": "experiment_finished_pending_independent_verification", "output_dir": str(run_dir),
        "elapsed_seconds": time.perf_counter() - started, "rows": len(rows),
        "core_source_hashes_after": after, "core_source_unchanged": before == after,
        "artifact_sha256": {str(path.relative_to(run_dir)): sha256(path) for path in files},
        "finished_utc": datetime.now(timezone.utc).isoformat()})
    print(f"Experiment finished: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
