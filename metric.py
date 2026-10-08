"""OPE, fixed-policy FQE, calibration, bootstrap and OOD metrics."""
from __future__ import annotations
import copy
import hashlib
import math
import numpy as np
import time
import torch
from model import FQECritic, HeparinFQECritic
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from typing import Callable, Optional
from util import NUM_ACTIONS, _cpu_threads, _trajectory_split, initial_state_indices




# Importance-sampling estimators

_LOG_MAX = math.log(np.finfo(np.float64).max)


def policy_probs(q, mode="greedy", temperature=1.0):
    """Freeze target probabilities from Q; greedy ties use the first argmax."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] == 0 or not np.isfinite(q).all():
        raise ValueError("q must be a nonempty finite [transitions, actions] array")
    if mode == "greedy":
        result = np.zeros_like(q)
        result[np.arange(len(q)), np.argmax(q, axis=1)] = 1.0
        return result
    if mode != "softmax":
        raise ValueError("mode must be 'greedy' or 'softmax'")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    # Subtract before dividing, avoiding a positive overflow from q / temperature.
    with np.errstate(over="ignore", under="ignore"):
        shifted = (q - q.max(axis=1, keepdims=True)) / temperature
        result = np.exp(shifted)
    return result / result.sum(axis=1, keepdims=True)


def _probabilities(values, shape, name, zero_rows=None):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
        raise ValueError(name + " must be finite probabilities with shape " + str(shape))
    sums = result.sum(axis=1)
    valid = np.isclose(sums, 1.0, rtol=0, atol=1e-6)
    if zero_rows is not None:
        valid |= np.asarray(zero_rows, dtype=bool) & (sums == 0)
    if not valid.all():
        raise ValueError(name + " rows must sum to one; probabilities are not silently renormalized")
    return result


def anchored_policy_probs(q, behavior_probs, strength=1.0):
    """Return a distinct stochastic policy anchored to a frozen behavior model.

    For a_best=argmax_a Q(s,a),
      p(a|s) = b(a|s) * (1 + strength * I[a=a_best])
               / (1 + strength * b(a_best|s)).
    strength must be finite and nonnegative. With strength=1, the one-step
    density ratio p/b is at most 2 wherever b>0. This is a different policy
    from greedy CQL or softmax(Q); it is not a correction of their OPE values.
    Zeros in b remain zero. No probability floor, epsilon or Q bound is used,
    and neither input array is modified. Support containment is relative to
    the supplied behavior model, not proof of the true behavior support.
    """
    values = np.asarray(q, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("q must be a nonempty finite [transitions, actions] array")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    behavior = _probabilities(behavior_probs, values.shape, "behavior_probs")
    best = np.argmax(values, axis=1)
    rows = np.arange(len(values))
    result = behavior.copy()
    result[rows, best] *= 1.0 + strength
    return result / (1.0 + strength * behavior[rows, best])[:, None]


def _signed_sum(log_weights, coefficients):
    """Return (float64 value or None, sign, log(abs(value)), numeric status)."""
    lw = np.asarray(log_weights, dtype=np.float64).ravel()
    coeff = np.asarray(coefficients, dtype=np.float64).ravel()
    mask = (coeff != 0) & np.isfinite(lw)
    if not mask.any():
        return 0.0, 0, None, "finite"
    logs = lw[mask] + np.log(np.abs(coeff[mask]))
    offset = float(logs.max())
    total = math.fsum((np.sign(coeff[mask]) * np.exp(logs - offset)).tolist())
    if total == 0:
        return 0.0, 0, None, "finite"
    sign = 1 if total > 0 else -1
    logabs = offset + math.log(abs(total))
    if logabs > _LOG_MAX:
        return None, sign, float(logabs), "float64_overflow"
    value = sign * math.exp(logabs)
    return float(value), sign, float(logabs), "underflow_to_zero" if value == 0 else "finite"


def _normalize(log_weights):
    """Log-sum-exp normalization along trajectories; no uniform fallback."""
    logs = np.asarray(log_weights, dtype=np.float64)
    if logs.ndim == 1:
        logs = logs[:, None]
    maximum = np.max(logs, axis=0)
    defined = np.isfinite(maximum)
    normalized = np.zeros_like(logs)
    if defined.any():
        scaled = np.exp(logs[:, defined] - maximum[defined])
        normalized[:, defined] = scaled / scaled.sum(axis=0, keepdims=True)
    return normalized, defined


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size), "finite_count": int(finite.size),
        "zero_count": int(np.sum(values == 0)),
        "min": float(finite.min()) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "mean": float(math.fsum((finite / finite.size).tolist())) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "p05": float(np.quantile(finite, .05)) if finite.size else None,
        "p95": float(np.quantile(finite, .95)) if finite.size else None,
    }


def _confidence_interval(samples, alpha, total):
    values = np.asarray([value for value in samples if value is not None and np.isfinite(value)], dtype=np.float64)
    return {
        "low": float(np.quantile(values, alpha / 2)) if values.size else None,
        "high": float(np.quantile(values, 1 - alpha / 2)) if values.size else None,
        "resamples": int(total), "defined_finite_resamples": int(values.size),
        "defined_finite_fraction": float(values.size / total) if total else None,
        "undefined_or_overflow_resamples": int(total - values.size),
        "status": "not_requested" if total == 0 else (
            "all_resamples_defined" if values.size == total else
            "interval_conditional_on_defined_resamples" if values.size else "no_defined_resamples"),
    }


def evaluate_ope(actions, rewards, dones, target_probs, behavior_probs, q_values, *,
                 gamma=0.98, next_target_probs=None, next_q_values=None,
                 ratio_cap: Optional[float] = None, cumulative_weight_cap: Optional[float] = None,
                 n_bootstrap=1000, seed=20261007,
                 alpha=0.05, low_probability_threshold=0.01, batch_size=None,
                 metadata=None, return_episode_details=False, episode_groups=None,
                 return_bootstrap_samples=False):
    """Return DR/WIS/WDR, support/weight diagnostics and episode-bootstrap intervals.

    q_values[N,A] must come from an independently trained/cross-fitted critic for
    the fixed target. This function cannot verify independence from arrays alone.
    Optional next_* arrays provide terminal-masked TD and adjacency diagnostics;
    estimators themselves use the standard non-recursive, current-state formula.
    All trajectories are included; batch_size is a recorded compatibility hint
    and never selects trajectories or changes normalization. Each nonzero done
    must equal 1 and ends a trajectory. A final unfinished trajectory is rejected.

    ratio_cap clips each step ratio BEFORE multiplication. cumulative_weight_cap
    instead clips each ORIGINAL cumulative prefix product AFTER multiplication;
    the clipped value is never fed into the next prefix product. These are separate
    biased sensitivity estimators, not changes to the actor or behavior policy.
    Choose at most one cap. Previous-time DR/WDR coefficients use the same capped
    prefixes; rho_-1=1 and w_-1=1/n. Absorbing padding carries the capped final weight.

    Bootstrap holds target, behavior, and critic arrays fixed. episode_groups may
    supply one subject/cluster label per trajectory; complete clusters are then
    resampled, keeping all episodes of each selected subject. The point estimand
    remains episode-weighted. Otherwise complete trajectories are resampled.
    This accounts for sampling only, not nuisance estimation, unmeasured confounding,
    support failure or selection bias. Undefined resamples are counted explicitly;
    a finite-resample percentile interval is labeled conditional on being defined.
    return_bootstrap_samples exposes the same resampled point values (None when
    undefined/overflow), for explicitly requested nested nuisance-refit analyses.
    The returned samples themselves still hold the supplied nuisance arrays fixed.
    """
    q = np.asarray(q_values, dtype=np.float64)
    if q.ndim != 2 or not q.size or not np.isfinite(q).all():
        raise ValueError("q_values must be a nonempty finite [N,A] critic array")
    nrows, nactions = q.shape
    action_raw = np.asarray(actions).reshape(-1)
    if len(action_raw) != nrows or not np.isfinite(action_raw).all() or np.any(action_raw != np.floor(action_raw)):
        raise ValueError("actions must contain one integer action per transition")
    action = action_raw.astype(np.int64)
    if np.any(action < 0) or np.any(action >= nactions):
        raise ValueError("action is outside the probability/critic action axis")
    reward = np.asarray(rewards, dtype=np.float64).reshape(-1)
    done = np.asarray(dones).reshape(-1)
    if len(reward) != nrows or not np.isfinite(reward).all():
        raise ValueError("rewards must be finite and aligned with transitions")
    if len(done) != nrows or not np.all((done == 0) | (done == 1)) or done[-1] != 1:
        raise ValueError("dones must be binary, aligned, and terminate the final trajectory")
    done = done.astype(bool)
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0,1] for these finite trajectories")
    if ratio_cap is not None and (not np.isfinite(ratio_cap) or ratio_cap <= 0):
        raise ValueError("ratio_cap must be None or finite and positive")
    if cumulative_weight_cap is not None and (not np.isfinite(cumulative_weight_cap) or cumulative_weight_cap <= 0):
        raise ValueError("cumulative_weight_cap must be None or finite and positive")
    if ratio_cap is not None and cumulative_weight_cap is not None:
        raise ValueError("Choose either ratio_cap or cumulative_weight_cap, not both")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a nonnegative integer")
    if not 0 < alpha < 1 or not 0 <= low_probability_threshold <= 1:
        raise ValueError("invalid alpha or low_probability_threshold")
    if batch_size is not None and (not isinstance(batch_size, (int, np.integer)) or batch_size <= 0):
        raise ValueError("batch_size must be None or a positive integer")
    pi = _probabilities(target_probs, q.shape, "target_probs")
    behavior = _probabilities(behavior_probs, q.shape, "behavior_probs")
    idx = np.arange(nrows)
    pa, ba = pi[idx, action], behavior[idx, action]
    if np.any(ba == 0):
        raise ValueError("behavior probability of a recorded action is zero; ratios are undefined (no epsilon is added)")
    v = np.sum(pi * q, axis=1)
    qa = q[idx, action]
    with np.errstate(divide="ignore"):
        log_ratio_raw = np.log(pa) - np.log(ba)
    log_ratio = log_ratio_raw.copy()
    if ratio_cap is not None:
        log_ratio = np.minimum(log_ratio, math.log(ratio_cap))
    ends = np.flatnonzero(done)
    starts = np.r_[0, ends[:-1] + 1]
    lengths = ends - starts + 1
    nepisodes, horizon = len(ends), int(lengths.max())
    group_index = None
    group_count = nepisodes
    if episode_groups is not None:
        groups = np.asarray(episode_groups).reshape(-1)
        if len(groups) != nepisodes:
            raise ValueError("episode_groups must contain one cluster label per complete trajectory")
        mapping, codes = {}, []
        for label in groups.tolist():
            if label is None or (isinstance(label, float) and not np.isfinite(label)):
                raise ValueError("episode_groups cannot contain missing cluster labels")
            try:
                if label not in mapping:
                    mapping[label] = len(mapping)
                codes.append(mapping[label])
            except TypeError as error:
                raise ValueError("episode_groups labels must be hashable") from error
        group_index = np.asarray(codes, dtype=np.int64)
        group_count = len(mapping)
    log_discount = np.zeros(horizon, dtype=np.float64)
    if gamma == 0:
        log_discount[1:] = -np.inf
    else:
        log_discount = np.arange(horizon, dtype=np.float64) * math.log(gamma)
    padded_lw = np.empty((nepisodes, horizon), dtype=np.float64)
    padded_reward = np.zeros_like(padded_lw)
    padded_qa = np.zeros_like(padded_lw)
    padded_v = np.zeros_like(padded_lw)
    episode_sign = np.zeros(nepisodes, dtype=np.float64)
    episode_logabs = np.full(nepisodes, -np.inf)
    episode_dr = []
    observed_return = []
    capped_prefix_count = 0
    capped_terminal_count = 0
    for i, (start, end, length) in enumerate(zip(starts, ends, lengths)):
        sl = slice(int(start), int(end) + 1)
        lw = np.cumsum(log_ratio[sl], dtype=np.float64)
        if cumulative_weight_cap is not None:
            log_cap = math.log(cumulative_weight_cap)
            capped_prefix_count += int(np.sum(lw > log_cap))
            capped_terminal_count += int(lw[-1] > log_cap)
            lw = np.minimum(lw, log_cap)
        padded_lw[i, :length] = lw
        padded_lw[i, length:] = lw[-1]  # absorbing-state ratio=1; never remove from denominator
        padded_reward[i, :length] = reward[sl]
        padded_qa[i, :length] = qa[sl]
        padded_v[i, :length] = v[sl]
        discount = log_discount[:length]
        previous_lw = np.r_[0., lw[:-1]]
        # Three separate coefficient blocks avoid overflow in r - Q.
        value, sign, logabs, status = _signed_sum(
            np.r_[discount + lw, discount + lw, discount + previous_lw],
            np.r_[reward[sl], -qa[sl], v[sl]])
        episode_dr.append({"value": value, "status": status})
        episode_sign[i] = sign
        if logabs is not None:
            episode_logabs[i] = logabs
        observed_return.append(_signed_sum(discount, reward[sl])[0])
    observed_return = np.asarray(observed_return, dtype=np.float64)
    if not np.isfinite(observed_return).all():
        raise ValueError("discounted observed return exceeds float64 range")
    terminal_logs = padded_lw[:, -1]

    def point(counts):
        count = np.asarray(counts, dtype=np.float64)
        total_count = float(count.sum())
        with np.errstate(divide="ignore"):
            log_count = np.log(count)
        dr_result = _signed_sum(episode_logabs + log_count - math.log(total_count), episode_sign)
        normalized, valid = _normalize(terminal_logs + log_count)
        wis = float(np.dot(normalized[:, 0], observed_return)) if valid[0] else None
        step_norm, step_defined = _normalize(padded_lw + log_count[:, None])
        wdr = None
        wdr_status = "zero_total_weight" if not step_defined.all() else "finite"
        if step_defined.all():
            previous_norm = np.column_stack((count / total_count, step_norm[:, :-1]))
            # Each column is a convex average. Combine the signed discounted blocks
            # in log space rather than exponentiating raw cumulative ratios.
            terms = np.r_[np.sum(step_norm * padded_reward, axis=0),
                          -np.sum(step_norm * padded_qa, axis=0),
                          np.sum(previous_norm * padded_v, axis=0)]
            wdr_result = _signed_sum(np.tile(log_discount, 3), terms)
            wdr, wdr_status = wdr_result[0], wdr_result[3]
        return {"dr": dr_result[0], "wis": wis, "wdr": wdr}, {
            "dr": dr_result[3], "dr_sign": dr_result[1], "dr_log_abs": dr_result[2],
            "wis": "finite" if valid[0] else "zero_total_weight", "wdr": wdr_status,
            "undefined_wdr_time_steps": np.flatnonzero(~step_defined).tolist()}, normalized[:, 0], step_norm, step_defined

    estimates, numeric, normalized, step_norm, step_defined = point(np.ones(nepisodes))
    samples = {key: [] for key in ["dr", "wis", "wdr"]}
    rng = np.random.RandomState(seed)
    for _ in range(n_bootstrap):
        draw = rng.choice(group_count, size=group_count, replace=True)
        group_multiplicity = np.bincount(draw, minlength=group_count)
        counts = group_multiplicity if group_index is None else group_multiplicity[group_index]
        values = point(counts)[0]
        for name in samples:
            samples[name].append(values[name])
    overlap = np.sum(pi * (behavior > 0), axis=1)
    unsupported = np.sum(pi * (behavior == 0), axis=1)
    low_mass = np.sum(pi * (behavior < low_probability_threshold), axis=1)
    with np.errstate(over="ignore", under="ignore"):
        raw_ratio = np.exp(log_ratio_raw)
    ess = float(1 / np.dot(normalized, normalized)) if numeric["wis"] == "finite" else None
    step_ess = [float(1 / np.dot(step_norm[:, t], step_norm[:, t])) if step_defined[t] else None
                for t in range(horizon)]
    next_diagnostics = None
    if (next_target_probs is None) != (next_q_values is None):
        raise ValueError("next_target_probs and next_q_values must be provided together")
    if next_q_values is not None:
        nq = np.asarray(next_q_values, dtype=np.float64)
        if nq.shape != q.shape or not np.isfinite(nq).all():
            raise ValueError("next_q_values must be finite and aligned with q_values")
        npi = _probabilities(next_target_probs, q.shape, "next_target_probs", zero_rows=done)
        vn = np.sum(npi * nq, axis=1)
        vn[done] = 0.0
        internal = np.flatnonzero(~done)
        error = np.abs(vn[internal] - v[internal + 1])
        next_diagnostics = {
            "terminal_mask_applied": True,
            "max_abs_internal_next_value_alignment_error": float(error.max()) if error.size else 0.,
            "td_residual": _summary(reward + gamma * vn - qa),
            "used_for_estimates": False,
        }
    result = {
        **estimates, "numerical_status": numeric,
        "gamma": float(gamma), "ratio_cap": ratio_cap,
        "cumulative_weight_cap": cumulative_weight_cap,
        "analysis_kind": "unclipped_primary" if ratio_cap is None and cumulative_weight_cap is None else "clipped_sensitivity",
        "clipping": {"scope": "step_ratio" if ratio_cap is not None else "cumulative_prefix" if cumulative_weight_cap is not None else "none",
                     "capped_observed_prefix_count": capped_prefix_count,
                     "capped_terminal_trajectory_count": capped_terminal_count,
                     "cumulative_cap_feedback": False,
                     "bias_variance_sensitivity_only": ratio_cap is not None or cumulative_weight_cap is not None},
        "episodes_used": int(nepisodes), "episodes_excluded": 0, "transitions": nrows,
        "horizon_after_absorbing_padding": horizon, "batch_size_hint": batch_size,
        "definitions": {"dr": "standard non-recursive per-decision DR, averaged over complete trajectories",
                        "wis": "self-normalized complete-trajectory importance sampling",
                        "wdr": "per-time self-normalized DR with absorbing padding and w_-1=1/n"},
        "bootstrap": {"seed": int(seed), "n_bootstrap": int(n_bootstrap), "alpha": float(alpha),
                      "unit": "whole trajectory" if group_index is None else "whole subject cluster",
                      "independent_resampling_units": int(group_count),
                      "point_estimand": "episode-weighted mean", "nuisance_models_held_fixed": True,
                      "intervals": {name: _confidence_interval(values, alpha, n_bootstrap)
                                    for name, values in samples.items()}},
        "support": {"assessed_states": "observed current states only; target-induced state coverage is unverified",
                    "target_behavior_overlap_mass": _summary(overlap),
                    "target_mass_at_behavior_zero": _summary(unsupported),
                    "rows_with_unsupported_target_mass": int(np.sum(unsupported > 0)),
                    "observed_state_positivity_compatible_fraction": float(np.mean(unsupported == 0)),
                    "support_violation_at_observed_states": bool(np.any(unsupported > 0)),
                    "low_behavior_probability_threshold": float(low_probability_threshold),
                    "target_mass_below_behavior_probability_threshold": _summary(low_mass),
                    "observed_action_behavior_probability": _summary(ba),
                    "observed_action_target_probability": _summary(pa),
                    "observed_action_log_raw_ratio": _summary(log_ratio_raw),
                    "observed_action_raw_ratio": _summary(raw_ratio),
                    "raw_ratio_float64_overflow_count": int(np.sum(np.isposinf(raw_ratio))),
                    "explicitly_capped_step_count": int(np.sum(log_ratio_raw > math.log(ratio_cap))) if ratio_cap is not None else 0},
        "weights": {"trajectory_log_weight": _summary(terminal_logs),
                    "exact_zero_trajectory_weight_count": int(np.sum(np.isneginf(terminal_logs))),
                    "trajectory_weights_below_float64_smallest_subnormal": int(np.sum(terminal_logs < math.log(np.nextafter(0., 1.)))),
                    "trajectory_ess": ess, "trajectory_ess_fraction": ess / nepisodes if ess is not None else None,
                    "maximum_normalized_trajectory_weight": float(normalized.max()) if ess is not None else None,
                    "top10_normalized_trajectory_weight_share": float(np.sort(normalized)[-10:].sum()) if ess is not None else None,
                    "per_decision_ess_with_absorbing_padding": step_ess},
        "observed_discounted_return": _summary(observed_return),
        "next_critic_diagnostics": next_diagnostics,
        "metadata": {} if metadata is None else dict(metadata),
        "limitations": ["Arrays alone cannot verify independent/cross-fitted critic or propensity provenance.",
                        "Bootstrap intervals omit nuisance-estimation error and systematic OPE bias.",
                        "Any explicit weight cap introduces potential bias; capped ESS does not restore policy support.",
                        "ESS measures realized weight concentration and does not certify positivity or outcome accuracy.",
                        "Support diagnostics at observed states do not establish target-policy state coverage."],
    }
    if return_episode_details:
        result["episodes"] = [{"start": int(start), "end": int(end), "length": int(length),
                               "dr": episode_dr[i]["value"], "dr_status": episode_dr[i]["status"],
                               "log_abs_dr": float(episode_logabs[i]) if np.isfinite(episode_logabs[i]) else None,
                               "dr_sign": int(episode_sign[i]),
                               "log_weight": float(terminal_logs[i]) if np.isfinite(terminal_logs[i]) else None,
                               "weight_exactly_zero": bool(np.isneginf(terminal_logs[i])),
                               "normalized_weight": float(normalized[i]),
                               "observed_discounted_return": float(observed_return[i])}
                              for i, (start, end, length) in enumerate(zip(starts, ends, lengths))]
    if return_bootstrap_samples:
        result["bootstrap"]["samples"] = samples
    return result


# Behavior probability diagnostics

def _metrics(action, probability):
    report = probability_metrics(action, probability)
    fields = ["rows", "log_loss", "brier_multiclass_sum", "top_label_ece",
              "accuracy", "observed_action_zero_probability_count"]
    report["per_action_recorded_action_strata"] = {
        str(a): ({key: value for key, value in probability_metrics(
            action[action == a], probability[action == a]).items() if key in fields}
                 if np.any(action == a) else {"rows": 0, **{key: None for key in fields if key != "rows"}})
        for a in range(NUM_ACTIONS)
    }
    report["per_action_metric_scope"] = "same multiclass metrics within each recorded-action stratum; ECE remains top-label ECE"
    return report


def probability_metrics(action, probabilities, ece_bins=15, num_actions=NUM_ACTIONS):
    """Multiclass log loss, sum-of-classes Brier, and top-label ECE."""
    action = np.asarray(action, dtype=np.int64).reshape(-1)
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.shape != (len(action), num_actions) or not len(action):
        raise ValueError(f"metrics require nonempty N x {num_actions} probabilities")
    if not np.isfinite(probability).all() or (probability < 0).any() or not np.allclose(probability.sum(1), 1):
        raise ValueError("invalid probability array")
    observed = probability[np.arange(len(action)), action]
    predicted = probability.argmax(1)
    confidence = probability.max(1)
    correct = predicted == action
    # Only the reporting logarithm uses a floor, not saved probabilities or OPE.
    epsilon = np.finfo(np.float64).eps
    log_loss = float(-np.log(np.clip(observed, epsilon, 1)).mean())
    brier = float((np.square(probability).sum(1) - 2 * observed + 1).mean())
    bin_id = np.minimum((confidence * ece_bins).astype(int), ece_bins - 1)
    bins = []
    ece = 0.
    for index in range(ece_bins):
        mask = bin_id == index
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(action) * abs(accuracy - mean_confidence)
        bins.append({"lower": index/ece_bins, "upper": (index+1)/ece_bins,
                     "count": count, "accuracy": accuracy, "confidence": mean_confidence})
    confusion = np.zeros((num_actions, num_actions), dtype=np.int64)
    np.add.at(confusion, (action, predicted), 1)
    return {"rows": len(action), "log_loss": log_loss, "brier_multiclass_sum": brier,
            "top_label_ece": float(ece), "accuracy": float(correct.mean()),
            "observed_action_zero_probability_count": int((observed == 0).sum()),
            "log_loss_reporting_floor": epsilon, "ece_bins": bins,
            "confusion_true_row_predicted_column": confusion.tolist()}


# FQE validation and fitted-iteration helpers

def _states(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"{name} must have nonempty shape [rows, features].")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite observations.")
    return np.ascontiguousarray(result)


def _vector(values, name: str, rows: int) -> np.ndarray:
    result = np.asarray(values)
    if result.shape == (rows, 1):
        result = result[:, 0]
    if result.shape != (rows,):
        raise ValueError(f"{name} must have shape [{rows}] or [{rows}, 1].")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values.")
    return result


def _policy_probs(values, rows: int, actions: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] != rows or result.shape[1] < 1:
        raise ValueError("policy probabilities must have shape [rows, actions].")
    if actions is not None and result.shape[1] != actions:
        raise ValueError("Policy action count does not match critic output.")
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Policy probabilities must be finite and nonnegative.")
    if not np.allclose(result.sum(axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("Each fixed-policy probability row must sum to one.")
    # Do not normalize or mutate the caller's evaluated policy.
    return np.ascontiguousarray(result)


def validate_transitions(state, next_state, action, reward, done, next_policy_probs):
    """Validate alignment and the terminal-only reward assumption for the bound.

    Episodes must be contiguous, end in done=1, have zero intermediate reward,
    and have exactly one terminal reward equal to -1 or +1.  Nonterminal next
    observations are checked against the following row to catch misalignment.
    Terminal next-state placeholders are allowed and are masked from targets.
    """
    s = _states(state, "state")
    ns = _states(next_state, "next_state")
    n = len(s)
    if ns.shape != s.shape:
        raise ValueError("state and next_state must have the same shape.")
    probs = _policy_probs(next_policy_probs, n)
    a = _vector(action, "action", n)
    if not np.equal(a, np.floor(a)).all():
        raise ValueError("Actions must be integer indices.")
    a = np.asarray(a, dtype=np.int64)
    if (a < 0).any() or (a >= probs.shape[1]).any():
        raise ValueError("Action indices fall outside the fixed policy action set.")
    r = np.asarray(_vector(reward, "reward", n), dtype=np.float32)
    d = np.asarray(_vector(done, "done", n), dtype=np.float32)
    starts = initial_state_indices(d)
    terminal = d == 1
    if not np.allclose(r[~terminal], 0.0, rtol=0, atol=1e-6):
        raise ValueError("Bounded FQE requires zero nonterminal rewards.")
    if not np.allclose(np.abs(r[terminal]), 1.0, rtol=0, atol=1e-6):
        raise ValueError("Every terminal reward must be -1 or +1.")
    continuing_rows = np.flatnonzero(d[:-1] == 0)
    if not np.allclose(ns[continuing_rows], s[continuing_rows + 1], rtol=1e-5, atol=1e-6):
        raise ValueError("Nonterminal next_state is not aligned with the next row.")
    return s, ns, a, r, d, probs, starts


def _batch_targets(target, ns, rewards, done, fixed_probs, gamma):
    next_q = target(ns)
    next_value = (fixed_probs * next_q).sum(dim=1)
    return rewards + gamma * (1.0 - done) * next_value


def _residual_metrics(model, tensors, indices, gamma, batch_size):
    s, ns, a, r, d, probs = tensors
    predictions = []
    residuals = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = torch.from_numpy(indices[start:start + batch_size])
            observed_q = model(s[idx]).gather(1, a[idx, None]).squeeze(1)
            # Self-consistency diagnostic, independent of the lagged fit target.
            target = _batch_targets(model, ns[idx], r[idx], d[idx], probs[idx], gamma)
            predictions.append(observed_q.numpy())
            residuals.append((observed_q - target).numpy())
    q = np.concatenate(predictions)
    error = np.concatenate(residuals).astype(np.float64)
    terminal = d[indices].numpy() == 1
    mse = float(np.mean(error ** 2))
    terminal_mse = float(np.mean(error[terminal] ** 2)) if terminal.any() else None
    nonterminal_mse = float(np.mean(error[~terminal] ** 2)) if (~terminal).any() else None
    strata = [x for x in [terminal_mse, nonterminal_mse] if x is not None]
    return {
        "bellman_mse": mse,
        "bellman_rmse": float(np.sqrt(mse)),
        "balanced_bellman_mse": float(np.mean(strata)),
        "terminal_bellman_mse": terminal_mse,
        "nonterminal_bellman_mse": nonterminal_mse,
        "observed_action_q_mean": float(np.mean(q)),
        "observed_action_q_min": float(np.min(q)),
        "observed_action_q_max": float(np.max(q)),
    }, q


# Terminal-reward FQE

def fit_fqe(
    state,
    next_state,
    action,
    reward,
    done,
    next_policy_probs,
    *,
    gamma: float = 0.98,
    hidden_dim: int = 128,
    epochs: int = 50,
    min_epochs: int = 20,
    patience: int = 10,
    batch_size: int = 4096,
    validation_fraction: float = 0.15,
    lr: float = 1e-3,
    seed: int = 42,
    num_threads: int = 4,
    min_improvement: float = 1e-6,
    groups=None,
    progress_callback: Callable[[dict], None] | None = None,
):
    """Fit a fresh bounded critic on a supplied training buffer only.

    ``next_policy_probs[k]`` must be the evaluated policy at
    ``next_state[k]``.  It is frozen throughout all Bellman iterations.
    Training episodes are split into fit/validation episodes once. Optional
    ``groups`` labels (per row or per episode, e.g. patient IDs) keep multiple
    episodes from one patient in the same split. Validation
    chooses a critic epoch using an equal-weight average of terminal and
    nonterminal Bellman MSE, so sparse terminal signals are not hidden by many
    zero-reward rows.  No test/evaluation observations are accepted here.

    Returns ``(critic, report)``.  The report separates a validation plateau
    from a claim of small residual; stopping alone is not proof of convergence.
    """
    fit_started = time.perf_counter()
    if not 0 <= gamma < 1:
        raise ValueError("gamma must lie in [0, 1).")
    if epochs < 1 or not 1 <= min_epochs <= epochs:
        raise ValueError("Require 1 <= min_epochs <= epochs.")
    if min(patience, batch_size) < 1 or lr <= 0 or min_improvement < 0:
        raise ValueError("Invalid optimization/stopping parameters.")
    arrays = validate_transitions(state, next_state, action, reward, done, next_policy_probs)
    s, ns, a, r, d, probs, starts = arrays
    train_idx, val_idx, val_episodes, group_count, val_groups = _trajectory_split(d, r, validation_fraction, seed, groups)
    tensors = tuple(torch.from_numpy(x) for x in [s, ns, a, r, d, probs])
    shuffle_rng = np.random.default_rng(seed + 1)
    # Preserve the caller's torch RNG state; only the fresh critic is initialized.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = FQECritic(s.shape[1], probs.shape[1], hidden_dim)
    target = copy.deepcopy(model).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    best_metric = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    previous_val_q = None
    stop_reason = "max_epochs"
    with _cpu_threads(num_threads):
        initial_train, _ = _residual_metrics(model, tensors, train_idx, gamma, batch_size)
        initial_val, _ = _residual_metrics(model, tensors, val_idx, gamma, batch_size)
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            total_loss = 0.0
            shuffled = shuffle_rng.permutation(train_idx)
            for start in range(0, len(shuffled), batch_size):
                idx = torch.from_numpy(shuffled[start:start + batch_size])
                observed_q = model(tensors[0][idx]).gather(1, tensors[2][idx, None]).squeeze(1)
                with torch.no_grad():
                    targets = _batch_targets(
                        target, tensors[1][idx], tensors[3][idx], tensors[4][idx], tensors[5][idx], gamma
                    )
                loss = torch.mean((observed_q - targets) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(idx)
            train_metrics, _ = _residual_metrics(model, tensors, train_idx, gamma, batch_size)
            val_metrics, val_q = _residual_metrics(model, tensors, val_idx, gamma, batch_size)
            log = {
                "epoch": epoch,
                "fit_mse_lagged_target": total_loss / len(train_idx),
                "train": train_metrics,
                "validation": val_metrics,
                "validation_observed_q_mean_abs_change": None if previous_val_q is None else float(np.mean(np.abs(val_q - previous_val_q))),
                "validation_observed_q_max_abs_change": None if previous_val_q is None else float(np.max(np.abs(val_q - previous_val_q))),
                "epoch_elapsed_seconds": time.perf_counter() - epoch_started,
                "fit_elapsed_seconds": time.perf_counter() - fit_started,
            }
            history.append(log)
            previous_val_q = val_q.copy()
            metric = val_metrics["balanced_bellman_mse"]
            if epoch >= min_epochs and metric < best_metric - min_improvement:
                best_metric = metric
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            elif epoch >= min_epochs:
                stale_epochs += 1
            # Hard update once per epoch: a frozen target for each fitted sweep.
            target.load_state_dict(model.state_dict())
            if progress_callback is not None:
                progress_callback(log)
            if epoch >= min_epochs and stale_epochs >= patience:
                stop_reason = "validation_plateau"
                break
        model.load_state_dict(best_state)
        model.eval()
        selected_train, _ = _residual_metrics(model, tensors, train_idx, gamma, batch_size)
        selected_val, _ = _residual_metrics(model, tensors, val_idx, gamma, batch_size)
    report = {
        "critic": "bounded auxiliary FQE critic; independent two-hidden-layer MLP",
        "physical_q_bounds": [-1.0, 1.0],
        "bound_assumption": "one terminal reward in {-1,+1}; zero intermediate rewards; gamma < 1",
        "fixed_policy": "caller-supplied next_policy_probs; never inferred from FQE critic",
        "fixed_next_policy_sha256_float32": hashlib.sha256(probs.tobytes()).hexdigest(),
        "target": "r + gamma * (1-done) * sum_a fixed_pi(a|next_state) * Q_target(next_state,a)",
        "config": {
            "gamma": gamma, "hidden_dim": hidden_dim, "epochs": epochs,
            "min_epochs": min_epochs, "patience": patience, "batch_size": batch_size,
            "validation_fraction": validation_fraction, "lr": lr, "seed": seed,
            "num_threads": num_threads, "min_improvement": min_improvement,
            "target_update": "hard copy once per epoch",
        },
        "validation_selection_metric": "mean(terminal Bellman MSE, nonterminal Bellman MSE)",
        "split": {
            "scope": "supplied training episodes only; no final test data",
            "unit": "trajectory" if groups is None else "caller-supplied group (e.g. patient)",
            "train_rows": len(train_idx), "validation_rows": len(val_idx),
            "train_episodes": len(starts) - len(val_episodes), "validation_episodes": len(val_episodes),
            "validation_episode_indices_sha256": hashlib.sha256(val_episodes.tobytes()).hexdigest(),
            "train_groups": group_count - len(val_groups), "validation_groups": len(val_groups),
            "validation_group_indices_sha256": hashlib.sha256(val_groups.tobytes()).hexdigest(),
            "outcome_stratified": "terminal-outcome sign summed within group; preserves mixed-outcome patients",
        },
        "data_validation": {
            "complete_contiguous_episodes": len(starts),
            "terminal_positive": int(np.sum((d == 1) & (r > 0))),
            "terminal_negative": int(np.sum((d == 1) & (r < 0))),
            "nonterminal_reward_count": int(np.count_nonzero(r[d == 0])),
            "nonterminal_next_state_alignment_checked": True,
            "done_semantics": "1=terminal, 0=continuing",
        },
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "initial_train": initial_train,
        "initial_validation": initial_val,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - fit_started,
        "selected_epoch": best_epoch,
        "stop_reason": stop_reason,
        "selected_train": selected_train,
        "selected_validation": selected_val,
        "convergence_note": "Plateau or max_epochs alone does not prove convergence; inspect held-out residuals and epoch-wise Q changes. Bellman fit also does not establish coverage of the evaluated policy.",
        "history": history,
    }
    return model, report


def predict_q(model: FQECritic, state, *, batch_size: int = 4096, num_threads: int = 4) -> np.ndarray:
    """Evaluate the frozen CPU critic, with no updates or policy construction."""
    s = _states(state, "state")
    if s.shape[1] != model.state_dim or batch_size < 1:
        raise ValueError("Invalid prediction feature count or batch size.")
    model.eval()
    values = []
    with _cpu_threads(num_threads), torch.no_grad():
        for start in range(0, len(s), batch_size):
            values.append(model(torch.from_numpy(s[start:start + batch_size])).cpu().numpy())
    return np.concatenate(values, axis=0)


def policy_values(model: FQECritic, state, policy_probs, *, batch_size: int = 4096, num_threads: int = 4) -> np.ndarray:
    """V(s) under caller-supplied fixed policy, not a policy induced by Q."""
    s = _states(state, "state")
    probs = _policy_probs(policy_probs, len(s), model.num_actions)
    return (predict_q(model, s, batch_size=batch_size, num_threads=num_threads) * probs).sum(axis=1).astype(np.float64)


def evaluate_initial_states(model: FQECritic, state, done, policy_probs, *, batch_size: int = 4096, num_threads: int = 4):
    """Evaluate each held-out episode's initial state without fitting/selecting.

    ``policy_probs`` is the frozen evaluated policy at each *current* state
    (or, alternatively, at the initial states only), not ``next_policy_probs``.
    Bootstrap uncertainty is intentionally a separate operation below.
    """
    s = _states(state, "state")
    d = _vector(done, "done", len(s))
    starts = initial_state_indices(d)
    raw = np.asarray(policy_probs)
    if raw.ndim != 2:
        raise ValueError("policy_probs must be a probability matrix.")
    if len(raw) == len(s):
        probs = _policy_probs(raw, len(s), model.num_actions)[starts]
    else:
        probs = _policy_probs(raw, len(starts), model.num_actions)
    values = policy_values(model, s[starts], probs, batch_size=batch_size, num_threads=num_threads)
    return {
        "mean": float(values.mean()),
        "episode_count": len(values),
        "initial_values": values,
        "initial_row_indices": starts,
        "min_initial_value": float(values.min()),
        "max_initial_value": float(values.max()),
        "evaluation_scope": "frozen critic and frozen evaluated policy; initial states only",
    }


# Per-transition-reward FQE

def validate_heparin(state, next_state, action, reward, done, probs):
    s, ns = _states(state, 'state'), _states(next_state, 'next_state')
    assert s.shape == ns.shape
    p = _policy_probs(probs, len(s))
    a = _vector(action, 'action', len(s))
    if not np.equal(a, np.floor(a)).all() or (a < 0).any() or (a >= p.shape[1]).any():
        raise ValueError('Invalid action index')
    r = _vector(reward, 'reward', len(s)).astype(np.float32)
    d = _vector(done, 'done', len(s)).astype(np.float32)
    starts = initial_state_indices(d)
    if (np.abs(r) > 1.000001).any():
        raise ValueError('Expected per-transition reward in [-1,1]')
    continuing = np.flatnonzero(d[:-1] == 0)
    np.testing.assert_allclose(ns[continuing], s[continuing + 1], rtol=1e-5, atol=1e-6)
    return s, ns, a.astype(np.int64), r, d, p, starts


def fit_heparin_fqe(arrays, next_probs, groups, *, gamma=.98, epochs=100,
                    min_epochs=50, patience=20, seed=42, hidden_dim=128,
                    batch_size=2048, lr=1e-3, num_threads=1, progress_callback=None):
    if not 0 <= gamma < 1 or not 1 <= min_epochs <= epochs:
        raise ValueError('Invalid gamma or epoch budget')
    s, ns, a, r, d, probs, starts = validate_heparin(
        *[arrays[k] for k in ['state', 'next_state', 'action', 'reward', 'done']], next_probs)
    train_idx, val_idx, _, group_count, val_groups = _trajectory_split(d, r, .15, seed, groups)
    tensors = tuple(torch.from_numpy(x) for x in [s, ns, a, r, d, probs])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = HeparinFQECritic(s.shape[1], probs.shape[1], hidden_dim, 1 / (1-gamma))
    target = copy.deepcopy(model).eval()
    for p in target.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 1)
    best, best_epoch, best_state, stale = float('inf'), 0, None, 0
    history, begun = [], time.perf_counter()
    with _cpu_threads(num_threads):
        for epoch in range(1, epochs + 1):
            model.train()
            order = rng.permutation(train_idx)
            for begin in range(0, len(order), batch_size):
                idx = torch.from_numpy(order[begin:begin+batch_size])
                # Freeze one target network for the entire fitted iteration.
                with torch.no_grad():
                    y = _batch_targets(target, tensors[1][idx], tensors[3][idx],
                                       tensors[4][idx], tensors[5][idx], gamma)
                q = model(tensors[0][idx]).gather(1, tensors[2][idx,None]).squeeze(1)
                loss = torch.nn.functional.mse_loss(q, y)
                if not torch.isfinite(loss): raise ValueError('Nonfinite FQE loss')
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.eval()
            target.load_state_dict(model.state_dict())
            metrics, _ = _residual_metrics(model, tensors, val_idx, gamma, batch_size)
            history.append({'epoch':epoch, **metrics})
            if epoch >= min_epochs:
                if metrics['bellman_mse'] < best - 1e-6:
                    best, best_epoch, stale = metrics['bellman_mse'], epoch, 0
                    best_state = copy.deepcopy(model.state_dict())
                else: stale += 1
            if progress_callback: progress_callback(history[-1])
            if stale >= patience: break
        model.load_state_dict(best_state); model.eval()
        selected_train, _ = _residual_metrics(model,tensors,train_idx,gamma,batch_size)
        selected_val, _ = _residual_metrics(model,tensors,val_idx,gamma,batch_size)
    report = {'seed':seed, 'gamma':gamma, 'value_bound':model.value_bound,
              'reward_assumption':'each transition in [-1,1]; intermediate rewards allowed',
              'selected_epoch':best_epoch, 'epochs_completed':len(history),
              'selected_train':selected_train, 'selected_validation':selected_val,
              'selection_metric':'training-internal held-out patient observed-action Bellman MSE',
              'fit_rows':len(train_idx),'validation_rows':len(val_idx),
              'patient_groups':group_count,'validation_patient_groups':len(val_groups),
              'fixed_next_policy_sha256_float32':hashlib.sha256(np.ascontiguousarray(probs).tobytes()).hexdigest(),
              'elapsed_seconds':time.perf_counter()-begun,'history':history,
              'uncertainty_note':'Bellman residual and early stopping do not establish policy coverage or consistency'}
    return model, report


# Conditional bootstrap intervals

def bootstrap_mean_ci(initial_values, *, n_bootstrap: int = 1000, alpha: float = 0.05, seed: int = 42):
    """Episode bootstrap conditional on the fitted critic and fixed policy.

    This interval covers variation in the sampled initial episodes only. It
    does not include critic fitting uncertainty, policy uncertainty, or bias
    from poor action/state coverage. Repeated subjects need a subject-cluster
    bootstrap instead if subject-level independence is the intended target.
    """
    values = np.asarray(initial_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("initial_values must be a nonempty finite vector.")
    if n_bootstrap < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid bootstrap count or alpha.")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    # Limit temporary allocation independently of the cohort size.
    batch = max(1, min(128, 1_000_000 // len(values)))
    for start in range(0, n_bootstrap, batch):
        count = min(batch, n_bootstrap - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(values.mean()), "ci_lower": float(low), "ci_upper": float(high),
        "alpha": alpha, "n_bootstrap": n_bootstrap, "seed": seed,
        "bootstrap_unit": "initial episode",
        "uncertainty_scope": "conditional on frozen fitted critic and fixed policy; no refitting",
    }


def bootstrap_initial_values(values, *, groups=None, n_bootstrap=1000, seed=20261007, alpha=.05, return_samples=False):
    """Conditional sampling CI; preserve all episodes within a subject."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite initial episode values')
    if n_bootstrap < 0 or not 0 < alpha < 1:
        raise ValueError('Invalid bootstrap settings')
    groups = np.arange(len(values)) if groups is None else np.asarray(groups)
    if len(groups) != len(values):
        raise ValueError('One bootstrap group per episode is required')
    unique = np.unique(groups)
    grouped = [np.flatnonzero(groups == group) for group in unique]
    # A subject bootstrap can have a varying number of episodes; ratio of sums
    # retains the episode-weighted estimand used for all OPE point estimates.
    sums = np.array([values[rows].sum() for rows in grouped])
    counts = np.array([len(rows) for rows in grouped])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, len(unique), size=len(unique))
        samples.append(float(sums[chosen].sum() / counts[chosen].sum()))
    report = {'value': float(values.mean()), 'low': float(np.quantile(samples, alpha/2)) if samples else None,
              'high': float(np.quantile(samples, 1-alpha/2)) if samples else None,
              'n_bootstrap': n_bootstrap, 'seed': seed, 'independent_groups': len(unique),
              'uncertainty_scope': 'conditional on fitted critic and fixed target policy; subject clusters when available'}
    if return_samples:
        report['samples'] = samples
    return report


# Uncertainty scores and policy diagnostics

def variance_scores(q_outputs):
    """Return the ensemble variance of the maximum action Q-value for each state."""
    q_max = q_outputs.max(axis=2)                    # (M, N)
    return q_max.var(axis=0)


def entropy_scores(q_outputs, temperature=1.0):
    """Return the variance of per-policy entropy for each state.
    
    Compute softmax probabilities with the supplied temperature, calculate entropy for each policy, and take its variance across the ensemble. The output has shape (N,); larger values indicate greater disagreement among policies about entropy."""
    # q_outputs: (M, N, A)

    # Compute softmax probabilities and entropy for each policy using vectorized operations.
    scaled = q_outputs / temperature              # (M, N, A)

    m = scaled.max(axis=2, keepdims=True)         # (M, N, 1)
    exp_shifted = np.exp(scaled - m)              # (M, N, A)
    sum_exp = exp_shifted.sum(axis=2, keepdims=True)  # (M, N, 1)

    probs = exp_shifted / sum_exp                 # (M, N, A)
    log_probs = (scaled - m) - np.log(sum_exp)    # (M, N, A)

    # Entropy for each policy and state has shape (M, N).
    entropy_per_policy = -np.sum(probs * log_probs, axis=2)

    # ensemble variance over policies: (N,)
    entropy_var_scores = entropy_per_policy.var(axis=0)

    return entropy_var_scores


def energy_scores(q_outputs, temperature=1.0):
    """Return the variance of per-policy energy for each state.
    
    Energy_k(s) = -T * log(sum_a exp(Q_k(s,a) / T)).
    The output has shape (N,); larger values indicate greater disagreement among policies about energy."""
    # q_outputs: (M, N, A)
    scaled = q_outputs / temperature              # (M, N, A)

    m = scaled.max(axis=2, keepdims=True)         # (M, N, 1)
    exp_shifted = np.exp(scaled - m)              # (M, N, A)
    sum_exp = exp_shifted.sum(axis=2, keepdims=True)  # (M, N, 1)

    log_sum_exp = np.log(sum_exp) + m             # (M, N, 1)

    # Energy for each policy and state has shape (M, N).
    energy_per_policy = -temperature * log_sum_exp.squeeze(axis=2)

    # ensemble variance over policies: (N,)
    energy_var_scores = energy_per_policy.var(axis=0)

    return energy_var_scores


def policy_diagnostics(q, target, behavior, action_counts, support_probability=.01, min_count=100):
    q, target, behavior = map(lambda x: np.asarray(x,dtype=np.float64), [q,target,behavior])
    rare = (behavior < support_probability) | (np.asarray(action_counts)[None,:] < min_count)
    gap = np.sort(q,axis=1)[:,-1] - np.sort(q,axis=1)[:,-2]
    ratio = np.divide(target,behavior,out=np.zeros_like(target),where=behavior>0)
    return dict(q_saturation_fraction=float(np.mean(np.abs(q)>=.99)),
        q_min=float(q.min()),q_max=float(q.max()), q_best_runnerup_gap_median=float(np.median(gap)),
        q_near_tie_fraction=float(np.mean(gap<1e-4)),
        total_variation_mean=float(np.mean(.5*np.sum(np.abs(target-behavior),axis=1))),
        max_ratio_to_frozen_actor_baseline=float(ratio.max()),
        rare_action_probability_max_change=float(np.max(np.abs(target-behavior)[rare])) if rare.any() else 0.,
        target_positive_behavior_zero_count=int(np.sum((target>0)&(behavior==0))))


# OOD metrics

def report_synthetic_masks(labels, test_mask, ood_mask, name: str, quantile_thr):
    preds = np.concatenate(
        [test_mask.astype(int), ood_mask.astype(int)], axis=0
    )  # (2 * N_test,)

    correct = (preds == labels).sum()
    accuracy = correct / len(labels)

    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)

    print(f"\n=== [{name}] ID vs. Gaussian OOD (thr={quantile_thr:.3f}) ===")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1-score : {f1*100:.2f}%")


def evaluate_ood_scores(labels, variance, entropy, energy):
    var_scores_all, ent_scores_all, eng_scores_all = variance, entropy, energy
    def normalize(x):
        m = x.mean()
        s = x.std() + 1e-8
        return (x - m) / s

    var_n = normalize(var_scores_all)
    ent_n = normalize(ent_scores_all)
    eng_n = normalize(eng_scores_all)

    fpr_var, tpr_var, _ = roc_curve(labels, var_scores_all)
    fpr_ent, tpr_ent, _ = roc_curve(labels, ent_scores_all)
    fpr_eng, tpr_eng, _ = roc_curve(labels, eng_scores_all)

    auc_var = roc_auc_score(labels, var_scores_all)
    auc_ent = roc_auc_score(labels, ent_scores_all)
    auc_eng = roc_auc_score(labels, eng_scores_all)

    # Use normalized scores for union (maximum) and majority (second-largest).
    union_scores_all = np.maximum.reduce([var_n, ent_n, eng_n])
    fpr_union, tpr_union, _ = roc_curve(labels, union_scores_all)
    auc_union = roc_auc_score(labels, union_scores_all)

    scores_stack = np.stack([var_n, ent_n, eng_n], axis=0)
    scores_sorted = np.sort(scores_stack, axis=0)
    maj_scores_all = scores_sorted[1]
    fpr_maj, tpr_maj, _ = roc_curve(labels, maj_scores_all)
    auc_maj = roc_auc_score(labels, maj_scores_all)

    return {'var': dict(fpr=fpr_var, tpr=tpr_var, auc=auc_var), 'ent': dict(fpr=fpr_ent, tpr=tpr_ent, auc=auc_ent), 'eng': dict(fpr=fpr_eng, tpr=tpr_eng, auc=auc_eng), 'union': dict(fpr=fpr_union, tpr=tpr_union, auc=auc_union), 'maj': dict(fpr=fpr_maj, tpr=tpr_maj, auc=auc_maj)}


def report_external_masks(labels, test_mask, ood_mask, name: str, quantile_thr):
    preds = np.concatenate(
        [test_mask.astype(int), ood_mask.astype(int)], axis=0
    )  # (2 * N_test,)

    correct = (preds == labels).sum()
    accuracy = correct / len(labels)

    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)

    print(f"\n=== [{name}] ID vs. Gaussian OOD (thr={quantile_thr:.3f}) ===")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1-score : {f1*100:.2f}%")


def evaluate_external_ood_scores(labels, variance, entropy, energy):
    var_scores_all, ent_scores_all, eng_scores_all = variance, entropy, energy
    fpr_var, tpr_var, _ = roc_curve(labels, var_scores_all)
    fpr_ent, tpr_ent, _ = roc_curve(labels, ent_scores_all)
    fpr_eng, tpr_eng, _ = roc_curve(labels, eng_scores_all)

    auc_var = roc_auc_score(labels, var_scores_all)
    auc_ent = roc_auc_score(labels, ent_scores_all)
    auc_eng = roc_auc_score(labels, eng_scores_all)

    # ---------- Union score (max) ----------
    # Soft union uses the largest of the three scores as the ensemble score.
    union_scores_all = np.maximum.reduce(
        [var_scores_all, ent_scores_all, eng_scores_all]
    )

    fpr_union, tpr_union, _ = roc_curve(labels, union_scores_all)
    auc_union = roc_auc_score(labels, union_scores_all)

    # ---------- Majority vote score (2nd largest) ----------
    # The continuous majority score is the second-largest score.
    scores_stack = np.stack(
        [var_scores_all, ent_scores_all, eng_scores_all], axis=0
    )  # (3, 2N)
    scores_sorted = np.sort(scores_stack, axis=0)       # ascending: [min, mid, max]
    maj_scores_all = scores_sorted[1]                   # mid = 2nd largest

    fpr_maj, tpr_maj, _ = roc_curve(labels, maj_scores_all)
    auc_maj = roc_auc_score(labels, maj_scores_all)

    return {'var': dict(fpr=fpr_var, tpr=tpr_var, auc=auc_var), 'ent': dict(fpr=fpr_ent, tpr=tpr_ent, auc=auc_ent), 'eng': dict(fpr=fpr_eng, tpr=tpr_eng, auc=auc_eng), 'union': dict(fpr=fpr_union, tpr=tpr_union, auc=auc_union), 'maj': dict(fpr=fpr_maj, tpr=tpr_maj, auc=auc_maj)}


def report_clustered_masks(labels, test_mask, ood_mask, name: str, quantile_thr):
    preds = np.concatenate(
        [test_mask.astype(int), ood_mask.astype(int)], axis=0
    )  # (2 * N_test,)

    correct = (preds == labels).sum()
    accuracy = correct / len(labels)

    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)

    print(f"\n=== [{name}] ID vs. Gaussian OOD (thr={quantile_thr:.3f}) ===")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1-score : {f1*100:.2f}%")


def embedding_quality(embedding, labels):
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    return davies_bouldin_score(embedding, labels), silhouette_score(embedding, labels)


# Shared policy evaluation

def _full_behavior(buffer, probabilities):
    result = probabilities if probabilities is not None else getattr(buffer, 'bc_prob_full', None)
    if result is None:
        raise ValueError('Full train-only behavior probabilities are required; old BC_prob scalars are not sufficient')
    return np.asarray(result)


def _evaluate(algorithm, policy, replay_buffer, *, critic, behavior_probs=None, policy_mode='greedy',
              gamma=.98, clip=None, batch_size=1024, n_bootstrap=1000, seed=20261007,
              alpha=.05, episode_groups=None):
    from agent import frozen_policy_arrays
    if critic is None:
        raise ValueError('DR requires an independently fitted fixed-policy critic; use fit_fqe on training data')
    n = replay_buffer.crt_size
    target = frozen_policy_arrays(algorithm, policy, replay_buffer.state[:n], mode=policy_mode, batch_size=batch_size)
    next_target = frozen_policy_arrays(algorithm, policy, replay_buffer.next_state[:n], mode=policy_mode, batch_size=batch_size)
    return evaluate_ope(replay_buffer.action[:n], replay_buffer.reward[:n], replay_buffer.done[:n],
                        target, _full_behavior(replay_buffer, behavior_probs), predict_q(critic, replay_buffer.state[:n]),
                        gamma=gamma, next_target_probs=next_target,
                        next_q_values=predict_q(critic, replay_buffer.next_state[:n]), ratio_cap=clip,
                        batch_size=batch_size, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha,
                        episode_groups=episode_groups,
                        metadata={'target_policy': policy_mode, 'critic': 'independent bounded train-only FQE'})


def eval_multi_step_doubly_robust_ci(algorithm, policy, replay_buffer, clip=None, batch_size=1024,
                                    gamma=.98, device=None, n_bootstrap=1000, alpha=.05, *,
                                    critic=None, behavior_probs=None, policy_mode='greedy',
                                    seed=20261007, episode_groups=None, return_details=False):
    result = _evaluate(algorithm, policy, replay_buffer, critic=critic, behavior_probs=behavior_probs,
                       policy_mode=policy_mode, gamma=gamma, clip=clip, batch_size=batch_size,
                       n_bootstrap=n_bootstrap, seed=seed, alpha=alpha, episode_groups=episode_groups)
    ci = result['bootstrap']['intervals']['dr']
    return result if return_details else (result['dr'], ci['low'], ci['high'])


def eval_wis_ci(algorithm, policy, replay_buffer, clip=None, gamma=.98, device=None,
                n_bootstrap=1000, alpha=.05, *, behavior_probs=None, policy_mode='greedy',
                seed=20261007, episode_groups=None, return_details=False):
    from agent import frozen_policy_arrays
    n = replay_buffer.crt_size
    target = frozen_policy_arrays(algorithm, policy, replay_buffer.state[:n], mode=policy_mode)
    # WIS is independent of the critic; zero arrays satisfy the shared interface.
    result = evaluate_ope(replay_buffer.action[:n], replay_buffer.reward[:n], replay_buffer.done[:n],
                          target, _full_behavior(replay_buffer, behavior_probs), np.zeros_like(target),
                          gamma=gamma, ratio_cap=clip, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha,
                          episode_groups=episode_groups, metadata={'target_policy': policy_mode})
    ci = result['bootstrap']['intervals']['wis']
    return result if return_details else (result['wis'], ci['low'], ci['high'])


def _fit_policy_critic(arrays, next_probs, groups, *, reward_type, gamma, seed,
                       epochs, batch_size=None, lr=1e-3):
    if reward_type == 'per_step':
        return fit_heparin_fqe(arrays, next_probs, groups, gamma=gamma, seed=seed,
                              epochs=epochs, min_epochs=min(50, epochs), patience=20,
                              batch_size=batch_size or 2048, lr=lr)
    if reward_type != 'terminal':
        raise ValueError('reward_type must be terminal or per_step')
    return fit_fqe(arrays['state'], arrays['next_state'], arrays['action'],
                   arrays['reward'], arrays['done'], next_probs,
                   gamma=gamma, seed=seed, groups=groups, epochs=epochs,
                   min_epochs=min(30, epochs), patience=20,
                   batch_size=batch_size or 4096, lr=lr)


def eval_fqe_ci(algorithm, policy, replay_buffer, gamma=.98, device=None, num_epochs=100,
                batch_size=None, lr=1e-3, target_update_freq=1, tau=1., n_bootstrap=1000,
                alpha=.05, *, fit_buffer=None, policy_mode='greedy', seed=42,
                train_groups=None, episode_groups=None, return_details=False, reward_type="terminal"):
    from agent import frozen_policy_arrays
    if fit_buffer is None or fit_buffer is replay_buffer:
        raise ValueError('An independent training buffer is required for fixed-policy FQE')
    if target_update_freq != 1 or tau != 1.:
        raise ValueError('Corrected FQE uses one frozen target per fitted epoch')
    if train_groups is not None and episode_groups is not None and np.intersect1d(train_groups, episode_groups).size:
        raise ValueError('Test subjects remain in FQE training data')
    nt = fit_buffer.crt_size
    fixed_next = frozen_policy_arrays(algorithm, policy, fit_buffer.next_state[:nt], mode=policy_mode)
    arrays = {name: getattr(fit_buffer, name)[:nt] for name in ['state', 'next_state', 'action', 'reward', 'done']}
    critic, fit_report = _fit_policy_critic(arrays, fixed_next, train_groups,
                                          reward_type=reward_type, gamma=gamma, epochs=num_epochs,
                                          batch_size=batch_size, lr=lr, seed=seed)
    n = replay_buffer.crt_size
    target = frozen_policy_arrays(algorithm, policy, replay_buffer.state[:n], mode=policy_mode)
    initial = evaluate_initial_states(critic, replay_buffer.state[:n], replay_buffer.done[:n], target)
    result = bootstrap_initial_values(initial['initial_values'], groups=episode_groups,
                                       n_bootstrap=n_bootstrap, seed=seed, alpha=alpha)
    if return_details:
        return {'fqe': result, 'fit': fit_report, 'critic': critic}
    return result['value'], result['low'], result['high']


def evaluate_policy_arrays(arrays, target_probs, behavior_probs, q_values, *,
                           next_target_probs=None, next_q_values=None, gamma=.98,
                           ratio_cap=None, cumulative_weight_cap=None, n_bootstrap=1000, seed=20261007,
                           alpha=.05, episode_groups=None, metadata=None,
                           return_bootstrap_samples=False, return_fqe_samples=False):
    """Evaluate DR/WDR/WIS/FQE on the same fixed arrays, with no fitting."""
    from util import initial_state_indices
    starts = initial_state_indices(arrays['done'])
    report = evaluate_ope(arrays['action'], arrays['reward'], arrays['done'],
                          target_probs, behavior_probs, q_values, gamma=gamma,
                          next_target_probs=next_target_probs, next_q_values=next_q_values,
                          ratio_cap=ratio_cap, cumulative_weight_cap=cumulative_weight_cap,
                          n_bootstrap=n_bootstrap, seed=seed,
                          alpha=alpha, episode_groups=episode_groups, metadata=metadata,
                          return_bootstrap_samples=return_bootstrap_samples)
    initial = (np.asarray(target_probs)[starts] * np.asarray(q_values)[starts]).sum(1)
    report['fqe'] = bootstrap_initial_values(initial, groups=episode_groups,
                                            n_bootstrap=n_bootstrap, seed=seed, alpha=alpha, return_samples=return_fqe_samples)
    return report


def evaluate_policy(algorithm, policy, train_buffer, test_buffer, *, reward_type,
                    train_groups=None, test_groups=None, policy_mode='greedy',
                    gamma=.98, seed=42, behavior_seed=None, fqe_epochs=100,
                    n_bootstrap=1000, output_dir=None):
    """Fit train-only nuisances and evaluate one fixed actor on every episode."""
    from agent import frozen_policy_arrays, pred_q_value
    if train_buffer is test_buffer:
        raise ValueError('An independent training buffer is required; train and test must be different buffers')
    if reward_type not in {'terminal', 'per_step'}:
        raise ValueError('reward_type must be terminal or per_step')
    from pathlib import Path
    import json
    import hashlib
    from agent import fit_behavior
    from model import full_action_proba
    from util import episode_slices
    nt, ne = train_buffer.crt_size, test_buffer.crt_size
    episode_slices(train_buffer.done[:nt]); episode_slices(test_buffer.done[:ne])
    if (train_groups is None) != (test_groups is None):
        raise ValueError('Provide both training and test patient groups, or neither')
    for groups, buffer in [(train_groups, train_buffer), (test_groups, test_buffer)]:
        if groups is not None:
            if np.asarray(groups).shape != (buffer.crt_size,):
                raise ValueError('One patient group per transition is required')
            for rows in episode_slices(buffer.done[:buffer.crt_size]):
                if len(np.unique(np.asarray(groups)[rows])) != 1:
                    raise ValueError('Each complete episode must belong to one patient')
    if train_groups is not None and np.intersect1d(train_groups, test_groups).size:
        raise ValueError('Test subjects remain in nuisance training data')
    if output_dir is not None:
        path = Path(output_dir)
        if any((path/name).exists() for name in ['evaluation.json', 'behavior.joblib', 'fqe.pt']):
            raise FileExistsError('Use a fresh evaluation output directory')
    policy.Q.eval()
    device = next(policy.Q.parameters()).device
    with torch.inference_mode():
        probe = torch.as_tensor(np.asarray(test_buffer.state[:1]), dtype=torch.float32, device=device)
        num_actions = pred_q_value(algorithm, policy, probe).shape[1]
    per_step = reward_type == 'per_step'
    behavior_seed = (53 if per_step else seed) if behavior_seed is None else behavior_seed
    raw, calibrated, behavior_report = fit_behavior(
        train_buffer.state[:nt], train_buffer.action[:nt], train_buffer.done[:nt],
        groups=train_groups, random_seed=behavior_seed, num_actions=num_actions,
        selection='calibrated' if per_step else 'validation', require_all_actions=per_step)
    behavior_report.pop('_partition_indices', None)
    behavior_model = calibrated if behavior_report['selected_behavior'] == 'calibrated' else raw
    behavior = full_action_proba(behavior_model, test_buffer.state[:ne], num_actions=num_actions)
    if policy_mode == 'behavior_anchor':
        def probabilities(state):
            qparts = []
            with torch.inference_mode():
                for start in range(0, len(state), 1024):
                    batch = torch.as_tensor(state[start:start+1024], dtype=torch.float32, device=device)
                    qparts.append(pred_q_value(algorithm, policy, batch).cpu().numpy())
            return anchored_policy_probs(np.concatenate(qparts),
                full_action_proba(behavior_model, state, num_actions=num_actions), strength=1.)
    else:
        def probabilities(state):
            return frozen_policy_arrays(algorithm, policy, state, mode=policy_mode)
    next_train = probabilities(train_buffer.next_state[:nt])
    target, next_target = probabilities(test_buffer.state[:ne]), probabilities(test_buffer.next_state[:ne])
    arrays = {name: getattr(train_buffer, name)[:nt] for name in ['state', 'next_state', 'action', 'reward', 'done']}
    critic, fit_report = _fit_policy_critic(arrays, next_train, train_groups,
                                          reward_type=reward_type, gamma=gamma, seed=seed, epochs=fqe_epochs)
    qhat = predict_q(critic, test_buffer.state[:ne], num_threads=1 if per_step else 4)
    next_qhat = predict_q(critic, test_buffer.next_state[:ne], num_threads=1 if per_step else 4)
    starts = np.r_[0, np.flatnonzero(np.asarray(test_buffer.done[:ne]).reshape(-1) == 1)[:-1] + 1]
    episode_groups = None if test_groups is None else np.asarray(test_groups)[starts]
    test_arrays = {name: getattr(test_buffer, name)[:ne] for name in arrays}
    report = evaluate_policy_arrays(test_arrays, target, behavior, qhat,
        gamma=gamma, next_target_probs=next_target, next_q_values=next_qhat,
        n_bootstrap=n_bootstrap, seed=seed, episode_groups=episode_groups,
        metadata={'target_policy': policy_mode, 'reward_type': reward_type,
                  'propensity': 'train-only with separate fit/calibration/validation',
                  'critic': 'independent fixed-policy bounded FQE',
                  'checkpoint': 'one final checkpoint for every estimator'})
    report['behavior_fit'], report['fqe_fit'] = behavior_report, fit_report
    report['critic_value_bound'] = float(1/(1-gamma)) if per_step else 1.
    report['uncertainty_refit_note'] = 'Intervals hold actor, behavior and critic fixed; they omit nuisance-fitting uncertainty.'
    if output_dir is not None:
        import joblib
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(behavior_model, path/'behavior.joblib', compress=3)
        np.save(path/'behavior_test_all_actions.npy', behavior)
        torch.save({'state_dict': critic.state_dict(), 'state_dim': critic.state_dim,
                    'num_actions': critic.num_actions, 'hidden_dim': critic.hidden_dim,
                    'reward_type': reward_type, 'value_bound': report['critic_value_bound']}, path/'fqe.pt')
        from util import source_hashes
        from util import project_root
        report['source_sha256'] = source_hashes(project_root())
        (path/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def evaluate_sepsis_policy(algorithm, policy, train_buffer, test_buffer, **kwargs):
    """Terminal-only ±1 reward protocol, including train-only RF selection."""
    return evaluate_policy(algorithm, policy, train_buffer, test_buffer, reward_type='terminal', **kwargs)


def evaluate_heparin_policy(algorithm, policy, train_buffer, test_buffer, **kwargs):
    """Per-transition reward protocol and fixed calibrated six-action behavior."""
    return evaluate_policy(algorithm, policy, train_buffer, test_buffer, reward_type='per_step', **kwargs)


# WDR interface

def _report(result):
    interval = result["bootstrap"]["intervals"]["wdr"]
    return {
        "estimator": "WDR",
        "value": result["wdr"],
        "low": interval["low"],
        "high": interval["high"],
        "interval": interval,
        "gamma": result["gamma"],
        "ratio_cap": result["ratio_cap"],
        "cumulative_weight_cap": result["cumulative_weight_cap"],
        "analysis_kind": result["analysis_kind"],
        "episodes_used": result["episodes_used"],
        "episodes_excluded": result["episodes_excluded"],
        "numerical_status": result["numerical_status"]["wdr"],
        "undefined_time_steps": result["numerical_status"]["undefined_wdr_time_steps"],
        "diagnostics": result,
    }


def evaluate_wdr(actions, rewards, dones, target_probs, behavior_probs, q_values,
                 *, gamma=.98, ratio_cap=None, cumulative_weight_cap=None, n_bootstrap=1000,
                 seed=20261007, alpha=.05, episode_groups=None,
                 return_bootstrap_samples=False, metadata=None):
    """Evaluate already frozen arrays; no policy or nuisance-model fitting.

    Probability and critic arrays cover all actions at every recorded state.
    Caller establishes independent fitting/cross-fitting and row alignment.
    ``episode_groups`` supplies one patient identifier per complete episode.
    The interval holds the policy and nuisance models fixed; undefined draws
    remain counted and finite-only percentiles are explicitly conditional.
    """
    return _report(evaluate_ope(
        actions, rewards, dones, target_probs, behavior_probs, q_values,
        gamma=gamma, ratio_cap=ratio_cap, cumulative_weight_cap=cumulative_weight_cap,
        n_bootstrap=n_bootstrap, seed=seed,
        alpha=alpha, episode_groups=episode_groups,
        return_bootstrap_samples=return_bootstrap_samples, metadata=metadata,
    ))


def eval_wdr_ci(algorithm, policy, replay_buffer, *, critic,
                behavior_probs=None, policy_mode="greedy", gamma=.98,
                clip=None, batch_size=1024, n_bootstrap=1000,
                seed=20261007, alpha=.05, episode_groups=None,
                return_details=False):
    """Fixed-policy buffer interface using an independently fit critic.

    Inference batch size never changes trajectory normalization.  Greedy
    freezes the actual repository action rule; softmax is a separate actor.
    Returns (WDR, low, high), or the full WDR report when requested.
    """

    report = _report(_evaluate(
        algorithm, policy, replay_buffer, critic=critic,
        behavior_probs=behavior_probs, policy_mode=policy_mode,
        gamma=gamma, clip=clip, batch_size=batch_size,
        n_bootstrap=n_bootstrap, seed=seed, alpha=alpha,
        episode_groups=episode_groups,
    ))
    if return_details:
        return report
    return report["value"], report["low"], report["high"]
