"""Independent, slow Decimal transcription of Thomas & Brunskill's WDR.

This oracle imports no production estimator or numerical weight helper.  It
multiplies ratios directly and evaluates each time-column of equation (1).
It is for verification, not clinical fitting or production bootstrap.
"""
from decimal import Decimal, localcontext


def reference_wdr(actions, rewards, dones, target_probs, behavior_probs,
                  q_values, *, gamma=.98, ratio_cap=None, precision=80):
    def dec(value):
        return Decimal(str(float(value)))

    with localcontext() as ctx:
        ctx.prec = precision
        episodes, start = [], 0
        for end, done in enumerate(dones):
            if done:
                episodes.append((start, end + 1))
                start = end + 1
        if not episodes or start != len(actions):
            raise ValueError("Expected complete episodes")
        count = len(episodes)
        horizon = max(end - begin for begin, end in episodes)
        cumulative = [Decimal(1)] * count
        previous = [Decimal(1) / Decimal(count)] * count
        discount, total = Decimal(1), Decimal(0)
        columns = []
        for t in range(horizon):
            current_r, current_q, current_v = [], [], []
            for i, (begin, end) in enumerate(episodes):
                row = begin + t
                if row >= end:
                    # Absorbing state: do not update the final cumulative ratio.
                    current_r.append(Decimal(0))
                    current_q.append(Decimal(0))
                    current_v.append(Decimal(0))
                    continue
                action = int(actions[row])
                denominator = dec(behavior_probs[row][action])
                if denominator <= 0:
                    raise ValueError("Recorded behavior probability must be positive")
                ratio = dec(target_probs[row][action]) / denominator
                if ratio_cap is not None:
                    ratio = min(ratio, dec(ratio_cap))
                cumulative[i] *= ratio
                current_r.append(dec(rewards[row]))
                current_q.append(dec(q_values[row][action]))
                current_v.append(sum(
                    (dec(p) * dec(q) for p, q in zip(target_probs[row], q_values[row])),
                    Decimal(0),
                ))
            denominator = sum(cumulative, Decimal(0))
            if denominator == 0:
                return {"value": None, "undefined_time_step": t, "columns": columns}
            weights = [value / denominator for value in cumulative]
            reward_term = sum((w * r for w, r in zip(weights, current_r)), Decimal(0))
            q_term = sum((w * q for w, q in zip(weights, current_q)), Decimal(0))
            v_term = sum((w * v for w, v in zip(previous, current_v)), Decimal(0))
            contribution = discount * (reward_term - q_term + v_term)
            total += contribution
            ess = Decimal(1) / sum((w * w for w in weights), Decimal(0))
            columns.append({
                "time_step": t, "discount": float(discount),
                "reward_term": float(reward_term), "q_term": float(q_term),
                "previous_weight_v_term": float(v_term),
                "discounted_contribution": float(contribution),
                "nonzero_cumulative_weights": sum(value > 0 for value in cumulative),
                "ess": float(ess), "max_normalized_weight": float(max(weights)),
            })
            previous = weights
            discount *= dec(gamma)
        return {"value": float(total), "undefined_time_step": None, "columns": columns}
