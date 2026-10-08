"""Check fitted Platt sigmoids with an independent finite-gradient optimizer.

This is a read-only model audit prompted by matmul RuntimeWarnings in the local
sklearn/NumPy backend. It never overwrites or refits the saved BC models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from gym_test.verify_bc_results import independent_features
from gym_test.verify_results import load_npz, write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    protocol = json.loads((run / "protocol.json").read_text())
    baseline = Path(protocol["baseline_dir"])
    reports = {}
    for seed in protocol["seeds"]:
        directory = run / f"seed_{seed}/behavior"
        train = load_npz(baseline / f"seed_{seed}/datasets/actor_train.npz")
        rows = load_npz(directory / "partition.npz")["calibration"]
        state = independent_features(train, protocol["horizon"])[rows]
        labels = train["action"][rows]
        raw = joblib.load(directory / "raw.joblib")
        fitted = joblib.load(directory / "calibrated.joblib").calibrated_classifiers_[0]
        inputs = raw.predict_proba(state)
        seed_reports = []
        for action, sigmoid in enumerate(fitted.calibrators):
            feature = inputs[:, action]
            positive = labels == action
            prior1 = int(np.sum(positive))
            prior0 = len(labels) - prior1
            target = np.where(positive, (prior1 + 1) / (prior1 + 2), 1 / (prior0 + 2))

            def loss_gradient(parameters):
                logits = -(parameters[0] * feature + parameters[1])
                loss = np.sum(np.logaddexp(0, logits) - target * logits)
                residual = expit(logits) - target
                # Explicit sums avoid the matmul operation that emitted warnings.
                gradient = np.array([-np.sum(residual * feature), -np.sum(residual)])
                assert np.isfinite(loss) and np.isfinite(gradient).all()
                return loss, gradient

            initial = np.array([0, np.log((prior0 + 1) / (prior1 + 1))])
            oracle = minimize(loss_gradient, initial, method="L-BFGS-B", jac=True,
                              options={"gtol": 1e-6, "ftol": 64 * np.finfo(float).eps})
            assert oracle.success, oracle.message
            saved = np.array([sigmoid.a_, sigmoid.b_])
            saved_loss, saved_gradient = loss_gradient(saved)
            prediction_difference = float(np.max(np.abs(expit(-(saved[0] * feature + saved[1]))
                                                        - expit(-(oracle.x[0] * feature + oracle.x[1])))))
            assert abs(saved_loss - float(oracle.fun)) < 1e-6
            assert prediction_difference < 1e-5
            seed_reports.append({"action": action, "saved_parameters": saved.tolist(),
                                 "independent_parameters": oracle.x.tolist(),
                                 "independent_optimizer_success": bool(oracle.success),
                                 "independent_optimizer_message": str(oracle.message),
                                 "saved_gradient_max_abs": float(np.max(np.abs(saved_gradient))),
                                 "loss_difference": float(saved_loss - oracle.fun),
                                 "maximum_calibration_prediction_difference": prediction_difference})
        reports[f"seed_{seed}"] = seed_reports
        print(f"seed={seed}: all four fitted calibration sigmoids match independent optimizer", flush=True)
    write_new_json(run / "calibration_audit.json", {
        "status": "passed", "saved_models_modified": False,
        "trigger": "sklearn calibration.py gradient matmul RuntimeWarnings during original fit",
        "check": "Same Platt-smoothed objective, independently implemented stable logistic loss and explicit-sum gradient",
        "result": "All saved sigmoid coefficients are finite; predictions agree with independently converged fits within 1e-5",
        "seeds": reports})


if __name__ == "__main__":
    main()
