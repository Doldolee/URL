"""Recompute paper WDR for five unchanged original greedy CQL checkpoints.

Uses frozen train-only RF probabilities and policy-specific FQE predictions
from the corrected archive.  No fitting, policy selection, or archive writes.
Independent 80-digit Decimal calculations check all five complete cohorts.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

STAGED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGED_ROOT))
sys.path.insert(0, str(STAGED_ROOT / "tests"))

from util import verify_source_manifest
import numpy as np
from metric import policy_probs
from metric import evaluate_wdr
from wdr_reference import reference_wdr


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=STAGED_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir.resolve()
    archive = root / "outputs/sepsis_ope_corrected_20261007"
    if out.exists():
        raise FileExistsError("Use a new output directory; completed artifacts are not overwritten")
    out.mkdir(parents=True)
    manifest = json.loads((archive / "behavior/manifest.json").read_text())
    paths = [root / "dataset/sepsis" / ("test_" + field + ".npy")
             for field in ["state", "next_state", "action", "reward", "done"]]
    paths += [archive / "behavior/manifest.json", archive / "behavior" / manifest["primary_behavior"],
              archive / "behavior/test_groups.npy", archive / "behavior/train_groups.npy",
              archive / "behavior/train_keep_mask.npy", archive / "unseen_test_mask.npy",
              archive / "completion_receipt.json",
              root / "outputs/sepsis_policy_retrain_20261007/completion_receipt.json"]
    paths += [root / name for name in ['agent.py', 'metric.py', 'metric.py', 'metric.py', 'agent.py', 'agent.py', 'agent.py', 'agent.py', 'agent.py', 'metric.py', 'metric.py', 'metric.py', 'model.py', 'util.py', 'util.py']]
    for i in range(5):
        directory = archive / "models" / (f"sepsis_CQL_{i}_greedy_seed42")
        paths += [root / "pth" / f"sepsis_CQL_{i}.pth",
                  root / "outputs/sepsis_dr_audit_20261007" / f"sepsis_CQL_{i}_q.npz",
                  directory / "critic_test_q.npz", directory / "fqe.pt",
                  directory / "fit.json", directory / "evaluation.json"]
    before = {str(path.relative_to(root)): sha(path) for path in paths}
    # The archive contains old mount paths. Verify the corresponding current
    # project files by content, without editing archived path metadata.
    for field, info in manifest["inputs"]["test"].items():
        relative = f"dataset/sepsis/test_{field}.npy"
        if before[relative] != info["sha256"]:
            raise ValueError(f"Input changed since behavior fitting: {relative}")
    # Original artifact checks remain above; current source hashes have a
    # migration manifest linked to the preserved pre-refactor receipts.
    verify_source_manifest(STAGED_ROOT)
    action = np.load(root / "dataset/sepsis/test_action.npy").reshape(-1)
    reward = np.load(root / "dataset/sepsis/test_reward.npy").reshape(-1)
    done = np.load(root / "dataset/sepsis/test_done.npy").reshape(-1)
    groups = np.load(archive / "behavior/test_groups.npy")
    mask = np.load(archive / "unseen_test_mask.npy")
    original_train_groups = np.load(archive / "behavior/train_groups.npy")
    keep = np.load(archive / "behavior/train_keep_mask.npy")
    if np.intersect1d(original_train_groups[keep], groups).size:
        raise ValueError("Nuisance train/test patients overlap")
    if not np.array_equal(mask, ~np.isin(groups, np.unique(original_train_groups))):
        raise ValueError("Primary-cohort mask changed")
    starts = np.r_[0, np.flatnonzero(done == 1)[:-1] + 1]
    ends = np.flatnonzero(done == 1)
    for begin, end in zip(starts, ends):
        if not np.all(mask[begin:end + 1] == mask[begin]):
            raise ValueError("Cohort mask splits an episode")
        if not np.all(groups[begin:end + 1] == groups[begin]):
            raise ValueError("Patient label changes within an episode")
    b = np.load(archive / "behavior" / manifest["primary_behavior"])[mask]
    action, reward, done, groups = action[mask], reward[mask], done[mask], groups[mask]
    starts = np.r_[0, np.flatnonzero(done == 1)[:-1] + 1]
    episode_groups = groups[starts]
    if not np.all(reward[done == 0] == 0) or not np.all(np.isin(reward[done == 1], [-1, 1])):
        raise ValueError("Expected one terminal +/-1 reward per complete episode")
    rows, traces, max_error = [], [], 0.
    for i in range(5):
        name = f"sepsis_CQL_{i}"
        directory = archive / "models" / (name + "_greedy_seed42")
        old = json.loads((directory / "evaluation.json").read_text())
        if sha(root / "pth" / (name + ".pth")) != old["checkpoint_sha256"]:
            raise ValueError("Original CQL checkpoint changed")
        with np.load(root / "outputs/sepsis_dr_audit_20261007" / (name + "_q.npz")) as cache:
            target = policy_probs(cache["q"][mask], mode="greedy")
        with np.load(directory / "critic_test_q.npz") as cache:
            qhat = cache["q"][mask]
        print("WDR evaluation", name, flush=True)
        report = evaluate_wdr(
            action, reward, done, target, b, qhat, gamma=.98,
            n_bootstrap=args.n_bootstrap, seed=20261007,
            episode_groups=episode_groups, return_bootstrap_samples=True,
            metadata={"target_policy": "unchanged original CQL greedy",
                      "nuisance_models": "frozen train-only calibrated RF and policy-specific FQE",
                      "cohort": "1322 complete episodes absent from current stored train patients"},
        )
        print("Independent Decimal equation", name, flush=True)
        reference = reference_wdr(action, reward, done, target, b, qhat, gamma=.98)
        error = abs(report["value"] - reference["value"])
        max_error = max(max_error, error)
        np.testing.assert_allclose(report["value"], reference["value"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(report["value"], old["unseen_subjects"]["wdr"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            report["diagnostics"]["weights"]["per_decision_ess_with_absorbing_padding"],
            [column["ess"] for column in reference["columns"]], rtol=1e-10, atol=1e-10)
        report["reference"] = {"method": "independent 80-digit Decimal direct-ratio equation",
                               "value": reference["value"], "absolute_error": error,
                               "old_archive_wdr": old["unseen_subjects"]["wdr"]}
        save_json(out / (name + ".json"), report)
        weights = report["diagnostics"]["weights"]
        interval = report["interval"]
        rows.append({"model": name, "policy": "original_greedy", "episodes": len(starts),
                     "subjects": len(np.unique(episode_groups)), "gamma": .98,
                     "ratio_cap": "none", "wdr": report["value"],
                     "conditional_low": report["low"], "conditional_high": report["high"],
                     "valid_bootstrap": interval["defined_finite_resamples"],
                     "planned_bootstrap": args.n_bootstrap,
                     "terminal_weight_ess": weights["trajectory_ess"],
                     "nonzero_terminal_paths": len(starts) - weights["exact_zero_trajectory_weight_count"],
                     "max_terminal_normalized_weight": weights["maximum_normalized_trajectory_weight"],
                     "decimal_absolute_error": error})
        traces.extend({"model": name, **column} for column in reference["columns"])
        print(json.dumps(rows[-1]), flush=True)
    for filename, values in [("metrics.csv", rows), ("time_step_diagnostics.csv", traces)]:
        with (out / filename).open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    after = {str(path.relative_to(root)): sha(path) for path in paths}
    if after != before:
        raise ValueError("A pre-existing input/source/archive artifact changed during evaluation")
    source_paths = [STAGED_ROOT / "metric.py", Path(__file__).resolve(),
                    STAGED_ROOT / "tests/wdr_reference.py", STAGED_ROOT / "tests/test_rl_wdr.py"]
    save_json(out / "input_preservation.json", {"status": "unchanged", "files": before,
               "scope": "All files read by this runner, including original policy/critic checkpoints and prior receipts"})
    save_json(out / "completion_receipt.json", {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Faithful WDR test of five unchanged original greedy CQL policies",
        "paper": "https://proceedings.mlr.press/v48/thomasa16.pdf",
        "implementation": "equation (1), section 5; time normalization; w_-1=1/N; absorbing padding",
        "policy_retrained": False, "behavior_refitted": False, "fqe_refitted": False,
        "gamma": .98, "ratio_cap": None, "probability_floor": None, "output_clipping": None,
        "episodes": len(starts), "subjects": len(np.unique(episode_groups)),
        "bootstrap": {"unit": "patient cluster retaining whole episodes", "planned_per_policy": args.n_bootstrap,
                      "nuisance_models_held_fixed": True, "intervals_condition_on_defined_draws": True},
        "decimal_reference_cases": 5, "max_absolute_equation_error": max_error,
        "previous_inputs_and_sources_preserved": before == after,
        "historical_checkpoint_training_membership_verified": False,
        "validated_policy_value": False,
        "new_source_sha256": {str(path.relative_to(STAGED_ROOT)): sha(path) for path in source_paths},
        "artifact_sha256": {path.name: sha(path) for path in out.iterdir() if path.is_file()},
    })
    table = "\n".join(
        f"| {row['model']} | {row['wdr']:.6f} | [{row['conditional_low']:.6f}, {row['conditional_high']:.6f}] | "
        f"{row['valid_bootstrap']}/{args.n_bootstrap} | {row['terminal_weight_ess']:.6f} | {row['nonzero_terminal_paths']} |"
        for row in rows)
    (out / "README.md").write_text(
        "# CQL WDR 논문식 검증 및 재평가\n\n"
        "Thomas & Brunskill (ICML 2016)의 식 (1), section 5를 사용했습니다. 기존 metric 엔진의 "
        "WDR 계산은 이미 이 식을 구현하고 있었으며, 이번에는 metric 전용 인터페이스와 독립 Decimal 기준 계산을 추가했습니다.\n\n"
        "평가 대상은 기존 CQL checkpoint 5개의 원 greedy 정책입니다. 새 제약 정책을 평가한 결과가 아닙니다. "
        "정책·behavior·FQE를 새로 학습하지 않고, 보존된 확률 및 Q cache에서 WDR와 환자 bootstrap을 실제 재계산했습니다. "
        "gamma=.98, clipping/floor 없음, 모든 complete episode 사용, 종료 후 absorbing-state 분모 유지입니다.\n\n"
        "| 정책 | WDR | 고정 nuisance 조건부 percentile | 유효 bootstrap | 최종 weight ESS | 최종 양의 weight 경로 |\n"
        "|---|---:|---|---:|---:|---:|\n" + table + "\n\n"
        f"5개 전체 cohort에서 독립 80자리 Decimal 식과의 최대 절대오차는 {max_error:.3e}입니다. "
        "time_step_diagnostics.csv에는 reward/Q/이전 시점 V 항, 시점별 ESS, 가중치가 남은 경로 수를 저장했습니다.\n\n"
        "이 구간은 정책과 nuisance 모델을 고정한 평가 환자 bootstrap이고, 계산 가능한 draw에 조건부입니다. "
        "Undefined draw를 0으로 대체하지 않습니다. 새 nuisance 재추정 구간이 아니며, 기존 corrected archive의 "
        "nuisance-refit 결과는 그대로 남아 있습니다. ESS는 최종 누적 가중치의 진단이지 DR/WDR 전체의 유효 표본 수가 아닙니다. "
        "낮은 ESS와 과거 checkpoint의 학습 membership 미확인 때문에 검증된 정책 가치 또는 임상 개선으로 해석하지 않습니다. "
        "WDR 출력도 [-1,1] 밖으로 나갈 수 있으며 강제로 자르지 않습니다.\n\n"
        "원 metric/metric 소스와 이번에 읽은 기존 데이터·checkpoint·cache·receipt의 SHA256은 실행 전후 동일했습니다.\n"
    )
    print("complete", str(out), "max Decimal error", max_error, flush=True)


if __name__ == "__main__":
    main()
