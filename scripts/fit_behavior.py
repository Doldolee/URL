"""Fit train-only calibrated behavior probabilities in a fresh output directory."""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import joblib
import numpy as np

from util import ReplayBuffer
from util import prepare_subject_split, reward_type
from util import project_root
from agent import fit_behavior
from model import full_action_proba


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['heparin', 'sepsis'], default='heparin')
    parser.add_argument('--project-root', type=Path, default=project_root())
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--cohort-csv', type=Path)
    parser.add_argument('--demog-csv', type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh output directory; dataset and existing models are read-only')
    root = args.project_root.resolve()
    if args.output_dir.resolve().is_relative_to(root/'dataset') or args.output_dir.resolve().is_relative_to(root/'pth'):
        raise ValueError('Behavior outputs must be outside dataset and pth')
    state_dim, num_actions = (16, 6) if args.dataset == 'heparin' else (43, 25)
    train = ReplayBuffer(state_dim, 64, args.dataset, root/'dataset'/args.dataset, device='cpu').load_data()
    test = ReplayBuffer(state_dim, 64, args.dataset, root/'dataset'/args.dataset, device='cpu').load_data(only_test_set=True)
    groups, _, mapping = prepare_subject_split(args.dataset, train, test, root, cohort_csv=args.cohort_csv, demog_csv=args.demog_csv)
    per_step = reward_type(args.dataset) == 'per_step'
    seed = (53 if per_step else 42) if args.seed is None else args.seed
    raw, calibrated, report = fit_behavior(train.state, train.action, train.done, groups=groups,
        random_seed=seed, num_actions=num_actions,
        selection='calibrated' if per_step else 'validation', require_all_actions=per_step)
    report.pop('_partition_indices')
    selected = calibrated if report['selected_behavior'] == 'calibrated' else raw
    probability = full_action_proba(selected, test.state, num_actions=num_actions)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(selected, args.output_dir/'behavior.joblib', compress=3)
    np.save(args.output_dir/'test_all_action_probs.npy', probability)
    (args.output_dir/'fit.json').write_text(json.dumps({'fit': report, 'subject_mapping': mapping}, indent=2, allow_nan=False))
    print(json.dumps({'output_dir': str(args.output_dir), 'selected_behavior': report['selected_behavior']}))


if __name__ == '__main__':
    main()
