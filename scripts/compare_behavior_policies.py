"""Compare train-only behavior estimators for unchanged MIMIC-III softmax actors.

Select one shared denominator per dataset by mean trajectory ESS across the
20 matched actors on the original behavior-validation patients. Test outcomes,
test actions and test ESS do not enter selection. This exploratory ESS criterion
does not establish that the selected estimator is the true clinician policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import POLICIES, HeparinBCQ, frozen_policy_arrays
from metric import evaluate_policy_arrays, probability_metrics
from model import full_action_proba
from util import episode_slices, grouped_partition, initial_state_indices, load_behavior_model

PREVIOUS = ROOT / 'outputs/mimic3_four_algorithm_softmax_ope_20261007'
FIELDS = ['state', 'next_state', 'action', 'reward', 'done']
ALGORITHMS = ['DQN', 'DDQN', 'BCQ', 'CQL']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trajectory_weight_diagnostics(action, done, target, behavior):
    """Independent, log-space terminal-weight diagnostic; no clipping or floor."""
    action = np.asarray(action).reshape(-1).astype(int)
    target = np.asarray(target, dtype=float)
    behavior = np.asarray(behavior, dtype=float)
    if target.shape != behavior.shape or len(action) != len(target):
        raise ValueError('Unaligned action probabilities')
    for value in [target, behavior]:
        if not np.isfinite(value).all() or (value < 0).any() or not np.allclose(value.sum(1), 1.):
            raise ValueError('Invalid probabilities')
    selected = np.arange(len(action))
    pa, ba = target[selected, action], behavior[selected, action]
    zero = int(np.sum(ba == 0))
    if zero:
        return {'trajectory_ess': None, 'observed_behavior_zero_count': zero,
                'status': 'undefined_recorded_action_probability_zero'}
    with np.errstate(divide='ignore'):
        logs = np.log(pa) - np.log(ba)
    slices = episode_slices(done)
    terminal = np.array([logs[sl].sum() for sl in slices])
    if np.isneginf(terminal).all():
        return {'trajectory_ess': None, 'observed_behavior_zero_count': 0,
                'status': 'undefined_all_trajectory_weights_zero'}
    weights = np.exp(terminal - logsumexp(terminal))
    ess = float(1. / np.dot(weights, weights))
    return {'trajectory_ess': ess, 'trajectory_ess_fraction': ess / len(slices),
            'maximum_normalized_trajectory_weight': float(weights.max()),
            'top10_normalized_trajectory_weight_share': float(np.sort(weights)[-10:].sum()),
            'observed_behavior_zero_count': 0, 'status': 'finite',
            'episodes': len(slices)}


def rank_candidates(rows):
    """Rank only predeclared matched-policy validation diagnostics."""
    names = list(dict.fromkeys(row['candidate'] for row in rows))
    ranked = []
    reference_policies = None
    for name in names:
        chosen = [r for r in rows if r['candidate'] == name and r['kind'] == 'matched_protocol']
        policies = {r['policy'] for r in chosen}
        if len(chosen) != 20 or len(policies) != 20:
            raise ValueError('Each candidate must have the same 20 matched policies')
        if reference_policies is None:
            reference_policies = policies
        elif policies != reference_policies:
            raise ValueError('Candidates must use the same policy identities')
        values = [r['trajectory_ess'] for r in chosen]
        defined = [v for v in values if v is not None and np.isfinite(v) and v > 0]
        ranked.append({'candidate': name, 'policies': 20, 'defined_policies': len(defined),
                       'eligible': len(defined) == 20,
                       'mean_ess': float(np.mean(defined)) if defined else None,
                       'median_ess': float(np.median(defined)) if defined else None,
                       'minimum_ess': min(defined) if defined else None,
                       'maximum_ess': max(defined) if defined else None})
    # Undefined propensities are not silently assigned ESS=0 or selectively omitted.
    ranked.sort(key=lambda r: (not r['eligible'], -(r['mean_ess'] or 0.), names.index(r['candidate'])))
    return ranked


def load_cohort(dataset, inputs):
    def tracked(path):
        path = Path(path)
        key = str(path.relative_to(ROOT))
        digest = sha(path)
        if key in inputs and inputs[key] != digest:
            raise ValueError('Previously recorded input changed: ' + key)
        inputs[key] = digest
        return path
    arrays = {split: {k: np.load(tracked(ROOT/'dataset'/dataset/f'{split}_{k}.npy'))
                      for k in FIELDS} for split in ['train', 'test']}
    prior = json.loads(tracked(PREVIOUS/dataset/'protocol.json').read_text())
    for key, digest in prior['inputs_sha256'].items():
        if key.startswith('dataset/' + dataset + '/') and inputs.get(key) != digest:
            raise ValueError('Dataset differs from frozen softmax cohort')
    if dataset == 'heparin':
        archive = ROOT/'outputs/heparin_mimic3_bcq_best_20261007'
        for split in arrays:
            arrays[split]['done'] = 1 - arrays[split]['done']
        with np.load(tracked(archive/'row_groups.npz')) as groups:
            train_groups, test_groups = groups['train_subjects'], groups['test_subjects']
        with np.load(tracked(archive/'selection_split.npz')) as split:
            keep = split['actor_train_mask']
        train_groups = train_groups[keep]
        with np.load(tracked(archive/'behavior/partitions.npz')) as split:
            partition = {k: split[k] for k in ['fit', 'calibration', 'validation']}
        calibrated = load_behavior_model(tracked(archive/'behavior/calibrated.joblib'))
        raw = calibrated.estimator.estimator
        seed, num_actions = 53, 6
    else:
        archive = ROOT/'outputs/sepsis_ope_corrected_20261007/behavior'
        train_groups = np.load(tracked(archive/'train_groups.npy'))
        test_groups = np.load(tracked(archive/'test_groups.npy'))
        keep = np.load(tracked(archive/'train_keep_mask.npy'))
        np.testing.assert_array_equal(keep, ~np.isin(train_groups, np.unique(test_groups)))
        original_to_local = np.full(len(keep), -1, dtype=int)
        original_to_local[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
        partition = {k: original_to_local[np.load(tracked(archive/f'{k}_train_row_indices.npy'))]
                     for k in ['fit', 'calibration', 'validation']}
        assert all(np.all(v >= 0) for v in partition.values())
        train_groups = train_groups[keep]
        calibrated = load_behavior_model(tracked(archive/'rf_sigmoid_calibrated.joblib'))
        raw = load_behavior_model(tracked(archive/'rf_raw.joblib'))
        seed, num_actions = 42, 25
    train = {k: v[keep] for k, v in arrays['train'].items()}
    assert not np.intersect1d(train_groups, test_groups).size
    replayed = grouped_partition(train['done'], train_groups, random_seed=seed)
    for key in partition:
        np.testing.assert_array_equal(np.sort(partition[key]), np.sort(replayed[key]))
    for split, groups in [(train, train_groups), (arrays['test'], test_groups)]:
        for sl in episode_slices(split['done']):
            assert np.all(groups[sl] == groups[sl.start])
    return train, arrays['test'], train_groups, test_groups, partition, seed, num_actions, raw, calibrated, tracked


def prepare(dataset, out):
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    train, test, tg, eg, part, seed, actions, raw, calibrated, tracked = load_cohort(dataset, inputs)
    np.savez_compressed(out/'partitions.npz', **part)
    val = {k: v[part['validation']] for k, v in train.items()}
    policy_dir = out/'validation_policies'
    policy_dir.mkdir()
    specs = []
    torch.set_num_threads(1)
    for directory in sorted((PREVIOUS/dataset/'policies').iterdir()):
        if not directory.is_dir():
            continue
        artifacts = json.loads(tracked(directory/'artifacts.json').read_text())
        metrics = json.loads(tracked(directory/'metrics.json').read_text())
        checkpoint = tracked(ROOT/artifacts['actor'])
        if sha(checkpoint) != artifacts['actor_sha256']:
            raise ValueError('Actor hash mismatch')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = saved.get('params', saved.get('config'))
        if config is None:
            raise ValueError('Missing actor config')
        algorithm = metrics['algorithm']
        actor = (HeparinBCQ if dataset == 'heparin' and algorithm == 'BCQ' else POLICIES[algorithm])(**config)
        actor.Q.load_state_dict(saved['model_state_dict'])
        actor.Q.eval()
        pi = frozen_policy_arrays(algorithm, actor, val['state'], mode='softmax', temperature=1.)
        np.save(policy_dir/f'{directory.name}.npy', pi)
        tracked(directory/'frozen_arrays.npz')
        tracked(directory/'evaluation.json')
        specs.append({'policy': directory.name, 'algorithm': algorithm,
                      'seed': metrics['seed'], 'kind': metrics['kind'],
                      'previous_directory': str(directory.relative_to(ROOT))})
        print('PREPARED', dataset, directory.name, flush=True)
    candidate_dir = out/'candidates'
    candidate_dir.mkdir()
    validation_metrics = {}
    for name, estimator in [('rf_raw', raw), ('rf_sigmoid', calibrated)]:
        directory = candidate_dir/name
        directory.mkdir()
        probability = full_action_proba(estimator, val['state'], num_actions=actions)
        np.save(directory/'validation_proba.npy', probability)
        validation_metrics[name] = probability_metrics(val['action'], probability, num_actions=actions)
    protocol = {'dataset': dataset, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'policy_mode': 'softmax', 'temperature': 1., 'gamma': .98,
                'candidate_order': ['rf_raw', 'rf_sigmoid'], 'split_seed': seed,
                'actor_retrained': False, 'fqe_refitted': False, 'num_actions': actions,
                'selection': 'maximum mean terminal trajectory ESS across 20 matched actors on original train-internal behavior-validation patients; ties use candidate order',
                'undefined_candidate_rule': 'ineligible if any of the 20 validation ESS values is undefined',
                'predictive_metrics_are_diagnostic_not_a_gate': True,
                'test_used_for_selection': False, 'historical_test': True,
                'validation_is_independent_of_behavior_fit_but_in_actor_training_cohort': True,
                'ratio_cap': None, 'probability_floor': None, 'output_clipping': None,
                'split': {k: {'rows': len(v), 'subjects': len(np.unique(tg[v])),
                             'episodes': len(episode_slices(train['done'][v]))} for k, v in part.items()},
                'test_rows': len(eg), 'test_subjects': len(np.unique(eg)),
                'test_episodes': len(episode_slices(test['done'])), 'policies': specs}
    write(out/'protocol.json', protocol)
    write(out/'rf_validation_metrics.json', validation_metrics)
    write(out/'input_preservation.json', inputs)


def fit(dataset, out):
    from agent import fit_behavior_sensitivity_candidates
    inputs = json.loads((out/'input_preservation.json').read_text())
    train, test, tg, eg, part, seed, actions, raw, calibrated, tracked = load_cohort(dataset, inputs)
    if (out/'candidate_fit.json').exists():
        raise FileExistsError('Candidate fitting already completed')
    def progress(event):
        print('FIT', dataset, json.dumps(event, default=str), flush=True)
    models, report = fit_behavior_sensitivity_candidates(
        train['state'], train['action'], train['done'], tg, num_actions=actions,
        partition=part, seed=seed, n_jobs=4, progress_callback=progress)
    validation_metrics = json.loads((out/'rf_validation_metrics.json').read_text())
    for name, predictor in models.items():
        directory = out/'candidates'/name
        directory.mkdir(exist_ok=False)
        joblib.dump(predictor, directory/'model.joblib', compress=3)
        validation = full_action_proba(predictor, train['state'][part['validation']], num_actions=actions)
        np.save(directory/'validation_proba.npy', validation)
        probe = joblib.load(directory/'model.joblib')
        np.testing.assert_allclose(probe.predict_proba(train['state'][part['validation'][:37]]),
                                   validation[:37], rtol=1e-6, atol=1e-9)
        validation_metrics[name] = probability_metrics(train['action'][part['validation']], validation, num_actions=actions)
        print('SAVED', dataset, name, 'validation_nll', validation_metrics[name]['log_loss'], flush=True)
    write(out/'candidate_fit.json', report)
    write(out/'validation_metrics.json', validation_metrics)
    protocol = json.loads((out/'protocol.json').read_text())
    protocol['candidate_order'] += list(models)
    protocol['candidate_families'] = ['random_forest', 'logistic', 'MLP', 'kNN_Dirichlet', 'cluster_counts_Dirichlet']
    protocol['smoothing_note'] = 'kNN/cluster Dirichlet concentration selected by calibration NLL; not a propensity floor or ESS-tuned smoothing'
    write(out/'protocol.json', protocol)
    write(out/'input_preservation.json', inputs)


def evaluate(dataset, out, n_bootstrap):
    inputs = json.loads((out/'input_preservation.json').read_text())
    train, test, tg, eg, part, seed, actions, raw, calibrated, tracked = load_cohort(dataset, inputs)
    protocol = json.loads((out/'protocol.json').read_text())
    metrics = json.loads((out/'validation_metrics.json').read_text())
    val = {k: v[part['validation']] for k, v in train.items()}
    validation_rows = []
    for candidate in protocol['candidate_order']:
        behavior = np.load(out/'candidates'/candidate/'validation_proba.npy')
        for spec in protocol['policies']:
            target = np.load(out/'validation_policies'/f"{spec['policy']}.npy")
            diag = trajectory_weight_diagnostics(val['action'], val['done'], target, behavior)
            validation_rows.append({'dataset': dataset, 'candidate': candidate,
                                    **{k: spec[k] for k in ['policy', 'algorithm', 'seed', 'kind']}, **diag})
    rankings = rank_candidates(validation_rows)
    for row in rankings:
        row.update(metrics[row['candidate']])
    if not rankings[0]['eligible']:
        raise ValueError('No fully defined candidate for validation ESS selection')
    winner = rankings[0]['candidate']
    nll_winner = min(metrics, key=lambda name: metrics[name]['log_loss'])
    selection = {'selected_behavior': winner, 'minimum_log_loss_behavior': nll_winner,
                 'rule': protocol['selection'], 'test_used_for_selection': False,
                 'selected_validation': rankings[0], 'baseline_validation': next(r for r in rankings if r['candidate']=='rf_sigmoid'),
                 'selected_log_loss_relative_change_vs_rf': metrics[winner]['log_loss']/metrics['rf_sigmoid']['log_loss'] - 1.,
                 'interpretation': 'Exploratory ESS-selected denominator, not a claim of improved clinician-policy accuracy or established policy value'}
    # Freeze and persist the decision BEFORE computing any candidate test predictions.
    write(out/'selection.json', selection)
    csv_write(out/'validation_policy_ess.csv', validation_rows)
    csv_write(out/'validation_candidate_ranking.csv', rankings)
    print('SELECTED', dataset, json.dumps(selection), flush=True)
    selected_dir = out/'selected'
    selected_dir.mkdir(exist_ok=False)
    primary_names = set([winner, nll_winner, 'rf_sigmoid'])
    test_rows = []
    starts = initial_state_indices(test['done'])
    for candidate in protocol['candidate_order']:
        if candidate == 'rf_raw':
            estimator = raw
        elif candidate == 'rf_sigmoid':
            estimator = calibrated
        else:
            estimator = joblib.load(out/'candidates'/candidate/'model.joblib')
        behavior = full_action_proba(estimator, test['state'], num_actions=actions)
        np.save(out/'candidates'/candidate/'test_proba.npy', behavior)
        if candidate == winner:
            joblib.dump(estimator, selected_dir/'behavior.joblib', compress=3)
            np.save(selected_dir/'test_proba.npy', behavior)
        for spec in protocol['policies']:
            directory = ROOT/spec['previous_directory']
            with np.load(directory/'frozen_arrays.npz') as frozen:
                cache = {k: frozen[k] for k in frozen.files}
            diag = trajectory_weight_diagnostics(test['action'], test['done'], cache['test_target_probs'], behavior)
            row = {'dataset': dataset, 'candidate': candidate,
                   **{k: spec[k] for k in ['policy', 'algorithm', 'seed', 'kind']}, **diag,
                   'selected': candidate == winner, 'minimum_log_loss': candidate == nll_winner}
            if diag['status'] == 'finite':
                bootstraps = n_bootstrap if candidate == winner else 0
                ev = evaluate_policy_arrays(test, cache['test_target_probs'], behavior, cache['test_q'],
                    next_target_probs=cache['test_next_target_probs'], next_q_values=cache['test_next_q'],
                    gamma=.98, ratio_cap=None, n_bootstrap=bootstraps, seed=20261007,
                    episode_groups=eg[starts], return_bootstrap_samples=candidate == winner,
                    return_fqe_samples=candidate == winner,
                    metadata={'dataset': dataset, 'policy': spec['policy'], 'behavior': candidate,
                              'behavior_selection': 'train-internal validation ESS', 'policy_mode': 'softmax',
                              'temperature': 1., 'actor_refitted': False, 'critic_refitted': False})
                np.testing.assert_allclose(ev['weights']['trajectory_ess'], diag['trajectory_ess'], rtol=1e-10)
                previous = json.loads((directory/'evaluation.json').read_text())
                np.testing.assert_allclose(ev['fqe']['value'], previous['fqe']['value'], rtol=0, atol=1e-12)
                if candidate == 'rf_sigmoid':
                    for key in ['dr','wdr','wis']:
                        np.testing.assert_allclose(ev[key], previous[key], rtol=1e-9, atol=1e-10)
                row.update({k: ev[k] for k in ['dr','wdr','wis']})
                row['fqe'] = ev['fqe']['value']
                if candidate in primary_names:
                    evdir = out/'evaluations'/candidate/spec['policy']
                    evdir.mkdir(parents=True, exist_ok=False)
                    write(evdir/'evaluation.json', ev)
                if candidate == winner:
                    for key in ['low', 'high']:
                        np.testing.assert_allclose(ev['fqe'][key], previous['fqe'][key], rtol=0, atol=1e-12)
                    for key in ['dr','wdr','wis']:
                        ci = ev['bootstrap']['intervals'][key]
                        row.update({key+'_low':ci['low'],key+'_high':ci['high'],key+'_bootstrap_defined':ci['defined_finite_resamples']})
                    row.update(fqe_low=ev['fqe']['low'],fqe_high=ev['fqe']['high'])
            test_rows.append(row)
            print('TEST', dataset, candidate, spec['policy'], 'ESS', row['trajectory_ess'], flush=True)
    csv_write(out/'test_policy_metrics.csv', test_rows)
    selected_rows = [r for r in test_rows if r['selected']]
    csv_write(out/'selected_policy_metrics.csv', selected_rows)
    test_rankings = rank_candidates(test_rows)
    csv_write(out/'test_candidate_summary.csv', test_rankings)
    summaries = []
    for candidate in dict.fromkeys(['rf_sigmoid', winner, nll_winner]):
        for algorithm in ALGORITHMS:
            chosen = [r for r in test_rows if r['candidate']==candidate and r['algorithm']==algorithm and r['kind']=='matched_protocol']
            summary = {'dataset':dataset,'candidate':candidate,'algorithm':algorithm,'policies':len(chosen)}
            for key in ['trajectory_ess','wdr','wis','dr','fqe']:
                values = [r[key] for r in chosen if r.get(key) is not None]
                summary.update({key+'_mean':float(np.mean(values)) if values else None,
                                key+'_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,
                                key+'_min':min(values) if values else None,
                                key+'_max':max(values) if values else None,
                                key+'_defined':len(values)})
            summaries.append(summary)
    csv_write(out/'algorithm_summary.csv', summaries)
    changed = {key: sha(ROOT/key) for key,value in inputs.items() if sha(ROOT/key)!=value}
    if changed:
        raise ValueError('Immutable inputs changed: ' + str(changed))
    write(out/'input_preservation.json', inputs)
    write(out/'completion_receipt.json', {'status':'complete','dataset':dataset,
        'completed_utc':datetime.now(timezone.utc).isoformat(),'selected_behavior':winner,
        'candidate_count':len(protocol['candidate_order']),'policy_count':len(protocol['policies']),
        'actor_refits':0,'critic_refits':0,'bootstrap_draws_selected':n_bootstrap,
        'previous_rf_ope_reproduced':True,'independent_log_ess_matches_ope':True,
        'fixed_fqe_point_estimates_unchanged':True,'input_sha256_preserved':True,
        'artifact_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and not p.name.startswith('._')}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['heparin','sepsis'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare','fit','evaluate'], required=True)
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()/args.dataset
    start = time.perf_counter()
    torch.set_num_threads(1)
    with threadpool_limits(limits=4):
        {'prepare':lambda:prepare(args.dataset,out), 'fit':lambda:fit(args.dataset,out),
         'evaluate':lambda:evaluate(args.dataset,out,args.n_bootstrap)}[args.stage]()
    print('STAGE_COMPLETE',args.dataset,args.stage,time.perf_counter()-start,flush=True)


if __name__ == '__main__':
    main()
