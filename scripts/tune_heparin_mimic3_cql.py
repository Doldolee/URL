"""Validation-selected MIMIC-III Heparin CQL search and softmax OPE.

The existing cql_alpha is a log-sum-exp temperature, not a loss multiplier.
Selection uses the mean validation FQE across paired actor seeds. Test data
are evaluated only after the setting and its single checkpoint are frozen.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import itertools
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agent import CQL, frozen_policy_arrays
from metric import (bootstrap_initial_values, evaluate_policy_arrays,
                    fit_heparin_fqe, predict_q)
from model import HeparinFQECritic, full_action_proba
from util import (ArrayTrainingBuffer, episode_slices, initial_state_indices,
                  load_behavior_model, sha256_file, verify_source_manifest)
from scripts.evaluate_clipping_sensitivity import direct_reference, assert_reference

SPLIT_SOURCE = Path('outputs/heparin_mimic3_bcq_best_20261007')
BASELINE = Path('outputs/mimic3_four_algorithm_ope_20261007/heparin')
SOFTMAX_BASELINE = Path('outputs/mimic3_four_algorithm_softmax_ope_20261007/heparin')
CAP5_BASELINE = Path('outputs/mimic3_four_algorithm_cumulative_cap5_ope_20261007')
LEARNING_RATES = (1e-6, 1e-5, 1e-4, 3e-4)
CQL_TEMPERATURES = (.1, 1., 5.)
ACTOR_SEEDS = (42, 43, 44)
CHECKPOINT_UPDATES = (1500, 5000)


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = Path(path).with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_training(root):
    """Reuse the hash-verified, episode-complete patient split from BCQ."""
    source = root / SPLIT_SOURCE
    arrays = {key: np.load(root / 'dataset/heparin' / f'train_{key}.npy', allow_pickle=False)
              for key in ('state', 'next_state', 'action', 'reward', 'done')}
    arrays['done'] = 1 - arrays['done']
    with np.load(source / 'row_groups.npz') as saved:
        groups, test_groups = saved['train_subjects'], saved['test_subjects']
        keep = saved['train_keep']
    with np.load(source / 'selection_split.npz') as saved:
        fit_mask, val_mask = saved['actor_train_mask'], saved['policy_validation_mask']
    assert len(groups) == len(arrays['state'])
    assert fit_mask.dtype == val_mask.dtype == np.dtype(bool)
    assert not np.any(fit_mask & val_mask)
    np.testing.assert_array_equal(fit_mask | val_mask, keep)
    assert not np.intersect1d(groups[fit_mask], groups[val_mask]).size
    assert not np.intersect1d(groups[keep], test_groups).size
    for rows in episode_slices(arrays['done']):
        assert np.all(fit_mask[rows] == fit_mask[rows.start])
        assert np.all(val_mask[rows] == val_mask[rows.start])
        assert np.all(groups[rows] == groups[rows.start])
    train = {key: value[fit_mask] for key, value in arrays.items()}
    val = {key: value[val_mask] for key, value in arrays.items()}
    episode_slices(train['done'])
    episode_slices(val['done'])
    return train, val, groups[fit_mask], groups[val_mask], test_groups


def config(lr, temperature):
    return dict(num_actions=6, state_dim=16, device='cpu', discount=.98,
                optimizer='Adam', optimizer_parameters={'lr': lr, 'weight_decay': 1e-5},
                use_polyak_target_update=False, target_update_frequency=25,
                tau=.005, hidden_node=1024, activation='relu', cql_alpha=temperature,
                max_timesteps=5000)


def choose_setting(rows):
    """Rank complete three-seed settings; then select one validation checkpoint."""
    grouped = {}
    for row in rows:
        if not np.isfinite(row['validation_fqe']):
            raise ValueError('Nonfinite validation selection metric')
        key = (row['learning_rate'], row['cql_temperature'], row['updates'])
        grouped.setdefault(key, []).append(row)
    settings = []
    for (lr, temperature, updates), members in grouped.items():
        if sorted(row['seed'] for row in members) != list(ACTOR_SEEDS):
            raise ValueError('Each setting must contain exactly the three paired actor seeds')
        values = [row['validation_fqe'] for row in members]
        settings.append({'learning_rate': lr, 'cql_temperature': temperature,
                         'updates': updates, 'seeds': list(ACTOR_SEEDS),
                         'validation_fqe_mean': float(np.mean(values)),
                         'validation_fqe_seed_sd': float(np.std(values, ddof=1)),
                         'validation_fqe_min': float(min(values)),
                         'validation_fqe_max': float(max(values))})
    ranked = sorted(settings, key=lambda row: (-row['validation_fqe_mean'],
                    row['updates'], row['learning_rate'], row['cql_temperature']))
    if not ranked:
        raise ValueError('No complete settings to select')
    setting = ranked[0]
    members = grouped[(setting['learning_rate'], setting['cql_temperature'], setting['updates'])]
    best = sorted(members, key=lambda row: (-row['validation_fqe'], row['seed']))[0]
    return ranked, best


def make_policy(checkpoint):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    policy = CQL(**saved['config'])
    policy.Q.load_state_dict(saved['model_state_dict'])
    policy.Q.eval()
    return policy, saved


def check_baseline(root, policy, seed, train, next_probs, critic):
    """Assert that the unchanged baseline reproduces the previously frozen run."""
    old = torch.load(root / BASELINE / 'policies' / f'heparin_CQL_seed{seed}' /
                     'policy_final.pth', map_location='cpu', weights_only=False)
    for key, value in policy.Q.state_dict().items():
        torch.testing.assert_close(value, old['model_state_dict'][key], rtol=0, atol=0)
    old_critic = torch.load(root / SOFTMAX_BASELINE / 'policies' /
                            f'heparin_CQL_seed{seed}' / 'fqe.pt',
                            map_location='cpu', weights_only=False)
    state = old_critic.get('model_state_dict', old_critic)
    for key, value in critic.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    return {'baseline_actor_state_dict_exact': True,
            'baseline_softmax_critic_state_dict_exact': True}


def train_run(task):
    root, out, lr, temperature, seed = task
    root, out = Path(root), Path(out)
    name = f'cql_lr{lr:g}_temperature{temperature:g}_seed{seed}'
    directory = out / 'candidates' / name
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / 'run_complete.json'
    if complete.exists():
        record = json.loads(complete.read_text())
        for relative, digest in record['artifacts_sha256'].items():
            assert sha256_file(directory / relative) == digest
        return record['candidates']
    with (directory / 'training.log').open('a') as log, redirect_stdout(log), threadpool_limits(limits=1):
        torch.set_num_threads(1)
        train, val, train_groups, val_groups, _ = load_training(root)
        behavior = np.load(out / 'validation_behavior_probs.npy', allow_pickle=False)
        starts = initial_state_indices(val['done'])
        cfg = config(lr, temperature)
        torch.manual_seed(seed)
        policy = CQL(**cfg)
        buffer = ArrayTrainingBuffer(train, seed=seed, batch_size=64)
        initial = {key: value.detach().clone() for key, value in policy.Q.named_parameters()}
        begun = time.perf_counter()
        rows = []
        for update in range(1, max(CHECKPOINT_UPDATES) + 1):
            # Match the completed four-algorithm protocol: freeze target BN buffers.
            policy.Q_target.eval()
            policy.train(buffer)
            if update % 250 == 0:
                write(directory / 'progress.json', {'run': name, 'update': update,
                      'elapsed_seconds': time.perf_counter() - begun})
                print('UPDATE', update, round(time.perf_counter() - begun, 2), flush=True)
            if update not in CHECKPOINT_UPDATES:
                continue
            assert all(torch.isfinite(value).all() for value in policy.Q.parameters())
            delta = sum(float((value.detach() - initial[key]).square().sum())
                        for key, value in policy.Q.named_parameters()) ** .5
            assert delta > 0
            checkpoint = directory / f'update{update}.pth'
            torch.save({'model_state_dict': policy.Q.state_dict(), 'config': cfg,
                        'seed': seed, 'updates': update, 'parameter_l2_change': delta,
                        'protocol_sha256': sha256_file(out / 'protocol.json')}, checkpoint)
            next_probs = frozen_policy_arrays('CQL', policy, train['next_state'], mode='softmax')
            critic, fitting = fit_heparin_fqe(train, next_probs, train_groups, seed=42)
            probs = frozen_policy_arrays('CQL', policy, val['state'], mode='softmax')
            q = predict_q(critic, val['state'], num_threads=1)
            evaluation = evaluate_policy_arrays(val, probs, behavior, q,
                gamma=.98, cumulative_weight_cap=5., n_bootstrap=0,
                episode_groups=val_groups[starts], metadata={'scope': 'selection validation only'})
            initial_values = (probs[starts] * q[starts]).sum(1)
            interval = bootstrap_initial_values(initial_values, groups=val_groups[starts],
                                                n_bootstrap=400, seed=20261007)
            verification = {}
            if lr == 1e-6 and temperature == 1. and update == 1500:
                verification = check_baseline(root, policy, seed, train, next_probs, critic)
            row = {'candidate': f'{name}_update{update}', 'learning_rate': lr,
                   'cql_temperature': temperature, 'seed': seed, 'updates': update,
                   'validation_fqe': interval['value'], 'validation_fqe_low': interval['low'],
                   'validation_fqe_high': interval['high'], 'validation_wdr_cap5': evaluation['wdr'],
                   'validation_dr_cap5': evaluation['dr'], 'validation_wis_cap5': evaluation['wis'],
                   'validation_trajectory_ess_cap5': evaluation['weights']['trajectory_ess'],
                   'validation_trajectory_ess_percent_cap5':
                       100 * evaluation['weights']['trajectory_ess'] / len(starts),
                   'critic_validation_bellman_mse': fitting['selected_validation']['bellman_mse'],
                   'critic_epoch': fitting['selected_epoch'], 'actor_parameter_l2_change': delta,
                   'checkpoint': checkpoint.relative_to(out).as_posix(),
                   'checkpoint_sha256': sha256_file(checkpoint)}
            torch.save(critic.state_dict(), directory / f'update{update}_fqe.pt')
            write(directory / f'update{update}_fqe_fit.json', fitting)
            write(directory / f'update{update}_validation.json', {
                'candidate': row, 'ope': evaluation, 'fqe': interval,
                'baseline_verification': verification})
            np.savez_compressed(directory / f'update{update}_validation_arrays.npz',
                                target_probs=probs, q=q, initial_values=initial_values)
            rows.append(row)
            print('CANDIDATE', json.dumps(row), flush=True)
        artifacts = {path.name: sha256_file(path) for path in directory.iterdir()
                     if path.is_file() and path.suffix in {'.pth', '.pt', '.npz', '.json'}
                     and path.name not in {'progress.json', 'run_complete.json'}}
        write(complete, {'status': 'complete', 'candidates': rows,
                        'elapsed_seconds': time.perf_counter() - begun,
                        'artifacts_sha256': artifacts})
        return rows


def final_evaluation(root, out, best):
    selected = out / 'selected'
    selected.mkdir(exist_ok=True)
    destination = selected / 'policy_best.pth'
    shutil.copyfile(out / best['checkpoint'], destination)
    write(selected / 'selection.json', {
        'status': 'frozen before loading test transition arrays', 'selected': best,
        'selected_setting': json.loads((out / 'ranked_settings.json').read_text())[0],
        'setting_selection': 'maximum mean external validation FQE across seeds 42,43,44',
        'checkpoint_selection': 'maximum validation FQE within the selected setting',
        'test_used_for_selection': False, 'no_actor_refit_after_selection': True,
        'checkpoint_sha256': sha256_file(destination)})
    print('SELECTED', json.dumps(best), flush=True)
    train, _, train_groups, _, test_groups = load_training(root)
    test = {key: np.load(root / 'dataset/heparin' / f'test_{key}.npy', allow_pickle=False)
            for key in train}
    test['done'] = 1 - test['done']
    starts = initial_state_indices(test['done'])
    for rows in episode_slices(test['done']):
        assert np.all(test_groups[rows] == test_groups[rows.start])
    policy, checkpoint = make_policy(destination)
    torch.set_num_threads(1)
    probs = frozen_policy_arrays('CQL', policy, test['state'], mode='softmax')
    next_probs = frozen_policy_arrays('CQL', policy, test['next_state'], mode='softmax')
    np.testing.assert_allclose(probs, frozen_policy_arrays('CQL', policy, test['state'],
                               mode='softmax', batch_size=257), rtol=2e-6, atol=2e-7)
    behavior_model = load_behavior_model(root / SPLIT_SOURCE / 'behavior/calibrated.joblib')
    predicted_behavior = full_action_proba(behavior_model, test['state'], num_actions=6)
    behavior = np.load(root / SPLIT_SOURCE / 'selected/behavior_test_probs.npy', allow_pickle=False)
    np.testing.assert_allclose(predicted_behavior, behavior, rtol=0, atol=5e-15)
    np.save(selected / 'behavior_test_probs.npy', behavior)
    train_next_probs = frozen_policy_arrays('CQL', policy, train['next_state'], mode='softmax')
    rows, verification = [], []
    for critic_seed in (42, 43, 44):
        if critic_seed == 42:
            candidate = (out / best['checkpoint']).parent
            fitting = json.loads((candidate / f"update{best['updates']}_fqe_fit.json").read_text())
            critic = HeparinFQECritic(16, 6, 128, 1 / (1 - .98))
            critic.load_state_dict(torch.load(candidate / f"update{best['updates']}_fqe.pt",
                                  map_location='cpu', weights_only=True))
            critic.eval()
        else:
            critic, fitting = fit_heparin_fqe(train, train_next_probs, train_groups, seed=critic_seed)
        torch.save(critic.state_dict(), selected / f'fqe_seed{critic_seed}.pt')
        write(selected / f'fqe_seed{critic_seed}_fit.json', fitting)
        q = predict_q(critic, test['state'], num_threads=1)
        next_q = predict_q(critic, test['next_state'], num_threads=1)
        evaluation = evaluate_policy_arrays(test, probs, behavior, q,
            next_target_probs=next_probs, next_q_values=next_q,
            gamma=.98, cumulative_weight_cap=5., n_bootstrap=1000,
            seed=20261007, episode_groups=test_groups[starts], return_bootstrap_samples=True,
            return_fqe_samples=True, metadata={'actor': best['candidate'], 'critic_seed': critic_seed,
                                               'policy': 'softmax temperature1'})
        reference = direct_reference(test['action'], test['reward'], test['done'],
                                     probs, behavior, q, cumulative_weight_cap=5.)
        error = assert_reference(evaluation, reference)
        raw = evaluate_policy_arrays(test, probs, behavior, q,
            gamma=.98, n_bootstrap=0, episode_groups=test_groups[starts])
        raw_reference = direct_reference(test['action'], test['reward'], test['done'],
                                        probs, behavior, q)
        raw_error = assert_reference(raw, raw_reference)
        write(selected / f'evaluation_seed{critic_seed}.json', evaluation)
        write(selected / f'unclipped_diagnostic_seed{critic_seed}.json', raw)
        np.savez_compressed(selected / f'frozen_arrays_seed{critic_seed}.npz',
            test_target_probs=probs, test_next_target_probs=next_probs,
            test_q=q, test_next_q=next_q)
        row = {'critic_seed': critic_seed, 'primary': critic_seed == 42,
               'fqe': evaluation['fqe']['value'], 'fqe_low': evaluation['fqe']['low'],
               'fqe_high': evaluation['fqe']['high']}
        for name in ('dr', 'wdr', 'wis'):
            interval = evaluation['bootstrap']['intervals'][name]
            row.update({name: evaluation[name], name + '_low': interval['low'],
                        name + '_high': interval['high'],
                        name + '_defined_resamples': interval['defined_finite_resamples']})
        row.update(trajectory_ess=evaluation['weights']['trajectory_ess'],
                   trajectory_ess_percent=100 * evaluation['weights']['trajectory_ess'] / len(starts),
                   unclipped_trajectory_ess=raw['weights']['trajectory_ess'],
                   unclipped_trajectory_ess_percent=100 * raw['weights']['trajectory_ess'] / len(starts),
                   episodes=len(starts))
        rows.append(row)
        csv_write(out / 'ope_metrics.csv', rows)
        verification.append({'critic_seed': critic_seed, 'cap5_relative_errors': error,
                             'unclipped_relative_errors': raw_error})
        reloaded = HeparinFQECritic(16, 6, 128, 1 / (1 - .98))
        reloaded.load_state_dict(torch.load(selected / f'fqe_seed{critic_seed}.pt',
                                map_location='cpu', weights_only=True))
        np.testing.assert_array_equal(predict_q(reloaded, test['state'], num_threads=1), q)
        print('OPE', json.dumps(row), flush=True)
    reloaded_policy, _ = make_policy(destination)
    np.testing.assert_array_equal(probs, frozen_policy_arrays('CQL', reloaded_policy,
                                                             test['state'], mode='softmax'))
    write(out / 'numerical_verification.json', {'status': 'passed',
          'independent_longdouble_direct_products': verification,
          'reloaded_actor_and_all_critics_match_cached_test_arrays': True,
          'test_behavior_identical_to_frozen_shared_behavior': True,
          'softmax_batch_invariance_atol': 2e-7})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError('Use one to three CPU workers, each with one torch thread')
    root, out = ROOT, args.output_dir.resolve()
    manifest = verify_source_manifest(root)
    existed = out.exists()
    if existed and not args.resume:
        raise FileExistsError('Use a fresh output directory or explicitly resume an incomplete run')
    if (out / 'completion_receipt.json').exists():
        raise FileExistsError('A completed run must never be overwritten')
    out.mkdir(parents=True, exist_ok=True)
    source = root / SPLIT_SOURCE
    prior = json.loads((source / 'completion_receipt.json').read_text())
    required = ['row_groups.npz', 'selection_split.npz', 'behavior/calibrated.joblib',
                'selected/behavior_test_probs.npy']
    for relative in required:
        assert sha256_file(source / relative) == prior['artifact_sha256'][relative]
    inputs = [root / 'source_manifest.json', source / 'completion_receipt.json']
    inputs += [root / relative for relative in manifest['sources']]
    inputs += [root / relative for relative in manifest['interfaces']]
    inputs += [source / relative for relative in required]
    inputs += [root / 'dataset/heparin' / f'{split}_{key}.npy'
               for split in ('train', 'test')
               for key in ('state', 'next_state', 'action', 'reward', 'done', 'BC_prob')]
    inputs += sorted((root / 'pth').glob('heparin_CQL_*.pth'))
    inputs += [root / BASELINE / 'protocol.json', root / CAP5_BASELINE / 'completion_receipt.json',
               root / CAP5_BASELINE / 'algorithm_summary.csv']
    for seed in ACTOR_SEEDS:
        inputs += [root / BASELINE / 'policies' / f'heparin_CQL_seed{seed}' / 'policy_final.pth',
                   root / SOFTMAX_BASELINE / 'policies' / f'heparin_CQL_seed{seed}' / 'fqe.pt']
    before = {str(path.relative_to(root)): sha256_file(path) for path in inputs}
    train, val, train_groups, val_groups, test_groups = load_training(root)
    starts = initial_state_indices(val['done'])
    protocol = {'dataset': 'MIMIC-III Heparin', 'train_rows': len(train_groups),
        'train_patients': len(np.unique(train_groups)), 'train_episodes': len(episode_slices(train['done'])),
        'selection_validation_rows': len(val_groups), 'selection_validation_patients': len(np.unique(val_groups)),
        'selection_validation_episodes': len(starts), 'test_rows': len(test_groups),
        'test_patients': len(np.unique(test_groups)), 'actor_seeds': list(ACTOR_SEEDS),
        'learning_rates': list(LEARNING_RATES), 'cql_temperatures': list(CQL_TEMPERATURES),
        'checkpoint_updates': list(CHECKPOINT_UPDATES), 'actor_runs': 36, 'checkpoints': 72,
        'cql_alpha_meaning': 'temperature in alpha*logsumexp(Q/alpha); conservative-loss multiplier fixed1',
        'loss': 'observed-action TD MSE + alpha*logsumexp(Q/alpha) - Q(s,a_data)',
        'target': 'online greedy action, frozen target value, target BN eval between hard copies',
        'actor_architecture': 'unchanged CQLNet, 16 inputs, 6 actions, hidden1024, ReLU, BatchNorm',
        'actor_batch_size': 64, 'optimizer': 'Adam', 'weight_decay': 1e-5, 'fixed_learning_rate': True,
        'target_update_frequency': 25, 'gamma': .98, 'policy_mode': 'softmax', 'policy_temperature': 1.,
        'selection_rule': 'maximum mean validation FQE across three paired actor seeds; tie fewer updates/lower lr/lower temperature',
        'checkpoint_selection_rule': 'maximum validation FQE within selected setting; tie lower seed',
        'fqe': {'hidden_dim': 128, 'seed': 42, 'max_epochs': 100, 'min_epochs': 50, 'patience': 20,
                'lr': .001, 'batch_size': 2048, 'internal_holdout': '15% actor-training patient Bellman MSE',
                'value_bound': 1 / (1 - .98)},
        'selected_actor_critic_sensitivity_seeds': [42, 43, 44], 'cumulative_weight_cap': 5.,
        'cumulative_cap_feedback': False, 'ratio_cap': None, 'probability_floor': None,
        'behavior': 'reuse exact train-only sigmoid-calibrated RF; no refit or selection',
        'n_bootstrap': 1000, 'bootstrap_seed': 20261007, 'bootstrap_unit': 'whole test patient',
        'bootstrap_scope': 'conditional on selected policy and fitted nuisances; excludes tuning and fitting uncertainty',
        'test_used_for_selection': False, 'test_historical_previously_analyzed': True,
        'original_preprocessing_arrays_unchanged': True, 'test_patient_overlap_removed_before_training': True,
        'inputs_sha256': before}
    protocol_path = out / 'protocol.json'
    if existed:
        assert json.loads(protocol_path.read_text()) == protocol, 'Resume protocol/input mismatch'
    else:
        write(protocol_path, protocol)
        behavior_model = load_behavior_model(source / 'behavior/calibrated.joblib')
        np.save(out / 'validation_behavior_probs.npy', full_action_proba(behavior_model, val['state'], num_actions=6))
        shutil.copyfile(source / 'selection_split.npz', out / 'selection_split.npz')
        shutil.copyfile(source / 'row_groups.npz', out / 'row_groups.npz')
    print('PROTOCOL', json.dumps({key: protocol[key] for key in ('actor_runs', 'checkpoints',
          'train_rows', 'selection_validation_patients', 'test_patients')}), flush=True)
    tasks = [(str(root), str(out), lr, temperature, seed)
             for lr, temperature, seed in itertools.product(LEARNING_RATES, CQL_TEMPERATURES, ACTOR_SEEDS)]
    candidates = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(train_run, task): task for task in tasks}
        for future in as_completed(pending):
            rows = future.result()
            candidates.extend(rows)
            csv_write(out / 'validation_candidates.csv', sorted(candidates, key=lambda row: row['candidate']))
            print('RUN_COMPLETE', len(candidates) // 2, '/36',
                  json.dumps({key: rows[-1][key] for key in ('candidate', 'validation_fqe')}), flush=True)
    assert len(candidates) == 72
    ranked, best = choose_setting(candidates)
    csv_write(out / 'setting_summary.csv', ranked)
    write(out / 'ranked_settings.json', ranked)
    final_evaluation(root, out, best)
    after = {relative: sha256_file(root / relative) for relative in before}
    assert before == after
    write(out / 'input_preservation.json', {'status': 'passed', 'all_input_hashes_unchanged': True,
                                            'inputs_sha256': after})
    artifacts = {path.relative_to(out).as_posix(): sha256_file(path) for path in out.rglob('*')
                 if path.is_file() and not path.name.startswith('._') and path.suffix != '.tmp'}
    write(out / 'completion_receipt.json', {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'actor_runs': 36, 'candidate_checkpoints': 72, 'settings_compared': 24,
        'single_selected_checkpoint': True, 'selected': best, 'inputs_preserved': True,
        'test_used_for_selection': False, 'artifact_sha256': artifacts})
    print('COMPLETE', str(out), flush=True)


if __name__ == '__main__':
    main()
