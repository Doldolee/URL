# coding: utf-8
"""Train five seeds of four algorithms, then fixed greedy-policy OPE.

All original inputs/artifacts are read only. Write to a new output directory.
Protocol is written before training; no test-based hyperparameter selection.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / 'tests'))
from util import source_hashes
import numpy as np
import torch
from agent import POLICIES
from util import ArrayTrainingBuffer
from agent import frozen_policy_arrays
from metric import evaluate_policy_arrays
from metric import fit_fqe, predict_q
from wdr_reference import reference_wdr


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.project_root.resolve(), args.output_dir.resolve()
    if out.exists():
        raise FileExistsError('Completed or partial output is not overwritten; use a new directory')
    out.mkdir(parents=True)
    archive = root / 'outputs/sepsis_ope_corrected_20261007'
    manifest = json.loads((archive / 'behavior/manifest.json').read_text())
    fields = ['state', 'next_state', 'action', 'reward', 'done']
    inputs = [root / 'dataset/sepsis' / (split + '_' + k + '.npy') for split in ['train', 'test'] for k in fields]
    inputs += [archive / 'behavior' / k for k in ['manifest.json', 'train_keep_mask.npy', 'train_groups.npy', 'test_groups.npy', manifest['primary_behavior']]]
    inputs += [root / k for k in ['agent.py', 'agent.py', 'agent.py', 'agent.py', 'agent.py', 'model.py', 'agent.py', 'metric.py', 'metric.py', 'metric.py', 'metric.py', 'metric.py', 'metric.py', 'model.py', 'util.py', 'util.py', 'configs/config.py', 'configs/config_base.yaml']]
    inputs += list((root / 'pth').glob('sepsis_*.pth'))
    inputs += [root / 'outputs' / k / 'completion_receipt.json' for k in ['sepsis_ope_corrected_20261007', 'sepsis_policy_retrain_20261007', 'sepsis_wdr_20261007', 'sepsis_wdr_softmax_20261007']]
    before = {str(p.relative_to(root)): sha(p) for p in inputs}
    train = {k: np.load(root / 'dataset/sepsis' / ('train_' + k + '.npy')) for k in fields}
    test = {k: np.load(root / 'dataset/sepsis' / ('test_' + k + '.npy')) for k in fields}
    for split, arrays in [('train', train), ('test', test)]:
        for k in fields:
            if sha(root / 'dataset/sepsis' / (split + '_' + k + '.npy')) != manifest['inputs'][split][k]['sha256']:
                raise ValueError('Dataset changed since frozen behavior fit')
    keep = np.load(archive / 'behavior/train_keep_mask.npy')
    tg = np.load(archive / 'behavior/train_groups.npy')
    eg = np.load(archive / 'behavior/test_groups.npy')
    if not np.array_equal(keep, ~np.isin(tg, np.unique(eg))):
        raise ValueError('Training mask must remove every test patient')
    train = {k: x[keep] for k, x in train.items()}
    tg = tg[keep]
    if np.intersect1d(tg, eg).size:
        raise ValueError('Training/test patients overlap')
    for arrays, groups in [(train, tg), (test, eg)]:
        d, r = arrays['done'].reshape(-1), arrays['reward'].reshape(-1)
        ends = np.flatnonzero(d == 1)
        starts = np.r_[0, ends[:-1] + 1]
        assert ends[-1] == len(d) - 1
        assert np.all(r[d == 0] == 0) and np.all(np.isin(r[d == 1], [-1, 1]))
        continuing = np.flatnonzero(d[:-1] == 0)
        np.testing.assert_allclose(arrays['next_state'][continuing], arrays['state'][continuing + 1], rtol=1e-5, atol=1e-6)
        for start, end in zip(starts, ends):
            assert np.all(groups[start:end + 1] == groups[start])
    starts = np.r_[0, np.flatnonzero(test['done'].reshape(-1) == 1)[:-1] + 1]
    episodes, subjects = len(starts), len(np.unique(eg))
    episode_groups = eg[starts]
    behavior = np.load(archive / 'behavior' / manifest['primary_behavior'])
    cfg = dict(num_actions=25, state_dim=43, device='cpu', discount=.98, optimizer='Adam',
               optimizer_parameters={'lr':1e-6, 'weight_decay':1e-5}, use_polyak_target_update=False,
               target_update_frequency=25, tau=.005, hidden_node=1024, activation='relu',
               cql_alpha=1., bcq_threshold=.3, max_timesteps=1500)
    fit_cfg = dict(gamma=.98, hidden_dim=128, epochs=100, min_epochs=30, patience=20,
                   batch_size=4096, validation_fraction=.15, lr=1e-3, seed=42, num_threads=4)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), algorithms=list(POLICIES),
                    policy_seeds=[42,43,44,45,46], training_updates=1500, batch_size=64,
                    policy_config=cfg, fqe_config=fit_cfg, evaluation_episodes=episodes,
                    evaluation_subjects=subjects, train_rows=len(tg), train_subjects=len(np.unique(tg)),
                    policy_mode='greedy', ratio_cap=None, probability_floor=None, output_clipping=None,
                    bootstrap_draws=1000, bootstrap_seed=20261007, bootstrap_unit='whole patient cluster',
                    behavior='frozen train-only calibrated RF shared by all 20 policies',
                    actor_selection='final update only, no test or validation policy selection',
                    fixes=['DQN target is max_a Q_target(s_next,a); DDQN online argmax with target evaluation',
                           'BCQ restore fixed configured Adam LR after unstepped scheduler initializes zero LR'],
                    original_rl_agent_preserved=True,
                    preprocessing='existing normalized arrays, including known upstream preprocessing limitations',
                    primary_cohort='all 1662 complete stored test episodes; all test patients excluded from training',
                    uncertainty_scope='conditional fixed-nuisance patient bootstrap; checkpoint SD is descriptive',
                    inputs_sha256=before)
    write(out / 'protocol.json', protocol)
    torch.set_num_threads(4)
    rows, traces = [], []
    for algorithm in POLICIES:
        for seed in protocol['policy_seeds']:
            name = f'sepsis_{algorithm}_seed{seed}'
            directory = out / 'models' / name
            directory.mkdir(parents=True)
            torch.manual_seed(seed)
            policy = POLICIES[algorithm](**cfg)
            buffer = ArrayTrainingBuffer(train, seed=seed, batch_size=64)
            initial = {k: v.detach().clone() for k,v in policy.Q.named_parameters()}
            begun = time.perf_counter()
            print('TRAIN', name, 'lr', policy.Q_optimizer.param_groups[0]['lr'], flush=True)
            for update in range(1500):
                policy.train(buffer)
                if (update + 1) % 250 == 0:
                    print(name, 'update', update + 1, 'seconds', round(time.perf_counter() - begun, 2), flush=True)
            delta = sum(float((v.detach() - initial[k]).square().sum()) for k,v in policy.Q.named_parameters()) ** .5
            if not delta > 0 or not all(torch.isfinite(v).all() for v in policy.Q.parameters()):
                raise ValueError('Policy did not train with finite parameter changes')
            training = dict(seed=seed, updates=policy.iterations, elapsed_seconds=time.perf_counter()-begun,
                            parameter_l2_change=delta, final_lr=policy.Q_optimizer.param_groups[0]['lr'],
                            train_patient_groups_sha256=array_sha(tg), test_patients_excluded=True,
                            action_sampling='uniform transitions, paired numpy default_rng(seed) stream')
            write(directory / 'training.json', training)
            policy.Q.eval()
            torch.save(dict(model_state_dict=policy.Q.state_dict(),algorithm=algorithm,seed=seed,
                            params=cfg,training=training,protocol_sha256=sha(out/'protocol.json')),
                       directory / 'policy_final.pth')
            next_pi = frozen_policy_arrays(algorithm, policy, train['next_state'], mode='greedy')
            pi = frozen_policy_arrays(algorithm, policy, test['state'], mode='greedy')
            # Changing inference batching must not change an eval-mode greedy actor.
            probe = frozen_policy_arrays(algorithm, policy, test['state'][:2049], mode='greedy', batch_size=257)
            np.testing.assert_array_equal(probe, pi[:2049])
            def callback(x):
                if x['epoch'] % 20 == 0:
                    print('FQE', name, 'epoch', x['epoch'], flush=True)
            critic, fit = fit_fqe(train['state'], train['next_state'], train['action'], train['reward'],
                                 train['done'], next_pi, groups=tg, progress_callback=callback, **fit_cfg)
            assert fit['fixed_next_policy_sha256_float32'] == array_sha(next_pi.astype(np.float32))
            write(directory / 'fit.json', fit)
            torch.save(critic.state_dict(), directory / 'fqe.pt')
            qhat = predict_q(critic, test['state'])
            np.savez_compressed(directory / 'frozen_arrays.npz', train_next_actions=next_pi.argmax(1),
                                test_actions=pi.argmax(1), critic_test_q=qhat)
            print('OPE', name, flush=True)
            ope = evaluate_policy_arrays(test, pi, behavior, qhat,
                               gamma=.98, ratio_cap=None, n_bootstrap=1000, seed=20261007,
                               episode_groups=episode_groups, return_bootstrap_samples=True, return_fqe_samples=True,
                               metadata={'actor':name,'training':'fresh common protocol, all test subjects excluded',
                                         'critic':'train-only fixed-policy bounded FQE','behavior':'same frozen calibrated RF'})
            fqe = ope.pop('fqe')
            reference = reference_wdr(test['action'].reshape(-1), test['reward'].reshape(-1),
                                      test['done'].reshape(-1), pi, behavior, qhat, gamma=.98)
            if reference['value'] is None:
                assert ope['wdr'] is None
                error = None
            else:
                error = abs(reference['value'] - ope['wdr'])
                np.testing.assert_allclose(reference['value'], ope['wdr'], rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose([x['ess'] for x in reference['columns']],
                                          ope['weights']['per_decision_ess_with_absorbing_padding'], rtol=1e-10, atol=1e-10)
            verification = dict(decimal_wdr=reference['value'],absolute_error=error,
                                undefined_time_step=reference['undefined_time_step'], actor_batch_invariant=True,
                                fqe_actor_hash_matches=True)
            write(directory / 'evaluation.json',dict(ope=ope,fqe=fqe,verification=verification))
            intervals = ope['bootstrap']['intervals']
            row = dict(algorithm=algorithm,seed=seed,policy=name,episodes=episodes,subjects=subjects,
                       wdr=ope['wdr'],fqe=fqe['value'],wis=ope['wis'],dr=ope['dr'],
                       wdr_low=intervals['wdr']['low'],wdr_high=intervals['wdr']['high'],
                       wis_low=intervals['wis']['low'],wis_high=intervals['wis']['high'],
                       fqe_low=fqe['low'],fqe_high=fqe['high'],
                       wdr_bootstrap_defined=intervals['wdr']['defined_finite_resamples'],
                       wis_bootstrap_defined=intervals['wis']['defined_finite_resamples'],
                       terminal_weight_ess=ope['weights']['trajectory_ess'],
                       nonzero_terminal_paths=episodes-ope['weights']['exact_zero_trajectory_weight_count'],
                       max_terminal_normalized_weight=ope['weights']['maximum_normalized_trajectory_weight'],
                       decimal_absolute_error=error,fqe_selected_epoch=fit['selected_epoch'])
            rows.append(row)
            traces.extend({'algorithm':algorithm,'seed':seed,**x} for x in reference['columns'])
            csv_write(out / 'metrics.csv',rows)
            print('RESULT',json.dumps(row),flush=True)
            del buffer, policy, critic, initial, next_pi, pi, qhat
    summary = []
    for algorithm in POLICIES:
        subset = [x for x in rows if x['algorithm'] == algorithm]
        result = dict(algorithm=algorithm,policies=len(subset))
        for key in ['wdr','fqe','wis','terminal_weight_ess']:
            values = [x[key] for x in subset if x[key] is not None]
            result[key+'_defined'] = len(values)
            result[key+'_mean'] = float(np.mean(values)) if values else None
            result[key+'_sd'] = float(np.std(values,ddof=1)) if len(values)>1 else None
            result[key+'_min'] = float(min(values)) if values else None
            result[key+'_max'] = float(max(values)) if values else None
        result['nonzero_terminal_paths_min'] = min(x['nonzero_terminal_paths'] for x in subset)
        result['nonzero_terminal_paths_max'] = max(x['nonzero_terminal_paths'] for x in subset)
        result['wdr_bootstrap_defined'] = sum(x['wdr_bootstrap_defined'] for x in subset)
        result['wis_bootstrap_defined'] = sum(x['wis_bootstrap_defined'] for x in subset)
        summary.append(result)
    csv_write(out / 'summary.csv',summary)
    csv_write(out / 'time_step_diagnostics.csv',traces)
    after = {str(p.relative_to(root)):sha(p) for p in inputs}
    if after != before:
        raise ValueError('Pre-existing inputs or source changed during run')
    write(out / 'input_preservation.json',dict(status='unchanged',files=before))
    write(out / 'completion_receipt.json',dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),
          trained_policies=20,fqe_fits=20,behavior_refitted=False,policy_mode='greedy',
          evaluation_episodes=episodes,evaluation_subjects=subjects,
          previous_inputs_preserved=True,validated_policy_value=False,
          decimal_reference_cases=20,max_absolute_equation_error=max(x['decimal_absolute_error'] or 0 for x in rows),
          source_sha256=source_hashes(SOURCE),
          artifact_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    print('COMPLETE',json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
