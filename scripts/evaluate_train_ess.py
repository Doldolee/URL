"""Diagnose ESS on the existing actor-training cohort without any refitting.

Compare the default calibrated RF and the validation-ESS-selected kNN for the
same temperature-1 softmax actors. Complete training trajectories, including
behavior-fit/calibration/validation partitions, are reported separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import POLICIES, HeparinBCQ, frozen_policy_arrays
from metric import evaluate_ope
from model import full_action_proba
from util import episode_slices, load_behavior_model, source_hashes, verify_source_manifest
from scripts.compare_behavior_policies import load_cohort, trajectory_weight_diagnostics, sha, write, csv_write

SENSITIVITY = ROOT/'outputs/behavior_policy_sensitivity_20261007'


def summarize(rows):
    summaries = []
    for dataset in dict.fromkeys(r['dataset'] for r in rows):
        for behavior in dict.fromkeys(r['behavior'] for r in rows if r['dataset']==dataset):
            for scope in dict.fromkeys(r['scope'] for r in rows if r['dataset']==dataset):
                for algorithm in ['ALL20','DQN','DDQN','BCQ','CQL']:
                    selected = [r for r in rows if r['dataset']==dataset and r['behavior']==behavior
                        and r['scope']==scope and r['kind']=='matched_protocol'
                        and (algorithm=='ALL20' or r['algorithm']==algorithm)]
                    if not selected:
                        continue
                    summary = {'dataset':dataset,'behavior':behavior,'scope':scope,'algorithm':algorithm,
                               'policies':len(selected),'episodes':selected[0]['episodes'],
                               'subjects':selected[0]['subjects'],'rows':selected[0]['rows']}
                    for key in ['trajectory_ess','trajectory_ess_fraction','maximum_normalized_trajectory_weight',
                                'top10_normalized_trajectory_weight_share']:
                        values = [r[key] for r in selected if r.get(key) is not None]
                        summary.update({key+'_mean':float(np.mean(values)) if values else None,
                                        key+'_min':min(values) if values else None,
                                        key+'_max':max(values) if values else None,
                                        key+'_defined':len(values)})
                    summaries.append(summary)
    return summaries


def run(dataset, out):
    verify_source_manifest(ROOT)
    out.mkdir(parents=True,exist_ok=False)
    inputs = {}
    train, test, tg, eg, partition, seed, actions, raw, rf, tracked = load_cohort(dataset,inputs)
    previous = SENSITIVITY/dataset
    old = json.loads(tracked(previous/'protocol.json').read_text())
    selection = json.loads(tracked(previous/'selection.json').read_text())
    candidate = selection['selected_behavior']
    fitted = load_behavior_model(tracked(previous/'selected/behavior.joblib'))
    tracked(SENSITIVITY/'completion_receipt.json')
    scopes = {'train_total':np.arange(len(tg)), 'behavior_fit':partition['fit'],
              'behavior_calibration':partition['calibration'], 'behavior_validation':partition['validation']}
    cohort = {name:{'rows':len(index),'subjects':len(np.unique(tg[index])),
                   'episodes':len(episode_slices(train['done'][index]))} for name,index in scopes.items()}
    np.savez_compressed(out/'cohort.npz',done=train['done'].reshape(-1),groups=tg,
                        **{name:index for name,index in partition.items()})
    behavior_probabilities = {}
    for name,model in [('rf_sigmoid',rf),(candidate,fitted)]:
        print('BEHAVIOR_PREDICT',dataset,name,flush=True)
        probabilities = full_action_proba(model,train['state'],num_actions=actions)
        behavior_probabilities[name] = probabilities
        np.save(out/f'{name}_train_observed_probability.npy',probabilities[
            np.arange(len(tg)),train['action'].reshape(-1).astype(int)])
        saved_validation = np.load(tracked(previous/'candidates'/name/'validation_proba.npy'))
        np.testing.assert_allclose(probabilities[partition['validation']],saved_validation,rtol=1e-10,atol=1e-12)
    protocol = {'dataset':dataset,'created_utc':datetime.now(timezone.utc).isoformat(),
        'cohort':'same actual actor-training patients as previous fixed softmax experiment',
        'scope_counts':cohort,'behaviors':['rf_sigmoid',candidate],
        'policy_mode':'softmax','temperature':1.,'bcq_imitation_mask_applied':False,
        'actor_refits':0,'behavior_refits':0,'fqe_refits':0,'model_reselection':False,
        'probability_floor':None,'ratio_cap':None,'discount_applied_to_importance_weights':False,
        'ess_formula':'1/sum_i(w_i^2), w_i=softmax_i(sum_t(log(pi(a_t|s_t))-log(b(a_t|s_t))))',
        'per_decision_ess':'all trajectories remain in normalization after termination using absorbing ratio1',
        'fit_is_in_sample_for_behavior':True,'knn_fit_query_includes_fit_records_themselves':True,
        'validation_is_behavior_heldout_but_part_of_actor_training':True,
        'heparin_outer_policy_selection_patients_excluded_from_actor_train':dataset=='heparin'}
    write(out/'protocol.json',protocol)
    rows,step_rows = [],[]
    for spec in old['policies']:
        directory = ROOT/spec['previous_directory']
        artifacts = json.loads(tracked(directory/'artifacts.json').read_text())
        checkpoint = tracked(ROOT/artifacts['actor'])
        assert sha(checkpoint)==artifacts['actor_sha256']
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        config = saved.get('params',saved.get('config'))
        algorithm = spec['algorithm']
        policy = (HeparinBCQ if dataset=='heparin' and algorithm=='BCQ' else POLICIES[algorithm])(**config)
        policy.Q.load_state_dict(saved['model_state_dict']);policy.Q.eval()
        pi = frozen_policy_arrays(algorithm,policy,train['state'],mode='softmax',temperature=1.)
        np.save(out/(spec['policy']+'_train_observed_probability.npy'),
                pi[np.arange(len(tg)),train['action'].reshape(-1).astype(int)])
        previous_validation = np.load(tracked(previous/'validation_policies'/f"{spec['policy']}.npy"))
        np.testing.assert_allclose(pi[partition['validation']],previous_validation,rtol=1e-7,atol=1e-10)
        for behavior,b in behavior_probabilities.items():
            for scope,index in scopes.items():
                diag = trajectory_weight_diagnostics(train['action'][index],train['done'][index],pi[index],b[index])
                rows.append({'dataset':dataset,'behavior':behavior,'scope':scope,
                             **{k:spec[k] for k in ['policy','algorithm','seed','kind']},
                             **cohort[scope],**diag})
            # Reuse the established OPE engine only for weight diagnostics. Zero
            # reward/critic values are diagnostic placeholders, not policy values.
            result = evaluate_ope(train['action'],np.zeros(len(tg)),train['done'],pi,b,np.zeros_like(pi),
                                  n_bootstrap=0,ratio_cap=None)
            direct = rows[-4]
            np.testing.assert_allclose(result['weights']['trajectory_ess'],direct['trajectory_ess'],rtol=1e-10,atol=1e-10)
            detail_dir = out/'policies'/spec['policy'];detail_dir.mkdir(parents=True,exist_ok=True)
            write(detail_dir/(behavior+'_weights.json'),{'weights':result['weights'],
                'support_at_observed_states':result['support'],
                'policy_values_computed_for_reporting':False})
            lengths = np.array([sl.stop-sl.start for sl in episode_slices(train['done'])])
            for t,ess in enumerate(result['weights']['per_decision_ess_with_absorbing_padding']):
                step_rows.append({'dataset':dataset,'behavior':behavior,'policy':spec['policy'],
                    'algorithm':algorithm,'kind':spec['kind'],'step_zero_based':t,
                    'ess_with_absorbing_padding':ess,'episodes_in_denominator':len(lengths),
                    'active_episodes':int(np.sum(lengths>t))})
        csv_write(out/'policy_ess.csv',rows)
        print('POLICY_COMPLETE',dataset,spec['policy'],flush=True)
    csv_write(out/'per_step_ess.csv',step_rows)
    summaries = summarize(rows)
    csv_write(out/'summary.csv',summaries)
    old_rows = list(csv.DictReader(tracked(previous/'validation_policy_ess.csv').open()))
    for row in [r for r in rows if r['scope']=='behavior_validation']:
        expected = next(r for r in old_rows if r['candidate']==row['behavior'] and r['policy']==row['policy'])
        np.testing.assert_allclose(row['trajectory_ess'],float(expected['trajectory_ess']),rtol=1e-9,atol=1e-9)
    assert all(sha(ROOT/name)==digest for name,digest in inputs.items())
    write(out/'input_preservation.json',inputs)
    write(out/'completion_receipt.json',{'status':'complete','dataset':dataset,'policies':len(old['policies']),
        'actor_refits':0,'behavior_refits':0,'fqe_refits':0,'cohort':cohort,
        'validation_ess_reproduced':True,'terminal_ess_matches_ope_engine':True,
        'runtime_source_sha256':source_hashes(ROOT),'input_sha256_preserved':True,
        'artifact_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and not p.name.startswith('._')}})
    print('DATASET_COMPLETE',dataset,json.dumps([r for r in summaries if r['algorithm']=='ALL20']),flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['heparin','sepsis'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    with threadpool_limits(limits=4):
        run(args.dataset,args.output_dir.resolve()/args.dataset)


if __name__=='__main__':
    main()
