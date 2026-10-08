#!/usr/bin/env python3
"""Prospective MIMIC-III Heparin BCQ tuning and single-checkpoint OPE."""
import argparse
import csv
import hashlib
import itertools
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util import ArrayTrainingBuffer
from agent import fit_behavior
from model import full_action_proba
from util import episode_slices
from util import sha256_file
from agent import HeparinBCQ
from model import HeparinFQECritic
from metric import fit_heparin_fqe
from metric import predict_q
from util import initial_state_indices
from agent import frozen_policy_arrays
from metric import evaluate_policy_arrays
from metric import evaluate_ope
from prepare_heparin_subjects import recover
sys.path.insert(0, str(ROOT / 'tests'))
from wdr_reference import reference_wdr


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def prob(model, state):
    return full_action_proba(model, state, num_actions=6)


def metrics(actions,probs):
    a=np.asarray(actions,dtype=int).reshape(-1)
    return {'log_loss':float(log_loss(a,probs,labels=np.arange(6))),
            'accuracy':float((probs.argmax(1)==a).mean()),
            'brier':float(np.mean(np.sum((probs-np.eye(6)[a])**2,axis=1))),
            'recorded_action_min_probability':float(probs[np.arange(len(a)),a].min())}


def make_policy(config, checkpoint=None):
    policy=HeparinBCQ(**config)
    if checkpoint is not None:
        policy.Q.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict'])
    policy.Q.eval()
    return policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--resume-dir',type=Path,help='Reuse complete actor runs only; refit every validation critic')
    args=parser.parse_args();root=args.project_root.resolve();out=args.output_dir.resolve()
    assert root==ROOT
    if out.exists():raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True)
    arrays,groups,stays,keep,mapping=recover(root,root.parent/'heparin_RL/preprocessing/mimic3/demog.csv')
    write(out/'subject_mapping.json',mapping)
    np.savez_compressed(out/'row_groups.npz',train_subjects=groups['train'],test_subjects=groups['test'],
                        train_stays=stays['train'],test_stays=stays['test'],train_keep=keep)
    unique=np.unique(groups['train'][keep]);rng=np.random.default_rng(20261007)
    val_subjects=rng.permutation(unique)[:int(np.ceil(.2*len(unique)))]
    val_mask=keep & np.isin(groups['train'],val_subjects)
    fit_mask=keep & ~val_mask
    for sl in episode_slices(arrays['train']['done']):
        assert np.all(fit_mask[sl]==fit_mask[sl.start]) and np.all(val_mask[sl]==val_mask[sl.start])
    train={k:v[fit_mask] for k,v in arrays['train'].items()}
    val={k:v[val_mask] for k,v in arrays['train'].items()}
    tg,vg=groups['train'][fit_mask],groups['train'][val_mask]
    assert not np.intersect1d(tg,vg).size and not np.intersect1d(tg,groups['test']).size
    np.savez_compressed(out/'selection_split.npz',actor_train_mask=fit_mask,policy_validation_mask=val_mask)
    input_files=[root/'dataset/heparin'/f'{split}_{k}.npy' for split in ['train','test']
                 for k in ['state','next_state','action','reward','done','BC_prob']]
    input_files += [Path(mapping['cohort_path']),Path(mapping['demog_path'])]
    input_files += [root/k for k in ['agent.py', 'agent.py', 'agent.py', 'agent.py', 'agent.py', 'model.py', 'agent.py', 'agent.py', 'agent.py', 'util.py', 'agent.py', 'metric.py', 'model.py', 'metric.py', 'metric.py', 'metric.py', 'model.py', 'util.py', 'util.py', 'agent.py', 'metric.py', 'metric.py', 'metric.py', 'agent.py', 'metric.py', 'model.py', 'util.py', 'util.py', 'util.py', 'util.py', 'util.py', 'util.py', 'scripts/tune_heparin_mimic3_bcq.py', 'scripts/prepare_heparin_subjects.py', 'tests/wdr_reference.py']]
    input_files += sorted((root/'pth').glob('heparin_BCQ_*.pth'))
    before={str(p):sha256_file(p) for p in input_files}
    grid=[(1e-6,.3)]+list(itertools.product([1e-5,1e-4],[.1,.3,.5]))
    reused={}
    if args.resume_dir:
        previous=args.resume_dir.resolve()
        with np.load(previous/'selection_split.npz') as prior:
            np.testing.assert_array_equal(prior['actor_train_mask'],fit_mask)
            np.testing.assert_array_equal(prior['policy_validation_mask'],val_mask)
        for lr,threshold in grid:
            for seed in [42,43,44]:
                name=f'bcq_lr{lr:g}_threshold{threshold:g}_seed{seed}'
                paths=[previous/'candidates'/name/f'update{u}.pth' for u in [1500,5000]]
                if all(p.is_file() for p in paths):
                    reused[name]=paths
                    input_files.extend(paths)
        before={str(p):sha256_file(p) for p in input_files}
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(), 'dataset':'MIMIC-III Heparin only',
              'train_rows':len(tg),'train_subjects':len(np.unique(tg)),
              'validation_rows':len(vg),'validation_subjects':len(np.unique(vg)),
              'test_rows':len(groups['test']),'test_subjects':len(np.unique(groups['test'])),
              'policy_seeds':[42,43,44], 'grid_lr_threshold':grid,'checkpoint_updates':[1500,5000],
              'actor_runs':len(grid)*3,'candidate_checkpoints':len(grid)*3*2,
              'architecture':'unchanged repository BCQNet, hidden1024, ReLU, 16 states, 6 actions',
              'actor_selection':'maximum external validation FQE initial-state mean; ties prefer fewer updates then candidate name',
              'validation_fqe':'train-only critic seed42, max100/min50 epochs; internal patient15% Bellman-MSE holdout',
              'target_policy':'single greedy BCQ with imitation threshold mask; no checkpoint ensemble',
              'gamma':.98,'actor_batch_size':64,'hard_target_update':25,'weight_decay':1e-5,
              'fixes':['restore nonzero fixed optimizer LR','freeze target BatchNorm buffers between hard copies','convert original done=0-terminal to standard 1-terminal in memory'],
              'behavior':'200-tree RF fit60%/sigmoid calibration20%/diagnostic holdout20% of actor-training patients only; calibrated model fixed primary',
              'primary_ratio_cap':None,'probability_floor':None,'output_clipping':None,
              'bootstrap_draws':1000,'bootstrap_unit':'whole test patient','bootstrap_scope':'conditional on selected actor and fitted nuisances',
              'preprocessing':'original states/actions/rewards/splits retained; remove overlapping test patients and set aside selection patients in memory',
              'test_not_used_for_policy_or_critic_selection':True,'test_is_historical_previously_analyzed':True,
              'inputs_sha256':before}
    protocol['reused_complete_actor_runs']=list(reused)
    protocol['critic_output']='50*tanh(raw_logits/50); unit gradient near zero; theoretical return bound'
    write(out/'protocol.json',protocol)
    print('PROTOCOL',json.dumps({k:protocol[k] for k in ['train_rows','train_subjects','validation_rows','validation_subjects','test_rows','actor_runs','candidate_checkpoints']}),flush=True)
    torch.set_num_threads(1)
    behavior_dir=out/'behavior';behavior_dir.mkdir()
    rf, calibrated, fitting = fit_behavior(train['state'], train['action'], train['done'],
        groups=tg, random_seed=53, num_actions=6, selection='calibrated', require_all_actions=True)
    part=fitting.pop('_partition_indices')
    a=train['action'].reshape(-1).astype(int)
    behavior_report={'primary':'frozen sigmoid calibrated RF','partitions':{k:{'rows':len(v),'patients':len(np.unique(tg[v]))} for k,v in part.items()},
                     'raw_validation':metrics(a[part['validation']],prob(rf,train['state'][part['validation']])),
                     'calibrated_validation':metrics(a[part['validation']],prob(calibrated,train['state'][part['validation']])),
                     'rf_parameters':rf.get_params(),'saved_probability_floor':None,
                     'note':'log-loss library clips only for diagnostic reporting; saved propensities are untouched'}
    write(behavior_dir/'fit.json',behavior_report);joblib.dump(calibrated,behavior_dir/'calibrated.joblib')
    np.savez_compressed(behavior_dir/'partitions.npz',**part)
    print('BEHAVIOR',json.dumps(behavior_report['calibrated_validation']),flush=True)
    candidates=[]
    for lr,threshold in grid:
        for seed in protocol['policy_seeds']:
            name=f'bcq_lr{lr:g}_threshold{threshold:g}_seed{seed}'
            d=out/'candidates'/name;d.mkdir(parents=True)
            cfg=dict(num_actions=6,state_dim=16,device='cpu',discount=.98,optimizer='Adam',
                     optimizer_parameters={'lr':lr,'weight_decay':1e-5},use_polyak_target_update=False,
                     target_update_frequency=25,tau=.005,hidden_node=1024,activation='relu',
                     bcq_threshold=threshold,max_timesteps=5000)
            torch.manual_seed(seed);policy=HeparinBCQ(**cfg)
            buffer=ArrayTrainingBuffer(train,seed=seed,batch_size=64)
            initial={k:v.detach().clone() for k,v in policy.Q.named_parameters()}
            begun=time.perf_counter()
            print('TRAIN',name,flush=True)
            for update in range(1,5001):
                if name not in reused:
                    policy.train(buffer)
                    if update%500==0:print('UPDATE',name,update,round(time.perf_counter()-begun,2),flush=True)
                if update not in protocol['checkpoint_updates']:continue
                policy.Q.eval();cp=d/f'update{update}.pth'
                if name in reused:
                    source=reused[name][0 if update==1500 else 1]
                    saved=torch.load(source,map_location='cpu',weights_only=False)
                    assert saved['config']==cfg and saved['updates']==update and saved['seed']==seed
                    policy.Q.load_state_dict(saved['model_state_dict']);delta=saved['parameter_l2_change']
                    shutil.copyfile(source,cp)
                    print('REUSED_ACTOR',name,update,flush=True)
                else:
                    delta=sum(float((v.detach()-initial[k]).square().sum()) for k,v in policy.Q.named_parameters())**.5
                    torch.save({'model_state_dict':policy.Q.state_dict(),'config':cfg,'seed':seed,'updates':update,
                                'parameter_l2_change':delta,'protocol_sha256':sha256_file(out/'protocol.json')},cp)
                assert delta>0 and all(torch.isfinite(v).all() for v in policy.Q.parameters())
                next_pi=frozen_policy_arrays('BCQ',policy,train['next_state'],mode='greedy')
                critic,fit=fit_heparin_fqe(train,next_pi,tg,seed=42)
                pi=frozen_policy_arrays('BCQ',policy,val['state'],mode='greedy')
                q=predict_q(critic,val['state'],num_threads=1)
                starts=initial_state_indices(val['done']);v=(pi[starts]*q[starts]).sum(1)
                ci=bootstrap_initial_values(v,groups=vg[starts],n_bootstrap=400,seed=20261007)
                row={'candidate':name+f'_update{update}','learning_rate':lr,'threshold':threshold,'seed':seed,
                     'updates':update,'validation_fqe':float(v.mean()),'validation_fqe_low':ci['low'],'validation_fqe_high':ci['high'],
                     'critic_validation_bellman_mse':fit['selected_validation']['bellman_mse'],
                     'critic_epoch':fit['selected_epoch'],'actor_parameter_l2_change':delta,
                     'checkpoint':cp.relative_to(out).as_posix(),'checkpoint_sha256':sha256_file(cp)}
                write(d/f'update{update}_fqe_fit.json',fit);torch.save(critic.state_dict(),d/f'update{update}_fqe.pt')
                candidates.append(row);csv_write(out/'validation_candidates.csv',candidates)
                print('CANDIDATE',json.dumps(row),flush=True)
            del policy,buffer,initial,critic
    best=sorted(candidates,key=lambda r:(-r['validation_fqe'],r['updates'],r['candidate']))[0]
    selected=out/'selected';selected.mkdir()
    shutil.copyfile(out/best['checkpoint'],selected/'policy_best.pth')
    chosen_cp=torch.load(selected/'policy_best.pth',map_location='cpu',weights_only=False)
    policy=make_policy(chosen_cp['config'],selected/'policy_best.pth')
    write(selected/'selection.json',{'status':'frozen before test OPE','selected':best,'candidates':len(candidates),
                                    'selection_rule':protocol['actor_selection'],'no_actor_refit_after_selection':True,
                                    'checkpoint_sha256':sha256_file(selected/'policy_best.pth')})
    print('SELECTED',json.dumps(best),flush=True)
    test=arrays['test'];eg=groups['test'];starts=initial_state_indices(test['done']);episode_groups=eg[starts]
    pi=frozen_policy_arrays('BCQ',policy,test['state'],mode='greedy')
    next_pi=frozen_policy_arrays('BCQ',policy,test['next_state'],mode='greedy')
    train_next_pi=frozen_policy_arrays('BCQ',policy,train['next_state'],mode='greedy')
    np.testing.assert_array_equal(pi,frozen_policy_arrays('BCQ',policy,test['state'],mode='greedy',batch_size=257))
    b=prob(calibrated,test['state']);np.save(selected/'behavior_test_probs.npy',b)
    results=[]
    for critic_seed in [42,43,44]:
        if critic_seed==42:
            candidate_dir=(out/best['checkpoint']).parent
            fit=json.loads((candidate_dir/f"update{best['updates']}_fqe_fit.json").read_text())
            critic=HeparinFQECritic(16,6,128,1/(1-.98))
            critic.load_state_dict(torch.load(candidate_dir/f"update{best['updates']}_fqe.pt",map_location='cpu',weights_only=True));critic.eval()
        else:critic,fit=fit_heparin_fqe(train,train_next_pi,tg,seed=critic_seed)
        assert fit['fixed_next_policy_sha256_float32']==hashlib.sha256(np.ascontiguousarray(train_next_pi,dtype=np.float32).tobytes()).hexdigest()
        torch.save(critic.state_dict(),selected/f'fqe_seed{critic_seed}.pt');write(selected/f'fqe_seed{critic_seed}_fit.json',fit)
        q=predict_q(critic,test['state'],num_threads=1);nq=predict_q(critic,test['next_state'],num_threads=1)
        ev=evaluate_policy_arrays(test,pi,b,q,gamma=.98,
                        next_target_probs=next_pi,next_q_values=nq,ratio_cap=None,n_bootstrap=1000,
                        seed=20261007,episode_groups=episode_groups,return_bootstrap_samples=True,
                        metadata={'dataset':'Heparin MIMIC-III','actor':best['candidate'],'critic_seed':critic_seed,'checkpoint':sha256_file(selected/'policy_best.pth')})
        fqe=ev.pop('fqe')
        ref=reference_wdr(test['action'].reshape(-1),test['reward'].reshape(-1),test['done'].reshape(-1),pi,b,q,gamma=.98)
        if ref['value'] is None:assert ev['wdr'] is None;error=None
        else:error=abs(ref['value']-ev['wdr']);np.testing.assert_allclose(ref['value'],ev['wdr'],rtol=1e-10,atol=1e-10)
        write(selected/f'evaluation_seed{critic_seed}.json',{'ope':ev,'fqe':fqe,'decimal_wdr_verification':{'reference':ref['value'],'absolute_error':error,'undefined_time_step':ref['undefined_time_step']}})
        np.savez_compressed(selected/f'frozen_arrays_seed{critic_seed}.npz',test_target_probs=pi,test_q=q,test_next_q=nq,test_next_target_probs=next_pi)
        ci=ev['bootstrap']['intervals'];row={'critic_seed':critic_seed,'primary':critic_seed==42,'fqe':fqe['value'],'fqe_low':fqe['low'],'fqe_high':fqe['high']}
        for metric in ['dr','wdr','wis']:
            row.update({metric:ev[metric],metric+'_low':ci[metric]['low'],metric+'_high':ci[metric]['high'],metric+'_bootstrap_defined':ci[metric]['defined_finite_resamples']})
        row.update(trajectory_ess=ev['weights']['trajectory_ess'],nonzero_terminal_paths=len(starts)-ev['weights']['exact_zero_trajectory_weight_count'])
        results.append(row);csv_write(out/'ope_metrics.csv',results);print('OPE_RESULT',json.dumps(row),flush=True)
        if critic_seed==42:
            csv_write(out/'time_step_diagnostics.csv',ref['columns'])
            cap_rows=[]
            for cap in [5.,10.,20.,50.]:
                cap_ev=evaluate_ope(test['action'],test['reward'],test['done'],pi,b,q,gamma=.98,ratio_cap=cap,n_bootstrap=0)
                cap_rows.append({'step_ratio_cap':cap,'dr':cap_ev['dr'],'wdr':cap_ev['wdr'],'wis':cap_ev['wis'],'trajectory_ess':cap_ev['weights']['trajectory_ess']})
            csv_write(out/'ratio_clipping_sensitivity.csv',cap_rows)
    # Reload the chosen actor and critics; verify all cached test predictions.
    loaded=make_policy(chosen_cp['config'],selected/'policy_best.pth')
    np.testing.assert_array_equal(frozen_policy_arrays('BCQ',loaded,test['state'],mode='greedy'),pi)
    for seed in [42,43,44]:
        c=HeparinFQECritic(16,6,128,1/(1-.98));c.load_state_dict(torch.load(selected/f'fqe_seed{seed}.pt',weights_only=True,map_location='cpu'));c.eval()
        with np.load(selected/f'frozen_arrays_seed{seed}.npz') as cache:
            np.testing.assert_array_equal(predict_q(c,test['state'],num_threads=1),cache['test_q'])
    after={str(p):sha256_file(p) for p in input_files};assert before==after
    write(out/'input_preservation.json',{'status':'passed','all_input_hashes_identical_before_and_after':True,'inputs_sha256':after})
    write(out/'serialized_verification.json',{'status':'passed','actor_test_actions_match_all_rows':True,'critic_test_predictions_match_all_rows':True,'inference_batch_invariant':True,'selected_actor_count':1})
    artifacts={p.relative_to(out).as_posix():sha256_file(p) for p in out.rglob('*') if p.is_file() and not p.name.startswith('._')}
    write(out/'completion_receipt.json',{'status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
                                       'actor_runs':len(grid)*3,'candidate_checkpoints':len(candidates),'selected':best,
                                       'single_selected_checkpoint':True,'inputs_preserved':True,'artifact_sha256':artifacts})
    print('COMPLETE',str(out),flush=True)


if __name__=='__main__':main()
