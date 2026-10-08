"""Frozen-stage retrospective Sepsis policy retraining experiment.

Each stage has a distinct input/output boundary. Validation locks the candidate
before test values are computed. Existing historical files are read only.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/sepsis_policy_retrain_20261007'
sys.path.insert(0, str(ROOT))
from util import active_source_paths
import joblib
from util import load_behavior_model
import numpy as np
import torch
from util import load_split
from model import full_action_proba
from util import episode_slices
from agent import fit_behavior
from metric import fit_fqe, predict_q
from model import FQECritic
from util import initial_state_indices
from metric import bootstrap_initial_values
from metric import evaluate_ope, policy_probs
from util import patient_split
from util import TrainScaler
from agent import constrained_probs, fit_policy
from metric import policy_diagnostics

FIELDS = ['state', 'next_state', 'action', 'reward', 'done']
FQE = dict(gamma=.98, epochs=100, min_epochs=30, patience=20,
           batch_size=4096, validation_fraction=.15, lr=1e-3, num_threads=4)
CRITICS = [(128,501),(256,502)]
BETAS = [.05,.1,.25]
SEEDS = [101,202,303]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def protocol():
    path = OUT/'protocol.json'
    if path.exists():
        return read(path)
    p = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='new retrospective policy learner; already examined test cohort; no claim of clinical benefit',
        reward='one terminal +/-1 from mortality_90d; intermediate rewards zero',gamma=.98,
        split='retain original test patients; exclude overlapping train patients; remaining patient train/selection-val 80/20',
        split_seed=20261007, normalization='physical-scale processed RAW CSV, training-state-only mean/std without clipping',
        unresolved_preprocessing='upstream interpolation and KNN imputation used future/cross-patient information; RAW does not undo it',
        behavior='internal patient fit/calibration/validation; RF raw/sigmoid and regularized multinomial raw/temperature; min internal validation logloss',
        alternate_behavior='independently partitioned sigmoid RF, seed314, fitted only on policy training patients',
        policy=dict(betas=BETAS,seeds=SEEDS,cql_coefficient=.1,hidden_dim=128,epochs=80,min_epochs=30,patience=15,
                    support_probability=.01,min_action_count=100,q_bounds=[-1,1],
                    actor='rare-action baseline mass fixed; exp(beta Q) tilt within supported actions',
                    interpretation='new stochastic constrained learner; not original greedy CQL'),
        primary_ope=dict(ratio_cap=None,output_clip=None,normalization='WIS terminal weights, WDR per-time weights; DR unnormalized'),
        validation_critics=[dict(hidden_dim=h,seed=s) for h,s in CRITICS],fqe=FQE,
        selection=dict(primary_ess_fraction=.10,primary_max_weight=.05,alternate_ess_fraction=.05,
            alternate_max_weight=.10,maximum_propensity_DR_and_WIS_span=.10,
            improvement='paired fixed-nuisance subject-bootstrap FQE difference lower percentile >0 for BOTH critic configurations',
            ranking='largest minimum paired lower percentile among candidates passing all gates',
            fallback='no supported improvement; still evaluate preselected best constrained candidate as exploratory and reference',
            caveat='gates and paired intervals are diagnostics, not positivity, unbiasedness, or safety guarantees',
            reference_identity='beta0 with the same fitted actor and denominator has ratio1/ESS=N algebraically; this does not validate true clinician propensities',
            interval_scope='validation paired percentiles condition on nuisance fits and are not selection-adjusted safety intervals'),
        final=dict(critics=[dict(hidden_dim=h,seed=s) for h,s in CRITICS+[(128,503)]],
                   conditional_bootstrap=1000,outer_training_patient_refits=30,inner_test_patient_resamples=100,
                   uncertainty='frozen target actor; behavior family fixed; FQE128 seed501 refit on each resampled train; failures retained; limited percentile precision'),
        retrospective_test='old test outcomes were previously analyzed; this is not an untouched external cohort')
    save(path,p)
    return p


def prepare():
    from util import recover_raw_states
    p = protocol()
    cache = OUT/'data'
    cache.mkdir(exist_ok=True)
    if (cache/'manifest.json').exists():
        manifest=read(cache/'manifest.json')
        for info in manifest['artifacts'].values():
            assert sha(info['path'])==info['sha256']
        return
    old = ROOT/'outputs/sepsis_ope_corrected_20261007/behavior'
    mapping=read(old/'subject_mapping_manifest.json')
    original={split:load_split(ROOT/'dataset/sepsis',split)[0] for split in ['train','test']}
    for split,fields in mapping['inputs'].items():
        for field,info in fields.items():
            assert sha(info['path'])==info['sha256']
    for name in ['train_groups.npy','test_groups.npy']:
        assert sha(old/name)==mapping['artifacts'][name]['sha256']
    tr_groups=np.load(old/'train_groups.npy')
    te_groups=np.load(old/'test_groups.npy')
    train_idx,val_idx,excluded=patient_split(tr_groups,te_groups,original['train']['reward'],original['train']['done'])
    raw,raw_report=recover_raw_states(original['train'],original['test'])
    scaler=TrainScaler().fit(raw['train']['state'][train_idx])
    np.savez(cache/'scaler.npz',mean=scaler.mean,scale=scaler.scale)
    masks={'train':('train',train_idx,tr_groups[train_idx]),
           'validation':('train',val_idx,tr_groups[val_idx]),
           'test':('test',np.arange(len(te_groups)),te_groups)}
    counts={}
    for name,(source,ix,groups) in masks.items():
        np.save(cache/(name+'_original_rows.npy'),ix)
        np.save(cache/(name+'_groups.npy'),groups)
        for field in FIELDS:
            values=scaler.transform(raw[source][field][ix]) if field in ['state','next_state'] else original[source][field][ix]
            np.save(cache/(name+'_'+field+'.npy'),values)
        counts[name]=dict(rows=len(ix),patients=len(np.unique(groups)),episodes=len(episode_slices(original[source]['done'][ix])),
                          terminal_rewards=np.unique(original[source]['reward'][ix],return_counts=True)[1].tolist())
    np.save(cache/'excluded_train_original_rows.npy',excluded)
    save(cache/'raw_recovery.json',raw_report)
    artifacts={path.name:dict(path=str(path),sha256=sha(path)) for path in sorted(cache.iterdir()) if path.is_file() and not path.name.startswith('._')}
    save(cache/'manifest.json',dict(protocol_sha256=sha(OUT/'protocol.json'),counts=counts,
        excluded=dict(rows=len(excluded),patients=len(np.unique(tr_groups[excluded]))),
        patient_overlap=0,raw=raw_report,artifacts=artifacts,source_mapping_sha256=sha(old/'subject_mapping_manifest.json')))
    print('Prepared',counts,flush=True)


def data(split):
    return {field:np.load(OUT/'data'/(split+'_'+field+'.npy')) for field in FIELDS},np.load(OUT/'data'/(split+'_groups.npy'))


def behavior():
    from agent import fit_behavior_candidates
    directory=OUT/'behavior'
    directory.mkdir(exist_ok=True)
    if (directory/'manifest.json').exists():
        return
    tr,groups=data('train')
    print('Fitting behavior candidates on training patients only',flush=True)
    selected,report,models=fit_behavior_candidates(tr['state'],tr['action'],tr['done'],groups,seed=42,n_jobs=4)
    partition=report.pop('_partition_indices')
    for key,ix in partition.items():
        np.save(directory/(key+'_rows.npy'),ix)
    joblib.dump(selected,directory/'selected.joblib',compress=3)
    # Alternate denominator is never substituted into the frozen actor.
    _,alternate,alt_report=fit_behavior(tr['state'],tr['action'],tr['done'],groups,random_seed=314,n_jobs=4)
    alt_report.pop('_partition_indices')
    joblib.dump(alternate,directory/'alternate.joblib',compress=3)
    save(directory/'manifest.json',dict(training=report,alternate=alt_report,
         selected_sha256=sha(directory/'selected.joblib'),alternate_sha256=sha(directory/'alternate.joblib'),
         train_data_sha256=sha(OUT/'data/train_state.npy')))
    print('Behavior selected',report.get('selected_behavior',report.get('selected_candidate')),flush=True)


def bprobs(split,alternate=False,next_state=False):
    # Test predictions are only requested by final/uncertainty stages.
    label='alternate' if alternate else 'selected'
    path=OUT/'behavior'/(label+'_'+split+('_next' if next_state else '')+'.npy')
    if not path.exists():
        model=load_behavior_model(OUT/'behavior'/(label+'.joblib'))
        d,_=data(split)
        np.save(path,full_action_proba(model,d['next_state' if next_state else 'state']))
    return np.load(path)


def candidates():
    return [dict(name='reference',beta=0.,seed=None,greedy=False)] + [
        dict(name='tilt'+str(beta)+'_seed'+str(seed),beta=beta,seed=seed,greedy=False) for beta in BETAS for seed in SEEDS]+[
        dict(name='greedy_seed'+str(seed),beta=0.,seed=seed,greedy=True) for seed in SEEDS]


def train():
    tr,groups=data('train')
    counts=np.bincount(np.asarray(tr['action']).reshape(-1).astype(int),minlength=25)
    np.save(OUT/'data/action_counts.npy',counts)
    for c in candidates()[1:]:
        directory=OUT/'policies'/c['name']
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/'fit.json').exists():
            continue
        print('Policy fit',c['name'],flush=True)
        def progress(log):
            if log['epoch']%10==0:
                print(c['name'],'epoch',log['epoch'],'balanced residual',log['validation']['balanced_mse'],flush=True)
        model,report=fit_policy(**tr,behavior_next=bprobs('train',next_state=True),groups=groups,
            beta=c['beta'],greedy=c['greedy'],seed=c['seed'],action_counts=counts,progress_callback=progress)
        torch.save(dict(state_dict=model.state_dict(),candidate=c,state_dim=43,hidden_dim=128,num_actions=25),directory/'policy.pt')
        save(directory/'fit.json',report)


def actor(c,split,next_state=False):
    b=bprobs(split,next_state=next_state)
    d,_=data(split)
    state=d['next_state' if next_state else 'state']
    if c['name']=='reference':
        return b,np.zeros_like(b)
    record=torch.load(OUT/'policies'/c['name']/'policy.pt',map_location='cpu',weights_only=True)
    model=FQECritic(43,25,128)
    model.load_state_dict(record['state_dict'])
    model.eval()
    q=predict_q(model,state)
    if c['greedy']:
        return policy_probs(q,mode='greedy'),q
    return constrained_probs(q,b,c['beta'],np.load(OUT/'data/action_counts.npy')),q


def fit_critic(c,hidden,seed,directory,train_data=None,groups=None,next_probs=None):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'fit.json').exists():
        record=torch.load(directory/'critic.pt',map_location='cpu',weights_only=True)
        model=FQECritic(43,25,hidden)
        model.load_state_dict(record['state_dict'])
        model.eval()
        return model
    if train_data is None:
        train_data,groups=data('train')
    if next_probs is None:
        next_probs=actor(c,'train',next_state=True)[0]
    print('FQE',c['name'],hidden,seed,flush=True)
    model,report=fit_fqe(**train_data,next_policy_probs=next_probs,groups=groups,hidden_dim=hidden,seed=seed,**FQE)
    torch.save(dict(state_dict=model.state_dict(),hidden_dim=hidden,seed=seed),directory/'critic.pt')
    save(directory/'fit.json',report)
    return model


def evaluate(c,split,critic,bootstrap=0,alternate=False,return_details=False):
    d,groups=data(split)
    target,_=actor(c,split)
    next_target,_=actor(c,split,next_state=True)
    q,nextq=predict_q(critic,d['state']),predict_q(critic,d['next_state'])
    starts=initial_state_indices(d['done'])
    initial=np.sum(target[starts]*q[starts],axis=1)
    fqe=bootstrap_initial_values(initial,groups=groups[starts],n_bootstrap=bootstrap)
    try:
        result=evaluate_ope(d['action'],d['reward'],d['done'],target,bprobs(split,alternate),q,
            gamma=.98,next_target_probs=next_target,next_q_values=nextq,n_bootstrap=bootstrap,
            episode_groups=groups[starts],return_episode_details=return_details,
            metadata=dict(candidate=c,denominator='alternate' if alternate else 'selected',actor_frozen=True))
    except (ValueError,FloatingPointError) as error:
        result=dict(status='undefined_ope',reason=str(error),dr=None,wdr=None,wis=None,
            weights=dict(trajectory_ess=None,trajectory_ess_fraction=None,maximum_normalized_trajectory_weight=None))
    result['fqe']=fqe
    return result,initial


def gates(primary,alternate,p):
    g=p['selection']
    def finite(r):
        return all(r[x] is not None and np.isfinite(r[x]) for x in ['dr','wis','wdr'])
    def weightcheck(r,key,bound,greater):
        value=r['weights'][key]
        return value is not None and np.isfinite(value) and (value>=bound if greater else value<=bound)
    checks=dict(primary_finite=finite(primary),alternate_finite=finite(alternate),
        primary_ess=weightcheck(primary,'trajectory_ess_fraction',g['primary_ess_fraction'],True),
        primary_max_weight=weightcheck(primary,'maximum_normalized_trajectory_weight',g['primary_max_weight'],False),
        alternate_ess=weightcheck(alternate,'trajectory_ess_fraction',g['alternate_ess_fraction'],True),
        alternate_max_weight=weightcheck(alternate,'maximum_normalized_trajectory_weight',g['alternate_max_weight'],False))
    checks['propensity_sensitivity']=finite(primary) and finite(alternate) and max(abs(primary[x]-alternate[x]) for x in ['dr','wis'])<=g['maximum_propensity_DR_and_WIS_span']
    return dict(passed=all(checks.values()),checks=checks)


def validate():
    if (OUT/'selection.json').exists():
        return
    p=protocol()
    d,groups=data('validation')
    starts=initial_state_indices(d['done'])
    references={}
    results=[]
    for c in candidates():
        row=dict(candidate=c,critics=[])
        for hidden,seed in CRITICS:
            directory=OUT/'validation'/c['name']/('h'+str(hidden)+'_s'+str(seed))
            critic=fit_critic(c,hidden,seed,directory)
            primary,initial=evaluate(c,'validation',critic,bootstrap=1000)
            alternate,_=evaluate(c,'validation',critic,alternate=True)
            if c['name']=='reference':
                references[(hidden,seed)]=initial
            difference=bootstrap_initial_values(initial-references[(hidden,seed)],groups=groups[starts],n_bootstrap=1000)
            entry=dict(hidden_dim=hidden,seed=seed,primary=primary,alternate=alternate,
                       paired_fqe_minus_reference=difference,gate=gates(primary,alternate,p))
            save(directory/'evaluation.json',entry)
            row['critics'].append(entry)
        row['minimum_paired_lower']=min(x['paired_fqe_minus_reference']['low'] for x in row['critics'])
        row['all_gates_passed']=all(x['gate']['passed'] for x in row['critics'])
        row['improvement_gate_passed']=row['minimum_paired_lower']>0
        row['eligible']=not c['greedy'] and c['name']!='reference' and row['all_gates_passed'] and row['improvement_gate_passed']
        results.append(row)
        save(OUT/'validation_results.json',dict(results=results))
        print('Validation',c['name'],'LCB',row['minimum_paired_lower'],'support gates',row['all_gates_passed'],flush=True)
    constrained=[r for r in results if r['candidate']['name']!='reference' and not r['candidate']['greedy']]
    eligible=[r for r in constrained if r['eligible']]
    winner=max(eligible or constrained,key=lambda r:r['minimum_paired_lower'])
    selected=dict(protocol_sha256=sha(OUT/'protocol.json'),data_manifest_sha256=sha(OUT/'data/manifest.json'),
        validation_results_sha256=sha(OUT/'validation_results.json'),
        behavior_sha256=sha(OUT/'behavior/selected.joblib'),
        policy_sha256={c['name']:sha(OUT/'policies'/c['name']/'policy.pt') for c in candidates()[1:]},
        action_counts_sha256=sha(OUT/'data/action_counts.npy'),
        status='diagnostic_gates_passed' if eligible else 'no_supported_improvement',
        selected_policy=winner['candidate'] if eligible else candidates()[0],
        exploratory_candidate=winner['candidate'],
        comparator=dict(name='greedy_seed101',beta=0.,seed=101,greedy=True),
        candidate_minimum_paired_lower=winner['minimum_paired_lower'],candidate_support_gates=winner['all_gates_passed'],
        test_values_used_for_selection=False,locked_utc=datetime.now(timezone.utc).isoformat())
    save(OUT/'selection.json',selected)
    print('Locked selection',selected,flush=True)


def final():
    selection=read(OUT/'selection.json')
    assert selection['validation_results_sha256']==sha(OUT/'validation_results.json')
    assert selection['protocol_sha256']==sha(OUT/'protocol.json')
    assert selection['behavior_sha256']==sha(OUT/'behavior/selected.joblib')
    assert selection['action_counts_sha256']==sha(OUT/'data/action_counts.npy')
    for name,digest in selection['policy_sha256'].items():
        assert sha(OUT/'policies'/name/'policy.pt')==digest
    chosen=[candidates()[0],selection['exploratory_candidate'],selection['comparator']]
    results=[]
    for c in chosen:
        role='fitted_behavior_reference' if c['name']=='reference' else ('new_greedy_comparator' if c['greedy'] else
            ('validation_selected_candidate' if selection['status']=='diagnostic_gates_passed' else 'exploratory_candidate_not_supported'))
        row=dict(candidate=c,role=role,critics=[])
        for hidden,seed in CRITICS+[(128,503)]:
            directory=OUT/'final'/c['name']/('h'+str(hidden)+'_s'+str(seed))
            # Fixed policy fits are identical to validation fits; reuse without test tuning.
            source=OUT/'validation'/c['name']/('h'+str(hidden)+'_s'+str(seed))
            critic=fit_critic(c,hidden,seed,source if source.exists() else directory)
            directory.mkdir(parents=True,exist_ok=True)
            primary,_=evaluate(c,'test',critic,bootstrap=1000,return_details=True)
            alternate,_=evaluate(c,'test',critic,alternate=True)
            qactor=actor(c,'test')[1]
            diagnosis=policy_diagnostics(qactor,actor(c,'test')[0],bprobs('test'),np.load(OUT/'data/action_counts.npy'))
            entry=dict(hidden_dim=hidden,seed=seed,primary=primary,alternate=alternate,
                gate=gates(primary,alternate,protocol()),actor_diagnostics=diagnosis,critic_source=str(source if source.exists() else directory))
            save(directory/'evaluation.json',entry)
            row['critics'].append(entry)
            print('Final',c['name'],hidden,seed,{x:primary[x] for x in ['dr','wdr','wis']},'FQE',primary['fqe']['value'],'ESS',primary['weights']['trajectory_ess'],flush=True)
        results.append(row)
        save(OUT/'final_results.json',dict(selection_sha256=sha(OUT/'selection.json'),results=results))


def uncertainty():
    from agent import fit_selected_behavior
    selection=read(OUT/'selection.json')
    c=selection['exploratory_candidate']
    tr,groups=data('train')
    te,tgroups=data('test')
    train_next=actor(c,'train',next_state=True)[0]
    target=actor(c,'test')[0]
    next_target=actor(c,'test',next_state=True)[0]
    selected_name=read(OUT/'behavior/manifest.json')['training']['selected_behavior']
    unique=np.unique(groups)
    episodes_by_group={int(g):[] for g in unique}
    for rows in episode_slices(tr['done']):
        episodes_by_group[int(groups[rows.start])].append(rows)
    rng=np.random.default_rng(73001)
    starts=initial_state_indices(te['done'])
    directory=OUT/'uncertainty'
    directory.mkdir(exist_ok=True)
    all_results=[]
    for draw in range(30):
        drawn=rng.choice(unique,size=len(unique),replace=True)
        ix=np.concatenate([np.concatenate([np.arange(s.start,s.stop) for s in episodes_by_group[int(g)]]) for g in drawn])
        dest=directory/('draw'+str(draw).zfill(3))
        dest.mkdir(exist_ok=True)
        path=dest/'result.json'
        if path.exists():
            result=read(path)
            assert result['draw_sha256']==hashlib.sha256(drawn.tobytes()).hexdigest()
            if result['status']=='started':
                result.update(status='interrupted_failure',reason='Interrupted first attempt; no hidden retry')
                save(path,result)
            all_results.append(result)
            continue
        print('Nuisance refit',draw+1,'/30',flush=True)
        sample={field:values[ix] for field,values in tr.items()}
        result=dict(draw=draw,draw_sha256=hashlib.sha256(drawn.tobytes()).hexdigest(),policy=c,
                    selected_behavior=selected_name,actor_refitted=False,status='started',
                    role='validation_selected_candidate' if selection['status']=='diagnostic_gates_passed' else 'exploratory_candidate_not_supported')
        save(path,result)
        try:
            model=fit_critic(c,128,501,dest,train_data=sample,groups=groups[ix],next_probs=train_next[ix])
            q,nextq=predict_q(model,te['state']),predict_q(model,te['next_state'])
            initial=np.sum(target[starts]*q[starts],axis=1)
            result['fqe']=bootstrap_initial_values(initial,groups=tgroups[starts],n_bootstrap=100,seed=74000+draw,return_samples=True)
        except (ValueError,FloatingPointError,RuntimeError) as error:
            result.update(status='undefined_fqe',reason=str(error))
            save(path,result)
            all_results.append(result)
            continue
        try:
            bmodel,b_report=fit_selected_behavior(sample['state'],sample['action'],sample['done'],groups[ix],selected_name,seed=75000+draw,n_jobs=4)
            bp=full_action_proba(bmodel,te['state'])
            ope=evaluate_ope(te['action'],te['reward'],te['done'],target,bp,q,gamma=.98,
                next_target_probs=next_target,next_q_values=nextq,n_bootstrap=100,seed=74000+draw,
                episode_groups=tgroups[starts],return_bootstrap_samples=True)
            result.update(status='defined',ope=ope,behavior_refit=b_report)
        except (ValueError,FloatingPointError) as error:
            result.update(status='undefined_ope',reason=str(error))
        save(path,result)
        all_results.append(result)
    def interval(values,total):
        valid=[v for v in values if v is not None and np.isfinite(v)]
        return dict(low=float(np.quantile(valid,.025)) if valid else None,
            high=float(np.quantile(valid,.975)) if valid else None,finite=len(valid),attempted=total,
            conditional_on_defined=len(valid)!=total)
    summary=dict(outer_draws=30,inner_draws=100,policy=c,actor_refitted=False,
        status_counts={status:sum(r['status']==status for r in all_results) for status in ['defined','undefined_ope','undefined_fqe','interrupted_failure']},
        intervals={'fqe':interval([v for r in all_results if 'fqe' in r for v in r['fqe']['samples']],3000)},
        interpretation='30 nuisance fits, not 3000 independent fits; approximate percentiles, frozen actor/scaler, historical imputation and systematic bias omitted')
    for metric in ['dr','wdr','wis']:
        values=[v for r in all_results if r['status']=='defined' for v in r['ope']['bootstrap']['samples'][metric]]
        summary['intervals'][metric]=interval(values,3000)
    save(directory/'summary.json',summary)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('stage',choices=['prepare','behavior','train','validate','final','uncertainty','all'])
    args=parser.parse_args()
    global OUT
    OUT = args.output_dir.expanduser().resolve()
    historical = ROOT / 'outputs/sepsis_policy_retrain_20261007'
    if OUT == historical or OUT.is_relative_to(historical):
        raise ValueError('Use a distinct experiment output directory; completed artifacts are read-only')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    protocol()
    sources=active_source_paths(ROOT)+[OUT/'protocol.json']
    binding=dict(sources={str(path):sha(path) for path in sources})
    frozen=OUT/'runtime_sources.json'
    if frozen.exists():
        assert read(frozen)==binding,'Frozen runtime source changed; preserve this run and start a distinct experiment'
    else:
        save(frozen,binding)
    for name,fn in [('prepare',prepare),('behavior',behavior),('train',train),('validate',validate),('final',final),('uncertainty',uncertainty)]:
        if args.stage in [name,'all']:
            fn()


if __name__=='__main__':
    main()
