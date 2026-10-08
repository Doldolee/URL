"""Fixed MIMIC-III softmax actors: step-ratio versus cumulative-prefix cap sweeps.

No actor, propensity, or critic fitting and no cap selection. Reuses frozen,
hash-verified test arrays and FQE from the completed temperature-1 experiment.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from metric import evaluate_ope
from util import sha256_file, verify_source_manifest

BASE=ROOT/'outputs/mimic3_four_algorithm_softmax_ope_20261007'
CAPS=(2.,5.,10.,20.,50.,100.)
ALGORITHMS=('DQN','DDQN','BCQ','CQL')


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def csv_write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def direct_reference(actions,rewards,dones,pi,behavior,q,*,gamma=.98,
                     ratio_cap=None,cumulative_weight_cap=None):
    """Independent long-double direct products, no production log-weight helpers."""
    dtype=np.longdouble
    action=np.asarray(actions).reshape(-1).astype(int)
    r=np.asarray(rewards,dtype=dtype).reshape(-1)
    done=np.asarray(dones).reshape(-1)
    p=np.asarray(pi,dtype=dtype);b=np.asarray(behavior,dtype=dtype);q=np.asarray(q,dtype=dtype)
    ends=np.flatnonzero(done);starts=np.r_[0,ends[:-1]+1]
    n=len(starts);h=int(np.max(ends-starts+1))
    weight=np.empty((n,h),dtype=dtype)
    reward=np.zeros((n,h),dtype=dtype);qa=reward.copy();v=reward.copy()
    ratio=p[np.arange(len(action)),action]/b[np.arange(len(action)),action]
    if ratio_cap is not None:ratio=np.minimum(ratio,dtype(ratio_cap))
    for i,(start,end) in enumerate(zip(starts,ends)):
        sl=slice(start,end+1);length=end-start+1
        prefix=np.cumprod(ratio[sl],dtype=dtype)
        if cumulative_weight_cap is not None:prefix=np.minimum(prefix,dtype(cumulative_weight_cap))
        weight[i,:length]=prefix;weight[i,length:]=prefix[-1]
        reward[i,:length]=r[sl];qa[i,:length]=q[np.arange(start,end+1),action[sl]]
        v[i,:length]=np.sum(p[sl]*q[sl],axis=1)
    if not np.isfinite(weight).all():raise ValueError('Direct reference overflow')
    discount=np.power(dtype(gamma),np.arange(h))
    previous=np.column_stack((np.ones(n,dtype=dtype),weight[:,:-1]))
    dr=np.sum(discount*(weight*(reward-qa)+previous*v))/n
    totals=weight.sum(0)
    if np.any(totals==0):return {'dr':float(dr),'wis':None,'wdr':None,'ess':None,'step_ess':None}
    normalized=weight/totals
    prev_norm=np.column_stack((np.full(n,dtype(1)/n,dtype=dtype),normalized[:,:-1]))
    wdr=np.sum(discount*np.sum(normalized*(reward-qa)+prev_norm*v,axis=0))
    returns=np.sum(discount*reward,axis=1)
    wis=np.sum(normalized[:,-1]*returns)
    ess=dtype(1)/np.sum(normalized[:,-1]**2)
    step_ess=dtype(1)/np.sum(normalized**2,axis=0)
    return {'dr':float(dr),'wdr':float(wdr),'wis':float(wis),'ess':float(ess),
            'step_ess':[float(x) for x in step_ess]}


def assert_reference(report,reference):
    errors={}
    for k in ['dr','wdr','wis']:
        if reference[k] is None:
            assert report[k] is None
            errors[k]=None
        else:
            np.testing.assert_allclose(report[k],reference[k],rtol=2e-10,atol=2e-10)
            errors[k]=abs(report[k]-reference[k])/max(1.,abs(reference[k]))
    np.testing.assert_allclose(report['weights']['trajectory_ess'],reference['ess'],rtol=2e-10,atol=2e-10)
    np.testing.assert_allclose(report['weights']['per_decision_ess_with_absorbing_padding'],reference['step_ess'],rtol=2e-10,atol=2e-10)
    return errors


def run_dataset(dataset,out,n_bootstrap):
    out.mkdir(parents=True,exist_ok=False)
    inputs={}
    baseline_receipt=json.loads((BASE/'completion_receipt.json').read_text())
    assert baseline_receipt['status']=='complete'
    def tracked(path):
        path=Path(path);key=str(path.relative_to(ROOT));digest=sha256_file(path)
        if key in inputs:assert inputs[key]==digest
        inputs[key]=digest
        if path.is_relative_to(BASE):
            rel=str(path.relative_to(BASE))
            if rel in baseline_receipt['artifact_sha256']:
                assert baseline_receipt['artifact_sha256'][rel]==digest,rel
        return path
    tracked(BASE/'completion_receipt.json')
    prior_protocol=json.loads(tracked(BASE/dataset/'protocol.json').read_text())
    with tracked(BASE/'policy_metrics.csv').open() as f:
        specs=[r for r in csv.DictReader(f) if r['dataset']==dataset]
    arrays={k:np.load(tracked(ROOT/'dataset'/dataset/f'test_{k}.npy')) for k in ['action','reward','done']}
    if dataset=='heparin':
        arrays['done']=1-arrays['done']
        source=ROOT/'outputs/heparin_mimic3_bcq_best_20261007'
        behavior=np.load(tracked(source/'selected/behavior_test_probs.npy'))
        with np.load(tracked(source/'row_groups.npz')) as g:groups=g['test_subjects']
        tracked(source/'behavior/calibrated.joblib')
    else:
        source=ROOT/'outputs/sepsis_ope_corrected_20261007/behavior'
        behavior=np.load(tracked(source/'calibrated_test_proba.npy'))
        groups=np.load(tracked(source/'test_groups.npy'))
        tracked(source/'rf_sigmoid_calibrated.joblib')
    done=arrays['done'].reshape(-1);ends=np.flatnonzero(done);starts=np.r_[0,ends[:-1]+1]
    for start,end in zip(starts,ends):assert np.all(groups[start:end+1]==groups[start])
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset':dataset,
        'policy_mode':'softmax','temperature':1.,'gamma':.98,'caps':list(CAPS),
        'scopes':['none','step_ratio','cumulative_prefix'],'cap_selection':None,
        'grid_fixed_before_evaluation':True,'percentile_thresholds_used':False,
        'actor_refits':0,'behavior_refits':0,'critic_refits':0,'probability_floor':None,
        'bias_variance_sensitivity_only':True,'original_cumulative_products_preserved_before_prefix_cap':True,
        'cumulative_cap_feedback':False,'absorbing_padding':True,'previous_weight_value_term':True,
        'n_bootstrap':n_bootstrap,'bootstrap_seed':20261007,
        'bootstrap_unit':'whole subject cluster','nuisance_models_held_fixed':True,
        'test_used_to_select_cap_or_policy':False,'historical_test':True,
        'policies':len(specs),'matched_policies':20,'episodes':len(starts),'subjects':len(np.unique(groups)),
        'transitions':len(done),'prior_protocol_sha256':sha256_file(BASE/dataset/'protocol.json')}
    write(out/'protocol.json',protocol)
    rows=[];steps=[];checks=[]
    for spec in specs:
        name=spec['policy'];directory=BASE/dataset/'policies'/name
        prior=json.loads(tracked(directory/'evaluation.json').read_text())
        assert prior['gamma']==.98 and prior['ratio_cap'] is None
        assert prior['bootstrap']['n_bootstrap']==n_bootstrap
        artifacts=json.loads(tracked(directory/'artifacts.json').read_text())
        assert sha256_file(tracked(ROOT/artifacts['actor']))==artifacts['actor_sha256']
        assert sha256_file(tracked(directory/'fqe.pt'))==artifacts['critic_sha256']
        cache=np.load(tracked(directory/'frozen_arrays.npz'))
        pi=cache['test_target_probs'];q=cache['test_q']
        assert np.all(pi>0)
        action=arrays['action'].reshape(-1).astype(int);ix=np.arange(len(action))
        raw_ratio=pi[ix,action]/behavior[ix,action]
        scenarios=[('none',None)]+[(s,c) for s in ['step_ratio','cumulative_prefix'] for c in CAPS]
        for scope,cap in scenarios:
            kw={'ratio_cap':cap} if scope=='step_ratio' else {'cumulative_weight_cap':cap} if scope=='cumulative_prefix' else {}
            tag='unclipped' if cap is None else scope+'_cap'+format(cap,'g')
            dest=out/'policies'/name/tag;dest.mkdir(parents=True)
            t=time.perf_counter()
            point=evaluate_ope(arrays['action'],arrays['reward'],done,pi,behavior,q,
                gamma=.98,n_bootstrap=0,**kw)
            no_change=scope=='none' or (scope=='step_ratio' and np.max(raw_ratio)<=cap) or (
                scope=='cumulative_prefix' and point['clipping']['capped_observed_prefix_count']==0)
            if no_change:
                for k in ['dr','wdr','wis']:np.testing.assert_allclose(point[k],prior[k],rtol=1e-12,atol=1e-12)
                report=point;report['bootstrap']=copy.deepcopy(prior['bootstrap'])
                report['bootstrap']['reused_identical_weight_reference']=True
            else:
                report=evaluate_ope(arrays['action'],arrays['reward'],done,pi,behavior,q,
                    gamma=.98,n_bootstrap=n_bootstrap,seed=20261007,episode_groups=groups[starts],
                    return_bootstrap_samples=True,**kw)
            report['fqe']=copy.deepcopy(prior['fqe'])
            report['metadata']={'dataset':dataset,'policy':name,'scope':scope,'cap':cap,
                                'frozen_nuisances':True,'cap_selected':False}
            ref=direct_reference(arrays['action'],arrays['reward'],done,pi,behavior,q,gamma=.98,**kw)
            errors=assert_reference(report,ref)
            weights=report['weights'];raw_ess=prior['weights']['trajectory_ess']
            row={k:spec[k] for k in ['dataset','algorithm','seed','policy','kind']}
            row.update(scope=scope,cap=cap,episodes=len(starts),subjects=len(np.unique(groups)),
                trajectory_ess=weights['trajectory_ess'],raw_trajectory_ess=raw_ess,
                ess_fraction=weights['trajectory_ess']/len(starts),
                max_terminal_weight=weights['maximum_normalized_trajectory_weight'],
                raw_max_terminal_weight=prior['weights']['maximum_normalized_trajectory_weight'],
                step_ess_min=min(weights['per_decision_ess_with_absorbing_padding']),
                step_ess_first=weights['per_decision_ess_with_absorbing_padding'][0],
                step_ratio_clipped_fraction=float(np.mean(raw_ratio>cap)) if scope=='step_ratio' else 0.,
                cumulative_prefix_clipped_fraction=report['clipping']['capped_observed_prefix_count']/len(done),
                cumulative_terminal_clipped_fraction=report['clipping']['capped_terminal_trajectory_count']/len(starts),
                fqe=prior['fqe']['value'],fqe_low=prior['fqe']['low'],fqe_high=prior['fqe']['high'])
            for k in ['dr','wdr','wis']:
                ci=report['bootstrap']['intervals'][k]
                row.update({k:report[k],k+'_low':ci['low'],k+'_high':ci['high'],
                    k+'_bootstrap_defined':ci['defined_finite_resamples'],k+'_delta_from_raw':report[k]-prior[k]})
            row['bootstrap_reused_identical_weights']=no_change
            write(dest/'evaluation.json',report)
            checks.append({'policy':name,'scope':scope,'cap':cap,'errors':errors,
                           'elapsed_seconds':time.perf_counter()-t})
            rows.append(row)
            for step,(ess,raw_step) in enumerate(zip(weights['per_decision_ess_with_absorbing_padding'],prior['weights']['per_decision_ess_with_absorbing_padding'])):
                steps.append({'dataset':dataset,'policy':name,'algorithm':spec['algorithm'],
                              'scope':scope,'cap':cap,'time_step':step,'ess':ess,'raw_ess':raw_step})
            print('RESULT',dataset,name,tag,'ESS',format(row['trajectory_ess'],'.5g'),
                  'WDR',format(row['wdr'],'.6g'),'WIS',format(row['wis'],'.6g'),flush=True)
        cache.close()
    csv_write(out/'policy_metrics.csv',rows);csv_write(out/'time_step_ess.csv',steps)
    write(out/'equation_verification.json',{'status':'passed','oracle':'longdouble direct products',
                                          'full_cohort_scenarios':len(checks),'checks':checks})
    preserved={k:sha256_file(ROOT/k)==v for k,v in inputs.items()}
    assert all(preserved.values())
    write(out/'input_preservation.json',{'status':'passed','inputs_sha256':inputs,'preserved':preserved})
    artifacts={str(p.relative_to(out)):sha256_file(p) for p in out.rglob('*') if p.is_file() and not p.name.startswith('._')}
    write(out/'completion_receipt.json',{'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),
        'dataset':dataset,'policies':len(specs),'scenarios':len(rows),'bootstrap_draws':n_bootstrap,
        'inputs_preserved':True,'artifact_sha256':artifacts})
    return rows


def summarize(rows):
    summaries=[]
    for dataset in ['heparin','sepsis']:
        for scope,cap in [('none',None)]+[(s,c) for s in ['step_ratio','cumulative_prefix'] for c in CAPS]:
            for algorithm in ['ALL20']+list(ALGORITHMS):
                selected=[r for r in rows if r['dataset']==dataset and r['scope']==scope and r['cap']==cap
                          and r['kind']=='matched_protocol' and (algorithm=='ALL20' or r['algorithm']==algorithm)]
                if not selected:continue
                row={'dataset':dataset,'scope':scope,'cap':cap,'algorithm':algorithm,'policies':len(selected)}
                for k in ['trajectory_ess','ess_fraction','dr','wdr','wis','fqe','max_terminal_weight']:
                    values=np.asarray([r[k] for r in selected],dtype=float)
                    row.update({k+'_mean':float(values.mean()),k+'_sd':float(values.std(ddof=1)),
                                k+'_min':float(values.min()),k+'_max':float(values.max())})
                summaries.append(row)
    return summaries


def build_report(out,rows):
    summary=summarize(rows);csv_write(out/'policy_metrics.csv',rows);csv_write(out/'algorithm_summary.csv',summary)
    ranks=[]
    for d in ['heparin','sepsis']:
        base=[r for r in summary if r['dataset']==d and r['scope']=='none' and r['algorithm']!='ALL20']
        for metric in ['wdr','wis','fqe']:
            prior_order=[r['algorithm'] for r in sorted(base,key=lambda x:-x[metric+'_mean'])]
            for scope,cap in [('none',None)]+[(s,c) for s in ['step_ratio','cumulative_prefix'] for c in CAPS]:
                group=[r for r in summary if r['dataset']==d and r['scope']==scope and r['cap']==cap and r['algorithm']!='ALL20']
                if not group:continue
                order=[r['algorithm'] for r in sorted(group,key=lambda x:-x[metric+'_mean'])]
                ranks.append({'dataset':d,'metric':metric,'scope':scope,'cap':cap,'order':' > '.join(order),
                              'unclipped_order':' > '.join(prior_order),'same_as_unclipped':order==prior_order})
    csv_write(out/'algorithm_rankings.csv',ranks)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for i,d in enumerate(['heparin','sepsis']):
        if not any(r['dataset']==d for r in summary):
            for ax in axes[i]:ax.set_visible(False)
            continue
        for alg_index,alg in enumerate(ALGORITHMS):
            base=next(r for r in summary if r['dataset']==d and r['scope']=='none' and r['algorithm']==alg)
            for j,key in enumerate(['trajectory_ess','wdr','wis']):
                for scope,style in [('cumulative_prefix','-'),('step_ratio','--')]:
                    values=[next(r for r in summary if r['dataset']==d and r['scope']==scope and r['cap']==c and r['algorithm']==alg)[key+'_mean'] for c in CAPS]
                    axes[i,j].plot(CAPS,values,style,color=f'C{alg_index}',marker='o',ms=3,label=alg+' '+('prefix' if scope=='cumulative_prefix' else 'step'))
                axes[i,j].axhline(base[key+'_mean'],color=f'C{alg_index}',linestyle=':',alpha=.45,lw=.8)
                axes[i,j].set_xscale('log');axes[i,j].set_xlabel('Fixed cap');axes[i,j].set_title(d+' / '+key)
                axes[i,j].grid(alpha=.2)
                if j==0:axes[i,j].set_yscale('log')
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('Fixed softmax policies: solid = prefix cap, dashed = step cap, dotted = unclipped',fontsize=11)
    fig.savefig(out/'clipping_sensitivity.png',dpi=180);plt.close(fig)
    lines=['# MIMIC-III softmax OPE: clipping 민감도 분석','',
        '기존 DQN/DDQN/BCQ/CQL 각5개 seed, 총40개 정책과 기존 Heparin BCQ best1개를 유지했습니다. '
        'Softmax temperature1, train-only calibrated RF, 기존 정책별 FQE, gamma.98을 그대로 사용했습니다. '
        'Actor·behavior·critic 학습 및 정책/상한 선정은 없습니다.','',
        '## 고정한 clipping 정의','',
        '- Step ratio: `prod_u min(pi(a_u|s_u)/b(a_u|s_u), c)`.',
        '- Cumulative prefix: `min(prod_u pi(a_u|s_u)/b(a_u|s_u), c)`; 원 누적곱을 먼저 계산하며 잘린 값을 다음 곱으로 전달하지 않습니다.',
        '- 상한2/5/10/20/50/100은 산출 전 고정했습니다. Test percentile이나 test ESS로 상한을 선택하지 않았습니다.',
        '- DR/WDR의 이전시점 V항에도 같은 이전 prefix를 사용합니다. 초기 rho=1, 초기 normalized weight=1/N이고 종료 후 최종weight를 absorbing 분모에 유지합니다.',
        '- WIS는 마지막 weight로 원래 discounted episode return을 정규화합니다. FQE 값과 기존 구간은 변경 없이 재사용합니다.','',
        '## 공통20개 정책 평균','',
        '정책 평균은 ensemble 가치가 아니며 알고리즘 표의 SD는 seed 간 편차입니다.','',
        '| Dataset | Scope | Cap | ESS mean | WDR mean | WIS mean |',
        '|---|---|---:|---:|---:|---:|']
    for r in summary:
        if r['algorithm']=='ALL20':lines.append(f"| {r['dataset']} | {r['scope']} | {r['cap'] or 'none'} | {r['trajectory_ess_mean']:.3f} | {r['wdr_mean']:.6f} | {r['wis_mean']:.6f} |")
    lines+=['','## 상한10의 알고리즘별 예시','',
        '아래는 전체 grid 중 하나의 표시 예시이며 권장 상한이나 선정 결과가 아닙니다. 각 알고리즘5개 seed 평균이며 기존 BCQ best는 제외합니다.','',
        '| Dataset | Algorithm | ESS mean | WDR mean | WIS mean | FQE mean |',
        '|---|---|---:|---:|---:|---:|']
    for r in summary:
        if r['scope']=='cumulative_prefix' and r['cap']==10. and r['algorithm']!='ALL20':
            lines.append(f"| {r['dataset']} | {r['algorithm']} | {r['trajectory_ess_mean']:.3f} | {r['wdr_mean']:.6f} | {r['wis_mean']:.6f} | {r['fqe_mean']:.6f} |")
    lines+=['','## 해석과 불확실성','',
        'Clipping은 편향-분산 절충입니다. Capped ESS 상승은 weight 집중 완화이며 추가 환자 정보나 실제 positivity 회복을 뜻하지 않습니다. '
        '95% 구간은 같은1000회 whole-patient bootstrap이며 actor/behavior/FQE를 고정합니다. Nuisance 추정, clipping 편향, 미측정 교란, 정책 선정 오차를 포함하지 않습니다. '
        '최종 trajectory ESS는 WDR 전체의 유효 표본 수와 동일하지 않습니다.','',
        '알고리즘 순위가 상한/방식에 따라 달라지는지 `algorithm_rankings.csv`에서 확인합니다. 낮은 raw ESS를 감추거나 test 순위로 새 checkpoint/상한을 선정하지 않았습니다.','',
        '## 산출물과 재현','',
        '- [정책별 수치와 조건부 구간](policy_metrics.csv)',
        '- [알고리즘별 평균/SD](algorithm_summary.csv)',
        '- [WDR/WIS/FQE 순위](algorithm_rankings.csv)',
        '- [그림](clipping_sensitivity.png)',
        '- Dataset별 `equation_verification.json`, `input_preservation.json`, `completion_receipt.json`.',
        '- 재현: `python -m scripts.evaluate_clipping_sensitivity --output-dir outputs/새경로`.',
        '- 기본 OPE 설정은 unclipped로 유지하며 이 결과는 별도 sensitivity 분석입니다.','']
    (out/'README.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--dataset',choices=['heparin','sepsis','both'],default='both')
    p.add_argument('--n-bootstrap',type=int,default=1000)
    p.add_argument('--resume',action='store_true',help='Verify and reuse completed datasets in an interrupted run')
    args=p.parse_args();out=args.output_dir.resolve()
    verify_source_manifest(ROOT)
    if out.exists() and not args.resume:raise FileExistsError('Use a new output directory or explicit --resume')
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),
        'fixed_caps':list(CAPS),'cap_selection':None,'policy_selection':None,'n_bootstrap':args.n_bootstrap,
        'source_manifest_sha256':sha256_file(ROOT/'source_manifest.json'),
        'source_manifest':json.loads((ROOT/'source_manifest.json').read_text()),
        'baseline_receipt_sha256':sha256_file(BASE/'completion_receipt.json')}
    if args.resume:
        if (out/'completion_receipt.json').exists():raise FileExistsError('Run already complete')
        old=json.loads((out/'protocol.json').read_text())
        for k in ['fixed_caps','cap_selection','policy_selection','n_bootstrap','baseline_receipt_sha256']:
            assert old[k]==protocol[k],k
        write(out/'resume_source_manifest.json',protocol)
    else:
        out.mkdir(parents=True)
        write(out/'protocol.json',protocol)
    rows=[]
    with threadpool_limits(limits=1):
        for d in (['heparin','sepsis'] if args.dataset=='both' else [args.dataset]):
            directory=out/d
            receipt=directory/'completion_receipt.json'
            if args.resume and receipt.exists():
                completed=json.loads(receipt.read_text());assert completed['status']=='complete'
                for rel,digest in completed['artifact_sha256'].items():assert sha256_file(directory/rel)==digest,rel
                preservation=json.loads((directory/'input_preservation.json').read_text())
                for rel,digest in preservation['inputs_sha256'].items():assert sha256_file(ROOT/rel)==digest,rel
                with (directory/'policy_metrics.csv').open() as f:
                    for row in csv.DictReader(f):
                        for k,v in row.items():
                            if k in ['dataset','algorithm','seed','policy','kind','scope']:continue
                            row[k]=None if v=='' else v=='True' if v in ['True','False'] else float(v)
                        rows.append(row)
                print('REUSE_VERIFIED',d,completed['scenarios'],flush=True)
            else:
                if directory.exists():directory.rmdir()  # only an empty interrupted directory is safe to remove
                rows+=run_dataset(d,directory,args.n_bootstrap)
    build_report(out,rows)
    artifacts={str(f.relative_to(out)):sha256_file(f) for f in out.rglob('*') if f.is_file() and not f.name.startswith('._')}
    write(out/'completion_receipt.json',{'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),
        'scenarios':len(rows),'actor_refits':0,'behavior_refits':0,'critic_refits':0,'cap_selection':None,
        'bootstrap_draws':args.n_bootstrap,'artifact_sha256':artifacts})
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
