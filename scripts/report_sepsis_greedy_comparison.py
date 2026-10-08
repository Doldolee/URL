# coding: utf-8
"""Read completed run, verify serialized models, and write a new report receipt."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
from agent import POLICIES
from agent import frozen_policy_arrays
from model import FQECritic
from metric import predict_q


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def fmt(value):
    return 'undefined' if value is None or value=='' else f'{float(value):.6f}'


def mean_sd(row,key):
    return f"{fmt(row[key+'_mean'])} ± {fmt(row[key+'_sd'])} ({row[key+'_defined']}/5)"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();root=args.project_root;out=args.output_dir
    if (out/'report_receipt.json').exists():raise FileExistsError('Use fresh result copies')
    receipt=json.loads((out/'completion_receipt.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    assert receipt['status']=='complete'
    for rel,digest in receipt['artifact_sha256'].items():assert sha(out/rel)==digest,rel
    for rel,digest in protocol['inputs_sha256'].items():assert sha(root/rel)==digest,rel
    state=np.load(root/'dataset/sepsis/test_state.npy')
    torch.set_num_threads(4)
    verification=[]
    for algorithm in protocol['algorithms']:
        for seed in protocol['policy_seeds']:
            d=out/'models'/f'sepsis_{algorithm}_seed{seed}'
            cp=torch.load(d/'policy_final.pth',map_location='cpu',weights_only=False)
            policy=POLICIES[algorithm](**cp['params'])
            policy.Q.load_state_dict(cp['model_state_dict']);policy.Q.eval()
            pi=frozen_policy_arrays(algorithm,policy,state,mode='greedy')
            with np.load(d/'frozen_arrays.npz') as saved:
                np.testing.assert_array_equal(pi.argmax(1),saved['test_actions'])
                next_pi=np.eye(25,dtype=np.float32)[saved['train_next_actions']]
                expected=json.loads((d/'fit.json').read_text())['fixed_next_policy_sha256_float32']
                assert hashlib.sha256(np.ascontiguousarray(next_pi).tobytes()).hexdigest()==expected
                critic=FQECritic(43,25,128)
                critic.load_state_dict(torch.load(d/'fqe.pt',map_location='cpu',weights_only=True))
                q=predict_q(critic,state)
                np.testing.assert_array_equal(q,saved['critic_test_q'])
            ev=json.loads((d/'evaluation.json').read_text())
            if ev['verification']['undefined_time_step'] is not None:
                assert ev['verification']['undefined_time_step']==min(ev['ope']['numerical_status']['undefined_wdr_time_steps'])
            verification.append(dict(algorithm=algorithm,seed=seed,serialized_actor_matches_all_test_rows=True,
                                     serialized_critic_matches_all_test_rows=True,fqe_fixed_actor_hash_matches=True))
            print('serialized verification',algorithm,seed,flush=True)
    write(out/'serialized_model_verification.json',dict(status='passed',cases=verification))
    summaries=list(csv.DictReader((out/'summary.csv').open()))
    rows=list(csv.DictReader((out/'metrics.csv').open()))
    table='\n'.join(f"| {r['algorithm']} | {mean_sd(r,'wdr')} | {mean_sd(r,'fqe')} | {mean_sd(r,'wis')} | "
                    f"{fmt(r['terminal_weight_ess_min'])}–{fmt(r['terminal_weight_ess_max'])} | "
                    f"{r['nonzero_terminal_paths_min']}–{r['nonzero_terminal_paths_max']} |" for r in summaries)
    details='\n'.join(f"| {r['algorithm']} | {r['seed']} | {fmt(r['wdr'])} | {fmt(r['fqe'])} | {fmt(r['wis'])} | "
                      f"{fmt(r['terminal_weight_ess'])} | {r['nonzero_terminal_paths']} | {r['wdr_bootstrap_defined']}/1000 |" for r in rows)
    support='\n'.join(f"| {r['algorithm']} | {r['wdr_bootstrap_defined']}/5000 | {r['wis_bootstrap_defined']}/5000 |" for r in summaries)
    readme=f'''# Sepsis greedy DDQN / DQN / BCQ / CQL 비교

네 알고리즘을 각각 seed 42–46으로 새로 학습하고, 각 **동일 greedy actor**의 WDR·FQE·WIS를 실제 산출했습니다. 이전 CQL·BCQ checkpoint를 재평가한 표가 아닙니다. 기존 결과는 보존했습니다.

## 알고리즘별 결과

평균 ± **seed 간 표본 표준편차**입니다. 괄호는 계산 가능한 정책 수/계획 5개입니다. Undefined 정책을 0으로 대체하지 않으며 일부 seed만 계산 가능하면 해당 평균은 그 subset에 조건부입니다. 정책 가치의 신뢰구간이나 ensemble 가치가 아닙니다.

| 알고리즘 | WDR mean ± SD (defined) | FQE mean ± SD (defined) | WIS mean ± SD (defined) | 최종 weight ESS 범위 | 양의 최종 weight trajectory 수 |
|---|---|---|---|---|---|
{table}

## 공통 학습·평가 조건

- 학습: 기존 `train`에서 test 환자 전원을 제외한 **{protocol['train_rows']:,}행/{protocol['train_subjects']:,}명**. 제외 mask는 환자 그룹으로 직접 재검증했습니다. 전체 episode를 유지했습니다.
- 평가: 기존 test의 **{protocol['evaluation_episodes']:,} complete episode/{protocol['evaluation_subjects']:,}명** 전체. 과거 원 CQL WDR의 1322 episode cohort와 다르므로 숫자의 단순 전후 차이를 성능 개선으로 해석하지 않습니다. 이미 분석한 test이며 untouched test가 아닙니다.
- 현재 configs의 실제 값을 고정: 43 state features, 25 actions, hidden1024, ReLU, Adam lr=1e-6/weight_decay=1e-5, batch64, **1500 updates**, gamma=.98, hard target copy25 updates. Uniform transition sampling을 seed별로 네 알고리즘에서 동일하게 사용했습니다. 최종 update 정책만 평가하고 test로 hyperparameter/checkpoint를 고르지 않았습니다. 1500은 epoch 수가 아닙니다.
- 기존 네트워크 구조 유지: DQN은 일반 MLP, DDQN/CQL은 dueling Q, BCQ는 dueling Q+imitation branch입니다. 따라서 결과에는 구조 차이도 포함되며 loss 하나만의 ablation이 아닙니다. CQL alpha=1, BCQ threshold=.3.
- **학습 수정 두 가지**는 별도 `agent.py`와 `agent.py`에 있습니다. 기존 agent 구현는 수정하지 않았습니다. 기존 DQN도 online argmax/target evaluation을 쓰므로 새 DQN에는 `max_a Q_target(s_next,a)`를 사용합니다. DDQN은 online argmax/target evaluation을 유지합니다. BCQ의 미실행 warmup scheduler가 초기 Adam LR를 0으로 만드는 문제는 다른 알고리즘과 동일한 고정 LR1e-6 복원으로 수정했습니다. 이번에는 cosine scheduler를 실행하지 않습니다.
- Greedy는 선택 행동 probability1, 나머지0입니다. BCQ는 imitation 확률/최대 확률>.3인 행동만 허용한 후 Q argmax를 취합니다. Softmax 정책 평가가 아닙니다. `eval()` 모드에서 actor를 고정하며 inference batch 변경으로 행동이 바뀌지 않는지 확인했습니다.
- Behavior: corrected archive의 **공통 train-only calibrated RF full25 probabilities**를 그대로 사용했습니다. 이번에 behavior를 새로 fit하지 않았습니다. 기존 train/test `BC_prob` scalar는 사용하지 않습니다.
- FQE: 각 고정 greedy actor마다 별도 train-only bounded128×128 critic을 새로 fit했습니다(**20 fits**). gamma=.98, 최대100 epochs, min30/patience20, batch4096, lr1e-3, seed42, train 내부 환자15% holdout의 balanced terminal/nonterminal Bellman MSE로 epoch를 선택했습니다. Actor 정책을 FQE의 argmax로 다시 정의하지 않습니다. Train next-state 정책의 float32 SHA256을 fit 기록과 대조했습니다.

## 추정식

`rho_it = product_(k<=t) pi(a_ik|s_ik)/b_hat(a_ik|s_ik)`.

`w_it = rho_it / sum_j rho_jt`, `w_i,-1=1/N`.

`WDR = sum_it gamma^t [w_it*(r_it - Qhat(s_it,a_it)) + w_i,t-1*sum_a pi(a|s_it)Qhat(s_it,a)]`.

WDR은 Thomas & Brunskill (ICML2016), [식(1) 및 section5](https://proceedings.mlr.press/v48/thomasa16.pdf)의 시간별 정규화를 사용합니다. 종료 trajectory는 이후 reward/Q/V=0이고 마지막 누적 rho를 absorbing 분모에 유지합니다. WIS는 마지막 trajectory rho로 관측 discounted return을 정규화합니다. FQE는 episode initial state에서 고정 actor의 기대 Q를 평균합니다.

Reward는 중간0, `mortality_90d` 기반 terminal survival+1/death−1 한 번입니다. gamma=.98은 저장 transition당이며 실제 경과시간 비례 할인이 아닙니다. **Primary step-ratio clipping, probability floor, 최종 출력 clipping은 없습니다.** WDR은 유한 표본에서 [-1,1] 밖으로 나갈 수 있습니다. 양의 누적 weight가 모두 소실되면 WDR/WIS를 undefined로 남깁니다.

## seed별 결과

| 알고리즘 | seed | WDR | FQE | WIS | 최종 weight ESS | 양의 최종 경로 | 유효 WDR bootstrap |
|---|---:|---:|---:|---:|---:|---:|---|
{details}

정책별 조건부 percentile 구간은 [metrics.csv](metrics.csv), 전체 bootstrap sample은 `models/*/evaluation.json`에 있습니다. `time_step_diagnostics.csv`에는 각 시점의 ESS, reward/Q/이전 weight V 항과 가중치 집중을 저장했습니다.

## 불확실성과 해석

정책별 환자 bootstrap1000회는 **actor/behavior/FQE를 고정한 조건부** 구간입니다. 환자의 모든 episode를 함께 resample하고 episode-weighted estimand를 유지합니다. FQE는 각 seed에서1000/1000회가 finite입니다. WDR/WIS가 정의되지 않은 draw는 0으로 채우거나 재시도하지 않으며 계획 분모에 남겼습니다. 유효 draw만의 percentile은 계산 가능 조건부 구간으로 읽어야 합니다.

| 알고리즘 | 유효 WDR draw / 계획 | 유효 WIS draw / 계획 |
|---|---:|---:|
{support}

Seed SD는 policy initialization/sampling 차이의 기술통계이며 nuisance 재추정 불확실성을 포함하지 않습니다. 새 behavior/FQE outer refit은 이번 비교에서 수행하지 않았습니다. 최종 weight ESS는 WDR 전체의 유효 표본 수가 아니라 최종 누적 가중치 집중 진단입니다. Greedy의 불일치 이후 누적 rho=0이 되고, 남은 경로도 fitted behavior 확률의 역수를 곱하므로 가중치가 집중됩니다. 이 표만으로 알고리즘 우월성이나 임상적 정책 가치를 확인했다고 해석하지 않습니다.

고정1500-update protocol의 수치 산출을 완료했으며 policy 수렴/튜닝을 완료했다는 주장도 아닙니다. 기존 정규화·결측 보완·빈 bin 생략·시간 경계 문제를 유지한 자료입니다. 새 전처리·외부 검증 실험과 구분합니다.

## 검증 및 파일

- 새 학습 회귀 test4개: DQN/DDQN target 차이의 analytic SGD 결과, terminal bootstrap 차단, BCQ zero-LR 회귀와 실제 parameter update, paired sampling 재현성.
- 전체20개 cohort WDR을 별도 80자리 Decimal 식과 대조했습니다. 정의된 WDR의 최대 절대오차 **{receipt['max_absolute_equation_error']:.3e}**. Undefined도 reference와 대조했습니다.
- Policy20/FQE20 checkpoint를 저장 후 재로드하여 **모든 test 행**의 actor 행동과 critic Q가 저장 cache와 정확히 일치하는지 검증했습니다. FQE fixed-policy hash20개도 재확인했습니다.
- 읽은 기존 source/data/checkpoint/receipt {len(protocol['inputs_sha256'])}개의 SHA256은 실행 전후 및 report 생성 시 동일했습니다. 원 모델·결과를 덮어쓰지 않았습니다.

[Protocol](protocol.json), [알고리즘 집계 CSV](summary.csv), [seed별 CSV](metrics.csv), [원 계산 receipt](completion_receipt.json), [모델 재로드 검증](serialized_model_verification.json), [입력 보존](input_preservation.json)을 함께 확인합니다.
'''
    (out/'README.md').write_text(readme)
    write(out/'report_receipt.json',dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),
          serialized_model_verifications=20,previous_inputs_preserved=True,
          prior_completion_receipt_sha256=sha(out/'completion_receipt.json'),
          artifact_sha256={name:sha(out/name) for name in ['README.md','serialized_model_verification.json','summary.csv','metrics.csv','protocol.json']},
          report_source_sha256=sha(Path(__file__))))
    print('report complete',out,flush=True)

if __name__=='__main__':main()
