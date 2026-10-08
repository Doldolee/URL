"""Build the completed, exploratory behavior-estimator ESS comparison report."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_csv(path):
    return list(csv.DictReader(Path(path).open()))


def numeric(row, key):
    return float(row[key]) if row.get(key) not in ['', None] else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    combined, selections, summaries = [], {}, []
    table = []
    details = []
    figure, axes = plt.subplots(1, 2, figsize=(13, 8), constrained_layout=True)
    for dataset, axis in zip(['heparin', 'sepsis'], axes):
        directory = out/dataset
        receipt = json.loads((directory/'completion_receipt.json').read_text())
        if receipt['status'] != 'complete':
            raise ValueError('Dataset experiment is incomplete')
        selection = json.loads((directory/'selection.json').read_text())
        validation = read_csv(directory/'validation_candidate_ranking.csv')
        test = read_csv(directory/'test_candidate_summary.csv')
        vm = {r['candidate']: r for r in validation}
        tm = {r['candidate']: r for r in test}
        winner = selection['selected_behavior']
        nll_winner = selection['minimum_log_loss_behavior']
        reference, selected = tm['rf_sigmoid'], tm[winner]
        best_test = next(r for r in test if r['eligible'] == 'True')
        selections[dataset] = {
            'selected_behavior': winner,
            'selection_metric': 'maximum validation mean trajectory ESS across 20 matched policies',
            'selected_model': str((directory/'selected/behavior.joblib').relative_to(ROOT)),
            'validation_mean_ess': numeric(vm[winner], 'mean_ess'),
            'test_mean_ess': numeric(selected, 'mean_ess'),
            'test_ess_maximum_candidate_diagnostic_only': best_test['candidate'],
            'test_used_for_selection': False,
            'minimum_log_loss_behavior': nll_winner,
            'default_fit_behavior_protocol_changed': False}
        table.append(f"| {dataset} | {winner} | {numeric(vm['rf_sigmoid'],'mean_ess'):.3f} → {numeric(vm[winner],'mean_ess'):.3f} | "
                     f"{numeric(reference,'mean_ess'):.3f} → {numeric(selected,'mean_ess'):.3f} | "
                     f"{numeric(vm['rf_sigmoid'],'log_loss'):.4f} → {numeric(vm[winner],'log_loss'):.4f} |")
        for row in validation:
            name = row['candidate']
            combined.append({'dataset':dataset, 'candidate':name,
                'selected':name==winner, 'validation_eligible':row['eligible'],
                'validation_mean_ess':numeric(row,'mean_ess'),
                'validation_median_ess':numeric(row,'median_ess'),
                'validation_log_loss':numeric(row,'log_loss'),
                'validation_brier':numeric(row,'brier_multiclass_sum'),
                'validation_ece':numeric(row,'top_label_ece'),
                'test_eligible':tm[name]['eligible'], 'test_mean_ess':numeric(tm[name],'mean_ess'),
                'test_median_ess':numeric(tm[name],'median_ess')})
        detail = [f"**{dataset}: {winner}**", '',
            f"Test에서 평균 ESS가 가장 높았던 후보는 `{best_test['candidate']}` "
            f"({numeric(best_test,'mean_ess'):.3f})입니다. 이는 선택 이후 계산한 탐색 결과이며, test에서 후보를 다시 선택하지 않았습니다.", '',
            f"Validation log-loss 최소 후보는 `{nll_winner}` "
            f"({numeric(vm[nll_winner],'log_loss'):.4f})입니다. ESS 최대 후보와 clinician 행동 예측 성능 최대 후보는 다릅니다.", '',
            '| 알고리즘 | Test ESS mean | WDR mean ± seed SD | FQE mean ± seed SD | WIS mean ± seed SD | DR mean |',
            '|---|---:|---:|---:|---:|---:|']
        algorithm_rows = read_csv(directory/'algorithm_summary.csv')
        for row in algorithm_rows:
            summaries.append(row)
            if row['candidate'] != winner:
                continue
            def mean_sd(key):
                return f"{numeric(row,key+'_mean'):.4f} ± {numeric(row,key+'_sd'):.4f}"
            detail.append(f"| {row['algorithm']} | {numeric(row,'trajectory_ess_mean'):.3f} | "
                          f"{mean_sd('wdr')} | {mean_sd('fqe')} | {mean_sd('wis')} | {numeric(row,'dr_mean'):.4g} |")
        detail += ['', f"[정책별 OPE·1000회 조건부 bootstrap]({dataset}/selected_policy_metrics.csv), "
                   f"[전체 후보 test 결과]({dataset}/test_policy_metrics.csv), "
                   f"[선택된 behavior 모델]({dataset}/selected/behavior.joblib).", '']
        if dataset == 'heparin':
            best = next(r for r in read_csv(directory/'selected_policy_metrics.csv') if r['kind']!='matched_protocol')
            detail += [f"기존 Heparin BCQ validation-best는 20개 공통 정책의 선정 점수에 포함하지 않았습니다. "
                       f"별도 softmax 평가: ESS {numeric(best,'trajectory_ess'):.3f}, "
                       f"WDR {numeric(best,'wdr'):.4f}, FQE {numeric(best,'fqe'):.4f}, WIS {numeric(best,'wis'):.4f}.", '']
        details.extend(detail)
        names = [r['candidate'] for r in validation]
        position = np.arange(len(names))
        axis.barh(position-.18, [numeric(vm[n],'mean_ess') or 0 for n in names], .36, label='Behavior validation',color='#2864aa')
        axis.barh(position+.18, [numeric(tm[n],'mean_ess') or 0 for n in names], .36, label='Historical test',color='#df9841')
        labels = [n+(' *' if n==winner else '')+
                  (' (undefined)' if vm[n]['eligible']!='True' else '') for n in names]
        axis.set_yticks(position, labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel('Mean terminal trajectory ESS across 20 fixed policies')
        axis.set_title('MIMIC-III '+dataset+'\n* Selected by validation ESS',fontsize=11)
        axis.grid(axis='x',alpha=.2)
        axis.legend(fontsize=8)
    figure.savefig(out/'ess_comparison.png', dpi=180)
    plt.close(figure)
    for filename, rows in [('candidate_comparison.csv',combined),('algorithm_summary.csv',summaries)]:
        with (out/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    (out/'selected_behavior.json').write_text(json.dumps(selections,indent=2,ensure_ascii=False)+'\n')
    text = ['# MIMIC-III behavior 모델 비교와 validation ESS 선정', '',
        '기존 temperature1 softmax actor41개와 해당 FQE41개를 그대로 사용하고, importance-weight 분모의 behavior 추정 방법만 바꿨습니다. ' 
        'Dataset별17개 후보를 같은 환자 분할에서 비교한 탐색 실험입니다. 공식 기본 RF 학습 프로토콜은 유지합니다.', '',
        '한 dataset에서 하나의 behavior 모델을 공유합니다. DQN/DDQN/BCQ/CQL 각5개 seed의 '
        '**train 내부 behavior-validation 평균 terminal trajectory ESS 최대**로 선택했습니다. '
        'Heparin BCQ best는 별도 보고하며 선택 평균에서 제외합니다. Test ESS·보상·OPE를 선택에 사용하지 않았습니다.', '',
        '| 데이터 | 선택 방법 | Validation ESS: RF → 선택 | Test ESS: RF → 선택 | Validation log-loss: RF → 선택 |',
        '|---|---|---:|---:|---:|', *table, '',
        'ESS는 최종 누적 importance weight의 집중도이며 WDR 전체의 유효 표본 수가 아닙니다. '
        '**ESS 최대 기준으로 모델을 선택해도 test ESS 개선이나 clinician 행동정책의 정확성을 보장하지 않습니다.** '
        '이번 test ESS는 여전히 매우 작아 정책 가치의 신뢰성이 확보됐다고 해석하지 않습니다.', '',
        '![17개 후보의 validation/test ESS](ess_comparison.png)', '', *details,
        '후보: 기존 RF raw/sigmoid(재학습 없음), logistic C1 raw/temperature, MLP64×64·128×128·1000 '
        '각 raw/temperature, kNN100·300·1000, cluster50·100·300·750. 총34개 후보 평가이며 새 학습은 dataset별15개 predictor 구성입니다. '
        'MLP는 구조별 하나의 CE 학습 모델을 raw/temperature에서 공유합니다.', '',
        'Scaler·행동 prior·모델 fitting은 fit 환자만 사용합니다. MLP early stopping, temperature, '
        'kNN/cluster Dirichlet concentration은 calibration 환자의 NLL로 결정합니다. '
        'Dirichlet은 fit 행동 prior로 count를 보정하며 concentration grid는 [.1,1,10,100,1000]입니다. '
        '이는 논문의 kNN/cluster 계열을 참고한 구현이며, 원 논문의 코드·보정식을 그대로 재현한 것은 아닙니다. '
        '저장 확률 floor, importance ratio cap, output clipping은 적용하지 않았습니다. '
        'Raw RF에서 기록 행동 확률이0이면 ESS/OPE를 미정의로 보고하고 선택에서 제외합니다.', '',
        'Heparin behavior fit/calibration/validation은23805/8349/7699행(789/264/264명), '
        'Sepsis는73775/25111/24414행(4748/1583/1583명)입니다. '
        'Validation은 behavior fitting과 독립이지만 기존 actor/FQE 학습 cohort에는 들어 있습니다. '
        'Test는 원 Heparin191episode·Sepsis1662episode이며, 기존에 분석했던 historical test입니다. '
        'Heparin 원 done0terminal을 메모리에서 변환했고 모든 원 배열은 유지합니다.', '',
        'Gamma=.98, WDR 시간별 normalization·absorbing padding을 유지했습니다. '
        '선택 모델의 policy별 환자 bootstrap1000회는 actor/behavior/FQE와 선택을 고정한 조건부 구간이며 '
        'behavior 재추정·모델 선정 불확실성을 포함하지 않습니다. Seed SD는 정책5개의 값의 표본 SD입니다. '
        'Heparin은 반복 reward이므로 return이[-1,1]로 제한되지 않고, Sepsis는 terminal±1입니다. '
        'Unclipped DR의 극단적 값은 그대로 기록했습니다.', '',
        '[전체 후보 비교](candidate_comparison.csv), [선택 모델 경로](selected_behavior.json), '
        '[Heparin 상세](heparin/protocol.json), [Sepsis 상세](sepsis/protocol.json), [전체 테스트 로그](tests.log).', '',
        '**검증:** 전체112개 테스트 통과, 기존 RF OPE 재현, 독립 log-space ESS 대조, '
        '선택 모델 재로드 전체 test 확률 동일, 기존41개 FQE 점추정·95% 구간 동일. '
        '9개 정책의80자리 Decimal WDR 대조 최대오차9.44e-15이며 원 입력 해시를 보존했습니다. '
        '[독립 검증](independent_verification.json). 초기 inference 수정 전 후보는 '
        'dataset별 `fit_attempt_01/`에 보존한 개발 중간 자료이며 최종 선정에는 사용하지 않았습니다.', '',
        '재실행은 새 output-dir에서 `python -m scripts.compare_behavior_policies --dataset heparin '
        '--output-dir outputs/새경로 --stage prepare`, 이어서 같은 dataset/output의 `--stage fit`, `--stage evaluate`입니다. '
        'Sepsis도 같은 순서로 실행합니다. 두 완료 후 `python -m scripts.report_behavior_comparison --output-dir outputs/새경로`를 실행합니다.', '']
    (out/'README.md').write_text('\n'.join(text))
    print(json.dumps(selections,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
