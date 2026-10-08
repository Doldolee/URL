from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util import project_root
from util import ReplayBuffer
try:
    import mlflow
except ImportError:
    mlflow = None
from agent import CQL
from agent import DDQN
from agent import FixedLearningRateBCQ as BCQ
from agent import StandardDQN as DQN
from agent import HeparinBCQ
from util import prepare_subject_split, reward_type
from metric import evaluate_sepsis_policy, evaluate_heparin_policy

from datetime import datetime, timezone
import hashlib
import json
import torch


from util import set_seed


def log_params(params):
    for name in ['optimizer_parameters', 'use_polyak_target_update', 'target_update_frequency',
                 'algorithm', 'max_timesteps', 'eval_freq', 'hidden_node',
                 'activation', 'batch_size', 'discount']:
        mlflow.log_param(name, params[name])
    mlflow.log_param('lr', params['optimizer_parameters']['lr'])
    mlflow.log_param('seed', params.get('seed', 42))
    mlflow.log_param('ope_policy_mode', params.get('ope_policy_mode', 'greedy'))
    mlflow.log_param('ope_primary_ratio_cap', 'none')
    mlflow.log_param('ope_fqe_epochs', params.get('ope_fqe_epochs', 100))


def _log_evaluation(dataset, report, step):
    for name in ['dr', 'wdr', 'wis']:
        interval = report['bootstrap']['intervals'][name]
        if report[name] is not None:
            mlflow.log_metric(f'{dataset} {name}', report[name], step=step)
        for suffix, key in [(' ci low', 'low'), (' ci high', 'high')]:
            if interval[key] is not None:
                mlflow.log_metric(f'{dataset} {name}{suffix}', interval[key], step=step)
    for suffix, key in [('', 'value'), (' ci low', 'low'), (' ci high', 'high')]:
        if report['fqe'][key] is not None:
            mlflow.log_metric(f'{dataset} fqe{suffix}', report['fqe'][key], step=step)
    if report['weights']['trajectory_ess'] is not None:
        mlflow.log_metric(f'{dataset} trajectory ess', report['weights']['trajectory_ess'], step=step)


def train(params):
    params = dict(params)
    dataset = params['target_data']
    profile = reward_type(dataset)
    if params.get('target_test_data', dataset) != dataset:
        raise ValueError('Training OPE requires the same verified dataset; evaluate external cohorts separately')
    root = Path(params.get('project_root', project_root())).resolve()
    data_path = root/'dataset'/dataset
    device = params.get('device')
    if device is None or device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params['device'] = device
    seed = params.get('seed', 42)
    set_seed(seed)
    train_buffer = ReplayBuffer(state_dim=params['state_dim'], batch_size=params['batch_size'],
                                target_data=dataset, buffer_path=data_path, device=device).load_data()
    test_buffer = ReplayBuffer(state_dim=params['state_dim'], batch_size=params['batch_size'],
                               target_data=dataset, buffer_path=data_path, device=device).load_data(only_test_set=True)
    train_groups, test_groups, mapping = prepare_subject_split(
        dataset, train_buffer, test_buffer, root,
        cohort_csv=params.get('cohort_csv'), demog_csv=params.get('demog_csv'))
    mlflow.log_param('train_test_subject_overlap_excluded', mapping['excluded_train_episodes'])
    mlflow.log_param('evaluation_unit', 'subject cluster')
    mlflow.log_param('ope_reward_type', profile)
    mlflow.log_param('done_terminal_value_in_memory', 1)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = Path(params.get('output_dir', root/'outputs'/f'{dataset}_training_{timestamp}_seed{seed}'))
    destination.mkdir(parents=True, exist_ok=False)
    policies = {'BCQ': HeparinBCQ if dataset == 'heparin' else BCQ, 'CQL': CQL, 'DDQN': DDQN, 'DQN': DQN}
    policy = policies[params['algorithm']](**params)
    for training_iters in range(params['max_timesteps']):
        policy.train(train_buffer)
        if (training_iters + 1) % params['eval_freq'] == 0:
            print(f"Training updates: {training_iters + 1}", flush=True)
    checkpoint = destination/'policy_final.pth'
    hashes = {split+'_'+name: hashlib.sha256((data_path/f'{split}_{name}.npy').read_bytes()).hexdigest()
              for split in ['train', 'test'] for name in ['state', 'next_state', 'action', 'reward', 'done']}
    torch.save({'model_state_dict': policy.Q.state_dict(), 'training_step': params['max_timesteps'],
                'params': params, 'seed': seed, 'dataset_sha256': hashes,
                'subject_mapping': mapping, 'done_convention': 'terminal_is_one',
                'reward_type': profile}, checkpoint)
    evaluator = evaluate_heparin_policy if dataset == 'heparin' else evaluate_sepsis_policy
    report = evaluator(params['algorithm'], policy, train_buffer, test_buffer,
                       train_groups=train_groups, test_groups=test_groups,
                       policy_mode=params.get('ope_policy_mode', 'greedy'),
                       gamma=params['discount'], seed=seed,
                       fqe_epochs=params.get('ope_fqe_epochs', 100),
                       n_bootstrap=params.get('ope_bootstrap_draws', 1000), output_dir=destination)
    report['policy_checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report['dataset_sha256'], report['subject_mapping'] = hashes, mapping
    (destination/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    _log_evaluation(dataset, report, params['max_timesteps'])
    mlflow.log_artifact(str(checkpoint))
    mlflow.log_artifact(str(destination/'evaluation.json'))
    return policy


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train and evaluate one policy with patient-separated OPE')
    parser.add_argument('--dataset', choices=['heparin', 'sepsis'], default='sepsis')
    parser.add_argument('--algorithm', choices=['BCQ', 'CQL', 'DDQN', 'DQN'], default='CQL')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    from configs.config import get_params
    if mlflow is None:
        import mlflow
    params = get_params(target_data=args.dataset, algorithm=args.algorithm)
    params['device'] = args.device
    if args.output_dir:
        params['output_dir'] = args.output_dir
    mlflow.set_experiment(params['target_data'])
    with mlflow.start_run(run_name=params['algorithm']):
        log_params(params)
        train(params)
