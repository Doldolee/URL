"""Current single-run defaults; estimator settings are separate from training."""
from pathlib import Path
import yaml


def get_params(target_data='sepsis', algorithm='CQL'):
    if target_data not in {'heparin', 'sepsis'} or algorithm not in {'BCQ', 'CQL', 'DDQN', 'DQN'}:
        raise ValueError('Choose a supported MIMIC-III dataset and algorithm')
    with (Path(__file__).parent/'config_base.yaml').open() as stream:
        params = yaml.safe_load(stream)
    params.update(target_data=target_data, target_test_data=target_data, algorithm=algorithm,
                  num_actions=6 if target_data == 'heparin' else 25,
                  state_dim=16 if target_data == 'heparin' else 43,
                  optimizer_parameters={'lr': 1e-6, 'weight_decay': 1e-5},
                  use_polyak_target_update=False, target_update_frequency=25,
                  max_timesteps=1500, eval_freq=50, hidden_node=1024,
                  activation='relu', batch_size=64,
                  ope_bootstrap_draws=1000)
    if target_data == 'heparin' and algorithm == 'BCQ':
        # Selected validation settings; training this config creates a new run,
        # while the existing selected checkpoint remains frozen in pth/.
        params['optimizer_parameters']['lr'] = 1e-4
        params.update(bcq_threshold=.1, seed=44, max_timesteps=5000)
    return params
