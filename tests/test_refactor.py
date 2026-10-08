"""Contracts for patient-separated Heparin routing and shared OPE."""
import agent
import metric
import model
import util
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

import agent as behavior
import util as dataset
import metric as metric
from util import ReplayBuffer
from model import HeparinFQECritic
from test_metric import buffer, TablePolicy


def store(folder, split, state, done, reward=None):
    state = np.asarray(state, dtype=np.float32)
    done = np.asarray(done, dtype=np.float32).reshape(-1, 1)
    n = len(state)
    data = {'state': state, 'next_state': state.copy(), 'action': np.zeros((n, 1), dtype=np.int64),
            'done': done, 'reward': np.asarray(reward if reward is not None else np.zeros(n), dtype=np.float32).reshape(-1, 1)}
    for name, values in data.items():
        np.save(folder/f'{split}_{name}.npy', values)
    return data


class RefactorTests(unittest.TestCase):
    def test_buffer_standardizes_iii_once_and_preserves_iv_terminal_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            for name, stored in [('heparin', [1, 0]), ('heparin4_observed_aptt', [0, 1]), ('sepsis', [0, 1])]:
                source=store(folder, 'train', [[0, 1], [1, 1]], stored)
                loaded=ReplayBuffer(2, 16, name, folder, device='cpu').load_data()
                np.testing.assert_array_equal(loaded.done.reshape(-1), [0, 1])
                np.testing.assert_array_equal(np.load(folder/'train_done.npy'), source['done'])
                self.assertEqual(loaded.done_convention, 'terminal_is_one')
                self.assertEqual(loaded.sample()[4].shape, (16, 1))
                self.assertFalse((folder/'train_BC_prob.npy').exists())
            store(folder, 'train', [[0, 1], [1, 1]], [1, 0])
            with self.assertRaisesRegex(ValueError, 'unfinished'):
                ReplayBuffer(2, 1, 'heparin', folder, device='cpu').load_data(size=1)
            np.save(folder/'train_action.npy', np.array([[.5], [0.]]))
            with self.assertRaisesRegex(ValueError, 'finite integers'):
                ReplayBuffer(2, 1, 'heparin', folder, device='cpu').load_data()

    def test_per_step_pipeline_uses_heparin_critic_and_same_actor_for_every_estimator(self):
        training=buffer([[0, 1], [1, 1]], [1, 1], [0, 1])
        evaluation=buffer([[2, 1], [3, 1]], [1, 1], [0, 1])
        policy=TablePolicy([[1, 0]]*4)
        critic=HeparinFQECritic(2, 2, 8, value_bound=2.)
        classifier=object()
        with mock.patch('agent.fit_behavior', return_value=(classifier, classifier, {'selected_behavior':'calibrated','_partition_indices':{}})) as fit_behavior, \
             mock.patch('model.full_action_proba', side_effect=lambda model, state, **kw: np.tile([1.,0.], (len(state),1))), \
             mock.patch('metric.fit_heparin_fqe', return_value=(critic, {'held_out':True})) as fit_heparin, \
             mock.patch.object(metric, 'fit_fqe', side_effect=AssertionError('Sepsis-only fitter must not run')):
            report=metric.evaluate_heparin_policy('CQL', policy, training, evaluation,
                train_groups=[1,1],test_groups=[2,2],gamma=.5,fqe_epochs=60,n_bootstrap=10)
        for estimator in ['dr','wdr','wis']:
            self.assertAlmostEqual(report[estimator],1.5)
        self.assertEqual(report['fqe']['value'],0.)
        self.assertEqual(report['critic_value_bound'],2.)
        self.assertEqual(report['episodes_used'],1)
        self.assertEqual(fit_behavior.call_args.kwargs['num_actions'],2)
        self.assertEqual(fit_behavior.call_args.kwargs['selection'],'calibrated')
        self.assertEqual(fit_behavior.call_args.kwargs['random_seed'],53)
        self.assertTrue(np.shares_memory(fit_heparin.call_args.args[0]['state'],training.state))
        np.testing.assert_array_equal(fit_heparin.call_args.args[1],[[1,0],[1,0]])
        self.assertEqual(fit_heparin.call_args.kwargs['gamma'],.5)
        self.assertEqual(fit_heparin.call_args.kwargs['min_epochs'],50)

    def test_patient_filter_rejects_partial_episode_removal(self):
        training=buffer([[0,1],[1,1]],[0,1],[0,1])
        evaluation=buffer([[2,1]],[1],[1])
        bad=(np.array([1,1]),np.array([2]),np.array([True,False]),{'excluded_train_episodes':0})
        with mock.patch('util.recover_subject_groups',return_value=bad):
            with self.assertRaisesRegex(ValueError,'whole episodes'):
                util.prepare_subject_split('sepsis',training,evaluation,Path('/private/tmp'))

    def test_unknown_dataset_is_rejected_before_training(self):
        with self.assertRaisesRegex(ValueError,'MIMIC-III'):
            util.reward_type('heparin4_observed_aptt')

    def test_six_action_behavior_has_disjoint_patient_partitions_and_full_probabilities(self):
        # Each synthetic episode covers all six actions, including calibration.
        n=180
        state=np.c_[np.tile(np.arange(6),30),np.repeat(np.arange(30),6)].astype(np.float32)
        action=np.tile(np.arange(6),30)
        done=np.tile([0,0,0,0,0,1],30)
        groups=np.repeat(np.arange(30),6)
        raw, calibrated, report=agent.fit_behavior(state,action,done,groups=groups,
            num_actions=6,selection='calibrated',require_all_actions=True,n_estimators=3,n_jobs=1)
        self.assertEqual(report['selected_behavior'],'calibrated')
        self.assertFalse(report['validation_used_for_selection'])
        p=model.full_action_proba(calibrated,state,num_actions=6)
        self.assertEqual(p.shape,(n,6));np.testing.assert_allclose(p.sum(1),1.)
        partitions=report['_partition_indices']
        for left,right in [('fit','calibration'),('fit','validation'),('calibration','validation')]:
            self.assertFalse(set(groups[partitions[left]]) & set(groups[partitions[right]]))
        with self.assertRaisesRegex(ValueError,'invalid behavior model class'):
            model.full_action_proba(raw,state,num_actions=5)

    def test_main_heparin_standardizes_done_filters_patients_and_logs_undefined_results(self):
        mlflow=types.ModuleType('mlflow')
        for name in ['log_param','log_metric','log_artifact']:
            setattr(mlflow,name,mock.Mock())
        config=types.ModuleType('configs.config');config.get_params=mock.Mock()
        source=Path(__file__).resolve().parents[1]/'scripts/train_policy.py'
        spec=importlib.util.spec_from_file_location('heparin_main_test',source)
        main=importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules,{'mlflow':mlflow,'configs.config':config}):
            spec.loader.exec_module(main)
        calls=[]
        class Agent:
            def __init__(self,**params):
                self.Q=torch.nn.Linear(16,6)
            def train(self,data):
                calls.append(data)
                np.testing.assert_array_equal(data.state[:,0],[2,3])
                np.testing.assert_array_equal(data.done.reshape(-1),[0,1])
        def recover(arrays,cohort,demog):
            # Both source exports have been standardized exactly once by the loader.
            np.testing.assert_array_equal(arrays['train']['done'].reshape(-1),[0,1,0,1])
            return {'train':np.array([7,7,8,8]),'test':np.array([7,7])}, {}, np.array([False,False,True,True]), {'excluded_train_episodes':1}
        def evaluate(algorithm,policy,training,evaluation,**kwargs):
            self.assertEqual(len(calls),1)
            np.testing.assert_array_equal(kwargs['train_groups'],[8,8])
            np.testing.assert_array_equal(kwargs['test_groups'],[7,7])
            return {'dr':None,'wdr':None,'wis':None,
                    'bootstrap':{'intervals':{name:{'low':None,'high':None} for name in ['dr','wdr','wis']}},
                    'fqe':{'value':0.,'low':None,'high':None},'weights':{'trajectory_ess':None}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'dataset/heparin';data.mkdir(parents=True)
            state=np.zeros((4,16));state[:,0]=np.arange(4)
            original=store(data,'train',state,[1,0,1,0])
            store(data,'test',np.zeros((2,16)),[1,0])
            params={'project_root':str(root),'target_data':'heparin','state_dim':16,'batch_size':2,
                    'algorithm':'BCQ','max_timesteps':1,'eval_freq':1,'discount':.98,'device':'cpu'}
            with mock.patch.object(main,'HeparinBCQ',Agent), \
                 mock.patch('util.recover_heparin_subject_groups',side_effect=recover), \
                 mock.patch.object(main,'evaluate_heparin_policy',side_effect=evaluate) as evaluator, \
                 mock.patch.object(main,'evaluate_sepsis_policy',side_effect=AssertionError('Wrong reward protocol')):
                main.train(params)
            self.assertEqual(evaluator.call_count,1)
            checkpoints=list((root/'outputs').glob('*/policy_final.pth'))
            self.assertEqual(len(checkpoints),1)
            cp=torch.load(checkpoints[0],weights_only=True,map_location='cpu')
            self.assertEqual(cp['reward_type'],'per_step')
            self.assertEqual(cp['done_convention'],'terminal_is_one')
            np.testing.assert_array_equal(np.load(data/'train_done.npy'),original['done'])
            self.assertEqual(len(mlflow.log_metric.call_args_list),1) # FQE point only
            report=json.loads((checkpoints[0].parent/'evaluation.json').read_text())
            self.assertEqual(report['subject_mapping']['excluded_train_episodes'],1)
            self.assertEqual(len(report['policy_checkpoint_sha256']),64)


if __name__=='__main__':
    unittest.main()
