import unittest
import numpy as np
import torch
from agent import DDQN
from agent import BCQ
from agent import StandardDQN
from agent import FixedLearningRateBCQ
from util import ArrayTrainingBuffer


class Table(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor(values,dtype=torch.float32))
    def forward(self, states):
        return self.values[states[:,0].long()]


class FixedBuffer:
    def __init__(self, terminal=False): self.terminal=terminal
    def sample(self):
        return (torch.tensor([[0.]]),torch.tensor([[0]]),torch.tensor([[1.]]),
                torch.tensor([[-1. if self.terminal else 0.]]),
                torch.tensor([[float(self.terminal)]]),torch.zeros(1,1))


class TrainingSemanticsTests(unittest.TestCase):
    def make(self, cls):
        p=cls(num_actions=2,state_dim=1,device='cpu',hidden_node=4,discount=1.,target_update_frequency=999)
        p.Q=Table([[0.,0.],[10.,0.]])
        p.Q_target=Table([[0.,0.],[2.,5.]])
        p.Q_optimizer=torch.optim.SGD(p.Q.parameters(),lr=.1)
        return p

    def test_dqn_and_double_dqn_differ_when_online_and_target_disagree(self):
        dqn,ddqn=self.make(StandardDQN),self.make(DDQN)
        dqn.train(FixedBuffer());ddqn.train(FixedBuffer())
        # Analytic SGD outcome: DQN uses 5, DDQN uses target action0 value2.
        self.assertAlmostEqual(dqn.Q.values[0,0].item(),1.)
        self.assertAlmostEqual(ddqn.Q.values[0,0].item(),.4,places=6)

    def test_both_mask_bootstrap_at_terminal(self):
        for cls in [StandardDQN,DDQN]:
            p=self.make(cls);p.train(FixedBuffer(terminal=True))
            self.assertAlmostEqual(p.Q.values[0,0].item(),-.2,places=6)

    def test_bcq_zero_lr_regression_and_actual_parameter_update(self):
        torch.set_num_threads(1)
        cfg=dict(num_actions=3,state_dim=2,device='cpu',hidden_node=8,activation='relu',
                 max_timesteps=1500,optimizer_parameters={'lr':1e-6,'weight_decay':1e-5})
        raw=BCQ(**cfg)
        self.assertEqual(raw.Q_optimizer.param_groups[0]['lr'],0.)
        p=FixedLearningRateBCQ(**cfg)
        self.assertEqual(p.Q_optimizer.param_groups[0]['lr'],1e-6)
        rng=np.random.default_rng(9)
        arrays=dict(state=rng.normal(size=(20,2)),next_state=rng.normal(size=(20,2)),
                    action=np.arange(20).reshape(-1,1)%3,reward=np.ones((20,1)),done=np.ones((20,1)))
        before={k:v.clone() for k,v in p.Q.named_parameters()}
        p.train(ArrayTrainingBuffer(arrays,seed=42))
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in p.Q.named_parameters()))

    def test_uniform_transition_stream_is_paired_across_algorithms(self):
        arrays={k:np.arange(20).reshape(-1,1) for k in ['state','action','next_state','reward','done']}
        x,y=ArrayTrainingBuffer(arrays,seed=43),ArrayTrainingBuffer(arrays,seed=43)
        for i in range(3):
            for a,b in zip(x.sample(),y.sample()):
                self.assertTrue(torch.equal(a,b))


if __name__=='__main__': unittest.main()
