import unittest
import numpy as np
import torch
from agent import HeparinBCQ
from metric import fit_heparin_fqe
from util import ArrayTrainingBuffer
from metric import predict_q


class HeparinFQETests(unittest.TestCase):
    def test_target_batchnorm_buffers_stay_frozen_between_hard_copies(self):
        torch.set_num_threads(1)
        p=HeparinBCQ(num_actions=2,state_dim=1,device='cpu',hidden_node=8,activation='relu',
                     max_timesteps=100,target_update_frequency=25,optimizer_parameters={'lr':.001})
        data=dict(state=np.arange(20)[:,None],next_state=np.arange(20)[:,None]+1,
                  action=np.zeros((20,1)),reward=np.ones((20,1)),done=np.ones((20,1)))
        before={k:v.clone() for k,v in p.Q_target.named_buffers()}
        p.train(ArrayTrainingBuffer(data,seed=42,batch_size=8))
        for k,v in p.Q_target.named_buffers():self.assertTrue(torch.equal(v,before[k]))

    def test_repeated_rewards_fit_the_discounted_return(self):
        # 40 identical complete episodes with two rewards: V(s0)=1+.5*1=1.5.
        s=np.tile([[0.],[1.]],(40,1)).astype(np.float32)
        ns=np.ones_like(s);r=np.ones((80,1));d=np.tile([[0.],[1.]],(40,1))
        data=dict(state=s,next_state=ns,action=np.zeros((80,1)),reward=r,done=d)
        q,report=fit_heparin_fqe(data,np.ones((80,1),dtype=np.float32),np.repeat(np.arange(40),2),
                               gamma=.5,epochs=200,min_epochs=100,patience=30,hidden_dim=16,lr=.005,batch_size=80)
        values=predict_q(q,np.array([[0.],[1.]],dtype=np.float32),num_threads=1)[:,0]
        np.testing.assert_allclose(values,[1.5,1.],atol=.05)
        self.assertEqual(report['value_bound'],2.)


if __name__=='__main__':unittest.main()
