"""Check cumulative caps against hand arithmetic and an independent product oracle."""
import unittest
import numpy as np

from metric import evaluate_ope, evaluate_wdr, evaluate_policy_arrays
from scripts.evaluate_clipping_sensitivity import direct_reference


class ClippingTests(unittest.TestCase):
    def test_cap_does_not_feed_back_into_later_prefix(self):
        # Original prefixes=(4,1), capped prefixes=(2,1), NOT (2,.5).
        action = np.zeros(4, dtype=int)
        pi = np.array([[.8,.2],[.05,.95],[.2,.8],[.2,.8]])
        b = np.tile([.2,.8], (4,1))
        result = evaluate_ope(action, [0,1,0,0], [0,1,0,1], pi, b,
                              np.zeros_like(pi), gamma=1, cumulative_weight_cap=2,
                              n_bootstrap=0)
        self.assertAlmostEqual(result['wis'], .5)
        self.assertAlmostEqual(result['wdr'], .5)
        self.assertAlmostEqual(result['weights']['trajectory_ess'], 2.)
        self.assertEqual(result['clipping']['capped_observed_prefix_count'], 1)

    def test_capped_absorbing_weight_stays_in_denominator(self):
        pi = np.array([[.8,.2],[.2,.8],[.4,.6]])
        b = np.tile([.2,.8], (3,1))
        result = evaluate_ope([0,0,0], [1,0,-1], [1,0,1], pi, b,
                              np.zeros_like(pi), gamma=1, cumulative_weight_cap=2,
                              n_bootstrap=0)
        self.assertAlmostEqual(result['wdr'], 2/3-1/2)
        self.assertAlmostEqual(result['wis'], 0)

    def test_previous_capped_prefix_is_used_for_value_term(self):
        action = [0,0,0,0]
        pi = np.array([[.8,.2],[.05,.95],[.2,.8],[.2,.8]])
        b = np.tile([.2,.8], (4,1))
        q = np.array([[1,2],[3,4],[5,6],[7,8]])
        kwargs = dict(gamma=.9, cumulative_weight_cap=2, n_bootstrap=0)
        result = evaluate_ope(action, [0,1,0,-1], [0,1,0,1], pi,b,q,**kwargs)
        # t0: V average=3.5, weighted Q=7/3 => 7/6.
        # t1: reward average=0, weighted Q=5, previous-weight V=157/30.
        self.assertAlmostEqual(result['wdr'], 7/6+.9*(157/30-5))
        self.assertAlmostEqual(result['dr'], direct_reference(
            action,[0,1,0,-1],[0,1,0,1],pi,b,q,**{k:v for k,v in kwargs.items() if k!='n_bootstrap'})['dr'])

    def test_ragged_random_cases_and_zero_gamma_match_direct_products(self):
        rng=np.random.RandomState(131)
        for gamma in [0.,.98,1.]:
            for scope in ['ratio_cap','cumulative_weight_cap']:
                for cap in [None,.5,2.,100.]:
                    lengths=rng.randint(1,6,size=6); n=int(lengths.sum())
                    done=np.zeros(n,dtype=int);done[np.cumsum(lengths)-1]=1
                    action=rng.randint(0,3,size=n);reward=rng.uniform(-1,1,n)
                    pi=rng.dirichlet([1,2,1],size=n);b=rng.dirichlet([2,1,2],size=n)
                    q=rng.uniform(-1,1,(n,3));kw={scope:cap,'gamma':gamma}
                    a=evaluate_ope(action,reward,done,pi,b,q,n_bootstrap=0,**kw)
                    ref=direct_reference(action,reward,done,pi,b,q,**kw)
                    for key in ['dr','wdr','wis']:
                        self.assertAlmostEqual(a[key],ref[key],places=10)
                    np.testing.assert_allclose(a['weights']['per_decision_ess_with_absorbing_padding'],ref['step_ess'],rtol=1e-12)

    def test_zero_target_mass_remains_zero_with_cap(self):
        result=evaluate_ope([1,1],[1,-1],[1,1],[[1,0]]*2,[[.5,.5]]*2,
                            [[0,0]]*2,cumulative_weight_cap=2,n_bootstrap=10)
        self.assertIsNone(result['wis']);self.assertIsNone(result['wdr'])

    def test_shared_patient_bootstrap_and_interfaces(self):
        pi=np.array([[.8,.2],[.2,.8]]);b=np.array([[.2,.8]]*2);q=np.zeros_like(pi)
        arrays={'action':[0,0],'reward':[1,-1],'done':[1,1]}
        kwargs=dict(cumulative_weight_cap=2,n_bootstrap=20,episode_groups=[7,7])
        a=evaluate_policy_arrays(arrays,pi,b,q,**kwargs)
        w=evaluate_wdr(arrays['action'],arrays['reward'],arrays['done'],pi,b,q,**kwargs)
        self.assertEqual(a['wdr'],w['value'])
        self.assertAlmostEqual(a['bootstrap']['intervals']['wdr']['low'],1/3)
        self.assertEqual(w['cumulative_weight_cap'],2)

    def test_caps_must_be_positive_finite_and_mutually_exclusive(self):
        for cap in [0,-1,np.inf,np.nan]:
            with self.assertRaisesRegex(ValueError,'cumulative_weight_cap'):
                evaluate_ope([0],[1],[1],[[1,0]],[[1,0]],[[0,0]],cumulative_weight_cap=cap,n_bootstrap=0)
        with self.assertRaisesRegex(ValueError,'not both'):
            evaluate_ope([0],[1],[1],[[1,0]],[[1,0]],[[0,0]],ratio_cap=2,cumulative_weight_cap=2,n_bootstrap=0)


if __name__=='__main__':
    unittest.main()
