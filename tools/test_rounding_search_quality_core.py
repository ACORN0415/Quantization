"""CPU checks for exact trace parity and finite binary subproblem enumeration."""
import itertools
import unittest
import torch
from transition_round_core import make_rtn
from postwo_round_core import optimize_rounding
from rounding_search_quality_core import trace_rounding,enumerate_binary_quadratic


class SearchQualityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        g=torch.Generator().manual_seed(971)
        self.v=torch.randn(11,2,3,generator=g)
        self.a0=torch.randn(2,5,11,generator=g).softmax(-1)*.6
        self.a1=self.a0*torch.tensor([1.2,1.8])[:,None,None]
        self.w=torch.randn(6,6,generator=g)*.3

    def test_trace_exact_final_parity_all_modes(self):
        initial=make_rtn(self.v,group=4)
        for local,lam in ((False,0.),(False,1.),(True,1.)):
            candidates=[]
            old,om=optimize_rounding(self.a0,self.a1,self.v,self.w,initial=initial,
                                    group=4,head_local=local,transition_weight=lam)
            new,nm=trace_rounding(self.a0,self.a1,self.v,self.w,initial=initial,
                group=4,head_local=local,transition_weight=lam,
                callback=lambda iteration,state:candidates.append((iteration,state)))
            self.assertTrue(torch.equal(old.codes,new.codes))
            self.assertTrue(torch.equal(old.scale,new.scale))
            self.assertEqual(om['training_objective_history'],nm['training_objective_history'])
            self.assertEqual([i for i,_ in candidates],list(range(33)))
            self.assertTrue(torch.equal(candidates[0][1].codes,initial.codes))
            for _,state in candidates:
                self.assertTrue(torch.equal(state.codes[11:],initial.codes[11:]))
                self.assertEqual(state.bytes(),initial.bytes())
            # Retained callback objects must not alias one another's mutable codes.
            self.assertNotEqual(candidates[0][1].codes.data_ptr(),candidates[1][1].codes.data_ptr())
            self.assertEqual(nm['callback_candidates'],33)

    def test_no_callback_and_zero_steps(self):
        old,om=optimize_rounding(self.a0,self.a1,self.v,self.w,group=4,steps=0)
        new,nm=trace_rounding(self.a0,self.a1,self.v,self.w,group=4,steps=0)
        self.assertTrue(torch.equal(old.codes,new.codes))
        self.assertEqual(nm['callback_candidates'],0)
        self.assertEqual(om['final'],nm['final'])

    def test_enumerator_matches_direct_output_errors(self):
        g=torch.Generator().manual_seed(191)
        p0=torch.randn(3,2,generator=g,dtype=torch.float64)
        p1=torch.randn(3,2,generator=g,dtype=torch.float64)
        effects0=torch.randn(4,3,2,generator=g,dtype=torch.float64)
        effects1=torch.randn(4,3,2,generator=g,dtype=torch.float64)
        dlinear=(effects0*p0+effects1*p1).sum((1,2))
        dquad=.5*(effects0.flatten(1)@effects0.flatten(1).T+effects1.flatten(1)@effects1.flatten(1).T)
        dconst=.5*(p0.square().sum()+p1.square().sum())
        jump=p1-p0;effects_jump=effects1-effects0
        jlinear=2*(effects_jump*jump).sum((1,2))
        jquad=effects_jump.flatten(1)@effects_jump.flatten(1).T
        jconst=jump.square().sum()
        result=enumerate_binary_quadratic(dlinear,dquad,dconst,jlinear,jquad,jconst,batch_size=3)
        ds=[];js=[]
        for index in range(16):
            bits=torch.tensor([(index>>j)&1 for j in range(4)],dtype=torch.float64)
            out0=p0+(bits[:,None,None]*effects0).sum(0)
            out1=p1+(bits[:,None,None]*effects1).sum(0)
            ds.append(.5*(out0.square().sum()+out1.square().sum()))
            js.append((out1-out0).square().sum())
        ds,js=torch.stack(ds),torch.stack(js)
        torch.testing.assert_close(result['d_values'],ds,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(result['j_values'],js,rtol=1e-12,atol=1e-12)
        self.assertEqual(result['best_mean_index'],int(ds.argmin()))
        self.assertEqual(result['best_transition_index'],int((ds+js).argmin()))
        self.assertEqual(result['best_mean_bits'].tolist(),[(int(ds.argmin())>>j)&1 for j in range(4)])

    def test_enumerator_16_coordinates_and_tie_order(self):
        linear=torch.arange(16,dtype=torch.float64)-7.5
        zero=torch.zeros_like(linear);matrix=torch.zeros(16,16,dtype=torch.float64)
        answer=enumerate_binary_quadratic(linear,matrix,0.,zero,matrix,0.,batch_size=1000)
        self.assertEqual(answer['count'],65536)
        self.assertEqual(answer['best_mean_index'],255)
        self.assertEqual(answer['best_transition_index'],255)
        tied=enumerate_binary_quadratic(zero,matrix,0.,zero,matrix,0.)
        self.assertEqual(tied['best_mean_index'],0)
        self.assertEqual(tied['best_transition_index'],0)

    def test_validation(self):
        x=torch.zeros(2,dtype=torch.float64);q=torch.zeros(2,2,dtype=torch.float64)
        for bad in (0,-1,True):
            with self.assertRaises(ValueError):enumerate_binary_quadratic(x,q,0.,x,q,0.,batch_size=bad)
        with self.assertRaises(ValueError):enumerate_binary_quadratic(x,q,float('nan'),x,q,0.)
        with self.assertRaises(ValueError):enumerate_binary_quadratic(x,q,0.,x.float(),q,0.)
        with self.assertRaises(ValueError):trace_rounding(self.a0,self.a1,self.v,self.w,callback=17)


if __name__=='__main__':unittest.main()
