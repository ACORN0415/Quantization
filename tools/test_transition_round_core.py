"""CPU contracts for two-state output loss and immutable-scale rounding."""
import unittest
import torch
from transition_round_core import make_rtn,output_error_metrics,optimize_rounding


class TransitionRoundTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(991)
        self.v=torch.randn(19,2,4)
        self.a0=torch.rand(2,5,19);self.a0/=self.a0.sum(-1,keepdim=True)
        self.a1=self.a0.clone();self.a1[:,:,:4]=0;self.a1/=self.a1.sum(-1,keepdim=True)

    def test_loss_matches_direct_attention_errors(self):
        state=make_rtn(self.v)
        x=state.restore().permute(1,0,2);v=self.v.permute(1,0,2)
        e0=self.a0@x-self.a0@v;e1=self.a1@x-self.a1@v
        m=output_error_metrics(self.a0,self.a1,state.restore(),self.v,transition_weight=1)
        expected=float((e0.square().sum()+e1.square().sum())/2+(e1-e0).square().sum())
        self.assertAlmostEqual(m['objective'],expected,places=7)

    def test_lossless_and_identical_states(self):
        m=output_error_metrics(self.a0,self.a1,self.v,self.v,1)
        self.assertEqual(m['objective'],0)
        m=output_error_metrics(self.a0,self.a0,make_rtn(self.v).restore(),self.v,1)
        self.assertEqual(m['transition_error_sq'],0)

    def test_prefix_scale_payload_and_neighbor_contract(self):
        initial=make_rtn(self.v)
        answers=[]
        for weight in (0.,1.):
            answer,meta=optimize_rounding(self.a0,self.a1,self.v,suffix_tokens=7,steps=32,transition_weight=weight,initial=initial)
            answers.append(answer)
            self.assertTrue(torch.equal(answer.codes[:12],initial.codes[:12]))
            self.assertTrue(torch.equal(answer.scale,initial.scale))
            self.assertEqual(meta['candidates_evaluated'],33)
            scale=initial.scale[:,0][torch.arange(12,19)//16].float()
            norm=self.v[12:]/scale
            lo=norm.floor().clamp(-7,7).add(7).to(torch.uint8)
            hi=norm.ceil().clamp(-7,7).add(7).to(torch.uint8)
            self.assertTrue(bool(((answer.codes[12:19]==lo)|(answer.codes[12:19]==hi)).all()))
            self.assertLessEqual(meta['final']['objective'],meta['initial']['objective'])
        self.assertEqual(answers[0].bytes(),answers[1].bytes())
        self.assertEqual(initial.bytes()['resident'],32*2*4+2*1*2*4*2)
        self.assertEqual(initial.bytes()['packed_estimate'],32*2*4//2+2*1*2*4*2)

    def test_zero_step_and_deterministic_fixed_budget(self):
        initial=make_rtn(self.v)
        unchanged,meta=optimize_rounding(self.a0,self.a1,self.v,suffix_tokens=7,steps=0,initial=initial)
        self.assertTrue(torch.equal(unchanged.codes,initial.codes))
        a,am=optimize_rounding(self.a0,self.a1,self.v,suffix_tokens=7,steps=8)
        b,bm=optimize_rounding(self.a0,self.a1,self.v,suffix_tokens=7,steps=8)
        self.assertTrue(torch.equal(a.codes,b.codes));self.assertEqual(am,bm)

    def test_mean_baseline_training_never_worse_than_rtn(self):
        for seed in (2,3,4):
            torch.manual_seed(seed)
            v=torch.randn_like(self.v)
            _,meta=optimize_rounding(self.a0,self.a1,v,suffix_tokens=19,steps=32,transition_weight=0)
            self.assertLessEqual(meta['final']['mean_error_sq'],meta['initial']['mean_error_sq'])

    def test_rejections_and_zero_weights(self):
        with self.assertRaises(ValueError):make_rtn(self.v*float('nan'))
        with self.assertRaises(ValueError):optimize_rounding(self.a0,self.a1,self.v,suffix_tokens=0)
        with self.assertRaises(ValueError):optimize_rounding(self.a0,self.a1,self.v,suffix_tokens=7,transition_weight=-1)
        with self.assertRaises(ValueError):output_error_metrics(self.a0,self.a1[:,:,:18],self.v,self.v)
        a,meta=optimize_rounding(torch.zeros_like(self.a0),torch.zeros_like(self.a1),self.v,suffix_tokens=7)
        self.assertTrue(torch.equal(a.codes,make_rtn(self.v).codes));self.assertEqual(meta['final']['objective'],0)


if __name__=='__main__':unittest.main()
