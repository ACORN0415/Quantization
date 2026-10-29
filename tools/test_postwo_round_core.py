"""CPU synthetic checks for the isolated post-Wo value-code optimizer."""
import unittest

import torch

from transition_round_core import make_rtn
from postwo_round_core import optimize_rounding,output_error_metrics,_objective,_project,_pullback


class PostWoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        g=torch.Generator().manual_seed(20261029)
        self.v=torch.randn(12,2,3,generator=g)
        self.a0=torch.randn(2,5,12,generator=g).softmax(-1)*.6
        self.a1=self.a0*torch.tensor([1.2,1.8])[:,None,None]
        self.w=torch.randn(6,6,generator=g)*.3

    def test_full_metric_formula(self):
        g=torch.Generator().manual_seed(23)
        e0=torch.randn(2,5,3,generator=g,dtype=torch.float64)
        c=torch.tensor([1.2,1.8],dtype=torch.float64).repeat_interleave(3)
        wo=self.w.double();m=wo.T@wo;C=torch.diag(c)
        lam=.7
        h=.5*(m+C.T@m@C)+lam*(C-torch.eye(6)) .T@m@(C-torch.eye(6))
        e=e0.permute(1,0,2).reshape(5,6)
        direct=_objective(e0,e0*torch.tensor([1.2,1.8],dtype=torch.float64)[:,None,None],wo,lam,False)[0]
        equivalent=torch.einsum('qi,ij,qj->',e,h,e)
        torch.testing.assert_close(direct,equivalent)

    def test_cross_head_cancellation(self):
        e0=torch.tensor([[[1.]], [[-1.]]]);e1=e0*torch.tensor([1.,2.])[:,None,None]
        wo=torch.tensor([[1.,1.]])
        full=_objective(e0,e1,wo,1.,False)
        local=_objective(e0,e1,wo,1.,True)
        self.assertNotEqual(float(full[0]),float(local[0]))
        self.assertEqual(float(full[1]),.5)
        self.assertEqual(float(local[1]),3.5)

    def test_manual_projection_gradient(self):
        g=torch.Generator().manual_seed(11)
        for local in (False,True):
            e0=torch.randn(2,5,3,generator=g,requires_grad=True)
            e1=torch.randn(2,5,3,generator=g,requires_grad=True)
            loss=_objective(e0,e1,self.w,.6,local)[0]
            d0,d1=torch.autograd.grad(loss,(e0,e1))
            p0,p1=_project(e0,self.w,local),_project(e1,self.w,local)
            expected0=_pullback(p0+1.2*(p0-p1),self.w,2,3,local)
            expected1=_pullback(p1+1.2*(p1-p0),self.w,2,3,local)
            torch.testing.assert_close(d0,expected0)
            torch.testing.assert_close(d1,expected1)

    def test_optimizer_codes_fixed_scale_and_budget(self):
        initial=make_rtn(self.v,group=4)
        for local,lam in ((False,0.),(False,1.),(True,1.)):
            answer,meta=optimize_rounding(self.a0,self.a1,self.v,self.w,
                initial=initial,group=4,head_local=local,transition_weight=lam,steps=32)
            self.assertTrue(torch.equal(answer.scale,initial.scale))
            self.assertEqual(answer.bytes(),initial.bytes())
            self.assertEqual(meta['candidates_evaluated'],33)
            self.assertLessEqual(meta['final']['objective'],meta['initial']['objective'])
            low=(self.v/initial.scale[:,0].repeat_interleave(4,0)).floor().clamp(-7,7)
            high=(self.v/initial.scale[:,0].repeat_interleave(4,0)).ceil().clamp(-7,7)
            signed=answer.codes[:12].float()-7
            self.assertTrue(bool(((signed==low)|(signed==high)).all()))

    def test_partial_token_group_keeps_padding(self):
        values=self.v[:11]
        a0,a1=self.a0[:,:,:11],self.a1[:,:,:11]
        initial=make_rtn(values,group=4)
        result,meta=optimize_rounding(a0,a1,values,self.w,group=4,initial=initial,
                                      transition_weight=1.,steps=4)
        self.assertEqual(tuple(result.restore().shape),tuple(values.shape))
        self.assertTrue(torch.equal(result.codes[11:],initial.codes[11:]))
        self.assertEqual(result.bytes(),initial.bytes())
        self.assertLessEqual(meta['final']['objective'],meta['initial']['objective'])

    def test_zero_error_and_zero_projection(self):
        state=make_rtn(self.v,group=4)
        zero=output_error_metrics(self.a0,self.a1,self.v,self.v,self.w,1.)
        self.assertEqual(zero['objective'],0.)
        result,meta=optimize_rounding(self.a0,self.a1,self.v,torch.zeros_like(self.w),group=4,steps=3)
        self.assertTrue(torch.equal(result.codes,state.codes))
        self.assertEqual(meta['final']['objective'],0.)

    def test_reject_nonfinite_and_bad_shape(self):
        with self.assertRaises(ValueError):
            optimize_rounding(self.a0,self.a1,self.v,self.w[:,:5])
        bad=self.w.clone();bad[0,0]=float('nan')
        with self.assertRaises(ValueError):
            optimize_rounding(self.a0,self.a1,self.v,bad)


if __name__=='__main__': unittest.main()
