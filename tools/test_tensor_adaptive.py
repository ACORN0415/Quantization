import unittest
import torch
from tensor_adaptive_core import select,encode
from tensor_cross_probe import decode

class Gates(unittest.TestCase):
 def test_projector_and_reproducibility(self):
  torch.manual_seed(77)
  v=torch.linalg.qr(torch.randn(2,2,13,4,dtype=torch.complex128)).Q
  ids,d=select(v,4,31)
  self.assertTrue(torch.equal(ids,select(v,4,31)[0]))
  self.assertLess(d['max_selected_residual'],1e-12)
  for row in ids:self.assertEqual(row.unique().numel(),4)
  # Independent explicit pseudoinverse residual annihilates the selected rows.
  a=v.gather(2,ids[:,None,:,None].expand(-1,2,-1,4))
  p=torch.eye(4,dtype=v.dtype)-torch.linalg.pinv(a)@a
  self.assertLess(float((v@p).abs().max()),1e-12)
 def test_full_rank_codec(self):
  torch.manual_seed(51);x=torch.randn(2,3,19,5)
  for greedy in (False,True):
   state,_=encode(x,5,greedy=greedy)
   target=x.permute(1,2,0,3).reshape(1,57,2,5)
   self.assertLess(float((decode(state)-target).norm()/target.norm()),.002)
 def test_shared_low_rank(self):
  torch.manual_seed(52)
  x=torch.randn(1,3,20,3)@torch.randn(1,1,3,9)
  state,_=encode(x,3)
  target=x.permute(1,2,0,3).reshape(1,60,1,9)
  self.assertLess(float((decode(state)-target).norm()/target.norm()),.003)
if __name__=='__main__':unittest.main()
