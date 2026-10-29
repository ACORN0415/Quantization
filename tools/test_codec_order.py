import unittest
import torch
from codec_order_core import *
class Gates(unittest.TestCase):
 def test_full_rank_matrix(self):
  torch.manual_seed(99);x=torch.randn(2,3,13,5)
  target=x.permute(1,2,0,3).reshape(1,39,2,5)
  for method in ('tarp','tsvd','pivot_qr','range'):
   raw=decompose(x,method,5)
   y=reconstruct(method,x.shape,raw['factors'])
   self.assertLess(float((y-target).norm()/target.norm()),2e-5,method)
 def test_low_rank_tensor(self):
  torch.manual_seed(32)
  x=torch.einsum('hsr,htr,hdr->htsd',torch.randn(2,13,2),torch.randn(2,3,2),torch.randn(2,7,2))
  target=x.permute(1,2,0,3).reshape(1,39,2,7)
  for method in ('tucker','tt'):
   raw=decompose(x,method,2)
   y=reconstruct(method,x.shape,raw['factors'])
   self.assertLess(float((y-target).norm()/target.norm()),2e-5,method)
 def test_factor_packing_reference(self):
  # Independent scalar reconstruction, nonmultiple-of-16 tail, negatives and zeros.
  x=torch.linspace(-3,2,65).reshape(5,13);x[0]=0
  for bits in (2,3,4):
   state=pack_factor(x,bits);out=unpack_factor(state).flatten();qmax=(1<<(bits-1))-1
   for i,value in enumerate(x.flatten()):
    scale=float(state['maximum'][i//16])/qmax
    expected=max(-qmax,min(qmax,round(float(value)/scale)))*scale
    self.assertAlmostEqual(float(out[i]),expected,places=5)
 def test_payload_and_ranks(self):
  torch.manual_seed(24);x=torch.randn(2,3,13,7)
  for method in METHODS:
   raw=decompose(x,method,3)
   for bits in (None,2,3,4):
    state=store(raw,bits)
    self.assertEqual(payload_bytes(state),expected_bytes(method,x.shape,3,bits))
    self.assertTrue(torch.isfinite(decode(state)).all())
    self.assertEqual(choose_rank(method,x.shape,expected_bytes(method,x.shape,3,bits),bits),3)
if __name__=='__main__':unittest.main()
