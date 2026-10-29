import unittest
import torch
from history_precision_core import schedules
class Gates(unittest.TestCase):
 def test_equal_budget_and_history(self):
  for seed in (9101,9102,9103):
   s=schedules(n=40,seed=seed)
   for name,x in s.items():
    self.assertTrue(torch.equal(x.sum(1),torch.full((4,),10)))
    self.assertTrue(torch.equal(x,schedules(n=40,seed=seed)[name]))
    self.assertTrue(torch.equal(x[0],s['fixed'][0]))
   self.assertTrue(torch.equal(s['fixed'],s['fixed'][0].expand(4,-1)))
   self.assertEqual(int((s['banlast1'][1:]&s['banlast1'][:-1]).sum()),0)
   self.assertTrue(torch.equal(s['roundrobin'].sum(0),torch.ones(40,dtype=torch.long)))
 def test_first_choices_do_not_use_future(self):
  # Scheduler takes no values, queries, model outputs or future RNG input.
  a=schedules(n=40,seed=9101);torch.randn(100)
  b=schedules(n=40,seed=9101)
  self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
if __name__=='__main__':unittest.main()
