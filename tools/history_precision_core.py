"""Causal equal-budget high precision schedules, inspired by BanLast."""
import torch
POLICIES=('fixed','independent','banlast1','roundrobin')
def schedules(n=4680,steps=4,fraction=.25,seed=9101):
 m=int(n*fraction)
 if m<=0 or n!=m*steps:raise ValueError('this pilot needs an exact one-pass round robin')
 result={}
 for policy in POLICIES:
  gen=torch.Generator().manual_seed(seed);permutation=torch.randperm(n,generator=gen)
  initial=permutation[:m];previous=None;masks=[]
  for step in range(steps):
   if step==0 or policy=='fixed':idx=initial
   elif policy=='roundrobin':idx=permutation[step*m:(step+1)*m]
   elif policy=='independent':idx=torch.randperm(n,generator=gen)[:m]
   else:
    candidates=(~previous).nonzero().flatten()
    idx=candidates[torch.randperm(len(candidates),generator=gen)[:m]]
   mask=torch.zeros(n,dtype=torch.bool);mask[idx]=True;masks.append(mask);previous=mask
  result[policy]=torch.stack(masks)
 return result
