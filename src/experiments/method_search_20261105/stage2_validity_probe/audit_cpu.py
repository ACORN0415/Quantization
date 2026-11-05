import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
from pathlib import Path
import hashlib,json,math,time
import torch
torch.set_num_threads(4);torch.set_grad_enabled(False)
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((P/'RESULT.json').read_text());lock=json.loads((P/'LOCK.json').read_text());start=time.monotonic();checks=[]
def ck(name,ok):assert ok,name;checks.append(name)
for p,h in lock['sha256'].items():ck(p,sha(p)==h)
ck('tensor hash',sha(P/'TENSORS.pt')==r['tensors']['sha256'])
t=torch.load(P/'TENSORS.pt',map_location='cpu',weights_only=True)
def metric(a,b):
 return {'all_element_bytes_equal':torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)), 'unequal_elements':int((a!=b).sum()),'max_abs_diff':float((a.float()-b.float()).abs().max())}
for impl,row in r['impl'].items():
 for check,a,b in [('first_reference','first','ref1'),('second_reference','second','ref2'),('first_preserved','first_after_second','first')]:
  for field,x,y in zip(('packed','scale','sentinel'),t[impl][a],t[impl][b]):ck(impl+'/'+check+'/'+field,metric(x,y)==row[check][field])
for kind in ('first','second'):
 for suffix in ('z','scale'):
  ck(kind+'/'+suffix,metric(t['preprocess_'+kind]['graph_'+suffix],t['preprocess_'+kind]['eager_'+suffix])==r['preprocess'][kind]['graph_vs_eager_'+suffix])
ck('no extra GPU',not torch.cuda.is_initialized())
# CPU-only dispatch observation, using identical source shapes/strides and helper expression.
d=torch.load(P.parent/'obq_gpu_cost_stage1/prepared_l18_c1.pt',map_location='cpu',weights_only=True);k=d['k'];h=torch.ones(1,1)
while h.shape[0]<128:h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
h=h/math.sqrt(128);second=(-k.float()*.75).bfloat16();dispatch={}
for name,x in (('first',k),('second',second)):
 xf=x.float()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:y=xf@h
 ops=[e.key for e in prof.key_averages() if e.key in ('aten::matmul','aten::mm','aten::bmm')]
 dispatch[name]={'BF16_stride':list(x.stride()),'FP32_stride':list(xf.stride()),'CPU_ATen_ops':ops,'contiguous':xf.is_contiguous()}
ck('completed exact investigation',r['status']=='completed' and r['compile_calls']==r['quality_calls']==r['timing_samples']==0)
out={'passed':True,'checks_passed':len(checks),'elapsed_seconds':time.monotonic()-start,'result_sha256':sha(P/'RESULT.json'),'tensor_sha256':sha(P/'TENSORS.pt'),'CPU_dispatch':dispatch,'GPU_reruns':0,
 'scope':'Packet/z/scale mismatch statistics recomputed from saved CPU tensors. Dispatch profile is CPU-only, not a CUDA kernel trace.',
 'ATen_source':'https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/LinearAlgebra.cpp#L1824'}
with (P/'AUDIT_CPU.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
