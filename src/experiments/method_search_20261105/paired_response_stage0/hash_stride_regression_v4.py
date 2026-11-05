"""Synthetic byte hashing tests; original tensor files and model are never read."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;torch.set_num_threads(4);torch.set_grad_enabled(False)
spec=importlib.util.spec_from_file_location('hash_fixed_driver',HERE/'diagnose.py');d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)
old=lambda x:hashlib.sha256(x.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
flatten_only=lambda x:hashlib.sha256(x.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def independent(x):return hashlib.sha256(x.detach().view({1:torch.int8,2:torch.int16,4:torch.int32,8:torch.int64}[x.element_size()]).numpy().tobytes(order='C')).hexdigest()
rows=[]
for dtype in (torch.bfloat16,torch.float16,torch.float32,torch.float64):
 raw=torch.arange(12*128,dtype=torch.float64).reshape(12,1,128).transpose(-1,-2).to(dtype)
 storage=torch.arange(300,dtype=torch.float64).to(dtype);singleton=torch.as_strided(storage,(1,1),(128,128),5)
 for name,x in [('singleton_last',raw),('all_singleton_stride128',singleton),('ordinary',raw.squeeze(-1).contiguous()),('transposed',raw.squeeze(-1).t()),('scalar',torch.tensor(1.25,dtype=dtype)),('empty',torch.empty(0,dtype=dtype))]:
  before=independent(x);failures={}
  for label,fn in [('old',old),('flatten_only',flatten_only)]:
   try:value=fn(x);failures[label]=None;assert value==before
   except RuntimeError as error:failures[label]=repr(error)
  assert d.tsha(x)==before and independent(x)==before
  rows.append({'dtype':str(dtype),'case':name,'shape':list(x.shape),'stride':list(x.stride()),'old_error':failures['old'],'flatten_only_error':failures['flatten_only'],'fixed_matches_independent':True,'input_preserved':True})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out={'status':'passed','cases':len(rows),'rows':rows,'driver_sha256':sha(HERE/'diagnose.py'),'regression_sha256':sha(Path(__file__)),'original_tensor_reads':0,'GPU_calls':0,'scope':'generic hash helper only; no full pipeline completion claim'}
with (HERE/'HASH_STRIDE_REGRESSION_V4.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({'status':'passed','cases':len(rows),'old_failures':sum(x['old_error'] is not None for x in rows),'flatten_only_failures':sum(x['flatten_only_error'] is not None for x in rows)}))
