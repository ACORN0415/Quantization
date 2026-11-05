"""Synthetic hash-only diagnosis. Does not edit/import the driver or read .pt."""
import hashlib,json,time
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;torch.set_num_threads(4);torch.set_grad_enabled(False)
old=lambda x:hashlib.sha256(x.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
proposed=lambda x:hashlib.sha256(x.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def independent(x):
 kinds={1:torch.int8,2:torch.int16,4:torch.int32,8:torch.int64}
 # Same-element-size reinterpretation has no last-stride==1 requirement.
 return hashlib.sha256(x.detach().view(kinds[x.element_size()]).numpy().tobytes(order='C')).hexdigest()
rows=[]
for dtype in (torch.bfloat16,torch.float16,torch.float32,torch.float64):
 raw=torch.arange(12*128,dtype=torch.float64).reshape(12,1,128).transpose(-1,-2).to(dtype)
 for name,x in [('singleton_last_stride128',raw),('ordinary',raw.squeeze(-1).contiguous()),('transposed',raw.squeeze(-1).t()),('scalar',torch.tensor(1.25,dtype=dtype)),('empty',torch.empty(0,dtype=dtype))]:
  before=independent(x)
  try:original=old(x);failure=None
  except Exception as exc:original=None;failure=repr(exc)
  replacement=proposed(x);assert replacement==before and independent(x)==before
  if original is not None:assert original==replacement
  rows.append({'dtype':str(dtype),'case':name,'shape':list(x.shape),'stride':list(x.stride()),'contiguous':x.is_contiguous(),'old_error':failure,'old_equals_new_if_defined':original is None or original==replacement,'new_equals_independent_byte_digest':True})
report={'scope':'synthetic hash-only cause diagnosis, no source changes','cases':rows,'cases_passed':len(rows),'proposed_change':"x.detach().contiguous().reshape(-1).view(torch.uint8)",'original_tensor_file_reads':0,'GPU_calls':0,'actual_method_effect_measurements':0}
(H/'HASH_STRIDE_PROBE_V1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'cases':len(rows),'old_failures':sum(x['old_error'] is not None for x in rows),'all_new_match_independent_bytes':True}))
