import sys
sys.dont_write_bytecode=True
import importlib.util,json,math,hashlib,time
from pathlib import Path
import torch
p=Path('/home/acorn0415/Quantization/src/experiments/method_search_20261105/paired_response_preparation/core.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
before=sha(p);spec=importlib.util.spec_from_file_location('paired_readonly_review',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
torch.set_num_threads(4);torch.set_grad_enabled(False)
cases=[]
for kind in ['real_gaussian_float64','real_gaussian_bfloat16']:
 model=m.PairedResponse.gaussian(1,128,1,17)
 # A legitimate Gaussian map; choose two finite synthetic keys near its feature peak.
 center=model.omega[:,:,0].double()*128**.25
 delta=torch.zeros_like(center);delta[0,0]=128**.25
 key=torch.stack((center,center+delta),1)
 if kind.endswith('bfloat16'):key=key.bfloat16()
 khat=key.flip(1).contiguous()
 vp=torch.tensor([[[1.],[0.]]],dtype=key.dtype);vn=-vp
 state1,d1=model.summarize('positive',key,khat,vp)
 state2,d2=model.summarize('negative',key,khat,vn)
 model.append(state1);model.append(state2)
 q=torch.zeros(1,1,128,dtype=key.dtype)
 # Four historical values sum to zero. One current value=5 yields exact base output=1.
 vals=torch.cat((vp,vn,torch.tensor([[[5.]]],dtype=key.dtype)),1).double()
 scores=torch.zeros(1,1,5,dtype=torch.float64)
 base=scores.softmax(-1)@vals;logz=scores.logsumexp(-1)
 got=model.read(q,base,logz,min_denominator_ratio=.1)
 # The two saved numerator states cancel bit-for-bit; each denominator state is zero.
 # Thus the correction of these saved states is exactly zero in real arithmetic.
 cancel_exact=torch.equal(state1.dn,-state2.dn) and torch.equal(state1.offset,state2.offset) and not bool(state1.dz.any()) and not bool(state2.dz.any())
 cases.append({'case':kind,'source_state_generated_by_summarize':True,'gaussian_seed':17,'heads':1,'dim':128,'rank':1,'offset':state1.offset.tolist(),'dz1':state1.dz.tolist(),'dn1':state1.dn.tolist(),'dn2':state2.dn.tolist(),'stored_corrections_exactly_cancel':cancel_exact,'base_output':base.tolist(),'output':got.output.tolist(),'expected_from_saved_states':base.tolist(),'absolute_error':float((got.output-base).abs().max()),'fallback':got.fallback.tolist(),'log_denominator_ratio':got.log_denominator_ratio.tolist(),'diagnostics':got.diagnostics,'preparation_diagnostics':[d1,d2]})
require_same=before==sha(p)
out={'source_sha256':before,'source_unchanged':require_same,'synthetic_only':True,'real_capture_reads':0,'GPU_calls':0,'cuda_initialized':torch.cuda.is_initialized(),'model_loads':0,'time_utc':time.time(),'cases':cases}
target=Path('/home/acorn0415/Quantization/docs/reviews/20261105_METHOD_SEARCH/PAIRED_CORE_INDEPENDENT_PROBE_CODEX.json')
with target.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2))
