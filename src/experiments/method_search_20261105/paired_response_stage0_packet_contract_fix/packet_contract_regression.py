"""Actual V4/V5 hash-predicate AST on independent synthetic producer payloads."""
import ast,copy,hashlib,importlib.util,json,sys,time
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;OLD=H.parent/'paired_response_stage0/diagnose.py';NEW=H/'diagnose.py'
torch.set_num_threads(4);torch.set_grad_enabled(False)
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
old=load(OLD,'packet_contract_old');new=load(NEW,'packet_contract_new')
def predicate(path):
 tree=ast.parse(path.read_text());f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='evaluate_layer')
 found=[x for x in ast.walk(f) if isinstance(x,ast.If) and any(isinstance(v,ast.Constant) and v.value=='packet tensor anchor' for v in ast.walk(x))]
 # The target if contains only the anchor ValueError body.
 node=next(x for x in found if len(x.body)==1 and isinstance(x.body[0],ast.Raise))
 return compile(ast.Module(body=[copy.deepcopy(node)],type_ignores=[]),str(path),'exec')
old_pred,new_pred=predicate(OLD),predicate(NEW)
def pack_independent(codes):
 values=codes.flatten().tolist();return torch.tensor([(values[i]&15)|((values[i+1]&15)<<4) for i in range(0,len(values),2)],dtype=torch.uint8)
def unpack_independent(p,shape):
 values=[]
 for b in p.tolist():
  for n in (b&15,b>>4):values.append(n-16 if n>=8 else n)
 return torch.tensor(values,dtype=torch.int8).reshape(shape)
class Codec:
 unpack=staticmethod(unpack_independent)
codes=torch.tensor([-8,-1,0,1,7,-3,2,-7],dtype=torch.int8).reshape(2,4);packed=pack_independent(codes);scale=torch.tensor([[.125],[.25]],dtype=torch.float16)
sha_bytes=lambda b:hashlib.sha256(b).hexdigest()
# The independent producer follows causal replay's packed uint8 bytes contract.
producer={'code_sha256':sha_bytes(bytes(packed.tolist())),'scale_sha256':sha_bytes(scale.view(torch.int16).numpy().tobytes())}
base={'packed':packed,'scale':scale,'shape':tuple(codes.shape)};rows=[]
def check(name,compiled,hashfn,kp,manifest,accept):
 before={k:hashfn(v) for k,v in kp.items() if isinstance(v,torch.Tensor)};error=None
 try:exec(compiled,{'kp':kp,'row':{'K':{'R_bf16':manifest}},'base_name':'R_bf16','tsha':hashfn,'c':Codec})
 except ValueError as exc:error=str(exc)
 assert (error is None)==accept,(name,error)
 assert before=={k:hashfn(v) for k,v in kp.items() if isinstance(v,torch.Tensor)}
 rows.append({'case':name,'accepted':error is None,'error':error,'expected':accept,'input_preserved':True})
check('V4 rejects valid original-producer packet',old_pred,old.tsha,base,producer,False)
check('V5 accepts valid packed producer anchor',new_pred,new.tsha,base,producer,True)
wrong={**producer,'code_sha256':sha_bytes(codes.view(torch.uint8).numpy().tobytes())}
check('V4 accepts mistaken unpacked schema counterexample',old_pred,old.tsha,base,wrong,True)
check('V5 rejects unpacked-hash manifest',new_pred,new.tsha,base,wrong,False)
corrupt={**base,'packed':packed.clone()};corrupt['packed'][0]^=1
check('V5 rejects changed packed payload',new_pred,new.tsha,corrupt,producer,False)
corruptscale={**base,'scale':scale.clone()};corruptscale['scale'][0,0]+=.125
check('V5 rejects changed scale',new_pred,new.tsha,corruptscale,producer,False)
assert torch.equal(unpack_independent(packed,codes.shape),codes)
# Exact source text inversion and AST equivalence after only the two authorized edits.
oldtext=OLD.read_text();newtext=NEW.read_text();restored=newtext.replace("tsha(kp['packed'])!=row['K'][base_name]['code_sha256']","tsha(c.unpack(kp['packed'],kp['shape']))!=row['K'][base_name]['code_sha256']").replace('SOURCES_V5.json','SOURCES_V4.json')
assert restored==oldtext;assert ast.dump(ast.parse(restored),include_attributes=False)==ast.dump(ast.parse(oldtext),include_attributes=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'status':'passed','cases':len(rows),'rows':rows,'independent_producer':'packed uint8 payload and stored FP16 scale; Python nibble loop','actual_source_predicate_executed':True,'authorized_edits_only_text_and_AST_equal':True,'source_sha256':{str(OLD):sha(OLD),str(NEW):sha(NEW),str(Path(__file__)):sha(Path(__file__))},'python_version':sys.version,'torch_version':str(torch.__version__),'real_tensor_reads':0,'GPU_calls':0,'actual_effect_measurements':0,'scope':'packet-anchor branch and exact source diff only; not a new full-pipeline run'}
with (H/'PACKET_CONTRACT_REGRESSION_V1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'status':'passed','cases':len(rows),'authorized_edits_only_AST':True}))
