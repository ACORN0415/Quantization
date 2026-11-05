"""CPU toy recurrence checked against the pinned upstream rollout function.

No checkpoint, dataset, real model, CUDA, or network is used.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
start=time.time()
spec=importlib.util.spec_from_file_location('_diagnostic',HERE/'experiment.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
try:driver.validate({'status':'UNAPPROVED'})
except RuntimeError:pass
else:raise AssertionError('unapproved diagnostic accepted')
import torch
torch.set_num_threads(2)
assert not torch.cuda.is_initialized()
source=Path('/home/acorn0415/sfq_data/dino_wm_planning/runtime_20261106_v1/sources/QuantWM/models/visual_world_model.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='931d44c0d541d95be8236f65247de9b40ac3843fe047c1a44d4a2482d1bf636f'
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VWorldModel')
function=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='rollout')
namespace={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(source),'exec'),namespace)
original=namespace['rollout']
class Toy:
    num_hist=3
    def __init__(self):self.predictions=0
    def encode_act(self,act):return act
    def encode(self,obs,act):
        b=act.shape[0];v=obs['visual'].expand(b,-1,-1,-1);p=obs['proprio'].expand(b,-1,-1)
        return torch.cat([v,p.unsqueeze(2).expand(-1,-1,v.shape[2],-1),act.unsqueeze(2).expand(-1,-1,v.shape[2],-1)],dim=3)
    def predict(self,z):
        self.predictions+=1
        return torch.tanh(z*.3+z.mean(dim=1,keepdim=True)*.1+z[..., -10:].mean(dim=-1,keepdim=True)*.07)
    def replace_actions_from_z(self,z,act):
        z[..., -10:]=act.unsqueeze(2).expand(-1,-1,z.shape[2],-1);return z
    def separate_emb(self,z):return {'visual':z[...,:384],'proprio':z[:,:,0,384:394]},z[...,394:]
obs={'visual':torch.linspace(-1,1,196*384).reshape(1,1,196,384),'proprio':torch.linspace(-1,1,10).reshape(1,1,10)}
actions=torch.linspace(-2,2,4*5*10).reshape(4,5,10)
before={k:v.clone() for k,v in obs.items()};counts=[]
for order in [torch.arange(4),torch.tensor([3,0,2,1])]:
    a=actions[order].clone();copy_a=a.clone();direct=Toy();cached=Toy()
    with torch.inference_mode():
        out1,z1=original(direct,obs,a)
        out2,z2=driver.reused_rollout(torch,cached,obs,a)
    assert z1.shape==z2.shape==(4,6,196,404)
    assert torch.equal(z1,z2)
    assert all(torch.equal(out1[k],out2[k]) for k in out1)
    assert direct.predictions==cached.predictions==5
    assert torch.equal(a,copy_a)
    assert all(torch.equal(obs[k],before[k]) for k in obs)
    counts.append((direct.predictions,cached.predictions))
changed=actions.clone();changed[:,-1]*=-1
_,old=original(Toy(),obs,actions);_,new=original(Toy(),obs,changed)
assert not torch.equal(old[:,-1],new[:,-1]),'toy does not detect missed final action'
assert not torch.cuda.is_initialized()
r={'status':'passed_CPU_toy_recurrence_not_real_model','cases':['ordered candidates','permuted candidates','final-action sensitivity negative control'],
   'original_function':'AST-extracted pinned VWorldModel.rollout, no module/model construction',
   'output_torch_equal':True,'source_call_counts':counts,'cached_input_and_actions_unchanged':True,
   'driver_sha256':hashlib.sha256((HERE/'experiment.py').read_bytes()).hexdigest(),
   'selftest_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'runtime_seconds':time.time()-start,'GPU':0,'weights_dataset_pickle_model_loads':0,'network':0}
with (HERE/'SELFTEST.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
