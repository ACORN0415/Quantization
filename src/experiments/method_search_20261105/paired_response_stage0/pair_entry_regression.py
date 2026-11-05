"""Execute evaluate_layer's actual entry control path with fake DataAccess.

No capture / model / GPU / real effect. Stops at the first synthetic write load.
The old invalid V2 is executed without patching its function body; only __file__
is set to the original live location so frozen relative helper bindings resolve.
"""
import hashlib,json,sys,time,types
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
OLD=HERE/'snapshots/before_amendment_v3/diagnose.py'
NEW=HERE/'diagnose.py'
def load_source(path,name):
    mod=types.ModuleType(name);mod.__file__=str(NEW);sys.modules[name]=mod
    exec(compile(path.read_text(),str(path),'exec'),mod.__dict__)
    return mod
new=load_source(NEW,'pair_entry_active');old=load_source(OLD,'pair_entry_preserved_V2');new.setup_cpu()
# Both sources are bound to the same actual frozen helpers. No function branch,
# make_paired_manifest implementation or evaluate_layer body is monkeypatched.
old._MODULES=new.modules()
class FirstWriteReached(Exception):pass
class FakeBudget:
    def __init__(self):self.calls=0
    def guard(self):self.calls+=1
class FakeAccess:
    def __init__(self):self.budget=FakeBudget();self.loads=[]
    def load(self,descriptor):
        tag=descriptor['synthetic'];self.loads.append(tag)
        if tag=='prepared':return {}
        if tag=='Wo':return {'wo':torch.zeros(1536,1536,dtype=torch.float32),'bias':torch.zeros(1536,dtype=torch.float32)}
        if tag=='first_write':raise FirstWriteReached('stop before any K/V data or score')
        raise AssertionError('unexpected fixture I/O')
rows=[];start=time.monotonic()
for name,module in [('preserved_invalid_V2',old),('active_V3',new)]:
 for arm in ('PAIR17','PAIR29','PAIR43'):
    a=FakeAccess();binding={'layer':18,'Wo':{'synthetic':'Wo'},'packets':[{'chunk':1}],'writes':[{'synthetic':'first_write'}]};prepared={'state':{'synthetic':'prepared'}}
    error=None
    try:module.evaluate_layer(binding,arm,a,prepared)
    except Exception as e:error=e
    if name=='preserved_invalid_V2':
        assert isinstance(error,UnboundLocalError) and 'reloaded' in str(error),(name,arm,repr(error))
        assert a.loads==['prepared','Wo'] and a.budget.calls==0
    else:
        assert isinstance(error,FirstWriteReached),(name,arm,repr(error))
        assert a.loads==['prepared','Wo','first_write'] and a.budget.calls==1
    rows.append({'source':name,'arm':arm,'exception_type':type(error).__name__,'exception_text':str(error),'synthetic_load_sequence':a.loads,'budget_guards':a.budget.calls,'expected_observation':True})
assert not torch.cuda.is_initialized()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'status':'passed','cases':6,'observations':rows,'source_sha256':{str(OLD):sha(OLD),str(NEW):sha(NEW),str(Path(__file__)):sha(Path(__file__))},'scope':'actual evaluate_layer entry path only, no completed effect loop or full real-data runtime validation','synthetic_source_locator':'__file__ is live original location for preserved source relative helper paths','original_tensor_file_reads':0,'new_GPU_calls':0,'actual_effect_measurements':0,'elapsed_seconds':time.monotonic()-start}
with (HERE/'PAIR_ENTRY_REGRESSION_V1.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'status':'passed','cases':6,'old_UnboundLocalError':3,'new_first_write_sentinel':3,'elapsed_seconds':report['elapsed_seconds']}))
