import os, json, importlib, time, resource, pathlib
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
r={"start_epoch":time.time(),"modules":{},"checkpoint_symbols":{},"tensor_or_pickle_loads":0,"installs":0,"network_requested":0}
for name in ["torch","torchvision","numpy","einops","yaml","accelerate","accelerate.optimizer","accelerate.state","accelerate.utils.dataclasses"]:
 try:
  m=importlib.import_module(name);r["modules"][name]={"status":"imported","file":getattr(m,"__file__",None),"version":getattr(m,"__version__",None)}
 except Exception as e:r["modules"][name]={"status":"failed","type":type(e).__name__,"message":str(e)}
for module,names in {"accelerate.optimizer":["AcceleratedOptimizer"],"accelerate.state":["AcceleratorState","GradientState"],"accelerate.utils.dataclasses":["DistributedType","DynamoBackend","TorchDynamoPlugin"]}.items():
 try:
  m=importlib.import_module(module);r["checkpoint_symbols"][module]={n:hasattr(m,n) for n in names}
 except Exception as e:r["checkpoint_symbols"][module]={"error":str(e)}
import torch
r["cuda_initialized"]=torch.cuda.is_initialized();r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;r["finish_epoch"]=time.time();r["scope"]="installed import/class existence only; no checkpoint restoration or constructor/model forward; compatibility still unproven"
pathlib.Path(__file__).with_name("ROOT_CPU_IMPORT_RESULT.json").write_text(json.dumps(r,indent=2)+"\n")
print(json.dumps(r))
assert not r["cuda_initialized"]
