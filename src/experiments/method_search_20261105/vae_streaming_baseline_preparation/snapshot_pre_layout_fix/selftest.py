"""CPU synthetic only: no native model construction, tensor files, or CUDA calls."""
import ast,copy,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('streaming_worker_under_test',HERE/'worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(4)
checks=[]
def check(name,condition):
    assert condition,name
    checks.append(name)
def rejects(name,fn):
    try:fn()
    except (AssertionError,ValueError,TypeError):checks.append(name);return
    raise AssertionError('accepted invalid: '+name)
started=time.monotonic()
native=w.native_decode_ast();changed,record=w.transform_decode(native)
check('native AST inverse exact',record['inverse_equals_native'])
mut=copy.deepcopy(native)
loop=[x for x in mut.body if isinstance(x,ast.For)][0]
loop.body[1].orelse[-1]=w.stmt('out = torch.cat([out, out_], 3)')
rejects('nonregistered output concat axis rejected',lambda:w.transform_decode(mut))
mut=copy.deepcopy(native);mut.body[-2]=w.stmt('self.other_clear_cache()')
rejects('changed cache ending rejected',lambda:w.transform_decode(mut))

class FakeConv(torch.nn.Module):
    def __init__(self):super().__init__();self.shapes=[]
    def forward(self,x):self.shapes.append(tuple(x.shape));return x*torch.tensor(.75,dtype=x.dtype)
class FakeDecoder(torch.nn.Module):
    def __init__(self,owner):super().__init__();object.__setattr__(self,'owner',owner);self.shapes=[]
    def forward(self,x,feat_cache,feat_idx):
        self.shapes.append(tuple(x.shape));idx=len(self.shapes)-1
        prior=feat_cache[0]
        y=x[:,:3].clone()
        if prior is not None:y=y+prior[:,:3]*torch.tensor(.125,dtype=x.dtype)
        feat_cache[0]=x.clone();feat_idx[0]+=1
        if idx>0:y=y.repeat_interleave(4,dim=2)
        return y
class FakeModel:
    def __init__(self):
        self.clear_records=[];self._feat_map=[None];self._conv_idx=[0]
        self.conv2=FakeConv();self.decoder=FakeDecoder(self)
    def clear_cache(self):self.clear_records.append(self._feat_map[0] is None);self._feat_map=[None];self._conv_idx=[0]

# Exercise the untouched and transformed real function AST on causal fake tensors.
for length in (1,3,126):
    torch.manual_seed(11)
    z=torch.randn((1,4,length,3,5),dtype=torch.bfloat16)
    original=z.clone();scale=[torch.arange(4,dtype=torch.bfloat16)*.125,torch.ones(4,dtype=torch.bfloat16)*1.25]
    # Native uses self.z_dim in tensor scale branch.
    a=FakeModel();b=FakeModel();a.z_dim=b.z_dim=4
    saved_scale=[x.clone() for x in scale]
    fn=w.compile_decode(torch,native)
    with torch.inference_mode():
        out=fn(a,z,scale);reference=w.normalize_slab(out)
    slabs=[];positions=[]
    def emit(x,index):
        positions.append(index);slabs.append(w.normalize_slab(x).to('cpu'))
    stream=w.compile_decode(torch,changed,emit)
    with torch.inference_mode():ret=stream(b,z,scale)
    cat=torch.cat(slabs,dim=2)
    check(f'{length}: full RGB shape, dtype, logical bytes',w.bytes_equal(reference,cat))
    check(f'{length}: decoder slice order and shapes identical',a.decoder.shapes==b.decoder.shapes==[(1,4,1,3,5)]*length)
    check(f'{length}: full conv2 remains one whole call',a.conv2.shapes==b.conv2.shapes==[(1,4,length,3,5)])
    check(f'{length}: first/final cache reset identical',a.clear_records==b.clear_records==[True,False])
    check(f'{length}: emitter count/index and return None',positions==list(range(length)) and ret is None)
    check(f'{length}: input preserved',w.bytes_equal(z,original) and all(w.bytes_equal(x,y) for x,y in zip(scale,saved_scale)))
    check(f'{length}: state cleared after decode',a._feat_map==b._feat_map==[None])
    check(f'{length}: synthetic no autograd outputs',not cat.requires_grad and cat.grad_fn is None)

# CPU toy only checks pointwise formula/coordinates, not CUDA arithmetic identity.
x=torch.tensor([[-2.,-0.,0.,.25,2.]],dtype=torch.bfloat16).reshape(1,1,1,1,5).expand(1,3,1,1,5).clone()
orig=x.clone();got=w.normalize_slab(x)
expected=x.float().clamp(-1,1);expected=expected*.5;expected=expected+.5;expected=expected.clamp(0,1)
check('normalization formula and input preservation',w.bytes_equal(got,expected) and w.bytes_equal(x,orig))
neg=torch.tensor([-0.],dtype=torch.float32);pos=torch.tensor([0.],dtype=torch.float32)
check('byte comparison distinguishes signed zero',not w.bytes_equal(neg,pos))
rejects('normalization rejects incorrect channel count',lambda:w.normalize_slab(torch.ones((1,2,1,1,1),dtype=torch.bfloat16)))
rejects('normalization rejects incorrect dtype',lambda:w.normalize_slab(torch.ones((1,3,1,1,1),dtype=torch.float32)))
check('empty byte view supported',w.flat_bytes(torch.empty(0)).numel()==0)
y=torch.arange(128,dtype=torch.bfloat16)[::128].reshape(1,1)
check('singleton strided byte view supported',w.flat_bytes(y).numel()==2)
class Cache:pass
c=Cache();base=torch.ones((1,4,2,3,5),dtype=torch.bfloat16);c._feat_map=[base,base[:,:,:1],None,'Rep']
r=w.cache_summary(c)
check('cache unique storage excludes aliased double count',r['tensor_count']==2 and r['unique_storage_count']==1 and r['unique_storage_bytes']==base.untyped_storage().nbytes())
check('cache logical count independently reported',r['logical_bytes']==base.numel()*2+base[:,:,:1].numel()*2)
check('cache string and unused slots recorded',r['non_tensor_slots']==[{'slot':2,'value':None},{'slot':3,'value':'Rep'}])
cls,prepare,cls_sha=w.exact_wrapper(torch)
class WrapperInput:pass
wi=WrapperInput();wi.mean=torch.arange(16,dtype=torch.float32)/10;wi.std=torch.arange(16,dtype=torch.float32)/10+1
zz=torch.randn((1,3,16,2,4),dtype=torch.bfloat16)
zs,ss=prepare(wi,zz)
check('native wrapper input assignments shape and values',w.bytes_equal(zs,zz.permute(0,2,1,3,4)))
check('native wrapper scale dtype and BF16 reciprocal',w.bytes_equal(ss[0],wi.mean.to(zz.dtype)) and w.bytes_equal(ss[1],1.0/wi.std.to(zz.dtype)))
check('native wrapper class extracted without model construction',cls.__name__=='WanVAEWrapper')
check('no CUDA initialized',not torch.cuda.is_initialized())
result={'status':'passed_CPU_synthetic_only','checks':checks,'passed':len(checks),'seconds':time.monotonic()-started,
        'source_sha256':w.sha(HERE/'worker.py'),'selftest_sha256':w.sha(__file__),'AST_contract':record,
        'torch':torch.__version__,'python':sys.version,'GPU_calls':0,'model_loads':0,'original_tensor_loads':0,
        'limitations':'Toy decoder and CPU pointwise arithmetic; no actual VAE, CUDA kernels, output equality, memory or performance verified.'}
out=HERE/'SELFTEST.json';assert not out.exists();w.save(out,result)
print(json.dumps({'status':result['status'],'passed':len(checks),'seconds':result['seconds'],'sha256':w.sha(out)}))
