"""Causal replay helpers; import has no GPU/environment or file side effects."""
import hashlib,json,math,os,resource,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
CHUNK,FRAME,HEADS,DIM=4680,1560,12,128
LAYERS=(18,24); READS=(0,3)
ARMS=('R_bf16','R_fp32','O_offline','O_own_refresh','O_first_reader')
QIDS=torch.arange(128)*CHUNK//128
TEXT='A brown dog walks across a shallow stream on smooth stones, pauses to look toward the riverbank, then continues forward. A continuous low tracking shot in soft afternoon light, with reeds moving gently in the background.'
SEED=307

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def tsha(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def save(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def record(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def hadamard():
    h=torch.ones(1,1)
    while h.shape[0]<DIM:h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    return h/math.sqrt(DIM)
def pack(c):
    c=c.contiguous().view(torch.uint8).flatten()
    assert not c.numel()%2
    return (c[::2]&15)|((c[1::2]&15)<<4)
def unpack(p,shape):
    z=torch.empty(p.numel()*2,dtype=torch.uint8);z[::2]=p&15;z[1::2]=p>>4
    z=z.to(torch.int16);z[z>=8]-=16
    return z.to(torch.int8).reshape(shape)
def hessian(q,return_raw=False):
    assert q.shape==(1,CHUNK,HEADS,DIM) and q.device.type=='cpu'
    qr=(q[0].float()@hadamard()).permute(1,0,2).double()
    h=torch.bmm(qr.transpose(1,2),qr)/CHUNK
    damp=.01*h.diagonal(dim1=1,dim2=2).mean(-1)
    hd=h+damp[:,None,None]*torch.eye(DIM).double()[None]
    return (hd,h) if return_raw else hd
def frames(k):
    assert k.shape==(1,CHUNK,HEADS,DIM) and k.dtype==torch.bfloat16 and k.device.type=='cpu'
    z=(k.float()@hadamard()).to(k.dtype)[0].reshape(3,FRAME,HEADS,DIM)
    z=torch.cat((z,z.new_zeros(3,8,HEADS,DIM)),1).reshape(3,98,16,HEADS,DIM)
    s=(z.abs().amax(2,keepdim=True)/7).clamp_min(1e-8)
    sf=s.to(torch.float16);assert torch.equal(sf.to(k.dtype),s)
    return z,s,sf

def factorize(h):
    us=[]
    for hh in h:
        inverse=torch.linalg.inv(hh);inverse=(inverse+inverse.t())*.5
        us.append(torch.linalg.cholesky(inverse,upper=True).float())
    return torch.stack(us)

def codes_obq(xg,sf,h,factors=None):
    x=xg.float().reshape(-1,HEADS,DIM)
    scales=sf.float().expand_as(xg).reshape(-1,HEADS,DIM)
    ans=torch.empty(x.shape,dtype=torch.int8)
    for head in range(HEADS):
        if factors is None:
            inverse=torch.linalg.inv(h[head]);inverse=(inverse+inverse.t())*.5
            u=torch.linalg.cholesky(inverse,upper=True).float()
        else:u=factors[head]
        w=x[:,head].clone()
        for j in range(DIM):
            c=(w[:,j]/scales[:,head,j]).round().clamp(-8,7)
            ans[:,head,j]=c.to(torch.int8)
            if j+1<DIM:
                e=(w[:,j]-c*scales[:,head,j])/u[j,j]
                w[:,j+1:]-=e[:,None]*u[j,j+1:][None]
    return ans.reshape(xg.shape)

def encode_k(k,arm,h=None,factors=None):
    z,s,sf=frames(k)
    if arm=='R_bf16':c=(z/s).round().clamp(-8,7).to(torch.int8)
    elif arm=='R_fp32':c=(z.float()/sf.float()).round().clamp(-8,7).to(torch.int8)
    else:
        assert arm in ARMS[2:] and h is not None
        c=codes_obq(z,sf,h,factors)
    p=pack(c);assert torch.equal(unpack(p,c.shape),c)
    return {'packed':p,'scale':sf,'sentinel':torch.tensor([255],dtype=torch.uint8),'shape':tuple(c.shape),'kind':'K'}
def encode_v(v):
    assert v.shape==(1,CHUNK,HEADS,DIM) and v.dtype==torch.bfloat16
    z=torch.cat((v,v.new_zeros(1,8,HEADS,DIM)),1).reshape(1,293,16,HEADS,DIM)
    s=(z.abs().amax(2,keepdim=True)/7).clamp_min(1e-8)
    c=(z/s).round().clamp(-8,7).to(torch.int8);p=pack(c)
    assert torch.equal(unpack(p,c.shape),c)
    return {'packed':p,'scale':s.to(torch.float16),'shape':tuple(c.shape),'kind':'V'}
def restore(p):
    c=unpack(p['packed'],p['shape']);z=c.bfloat16()*p['scale'].bfloat16()
    if p['kind']=='V':return z.reshape(1,-1,HEADS,DIM)[:,:CHUNK].contiguous()
    z=z.reshape(3,1568,HEADS,DIM)[:,:FRAME].reshape(1,CHUNK,HEADS,DIM)
    return (z.float()@hadamard().t()).bfloat16()
def nbytes(p):return sum(x.numel()*x.element_size() for x in p.values() if isinstance(x,torch.Tensor))
def lock_files():
    return [HERE/x for x in ('REGISTRATION_DRAFT_KO.txt','common.py','capture.py','replay.py','selftest.py')]+[
        ROOT/'src/kvq/lockstep.py',ROOT/'src/kvq/store.py',ROOT/'src/wan/modules/causal_model.py',
        ROOT/'src/scripts/p3_tf_fr.py',ROOT/'src/wan/modules/attention.py']
def check_lock():
    p=HERE/'LOCK.json';l=json.loads(p.read_text())
    assert l['fable_reviewed'] and l['root_accepted']
    for path,h in l['sha256'].items():assert sha(path)==h,path
    return l

def guard(capture=False):
    run=HERE/'RUN.json'
    if run.exists() and time.time()-json.loads(run.read_text())['started_utc']>1800:raise TimeoutError('30min total budget')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>(40 if capture else 8)*1024**3:raise MemoryError('registered RSS budget')
    if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()>22*1024**3:raise MemoryError('22GiB CUDA allocated budget')
    if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())>4*1024**3:raise OSError('4GiB new disk budget')
