"""Small synthetic state tests plus one production-shaped synthetic chunk."""
import argparse,hashlib,json,os,resource,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import torch
import mean_baseline as m


def run(output):
    begin=time.monotonic();torch.set_num_threads(4)
    here=Path(__file__).resolve().parent;checks=[]
    def check(name,ok,**details):
        checks.append({'name':name,'passed':bool(ok),**details})
        if not ok:raise AssertionError(name)
    def by(x):return m.tensor_bytes(x.detach())
    def rejects(name,fn):
        try:fn()
        except ValueError:check(name,True)
        else:check(name,False)
    def packet_equal(a,b):
        return a['shape']==b['shape'] and a['kind']==b['kind'] and all(by(a[k])==by(b[k]) for k in ('packed','scale','sentinel'))
    gen=torch.Generator().manual_seed(1903)
    k=torch.randn(1,4680,12,128,generator=gen).bfloat16()
    kb=hashlib.sha256(by(k)).hexdigest()
    zero=m.prepare_mean(torch.zeros(12,128))
    plain=m.c.encode_k(k,'R_fp32');zp=m.encode(k,zero)
    check('zero_mean_original_packet_bytes',packet_equal(plain,zp))
    check('zero_mean_original_restore_bytes',by(m.restore(zp,zero))==by(m.c.restore(plain)))
    zminus=m.prepare_mean(torch.full((12,128),-0.))
    check('negative_zero_state_original_bytes',packet_equal(m.encode(k,zminus),plain) and by(m.restore(plain,zminus))==by(m.c.restore(plain)))
    mu=torch.randn(12,128,generator=gen)*.17
    state=m.prepare_mean(mu);mb=by(mu)
    p=m.encode(k,state);r=m.restore(p,state)
    centered=(k.float()-mu[None,None]).bfloat16()
    direct=m.c.encode_k(centered,'R_fp32')
    expected=(m.c.restore(direct).float()+mu[None,None]).bfloat16()
    check('nonzero_mean_explicit_packet',packet_equal(p,direct))
    check('nonzero_mean_explicit_restore',by(r)==by(expected))
    check('input_key_mean_preserved',kb==hashlib.sha256(by(k)).hexdigest() and by(mu)==mb)
    check('output_shape_dtype',r.shape==k.shape and r.dtype==torch.bfloat16)
    check('shared_mu_owned',state.mean.data_ptr()!=mu.data_ptr() and state.storage_bytes()==state.nbytes()==6144)
    mu.add_(4)
    check('external_mu_mutation_does_not_change_state',by(state.mean)==mb)
    reloaded=m.MeanState.from_bytes(state.to_bytes())
    check('shared_state_bytes_roundtrip',reloaded.to_bytes()==state.to_bytes() and reloaded.mean.data_ptr()!=state.mean.data_ptr())
    check('reloaded_state_restore_exact',by(m.restore(p,reloaded))==by(r))
    # Fully preserve both state and the ordinary packet across actual disk save/reload.
    file=here/'MEAN_SYNTHETIC_STATE.pt'
    if file.exists():raise FileExistsError(file)
    torch.save({'mean':state.mean,'packet':p},file)
    saved=torch.load(file,map_location='cpu',weights_only=True)
    sr=m.prepare_mean(saved['mean']);sp=saved['packet']
    check('disk_state_packet_roundtrip',sr.to_bytes()==state.to_bytes() and packet_equal(sp,p) and by(m.restore(sp,sr))==by(r))
    check('disk_payload_under20MiB',file.stat().st_size<20*1024**2,bytes=file.stat().st_size)
    check('packet_size_unchanged',m.c.nbytes(p)==m.c.nbytes(plain)==4515841)
    check('six_chunk_extra_only_shared_mu',6*m.c.nbytes(p)+state.nbytes()-6*m.c.nbytes(plain)==6144)
    # Token weighted offline mean, unequal chunk lengths; no production data.
    a=torch.randn(3,12,128,generator=gen).bfloat16();b=torch.randn(1,7,12,128,generator=gen).bfloat16()
    ab,bb=by(a),by(b)
    mean=m.make_mean([a,b])
    expected_mu=torch.cat((a,b[0])).double().mean(0).float()
    check('offline_mean_token_weighted',by(mean.mean)==by(expected_mu))
    check('offline_mean_input_formats',by(m.make_mean(torch.cat((a,b[0]))[None]).mean)==by(expected_mu))
    check('offline_mean_inputs_preserved',by(a)==ab and by(b)==bb)
    with torch.enable_grad():
        kg=k.clone().requires_grad_();mg=state.mean.clone().requires_grad_()
        s=m.prepare_mean(mg);q=m.encode(kg,s);y=m.restore(q,s);fit=m.make_mean(a.clone().requires_grad_())
        tensors=(s.mean,fit.mean,q['packed'],q['scale'],q['sentinel'],y)
        check('no_retained_autograd_graph',all(not x.requires_grad and x.grad_fn is None for x in tensors))
        check('caller_grad_flags_preserved',torch.is_grad_enabled() and kg.requires_grad and mg.requires_grad and kg.grad is None and mg.grad is None)
    rejects('zero_K_scale_rejected',lambda:m.encode(torch.zeros_like(k),zero))
    rejects('tiny_scale_rejected',lambda:m.encode((k.float()*1e-10).bfloat16(),zero))
    rejects('overflow_scale_rejected',lambda:m.encode((k.float()*1e8).bfloat16(),zero))
    rejects('nonfinite_key_rejected',lambda:m.encode(torch.full_like(k,float('inf')),zero))
    for label,value in [('negative',-1.),('zero',0.),('nonfinite',float('nan'))]:
        bad={key:(val.clone() if isinstance(val,torch.Tensor) else val) for key,val in p.items()}
        bad['scale'].flatten()[0]=value
        rejects(label+'_packet_scale_rejected',lambda bad=bad:m.restore(bad,state))
    rejects('nonfinite_mu_rejected',lambda:m.prepare_mean(torch.full((12,128),float('nan'))))
    rejects('wrong_mu_dtype_rejected',lambda:m.prepare_mean(torch.zeros(12,128,dtype=torch.float64)))
    rejects('empty_mean_input_rejected',lambda:m.make_mean([]))
    rejects('truncated_state_rejected',lambda:m.MeanState.from_bytes(state.to_bytes()[:-1]))
    # Nonzero centering roundtrip is intentionally not an exact-identity promise.
    # A deterministic direct two-stage BF16 example, without an encoder.
    value2=torch.tensor([.001953125],dtype=torch.bfloat16);shift2=torch.tensor([1.],dtype=torch.float32)
    changed=((value2.float()-shift2).bfloat16().float()+shift2).bfloat16()
    check('nonzero_centering_not_identity_claim',not torch.equal(changed,value2),before=float(value2),after=float(changed))
    check('CPU_only',not torch.cuda.is_initialized())
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    check('RSS_under2GiB',rss<2*1024**3,bytes=rss)
    result={'status':'passed','synthetic_only':True,'checks_total':len(checks),'checks':checks,'real_capture_reads':0,'GPU_calls':0,'model_loads':0,'threads':torch.get_num_threads(),'wall_seconds':time.monotonic()-begin,'max_rss_bytes':rss,'persistent_extra_bytes':6144,'shared_hadamard_excluded_bytes':65536,'artifact':{'path':str(file),'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()},'source_hashes':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (here/'mean_baseline.py',Path(__file__),m.COMMON_PATH)}}
    with Path(output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'checks':len(checks),'wall_seconds':result['wall_seconds'],'rss':rss,'persistent_extra_bytes':6144}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    if Path(args.output).resolve().parent!=Path(__file__).resolve().parent:raise ValueError('output directory')
    run(args.output)
