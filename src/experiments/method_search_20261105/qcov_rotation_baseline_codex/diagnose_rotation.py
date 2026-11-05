"""CPU development-only q-covariance rotations against frozen OBQ capacity result."""
import argparse
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
sys.dont_write_bytecode = True
import torch

HERE = Path(__file__).resolve().parent
OLD_DIR = HERE.parent / 'obq_capacity_codex'
spec = importlib.util.spec_from_file_location('frozen_obq_capacity', OLD_DIR/'diagnose_obq.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
torch.set_num_threads(4)
torch.set_grad_enabled(False)
ARMS = ('U_FP32_RTN', 'UHP_FP32_RTN')
BEGIN = None


def guard():
    if BEGIN is not None and time.monotonic()-BEGIN > 1200:
        raise TimeoutError('1200-second actual-capture limit')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 8*1024**3:
        raise MemoryError('8-GiB RSS limit')


def rotate(x, r, inverse=False):
    # [B,T,H,D] or [T,H,D], per-head matrix. All runtime matmuls FP32.
    if inverse: r = r.transpose(-1,-2)
    return torch.einsum('...hd,hde->...he',x.float(),r.float())


def had64(d):
    h=torch.ones(1,1,dtype=torch.float64)
    while len(h)<d:
        h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)/math.sqrt(2)
    return h


def make_rotations(q):
    qh=q[0].permute(1,0,2).double()
    c=torch.bmm(qh.transpose(1,2),qh)/qh.shape[1]
    c=(c+c.transpose(1,2))*.5
    ev,u=torch.linalg.eigh(c)
    d=q.shape[-1]
    br=torch.tensor([int(format(i,f'0{int(math.log2(d))}b')[::-1],2) for i in range(d)])
    # Mirrors official make_br_perm_matrix: perm[br[i]]=descending_eigenvalue_index[i].
    perm=torch.empty_like(ev,dtype=torch.long)
    perm[:,br]=torch.argsort(ev,descending=True,dim=-1)
    uh=u@had64(d)
    uhp=torch.gather(uh,2,perm[:,None,:].expand(-1,d,-1))
    r64={'U_FP32_RTN':u,'UHP_FP32_RTN':uhp}
    ortho={name:float((r.transpose(1,2)@r-torch.eye(d)).abs().max()) for name,r in r64.items()}
    assert all(v<=1e-10 for v in ortho.values())
    return {name:r.float() for name,r in r64.items()}, {'eigenvalues':ev.tolist(),
        'covariance_sha256':old.tensor_sha(c),'orthogonality_fp64':ortho,
        'rotation_sha256':{n:old.tensor_sha(r.float()) for n,r in r64.items()},
        'per_head_adaptation':True,'sign_policy':'torch.linalg.eigh returned signs; no canonicalization'},r64


def make_frames(k,r):
    kr=rotate(k,r).to(k.dtype)[0]
    fr=kr.reshape(-1,old.FRAME,old.HEADS,old.DIM)
    pad=torch.cat((fr,fr.new_zeros(len(fr),8,old.HEADS,old.DIM)),1)
    xg=pad.reshape(len(fr),98,old.GROUP,old.HEADS,old.DIM)
    sn=(xg.abs().amax(2,keepdim=True)/7).clamp_min(1e-8)
    ss=sn.to(torch.float16)
    assert torch.equal(sn,ss.to(k.dtype))
    return xg,sn,ss


def packed_restore(code,ss,r):
    packed=old.lib.pack_nibbles(code.contiguous().view(torch.uint8))
    un=old.lib.unpack_nibbles(packed,code.shape).to(torch.int16)
    un[un>=8]-=16
    un=un.to(torch.int8)
    assert torch.equal(un,code)
    rec=(un.to(torch.bfloat16)*ss.to(torch.bfloat16)).reshape(-1,1568,old.HEADS,old.DIM)[:,:old.FRAME]
    rec=rec.reshape(1,-1,old.HEADS,old.DIM)
    restored=rotate(rec,r,True).to(torch.bfloat16)
    sentinels=torch.full((restored.shape[1]//old.CHUNK,),255,dtype=torch.uint8)
    return restored,packed.numel()+ss.numel()*ss.element_size()+sentinels.numel()


def refs():
    return [Path(__file__),HERE/'REGISTRATION_KO.txt',OLD_DIR/'diagnose_obq.py',
        OLD_DIR/'REGISTRATION_KO.txt',OLD_DIR/'REGISTRATION_LOCK.json',OLD_DIR/'RESULT.json',
        old.LIB_PATH,old.ROOT/'src/experiments/kv_headrule_rollout/headrule_quantizer.py',
        old.ROOT/'src/kvq/quant.py']


def selftest():
    assert not torch.cuda.is_available()
    gen=torch.Generator().manual_seed(20261105)
    q=torch.randn((1,256,old.HEADS,old.DIM),generator=gen)
    rs,meta,r64=make_rotations(q)
    x=torch.randn((1,old.CHUNK,old.HEADS,old.DIM),generator=gen).to(torch.bfloat16)
    before=old.tensor_sha(x)
    checks={}
    for name,r in rs.items():
        qa=q[0,:8].double();ka=x[0,:12].double()
        qr=torch.einsum('thd,hde->the',qa,r64[name])
        kr=torch.einsum('thd,hde->the',ka,r64[name])
        a=torch.einsum('thd,shd->hts',qa,ka)
        b=torch.einsum('thd,shd->hts',qr,kr)
        err=float((a-b).abs().max()); assert err<=1e-10
        x32=x.float();back=rotate(rotate(x32,r),r,True)
        torch.testing.assert_close(back,x32,rtol=5e-5,atol=5e-5)
        qrf=rotate(q[0,:8],r);krf=rotate(x[0,:12],r)
        torch.testing.assert_close(torch.einsum('thd,shd->hts',qrf,krf),a.float(),rtol=5e-5,atol=5e-5)
        bfback=rotate(rotate(x,r).to(torch.bfloat16),r,True).to(torch.bfloat16)
        xg,_,ss=make_frames(x,r);code=old.rtn_codes(xg,ss,True)
        rec,kb=packed_restore(code,ss,r)
        assert torch.isfinite(rec).all()
        checks[name]={'fp64_logit_maxabs':err,'fp32_roundtrip_maxabs':float((back-x32).abs().max()),
            'bf16_noquant_relative_sse':old.energy(bfback.float()-x32)/old.energy(x32),'payload_bytes':kb}
    hm=old.hadamard(old.DIM,'cpu');r=hm.repeat(old.HEADS,1,1)
    a,_,sa=make_frames(x,r);b,_,sb=old.make_frames(x,hm)
    assert torch.equal(a,b) and torch.equal(sa,sb)
    code=old.rtn_codes(a,sa,True)
    ra,ba=packed_restore(code,sa,r);rb,bb=old.packed_restore(code,sb,hm)
    assert torch.equal(ra,rb) and ba==bb
    eye=torch.eye(old.DIM).repeat(old.HEADS,1,1)
    a,_,ss=make_frames(x,eye);c=old.rtn_codes(a,ss,True)
    rec,kb=packed_restore(c,ss,eye)
    expect=(c.to(torch.bfloat16)*ss.to(torch.bfloat16)).reshape(-1,1568,old.HEADS,old.DIM)[:,:1560].reshape_as(x)
    assert torch.equal(rec,expect)
    perm=torch.randperm(old.DIM,generator=gen)
    xp=x[...,perm]
    ap,_,sp=make_frames(xp,eye);cp=old.rtn_codes(ap,sp,True)
    rp,_=packed_restore(cp,sp,eye)
    assert torch.equal(cp,c[...,perm]) and torch.equal(sp,ss[...,perm])
    assert torch.equal(rp[...,torch.argsort(perm)],rec)
    assert old.tensor_sha(x)==before
    result={'passed':True,'code_sha256':old.sha(__file__),'torch_version':torch.__version__,
        'checks':checks,'rotation':meta,'hadamard_compatibility':'codes/scales/restore/payload exact',
        'identity_pack_roundtrip':True,'permutation_equivariance_exact':True,
        'input_preserved':True,'cuda_visible':os.environ['CUDA_VISIBLE_DEVICES']}
    old.save(HERE/'SELFTEST.json',result)
    print(json.dumps({key:result[key] for key in ('passed','code_sha256','checks',
        'hadamard_compatibility','identity_pack_roundtrip','permutation_equivariance_exact')},indent=2))


def lock():
    path=HERE/'REGISTRATION_LOCK.json'
    if path.exists():raise RuntimeError('refuse overwrite lock')
    st=json.loads((HERE/'SELFTEST.json').read_text())
    assert st['passed'] and st['code_sha256']==old.sha(__file__)
    old.save(path,{'registered_utc':time.time(),'sha256':{str(p):old.sha(p) for p in refs()},
        'selftest_sha256':old.sha(HERE/'SELFTEST.json'),'arms':ARMS,'layers':old.LAYERS,
        'runtime_seconds_cap':1200,'rss_bytes_cap':8*1024**3,'q_indices':old.QIDS.tolist()})


def near(a,b):
    return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)


def run_layer(layer,prior):
    guard();start=time.monotonic()
    cal,csrc=old.load_capture(0,layer)
    rs,meta,r64=make_rotations(cal['q'])
    del cal,r64;gc.collect()
    cal_s=time.monotonic()-start
    data,src=old.load_capture(4,layer)
    assert csrc==prior['calibration'] and src==prior['evaluation']
    hashes={k:old.tensor_sha(data[k]) for k in ('q','k','v')}
    q=data['q'][0,old.QIDS].float();k=data['k'];v=data['v'];wo=data['wo'].float();bias=data['bias'].float()
    ref=old.attn(q,k[0].float(),v[0].float())
    eref=old.energy(ref.reshape(len(q),-1)@wo.t()+bias)
    assert near(eref,prior['reference_energy'])
    v4,vb=old.lib.fixed_v4(v)
    kref=old.attn(q,k[0].float(),v4)
    hm=old.hadamard(old.DIM,'cpu');hrot=hm.repeat(old.HEADS,1,1)
    xg,_,ss=make_frames(k[:,:old.PAST],hrot);code=old.rtn_codes(xg,ss,True)
    rec,kb=packed_restore(code,ss,hrot)
    anchor=prior['arms']['B0_FP32_RTN']
    assert old.tensor_sha(code)==anchor['code_sha256'] and old.tensor_sha(ss)==anchor['scale_sha256']
    assert kb+vb==anchor['packed_payload_bytes']
    out=old.attn(q,torch.cat((rec,k[:,old.PAST:]),1)[0].float(),v4)
    assert near(old.metric(out,ref,wo,eref)['sse'],anchor['sse'])
    del xg,ss,code,rec,out;gc.collect()
    row={'layer':layer,'calibration':csrc,'evaluation':src,'rotation':meta,'calibration_seconds':cal_s,
        'prior_arms_reused':{a:prior['arms'][a] for a in ('B0_BF16_RTN','B0_FP32_RTN','OBQ_FULL')},
        'reference_energy':eref,'anchor_reproduction_passed':True,'arms':{}}
    for name,r in rs.items():
        guard();t=time.monotonic()
        xg,_,ss=make_frames(k[:,:old.PAST],r)
        rotate_scale_s=time.monotonic()-t;tcode=time.monotonic()
        code=old.rtn_codes(xg,ss,True)
        code_s=time.monotonic()-tcode;td=time.monotonic()
        rec,kb=packed_restore(code,ss,r)
        packed_decode_s=time.monotonic()-td;ta=time.monotonic()
        kfull=torch.cat((rec,k[:,old.PAST:]),1)[0].float()
        out=old.attn(q,kfull,v4)
        attention_s=time.monotonic()-ta
        assert kb+vb==anchor['packed_payload_bytes']
        e=kfull[:old.PAST]-k[0,:old.PAST].float()
        qe=torch.bmm(q.permute(1,0,2),e.permute(1,2,0))/math.sqrt(old.DIM)
        item={**old.metric(out,ref,wo,eref),'k_only_reference':old.metric(out,kref,wo,eref),
            'past_key_sse':old.energy(e),'logit_mse':old.energy(qe)/qe.numel(),
            'packed_payload_bytes':kb+vb,'key_payload_bytes':kb,'value_payload_bytes':vb,
            'rotation_persistent_bytes':r.numel()*r.element_size(),'scale_sha256':old.tensor_sha(ss),
            'code_sha256':old.tensor_sha(code),'rotate_scale_seconds':rotate_scale_s,'code_seconds':code_s,
            'packed_encode_decode_seconds':packed_decode_s,'attention_seconds':attention_s}
        del xg,ss,code,rec,kfull,out,e,qe;gc.collect();guard()
        noquant=rotate(rotate(k[:,:old.PAST],r).to(torch.bfloat16),r,True).to(torch.bfloat16)
        nk=torch.cat((noquant,k[:,old.PAST:]),1)[0].float()
        no=old.attn(q,nk,v4)
        item['noquant_rotation_error_against_k_reference']=old.metric(no,kref,wo,eref)
        nq=item['noquant_rotation_error_against_k_reference']['sse']
        item['noquant_fraction_of_B0_FP32_RTN']=nq/anchor['sse']
        item['noquant_ratio_to_cpu_native_floor']=nq/prior['cpu_vs_native']['sse']
        assert nq<=.01*anchor['sse'],'noquant arithmetic perturbation exceeds registered 1% budget'
        item['total_arm_seconds']=time.monotonic()-t
        row['arms'][name]=item
        print('RESULT',layer,name,json.dumps({key:item[key] for key in ('sse','logit_mse','packed_payload_bytes')}),flush=True)
        del noquant,nk,no;gc.collect()
    assert hashes=={key:old.tensor_sha(data[key]) for key in hashes}
    row['input_preserved']=True;row['seconds']=time.monotonic()-start
    row['obq_gains_vs_rotations']={name:1-prior['arms']['OBQ_FULL']['sse']/row['arms'][name]['sse'] for name in ARMS}
    return row


def run():
    global BEGIN
    dest=HERE/'RESULT.json'
    if dest.exists():raise RuntimeError('refuse overwrite result')
    reg=json.loads((HERE/'REGISTRATION_LOCK.json').read_text())
    assert all(old.sha(p)==h for p,h in reg['sha256'].items())
    assert old.sha(HERE/'SELFTEST.json')==reg['selftest_sha256']
    prior=json.loads((OLD_DIR/'RESULT.json').read_text())
    assert prior['status']=='completed'
    BEGIN=time.monotonic();old.RUN_START=BEGIN
    result={'status':'running','started_utc':time.time(),'torch_version':torch.__version__,
        'registration_sha256':old.sha(HERE/'REGISTRATION_LOCK.json'),'prior_result_sha256':old.sha(OLD_DIR/'RESULT.json'),
        'cpu_threads':torch.get_num_threads(),'cuda_visible':os.environ['CUDA_VISIBLE_DEVICES'],'rows':[]}
    old.save(dest,result)
    try:
        for layer in old.LAYERS:
            prev=next(r for r in prior['rows'] if r['layer']==layer)
            result['rows'].append(run_layer(layer,prev));old.save(dest,result);gc.collect();guard()
        gains=[g for row in result['rows'] for g in row['obq_gains_vs_rotations'].values()]
        worse=any(row['prior_arms_reused']['OBQ_FULL']['sse']>row['arms'][name]['sse'] and
            not near(row['prior_arms_reused']['OBQ_FULL']['sse'],row['arms'][name]['sse'])
            for row in result['rows'] for name in ARMS)
        result.update(status='completed',decision='continue_causal_feasibility' if min(gains)>=.1 else
            ('gap_uncertain_simple_baseline_better' if worse else 'undecided_below_registered_10_percent'))
    except BaseException as e:
        result.update(status='incomplete_or_invalid',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        result.update(finished_utc=time.time(),elapsed_seconds=time.monotonic()-BEGIN,
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,completed_layers=len(result['rows']))
        old.save(dest,result)
    print(json.dumps({k:result[k] for k in ('status','decision','elapsed_seconds','max_rss_bytes')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['selftest','lock','run']);a=p.parse_args()
    {'selftest':selftest,'lock':lock,'run':run}[a.action]()
