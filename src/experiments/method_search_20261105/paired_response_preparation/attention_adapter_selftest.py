"""Synthetic checks only; no old helper import, capture, model or CUDA execution."""
import argparse,hashlib,json,math,os,resource,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import torch
import attention_adapter as a


def bits(x):return x.detach().contiguous().view(torch.uint8).numpy().tobytes()

def run():
    start=time.monotonic();torch.set_num_threads(4);torch.set_grad_enabled(False);checks=[]
    def check(name,ok,**detail):
        checks.append({'name':name,'passed':bool(ok),**detail})
        if not ok:raise AssertionError(name)
    def rejects(name,fn):
        try:fn()
        except ValueError:check(name,True)
        else:check(name,False)
    g=torch.Generator().manual_seed(719)
    q=torch.randn(70,3,8,generator=g);pk=torch.randn(11,3,8,generator=g);pv=torch.randn(11,3,8,generator=g)
    ck=torch.randn(5,3,8,generator=g);cv=torch.randn(5,3,8,generator=g)
    inputs=(q,pk,pv,ck,cv);before=[bits(x) for x in inputs]
    out=a.attention_window(*inputs);key=torch.cat((pk,ck));val=torch.cat((pv,cv))
    # Local source-equivalent old arithmetic, deliberately no legacy imports/guard.
    kt=key.permute(1,2,0).contiguous();vt=val.permute(1,0,2).contiguous();old=torch.empty_like(q);ls=torch.empty(3,70)
    for begin in range(0,len(q),32):
        logits=torch.bmm(q[begin:begin+32].permute(1,0,2),kt)/math.sqrt(8)
        old[begin:begin+32]=torch.bmm(logits.softmax(-1),vt).permute(1,0,2)
        ls[:,begin:begin+32]=logits.logsumexp(-1)
    check('qblock32_old_arithmetic_bytes',bits(old)==bits(out.original_output))
    check('FP32_same_logit_LSE_bytes',bits(ls)==bits(out.logz) and out.logz.dtype==torch.float32)
    check('paired_axis_roundtrip_bytes',out.paired_output.shape==(3,70,8) and bits(a.paired_to_original(out.paired_output))==bits(old))
    check('past_plus_current_counts',out.past_tokens==11 and out.current_tokens==5)
    check('attention_inputs_preserved',before==[bits(x) for x in inputs])
    # Independent full-matrix FP64 arithmetic checks numerical scale/token coverage.
    score=torch.einsum('qhd,khd->hqk',q.double(),key.double())/math.sqrt(8)
    expected=torch.einsum('hqk,khd->qhd',score.softmax(-1),val.double())
    check('independent_FP64_attention',torch.allclose(out.original_output.double(),expected,atol=1e-6,rtol=2e-6),max_absolute_error=float((out.original_output.double()-expected).abs().max()))
    check('independent_FP64_LSE',torch.allclose(out.logz.double(),score.logsumexp(-1),atol=1e-6,rtol=2e-6),max_absolute_error=float((out.logz.double()-score.logsumexp(-1)).abs().max()))
    zq=torch.zeros(4,2,3);zp=torch.zeros(2,2,3);zc=torch.zeros(3,2,3)
    pvalue=torch.ones(2,2,2)*2;cvalue=torch.ones(3,2,2)*7
    full=a.attention_window(zq,zp,pvalue,zc,cvalue)
    check('all_past_current_value_scope',torch.allclose(full.original_output,torch.full((4,2,2),5.)))
    check('all_past_current_LSE_scope',torch.equal(full.logz,torch.full((2,4),math.log(5),dtype=torch.float32)))
    only=a.attention_window(zq,zp[:0],pvalue[:0],zc,cvalue)
    check('empty_past_current_still_included',torch.allclose(only.original_output,torch.full((4,2,2),7.)) and torch.equal(only.logz,torch.full((2,4),math.log(3),dtype=torch.float32)))
    # Head cross terms must survive flattening and W_o multiplication.
    cand=torch.ones(1,2,1);ref=torch.zeros_like(cand);weight=torch.tensor([[1.,-1.]]);bias=torch.tensor([3.])
    metric=a.post_wo_difference(cand,ref,weight,bias)
    check('Wo_head_cross_terms_preserved',metric['sse']==0. and float((cand-ref).square().sum())==2. and metric['reference_energy']==9.)
    cand=torch.tensor([[[1.,2.],[3.,4.]],[[2.,1.],[0.,3.]]]);ref=cand*.5
    w=torch.tensor([[1.,0.,2.,-1.],[0.,1.,1.,0.]])
    b=torch.tensor([4.,-2.]);b2=torch.tensor([1e8,-1e8]);protected=(cand,ref,w,b,b2);initial=[bits(x) for x in protected]
    m=a.post_wo_difference(cand,ref,w,b);m2=a.post_wo_difference(cand,ref,w,b2)
    direct=(cand-ref).reshape(2,-1)@w.t()
    check('post_Wo_original_difference_bytes',bits(m['delta'])==bits(direct) and m['sse']==float(direct.double().square().sum()))
    check('bias_cancels_only_from_difference',bits(m['delta'])==bits(m2['delta']) and m['sse']==m2['sse'] and m['reference_energy']!=m2['reference_energy'])
    check('reference_energy_includes_bias',m['reference_energy']==float((ref.reshape(2,-1)@w.t()+b).double().square().sum()))
    check('post_Wo_inputs_preserved',initial==[bits(x) for x in protected])
    zero=a.post_wo_difference(torch.zeros(1,1,1),torch.zeros(1,1,1),torch.ones(1,1),torch.zeros(1))
    check('zero_reference_energy_explicit',zero['reference_energy']==0 and zero['nse'] is None)
    with torch.enable_grad():
        grad_inputs=tuple(x.detach().clone().requires_grad_() for x in inputs);initial=[bits(x) for x in grad_inputs]
        gout=a.attention_window(*grad_inputs)
        cm=cand.detach().clone().requires_grad_();rm=ref.detach().clone().requires_grad_();wm=w.detach().clone().requires_grad_();bm=b.detach().clone().requires_grad_()
        gm=a.post_wo_difference(cm,rm,wm,bm)
        check('no_grad_output_contract',all(not x.requires_grad and x.grad_fn is None for x in (gout.original_output,gout.paired_output,gout.logz,gm['delta'],a.paired_to_original(gout.paired_output))))
        check('grad_inputs_and_mode_preserved',torch.is_grad_enabled() and initial==[bits(x) for x in grad_inputs] and all(x.requires_grad and x.grad is None for x in grad_inputs))
    rejects('bad_query_rank',lambda:a.attention_window(q[None],pk,pv,ck,cv))
    rejects('BF16_conversion_must_be_explicit',lambda:a.attention_window(q.bfloat16(),pk,pv,ck,cv))
    rejects('key_head_mismatch',lambda:a.attention_window(q,pk[:,:2],pv,ck,cv))
    rejects('value_token_mismatch',lambda:a.attention_window(q,pk,pv[:-1],ck,cv))
    rejects('empty_current_rejected',lambda:a.attention_window(q,pk,pv,ck[:0],cv[:0]))
    rejects('invalid_Wo_shape',lambda:a.post_wo_difference(cand,ref,w[:,:-1],b))
    rejects('invalid_bias_shape',lambda:a.post_wo_difference(cand,ref,w,b[:1]))
    bad=q.clone();bad[0,0,0]=float('nan')
    rejects('nonfinite_query_rejected',lambda:a.attention_window(bad,pk,pv,ck,cv))
    check('CUDA_uninitialized',not torch.cuda.is_initialized())
    return {'status':'passed','synthetic_only':True,'checks_total':len(checks),'checks':checks,'real_capture_reads':0,'GPU_calls':0,'model_loads':0,'effect_experiments':0,'threads':torch.get_num_threads(),'cuda_initialized':torch.cuda.is_initialized(),'wall_seconds':time.monotonic()-start,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args();here=Path(__file__).resolve().parent
    result=run();result['source_hashes']={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in ('attention_adapter.py','attention_adapter_selftest.py')}
    out=Path(args.output).resolve()
    if out.parent!=here:raise ValueError('output outside preparation folder')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'checks_total':result['checks_total'],'wall_seconds':result['wall_seconds'],'output':str(out)}))
