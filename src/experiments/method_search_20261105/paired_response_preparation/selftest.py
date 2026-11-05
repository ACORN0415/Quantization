"""Synthetic only; no imports of capture/model/GPU paths."""
import argparse, hashlib, json, math, os, platform, resource, time
from pathlib import Path
import torch
from core import PairedResponse, ChunkState, PreparationError

HERE = Path(__file__).resolve().parent

def digest(x):
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def filehash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run():
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    checks = []
    def check(name, condition, **details):
        checks.append({'name': name, 'passed': bool(condition), **details})
        if not condition:
            raise AssertionError(name)
    gen = torch.Generator().manual_seed(113)
    rnd = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float64)
    h, dim, dv, r, nq = 2, 5, 3, 13, 7
    model = PairedResponse.gaussian(h, dim, r, 211)
    q = rnd(h,nq,dim)*.2
    current_k, current_v = rnd(h,4,dim)*.3, rnd(h,4,dv)*.4
    chunks = []
    for n in (11,9,13):
        k = rnd(h,n,dim)*.3
        kh = k + rnd(h,n,dim)*.05
        v = rnd(h,n,dv)*.4
        chunks.append((k,kh,v))
    protected = [model.omega,q,current_k,current_v]+[x for row in chunks for x in row]
    before = [digest(x) for x in protected]

    def baseline(rows):
        k = torch.cat([z[1] for z in rows]+[current_k],1)
        v = torch.cat([z[2] for z in rows]+[current_v],1)
        score = q @ k.transpose(-1,-2) / math.sqrt(dim)
        return score.softmax(-1) @ v, score.logsumexp(-1)

    def independent_feature(x):
        # Independent formula/layout, no summarize/_logs or offsets reused.
        scaled = x / (dim**.25)
        lin = torch.einsum('hnd,hdr->hnr', scaled, model.omega.double())
        return torch.exp(lin - torch.einsum('hnd,hnd->hn',scaled,scaled)[...,None]/2) / math.sqrt(r)

    def direct(rows):
        base, logz = baseline(rows)
        phi = independent_feature(q)
        denominator_delta = torch.zeros_like(logz)
        numerator_delta = torch.zeros_like(base)
        for k,kh,v in rows:
            # Two complete kernels formed independently, then differenced.
            difference = (phi @ independent_feature(k).transpose(-1,-2)) - (phi @ independent_feature(kh).transpose(-1,-2))
            denominator_delta += difference.sum(-1)
            numerator_delta += difference @ v
        return (base + numerator_delta/logz.exp()[...,None])/(1+denominator_delta/logz.exp())[...,None]

    errors=[]; diagnostics=[]
    for idx,row in enumerate(chunks):
        state, diag = model.summarize(str(idx),*row)
        check(f'persistent_dtypes_{idx}',state.dz.dtype==torch.float32 and state.dn.dtype==torch.float32 and state.offset.dtype==torch.float64)
        model.append(state);diagnostics.append(diag)
        base,lz=baseline(chunks[:idx+1])
        reader_inputs=[base,lz]+[x for s in model.states.values() for x in (s.dz,s.dn,s.offset)]
        reader_before=[digest(x) for x in reader_inputs]
        out=model.read(q,base,lz,min_denominator_ratio=.1)
        check(f'reader_inputs_and_states_preserved_{idx}',reader_before==[digest(x) for x in reader_inputs])
        expected=direct(chunks[:idx+1]);err=float((out.output-expected).abs().max());errors.append(err)
        check(f'append_direct_kernel_{idx}',torch.allclose(out.output,expected,atol=2e-7,rtol=2e-6) and not out.fallback.any(),max_absolute_difference=err)
    model.evict('0')
    base,lz=baseline(chunks[1:]);out=model.read(q,base,lz,min_denominator_ratio=.1)
    err=float((out.output-direct(chunks[1:])).abs().max());errors.append(err)
    check('evict_direct_kernel',torch.allclose(out.output,direct(chunks[1:]),atol=2e-7,rtol=2e-6) and not out.fallback.any(),max_absolute_difference=err)
    independent_rebuild=PairedResponse(model.omega)
    for idx,row in enumerate(chunks[1:],1):
        st,_=independent_rebuild.summarize(str(idx),*row);independent_rebuild.append(st)
    check('evict_equals_retained_rebuild_bits',digest(out.output)==digest(independent_rebuild.read(q,base,lz,min_denominator_ratio=.1).output))
    check('inputs_and_map_unchanged',before==[digest(x) for x in protected])
    check('shared_map_single_persistent_tensor',[(key,tuple(v.shape)) for key,v in vars(model).items() if isinstance(v,torch.Tensor)]==[('omega',(h,dim,r))])
    grad_inputs=[x.clone().requires_grad_() for x in chunks[0]]
    no_graph,_=model.summarize('no_graph',*grad_inputs)
    check('stored_state_does_not_retain_autograd_inputs',all(x.grad_fn is None and not x.requires_grad for x in (no_graph.dz,no_graph.dn,no_graph.offset)))

    nochange=PairedResponse(model.omega)
    st,diag=nochange.summarize('same',chunks[0][0],chunks[0][0],chunks[0][2]);nochange.append(st)
    check('no_intervention_exact_zero_state',torch.count_nonzero(st.dz)==0 and torch.count_nonzero(st.dn)==0)
    b=base.to(torch.bfloat16);zout=nochange.read(q,b,lz,min_denominator_ratio=.1)
    check('no_intervention_base_bytes',digest(b)==digest(zout.output) and zout.diagnostics['exact_zero_bypass'] and not zout.fallback.any())
    nochange.evict('same');empty=nochange.read(q,b,lz,min_denominator_ratio=.1)
    check('empty_state_base_bytes',digest(b)==digest(empty.output) and empty.diagnostics['exact_zero_bypass'])
    duplicate=False
    try:model.append(next(iter(model.states.values())))
    except ValueError:duplicate=True
    check('duplicate_append_rejected',duplicate)

    # Synthetic budget allocation: no model/capture tensor is read.
    sized=PairedResponse.gaussian(12,128,32,17)
    for i in range(6):
        sized.append(ChunkState(str(i),torch.zeros(12,32),torch.zeros(12,32,128),torch.zeros(12,32,dtype=torch.float64)))
    tensors=[sized.omega]+[x for s in sized.states.values() for x in (s.dz,s.dn,s.offset)]
    raw_allocated=sum(x.untyped_storage().nbytes() for x in tensors)
    check('actual_tensor_payload_budget',sized.tensor_payload_bytes()==1403904 and raw_allocated==1403904 and len({x.untyped_storage().data_ptr() for x in tensors})==len(tensors),logical_tensor_bytes=sized.tensor_payload_bytes(),unique_storage_bytes=raw_allocated,allocator_metadata_included=False)

    small=PairedResponse(torch.zeros(1,1,1))
    kval=torch.tensor([[[0.],[math.sqrt(230.)]]],dtype=torch.float64)
    kh=torch.tensor([[[0.],[math.sqrt(232.)]]],dtype=torch.float64)
    _,dcast=small.summarize('cast',kval,kh,torch.ones(1,2,1,dtype=torch.float64))
    check('FP32_cast_underflow_recorded',dcast['state_cast_underflow']>0,diagnostics=dcast)
    _,dexp=small.summarize('underflow',torch.tensor([[[0.],[100.]]],dtype=torch.float64),torch.tensor([[[0.],[101.]]],dtype=torch.float64),torch.ones(1,2,1,dtype=torch.float64))
    check('feature_exp_underflow_recorded',dexp['feature_exp_underflow']>0,diagnostics=dexp)
    # Near-zero exponent: exp(-5e-21) rounds to 1, expm1 keeps the difference.
    close,dcancel=small.summarize('nearby',torch.zeros(1,1,1,dtype=torch.float64),torch.full((1,1,1),1e-10,dtype=torch.float64),torch.ones(1,1,1,dtype=torch.float64))
    check('nearby_exponent_cancellation_stabilized',dcancel['naive_exp_subtraction_collision']==1 and close.dz.item()>0 and abs(close.dz.item()/5e-21-1)<1e-6,diagnostics=dcancel)
    preparation_rejections=[]
    for name,k,kh,v in (
        ('nonfinite_input',torch.tensor([[[float('nan')]]]),torch.zeros(1,1,1),torch.ones(1,1,1)),
        ('feature_log_overflow',torch.tensor([[[1e308]]],dtype=torch.float64),torch.zeros(1,1,1),torch.ones(1,1,1)),
        ('FP32_state_overflow',torch.ones(1,1,1),torch.zeros(1,1,1),torch.full((1,1,1),1e39,dtype=torch.float64))):
        try:small.summarize(name,k,kh,v)
        except PreparationError as e:preparation_rejections.append({'name':name,'reason':e.reason,'diagnostics':e.diagnostics})
    check('invalid_preparation_rejected',len(preparation_rejections)==3,cases=preparation_rejections)

    native=torch.tensor([[[.25]]],dtype=torch.float64);q1=torch.zeros(1,1,1,dtype=torch.float64);lz1=torch.zeros(1,1,dtype=torch.float64)
    def artificial(dz,dn,offset=0.,omega=0.):
        a=PairedResponse(torch.tensor([[[omega]]],dtype=torch.float32))
        a.append(ChunkState('x',torch.tensor([[dz]],dtype=torch.float32),torch.tensor([[[dn]]],dtype=torch.float32),torch.tensor([[offset]],dtype=torch.float64)))
        return a
    for name,dz in (('negative_denominator',-2.),('small_positive_denominator',-.99)):
        a=artificial(dz,1.);res=a.read(q1,native,lz1,min_denominator_ratio=.1)
        check(name,res.fallback.all() and digest(res.output)==digest(native) and res.diagnostics['unsafe_denominator_queries']==1,diagnostics=res.diagnostics)
    a=artificial(1.,1.,omega=40.)
    over=a.read(torch.tensor([[[28.]]]),native,lz1,min_denominator_ratio=.1)
    check('large_exp_stabilized_without_raw_overflow',not over.fallback.any() and torch.isfinite(over.output).all() and over.diagnostics['potential_unscaled_exp_overflow']==1 and torch.allclose(over.output,torch.ones_like(native)),diagnostics=over.diagnostics)
    lost=a.read(torch.tensor([[[40.]]]),native,lz1,min_denominator_ratio=.1)
    check('lost_native_base_fallback',lost.fallback.all() and digest(lost.output)==digest(native) and lost.diagnostics['base_exp_underflow']==1,diagnostics=lost.diagnostics)
    a=PairedResponse(torch.zeros(1,1,2));a.append(ChunkState('x',torch.ones(1,2),torch.ones(1,2,1),torch.tensor([[0.,-800.]],dtype=torch.float64)))
    under=a.read(q1,native,lz1,min_denominator_ratio=.1)
    check('coefficient_underflow_recorded',under.diagnostics['coefficient_exp_underflow']==1 and not under.fallback.any(),diagnostics=under.diagnostics)
    a=artificial(1.,1.);a.states['x'].dn.fill_(float('nan'))
    bad=a.read(q1,native,lz1,min_denominator_ratio=.1)
    check('corrupted_state_fallback',bad.fallback.all() and digest(bad.output)==digest(native) and bad.diagnostics['state_nonfinite']==1)
    a=artificial(1.,1.)
    for name,query,logz in (('nonfinite_query',torch.full_like(q1,float('inf')),lz1),('nonfinite_logz',q1,torch.full_like(lz1,float('nan'))),('computed_query_overflow',torch.full_like(q1,1e308),lz1)):
        bad=a.read(query,native,logz,min_denominator_ratio=.1)
        check(name,bad.fallback.all() and digest(bad.output)==digest(native),diagnostics=bad.diagnostics)
    no_base=False
    try:a.read(q1,torch.full_like(native,float('nan')),lz1,min_denominator_ratio=.1)
    except ValueError:no_base=True
    check('nonfinite_base_rejected',no_base)
    large_output=artificial(1.,2e5).read(q1,native.half(),lz1,min_denominator_ratio=.1)
    check('output_dtype_overflow_fallback',large_output.fallback.all() and digest(large_output.output)==digest(native.half()) and large_output.diagnostics['output_cast_nonfinite_queries']==1,diagnostics=large_output.diagnostics)
    check('cuda_uninitialized',not torch.cuda.is_initialized())
    return {'status':'passed','scope':'synthetic algebra and numerical guard tests only; no real-effect evidence',
            'cpu_threads':torch.get_num_threads(),'python':platform.python_version(),'torch':torch.__version__,
            'cuda_initialized':torch.cuda.is_initialized(),'wall_seconds':time.monotonic()-start,
            'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'check_count':len(checks),'checks':checks,'summary_diagnostics':diagnostics,
            'maximum_direct_kernel_absolute_difference':max(errors),
            'source_hashes':{p.name:filehash(p) for p in (HERE/'core.py',HERE/'selftest.py')}}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output).resolve()
    if out.parent!=HERE:raise ValueError('output must stay in preparation directory')
    with out.open('x') as f:
        result=run();json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'check_count':result['check_count'],'max_absolute_difference':result['maximum_direct_kernel_absolute_difference'],'wall_seconds':result['wall_seconds'],'output':str(out)}))
