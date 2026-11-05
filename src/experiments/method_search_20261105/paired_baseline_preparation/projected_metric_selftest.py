"""Synthetic equivalence to explicit constructed M_r; no capture/model/GPU I/O."""
import argparse,hashlib,json,math,os,resource,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import torch
import projected_metric as p
c=p.c


def run():
    started=time.monotonic();torch.set_num_threads(4);torch.set_grad_enabled(False)
    checks=[];rows=[]
    def check(name, ok, **detail):
        checks.append({'name':name,'passed':bool(ok),**detail})
        if not ok:raise AssertionError(name)
    def rejects(name,fn):
        try:fn()
        except ValueError:check(name,True)
        else:check(name,False)
    g=torch.Generator().manual_seed(411)
    for case,(n,h,d,r) in enumerate(((17,2,8,3),(19,3,12,1),(7,1,4,4))):
        ideal=torch.stack([torch.linalg.qr(torch.randn(d,d,generator=g,dtype=torch.float64)).Q[:,:r] for _ in range(h)])
        u=c.prepare_basis(ideal)
        lam=(torch.arange(r,0,-1,dtype=torch.float32)[None].repeat(h,1)*.7+.1).contiguous()
        state=p.prepare_metric(u,lam)
        k=torch.randn(n,h,d,generator=g).bfloat16()
        kh=(k.float()+torch.randn(n,h,d,generator=g)*.17).bfloat16()
        inputs=(k,kh,u,lam,state.basis,state.eigenvalues)
        before=[c.as_bytes(x) for x in inputs]
        # Independent explicit moment from exactly the stored U and lambda.
        matrix=(state.basis.double()*state.eigenvalues.double()[:,None,:]) @ state.basis.double().transpose(-1,-2)
        projected,pi=p.psac_encode_projected(k,kh,state)
        full,fi=c.psac_encode(k,kh,state.basis,matrix)
        check(f'packet_exact_case{case}',projected.to_bytes()==full.to_bytes())
        check(f'active_rows_exact_case{case}',pi['active_rows']==fi['active_rows'])
        pr=p.psac_restore(kh,projected,state);fr=c.psac_restore(kh,full,state.basis)
        check(f'BF16_restore_exact_case{case}',c.as_bytes(pr)==c.as_bytes(fr))
        max_score_difference=0.
        for label,err in (('before',k.float()-kh.float()),('after',k.float()-pr.float())):
            expected=torch.einsum('nhd,hde,nhe->nh',err.double(),matrix,err.double())
            actual=p.projected_row_error(err,state)
            diff=float((actual-expected).abs().max());max_score_difference=max(max_score_difference,diff)
            check(f'row_score_{label}_case{case}',torch.allclose(actual,expected,atol=1e-10,rtol=1e-11),max_absolute_difference=diff)
        check(f'aggregate_scores_case{case}',all(math.isclose(pi[x],fi[x],rel_tol=1e-11,abs_tol=1e-10) for x in ('before_query_error','after_screened_query_error')))
        check(f'input_bytes_preserved_case{case}',before==[c.as_bytes(x) for x in inputs])
        check(f'shared_state_bytes_case{case}',state.tensor_payload_bytes()==h*d*r*2+h*r*4 and len(vars(state))==2)
        roundtrip=c.PsacPacket.from_bytes(projected.to_bytes())
        check(f'packet_roundtrip_case{case}',c.as_bytes(p.psac_restore(kh,roundtrip,state))==c.as_bytes(pr))
        rows.append({'shape':[n,h,d,r],'active_rows':pi['active_rows'],'packet_bytes':projected.nbytes(),'basis_bytes':pi['basis_bytes_shared'],'lambda_bytes':pi['eigenvalue_bytes_shared'],'maximum_score_difference':max_score_difference})

    # BF16 rounding tie must still discard a correction, as in full-M review.
    u=c.prepare_basis(torch.tensor([[[1/math.sqrt(2)],[1/math.sqrt(2)]]],dtype=torch.float64))
    st=p.prepare_metric(u,torch.tensor([[2.]],dtype=torch.float32))
    k=torch.tensor([[[1.,1.0078125]]],dtype=torch.bfloat16);kh=torch.ones_like(k)
    packet,info=p.psac_encode_projected(k,kh,st)
    explicit=st.basis.double() @ torch.diag_embed(st.eigenvalues.double()) @ st.basis.double().transpose(-1,-2)
    fp,fi=c.psac_encode(k,kh,st.basis,explicit)
    check('BF16_tie_dropped',info['active_rows']==0 and packet.to_bytes()==fp.to_bytes() and c.as_bytes(p.psac_restore(kh,packet,st))==c.as_bytes(kh))
    zero=p.prepare_metric(st.basis,torch.zeros_like(st.eigenvalues))
    zp,zi=p.psac_encode_projected(k,kh,zero)
    check('zero_lambda_all_ties_drop',zi['active_rows']==0 and c.as_bytes(p.psac_restore(kh,zp,zero))==c.as_bytes(kh))
    signed=kh.clone();signed[0,0,0]=-0.
    qp,qi=p.psac_encode_projected(signed,signed,st)
    check('noquant_signed_zero_bytes',qi['active_rows']==0 and c.as_bytes(p.psac_restore(signed,qp,st))==c.as_bytes(signed))
    with torch.enable_grad():
        ug=st.basis.detach().clone().requires_grad_();lg=st.eigenvalues.detach().clone().requires_grad_()
        kg=k.detach().clone().requires_grad_();khg=kh.detach().clone().requires_grad_()
        inputs=(ug,lg,kg,khg);before=[c.as_bytes(x) for x in inputs]
        sg=p.prepare_metric(ug,lg);gp,gi=p.psac_encode_projected(kg,khg,sg)
        tensors=[sg.basis,sg.eigenvalues,gp.header,gp.codes,gp.scales,gp.indices]
        check('grad_enabled_persistent_state_detached',all(not x.requires_grad and x.grad_fn is None for x in tensors))
        check('owned_metric_storage',sg.basis.data_ptr()!=ug.data_ptr() and sg.eigenvalues.data_ptr()!=lg.data_ptr())
        check('grad_enabled_inputs_preserved',before==[c.as_bytes(x) for x in inputs] and all(x.requires_grad and x.grad is None for x in inputs))
        check('caller_grad_mode_preserved',torch.is_grad_enabled())
    rejects('negative_lambda',lambda:p.prepare_metric(st.basis,-torch.ones_like(st.eigenvalues)))
    rejects('nonfinite_lambda',lambda:p.prepare_metric(st.basis,torch.full_like(st.eigenvalues,float('nan'))))
    rejects('FP64_lambda_rejected_not_silently_converted',lambda:p.prepare_metric(st.basis,st.eigenvalues.double()))
    rejects('FP32_basis_rejected_not_silently_converted',lambda:p.prepare_metric(st.basis.float(),st.eigenvalues))
    bad=k.clone();bad[0,0,0]=float('inf')
    rejects('nonfinite_key',lambda:p.psac_encode_projected(bad,kh,st))
    check('no_full_moment_retained',set(vars(st))=={'basis','eigenvalues'})
    # The stored BF16/FP32 metric is not silently treated as an ideal full moment.
    ideal_u=torch.tensor([[[1/math.sqrt(2)],[1/math.sqrt(2)]]],dtype=torch.float64)
    ideal_m=2*(ideal_u @ ideal_u.transpose(-1,-2))
    check('stored_BF16_metric_differs_from_ideal',not torch.equal(explicit,ideal_m),maximum_difference=float((explicit-ideal_m).abs().max()))
    budgets={}
    for rank in (1,2,3,4,8):
        u0=torch.eye(128,dtype=torch.bfloat16)[None,:,:rank].repeat(12,1,1).contiguous()
        l0=torch.ones(12,rank,dtype=torch.float32)
        ss=p.prepare_metric(u0,l0)
        codes=torch.zeros(4680*12,rank,dtype=torch.int8);scales=torch.ones(4680*12,1,dtype=torch.float16)
        active=torch.ones(4680*12,dtype=torch.bool)
        dense,_=c.packet_from_rows(codes,scales,active,4680,12,128)
        total=6*dense.nbytes()+ss.tensor_payload_bytes()
        expected=6*(28+4680*12*(rank+2))+12*128*rank*2+12*rank*4
        check(f'dense_budget_includes_lambda_rank{rank}',total==expected and ss.tensor_payload_bytes()==sum(x.untyped_storage().nbytes() for x in vars(ss).values()),total_bytes=total)
        budgets[str(rank)]={'six_dense_packets_and_shared_U_lambda':total,'lambda_bytes':12*rank*4}
    check('FULL_plus_rank1_arithmetic',786432+budgets['1']['six_dense_packets_and_shared_U_lambda']==1800600)
    check('cuda_not_initialized',not torch.cuda.is_initialized())
    return {'status':'passed','synthetic_only':True,'real_capture_reads':0,'GPU_calls':0,'model_loads':0,'effect_experiments':0,'checks_total':len(checks),'checks':checks,'cases':rows,'budgets':budgets,'wall_seconds':time.monotonic()-started,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'threads':torch.get_num_threads(),'cuda_initialized':torch.cuda.is_initialized(),'torch':torch.__version__}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run();here=Path(__file__).resolve().parent
    result['source_hashes']={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in ('core.py','projected_metric.py','projected_metric_selftest.py')}
    output=Path(args.output).resolve()
    if output.parent!=here:raise ValueError('output must stay in preparation folder')
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'checks_total':result['checks_total'],'wall_seconds':result['wall_seconds'],'budgets':result['budgets'],'output':str(output)}))
