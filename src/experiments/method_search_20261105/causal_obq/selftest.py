"""CPU-only comparison to locked capacity code and original quantizer, no real captures."""
import importlib.util,json,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *

def test():
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    p=HERE.parent/'obq_capacity_codex/diagnose_obq.py'
    assert sha(p)=='07324360458a1d5687cff8932db316af97f9945ab75a689c7f3f7a0f0968af0d'
    spec=importlib.util.spec_from_file_location('locked_obq',p);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    gen=torch.Generator().manual_seed(20261106)
    k=torch.randn(1,CHUNK,HEADS,DIM,generator=gen).bfloat16();q=torch.randn(k.shape,generator=gen).bfloat16()
    original=tsha(k);xg,s,ss=old.make_frames(k,old.hadamard(DIM,'cpu'));h=hessian(q)
    full,_=old.obq_codes(xg,ss,h)
    for arm,c in [('R_bf16',old.rtn_codes(xg,s,False)),('R_fp32',old.rtn_codes(xg,ss,True)),('O_offline',full),('O_own_refresh',full),('O_first_reader',full)]:
        pnew=encode_k(k,arm,h,factorize(h) if arm=='O_offline' else None)
        assert torch.equal(unpack(pnew['packed'],pnew['shape']),c)
        assert torch.equal(pnew['scale'],ss)
        restored,bytesold=old.packed_restore(c,ss,old.hadamard(DIM,'cpu'))
        assert torch.equal(restore(pnew),restored) and nbytes(pnew)==bytesold
    diag=torch.diag_embed(torch.rand(HEADS,DIM,generator=gen).double()+.5)
    cdiag=codes_obq(xg,ss,diag)
    assert torch.equal(cdiag,old.rtn_codes(xg,ss,True))
    original_v=old.lib.Quantizer('rtn',4,4,group=16).encode(k,value=True)
    vp=encode_v(k)
    assert torch.equal(restore(vp),original_v.restore()) and nbytes(vp)==original_v.bytes()['packed_estimate']
    assert tsha(k)==original
    assert list(range(1,7))==[c-1 for c in range(2,8)]
    qids=[i*CHUNK//128 for i in range(128)]
    assert QIDS.tolist()==qids
    result={'passed':True,'checks':['all five K arms exact codes/scales/restore/payload vs locked capacity',
       'nonuniform diagonal full OBQ exact FP32 RTN','V actual pack/restoration exact original Quantizer',
       'input preserved','first-reader source mapping fixed','query indices fixed'],
       'time_utc':time.time(),'source_sha256':{str(HERE/x):sha(HERE/x) for x in ('common.py','capture.py','replay.py','selftest.py')},
       'new_k_payload_bytes':nbytes(pnew),'v_payload_bytes':nbytes(vp),'gpu_calls':0}
    target=HERE/'SELFTEST_V2.json'
    if target.exists():raise RuntimeError('Refuse overwrite selftest record')
    save(target,result);print(json.dumps(result,indent=2))
if __name__=='__main__':test()
