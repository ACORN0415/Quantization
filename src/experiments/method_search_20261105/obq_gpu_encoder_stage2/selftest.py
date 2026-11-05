"""CPU-only check of warp row/index arithmetic, sequential math and packing.
This is not a CUDA compilation or GPU validation of the .cu source.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,hashlib,importlib.util,json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stage2_cpu_only',P/'experiment.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(4);torch.set_grad_enabled(False)
g=torch.Generator().manual_seed(20261105)
z=torch.randn((1,2,16,12,128),generator=g).bfloat16()
s=(z.abs().amax(2,keepdim=True)/7).clamp_min(1e-8).half()
a=torch.randn((12,128,128),generator=g).double();h=a@a.transpose(1,2)+torch.eye(128)*10;u=m.c.factorize(h)

def warp_map(z,s,u):
    rows=z.numel()//128;w=z.float().reshape(rows,4,32);lane=torch.arange(32);head=torch.arange(rows)%12
    scale_row=(torch.arange(rows)//12//16)*12+head
    scales=s.float().reshape(-1,128)[scale_row].reshape(rows,4,32)
    code=torch.empty_like(w,dtype=torch.int8)
    for j in range(128):
        slot=j//32;owner=j%32
        c=(w[:,slot,owner]/scales[:,slot,owner]).round().clamp(-8,7)
        code[:,slot,owner]=c.to(torch.int8)
        if j<127:
            e=(w[:,slot,owner]-c*scales[:,slot,owner])/u[head,j,j]
            for k in range(4):
                active=k*32+lane>j;feature=(k*32+lane)[active]
                w[:,k,active]-=e[:,None]*u[head[:,None],j,feature[None,:]]
    byte=(code[:,:,::2].to(torch.uint8)&15)|((code[:,:,1::2].to(torch.uint8)&15)<<4)
    return code.reshape(z.shape),byte.flatten()
checks=[];before=m.c.tsha(z)
for label,factor in [('full',u),('identity',torch.eye(128).repeat(12,1,1))]:
    got,packed=warp_map(z,s,factor);ref=m.v1.batch_codes(z,s,factor)
    assert torch.equal(got,ref) and torch.equal(packed,m.c.pack(ref));checks.append(label+'_warp_mapping_matches_reference_codes_and_pack')
    if label=='identity':assert torch.equal(ref,(z.float()/s.float()).round().clamp(-8,7).to(torch.int8));checks.append('identity_factor_is_RTN')
assert before==m.c.tsha(z);checks.append('input_preserved')
k=torch.randn((1,4680,12,128),generator=g).bfloat16()
for kind,packet in [('K',m.c.encode_k(k,'R_fp32')),('V',m.c.encode_v(k))]:
    assert torch.equal(m.restore_gpu(packet,m.c.hadamard()),m.c.restore(packet));checks.append(kind+'_reader_operations_CPU_reference_exact')
for path in (P/'experiment.py',Path(__file__)):ast.parse(path.read_text())
checks.append('python_syntax')
assert not torch.cuda.is_initialized();checks.append('no_GPU_initialization')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result={'passed':True,'checks':checks,'actual_capture_reads':0,'GPU_initialized':False,'CUDA_compiles':0,
        'python_sha256':sha(P/'experiment.py'),'cuda_sha256':sha(P/'obq_pack.cu'),'selftest_sha256':sha(__file__)}
m.c.save(P/'SELFTEST.json',result);print(json.dumps(result,indent=2))
