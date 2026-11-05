"""Synthetic source-extracted cache contracts, no actual PRQ/model/GPU imports."""
import ast
from enum import IntEnum
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Optional
import torch

ROOT=Path('/home/acorn0415/sfq_data/qvg-reuse-reference')
CACHE=ROOT/'quant_videogen/kv_cache.py'
ROPE=ROOT/'experiments/Self-Forcing/wan/modules/causal_model.py'


def extracted(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)


def fake_decode(a,b):
    # Deliberately fake whole-span codec output. Cache read/eviction is original.
    return a['decoded'].permute(0,2,1,3),b['decoded'].permute(0,2,1,3)


def fake_pack(x):
    # Span-dependent output makes any lost fitting input visible in this contract.
    return {'decoded':(x.float()+x.float().mean()*.001).bfloat16().clone()}


def main():
    tick=time.monotonic();torch.set_num_threads(4);torch.set_grad_enabled(False)
    ns={'torch':torch,'IntEnum':IntEnum,'Optional':Optional,'uncompress_kv_cache':fake_decode}
    extracted(CACHE,{'ChunkState','ChunkedKVCache'},ns)
    Store=ns['ChunkedKVCache'];CS=ns['ChunkState'];cases=[]
    # Small spatial frame length2, H2,D12, production temporal geometry126/42/24.
    for window in (21,63):
        for sink in (0,1,3):
            ref=Store(1,2,2,12,126,torch.bfloat16,'cpu',layout='BSHD')
            actual=Store(1,2,2,12,126,torch.bfloat16,'cpu',layout='BSHD')
            writes=reads=quantizations=0;peak=0;partial=0;pending=0
            for block in range(42):
                start,end=block*3,(block+1)*3
                if block and block%8==0:
                    a,b=(block-8)*3,start
                    old=ref.read(a*2,b*2);new=actual.read(a*2,b*2)
                    assert torch.equal(old,new),'pending fitting source lost'
                    ref.store_quantized(a*2,b*2,fake_pack(old))
                    actual.store_quantized(a*2,b*2,fake_pack(new))
                    pending=b;quantizations+=1
                assert pending==(block//8)*24
                for step in range(5):
                    data=(torch.arange(start*2*24,end*2*24).reshape(1,6,2,12).float()/128+step/64).bfloat16()
                    # Only current BF16 frames may be repeatedly overwritten.
                    assert all(actual.chunk_state[i]!=CS.QUANTIZED for i in range(start,end))
                    ref.write(start*2,end*2,data);actual.write(start*2,end*2,data);writes+=1
                    prefix=min(sink,end);tail=max(prefix,end-(window-prefix))
                    release=min(tail,pending)
                    if release>prefix:actual.evict_range(prefix*2,release*2)
                    ids=list(range(prefix))+list(range(tail,end))
                    assert len(ids)==min(window,end) and len(ids)==len(set(ids))
                    ranges=[(0,prefix),(tail,end)] if prefix else [(tail,end)]
                    got=torch.cat([actual.read(a*2,b*2) for a,b in ranges if b>a],1)
                    full=ref.read(0,end*2).reshape(1,end,2,2,12)
                    expected=full[:,ids].reshape_as(got)
                    assert torch.equal(got,expected),'visible values differ'
                    reads+=1
                    if any(s['start_chunk']<tail<s['end_chunk'] for s in actual.quantized_spans):partial+=1
                    frames=sum(s==CS.BF16 for s in actual.chunk_state)+sum(s['end_chunk']-s['start_chunk'] for s in actual.quantized_spans)
                    peak=max(peak,frames)
            cases.append({'window':window,'sink':sink,'generation_blocks':42,'current_overwrites':writes,'visible_reads_checked':reads,'24frame_quantizations':quantizations,'partial_span_read_calls':partial,'peak_physically_retained_frame_equivalents':peak,'note':'BF16 frames plus full fake compressed span lengths; not actual bytes'})
    # Naive visible-window eviction loses the first pending 24-frame codec input.
    naive=Store(1,2,2,12,126,torch.bfloat16,'cpu',layout='BSHD')
    for block in range(8):
        naive.write(block*6,(block+1)*6,torch.ones(1,6,2,12,dtype=torch.bfloat16))
        if (block+1)*3>21:naive.evict_range(0,((block+1)*3-21)*2)
    try:naive.read(0,24*2)
    except ValueError as e:naive_error=str(e)
    else:raise AssertionError('naive eviction unexpectedly retained fitting source')
    # Extract original RoPE functions, without importing the model or torch.compile.
    rn={'torch':torch,'math':math};extracted(ROPE,{'causal_rope_apply','causal_rope_apply_long_input'},rn)
    rope=rn['causal_rope_apply'];long=rn['causal_rope_apply_long_input']
    gen=torch.Generator().manual_seed(17)
    x=torch.randn(1,126*6,2,12,generator=gen).bfloat16()
    phase=torch.arange(128,dtype=torch.float64)[:,None]*torch.linspace(.01,.06,6,dtype=torch.float64)[None]
    freq=torch.polar(torch.ones_like(phase),phase);grid=torch.tensor([[3,2,3]])
    all_rope=long(x,grid,freq);rope_checks=0;wrong_zero_origin_diff=False
    for window in (21,63):
        for end in (24,27,66,126):
            start=max(0,end-window)
            sliced=x[:,start*6:end*6]
            corrected=rope(sliced,torch.tensor([[end-start,2,3]]),freq,start_frame=start)
            expected=all_rope[:,start*6:end*6]
            assert torch.equal(corrected,expected);rope_checks+=1
            wrong=long(sliced,grid,freq)
            if start>0:wrong_zero_origin_diff |= not torch.equal(wrong,expected)
    assert wrong_zero_origin_diff
    # Arbitrary sink needs absolute frame IDs/segments; tail-only offset cannot encode it.
    ids=[0]+list(range(106,126));segments=torch.cat((rope(x[:,:6],torch.tensor([[1,2,3]]),freq,0),rope(x[:,106*6:126*6],torch.tensor([[20,2,3]]),freq,106)),1)
    assert torch.equal(segments,all_rope.reshape(1,126,6,2,12)[:,ids].reshape_as(segments))
    print(json.dumps({'status':'passed','cases':cases,'naive_eviction_expected_failure':naive_error,'rope_checks':rope_checks,'wrong_zero_origin_detected':wrong_zero_origin_diff,'sink_segment_absolute_positions_checked':True,'actual_PRQ_executions':0,'actual_model_loads':0,'actual_tensor_files_read':0,'GPU_calls':0,'cuda_initialized':torch.cuda.is_initialized(),'wall_seconds':time.monotonic()-tick,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (CACHE,ROPE,Path(__file__))}},indent=2))


if __name__=='__main__':main()
