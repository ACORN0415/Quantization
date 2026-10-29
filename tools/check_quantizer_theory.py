"""CPU-only exact rational examples and native grouping differential checks.

This is not a floating-point proof. It does not change experimental code or data.
"""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import re
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kvq import Quantizer


def ideal(x,group,q):
    values=list(x);out=[]
    for lo in range(0,len(values),group):
        block=values[lo:lo+group];maximum=max(abs(v) for v in block)
        if not maximum:out.extend([F(0)]*len(block));continue
        scale=maximum/q
        out.extend(scale*round(v/scale) for v in block)
    return out


def squared(a,b):return sum((x-y)**2 for x,y in zip(a,b))


def native_group(group,position):
    x=torch.zeros(1,position+1,1,1,dtype=torch.float32);x[:,position]=7
    state=Quantizer('rtn',4,4,group).encode(x)
    nz=torch.nonzero(state.codes,as_tuple=False)
    if nz.shape[0]!=1:raise AssertionError('impulse produced unexpected nonzero codes')
    return int(nz[0,1]),int(nz[0,2])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lean-log',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=ROOT/'src/experiments/mechanism_diagnostic_results/theory_checks.json')
    args=ap.parse_args();torch.set_num_threads(1)
    pattern=r'VECTOR group=(\d+) shift=(\d+) position=(\d+) beforeGroup=(\d+) afterGroup=(\d+) beforeOffset=(\d+) afterOffset=(\d+)'
    vectors=[tuple(map(int,m)) for m in re.findall(pattern,args.lean_log.read_text())]
    if len(vectors)!=24:raise AssertionError(f'expected 24 Lean vectors, got {len(vectors)}')
    for g,shift,p,bg,ag,bo,ao in vectors:
        assert native_group(g,p)==(bg,bo)
        assert native_group(g,p+shift)==(ag,ao)
    # Exact rational arithmetic, no float approximations in these examples.
    examples=[]
    for name,x in [('increase',[F(49,100),F(49,100),F(1),F(49,100)]),
                   ('decrease',[F(1),F(4,5),F(3,2),F(2)])]:
        y=ideal(x,2,7);assert ideal(y,2,7)==y
        ref=x[1:];before=y[1:];after=ideal(before,2,7)
        old=squared(before,ref);new=squared(after,ref)
        assert (new>old) if name=='increase' else (new<old)
        examples.append(dict(name=name,source=list(map(str,x)),quantized=list(map(str,y)),
                             shifted=list(map(str,before)),requantized=list(map(str,after)),
                             before_sse=str(old),after_sse=str(new)))
    # Reach the actual cache shape and eviction remainder with constructed K/V.
    native=[];q=Quantizer('rtn',4,4,16)
    for name in ('increase','decrease'):
        x=torch.zeros(1,32760,1,1,dtype=torch.bfloat16)
        if name=='increase':
            x[:,4672:4688]=.49;x[:,4688:4704]=.49;x[:,4688]=1
        else:
            x[:,4672:4680]=1;x[:,4680:4688]=.8;x[:,4688:4703]=1.5;x[:,4703]=2
        y=q.encode(x).restore()
        def evict(z):return torch.cat((z[:,4680:],torch.zeros_like(z[:,:4680])),1)
        ref=evict(x);before=evict(y);after=q.encode(before).restore()
        old=float((before.float()-ref.float()).square().sum());new=float((after.float()-ref.float()).square().sum())
        assert (new>old) if name=='increase' else (new<old)
        native.append(dict(name=name,shape=list(x.shape),group=16,shift=4680,bits=4,dtype='bfloat16',
                           before_sse=old,after_sse=new,changed_tokens=int((before!=after).sum()),
                           scope='Production quantizer on synthetic buffers; actual Wan activation reachability not established'))
    result=dict(lean_vectors_matched=24,native_encode_calls_for_vectors=48,
                integer_domains='Lean Nat and Python arbitrary precision; tested tensor lengths <=37440',
                rational_examples=examples,native_synthetic_examples=native,
                scope='Geometry proof and human-readable ideal-arithmetic RTN proof have different scope. No formal BF16, CUDA, or video-quality proof.')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS 24 Lean vectors vs native Quantizer.encode; two exact rational and two shipped-shape BF16 regrouping examples')


if __name__=='__main__':main()
