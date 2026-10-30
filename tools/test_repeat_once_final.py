"""CPU verification of observer non-interference and failure detection."""
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import torch
from kvq.quant import Quantized
from mechanism_core import make
from repeat_once_audit import OperationAudit, InvalidExecution, compare_bits


def trajectory(repeat, observed):
    torch.manual_seed(417)
    store=make('C_repeat' if repeat else 'C_once',observe=False)
    digest=hashlib.sha256()
    calls={'restore':0,'encode':0}
    restore_old=Quantized.restore
    encode_old=store.quantizer.target.encode
    def restore(state):
        calls['restore']+=1
        return restore_old(state)
    def encode(x,*,value=False):
        calls['encode']+=1
        return encode_old(x,value=value)
    Quantized.restore=restore
    store.quantizer.target.encode=encode
    audit=OperationAudit([store]) if observed else None
    try:
        if audit:audit.__enter__()
        for c in range(42):
            for call in range(5):
                k=torch.randn(1,4680,1,2).bfloat16()
                v=torch.randn_like(k)
                def attention(q,k,v):
                    for x in (k,v):digest.update(x.contiguous().view(torch.uint8).numpy().tobytes())
                store.attend(k,k,v,c*4680,attention)
        result=audit.validate() if audit else None
        return digest.hexdigest(),calls,result
    finally:
        if audit:audit.__exit__(None,None,None)
        Quantized.restore=restore_old


def main():
    torch.set_num_threads(2)
    for repeat in (False,True):
        plain=trajectory(repeat,False)
        audited=trajectory(repeat,True)
        assert plain[:2]==audited[:2], (plain[:2],audited[:2])
        row=audited[2]['layers'][0]
        assert row['pair_calls']==(1155 if repeat else 41)
        assert row['fresh_k']==row['fresh_v']==41
        assert row['repeat_k']==row['repeat_v']==(1114 if repeat else 0)
        print('PASS observed/unobserved outputs and actual codec counts:',repeat,plain[1])
    x=torch.tensor([0.,1.],dtype=torch.bfloat16)
    y=torch.tensor([-0.,1.],dtype=torch.bfloat16)
    assert torch.equal(x,y) and not compare_bits(x,y)['equal']
    assert compare_bits(x,x)['equal']
    assert not compare_bits(x,x.float())['equal']
    assert not compare_bits(x,x.reshape(1,2))['equal']
    try:compare_bits(torch.tensor([float('nan')]),torch.tensor([float('nan')]))
    except InvalidExecution:pass
    else:raise AssertionError('nonfinite accepted')
    print('PASS byte comparison detects signed zero, shape, dtype and nonfinite inputs')
    store=make('C_repeat',observe=False)
    with OperationAudit([store]) as audit:
        store.repeat=False
        try:
            x=torch.zeros(1,4680,1,1,dtype=torch.bfloat16)
            store.attend(x,x,x,0,lambda *args:None)
        except InvalidExecution:pass
        else:raise AssertionError('arm mutation missed')
    print('PASS disabling repeated path is rejected')
    store=make('C_once',observe=False)
    original=store.quantizer.encode_pair
    def broken(k,v):
        return store.quantizer.target.encode(k),store.quantizer.target.encode(v)
    store.quantizer.encode_pair=broken
    with OperationAudit([store]) as audit:
        x=torch.zeros(1,4680,1,1,dtype=torch.bfloat16)
        for call in range(5):store.attend(x,x,x,0,lambda *args:None)
        try:store.attend(x,x,x,4680,lambda *args:None)
        except InvalidExecution:pass
        else:raise AssertionError('duplicate K instead of V missed')
    print('PASS incorrect actual K/V encoding is rejected')


if __name__=='__main__':main()
