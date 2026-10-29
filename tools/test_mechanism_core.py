"""Meaningful gates: eviction, clean refresh, group alignment and old adapters."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'src/scripts'))
import torch
from anchor_identity_gate import IdentityQuantizer
from kvq import Quantizer
from kvq.store import ChunkStore
from mechanism_core import make,state_bytes,histories,region_layout


def run():
    torch.manual_seed(872341);torch.set_num_threads(2)
    for chunk in (1,6,7,8,41):
        n=min(chunk,6)*4680
        for group in (16,24,32):
            for call in (0,1,4):
                layouts={a:region_layout(a,group,chunk,call,n) for a in ('L','C_repeat','C_once')}
                for masks,signature in layouts.values():
                    assert (sum(m.int() for m in masks.values())==1).all()
                if group==24:assert len({v[1] for v in layouts.values()})==1
                assert not layouts['C_once'][0]['mixed_current_group'].any()
    records=[]
    for group in (16,24,32):
        stores={a:make(a,group,capacity=168) for a in ('L','C_repeat','C_once')}
        old=ChunkStore(Quantizer('rtn',4,4,group),168)
        identities={a:make(a,group,capacity=168,identity=IdentityQuantizer()) for a in stores}
        for chunk in range(12):
            for call in range(5):
                k=torch.randn(1,24,2,4).bfloat16();v=torch.randn_like(k)
                seen={};noquant={}
                for arm,store in stores.items():
                    store.attend(k,k,v,chunk*24,lambda q,a,b,arm=arm:seen.update({arm:(a.clone(),b.clone())}))
                old_seen=[];old.attend(k,k,v,chunk*24,lambda q,a,b:old_seen.extend([a,b]))
                assert all(torch.equal(a,b) for a,b in zip(seen['C_once'],old_seen))
                if group==24:
                    assert all(torch.equal(a,b) for a,b in zip(seen['L'],seen['C_repeat'])),(chunk,call)
                    # Rolling counts now include its post-attention commit.
                    assert {s:n-1 for s,n in histories(stores['L']).items()}==histories(stores['C_repeat'])
                for arm,store in identities.items():
                    store.attend(k,k,v,chunk*24,lambda q,a,b,arm=arm:noquant.update({arm:(a.clone(),b.clone())}))
                assert all(torch.equal(a,b) for arm in identities for a,b in zip(noquant['L'],noquant[arm]))
        records.append(dict(group=group,old_chunk_equivalent=True,noquant_all_equal=True,
                            aligned_rolling_repeat_equal=group==24,bytes={a:state_bytes(s) for a,s in stores.items()}))
    print('PASS 3 groups × 12 chunks × 5 writes; eviction, pending refresh, aligned read identity, counts and component bytes')
    return records


if __name__=='__main__':run()
