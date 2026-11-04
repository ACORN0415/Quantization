"""CPU-only serialized per-page channel mixed precision; diagnostic, not GPU codec."""
import struct
import numpy as np
import torch

HEADER=struct.Struct('<4s6I')


def pack(codes,bits):
    a=np.asarray(codes,dtype=np.uint8).reshape(-1)
    assert bits in (1,2,3,4) and (a.size==0 or int(a.max())<2**bits)
    padded=np.pad(a,(0,(-a.size)%8)).reshape(-1,8).astype(np.uint32)
    words=np.sum(padded << (bits*np.arange(8,dtype=np.uint32)),axis=1,dtype=np.uint32)
    byte=((words[:,None] >> (8*np.arange(bits,dtype=np.uint32))) & 255).astype(np.uint8)
    return byte.reshape(-1)[:(a.size*bits+7)//8].tobytes()


def unpack(blob,bits,count):
    assert bits in (1,2,3,4) and len(blob)==(count*bits+7)//8
    b=np.frombuffer(blob,dtype=np.uint8)
    b=np.pad(b,(0,(-b.size)%bits)).reshape(-1,bits).astype(np.uint32)
    words=np.sum(b << (8*np.arange(bits,dtype=np.uint32)),axis=1,dtype=np.uint32)
    return ((words[:,None] >> (bits*np.arange(8,dtype=np.uint32))) & (2**bits-1)).astype(np.uint8).reshape(-1)[:count]


def encode(key,group,mode,seed):
    assert key.ndim==4 and key.shape[0]==1 and key.dtype==torch.bfloat16
    _,n,h,d=key.shape;assert n%group==0 and d==128
    x=key[0].float().reshape(n//group,group,h,d).permute(0,2,1,3).contiguous()
    p=x.shape[0];boost=63 if mode in ('selected','random') else 0;bits=0 if boost else 3
    mask=torch.zeros((p,h,d),dtype=torch.bool)
    if boost:
        score=x.abs().mean(2) if mode=='selected' else torch.rand((p,h,d),generator=torch.Generator().manual_seed(seed))
        ids=torch.argsort(score,dim=-1,descending=True,stable=True)[...,:boost]
        mask.scatter_(-1,ids,True)
    high=torch.where(mask,15.,3.).unsqueeze(2) if boost else 7.
    offset=x.amin(2,keepdim=True).half()
    scale=((x.amax(2,keepdim=True)-offset.float())/high).clamp_min(2**-24).half()
    assert torch.isfinite(offset).all() and torch.isfinite(scale).all() and (scale>0).all()
    codes=((x-offset.float())/scale.float()).round().clamp_min(0)
    codes=torch.minimum(codes,torch.as_tensor(high)).byte()
    header=HEADER.pack(b'KCB1',n,h,d,group,bits,boost)
    meta=offset.numpy().astype('<f2').tobytes()+scale.numpy().astype('<f2').tobytes()
    if boost:
        mask_blob=pack(mask.numpy(),1)
        # Stable canonical ascending channel order, recoverable from the stored mask.
        ids=mask.nonzero(as_tuple=False)[:,-1].reshape(p,h,boost)
        sparse=codes.gather(3,ids.unsqueeze(2).expand(p,h,group,boost)) >> 2
        low=pack((codes & 3).numpy(),2);extra=pack(sparse.numpy(),2)
        assert np.array_equal(unpack(low,2,codes.numel()),(codes & 3).numpy().reshape(-1))
        assert np.array_equal(unpack(extra,2,sparse.numel()),sparse.numpy().reshape(-1))
        payload=header+meta+mask_blob+low+extra
    else:
        stream=pack(codes.numpy(),bits)
        assert np.array_equal(unpack(stream,bits,codes.numel()),codes.numpy().reshape(-1))
        payload=header+meta+stream
    direct=(codes.float()*scale.float()+offset.float()).permute(0,2,1,3).reshape(1,n,h,d).bfloat16()
    restored=decode(payload)
    assert torch.equal(restored.view(torch.uint8),direct.view(torch.uint8))
    return payload,restored


def decode(payload):
    magic,n,h,d,g,bits,boost=HEADER.unpack_from(payload)
    assert magic==b'KCB1' and n%g==0 and ((bits==3 and boost==0) or (bits==0 and 0<boost<=d))
    p=n//g;size=p*h*d;pos=HEADER.size
    def take(length):
        nonlocal pos
        b=payload[pos:pos+length];assert len(b)==length;pos+=length;return b
    off=torch.from_numpy(np.frombuffer(take(size*2),dtype='<f2').copy()).view(p,h,1,d).float()
    scale=torch.from_numpy(np.frombuffer(take(size*2),dtype='<f2').copy()).view(p,h,1,d).float()
    count=n*h*d
    if boost:
        mask=torch.from_numpy(unpack(take((size+7)//8),1,size).copy()).bool().view(p,h,d)
        assert (mask.sum(-1)==boost).all()
        codes=torch.from_numpy(unpack(take((count*2+7)//8),2,count).copy()).view(p,h,g,d)
        n_hi=p*h*g*boost
        high=torch.from_numpy(unpack(take((n_hi*2+7)//8),2,n_hi).copy()).view(p,h,g,boost)
        ids=mask.nonzero(as_tuple=False)[:,-1].reshape(p,h,boost).unsqueeze(2).expand(p,h,g,boost)
        values=codes.gather(3,ids) | (high << 2);codes.scatter_(3,ids,values)
    else:codes=torch.from_numpy(unpack(take((count*bits+7)//8),bits,count).copy()).view(p,h,g,d)
    assert pos==len(payload)
    return (codes.float()*scale+off).permute(0,2,1,3).reshape(1,n,h,d).bfloat16()


def unit_checks():
    checked=0
    for bits in (1,2,3,4):
        for n in (0,1,2,3,7,8,9,15,32):
            a=np.arange(n,dtype=np.uint8) % (2**bits);ref=bytearray((n*bits+7)//8)
            for i,value in enumerate(a):
                for j in range(bits):
                    if int(value)&(1<<j):ref[(i*bits+j)//8] |= 1<<((i*bits+j)%8)
            b=pack(a,bits);assert b==bytes(ref);assert np.array_equal(a,unpack(b,bits,n));checked+=1
    gen=torch.Generator().manual_seed(17)
    for value in (torch.zeros(1,12,2,128),torch.full((1,12,2,128),3.),torch.randn((1,12,2,128),generator=gen)*200):
        for mode in ('uniform','selected','random'):
            b,y=encode(value.bfloat16(),6,mode,303);assert torch.isfinite(y).all() and y.shape==value.shape;checked+=1
    return {'status':'passed','bitstream_oracle_cases':36,'toy_roundtrips':checked-36}
