"""Native-span CUDA validity gates, called inside the stage-1 budget."""
import copy
import gc
import torch
from quant_videogen import compress_kv_cache,get_quantize_fn,QuantizeConfig
from kv_reuse_core import ReusePool,ReadController,make_store_class


def exact(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(
        a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def run():
    cfg=QuantizeConfig(quant_type='triton-nstages-kmeans-int2',quant_block_size=64,
                       num_prq_stages=1,kmeans_max_iters=2)
    torch.manual_seed(303)
    x=torch.randn((1,12,37440,128),device='cuda',dtype=torch.bfloat16)
    y=torch.randn_like(x)
    states=compress_kv_cache(x,y,cfg.quant_type,cfg,get_quantize_fn(cfg.quant_type,cfg))
    del x,y
    for s in states:s['info']=dict(quant_config=cfg,output_dtype=torch.bfloat16)
    controls={arm:ReadController(arm,ReusePool(2**30) if arm=='R' else None) for arm in ('U','B','R')}
    stores={}
    for arm,c in controls.items():
        cls=make_store_class(c)
        stores[arm]=[cls(1,1560,12,128,27,torch.bfloat16,torch.device('cuda'),layout='BSHD') for _ in range(2)]
        for store,s in zip(stores[arm],states):store.store_quantized(0,37440,s)
    checks=[]
    for start,end in ((0,37440),(1560,35880),(0,37440)):
        for kind in range(2):
            ref=stores['U'][kind].read(start,end)
            for arm in ('B','R'):
                got=stores[arm][kind].read(start,end)
                assert exact(got,ref),(arm,start,end)
                del got
            del ref
        checks.append(f'full/partial read {start}:{end}')
    # Same state identity, changed underlying tensor version must miss the cache.
    states[0]['centroids_list'][0].add_(0.25)
    ref=stores['B'][0].read(0,37440);got=stores['R'][0].read(0,37440)
    assert exact(ref,got);del ref,got
    checks.append('in-place tensor mutation invalidates version')
    replacement=copy.deepcopy(states[1])
    replacement['centroids_list'][0].add_(0.125)
    for arm in stores:stores[arm][1].store_quantized(0,37440,replacement)
    ref=stores['B'][1].read(0,37440);got=stores['R'][1].read(0,37440)
    assert exact(ref,got);del ref,got
    checks.append('state replacement invalidates held span')
    # No selection is possible if the entire K/V pair cannot fit.
    c=ReadController('R',ReusePool(0));cls=make_store_class(c)
    tiny=cls(1,1560,12,128,27,torch.bfloat16,torch.device('cuda'),layout='BSHD')
    tiny.store_quantized(0,37440,states[0])
    ref=stores['B'][0].read(0,37440);got=tiny.read(0,37440)
    assert exact(ref,got) and c.pool.used==0;del ref,got,tiny
    checks.append('insufficient pair budget falls back without truncating context')
    raw=torch.randn((1,4680,12,128),device='cuda',dtype=torch.bfloat16)
    for arm in stores:
        for store in stores[arm]:store.write(37440,42120,raw)
    ref=stores['B'][0].read(0,42120);got=stores['R'][0].read(0,42120)
    assert exact(ref,got);del ref,got
    checks.append('new raw chunk concatenation')
    for arm in ('B','R'):
        try:stores[arm][0].write(0,4680,raw)
        except ValueError as exc:assert 'Partial overwrite' in str(exc)
        else:raise AssertionError('partial quantized overwrite accepted')
    checks.append('unsupported partial span overwrite rejected before mutation')
    for arm in stores:
        for store in stores[arm]:store.evict_range(0,1560)
    ref=stores['B'][0].read(1560,37440);got=stores['R'][0].read(1560,37440)
    assert exact(ref,got);del ref,got
    checks.append('partial eviction retains a still-live native span')
    for arm in stores:
        for store in stores[arm]:store.evict_range(0,37440)
    assert all(entry['states']==[None,None] for entry in controls['R'].pool.entries.values())
    for arm in stores:
        for store,s in zip(stores[arm],states):store.store_quantized(0,37440,s)
    ref=stores['B'][0].read(0,37440);got=stores['R'][0].read(0,37440)
    assert exact(ref,got);del ref,got
    checks.append('full eviction and reinsert')
    # Full-span overwrite returns the newly written values, never held data.
    raw_full=torch.zeros((1,37440,12,128),device='cuda',dtype=torch.bfloat16)
    for arm in stores:stores[arm][0].write(0,37440,raw_full)
    assert exact(stores['R'][0].read(0,37440),raw_full)
    checks.append('full overwrite invalidation')
    stats=controls['R'].pool.stats()
    del stores,controls,states,raw,raw_full,replacement,c,cls
    gc.collect();torch.cuda.empty_cache()
    return dict(passed=True,shape=[1,12,37440,128],checks=checks,pool_stats=stats)
