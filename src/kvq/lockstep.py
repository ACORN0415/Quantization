"""Snapshot BF16 context before advancing reference; no global model hooks."""
import torch
from .store import ChunkStore


def bf16_cache(pipe, device):
    capacity = pipe.local_attn_size * pipe.frame_seq_length
    if capacity <= 0: raise ValueError('finite window required')
    return [{'k':torch.zeros(1,capacity,12,128,device=device,dtype=torch.bfloat16),
             'v':torch.zeros(1,capacity,12,128,device=device,dtype=torch.bfloat16),
             'global_end_index':torch.zeros(1,device=device,dtype=torch.long),
             'local_end_index':torch.zeros(1,device=device,dtype=torch.long)}
            for _ in range(pipe.num_transformer_blocks)]


def cross_cache(pipe, device):
    return [{'k':torch.zeros(1,512,12,128,device=device,dtype=torch.bfloat16),
             'v':torch.zeros(1,512,12,128,device=device,dtype=torch.bfloat16),'is_init':False}
            for _ in range(pipe.num_transformer_blocks)]


def plain_snapshot(reference):
    return [{name:t.clone() if isinstance(t,torch.Tensor) else t for name,t in cache.items()}
            for cache in reference]


def anchor_store(reference, quantizer, capacity, chunk_tokens, sink_tokens=0):
    """Last previous chunk remains independent pending BF16 until next write."""
    end=int(reference['local_end_index'].item())
    global_end=int(reference['global_end_index'].item())
    if end % chunk_tokens or global_end % chunk_tokens or sink_tokens % chunk_tokens:
        raise ValueError('unaligned context')
    count,global_count=end//chunk_tokens,global_end//chunk_tokens
    pinned=min(sink_tokens//chunk_tokens,count)
    ids=list(range(pinned))+list(range(global_count-(count-pinned),global_count))
    store=ChunkStore(quantizer,capacity,sink_tokens)
    store.global_end,store.local_end=global_end,end
    store.chunk_tokens=chunk_tokens
    for i,idx in enumerate(ids):
        k=reference['k'][:,i*chunk_tokens:(i+1)*chunk_tokens]
        v=reference['v'][:,i*chunk_tokens:(i+1)*chunk_tokens]
        if i==count-1:
            store.pending=(k.clone(),v.clone());store.pending_start=idx*chunk_tokens
        else:
            state=quantizer.encode_pair(k,v)
            stats=[]
            for original,encoded in zip((k,v),state):
                x=original.float()
                stats.append(((encoded.restore().float()-x).square().sum().item(),x.square().sum().item()))
            store.segments.append({'start':idx*chunk_tokens,'raw':None,'state':state,'error_stats':stats})
    return store


def compressed_cache(pipe, quantizer, device, reference=None):
    capacity=pipe.local_attn_size*pipe.frame_seq_length
    count=pipe.num_frame_per_block*pipe.frame_seq_length
    sink=int(getattr(pipe.args.model_kwargs,'sink_size',0))*pipe.frame_seq_length
    result=[]
    for layer in range(pipe.num_transformer_blocks):
        store=(anchor_store(reference[layer],quantizer,capacity,count,sink) if reference is not None
               else ChunkStore(quantizer,capacity,sink))
        result.append({'store':store,'k':torch.empty(0,device=device,dtype=torch.bfloat16),
                       'v':torch.empty(0,device=device,dtype=torch.bfloat16),
                       'global_end_index':torch.tensor([store.global_end],device=device),
                       'local_end_index':torch.tensor([store.local_end],device=device),'layer':layer})
    return result


def chunk_noise(pipe, frames, seed, device):
    n=pipe.num_frame_per_block
    if frames % n or pipe.independent_first_frame:raise ValueError('fixed chunks required')
    return torch.cat([torch.randn((1,n,16,60,104),device=device,dtype=torch.bfloat16,
                                 generator=torch.Generator(device=device).manual_seed(seed*1000003+c))
                      for c in range(frames//n)],1)


def run_chunk(pipe, cache, xcache, cond, noise, chunk, seed):
    n=pipe.num_frame_per_block
    x=noise[:,chunk*n:(chunk+1)*n]
    generator=torch.Generator(device=noise.device).manual_seed(seed*1000003+7919+chunk)
    for index,t in enumerate(pipe.denoising_step_list):
        timestep=torch.ones((1,n),device=noise.device,dtype=torch.int64)*t
        _,pred=pipe.generator(noisy_image_or_video=x,conditional_dict=cond,timestep=timestep,
                              kv_cache=cache,crossattn_cache=xcache,current_start=chunk*n*pipe.frame_seq_length)
        if index+1<len(pipe.denoising_step_list):
            flat=pred.flatten(0,1)
            eps=torch.randn(flat.shape,device=flat.device,dtype=flat.dtype,generator=generator)
            x=pipe.scheduler.add_noise(flat,eps,pipe.denoising_step_list[index+1]*
                                       torch.ones(n,device=noise.device,dtype=torch.long)).unflatten(0,pred.shape[:2])
    pipe.generator(noisy_image_or_video=pred,conditional_dict=cond,
                   timestep=torch.ones_like(timestep)*pipe.args.context_noise,
                   kv_cache=cache,crossattn_cache=xcache,current_start=chunk*n*pipe.frame_seq_length)
    return pred


def relative(a,b):
    a,b=a.float(),b.float()
    return ((a-b).norm()/b.norm().clamp_min(1e-12)).item()
