"""Two-generation passive BF16 trace. Never replaces model attention outputs."""
import argparse,gc,json,os,resource,sys,time,traceback
from pathlib import Path
from common import *
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src'),str(ROOT/'src/scripts')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from kv_channel_full_attention import free_gpu,same

def capture(mode):
    check_lock();free_gpu();os.chdir(ROOT/'src')
    import p3_tf_fr as pilot
    import wan.modules.causal_model as model
    folder=HERE/mode
    if folder.exists():raise RuntimeError('Refuse overwrite generation')
    folder.mkdir()
    if mode=='reference':
        if (HERE/'RUN.json').exists():raise RuntimeError('Refuse restart experiment clock')
        save(HERE/'RUN.json',{'started_utc':time.time(),'maximum_generations':2,'seed':SEED,'text':TEXT})
    else:assert json.loads((HERE/'reference/manifest.json').read_text())['status']=='completed'
    meta={'status':'running','mode':mode,'started_utc':time.time(),'seed':SEED,'text':TEXT,
          'model_forwards':0,'artifacts':[],'chunks':[],'events':[],'source_sha256':pilot.source_hash()}
    save(folder/'manifest.json',meta)
    original=model.attention;handles=[];pipe=None
    try:
        guard(capture=True)
        torch.cuda.init();torch.cuda.reset_peak_memory_stats()
        pipe=pilot.build(SEED);guard(capture=True)
        meta['after_build_peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        cond=pilot.encode(pipe,[TEXT])[0];guard(capture=True)
        meta['after_encode_peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        assert len(pipe.denoising_step_list)==4 and pipe.args.context_noise==0
        assert pipe.num_frame_per_block==3 and pipe.frame_seq_length==1560 and pipe.local_attn_size==21
        import inspect
        from kvq.store import ChunkStore
        assert inspect.signature(ChunkStore).parameters['residual_tokens'].default==0
        meta['replay_assumed_residual_tokens']=0
        cache=pilot.bf16_cache(pipe,pilot.gpu);xc=pilot.cross_cache(pipe,pilot.gpu)
        noise=pilot.chunk_noise(pipe,24,SEED,pilot.gpu)
        modules=[m for m in pipe.generator.modules() if isinstance(m,model.CausalWanSelfAttention)]
        assert len(modules)==30
        active={};global_call={};counts={l:0 for l in LAYERS};writes={l:{} for l in LAYERS}
        expected=None if mode=='reference' else torch.load(HERE/'reference/latent.pt',map_location='cpu',weights_only=True)['latent']
        def model_pre(module,args,kwargs):
            guard(capture=True);n=meta['model_forwards'];meta['model_forwards']+=1
            assert meta['model_forwards']<=40
            chunk,call=divmod(n,5)
            ts=kwargs['timestep'];want=pipe.denoising_step_list[call] if call<4 else 0
            assert bool((ts==want).all()) and kwargs['current_start']==chunk*CHUNK
            global_call.update(chunk=chunk,call=call,timestep=float(ts.flatten()[0]))
        handles.append(pipe.generator.register_forward_pre_hook(model_pre,with_kwargs=True))
        def pre(layer):
            def hook(module,args):
                count=counts[layer];counts[layer]+=1
                assert (count//5,count%5)==(global_call['chunk'],global_call['call'])
                active.clear();active.update(layer=layer,**global_call)
            return hook
        def persist(name,data):
            path=folder/name;assert not path.exists();torch.save(data,path)
            meta['artifacts'].append(record(path));guard(capture=True)
        if mode=='capture':
            for layer in LAYERS:
                handles.append(modules[layer].register_forward_pre_hook(pre(layer)))
                persist(f'l{layer}_wo.pt',{'wo':modules[layer].o.weight.detach().cpu(),'bias':modules[layer].o.bias.detach().cpu()})
            def attention(q,k,v,*args,**kwargs):
                ctx=dict(active);active.clear()
                y=original(q,k,v,*args,**kwargs)
                if not ctx:return y
                l,c,call=ctx['layer'],ctx['chunk'],ctx['call']
                assert q.shape==(1,CHUNK,HEADS,DIM)
                if call==4 and 1<=c<=6:
                    payload={'q':q.detach().cpu(),'k':k[:,-CHUNK:].detach().cpu().clone(),
                             'v':v[:,-CHUNK:].detach().cpu().clone(),'context':ctx}
                    writes[l][c]=(payload['k'],payload['v'])
                    persist(f'l{l}_write{c}.pt',payload)
                    meta['events'].append({'kind':'refresh_write',**ctx,'q_sha256':tsha(payload['q'])})
                if call==0 and 2<=c<=7:
                    payload={'q':q.detach().cpu(),'context':ctx,'encoded_source_chunk':c-1}
                    persist(f'l{l}_first_reader{c-1}.pt',payload)
                    meta['events'].append({'kind':'first_reader_available',**ctx,'source_chunk':c-1,'q_sha256':tsha(payload['q'])})
                if c==7 and call in READS:
                    assert k.shape==(1,7*CHUNK,HEADS,DIM) and sorted(writes[l])==list(range(1,7))
                    pastk=torch.cat([writes[l][j][0] for j in range(1,7)],1)
                    pastv=torch.cat([writes[l][j][1] for j in range(1,7)],1)
                    assert same(pastk,k[:,:6*CHUNK].detach().cpu()) and same(pastv,v[:,:6*CHUNK].detach().cpu())
                    payload={'q':q.detach().cpu(),'k_current':k[:,-CHUNK:].detach().cpu().clone(),
                        'v_current':v[:,-CHUNK:].detach().cpu().clone(),'native_av':y[:,QIDS].detach().cpu(),
                        'context':ctx,'past_chunk_ids':list(range(1,7))}
                    persist(f'l{l}_read{call}.pt',payload)
                    meta['events'].append({'kind':'evaluation_read',**ctx,'past_raw_byte_equal':True,'q_sha256':tsha(payload['q'])})
                return y
            model.attention=attention
        ys=[];t0=time.perf_counter()
        for c in range(8):
            begin=time.perf_counter();y=pilot.run_chunk(pipe,cache,xc,cond,noise,c,SEED).cpu()
            row={'chunk':c,'seconds':time.perf_counter()-begin,'finite':bool(torch.isfinite(y).all())}
            assert row['finite']
            if expected is not None:
                row['reference_byte_equal']=bool(same(y,expected[:,c*3:(c+1)*3]));assert row['reference_byte_equal']
            ys.append(y);meta['chunks'].append(row);save(folder/'manifest.json',meta)
        torch.cuda.synchronize();meta['sampling_seconds']=time.perf_counter()-t0
        persist('latent.pt',{'latent':torch.cat(ys,1)})
        assert meta['model_forwards']==40 and pilot.source_hash()==meta['source_sha256']
        if mode=='capture':assert all(n==40 for n in counts.values()) and len(meta['artifacts'])==31
        meta.update(status='completed',layer_call_counts=counts)
    except BaseException as e:
        meta.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        model.attention=original
        for h in handles:h.remove()
        meta.update(finished_utc=time.time(),max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0)
        save(folder/'manifest.json',meta)

if __name__=='__main__':
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['reference','capture']);a=ap.parse_args();capture(a.mode)
