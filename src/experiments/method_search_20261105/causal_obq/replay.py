"""Build CPU packets using only causally available files; separate native GPU reader."""
import argparse,gc,json,math,statistics,sys,time,traceback,resource
from common import *

def verified(p,manifest):
    row=next(a for a in manifest['artifacts'] if Path(a['path']).name==p.name)
    assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
    return torch.load(p,map_location='cpu',weights_only=True)

def packets():
    check_lock();guard()
    cap=HERE/'capture';cm=json.loads((cap/'manifest.json').read_text());assert cm['status']=='completed'
    out=HERE/'packets'
    if out.exists():raise RuntimeError('Refuse overwrite packet directory')
    out.mkdir();r={'status':'running','started_utc':time.time(),'rows':[],'artifacts':[],'offline':[]}
    save(out/'manifest.json',r)
    try:
        for layer in LAYERS:
            path=Path(f'/home/acorn0415/sfq_data/kv_alllayer_capture/p00_s101/l{layer}_c7.pt')
            manifest=json.loads((path.parent/'manifest.json').read_text());t0=time.perf_counter()
            raw=verified(path,manifest);hoff,hraw=hessian(raw['q'],return_raw=True);del raw
            capacity=json.loads((HERE.parent/'obq_capacity_codex/RESULT.json').read_text())
            old=next(x for x in capacity['rows'] if x['layer']==layer)
            assert tsha(hraw)==old['hessian_raw_sha256'] and sha(path)==old['calibration']['sha256']
            factor_start=time.perf_counter();foff=factorize(hoff);factor_seconds=time.perf_counter()-factor_start
            r['offline'].append({'layer':layer,'source':record(path),'offline_prepare_including_validation_factor_seconds':time.perf_counter()-t0,'H_sha256':tsha(hoff),'H_bytes':hoff.numel()*hoff.element_size(),'raw_H_sha256':tsha(hraw),'factor_bytes':foff.numel()*foff.element_size(),'factor_seconds':factor_seconds})
            for chunk in range(1,7):
                guard();w=verified(cap/f'l{layer}_write{chunk}.pt',cm)
                fr=verified(cap/f'l{layer}_first_reader{chunk}.pt',cm)
                assert w['context']=={'layer':layer,'chunk':chunk,'call':4,'timestep':0.0}
                assert fr['context']['chunk']==chunk+1 and fr['context']['call']==0 and fr['encoded_source_chunk']==chunk
                hs={'O_offline':hoff};htimes={}
                for arm,q in [('O_own_refresh',w['q']),('O_first_reader',fr['q'])]:
                    hessian(q) # fixed one warmup, no candidate choice
                    samples=[];digests=[]
                    for _ in range(3):
                        t0=time.perf_counter();hs[arm]=hessian(q);samples.append(time.perf_counter()-t0);digests.append(tsha(hs[arm]))
                    assert len(set(digests))==1
                    htimes[arm]={'seconds':samples,'median':statistics.median(samples),'input_q_sha256':tsha(q),'H_sha256':digests[0]}
                v=encode_v(w['v']);vp=out/f'l{layer}_c{chunk}_V.pt';torch.save(v,vp);r['artifacts'].append(record(vp))
                row={'layer':layer,'source_chunk':chunk,'first_reader_chunk':chunk+1,'first_reader_call':0,
                     'H_times':htimes,'arms':{},'v_payload_bytes':nbytes(v)}
                # Fixed warmup, then three rotated-order passes; all arms include K transform, scales, code, pack and roundtrip check.
                for arm in ARMS:encode_k(w['k'],arm,hs.get(arm),foff if arm=='O_offline' else None)
                reps={a:[] for a in ARMS};hashes={a:[] for a in ARMS};ps={}
                for rep in range(3):
                    order=ARMS[(chunk+rep)%len(ARMS):]+ARMS[:(chunk+rep)%len(ARMS)]
                    for arm in order:
                        guard();t0=time.perf_counter();packet=encode_k(w['k'],arm,hs.get(arm),foff if arm=='O_offline' else None);reps[arm].append(time.perf_counter()-t0)
                        hashes[arm].append(tsha(packet['packed']));ps[arm]=packet
                assert len({tsha(p['scale']) for p in ps.values()})==1
                assert len({nbytes(p) for p in ps.values()})==1
                for arm in ARMS:
                    assert len(set(hashes[arm]))==1
                    p=out/f'l{layer}_c{chunk}_{arm}.pt';torch.save(ps[arm],p);r['artifacts'].append(record(p))
                    row['arms'][arm]={'seconds':reps[arm],'median':statistics.median(reps[arm]),'payload_bytes':nbytes(ps[arm]),
                        'code_sha256':hashes[arm][0],'scale_sha256':tsha(ps[arm]['scale'])}
                r['rows'].append(row);save(out/'manifest.json',r)
                del w,fr,hs,ps,packet;gc.collect()
        r['status']='completed'
    except BaseException as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        r.update(finished_utc=time.time(),max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        save(out/'manifest.json',r)

def post_metric(out,ref,wo,bias):
    y=ref.float().reshape(128,-1)@wo.float().t()+bias.float()
    e=(out.float()-ref.float()).reshape(128,-1)@wo.float().t()
    sse=float(e.double().square().sum());return {'sse':sse,'nse':sse/float(y.double().square().sum())}

def native():
    check_lock();guard()
    sys.path.insert(0,str(ROOT/'src'))
    from wan.modules.attention import attention
    cap=HERE/'capture';pk=HERE/'packets'
    cm=json.loads((cap/'manifest.json').read_text());pm=json.loads((pk/'manifest.json').read_text())
    assert cm['status']==pm['status']=='completed'
    target=HERE/'NATIVE_RESULT.json'
    if target.exists():raise RuntimeError('Refuse overwrite native result')
    r={'status':'running','started_utc':time.time(),'endpoints':[],'packet_manifest_sha256':sha(pk/'manifest.json')}
    save(target,r)
    try:
        for layer in LAYERS:
            wo=verified(cap/f'l{layer}_wo.pt',cm)
            ks=[];vs=[];v4=[]
            for c in range(1,7):
                w=verified(cap/f'l{layer}_write{c}.pt',cm);ks.append(w['k']);vs.append(w['v'])
                v4.append(restore(verified(pk/f'l{layer}_c{c}_V.pt',pm)))
            kp,vp,vq=torch.cat(ks,1),torch.cat(vs,1),torch.cat(v4,1);del ks,vs,v4,w
            for call in READS:
                guard();d=verified(cap/f'l{layer}_read{call}.pt',cm)
                q=d['q'].cuda();kf=torch.cat((kp,d['k_current']),1).cuda();vf=torch.cat((vp,d['v_current']),1).cuda()
                actual=attention(q,kf,vf);ref=actual[:,QIDS].cpu()[0]
                assert torch.equal(ref.contiguous().view(torch.uint8),d['native_av'][0].contiguous().view(torch.uint8)), 'native BF16 replay byte mismatch'
                row={'layer':layer,'call':call,'native_replay_byte_equal':True,'arms':{}}
                del actual,kf,vf
                vf=torch.cat((vq,d['v_current']),1).cuda()
                for arm in ARMS:
                    guard();ps=[verified(pk/f'l{layer}_c{c}_{arm}.pt',pm) for c in range(1,7)]
                    t0=time.perf_counter();kr=torch.cat([restore(p) for p in ps]+[d['k_current']],1);restore_s=time.perf_counter()-t0
                    torch.cuda.synchronize();t0=time.perf_counter();kg=kr.cuda();torch.cuda.synchronize();transfer_s=time.perf_counter()-t0
                    for _ in range(2):attention(q,kg,vf)
                    torch.cuda.synchronize();samples=[];digests=[];out=None
                    for _ in range(5):
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        start.record();out=attention(q,kg,vf);end.record();end.synchronize();samples.append(start.elapsed_time(end)/1000)
                        digests.append(tsha(out[:,QIDS].cpu()))
                    assert len(set(digests))==1
                    row['arms'][arm]={**post_metric(out[0,QIDS].cpu(),ref,wo['wo'],wo['bias']),
                       'native_attention_seconds':samples,'native_attention_median':statistics.median(samples),
                       'cpu_restore_seconds':restore_s,'key_h2d_seconds':transfer_s,'payload_bytes':sum(nbytes(p) for p in ps)+sum(x['v_payload_bytes'] for x in pm['rows'] if x['layer']==layer),
                       'repeated_outputs_byte_equal':True}
                    del kr,kg,out,ps;gc.collect()
                r['endpoints'].append(row);save(target,r);del q,vf,d
            del kp,vp,vq,wo;gc.collect()
        decisions={}
        for arm in ARMS[2:]:
            gains=[{'layer':e['layer'],'call':e['call'],**{a:1-e['arms'][arm]['sse']/e['arms'][a]['sse'] for a in ARMS[:2]}} for e in r['endpoints']]
            values=[g[a] for g in gains for a in ARMS[:2]]
            decisions[arm]={'gains':gains,'decision':'signal' if min(values)>=.2 else ('stop_arm' if min(values)<=-.1 else 'undecided')}
        r.update(status='completed',decisions=decisions)
        assert sha(pk/'manifest.json')==r['packet_manifest_sha256']
    except BaseException as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        r.update(finished_utc=time.time(),peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        save(target,r)

if __name__=='__main__':
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['packets','native']);a=p.parse_args()
    packets() if a.mode=='packets' else native()
