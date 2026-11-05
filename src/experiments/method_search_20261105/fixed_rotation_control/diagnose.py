"""Fixed17 rotation control. Import/selftest never reads a real capture or uses CUDA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import argparse,gc,importlib.util,json,math,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
import torch
torch.set_num_threads(4);torch.set_grad_enabled(False)
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
rot=module('fixed_rotation_helpers',BASE/'qcov_rotation_baseline_codex/diagnose_rotation.py')
old=rot.old
c=module('fixed_rotation_causal',BASE/'causal_obq/common.py')
ARMS=('RTN_BF16','RTN_FP32','OBQ_FULL','FIXED17_FP32_RTN')
START=None

def guard():
    if START is not None and time.monotonic()-START>1195:raise TimeoutError('1200s cap, 5s cleanup reserve')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3:raise MemoryError('8GiB RSS cap')
    if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())>1024**3:raise OSError('1GiB disk cap')

def family():
    out=[];h32=rot.had64(32).float();eye=torch.eye(32)
    for mask in range(16):out.append(torch.block_diag(*[h32 if mask>>j&1 else eye for j in range(4)]))
    out.append(c.hadamard())
    return torch.stack(out)

def raw_moment(q):
    qh=q[0].permute(1,0,2).double()
    return torch.bmm(qh.transpose(1,2),qh)/qh.shape[1]

def score_error(error,moment):
    # Sum actual reconstructed key errors, never future evaluation queries.
    e=error[0].permute(1,0,2).double();s=torch.einsum('htd,hde,hte->h',e,moment,e)
    assert torch.isfinite(s).all() and (s>=-1e-9).all()
    return s

def encode(k,r,arm='FIXED17_FP32_RTN',u=None):
    guard();tick=time.monotonic();z,sn,sf=rot.make_frames(k,r)
    assert bool(torch.isfinite(z).all() & torch.isfinite(sf).all() & (sf>0).all()),'invalid stored scale'
    prepare_s=time.monotonic()-tick;tick=time.monotonic()
    if arm=='RTN_BF16':codes=old.rtn_codes(z,sn,False)
    elif arm=='OBQ_FULL':codes=c.codes_obq(z,sf,None,u)
    else:codes=old.rtn_codes(z,sf,True)
    packed=c.pack(codes);pack_s=time.monotonic()-tick;tick=time.monotonic()
    un=c.unpack(packed,codes.shape);assert torch.equal(un,codes)
    zhat=(un.bfloat16()*sf.bfloat16()).reshape(-1,1568,12,128)[:,:1560].reshape_as(k)
    restored=rot.rotate(zhat,r,True).bfloat16()
    packet={'packed':packed,'scale':sf,'sentinel':torch.full((k.shape[1]//4680,),255,dtype=torch.uint8)}
    restore_s=time.monotonic()-tick
    info={'code_sha256':c.tsha(codes),'scale_sha256':c.tsha(sf),'payload_bytes':c.nbytes(packet),
      'rotate_scale_seconds':prepare_s,'code_pack_seconds':pack_s,'verified_unpack_restore_seconds':restore_s}
    return restored,packet,info

def calibrate(q,k):
    before=(c.tsha(q),c.tsha(k));begin=time.monotonic();m=raw_moment(q)
    hd,hr=c.hessian(q,return_raw=True);u=c.factorize(hd)
    prep=time.monotonic()-begin;fs=family();scores=[];candidate_times=[]
    for i,r0 in enumerate(fs):
        guard();tick=time.monotonic();r=r0.repeat(12,1,1)
        rec,packet,_=encode(k,r);scores.append(score_error(rec.float()-k.float(),m))
        candidate_times.append(time.monotonic()-tick);del rec,packet;gc.collect()
    scores=torch.stack(scores);choice=scores.argmin(0);r=fs[choice].contiguous()
    assert before==(c.tsha(q),c.tsha(k))
    info={'raw_original_C_sha256':c.tsha(m),'raw_rotated_H_sha256':c.tsha(hr),
      'damped_H_sha256':c.tsha(hd),'U_sha256':c.tsha(u),
      'candidate_scores_by_head':scores.tolist(),'choice_by_head':choice.tolist(),
      'candidate_order':'0..15 bitmask I/H32, then 16=H128; lowest index on exact ties',
      'matrix_sha256':c.tsha(r),'moment_factor_seconds':prep,'candidate_seconds':candidate_times,
      'total_seconds':time.monotonic()-begin,'rotation_state_bytes':r.numel()*r.element_size(),
      'label_state_bytes':choice.numel(),'U_state_bytes':u.numel()*u.element_size(),
      'shared_H128_bytes':c.hadamard().numel()*4,'source_tensors_preserved':True}
    return r,u,info

def near(x,y):return math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-9)

def verified(path,manifest):
    guard();rows=[a for a in manifest['artifacts'] if Path(a['path']).name==path.name];assert len(rows)==1
    assert c.sha(path)==rows[0]['sha256'] and path.stat().st_size==rows[0]['bytes']
    return torch.load(path,map_location='cpu',weights_only=True)

def restore_v_checked(packet):
    # Validate the stored FP16 representation before either p4 or causal V is decoded.
    assert packet['kind']=='V' and tuple(packet['shape'])==(1,293,16,12,128)
    scale=packet['scale']
    assert scale.shape==(1,293,1,12,128) and scale.dtype==torch.float16
    assert packet['packed'].dtype==torch.uint8 and c.nbytes(packet)==4500480
    assert bool(torch.isfinite(scale).all() & (scale>0).all()),'invalid stored V FP16 scale'
    return c.restore(packet)

def fixed_v4_checked(value):
    assert value.shape==(1,7*4680,12,128) and value.dtype==torch.bfloat16
    parts=[];payload=0
    for chunk in range(6):
        packet=c.encode_v(value[:,chunk*4680:(chunk+1)*4680])
        parts.append(restore_v_checked(packet));payload+=c.nbytes(packet)
    return torch.cat(parts+[value[:,6*4680:]],1)[0].float(),payload

def evaluation(q,k,v,v4,vbytes,wo,bias,native,r,u,tag,anchor=None,packet_refs=None):
    before={n:c.tsha(t) for n,t in [('q',q),('k',k),('v',v),('v4',v4)]}
    q=q[0,old.QIDS].float();wo=wo.float();bias=bias.float()
    reference=old.attn(q,k[0].float(),v[0].float())
    energy=old.energy(reference.reshape(128,-1)@wo.t()+bias)
    kref=old.attn(q,k[0].float(),v4)
    row={'endpoint':tag,'reference_energy':energy,'cpu_vs_native':old.metric(reference,native.float(),wo,energy),'arms':{}}
    hm=c.hadamard().repeat(12,1,1)
    for arm in ARMS:
        guard();tick=time.monotonic();rr=r if arm=='FIXED17_FP32_RTN' else hm
        rec,packet,inf=encode(k[:,:old.PAST],rr,arm,u);enc=time.monotonic()-tick
        if anchor is not None and arm!='FIXED17_FP32_RTN':
            a=anchor['arms'][{'RTN_BF16':'B0_BF16_RTN','RTN_FP32':'B0_FP32_RTN','OBQ_FULL':'OBQ_FULL'}[arm]]
            assert inf['code_sha256']==a['code_sha256'] and inf['scale_sha256']==a['scale_sha256']
            assert inf['payload_bytes']+vbytes==a['packed_payload_bytes']
        if packet_refs is not None and arm!='FIXED17_FP32_RTN':
            patharm={'RTN_BF16':'R_bf16','RTN_FP32':'R_fp32','OBQ_FULL':'O_offline'}[arm]
            refs=packet_refs(patharm)
            assert torch.equal(packet['packed'],torch.cat([p['packed'] for p in refs]))
            assert torch.equal(packet['scale'],torch.cat([p['scale'] for p in refs]))
            assert torch.equal(packet['sentinel'],torch.cat([p['sentinel'] for p in refs]))
            del refs
        full=torch.cat((rec,k[:,old.PAST:]),1)[0].float();ta=time.monotonic()
        out=old.attn(q,full,v4);att=time.monotonic()-ta
        item={**old.metric(out,reference,wo,energy),**inf,'total_KV_payload_bytes':inf['payload_bytes']+vbytes,
            'value_payload_bytes':vbytes,'encode_pack_restore_seconds':enc,'attention_seconds':att,
            'k_only_reference':old.metric(out,kref,wo,energy)}
        if anchor is not None and arm!='FIXED17_FP32_RTN':assert near(item['sse'],a['sse']) and near(energy,anchor['reference_energy'])
        row['arms'][arm]=item;del rec,packet,full,out;gc.collect()
    assert len({a['total_KV_payload_bytes'] for a in row['arms'].values()})==1
    nq=rot.rotate(rot.rotate(k[:,:old.PAST],r).bfloat16(),r,True).bfloat16()
    no=old.attn(q,torch.cat((nq,k[:,old.PAST:]),1)[0].float(),v4)
    row['noquant_vs_K_reference']=old.metric(no,kref,wo,energy)
    row['noquant_fraction_of_RTN_FP32']=row['noquant_vs_K_reference']['sse']/row['arms']['RTN_FP32']['sse']
    assert row['noquant_fraction_of_RTN_FP32']<=.01,'noquant arithmetic exceeds registered 1%'
    after={n:c.tsha(t) for n,t in [('k',k),('v',v),('v4',v4)]}
    assert all(after[n]==before[n] for n in after)
    row['source_tensors_preserved']=True
    a=row['arms'];row['OBQ_gain_vs_FIXED17']=1-a['OBQ_FULL']['sse']/a['FIXED17_FP32_RTN']['sse']
    return row

def decide(rows):
    assert len(rows)==6
    if all(r['OBQ_gain_vs_FIXED17']>=.1 for r in rows):return 'continue_beyond_simple_fixed_rotation'
    if any(r['arms']['OBQ_FULL']['sse']>r['arms']['FIXED17_FP32_RTN']['sse'] and not near(r['arms']['OBQ_FULL']['sse'],r['arms']['FIXED17_FP32_RTN']['sse']) for r in rows):
        return 'broader_quality_gap_uncertain_simple_control_better'
    return 'undecided_below_registered_10_percent'

def local_python_dependencies():
    # Include package __init__ files as well as dynamically imported helpers.
    paths={Path(__file__).resolve()}
    for m in [rot,old,c,old.lib,*list(sys.modules.values())]:
        filename=getattr(m,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.suffix=='.py' and path.is_file() and path.is_relative_to(old.ROOT):paths.add(path)
    return sorted(paths)

def sources():
    files=[Path(__file__),HERE/'REGISTRATION_PROPOSAL_KO.txt',HERE/'SELFTEST_V3.json',
      BASE/'qcov_rotation_baseline_codex/diagnose_rotation.py',BASE/'obq_capacity_codex/diagnose_obq.py',
      BASE/'obq_capacity_codex/RESULT.json',BASE/'causal_obq/common.py',old.LIB_PATH,
      old.ROOT/'src/experiments/kv_headrule_rollout/headrule_quantizer.py',old.ROOT/'src/kvq/quant.py',
      old.ROOT/'src/kvq/__init__.py',old.ROOT/'src/kvq/store.py',
      old.DATA/'p00_s101/manifest.json',old.DATA/'p04_s101/manifest.json',
      BASE/'causal_obq/capture/manifest.json',BASE/'causal_obq/packets/manifest.json']
    return list(dict.fromkeys(files+local_python_dependencies()))

def verify_lock():
    lock=json.loads((HERE/'LOCK.json').read_text())
    assert lock['root_execution_authorized'] and lock['pilot_finished_before_start']
    assert lock['review_status'] in ('actual_independent_review_completed','explicit_root_review_deferral')
    for p in sources():assert c.sha(p)==lock['sha256'][str(p)],str(p)
    # The authority and review snapshots added by the coordinator are protected too.
    for filename,digest in lock['sha256'].items():assert c.sha(Path(filename))==digest,filename
    assert lock['limit_seconds']==1200 and lock['cpu_threads']==4

def worker():
    global START
    START=float(json.loads((HERE/'SUPERVISOR.json').read_text())['started_monotonic']);old.RUN_START=START
    verify_lock();guard();target=HERE/'RESULT.json';assert not target.exists()
    result={'status':'running','started_utc':time.time(),'rows':[],'calibration':[],
      'GPU_calls':0,'threads':torch.get_num_threads(),'torch_version':torch.__version__,
      'lock_sha256':c.sha(HERE/'LOCK.json'),'scope':'existing development captures; CPU attention, not native GPU endpoints'}
    c.save(target,result)
    try:
        prior=json.loads((BASE/'obq_capacity_codex/RESULT.json').read_text())
        cap=BASE/'causal_obq/capture';pk=BASE/'causal_obq/packets'
        cm=json.loads((cap/'manifest.json').read_text());pm=json.loads((pk/'manifest.json').read_text())
        for layer in (18,24):
            guard();tc=time.monotonic();cal,csrc=old.load_capture(0,layer);load_s=time.monotonic()-tc
            r,u,meta=calibrate(cal['q'],cal['k'][:,:old.PAST]);meta['source_load_and_validation_seconds']=load_s
            expected=next(x for x in prior['rows'] if x['layer']==layer)
            assert meta['raw_rotated_H_sha256']==expected['hessian_raw_sha256'] and csrc==expected['calibration']
            meta.update(layer=layer,source=csrc);state=HERE/f'layer{layer}_fixed.pt'
            assert not state.exists();torch.save({'R':r,'choice':torch.tensor(meta['choice_by_head'],dtype=torch.uint8),'U':u},state)
            meta['state_artifact']=c.record(state);result['calibration'].append(meta);del cal;gc.collect()
            data,src=old.load_capture(4,layer);assert src==expected['evaluation'];qhash=c.tsha(data['q'])
            v4,vbytes=fixed_v4_checked(data['v'])
            row=evaluation(data['q'],data['k'],data['v'],v4,vbytes,data['wo'],data['bias'],data['native_av'][0,old.QIDS],r,u,
                f'p04_layer{layer}_c7_call0',anchor=expected)
            assert c.tsha(data['q'])==qhash;row['source']=src;result['rows'].append(row);c.save(target,result)
            print('endpoint',row['endpoint'],'completed',flush=True);del data,v4;gc.collect()
            writes=[verified(cap/f'l{layer}_write{i}.pt',cm) for i in range(1,7)]
            assert all(w['context']=={'layer':layer,'chunk':i,'call':4,'timestep':0.0} for i,w in enumerate(writes,1))
            pastk=torch.cat([w['k'] for w in writes],1);pastv=torch.cat([w['v'] for w in writes],1);del writes
            vp=[verified(pk/f'l{layer}_c{i}_V.pt',pm) for i in range(1,7)]
            vv=torch.cat([restore_v_checked(p) for p in vp],1);vbytes=sum(c.nbytes(p) for p in vp);del vp
            wo=verified(cap/f'l{layer}_wo.pt',cm)
            def packet_refs(arm):return [verified(pk/f'l{layer}_c{i}_{arm}.pt',pm) for i in range(1,7)]
            for call in (0,3):
                guard();data=verified(cap/f'l{layer}_read{call}.pt',cm);qhash=c.tsha(data['q'])
                assert data['context']['layer']==layer and data['context']['chunk']==7 and data['context']['call']==call
                assert data['past_chunk_ids']==list(range(1,7))
                k=torch.cat((pastk,data['k_current']),1);v=torch.cat((pastv,data['v_current']),1)
                v4=torch.cat((vv,data['v_current']),1)[0].float()
                row=evaluation(data['q'],k,v,v4,vbytes,wo['wo'],wo['bias'],data['native_av'][0],r,u,
                    f'causal_layer{layer}_c7_call{call}',packet_refs=packet_refs)
                assert c.tsha(data['q'])==qhash;result['rows'].append(row);c.save(target,result)
                print('endpoint',row['endpoint'],'completed',flush=True);del data,k,v,v4;gc.collect()
            del pastk,pastv,vv,wo,r,u;gc.collect();guard()
        result.update(status='completed',decision=decide(result['rows']));verify_lock()
    except BaseException as e:result.update(status='invalid_or_incomplete',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-START,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        c.save(target,result)

def supervise():
    assert not (HERE/'SUPERVISOR.json').exists() and (HERE/'LOCK.json').exists()
    start=time.monotonic();status={'started_monotonic':start,'started_utc':time.time(),'status':'running'}
    c.save(HERE/'SUPERVISOR.json',status);child=None
    def stop():
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    try:
        def interrupted(signum,frame):raise KeyboardInterrupt(f'Supervisor signal {signum}')
        signal.signal(signal.SIGTERM,interrupted)
        verify_lock()
        with (HERE/'RUN_LOG.txt').open('x') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker'],cwd=old.ROOT,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=child.wait(timeout=max(.1,1195-(time.monotonic()-start)));assert code==0,f'worker exit {code}'
        status['status']='completed'
    except BaseException as e:stop();status.update(status='invalid_or_incomplete',error=repr(e));raise
    finally:
        status.update(elapsed_seconds=time.monotonic()-start,finished_utc=time.time());c.save(HERE/'SUPERVISOR.json',status)

def selftest():
    assert not torch.cuda.is_initialized();assert not (HERE/'SELFTEST_V3.json').exists()
    gen=torch.Generator().manual_seed(20261105);fs=family();identity=torch.eye(128)
    allcodes=torch.arange(-8,8,dtype=torch.int8);assert torch.equal(c.unpack(c.pack(allcodes),allcodes.shape),allcodes)
    assert fs.shape==(17,128,128) and torch.equal(fs[0],identity) and torch.equal(fs[16],c.hadamard())
    # Use exact FP64 mathematical family for the FP64 contract.
    f64=[torch.block_diag(*[rot.had64(32) if mask>>j&1 else torch.eye(32,dtype=torch.float64) for j in range(4)]) for mask in range(16)]+[rot.had64(128)]
    q=torch.randn(7,128,generator=gen).double();x=torch.randn(9,128,generator=gen).double()
    for f in f64:
        assert float((f.t()@f-torch.eye(128)).abs().max())<=1e-10
        assert float(((q@f)@(x@f).t()-q@x.t()).abs().max())<=1e-10
    for f in fs:
        torch.testing.assert_close((x.float()@f)@f.t(),x.float(),rtol=5e-5,atol=5e-5)
        torch.testing.assert_close((q.float()@f)@(x.float()@f).t(),(q@x.t()).float(),rtol=5e-5,atol=5e-5)
    qs=torch.randn(1,19,12,128,generator=gen);es=torch.randn(1,23,12,128,generator=gen)
    score=score_error(es,raw_moment(qs))
    direct=torch.einsum('qhd,khd->hqk',qs[0].double(),es[0].double()).square().sum((1,2))/19
    torch.testing.assert_close(score,direct,rtol=1e-12,atol=1e-10)
    k=torch.randn(1,4680,12,128,generator=gen).bfloat16();before=c.tsha(k)
    h=fs[16].repeat(12,1,1);rec,packet,info=encode(k,h,'RTN_FP32')
    z,sn,sf=old.make_frames(k,c.hadamard());codes=old.rtn_codes(z,sf,True);oref,obytes=old.packed_restore(codes,sf,c.hadamard())
    assert torch.equal(rec,oref) and info['payload_bytes']==obytes==4515841
    assert torch.equal(packet['packed'],c.pack(codes)) and torch.equal(packet['scale'],sf)
    choice=torch.arange(12)%17;r=fs[choice];rr,pp,ii=encode(k,r)
    assert ii['payload_bytes']==obytes and torch.isfinite(rr).all()
    oldbytes={n:c.tsha(t) for n,t in pp.items()}
    encode((-k*.75).bfloat16(),r)
    assert oldbytes=={n:c.tsha(t) for n,t in pp.items()} and c.tsha(k)==before
    fixture=torch.ones(17,12);fixture[3,0]=0.;fixture[8,1]=0.
    assert fixture.argmin(0).tolist()==[3,8]+[0]*10
    # The guarded V path must preserve the previous valid packet/arithmetic exactly.
    vv=torch.randn(1,7*4680,12,128,generator=gen).bfloat16();vbefore=c.tsha(vv)
    checked,vbytes=fixed_v4_checked(vv);previous,previous_bytes=old.lib.fixed_v4(vv)
    assert torch.equal(checked,previous) and vbytes==previous_bytes==6*4500480 and c.tsha(vv)==vbefore
    packet_v=c.encode_v(vv[:,:4680]);old_state=old.lib.Quantizer('rtn',4,4,group=16).encode(vv[:,:4680],value=True)
    assert torch.equal(packet_v['packed'],c.pack(old_state.codes)) and torch.equal(packet_v['scale'],old_state.scale)
    assert torch.equal(restore_v_checked(packet_v),old_state.restore())
    from unittest.mock import patch
    invalid_v_checks=[]
    for label,fill in (('zero',0.),('tiny',1e-10)):
        bad=c.encode_v(torch.full_like(vv[:,:4680],fill))
        assert bool((bad['scale']==0).all())
        with patch.object(c,'restore',side_effect=RuntimeError('decoder reached before V validity guard')):
            try:restore_v_checked(bad)
            except AssertionError as exc:assert str(exc)=='invalid stored V FP16 scale';invalid_v_checks.append(label)
            else:raise AssertionError('invalid V scale accepted')
    for label,value in (('nan',float('nan')),('inf',float('inf')),('negative',-1.)):
        bad=dict(packet_v);bad['scale']=packet_v['scale'].clone();bad['scale'].view(-1)[0]=value
        with patch.object(c,'restore',side_effect=RuntimeError('decoder reached before V validity guard')):
            try:restore_v_checked(bad)
            except AssertionError as exc:assert str(exc)=='invalid stored V FP16 scale';invalid_v_checks.append(label)
            else:raise AssertionError('invalid V scale accepted')
    assert len(invalid_v_checks)==5
    dependencies=local_python_dependencies()
    assert all(old.ROOT/('src/kvq/'+name) in dependencies for name in ('__init__.py','quant.py','store.py'))
    assert all(path in sources() for path in dependencies)
    def cases(ratio):return [{'OBQ_gain_vs_FIXED17':1-ratio,'arms':{'OBQ_FULL':{'sse':ratio},'FIXED17_FP32_RTN':{'sse':1.}}} for _ in range(6)]
    assert decide(cases(.8))=='continue_beyond_simple_fixed_rotation'
    assert decide(cases(1.1))=='broader_quality_gap_uncertain_simple_control_better'
    assert decide(cases(.95))=='undecided_below_registered_10_percent'
    assert not torch.cuda.is_initialized()
    c.save(HERE/'SELFTEST_V3.json',{'passed':True,'GPU_calls':0,'real_capture_reads':0,'model_loads':0,
      'family_17':True,'raw_score_projection_equivalence':True,'exact_H128_anchor':True,
      'owned_output_preserved':True,'packed_payload_bytes':obytes,'input_preserved':True,'decision_cases':3,
      'valid_V_path_exact_previous':True,'V_payload_bytes_six_chunks':vbytes,'invalid_V_rejected_before_decoder':invalid_v_checks,
      'local_import_dependency_sha256':{str(path):c.sha(path) for path in dependencies},
      'source_sha256':c.sha(__file__),'registration_sha256':c.sha(HERE/'REGISTRATION_PROPOSAL_KO.txt'),
      'torch_version':torch.__version__,'threads':torch.get_num_threads(),'time_utc':time.time()})
    print('synthetic-only selftest passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['selftest','run','worker']);a=p.parse_args()
    {'selftest':selftest,'run':supervise,'worker':worker}[a.mode]()
