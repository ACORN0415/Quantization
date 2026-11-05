"""CPU-only locked D+mean versus scalar-I+mean diagnostic; no capture access in selftest."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import argparse,ast,gc,hashlib,importlib.util,json,math,resource,statistics,time,traceback
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];CAUSAL=HERE.parent/'causal_obq'
spec=importlib.util.spec_from_file_location('causal_helpers',CAUSAL/'common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
torch.set_num_threads(4);torch.set_grad_enabled(False)
ARMS=('R_bf16','R_fp32','FULL','D_MEAN','C_MEAN');LAYERS=(18,24);CALLS=(0,3);QIDS=torch.arange(128)*4680//128
IMPLEMENTATIONS=('R_bf16','R_fp32','FULL','FULL_BATCH','D_MEAN','C_MEAN')
BATCH_SOURCE=HERE.parent/'obq_gpu_cost_stage1'/'benchmark.py'
BATCH_SOURCE_SHA='3f1fc5b32a52a406a99f470ba81b93a9fc856021ef1baed455447cc43f189f10'
START=None

def guard():
 if START is not None and time.monotonic()-START>600:raise TimeoutError('600s cap')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4*1024**3:raise MemoryError('4GiB RSS cap')
 if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())>1024**3:raise OSError('1GiB new disk cap')
def sha(p):return c.sha(p)
def save(p,x):return c.save(p,x)
def tsha(t):return c.tsha(t)
def load(p,m):
 guard();entry=next(x for x in m['artifacts'] if Path(x['path']).name==p.name)
 assert sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes']
 return torch.load(p,map_location='cpu',weights_only=True)

def statistics_full(q):
 h,raw=c.hessian(q,return_raw=True)
 return {'h':h,'factor':c.factorize(h),'lambda':.01*raw.diagonal(dim1=1,dim2=2).mean(-1),'raw':raw}
def statistics_mean(q,mode):
 qr=(q[0].float()@c.hadamard()).permute(1,0,2).double()
 mu=qr.mean(1);var=(qr-mu[:,None]).square().mean(1)
 damp=.01*qr.square().mean(1).mean(-1)
 d=var+damp[:,None] if mode=='D_MEAN' else (var.mean(-1)+damp)[:,None].expand_as(var)
 d=d.float().contiguous();u=mu.float();assert (d>0).all() and torch.isfinite(d).all()
 term=u.square()/d;inclusive=term.flip(-1).cumsum(-1).flip(-1)
 tail=torch.cat((inclusive[:,1:],inclusive.new_zeros(len(u),1)),-1)
 return {'d':d,'u':u,'alpha':u/d,'beta':u/(1+tail),'lambda':damp}

def scalar_codes(z,sf,s):
 shape=z.shape;x=z.float().reshape(-1,12,128);sc=sf.float().expand_as(z).reshape(-1,12,128)
 state=torch.zeros(x.shape[:2]);ans=torch.empty(x.shape,dtype=torch.int8)
 for j in range(128):
  w=x[:,:,j]+s['alpha'][:,j]*state
  code=(w/sc[:,:,j]).round().clamp(-8,7);ans[:,:,j]=code.to(torch.int8)
  if j<127:state+=(w-code*sc[:,:,j])*s['beta'][:,j]
 return ans.reshape(shape)
def dense_same_h(z,sf,s):
 u=s['u'].double();h=torch.diag_embed(s['d'].double())+u[:,:,None]*u[:,None,:]
 return c.codes_obq(z,sf,h)

# AST-exact copy of the already checked batched implementation, with fixed source hash.
def batch_codes(z,s,u):
    w=z.float().reshape(-1,12,128)
    scales=s.float().expand_as(z).reshape(-1,12,128)
    out=torch.empty_like(w,dtype=torch.int8)
    for j in range(128):
        c=(w[:,:,j]/scales[:,:,j]).round().clamp(-8,7)
        out[:,:,j]=c.to(torch.int8)
        if j<127:
            e=(w[:,:,j]-c*scales[:,:,j])/u[:,j,j][None,:]
            w[:,:,j+1:]-=e[:,:,None]*u[:,j,j+1:][None,:,:]
    return out.reshape(z.shape)

def check_batch_source():
 assert sha(BATCH_SOURCE)==BATCH_SOURCE_SHA
 trees=[ast.parse(p.read_text()) for p in (Path(__file__),BATCH_SOURCE)]
 funcs=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='batch_codes') for t in trees]
 assert ast.dump(funcs[0],include_attributes=False)==ast.dump(funcs[1],include_attributes=False)

def encode(k,arm,stats):
 begin=time.perf_counter();z,s,sf=c.frames(k);a=time.perf_counter()
 if arm=='R_bf16':code=(z/s).round().clamp(-8,7).to(torch.int8)
 elif arm=='R_fp32':code=(z.float()/sf.float()).round().clamp(-8,7).to(torch.int8)
 elif arm=='FULL':code=c.codes_obq(z,sf,stats['h'],stats['factor'])
 elif arm=='FULL_BATCH':code=batch_codes(z,sf,stats['factor'])
 else:code=scalar_codes(z,sf,stats)
 b=time.perf_counter();packed=c.pack(code);assert torch.equal(c.unpack(packed,code.shape),code)
 packet={'kind':'K','shape':tuple(code.shape),'packed':packed,'scale':sf,'sentinel':torch.tensor([255],dtype=torch.uint8)}
 end=time.perf_counter()
 return packet,{'end_to_end':end-begin,'rotation_scale':a-begin,'code_with_reset_allocation':b-a,'pack_and_roundtrip':end-b}

def attention(q,k,v):
 out=torch.empty_like(q);kt=k.permute(1,2,0).contiguous();vt=v.permute(1,0,2).contiguous()
 for start in range(0,len(q),32):
  guard();p=(torch.bmm(q[start:start+32].permute(1,0,2),kt)/math.sqrt(128)).softmax(-1)
  out[start:start+32]=torch.bmm(p,vt).permute(1,0,2)
 return out

def metric(out,ref,wo,energy):
 e=(out-ref).reshape(128,-1)@wo.t();sse=float(e.double().square().sum())
 return {'sse':sse,'nse':sse/energy}

def independent_fp64_test():
 # Reference independently minimizes the remaining quadratic using its inverse.
 # It does not use the cached scalar coefficients or the Cholesky encoder.
 g=torch.Generator().manual_seed(9000);cases=0;count=0;peak=0.
 for dim in (1,2,8,32):
  for strength in (0.,.2,2.):
   for rep in range(2):
    d=torch.exp(torch.rand(dim,generator=g,dtype=torch.float64)*3-1.5)
    u=torch.randn(dim,generator=g,dtype=torch.float64)*strength
    h=torch.diag(d)+u[:,None]*u[None,:]
    x=torch.randn(7,dim,generator=g,dtype=torch.float64)*5
    scale=torch.rand(7,dim,generator=g,dtype=torch.float64)+.15
    state=torch.zeros(7,dtype=torch.float64);work=x.clone();sc=[];dc=[]
    for j in range(dim):
     w=x[:,j]+u[j]/d[j]*state
     peak=max(peak,float((w-work[:,j]).abs().max()))
     a=(w/scale[:,j]).round().clamp(-8,7)
     b=(work[:,j]/scale[:,j]).round().clamp(-8,7)
     sc.append(a);dc.append(b)
     e=w-a*scale[:,j]
     state+=e*u[j]/(1+(u[j+1:].square()/d[j+1:]).sum())
     if j+1<dim:
      inv=torch.linalg.inv(h[j:,j:])
      work[:,j+1:]-=(work[:,j]-b*scale[:,j])[:,None]*inv[1:,0][None,:]/inv[0,0]
    assert torch.equal(torch.stack(sc,1),torch.stack(dc,1))
    cases+=1;count+=x.numel()
 return {'cases':cases,'codes':count,'code_differences':0,'max_prequant_abs_difference':peak}

def math_test():
 check_batch_source()
 gen=torch.Generator().manual_seed(20551105)
 k=torch.randn(1,4680,12,128,generator=gen).bfloat16();q=torch.randn(k.shape,generator=gen).bfloat16()
 # Add channel means to avoid a near-diagonal-only synthetic test.
 q=(q.float()+torch.randn(1,1,12,128,generator=gen)).bfloat16()
 before=tsha(k);z,ns,sf=c.frames(k);checks=[]
 for mode in ('D_MEAN','C_MEAN'):
  st=statistics_mean(q,mode);co=scalar_codes(z,sf,st);de=dense_same_h(z,sf,st)
  delta=int((co!=de).sum());assert co.dtype==de.dtype==torch.int8 and co.min()>=-8 and co.max()<=7
  packet,t=encode(k,mode,st);assert torch.equal(c.unpack(packet['packed'],packet['shape']),co)
  assert c.nbytes(packet)==4515841
  checks.append({'mode':mode,'fp32_dense_code_differences':delta,'codes':co.numel()})
 for mode in ('R_bf16','R_fp32','FULL'):
  stats=statistics_full(q) if mode=='FULL' else None;p,t=encode(k,mode,stats)
  arm='O_offline' if mode=='FULL' else mode
  original=c.encode_k(k,arm,None if stats is None else stats['h'],None if stats is None else stats['factor'])
  assert all(torch.equal(p[x],original[x]) for x in ('packed','scale','sentinel'))
  if mode=='FULL':
   batch,_=encode(k,'FULL_BATCH',stats)
   assert all(torch.equal(batch[x],original[x]) for x in ('packed','scale','sentinel'))
 # Arbitrary diagonal drops exactly to FP32 RTN because mu=0.
 d=torch.rand(12,128,generator=gen)+.1;s={'d':d,'u':torch.zeros_like(d),'alpha':torch.zeros_like(d),'beta':torch.zeros_like(d)}
 assert torch.equal(scalar_codes(z,sf,s),(z.float()/sf.float()).round().clamp(-8,7).to(torch.int8))
 vp=c.encode_v(k);assert vp['packed'].dtype==torch.uint8 and c.nbytes(vp)==4500480
 assert tsha(k)==before
 return {'passed':True,'independent_fp64':independent_fp64_test(),'synthetic':checks,'diagonal_exact':True,'baseline_packets_exact':True,'input_preserved':True,
   'code_sha256':sha(__file__),'common_sha256':sha(CAUSAL/'common.py'),'batch_source_sha256':BATCH_SOURCE_SHA,
   'full_batched_packet_exact':True,'time_utc':time.time(),'GPU_calls':0}

def check_lock():
 check_batch_source()
 lock=json.loads((HERE/'LOCK.json').read_text());assert lock['root_accepted'] and lock['fable_reviewed']
 for path,h in lock['sha256'].items():
  guard();assert sha(path)==h,path
 assert lock['registered_utc']<time.time()
 return lock

def run():
 global START
 START=time.monotonic();started_utc=time.time();guard()
 lock_begin=time.monotonic();check_lock();lock_seconds=time.monotonic()-lock_begin;lock_checked_utc=time.time()
 guard();assert not torch.cuda.is_available();target=HERE/'RESULT.json'
 if target.exists():raise RuntimeError('Refuse overwrite result')
 r={'status':'running','started_utc':started_utc,'lock_validation_seconds':lock_seconds,'lock_checked_utc':lock_checked_utc,
    'rows':[],'timing':[],'stats':[],'numeric_deltas':[],'artifacts':[]}
 save(target,r)
 try:
  cap=CAUSAL/'capture';pk=CAUSAL/'packets';cm=json.loads((cap/'manifest.json').read_text());pm=json.loads((pk/'manifest.json').read_text())
  assert cm['status']==pm['status']=='completed'
  for layer in LAYERS:
   calpath=Path(f'/home/acorn0415/sfq_data/kv_alllayer_capture/p00_s101/l{layer}_c7.pt')
   calmeta=json.loads((calpath.parent/'manifest.json').read_text());t0=time.perf_counter();cal=load(calpath,calmeta);io=time.perf_counter()-t0
   info={};stat_times={}
   for arm in ('FULL','D_MEAN','C_MEAN'):
    fn=statistics_full if arm=='FULL' else lambda q,arm=arm:statistics_mean(q,arm)
    fn(cal['q']);times=[];hashes=[]
    for _ in range(3):
     guard();t0=time.perf_counter();info[arm]=fn(cal['q']);times.append(time.perf_counter()-t0)
     hashes.append(tsha(info[arm]['h'] if arm=='FULL' else info[arm]['d']))
    assert len(set(hashes))==1;stat_times[arm]=statistics.median(times)
    r['stats'].append({'layer':layer,'arm':arm,'seconds':times,'median':stat_times[arm],
       'resident_encoder_state_bytes':sum(x.numel()*x.element_size() for k,x in info[arm].items() if isinstance(x,torch.Tensor) and k!='raw')})
   old=next(o for o in pm['offline'] if o['layer']==layer)
   assert tsha(info['FULL'].pop('raw'))==old['raw_H_sha256']
   for arm in ('D_MEAN','C_MEAN'):assert torch.allclose(info[arm]['lambda'],info['FULL']['lambda'],atol=1e-12,rtol=0)
   r.setdefault('calibration_sources',[]).append({'layer':layer,'path':str(calpath),'sha256':sha(calpath),'load_verify_seconds':io,
       'lambda_max_abs_delta':max(float((info[x]['lambda']-info['FULL']['lambda']).abs().max()) for x in ('D_MEAN','C_MEAN'))})
   del cal;gc.collect();pastk=[];pastv=[];v4=[];states={a:[] for a in ARMS}
   for chunk in range(1,7):
    w=load(cap/f'l{layer}_write{chunk}.pt',cm);before=tsha(w['k']);pastk.append(w['k']);pastv.append(w['v']);v4.append(c.restore(load(pk/f'l{layer}_c{chunk}_V.pt',pm)))
    original={arm:load(pk/f'l{layer}_c{chunk}_{"O_offline" if arm=="FULL" else arm}.pt',pm) for arm in ARMS[:3]}
    for arm in IMPLEMENTATIONS:encode(w['k'],arm,info.get('FULL' if arm=='FULL_BATCH' else arm))
    times={a:[] for a in IMPLEMENTATIONS};codes={a:[] for a in IMPLEMENTATIONS};last={}
    for rep in range(3):
     shift=(chunk+rep)%len(IMPLEMENTATIONS)
     for arm in IMPLEMENTATIONS[shift:]+IMPLEMENTATIONS[:shift]:
      guard();last[arm],tim=encode(w['k'],arm,info.get('FULL' if arm=='FULL_BATCH' else arm));times[arm].append(tim);codes[arm].append(tsha(last[arm]['packed']))
    for arm in IMPLEMENTATIONS:
     assert len(set(codes[arm]))==1
     if arm=='FULL_BATCH':
      assert all(torch.equal(last[arm][x],original['FULL'][x]) for x in ('packed','scale','sentinel'))
     elif arm in original:
      assert all(torch.equal(last[arm][x],original[arm][x]) for x in ('packed','scale','sentinel'));states[arm].append(original[arm])
     else:
      path=HERE/f'l{layer}_c{chunk}_{arm}.pt';assert not path.exists();torch.save(last[arm],path);r['artifacts'].append(c.record(path));states[arm].append(last[arm])
      z,_,sf=c.frames(w['k']);dense=dense_same_h(z,sf,info[arm]);scalar=c.unpack(last[arm]['packed'],last[arm]['shape'])
      r['numeric_deltas'].append({'layer':layer,'chunk':chunk,'arm':arm,'codes':dense.numel(),'different_codes':int((dense!=scalar).sum()),
         'max_abs_code_difference':int((dense.to(torch.int16)-scalar.to(torch.int16)).abs().max()),'dense_reference_evaluated_as_arm':False})
      del z,dense,scalar
     r['timing'].append({'layer':layer,'chunk':chunk,'arm':arm,'samples':times[arm],
        'medians':{k:statistics.median([t[k] for t in times[arm]]) for k in times[arm][0]},'payload_bytes':c.nbytes(last[arm])})
    assert len({tsha(x['scale']) for x in last.values()})==1 and len({c.nbytes(x) for x in last.values()})==1
    assert tsha(w['k'])==before;del w,original,last;gc.collect()
   kp,vp,vq=torch.cat(pastk,1)[0].float(),torch.cat(pastv,1)[0].float(),torch.cat(v4,1)[0].float();del pastk,pastv,v4
   weights=load(cap/f'l{layer}_wo.pt',cm);wo=weights['wo'].float();bias=weights['bias'].float()
   for call in CALLS:
    d=load(cap/f'l{layer}_read{call}.pt',cm);q=d['q'][0,QIDS].float();kref=torch.cat((kp,d['k_current'][0].float()));vref=torch.cat((vp,d['v_current'][0].float()));vqfull=torch.cat((vq,d['v_current'][0].float()))
    ref=attention(q,kref,vref);refy=ref.reshape(128,-1)@wo.t()+bias;energy=float(refy.double().square().sum())
    row={'layer':layer,'call':call,'reference_energy':energy,'cpu_vs_saved_native':metric(ref,d['native_av'][0].float(),wo,energy),'arms':{}}
    for arm in ARMS:
     k=torch.cat([c.restore(p)[0].float() for p in states[arm]]+[d['k_current'][0].float()]);out=attention(q,k,vqfull)
     row['arms'][arm]={**metric(out,ref,wo,energy),'payload_bytes':sum(c.nbytes(p) for p in states[arm])+6*4500480};del k,out
    r['rows'].append(row);save(target,r);del d,q,kref,vref,vqfull,ref
   del states,kp,vp,vq,weights,wo,bias,info;gc.collect()
  guard();decisions={}
  for arm in ('D_MEAN','C_MEAN'):
   gains=[];eligible=True;worse=False
   for row in r['rows']:
    a=row['arms'];g={};recovery={}
    for base in ('R_bf16','R_fp32'):
     g[base]=1-a[arm]['sse']/a[base]['sse'];den=a[base]['sse']-a['FULL']['sse']
     recovery[base]=(a[base]['sse']-a[arm]['sse'])/den if den>0 else None
     eligible &= g[base]>=.2 and recovery[base] is not None and recovery[base]>=.8;worse |= g[base]<=-.1
    gains.append({'layer':row['layer'],'call':row['call'],'gain':g,'full_gain_recovery':recovery})
   costs=[]
   for layer in LAYERS:
    cand=sum(x['medians']['end_to_end'] for x in r['timing'] if x['layer']==layer and x['arm']==arm)+next(x['median'] for x in r['stats'] if x['layer']==layer and x['arm']==arm)
    full_impl={a:sum(x['medians']['end_to_end'] for x in r['timing'] if x['layer']==layer and x['arm']==a)+next(x['median'] for x in r['stats'] if x['layer']==layer and x['arm']=='FULL') for a in ('FULL','FULL_BATCH')}
    chosen=min(full_impl,key=full_impl.get);full=full_impl[chosen]
    costs.append({'layer':layer,'candidate_seconds':cand,'full_seconds':full,'full_implementation_seconds':full_impl,
       'chosen_full_implementation':chosen,'relative_reduction':1-cand/full})
   decisions[arm]={'output_decision':'signal' if eligible else ('stop_arm' if worse else 'undecided'),'cpu_cost_reduced_both_layers':all(x['candidate_seconds']<x['full_seconds'] for x in costs),'gains':gains,'costs':costs}
  choose=lambda a:decisions[a]['output_decision']=='signal' and decisions[a]['cpu_cost_reduced_both_layers']
  r.update(status='completed',decisions=decisions,preferred='C_MEAN' if choose('C_MEAN') else ('D_MEAN' if choose('D_MEAN') else None))
 except BaseException as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
 finally:
  r.update(finished_utc=time.time(),elapsed_seconds=time.monotonic()-START,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);save(target,r)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['selftest','run']);args=ap.parse_args()
 if args.mode=='selftest':
  target=HERE/'SELFTEST_V3.json'
  if target.exists():raise RuntimeError('Refuse overwrite selftest')
  x=math_test();save(target,x);print(json.dumps(x,indent=2))
 else:run()
