"""Read-only audit of recorded hashes, CPU packet fixtures and GPU timing records.
No benchmark import, CUDA calls or source mutation. Writes a new audit result only.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
from pathlib import Path
import hashlib,json,math,time
import torch
torch.set_num_threads(4)
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
checks=[]
def check(name,condition):
    assert condition,name
    checks.append(name)
def bits(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def quantile(values,p):
    ordered=sorted(values);r=(len(ordered)-1)*p;i=int(r);f=r-i
    return ordered[i]*(1-f)+ordered[min(i+1,len(ordered)-1)]*f
def qstats(values):
    q1,q2,q3=[quantile(values,p) for p in (.25,.5,.75)]
    return dict(median=q2,q1=q1,q3=q3,iqr=q3-q1)
def equal_stats(a,b):
    return a.keys()==b.keys() and all(math.isclose(a[k],b[k],rel_tol=1e-12,abs_tol=1e-14) for k in a)
start=time.monotonic();lock=read(P/'LOCK.json');result=read(P/'RESULT.json');review=read(P/'REVIEW_LINK.json');st=read(P/'SELFTEST.json')
check('result completed within 900 seconds',result['status']=='completed' and result['elapsed_seconds']<=900)
check('lock hash',sha(P/'LOCK.json')==result['lock_sha256'])
for path,digest in lock['sha256'].items():check('locked source '+path,sha(path)==digest)
check('selftest hash and code',sha(P/'SELFTEST.json')==lock['selftest_sha256'] and st['passed'] and st['code_sha256']==sha(P/'benchmark.py'))
check('historical review snapshot',sha(P/'REVIEW_AT_LOCK_KO.txt')==review['review_sha256'] and Path(review['review_path']).read_bytes().startswith((P/'REVIEW_AT_LOCK_KO.txt').read_bytes()) and review['code_sha256']==sha(P/'benchmark.py') and review['registration_sha256']==sha(P/'REGISTRATION_DRAFT_KO.txt'))
check('no model/generation',result['models_loaded']==result['generations']==0)
check('parent RSS/disk caps',result['cpu_parent_peak_rss_bytes']<=8*1024**3 and result['new_directory_bytes']<=2*1024**3)
check('two fixed cases',[(p['layer'],p['chunk']) for p in result['prepared']]==[(18,1),(24,6)])
fixtures={};packet_fields=0
for item in result['prepared']:
    path=Path(item['path']);check('prepared file '+path.name,sha(path)==item['sha256'] and path.stat().st_size==item['bytes'])
    d=torch.load(path,map_location='cpu',weights_only=True);fixtures[(d['layer'],d['chunk'])]=d
    check('prepared original K '+path.name,bits(d['k'],torch.load(P.parent/'causal_obq'/f"capture/l{d['layer']}_write{d['chunk']}.pt",map_location='cpu',weights_only=True)['k']))
    for alg in ('R_fp32','O_offline','R_bf16'):
        ref=torch.load(P.parent/'causal_obq'/f"packets/l{d['layer']}_c{d['chunk']}_{alg}.pt",map_location='cpu',weights_only=True)
        actual=d['expected'][alg]
        check('prepared packet '+path.name+'/'+alg,all(bits(a,ref[k]) for a,k in zip(actual,('packed','scale','sentinel'))))
        check('actual packet byte count '+path.name+'/'+alg,sum(t.numel()*t.element_size() for t in actual)==4515841)
        packet_fields+=3
    check('resident input shapes '+path.name,tuple(d['z'].shape)==(3,98,16,12,128) and tuple(d['s'].shape)==(3,98,1,12,128) and tuple(d['u'].shape)==(12,128,128))
expected=[(18,1,'eager'),(18,1,'graph'),(24,6,'eager'),(24,6,'graph')]
check('four expected workers',[(w['layer'],w['chunk'],w['mode']) for w in result['workers']]==expected)
rows=[];pair_count=0;algorithm_count=0;memory=[]
for w in result['workers']:
    x=read(w['path']);name=Path(w['path']).name;case=(w['layer'],w['chunk']);d=fixtures[case]
    check('worker file '+name,sha(w['path'])==w['sha256'] and w['status']==x['status']=='completed' and w['exit_code']==0)
    item=next(p for p in result['prepared'] if (p['layer'],p['chunk'])==case)
    check('worker source '+name,x['prepared_sha256']==item['sha256'])
    check('worker GPU assertions '+name,x['input_bytes_preserved'] and all(v=='all_packet_bytes_equal' for v in x['eager_code_validation'].values()) and set(x['eager_code_validation'])=={'R_fp32','R_bf16','O_offline'})
    check('worker cap '+name,x['max_rss_bytes']<=8*1024**3 and not x['last_stage_gpu_cap_exceeded'] and x['last_stage_peak_allocated_bytes']<=4*1024**3)
    s=x['stages']['T_code'];enc=x['stages']['T_encoder'];p=enc['preprocess']
    check('stage validity '+name,s['status']=='completed' and s['correctness_all_packet_bytes'] and enc['status']=='unsupported_numerical_equivalence')
    check('unsupported preprocessing count '+name,p['rotated_unequal_elements']>0 and p['scale_unequal_elements']>0 and not p['rotated_bytes_equal'] and not p['scale_bytes_equal'])
    check('twenty paired rounds '+name,len(s['samples'])==20)
    check('GPU stage peak cap '+name,s['peak_allocated_bytes']<=4*1024**3)
    check('timed copy bytes '+name,s['setup']['replay_input_copy_bytes']==(15353856 if w['mode']=='graph' else 0))
    for n,r in enumerate(s['samples']):
        check('AB BA '+name+'/'+str(n),r['round']==n and r['order']==(['R_fp32','O_offline'] if n%2==0 else ['O_offline','R_fp32']))
        check('positive finite timings '+name+'/'+str(n),all(v>0 and math.isfinite(v) for t in r['times'].values() for v in t.values()))
    recalculated={}
    for kind in ('gpu_seconds','host_seconds'):
        recalculated[kind]={}
        for alg in ('R_fp32','O_offline'):
            q=qstats([r['times'][alg][kind] for r in s['samples']]);recalculated[kind][alg]=q
            check('summary '+name+'/'+alg+'/'+kind,equal_stats(q,s['summary'][alg][kind]))
        q=qstats([r['times']['O_offline'][kind]-r['times']['R_fp32'][kind] for r in s['samples']]);recalculated[kind]['paired']=q
        check('paired summary '+name+'/'+kind,equal_stats(q,s['paired_OBQ_minus_RTN'][kind]))
    pair_count+=20;algorithm_count+=40
    med=recalculated['host_seconds']['paired']['median'];ref=item['BF16_reference_consumer_chunk_seconds']
    rows.append(dict(layer=w['layer'],chunk=w['chunk'],mode=w['mode'],statistics=recalculated,reference_BF16_chunk_seconds=ref,
        per_layer_increment_percent_of_BF16_chunk=100*med/ref,conditional_30_equal_cost_layers_increment_percent=100*30*med/ref,
        encoder_preprocessing=p,graph_output_ownership='borrowed_replayed_static_buffer' if w['mode']=='graph' else 'fresh_packed_output_scale_aliases_immutable_input',
        peak_allocated_bytes=s['peak_allocated_bytes'],peak_reserved_bytes=s['peak_reserved_bytes']))
    memory.append(x['max_rss_bytes'])
check('no CUDA audit execution',not torch.cuda.is_initialized())
out={'passed':True,'audit_sha256':sha(__file__),'result_sha256':sha(P/'RESULT.json'),'checks_passed':len(checks),'locked_source_count':len(lock['sha256']),
    'CPU_packet_fields_bytes_compared':packet_fields,'GPU_reexecutions':0,'models_loaded':0,'generations':0,'paired_rounds':pair_count,'algorithm_timing_observations':algorithm_count,
    'elapsed_seconds':time.monotonic()-start,'rows':rows,'max_worker_RSS_bytes':max(memory),
    'limits':['GPU numerical assertions are read from recorded worker results, not independently rerun.','No persistent graph-output ownership in locked implementation; graph timing is borrowed-output only.','No valid full GPU encoder cost: preprocessing differs from frozen CPU bytes.','Conditional 30-layer cost extrapolation is not a whole-model benchmark.']}
with (P/'AUDIT_RESULT.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
