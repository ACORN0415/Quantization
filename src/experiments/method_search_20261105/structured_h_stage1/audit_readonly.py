"""One read-only audit of frozen outputs; does not rerun quantization or attention."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,math,statistics,time
from pathlib import Path
import torch
torch.set_num_threads(4)
HERE=Path(__file__).resolve().parent;CAUSAL=HERE.parent/'causal_obq'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def payload(p):return sum(v.numel()*v.element_size() for v in p.values() if isinstance(v,torch.Tensor))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
lock=json.loads((HERE/'LOCK.json').read_text());r=json.loads((HERE/'RESULT.json').read_text())
assert r['status']=='completed' and r['elapsed_seconds']<600 and r['max_rss_bytes']<4*1024**3
assert len(r['rows'])==4 and len(r['timing'])==72 and len(r['stats'])==6 and len(r['artifacts'])==24
protected={Path(p):v for p,v in lock['sha256'].items()}
protected.update({Path(a['path']):a['sha256'] for a in r['artifacts']})
protected.update({HERE/n:sha(HERE/n) for n in ('RESULT.json','LOCK.json')})
for p,h in protected.items():assert sha(p)==h,str(p)
for a in r['artifacts']:assert Path(a['path']).stat().st_size==a['bytes']
stats={}
for row in r['stats']:
 assert len(row['seconds'])==3;close(statistics.median(row['seconds']),row['median'])
 stats[(row['layer'],row['arm'])]=row['median']
for row in r['timing']:
 assert len(row['samples'])==3 and row['payload_bytes']==4515841
 for key,value in row['medians'].items():close(value,statistics.median(x[key] for x in row['samples']))
packet_checks=[]
for layer in (18,24):
 for chunk in range(1,7):
  originals={a:torch.load(CAUSAL/f'packets/l{layer}_c{chunk}_{a}.pt',map_location='cpu',weights_only=True) for a in ('R_bf16','R_fp32','O_offline')}
  for p in originals.values():assert payload(p)==4515841
  base=originals['O_offline']
  assert all(torch.equal(v['scale'],base['scale']) for v in originals.values())
  v=torch.load(CAUSAL/f'packets/l{layer}_c{chunk}_V.pt',map_location='cpu',weights_only=True)
  assert payload(v)==4500480
  for arm in ('D_MEAN','C_MEAN'):
   path=HERE/f'l{layer}_c{chunk}_{arm}.pt';p=torch.load(path,map_location='cpu',weights_only=True)
   assert p['kind']=='K' and tuple(p['shape'])==(3,98,16,12,128)
   assert p['packed'].dtype==torch.uint8 and p['packed'].numel()*2==math.prod(p['shape'])
   assert p['scale'].dtype==torch.float16 and torch.equal(p['scale'],base['scale'])
   assert torch.equal(p['sentinel'],base['sentinel']) and payload(p)==4515841
   packet_checks.append({'layer':layer,'chunk':chunk,'arm':arm,'payload_bytes':payload(p),'scale_bytes_equal_full':True})
decisions={}
for arm in ('D_MEAN','C_MEAN'):
 tests=[];signal=True;worse=False
 for row in r['rows']:
  a=row['arms'];assert set(a)=={'R_bf16','R_fp32','FULL','D_MEAN','C_MEAN'}
  for v in a.values():
   assert v['payload_bytes']==54097926;close(v['nse'],v['sse']/row['reference_energy'])
  saved=next(g for g in r['decisions'][arm]['gains'] if g['layer']==row['layer'] and g['call']==row['call'])
  for baseline in ('R_bf16','R_fp32'):
   gain=(a[baseline]['sse']-a[arm]['sse'])/a[baseline]['sse']
   denominator=a[baseline]['sse']-a['FULL']['sse'];recovery=(a[baseline]['sse']-a[arm]['sse'])/denominator if denominator>0 else None
   close(gain,saved['gain'][baseline]);close(recovery,saved['full_gain_recovery'][baseline])
   passed=gain>=.2 and recovery is not None and recovery>=.8;signal &= passed;worse |= gain<=-.1
   tests.append({'layer':row['layer'],'call':row['call'],'baseline':baseline,'gain':gain,'recovery':recovery,'passed':passed})
 label='signal' if signal else ('stop_arm' if worse else 'undecided')
 assert label==r['decisions'][arm]['output_decision']
 costs=[]
 for layer in (18,24):
  total=lambda a:sum(t['medians']['end_to_end'] for t in r['timing'] if t['layer']==layer and t['arm']==a)
  fulls={a:total(a)+stats[(layer,'FULL')] for a in ('FULL','FULL_BATCH')}
  full=min(fulls.values());candidate=total(arm)+stats[(layer,arm)]
  s=next(x for x in r['decisions'][arm]['costs'] if x['layer']==layer)
  close(candidate,s['candidate_seconds']);close(full,s['full_seconds']);close(1-candidate/full,s['relative_reduction'])
  assert min(fulls,key=fulls.get)==s['chosen_full_implementation']
  costs.append(candidate<full)
 assert all(costs)==r['decisions'][arm]['cpu_cost_reduced_both_layers']
 decisions[arm]={'decision':label,'endpoint_baseline_checks':tests,'cost_reduced_both_layers':all(costs)}
assert r['preferred'] is None and all(d['decision']=='undecided' for d in decisions.values())
for p,h in protected.items():assert sha(p)==h,str(p)
numeric={a:{'codes':sum(x['codes'] for x in r['numeric_deltas'] if x['arm']==a),
 'different_codes':sum(x['different_codes'] for x in r['numeric_deltas'] if x['arm']==a),
 'max_abs_difference':max(x['max_abs_code_difference'] for x in r['numeric_deltas'] if x['arm']==a)} for a in ('D_MEAN','C_MEAN')}
out={'passed':True,'auditor_sha256':sha(__file__),'result_sha256':sha(HERE/'RESULT.json'),
 'protected_files_hashed_before_and_after':len(protected),'locked_source_files':len(lock['sha256']),
 'new_packet_files':len(packet_checks),'existing_packet_files_inspected':48,'new_packet_checks':packet_checks,
 'decisions':decisions,'numeric_differences_from_execution_record':numeric,
 'disk_bytes_before_audit_output':sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
 'time_utc':time.time(),'GPU_calls':0,
 'scope':'SHA and sizes; actual packet dtype/shape/payload/scales; medians and decision arithmetic. No H/encoder/attention SSE rerun. Runtime code assertions are execution evidence, not a second independent effect measurement.'}
print(json.dumps(out,indent=2))
