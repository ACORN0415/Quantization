"""Prepare immutable review/binding/LOCK files only. Never starts RUN or CUDA.

Default is a dry run. --write requires final stage2 results, actual reviewer files
and an explicit root decision JSON. Code/model fingerprints are preparation;
the supervisor rechecks them inside the complete 60-minute execution budget.
"""
import argparse,json,time
from pathlib import Path
import pilot_common as p

def exclusive(path,content):
 with path.open('xb') as out:out.write(content)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage2',type=Path,required=True)
 ap.add_argument('--root-decision',type=Path,required=True)
 ap.add_argument('--stage2-review',type=Path,required=True);ap.add_argument('--pilot-review',type=Path,required=True)
 ap.add_argument('--codex-review',type=Path,required=True)
 ap.add_argument('--write',action='store_true');args=ap.parse_args();start=time.monotonic()
 assert not any((p.HERE/name).exists() for name in ('RUN.json','LOCK.json','COUNTERS.json'))
 assert json.loads((p.HERE/'STAGE2_BINDING.json').read_text())['status']=='pending'
 root=json.loads(args.root_decision.read_text())
 p.validate_review_status(root)
 for review in (args.stage2_review,args.pilot_review,args.codex_review):assert review.is_file() and review.stat().st_size>0
 assert p.sha(args.codex_review)==root['independent_codex_review_sha256']
 for path in p.HERE.glob('*.py'):assert p.sha(path)==root['pilot_source_sha256'][str(path)],path
 source=args.stage2.resolve();result=source/'RESULT.json';quality=source/'QUALITY.json';build=source/'BUILD.json'
 r=json.loads(result.read_text());q=json.loads(quality.read_text());b=json.loads(build.read_text())
 assert r['status']=='completed' and q['status']=='completed' and q['quality_gate'] is True
 assert len(q['endpoints'])==4 and all(x['native_replay_byte_equal'] and all(x['gains'][a]>=.2 for a in ('R_bf16','R_fp32')) for x in q['endpoints'])
 binary=Path(b['module']);assert binary.is_file()
 api=source/'experiment.py';assert '\ndef gpu_frames(k,hm):' in api.read_text(), 'Canonical stage2 GPU layout helper missing'
 st=json.loads((p.HERE/'SELFTEST_V3.json').read_text());assert st['passed'] and st['GPU_calls']==0
 for filename,digest in st['source_sha256'].items():assert p.sha(filename)==digest,filename
 source_lock=source/'LOCK.json';sl=json.loads(source_lock.read_text())
 # Every transitive dependency already locked by stage2 remains bound here.
 stage_hash={}
 for filename,digest in sl['sha256'].items():
  path=Path(filename);assert p.sha(path)==digest,path;stage_hash[str(path)]=digest
 for path in (api,source/'obq_pack.cu',source_lock,result,quality,build,binary,
  p.HERE.parent/'obq_gpu_cost_stage1/benchmark.py',p.HERE.parent/'causal_obq/common.py'):
  stage_hash[str(path)]=p.sha(path)
 snapshots={'ROOT_DECISION_AT_LOCK.json':args.root_decision,'STAGE2_PRIOR_FABLE_AT_LOCK_KO.txt':args.stage2_review,
  'PILOT_PRIOR_FABLE_AT_LOCK_KO.txt':args.pilot_review,'INDEPENDENT_CODEX_AT_LOCK_KO.txt':args.codex_review,
  'REVIEW_AVAILABILITY_AT_LOCK.json':p.ROOT/'docs/reviews/20261105_METHOD_SEARCH/REVIEW_AVAILABILITY.json',
  'USER_REVIEW_PREFERENCE_AT_LOCK_KO.txt':p.ROOT/'AGENTS.md'}
 snapshot_bytes={name:path.read_bytes() for name,path in snapshots.items()}
 for name,data in snapshot_bytes.items():
  import hashlib
  stage_hash[str(p.HERE/name)]=hashlib.sha256(data).hexdigest()
 review_status={key:root[key] for key in ('root_execution_authorized','root_code_review_accepted','independent_codex_code_review_accepted',
  'root_stage2_accepted','cost_review_accepted','historical_fable_design_reviewed','review_route','review_status',
  'claude_final_source_reviewed','claude_final_result_reviewed','final_reviewer_model','claude_not_before','administrative_amendment_adopted')}
 binding={'status':'ready',**review_status,
  'api_source':str(api),'extension_binary':str(binary),'stage2_result':str(result),'quality_result':str(quality),
  'source_helper':'api.gpu_frames; api.v1.gpu_frames forbidden','sha256':stage_hash,'prepared_utc':time.time()}
 files=list(dict.fromkeys(p.required_sources()+[Path(__file__).resolve(),p.HERE/'audit_posthoc.py']))
 print(json.dumps({'action':'write_bundle' if args.write else 'dry_run','required_sources':len(files),
  'required_source_bytes':sum(x.stat().st_size for x in files),'stage2_bound_dependencies':len(stage_hash),
  'GPU_calls':0,'starts_RUN':False},indent=2),flush=True)
 if not args.write:return
 for name,data in snapshot_bytes.items():exclusive(p.HERE/name,data)
 # Preserve the original pending state rather than erasing its provenance.
 pending=p.HERE/'STAGE2_BINDING.json';exclusive(p.HERE/'STAGE2_BINDING_PENDING_SNAPSHOT.json',pending.read_bytes())
 p.save(pending,binding)
 digest={str(path):p.sha(path) for path in files}
 digest.update(stage_hash)
 digest[str(p.HERE/'STAGE2_BINDING_PENDING_SNAPSHOT.json')]=p.sha(p.HERE/'STAGE2_BINDING_PENDING_SNAPSHOT.json')
 lock={'registered_utc':time.time(),**review_status,'sha256':digest,
  'limit_seconds':3600,'generation_limit':8,'VAE_limit':8,'review_snapshots':list(snapshots),
  'binding_preparation_seconds':time.monotonic()-start,'binding_is_not_experiment_start':True}
 exclusive(p.HERE/'LOCK.json',(json.dumps(lock,indent=2)+'\n').encode())
 print(json.dumps({'LOCK':p.record(p.HERE/'LOCK.json'),'run_created':False},indent=2))
if __name__=='__main__':main()
