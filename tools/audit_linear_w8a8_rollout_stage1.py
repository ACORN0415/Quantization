"""Read-only CPU verification of the stopped rollout and post-hoc chunk errors."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
import torch
from audit_linear_w8a8_fused_stage1 import sha,eq
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'src/experiments/linear_w8a8_rollout_stage1'


def main():
    torch.set_num_threads(4)
    r=json.loads((OUT/'results.json').read_text());assert r['status']=='stopped_latent_gate'
    for f,k in [('tools/linear_w8a8_rollout_stage1.py','code_sha256'),('tools/linear_w8a8_fused_stage1.py','helper_sha256'),('tools/kv_reuse_stage1.py','native_loader_sha256'),('tools/kv_reuse_core.py','reuse_core_sha256'),('src/experiments/linear_w8a8_rollout_stage1/REGISTRATION_KO.txt','registration_sha256')]:assert sha(ROOT/f)==r[k]
    assert r['model_forward_started']==r['model_forward_finished']==120 and len(r['runs'])==2
    assert r['finished_utc']<r['deadline_utc']
    values=[]
    for i,row in enumerate(r['runs']):
        assert row['name']==('warm_B','warm_Q')[i] and row['status']=='valid'
        assert row['schedule']==[(j//5)*4680 for j in range(60)]
        assert row['counts']['decode_calls']==1200 and row['counts']['read_calls']==3660
        e=row['latent'];assert sha(e['path'])==e['sha256'];v=torch.load(e['path'],map_location='cpu',weights_only=True)['latent'];assert torch.isfinite(v).all();values.append(v)
        e=row['telemetry'];assert sha(e['path'])==e['sha256'];tele=json.loads(Path(e['path']).read_text())
        assert tele and not any(x['external_pids'] for x in tele)
        assert max(x['process_bytes'] for x in tele)==row['process_sample_peak']
    b,q=values;assert b.shape==q.shape and b.dtype==q.dtype
    rel=float((q.float()-b.float()).double().norm()/b.double().norm());assert abs(rel-r['runs'][1]['latent_relative_l2_to_B'])<1e-12 and rel>.05
    old=Path('/home/acorn0415/sfq_data/kv_reuse_stage1/gate_U/latent.pt')
    assert old.exists();previous=torch.load(old,map_location='cpu',weights_only=True)['latent'];assert eq(b,previous)
    per_chunk=[]
    for i in range(12):
        bs=b[:,3*i:3*(i+1)];qs=q[:,3*i:3*(i+1)]
        per_chunk.append(float((qs.float()-bs.float()).double().norm()/bs.double().norm()))
    print(json.dumps({'status':'passed','gpu_calls':0,'forward_ledger':120,'valid_timed_pairs':0,
                      'baseline_all_tensor_bytes_equal_to_previous_gate_U':True,'previous_baseline_path':str(old),
                      'overall_latent_relative_l2':rel,'per_chunk_relative_l2':per_chunk,
                      'posthoc_scope':'chunk errors describe saved outputs; not a causal attribution or image-quality metric',
                      'original_decision_preserved':r['decision']},indent=2))


if __name__=='__main__':main()
