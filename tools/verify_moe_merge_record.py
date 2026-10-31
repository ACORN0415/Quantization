"""Read-only audit of saved merge outputs; prints JSON and never initializes CUDA.

The registered deadline applies to the recorded experiment, not to this audit.
The original audit.json and completion.json remain historical records.
"""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'src/experiments/moe_chunk_merge_gpu'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_recorded_window(reg, result, completion):
    """Check the original run's timestamps without consulting the current clock."""
    start = datetime.fromisoformat(reg['started_at_utc'].replace('Z','+00:00'))
    deadline = datetime.fromisoformat(reg['deadline_utc'].replace('Z','+00:00'))
    gpu_start = datetime.fromisoformat(result['started_at_utc'])
    gpu_end = datetime.fromisoformat(result['completed_at_utc'])
    end = datetime.fromisoformat(completion['completed_at_utc'])
    assert completion['status'] == 'complete'
    assert datetime.fromisoformat(completion['started_at_utc']) == start
    assert start <= gpu_start <= gpu_end <= end < deadline, 'Recorded experiment exceeded its time window'
    elapsed = (end-start).total_seconds()
    assert elapsed < reg['caps']['total_wall_seconds']
    assert abs(elapsed-completion['elapsed_seconds_including_preparation_execution_audit_report']) < 1e-6
    return dict(started_at_utc=start.isoformat(), completed_at_utc=end.isoformat(),
                deadline_utc=deadline.isoformat(), elapsed_seconds=elapsed, cap_met=True)


def audit_saved_record():
    reg = json.loads((RECORD/'registration.json').read_text())
    result = json.loads((RECORD/'result.json').read_text())
    calls = json.loads((RECORD/'calls.json').read_text())
    manifest = json.loads((RECORD/'raw_manifest.json').read_text())
    completion = json.loads((RECORD/'completion.json').read_text())
    recorded_window = validate_recorded_window(reg, result, completion)
    assert result['status'] == 'PASS'
    assert result['regression_code_sha256'] == digest(ROOT/'tools/reproduce_moe_chunk_merge.py')
    assert result['registration_sha256'] == digest(RECORD/'registration.json')
    assert result['source_sha256'] == reg['source']['sha256'] == digest(ROOT/reg['source']['local_path'])
    source = (ROOT/reg['source']['local_path']).read_bytes()
    assert hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest() == reg['source']['git_blob_sha']
    frozen_epoch = int(subprocess.check_output(['git','show','-s','--format=%ct','ea3feff'],cwd=ROOT))
    assert frozen_epoch < datetime.fromisoformat(result['started_at_utc']).timestamp()

    def read(name):
        info = manifest[name]
        path = RECORD/'raw'/name
        assert digest(path) == info['sha256']
        return np.frombuffer(path.read_bytes(),dtype=np.dtype(info['dtype'])).reshape(info['shape'])

    references = {}
    for case in reg['fixtures']['token_0_vectors']:
        bank = read(case+'_input_bank.bin')
        weights = read(case+'_input_weights.bin')
        ids = read(case+'_input_ids.bin')
        inverse = read(case+'_input_inverse.bin')
        assert bank.shape == (32,256) and bank.dtype == np.float16
        assert all(len(set(row)) == 8 for row in ids.tolist())
        assert np.array_equal(np.sort(inverse.ravel()),np.arange(32))
        values = bank[inverse]
        products = values.astype(np.float64) * weights.astype(np.float64)[:,:,None]
        exact64 = products.sum(axis=1)
        # Independently prove float64 sums are exact for each of this fixture's four lane families.
        for token in range(4):
            for lane in range(4):
                rational = sum((Fraction.from_float(float(values[token,s,lane])) *
                                Fraction.from_float(float(weights[token,s])) for s in range(8)),Fraction())
                assert Fraction.from_float(float(exact64[token,lane])) == rational
                assert np.all(exact64[token,lane::4] == exact64[token,lane])
        exact = exact64.astype(np.float16)

        def accumulate(chunk_tokens, canonical):
            acc = np.zeros((4,256),dtype=np.float32)
            perm = np.argsort(ids,axis=1,kind='stable') if canonical else np.tile(np.arange(8),(4,1))
            for start in range(0,32,chunk_tokens*8):
                for s in range(8):
                    slots = perm[:,s]
                    indices = inverse[np.arange(4),slots]
                    inside = (indices >= start) & (indices < start+chunk_tokens*8)
                    vals = np.where(inside[:,None],bank[indices].astype(np.float32),np.float32(0))
                    gates = weights[np.arange(4),slots].astype(np.float32)[:,None]
                    product = np.multiply(vals,gates,dtype=np.float32)
                    assert np.array_equal(product.astype(np.float64), vals.astype(np.float64)*gates.astype(np.float64))
                    np.add(acc,product,out=acc)
            return acc.astype(np.float16)

        references[case] = dict(exact=exact, slot=accumulate(4,False),
                                path={(mode,c):accumulate(4 if mode=='slot_buffer' else c,mode=='expert_order')
                                      for mode in reg['paths'] for c in [1,2,4]})

    repeat_bytes = {}
    agreement = Counter()
    for row in result['rows']:
        got = read(Path(row['raw_file']).name)
        ref = references[row['case']]
        expected = ref['path'][row['mode'],row['chunk_tokens']]
        same = lambda a,b:a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
        assert same(got,expected) and row['matches_path_cpu_reference']
        for field,target in [('matches_original_slot_reference',ref['slot']),
                             ('matches_correctly_rounded_exact_sum',ref['exact'])]:
            assert same(got,target) == row[field]
            agreement[field] += int(same(got,target))
        close = np.allclose(got.astype(np.float64),ref['slot'].astype(np.float64),atol=.002,rtol=.002)
        assert bool(close) == row['merge_only_passes_documented_allclose']
        agreement['merge_only_allclose_passes'] += int(close)
        key = row['case'],row['mode'],row['chunk_tokens']
        if key in repeat_bytes:
            assert repeat_bytes[key] == got.tobytes()
            agreement['repeat_pairs_bitwise_equal'] += 1
        else:
            repeat_bytes[key] = got.tobytes()
    assert len(result['rows']) == 36 and agreement['repeat_pairs_bitwise_equal'] == 18
    assert all(c['status']=='launched' for c in calls)
    host = Counter(c['kernel'] for c in calls)
    trace = json.loads((RECORD/'cuda_trace.json').read_text())
    assert digest(RECORD/'cuda_trace.json') == result['raw_trace_sha256']
    device = Counter(e.get('name','') for e in trace['traceEvents'] if e.get('cat')=='kernel')
    matched = {k:sum(n for name,n in device.items() if k in name) for k in host}
    assert dict(host) == matched == result['host_launch_counts'] == result['cuda_profiler_target_counts']
    merges = [c for c in calls if c['kernel']=='fused_weighted_unpermute_reduce_kernel']
    for c in merges:
        assert c['observed_chunk_end']-c['observed_chunk_start'] == c['expert_output_shape'][0]
        assert c['expert_output_dtype'] == 'torch.float16' and c['accumulator_dtype'] == 'torch.float32'
    assert len(merges) == 68 <= reg['caps']['merge_kernel_launches']
    assert result['gpu_stage_wall_seconds'] < reg['caps']['gpu_stage_wall_seconds']
    assert result['model_forwards'] == 0
    now = datetime.now(timezone.utc)
    files = ['registration.json','cpu_reference.json','result.json','calls.json','raw_manifest.json','cuda_trace.json','gpu_run.log',
             'audit.json','completion.json']
    audit = dict(status='PASS',checked_at_utc=now.isoformat(),read_only=True,
                 recorded_experiment_window=recorded_window,
                 preregistration_commit='ea3feff', source_unchanged=True, raw_files_verified=len(manifest),
                 output_arrays_verified=36, full_array_comparison=True, agreement=dict(agreement),
                 merge_launches_by_mode=dict(Counter(c['context']['mode'] for c in merges)),
                 host_launch_counts=dict(host),independent_cuda_trace_counts=matched,
                 invalid_or_incomplete_runs=0,new_gpu_calls_by_auditor=0,
                 files_sha256={f:digest(RECORD/f) for f in files},
                 auditor_sha256=digest(__file__),research_expansion=False)
    return audit


def main():
    print(json.dumps(audit_saved_record(),indent=2))


if __name__ == '__main__':
    main()
