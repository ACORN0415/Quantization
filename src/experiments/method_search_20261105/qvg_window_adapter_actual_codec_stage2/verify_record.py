"""Read-only structural and byte-integrity audit of the stage2 stored record.

Does not import torch, a codec, a model or a GPU library. It cannot independently
recompute codec/attention numerical verdicts; it checks their recorded coverage,
consistency, source/artifact integrity and registered limits. Output is stdout or
an explicitly NEW file, never RESULT/LOCK/RUN or a packet artifact.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
EXPECTED_WINDOWS = (21, 63, 126)
EXPECTED_POINTS = {21:{7:'precompression',8:'postcompression',9:'partial_span',41:'final'},
                   63:{7:'precompression',8:'postcompression',21:'partial_span',41:'final'},
                   126:{7:'precompression',8:'postcompression',21:'intermediate_full_horizon_not_partial',41:'final'}}
EXPECTED_ADAPTER = 'cce683d4a2ec1db16dfe986d0202970d1ce3f83458427597f3145e55e8a46d09'
EXPECTED_CONFIG = {'quant_type': 'triton-nstages-kmeans-int2', 'quant_block_size': 64,
                   'num_prq_stages': 1, 'kmeans_max_iters': 2,
                   'cache_num_k_centroids': 256, 'cache_num_v_centroids': 256,
                   'asymmetric': False}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    def nonfinite(x):
        raise ValueError('Non-finite JSON constant: ' + x)
    return json.loads(Path(path).read_text(), parse_constant=nonfinite)


class Audit:
    def __init__(self):
        self.checks = []
        self.files = {}

    def check(self, name, ok, detail=None):
        row = {'name': name, 'passed': bool(ok)}
        if detail is not None:
            row['detail'] = detail
        self.checks.append(row)
        return bool(ok)

    def artifact(self, item, base=HERE):
        if not isinstance(item, dict) or not isinstance(item.get('path'), str):
            self.check('artifact_descriptor', False, item)
            return
        p = Path(item['path'])
        if not p.is_absolute():
            p = base / p
        p = p.resolve()
        if not self.check('artifact_exists:' + str(p), p.is_file()):
            return
        digest = sha(p)
        self.check('artifact_sha256:' + str(p), digest == item.get('sha256'))
        expected_size = item.get('size', item.get('bytes'))
        if expected_size is not None:
            self.check('artifact_size:' + str(p), p.stat().st_size == expected_size)
        self.files[str(p)] = {'sha256': digest, 'bytes': p.stat().st_size}


def check_success(record, audit):
    reads = record.get('reads', [])
    audit.check('630_read_records', len(reads) == 630)
    keys = [(r.get('window_frames'), r.get('block'), r.get('step')) for r in reads]
    expected = {(w, b, s) for w in EXPECTED_WINDOWS for b in range(42) for s in range(5)}
    audit.check('read_coverage_unique_exact', len(keys) == len(set(keys)) and set(keys) == expected)
    for r in reads:
        key = (r.get('window_frames'), r.get('block'), r.get('step'))
        if key not in expected:
            continue
        w, b, s = key
        end = (b + 1) * 3 * 1560
        left = max(0, end - w * 1560)
        audit.check(f'absolute_window:{key}', r.get('start_token') == left and r.get('end_token') == end)
        audit.check(f'absolute_capacity:{key}', r.get('capacity_tokens') == 126*1560)
        audit.check(f'committed_frontier:{key}', r.get('frontier_tokens') == (b//8)*24*1560)
        audit.check(f'committed_indices:{key}', r.get('global_end_index') == end and r.get('local_end_index') == end)
        for field in ('raw_k_equal', 'raw_v_equal', 'q_rope_equal', 'k_rope_equal'):
            audit.check(f'{field}:{key}', r.get(field) is True)
    compression = record.get('compressions', [])
    audit.check('five_compression_records', len(compression) == 5)
    ranges = [(r.get('start_frame'), r.get('end_frame')) for r in compression]
    audit.check('original_24frame_group_schedule', ranges == [(x, x + 24) for x in (0,24,48,72,96)])
    serials = [r.get('call_serial') for r in compression]
    audit.check('compress_call_serials', serials == list(range(1,6)))
    for r in compression:
        label = str((r.get('start_frame'), r.get('end_frame')))
        audit.check('same_encoder_input_all_windows:' + label, r.get('source_matches_windows') == list(EXPECTED_WINDOWS))
        audit.check('actual_encoder_arguments_preserved:' + label, r.get('codec_arguments_unchanged') is True)
        for field in ('source_sha256','actual_codec_argument_sha256'):
            values=r.get(field,[])
            audit.check(field+':'+label,isinstance(values,list) and len(values)==2 and all(isinstance(x,str) and len(x)==64 for x in values))
        manifest=r.get('tensor_manifest',[])
        names=[x.get('name') for x in manifest]
        audit.check('packet_manifest_unique:'+label,len(names)==len(set(names)))
        by_name={x.get('name'):x for x in manifest}
        for side in (0,1):
            for suffix,dtype,shape in (('residual_quant','torch.uint8',[1,12,24*1560,32]),
                                       ('scales','torch.float8_e4m3fn',[1,12,24*1560,2])):
                item=by_name.get(f'/{side}/{suffix}',{})
                audit.check(f'packet_native_format:{label}:{side}:{suffix}',item.get('dtype')==dtype and item.get('shape')==shape and item.get('logical_bytes')==math.prod(shape))
        decode = r.get('independent_decode_equal', {})
        for side in ('k', 'v'):
            audit.check('independent_CPU_decode:' + label + ':' + side, decode.get(side) is True)
        before, after = r.get('packet_before_sha256'), r.get('packet_after_sha256')
        audit.check('packet_immutable:' + label, isinstance(before, str) and len(before) == 64 and before == after)
        audit.artifact(r.get('artifact'))
    native = record.get('attention_records', [])
    audit.check('12_native_endpoint_records', len(native) == 12)
    native_keys = [(r.get('window_frames'), r.get('block'), r.get('step'), r.get('label')) for r in native]
    expected_native = {(w,b,4,label) for w, points in EXPECTED_POINTS.items() for b,label in points.items()}
    audit.check('exact_native_endpoint_coverage', len(native_keys) == len(set(native_keys)) and set(native_keys) == expected_native)
    for r in native:
        label = str((r.get('window_frames'), r.get('block'), r.get('step'), r.get('label')))
        audit.check('native_repeat_exact:' + label, r.get('repeat_equal') is True)
        audit.check('native_adapter_reference_exact:' + label, r.get('adapter_reference_equal') is True)
    counters = record.get('counters', {})
    audit.check('exact_five_original_compressor_calls', counters.get('compress') == 5)
    audit.check('exact_ten_KV_PRQ_calls', counters.get('prq_tensor_encode') == 10)
    audit.check('no_SDPA_fallback', counters.get('sdpa_fallback') == 0)
    for field in ('decode_total', 'uncompress_total', 'rope_apply_total', 'rope_long_reference', 'fa2_kernel'):
        audit.check('actual_count_positive:' + field, isinstance(counters.get(field), int) and counters[field] > 0)
    audit.check('native_call_count_three_per_endpoint', counters.get('fa2_kernel') == 3*len(native) == 36)
    audit.check('original_long_RoPE_210', counters.get('rope_long_reference') == 42*5)
    expected_rope = 42*5 + sum(5*(1+min(b+1,w//3)) for w in EXPECTED_WINDOWS for b in range(42))
    audit.check('actual_segment_query_RoPE_count', counters.get('rope_apply_total') == expected_rope)
    cache_reads = record.get('cache_read_records', [])
    audit.check('actual_cache_read_trace_present', len(cache_reads) >= 2*630)
    expected_decode = 5
    for row in cache_reads:
        spans=row.get('packed_spans')
        if not audit.check('cache_read_span_count_nonnegative', isinstance(spans,int) and spans>=0):
            continue
        expected_decode += spans
    audit.check('uncompress_from_observed_span_reads', counters.get('uncompress_total') == expected_decode == record.get('expected_uncompress_calls'))
    audit.check('two_decoder_calls_per_uncompress', counters.get('decode_total') == 2*expected_decode)
    config = record.get('codec_config', {})
    for key, expected_value in EXPECTED_CONFIG.items():
        audit.check('fixed_config:' + key, config.get(key) == expected_value)
    audit.check('fixed_seed', record.get('seed') == 20261106)
    effective=record.get('effective_config',{})
    audit.check('native_effective_FP8_INT2_no_clipping',effective.get('scale_precision')=='torch.float8_e4m3fn' and effective.get('num_bits')==2 and effective.get('use_percentile_clipping') is False)
    rope=record.get('rope_input',{})
    audit.check('original_model_frequency_shape_dtype',rope.get('grid_sizes')==[[3,30,52]] and rope.get('freqs_shape')==[1024,64] and rope.get('freqs_dtype')=='torch.complex128' and rope.get('model_frequency_values') is True and rope.get('absolute_frame_range')==[0,126])


def audit_record(path):
    tick = time.monotonic()
    p = Path(path).resolve()
    before = sha(p)
    record = load_json(p)
    audit = Audit()
    status = record.get('status')
    audit.check('known_record_status', status in ('passed', 'failed', 'inconclusive'))
    # Successful numerical evidence is audited strictly. An unsuccessful record
    # is preserved/reported as such; missing expected rows are not fabricated.
    if status == 'passed':
        check_success(record, audit)
    else:
        audit.check('non_success_has_reason', any(record.get(k) for k in ('error', 'failure', 'reason')))
    binding = record.get('source_sha256', {})
    if isinstance(binding, dict):
        for name, item in binding.items():
            if isinstance(item, str) and len(item) == 64:
                audit.artifact({'path': name, 'sha256': item})
            elif isinstance(item, dict) and 'path' in item:
                audit.artifact(item)
    elif isinstance(binding, list):
        for item in binding:
            audit.artifact(item)
    else:
        audit.check('source_binding_present', False)
    if status == 'passed':
        audit.check('some_bound_sources_verified', len(audit.files) > 5)
        adapter_files = [v for k, v in audit.files.items() if Path(k).name == 'adapter.py']
        audit.check('frozen_CPU_stage1_adapter', len(adapter_files) == 1 and adapter_files[0]['sha256'] == EXPECTED_ADAPTER)
    if status == 'passed':
        audit.check('sources_unchanged_in_execution', record.get('source_hashes_unchanged') is True)
        audit.check('model_loads_zero', record.get('model_loads') == 0)
        audit.check('video_generations_zero', record.get('video_generations') == 0)
        audit.check('CPU_threads_cap', isinstance(record.get('CPU_threads'),int) and 0<record['CPU_threads']<=4)
        for field, cap in [('peak_RSS_bytes',8*2**30),('GPU_peak_allocated_bytes',8*2**30),('output_disk_bytes',2**30)]:
            v=record.get(field)
            audit.check(field+'_cap', isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=cap)
        audit.check('worker_wall_cap', isinstance(record.get('wall_seconds'),(int,float)) and 0<=record['wall_seconds']<=600)
        started=record.get('started_utc',float('inf'));finished=record.get('completed_utc',float('-inf'));deadline=record.get('absolute_deadline_utc',float('-inf'))
        audit.check('worker_clock_consistency', started<=finished<=deadline and math.isclose(record.get('wall_seconds',-1),finished-started,abs_tol=.01))
        native_bind=record.get('binding',{})
        expected_modules={'cache_class_module':'quant_videogen.kv_cache','compress_module':'quant_videogen.compress',
                          'decoder_module':'quant_videogen.functions','fa2_function_module':'flash_attn.flash_attn_interface'}
        for key, val in expected_modules.items():audit.check('native_binding:'+key,native_bind.get(key)==val)
        basenames={Path(k).name for k in audit.files}
        required_names={'compress.py','uncompress.py','functions.py','kv_cache.py','prq.py','quant_pack.py','accumulate.py','kmeans_euclid.py','causal_model.py','attention.py','flash_attn_interface.py'}
        audit.check('native_transitive_source_files_bound',required_names<=basenames)
        audit.check('native_flash_extension_bound',any('flash_attn' in Path(k).name and Path(k).suffix=='.so' for k in audit.files))
        for name in ('SUPERVISOR.json','ATTEMPT_CLAIM.json'):
            audit.check('required_sidecar:'+name,(p.parent/name).is_file())
        if (p.parent/'SUPERVISOR.json').is_file() and (p.parent/'ATTEMPT_CLAIM.json').is_file():
            sup=load_json(p.parent/'SUPERVISOR.json');claim=load_json(p.parent/'ATTEMPT_CLAIM.json')
            audit.artifact({'path':str(p.parent/'SUPERVISOR.json'),'sha256':sha(p.parent/'SUPERVISOR.json')})
            audit.artifact({'path':str(p.parent/'ATTEMPT_CLAIM.json'),'sha256':sha(p.parent/'ATTEMPT_CLAIM.json')})
            audit.check('supervisor_completed',sup.get('status')=='completed')
            audit.check('worker_exit_zero',sup.get('worker_exit_code')==0)
            audit.check('supervisor_binds_result',sup.get('result_sha256')==before)
            audit.check('one_registered_attempt',claim.get('attempt')==1)
            audit.check('same_absolute_deadline',claim.get('absolute_deadline_utc')==deadline==sup.get('absolute_deadline_utc'))
            audit.check('total_600s_start_includes_supervisor',claim.get('started_utc')==sup.get('started_utc') and math.isclose(deadline-claim.get('started_utc',0),600,abs_tol=1e-6) and sup.get('started_utc',float('inf'))<=started)
            audit.check('supervisor_wall_cap',0<=sup.get('wall_seconds',float('inf'))<=600 and sup.get('completed_utc',float('inf'))<=deadline)
            audit.check('process_group_RSS_cap',0<=sup.get('peak_group_RSS_bytes',float('inf'))<=8*2**30)
            audit.check('owned_process_cleanup',sup.get('owned_cleanup_verified') is True and sup.get('remaining_group_members')=={})
        lock_path=p.parent/'LOCK.json'
        audit.check('execution_lock_present',lock_path.is_file())
        if lock_path.is_file():
            lock=load_json(lock_path)
            audit.check('root_one_probe_lock_status',lock.get('status')=='approved_for_one_GPU_probe')
            locked=lock.get('sha256',{})
            audit.check('observed_native_sources_equal_locked_sources',isinstance(binding,dict) and all(locked.get(k)==v for k,v in binding.items()))
            audit.artifact({'path':str(lock_path),'sha256':sha(lock_path)})
        actual_disk=sum(f.stat().st_size for f in p.parent.rglob('*') if f.is_file())
        audit.check('current_output_directory_disk_cap',actual_disk<=2**30,actual_disk)
    audit.check('record_unchanged_during_audit', sha(p) == before)
    return {'status': 'passed' if all(c['passed'] for c in audit.checks) else 'failed',
            'record_status': status, 'record_path': str(p), 'record_sha256': before,
            'created_utc': time.time(), 'audit_wall_seconds': time.monotonic() - tick,
            'checks_total': len(audit.checks), 'checks': audit.checks, 'files': audit.files,
            'scope': 'Read-only stored JSON coverage and source/artifact byte-integrity audit. No independent CPU reconstruction of saved tensor packets, no decoder/attention numerical recomputation, no torch/GPU/model import. A passed audit is not a successful experiment if record_status is not passed.',
            'auditor_source_sha256': sha(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    result = audit_record(args.record)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output is None:
        sys.stdout.write(text)
    else:
        out = args.output.resolve()
        protected = {args.record.resolve(), HERE/'RESULT.json', HERE/'LOCK.json', HERE/'RUN.json', Path(__file__).resolve()}
        if out in protected or out.suffix != '.json':
            raise ValueError('Output must be a new audit JSON, never source or original record')
        with out.open('x') as f:
            f.write(text)
        print(json.dumps({'status':result['status'], 'record_status':result['record_status'], 'checks_total':result['checks_total'], 'output':str(out)}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
