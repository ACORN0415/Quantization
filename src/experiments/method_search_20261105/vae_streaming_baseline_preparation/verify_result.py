"""Independent stdlib-only audit of saved streaming records; no tensor loads."""
import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPECTED_LOCK = '4251016b7d044491fc031c5800fbe13cc40265ccdb76f3e4ec2460464ea6e2f5'
LIMITS = dict(wall_seconds=600, CPU_threads=4, RSS_bytes=12*1024**3,
              GPU_allocator_bytes=22*1024**3, output_bytes=256*1024**2,
              whole_video_decodes=2, decoder_forward_calls=252, generator_calls=0)
SHA_RE = re.compile(r'^[0-9a-f]{64}$')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def astsha(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def audit():
    begin = time.time()
    out = {'status': 'running', 'audit_started_utc': begin, 'checks': 0,
           'failures': [], 'hashed_files': {}, 'tensor_loads': 0,
           'GPU_calls': 0, 'model_loads': 0, 'new_decodes': 0}

    def check(ok, label):
        out['checks'] += 1
        if not ok:
            out['failures'].append(label)

    paths = {name: HERE/name for name in ('LOCK.json', 'RESULT.json', 'RUN.json',
             'ROOT_EXECUTION_STARTED.json', 'ROOT_EXECUTION_FINAL.json')}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise RuntimeError('Final records unavailable; do not poll: '+repr(missing))
    initial_hashes = {str(p): sha(p) for p in paths.values()}
    lock, result, run, start, final = (load(paths[n]) for n in paths)
    check(initial_hashes[str(paths['LOCK.json'])] == EXPECTED_LOCK, 'frozen LOCK hash')
    check(lock['status'] == 'approved_for_one_native_streaming_probe', 'LOCK status')
    check(lock['root_execution_authorized'] is True, 'LOCK authorization')
    check(lock['gate'] == 'all501_RGB_logical_bytes_equal_and_trace_contract', 'LOCK gate')
    check(lock['limits'] == result['limits'] == LIMITS, 'limits agreement')
    check(start['source_commit'] == 'b89ad1d1f3f7f1c4ac50ee64ca7e24683d61d9f1', 'pre-run commit record')
    check(run['attempt'] == start['attempt'] == final['ROOT_attempts'] == 1, 'single attempt records')
    check(run['status'] == 'started', 'exclusive RUN record')
    check(final['status'] == 'completed' and final['exit_code'] == 0, 'root terminal exit')
    check(final['worker_result_exists'] is True and final.get('exception') is None, 'root result/exception')
    check(final['wall_seconds'] <= 600 and final['wall_seconds'] > 0, 'root wall cap')
    check(abs(final['wall_seconds']-(final['finished_epoch']-final['start_epoch'])) <= .001,
          'root elapsed arithmetic')
    check(abs(start['start_epoch']-final['start_epoch']) < .001, 'root start consistency')
    check(run['started_utc'] == result['started_utc'], 'worker start consistency')
    check(run['absolute_deadline_utc'] == result['deadline_utc'], 'worker deadline consistency')
    check(result['deadline_utc']-result['started_utc'] == 600, 'worker cap definition')
    check(final['start_epoch'] <= result['started_utc'] <= result['finished_utc'] <= final['finished_epoch'],
          'worker contained within root run')
    check(result['finished_utc'] <= result['deadline_utc'], 'worker deadline respected')
    for name in ('FAILURE.json', 'WATCHDOG_FAILURE.json'):
        check(not (HERE/name).exists(), 'no '+name)
    check(result['status'] == 'completed_exact_streaming_baseline', 'worker terminal status')
    check(result['decision'] == 'exact_output_baseline_valid_resource_effect_descriptive_not_new_method',
          'registered decision')
    for field in ('source_and_input_hashes_unchanged', 'watchdog_completed_cleanly'):
        check(result[field] is True, field)
    for field in ('TF32', 'autocast', 'benchmark'):
        check(result['runtime'][field] is False, 'runtime '+field)
    check(result['runtime']['deterministic'] is True, 'deterministic record')
    check(result['runtime']['torch'] == lock['environment_packages']['torch'], 'torch version record')
    for field in ('whole_video_decodes', 'decoder_forward_calls', 'generator_calls'):
        check(result[field] == LIMITS[field], 'counter '+field)

    # Stream SHA only. Never deserialize latent/weights/RGB or execute worker.
    hashed = {}
    for path, expected in lock['sha256'].items():
        resolved = str(Path(path).resolve())
        if resolved not in hashed:
            hashed[resolved] = sha(resolved)
        check(hashed[resolved] == expected, 'LOCK source/input '+path)
        out['hashed_files'][path] = {'sha256': hashed[resolved], 'bytes': Path(path).stat().st_size}
    out['unique_hashed_source_paths'] = len(hashed)
    out['unique_hashed_source_bytes'] = sum(Path(p).stat().st_size for p in hashed)
    # Independently reconstruct the whitelisted AST transformation, without exec.
    tree = ast.parse((ROOT/'src/wan/modules/vae.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WanVAE_')
    native = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'decode')
    check(astsha(native) == result['decode_AST_contract']['original_ast_sha256'], 'native AST hash')
    stream = copy.deepcopy(native)
    loop = next(n for n in stream.body if isinstance(n, ast.For))
    branch = loop.body[1]
    oldcat = copy.deepcopy(branch.orelse[1])
    check(ast.dump(oldcat, include_attributes=False) == ast.dump(ast.parse('out = torch.cat([out, out_], 2)').body[0], include_attributes=False), 'native output cat')
    branch.body += ast.parse('_emit_slab(out, i)\ndel out').body
    branch.orelse[1:] = ast.parse('_emit_slab(out_, i)\ndel out_').body
    stream.body[-1] = ast.parse('return None').body[0]
    check(astsha(stream) == result['decode_AST_contract']['stream_ast_sha256'], 'whitelisted transformed AST hash')
    check(result['decode_AST_contract']['inverse_equals_native'] is True, 'worker AST inversion record')
    wrapper_tree = ast.parse((ROOT/'src/utils/wan_wrapper.py').read_text())
    wrapper = next(n for n in wrapper_tree.body if isinstance(n, ast.ClassDef) and n.name == 'WanVAEWrapper')
    check(astsha(wrapper) == result['wrapper_AST_sha256'], 'wrapper AST hash')

    arms = result['arms']
    check(len(arms) == 2, 'two arms')
    check([a['name'] for a in arms] == ['native_BF16', 'stream_BF16'], 'fixed arm order')
    logical_cache = []
    for arm in arms:
        name = arm['name']; trace = arm['trace']
        check(arm['fresh_checkpoint_loaded'] is True, name+' fresh model record')
        for key in ('setup_seconds', 'decode_emit_validation_seconds'):
            check(math.isfinite(arm[key]) and arm[key] > 0, name+' '+key)
        check(0 < arm['GPU_peak_allocated_bytes'] <= LIMITS['GPU_allocator_bytes'], name+' allocated cap')
        check(arm['GPU_peak_reserved_bytes'] >= arm['GPU_peak_allocated_bytes'], name+' reserved >= allocated')
        check(trace['conv2'] == [{'input_shape':[1,16,126,60,104], 'output_shape':[1,16,126,60,104], 'dtype':'torch.bfloat16'}], name+' full conv2')
        check(len(trace['decoder']) == 126, name+' decoder count')
        for i, row in enumerate(trace['decoder']):
            check(row == {'index':i, 'input_shape':[1,16,1,60,104], 'output_shape':[1,3,1 if i == 0 else 4,480,832], 'dtype':'torch.bfloat16'}, name+f' decoder {i}')
        check(sum(row['output_shape'][2] for row in trace['decoder']) == 501, name+' output frame arithmetic')
        check(trace['clear'] == [{'before_tensor_count':0,'after_tensor_count':0}, {'before_tensor_count':32,'after_tensor_count':0}], name+' clear contract')
        samples = trace['cache_samples']
        check([s['index'] for s in samples] == [0,64,125], name+' cache sample positions')
        check([s['tensor_count'] for s in samples] == [30,32,32], name+' cache counts')
        check([s['unique_storage_bytes'] for s in samples] == [920325120,1888573440,1888573440], name+' recorded storage totals')
        for s in samples:
            entries = s['entries']; groups = {e['storage_group'] for e in entries}
            check(len(entries) == s['tensor_count'], name+' entry count')
            check(len(groups) == s['unique_storage_count'], name+' observed storage group count')
            check(len({e['slot'] for e in entries}) == len(entries), name+' unique tensor slots')
            total = 0
            for e in entries:
                check(e['dtype'] == 'torch.bfloat16', name+' state BF16')
                calculated = math.prod(e['shape'])*2
                check(calculated == e['logical_bytes'], name+f" logical byte slot {e['slot']}")
                total += calculated
            check(total == s['logical_bytes'], name+' logical byte sum')
            check(len(entries)+len(s['non_tensor_slots']) == s['list_slots'] == 33, name+' slot coverage')
            logical_cache.append({'arm': name, 'index': s['index'], 'recomputed_logical_bytes': total,
                                  'recorded_unique_storage_bytes':s['unique_storage_bytes']})
        check(len(arm['frame_sha256']) == 501, name+' frame hash count')
        check(all(SHA_RE.fullmatch(s) for s in arm['frame_sha256']), name+' SHA format')
    check(arms[0]['trace'] == arms[1]['trace'], 'two arm trace identity')
    check(sum(len(a['trace']['decoder']) for a in arms) == result['decoder_forward_calls'], 'counter vs hook row sum')
    rows = result['per_frame']
    check(len(rows) == 501 and [r['frame'] for r in rows] == list(range(501)), 'exact 501 row coverage')
    for row in rows:
        i = row['frame']
        check(row['logical_bytes_equal'] is True, f'frame {i} original comparison record')
        check(row['native_sha256'] == row['stream_sha256'] == arms[0]['frame_sha256'][i] == arms[1]['frame_sha256'][i], f'frame {i} four hash records agree')
    for field in ('monitor_peak_RSS_bytes', 'process_lifetime_peak_RSS_bytes'):
        check(0 < result[field] <= LIMITS['RSS_bytes'], field+' cap')
    check(max(a['GPU_peak_allocated_bytes'] for a in arms) <= result['monitor_peak_GPU_allocator_bytes'] <= LIMITS['GPU_allocator_bytes'], 'global GPU peak and cap')
    check(0 < result['monitor_peak_output_bytes'] <= LIMITS['output_bytes'], 'recorded disk cap')
    check(math.fsum(a['setup_seconds']+a['decode_emit_validation_seconds'] for a in arms) <= result['finished_utc']-result['started_utc'], 'arm timed intervals within worker wall')

    n, s = arms
    out['derived'] = {
        'worker_wall_seconds': result['finished_utc']-result['started_utc'],
        'root_wall_seconds': final['finished_epoch']-final['start_epoch'],
        'native_peak_allocated_bytes': n['GPU_peak_allocated_bytes'],
        'stream_peak_allocated_bytes': s['GPU_peak_allocated_bytes'],
        'peak_allocated_reduction_bytes': n['GPU_peak_allocated_bytes']-s['GPU_peak_allocated_bytes'],
        'peak_allocated_reduction_percent': 100*(1-s['GPU_peak_allocated_bytes']/n['GPU_peak_allocated_bytes']),
        'native_decode_emit_validation_seconds': n['decode_emit_validation_seconds'],
        'stream_decode_emit_validation_seconds': s['decode_emit_validation_seconds'],
        'descriptive_timed_reduction_percent': 100*(1-s['decode_emit_validation_seconds']/n['decode_emit_validation_seconds']),
        'frame_RGB_FP32_logical_bytes': 3*480*832*4,
        'whole_501_RGB_logical_bytes': 501*3*480*832*4,
        'total_recorded_equal_frames': sum(r['logical_bytes_equal'] is True for r in rows),
        'cache_logical_byte_recomputation': logical_cache,
        'RSS_monitor_minus_ru_maxrss_bytes': result['monitor_peak_RSS_bytes']-result['process_lifetime_peak_RSS_bytes']}
    # Do not conflate sampled RSS with getrusage, storage with logical bytes,
    # or the saved comparison verdict with a second tensor comparison.
    out['limitations'] = [
        'Only saved scalar/hash/trace records and source files independently audited; no RGB, latent or model tensor was loaded.',
        '501 per-frame byte comparisons are original worker observations; this audit checks their coverage and four-way hash consistency, not independent RGB re-execution.',
        'Unique storage nbytes per group were not saved; only logical bytes and storage-group counts are independently recomputed.',
        'Generator zero is source-path evidence and a recorded constant, not a generator hook observation.',
        'Fresh model, allocated-zero cleanup and watchdog join are original-worker assertions plus root exit0; no process/GPU inspection was performed by auditor.',
        'Single fixed order; times include setup separately and validation/hash/D2H, with extra byte comparison in streaming. No production speedup or statistical claim.',
        'Current source/input hashes do not prove unrecorded intermediate or external library state; binary dependency hashes were not in this LOCK.',
        'Audit ran after experiment completion and is not an extra experimental decode.']
    out['record_sha256'] = initial_hashes
    for p, digest in initial_hashes.items():
        check(sha(p) == digest, 'record unchanged during audit '+p)
    out['audit_finished_utc'] = time.time()
    out['audit_wall_seconds'] = out['audit_finished_utc']-begin
    out['audit_peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    check(out['audit_wall_seconds'] <= 900, 'post-run audit time cap')
    check(out['audit_peak_RSS_bytes'] <= 2*1024**3, 'post-run audit RSS cap')
    out['auditor_source_sha256'] = sha(__file__)
    out['status'] = 'passed_saved_record_audit' if not out['failures'] else 'failed_saved_record_audit'
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(HERE/'AUDIT_CODEX.json'))
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    if Path(args.output).exists():
        raise FileExistsError('Do not overwrite an existing audit: '+args.output)
    result = audit()
    data = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if len(data.encode()) > 8*1024**2:
        raise RuntimeError('Audit output cap')
    with Path(args.output).open('x') as f:
        f.write(data)
    print(json.dumps({k: result[k] for k in ('status','checks','failures','audit_wall_seconds','audit_peak_RSS_bytes','derived')}, ensure_ascii=False))
    sys.exit(0 if not result['failures'] else 1)
