"""Read-only independent byte/record audit; never deserialize model/dataset payloads."""
import collections
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import signal
import stat
import sys
import time
import zipfile
import zlib

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / 'ROOT_SPEC.json').read_text())
START = time.time()
DEADLINE = SPEC['Codex_deadline_epoch']
CHECKS = []
HASHES = {}
BYTES_READ = 0

def guard():
    if time.time() >= DEADLINE:
        raise TimeoutError('original audit deadline')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > SPEC['limits']['RSS_bytes']:
        raise MemoryError('audit RSS cap')

def check(name, ok, detail=None):
    guard()
    CHECKS.append({'name': name, 'passed': bool(ok), 'detail': detail})
    if not ok:
        raise AssertionError(name + ': ' + repr(detail))

def digest(path):
    global BYTES_READ
    path = Path(path).resolve()
    if str(path) in HASHES:
        return HASHES[str(path)]['sha256']
    h = hashlib.sha256(); n = 0
    with path.open('rb') as src:
        while True:
            guard(); b = src.read(1024 * 1024)
            if not b: break
            h.update(b); n += len(b); BYTES_READ += len(b)
    HASHES[str(path)] = {'sha256': h.hexdigest(), 'bytes': n}
    return h.hexdigest()

def read_json(path):
    path = Path(path)
    digest(path)
    return json.loads(path.read_text())

def tree_files(path):
    result = []
    for base, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            p = Path(base) / name
            check('filesystem_not_symlink:' + str(p), not p.is_symlink())
        for name in files:
            p = Path(base) / name
            check('regular_file:' + str(p), stat.S_ISREG(p.stat().st_mode))
            result.append(p)
    return result

def run():
    global BYTES_READ
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (SPEC['limits']['RSS_bytes'], SPEC['limits']['RSS_bytes']))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('audit alarm')))
    signal.setitimer(signal.ITIMER_REAL, max(0.01, DEADLINE - time.time()))
    src = Path(SPEC['input_folder']); root = Path(SPEC['asset_root'])
    manifest_path = src / 'APPROVED_ACQUISITION.json'
    approved = read_json(manifest_path)
    result = read_json(src / 'RESULT.json')
    claim = read_json(src / 'RUN.json')
    final = read_json(src / 'ROOT_EXECUTION_FINAL.json')
    started = read_json(src / 'ROOT_EXECUTION_STARTED.json')
    acquisition_spec = read_json(src / 'ROOT_SPEC.json')
    metadata = read_json(acquisition_spec['inputs'])
    ready = read_json(src / 'READY.json')
    checksums = {src / 'acquire_assets.py': approved['source_sha256'], src / 'ROOT_SPEC.json': approved['root_spec_sha256'], Path(acquisition_spec['inputs']): approved['metadata_result_sha256']}
    checksums.update({Path(k): v for k, v in approved['review_evidence_sha256'].items()})
    checksums.update({src / k: v for k, v in ready['files'].items()})
    for path, expected in checksums.items():
        check('source_or_review_hash:' + str(path), digest(path) == expected)
    check('approved_status', approved['status'] == 'approved_for_one_dino_wm_asset_acquisition')
    check('approval_flags', approved['root_execution_authorized'] is True and approved['actual_claude_source_review_completed'] is True)
    check('worker_terminal', result['status'] == 'completed_verified_selected_assets_not_model_execution')
    check('root_terminal', final['status'] == 'completed' and final['exit_code'] == 0 and final['exception'] is None)
    check('one_recorded_attempt', claim['attempt'] == started['attempt'] == final['ROOT_attempts'] == 1)
    check('exact_run_claim_file', [p.name for p in src.glob('RUN*')] == ['RUN.json'])
    check('claim_manifest_binding', claim['manifest_sha256'] == result['manifest_sha256'] == digest(manifest_path))
    check('worker_source_binding', result['source_sha256'] == approved['source_sha256'])
    check('destination_binding', claim['destination'] == result['destination'] == approved['destination'] == str(root))
    limits = approved['limits']
    check('limits_unchanged', limits == acquisition_spec['planned_future_execution_limits_not_authorized'])
    check('approval_precedes_execution', approved['approved_epoch'] <= started['started_epoch'])
    check('clock_order', started['started_epoch'] <= claim['started_utc'] == result['started_utc'] <= result['finished_utc'] <= final['finished_epoch'])
    check('root_clock_same', final['started_epoch'] == started['started_epoch'])
    root_wall = final['finished_epoch'] - final['started_epoch']
    check('root_wall_arithmetic', abs(root_wall - final['wall_seconds']) <= 0.001, {'recomputed': root_wall, 'recorded': final['wall_seconds']})
    check('root_time_cap', root_wall <= limits['wall_seconds'] and final['finished_epoch'] <= started['absolute_deadline_epoch'])
    check('deadlines', abs(started['absolute_deadline_epoch'] - started['started_epoch'] - limits['wall_seconds']) < 1e-5 and abs(claim['deadline_utc'] - claim['started_utc'] - limits['wall_seconds']) < 1e-5 and result['deadline_utc'] == claim['deadline_utc'])
    check('worker_time_cap', result['finished_utc'] <= result['deadline_utc'])
    check('worker_recorded_RSS_cap', 0 <= result['peak_RSS_bytes'] <= limits['RSS_bytes'])
    check('recorded_zero_prohibited_calls', result['GPU_calls'] == result['payload_deserializations'] == result['installations'] == final['GPU_calls'] == 0)
    metadata_assets = {a['path']: a for a in metadata['archives']}
    check('revision', approved['revision'] == metadata['HF_metadata']['revision'])
    check('exact_two_archives', len(approved['archives']) == len(result['archives']) == 2 and {a['path'] for a in approved['archives']} == set(metadata_assets))
    expected_members = [m for a in approved['archives'] for m in a['members']]
    check('selectors_31_unique', len(expected_members) == len(set(expected_members)) == 31)
    # Independent semantic selector list, not merely trusting the approved record.
    known = ['outputs/pusht/hydra.yaml', 'outputs/pusht/checkpoints/model_latest.pth']
    known += [f'pusht_noise/val/{n}' for n in ('states.pth', 'rel_actions.pth', 'velocities.pth', 'seq_lengths.pkl')]
    known += [f'pusht_noise/val/obses/episode_{i:03d}.mp4' for i in range(21)]
    known += [f'pusht_noise/train/{n}' for n in ('states.pth', 'rel_actions.pth', 'velocities.pth', 'seq_lengths.pkl')]
    check('exact_semantic_selectors', set(expected_members) == set(known))
    archive_rows = []; member_rows = []; total_archive = total_extracted = 0
    for expected, recorded in zip(approved['archives'], result['archives']):
        md = metadata_assets[expected['path']]['HF_file_metadata']
        check('LFS_size_SHA:' + expected['path'], expected['size'] == md['size'] == md['lfs']['size'] and expected['sha256'] == md['lfs']['sha256'])
        expected_url = f"https://huggingface.co/datasets/{metadata['HF_metadata']['repo']}/resolve/{approved['revision']}/{expected['path']}"
        check('pinned_url:' + expected['path'], expected['url'] == expected_url)
        archive = root / 'archives' / PurePosixPath(expected['path']).name
        check('archive_path:' + expected['path'], recorded['path'] == str(archive) and not archive.is_symlink())
        check('archive_size:' + expected['path'], archive.stat().st_size == expected['size'] == recorded['bytes'])
        check('archive_SHA:' + expected['path'], digest(archive) == expected['sha256'] == recorded['sha256'])
        total_archive += archive.stat().st_size
        archive_rows.append({'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive)})
        recorded_members = {m['member']: m for m in recorded['selected_members']}
        check('recorded_members:' + expected['path'], len(recorded_members) == len(recorded['selected_members']) == len(expected['members']) and set(recorded_members) == set(expected['members']))
        with zipfile.ZipFile(archive, 'r') as zf:
            all_names = [i.filename for i in zf.infolist()]
            check('zip_unique_names:' + expected['path'], len(all_names) == len(set(all_names)))
            check('central_count:' + expected['path'], len(all_names) == metadata_assets[expected['path']]['member_count'])
            for name in expected['members']:
                item = recorded_members[name]; info = zf.getinfo(name); dest = root / 'extracted' / name
                check('safe_member:' + name, not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts and not info.is_dir() and not info.flag_bits & 1 and stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG))
                check('member_destination:' + name, item['path'] == str(dest) and not dest.is_symlink() and dest.resolve().is_relative_to((root / 'extracted').resolve()))
                check('member_size:' + name, dest.stat().st_size == info.file_size == item['bytes'])
                h_file = hashlib.sha256(); h_zip = hashlib.sha256(); c_file = c_zip = n = 0
                with dest.open('rb') as extracted, zf.open(info, 'r') as member:
                    while True:
                        guard(); a = extracted.read(1024 * 1024); b = member.read(1024 * 1024)
                        check('direct_member_bytes:' + name + ':' + str(n), a == b)
                        if not a: break
                        n += len(a); BYTES_READ += len(a) + len(b)
                        h_file.update(a); h_zip.update(b); c_file = zlib.crc32(a, c_file); c_zip = zlib.crc32(b, c_zip)
                check('member_SHA:' + name, h_file.hexdigest() == h_zip.hexdigest() == item['sha256'])
                check('member_CRC:' + name, f'{c_file:08x}' == f'{c_zip:08x}' == f'{info.CRC:08x}' == item['CRC32'])
                check('member_count:' + name, n == item['bytes'])
                HASHES[str(dest.resolve())] = {'sha256': h_file.hexdigest(), 'bytes': n}
                member_rows.append({'member': name, 'bytes': n, 'sha256': h_file.hexdigest(), 'CRC32': f'{c_file:08x}', 'direct_zip_extracted_byte_equal': True})
                total_extracted += n
    actual_files = tree_files(root)
    expected_paths = {root / 'archives' / PurePosixPath(a['path']).name for a in approved['archives']}
    expected_paths |= {root / 'extracted' / name for name in expected_members}
    check('no_unrequested_files_or_parts', set(actual_files) == expected_paths, {'actual': len(actual_files), 'expected': len(expected_paths)})
    check('31_extracted_files', len([p for p in actual_files if p.is_relative_to(root / 'extracted')]) == 31)
    actual_bytes = sum(p.stat().st_size for p in actual_files)
    check('filesystem_payload_sum', actual_bytes == total_archive + total_extracted)
    check('worker_extracted_sum', total_extracted == result['extracted_member_bytes'])
    events = result['HTTP_events']
    check('recorded_HTTP_sum', sum(x['body_bytes'] for x in events) == result['HTTP_body_bytes'] == total_archive)
    check('recorded_HTTP_successes', sorted((x['body_bytes'] for x in events if x['status'] == 200)) == sorted(a['size'] for a in approved['archives']))
    check('recorded_HTTP_four_hops', len(events) == 4 and [x['status'] for x in events] == [302, 200, 302, 200] and all(x['body_bytes'] == 0 for x in events if x['status'] == 302))
    check('bytes_within_limits', total_archive <= limits['HTTP_body_bytes'] and total_extracted <= limits['extracted_member_bytes'] and actual_bytes <= limits['disk_bytes'])
    source_dir_bytes = sum(p.stat().st_size for p in tree_files(src))
    check('recorded_disk_plausible', actual_bytes <= result['disk_bytes_observed'] <= limits['disk_bytes'])
    check('current_combined_disk_cap', actual_bytes + source_dir_bytes <= limits['disk_bytes'])
    check('audit_no_prohibited_imports', not any(x in sys.modules for x in ('torch', 'pickle', 'numpy', 'requests')))
    return {'archives': archive_rows, 'members': member_rows, 'totals': {'archive_bytes_independent': total_archive, 'extracted_bytes_independent': total_extracted, 'asset_tree_bytes_independent': actual_bytes, 'source_record_tree_bytes_now': source_dir_bytes, 'worker_recorded_combined_disk_bytes': result['disk_bytes_observed'], 'worker_recorded_disk_minus_payload': result['disk_bytes_observed'] - actual_bytes, 'root_wall_recomputed_seconds': root_wall, 'worker_wall_recomputed_seconds': result['finished_utc'] - result['started_utc'], 'worker_recorded_peak_RSS_bytes': result['peak_RSS_bytes']}, 'scope': {'independent': 'Local full ZIP SHA256 plus selected member direct bytes/SHA256/CRC32 versus extracted files; names/counts/sizes and stored scalar arithmetic.', 'record_only': 'HTTP traffic/timing/RSS/attempt count/zero prohibited calls and temporal order of verification are worker/root records plus source review, not independent runtime instrumentation.', 'not_checked': 'Unselected ZIP members decompression/CRC, payload semantic safety, tensor validity, model loading, DINO dependencies, planning quality or runtime readiness.'}}

if __name__ == '__main__':
    out = {'status': 'failed', 'audit_started_epoch': START, 'original_audit_deadline_epoch': DEADLINE}
    try:
        out.update(run()); out['status'] = 'passed'
    except BaseException as exc:
        out['error'] = repr(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        out.update(checks=CHECKS, checks_count=len(CHECKS), checked_file_hashes=HASHES, independently_streamed_bytes=BYTES_READ, audit_finished_epoch=time.time(), audit_wall_seconds=time.time() - START, audit_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), GPU_calls=0, model_tensor_deserializations=0, network_calls=0, installations=0)
        with (HERE / 'AUDIT_CODEX.json').open('x') as f:
            json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
        print(json.dumps({'status': out['status'], 'checks': len(CHECKS), 'error': out.get('error'), 'audit_wall_seconds': out['audit_wall_seconds']}, ensure_ascii=False))
    raise SystemExit(0 if out['status'] == 'passed' else 1)
