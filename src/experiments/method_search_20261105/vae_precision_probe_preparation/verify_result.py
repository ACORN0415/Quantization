"""Post-run CPU audit: saved scalar aggregation and eight saved RGB frames only."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                  OPENBLAS_NUM_THREADS='4')
HERE = Path(__file__).resolve().parent
SELECTED = (0, 80, 160, 240, 320, 400, 480, 500)
# Numerical comparison policy: FP64 summation-order roundoff, not a quality gate.
# Bounds apply identically to all fields, and are fixed before selected RGB loads.
ABS_TOL, REL_TOL = 2e-14, 2e-12
AUDIT_RSS_CAP, AUDIT_SECONDS = 2 * 1024**3, 600
EXPECTED_WORKER = 'bc3050540c89beedfc66943ba753f640293ca7347f04c2969aaf9fc8d327e970'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    begin = time.time()
    result = {'status': 'inconclusive', 'audit_started_epoch': begin,
              'audit_scope': 'Post-experiment CPU; scalar records for 501 frames, direct RGB for 8 selected frames only',
              'new_GPU_model_generator_VAE_calls': 0,
              'tensor_loads_allowed': ['BF16_selected_RGB.pt', 'FP32_selected_RGB.pt'],
              'FP64_comparison': {'absolute_tolerance': ABS_TOL, 'relative_tolerance': REL_TOL},
              'checks': 0, 'hashes': {}, 'selected_recalculation': []}
    output = HERE / 'AUDIT_CODEX.json'
    if output.exists():
        raise RuntimeError('Audit output already exists; no automatic overwrite/retry')

    def check(condition, label):
        result['checks'] += 1
        if not condition:
            raise AssertionError(label)

    def close(a, b, label):
        check(math.isfinite(a) and math.isfinite(b) and
              abs(a - b) <= ABS_TOL + REL_TOL * max(abs(a), abs(b)), label)

    def hashed(path, expected=None):
        p = Path(path).resolve()
        if str(p) not in result['hashes']:
            result['hashes'][str(p)] = sha(p)
        value = result['hashes'][str(p)]
        if expected is not None:
            check(value == expected, 'SHA256: ' + str(p))
        return value

    try:
        r, final, run, claim, lock = (read(n) for n in
            ['RESULT.json', 'ROOT_EXECUTION_FINAL.json', 'RUN.json', 'ROOT_ATTEMPT_CLAIM.json', 'LOCK.json'])
        check(final['status'] == 'completed' and final['exit_code'] == 0, 'root completion/exit')
        check(final['worker_result_exists'] is True and final['ROOT_attempts'] == 1, 'one root attempt')
        check(r['status'] == 'completed_two_arm_sensitivity_diagnostic', 'worker completed')
        check(run['attempt'] == claim['attempt'] == 1, 'exclusive attempt records')
        check(run['status'] == 'started', 'RUN remains historical start record')
        check(lock['root_execution_authorized'] is True and lock['attempt_limit'] == 1, 'locked authority')
        check(lock['status'] == 'approved_for_one_two_arm_precision_probe', 'locked status')
        check(lock['decision_rule'] == 'late_RGB_RMSE_ge_0.02_sensitivity_only', 'locked gate')
        limits = lock['limits']
        check(r['limits'] == limits and limits['wall_seconds'] == 600 and limits['CPU_threads'] == 4,
              'limits identity')
        close(final['wall_seconds'], final['finished_epoch'] - final['start_epoch'], 'root elapsed')
        check(final['start_epoch'] == claim['start_epoch'], 'root start binding')
        close(claim['deadline_epoch'] - claim['start_epoch'], 600., 'root deadline')
        check(final['start_epoch'] <= r['started_utc'] <= r['finished_utc'] <= final['finished_epoch'],
              'nested run chronology')
        check(r['started_utc'] == run['started_utc'] and r['deadline_utc'] == run['absolute_deadline_utc'],
              'worker start/deadline binding')
        check(final['finished_epoch'] <= claim['deadline_epoch'] and final['wall_seconds'] <= 600,
              'root total 600 second cap')
        check(r['finished_utc'] <= r['deadline_utc'] and r['finished_utc'] - r['started_utc'] <= 600,
              'worker total 600 second cap')
        check(r['vae_decodes'] == 2 and r['generator_calls'] == 0, 'recorded operation counts')
        check(r['watchdog_completed_cleanly'] and r['inputs_unchanged'] and not r['whole_RGB_saved_to_disk'],
              'watchdog/input/storage record')
        for name in ['WATCHDOG_FAILURE.json', 'FAILURE.json']:
            check(not (HERE / name).exists(), 'no failure marker: ' + name)
        for field in ['TF32_matmul', 'TF32_cudnn', 'autocast', 'cudnn_benchmark']:
            check(r['runtime'][field] is False, 'runtime ' + field)
        check(r['runtime']['cudnn_deterministic'] is True, 'cudnn deterministic record')
        check(r['monitor_peak_RSS_bytes'] <= limits['RSS_bytes'] and
              r['process_lifetime_peak_RSS_bytes'] <= limits['RSS_bytes'], 'experiment RSS cap')
        check(r['monitor_peak_GPU_allocator_bytes'] <= limits['GPU_allocator_bytes'], 'GPU allocator cap')
        check(r['monitor_peak_output_bytes'] <= limits['output_bytes'], 'monitored disk cap')
        for path, expected in lock['sha256'].items():
            hashed(path, expected)
        hashed(HERE / 'worker.py', EXPECTED_WORKER)
        for name in ['RESULT.json', 'ROOT_EXECUTION_FINAL.json', 'RUN.json',
                     'ROOT_ATTEMPT_CLAIM.json', 'LOCK.json', 'verify_result.py']:
            hashed(HERE / name)
        wrapper = next(Path(p) for p in lock['sha256'] if p.endswith('/utils/wan_wrapper.py'))
        tree = ast.parse(wrapper.read_text())
        cls = [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'WanVAEWrapper']
        check(len(cls) == 1 and hashlib.sha256(ast.dump(cls[0], include_attributes=False).encode()).hexdigest()
              == r['wrapper_AST_sha256'], 'native wrapper class AST')
        check([a['name'] for a in r['arms']] == ['BF16', 'FP32'], 'arm order and count')
        checkpoint = next(v for p, v in lock['sha256'].items() if p.endswith('/Wan2.1_VAE.pth'))
        for arm, dtype in zip(r['arms'], ['torch.bfloat16', 'torch.float32']):
            check(arm['fresh_checkpoint_loaded'] and arm['checkpoint_sha256'] == checkpoint,
                  'fresh checkpoint record: ' + arm['name'])
            hist = arm['loaded_parameter_dtype_histogram']
            check(set(hist) == {'torch.float32'} and hist['torch.float32'] > 0, 'original parameter dtypes')
            check(arm['latent_dtype'] == dtype and arm['use_cache'] is False, 'arm input dtype/cache')
            check(arm['raw_RGB_sha256'] is None, 'no whole RGB hash claim')
            check(arm['GPU_peak_allocated_bytes_cumulative'] <= limits['GPU_allocator_bytes'], 'arm GPU peak')
            check(math.isfinite(arm['decode_seconds']) and arm['decode_seconds'] > 0, 'positive decode time')
            expected_files = {str(HERE / f"{arm['name']}_selected_RGB.pt")}
            expected_files |= {str(HERE / f"{arm['name']}_frame{n:03d}.png") for n in SELECTED}
            check({a['path'] for a in arm['selected_artifacts']} == expected_files and
                  len(arm['selected_artifacts']) == 9, 'exact selected artifact coverage')
            for a in arm['selected_artifacts']:
                check(Path(a['path']).stat().st_size == a['bytes'], 'artifact file size')
                hashed(a['path'], a['sha256'])
        rows = r['per_frame']
        check(len(rows) == 501 and [x['frame'] for x in rows] == list(range(501)), '501 unique ordered rows')
        for row in rows:
            for key in ['MSE', 'MAE', 'q99_abs']:
                check(math.isfinite(row[key]) and 0 <= row[key] <= 1, 'bounded error scalar')
            for key in ['mean_RGB_bf16', 'mean_RGB_fp32']:
                check(len(row[key]) == 3 and all(math.isfinite(x) and 0 <= x <= 1 for x in row[key]),
                      'bounded RGB mean vector')
        late_rmse = math.sqrt(math.fsum(row['MSE'] for row in rows[249:501]) / 252)
        summary = r['summary']
        close(late_rmse, summary['late_RGB_RMSE'], 'independent pooled late RMSE')
        check(summary['late_frames'] == [249,500] and summary['early_mean_frames'] == [0,125]
              and summary['late_mean_frames'] == [375,500], 'summary ranges')
        recomputed = {}
        for suffix in ['bf16', 'fp32']:
            key = 'mean_RGB_' + suffix
            vec = [math.fsum(row[key][c] for row in rows[375:501])/126 -
                   math.fsum(row[key][c] for row in rows[:126])/126 for c in range(3)]
            norm = math.sqrt(math.fsum(v*v for v in vec))
            stored = summary['mean_RGB_change_' + suffix]
            for a,b in zip(vec, stored['vector']):
                close(a,b,'independent RGB drift vector')
            close(norm,stored['L2'],'independent RGB drift norm')
            recomputed[suffix] = {'vector':vec,'L2':norm}
        reduction = None if recomputed['bf16']['L2'] == 0 else 1 - recomputed['fp32']['L2']/recomputed['bf16']['L2']
        if reduction is None:
            check(summary['mean_RGB_change_reduction_descriptive'] is None, 'undefined descriptive ratio')
        else:
            close(reduction,summary['mean_RGB_change_reduction_descriptive'],'descriptive ratio')
        decision = 'review_precision_sensitivity' if late_rmse >= .02 else 'stop_precision_expansion'
        check(summary['decision'] == decision, 'registered decision independently recomputed')
        result['scalar_reaggregation'] = {'late_RGB_RMSE':late_rmse, 'decision':decision,
            'threshold':.02, 'frames':252, 'drift':recomputed, 'reduction_descriptive':reduction}

        import numpy as np
        import torch
        from PIL import Image
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        check(not torch.cuda.is_initialized(), 'no CUDA context before CPU audit')
        arrays = []
        for name in ['BF16', 'FP32']:
            data = torch.load(HERE/f'{name}_selected_RGB.pt', map_location='cpu', weights_only=True)
            check(set(data) == {'frames','RGB'} and tuple(data['frames']) == SELECTED, 'selected state keys/frames')
            tensor = data['RGB']
            check(tensor.device.type == 'cpu' and tensor.dtype == torch.float32 and
                  tuple(tensor.shape) == (8,3,480,832), 'selected tensor contract')
            x = tensor.numpy()
            check(np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1, 'selected RGB finite/range')
            arrays.append(x)
            for i, frame in enumerate(SELECTED):
                expected = (x[i].transpose(1,2,0) * np.float32(255)).astype(np.uint8)
                with Image.open(HERE/f'{name}_frame{frame:03d}.png') as im:
                    check(im.mode == 'RGB' and im.size == (832,480), 'PNG mode/shape')
                    check(im.tobytes() == expected.tobytes(), 'PNG equals selected RGB8 conversion')
        for j, frame in enumerate(SELECTED):
            a, b = (x[j].astype(np.float64) for x in arrays)
            delta = b-a
            count = delta.size
            mse = float(np.sum(np.square(delta), dtype=np.float64) / count)
            ad = np.abs(delta).reshape(-1)
            mae = float(np.sum(ad, dtype=np.float64) / count)
            pos = .99*(count-1)
            lo, hi = math.floor(pos), math.ceil(pos)
            ordered = np.partition(ad, (lo,hi))
            q99 = float(ordered[lo] + (pos-lo)*(ordered[hi]-ordered[lo]))
            got = {'MSE':mse,'MAE':mae,'q99_abs':q99}
            errors = {}
            for key, value in got.items():
                close(value, rows[frame][key], f'selected frame{frame} {key}')
                errors[key] = abs(value - rows[frame][key])
            means = {}
            for name, x in [('bf16', a),('fp32', b)]:
                values = [float(np.sum(x[c], dtype=np.float64) / x[c].size) for c in range(3)]
                for v, stored in zip(values, rows[frame]['mean_RGB_'+name]):
                    close(v,stored,f'selected frame{frame} mean RGB')
                means[name] = values
            result['selected_recalculation'].append({'frame':frame,'elements':count,
                **got,'mean_RGB':means,'absolute_difference_from_record':errors})
        check(not torch.cuda.is_initialized(), 'no CUDA context after CPU audit')
        disk = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
        check(disk <= limits['output_bytes'], 'current total folder size cap')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        check(rss <= AUDIT_RSS_CAP, 'post-run audit RSS <=2GiB')
        check(time.time()-begin <= AUDIT_SECONDS, 'post-run audit <=10min')
        result.update(status='passed', worker_wall_seconds=r['finished_utc']-r['started_utc'],
            root_wall_seconds=final['wall_seconds'], experiment_peaks={
                k:r[k] for k in ['process_lifetime_peak_RSS_bytes','monitor_peak_GPU_allocator_bytes','monitor_peak_output_bytes']},
            audit_RSS_bytes=rss, folder_bytes_before_audit_output=disk,
            whole_501_RGB_independently_recomputed=False, direct_RGB_frames=8,
            fresh_load_and_execution_counts='source/record verification; not a new trace or model replay')
    except BaseException as exc:
        result.update(status='audit_failed_or_inconclusive',error=repr(exc))
    result.update(audit_finished_epoch=time.time(),audit_wall_seconds=time.time()-begin,
                  auditor_sha256=sha(__file__))
    with output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({k:result.get(k) for k in ['status','checks','error','scalar_reaggregation','audit_wall_seconds','audit_RSS_bytes']}))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
