"""Serialize the registered diagnostic after the existing R8d GPU work.

The waiting period is separate from the 30-minute experiment budget. This
driver never stops another process and never retries a failed generation.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PYTHON = Path('/home/acorn0415/sfq_data/env/bin/python')
PRIOR = ROOT / 'src/experiments/kv_alignment_compounding/r8c/quality.json'
STATUS = HERE / 'DRIVER_STATUS.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(state):
    state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATUS)


def verify_lock():
    lock = json.loads((HERE / 'LOCK.json').read_text())
    assert lock['root_accepted'] and lock['fable_reviewed']
    assert str(Path(__file__).resolve()) in lock['sha256']
    for path, digest in lock['sha256'].items():
        assert sha(path) == digest, path


def gpu_idle():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        text=True, timeout=10)
    return not output.strip()


def prior_terminal():
    try:
        result = json.loads(PRIOR.read_text())
    except (OSError, ValueError):
        return False
    # R8d began 2026-11-05 15:53 KST. Do not accept the older 14-row R8c file.
    return (result.get('started_utc', 0) >= 1793861586
            and result.get('status') in ('completed', 'failed'))


def stop_own_child(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def main(wait_seconds):
    if not 1 <= wait_seconds <= 7200:
        raise ValueError('wait-seconds must be between 1 and 7200')
    verify_lock()
    if STATUS.exists() or (HERE / 'RUN.json').exists():
        raise RuntimeError('Refuse a second driver or experiment restart')
    # Exclusive creation prevents two simultaneous launchers passing the checks.
    with (HERE / 'DRIVER_STARTED.json').open('x') as marker:
        json.dump({'pid': os.getpid(), 'started_utc': time.time()}, marker)
    state = {'status': 'waiting_for_existing_gpu_work', 'stages': [],
             'driver_sha256': sha(__file__), 'prior_quality': str(PRIOR),
             'maximum_wait_seconds': wait_seconds, 'experiment_limit_seconds': 1800}
    save(state)
    wait_start = time.monotonic()
    child = None
    try:
        while not (prior_terminal() and gpu_idle()):
            remaining_wait = wait_seconds - (time.monotonic() - wait_start)
            if remaining_wait <= 0:
                raise TimeoutError('Wait limit reached before experiment; no generation started')
            time.sleep(min(15, remaining_wait))
        # A second idle check avoids the brief gap between external GPU stages.
        remaining_wait = wait_seconds - (time.monotonic() - wait_start)
        if remaining_wait < 3:
            raise TimeoutError('Wait limit leaves no idle-confirmation interval')
        time.sleep(3)
        if not gpu_idle():
            raise RuntimeError('Another GPU job started; no generation started')
        verify_lock()
        start = time.monotonic()
        state.update(status='running', waiting_seconds=start-wait_start,
                     experiment_started_utc=time.time())
        save(state)
        stages = [('reference', 'capture.py', 'reference'),
                  ('capture', 'capture.py', 'capture'),
                  ('cpu_packets', 'replay.py', 'packets'),
                  ('native_read', 'replay.py', 'native')]
        for name, script, mode in stages:
            verify_lock()
            if name != 'cpu_packets' and not gpu_idle():
                raise RuntimeError(f'GPU became busy before {name}; no overlapping work started')
            remaining = 1800 - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError('Experiment wall-time limit reached')
            entry = {'name': name, 'started_utc': time.time(), 'status': 'running'}
            state['stages'].append(entry)
            save(state)
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            if name == 'cpu_packets':
                env['CUDA_VISIBLE_DEVICES'] = ''
            with (HERE / f'{name}.log').open('x') as log:
                child = subprocess.Popen([str(PYTHON), str(HERE/script), mode],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                try:
                    code = child.wait(timeout=remaining)
                except BaseException as error:
                    entry.update(status='stopped', error=repr(error))
                    stop_own_child(child)
                    raise
                finally:
                    entry['finished_utc'] = time.time()
                entry.update(returncode=code, status='completed' if code == 0 else 'failed')
                save(state)
                if code:
                    raise RuntimeError(f'{name} failed; no automatic retry')
                child = None
        result = json.loads((HERE/'NATIVE_RESULT.json').read_text())
        assert result['status'] == 'completed'
        state.update(status='completed', elapsed_experiment_seconds=time.monotonic()-start,
                     native_result_sha256=sha(HERE/'NATIVE_RESULT.json'))
    except BaseException as error:
        state.update(status='stopped', error=repr(error))
        save(state)
        raise
    finally:
        stop_own_child(child)
        save(state)


if __name__ == '__main__':
    def stop_requested(signum, frame):
        raise KeyboardInterrupt(f'Driver stop requested by signal {signum}')
    signal.signal(signal.SIGTERM, stop_requested)
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-seconds', type=int, default=7200)
    arguments = parser.parse_args()
    main(arguments.wait_seconds)
