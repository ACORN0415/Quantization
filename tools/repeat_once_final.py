"""Bounded final verification. Existing registered sources/artifacts are read-only."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home()/'sfq_data/repeat_once_final'
OLD = Path.home()/'sfq_data/mechanism'
OUT = ROOT/'src/experiments/repeat_once_final'
PLAN = ROOT/'src/experiments/REPEAT_ONCE_FINAL.md'
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'src/scripts'), str(ROOT/'tools')]


def stamp():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def own_files():
    names = ['tools/repeat_once_final.py', 'tools/repeat_once_audit.py',
             'tools/test_repeat_once_final.py', 'tools/repeat_once_paper.py',
             'src/experiments/REPEAT_ONCE_FINAL.md']
    return {name:sha(ROOT/name) for name in names}


def emit(message):
    print(stamp(), message, flush=True)


def gpu_pids():
    p = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                       text=True, capture_output=True, check=True)
    return [int(x) for x in p.stdout.splitlines() if x.strip().isdigit()]


def tt_state():
    p = subprocess.run(['systemctl','--user','show','tt-kv-final.service',
                        '-p','ActiveState','-p','SubState','-p','Result','-p','LoadState'],
                       text=True,capture_output=True)
    values = dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)
    if p.returncode and values.get('LoadState')!='not-found':
        raise RuntimeError('cannot establish TT service state: '+p.stderr)
    log = Path.home()/'sfq_data/tt_kv_final/run.log'
    values['completion_marker'] = log.exists() and 'COMPLETE 18/18 generation, quality and report' in log.read_text()
    return values


def wait_for_tt():
    while True:
        state = tt_state()
        pids = gpu_pids()
        ready = (state.get('ActiveState') in ('inactive','failed') or state.get('LoadState')=='not-found')
        status = dict(state='waiting', updated_kst=stamp(), tt=state, gpu_pids=pids)
        save(DATA/'status.json',status)
        if ready and state['completion_marker'] and not pids:
            return status
        if ready and not state['completion_marker']:
            raise RuntimeError('TT ended without successful completion marker; final verification not started')
        emit('TT 종료 및 GPU 유휴 대기: '+json.dumps(state)+' GPU='+str(pids))
        time.sleep(45)


def verify_archives():
    evidence = read(ROOT/'src/experiments/mechanism_fr_results/evidence.json')
    artifacts = [r for r in evidence['artifacts'] if r['path'].endswith('_C_repeat/latent.pt')
                 or r['path'].endswith('_C_repeat/sampling.json')]
    if len(artifacts)!=48:
        raise RuntimeError('expected 48 archived reference artifacts')
    for item in artifacts:
        path = OLD/item['path']
        if not path.exists() or sha(path)!=item['sha256']:
            raise RuntimeError('reference missing/changed: '+str(path))
    return artifacts


def preflight():
    import mechanism_worker as original
    seal = read(ROOT/'src/experiments/mechanism_fr_registration.json')
    current = original.source_hash()
    if current!=seal['source_sha256']:
        raise RuntimeError('registered original implementation/model hash mismatch')
    archive = verify_archives()
    return dict(original_source_sha256=current, registered=seal, archive=archive,
                new_source_sha256=own_files(), verified_kst=stamp(),
                environment_command=subprocess.run(['nvidia-smi'],text=True,capture_output=True).stdout)


def tensor_digest(x):
    x = x.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode())
    h.update(x.view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def worker(case, arm, prompt, seed):
    import torch
    import mechanism_worker as original
    import p3_tf_fr as pilot
    from mechanism_core import make
    from kvq.replay import store_cache
    from repeat_once_audit import OperationAudit, compare_bits, require
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    os.chdir(ROOT/'src')
    manifest = read(DATA/'manifest.json')
    require(own_files()==manifest['new_source_sha256'], 'new implementation changed')
    require(original.source_hash()==manifest['original_source_sha256'], 'old implementation changed')
    require(not [p for p in gpu_pids() if p!=os.getpid()], 'GPU occupied before model load')
    folder = DATA/case
    folder.mkdir(parents=True,exist_ok=True)
    emit(f'실행 시작 {case}')
    begin = time.monotonic()
    pipe = pilot.build(seed)
    texts=(ROOT/'src/experiments/mechanism_prompts_v0.txt').read_text().splitlines()
    cond=pilot.encode(pipe,[texts[prompt]])[0]
    require(len(pipe.denoising_step_list)==4 and pipe.num_frame_per_block==3 and pipe.frame_seq_length==1560,
            'model schedule/geometry changed')
    stores=[make(arm,16,observe=False) for _ in range(30)]
    require(all(s.repeat == (arm=='C_repeat') for s in stores), 'requested arm does not match stores')
    cache=store_cache(stores,pilot.gpu)
    xc=pilot.cross_cache(pipe,pilot.gpu)
    noise=pilot.chunk_noise(pipe,126,seed,pilot.gpu)
    outputs=[]
    torch.cuda.synchronize()
    generation_start=time.monotonic()
    audit=OperationAudit(stores,folder/'events.jsonl')
    try:
        with audit:
            for chunk in range(42):
                outputs.append(pilot.run_chunk(pipe,cache,xc,cond,noise,chunk,seed))
                if chunk%7==0 or chunk==41:
                    save(folder/'progress.json',dict(chunk_completed=chunk+1,updated_kst=stamp(),layers=audit.records))
                    emit(f'{case} 청크 {chunk+1}/42')
            call_record=audit.validate()
        latent=torch.cat(outputs,1)
        torch.cuda.synchronize()
        generation_s=time.monotonic()-generation_start
        cpu=latent.cpu()
        require(bool(torch.isfinite(cpu).all()), 'nonfinite generated latent')
        torch.save({'latent':cpu},folder/'latent.pt')
        reference_folder=OLD/'fr'/f'p{prompt:02d}_s{seed}_C_repeat'
        previous=read(reference_folder/'sampling.json')
        require(previous['source_sha256']==manifest['original_source_sha256'], 'reference source mismatch')
        require(sha(reference_folder/'latent.pt')==previous['latent_file_sha256'], 'reference file changed')
        ref=torch.load(reference_folder/'latent.pt',map_location='cpu',weights_only=True)['latent']
        comparison=compare_bits(cpu,ref)
        require(tensor_digest(ref)==previous['latent_sha256'], 'reference tensor hash mismatch')
        require(own_files()==manifest['new_source_sha256'], 'implementation changed during run')
        require(original.source_hash()==manifest['original_source_sha256'], 'original source changed during run')
        # Reference-file and tensor-byte hashes are explicitly separate.
        result=dict(case=case,arm=arm,prompt=prompt,seed=seed,valid=True,comparison=comparison,
                    execution=call_record,finished_kst=stamp(),generation_s=generation_s,
                    worker_elapsed_s=time.monotonic()-begin,
                    latent_file_sha256=sha(folder/'latent.pt'),latent_tensor_sha256=tensor_digest(cpu),
                    reference_file_sha256=previous['latent_file_sha256'],
                    event_file_sha256=sha(folder/'events.jsonl'),
                    chunks=[tensor_digest(cpu[:,3*c:3*c+3]) for c in range(42)],
                    original_source_sha256=manifest['original_source_sha256'],
                    torch_version=torch.__version__,cuda_version=torch.version.cuda,
                    device=torch.cuda.get_device_name(),
                    peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                    note='instrumented validation run, not a performance benchmark')
        save(folder/'result.json',result)
        emit(f'{case} 유효 실행 완료, 비트 일치={comparison["equal"]}')
    except Exception:
        save(folder/'partial_counts.json',audit.records)
        raise


def terminate_worker(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()


def report(budget, reason):
    OUT.mkdir(parents=True,exist_ok=True)
    comparisons=[]
    gates=[]
    artifacts=[]
    for attempt in budget['attempts']:
        folder=DATA/attempt['case']
        result_path=folder/'result.json'
        result=read(result_path) if result_path.exists() else None
        if result is not None:
            for name in ('result.json','events.jsonl','latent.pt'):
                artifacts.append(dict(path=str((folder/name).relative_to(DATA)),sha256=sha(folder/name)))
        entry=dict(case=attempt['case'],prompt=attempt['prompt'],seed=attempt['seed'],
                   status='valid_equal' if result and result['valid'] and result['comparison']['equal'] else
                          'valid_different' if result and result['valid'] else
                          'incomplete' if attempt.get('timeout') else 'invalid',
                   comparison=result['comparison'] if result else None,
                   generation_s=result['generation_s'] if result else None)
        (gates if attempt['arm']=='C_repeat' else comparisons).append(entry)
    seen={(c['prompt'],c['seed']) for c in comparisons}
    for p in range(12):
        for s in (303,404):
            if (p,s) not in seen:
                comparisons.append(dict(case=f'once_p{p:02d}_s{s}',prompt=p,seed=s,status='not_started',comparison=None))
    comparisons.sort(key=lambda x:(x['prompt'],x['seed']))
    exact=sum(x['status']=='valid_equal' for x in comparisons)
    different=sum(x['status']=='valid_different' for x in comparisons)
    conclusion='all_24_exact' if exact==24 else 'exact_hypothesis_failed' if different else 'inconclusive'
    if len(gates)!=2 or any(g['status']!='valid_equal' for g in gates):
        conclusion='gate_failed_or_incomplete'
    budget['reported_kst']=stamp()
    budget['elapsed_through_report_s']=time.monotonic()-budget['started_monotonic']
    summary=dict(conclusion=conclusion,stop_reason=reason,comparisons=comparisons,gates=gates,budget=budget,
                 completed_valid=sum(x['status'].startswith('valid_') for x in gates+comparisons),
                 direct_observation='new repeated reproductions and new once runs only',
                 historical_observation='24 archived repeat latents; historical call logs unavailable',
                 artifacts=artifacts)
    save(DATA/'summary.json',summary)
    save(OUT/'summary.json',summary)
    if (DATA/'manifest.json').exists():save(OUT/'manifest.json',read(DATA/'manifest.json'))
    with (OUT/'comparisons.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['prompt','seed','status','different_elements','max_abs_difference'])
        writer.writeheader()
        for c in comparisons:
            detail=c['comparison'] or {}
            writer.writerow(dict(prompt=c['prompt'],seed=c['seed'],status=c['status'],
                                 different_elements=detail.get('different_elements'),
                                 max_abs_difference=detail.get('max_abs_difference')))
    lines=['# 단회·반복 압축 마지막 검증 결과','',
           f'판정: `{conclusion}`. 종료 사유: `{reason}`.',
           f'24개 예정 쌍 중 비트 일치 {exact}, 유효 불일치 {different}, 나머지 {24-exact-different}.',
           f'새 worker 시도 {len(budget["attempts"])}/26, 유효 완료 {summary["completed_valid"]}.',
           f'시작 {budget["started_kst"]}, 마지막 실행 종료 {budget.get("workers_finished_kst")},',
           f'집계 시점 {budget["reported_kst"]}, 집계까지 경과 {budget["elapsed_through_report_s"]:.1f}초 / 14400초.',
           '', '## 직접 계측 범위', '',
           '실제 Quantizer.encode 호출을 K/V·층·최초/재인코딩별로 계측했다. 복원/양자화를 계측용으로 추가하지 않았다.',
           '재인코딩 입력은 실제 restore 반환 객체의 전달 경로로 검사했다. 입력 통계/해시의 전체 원소 일치 주장이 아니다.',
           '과거 반복24개는 산출물과 소스 기록을 재사용했다. 그 실행들의 호출 로그는 없으며 사후 복원했다고 주장하지 않는다.',
           '', '## 두 반복 재현', '', '|사례|판정|','|---|---|']
    lines += [f'|{g["case"]}|{g["status"]}|' for g in gates]
    lines += ['', '## 24개 대응 비교', '', '|프롬프트|시드|판정|다른 원소|','|---|---|---|---|']
    lines += [f'|{c["prompt"]}|{c["seed"]}|{c["status"]}|{(c["comparison"] or {}).get("different_elements","—")}|' for c in comparisons]
    lines += ['', '## 해석', '',
              '논문의 중심은 기존 그룹 규칙 개입의 후반 MUSIQ +5.3935점이며, 이번 결과는 구현 검증이다.',
              '동일 결과는 시험한 구현/환경/입력 범위에 한정한다. 보편 멱등성·시간 품질·같은 메모리 예산 우위·새 알고리즘 신규성을 뜻하지 않는다.',
              '유효 불일치는 허용 오차 없이 실패로 남긴다. 미완료/무효를 일치 사례로 계산하지 않는다.',
              '이번 고정 계획 이후 새 후보·추가 시드·추가 rank 탐색을 시작하지 않는다.',
              '', '[상세 판정·예산·산출물 해시](summary.json) · [전체 비교](comparisons.csv) · [실행 계획](../REPEAT_ONCE_FINAL.md)']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    save(DATA/'status.json',dict(state='finished',conclusion=conclusion,updated_kst=stamp(),
                                  attempts=len(budget['attempts']),equal=exact,different=different))
    emit(f'최종 검증 종료: {conclusion}, 시도 {len(budget["attempts"])}회')


def supervise():
    DATA.mkdir(parents=True,exist_ok=True)
    # Never reset an already-started experiment's clock/counter.
    if (DATA/'budget.json').exists():
        raise RuntimeError('budget already exists; refusing to reset or implicitly resume')
    frozen=own_files()
    save(DATA/'queued_plan.json',dict(queued_kst=stamp(),sources=frozen))
    readiness=wait_for_tt()
    started=time.monotonic()
    budget=dict(started_kst=stamp(),started_monotonic=started,deadline_monotonic=started+14400,
                time_limit_s=14400,attempt_limit=26,attempts=[],readiness=readiness)
    save(DATA/'budget.json',budget)
    reason='complete'
    proc=None
    def time_limit_handler(*_):
        raise TimeoutError('global verification deadline reached')
    old_alarm=signal.signal(signal.SIGALRM,time_limit_handler)
    signal.alarm(14340)
    try:
        if own_files()!=frozen:raise RuntimeError('queued plan/code changed before start')
        manifest=preflight()
        save(DATA/'manifest.json',manifest)
        jobs=[('repeat_p00_s303','C_repeat',0,303),('repeat_p11_s404','C_repeat',11,404)]
        jobs += [(f'once_p{p:02d}_s{s}','C_once',p,s) for p in range(12) for s in (303,404)]
        for case,arm,prompt,seed in jobs:
            # Reserve one minute for worker termination and mandatory final accounting.
            if time.monotonic()>=budget['deadline_monotonic']-60:
                reason='time_limit';break
            if len(budget['attempts'])>=26:
                reason='attempt_limit';break
            if gpu_pids():
                reason='gpu_became_occupied';break
            if arm=='C_once':
                gate_results=[read(DATA/g[0]/'result.json') for g in jobs[:2]]
                if not all(g['valid'] and g['comparison']['equal'] and g['execution']['valid'] for g in gate_results):
                    reason='repeat_gate_failed';break
            folder=DATA/case
            folder.mkdir(parents=True,exist_ok=True)
            attempt=dict(case=case,arm=arm,prompt=prompt,seed=seed,started_kst=stamp())
            budget['attempts'].append(attempt)
            save(DATA/'budget.json',budget)
            save(DATA/'status.json',dict(state='running',case=case,attempt=len(budget['attempts']),updated_kst=stamp()))
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--mode','worker',
                     '--case',case,'--arm',arm,'--prompt',str(prompt),'--seed',str(seed)]
            with (folder/'run.log').open('w') as log:
                proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    code=proc.wait(timeout=max(.001,budget['deadline_monotonic']-60-time.monotonic()))
                except subprocess.TimeoutExpired:
                    attempt['timeout']=True;reason='time_limit'
                    terminate_worker(proc);code=proc.returncode
            attempt.update(finished_kst=stamp(),exit_code=code)
            save(DATA/'budget.json',budget)
            proc=None
            if code:
                if reason!='time_limit':reason='invalid_execution'
                break
            result=read(folder/'result.json')
            emit(f'{case} 완료: 비트 일치={result["comparison"]["equal"]}')
            if arm=='C_repeat' and not result['comparison']['equal']:
                reason='repeat_reproduction_mismatch';break
    except TimeoutError as exc:
        reason='time_limit'
        if proc is not None and budget['attempts']:
            budget['attempts'][-1]['timeout']=True
        save(DATA/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),kst=stamp()))
    except Exception as exc:
        reason='preflight_or_supervisor_failure'
        save(DATA/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),kst=stamp()))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,old_alarm)
        if proc is not None:
            terminate_worker(proc)
            if budget['attempts']:
                budget['attempts'][-1].update(finished_kst=stamp(),exit_code=proc.returncode)
        budget['workers_finished_kst']=stamp()
        budget['elapsed_to_workers_finished_s']=time.monotonic()-started
        save(DATA/'budget.json',budget)
        report(budget,reason)
        # Static figure/draft composition is reported separately from the bounded experiment.
        try:
            from repeat_once_paper import render
            post_started=time.monotonic()
            render()
            save(DATA/'postprocessing.json',dict(seconds=time.monotonic()-post_started,finished_kst=stamp()))
        except Exception as exc:
            save(DATA/'postprocessing_failure.json',dict(error=str(exc),traceback=traceback.format_exc()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['all','worker'],default='all')
    p.add_argument('--case');p.add_argument('--arm');p.add_argument('--prompt',type=int);p.add_argument('--seed',type=int)
    args=p.parse_args()
    if args.mode=='all':supervise()
    else:
        try:worker(args.case,args.arm,args.prompt,args.seed)
        except Exception as exc:
            save(DATA/args.case/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),kst=stamp()))
            raise


if __name__=='__main__':main()
