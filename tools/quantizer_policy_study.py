"""Development only. A GO prepares a decision; it never accesses new held-out text."""
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import quantizer_policy_worker as worker
from quantizer_policy_core import interaction

DATA = worker.DATA
OUT = ROOT/'src/experiments/quantizer_policy_cross_development'


def job(*args):
    command = [sys.executable, ROOT/'tools/quantizer_policy_worker.py', *args]
    print('RUN', *command, flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def matching(path, source):
    return path.exists() and worker.read(path)['source_sha256'] == source


def analyze(source):
    rows = []; artifacts = []; bounds = []
    for prompt in (0, 4):
        for method in ('rtn', 'kivi'):
            for arm in ('L', 'C_repeat'):
                folder = DATA/'development'/f'p{prompt:02d}_s101_{method}_{arm}'
                s = worker.read(folder/'sampling.json'); q = worker.read(folder/'quality.json')
                resource = worker.read(folder/'quality_resource.json')
                if any(v['source_sha256'] != source for v in (s, q, resource)):
                    raise RuntimeError('mixed source in factorial screen')
                if worker.common.sha(folder/'latent.pt') != s['latent_file_sha256']:
                    raise RuntimeError('changed development tensor')
                scores = q['per_frame_musiq']
                if len(scores) != 501 or not all(math.isfinite(v) for v in scores):
                    raise RuntimeError('incomplete development frame scores')
                whole_s = s['setup_s']+s['generation_s']+resource['seconds']
                maximum = max(s['peak']['allocated'], resource['peak_allocated_bytes'])
                cpu_max = max(s['cpu_lifetime_high_water_bytes'], resource['cpu_lifetime_high_water_bytes'])
                within = whole_s < 600 and maximum < 22*2**30 and cpu_max < 48*2**30
                bounds.append(within)
                rows.append(dict(prompt=prompt, seed=101, method=method, arm=arm,
                    late_raw_musiq=statistics.mean(scores[249:501]),
                    boundary_raw_musiq=q['boundary_raw_musiq'], inside_raw_musiq=q['inside_raw_musiq'],
                    generation_s=s['generation_s'], setup_s=s['setup_s'], quality_wall_s=resource['seconds'],
                    case_wall_s=whole_s, peak_allocated_bytes=maximum, cpu_high_water_bytes=cpu_max,
                    budget_pass=within, **s['cache']))
                for name in ('latent.pt', 'sampling.json', 'quality.json', 'quality_resource.json', 'video.mp4'):
                    path = folder/name
                    artifacts.append(dict(path=str(path.relative_to(DATA)), bytes=path.stat().st_size,
                        sha256=worker.common.sha(path)))
    effects = [dict(prompt=p, interaction=interaction([r for r in rows if r['prompt'] == p])) for p in (0, 4)]
    sign_agrees = effects[0]['interaction']*effects[1]['interaction'] > 0
    effect_pass = sign_agrees and all(abs(v['interaction']) >= 1 for v in effects)
    gate = worker.read(DATA/'gates.json')
    if not gate['passed'] or gate['source_sha256'] != source:
        raise RuntimeError('development gate missing')
    replay_summary = []
    for prompt in (0, 4):
        replay = worker.read(DATA/'replay'/f'p{prompt:02d}_s101.json')
        if replay['source_sha256'] != source or replay['teacher_bitexact_chunks'] != 42:
            raise RuntimeError('teacher replay not validated')
        for method in ('rtn', 'kivi'):
            for arm in ('L', 'C_repeat'):
                selected = [r for r in replay['rows'] if r['method'] == method and r['arm'] == arm and r['chunk'] >= 7]
                if len(selected) != 6*35*5:
                    raise RuntimeError('six-layer replay incomplete')
                replay_summary.append(dict(prompt=prompt, method=method, arm=arm,
                    k_rel=statistics.mean(r['k_rel'] for r in selected),
                    v_rel=statistics.mean(r['v_rel'] for r in selected), layers=replay['layers']))
    if source != worker.source_hash():
        raise RuntimeError('source changed before development analysis')
    advance = effect_pass and all(bounds)
    result = dict(source_sha256=source, development_only=True, generated_cases=8,
        independent_development_prompts=2, effect_rule='same sign and |interaction| >= 1.0 in both prompts',
        interactions=effects, effect_pass=effect_pass, resource_pass=all(bounds), advance=advance,
        next_action='prepare separate heldout registration' if advance else 'stop; no new heldout generation',
        gates=gate, replay_summary=replay_summary, artifacts=artifacts)
    OUT.mkdir(parents=True, exist_ok=True)
    worker.save(OUT/'decision.json', result)
    with (OUT/'cases.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# 양자화기 × 저장 정책 개발 비교', '',
        '**개발 데이터이며 p값·새 보류 표본 검증이 아니다.**', '',
        '| prompt | RTN L | RTN C-repeat | KIVI-style L | KIVI-style C-repeat | interaction |',
        '|---|---:|---:|---:|---:|---:|']
    for item in effects:
        p = item['prompt']; selected = {(r['method'],r['arm']):r for r in rows if r['prompt'] == p}
        numbers = [selected[m,a]['late_raw_musiq'] for m,a in [('rtn','L'),('rtn','C_repeat'),('kivi','L'),('kivi','C_repeat')]]
        lines.append('| '+str(p)+' | '+' | '.join(f'{v:.4f}' for v in numbers+[item['interaction']])+' |')
    lines += ['', 'interaction = (KIVI_C-repeat−RTN_C-repeat)−(KIVI_L−RTN_L).',
        f'고정 개발 진행 기준: **{"GO" if advance else "NO-GO"}**. 효과 기준 {effect_pass}, 비용 기준 {all(bounds)}.',
        'GO는 후속 계획 준비 허용이며 새 방법·일반성·논문 기여의 입증이 아니다.',
        'KIVI-style은 현재 재구현의 고유 그룹 축·offset을 유지하며 원본 KIVI 전체 방법의 재현이 아니다.',
        '같은 명목 비트폭이며 같은 메모리 예산이 아니다. 실제 상주량·offset·pending을 cases.csv에 기록했다.',
        '이 결과를 보고 양자화기·표본·INT2로 교체해 같은 개발 기준을 다시 통과시키지 않는다.', '',
        '[고정 개발 계획](../QUANTIZER_POLICY_CROSS_DEVELOPMENT.md) · [전체 사례·비용·바이트](cases.csv) · [판단·재생·해시](decision.json)']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    source = worker.source_hash()
    worker.report('교차 비교 개발 게이트 시작', completed=0, total=8)
    if not matching(DATA/'gates.json', source):
        job('--task', 'gates')
    for prompt in (0, 4):
        if not matching(DATA/'replay'/f'p{prompt:02d}_s101.json', source):
            job('--task', 'replay', '--prompt', str(prompt))
    completed = 0
    order = [('rtn','L'), ('kivi','L'), ('rtn','C_repeat'), ('kivi','C_repeat')]
    for prompt in (0, 4):
        for method, arm in (order if prompt == 0 else order[2:]+order[:2]):
            if source != worker.source_hash():
                raise RuntimeError('frozen development source changed')
            worker.report(f'DEVELOPMENT p{prompt} {method}/{arm}', completed=completed, total=8)
            args = ['--prompt', str(prompt), '--method', method, '--arm', arm]
            job('--task', 'generate', *args)
            job('--task', 'quality', *args)
            completed += 1
            worker.report('개발 사례 완료', completed=completed, total=8)
    result = analyze(source)
    subprocess.run(['git','add','--', str(OUT.relative_to(ROOT))], cwd=ROOT, check=True)
    if subprocess.run(['git','diff','--cached','--quiet','--',str(OUT.relative_to(ROOT))], cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m','Publish fixed quantizer-policy development screen and decision',
                        '--',str(OUT.relative_to(ROOT))], cwd=ROOT, check=True)
    subprocess.run(['git','push','origin','main'], cwd=ROOT, check=True)
    worker.report('개발 비교 완료: '+('GO — 별도 보류 계획 준비' if result['advance'] else 'NO-GO — 보류 생성 없음'),
                  state='pass', completed=8, total=8)
    print('QUANTIZER POLICY DEVELOPMENT COMPLETE', result['advance'], flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        worker.report(f'Coordinator {type(exc).__name__}: {exc}', state='failed')
        raise
