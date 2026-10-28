"""Detached evaluation -> integrated report -> commit/push; no generation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DATA=Path.home()/'sfq_data/policy_supplement'
OUT=ROOT/'src/experiments/policy_supplement_results'


def run(*args):
    print('RUN',*map(str,args),flush=True)
    subprocess.run(list(map(str,args)),cwd=ROOT,check=True)


def status(detail,state='running'):
    p=DATA/'status.json';v=json.loads(p.read_text()) if p.exists() else {}
    v.update(state=state,detail=detail,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    status('추가 평가 모델 로딩 및 260개 영상 처리 시작')
    run(sys.executable,ROOT/'tools/evaluate_policy_supplement.py','--mode','run','--data',DATA)
    status('평가 완료; 통합 보고서 및 GitHub 반영')
    summary=json.loads((OUT/'summary.json').read_text())
    # Keep the gate certificate with the published method and evidence.
    gate=json.loads((DATA/'gates.json').read_text())
    (OUT/'gates.json').write_text(json.dumps(gate,indent=2)+'\n')
    report=ROOT/'src/experiments/policy_study_results/REPORT_KO.md'
    marker='## 추가 평가 완료: 주체 특징과 시간 진단'
    body=report.read_text().split('\n'+marker)[0]
    lines=['',marker,'',
      '기존 영상 260개 전체에 대한 사후 탐색 평가를 완료했다. 원본 생성 실험을 반복하지 않았다.',
      '각 값은 프롬프트 안에서 시드를 평균한 뒤 프롬프트 10개를 짝지어 분석했다.',
      '', '| 비트 | 지표 | 후반 P3−P1 FR 차이 | 탐색적 95% 구간 |','|---|---|---:|---|']
    for r in summary['contrasts']:
        if r['arm']=='fr' and r['window']=='late':
            if r['mean'] is None:lines.append(f'| {r["bits"]} | {r["metric"]} | 측정 불가 | 빈 대응 영역이 있는 영상 {r["undefined_videos"]}개; 전체 표본 비교 미정의 |')
            else:lines.append(f'| {r["bits"]} | {r["metric"]} | {r["mean"]:+.6f} | [{r["ci_low"]:+.6f}, {r["ci_high"]:+.6f}] |')
    lines += ['',
      'subject는 높을수록 프레임 간 특징이 비슷하고, warp_error는 낮을수록 움직임 보정 후 잔차가 작다.',
      '두 값 모두 주체 유지나 시간 품질의 완전한 척도는 아니다. 움직임(flow_pixels)과 유효 영역(coverage)을 함께 봐야 한다.',
      '단순 프레임 차이(raw_difference)를 시간 품질 점수로 사용하지 않는다. 동작 소실이 점수를 높일 수 있다.',
      '이 추가 평가는 압축 MP4에 기반하며, 사전 등록한 원본 프레임 MUSIQ와 측정 단계가 다르다.',
      '지표 간 방향이 다르면 전체 영상 품질의 단일 우위로 요약하지 않는다. 확증적 발견이나 새 방법의 입증으로 해석하지 않는다.',
      '', '[추가 평가 상세 결과](../policy_supplement_results/REPORT_KO.md) · [고정한 평가 방식과 출처](../POLICY_STUDY_SUPPLEMENT.md)']
    report.write_text(body+'\n'.join(lines)+'\n')
    readme=ROOT/'README.md';text=readme.read_text()
    link='> 주체·시간 진단 추가 평가 완료: [사후 탐색 결과와 한계](src/experiments/policy_supplement_results/REPORT_KO.md).\n'
    if link not in text:readme.write_text(link+text)
    paths=['README.md',str(report.relative_to(ROOT)),str(OUT.relative_to(ROOT))]
    run('git','diff','--check')
    run('git','add','--',*paths)
    run('git','commit','-m','Integrate subject and temporal diagnostics for all 260 policy-study videos','--',*paths)
    for attempt in range(10):
        done=subprocess.run(['git','push','origin','main'],cwd=ROOT,env=dict(os.environ,GIT_TERMINAL_PROMPT='0'))
        if done.returncode==0:break
        if attempt==9:raise RuntimeError('GitHub push failed; results and local commit retained')
        time.sleep(30)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status('260개 추가 평가·통합 보고서·커밋·GitHub 푸시 완료: '+head,'pass')
    (DATA.parent/'WORK_STATUS.md').write_text('COMPLETE: policy study 80/80; supplementary video evaluation 260/260; Korean integrated report and GitHub push complete.\nCommit: '+head+'\nReport: Quantization/src/experiments/policy_study_results/REPORT_KO.md\n')
    print('SUPPLEMENT PIPELINE COMPLETE',head,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status(f'{type(exc).__name__}: {exc}','failed');raise
