"""Report existing stage-1 artifacts, without starting GPU work."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'src/experiments/kv_reuse_stage1'
DATA=Path.home()/'sfq_data/kv_reuse_stage1'


def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    s=read(OUT/'summary.json');m=read(OUT/'manifest.json')
    results=[read(OUT/(name+'.json')) for name in s['results']]
    byname={r['name']:r for r in results}
    for name,h in m['own_files'].items():
        assert sha(ROOT/name)==h,name
    for r in results:
        assert sha(DATA/r['name']/'latent.pt')==r['latent_sha256'],r['name']
    rows=[]
    for r in results:
        rows.append(dict(name=r['name'],arm=r['arm'],mode=r['mode'],generation_s=r['generation_s'],
                         peak_allocated_gib=r['peak_allocated']/2**30,
                         peak_reserved_gib=r['peak_reserved']/2**30,
                         process_sample_peak_gib=r['process_sample_peak']/2**30,
                         decoder_calls=r['counts']['decode_calls'],latent_equal_to_U=r['latent_equal_to_U']))
    if rows:
        with (OUT/'measurements.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    decisions={
      'stop_performance_policy_for_this_setting':'이 설정의 복잡한 성능 정책 연구는 종료한다. H2를 추가 실행하지 않는다.',
      'inconclusive_do_not_expand':'판단 불가. 단순 규칙을 넘어선 연구 확장은 보류하며 자동 추가 실행하지 않는다.',
      'consider_H2_separate_registration':'별도 예산·모델·미사용 조건을 봉인한 뒤 H2 검토를 진행할 근거가 있다.',
      'incomplete_or_invalid':'정확성/실행 유효성 게이트가 완료되지 않아 성능 판단을 보류한다.'}
    lines=['# QVG 복원값 재사용 단계1 진단 보고','',
      f'**판정: {decisions[s["decision"]]}**','',
      f'종료 사유: `{s["reason"]}`. 새 생성 시도 {len(s["budget"]["attempts"])}회, '
      f'유효 완료 {s["completed"]}회, 단계 경과 {s["budget"]["elapsed_s"]/60:.2f}분. '
      '상한은12회·120분이며 native-span gate와 모델 연결·검증·계측·warmup을 포함한다.','',
      '## 고정 조건','',
      'RTX4090, Self-Forcing1.3B, 기존 개발 p00/seed303,36잠재프레임. QVG 공식 revision '
      f'`{m["revision"]}`, INT2/group64/centroids256/1-stage/k-means2회, '
      '8생성청크 단위 압축이다. 모델·코덱·문맥·RoPE·attention·원본 난수 일정을 고정했다.','',
      'U는 공식 read, B는 중복 decoder 호출만 제거, R은 B에서1GiB pool에 완전한 K/V '
      'span을 처음4층까지 유지한다. 주 비교는B/R이다. 모든 팔에서 진단용 CPU offload/reload를 '
      '끄고 text를 선계산했다. VAE는 실행하지 않았다. 타이밍은 pool/cache 초기화부터 최종 '
      'latent까지이며 최초 복원·압축·연결·관리 비용을 포함한다.','',
      '질문은 **복원한 값을 추가 메모리에 보관하면, 다음 읽기에서 복원을 생략하는 이득이 '
      '전체 생성 시간에서도 남는가**이다. U→B는 중복 계산 제거이고, B→R이 재사용의 '
      '추가 효과다. U의 호출 수 감소를 R의 성능 개선으로 계산하지 않는다.','',
      '```text\n'
      'U: packed span → decoder 두 번 → 복원값 연결 → RoPE/attention\n'
      'B: packed span → decoder 한 번 → 복원값 연결 → RoPE/attention\n'
      'R: 보관값이 있으면 사용 / 없으면 B처럼 복원 후 보관 → 연결 → RoPE/attention\n'
      '```','',
      '## 정확성과 실제 호출','']
    lines += ['| 실행 | decoder 호출 | attention 입력 검사 | 최종 latent |',
              '|---|---:|---|---|']
    for name in ('gate_U','gate_B','gate_R'):
        if name not in byname:lines.append(f'| {name} | — | 미완료 | 미확인 |');continue
        r=byname[name]
        tape=r['tape']
        text=f'{tape["records"]} tensors 원소 바이트 '+('기록' if name=='gate_U' else '직접 일치')
        lines.append(f'| {name} | {r["counts"]["decode_calls"]} | {text} | '+
                     ('기준' if name=='gate_U' else str(r['latent_equal_to_U']))+' |')
    lines += ['', '입력 검사는 실제 native self-attention의 Q/K/V 전체를 대상으로 한다. 무손실 '
       'prefix-delta/zstd 테이프를 복원해 전체 원소 바이트를 비교하며 해시만 비교하지 않는다. '
       '타이밍에서는 이 검사를 끈다. 생성60forward×30층이며 decoder 호출은 실제 entry에서 계수한다.', '',
       '**검증 실행의 시간은 성능 비교에 쓰지 않는다.** gate_U/B/R의 `generation_s`에는 '
       '입력 테이프의 저장·복원·CPU 전송·전 원소 비교가 포함된다. 결과 JSON의 '
       '`time_scope`에 있는 파일 I/O 제외 문구는 latent 저장과 일반 timed 경계에 대한 '
       '설명이며, 검증 테이프 I/O까지 제외했다는 뜻으로 읽으면 안 된다. U는 별도 timed '
       '표본이 없으므로 U→B의 전체 생성 가속률도 이번 자료로 추정하지 않는다. '
       '모든 timed 실행에도 공통 청크 진행 로그를 기록하는 비용은 포함된다.', '',
       '## 메모리와 전체 생성 시간','',
       '공통 프로세스 상한20GiB, allocator cap18GiB, R pool1GiB. 동일 상한과 동일 실측 '
       '사용량은 다른 조건이다. 아래는 별도 검증·profile·warmup을 제외한 timed 실행이다.', '',
       '| 실행 | 생성 초 | allocated peak GiB | reserved peak GiB | NVML 표본 최대 GiB |',
       '|---|---:|---:|---:|---:|']
    for r in rows:
        if r['mode']=='timed':lines.append(f'| {r["name"]} | {r["generation_s"]:.4f} | '
          f'{r["peak_allocated_gib"]:.4f} | {r["peak_reserved_gib"]:.4f} | {r["process_sample_peak_gib"]:.4f} |')
    timed=[r for r in results if r['mode']=='timed']
    if timed:
        lines += ['', f'위 timed 실행 중 GPU 온도 표본 범위는 '
                  f'{min(r["temp_min"] for r in timed)}–{max(r["temp_max"] for r in timed)}°C, '
                  f'SM 클럭 표본 범위는 {min(r["clock_min"] for r in timed)}–'
                  f'{max(r["clock_max"] for r in timed)} MHz였다. '
                  '상한 위반이나 외부 GPU 작업이 감지되면 실행 전체를 중단하도록 했다.']
    lines += ['', 'NVML은0.5초 표본이므로 연속적인 실제 peak의 정확한 관측이라고 주장하지 않는다. '
              '실행 순서는BR/RB/BR이며 표의 시간은 계측을 켠 profile 실행 시간이 아니다.', '',
              '| 대응 쌍 | B 초 | R 초 | R의 시간 감소율 |','|---|---:|---:|---:|']
    for p in s['pairs']:lines.append(f'| {p["pair"]} | {p["B_s"]:.4f} | {p["R_s"]:.4f} | {p["reduction_fraction"]*100:.3f}% |')
    if s['pairs']:
        x=[p['reduction_fraction']*100 for p in s['pairs']]
        lines += ['',f'대응 감소율 중앙값 {statistics.median(x):.3f}%, 범위 [{min(x):.3f}%, {max(x):.3f}%]. '
                   '3쌍은 제한된 진단이며 모집단의 유의한 속도 향상을 입증하는 표본으로 취급하지 않는다.']
        lines += ['', '감소율은 `100 × (B 시간 − R 시간) / B 시간`이다. 양수는 R이 빠르고 '
                  '음수는 R이 느리다는 뜻이다. 5%는 사전 등록한 연구 진행 기준이며, '
                  '통계적 유의성의 경계가 아니다.']
    held=byname.get('gate_R')
    if held:
        p=held['counts']['pool']
        lines += ['', f'R의 보관 공간은 {p["capacity_bytes"]/2**30:.3f} GiB를 선할당했고, '
                  f'그중 완전한 K/V 쌍 {p["pairs"]}개에 {p["reserved_pair_bytes"]/2**30:.3f} GiB를 배정했다. '
                  f'검증 실행의 실제 hit는 {p["hits"]}회, miss는 {p["misses"]}회였다. '
                  'miss에는 보관 예산 밖의 층에서 매번 복원한 경우가 포함된다.']
    if s['optimistic']:
        o=s['optimistic'];profile=byname['profile_B']
        lines += ['', '## 프로파일과 진행 판단','',
        f'별도 profile_B 전체 생성 시간은 {profile["generation_s"]:.4f}초다. '
        f'그 안에서 decoder event 구간 합은 {profile["counts"]["event_seconds"]["decoder"]:.4f}초, '
        f'decode+layout 구간 합은 {profile["counts"]["event_seconds"]["decode_and_layout"]:.4f}초였다. '
        '두 구간은 포함 관계이므로 더하지 않는다.', '',
        f'별도 B 프로파일에서 decoder 구간은 전체의 {100*o["decoder_fraction"]:.3f}%, '
        f'decode+layout 구간은 {100*o["decode_layout_fraction"]:.3f}%였다. '
        f'현재 pool이 제거하는 호출 비율152/1200을 곱한 추정은 {100*o["scaled_decode_layout_fraction"]:.3f}%다.', '',
        '이는 층별 비용이 균일하다는 가정의 낙관적 비용 추정이며 엄밀한 절약 상한이 아니다. '
        '중첩된 event 구간을 합산하지 않았다. 이 추정과 실제 전체 시간 차이를 함께 보고 판단한다.']
    if s['decision']=='stop_performance_policy_for_this_setting':
        lines += ['', '## 연구 판단','',
          '등록한 세 대응 쌍의 시간 감소율과 프로파일 추정이 모두 5% 진행 기준 미만이다. '
          '따라서 **이 입력·길이·1 GiB 예산에서 복잡한 선택기를 개발하는 단계로 진행하지 않는다.** '
          '작은 실제 구현으로 확인한 이득과 제거 가능한 비용의 추정이 모두 연구 확장을 '
          '뒷받침하지 못했다. 다른 문맥 길이나 더 큰 유지 예산의 불가능성을 증명한 것은 아니다.', '',
          '남는 결과는 동일 출력을 유지하는 단순 구현과 그 비용의 측정이다. 공식 read의 '
          '중복 복원을 제거할 수 있다는 관측은 구현 정리의 근거이지만, U의 별도 timed '
          '측정이 없으므로 그 자체의 가속률은 보고하지 않는다. 이 진단만으로 새 양자화 '
          '방법이나 논문 신규성을 확보했다고 판단하지 않는다.']
    elif s['decision']=='inconclusive_do_not_expand':
        lines += ['', '## 연구 판단','',
          '실제 대응 시간과 프로파일 추정이 함께 진행 조건 또는 종료 조건을 만족하지 않았다. '
          '따라서 성공으로 분류하거나 추가 조건을 붙여 탐색을 연장하지 않는다. 이번 진단의 '
          '결론은 판단 불가이며, 비용 모델의 예측력이나 동적 선택기의 필요성은 미검증이다.']
    lines += ['', '## 해석 범위','',
      '- 한 입력·한 생성 길이·한 유지 예산·한 GPU의 진단이다. 다른 조건의 가능성을 부정하지 않는다.',
      '- QVG 복원 후 attention 경로의 결과이며 동일 코덱 fused attention과 비교하지 않았다.',
      '- 유지 공간은 추가 메모리 사용이다. 같은 실측 사용량에서의 우위나 Pareto 개선으로 바꾸지 않는다.',
      '- H2의 구성용/미사용 조건 예측이나 regret은 이번 단계에서 수행하지 않았다.',
      '- 코드·입력·예산·무효/미완료 기록은 아래 자료에 남기며 완료한 증거만 주장한다.', '',
      '## 실행 상태와 보존','',
      f'시작: {s["budget"]["started"]}. 종료 보고: {s["budget"]["reported"]}. '
      f'시도 후 유효 완료되지 않은 실행: '
      f'{sum(a.get("status")!="valid" for a in s["budget"]["attempts"])}회. '
      f'등록한 12회 중 미시도: {12-len(s["budget"]["attempts"])}회. '
      '실행 중 소스·설정 지문을 반복 확인했으며, 보고서 생성 시 소스와 저장 latent의 '
      '파일 해시를 다시 확인했다. 파일 해시 검사는 위 원소 바이트 검사를 대신하지 않는다.', '',
      '[등록](REGISTRATION_KO.md) · [요약](summary.json) · [수치표](measurements.csv) · '
      '[소스/입력 지문](manifest.json) · [native-span 검사](unit_gates.json)', '',
      '[실행 패키지 버전](environment.json). 모델 설정은 [model_setup.json](model_setup.json), '
      '각 실행의 상세 집계는 이름이 같은 JSON에 있다. 원자료 디렉터리에는 실행별 NVML '
      '표본과 latent, 전체 실행 로그도 보존한다.', '',
      '[계정 폴더 정리 기록](DISK_CLEANUP_KO.md). 검증 도중 승인받은 미사용 인덱스와 '
      '불완전한 Git 임시 pack, pip 캐시를 정리했다. 성능 측정 전에 완료했으며 연구 원자료는 보존했다.', '',
      f'큰 원자료와 입력 테이프: `{DATA}`. 로컬 보존이며 원격 백업을 뜻하지 않는다.']
    if (OUT/'verification.json').exists():
        audit=read(OUT/'verification.json')
        lines += ['', '[저장 증거 재검증](verification.json): CPU에서 기준을 포함한 '
          f'{len(audit["completed_cases"])}개 latent를 다시 읽어 형상·dtype·원소 바이트를 '
          '비교했다. 소스 지문, 저장 호출 집계, NVML 표본, 테이프 색인도 확인하고 '
          '테이프 파일의 SHA-256을 보존했다. GPU 연산을 재실행하거나 attention 입력 '
          '전체를 두 번째로 직접 비교한 검증은 아니다.', '',
          '재현용 명령(저장된 원자료 필요):', '',
          '```bash\n'
          '~/sfq_data/py tools/verify_kv_reuse_stage1.py\n'
          'python tools/report_kv_reuse_stage1.py\n'
          '```']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(s['decision'])


if __name__=='__main__':main()
