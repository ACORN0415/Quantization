"""CPU-only audit and report of the completed, registered MoE diagnostic."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from moe_allocation_objectives import fixed_multiset_candidates, lexicographic_choice
from moe_allocation_runtime import ROOT, RECORD, REPO, digest_tensor, routing_summary, write_json


def read(name):
    return json.loads((RECORD/f'{name}.json').read_text())


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):
            h.update(block)
    return h.hexdigest()


def median(values):
    return float(statistics.median(values))


def main():
    identity=read('identity');packing=read('packing');calibration=read('calibration')
    diagnostic=read('diagnostic');routing=read('routing');preparation=read('preparation')
    manifest=read('input_manifest');capture_hashes=read('capture_hashes')
    assert identity['status']==packing['status']==preparation['status']=='PASS'
    assert len(identity['documents'])==24 and all(r['bitwise_equal'] and r['max_absolute_error']==0 for r in identity['documents'])
    assert len(packing['projections'])==36
    assert all(r['code_equal'] and r['metadata_equal'] and r['decoded_bits_equal'] for r in packing['projections'])
    tokens=torch.load(ROOT/'tokens.pt',weights_only=True,map_location='cpu')
    captures=torch.load(ROOT/'captures.pt',weights_only=True,map_location='cpu')
    by_token={d['id']:d for d in tokens}
    by_capture={d['id']:d for d in captures}
    assert len(by_token)==len(by_capture)==24
    assert len({d['text_sha256'] for d in manifest['documents']})==24
    for d in manifest['documents']:
        assert digest_tensor(by_token[d['id']]['input_ids'])==d['token_sha256']
    for d in capture_hashes:
        for key in ('x','gates','indices','reference'):
            assert digest_tensor(by_capture[d['id']][key])==d[key]
        assert torch.all(by_capture[d['id']]['gates']>0)
    selected=routing['selected_experts']
    for split,key in [('train','calibration'),('validation','held_out')]:
        assert routing_summary([d for d in captures if d['split']==split],selected)==routing[key]

    candidates=fixed_multiset_candidates()
    for method in ('A','B','C','reference'):
        scores=np.mean([r[method] for r in calibration['documents']],axis=0)
        np.testing.assert_array_equal(scores,calibration['means'][method])
        assert list(lexicographic_choice(scores,candidates)[0])==calibration['choices'][method]
    assert calibration['choices']['B']==calibration['choices']['C']==calibration['choices']['reference']

    # Instantiate the declared fixed-size metadata and count the actual unique
    # tensor storage for every reference allocation, including padding.
    archive=torch.load(ROOT/'packed.pt',weights_only=True,map_location='cpu')
    payloads=[]
    for candidate in candidates:
        payload={'expert_ids':torch.tensor(selected,dtype=torch.int32),
                 'bits':torch.tensor(candidate,dtype=torch.uint8),
                 'format':torch.tensor([8,128,1024,2048,1024,2048,2048,1024],dtype=torch.int32)}
        for e,b in zip(selected,candidate):
            for projection,item in archive[(e,b)].items():
                assert digest_tensor(item['packed'])==next(r['packed_sha256'] for r in packing['projections']
                    if r['expert']==e and r['bits']==b and r['projection']==projection)
                for name in ('packed','scale','zero'):
                    payload[f'{e}.{projection}.{name}']=item[name]
        unique={value.untyped_storage().data_ptr():value.untyped_storage().nbytes() for value in payload.values()}
        payloads.append(sum(unique.values()))
        if list(candidate)==calibration['choices']['B']:
            torch.save(payload,ROOT/'selected_reference_payload.pt')
    assert set(payloads)=={10551296+20+32}
    del archive,payload

    common=read('download_result')['elapsed_seconds']+read('prepare_process')['complete_process_seconds']
    workers={m:[read(f'pair{i}_{m}') for i in (1,2,3)] for m in ('B','C')}
    process={m:[read(f'pair{i}_{m}_process') for i in (1,2,3)] for m in ('B','C')}
    all_names=['warmup_B','warmup_C']+[f'pair{i}_{m}' for i,order in enumerate([('B','C'),('C','B'),('B','C')],1) for m in order]
    all_workers=[read(n) for n in all_names]
    assert len({w['pid'] for w in all_workers})==8
    for w in all_workers:
        assert w['status']=='PASS' and w['bank_hashes']==read('bank_hashes')
        np.testing.assert_allclose(w['scores'],calibration['means']['B'],atol=1e-10,rtol=1e-9)
        assert w['choice']==calibration['choices']['B']
    cost={}
    for m in ('B','C'):
        cost[m]=dict(
            complete_worker_seconds=[p['complete_process_seconds'] for p in process[m]],
            one_shot_seconds=[common+p['complete_process_seconds'] for p in process[m]],
            median_complete_worker_seconds=median(p['complete_process_seconds'] for p in process[m]),
            median_one_shot_seconds=common+median(p['complete_process_seconds'] for p in process[m]),
            median_output_collection_seconds=median(w['output_collection_seconds'] for w in workers[m]),
            median_selection_seconds=median(w['selection_seconds'] for w in workers[m]),
            median_worker_peak_RSS_bytes=median(w['memory']['process_peak_rss_bytes'] for w in workers[m]),
            median_full_peak_RSS_bytes=median(max(preparation['memory']['process_peak_rss_bytes'],w['memory']['process_peak_rss_bytes']) for w in workers[m]),
            worker_peak_GPU_allocated_bytes=max(w['memory']['gpu_peak_allocated_bytes'] for w in workers[m]),
            full_peak_GPU_allocated_bytes=max(preparation['memory']['gpu_peak_allocated_bytes'],max(w['memory']['gpu_peak_allocated_bytes'] for w in workers[m])))
    paired_gain=[1-c/b for b,c in zip(cost['B']['one_shot_seconds'],cost['C']['one_shot_seconds'])]
    gain=1-cost['C']['median_one_shot_seconds']/cost['B']['median_one_shot_seconds']
    rss_gain=1-cost['C']['median_full_peak_RSS_bytes']/cost['B']['median_full_peak_RSS_bytes']
    time_pass=gain>=.10 and all(v>0 for v in paired_gain)
    memory_pass=rss_gain>=.20 and gain>=-.05 and cost['C']['full_peak_GPU_allocated_bytes']<=cost['B']['full_peak_GPU_allocated_bytes']
    assert not diagnostic['allocations_differ']
    assert all(d['A']==d['B'] and d['native_A']==d['native_B'] for d in diagnostic['held_out'])
    assert diagnostic['fp64']['gain']==0 and diagnostic['fp64']['ci95']==[0,0]
    assert not diagnostic['continue_quality']

    progress=[json.loads(line) for line in (RECORD/'progress.jsonl').read_text().splitlines()]
    stages=[p['stage'] for p in progress]
    assert stages.index('input_manifest_frozen')<stages.index('identity_passed')<stages.index('expert_packed')<stages.index('allocations_frozen')<stages.index('heldout_scored')
    assert not any(p['stage'] in ('FAILED','RESOURCE_CAP') for p in progress)
    measurements=[preparation,diagnostic,*all_workers]
    peak_RSS=max(r['memory']['process_peak_rss_bytes'] for r in measurements)
    peak_GPU=max(r['memory']['gpu_peak_allocated_bytes'] for r in measurements)
    assert peak_RSS<48*2**30 and peak_GPU<20*2**30
    disk=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    assert disk<20*2**30
    assert not torch.cuda.is_initialized(), 'Report must not run CUDA'
    sources=['moe_allocation_objectives.py','moe_allocation_runtime.py','moe_allocation_stage1.py','run_moe_allocation_stage1.py','report_moe_allocation_stage1.py']
    preserved={name:dict(bytes=(ROOT/name).stat().st_size,sha256=file_hash(ROOT/name))
               for name in ('tokens.pt','captures.pt','block.pt','packed.pt','selected_reference_payload.pt')}
    audit=dict(status='PASS',audit_scope='CPU tensor/hash/accounting/routing/scalar reaggregation; no repeated GPU forwards',
               source_sha256={name:file_hash(REPO/'tools'/name) for name in sources},
               actual_input_files=preserved,identity_documents=24,packing_projections=36,
               full_reference_candidate_bytes=payloads,isolated_worker_pids=[w['pid'] for w in all_workers],
               fresh_workers_all_bank_hashes_equal=True,stage_order_valid=True,
               calculation_execution_commits=dict(prepare='8ec4374',diagnose='9856213',cost='9856213'),
               GPU_used_by_report=False)
    write_json(RECORD/'audit.json',audit)
    end=datetime.now(timezone.utc)
    state=read('run_state')
    state.update(status='COMPLETE_VALID_DIAGNOSTIC',completed_utc=end.isoformat(),
                 elapsed_wall_seconds=end.timestamp()-state['start_epoch'],
                 valid_preparation_runs=1,valid_diagnostic_runs=1,warmup_workers=2,timed_workers=6,
                 invalid_runs=0,incomplete_runs=0,continue_quality=False,continue_cost=bool(time_pass or memory_pass))
    assert end.timestamp()<state['deadline_epoch']
    write_json(RECORD/'run_state.json',state)
    summary=dict(status='COMPLETE',decision='Close this restricted candidate: no allocation improvement and no registered cost benefit',
        selected_experts=selected,allocation=calibration['choices']['B'],allocations_differ=False,
        fp64_heldout=diagnostic['fp64'],native_heldout=diagnostic['native'],
        native_mean_error=diagnostic['native_mean_A'],
        native_execution=dict(unique_allocations=1,document_forwards=16,B_reuses_A=True,
                              explanation='Same selected allocation: evaluated once per held-out document, not two independent native executions'),
        interval_interpretation='Degenerate zero interval because the allocations are identical; not a population equivalence claim',
        routing=routing,common_preparation_seconds=common,cost=cost,
        paired_one_shot_C_gain=paired_gain,median_total_C_gain=gain,full_RSS_C_gain=rss_gain,
        continue_quality=False,continue_cost=bool(time_pass or memory_pass),
        measured_limits=dict(peak_process_RSS_GiB=peak_RSS/2**30,peak_GPU_allocated_GiB=peak_GPU/2**30,
                             incremental_disk_GiB=disk/2**30,elapsed_wall_seconds=state['elapsed_wall_seconds']),
        gates=dict(identity_documents=24,packing_projections=36,
                   max_B_reference=max(r['max_B_reference'] for r in calibration['documents']),
                   max_C_reference=max(r['max_C_reference'] for r in calibration['documents'])),
        scope='OLMoE-0924 layer 8, four co-routing-selected experts, 12 permutations of fixed bit multiset; not whole-model or all-MoE evidence')
    write_json(RECORD/'summary.json',summary)
    write_report(summary,state)
    print(json.dumps(dict(status='PASS',summary=summary),ensure_ascii=False,indent=2))


def write_report(s,state):
    B,C=s['cost']['B'],s['cost']['C']
    held=s['routing']['held_out']
    pair_lines='\n'.join(f"  {key}: {value:,}토큰" for key,value in held['pair_counts'].items())
    report=f'''MoE 배분 선택 제한 진단 결과
2026-10-31

판단: 이번 후보의 품질 개선 방향과 계산 가속 방향을 모두 종료한다.
A와 B가 고른 배분이 같았고, C도 등록한 총시간·최대 메모리 기준을 통과하지 못했다.
이는 한 층·선택한 네 전문가·고정 비트 구성의 판단이다. MoE 양자화 전체나
모든 교차항 활용 방법에 이득이 없다는 결론은 아니다.

요청한 네 질문
1. A와 B의 선택은 달랐는가? 아니다.
   전문가 [5, 11, 37, 45]에 A/B/C 모두 [3, 2, 3, 4]비트를 선택했다.
   각 목적함수의 최솟값 후보는 하나였다. 동점 규칙이 서로 다른 선택을 숨긴 것이 아니다.
2. 보류 문서의 개선은 얼마인가? 16문서에서 0.000%, 문서 bootstrap 95% 구간 [0, 0]%다.
   같은 배분을 비교하므로 생긴 퇴화 구간이다. 새로운 입력 전체에서 효과가 0임을
   정밀하게 추정한 결과나 모집단 동등성 증명으로 해석하지 않는다.
3. native BF16에서도 감소가 남았는가? 배분이 같아 감소가 없다.
   실제 native 오차는 토큰별 squared L2의 평균 {s['native_mean_error']:.12g}다.
   같은 배분이므로 각 보류 문서에서 native 블록을 한 번 실행해 A/B에 공유했다.
   16개를 각각 두 번 독립 실행해 동등성을 입증했다고 쓰지 않는다.
4. C가 준비 비용을 포함해 B보다 유리했는가? 등록한 기준을 통과하지 못했다.

시간: 3쌍의 독립 프로세스 측정, 순서는 B-C / C-B / B-C.
  공통 준비: {s['common_preparation_seconds']:.3f}초.
  구성은 모델 다운로드 177.521초와 준비 프로세스 전체 28.048초다.
  입력·모델 준비·무개입 검증·양자화·패킹·저장을 포함하며 사람의 코드 작성 대기는 포함하지 않는다.
  준비 포함 일회 총시간 중앙값: B {B['median_one_shot_seconds']:.3f}초 / C {C['median_one_shot_seconds']:.3f}초.
  공통 준비 이후 독립 작업 전체: B {B['median_complete_worker_seconds']:.3f}초 / C {C['median_complete_worker_seconds']:.3f}초.
  전문가 출력 수집 중앙값: B {B['median_output_collection_seconds']:.3f}초 / C {C['median_output_collection_seconds']:.3f}초.
  목적함수 평가 중앙값: B {B['median_selection_seconds']:.3f}초 / C {C['median_selection_seconds']:.3f}초.
  대응 총시간 감소율(C 기준): {', '.join(f'{100*x:+.4f}%' for x in s['paired_one_shot_C_gain'])}.
  총시간 차이는 작고 방향도 일관되지 않는다. 이 작은 차이를 유의한 가속/감속으로 주장하지 않는다.
  다운로드를 공통 비용에서 제외해도 독립 작업 전체에서 C의 일관된 이득은 없다.

메모리: 각 B/C 작업의 PID가 다르며 앞 실행의 최대 RSS를 물려받지 않았다.
  평가 작업 최대 RSS 중앙값: B {B['median_worker_peak_RSS_bytes']/2**30:.6f} GiB / C {C['median_worker_peak_RSS_bytes']/2**30:.6f} GiB.
  준비 단계를 포함한 최대 RSS: 두 경로 모두 {B['median_full_peak_RSS_bytes']/2**30:.6f} GiB.
  준비 단계 포함 GPU allocated 최대: 두 경로 모두 {B['full_peak_GPU_allocated_bytes']/2**30:.6f} GiB.
  준비 중에는 전체 BF16 모델이 상주했고, 평가 작업에는 대상 블록·후보 가중치·보정 입력이 상주했다.
  B와 C 모두 한 문서의 오차만 보유하고 64토큰씩 평가했다. 전체 자료의 오차 캐시를 B에만 강제하지 않았다.
  경로별 생성·전송·캐시/교차항 계산 비용과 각 시점의 상주 상태는 pair*_*.json에 있다.

해석 범위와 실제 공동 선택
모델은 OLMoE-1B-7B-0924의 0부터 세는 8번 층이다. 전문가 네 개는 보정
routing의 gate 가중 공동 선택 관계가 큰 집합을 골랐다. 대표성 있는 무작위 표본이 아니다.
나머지 60개 전문가는 BF16이다. 8개 보정 문서 4,096토큰으로 배분을 고르고,
다른 16개 문서 8,192토큰에서 확인했다. 보류 결과로 전문가나 배분을 다시 고르지 않았다.
보류 토큰 중 네 전문가의 둘 이상이 선택된 토큰은 {held['tokens_with_at_least_two']:,}/{held['tokens']:,}
({100*held['tokens_with_at_least_two']/held['tokens']:.2f}%)다. 쌍별 공동 선택은 다음과 같다.
{pair_lines}
공동 선택이 없어서 검사가 공허했던 것은 아니다. 그렇더라도 이번 평균 배분의 선택 이득은 나오지 않았다.

실행 유효성
측정 전 승인 보완은 d04fbec, 실행 코드는 8ec4374에 커밋했다. 독립 작업의
종료 시각 계측과 RSS 최고값 감시는 9856213에서 보완했으며 배분/비용 결과 전에 적용했다.
24개 입력 모두 원래 MoE 블록과 무개입 재조립 출력의 형상·dtype·원소 비트가 같았다.
실제 가중치의 4전문가 × 3비트폭 × 3행렬 = 36개 패킹 왕복이 통과했다.
12개 후보의 실제 code/scale/zero는 각각 10,551,296바이트다. 실제 할당한 선택
메타데이터 20바이트와 공통 형식 메타데이터 32바이트까지 포함하면 모두 10,551,348바이트다.
파일 컨테이너와 나머지 BF16 모델 저장량은 별도이며 전체 모델 압축률을 주장하지 않는다.
보정의 B/C 점수와 독립 합산의 최대 절대 차이는 각각
{s['gates']['max_B_reference']:.3g} / {s['gates']['max_C_reference']:.3g}였다.
워밍업 2개와 측정 6개는 같은 입력 오차 은행의 전체 바이트 해시와 같은 배분을 확인했다.

자원과 종료
시작: {state['start_utc']} (한국 시각 21:17:30).
검증·보고 생성 종료: {state['completed_utc']}.
코드 준비·다운로드·실행·감사를 포함한 경과: {state['elapsed_wall_seconds']/60:.2f}분 / 상한 120분.
준비 1회·진단 1회·워밍업 2회·측정 6회 유효 완료, 무효·미완료·재시도 0회.
모델 prefix 실행 24회이며 전체 모델 perplexity·과제 평가나 영상 생성은 수행하지 않았다.
실측 최대 RSS {s['measured_limits']['peak_process_RSS_GiB']:.3f} GiB,
GPU allocated {s['measured_limits']['peak_GPU_allocated_GiB']:.3f} GiB,
추가 디스크 {s['measured_limits']['incremental_disk_GiB']:.3f} GiB로 각 상한 안에서 끝났다.
큰 입력·블록 가중치·packed 후보는 /home/acorn0415/sfq_data/moe_allocation_stage1에
보존하며 작은 결과와 코드만 Git에 둔다. audit.json에 파일과 실행 코드 해시가 있다.

후속 판단
현재 검사에서 개별 오차 기준선 A로도 같은 배분을 골랐고, 현재 FP64 CPU 평가기의
C 전처리가 이 12개 후보에 비용 이득을 만들지 못했다. 등록대로 이 후보의 개발을 종료한다.
다른 층·전문가 집합·비트 구성이나 전체 모델 평가를 추가해 결과를 구제하지 않는다.
'''
    (RECORD/'REPORT_KO.txt').write_text(report)


if __name__=='__main__':
    main()
