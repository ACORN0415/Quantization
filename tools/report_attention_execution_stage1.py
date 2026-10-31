"""Audit the frozen attention experiment and write its bounded conclusion."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics as st


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'src/experiments/attention_execution_stage1'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    regpath = OUT / 'registration.json'
    rawpath = OUT / 'run1/results.json'
    reg, raw = load(regpath), load(rawpath)
    assert raw['status'] == 'completed' and not raw['errors']
    assert raw['registration_sha256'] == sha(regpath)
    assert raw['script_sha256'] == reg['script_sha256'] == sha(ROOT / 'tools/attention_execution_stage1.py')
    assert reg['attention_source_sha256'] == sha(Path(reg['attention_source']))
    schedule = load(ROOT / reg['schedule_source'])['schedule']
    counts = Counter(x['start'] + 4680 for x in schedule)
    assert reg['shape_calls'] == {str(k): counts[k] * 30 for k in reg['k_lengths']}
    assert sum(reg['shape_calls'].values()) == 600
    assert len(raw['small_gates']) == 12
    assert all(x['passed'] and x['shape_matches'] for x in raw['small_gates'])
    assert len(raw['shapes']) == 4 and len(raw['replay_rounds']) == 3
    for row in raw['shapes']:
        assert all(g['passed'] and g['shape_matches'] for g in row['large_gates'].values())
        for name, samples in row['evaluation'].items():
            assert len(samples) == 20
            assert all(x['wall_ms'] > 0 and x['cuda_interval_ms'] > 0 for x in samples)
            assert row['profiles'][name]['cuda_kernels']
        choice = min(row['selection'], key=lambda name: st.median(
            x['wall_ms'] for x in row['selection'][name]['samples']))
        assert choice == row['selected_before_evaluation']
        for name, key in [('external_fa2', 'flash_attn::_flash_attn_varlen_forward'),
                          ('sdpa_flash', 'aten::_scaled_dot_product_flash_attention'),
                          ('sdpa_efficient', 'aten::_scaled_dot_product_efficient_attention'),
                          ('sdpa_auto', 'aten::_scaled_dot_product_flash_attention')]:
            assert key in row['profiles'][name]['operators']
    assert sum(raw['call_counts'].values()) == 7704
    assert not any(t['other_compute_pids'] for t in raw['telemetry'])
    events = [json.loads(x) for x in (OUT / 'run1/progress.jsonl').read_text().splitlines()]
    for kn in reg['k_lengths']:
        selected = next(i for i, x in enumerate(events)
                        if x['event'] == 'selection_frozen' and x['k_tokens'] == kn)
        evaluated = next(i for i, x in enumerate(events)
                         if x['event'] == 'shape_completed' and x['k_tokens'] == kn)
        assert selected < evaluated
    summaries = {}
    for name in raw['policies']:
        values = [r['policies'][name] for r in raw['replay_rounds']]
        assert all(v['calls'] == 600 for v in values)
        for v in values:
            assert abs(v['cost_inclusive_ms'] - v['replay_wall_ms'] - v['one_time_cost_ms']) < 1e-8
        cost = [v['cost_inclusive_ms'] for v in values]
        relative = [100 * (r['policies'][name]['cost_inclusive_ms'] /
                           r['policies']['fixed_external_fa2']['cost_inclusive_ms'] - 1)
                    for r in raw['replay_rounds']]
        summaries[name] = dict(median_ms=st.median(cost), min_ms=min(cost), max_ms=max(cost),
            median_replay_only_ms=st.median(v['replay_wall_ms'] for v in values),
            median_percent_vs_fixed=st.median(relative), per_round_percent_vs_fixed=relative)
    fixed_vs_reference = [100 * (r['policies']['fixed_external_fa2']['cost_inclusive_ms'] /
                                 r['policies']['posthoc_reference']['cost_inclusive_ms'] - 1)
                          for r in raw['replay_rounds']]
    # This is the pre-registered sufficient stopping condition, not a test that
    # all possible selectors/inputs have been exhausted.
    assert all(x <= 5 for x in fixed_vs_reference)
    checked = datetime.now(timezone.utc)
    assert datetime.fromisoformat(raw['finished_at']) < datetime.fromisoformat(reg['overall_deadline'])
    assert raw['elapsed_seconds'] < reg['gpu_seconds_limit']
    summary = dict(status='closed_current_scope', checked_at=checked.isoformat(),
        decision_ko='현재 네 형상에서는 고정 FA2 경로로 충분하다. 복잡한 실행 선택기 개발로 진행하지 않는다.',
        action_ko='현재 생성 경로의 backend 설정을 유지한다. 다음 MoE 후보는 별도 검토 자료로 남기며 이번 검사에 혼합하지 않는다.',
        software=raw['environment'], tested_calls=600, full_schedule_calls=1800,
        covered_k_lengths=reg['k_lengths'], replay_rounds=3, policies=summaries,
        selector_cost_ms=raw['selector_overhead_ms'], fixed_vs_reference_percent=fixed_vs_reference,
        per_shape=[dict(k_tokens=r['k_tokens'], selected=r['selected_before_evaluation'],
            median_wall_ms={n: st.median(x['wall_ms'] for x in a) for n, a in r['evaluation'].items()})
            for r in raw['shapes']],
        correctness=dict(small_fp32_reference_comparisons=12, large_nontrivial_comparisons=12,
            large_self_checks=4, rtol=reg['rtol'], atol=reg['atol'],
            max_small_abs=max(x['max_abs'] for x in raw['small_gates']),
            max_large_abs=max(g['max_abs'] for r in raw['shapes'] for g in r['large_gates'].values())),
        resource=dict(preparation_seconds=(datetime.fromisoformat(raw['started_at']) -
            datetime.fromisoformat(reg['preparation_started_at'])).total_seconds(),
            gpu_phase_seconds=raw['elapsed_seconds'],
            overall_through_audit_seconds=(checked-datetime.fromisoformat(reg['preparation_started_at'])).total_seconds(),
            attention_calls=sum(raw['call_counts'].values()), model_generations=0,
            torch_peak_allocated_gib=raw['peak_allocated_bytes'] / 2**30,
            torch_peak_reserved_gib=raw['peak_reserved_bytes'] / 2**30),
        limitations=[
            '합성 고정 입력을 반복한 연산 검사다. 실제 여러 층의 작업 집합·출력 품질을 재현하지 않았다.',
            '첫 형상 호출은 작은 정확성 검사 뒤의 관측이다. 완전한 프로세스 cold start가 아니다.',
            '600회 재생 자체는 직접 계측했지만 준비 비용을 포함한 값은 별도 계측값을 합산한 것이다.',
            '사후 참고는 평가 묶음에서 형상별로 고른 경로다. 실제 재생에서 최단이라는 보장이나 엄밀한 하한이 아니다.',
            '고정 경로 대비 25.16%는 선택 비용을 포함한 해당 600회 묶음의 차이다. 모든 장기 서비스의 비용이나 전체 생성 지연이 아니다.',
            '선택 결과를 여러 작업에서 재사용하면 선택 비용은 더 분산된다. 이번에는 단 한 번의 보정 비용을 한 묶음에 부과했다.',
            '선택 비용을 제외한 경로 재생도 이번 세 라운드에서는 고정 FA2보다 빠르지 않았다. 장기 재사용의 보편적 우열을 주장하지 않는다.',
            '네 길이만 검사했다. 미측정 여덟 길이의 성능을 보간하지 않는다.',
            '계측 시 replay 전후 GPU SM clock은 2685MHz로 기록됐으나 연속 측정·고정은 아니다.',
            '모든 GPU attention 구현이나 다른 장비에서의 실행 선택 문제를 종료하는 결론이 아니다.'
        ])
    write('summary.json', summary)
    tracked = [regpath, rawpath, OUT/'run1/progress.jsonl', OUT/'run1.log',
               ROOT/'tools/attention_execution_stage1.py', Path(reg['attention_source']),
               ROOT/reg['schedule_source']]
    write('verification.json', dict(checked_at=checked.isoformat(), valid=True,
        checks=['사전 등록과 소스 해시 일치', '선택 기록이 평가보다 앞섬',
                '합성 입력 정확성 기준 통과', '네 경로의 실제 CUDA 커널 식별',
                '7704회 실제 attention 호출 집계와 계획 수 일치', '세 라운드 각각600회 재생',
                '실행 중 외부 GPU 작업 없음', 'GPU30분·전체90분 상한 이내'],
        input_sha256={str(p): sha(p) for p in tracked}))
    lines = [
        'Attention 실행 선택 진단 결과', '',
        summary['decision_ko'], summary['action_ko'], '',
        '조건: RTX4090, BF16, B=1/H=12/D=128, Q=4680, K=4680/18720/37440/56160.',
        '기존 스케줄의 해당 길이별150회, 총600회 합성 입력 재생. 순서를 바꿔3라운드.',
        '전체 생성시간 측정이나 영상 품질 검사가 아니다.', '',
        '선택/최초 형상 비용 포함 경과시간(3라운드 중앙값):'
    ]
    labels = dict(fixed_external_fa2='현재 FA2 고정', sdpa_auto='SDPA 자동',
                  measured_selector='직접 측정 선택', posthoc_reference='사후 참고 조합')
    for name, value in summaries.items():
        lines.append(f"- {labels[name]}: {value['median_ms']/1000:.6f}초 "
                     f"(범위{value['min_ms']/1000:.6f}~{value['max_ms']/1000:.6f}), "
                     f"고정 대비{value['median_percent_vs_fixed']:+.3f}%")
    lines += ['', f"선택 과정의 추가 비용: {raw['selector_overhead_ms']/1000:.6f}초.",
              'SDPA 자동은 네 형상 모두 PyTorch FlashAttention 경로를 실행했다.',
              '준비 비용을 제외한 직접 측정 선택의 재생 중앙값은 '
              f"{summaries['measured_selector']['median_replay_only_ms']/1000:.6f}초였다.",
              '이번 범위에서는 측정 선택으로 회수할 시간 이득을 확인하지 못했다.', '',
              '검증: 작은 FP32 기준 비교12개, 큰 입력의 실질 대조12개 모두 고정 허용오차 통과.',
              f"GPU 단계 {raw['elapsed_seconds']:.2f}초, attention 호출7704회, 모델 생성0회.",
              '', '해석 범위:']
    lines += ['- ' + x for x in summary['limitations']]
    (OUT/'REPORT_KO.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(decision=summary['decision_ko'], policies=summaries,
                         resource=summary['resource']), ensure_ascii=False))


if __name__ == '__main__':
    main()
