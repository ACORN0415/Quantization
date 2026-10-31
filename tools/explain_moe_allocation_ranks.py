"""Post-hoc description of frozen calibration scores; standard library only.

Does not run a model, change decision criteria, or rewrite the original audit,
summary, registration, or completion timestamps.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'src/experiments/moe_allocation_stage1'
REGISTRATION = REPO / 'src/experiments/moe_allocation_baselines/registration.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    preserved = [ROOT / name for name in ('calibration.json', 'summary.json', 'run_state.json', 'audit.json')]
    preserved.append(REGISTRATION)
    hashes = {str(path.relative_to(REPO)): digest(path) for path in preserved}
    data = json.loads((ROOT / 'calibration.json').read_text())
    registration = json.loads(REGISTRATION.read_text())
    a, b = data['means']['A'], data['means']['B']
    candidates = data['candidates']
    assert len(a) == len(b) == len(candidates) == 12
    for method in ('A', 'B'):
        for i, value in enumerate(data['means'][method]):
            independently_averaged = math.fsum(d[method][i] for d in data['documents']) / len(data['documents'])
            assert math.isclose(value, independently_averaged, rel_tol=1e-12, abs_tol=1e-16)
    assert all(math.isfinite(value) and value > 0 for value in a + b)
    orders = {method: sorted(range(12), key=lambda i: (data['means'][method][i], candidates[i]))
              for method in ('A', 'B')}
    best, runner_up = orders['B'][:2]
    net = [(bv - av) / bv for av, bv in zip(a, b)]
    tie_rule = registration['objectives']['tie_rule']
    gaps = {}
    for method, order in orders.items():
        scores = data['means'][method]
        adjacent = [scores[right] - scores[left] for left, right in zip(order, order[1:])]
        separated = all(scores[right] - scores[left] > tie_rule['atol'] + tie_rule['rtol'] * max(abs(scores[left]), abs(scores[right]))
                        for left, right in zip(order, order[1:]))
        gaps[method] = dict(minimum_adjacent_gap=min(adjacent), all_adjacent_gaps_exceed_tie_band=separated)
    result = dict(
        kind='POST_HOC_DESCRIPTIVE_ONLY',
        timestamp_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        aggregation='Ratio of document-mean scores across the eight calibration documents; not the mean of tokenwise ratios',
        net_cross_fraction_formula='(mean_B - mean_A) / mean_B',
        runner_up_gap_formula='(B_second - B_best) / B_best',
        all_12_ranks_identical=orders['A'] == orders['B'],
        order_candidate_indices_zero_based=orders,
        adjacent_score_separation=gaps,
        selected_candidate=candidates[best],
        selected_net_cross_fraction=net[best],
        net_cross_fraction_range=[min(net), max(net)],
        B_runner_up_relative_gap=(b[runner_up] - b[best]) / b[best],
        candidate_rows=[dict(candidate=candidates[i], A=a[i], B=b[i], net_cross_term=b[i]-a[i],
                             net_cross_fraction=net[i], rank_A=orders['A'].index(i)+1,
                             rank_B=orders['B'].index(i)+1) for i in range(12)],
        source_sha256=hashes,
        analysis_code_sha256=digest(Path(__file__)),
        gpu_calls=0, model_forwards=0, registered_criteria_changed=False,
        limitations='Aggregated net cross terms do not bound individual-token cross terms or exclude positive/negative cancellation. Rank identity is for calibration mean scores, not every document or token.',
        conclusion='For the tested subproblem, including cross terms changed neither the ranking of the 12 candidates nor the selected allocation; retain the existing closure decision.',
    )
    assert result['all_12_ranks_identical']
    assert candidates[best] == data['choices']['A'] == data['choices']['B']
    assert all(g['all_adjacent_gaps_exceed_tie_band'] for g in gaps.values())
    (ROOT / 'posthoc_rank_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')

    marker = '\n사후 설명 추가: 보정 평균 점수의 전체 후보 순위\n'
    report_path = ROOT / 'REPORT_KO.txt'
    original = report_path.read_text().split(marker)[0].rstrip()
    note = f'''{marker}
저장된 calibration.json의 8개 보정 문서 평균을 재계산했다. 새 모델 실행은 없으며,
이는 등록된 진행/종료 기준을 바꾸지 않는 사후 설명용 계산이다.
  A/B의 12개 후보 전체 순위: 모두 동일. 인접 점수 차이도 수치 동점 범위를 넘는다.
  선택된 후보의 순교차항 비중 (B-A)/B: {100*net[best]:+.6f}%.
  전체 12개 후보의 순교차항 비중: {100*min(net):+.6f}% ~ {100*max(net):+.6f}%.
  B 기준 1위와 2위의 상대 점수 차이 (B_second-B_best)/B_best:
  {100*result['B_runner_up_relative_gap']:.6f}%.
위 비중은 문서·토큰에 걸쳐 합산된 순교차항이다. 개별 토큰의 교차항이 항상
작다거나 양수·음수 상쇄가 없었다는 뜻은 아니다. 순위 일치 역시 보정 평균에
대한 결과이며 각 문서·토큰의 순위가 모두 같다는 주장이 아니다.
최종 표현: 시험한 부분문제의 보정 평균에서 교차항을 포함해도 12개 후보의
순위와 선택이 바뀌지 않았다. 기존 품질·비용 방향의 종료 결론을 유지한다.
후보별 수치·계산식·원자료 해시: posthoc_rank_analysis.json.
'''
    report_path.write_text(original + '\n' + note)
    for path in preserved:
        assert digest(path) == hashes[str(path.relative_to(REPO))]
    print(json.dumps({key: result[key] for key in (
        'all_12_ranks_identical', 'selected_net_cross_fraction', 'net_cross_fraction_range',
        'B_runner_up_relative_gap', 'registered_criteria_changed', 'gpu_calls')}, indent=2))


if __name__ == '__main__':
    main()
