# -*- coding: utf-8 -*-
"""확인 절차 1 — figure_data.json 의 값을 claim_table 의 수치와 대조한다.

claim_table 이 정본이다. 어긋나면 그림이 틀린 것이다.
기대값은 claim_table 본문에서 사람이 옮긴 것이므로, 이 파일 자체가
"손으로 옮긴 수치"다 — 그래서 **대조 전용**이고 그림 생성에는 쓰지 않는다.
"""
import json, os, sys

D = json.load(open(os.path.expanduser("~/gpu/Self-Forcing/figs/figure_data.json")))

CHECKS = [
    # (설명, 실제값, 기대값, 허용오차)
    ("Q1 정렬 A1 INT2 평균", D["fig1"]["int2"]["alignment_only"]["mean"], 2.28, 0.005),
    ("Q1 정렬 A1 INT2 lo",   D["fig1"]["int2"]["alignment_only"]["lo"], 0.235, 0.005),
    ("Q1 정렬 A1 INT2 hi",   D["fig1"]["int2"]["alignment_only"]["hi"], 4.329, 0.005),
    ("Q3 sink|OFF INT2",     D["fig1"]["int2"]["sink_only"]["mean"], -1.54, 0.005),
    ("Q5 상호작용 INT2",      D["fig1"]["int2"]["interaction"]["mean"], 9.22, 0.005),
    ("Q5 상호작용 INT2 lo",   D["fig1"]["int2"]["interaction"]["lo"], 5.85, 0.005),
    ("Q5 상호작용 INT2 hi",   D["fig1"]["int2"]["interaction"]["hi"], 12.59, 0.005),
    ("Q6 상호작용 INT4",      D["fig1"]["int4"]["interaction"]["mean"], -0.50, 0.005),
    ("가산 예측 INT2",         D["fig1"]["int2"]["sum_predicted"]["mean"], 0.74, 0.005),
    ("실제 결합 INT2",         D["fig1"]["int2"]["both"]["mean"], 9.96, 0.005),
    ("G3 정렬 ON 재양자화",    D["fig2"]["align_on"]["total"], 0, 0),
    ("G3 비교 총수",           D["fig2"]["align_off"]["total_compared"], 47250, 0),
    ("OFF 이동없음 검출",       D["fig2"]["align_off"]["no_shift"], 1350, 0),
    ("OFF 이동없음 비교",       D["fig2"]["align_off"]["no_shift_compared"], 39690, 0),
    ("OFF 이동있음 검출",       D["fig2"]["align_off"]["shifted"], 7560, 0),
    ("OFF 이동있음 비교",       D["fig2"]["align_off"]["shifted_compared"], 7560, 0),
    ("OFF 이동있음 비율",       D["fig2"]["align_off"]["shifted_rate"], 1.0, 0),
    ("A2 sink 변화(INT2)",    D["fig2"]["a2_int2_sink_reference"]["align_off"]["sink_changed"], 804, 0),
    ("A2 sink 비교(INT2)",    D["fig2"]["a2_int2_sink_reference"]["align_off"]["sink_compared"], 9000, 0),
    ("G1 같은격자 INT2",       D["fig3"]["same_grid"]["int2"], 0.0, 0),
    ("G2 밀린격자 INT2",       D["fig3"]["shifted_grid"]["int2"], 0.201, 0.0005),
    ("G2 밀린격자 INT4",       D["fig3"]["shifted_grid"]["int4"], 0.0547, 0.0005),
    ("Q8 앵커 존재",           D["fig4"]["anchor_presence"]["mean"], 7.68, 0.005),
    ("Q8 예산 1→3",           D["fig4"]["anchor_budget_1to3"]["mean"], 3.50, 0.005),
    ("Q7 적응 선택",           D["fig4"]["adaptive_selection"]["mean"], 0.00, 0),
    ("M1 λ4 subj",           D["fig5"]["subject_consistency"]["4.0"]["mean"], -0.1602, 0.0005),
    ("M2 λ0.5−λ1 subj lo",   D["fig5"]["direct_lam0p5_minus_lam1"]["subject_consistency"]["lo"], -0.0057, 0.0005),
    ("M2 λ0.5−λ1 subj hi",   D["fig5"]["direct_lam0p5_minus_lam1"]["subject_consistency"]["hi"], 0.0095, 0.0005),
    ("B1 보정 회복 λ1 MUSIQ", D["fig5"]["musiq"]["1.0"]["mean"], 10.04, 0.01),
    ("B1 회복 lo",            D["fig5"]["musiq"]["1.0"]["lo"], 3.09, 0.01),
    ("B1 회복 hi",            D["fig5"]["musiq"]["1.0"]["hi"], 16.99, 0.01),
    ("S1 BF16 상주 GB",       D["fig6"]["memory_bytes"]["bf16"] / 1e9, 6.038, 0.001),
    ("S1 최종 상주 GB",        D["fig6"]["memory_bytes"]["final"] / 1e9, 1.510, 0.001),
    ("S3 pending 비율",        D["fig6"]["memory_bytes"]["final_bf16_pending"]
                              / D["fig6"]["memory_bytes"]["final"], 0.571, 0.005),
    ("S2 BF16 s",             D["fig6"]["inference_s"]["bf16"], 465.6, 0.05),
    ("S2 최종 s",              D["fig6"]["inference_s"]["final"], 545.2, 0.05),
]

bad = 0
for name, got, exp, tol in CHECKS:
    ok = abs(got - exp) <= tol
    if not ok:
        bad += 1
    print(f"  {'OK ' if ok else 'FAIL'} {name:24s} 값 {got!r:>24}  기대 {exp}")
print(f"\n{len(CHECKS) - bad}/{len(CHECKS)} 일치" + ("" if not bad else "  ← 불일치 있음"))
sys.exit(1 if bad else 0)
