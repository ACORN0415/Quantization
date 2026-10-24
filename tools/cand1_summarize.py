# -*- coding: utf-8 -*-
"""cand1_diag.py 결과(JSON) 집계. 상태 = (prompt, layer) 반복 측정, 독립 표본 아님.
평가 = chunk 21 step 1~3 평균 상대오차 (step 0 은 D 의 표본 내 → 따로).
출력: 팔별 평균, 기준선 R1(P3+λ1 대응) 대비 짝지은 차이(중앙·양수 수·층별 최악), D 이득 회복률(B·E), 층 18·22·29,
outlier head(층 29 head 0·2·9) 캐시 mass·b, 잔차 통계, 바이트."""
import json, statistics as st, sys
H = 12

def m13(a, k): return sum(a[k][f"step{s}"]["rel"] for s in (1, 2, 3)) / 3
def med(xs): return st.median(xs) if xs else float("nan")

def main(paths):
    rows = []; extra = []
    for p in paths:
        d = json.load(open(p)); (extra if "k4_" in p else rows).extend(d["rows"])
    for e in extra:   # K4 팔(별도 실행)을 같은 (prompt, layer) 행에 병합 — 캡처 root 로 짝을 맞춘다
        for r in rows:
            if r["prompt"] == e["prompt"] and r["layer"] == e["layer"] and r["int8_vs_write"] == e["int8_vs_write"]:
                r["arms"].update({k: v for k, v in e["arms"].items() if k.startswith("K4_")}); r["resid_K4"] = e.get("resid_K4")
    layers = sorted({r["layer"] for r in rows}); arms = list(rows[0]["arms"].keys())
    print(f"상태 {len(rows)}개 ({len(paths)} 파일), 층 {layers}, scope {sorted({r.get('scope','all') for r in rows})}")
    by_layer = lambda name: {l: [m13(r["arms"], name) for r in rows if r["layer"] == l] for l in layers}

    # ---- 1. 잔차 통계 (chunk 20, head 평균 → 층별 중앙)
    print("\n## 1. 잔차 E0 (RTN INT2, chunk 20) — head 평균을 층별로 중앙값")
    print("층 | var/uniform(Δ²/12) med | p90 | 평균노름/RMS | 공분산 유효rank | 상위12 고유값 비율 | 분산 상위12채널 비율 | 첨도")
    for l in layers:
        rs = [r["resid_E0"] for r in rows if r["layer"] == l]
        f = lambda k: med([sum(x[k] for x in e) / H for e in rs])
        print(f"{l:2d} | {f('var_ratio_med'):.3f} | {f('var_ratio_p90'):.3f} | {f('mean_rel'):.3f} | {f('cov_eff_rank'):.1f} | {f('cov_top12_frac'):.3f} | {f('var_top12_frac'):.3f} | {f('kurt_med'):.2f}")
    print("\n## 1b. PSAC 후 잔차 E_rem (r8) — 같은 지표, 변형별 (층 전체 중앙)")
    for tag in sorted({k for r in rows for k in r["resid_Erem"]}):
        if "_r8_" not in f"_{tag}_": continue
        rs = [r["resid_Erem"][tag] for r in rows]
        f = lambda k: med([sum(x[k] for x in e) / H for e in rs])
        kept = med([sum(r["psac_energy_kept"][tag]) / H for r in rows])
        print(f"{tag}: 보존 에너지 {kept:.3f} | var/uniform {f('var_ratio_med'):.3f} | 평균노름/RMS {f('mean_rel'):.3f} | 유효rank {f('cov_eff_rank'):.1f} | 상위12 {f('cov_top12_frac'):.3f} | 첨도 {f('kurt_med'):.2f}")

    # ---- 2. 팔별 오차
    print("\n## 2. 팔별 attention 상대오차 (step1~3 평균) — 전체 평균 / 중앙 / 층별 최악(층 평균의 최대)")
    print("팔 | 평균 | 중앙 | 최악층(값) | 층18 | 층22 | 층29")
    for name in arms:
        bl = by_layer(name); vals = [m13(r["arms"], name) for r in rows]
        worst = max(layers, key=lambda l: st.mean(bl[l]))
        g = lambda l: f"{st.mean(bl[l]):.4f}" if l in bl else "-"
        print(f"{name} | {st.mean(vals):.4f} | {med(vals):.4f} | L{worst}({st.mean(bl[worst]):.4f}) | {g(18)} | {g(22)} | {g(29)}")

    # ---- 3. 짝지은 차이 vs R1
    print("\n## 3. R1(INT2+TUM λ1, 기준선) − 팔 : 양수 = 팔이 더 낫다. 중앙 / 최소 / 양수 상태 수 / 층별(평균) 양수 층 수")
    for name in arms:
        if name == "R1": continue
        diff = [m13(r["arms"], "R1") - m13(r["arms"], name) for r in rows]
        bl1, bln = by_layer("R1"), by_layer(name)
        lpos = sum(st.mean(bl1[l]) > st.mean(bln[l]) for l in layers)
        print(f"{name:16s} | {med(diff):+.4f} | {min(diff):+.4f} | {sum(x > 0 for x in diff)}/{len(diff)} | {lpos}/{len(layers)}")

    # ---- 4. D 이득 회복률
    print("\n## 4. 배포형 변형의 oracle(D) 이득 회복률: (R2_r·R2_D 와 같은 기준 R0) → (R0−R2_v)/(R0−R2_D). 중앙 / 층별 18·22·29")
    for r_ in sorted({k.split("_")[1] for k in arms if k.startswith("R2_")}):
        for v in ("B", "E"):
            if f"R2_{r_}_D" not in arms or f"R2_{r_}_{v}" not in arms: continue
            rec = []
            for r in rows:
                a = r["arms"]; dD = m13(a, "R0") - m13(a, f"R2_{r_}_D"); dv = m13(a, "R0") - m13(a, f"R2_{r_}_{v}")
                rec.append((r["layer"], dv / dD if abs(dD) > 1e-9 else float("nan")))
            lay = lambda l: med([x for ll, x in rec if ll == l]) if any(ll == l for ll, _ in rec) else float('nan')
            print(f"R2_{r_}_{v}: 회복률 중앙 {med([x for _, x in rec]):.2f} | L18 {lay(18):.2f} L22 {lay(22):.2f} L29 {lay(29):.2f}")

    # ---- 5. R3 vs R4 (같은 K̂1 위에서 보정 방식만)
    print("\n## 5. 같은 K̂1(r8) 위에서 보정 방식: R2(없음) / R3 λ1 / R3 λ0.5 / R3 λ0.25 / R4d / R4f — 평균 (변형별)")
    for v in ("B", "E", "D"):
        ks = [k for k in (f"R2_r8_{v}", f"R3_r8_{v}_l1.0", f"R3_r8_{v}_l0.5", f"R3_r8_{v}_l0.25", f"R4d_r8_{v}", f"R4m_r8_{v}", f"R4f_r8_{v}") if k in arms]
        if not ks: continue
        print(f"{v}: " + " | ".join(f"{k.split('_')[0]}{k.split('_l')[-1] if '_l' in k else ''} {st.mean([m13(r['arms'], k) for r in rows]):.4f}" for k in ks))
        if f"R3_r8_{v}_l1.0" not in arms or f"R4f_r8_{v}" not in arms: continue
        d43 = [m13(r["arms"], f"R3_r8_{v}_l1.0") - m13(r["arms"], f"R4f_r8_{v}") for r in rows]
        print(f"   R3λ1 − R4f: 중앙 {med(d43):+.4f} 최소 {min(d43):+.4f} 양수 {sum(x>0 for x in d43)}/{len(d43)}")

    # ---- 5b. Δ 정의를 바로잡은 기준선과의 비교 (v2 팔)
    if "R1_l0.25" in arms:
        print("\n## 5b. Δ 보정 기준선: R1(λ1, Δ=2·scale) vs R1_l0.25(= Δ=scale 균등 모형) vs R4d0(PSAC 없이 실측 분산) — 평균 / 중앙 / 층별 최악")
        for k in [k for k in ("R0", "R1", "R1_l0.5", "R1_l0.25", "R4d0", "R4m0", "R4f0") if k in arms]:
            vals = [m13(r["arms"], k) for r in rows]; bl = by_layer(k); worst = max(layers, key=lambda l: st.mean(bl[l]))
            print(f"{k:10s} {st.mean(vals):.4f} / {med(vals):.4f} / L{worst} {st.mean(bl[worst]):.4f}")
        print("\n## 5c. PSAC 의 추가 몫 (같은 보정식 안에서): 기준 − 팔, 양수 = PSAC 이 낫다. 중앙 / 최소 / 양수 / 층별 양수")
        pairs = []
        for r_ in sorted({k.split("_")[1] for k in arms if k.startswith("R2_")}):
            for v in ("B", "E", "D", "N"):
                if f"R3_{r_}_{v}_l0.25" in arms: pairs.append((f"R1_l0.25", f"R3_{r_}_{v}_l0.25"))
                if f"R4d_{r_}_{v}" in arms: pairs.append(("R4d0", f"R4d_{r_}_{v}"))
        for base, name in pairs:
            diff = [m13(r["arms"], base) - m13(r["arms"], name) for r in rows]
            blb, bln = by_layer(base), by_layer(name); lpos = sum(st.mean(blb[l]) > st.mean(bln[l]) for l in layers)
            print(f"{base:9s} − {name:16s} | {med(diff):+.4f} | {min(diff):+.4f} | {sum(x>0 for x in diff)}/{len(diff)} | {lpos}/{len(layers)}")
        print("\n## 5d. 방향 선택의 몫: 무작위 기저(N) − query 기저(v), 같은 r·같은 보정. 양수 = query 방향이 낫다")
        for r_ in sorted({k.split("_")[1] for k in arms if k.startswith("R2_")}):
            for stem in ("R2", "R4d"):
                for v in ("B", "E", "D"):
                    kn, kv = f"{stem}_{r_}_N", f"{stem}_{r_}_{v}"
                    if kn not in arms or kv not in arms: continue
                    diff = [m13(r["arms"], kn) - m13(r["arms"], kv) for r in rows]
                    bln, blv = by_layer(kn), by_layer(kv); lpos = sum(st.mean(bln[l]) > st.mean(blv[l]) for l in layers)
                    lay = lambda l: (st.mean(bln[l]) - st.mean(blv[l])) if l in bln and bln[l] else float('nan')
                    print(f"{kn:10s} − {kv:10s} | 중앙 {med(diff):+.4f} 최소 {min(diff):+.4f} 양수 {sum(x>0 for x in diff)}/{len(diff)} 층 {lpos}/{len(layers)} | L0 {lay(0):+.4f} L18 {lay(18):+.4f} L22 {lay(22):+.4f} L29 {lay(29):+.4f}")

    # ---- 5e. K4V2 기준선 (0.1 목표): INT2 계열 최선 vs K INT4
    if "K4_R1s" in arms or any("K4_R1s" in r["arms"] for r in rows):
        rows_k = [r for r in rows if "K4_R1s" in r["arms"]]
        print(f"\n## 5e. 0.1 목표 — K4V2 λ1 (K INT4 + TUM Δ=scale λ1; V INT2 동일) 대비. 상태 {len(rows_k)}개")
        print("팔 | 평균 | 중앙 | 층별 최악 | K4_R1s − 팔 (중앙 / 최소 / 팔이 나은 상태) | K 바이트 % of K4")
        bk = rows_k[0]["bytes_k_head_chunk"]
        def pct(name):
            if name.startswith("K4"): return "100"
            r_ = name.split("_r")[1].split("_")[0] if "_r" in name else None
            if name.startswith("R4d_"): return f"{100*bk[f'INT2+PSAC{r_}+R4d']/bk['K4']:.0f}"
            if name.startswith(("R2_", "R3_")): return f"{100*bk[f'INT2+PSAC{r_}']/bk['K4']:.0f}"
            return f"{100*bk['INT2']/bk['K4']:.0f}"
        for name in ["K4_R0", "K4_R1s", "K4_R1", "R0", "R1_l0.25", "R2_r8_E", "R4d_r8_E", "R2_r16_E", "R4d_r16_E", "R3_r16_E_l0.25"]:
            if not all(name in r["arms"] for r in rows_k): continue
            vals = [m13(r["arms"], name) for r in rows_k]; bl = {l: [m13(r["arms"], name) for r in rows_k if r["layer"] == l] for l in layers}
            worst = max(layers, key=lambda l: st.mean(bl[l])); diff = [m13(r["arms"], "K4_R1s") - m13(r["arms"], name) for r in rows_k]
            print(f"{name:16s} | {st.mean(vals):.4f} | {med(vals):.4f} | L{worst} {st.mean(bl[worst]):.4f} | {med(diff):+.4f} / {min(diff):+.4f} / {sum(x>0 for x in diff)}/{len(diff)} | {pct(name)}%")
        if rows_k[0].get("resid_K4"):
            vr = med([sum(x["var_ratio_med"] for x in r["resid_K4"]) / H for r in rows_k])
            print(f"K INT4 잔차 var/uniform(Δ=2·scale 기준) 중앙 {vr:.3f} (×4 = Δ=scale 기준)")

    # ---- 6. outlier heads (층 29 head 0,2,9): 캐시 mass (참조 / R0 / R1 / R4f_E) 와 b 중앙·최대
    print("\n## 6. 층 29 outlier head (0·2·9) — step1 캐시 mass: 참조 / R0 / R1 / R2_r8_E / R3_r8_E_l1.0 / R4f_r8_E ; b_med(R1) b_max(R1) ; b_med(R4f_E)")
    for r in rows:
        if r["layer"] != 29: continue
        rr = "r8" if "R4f_r8_E" in r["arms"] else "r16"
        for h in (0, 2, 9):
            g = lambda k: r["arms"][k.replace("r8", rr)]["step1"]["cache_mass"][h]
            a1 = r["arms"]["R1"]["step1"]; a4 = r["arms"][f"R4f_{rr}_E"]["step1"]
            extra = f" R1λ.25 {g('R1_l0.25'):.3f} R4d0 {g('R4d0'):.3f}" if "R1_l0.25" in r["arms"] else ""
            print(f"p{r['prompt']} h{h}: ref {r['ref_cache_mass']['step1'][h]:.3f} | R0 {g('R0'):.3f} R1 {g('R1'):.3f}{extra} R2E {g('R2_r8_E'):.3f} R3E {g('R3_r8_E_l1.0'):.3f} R4fE {g('R4f_r8_E'):.3f} | b R1 med {a1['b_med'][h]:.2f} max {a1['b_max'][h]:.1f} | b R4f med {a4['b_med'][h]:.2f}")

    # ---- 6b. R4 변형 분리 (부호 수정 후): R2 / R4d(대각) / R4m(평균+대각) / R4f(평균+공분산), 변형 E
    for r_ in sorted({k.split("_")[1] for k in arms if k.startswith("R2_")}):
        ks = [f"R2_{r_}_E", f"R4d_{r_}_E", f"R4m_{r_}_E", f"R4f_{r_}_E"]
        if all(k in arms for k in ks):
            print(f"\n## 6b. 보정 통계의 몫 ({r_}, 변형 E): " + " | ".join(f"{k.split('_')[0]} {st.mean([m13(r['arms'], k) for r in rows]):.4f}" for k in ks))
            for a_, b_ in ((ks[1], ks[2]), (ks[2], ks[3])):
                diff = [m13(r["arms"], a_) - m13(r["arms"], b_) for r in rows]
                print(f"   {a_} − {b_}: 중앙 {med(diff):+.4f} 최소 {min(diff):+.4f} 양수(뒤가 나음) {sum(x>0 for x in diff)}/{len(diff)}")
    if "R4m0" in arms:
        print("   PSAC 없음: " + " | ".join(f"{k} {st.mean([m13(r['arms'], k) for r in rows]):.4f}" for k in ("R1_l0.25", "R4d0", "R4m0", "R4f0")))

    # ---- 7. 바이트
    b = rows[0]["bytes_k_head_chunk"]
    print("\n## 7. K 바이트 head·chunk 당 (V 는 모든 팔 INT2 동일): " + " | ".join(f"{k} {v:,} ({100*v/b['K4']:.0f}% of K4)" for k, v in b.items()))
    # 전체 상주 추정 (30층, 과거 6 chunk + BF16 pending 1 chunk, K·V): 리뷰 지적 — K+V 압축부만의 비율은 전체 상주 비율이 아니다
    L_, D_, Hh, NL, NCH = 4680, 128, 12, 30, 6
    pend = L_ * D_ * 2 * 2 * Hh * NL
    tot = lambda kb, vb: (kb + vb) * NCH * Hh * NL + pend
    k4v2 = tot(b["K4"], b["INT2"]); print(f"전체 상주 추정(GB, 30층·W=7·pending BF16 포함): K4V2 {k4v2/2**30:.3f} | " +
          " | ".join(f"{k} {tot(v, b['INT2'])/2**30:.3f} ({100*tot(v, b['INT2'])/k4v2:.1f}%)" for k, v in b.items() if k.startswith("INT2+PSAC")))
    print(f"검산 R4d↔R1pooled 최대차: {max(r['selfcheck_R4d_vs_R1pooled_maxdiff'] for r in rows):.1e} ; INT8캡처 vs 실제쓰기 차 K 중앙 {med([r['int8_vs_write']['K'] for r in rows]):.4f}")

if __name__ == "__main__":
    main(sys.argv[1:])
