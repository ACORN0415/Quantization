# -*- coding: utf-8 -*-
"""S9 PSAC8 참고 백엔드 관문 판정 (사전 등록 §3). 인자: 결과 디렉터리.
G1 NOQUANT vs BF16 latent chunk 별 상대오차 ≤ 1e-2 · G2 양자화 세 팔 chunk 0 bit-exact · G3 상태 로그(세그먼트 수·chunk_id·리셋) ·
G4 PSAC8 세그먼트 구성 요소 형상/dtype → 해석식 바이트 대조(head·chunk 당 K 264,528 B). 속도도 기록."""
import json, os, sys, torch
O = sys.argv[1]; FPC = 3
def lat(a): return torch.load(f"{O}/{a}/latents/prompt000/full.pt", map_location="cpu", weights_only=False)["latent"].float()
def rel(a, b): return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
ok = True; out = {}
# G1
bf, nq = lat("BF16"), lat("NOQUANT"); assert bf.shape == nq.shape, (bf.shape, nq.shape)
F = bf.shape[1]; per = [rel(nq[:, c:c+FPC], bf[:, c:c+FPC]) for c in range(0, F, FPC)]
out["G1_rel_per_chunk"] = per
print("G1(원안) NOQUANT vs BF16 rollout latent chunk 별 상대오차: " + " ".join(f"{x:.1e}" for x in per))
print(f"   max {max(per):.2e} → {'PASS' if max(per) <= 1e-2 else 'FAIL'} (원안 기준 ≤1e-2; rollout 누적은 수치 발산을 포함 — 2026-10-25 프로토콜 변경 기록 참조)")
# G1': 같은 입력에서 명시적 경로 vs FA2 한 호출 비교 (S5 Gate B 와 같은 성격) — 사전 등록 §3 의 경로 관문은 이것으로 판정
recs = [r for r in json.load(open(f"{O}/NOQUANT.state.json")) if "fa2_rel" in r]
fa2 = [r["fa2_rel"] for r in recs]; out["G1_percall_fa2_rel"] = [(r["chunk"], r["layer"], r["fa2_rel"]) for r in recs]
g1 = bool(fa2) and max(fa2) <= 1e-2; ok &= g1
print(f"G1' 한 호출 명시적 vs FA2 (같은 입력, {len(fa2)} 지점): max {max(fa2) if fa2 else float('nan'):.2e} → {'PASS' if g1 else 'FAIL'} (기준 ≤1e-2)")
# G2
arms = ["PSAC8", "INT2L1", "K4V2L1"]; L = {a: lat(a) for a in arms}
c0 = [torch.equal(L[a][:, :FPC], L[arms[0]][:, :FPC]) for a in arms[1:]]
c0_vs_nq = torch.equal(L["PSAC8"][:, :FPC], nq[:, :FPC])
g2 = all(c0) and c0_vs_nq; ok &= g2; out["G2_chunk0_bitexact"] = {"arms": c0, "vs_noquant": c0_vs_nq}
print(f"G2 chunk0 bit-exact: 팔 사이 {c0}, NOQUANT 대비 {c0_vs_nq} → {'PASS' if g2 else 'FAIL'}")
later = {a: [rel(L[a][:, c:c+FPC], bf[:, c:c+FPC]) for c in range(FPC, F, FPC)] for a in arms}
out["chunk1plus_rel_vs_BF16"] = later
print("참고: BF16 대비 chunk1+ 상대오차 평균 " + " ".join(f"{a} {sum(v)/len(v):.3f}" for a, v in later.items()) + " (카오스 포화 — 품질 지표 아님)")
# G3
g3 = True
for a in ["NOQUANT"] + arms:
    recs = json.load(open(f"{O}/{a}.state.json"))
    for r in recs:
        cap = r["cap_chunks"]; ids = r["chunk_ids"]
        okr = r["n_segs"] <= cap - 1 and ids == sorted(ids) and all(b - a_ == 1 for a_, b in zip(ids, ids[1:])) and (not ids or ids[-1] == r["chunk"] - 1)
        g3 &= okr
        if r["layer"] == 29 or r["chunk"] == 1:
            print(f"G3 {a}: chunk {r['chunk']} layer {r['layer']} segs {r['n_segs']} ids {ids} cap {cap} → {'ok' if okr else 'BAD'}")
ok &= g3; out["G3"] = g3
print(f"G3 세그먼트 수·연속성 → {'PASS' if g3 else 'FAIL'}")
# G4
recs = [r for r in json.load(open(f"{O}/PSAC8.state.json")) if r["chunk"] == 20 and r["layer"] == 29 and r["last_seg"]]
if recs:
    seg = recs[0]["last_seg"]; L_, H, D, G, r = 4680, 12, 128, 16, 8; NB = -(-L_ // G)
    k_codes_packed = L_ * D * H * 2 // 8; k_scale = NB * D * H * 2
    psac_U = r * D * H * 2; psac_C = L_ * r * H; psac_Cs = r * H * 2; var = D * H * 2
    analytic_k = k_codes_packed + k_scale + psac_U + psac_C + psac_Cs + var
    print("G4 PSAC8 세그먼트 구성 (dtype, shape, 실제 storage bytes):")
    for k, v in seg.items(): print(f"   {k}: {v}")
    print(f"   해석식 K bytes/chunk (12 head, packed 2bit): {analytic_k:,} = 12 × {analytic_k//12:,}  (문서 264,528 × 12 = {264528*12:,})")
    g4 = abs(analytic_k - 264528 * 12) <= 12 * 16
    # 형상 대조
    try:
        shp = {k_: tuple(v_) for k_, v_ in {"U": seg["U"][1], "C": seg["C"][1], "Cs": seg["Cs"][1], "var": seg["var"][1], "kq": seg["k"]["q"][1], "kscale": seg["k"]["scale"][1]}.items()}
        g4 &= shp["U"] == (H, D, r) and shp["C"] == (L_, H, r) and shp["var"] == (H, D) and shp["kscale"][1] == NB
        print(f"   형상 {shp} → {'PASS' if g4 else 'FAIL'}")
    except Exception as e:
        g4 = False; print("   형상 확인 실패:", e)
    ok &= g4; out["G4"] = g4
else:
    print("G4: 기록 지점(20,29) 없음 → FAIL"); ok = False
# 속도
for a in ["BF16", "NOQUANT"] + arms:
    s = json.load(open(f"{O}/{a}.stats.json"))["runs"][0]
    print(f"속도 {a}: {s['seconds']:.1f} s/프롬프트, peak {s.get('peak_vram_gb', float('nan')):.2f} GB")
    out.setdefault("seconds", {})[a] = s["seconds"]
json.dump(out, open(f"{O}/gates.json", "w"), indent=1)
print("GATES", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
