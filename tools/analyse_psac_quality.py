# -*- coding: utf-8 -*-
"""S9 PSAC8 품질 실험 분석 — 사전 등록 §2 규칙을 그대로 적용. 인자: 결과 디렉터리 O.
입력: O/musiq_per_prompt.json, O/vbench_per_prompt.json, (선택) O/vbench_dyn_per_prompt.json, O/motion_proxy.json
출력: O/contrasts.json, 표준출력 표 + 판정."""
import json, os, statistics as st, sys
O = sys.argv[1]; T = 2.262  # t(9, .975)
ARMS = ["PSAC8", "K4V2L1", "INT2L1", "BF16"]
MARGIN = {"subject_consistency": -0.010, "musiq": -1.0, "temporal_flickering": -0.003}

def musiq_pp(tag):
    ch = json.load(open(f"{O}/musiq_per_prompt.json"))[tag]; n = len(ch[next(iter(ch))]["musiq"])
    return [st.fmean(ch[c]["musiq"][i] for c in ch) for i in range(n)]
vb = json.load(open(f"{O}/vbench_per_prompt.json"))
data = {a: {"musiq": musiq_pp(a), **{d: vb[a][d] for d in ("subject_consistency", "temporal_flickering")}} for a in ARMS}
dyn = None
if os.path.exists(f"{O}/vbench_dyn_per_prompt.json"):
    try: dyn = {a: json.load(open(f"{O}/vbench_dyn_per_prompt.json"))[a]["dynamic_degree"] for a in ARMS}
    except Exception as e: print("dynamic_degree 읽기 실패:", e)
mot = json.load(open(f"{O}/motion_proxy.json")) if os.path.exists(f"{O}/motion_proxy.json") else None

def ci(d):
    n = len(d); m = st.fmean(d); h = T * st.stdev(d) / n ** 0.5 if n > 1 else float("nan")
    return m, m - h, m + h, sum(x > 0 for x in d), m / (st.stdev(d) / n ** 0.5) if n > 1 and st.stdev(d) > 0 else float("nan")

out = {"n": len(data["PSAC8"]["musiq"]), "contrasts": {}, "rules": {}}
print(f"n = {out['n']} 프롬프트 (prompts_eval_psac10). 팔 평균: " + " | ".join(
    f"{a}: musiq {st.fmean(data[a]['musiq']):.2f} subj {st.fmean(data[a]['subject_consistency']):.4f} flick {st.fmean(data[a]['temporal_flickering']):.4f}" for a in ARMS))
for a, b in (("PSAC8", "K4V2L1"), ("PSAC8", "INT2L1"), ("PSAC8", "BF16"), ("K4V2L1", "BF16"), ("INT2L1", "BF16")):
    print(f"\n## {a} − {b}")
    for m in ("musiq", "subject_consistency", "temporal_flickering"):
        d = [x - y for x, y in zip(data[a][m], data[b][m])]; mean, lo, hi, pos, t = ci(d)
        out["contrasts"][f"{a}-{b}"] = out["contrasts"].get(f"{a}-{b}", {}); out["contrasts"][f"{a}-{b}"][m] = {"mean": mean, "lo": lo, "hi": hi, "pos": pos, "t": t, "per_prompt": d}
        fmt = "{:+.3f}" if m == "musiq" else "{:+.4f}"
        print(f"  {m:22s} " + fmt.format(mean) + f" [{fmt.format(lo)}, {fmt.format(hi)}] t={t:+.2f} 양수 {pos}/{len(d)} | " + " ".join(fmt.format(x) for x in d))

# ---- 사전 등록 규칙
k = out["contrasts"]["PSAC8-K4V2L1"]; rules = {}
for m, mg in MARGIN.items():
    rules[f"noninf_{m}"] = {"lo": k[m]["lo"], "margin": mg, "pass": k[m]["lo"] >= mg}
# 움직임
if dyn is not None:
    cP, cK = sum(1 for x in dyn["PSAC8"] if x), sum(1 for x in dyn["K4V2L1"] if x)
    rules["motion_dynamic_degree"] = {"PSAC8_dynamic": cP, "K4V2L1_dynamic": cK, "pass": cP >= cK - 1, "source": "vbench dynamic_degree"}
elif mot is not None:
    r = [p / q if q > 0 else float("nan") for p, q in zip(mot["PSAC8"]["mean_absdiff"], mot["K4V2L1"]["mean_absdiff"])]
    rules["motion_proxy"] = {"ratio_per_prompt": r, "n_ge_0.9": sum(x >= 0.9 for x in r), "pass": sum(x >= 0.9 for x in r) >= 8, "source": "mean |frame diff| PSAC8/K4V2L1"}
else:
    rules["motion"] = {"pass": False, "note": "움직임 지표 없음"}
# sanity: PSAC8 − INT2L1
s_ = out["contrasts"]["PSAC8-INT2L1"]
rules["sanity_vs_INT2L1"] = {"subj_lo": s_["subject_consistency"]["lo"], "musiq_lo": s_["musiq"]["lo"],
                             "pass": s_["subject_consistency"]["lo"] > 0 or s_["musiq"]["lo"] > 0}
out["rules"] = rules
allpass = all(v["pass"] for v in rules.values())
out["verdict"] = "PASS(비열등, 사전 한계 안)" if allpass else "FAIL"
print("\n## 사전 등록 규칙")
for kname, v in rules.items(): print(f"  {kname}: {'PASS' if v['pass'] else 'FAIL'}  {json.dumps({kk: vv for kk, vv in v.items() if kk not in ('pass','ratio_per_prompt')}, ensure_ascii=False)}")
print(f"\n판정: {out['verdict']}  (통과 = 비열등 세 지표 + 움직임 + INT2 λ1 대비 sanity 모두)")
json.dump(out, open(f"{O}/contrasts.json", "w"), indent=1, ensure_ascii=False); print("→", f"{O}/contrasts.json")
