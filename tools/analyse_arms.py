# -*- coding: utf-8 -*-
"""범용 팔 비교: musiq_per_prompt.json + vbench_per_prompt.json (+ vbench_dyn_per_prompt.json, motion_proxy.json) 에서
짝지은 차이·t(n-1,.975) CI·양수 개수. 사용: analyse_arms.py DIR --contrasts A-B C-D ... [--arms ...]"""
import argparse, json, math, os, statistics as st
T = {5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.262, 10: 2.228}
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--contrasts", nargs="+", required=True); ap.add_argument("--arms", nargs="*", default=None)
a = ap.parse_args(); O = a.dir
mp = json.load(open(f"{O}/musiq_per_prompt.json")); vb = json.load(open(f"{O}/vbench_per_prompt.json"))
arms = a.arms or list(mp.keys())
def musiq_pp(tag):
    ch = mp[tag]; n = len(ch[next(iter(ch))]["musiq"]); return [st.fmean(ch[c]["musiq"][i] for c in ch) for i in range(n)]
data = {t: {"musiq": musiq_pp(t), **{d: vb[t][d] for d in vb[t]}} for t in arms}
dyn = json.load(open(f"{O}/vbench_dyn_per_prompt.json")) if os.path.exists(f"{O}/vbench_dyn_per_prompt.json") else None
mot = json.load(open(f"{O}/motion_proxy.json")) if os.path.exists(f"{O}/motion_proxy.json") else None
n = len(data[arms[0]]["musiq"]); t = T.get(n - 1, 2.0)
def ci(d):
    m = st.fmean(d); h = t * st.stdev(d) / math.sqrt(len(d)); return m, m - h, m + h, sum(x > 0 for x in d)
print(f"n={n}. 팔 평균:")
for tname in arms:
    extra = f" dyn {sum(1 for x in dyn[tname]['dynamic_degree'] if x)}/{n}" if dyn else ""
    extra += f" motion {st.fmean(mot[tname]['mean_absdiff']):.4f}" if mot else ""
    print(f"  {tname:12s} musiq {st.fmean(data[tname]['musiq']):.2f} subj {st.fmean(data[tname]['subject_consistency']):.4f} flick {st.fmean(data[tname]['temporal_flickering']):.4f}{extra}")
out = {"n": n, "contrasts": {}}
for c in a.contrasts:
    A, B = c.split("-"); print(f"\n## {A} − {B}"); out["contrasts"][c] = {}
    for m in ("musiq", "subject_consistency", "temporal_flickering"):
        d = [x - y for x, y in zip(data[A][m], data[B][m])]; mean, lo, hi, pos = ci(d)
        out["contrasts"][c][m] = {"mean": mean, "lo": lo, "hi": hi, "pos": pos, "per_prompt": d}
        fmt = "{:+.3f}" if m == "musiq" else "{:+.4f}"
        print(f"  {m:22s} " + fmt.format(mean) + f" [{fmt.format(lo)}, {fmt.format(hi)}] 양수 {pos}/{n} | " + " ".join(fmt.format(x) for x in d))
    if mot:
        d = [x - y for x, y in zip(mot[A]["mean_absdiff"], mot[B]["mean_absdiff"])]; mean, lo, hi, pos = ci(d)
        out["contrasts"][c]["motion_absdiff"] = {"mean": mean, "lo": lo, "hi": hi, "pos": pos}
        print(f"  {'motion_absdiff':22s} {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] 양수 {pos}/{n}")
    if dyn:
        print(f"  dynamic_degree 동적 영상 수: {A} {sum(1 for x in dyn[A]['dynamic_degree'] if x)} vs {B} {sum(1 for x in dyn[B]['dynamic_degree'] if x)}")
json.dump(out, open(f"{O}/contrasts.json", "w"), indent=1, ensure_ascii=False); print("→", f"{O}/contrasts.json")
