"""Gate H — paired analysis of the lambda sweep.

Reports BOTH aggregations for every run, because they disagree (session-5
finding: the audit's '+10.40 recovery' is the drop metric, which improves when
the correction damages chunk 0 and flattens the curve):
  mean  — per-prompt mean MUSIQ over chunks 0..20 vs A1_int2 (level)
  drop  — per-prompt MUSIQ(c20) - MUSIQ(c0) vs A1_int2 (slope, audit's metric)
Also vs BF16, and the post-correction self-mass at chunk 20 from tum_mass.json.

Usage: analyze_gateH.py <musiq_per_prompt.json with gateH tags> [baseline_cfg]
"""
import json
import math
import os
import sys

R = os.path.expanduser("~/gpu/Self-Forcing/results")


def mat(d, c):
    n = len(d[c])
    return [[d[c][str(i)]["musiq"][p] for i in range(n)]
            for p in range(len(d[c]["0"]["musiq"]))]


def paired(da):
    n = len(da)
    m = sum(da) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in da) / (n - 1))
    se = sd / math.sqrt(n)
    t = m / se if se else float("inf")
    sg = sum(1 for x in da if (x > 0) == (m > 0))
    return m, sd, t, sg


def report(tag, new, base):
    mean_d = [sum(a) / len(a) - sum(b) / len(b) for a, b in zip(new, base)]
    drop_d = [(a[20] - a[0]) - (b[20] - b[0]) for a, b in zip(new, base)]
    for name, da in (("mean", mean_d), ("drop", drop_d)):
        m, sd, t, sg = paired(da)
        print(f"  {tag:28s} {name}: {m:+7.2f} ± {sd:5.2f}  t={t:+6.2f}  "
              f"sign {sg}/{len(da)}")


def main():
    per_prompt = json.load(open(sys.argv[1]))
    ga = json.load(open(f"{R}/gateA_metrics_A1_per_prompt.json"))
    bf, i2 = mat(ga, "A1_bf16"), mat(ga, "A1_int2")
    print("== vs A1_int2 (uncorrected) ==")
    for cfg in per_prompt:
        report(cfg, mat(per_prompt, cfg), i2)
    print("== vs A1_bf16 ==")
    for cfg in per_prompt:
        report(cfg, mat(per_prompt, cfg), bf)
    print("== reference points (legacy, self-INCLUSIVE) ==")
    b2 = json.load(open(os.path.expanduser("~/gpu/audit/batch2_musiq_per_prompt.json")))
    f2 = json.load(open(f"{R}/gateF2_metrics_per_prompt.json"))
    report("lam0 tum_zero", mat(b2, "tum_zero"), i2)
    report("lam1 tum_paper", mat(b2, "tum_paper"), i2)
    report("lam4 gateF2(pre-fix)", mat(f2, "tum_a1_int2"), i2)

    # post-correction self mass at chunk 20, mean over layers/steps/prompts
    for cfg in per_prompt:
        mp = f"{R}/gateH/{cfg}/tum_mass.json"
        if not os.path.exists(mp):
            continue
        e = json.load(open(mp))["entries"]
        by_chunk = {}
        for r in e:
            if not isinstance(r.get("mass"), dict):
                continue
            slots = r["mass"]
            last = f"slot{max(int(s[4:]) for s in slots)}"
            by_chunk.setdefault(r["chunk"], []).append(slots[last])
        for c in sorted(by_chunk):
            v = by_chunk[c]
            print(f"  {cfg}: chunk {c} post-bias self mass "
                  f"{100 * sum(v) / len(v):.1f}%  (n={len(v)})")


if __name__ == "__main__":
    main()
