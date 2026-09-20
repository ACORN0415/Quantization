"""Paired comparison of VBench dimensions between runs (prompts are the unit).

Usage: analyze_vbench.py <per_prompt.json> --pairs NEW:BASE [NEW:BASE ...]
       analyze_vbench.py <per_prompt.json> --vs BASE          (all others vs BASE)
"""
import argparse, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s6stats import paired as _paired   # addendum A2


def paired(v):
    r = _paired(v)
    return r["mean"], r["ci"], r["t"], r["sign"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("per_prompt")
    ap.add_argument("--vs", required=True, help="baseline tag")
    ap.add_argument("--out_csv", default=None)
    a = ap.parse_args()
    d = json.load(open(a.per_prompt))
    if a.vs not in d:
        sys.exit(f"{a.vs} not in {sorted(d)}")
    base = d[a.vs]
    dims = list(base)
    rows = []
    print(f"baseline: {a.vs}   (paired over prompts)\n")
    hdr = f"{'run':24s}" + "".join(f"{x[:18]:>20s}" for x in dims)
    print(hdr)
    for tag in d:
        if tag == a.vs:
            continue
        cells = []
        for dim in dims:
            if dim not in d[tag]:
                cells.append("        --        ")
                continue
            v = [x - y for x, y in zip(d[tag][dim], base[dim])]
            m, ci, t, sg = paired(v)
            cells.append(f"{m:+8.4f} (t{t:+5.1f})")
            rows.append((tag, dim, m, ci, t, sg, len(v)))
        print(f"{tag:24s}" + "".join(f"{c:>20s}" for c in cells))
    if a.out_csv:
        with open(a.out_csv, "w") as f:
            f.write("config,dimension,delta,ci95,t,sign,n\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.3f},{r[5]},{r[6]}\n")
        print("\nwrote", a.out_csv)


if __name__ == "__main__":
    main()
