"""Session 6 — paired comparison of any two runs from per-prompt MUSIQ JSONs.

Both aggregations are always reported, because session 5 showed they disagree
and that the drop metric alone reads a destroyed chunk 0 as a reward:
  mean  per-prompt mean MUSIQ over chunks 0..20   (primary)
  drop  per-prompt MUSIQ(c20) - MUSIQ(c0)         (reported alongside)

Usage:
  analyze_s6.py --pairs NEW_JSON:tag BASE_JSON:tag [label] ...
  analyze_s6.py --spec spec.json
"""
import argparse, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s6stats import paired as _paired   # addendum A2: t-based CI, not 1.96

R = os.path.expanduser("~/gpu/Self-Forcing/results")
_cache = {}


def load(path):
    if path not in _cache:
        _cache[path] = json.load(open(path if os.path.isabs(path)
                                      else os.path.join(R, path)))
    return _cache[path]


def mat(path, tag):
    """-> [prompt][chunk] MUSIQ"""
    d = load(path)
    if tag not in d:
        raise KeyError(f"{tag} not in {path} (have {sorted(d)})")
    d = d[tag]
    nch = len(d)
    npr = len(d["0"]["musiq"])
    return [[d[str(i)]["musiq"][p] for i in range(nch)] for p in range(npr)]


def paired(da):
    r = _paired(da)
    return r["mean"], r["sd"], r["t"], r["sign"]


def ci95(sd, n):
    """t(n-1, .975) * SE -- addendum A2. 1.96 understates by ~15% at n=10."""
    from s6stats import tcrit
    return tcrit(n) * sd / math.sqrt(n) if n > 1 else float("nan")


def compare(new, base, label, out=None):
    mean_d = [sum(a) / len(a) - sum(b) / len(b) for a, b in zip(new, base)]
    drop_d = [(a[-1] - a[0]) - (b[-1] - b[0]) for a, b in zip(new, base)]
    row = {"label": label, "n": len(mean_d)}
    for name, da in (("mean", mean_d), ("drop", drop_d)):
        m, sd, t, sg = paired(da)
        row[name] = {"delta": m, "sd": sd, "t": t, "sign": sg, "ci95": ci95(sd, len(da))}
        print(f"  {label:38s} {name}: {m:+7.2f} ±{ci95(sd, len(da)):5.2f}(CI95)  "
              f"t={t:+6.2f}  sign {sg}/{len(da)}")
    row["new_mean_abs"] = sum(sum(a) / len(a) for a in new) / len(new)
    row["base_mean_abs"] = sum(sum(b) / len(b) for b in base) / len(base)
    if out is not None:
        out.append(row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True,
                    help="JSON: [{label, new:[file,tag], base:[file,tag]}, ...]")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_json", default=None)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    rows = []
    for s in spec:
        try:
            compare(mat(*s["new"]), mat(*s["base"]), s["label"], rows)
        except (KeyError, FileNotFoundError) as e:
            print(f"  {s['label']:38s} SKIP ({e})")
    if a.out_json:
        json.dump(rows, open(a.out_json, "w"), indent=1)
    if a.out_csv:
        import csv as _csv
        with open(a.out_csv, "w", newline="") as f:
            # labels contain commas; csv.writer quotes them, a manual f-string
            # did not and shifted every column to the right of the label.
            w = _csv.writer(f)
            w.writerow(["label", "n", "mean_delta", "mean_ci95", "mean_t",
                        "mean_sign", "drop_delta", "drop_ci95", "drop_t",
                        "drop_sign", "new_mean_abs", "base_mean_abs"])
            for r in rows:
                w.writerow([r["label"], r["n"],
                            f"{r['mean']['delta']:.4f}", f"{r['mean']['ci95']:.4f}",
                            f"{r['mean']['t']:.3f}", r["mean"]["sign"],
                            f"{r['drop']['delta']:.4f}", f"{r['drop']['ci95']:.4f}",
                            f"{r['drop']['t']:.3f}", r["drop"]["sign"],
                            f"{r['new_mean_abs']:.4f}", f"{r['base_mean_abs']:.4f}"])
        print("wrote", a.out_csv)


if __name__ == "__main__":
    main()
