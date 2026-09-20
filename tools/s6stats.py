"""Shared paired statistics for session 6.

Addendum A2: the 95% CI must use t(n-1, .975), not 1.96. With n = 10 that is
2.262, so every interval computed with the normal quantile was ~15% too narrow.
TOST (addendum B3) needs the 90% interval, t(n-1, .95).
"""
import math

try:
    from scipy import stats as _st

    def tcrit(n, conf=0.95):
        return float(_st.t.ppf(0.5 + conf / 2.0, n - 1))
except Exception:                                              # noqa: BLE001
    _T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
            8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 15: 2.145,
            20: 2.093, 30: 2.045}
    _T90 = {2: 6.314, 3: 2.920, 4: 2.353, 5: 2.132, 6: 2.015, 7: 1.943,
            8: 1.895, 9: 1.860, 10: 1.833, 11: 1.812, 12: 1.796, 15: 1.753,
            20: 1.729, 30: 1.699}

    def tcrit(n, conf=0.95):
        tbl = _T95 if abs(conf - 0.95) < 1e-9 else _T90
        if n in tbl:
            return tbl[n]
        ks = sorted(tbl)
        return tbl[min(ks, key=lambda k: abs(k - n))]


def paired(v, conf=0.95):
    """-> dict(mean, sd, se, ci, t, n, sign, lo, hi) with a t-based interval."""
    n = len(v)
    m = sum(v) / n
    if n < 2:
        return {"mean": m, "sd": 0.0, "se": 0.0, "ci": float("nan"),
                "t": float("nan"), "n": n, "sign": n, "lo": m, "hi": m}
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    se = sd / math.sqrt(n)
    ci = tcrit(n, conf) * se
    return {"mean": m, "sd": sd, "se": se, "ci": ci,
            "t": (m / se if se else float("inf")), "n": n,
            "sign": sum(1 for x in v if (x > 0) == (m > 0)),
            "lo": m - ci, "hi": m + ci}


def tost(v, margin):
    """Two one-sided tests for equivalence within +/- margin.

    Equivalent iff the 90% CI of the mean difference lies inside the margin.
    Returns the verdict plus the interval, so a near miss is visible.
    """
    r = paired(v, conf=0.90)
    r["margin"] = margin
    r["equivalent"] = (r["lo"] > -margin) and (r["hi"] < margin)
    return r
