# -*- coding: utf-8 -*-
"""그림 5 자료 — λ sweep, 두 지표.

**한 배치·한 평가에서 읽는다.** 앞서 MUSIQ λ 값은 두 파일에 흩어져 있었다
(`gateH_fr_musiq` 에 λ0.5·2·8, `gateH_pilot_musiq` 에 λ1·4, 공통 기준선 없음).
그대로 이으면 배치 효과와 λ 반응을 구분할 수 없다. VBench 가 쓴 **같은 비디오**로
MUSIQ 를 8팔 한 번에 다시 평가해(`fig5_lambda_musiq*`) 두 패널의 출처를 맞췄다.
생성은 다시 하지 않았다 — 기존 비디오에 지표만 다시 걸었다.
"""
import json, os, statistics as st, sys

S6 = os.path.expanduser("~/gpu/Self-Forcing/results/session6")
sys.path.insert(0, "/home/devel/gpu/Quantization/tools")
from s6stats import paired

LAMS = [("lam0p5", 0.5), ("lam1", 1.0), ("lam2", 2.0), ("lam4", 4.0), ("lam8", 8.0)]


def _pp(path, arm, key):
    """프롬프트별 값. MUSIQ 는 chunk 별 dict, VBench 는 프롬프트 리스트다."""
    a = json.load(open(path))[arm]
    v = a[key] if isinstance(a, dict) and key in a else None
    if isinstance(v, list):                       # VBench: 프롬프트 리스트
        return list(v)
    ch = a                                        # MUSIQ: {chunk: {key: [...]}}
    n = len(ch[next(iter(ch))][key])
    return [st.fmean(ch[c][key][i] for c in ch) for i in range(n)]


def _c(a, b):
    d = [x - y for x, y in zip(a, b)]
    r = paired(d, 0.95)
    return {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "t": r["t"],
            "n": r["n"], "sign_pos": sum(1 for v in d if v > 0)}


def fig5():
    MQ = f"{S6}/fig5_lambda_musiq_per_prompt.json"
    VB = f"{S6}/vbench_lambda_per_prompt.json"
    base_mq = _pp(MQ, "A1_int2", "musiq")
    base_sc = _pp(VB, "A1_int2", "subject_consistency")
    out = {"musiq": {}, "subject_consistency": {},
           "baseline_arm": "A1_int2 (무보정 INT2)",
           "src": {"musiq": "fig5_lambda_musiq_per_prompt.json (8팔 동시 평가)",
                   "subject_consistency": "vbench_lambda_per_prompt.json"},
           "note": "두 패널 모두 같은 비디오에서 나왔다. MUSIQ 는 λ 팔이 두 파일에 "
                   "흩어져 있던 문제를 없애려고 한 번에 다시 평가한 것이다."}
    for arm, lam in LAMS:
        out["musiq"][str(lam)] = _c(_pp(MQ, arm, "musiq"), base_mq)
        out["subject_consistency"][str(lam)] = _c(
            _pp(VB, arm, "subject_consistency"), base_sc)
    # flatten 은 λ 축 위의 점이 아니다 — 별도 마커
    out["flat1"] = {
        "musiq": _c(_pp(MQ, "flat1", "musiq"), base_mq),
        "subject_consistency": _c(_pp(VB, "flat1", "subject_consistency"), base_sc)}
    # BF16 참고선 (λ 축 밖)
    out["bf16_ref"] = {
        "musiq": _c(_pp(MQ, "A1_bf16", "musiq"), base_mq),
        "subject_consistency": _c(_pp(VB, "A1_bf16", "subject_consistency"), base_sc)}
    # 캡션이 인용하는 직접 대비
    out["direct_lam0p5_minus_lam1"] = {
        "musiq": _c(_pp(MQ, "lam0p5", "musiq"), _pp(MQ, "lam1", "musiq")),
        "subject_consistency": _c(_pp(VB, "lam0p5", "subject_consistency"),
                                  _pp(VB, "lam1", "subject_consistency"))}
    return out


if __name__ == "__main__":
    d = fig5()
    print(f"{'λ':>5} {'MUSIQ 차':>10} {'CI':>22}   {'subj 차':>10} {'CI':>22}")
    for _, lam in LAMS:
        m = d["musiq"][str(lam)]; s = d["subject_consistency"][str(lam)]
        print(f"{lam:5.1f} {m['mean']:+10.3f} [{m['lo']:+8.3f},{m['hi']:+8.3f}]   "
              f"{s['mean']:+10.4f} [{s['lo']:+8.4f},{s['hi']:+8.4f}]")
    for k in ("flat1", "bf16_ref", "direct_lam0p5_minus_lam1"):
        m = d[k]["musiq"]; s = d[k]["subject_consistency"]
        print(f"{k:>26s} MUSIQ {m['mean']:+7.3f} [{m['lo']:+7.3f},{m['hi']:+7.3f}]  "
              f"subj {s['mean']:+8.4f} [{s['lo']:+8.4f},{s['hi']:+8.4f}]")
