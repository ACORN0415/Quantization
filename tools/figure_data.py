# -*- coding: utf-8 -*-
"""그림 수치를 **원자료에서만** 읽는다. 스크립트에 손으로 적지 않는다.

본문 표를 CSV 에서 생성하는 것과 같은 이유다 — 손으로 옮기는 순간 정정이
문서와 그림에서 따로 살아남는다(세션 6·7에서 실제로 그랬다).
"""
import csv, json, os, statistics as st, sys

R = os.path.expanduser("~/gpu/Self-Forcing/results")
S6 = f"{R}/session6"
sys.path.insert(0, "/home/devel/gpu/Quantization/tools")
from s6stats import paired


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _k3(bits, quantity):
    for r in _rows(f"{S6}/gateK3_interaction.csv"):
        if r["bits"] == bits and r["quantity"] == quantity:
            return {"mean": float(r["mean"]), "lo": float(r["lo"]),
                    "hi": float(r["hi"]), "t": float(r["t"]),
                    "n": int(r["n"]), "sign": int(r["sign"]),
                    "src": "gateK3_interaction.csv", "quantity": quantity}
    raise KeyError((bits, quantity))


def _ka(label):
    for r in _rows(f"{S6}/gateK_aligned.csv"):
        if r["label"] == label:
            m, ci = float(r["mean_delta"]), float(r["mean_ci95"])
            return {"mean": m, "lo": m - ci, "hi": m + ci,
                    "t": float(r["mean_t"]), "n": int(r["n"]),
                    "sign": int(r["mean_sign"]),
                    "src": "gateK_aligned.csv", "label": label}
    raise KeyError(label)


def _per_prompt(path, arm, key="musiq"):
    ch = json.load(open(path))[arm]
    n = len(ch[next(iter(ch))][key])
    return [st.fmean(ch[c][key][i] for c in ch) for i in range(n)]


def _contrast(a, b, src):
    d = [x - y for x, y in zip(a, b)]
    r = paired(d, 0.95)
    out = {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "t": r["t"],
           "n": r["n"], "sign_pos": sum(1 for v in d if v > 0),
           "per_prompt": d, "src": src}
    if r["se"] == 0:
        # 모든 차이가 0 이면 t = 0/0 으로 정의되지 않는다. inf 를 그대로 두면
        # 재현 자료를 읽는 쪽이 "무한히 유의"로 오독한다.
        out["t"] = None
        out["t_note"] = "all per-prompt differences are exactly 0; t is 0/0 (undefined), not infinite"
        out["lo"] = out["hi"] = None
        out["ci_note"] = "no interval: identical outcomes, not an estimate with uncertainty"
    return out


# ---------------------------------------------------------------- 그림 1
def fig1():
    out = {"int2": {}, "int4": {}}
    out["int2"]["alignment_only"] = _ka("A1 INT2  align ON vs OFF")
    out["int2"]["sink_only"] = _k3("int2", "simple: sink | align OFF")
    out["int2"]["sum_predicted"] = _k3("int2", "sum of simple effects (if additive)")
    out["int2"]["both"] = _k3("int2", "actual joint effect (ON,s3 - OFF,s0)")
    out["int2"]["interaction"] = _k3("int2", "INTERACTION (align x sink)")
    out["int4"]["alignment_only"] = _ka("A1 INT4  align ON vs OFF")
    out["int4"]["sink_only"] = _k3("int4", "simple: sink | align OFF")
    out["int4"]["sum_predicted"] = _k3("int4", "sum of simple effects (if additive)")
    out["int4"]["both"] = _k3("int4", "actual joint effect (ON,s3 - OFF,s0)")
    out["int4"]["interaction"] = _k3("int4", "INTERACTION (align x sink)")
    # 일관성 검사: 단순효과의 합이 실제로 합인가
    for b in ("int2", "int4"):
        s = out[b]["alignment_only"]["mean"] + out[b]["sink_only"]["mean"]
        assert abs(s - out[b]["sum_predicted"]["mean"]) < 5e-4, (b, s)
    return out


# ---------------------------------------------------------------- 그림 2
def fig2():
    """A1 **INT4** 생성 실행 중 audit. 단위는 **생존 슬롯 비교**이고 비교 대상은
    **K 코드만**이다(`_requant_audit` 는 state["k"]["q"] 만 본다).
    x 축은 시간상의 퇴출 전/후가 아니라 그 비교에서 내용이 **이동했는가**다
    (`shifted` 필드). A2 는 **INT2** 실행이라 같은 축에 올리지 않는다."""
    out = {"bits": 4, "unit": "surviving-slot comparisons (K codes only)",
           "axis": "no shift / shifted"}
    for tag, path in (("align_off", f"{S6}/requant/align_off.json"),
                      ("align_on", f"{S6}/requant/align_on.json")):
        d = json.load(open(path))
        rec = d["records"]
        ns = sum(x["slots_requantized"] for x in rec if x["shifted"] == 0)
        sh = sum(x["slots_requantized"] for x in rec if x["shifted"] > 0)
        cns = sum(x["slots_compared"] for x in rec if x["shifted"] == 0)
        csh = sum(x["slots_compared"] for x in rec if x["shifted"] > 0)
        out[tag] = {"no_shift": ns, "shifted": sh,
                    "no_shift_compared": cns, "shifted_compared": csh,
                    "no_shift_rate": ns / cns, "shifted_rate": sh / csh if csh else 0.0,
                    "total": d["total_slots_requantized"],
                    "total_compared": d["total_slot_comparisons"],
                    "src": os.path.basename(path)}
        assert ns + sh == d["total_slots_requantized"]
        assert cns + csh == d["total_slot_comparisons"]
    # A2(INT2)는 참고용으로만 담는다 — 그림에 같은 축으로 올리지 않는다
    a2 = {}
    for tag, path in (("align_off", f"{S6}/requant_a2/a2_off.json"),
                      ("align_on", f"{S6}/requant_a2/a2_on.json")):
        d = json.load(open(path))
        rec = d["records"]
        a2[tag] = {"sink_changed": sum(1 for x in rec
                                       for i in x.get("changed_slots", []) if i == 0),
                   "sink_compared": sum(1 for x in rec if x["slots_compared"] >= 1),
                   "src": os.path.basename(path)}
    out["a2_int2_sink_reference"] = a2
    out["a2_note"] = ("A2 는 INT2 실행이고 대상이 sink 슬롯 하나다. "
                      "비트폭이 다르므로 그림 2의 축에 올리지 않는다.")
    return out


# ---------------------------------------------------------------- 그림 3
def fig3():
    """`gateK1_lineage.py` 의 멱등 probe. 재는 양은 **재양자화 전후 복원값의 상대
    L2 차이**이지 최초 BF16 원본 대비 오차가 아니다 —
    `same_grid = rel(d2, d1)` 에서 d1 은 1차 복원, d2 는 그것을 다시 양자화·복원한
    값이다. 따라서 0 의 뜻은 "원본 오차 없음"이 아니라 **"재양자화가 저장 내용을
    바꾸지 않았다(멱등)"**이다."""
    d = json.load(open(f"{S6}/gateK1_bitdiff.json"))
    out = {"same_grid": {}, "shifted_grid": {}, "shift_mod_block": d["shift_mod_block"],
           "src": "gateK1_bitdiff.json (gateK1_lineage.py)", "layers": d["layers"],
           "quantity": "relative L2 change of the reconstruction across re-quantization",
           "zero_means": "idempotent (re-quantization did not change stored content)"}
    for bits in d["bits"]:
        runs = [r for r in d["runs"] if r["bits"] == bits]
        out["same_grid"][f"int{bits}"] = max(r["same_grid_idempotence"] for r in runs)
        out["shifted_grid"][f"int{bits}"] = max(r["shifted_grid_idempotence"] for r in runs)
    return out


# ---------------------------------------------------------------- 그림 4
def fig4():
    P = f"{S6}/gateK0_anchor_control_per_prompt.json"
    oldest = _per_prompt(P, "k0_oldest")
    fixed0 = _per_prompt(P, "k0_fixed0")
    fixed012 = _per_prompt(P, "k0_fixed012")
    oracle3 = _per_prompt(P, "k0_oracle3")
    src = "gateK0_anchor_control_per_prompt.json"
    out = {
        "anchor_presence": _contrast(fixed0, oldest, src),
        "anchor_budget_1to3": _contrast(fixed012, fixed0, src),
        "adaptive_selection": _contrast(oracle3, fixed012, src),
        "total_oracle_vs_oldest": _contrast(oracle3, oldest, src),
        "order": ["anchor_presence", "anchor_budget_1to3", "adaptive_selection"],
    }
    # 적응 선택은 '구간'이 아니라 '선택이 동일했다'는 관측이다 — CI 를 그리지 않는다.
    ad = out["adaptive_selection"]
    out["adaptive_is_exact_zero"] = all(abs(v) == 0.0 for v in ad["per_prompt"])
    return out


# ---------------------------------------------------------------- 그림 6
def fig6():
    fin = json.load(open(f"{S6}/b2/final_perf.bytes.json"))
    bf = json.load(open(f"{S6}/b2/bf16_perf.bytes.json"))
    man = json.load(open(f"{S6}/final_recipe_manifest.json"))
    dec = man["compression"]["decomposition_bytes"]
    return {
        "memory_bytes": {"bf16": bf["kv_resident_bytes"],
                         "final": fin["kv_resident_bytes"],
                         "final_packed_segments": dec["packed_segments"],
                         "final_bf16_pending": dec["bf16_pending_chunk"],
                         "residual": dec["residual"]},
        "peak_alloc_bytes": {"bf16": bf["peak_allocated_bytes"],
                             "final": fin["peak_allocated_bytes"]},
        "inference_s": {"bf16": man["latency"]["inference_s"]["bf16"],
                        "final": man["latency"]["inference_s"]["final"]},
        "provenance": {"commit": fin["commit"], "diff_sha256": fin["diff_sha256"]},
        "src": ["b2/final_perf.bytes.json", "b2/bf16_perf.bytes.json",
                "final_recipe_manifest.json"],
    }


def collect(with_fig5=True):
    data = {"fig1": fig1(), "fig2": fig2(), "fig3": fig3(),
            "fig4": fig4(), "fig6": fig6()}
    if with_fig5:
        from figure_data_fig5 import fig5
        data["fig5"] = fig5()
    return data


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_fig5", action="store_true")
    ap.add_argument("--out", default="figs/figure_data.json")
    a = ap.parse_args()
    d = collect(not a.no_fig5)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(d, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: list(v) for k, v in d.items()}, ensure_ascii=False, indent=1))
    print(f"wrote {a.out}")
