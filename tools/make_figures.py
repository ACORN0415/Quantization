# -*- coding: utf-8 -*-
"""본문 그림 6장. 사양: 문서/지시서/FIGURES_spec.md

수치는 전부 `figs/figure_data.json` 에서 읽는다 — 이 파일은 `tools/figure_data.py`
가 원자료에서 생성한다. **스크립트에 손으로 적힌 수치는 없다.**

전역 규칙(사양 §0): 벡터 + PNG 미리보기 · 흑백 안전(해치·마커 병용) ·
이중 y축 금지 · 0 기준선 실선 · 점선 금지 · 선택적 직접 라벨 ·
CI 가 0 을 포함하는 막대는 윤곽선만 · 계산값은 해치 + 라벨.
"""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

FIGS = os.path.expanduser("~/gpu/Self-Forcing/figs")
D = json.load(open(f"{FIGS}/figure_data.json"))

GRID = "#c8c8c8"      # 배경에서 한 단계만 어두운 회색
INK = "#1a1a1a"
FILL = "#7f7f7f"      # 측정값 실채움
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK, "axes.linewidth": 0.6,
    "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "hatch.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def style_ax(ax, zero=True):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if zero:
        ax.axhline(0, color=INK, lw=0.9, zorder=3)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIGS}/{name}.pdf")
    fig.savefig(f"{FIGS}/{name}.png")
    plt.close(fig)
    print(f"  wrote figs/{name}.pdf + .png")


def bar_with_ci(ax, x, d, *, hatch=None, label=None, width=0.68):
    """CI 가 0 을 포함하면 윤곽선만 — 이 논문 여러 곳의 요점이다."""
    crosses = d["lo"] <= 0 <= d["hi"]
    kw = dict(width=width, edgecolor=INK, linewidth=0.9, zorder=2)
    if hatch:
        ax.bar(x, d["mean"], facecolor="white", hatch=hatch, **kw)
    elif crosses:
        ax.bar(x, d["mean"], facecolor="white", **kw)
    else:
        ax.bar(x, d["mean"], facecolor=FILL, **kw)
    ax.errorbar(x, d["mean"], yerr=[[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]],
                fmt="none", ecolor=INK, elinewidth=0.9, capsize=3, zorder=4)
    return crosses


# ------------------------------------------------------------------ 그림 1
def fig1():
    """막대 5개 — 상호작용을 **직접** 그린다. 가산 예측과 결합 효과의 CI 를 눈으로
    비교하게 두면 독자가 상호작용 검정을 추론해야 한다. 상호작용은 그 자체로
    추정량이고 CI 가 있다."""
    f = D["fig1"]
    keys = ["alignment_only", "sink_only", "sum_predicted", "both", "interaction"]
    names = ["alignment\nonly", "sink\nonly", "sum\npredicted",
             "both\nmeasured", "interaction"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), sharey=True)
    for ax, bits, title in zip(axes, ("int2", "int4"), ("INT2", "INT4")):
        crossed = []
        for i, k in enumerate(keys):
            h = "//////" if k == "sum_predicted" else (
                "xxxxxx" if k == "interaction" else None)
            crossed.append(bar_with_ci(ax, i, f[bits][k], hatch=h))
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(names)
        style_ax(ax)
        ax.set_title(title)
        for i, k in enumerate(keys):
            d = f[bits][k]
            if crossed[i] or k in ("sum_predicted", "interaction", "both"):
                lab = f"{d['mean']:+.2f}"
                if crossed[i]:
                    lab += "\nCI incl. 0"
                ax.annotate(lab, (i, d["hi"]), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=7.5)
    axes[0].set_ylabel("MUSIQ difference")
    fig.legend(handles=[
        Patch(facecolor=FILL, edgecolor=INK, label="measured, CI excludes 0"),
        Patch(facecolor="white", edgecolor=INK, label="measured, CI includes 0"),
        Patch(facecolor="white", edgecolor=INK, hatch="//////",
              label="computed (additivity assumption)"),
        Patch(facecolor="white", edgecolor=INK, hatch="xxxxxx",
              label="interaction estimate")],
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.015), fontsize=7.5)
    fig.subplots_adjust(bottom=0.30)
    save(fig, "fig1_interaction")


# ------------------------------------------------------------------ 그림 2
def fig2():
    """y 축은 **검출 비율**이다. 두 집단의 분모가 다르므로(39,690 vs 7,560)
    원수치를 같은 높이 축에 올리면 "5.6배"로 오독된다."""
    f = D["fig2"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    xs = [0, 1]
    w = 0.36
    off = [100 * f["align_off"]["no_shift_rate"], 100 * f["align_off"]["shifted_rate"]]
    on = [100 * f["align_on"]["no_shift_rate"], 100 * f["align_on"]["shifted_rate"]]
    raw_off = [(f["align_off"]["no_shift"], f["align_off"]["no_shift_compared"]),
               (f["align_off"]["shifted"], f["align_off"]["shifted_compared"])]
    raw_on = [(f["align_on"]["no_shift"], f["align_on"]["no_shift_compared"]),
              (f["align_on"]["shifted"], f["align_on"]["shifted_compared"])]
    ax.bar([x - w / 2 for x in xs], off, width=w, facecolor=FILL, edgecolor=INK,
           linewidth=0.9, label="grid alignment OFF", zorder=2)
    ax.bar([x + w / 2 for x in xs], on, width=w, facecolor="white", edgecolor=INK,
           linewidth=0.9, hatch="\\\\\\", label="grid alignment ON", zorder=2)
    for x, v, (a, b) in zip(xs, off, raw_off):
        ax.annotate(f"{a:,} / {b:,}", (x - w / 2, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8)
    for x, v, (a, b) in zip(xs, on, raw_on):
        ax.annotate(f"{a} / {b:,}", (x + w / 2, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(["no shift", "shifted"])
    ax.set_ylabel("surviving-slot comparisons with\nchanged K codes (%)")
    ax.set_ylim(0, 118)
    ax.text(0.99, 0.97, f"A1 · INT{f['bits']} generation run\nK codes only · unit = surviving-slot comparison",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.6))
    style_ax(ax, zero=False)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig2_requant_counts")


# ------------------------------------------------------------------ 그림 3
def fig3():
    f = D["fig3"]
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    xs = [0, 1]
    w = 0.36
    same = [f["same_grid"]["int4"], f["same_grid"]["int2"]]
    shift = [f["shifted_grid"]["int4"], f["shifted_grid"]["int2"]]
    ax.bar([x - w / 2 for x in xs], same, width=w, facecolor=FILL,
           edgecolor=INK, linewidth=0.9, label="same grid (0 = idempotent)", zorder=2)
    ax.bar([x + w / 2 for x in xs], shift, width=w, facecolor="white",
           edgecolor=INK, linewidth=0.9, hatch="//////",
           label=f"grid shifted by {f['shift_mod_block']} tokens", zorder=2)
    for x in xs:
        ax.annotate("0", (x - w / 2, 0), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=9, weight="bold")
    for x, v in zip(xs, shift):
        ax.annotate(f"{v:.3f}", (x + w / 2, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["INT4", "INT2"])
    ax.set_ylabel("relative L2 change of the reconstruction\nacross re-quantization")
    ax.set_ylim(0, max(shift) * 1.20)
    style_ax(ax, zero=False)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig3_idempotence")


# ------------------------------------------------------------------ 그림 4
def fig4():
    """폭포. 각 성분이 자기 행·자기 CI 를 갖는다. 누적 끝점의 CI 는 프롬프트별 총
    대비에서 다시 계산한 것이며 증분 CI 와 다른 모양(마름모 끝, 캡 없음)으로 그린다.
    적응 선택은 '구간 없음' — 점과 글자만 두고 막대·오차막대 모양을 피한다."""
    f = D["fig4"]
    order = f["order"]
    names = ["anchor presence\n(1 chunk pinned)", "anchor budget\n1 → 3 chunks",
             "adaptive selection"]
    hatches = [None, "//////", None]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    left = 0.0
    rows = []
    for i, (k, nm, h) in enumerate(zip(order, names, hatches)):
        d = f[k]
        y = len(order) - i
        if k != "adaptive_selection":
            kw = dict(height=0.55, edgecolor=INK, linewidth=0.9, zorder=2)
            if h:
                ax.barh(y, d["mean"], left=left, facecolor="white", hatch=h, **kw)
            else:
                ax.barh(y, d["mean"], left=left, facecolor=FILL, **kw)
            ax.errorbar(left + d["mean"], y,
                        xerr=[[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]],
                        fmt="none", ecolor=INK, elinewidth=0.9, capsize=3, zorder=4)
            ax.annotate(f"{d['mean']:+.2f}  [{d['lo']:+.2f}, {d['hi']:+.2f}]",
                        (left + d["mean"] / 2, y + 0.33), va="bottom", ha="center",
                        fontsize=8)
        else:
            ax.plot([left], [y], marker="o", ms=4, color=INK, zorder=5)
            ax.annotate(f"{d['mean']:+.2f}   (identical selections, 140/140)",
                        (left, y + 0.22), va="bottom", ha="center", fontsize=8)
        rows.append((y, nm))
        left += d["mean"]
    tot = f["total_oracle_vs_oldest"]
    ax.barh(0, tot["mean"], height=0.55, facecolor="white", edgecolor=INK,
            linewidth=1.4, zorder=2)
    ax.errorbar(tot["mean"], 0, xerr=[[tot["mean"] - tot["lo"]],
                                      [tot["hi"] - tot["mean"]]],
                fmt="none", ecolor=INK, elinewidth=1.2, capsize=0, zorder=4)
    ax.plot([tot["lo"], tot["hi"]], [0, 0], linestyle="none", marker="D", ms=4,
            color=INK, mfc="white", zorder=5)
    ax.annotate(f"{tot['mean']:+.2f}  [{tot['lo']:+.2f}, {tot['hi']:+.2f}]  (cumulative)",
                (tot["mean"] / 2, 0.33), va="bottom", ha="center", fontsize=8,
                weight="bold")
    rows.append((0, "total\n(oracle − oldest)"))
    ax.set_yticks([y for y, _ in rows])
    ax.set_yticklabels([n for _, n in rows], fontsize=8)
    ax.set_xlabel("MUSIQ difference   (components applied in this order, top to bottom)")
    ax.set_xlim(0, 15.5)
    ax.set_ylim(-0.7, len(order) + 0.9)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.axvline(0, color=INK, lw=0.9, zorder=3)
    save(fig, "fig4_anchor_decomposition")


# ------------------------------------------------------------------ 그림 5
def fig5():
    """세로 2행. 각 행은 [λ sweep | flatten] 두 패널이 y 축을 공유한다.
    flatten 은 λ 축 위의 점이 아니므로 λ 축에 놓지 않는다 — 놓으면 λ1 과 λ2 사이의
    실험으로 읽힌다."""
    f = D["fig5"]
    lams = [0.5, 1.0, 2.0, 4.0, 8.0]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.4), sharex="col",
                             gridspec_kw=dict(width_ratios=[5, 1], wspace=0.06,
                                              hspace=0.12))
    panels = ((0, "musiq", "MUSIQ difference", "o"),
              (1, "subject_consistency", "subject_consist.\ndifference", "s"))
    for row, key, ylab, mk in panels:
        ax, axf = axes[row, 0], axes[row, 1]
        ys = [f[key][str(l)]["mean"] for l in lams]
        lo = [f[key][str(l)]["mean"] - f[key][str(l)]["lo"] for l in lams]
        hi = [f[key][str(l)]["hi"] - f[key][str(l)]["mean"] for l in lams]
        ax.errorbar(lams, ys, yerr=[lo, hi], marker=mk, ms=5, color=INK,
                    lw=1.1, elinewidth=0.9, capsize=3, zorder=4, mfc="white")
        ax.axvline(1.0, color=GRID, lw=0.8, zorder=1)
        style_ax(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks(lams)
        ax.set_xticklabels([f"{l:g}" for l in lams])
        ax.set_ylabel(ylab)
        # flatten 소패널 — 같은 y 축
        fl = f["flat1"][key]
        axf.errorbar([0], [fl["mean"]], yerr=[[fl["mean"] - fl["lo"]],
                                             [fl["hi"] - fl["mean"]]],
                     marker="^", ms=6, color=INK, lw=0, elinewidth=0.9,
                     capsize=3, zorder=5, mfc="white")
        axf.sharey(ax)
        axf.set_xlim(-0.8, 0.8)
        axf.set_xticks([0])
        axf.set_xticklabels(["flatten\n(λ=1)"], fontsize=8)
        axf.tick_params(axis="y", left=False, labelleft=False)
        style_ax(axf)
        axf.spines["left"].set_visible(False)
    axes[1, 0].set_xlabel("λ  (correction strength)")
    d4 = f["subject_consistency"]["4.0"]
    # 빈 자리는 아래 패널의 오른쪽 위(λ4~8, y>0)다. 곡선 아래·눈금 근처는 겹친다.
    axes[1, 0].annotate(f"λ=4: {d4['mean']:+.3f}  (below uncorrected)",
                        (4.0, d4["mean"]), textcoords="offset points",
                        xytext=(0, 96), ha="center", va="bottom", fontsize=7.5,
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=INK,
                                        shrinkB=5))
    axes[0, 0].set_title("difference vs uncorrected INT2 (λ=0), n=10", fontsize=9,
                         loc="left")
    fig.subplots_adjust(left=0.17)
    save(fig, "fig5_lambda_two_metrics")


# ------------------------------------------------------------------ 그림 6
def fig6():
    f = D["fig6"]
    GB = 1e9
    m = f["memory_bytes"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    ax = axes[0]
    ax.bar(0, m["bf16"] / GB, width=0.55, facecolor=FILL, edgecolor=INK,
           linewidth=0.9, zorder=2)
    seg = m["final_packed_segments"] / GB
    pend = m["final_bf16_pending"] / GB
    ax.bar(1, seg, width=0.55, facecolor=FILL, edgecolor=INK, linewidth=0.9, zorder=2)
    ax.bar(1, pend, bottom=seg, width=0.55, facecolor="white", edgecolor=INK,
           linewidth=0.9, hatch="//////", zorder=2)
    ax.annotate(f"{m['bf16']/GB:.3f}", (0, m["bf16"] / GB),
                textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.annotate(f"{m['final']/GB:.3f}", (1, (seg + pend)),
                textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    # 범례를 두지 않는다 — BF16 막대가 "packed segments" 로 오독된다.
    # 쌓기는 'final' 막대에만 해당하므로 그 막대에 직접 적는다.
    ax.annotate(f"BF16 pending\n{pend:.3f} GB", (1.32, seg + pend / 2),
                ha="left", va="center", fontsize=7.5)
    ax.annotate(f"packed segments\n{seg:.3f} GB", (1.32, seg / 2),
                ha="left", va="center", fontsize=7.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["BF16\nA1 window", "final\nA4 + sink 3"])
    ax.set_ylabel("resident KV cache (GB)")
    ax.set_xlim(-0.6, 2.6)
    ratio = m["bf16"] / m["final"]
    ax.annotate(f"{ratio:.2f}× smaller", (0.5, m["bf16"] / GB * 0.62), ha="center",
                fontsize=9, weight="bold")
    style_ax(ax, zero=False)
    ax = axes[1]
    t = f["inference_s"]
    ax.bar([0, 1], [t["bf16"], t["final"]], width=0.55, facecolor=FILL,
           edgecolor=INK, linewidth=0.9, zorder=2)
    for x, k in ((0, "bf16"), (1, "final")):
        ax.annotate(f"{t[k]:.1f}", (x, t[k]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["BF16\nA1 window", "final\nA4 + sink 3"])
    ax.set_ylabel("inference() total (s)")
    pct = 100 * (t["final"] / t["bf16"] - 1)
    ax.annotate(f"{pct:+.1f}%", (0.5, t["final"] * 0.55), ha="center",
                fontsize=9, weight="bold")
    style_ax(ax, zero=False)
    save(fig, "fig6_cost")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    os.makedirs(FIGS, exist_ok=True)
    for w in which:
        print(f"fig{w}:")
        globals()[f"fig{w}"]()
