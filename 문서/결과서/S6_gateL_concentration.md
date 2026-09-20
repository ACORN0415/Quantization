# Gate L3 — why one scalar reproduces the per-key Jensen correction

Session 5 found that erasing the per-key structure of the Taylor bias (the
`--tum_flatten_slots` control) reproduced the structured correction: +10.49 vs
+10.04 mean MUSIQ, the difference not significant. That was an empirical
coincidence with no explanation. This gate asks whether it is forced — whether
the per-key bias *must* collapse to a constant once the head dimension is large
enough — and what the condition is.

## 1. Setup

For a query `q` (one head, `d = 128` channels) and a cached key `i`, the
Taylor/Jensen bias the correction subtracts from the logit is

    b_i = (1 / 24d) * sum_c q_c^2 * Delta_ic^2                                (1)

where `Delta_ic` is the RTN quantization step of key `i` in channel `c`
(constant inside each 16-token block, so `i` indexes blocks). Write

    w_c = q_c^2 / (24 d)   (>= 0, fixed by the query)
    X_ic = Delta_ic^2      (>= 0, fixed by the cache)
    =>  b_i = sum_c w_c X_ic

Across the cached keys let `m_c = E_i[X_ic]`, `s_c = sd_i(X_ic)`,
`kappa_c = s_c / m_c` (the per-channel coefficient of variation of the squared
step), and `rho_cc'= Corr_i(X_ic, X_ic')`. Then

    mu_b      = sum_c a_c,                 a_c = w_c m_c                      (2)
    sigma_b^2 = sum_{c,c'} a_c a_c' kappa_c kappa_c' rho_cc'                  (3)

`a_c` is channel `c`'s contribution to the mean bias, so (2) says the mean is
just the total contribution and (3) says the spread is governed by how those
contributions co-vary across keys.

## 2. Two regimes

**(i) Channel-idiosyncratic variation (`rho = I`).** Then (3) collapses to a
sum of squares and

    sigma_b / mu_b = sqrt( sum_c a_c^2 kappa_c^2 ) / sum_c a_c
                   <= kappa_max / sqrt(d_eff),
    d_eff = (sum_c a_c)^2 / sum_c a_c^2                                       (4)

`d_eff` is the participation ratio of the contributions: it equals `d` when all
channels contribute equally and drops when a few channels dominate. So in this
regime the *relative* spread of the bias is suppressed by `sqrt(d)`. With
`d = 128` and `kappa ~ 1` that is about 0.09 — the bias is the same number for
every key to within ~9%, and per-key structure has nothing to do.

**(ii) A common per-key factor.** If instead `X_ic = g_i m_c (1 + eta_ic)` with
`g_i` a per-key scale (`E[g] = 1`) and `eta` idiosyncratic, then

    sigma_b / mu_b ~ sqrt( CV(g)^2 + kappa_eta^2 / d_eff )                    (5)

and the first term does **not** shrink with `d`. A block whose keys are large
in every channel has a large step in every channel, which is exactly a common
mode. So the concentration claim is not automatic: it needs `CV(g)` to be
small, and `CV(g)` is an empirical property of the cache, not of the algebra.

**Proposition (conditional).** If the per-key variation of `Delta^2` is
predominantly channel-idiosyncratic — `CV(g) <~ kappa_eta / sqrt(d_eff)` — then
`sigma_b / mu_b = O(1/sqrt(d))` and the per-key Jensen bias is, to first order
in `sigma_b`, a single constant. The condition is measured in §4.

## 3. What actually reaches the softmax

Concentration of `b_i` is necessary but not sufficient: what matters is the
effect on the attention distribution. Softmax is invariant to a constant shift
within the cached block, so the scalar correction already reproduces the whole
mean effect `mu_b` — the *only* thing the structured bias can do that a scalar
cannot is redistribute mass *within* the cached region. To first order,

    Delta p_i ~= - p_i ( b_i - bar{b}_p ),    bar{b}_p = sum_j p_j b_j        (6)

with `p` restricted and renormalized to the cached keys. The total variation
between the two cached distributions is therefore about

    (1/2) sum_i p_i | b_i - bar{b}_p |  <=  (1/2) sigma_b^(p)                 (7)

with `sigma_b^(p) = sqrt( E_p[b^2] - E_p[b]^2 )` the attention-weighted spread.

So the structured-vs-scalar question reduces to one ratio:

    structure : scalar  ~  sigma_b^(p) : mu_b                                 (8)

Both sides of (8) are measured in-flight (`bias_scale.json`:
`effective_cached_bias_mean` = `mu_b`, `attn_weighted_bias_spread` =
`sigma_b^(p)`), so this is a prediction that can be checked rather than a
plausibility argument.

## 4. Measurement

`--tum_bias_dump` records the real `q` and the per-block `Delta^2` at chunks
1/6/20 for layers 0/15/29 of an A1 INT2 run with the lambda=1 correction;
`gateL3_concentration.py` evaluates (2)-(5) on them.

Measured on an A1 INT2 run with the lambda=1 correction, 3 prompts, chunks
1/6/20, layers 0/15/29, 12 heads each (36 rows).

| quantity | median | mean | p10 | p90 | max |
|---|---|---|---|---|---|
| `mu_b` | 0.604 | **130.4** | 0.035 | 4.26 | 6496 |
| `sigma_b` | 0.083 | 5.22 | 0.009 | 0.816 | 272.7 |
| `sigma_b/mu_b` | **0.150** | 0.173 | 0.091 | 0.310 | 0.420 |
| independent-channel prediction (4) | **0.151** | 0.165 | 0.076 | 0.282 | 0.539 |
| `d_eff` | 12.1 | 15.8 | 4.65 | 33.6 | 44.4 |
| `CV(g)` | 0.109 | 0.126 | 0.069 | 0.212 | 0.285 |
| `kappa` (median over channels) | 0.539 | 0.519 | 0.409 | 0.600 | 0.697 |

### Row-matched agreement (session-7 C1) — read this first

The earlier scatter plot **sorted the predicted and the measured column
separately and then paired them**. Marginals that overlap produce a good-looking
plot regardless of per-observation agreement, so it was not evidence; and for
the same reason **the closeness of the medians, 0.150 vs 0.151, is not evidence
either**. Recomputed with the row correspondence kept (n = 108 = 3 chunks x 3
layers x 12 heads):

| | value |
|---|---|
| Pearson r | **+0.783** |
| residual (measured - predicted), mean | +0.0082 |
| residual SD | 0.0585 |
| Bland-Altman limits | [-0.1065, +0.1229] |

**Supportable**: the independent-channel prediction correlates with the
measurement observation by observation (r = 0.78).
**Not supportable**: "the prediction is verified per observation". The
Bland-Altman limits (+-0.11) are the same order as the predicted value itself
(median 0.151). Claim agreement at the level of the distribution, no further.

### Within a head: the proposition holds

`sigma_b/mu_b` = **0.150** measured against **0.151** predicted by the
independent-channel formula (4). The prediction is `kappa/sqrt(d_eff)` =
0.539/sqrt(12.1) = 0.155, which is what comes out. `CV(g)` = 0.109 is not zero,
so a common per-key mode exists, but it is subdominant: the condition
`CV(g) <~ kappa/sqrt(d_eff)` is satisfied (0.109 vs 0.155).

Note `d_eff` is **12**, not 128. The sum is dominated by roughly a dozen
effective channels, because `q_c^2 Delta_c^2` is concentrated on outlier
channels — so the suppression is `sqrt(12) ~ 3.5`, not `sqrt(128) ~ 11`. A naive
"128 channels, so it must concentrate" argument overstates the effect by 3x and
happens to reach the right conclusion for the wrong reason.

**Within one head, the per-key Jensen bias is one number to within 15%.** That
is why erasing its per-key structure changed nothing (session 5 flatten: +10.49
vs taylor +10.04).

### Across heads: four orders of magnitude

The same table's `mu_b` mean (130) and median (0.60) differ by 200x, and the
reason is not noise:

| chunk | layer | min head | median head | max head | max/median |
|---|---|---|---|---|---|
| 1 | 0 | 0.009 | 0.335 | 1.3 | 4x |
| 1 | 15 | 0.291 | 0.603 | 2.6 | 4x |
| 1 | 29 | 0.080 | 0.617 | **6496** | **10527x** |
| 6 | 29 | 0.089 | 0.720 | **5004** | **6946x** |
| 20 | 0 | 0.015 | 0.411 | 1.5 | 4x |
| 20 | 15 | 0.420 | 0.849 | 1.7 | 2x |
| 20 | 29 | 0.151 | 1.040 | **2461** | **2366x** |

Layers 0 and 15 are well behaved (2-4x across heads). **One head in layer 29
carries a bias near 2500-6500 logits**, which suppresses its cached region
completely — for that head the lambda=1 correction is not a reweighting, it is
"ignore the cache, attend only to self". This is the massive-activation channel
phenomenon showing up in the correction.

So the structure of the correction is:
- **across keys within a head**: flat to 15% -> a constant suffices
- **across heads**: spans 4 orders of magnitude -> a *single global* constant
  cannot match it unless the extreme heads are saturated either way

### The in-flight numbers, and which one is the target

| statistic | value | what it is |
|---|---|---|
| uniform mean over cached keys | **70.08** | outlier-head-driven; describes almost no head |
| attention-weighted mean `E_p[b]` | *(re-measured in gate L1)* | the constant a scalar must match |
| attention-weighted spread `sigma_b^(p)` | **0.155** | the part no constant can reproduce |

The first instrumentation recorded only the uniform mean, and taking it at face
value would have centred the gate-L1 sweep at c = 70 — a constant that
annihilates the cache. Softmax only sees the bias where probability lands, so
the target is the attention-weighted mean; `tum_correction.py` now records all
three separately.

### What the flatten control actually controlled

Session 5 described `--tum_flatten_slots` as "structure removed", and read
flatten = taylor as "the Jensen structure contributes nothing". The measurement
above makes that statement more specific, and narrower. Flatten computes

```python
flat = bias.sum(dim=2, keepdim=True) / n_cached     # per (query, head)
```

so it is a **per-head, per-query** constant. It erases variation across cached
keys and **keeps** the variation across heads -- including the layer-29 head at
2500. So what session 5 established is:

> the **per-key** structure of the Jensen bias contributes nothing

and NOT

> the whole bias is equivalent to one number.

Those are different claims, and the second does not follow from the first when
the per-head scale spans four orders of magnitude. The handoff's framing ("the
whole effect is lowering the cache by a scalar relative to self", "the same
effect is available from a single constant") is right about the first and
assumes the second. Gate L1 separates them:

| control | per-key structure | per-head scale | result |
|---|---|---|---|
| taylor | kept | kept | +10.04 (session 5) |
| flatten | **erased** | kept | +10.49 (session 5) -- tied |
| scalar (gate L1) | erased | **erased** | measuring |

If scalar ties as well, one constant is genuinely enough. If it falls short, the
deployable form is one constant **per head**, which flatten already shows is
sufficient -- still far cheaper than the structured bias, but not a single
number.

### Prediction for gate L1

A single global `c` reproduces the correction **iff** the layer-29 outlier head
does not matter for output quality. If it does, the sweep will show a scalar
plateauing below the structured bias's +10.04, and the fix is a per-head
constant rather than a global one.

## 5. Control: does the estimator recover a known answer?

Before trusting the measurement, the same script was run on a synthetic dump
with `Delta^2` drawn independently per channel — the regime where (4) is exact
by construction. Measured `sigma_b/mu_b` = 0.0730, independent-channel
prediction = 0.0730, `CV(g)` = 0.043, `d_eff` = 44. The estimator reproduces the
regime it is supposed to detect, and `d_eff` lands well below `d = 128` because
the chi-square weights `q_c^2` are themselves uneven — a reminder that the
suppression is `1/sqrt(d_eff)`, not `1/sqrt(d)`.

## 6. Status

Pending the driver-A stage-3 run.
