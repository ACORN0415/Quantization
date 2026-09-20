# Gate K1 — the lineage effect at bit level

A chunk is 4680 tokens = 292.5 blocks of 16, so every eviction shifts surviving content **8 tokens** relative to a block grid anchored at token 0. Cache = 7 chunk slots, sink = 0 tokens.

Storage-only measurement: clean BF16 chunk contents captured from a BF16 rollout are fed into each storage regime, so generation feedback is excluded and what is left is the storage error alone. `ref` = the same surviving content quantized once on the regime's own grid.

## 1. Is RTN idempotent?

| bits | re-quantize on the SAME grid | re-quantize 8 tokens off |
|---|---|---|
| INT4 | **0.000e+00** | 5.472e-02 |
| INT2 | **0.000e+00** | 2.008e-01 |

Exactly zero on the same grid — the audit's idempotence claim is bit-exact, not approximate. Shifted by 8 tokens it is not idempotent at all: the injected error is of the same order as the quantization error itself.

## 2. Does the error accumulate over evictions?

Relative key error of the cache against the single-quantization reference (`rel_vs_ref`), and against the clean BF16 content (`rel_vs_clean`), at the last chunk. `ref_vs_clean` is the floor set by quantizing once.

| bits | layer | regime | c20 rel_vs_ref | c20 rel_vs_clean | floor | excess |
|---|---|---|---|---|---|---|
| INT4 | 0 | lineage | 5.162e-02 | 0.0653 | 0.0528 | +23.8% |
| INT4 | 0 | aligned | 0.000e+00 | 0.0527 | 0.0527 | +0.0% |
| INT4 | 15 | lineage | 7.381e-02 | 0.0918 | 0.0740 | +24.0% |
| INT4 | 15 | aligned | 0.000e+00 | 0.0740 | 0.0740 | +0.0% |
| INT4 | 29 | lineage | 3.972e-02 | 0.0620 | 0.0543 | +14.2% |
| INT4 | 29 | aligned | 0.000e+00 | 0.0543 | 0.0543 | +0.0% |
| INT2 | 0 | lineage | 1.715e-01 | 0.3684 | 0.2952 | +24.8% |
| INT2 | 0 | aligned | 0.000e+00 | 0.2950 | 0.2950 | +0.0% |
| INT2 | 15 | lineage | 3.796e-01 | 0.6927 | 0.5359 | +29.3% |
| INT2 | 15 | aligned | 0.000e+00 | 0.5358 | 0.5358 | +0.0% |
| INT2 | 29 | lineage | 1.375e-01 | 0.3744 | 0.3311 | +13.1% |
| INT2 | 29 | aligned | 0.000e+00 | 0.3311 | 0.3311 | +0.0% |

`aligned` is bit-identical to the reference at every chunk (`rel_vs_ref` = 0), i.e. **zero accumulation after 14 evictions**. `lineage` carries a real excess over the quantization floor.

## 3. Growth with the number of evictions

`rel_vs_ref` per chunk, lineage regime (the cache is full after chunk 6, so eviction n happens at chunk 6+n):

- INT4 L0: c6=1.514e-03  c8=4.034e-02  c10=4.703e-02  c12=5.149e-02  c14=5.136e-02  c16=5.146e-02  c18=5.157e-02  c20=5.162e-02
- INT4 L15: c6=1.786e-03  c8=6.014e-02  c10=6.755e-02  c12=7.379e-02  c14=7.377e-02  c16=7.378e-02  c18=7.384e-02  c20=7.381e-02
- INT4 L29: c6=1.496e-03  c8=2.974e-02  c10=3.566e-02  c12=4.026e-02  c14=4.103e-02  c16=4.064e-02  c18=4.061e-02  c20=3.972e-02
- INT2 L0: c6=6.501e-03  c8=1.263e-01  c10=1.538e-01  c12=1.689e-01  c14=1.699e-01  c16=1.711e-01  c18=1.715e-01  c20=1.715e-01
- INT2 L15: c6=1.022e-02  c8=2.747e-01  c10=3.413e-01  c12=3.815e-01  c14=3.817e-01  c16=3.813e-01  c18=3.809e-01  c20=3.796e-01
- INT2 L29: c6=7.404e-03  c8=1.013e-01  c10=1.252e-01  c12=1.382e-01  c14=1.405e-01  c16=1.400e-01  c18=1.401e-01  c20=1.375e-01

The error **saturates** rather than growing without bound: at INT2 L15 it is 0.010 after the first eviction, 0.38 by chunk 12, and flat thereafter. That is a consequence of the window, not of the mechanism -- a slot survives at most 6 evictions before it is dropped, so each slot can only be re-quantized 6 times. A longer window would accumulate more.

## 4. Where the difference sits

Per-slot key error against the clean content at chunk 20 (slot 0 = oldest surviving chunk, last slot = the chunk just written):

- INT4 L0 lineage : 0.069 0.069 0.067 0.068 0.066 0.064 0.053
- INT4 L0 aligned : 0.053 0.053 0.053 0.053 0.053 0.053 0.053
- INT4 L15 lineage : 0.095 0.096 0.095 0.095 0.093 0.093 0.074
- INT4 L15 aligned : 0.074 0.074 0.074 0.074 0.074 0.074 0.074
- INT4 L29 lineage : 0.066 0.064 0.063 0.064 0.061 0.060 0.054
- INT4 L29 aligned : 0.054 0.055 0.054 0.054 0.054 0.054 0.054
- INT2 L0 lineage : 0.411 0.391 0.381 0.382 0.364 0.344 0.293
- INT2 L0 aligned : 0.296 0.291 0.292 0.298 0.296 0.298 0.293
- INT2 L15 lineage : 0.768 0.753 0.739 0.714 0.680 0.631 0.536
- INT2 L15 aligned : 0.536 0.536 0.537 0.536 0.534 0.536 0.536
- INT2 L29 lineage : 0.405 0.392 0.384 0.379 0.369 0.356 0.331
- INT2 L29 aligned : 0.331 0.331 0.331 0.330 0.331 0.332 0.331

The last slot is the chunk written this step: it is quantized once in both regimes, so it matches. Every older slot is where the regimes diverge, and the gap widens with slot age — which is what an error injected once per eviction looks like.

