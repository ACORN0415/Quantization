# Verification results

| Gate | State | Evidence |
|---|---|---|
| G1 | PASS | CPU and CUDA: RTN/KIVI × bits {2,3,4,8} × lengths {32,4680,32760} × random/constant/positive/zero; exact restored tensors. Independent key/value widths also compared. |
| G1b | PASS | Synthetic eviction, sink, reset, immutable segments, live-prefix residual; small actual attention matches legacy and pristine paths. |
| G2 | PENDING | Downloads must complete before full-model validation. |
| G3 | PENDING | Tiny-model pipeline RNG regression passed; full model still required. |
| G4 | PENDING | Small attention compatibility passed; full latents still required. |
| G5 | PENDING | Historical aggregation comparison prepared; not yet measured. |

All 11 unit/integration tests passed in 40.084 s. CUDA quantizer suite: 3 tests,
0.619 s, PASS. These do not replace full-model gates.

Logs: `/home/acorn0415/sfq_data/{all_tests,unit_cuda,integration}.log`.
Full-model runner: `scripts/verify.py`; reports and exact commands persist under
`/home/acorn0415/sfq_data/verify`. Criteria are in `docs/HANDOFF_codex.md`.
