# The two codebases do not simulate KV quantization the same way

Found while preparing gate M1. This is a code-level fact, verified by reading
both implementations, and it confounds the session-3 cross-model comparison.

## Self-Forcing (UCSD patch, `wan/modules/causal_model.py`)

There is **no BF16 master**. After each layer call the BF16 buffers are dropped:

```python
kv_cache["quant_state"] = quantizer.quantize_kv(cache_k, cache_v, ...)
kv_cache["k"] = cache_k.new_empty(0)
kv_cache["v"] = cache_v.new_empty(0)
```

and the next call reconstructs the cache by dequantizing that state. So the
cycle is dequantize -> shift -> write -> re-quantize, and because the shift is
8 tokens modulo the block grid (gate K1) the re-quantization is not idempotent:
error **compounds**, once per eviction.

## LongLive (`~/gpu/LongLive/wan/modules/causal_model.py`)

`cache["k"]` / `cache["v"]` stay **BF16 for the whole rollout**. Reads go
through `_kv_read`, which dequantizes into a throwaway `temp_k`; the write-back
rolls and writes the BF16 master itself, and only then

```python
cache["quant_state"] = KV_QUANTIZER.quantize_kv(cache["k"], cache["v"], ...)
```

Every quantization therefore starts from clean BF16. The grid still shifts, so
the realized error changes from step to step, but it **never compounds**.

## The signature in the logged cache error

`kv_err.json` records `k_rel`, the error of the dequantized cache against
whatever the writer held at that moment. The two implementations produce
qualitatively different curves, exactly as the code predicts:

| run | c0 | c1 | c4 | c8 | c12 | c20 |
|---|---|---|---|---|---|---|
| LongLive INT2 (BF16 master) | 0.4419 | 0.4408 | 0.4403 | 0.4403 | 0.4420 | **0.4418** |
| LongLive INT4 | 0.0638 | 0.0641 | 0.0642 | 0.0643 | 0.0643 | 0.0644 |
| Self-Forcing A1 INT2 (no master) | 0.3842 | 0.3142 | 0.1745 | 0.1402 | 0.1330 | **0.1309** |
| Self-Forcing A1 INT4 | 0.0565 | 0.0491 | 0.0293 | 0.0282 | 0.0281 | 0.0280 |

LongLive is flat at the single-quantization error, because it always measures a
fresh quantization of clean BF16. Self-Forcing decays, because it measures the
*incremental* cost of quantizing content that is already quantized — the
denominator is the already-degraded cache, not clean content. **These two
numbers are not the same quantity**, so the table shows that the two setups
differ in kind; it does not by itself measure the size of the accumulation.
That measurement is gate K1 (13-29% excess key error against a clean
reference), which only the Self-Forcing path exhibits.

## Consequence for the cross-model claim

Session 3 concluded that LongLive is more robust at INT2 (loss 21.68 vs 29.23
in mean MUSIQ). Part of that gap may be the storage simulation rather than
retraining, the sink, or the shorter window. The comparison as it stands is
**confounded**.

## What makes it fair

`--kv_grid_align segment` removes the compounding from the Self-Forcing path
(gate K1: bit-identical to quantizing the surviving content once). That puts it
in the same storage regime as LongLive — both are "single quantization, no
compounding" — differing only in grid partition, whose size is bounded by the
21-frame no-eviction control in gate K3. **So the apples-to-apples cross-model
row is aligned Self-Forcing vs LongLive, not rolling Self-Forcing vs LongLive.**
Gate M1 should be read against the aligned number from K3.
