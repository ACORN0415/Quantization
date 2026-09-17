# Gate F0 — is the self bucket read quantized or in BF16?

**Answer: BF16.** The current chunk's K/V are always read at full precision
during its own generation, and become quantized only from the next chunk on.

Order in the patched attention path (`wan/modules/causal_model.py`):

1. `cache_k, cache_v = quantizer.dequantize_kv(kv_cache["quant_state"], ...)`
   — the past cache comes back dequantized
2. `cache_k[:, local_start:local_end] = roped_key` — the current chunk is
   written in as raw BF16, with no quantization applied
3. `x = attention(roped_query, cache_k[...], cache_v[...])` — so what attention
   reads for the current chunk is BF16
4. **after** attention: `kv_cache["quant_state"] = quantizer.quantize_kv(cache_k, cache_v)`
   — only now does the current chunk get quantized, for the next chunk to read

Each chunk calls the generator five times (four denoising steps plus the
clean-context refresh). Step 4 of call *n* quantizes the current chunk, but step
2 of call *n+1* overwrites that slot with fresh BF16 before attention runs. So
self is BF16 at read time on every call, without exception.

## Consequence for the Jensen-bias framing

This matches the TUM setting exactly: a quantized cache competing against an
unquantized present. The claim to make is the specific one —

> the quantized cache steals attention mass from the unquantized current chunk

rather than the weaker general form ("slots with larger noise variance steal
from smaller ones"), which would have been the right phrasing had self also
been quantized.

It also sharpens what session 3 measured. The INT2 inversion — self collapsing
from 48% to 19% while the oldest cached chunk rises from 7% to 30% — is mass
moving from a BF16 bucket to quantized ones, which is the direction
E[exp(s+eps)] = exp(s)exp(sigma^2/2) predicts.
