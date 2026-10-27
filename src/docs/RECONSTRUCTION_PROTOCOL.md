# Reconstruction protocol

Scope: Self-Forcing / Wan2.1-T2V-1.3B inference, separate RTN/KIVI-style
quantizers, rolling and chunk-local cache storage, deterministic chunk noise,
and numerical validation. Upstream sources, pinned commits and licenses are
listed in `vendor/UPSTREAM.md`; no unlicensed reference source is redistributed.

Quantization operates on BF16 [B,L,H,D] tensors, grouping sequence tokens by 16
with zero padding. RTN uses symmetric rounding; KIVI-style keys use per-channel
sequence groups, values use per-token extrema across head/channel dimensions.
Scales and asymmetric minima are stored in fp16. Byte codes are simulated,
not packed. The current default unsigned KIVI representation fixes 8-bit code
wraparound; explicit legacy mode exists only for historical comparisons.

Rolling restores, rolls/inserts, attends, and requantizes the entire buffer per
call. Chunk storage quantizes completed segments once and keeps current/recent
residual chunks in BF16. Sink and window lengths must align with chunk sizes.
Original BF16 execution remains the path when no store is configured.

Chunk initial noise seed: `seed * 1000003 + chunk`.
Re-noising seed: `seed * 1000003 + 7919 + chunk`.
Release mode retains the upstream global noise stream and DynamicSwap behavior.

## Fixed validation criteria

| Gate | Criterion |
|---|---|
| G1 | CPU/CUDA reconstructed tensors bit-exact against separately supplied patched reference, bits 2/3/4/8, lengths 32/4680/32760 and edge cases. Legacy signed-int8 mode explicitly enabled only for compatibility. |
| G1b | Synthetic eviction, sink, prompt reset, immutable completed segments and live-prefix residual assertions. |
| G2 | Original release, seed 0, 21 frames, three prompts: final latents bit-exact. |
| G3 | Chunk RNG, 63 frames, two prompts: with/without 7777 global CUDA draws per chunk, bit-exact. |
| G4 | Legacy reconstruction vs rolling BF16/RTN4/RTN2/KIVI4 residual 0, 63 frames, two prompts each: every chunk bit-exact. |
| ANCHOR_IDENTITY | Original pilot code with an identity quantizer through `anchor_store` and `ChunkStore`: 42 TF/reference chunks bit-exact; segment encoding must be exercised. |
| G5 | Ten historical prompts, KIVI4 residual 0: all 210 rows within relative 1e-6, including historical record counts and active-token counts. |
| IDENTITY | Both storage policies with a residual large enough to bypass compression, two 63-frame prompts: all 42 chunks per policy bit-exact with previously verified BF16 captures. Unit counterpart includes both policies, sink 0/1, ten frames and five calls/frame. |

The old KV error aggregation is retained only for reproducing historical rows.
Schema-2 k_rel/v_rel compare only historical tokens immediately before attention; their
originals belong to each trajectory and do not measure an independent BF16
counterfactual. Peak CUDA allocation includes temporary reconstruction and
logging buffers. Neither it nor byte-container size establishes packed-kernel
memory savings. See `VERIFY.md` for measured evidence and scope restrictions.

`verify.py` writes live progress outside source documentation. Raw command/log
paths remain local; published evidence substitutes portable path placeholders
while preserving numerical values and raw-file digests.
