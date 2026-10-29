# Independent KIVI grouping candidate (CPU preparation only)

> 현재 상태: legacy 교차 개발 8/8이 고정 기준 **NO-GO**로 종료됐다.
> 이 후보는 CPU 준비물로만 보존하며, 후속 GPU 개발과 새 보류 확인에는 사용하지 않았다.

This candidate is **not registered in the running numerical study**. That study
and its development GO/NO-GO remain frozen. Preparing a different quantizer does
not rescue a failed screen, authorize GPU work, or provide new quality evidence.
The parent study owner must decide whether any later experiment is justified
after the fixed development decision. No GPU experiment was run for this work.

## Source and scope

Official upstream HEAD was resolved with `git ls-remote` on 2026-10-29 to
`876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`. The following official files were
read at that immutable commit:

- [quant/new_pack.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/quant/new_pack.py),
  `quant_and_pack_kcache`, `quant_and_pack_vcache`, their unpack helpers, and
  `triton_quantize_and_pack_along_last_dim`.
- [models/llama_kivi.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/models/llama_kivi.py),
  key transposition before token grouping, value grouping along the last channel
  dimension, and the residual-cache branches.

`tools/kivi_grouped_candidate.py` is a small independent implementation of
standard affine min/max quantization. No upstream source, kernels, packing
routines, or cache lifecycle code was copied. These links attribute the grouping
design; this is **not a full KIVI reproduction**, official numerical equivalence
claim, or port of upstream software.

## Interface and grouping

`KiviGroupedCandidate(key_bits=4, value_bits=4, group=16, arithmetic="fp32")`
exposes `encode` and `encode_pair(k, v)` for finite, nonempty floating tensors
`[B,L,H,D]`. Only CPU execution has been performed; the implementation uses
ordinary PyTorch tensor operations without a device-specific kernel.
Pairs must have equal shapes. Supported bit widths are 2, 4, and 8. Every restored
tensor is BF16, even if the input uses another floating dtype.

| Tensor | Codes layout | Extrema reduced over | Scale and offset layout |
| --- | --- | --- | --- |
| Key | `[B,ceil(L/G),G,H,D]` | token position within each G-token group | `[B,ceil(L/G),1,H,D]` |
| Value | `[B,L,H,D/G,G]` | G adjacent channels within one head and one token | `[B,L,H,D/G,1]` |

The axes match official key per-channel token groups and value per-token,
per-head channel groups after adapting its `[B,H,L,D]` interface. The existing
legacy `kvq.quant.Quantizer(method="kivi")` reduces value extrema over **H and D
together**; this candidate deliberately has no such cross-head reduction.

`GroupedQuantized` subclasses the existing `Quantized` dataclass and overrides
restoration for these layouts. `encode_pair` returns two such states. Existing
`memory_report`, `storage_bytes`, and `Quantized.bytes()` can account for them
without edits. No store is created by the candidate, and no experimental runner
imports or selects it. A future experiment would choose its storage policy
independently; original KIVI residual windows, immutable quantized history,
attention kernels, packing, and eviction behavior are not implemented here.

## Arithmetic and differences from the official implementation

For each group, `Q = 2**bits - 1`, `m = min(x)`, and
`s = max((max(x)-m)/Q, 2**-24)`. Codes are
`round(clamp((x-m)/s, 0, Q))`, using PyTorch nearest-even rounding. The standard
affine equation and clamp-before-round order agree with the cited source.
The following choices differ or restrict the scope:

1. In the default `arithmetic="fp32"` mode, all extrema, subtraction, division, and code selection run in FP32 after
   converting input to FP32. Official helpers keep input arithmetic dtype and
   the model uses Triton/CUDA kernels. Float64 inputs also lose precision on
   entry here. This candidate makes no bitwise-equivalence claim.
2. Codes are **uint8 containers**, with one actual byte per code at every bit
   width. Official code converts to int32 and actually packs codes into words.
   There is no packing or pack-word alignment requirement here.
3. Scales and offsets are explicitly rounded to FP16 *after* code selection.
   Codes therefore use unrounded FP32 metadata, while restoration uses persisted
   FP16 metadata. Official scale/offset dtype follows its input/kernel path.
4. Default-mode restoration converts persisted codes and metadata to FP32, multiplies and
   adds in FP32, then casts once to BF16. Official unpack helpers convert codes
   to FP16 before affine reconstruction, and the model can consume packed codes
   directly in attention kernels. Rounding and accumulation can differ.
5. Official cited quantization functions divide by the raw range without an
   epsilon or explicit constant-group branch. In default mode `2**-24` is the smallest
   positive FP16 subnormal, chosen to prevent stored scales rounding to zero.
   A constant group gets zero codes and restores its FP16-rounded offset cast
   to BF16; zero groups restore zero. Tiny nonconstant ranges use the same
   floor, so their codes and errors can differ from upstream. This floor is
   also different from the existing legacy implementation's `1e-8` floor.
6. Official key helpers require token-length divisibility and the model keeps
   an unquantized residual tail. Here a non-divisible key length is padded
   **with zeros before extrema** to the next full token group. The zeros
   participate in that last group's min/max, and their codes and metadata are
   physically stored. They are removed only from the restored output. For an
   all-positive or all-negative tail this changes its extrema and quantization
   error. There is no hidden BF16 residual. Values keep exactly L tokens and
   have no token padding; each token is independently quantized.
7. As in official value grouping, D must be divisible by G. A partial D group
   raises `ValueError`; channels are never padded or allowed to cross heads.
8. This implementation rejects empty/nonfinite inputs and FP16
   metadata overflow with `ValueError`. Valid inputs must have representable
   finite FP16 offsets and scales after conversion. It does not saturate
   overflowing metadata or claim support for the entire BF16 dynamic range.

An explicit `arithmetic="input"` mode is provided for the axis comparison with
the existing legacy quantizer. It requires BF16 inputs, keeps extrema and all
quantization arithmetic in BF16, uses the legacy `1e-8` floor, persists metadata
in FP16, and performs restoration multiply/add in BF16 as the legacy code does.
The `1e-8` scale can underflow to zero in FP16; this behavior is deliberately
preserved in this mode, including for constant groups. Zero-code constant groups
still restore finite offsets. This mode is not official arithmetic equivalence:
the official source has no epsilon, and its kernel/dtype path is not simulated.
Both modes retain the grouping, padding, BF16 output, and byte-code conventions
above. Default FP32 results cannot attribute changes solely to the V grouping
axis; the input mode supplies the arithmetic control for that purpose.

The returned state stores no original input or working buffers. Its `dtype`
field is BF16 and its codes, scales, and offsets are the only resident tensors.
FP32 scratch and restored BF16 tensors exist during calls but are not included
in persistent-state accounting or presented as a peak-memory measurement.

## CPU validation and readiness

Run using the existing isolated interpreter, without installing packages:

```sh
/home/acorn0415/sfq_data/py tools/test_kivi_grouped_candidate.py -v
```

On 2026-10-29: **7 tests passed** (CPU). They compare codes, metadata, and BF16
reconstruction with an independent scalar/group reference for both axes at 2
and 4 bits; test head isolation, constants/zeros, partial key groups, value
divisibility, finite restoration, and rejected inputs. They check actual tensor
storage, scale/offset bytes, key padding, hypothetical packed estimates, and
storage deduplication through the existing accounting interfaces.

For input mode, exact code/scale/offset/restoration agreement with legacy is
tested only on BF16 `[2,5,2,8]` bounded integer-valued inputs at bits 2/4 and G=4:
the full key tensor (including its padded tail), and value subproblems restricted
to one head and one 4-channel group at a time. Legacy value token padding is
trimmed before comparing the five valid tokens. This tests the legacy arithmetic
control without falsely claiming whole-tensor value equivalence across heads or
channel groups. It does not establish equivalence for arbitrary floating inputs,
other dtypes, upstream Triton kernels, or GPU execution. Zero-group FP16 scale
underflow and finite BF16 restoration are also tested in input mode.

For the tested `[2,5,2,8]`, G=4 pair: 320 valid code bytes + 96 key-padding
code bytes + 208 scale bytes + 208 offset bytes = **832 resident bytes**, both
at 2 and 4 bits. Hypothetical packed totals are 520 and 624 bytes respectively;
those totals do not describe actual resident storage.

Ready for CPU review and possible later experiment design only. No throughput,
GPU compatibility, attention correctness, downstream quality, or decision-gate
performance has been established. The default bit width is 4 and 2/4 are tested;
8-bit is accepted for interface parity with official supported widths but is
not part of this CPU validation claim.
