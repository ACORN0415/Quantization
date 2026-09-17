# `quantize_asym` clamps the zero point into the quantized range, which destroys any group that does not straddle zero

**Affects:** `kv_quant/utils.py::quantize_asym`, and therefore every method that calls it — `KIVIQuantizer` (both keys and values) and `QAQQuantizer`. `RTNQuantizer` is unaffected because it uses `quantize_sym`.

## The problem

```python
# kv_quant/utils.py
def quantize_asym(x: torch.Tensor, bits: int, reduce_dims: Tuple[int, ...]):
    qmin, qmax = 0, (1 << bits) - 1
    x_min = x.amin(dim=reduce_dims, keepdim=True)
    x_max = x.amax(dim=reduce_dims, keepdim=True)
    scale = ((x_max - x_min) / max(qmax - qmin, 1)).clamp_min(EPS)
    zp = torch.round(qmin - x_min / scale).clamp(qmin, qmax)   # <-- here
    q = torch.round(x / scale + zp).clamp(qmin, qmax).to(torch.int8)
    return q, scale.to(torch.float16), zp.to(torch.float16)
```

The zero point is an **offset**, not a quantization code. Clamping it to `[qmin, qmax]` is only harmless when the group straddles zero:

- `x_min < 0 < x_max` → `zp = -x_min·qmax/(x_max - x_min) ∈ [0, qmax]`, the clamp is a no-op.
- all values positive → the true `zp` is **negative**, gets clamped to `0`, and every code then evaluates to `round(x/scale)`, which is `≫ qmax` and saturates at `qmax`. The whole group collapses onto one code.
- all values negative → symmetric failure, everything collapses onto `0`.

Post-RoPE keys hit this constantly: within a 16-token block a given `(head, channel)` often keeps one sign.

## Reproduction

`repro_zp_clamp.py`, run from the repo root:

```python
import torch
from kv_quant.utils import quantize_asym, dequantize_asym, quantize_sym, dequantize_sym

torch.manual_seed(0)
def rel(a, b): return ((a - b).norm() / b.norm()).item()

# Block layout produced by _reshape_blocks: [B, n_blocks, block_size, heads, dim]
x = torch.randn(1, 8, 16, 12, 128) * 0.1 + 5.0     # a group that never crosses zero

q, s, z = quantize_asym(x, bits=4, reduce_dims=(2,))
print(f"asymmetric (KIVI/QAQ path): rel_err = {rel(dequantize_asym(q, s, z, dtype=torch.float32), x):.4f}")

q2, s2 = quantize_sym(x, bits=4, reduce_dims=(2,))
print(f"symmetric  (RTN path)     : rel_err = {rel(dequantize_sym(q2, s2, dtype=torch.float32), x):.4f}")

qmin, qmax = 0, 15
x_min, x_max = x.amin(dim=(2,), keepdim=True), x.amax(dim=(2,), keepdim=True)
scale = ((x_max - x_min) / (qmax - qmin)).clamp_min(1e-8)
zp_true = torch.round(qmin - x_min / scale)
zp_used = zp_true.clamp(qmin, qmax)
print(f"\ntrue zero point range   : [{zp_true.min():.0f}, {zp_true.max():.0f}]")
print(f"after .clamp(0, 15)     : [{zp_used.min():.0f}, {zp_used.max():.0f}]")
print(f"groups altered by clamp : {(zp_true != zp_used).float().mean() * 100:.1f}%")
```

Output on `b4c0936`:

```
asymmetric (KIVI/QAQ path): rel_err = 0.9296
symmetric  (RTN path)     : rel_err = 0.0399

true zero point range   : [-516, -97]
after .clamp(0, 15)     : [0, 0]
groups altered by clamp : 100.0%
```

4-bit asymmetric quantization is **23× worse than 4-bit symmetric** on this input, and 100% of groups are affected. For a group centred on zero the two agree as expected (asym 0.069 vs sym 0.085), so the failure is specific to one-signed groups.

## A second bug, which the first one was hiding

Simply dropping the clamp is **not** enough. `EPS = 1e-8` underflows to zero in
the `float16` the scale is stored in (smallest fp16 subnormal is ~6e-8). So for a
group whose spread falls below bf16 resolution:

- `x_max - x_min == 0` → `scale = EPS = 1e-8` → `scale.to(torch.float16) == 0`
- `zp = -x_min / 1e-8` → on the order of `1e8` → `zp.to(torch.float16) == inf`
- `dequantize_asym` → `(q - inf) * 0` → **NaN**

With the original clamp in place this never surfaced as NaN: `zp` was pinned to
`[0, 15]`, so the group silently dequantized to zero instead. Removing the clamp
turns a silent wrong answer into a loud one. We hit this on real 63-frame
rollouts — whole runs came back NaN.

## Fix

Store the group minimum instead of a zero-point code. Mathematically the same
asymmetric quantization, but numerically safe, and the same storage cost (one
fp16 per group either way):

```python
 def quantize_asym(x, bits, reduce_dims):
     qmin, qmax = 0, (1 << bits) - 1
     x_min = x.amin(dim=reduce_dims, keepdim=True)
     x_max = x.amax(dim=reduce_dims, keepdim=True)
     scale = ((x_max - x_min) / max(qmax - qmin, 1)).clamp_min(EPS)
-    zp = torch.round(qmin - x_min / scale).clamp(qmin, qmax)
-    q = torch.round(x / scale + zp).clamp(qmin, qmax).to(torch.int8)
-    return q, scale.to(torch.float16), zp.to(torch.float16)
+    q = torch.round((x - x_min) / scale).clamp(qmin, qmax).to(torch.int8)
+    return q, scale.to(torch.float16), x_min.to(torch.float16)

 def dequantize_asym(q, scale, zp, dtype=torch.float32):
-    return (q.to(dtype) - zp.to(dtype)) * scale.to(dtype)
+    return q.to(dtype) * scale.to(dtype) + zp.to(dtype)
```

`x_min` is a data value, so fp16 always represents it, and a constant group
round-trips exactly (`q = 0` → `0 * scale + x_min`).

After the fix, on the same block shape:

| group | asymmetric | symmetric (RTN) |
|---|---:|---:|
| zero-centred | **0.0650** | 0.0852 |
| all-positive (`+5.0`) | **0.0013** | 0.0400 |
| all-negative (`-5.0`) | **0.0013** | 0.0400 |
| spread below bf16 resolution | **0.0000**, no NaN | — |

Asymmetric now beats symmetric everywhere, which is what you would expect at
equal bit-width.

## Impact on measured results

We hit this while reproducing your KV-cache quantization study on Self-Forcing. In-situ cache error (relative L2 between what is written to the KV cache and what is read back, averaged over 30 layers and 3 prompts, 63-frame rollouts):

| method | key error | value error |
|---|---:|---:|
| `RTN_INT4` (`quantize_sym`) | 0.0278 | 0.0371 |
| `RTN_INT2` (`quantize_sym`) | 0.1298 | 0.1728 |
| `KIVI_INT4` (`quantize_asym`) | **0.1841** | 0.0569 |

`KIVI_INT4` keys are 6.6× worse than `RTN_INT4` keys at the same nominal bit-width — and worse than `RTN_INT2`. That inversion is what led us to the clamp. Any published comparison that ranks KIVI or QAQ against RTN using this code is likely measuring the bug rather than the method.

## Unrelated, but in the same area

`create_quantizer` in `kv_quant/factory.py` passes `key_bits`, `value_bits` and `name` to `RTNQuantizer` and `KIVIQuantizer` without binding them — they arrive via `**kwargs` but are never popped, so both paths raise `NameError: name 'key_bits' is not defined`:

```python
if method == "RTN":
    from .rtn import RTNQuantizer
    return RTNQuantizer(bits=bits, block_size=block_size, key_bits=key_bits, value_bits=value_bits, name=name)
```

```python
>>> create_quantizer('RTN', bits=4, block_size=16, name='RTN_INT4')
NameError: name 'key_bits' is not defined
```

Fix is to pop them from `kwargs` before the branches:

```python
key_bits = kwargs.pop("key_bits", None)
value_bits = kwargs.pop("value_bits", None)
name = kwargs.pop("name", None)
```

Happy to open PRs for either of these if useful.

---
Environment: `b4c09363735c89947222781db899862bc0dc55c5`, torch 2.6.0+cu124, Python 3.10.12, RTX 4090.
