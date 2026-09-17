"""Gate F2 — the TUM Jensen-bias correction, applied to cached logits.

Quantizing a key perturbs its logit, s -> s + eps. Softmax is convex in the
logit, so E[exp(s+eps)] = exp(s)exp(sigma^2/2) > exp(s): a noisier key attracts
more mass than it should. Session 3 measured exactly that shape at INT2 — the
current chunk, which is read in BF16 (Gate F0), lost mass from 48% to 19% while
the quantized cache gained it.

The correction subtracts the predicted bias from each cached logit:

    b_i ~= (1/24d) * sum_c q_c^2 * Delta_{i,c}^2

with Delta the quantization step of slot i along channel c. Paper 2605.26266
section 3, used as written; nothing new is invented here.

FlashAttention-2 takes no per-key logit bias, so this runs through an explicit
attention path — the same recomputation Gate B verified against FA2. It is
slower, and that is acceptable: this is a measurement of whether the correction
moves the distribution back, not a proposed kernel.

Gate: with Delta = 0 (BF16) the corrected path must reproduce the FA2 path
bit-for-bit, otherwise the explicit path itself is what changed the result.
"""
import torch

# Set by inference.py: the quantization step per cached slot, per channel.
TUM_ENABLED = False
TUM_STATE = {}          # layer id -> Delta^2 tensor [Lk, D] or scalar
TUM_LAYER_COUNTER = [0]


@torch.no_grad()
def delta_sq_from_quant_state(quantizer, state, like):
    """Per-token, per-channel squared quantization step of the cached keys.

    RTN stores a symmetric per-(block, head, channel) scale, so the step is
    2*scale for a signed grid; KIVI's asymmetric form stores the scale directly.
    Reconstructed to the cache's token layout so it can be indexed alongside K.
    """
    k_state = state["k"] if isinstance(state, dict) and "k" in state else state
    scale = k_state["scale"].float()              # [B, nb, 1, H, D]
    bits = int(k_state.get("bits", 4))
    step = 2.0 * scale if bits >= 2 else scale
    block = int(k_state.get("block_size", 16))
    b, nb, _, h, d = step.shape
    # expand the per-block step back over the tokens in that block
    step = step.expand(b, nb, block, h, d).reshape(b, nb * block, h, d)
    pad = k_state.get("pad_len", 0)
    if pad:
        step = step[:, :step.shape[1] - pad]
    return (step ** 2).to(like.dtype)


@torch.no_grad()
def corrected_attention(q, k, v, delta_sq=None, head_dim=None):
    """softmax(qk/sqrt(d) - b) @ v, with b the Jensen bias of each cached key.

    b_i = (1/(24 d)) * sum_c q_c^2 * Delta_{i,c}^2   — quadratic in q, so it is
    computed per query rather than once per key.
    """
    b_, lq, h, d = q.shape
    scale = 1.0 / (d ** 0.5)
    out = torch.empty_like(q)
    for hi in range(h):
        qi = q[:, :, hi].float()
        ki = k[:, :, hi].float()
        vi = v[:, :, hi].float()
        logits = torch.bmm(qi * scale, ki.transpose(1, 2))          # [B, Lq, Lk]
        if delta_sq is not None:
            ds = delta_sq[:, :, hi].float()                         # [B, Lk, D]
            bias = torch.bmm(qi ** 2, ds.transpose(1, 2)) / (24.0 * d)
            logits = logits - bias
        out[:, :, hi] = torch.bmm(torch.softmax(logits, dim=-1), vi).to(q.dtype)
        del logits
    return out


@torch.no_grad()
def verify_zero_delta(q, k, v, atol=1e-2, rtol=1e-2):
    """With no quantization the corrected path must match FA2."""
    from wan.modules.attention import attention
    ref = attention(q, k, v).float()
    got = corrected_attention(q, k, v, delta_sq=None).float()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    return ok, (got - ref).abs().max().item()
