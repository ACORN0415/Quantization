"""CPU-only matched-boundary diagnostic; real captures require an explicit registered run.

K payload: low nibbles for every head, extra high nibbles for one selected
head, FP16 scales, and one uint8 head index per chunk. Padding is retained.
All fixed/norm alternatives under one grouping have exactly equal payload bytes.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/experiments/kv_headrule_rollout")]
from kvq.quant import Quantizer
from headrule_quantizer import HeadRuleQuantizer, hadamard

torch.set_num_threads(4)
torch.set_grad_enabled(False)
FRAME, CHUNK, NH, DIM, GROUP = 1560, 4680, 12, 128, 16
PAST, FULL = 6 * CHUNK, 7 * CHUNK
PROMPTS, LAYERS = (0, 4), (8, 20, 28, 29)
DATA = Path("/home/acorn0415/sfq_data/kv_alllayer_capture")
START = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def guard():
    if time.monotonic() - START > 45 * 60:
        raise TimeoutError("45-minute CPU budget")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 8 * 1024**3:
        raise MemoryError("8-GiB max RSS budget")


def pack_nibbles(x):
    x = x.to(torch.uint8).reshape(-1)
    if x.numel() % 2:
        raise ValueError("nibble payload must have an even number of codes")
    return ((x[0::2] & 15) | ((x[1::2] & 15) << 4)).contiguous()


def unpack_nibbles(x, shape):
    y = torch.empty(x.numel() * 2, dtype=torch.uint8)
    y[0::2], y[1::2] = x & 15, x >> 4
    return y.reshape(shape)


def quant(x, bits):
    """Same operation ordering/dtypes as existing RTN, returning padded codes."""
    pad = (-x.shape[1]) % GROUP
    if pad:
        x = torch.cat([x, x.new_zeros(1, pad, NH, DIM)], 1)
    x = x.reshape(1, -1, GROUP, NH, DIM)
    high = (1 << (bits - 1)) - 1
    scale = (x.abs().amax(2, keepdim=True) / high).clamp_min(1e-8)
    codes = (x / scale).round().clamp(-high - 1, high).to(torch.int8)
    return codes, scale.to(torch.float16)


class PackedChunk:
    def __init__(self, x, hm, grouping, head=None, all_bits=4):
        if grouping not in ("frame", "chunk"):
            raise ValueError(grouping)
        self.dtype, self.length, self.hm = x.dtype, x.shape[1], hm
        self.head = torch.tensor([255 if head is None else head], dtype=torch.uint8)
        self.parts, self.all_bits = [], all_bits
        unit = FRAME if grouping == "frame" else CHUNK
        xr = (x.float() @ hm).to(x.dtype)
        for offset in range(0, x.shape[1], unit):
            part = xr[:, offset:offset + unit]
            c, s = quant(part, all_bits)
            if head is not None:
                assert all_bits == 4 and 0 <= head < NH
                c8, s8 = quant(part, 8)
                c[:, :, :, head] = c8[:, :, :, head]
                s[:, :, :, head] = s8[:, :, :, head]
            shape = tuple(c.shape)
            if all_bits == 8:
                low, high = c.view(torch.uint8).contiguous(), None
            else:
                low = pack_nibbles(c.view(torch.uint8))
                high = None if head is None else pack_nibbles(c[:, :, :, head].contiguous().view(torch.uint8) >> 4)
            self.parts.append((low, high, s, shape, part.shape[1]))

    def payload_bytes(self):
        return self.head.numel() + sum(
            lo.numel() * lo.element_size() + (0 if hi is None else hi.numel()) + sc.numel() * sc.element_size()
            for lo, hi, sc, _, _ in self.parts)

    def restore(self):
        restored = []
        head = int(self.head.item())
        for lo, hi, sc, shape, valid in self.parts:
            if self.all_bits == 8:
                c = lo.view(torch.int8).reshape(shape)
            else:
                u = unpack_nibbles(lo, shape)
                c = u.to(torch.int16)
                c[c >= 8] -= 16
                if hi is not None:
                    hs = shape[:3] + shape[4:]
                    uh = u[:, :, :, head] | (unpack_nibbles(hi, hs) << 4)
                    c[:, :, :, head] = uh.contiguous().view(torch.int8).to(torch.int16)
                c = c.to(torch.int8)
            y = (c.to(self.dtype) * sc.to(self.dtype)).reshape(1, -1, NH, DIM)[:, :valid]
            restored.append(y)
        return (torch.cat(restored, 1).float() @ self.hm.t()).to(self.dtype)


def attn(q, k, v, block=64):
    """Full-key FP32 attention in bounded query blocks; no GPU operations."""
    kt, vt = k.permute(1, 2, 0).contiguous(), v.permute(1, 0, 2).contiguous()
    out = torch.empty_like(q)
    for s in range(0, len(q), block):
        guard()
        logits = torch.bmm(q[s:s + block].permute(1, 0, 2), kt) / math.sqrt(q.shape[-1])
        p = logits.softmax(-1)
        out[s:s + block] = torch.bmm(p, vt).permute(1, 0, 2)
    return out


def energy(x):
    return float(x.double().square().sum())


def score(o, ref_o, wo, ref_energy):
    dy = (o - ref_o).reshape(o.shape[0], -1) @ wo.t()
    sse = energy(dy)
    return {"sse": sse, "nse": sse / ref_energy}


def build_k(k, hm, grouping, heads=None, all_bits=4):
    out, packets = [], []
    for c in range(6):
        packet = PackedChunk(k[:, c * CHUNK:(c + 1) * CHUNK], hm, grouping,
                             head=None if heads is None else heads[c], all_bits=all_bits)
        out.append(packet.restore())
        packets.append(packet)
    kfull = torch.cat(out + [k[:, PAST:]], 1)[0].float()
    return kfull, packets


def fixed_v4(v):
    qz = Quantizer("rtn", 4, 4, group=GROUP)
    vs, state_bytes = [], 0
    for c in range(6):
        state = qz.encode(v[:, c * CHUNK:(c + 1) * CHUNK], value=True)
        # Materialize the actual nibble payload, then verify it round-trips codes.
        packed = pack_nibbles(state.codes.contiguous().view(torch.uint8))
        u = unpack_nibbles(packed, state.codes.shape).to(torch.int16)
        u[u >= 8] -= 16
        assert torch.equal(u.to(torch.int8), state.codes)
        state_bytes += packed.numel() + state.scale.numel() * state.scale.element_size()
        vs.append(state.restore())
    return torch.cat(vs + [v[:, PAST:]], 1)[0].float(), state_bytes


def run_case(prompt, layer):
    guard()
    t0 = time.monotonic()
    path = DATA / f"p{prompt:02d}_s101/l{layer:02d}_c7.pt"
    source_hash = sha(path)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    saved = [a for a in manifest["artifacts"] if Path(a["path"]).name == path.name]
    assert len(saved) == 1 and saved[0]["sha256"] == source_hash and saved[0]["bytes"] == path.stat().st_size
    d = torch.load(path, map_location="cpu", weights_only=True)
    q, k, v = d["q"][0].float(), d["k"], d["v"]
    wo, bias = d["wo"].float(), d["bias"].float()
    assert tuple(k.shape) == (1, FULL, NH, DIM) and tuple(q.shape) == (CHUNK, NH, DIM)
    assert k.dtype == v.dtype == torch.bfloat16
    hm = hadamard(DIM, "cpu")
    ref_o = attn(q, k[0].float(), v[0].float())
    ref_y = ref_o.reshape(CHUNK, NH * DIM) @ wo.t() + bias
    ref_energy = energy(ref_y)
    assert ref_energy > 0
    native_delta = (ref_o - d["native_av"][0].float()).reshape(CHUNK, -1) @ wo.t()
    norm_heads = [int(k[:, c * CHUNK:(c + 1) * CHUNK].float().norm(dim=-1).mean((0, 1)).argmax()) for c in range(6)]
    mean_norm = k[:, :PAST].float().norm(dim=-1).mean((0, 1))
    v4, vbytes = fixed_v4(v)
    row = {"prompt": prompt, "layer": layer, "path": str(path), "source_sha256": source_hash, "manifest_verified": True,
           "reference_energy": ref_energy, "cpu_vs_native_post_nse": energy(native_delta) / ref_energy,
           "norm_heads": norm_heads, "mean_key_norm": mean_norm.tolist(), "groupings": {}}
    for grouping in ("frame", "chunk"):
        guard()
        kl, pl = build_k(k, hm, grouping)
        ol = attn(q, kl, v4)
        del kl
        kh, ph = build_k(k, hm, grouping, all_bits=8)
        oh = attn(q, kh, v4)
        del kh, ph
        result = {"base4": score(ol, ref_o, wo, ref_energy), "fixed_heads": [],
                  "base_payload_bytes": sum(p.payload_bytes() for p in pl) + vbytes}
        del pl
        expected_payload = None
        for h in range(NH):
            # Attention is independent across heads; preserve all W_o cross terms.
            mixed = ol.clone()
            mixed[:, h] = oh[:, h]
            metric = score(mixed, ref_o, wo, ref_energy)
            packet_bytes = 0
            for c in range(6):
                packet = PackedChunk(k[:, c * CHUNK:(c + 1) * CHUNK], hm, grouping, h)
                packet_bytes += packet.payload_bytes()
            candidate_bytes = packet_bytes + vbytes
            if expected_payload is None:
                expected_payload = candidate_bytes
            assert candidate_bytes == expected_payload
            result["fixed_heads"].append({"head": h, **metric, "payload_bytes": candidate_bytes})
        kn, pn = build_k(k, hm, grouping, norm_heads)
        on = attn(q, kn, v4)
        norm_bytes = sum(p.payload_bytes() for p in pn) + vbytes
        assert norm_bytes == expected_payload
        result["norm"] = {**score(on, ref_o, wo, ref_energy), "payload_bytes": norm_bytes}
        result["uniform_fixed_expectation_sse"] = sum(z["sse"] for z in result["fixed_heads"]) / NH
        result["best_same_input_diagnostic_head"] = min(range(NH), key=lambda h: (result["fixed_heads"][h]["sse"], h))
        row["groupings"][grouping] = result
        del kn, pn, on, ol, oh, mixed
    row["seconds"] = time.monotonic() - t0
    row["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return row


def summarize(rows):
    keyed = {(r["prompt"], r["layer"]): r for r in rows}
    result = {}
    for grouping in ("frame", "chunk"):
        comparisons = []
        for row in rows:
            other = keyed.get((4 if row["prompt"] == 0 else 0, row["layer"]))
            if other is None:
                continue
            calib = other["groupings"][grouping]
            h = min(range(NH), key=lambda j: (calib["fixed_heads"][j]["sse"], j))
            hnorm = max(range(NH), key=lambda j: (other["mean_key_norm"][j], -j))
            r = row["groupings"][grouping]
            comparisons.append({"prompt": row["prompt"], "layer": row["layer"],
                                "calibration_prompt": other["prompt"], "fixed_head": h,
                                "calibration_norm_head": hnorm, "norm_sse": r["norm"]["sse"],
                                "fixed_sse": r["fixed_heads"][h]["sse"],
                                "fixed_norm_sse": r["fixed_heads"][hnorm]["sse"],
                                "reference_energy": row["reference_energy"]})
        ns = sum(z["norm_sse"] for z in comparisons)
        fs = sum(z["fixed_sse"] for z in comparisons)
        per_prompt = {str(p): {"norm_sse": sum(z["norm_sse"] for z in comparisons if z["prompt"] == p),
                              "fixed_sse": sum(z["fixed_sse"] for z in comparisons if z["prompt"] == p)} for p in PROMPTS}
        valid = len(comparisons) == len(PROMPTS) * len(LAYERS)
        gain = None if fs == 0 else 1 - ns / fs
        nonworse = sum(z["norm_sse"] <= z["fixed_sse"] * (1 + 1e-5) for z in comparisons)
        proceed = valid and gain is not None and gain >= .05 and nonworse >= 6 and all(z["norm_sse"] < z["fixed_sse"] for z in per_prompt.values())
        # The proposal's 'fixed sufficient' judgment also needs selection-identity evidence; do not auto-close.
        result[grouping] = {"complete": valid, "comparisons": comparisons, "per_prompt": per_prompt,
                            "relative_sse_reduction": gain, "nonworse_cases": nonworse,
                            "continuation_signal": bool(proceed), "independent_holdout": False}
    return result


def selftest():
    torch.manual_seed(411)
    hm = hadamard(DIM, "cpu")
    # A full synthetic chunk tests the actual 1560/4680 group boundary and padding.
    x = (torch.randn(1, CHUNK, NH, DIM) * .3).to(torch.bfloat16)
    x[:, :, 7] *= 12
    original = HeadRuleQuantizer(k_bits=4, v_bits=4, k_delta=False, head_upgrade=True, device="cpu").encode_k(x)
    packed = PackedChunk(x, hm, "frame", original.head)
    assert torch.equal(packed.restore(), original.restore())
    qz = Quantizer("rtn", 4, 4, group=GROUP)
    direct_chunk = (qz.encode((x.float() @ hm).to(x.dtype)).restore().float() @ hm.t()).to(x.dtype)
    assert torch.equal(PackedChunk(x, hm, "chunk").restore(), direct_chunk)
    original_no = HeadRuleQuantizer(k_bits=4, v_bits=4, k_delta=False, head_upgrade=False, device="cpu").encode_k(x)
    assert torch.equal(PackedChunk(x, hm, "frame").restore(), original_no.restore())
    for bits in (4, 8):
        codes = torch.arange(-128 if bits == 8 else -8, 128 if bits == 8 else 8, dtype=torch.int16).to(torch.int8)
        if bits == 4:
            u = unpack_nibbles(pack_nibbles(codes.view(torch.uint8)), codes.shape).to(torch.int16)
            u[u >= 8] -= 16
            assert torch.equal(u.to(torch.int8), codes)
        else:
            raw = codes.view(torch.uint8)
            low, high = pack_nibbles(raw & 15), pack_nibbles(raw >> 4)
            rebuilt = unpack_nibbles(low, raw.shape) | (unpack_nibbles(high, raw.shape) << 4)
            assert torch.equal(rebuilt.view(torch.int8), codes)
    bytes_all = [PackedChunk(x, hm, "frame", h).payload_bytes() for h in range(NH)]
    assert len(set(bytes_all)) == 1
    # Exact head factorization with direct replacement, including W_o cross-head mixing.
    qq, kk, vv = torch.randn(9, NH, 8), torch.randn(17, NH, 8), torch.randn(17, NH, 8)
    kk2, w = kk + .1 * torch.randn_like(kk), torch.randn(31, NH * 8)
    ref = attn(qq, kk, vv, block=4)
    high = attn(qq, kk2, vv, block=4)
    assert torch.equal(ref, attn(qq, kk, vv, block=4))
    errors = []
    for h in range(NH):
        kh = kk.clone(); kh[:, h] = kk2[:, h]
        direct = attn(qq, kh, vv, block=4)
        joined = ref.clone(); joined[:, h] = high[:, h]
        assert torch.equal(direct, joined)
        errors.append(energy((direct - joined).reshape(9, -1) @ w.t()))
    return {"status": "pass", "real_captures_evaluated": 0, "gpu_calls": 0,
            "same_frame_existing_codec_bit_equal": True, "same_chunk_existing_codec_bit_equal": True,
            "mixed_frame_k_payload_bytes": bytes_all[0], "frame_group_count": 294, "chunk_group_count": 293,
            "all_fixed_heads_equal_payload": True, "head_reassembly_max_post_sse": max(errors),
            "source_sha256": {"diagnose.py": sha(__file__), "headrule_quantizer.py": sha(ROOT / "src/experiments/kv_headrule_rollout/headrule_quantizer.py"),
                              "quant.py": sha(ROOT / "src/kvq/quant.py")},
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("selftest", "run-captures"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registration-sha256")
    args = parser.parse_args()
    target = args.output.resolve()
    if not target.is_relative_to(HERE) or target.exists():
        raise RuntimeError("Output must be a new file inside this task directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "selftest":
        result = selftest()
        target.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    registration = HERE / "REGISTRATION_PROPOSAL_KO.txt"
    if not args.registration_sha256 or args.registration_sha256 != sha(registration):
        raise RuntimeError("An explicit reviewed registration SHA256 is required")
    r = {"status": "running", "registration_sha256": sha(registration), "code_sha256": sha(__file__),
         "kind": "development-capture diagnostic, not independent holdout", "rows": []}
    target.write_text(json.dumps(r, indent=2))
    try:
        for layer in LAYERS:
            for prompt in PROMPTS:
                row = run_case(prompt, layer)
                r["rows"].append(row)
                target.write_text(json.dumps(r, indent=2))
                print(json.dumps({"prompt": prompt, "layer": layer, "seconds": row["seconds"]}), flush=True)
        r.update(status="completed", summary=summarize(r["rows"]))
    except BaseException as e:
        r.update(status="incomplete", error=repr(e), summary=summarize(r["rows"]))
        raise
    finally:
        r["elapsed_seconds"] = time.monotonic() - START
        r["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        target.write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
