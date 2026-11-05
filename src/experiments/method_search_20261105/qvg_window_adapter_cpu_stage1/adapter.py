"""Finite absolute-address adapter for the pinned QVG cache API.

This module never imports a model, codec or upstream module. Callers inject them.
CPU tests with a fake codec verify lifecycle semantics, not actual PRQ fidelity.
"""
from __future__ import annotations

from functools import wraps
from types import MethodType
from typing import Callable

import torch


class AdapterContractError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise AdapterContractError(message)


def rope_absolute_segments(x, grid_sizes, freqs, *, start_frame, rope_fn):
    """Same 3-frame arithmetic as upstream long RoPE, with an absolute origin."""
    rows = grid_sizes.tolist()
    _require(len(rows) == x.shape[0] and len(rows) > 0, "grid batch mismatch")
    _require(all(row == rows[0] for row in rows), "heterogeneous grids unsupported")
    f, h, w = rows[0]
    _require(f == 3 and h > 0 and w > 0, "requires three-frame positive grid")
    width = f * h * w
    _require(x.shape[1] > 0 and x.shape[1] % width == 0, "partial block RoPE unsupported")
    _require(start_frame >= 0 and start_frame % 3 == 0, "unaligned absolute RoPE origin")
    _require(x.shape[-1] % 2 == 0, "odd head dimension unsupported")
    end_frame = start_frame + x.shape[1] // (h * w)
    _require(freqs.ndim == 2 and freqs.shape[0] >= max(end_frame, h, w)
             and freqs.shape[1] == x.shape[-1] // 2, "frequency table too short or wrong width")
    return torch.cat([
        rope_fn(x[:, p:p + width], grid_sizes, freqs,
                start_frame=start_frame + p // (h * w))
        for p in range(0, x.shape[1], width)
    ], dim=1)


class FiniteWindowAdapter:
    """Single-generation controller; all positions and span metadata stay absolute.

    A failed mutating callback poisons the generation. It does not roll back the
    upstream quantizer's already committed layers. reset() starts a fresh run.
    """

    def __init__(self, window_frames, horizon_frames=126, block_frames=3,
                 quant_group_frames=24, calls_per_block=5, sink_frames=0):
        _require(window_frames in (21, 63, 126), "only windows 21/63/126 registered")
        _require(horizon_frames == 126 and block_frames == 3 and quant_group_frames == 24
                 and calls_per_block == 5 and sink_frames == 0,
                 "unsupported horizon/block/group/call/sink configuration")
        self.window_frames = window_frames
        self.horizon_frames = horizon_frames
        self.block_frames = block_frames
        self.quant_group_frames = quant_group_frames
        self.calls_per_block = calls_per_block
        self.sink_frames = sink_frames
        self.caches = None
        self.layers = []
        self.frontier_tokens = 0
        self.poisoned = False
        self.failure = None
        self.frame_seq_length = None

    def _usable(self):
        _require(not self.poisoned, "generation poisoned; reset before reuse")
        _require(self.caches is not None, "initialize first")

    def _poison(self, exc):
        self.poisoned = True
        self.failure = f"{type(exc).__name__}: {exc}"

    def reset(self):
        """Drop this controller's old generation, including quantized span metadata."""
        if self.caches is not None:
            for cache in self.caches:
                cache["k"].clear()
                cache["v"].clear()
        self.caches = None
        self.layers = []
        self.frontier_tokens = 0
        self.frame_seq_length = None
        self.poisoned = False
        self.failure = None

    def initialize(self, cache_cls, *, batch_size, frame_seq_length, num_heads,
                   head_dim, num_layers, dtype, device):
        _require(self.caches is None and not self.poisoned, "reset before reinitialization")
        _require(all(isinstance(i, int) and i > 0 for i in
                     (batch_size, frame_seq_length, num_heads, head_dim, num_layers)),
                 "invalid cache dimensions")
        _require(head_dim % 2 == 0, "odd head dimension unsupported")
        new = []
        for _ in range(num_layers):
            def make():
                return cache_cls(batch_size, frame_seq_length, num_heads, head_dim,
                                 self.horizon_frames, dtype, device, layout="BSHD")
            new.append({"k": make(), "v": make(),
                        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                        "local_end_index": torch.tensor([0], dtype=torch.long, device=device)})
        self.frame_seq_length = frame_seq_length
        self.caches = new
        self.layers = [{"start": None, "end": 0, "calls": 0, "visible": (0, 0)}
                       for _ in range(num_layers)]
        return new

    def quantize(self, start, end, max_tokens, callback: Callable[[], object]):
        """Call the unchanged whole-group quantizer, then advance success frontier."""
        self._usable()
        fs = self.frame_seq_length
        _require(start == self.frontier_tokens and end == start + 24 * fs,
                 "quantization must cover the next original 24-frame group")
        _require(end <= self.horizon_frames * fs and max_tokens >= end,
                 "quantization outside committed capacity")
        _require(all(s["end"] == end and s["calls"] == 5 for s in self.layers),
                 "quantization before complete current refresh or inconsistent layers")
        sc, ec = start // fs, end // fs
        for cache in self.caches:
            _require(int(cache["global_end_index"].item()) == end
                     and int(cache["local_end_index"].item()) == end,
                     "quantization index mismatch")
            for side in ("k", "v"):
                _require(all(int(v) == 1 for v in cache[side].chunk_state[sc:ec]),
                         "pending group must retain original uncompressed values")
        try:
            result = callback()
            for cache in self.caches:
                for side in ("k", "v"):
                    obj = cache[side]
                    _require(all(int(v) == 2 for v in obj.chunk_state[sc:ec]),
                             "callback did not commit packed K/V; quant_type none/fake unsupported")
                    _require(any(s["start_chunk"] == sc and s["end_chunk"] == ec
                                 and isinstance(s["quant_data"], dict)
                                 for s in obj.quantized_spans), "missing whole-group packed span")
            self.frontier_tokens = end
            return result
        except BaseException as exc:
            self._poison(exc)
            raise

    def visible_range(self, layer_index):
        return tuple(self.layers[layer_index]["visible"])

    def inspect(self):
        return {"window_frames": self.window_frames, "capacity_frames": self.horizon_frames,
                "frontier_tokens": self.frontier_tokens, "poisoned": self.poisoned,
                "failure": self.failure, "layers": [dict(s) for s in self.layers]}

    def attend(self, layer_index, kv_cache, *, current_start, q, k, v, grid_sizes,
               freqs, rope_fn, attention_fn):
        self._usable()
        _require(isinstance(layer_index, int) and 0 <= layer_index < len(self.layers), "bad layer")
        _require(kv_cache is self.caches[layer_index], "cache identity mismatch")
        fs = self.frame_seq_length
        _require(isinstance(current_start, int) and current_start >= 0
                 and current_start % (3 * fs) == 0, "unaligned absolute block address")
        end = current_start + 3 * fs
        _require(end <= self.horizon_frames * fs, "finite horizon exceeded")
        rows = grid_sizes.tolist()
        _require(len(rows) == kv_cache["k"].batch_size and rows
                 and all(row == rows[0] for row in rows)
                 and rows[0][0] == 3 and rows[0][1] > 0 and rows[0][2] > 0
                 and rows[0][1] * rows[0][2] == fs, "unsupported grid")
        shape = (len(rows), 3 * fs, kv_cache["k"].num_heads, kv_cache["k"].head_dim)
        _require(tuple(q.shape) == shape and tuple(k.shape) == shape and tuple(v.shape) == shape,
                 "Q/K/V shape mismatch")
        _require(k.dtype == kv_cache["k"].dtype and v.dtype == kv_cache["v"].dtype
                 and q.device == k.device == v.device, "dtype/device mismatch")
        _require(freqs.ndim == 2 and freqs.shape[0] >= max(end // fs, rows[0][1], rows[0][2])
                 and freqs.shape[1] == shape[-1] // 2, "invalid frequency extent")
        state = self.layers[layer_index]
        if state["start"] is None:
            _require(current_start == 0, "first block must start at zero")
            next_calls = 1
        elif current_start == state["start"]:
            _require(state["calls"] < 5, "more than five current-block writes")
            next_calls = state["calls"] + 1
        else:
            _require(current_start == state["end"] and state["calls"] == 5,
                     "nonsequential block or unfinished current refresh")
            next_calls = 1
        _require(int(kv_cache["global_end_index"].item()) == state["end"]
                 and int(kv_cache["local_end_index"].item()) == state["end"], "external index mutation")
        _require(self.frontier_tokens == (current_start // (24 * fs)) * 24 * fs,
                 "scheduled whole-group quantization missing or frontier advanced early")
        sc, ec = current_start // fs, end // fs
        for side in ("k", "v"):
            obj = kv_cache[side]
            _require(obj.layout == "BSHD" and obj.kv_cache_size >= self.horizon_frames * fs,
                     "absolute-address capacity/layout mismatch")
            _require(not any(int(a) == 2 for a in obj.chunk_state[sc:ec]),
                     "packed historical overwrite forbidden")
            _require(not any(s["start_chunk"] < ec and s["end_chunk"] > sc
                             for s in obj.quantized_spans), "overlapping packed span forbidden")
        left = max(0, end - self.window_frames * fs)
        free_end = min(left, self.frontier_tokens)
        try:
            query = rope_fn(q, grid_sizes, freqs, start_frame=current_start // fs).type_as(v)
            kv_cache["k"].write(current_start, end, k)
            kv_cache["v"].write(current_start, end, v)
            for side in ("k", "v"):
                kv_cache[side].evict_range(0, free_end)
            key = rope_absolute_segments(kv_cache["k"].read(left, end), grid_sizes, freqs,
                                         start_frame=left // fs, rope_fn=rope_fn).type_as(v)
            value = kv_cache["v"].read(left, end)
            output = attention_fn(query, key, value)
            kv_cache["global_end_index"].fill_(end)
            kv_cache["local_end_index"].fill_(end)
            state.update(start=current_start, end=end, calls=next_calls, visible=(left, end))
            return output, kv_cache
        except BaseException as exc:
            self._poison(exc)
            raise


class PipelineInstallation:
    def __init__(self, pipe, controller, patches):
        self.pipeline = pipe
        self.controller = controller
        self._patches = patches
        self.installed = True

    def uninstall(self):
        if self.installed:
            self.controller.reset()
            self.pipeline.kv_cache1 = None
            for obj, name, was_local, old in reversed(self._patches):
                if was_local:
                    setattr(obj, name, old)
                else:
                    delattr(obj, name)
            self.installed = False


def install_pipeline(pipe, *, cache_cls, rope_fn, attention_fn,
                     window_frames, horizon_frames=126):
    """Mount onto the pinned pipeline without changing source, codec or schedule.

    Every inference validates unsupported conditions before clearing old owned
    cache. The original inference still calls the original quantizer at j=8,16,...
    The adapter replaces only initialization, cache attention and success tracking.
    """
    ctl = FiniteWindowAdapter(window_frames, horizon_frames)
    model = pipe.generator.model
    blocks = list(model.blocks)
    _require(pipe.kv_cache1 is None, "install only before the first cache allocation")
    _require(len(blocks) == pipe.num_transformer_blocks and len(blocks) > 0, "layer count mismatch")
    _require(pipe.num_frame_per_block == 3 and not pipe.independent_first_frame,
             "three-frame blocks without independent first frame required")
    _require(len(pipe.denoising_step_list) == 4, "four denoising calls plus refresh required")
    _require(getattr(pipe.quant_config, "quant_type", None) not in (None, "none"),
             "packed quantizer required; quant_type none unsupported")
    _require(pipe.local_attn_size == window_frames and model.local_attn_size == window_frames,
             "pipeline/model window mismatch")
    modules = [b.self_attn for b in blocks]
    for module in modules:
        _require(module.sink_size == 0 and module.local_attn_size == window_frames,
                 "sink/window mismatch")
        _require(module.num_heads == modules[0].num_heads and module.head_dim == modules[0].head_dim,
                 "heterogeneous head shapes unsupported")
    old_init = pipe._initialize_kv_cache
    old_quantize = pipe.quantize_kv_cache
    old_inference = pipe.inference
    patches = []

    def patch(obj, name, fn):
        was_local = name in vars(obj)
        previous = vars(obj)[name] if was_local else None
        patches.append((obj, name, was_local, previous))
        setattr(obj, name, MethodType(fn, obj))

    @wraps(old_init)
    def init(self, batch_size, dtype, device, target_frames=None):
        # Upstream ignores target_frames for local windows: deliberately replace it.
        self.kv_cache1 = ctl.initialize(cache_cls, batch_size=batch_size,
                                       frame_seq_length=self.frame_seq_length,
                                       num_heads=modules[0].num_heads, head_dim=modules[0].head_dim,
                                       num_layers=len(modules), dtype=dtype, device=device)

    @wraps(old_quantize)
    def quantize(self, tokens_to_quantize_start, tokens_to_quantize_end, max_tokens_to_quantize):
        return ctl.quantize(tokens_to_quantize_start, tokens_to_quantize_end, max_tokens_to_quantize,
                            lambda: old_quantize(tokens_to_quantize_start,
                                                 tokens_to_quantize_end, max_tokens_to_quantize))

    @wraps(old_inference)
    def inference(self, noise, text_prompts, initial_latent=None, return_latents=False,
                  profile=False, low_memory=False):
        _require(initial_latent is None, "initial latent unsupported")
        _require(noise.ndim == 5 and noise.shape[1] == 126, "requires exactly 126 latent frames")
        _require(self.num_frame_per_block == 3 and not self.independent_first_frame
                 and len(self.denoising_step_list) == 4, "generation configuration changed")
        _require(self.local_attn_size == window_frames and model.local_attn_size == window_frames
                 and all(m.sink_size == 0 and m.local_attn_size == window_frames for m in modules),
                 "window/sink configuration changed")
        _require(getattr(self.quant_config, "quant_type", None) not in (None, "none"),
                 "packed quantizer configuration changed")
        _require(self.kv_cache1 is None or self.kv_cache1 is ctl.caches, "foreign cache cannot be reset")
        ctl.reset()
        self.kv_cache1 = None
        try:
            return old_inference(noise, text_prompts, initial_latent=initial_latent,
                                 return_latents=return_latents, profile=profile, low_memory=low_memory)
        except BaseException as exc:
            ctl._poison(exc)
            raise

    patch(pipe, "_initialize_kv_cache", init)
    patch(pipe, "quantize_kv_cache", quantize)
    patch(pipe, "inference", inference)
    for index, module in enumerate(modules):
        def make_attend(layer_index):
            def attend(self, kv_cache, current_start, q, k, v, block_mask,
                       grid_sizes, freqs, seq_lens, s):
                return ctl.attend(layer_index, kv_cache, current_start=current_start,
                                  q=q, k=k, v=v, grid_sizes=grid_sizes, freqs=freqs,
                                  rope_fn=rope_fn, attention_fn=attention_fn)
            return attend
        patch(module, "attn_kv_cache_prerope", make_attend(index))
    return PipelineInstallation(pipe, ctl, patches)
