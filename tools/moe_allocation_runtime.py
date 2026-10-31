"""Runtime utilities for the registered, bounded OLMoE diagnostic.

The eager expert reassembly follows Hugging Face Transformers' OlmoeExperts
forward order (Apache-2.0, transformers/models/olmoe/modeling_olmoe.py).
This module does not implement GEMQ's inference kernels.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import threading
import time
from datetime import datetime, timezone

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from moe_allocation_objectives import BITS, pack_reference, unpack_reference

ROOT = Path('/home/acorn0415/sfq_data/moe_allocation_stage1')
REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / 'src/experiments/moe_allocation_stage1'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def event(stage, **kwargs):
    record = dict(utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), stage=stage, **kwargs)
    with (RECORD / 'progress.jsonl').open('a') as out:
        out.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


def digest_tensor(tensor):
    t = tensor.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def setup():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261031)
    np.random.seed(20261031)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(20 * 2**30 / torch.cuda.get_device_properties(0).total_memory)


def memory():
    return dict(rss_bytes=psutil.Process().memory_info().rss,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                gpu_allocated_bytes=torch.cuda.memory_allocated(),
                gpu_reserved_bytes=torch.cuda.memory_reserved(),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())


class Guard:
    def __init__(self, name):
        self.name = name
        self.deadline = json.loads((RECORD / 'run_state.json').read_text())['deadline_epoch']
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        while not self.finished.wait(0.25):
            reason = None
            if time.time() >= self.deadline:
                reason = '120 minute wall cap'
            elif psutil.Process().memory_info().rss > 48 * 2**30:
                reason = '48 GiB process RSS cap'
            elif torch.cuda.is_initialized() and torch.cuda.max_memory_allocated() > 20 * 2**30:
                reason = '20 GiB CUDA allocated cap'
            if reason:
                event('RESOURCE_CAP', run=self.name, reason=reason)
                write_json(RECORD / f'{self.name}_failure.json', dict(status='INCOMPLETE', reason=reason))
                os._exit(90)

    def close(self):
        self.finished.set()


def load_block():
    from transformers import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
    config = OlmoeConfig.from_pretrained(ROOT / 'model', local_files_only=True)
    config._experts_implementation = 'eager'
    with torch.device('meta'):
        block = OlmoeSparseMoeBlock(config)
    state = torch.load(ROOT / 'block.pt', map_location='cpu', weights_only=True, mmap=True)
    block.load_state_dict(state, assign=True)
    del state
    return block.to('cuda').eval()


def routes(indices, weights, count=64):
    result = np.zeros((indices.shape[0], count), dtype=np.float64)
    np.put_along_axis(result, indices.cpu().numpy(), weights.double().cpu().numpy(), axis=1)
    return result


def routing_summary(documents, selected):
    g = np.concatenate([routes(d['indices'], d['gates']) for d in documents])
    active = g[:, selected] != 0
    return dict(tokens=len(g), experts=selected,
                activations=active.sum(axis=0).tolist(),
                pair_counts={f'{selected[e]}-{selected[f]}':int((active[:, e] & active[:, f]).sum())
                             for e in range(4) for f in range(e+1, 4)},
                tokens_with_at_least_two=int((active.sum(axis=1) >= 2).sum()),
                weighted_pair_sums=(g[:, selected].T @ g[:, selected]).tolist())


def choose_experts(documents):
    g = np.concatenate([routes(d['indices'], d['gates']) for d in documents])
    w = g.T @ g
    pairs = [(float(w[e, f]), e, f) for e in range(64) for f in range(e+1, 64)]
    best = sorted(pairs, key=lambda x:(-x[0], x[1], x[2]))[0]
    if best[0] == 0:
        raise RuntimeError('No calibration co-routing opportunity; stop the registered layer')
    selected = [best[1], best[2]]
    while len(selected) < 4:
        candidates = [(sum(w[e, f] for f in selected), e) for e in range(64) if e not in selected]
        selected.append(sorted(candidates, key=lambda x:(-x[0], x[1]))[0][1])
    return sorted(selected)


@torch.inference_mode()
def reassemble(block, x, indices, gates):
    """Preserve the native eager expert order, shapes and BF16 operations."""
    output = torch.zeros_like(x)
    expert_mask = F.one_hot(indices, num_classes=block.experts.num_experts+1).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == block.experts.num_experts:
            continue
        position, tokens = torch.where(expert_mask[expert_idx])
        gate, up = F.linear(x[tokens], block.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        y = F.linear(block.experts.act_fn(gate)*up, block.experts.down_proj[expert_idx])
        y = y*gates[tokens, position, None]
        output.index_add_(0, tokens, y.to(output.dtype))
    return output


def decode_projection(item, device='cuda'):
    codes = unpack_reference(item['packed'].numpy(), item['bits'], item['shape'][1])
    q = torch.from_numpy(codes).to(device=device, dtype=torch.bfloat16).reshape(-1, 128)
    scales = item['scale'].to(device)
    zeros = item['zero'].to(device)
    return ((q-zeros)*scales).reshape(item['shape'])


@torch.inference_mode()
def build_quantized_candidates(block, selected):
    archive = {}
    rows = []
    for e in selected:
        gu = block.experts.gate_up_proj[e]
        matrices = {'gate':gu[:1024], 'up':gu[1024:], 'down':block.experts.down_proj[e]}
        for b in BITS:
            archive[(e,b)] = {}
            for name, w in matrices.items():
                grouped = w.reshape(-1,128)
                scale = (grouped.amax(1,keepdim=True)-grouped.amin(1,keepdim=True)).clamp(min=1e-5)/(2**b-1)
                zero = (-torch.round(grouped.amin(1,keepdim=True)/scale)).clamp(0,2**b-1)
                q = (torch.round(grouped/scale)+zero).clamp(0,2**b-1)
                assert torch.isfinite(scale).all() and torch.isfinite(q).all()
                reference = ((q-zero)*scale).reshape(w.shape)
                codes = q.to(torch.uint8).reshape(w.shape).cpu().numpy()
                packed = torch.from_numpy(pack_reference(codes,b).copy())
                item = dict(bits=b,shape=list(w.shape),packed=packed,scale=scale.cpu(),zero=zero.cpu())
                decoded = decode_projection(item)
                code_ok = np.array_equal(unpack_reference(packed.numpy(),b,w.shape[1]),codes)
                metadata_ok = bits_equal(item['scale'].cuda(),scale) and bits_equal(item['zero'].cuda(),zero)
                decoded_ok = bits_equal(reference,decoded)
                assert code_ok and metadata_ok and decoded_ok,(e,b,name)
                payload = sum(item[k].numel()*item[k].element_size() for k in ('packed','scale','zero'))
                rows.append(dict(expert=e,bits=b,projection=name,code_equal=code_ok,
                                 metadata_equal=metadata_ok,decoded_bits_equal=decoded_ok,
                                 payload_bytes=payload,packed_sha256=digest_tensor(packed),
                                 decoded_sha256=digest_tensor(decoded)))
                archive[(e,b)][name] = item
        event('expert_packed',expert=e)
    torch.save(archive,ROOT/'packed.pt')
    return rows


def load_versions():
    archive = torch.load(ROOT/'packed.pt',map_location='cpu',weights_only=True)
    versions = {}
    for key, entry in archive.items():
        versions[key] = (torch.cat([decode_projection(entry['gate']),decode_projection(entry['up'])]),
                         decode_projection(entry['down']))
    return versions


@torch.inference_mode()
def collect_bank(block, versions, document, selected):
    """Use each expert's full document dispatch batch, as the native block does.

    Scoring streams 64-token slices. Only one document's dense error bank is
    resident; expert GEMMs are not retiled, which could change BF16 rounding.
    """
    x = document['x'].cuda()
    indices = document['indices'].cuda()
    gates = document['gates'].cuda()
    bank = np.zeros((len(x),4,3,x.shape[1]),dtype=np.float64)
    for slot,e in enumerate(selected):
        positions,tokens = torch.where((indices==e).T)
        if len(tokens)==0:
            continue
        state = x[tokens]
        gate,up = F.linear(state,block.experts.gate_up_proj[e]).chunk(2,dim=-1)
        original = F.linear(block.experts.act_fn(gate)*up,block.experts.down_proj[e])
        original_cpu = original.double().cpu().numpy()
        g = gates[tokens,positions].double().cpu().numpy()
        token_cpu = tokens.cpu().numpy()
        for p,b in enumerate(BITS):
            gu,down = versions[(e,b)]
            gate,up = F.linear(state,gu).chunk(2,dim=-1)
            quant = F.linear(block.experts.act_fn(gate)*up,down)
            bank[token_cpu,slot,p] = g[:,None]*(quant.double().cpu().numpy()-original_cpu)
    return bank


@torch.inference_mode()
def native_output(block, versions, document, selected, allocation):
    originals = {e:(block.experts.gate_up_proj[e].clone(),block.experts.down_proj[e].clone()) for e in selected}
    try:
        for e,b in zip(selected,allocation):
            gu,down = versions[(e,b)]
            block.experts.gate_up_proj[e].copy_(gu)
            block.experts.down_proj[e].copy_(down)
        output = block.experts(document['x'].cuda(),document['indices'].cuda(),document['gates'].cuda()).cpu()
    finally:
        for e,(gu,down) in originals.items():
            block.experts.gate_up_proj[e].copy_(gu)
            block.experts.down_proj[e].copy_(down)
    return output
