"""Inspect existing train routing on CPU; no quantization or model forward.

Prints JSON only. Rules are analytic policy replays, not executions of either
upstream quantization package, which can also have numerical failure fallbacks.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch


REPO = Path(__file__).resolve().parents[1]
DATA = Path('/home/acorn0415/sfq_data/moe_allocation_stage1')
PRIOR = REPO/'src/experiments/moe_allocation_stage1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    source = DATA/'captures.pt'
    source_sha = sha(source)
    prior_audit = json.loads((PRIOR/'audit.json').read_text())
    assert source_sha == prior_audit['actual_input_files']['captures.pt']['sha256']
    config = json.loads((DATA/'model/config.json').read_text())
    num_experts, top_k = config['num_experts'], config['num_experts_per_tok']
    hashes = {r['id']: r['indices'] for r in json.loads((PRIOR/'capture_hashes.json').read_text())}
    captures = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    docs, counts = [], [0]*num_experts
    for item in captures:
        if item['split'] != 'train':
            continue
        idx = item['indices'].contiguous()
        assert hashlib.sha256(idx.view(torch.uint8).numpy().tobytes()).hexdigest() == hashes[item['id']]
        assert idx.shape == (512, top_k)
        assert 0 <= int(idx.min()) <= int(idx.max()) < num_experts
        assert all(len(set(row)) == top_k for row in idx.tolist())
        freq = torch.bincount(idx.flatten().long(), minlength=num_experts).tolist()
        independent = Counter(e for row in idx.tolist() for e in row)
        assert freq == [independent[e] for e in range(num_experts)]
        counts = [a+b for a,b in zip(counts,freq)]
        docs.append(dict(id=item['id'],tokens=len(idx),counts=freq))
    assert len(docs) == 8
    total = sum(d['tokens'] for d in docs)
    assert sum(counts) == total*top_k
    coverage_threshold = .005*total
    coverage_only = [n >= coverage_threshold for n in counts]
    projections = {}
    for name,k in [('gate_up',config['hidden_size']),('down',config['intermediate_size'])]:
        combined = [n >= coverage_threshold and n >= 2*k for n in counts]
        projections[name] = dict(input_width=k,absolute_threshold=2*k,
            coverage_only_gptq_count=sum(coverage_only),combined_rule_gptq_count=sum(combined),
            disagreement_experts=[e for e in range(num_experts) if coverage_only[e] != combined[e]],
            below_input_width=sum(n<k for n in counts),
            below_twice_input_width=sum(n<2*k for n in counts),
            hessian_payload_bytes_if_all_observed_experts_collected=sum(n>0 for n in counts)*k*k*4)
    ranked = sorted(range(num_experts),key=lambda e:(counts[e],e))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='PASS',checked_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=source_sha,code_sha256=sha(__file__),
        split='train only',documents=docs,tokens=total,num_experts=num_experts,top_k=top_k,
        counts=counts,min_count=min(counts),median_count=statistics.median(counts),max_count=max(counts),
        coverage_threshold=coverage_threshold,projections=projections,
        count_rank_order=ranked,
        model_forwards=0,new_quantization=0,cuda_initialized=False,
        limitations='Policy decisions only. Does not establish quality loss, numerical rank, observed memory/time, or full-package execution.',
        elapsed_seconds=time.perf_counter()-started),indent=2))


if __name__ == '__main__':
    main()
