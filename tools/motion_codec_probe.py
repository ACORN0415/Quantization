"""Isolated, development-only capture and bounded translation codec screen."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'src'), str(ROOT/'src/scripts')]
import torch
from motion_codec_core import make_anchor, encode_residual, decode_residual, translate_reference, matched_border_reference

DATA = Path.home()/'sfq_data/motion_codec'
RESULT = ROOT/'src/experiments/motion_codec_development'
LAYERS = (0, 14, 29)
CHUNKS = (0, 1, 6, 7)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def source_hash():
    h = hashlib.sha256()
    for rel in ('tools/motion_codec_probe.py', 'tools/motion_codec_core.py',
                'tools/test_motion_codec_core.py', 'src/experiments/MOTION_CODEC_DEVELOPMENT.md'):
        h.update(rel.encode()); h.update((ROOT/rel).read_bytes())
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)); tmp.replace(path)


def status(detail, state='running', **extra):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), state, detail, flush=True)
    save(DATA/'status.json', dict(state=state, detail=detail, pid=os.getpid(),
         updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra))


def capture(prompt):
    import mechanism_worker as common
    import wan.modules.causal_model as model
    frozen = source_hash()
    model_frozen = common.pilot.source_hash()
    out = DATA/f'p{prompt:02d}_s101'
    if out.exists(): raise RuntimeError('refuse to overwrite capture directory')
    out.mkdir(parents=True)
    others = common.subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True)
    if any(p.strip().isdigit() and int(p.strip()) != os.getpid() for p in others.splitlines()):
        raise RuntimeError('GPU already occupied')
    started = time.monotonic(); status(f'CAPTURE p{prompt}: model loading')
    pilot = common.pilot
    pipe = pilot.build(101)
    text = (ROOT/'src/experiments/policy_study_prompts.txt').read_text().splitlines()[prompt]
    cond = pilot.encode(pipe, [text])[0]
    cache = pilot.bf16_cache(pipe, pilot.gpu)
    xc = pilot.cross_cache(pipe, pilot.gpu)
    noise = pilot.chunk_noise(pipe, 126, 101, pilot.gpu)
    expected = torch.load(Path.home()/f'sfq_data/policy_study/reference/p{prompt:02d}_s101/bf16.pt',
                          map_location='cpu', weights_only=True)['latent']
    original = model.causal_rope_apply
    pending = {}; counts = {layer: 0 for layer in LAYERS}; artifacts = []

    def rope(x, grid_sizes, freqs, start_frame=0):
        result = original(x, grid_sizes, freqs, start_frame=start_frame)
        # Every attention calls q then k; the write observer consumes the latest k call.
        pending.update(pre=x, post=result, grid=grid_sizes, freqs=freqs, start=start_frame)
        return result

    def observer(layer):
        def observe(k, v, start, *, local=True):
            call_index = counts[layer]; counts[layer] += 1
            chunk, call = divmod(call_index, 5)
            if call != 4 or chunk not in CHUNKS: return
            if pending['post'].data_ptr() != k.data_ptr():
                raise RuntimeError('pre/post RoPE association failed')
            pre = pending['pre']
            grid = pending['grid'][0].tolist()
            n = grid[1]*grid[2]
            restored_pre = pre.clone()
            for frame in (1,2):
                reference = make_anchor(pre[:, (frame-1)*n:frame*n]).restore()
                target = pre[:, frame*n:(frame+1)*n]
                code = encode_residual(target, reference, height=grid[1], width=grid[2],
                                       lossless=True, dy=1, dx=-1)
                restored = decode_residual(code, reference)
                if not torch.equal(restored,target): raise RuntimeError('lossless residual identity failed')
                restored_pre[:, frame*n:(frame+1)*n] = restored
            reapplied = original(restored_pre, pending['grid'], pending['freqs'], start_frame=pending['start'])
            if not torch.equal(reapplied, k): raise RuntimeError('RoPE identity failed')
            grid = pending['grid'][0].tolist()
            if grid[0] != 3 or grid[1]*grid[2] != pipe.frame_seq_length:
                raise RuntimeError('unexpected token geometry')
            path = out/f'c{chunk:02d}_l{layer:02d}.pt'
            torch.save(dict(k=pre.cpu(), v=v.cpu(), grid=grid, start_frame=pending['start'],
                            freqs=pending['freqs'].cpu()), path)
            artifacts.append(dict(file=str(path.relative_to(DATA)), sha256=sha(path),
                                  bytes=path.stat().st_size, layer=layer, chunk=chunk))
        return observe

    for layer in LAYERS: cache[layer]['write_observer'] = observer(layer)
    model.causal_rope_apply = rope
    try:
        for chunk in range(8):
            result = pilot.run_chunk(pipe, cache, xc, cond, noise, chunk, 101)
            if not torch.equal(result.cpu(), expected[:, chunk*3:(chunk+1)*3]):
                raise RuntimeError(f'capture changed BF16 reference at chunk {chunk}')
            status(f'CAPTURE p{prompt}: {chunk+1}/8 chunks; BF16 bitexact')
    finally:
        model.causal_rope_apply = original
    if any(n != 40 for n in counts.values()) or len(artifacts) != 12:
        raise RuntimeError('capture incomplete')
    if source_hash() != frozen or pilot.source_hash() != model_frozen:
        raise RuntimeError('diagnostic/model source changed')
    save(out/'manifest.json', dict(prompt=prompt, seed=101, source_sha256=frozen,
        teacher_bitexact_chunks=8, rope_identity_records=12, artifacts=artifacts,
        seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        cpu_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_model_sha256=model_frozen, domain='pre-RoPE K / unchanged V; refresh only'))


def relative(x, ref):
    return float((x.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-12))


def shift_search(target, ref, height, width):
    # Full-feature prediction MSE on current K, not codec endpoint/attention error.
    # Tie order prefers zero and then short translations; no future information.
    shifts = sorted(((y,x) for y in range(-2,3) for x in range(-2,3)),
                    key=lambda s:(s[0]*s[0]+s[1]*s[1],s))
    scored = []
    for dy, dx in shifts:
        predicted = translate_reference(ref, height, width, dy, dx)
        scored.append(float((target.float()-predicted.float()).square().mean()))
    return shifts[min(range(len(shifts)), key=lambda i:scored[i])]


def analyze():
    from wan.modules.causal_model import causal_rope_apply
    frozen = source_hash(); rows = []; manifests = []
    started = time.monotonic()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for prompt in (0,4):
        out = DATA/f'p{prompt:02d}_s101'
        manifest = json.loads((out/'manifest.json').read_text())
        if manifest['source_sha256'] != frozen: raise RuntimeError('source mismatch')
        for item in manifest['artifacts']:
            if sha(DATA/item['file']) != item['sha256']: raise RuntimeError('capture SHA mismatch')
        manifests.append(manifest)
        for chunk in CHUNKS:
            samples = {layer:torch.load(out/f'c{chunk:02d}_l{layer:02d}.pt', map_location=device,
                                      weights_only=True) for layer in LAYERS}
            _, height, width = samples[0]['grid']; n = height*width
            for pair in (0,1):
                frames = {}
                for layer, sample in samples.items():
                    frames[layer] = {}
                    for kind in ('k','v'):
                        raw = sample[kind][:, pair*n:(pair+1)*n]
                        target = sample[kind][:, (pair+1)*n:(pair+2)*n]
                        anchor = make_anchor(raw)
                        frames[layer][kind] = (anchor, anchor.restore(), target)
                begin = time.monotonic()
                shared = shift_search(frames[0]['k'][2], frames[0]['k'][1], height, width)
                torch.cuda.synchronize() if device == 'cuda' else None
                shared_search_s = time.monotonic()-begin
                for layer in LAYERS:
                    begin = time.monotonic()
                    oracle = shift_search(frames[layer]['k'][2], frames[layer]['k'][1], height, width)
                    torch.cuda.synchronize() if device == 'cuda' else None
                    oracle_search_s = time.monotonic()-begin
                    for arm, shift in (('direct',None),('same_position',(0,0)),
                                       ('shared_translation',shared),('matched_border_same_position',shared),
                                       ('per_layer_translation',oracle)):
                        row = dict(prompt=prompt, chunk=chunk, pair=pair, layer=layer, arm=arm,
                                   dy=None if shift is None else shift[0], dx=None if shift is None else shift[1],
                                   search_s=shared_search_s if arm=='shared_translation' else
                                            oracle_search_s if arm=='per_layer_translation' else 0.)
                        begin = time.monotonic(); resident=0; packed=0; anchor_bytes=0; workspace=0
                        for kind in ('k','v'):
                            anchor, reference, target = frames[layer][kind]
                            anchor_bytes += anchor.bytes()['resident']
                            workspace += reference.numel()*reference.element_size()
                            if shift is None:
                                code=make_anchor(target); restored=code.restore()
                            else:
                                if arm == 'matched_border_same_position':
                                    active_reference = matched_border_reference(reference,height,width,*shift)
                                    coded_shift = (0,0)
                                else:
                                    active_reference = reference
                                    coded_shift = shift
                                code=encode_residual(target, active_reference,height=height,width=width,
                                                     dy=coded_shift[0],dx=coded_shift[1])
                                restored=decode_residual(code,active_reference)
                            row[kind+'_rel']=relative(restored,target)
                            if kind == 'k':
                                position = samples[layer]['start_frame']+pair+1
                                grid = torch.tensor([[1,height,width]],device=device)
                                freqs = samples[layer]['freqs']
                                actual_post = causal_rope_apply(restored,grid,freqs,start_frame=position)
                                target_post = causal_rope_apply(target,grid,freqs,start_frame=position)
                                row['k_post_rope_rel']=relative(actual_post,target_post)
                            resident += code.bytes()['resident']+anchor.bytes()['resident']
                            packed += code.bytes()['packed_estimate']+anchor.bytes()['packed_estimate']
                            if arm == 'matched_border_same_position':
                                # Additional int16 dy/dx mask geometry; zero translation remains explicit.
                                border_metadata = torch.tensor(shift,dtype=torch.int16,device=device)
                                resident += border_metadata.numel()*border_metadata.element_size()
                                packed += border_metadata.numel()*border_metadata.element_size()
                        torch.cuda.synchronize() if device=='cuda' else None
                        row.update(error_mean=(row['k_rel']+row['v_rel'])/2,
                            pair_resident_payload_bytes=resident,pair_packed_estimate_bytes=packed,
                            anchor_resident_bytes=anchor_bytes,reference_restore_workspace_bytes=workspace,
                            encode_decode_metric_wall_s=time.monotonic()-begin)
                        rows.append(row)
            status(f'ANALYZE p{prompt} c{chunk}: {len(rows)}/240 rows')
    decisions=[]
    for prompt in (0,4):
        means={arm:sum(r['error_mean'] for r in rows if r['prompt']==prompt and r['arm']==arm)/24
               for arm in ('direct','same_position','shared_translation','matched_border_same_position','per_layer_translation')}
        improvement=1-means['shared_translation']/means['same_position']
        border_improvement=1-means['shared_translation']/means['matched_border_same_position']
        decisions.append(dict(prompt=prompt,means=means,shared_vs_same_relative_reduction=improvement,
                              shared_vs_matched_border_relative_reduction=border_improvement,
                              passes=improvement>=.05 and border_improvement>=.05))
    if any(not math.isfinite(r[k]) for r in rows for k in ('k_rel','v_rel','k_post_rope_rel')):
        raise RuntimeError('nonfinite diagnostic metric')
    if len(rows)!=240 or source_hash()!=frozen: raise RuntimeError('analysis incomplete/source changed')
    RESULT.mkdir(parents=True,exist_ok=True)
    with (RESULT/'rows.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    decision=dict(source_sha256=frozen,advance_restricted_variant=all(d['passes'] for d in decisions),
        decisions=decisions,analysis_seconds=time.monotonic()-started,
        scope='development only; global translations +/-2; NOT a TempoKV comparison or dense-motion rejection',
        timing_scope='unwarmed instrumented codec+metric wall; not deployment throughput; shared search repeated in rows',
        byte_scope='pair anchor+target tensor payload; maps duplicated per K/V/layer; restore workspace separate; packed hypothetical',
        captures=manifests, analysis_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        analysis_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None)
    save(RESULT/'decision.json',decision)
    status('COMPLETE: restricted translation '+('GO' if decision['advance_restricted_variant'] else 'NO-GO'),
           state='pass',advance_restricted_variant=decision['advance_restricted_variant'])
    print(json.dumps(decisions,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['capture','analyze']);p.add_argument('--prompt',type=int,choices=[0,4])
    a=p.parse_args();torch.set_grad_enabled(False);torch.set_num_threads(4)
    try:
        if a.mode=='capture':
            if a.prompt is None: p.error('--prompt required')
            capture(a.prompt)
        else: analyze()
    except Exception as exc:
        status(str(exc),'failed');raise

if __name__=='__main__': main()
