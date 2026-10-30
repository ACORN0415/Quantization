"""Raw-frame quality evaluation for the TT video development study.

Inputs are retained latent.pt files, never compressed MP4s. This module owns no
sampling state. Source/input/preview hashes make completed evaluations resumable.
Run from src/ so the existing Wan VAE wrapper finds its offline checkpoint.
"""
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / 'sfq_data/tt_video'
ARMS = ('tt', 'quant', 'quant_tt', 'tt_quant')
FRAMES = 501
LATE_START = 249


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def cases(prompt=None):
    prompts = (0, 4) if prompt is None else (prompt,)
    return [dict(prompt=p, bits=b, arm=a, key=f'p{p}_b{b}_{a}')
            for p in prompts for b, a in [(0, 'bf16')] + [(b, a) for b in (2, 3, 4) for a in ARMS]]


def window_indices():
    whole = list(range(FRAMES))
    boundary = [12*c-3 for c in range(1, 42)]
    boundary_set = set(boundary)
    inside = [i for i in whole if i not in boundary_set]
    return dict(whole=whole, late=list(range(LATE_START, FRAMES)), boundary=boundary,
                late_boundary=[i for i in boundary if i >= LATE_START], inside=inside,
                late_inside=[i for i in inside if i >= LATE_START])


def transition_pairs():
    pairs = [dict(kind='inside', chunk=c, i=12*c+offset, j=12*c+offset+1)
             for c in range(42) for offset in (2, 3)]
    pairs += [dict(kind='boundary', chunk=c, i=12*c-4, j=12*c-3) for c in range(1, 42)]
    assert len(pairs) == 125 and all(0 <= p['i'] < p['j'] < FRAMES for p in pairs)
    return pairs


def strict_mean(values):
    if not values:
        raise ValueError('empty metric window')
    return None if any(x is None for x in values) else float(np.mean(values, dtype=np.float64))


def summarize_frames(musiq, dino, pixel_mse):
    if (len(musiq), len(dino), len(pixel_mse)) != (501, 500, 501):
        raise ValueError('unexpected metric lengths')
    result = {}
    for name, indices in window_indices().items():
        mse = float(np.mean([pixel_mse[i] for i in indices], dtype=np.float64))
        # Transition index i-1 ends at raw frame i. Frame 0 has no transition.
        transitions = [dino[i-1] for i in indices if i > 0]
        result[name] = dict(frame_count=len(indices), musiq=float(np.mean([musiq[i] for i in indices])),
                            dino_subject=float(np.mean(transitions)), pixel_mse=mse,
                            pixel_psnr_db=None if mse == 0 else -10*math.log10(mse),
                            pixel_psnr_infinite=mse == 0)
    return result


def summarize_temporal(rows):
    result = {}
    for region, late in (('whole', False), ('late', True)):
        for kind in ('boundary', 'inside'):
            selected = [r for r in rows if r['kind'] == kind and (not late or r['chunk'] >= 21)]
            result[f'{region}_{kind}'] = dict(
                pairs=len(selected), empty_mask_pairs=sum(r['coverage'] == 0 for r in selected),
                **{key: strict_mean([r[key] for r in selected])
                   for key in ('warp_error', 'coverage', 'flow_pixels', 'raw_difference')})
    return result


def provenance():
    import cv2
    import torchvision
    import pyiqa
    hub = Path(torch.hub.get_dir())
    weights = {
        'vae': ROOT/'src/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',
        'musiq': hub/'pyiqa/musiq_koniq_ckpt-e95806b9.pth',
        'dino': hub/'checkpoints/dino_vitbase16_pretrain.pth',
        'raft': hub/'checkpoints/raft_small_C_T_V2-01064c6d.pth',
    }
    return dict(
        files_sha256={name: digest(ROOT/name) for name in (
            'tools/tt_video_quality.py', 'tools/evaluate_policy_supplement.py',
            'tools/mechanism_worker.py', 'src/utils/wan_wrapper.py',
            'src/experiments/TT_VIDEO_DEVELOPMENT.md')},
        checkpoints_sha256={name: digest(path) for name, path in weights.items()},
        dino_commit=subprocess.check_output(
            ['git', '-C', str(Path.home()/'sfq_data/eval_models/dino'), 'rev-parse', 'HEAD'],
            text=True).strip(),
        torch=torch.__version__, torchvision=torchvision.__version__,
        opencv=cv2.__version__, pyiqa=getattr(pyiqa, '__version__', 'unknown'))


def sampling_and_hashes(case, source_hash):
    folder = DATA/case['key']
    latent = folder/'latent.pt'
    sampling_path = folder/'sampling.json'
    sampling = json.loads(sampling_path.read_text())
    recorded = sampling.get('source_sha256', sampling.get('source_hash'))
    if recorded is not None and recorded != source_hash:
        raise RuntimeError(f'sampling source mismatch: {case["key"]}')
    return digest(latent), digest(sampling_path)


def reusable(case, source_hash, own_hash, reference_hash, current_provenance):
    folder = DATA/case['key']
    path = folder/'quality.json'
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    latent_hash, sampling_hash = sampling_and_hashes(case, source_hash)
    expected = dict(source_sha256=source_hash, quality_source_sha256=own_hash,
                    latent_sha256=latent_hash, sampling_sha256=sampling_hash,
                    reference_latent_sha256=reference_hash)
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError(f'refuse mixed completed evaluation: {case["key"]}, {key}')
    if not (folder/'preview.mp4').exists() or digest(folder/'preview.mp4') != row.get('preview_sha256'):
        raise RuntimeError(f'completed evaluation preview missing/changed: {case["key"]}')
    if row.get('provenance') != current_provenance:
        raise RuntimeError(f'evaluation provenance changed: {case["key"]}')
    if row.get('frames') != FRAMES:
        raise RuntimeError('completed evaluation is incomplete')
    return row


@torch.inference_mode()
def decode_pixels(vae, path):
    obj = torch.load(path, map_location='cpu', weights_only=True)
    latent = obj['latent']
    if tuple(latent.shape) != (1, 126, 16, 60, 104) or not bool(torch.isfinite(latent).all()):
        raise RuntimeError(f'invalid latent {path.name}')
    torch.cuda.synchronize()
    started = time.perf_counter()
    decoded = vae.decode_to_pixel(latent.cuda().to(torch.bfloat16), use_cache=False)
    if tuple(decoded.shape) != (1, 501, 3, 480, 832) or not bool(torch.isfinite(decoded).all()):
        raise RuntimeError('unexpected/nonfinite decoded video')
    # Match the existing evaluator's RGB8 convention; no MP4 is read for metrics.
    pixels = ((decoded[0]*.5+.5).clamp(0, 1)*255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu()
    vae.model.clear_cache()
    del decoded, latent, obj
    torch.cuda.synchronize()
    return pixels, time.perf_counter()-started


@torch.inference_mode()
def score_pixels(pixels, reference, metric, evaluator, callback):
    import cv2
    from evaluate_policy_supplement import subject_values, temporal
    images = pixels.numpy()
    musiq = []
    features = []
    started = time.perf_counter()
    for start in range(0, FRAMES, 4):
        batch = pixels[start:start+4].permute(0, 3, 1, 2).cuda().float()/255
        values = metric(batch).reshape(-1).detach().cpu().tolist()
        if len(values) != len(batch) or not all(math.isfinite(x) for x in values):
            raise RuntimeError('invalid MUSIQ output')
        musiq.extend(values)
        if start and start % 128 == 0:
            callback(f'MUSIQ {min(start+4, FRAMES)}/{FRAMES}')
    del batch
    for start in range(0, FRAMES, 16):
        features.append(evaluator.features(list(images[start:start+16])))
    features = torch.cat(features)
    dino = subject_values(features).tolist()
    if len(dino) != 500 or not all(math.isfinite(x) for x in dino):
        raise RuntimeError('invalid DINO output')
    pixel_mse = []
    # Accumulate per frame without allocating another complete float RGB video.
    for start in range(0, FRAMES, 4):
        difference = pixels[start:start+4].float()-reference[start:start+4].float()
        pixel_mse.extend((difference.square().mean((1, 2, 3))/255**2).tolist())
    pairs = transition_pairs()
    rows = []
    for start in range(0, len(pairs), 4):
        group = pairs[start:start+4]
        a = torch.from_numpy(np.stack([cv2.resize(images[r['i']], (448, 256)) for r in group])).permute(0, 3, 1, 2).cuda().float()/255
        b = torch.from_numpy(np.stack([cv2.resize(images[r['j']], (448, 256)) for r in group])).permute(0, 3, 1, 2).cuda().float()/255
        count = len(group)
        flows = evaluator.raft(torch.cat((a, b))*2-1, torch.cat((b, a))*2-1, num_flow_updates=12)[-1]
        values = temporal(a, b, flows[:count], flows[count:])
        for index, pair in enumerate(group):
            row = dict(pair)
            for key, value in values.items():
                row[key] = None if key == 'warp_error' and float(values['coverage'][index]) == 0 else float(value[index])
                if row[key] is not None and not math.isfinite(row[key]):
                    raise RuntimeError('nonfinite temporal metric outside empty mask')
            rows.append(row)
    torch.cuda.synchronize()
    return dict(windows=summarize_frames(musiq, dino, pixel_mse),
                temporal_windows=summarize_temporal(rows), per_frame_musiq=musiq,
                per_transition_dino=dino, per_frame_pixel_mse=pixel_mse, pairs=rows,
                dino_features_sha256=hashlib.sha256(features.numpy().tobytes()).hexdigest(),
                scoring_seconds=time.perf_counter()-started)


@torch.inference_mode()
def evaluate_all(source_hash, status_callback):
    """Evaluate/resume all 26 completed generation cases; callback accepts str."""
    import pyiqa
    from torchvision.io import write_video
    from utils.wan_wrapper import WanVAEWrapper
    from mechanism_worker import auxiliary_evaluator

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    own_hash = digest(__file__)
    # Check current evaluator sources and checkpoint hashes even when all cases
    # are cached. This does not instantiate any CUDA models.
    sources = provenance()
    reference_hashes = {p: digest(DATA/f'p{p}_b0_bf16/latent.pt') for p in (0, 4)}
    existing = {}
    for case in cases():
        row = reusable(case, source_hash, own_hash, reference_hashes[case['prompt']], sources)
        if row is not None:
            existing[case['key']] = row
    if len(existing) == 26:
        status_callback('QUALITY complete: 26 verified cached videos')
        return [existing[c['key']] for c in cases()]
    status_callback(f'QUALITY load offline VAE/MUSIQ/DINO/RAFT; {len(existing)}/26 cached')
    vae = WanVAEWrapper().to(device='cuda', dtype=torch.bfloat16).eval()
    metric = pyiqa.create_metric('musiq', device='cuda')
    evaluator = auxiliary_evaluator()
    records = []
    try:
        for prompt in (0, 4):
            group = cases(prompt)
            if all(c['key'] in existing for c in group):
                records.extend(existing[c['key']] for c in group)
                continue
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            reference, reference_decode_seconds = decode_pixels(vae, DATA/f'p{prompt}_b0_bf16/latent.pt')
            reference_decode_peak_allocated = torch.cuda.max_memory_allocated()
            reference_decode_peak_reserved = torch.cuda.max_memory_reserved()
            for case in group:
                key = case['key']
                if key in existing:
                    records.append(existing[key])
                    continue
                folder = DATA/key
                latent_hash, sampling_hash = sampling_and_hashes(case, source_hash)
                status_callback(f'QUALITY {len(records)+1}/26 {key}: raw decode/evaluation')
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                wall = time.perf_counter()
                if case['arm'] == 'bf16':
                    pixels = reference
                    decode_seconds = reference_decode_seconds
                else:
                    pixels, decode_seconds = decode_pixels(vae, folder/'latent.pt')
                scores = score_pixels(pixels, reference, metric, evaluator,
                                      lambda msg: status_callback(f'QUALITY {key}: {msg}'))
                preview_started = time.perf_counter()
                preview_tmp = folder/'preview.tmp.mp4'
                write_video(str(preview_tmp), pixels, fps=16, video_codec='libx264', options={'crf': '18'})
                preview_tmp.replace(folder/'preview.mp4')
                preview_seconds = time.perf_counter()-preview_started
                record = dict(**case, frames=FRAMES, source_sha256=source_hash,
                              quality_source_sha256=own_hash, latent_sha256=latent_hash,
                              sampling_sha256=sampling_hash,
                              reference_latent_sha256=reference_hashes[prompt],
                              preview_sha256=digest(folder/'preview.mp4'),
                              raw_rgb_sha256=hashlib.sha256(pixels.numpy().tobytes()).hexdigest(),
                              metric_input='raw decoded RGB8 before MP4',
                              decode_seconds=decode_seconds, preview_seconds=preview_seconds,
                              quality_seconds=time.perf_counter()-wall+(reference_decode_seconds if case['arm']=='bf16' else 0),
                              quality_peak_allocated_bytes=max(torch.cuda.max_memory_allocated(), reference_decode_peak_allocated if case['arm']=='bf16' else 0),
                              quality_peak_reserved_bytes=max(torch.cuda.max_memory_reserved(), reference_decode_peak_reserved if case['arm']=='bf16' else 0),
                              provenance=sources, **scores)
                if digest(__file__) != own_hash or digest(folder/'latent.pt') != latent_hash or digest(folder/'sampling.json') != sampling_hash:
                    raise RuntimeError('evaluation input/source changed during scoring')
                save(folder/'quality.json', record)
                records.append(record)
                status_callback(f'QUALITY {len(records)}/26 {key} complete: late MUSIQ={record["windows"]["late"]["musiq"]:.3f}')
                del pixels
            del reference
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del evaluator, metric, vae
        gc.collect()
        torch.cuda.empty_cache()
    if len(records) != 26:
        raise RuntimeError('incomplete video evaluation')
    return records
