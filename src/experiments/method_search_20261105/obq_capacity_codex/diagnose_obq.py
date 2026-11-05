"""CPU-only frozen-p0 Hessian -> p4 OBQ capacity diagnostic; all K heads INT4."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LIB_PATH = HERE.parent / 'fair_comparison_codex/diagnose.py'
spec = importlib.util.spec_from_file_location('fair_diagnostic_library', LIB_PATH)
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)
from headrule_quantizer import HeadRuleQuantizer, hadamard
torch.set_num_threads(4)
torch.set_grad_enabled(False)

DATA = Path('/home/acorn0415/sfq_data/kv_alllayer_capture')
FRAME, GROUP, HEADS, DIM, CHUNK, PAST = 1560, 16, 12, 128, 4680, 28080
LAYERS = (18, 24)
ARMS = ('B0_BF16_RTN', 'B0_FP32_RTN', 'OBQ_FULL', 'OBQ_DIAG')
QIDS = (torch.arange(128) * CHUNK // 128).long()
RUN_START = None


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def save(path, d):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(d, indent=2) + '\n')
    tmp.replace(path)


def guard():
    if RUN_START is not None and time.monotonic() - RUN_START > 1200:
        raise TimeoutError('registered 1200-second actual-capture limit')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 8 * 1024**3:
        raise MemoryError('registered 8-GiB max RSS limit')


def make_frames(k, hm):
    assert k.shape[0] == 1 and k.shape[1] % FRAME == 0
    kr = (k.float() @ hm).to(k.dtype)[0]
    frames = kr.reshape(-1, FRAME, HEADS, DIM)
    padded = torch.cat((frames, frames.new_zeros(len(frames), 8, HEADS, DIM)), 1)
    xg = padded.reshape(len(frames), 98, GROUP, HEADS, DIM)
    scale_native = (xg.abs().amax(2, keepdim=True) / 7).clamp_min(1e-8)
    scale_stored = scale_native.to(torch.float16)
    assert torch.equal(scale_native, scale_stored.to(k.dtype))
    return xg, scale_native, scale_stored


def rtn_codes(xg, scale, fp32):
    if fp32: return (xg.float() / scale.float()).round().clamp(-8, 7).to(torch.int8)
    return (xg / scale).round().clamp(-8, 7).to(torch.int8)


def obq_codes(xg, scale_stored, hessian, diagonal=False):
    # Groups remain fixed; all tokens are independent rows under the same per-head H.
    dims = xg.shape
    x = xg.float().reshape(-1, HEADS, DIM)
    scales = scale_stored.float().expand_as(xg).reshape(-1, HEADS, DIM)
    answer = torch.empty(x.shape, dtype=torch.int8)
    work_stats = []
    for h in range(HEADS):
        guard()
        hh = hessian[h]
        if diagonal:
            assert torch.count_nonzero(hh - torch.diag(hh.diag())) == 0
            answer[:, h] = (x[:, h] / scales[:, h]).round().clamp(-8, 7).to(torch.int8)
            work_stats.append({'head': h, 'diagonal_zero_updates_skipped': True})
            continue
        inverse = torch.linalg.inv(hh)
        inverse = (inverse + inverse.t()) * .5
        upper = torch.linalg.cholesky(inverse, upper=True).float()
        assert torch.isfinite(upper).all() and (upper.diag() > 0).all()
        w = x[:, h].clone()
        for j in range(DIM):
            code = (w[:, j] / scales[:, h, j]).round().clamp(-8, 7)
            quant = code * scales[:, h, j]
            answer[:, h, j] = code.to(torch.int8)
            if j + 1 < DIM:
                err = (w[:, j] - quant) / upper[j, j]
                w[:, j + 1:] -= err[:, None] * upper[j, j + 1:][None]
            if j % 32 == 0: guard()
        work_stats.append({'head': h, 'factor_min_diag': float(upper.diag().min()),
                           'factor_max_diag': float(upper.diag().max())})
        del w, inverse, upper
    assert int(answer.min()) >= -8 and int(answer.max()) <= 7
    return answer.reshape(dims), work_stats


def packed_restore(codes, scale_stored, hm, dtype=torch.bfloat16):
    packed = lib.pack_nibbles(codes.contiguous().view(torch.uint8))
    unpacked = lib.unpack_nibbles(packed, codes.shape).to(torch.int16)
    unpacked[unpacked >= 8] -= 16
    unpacked = unpacked.to(torch.int8)
    assert torch.equal(codes, unpacked), 'packed code bytes mismatch'
    rec = (unpacked.to(dtype) * scale_stored.to(dtype)).reshape(-1, 1568, HEADS, DIM)[:, :FRAME]
    rec = (rec.reshape(1, -1, HEADS, DIM).float() @ hm.t()).to(dtype)
    # One sentinel head byte per chunk matches PackedChunk baseline payload representation.
    sentinels = torch.full((rec.shape[1] // CHUNK,), 255, dtype=torch.uint8)
    payload = packed.numel() + scale_stored.numel() * scale_stored.element_size() + sentinels.numel()
    return rec, payload


def attn(q, k, v):
    kt, vt = k.permute(1, 2, 0).contiguous(), v.permute(1, 0, 2).contiguous()
    out = torch.empty_like(q)
    for begin in range(0, len(q), 32):
        guard()
        logits = torch.bmm(q[begin:begin+32].permute(1, 0, 2), kt) / math.sqrt(DIM)
        out[begin:begin+32] = torch.bmm(logits.softmax(-1), vt).permute(1, 0, 2)
    return out


def energy(t): return float(t.double().square().sum())


def metric(out, reference, wo, reference_energy):
    delta = (out - reference).reshape(len(out), -1) @ wo.t()
    sse = energy(delta)
    return {'sse': sse, 'nse': sse / reference_energy}


def reference_files():
    return [Path(__file__), HERE / 'REGISTRATION_KO.txt', LIB_PATH,
            ROOT / 'src/experiments/kv_headrule_rollout/headrule_quantizer.py', ROOT / 'src/kvq/quant.py']


def selftest():
    assert not torch.cuda.is_available(), 'GPU visibility must be disabled'
    gen = torch.Generator().manual_seed(20261105)
    hm = hadamard(DIM, 'cpu')
    x = torch.randn((1, CHUNK, HEADS, DIM), generator=gen).to(torch.bfloat16)
    xg, sn, ss = make_frames(x, hm)
    orig = tensor_sha(x)
    c0 = rtn_codes(xg, sn, False)
    cf = rtn_codes(xg, ss, True)
    diag = torch.diag_embed(torch.rand(HEADS, DIM, generator=gen).double() + .1)
    cd, _ = obq_codes(xg, ss, diag, diagonal=True)
    assert torch.equal(cf, cd)
    old = HeadRuleQuantizer(4, 4, k_delta=False, head_upgrade=False, device='cpu').encode_k(x)
    assert torch.equal(c0, torch.cat([v[0] for v in old.frames], 0))
    assert torch.equal(ss, torch.cat([v[1] for v in old.frames], 0))
    rec, size = packed_restore(c0, ss, hm)
    assert torch.equal(rec, old.restore())
    lib_packet = lib.PackedChunk(x, hm, 'frame')
    assert size == lib_packet.payload_bytes() and torch.equal(rec, lib_packet.restore())
    # A small complete full-H path must collapse exactly to FP32 RTN for identity H.
    tiny = xg[:1, :1]
    tiny_scale = ss[:1, :1]
    cfull, _ = obq_codes(tiny, tiny_scale, torch.eye(DIM).repeat(HEADS,1,1).double())
    assert torch.equal(cfull, rtn_codes(tiny, tiny_scale, True))
    q = torch.randn((4,HEADS,DIM),generator=gen)
    k = torch.randn((20,HEADS,DIM),generator=gen)
    v = torch.randn((20,HEADS,DIM),generator=gen)
    assert torch.equal(attn(q,k,v),attn(q,k.clone(),v.clone()))
    assert tensor_sha(x) == orig
    result = {'passed': True, 'checks': ['existing HeadRule codes/scales/restore exact',
                'PackedChunk actual payload and restoration exact','diagonal==FP32 RTN all codes',
                'full algorithm identity-H==FP32 RTN','no-quant attention exact','input preserved'],
              'payload_bytes_one_chunk': size, 'code_sha256': sha(__file__), 'time_utc': time.time()}
    save(HERE / 'SELFTEST.json', result)
    print(json.dumps(result, indent=2))


def lock():
    path = HERE / 'REGISTRATION_LOCK.json'
    if path.exists(): raise RuntimeError('Refuse overwrite registration lock')
    st = json.loads((HERE / 'SELFTEST.json').read_text())
    assert st['passed'] and st['code_sha256'] == sha(__file__)
    result = {'registered_utc': time.time(), 'sha256': {str(p): sha(p) for p in reference_files()},
              'q_indices': QIDS.tolist(), 'layers': list(LAYERS), 'calibration_prompt': 0,
              'evaluation_prompt': 4, 'actual_capture_seconds_cap': 1200}
    save(path, result)
    print(json.dumps(result, indent=2))


def load_capture(prompt, layer):
    guard()
    p = DATA / f'p{prompt:02d}_s101/l{layer:02d}_c7.pt'
    manifest = json.loads((p.parent / 'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    entries = [a for a in manifest['artifacts'] if Path(a['path']).name == p.name]
    assert len(entries) == 1
    h = sha(p)
    assert entries[0]['sha256'] == h and entries[0]['bytes'] == p.stat().st_size
    data = torch.load(p, map_location='cpu', weights_only=True)
    assert data['q'].shape == (1,CHUNK,HEADS,DIM) and data['k'].shape == (1,7*CHUNK,HEADS,DIM)
    return data, {'path':str(p),'sha256':h,'manifest_verified':True}


def run_layer(layer):
    begin = time.monotonic()
    hm = hadamard(DIM, 'cpu')
    cal, cal_src = load_capture(0, layer)
    qr = (cal['q'][0].float() @ hm).permute(1,0,2).double()
    hraw = torch.bmm(qr.transpose(1,2), qr) / CHUNK
    damp = .01 * hraw.diagonal(dim1=1,dim2=2).mean(-1)
    hd = hraw + damp[:,None,None] * torch.eye(DIM).double()[None]
    hdiag = torch.diag_embed(hd.diagonal(dim1=1,dim2=2))
    eigen = torch.linalg.eigvalsh(hraw)
    spectral = (eigen[:, -16:].sum(-1) / eigen.sum(-1)).tolist()
    del cal, qr, eigen
    gc.collect()
    calibration_seconds = time.monotonic() - begin
    data, ev_src = load_capture(4,layer)
    source_hashes = {x:tensor_sha(data[x]) for x in ('q','k','v')}
    q = data['q'][0,QIDS].float()
    k, v = data['k'], data['v']
    wo, bias = data['wo'].float(), data['bias'].float()
    reference = attn(q,k[0].float(),v[0].float())
    ref_y = reference.reshape(len(q),-1) @ wo.t() + bias
    ref_energy = energy(ref_y)
    v4, vbytes = lib.fixed_v4(v)
    konly_reference = attn(q,k[0].float(),v4)
    native = data['native_av'][0,QIDS].float()
    row = {'layer':layer,'calibration':cal_src,'evaluation':ev_src,
           'reference_energy':ref_energy,'cpu_vs_native':metric(reference,native,wo,ref_energy),
           'v_only_error':metric(konly_reference,reference,wo,ref_energy),
           'spectral_top16_descriptive_only':spectral,'damping_per_head':damp.tolist(),
           'hessian_raw_sha256':tensor_sha(hraw),'calibration_load_and_hessian_seconds':calibration_seconds,
           'arms':{}}
    xg, scale_native, scales = make_frames(k[:,:PAST],hm)
    scale_hash = tensor_sha(scales)
    cbf = rtn_codes(xg,scale_native,False)
    cf = rtn_codes(xg,scales,True)
    row['bf16_vs_fp32_rtn_code_different'] = int((cbf != cf).sum())
    expected_bytes = None
    for arm in ARMS:
        guard(); t0 = time.monotonic()
        if arm == 'B0_BF16_RTN': code, stats = cbf, []
        elif arm == 'B0_FP32_RTN': code, stats = cf, []
        elif arm == 'OBQ_FULL': code, stats = obq_codes(xg,scales,hd)
        else:
            code, stats = obq_codes(xg,scales,hdiag,diagonal=True)
            assert torch.equal(code,cf), 'diagonal compensation differs from FP32 RTN'
        quant_s = time.monotonic()-t0
        rec, kb = packed_restore(code,scales,hm)
        assert tensor_sha(scales) == scale_hash
        kfull = torch.cat((rec,k[:,PAST:]),1)[0].float()
        out = attn(q,kfull,v4)
        bytecount = kb+vbytes
        if expected_bytes is None: expected_bytes = bytecount
        assert bytecount == expected_bytes
        # Diagnostic future-query logit error, never used to choose codes or H.
        e = kfull[:PAST] - k[0,:PAST].float()
        qe = torch.bmm(q.permute(1,0,2),e.permute(1,2,0)) / math.sqrt(DIM)
        row['arms'][arm] = {**metric(out,reference,wo,ref_energy),
            'k_only_reference':metric(out,konly_reference,wo,ref_energy),
            'past_key_sse':energy(e),'logit_mse':energy(qe)/qe.numel(),
            'packed_payload_bytes':bytecount,'key_payload_bytes':kb,'value_payload_bytes':vbytes,
            'code_sha256':tensor_sha(code),'scale_sha256':scale_hash,
            'codes_different_from_B0f32':int((code!=cf).sum()),
            'quantize_seconds':quant_s,'total_arm_seconds':time.monotonic()-t0,
            'factor_stats':stats}
        print('RESULT',layer,arm,json.dumps({key:row['arms'][arm][key] for key in ('nse','logit_mse','quantize_seconds','packed_payload_bytes')}),flush=True)
        del rec,kfull,out,e,qe
        if arm == 'OBQ_FULL': del code
        gc.collect()
    assert source_hashes == {x:tensor_sha(data[x]) for x in source_hashes}
    assert row['arms']['OBQ_DIAG']['code_sha256'] == row['arms']['B0_FP32_RTN']['code_sha256']
    row['source_tensors_preserved'] = True
    row['held_raw_damped_diagonal_hessian_bytes'] = sum(t.numel()*t.element_size() for t in (hraw,hd,hdiag))
    row['one_head_factor_fp32_bytes'] = DIM*DIM*4
    row['one_head_inverse_fp64_bytes'] = DIM*DIM*8
    row['one_head_compensation_workspace_fp32_bytes'] = xg.numel()//HEADS*4
    row['shared_hadamard_bytes'] = hm.numel()*hm.element_size()
    row['seconds'] = time.monotonic()-begin
    return row


def run():
    global RUN_START
    path = HERE/'RESULT.json'
    if path.exists(): raise RuntimeError('Refuse overwrite result')
    registration = json.loads((HERE/'REGISTRATION_LOCK.json').read_text())
    assert all(sha(p)==h for p,h in registration['sha256'].items())
    RUN_START = time.monotonic()
    result = {'status':'running','started_utc':time.time(),'registration_sha256':sha(HERE/'REGISTRATION_LOCK.json'),
              'cpu_threads':torch.get_num_threads(),'cuda_visible':os.environ['CUDA_VISIBLE_DEVICES'],'rows':[]}
    save(path,result)
    try:
        for layer in LAYERS:
            result['rows'].append(run_layer(layer));save(path,result);gc.collect();guard()
        gains = [{'layer':r['layer'],**{base:1-r['arms']['OBQ_FULL']['sse']/r['arms'][base]['sse'] for base in ARMS[:2]}} for r in result['rows']]
        success = all(g[b]>=.2 for g in gains for b in ARMS[:2])
        stop = any(g[b]<=-.1 for g in gains for b in ARMS[:2])
        result.update(status='completed',gains=gains,decision='proceed_causal_capture' if success else ('stop_this_instance' if stop else 'undecided'))
    except BaseException as exc:
        result.update(status='incomplete_or_invalid',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        result.update(finished_utc=time.time(),elapsed_seconds=time.monotonic()-RUN_START,
                      max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      completed_layers=len(result['rows']))
        save(path,result)
    print(json.dumps({k:result[k] for k in ('status','decision','gains','elapsed_seconds','max_rss_bytes')},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['selftest','lock','run']);args=parser.parse_args()
    {'selftest':selftest,'lock':lock,'run':run}[args.action]()
