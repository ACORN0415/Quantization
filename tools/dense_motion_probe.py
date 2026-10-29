"""Separate exploratory local-KV-correspondence screen; frozen prior result untouched."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import torch
from motion_codec_core import make_anchor
from dense_motion_core import AffineStats, choose_map, gather_reference, map_bytes

DATA=Path.home()/'sfq_data/dense_motion'
INPUT=Path.home()/'sfq_data/motion_codec'
OUT=ROOT/'src/experiments/dense_motion_development'
LAYERS=(0,14,29)
ARMS=('direct','same_position','affine_same','dense_same','dense_affine')
FILES=('tools/dense_motion_core.py','tools/test_dense_motion_core.py','tools/dense_motion_probe.py',
       'tools/motion_codec_core.py','src/experiments/DENSE_MOTION_DEVELOPMENT.md')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def fingerprint():
    h=hashlib.sha256()
    for rel in FILES:h.update(rel.encode());h.update((ROOT/rel).read_bytes())
    return h.hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)


def status(detail,state='running'):
    print(time.strftime('%Y-%m-%d %H:%M:%S'),state,detail,flush=True)
    save(DATA/'status.json',dict(state=state,detail=detail,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


def sync():torch.cuda.synchronize()


def timed(fn):
    value=fn();sync()  # one warm-up
    durations=[]
    for _ in range(3):
        sync();started=time.perf_counter();value=fn();sync();durations.append(time.perf_counter()-started)
    return value,statistics.median(durations)


def relative(x,y):return float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-12))


def samples(prompt,chunk):
    return {layer:torch.load(INPUT/f'p{prompt:02d}_s101/c{chunk:02d}_l{layer:02d}.pt',
                            map_location='cuda',weights_only=True) for layer in LAYERS}


def prepare(sample,pair):
    n=sample['grid'][1]*sample['grid'][2]
    result={}
    for kind in ('k','v'):
        raw=sample[kind][:,pair*n:(pair+1)*n]
        target=sample[kind][:,(pair+1)*n:(pair+2)*n]
        anchor=make_anchor(raw)
        result[kind]=dict(anchor=anchor,reference=anchor.restore(),target=target)
    return result


def main():
    from wan.modules.causal_model import causal_rope_apply
    started=time.monotonic();source=fingerprint()
    if OUT.exists():raise RuntimeError('refuse overwrite existing numerical results')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for fixed benchmark')
    import subprocess
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if any(s.strip().isdigit() and int(s.strip())!=os.getpid() for s in occupied.splitlines()):raise RuntimeError('GPU occupied')
    manifests=[];rows=[];map_records=[];fit_records=[];timings=[];identity_checks=0
    for prompt in (0,4):
        manifest=json.loads((INPUT/f'p{prompt:02d}_s101/manifest.json').read_text())
        for item in manifest['artifacts']:
            if sha(INPUT/item['file'])!=item['sha256']:raise RuntimeError('input SHA mismatch')
        manifests.append(manifest)
        status(f'p{prompt}: training coefficients on chunks0/1')
        stats={(layer,kind):AffineStats() for layer in LAYERS for kind in ('k','v')}
        sync();fit_start=time.perf_counter()
        for chunk in (0,1):
            batch=samples(prompt,chunk)
            for pair in (0,1):
                for layer in LAYERS:
                    frame=prepare(batch[layer],pair)
                    for kind,item in frame.items():stats[layer,kind].update(item['reference'],item['target'])
        coefficients={key:value.fit(ridge=1e-4) for key,value in stats.items()}
        sync();fit_seconds=time.perf_counter()-fit_start
        coeff_path=DATA/f'p{prompt:02d}_coefficients.pt';coeff_path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({f'{l}_{k}':dict(a=c.a.cpu(),b=c.b.cpu(),training_tokens=c.training_tokens,ridge=c.ridge)
                    for (l,k),c in coefficients.items()},coeff_path)
        fit_records.append(dict(prompt=prompt,training_chunks=[0,1],evaluation_chunks=[6,7],fit_seconds=fit_seconds,
            coefficients_bytes=sum(c.bytes()['resident'] for c in coefficients.values()),
            coefficients_file=coeff_path.name,coefficients_sha256=sha(coeff_path)))
        del stats,batch,frame
        for chunk in (6,7):
            batch=samples(prompt,chunk);_,height,width=batch[0]['grid']
            for pair in (0,1):
                frames,reference_s=timed(lambda:{layer:prepare(batch[layer],pair) for layer in LAYERS})
                offsets,map_s=timed(lambda:choose_map(frames[0]['k']['target'],frames[0]['k']['reference'],height,width,radius=2))
                encoded_offsets=offsets.cpu()
                map_path=DATA/f'p{prompt:02d}_c{chunk}_pair{pair}_map.pt';torch.save(encoded_offsets,map_path)
                counts={}
                for dy,dx in encoded_offsets.reshape(-1,2).tolist():
                    key=f'{dy},{dx}';counts[key]=counts.get(key,0)+1
                nonzero=float((offsets!=0).any(-1).float().mean())
                map_records.append(dict(prompt=prompt,chunk=chunk,pair=pair,nonzero_fraction=nonzero,
                    map_bytes=map_bytes(offsets),shift_counts=counts,file=map_path.name,sha256=sha(map_path),
                    search_seconds_median3=map_s,reference_encode_restore_seconds_median3=reference_s))
                for arm in ARMS:
                    arm_rows=[]
                    for layer in LAYERS:
                        def prediction(kind):
                            ref=frames[layer][kind]['reference']
                            if arm.startswith('dense_'):ref=gather_reference(ref,offsets,height,width)
                            if arm in ('affine_same','dense_affine'):return coefficients[layer,kind].predict(ref)
                            return ref.float()
                        def encode():
                            result={}
                            for kind,item in frames[layer].items():
                                x=item['target'] if arm=='direct' else item['target'].float()-prediction(kind)
                                result[kind]=make_anchor(x)
                            return result
                        codes,encode_s=timed(encode)
                        def decode():
                            result={}
                            for kind,code in codes.items():
                                x=code.restore() if arm=='direct' else prediction(kind)+code.restore()
                                result[kind]=x.to(frames[layer][kind]['target'].dtype)
                            return result
                        restored,decode_s=timed(decode)
                        # Decoder requires only persisted codes/coefficients/map and reconstructed anchor.
                        errors={kind+'_rel':relative(restored[kind],frames[layer][kind]['target']) for kind in ('k','v')}
                        grid=torch.tensor([[1,height,width]],device='cuda');position=batch[layer]['start_frame']+pair+1
                        actual=causal_rope_apply(restored['k'],grid,batch[layer]['freqs'],start_frame=position)
                        target=causal_rope_apply(frames[layer]['k']['target'],grid,batch[layer]['freqs'],start_frame=position)
                        errors['k_post_rope_rel']=relative(actual,target)
                        if arm.startswith('dense_'):
                            for kind,item in frames[layer].items():
                                identity_ref=gather_reference(item['reference'],torch.zeros_like(offsets),height,width)
                                plain=item['reference']
                                if arm=='dense_affine':
                                    identity_pred=coefficients[layer,kind].predict(identity_ref)
                                    plain_pred=coefficients[layer,kind].predict(plain)
                                else:
                                    identity_pred=identity_ref.float();plain_pred=plain.float()
                                recovered=(identity_pred+make_anchor(item['target'].float()-identity_pred).restore()).to(item['target'].dtype)
                                baseline=(plain_pred+make_anchor(item['target'].float()-plain_pred).restore()).to(item['target'].dtype)
                                if not torch.equal(recovered,baseline):raise RuntimeError('decoded identity map gate failed')
                                identity_checks+=1
                        payloads=[item['anchor'].bytes() for item in frames[layer].values()]+[c.bytes() for c in codes.values()]
                        resident=sum(v['resident'] for v in payloads);packed=sum(v['packed_estimate'] for v in payloads)
                        coeff_bytes=0
                        if arm in ('affine_same','dense_affine'):
                            coeff_bytes=sum(coefficients[layer,k].bytes()['resident'] for k in ('k','v'))
                            resident+=coeff_bytes;packed+=coeff_bytes
                        arm_rows.append(dict(prompt=prompt,chunk=chunk,pair=pair,layer=layer,arm=arm,**errors,
                            error_mean=(errors['k_rel']+errors['v_rel'])/2,
                            layer_pair_resident_without_shared_map=resident,layer_pair_packed_without_shared_map=packed,
                            coefficients_bytes=coeff_bytes,reference_workspace_bytes=sum(item['reference'].numel()*item['reference'].element_size() for item in frames[layer].values()),
                            encode_seconds_median3=encode_s,decode_seconds_median3=decode_s))
                    rows.extend(arm_rows)
                    timings.append(dict(prompt=prompt,chunk=chunk,pair=pair,arm=arm,
                        resident_payload_bytes=sum(r['layer_pair_resident_without_shared_map'] for r in arm_rows)+(map_bytes(offsets) if arm.startswith('dense_') else 0),
                        packed_estimate_bytes=sum(r['layer_pair_packed_without_shared_map'] for r in arm_rows)+(map_bytes(offsets) if arm.startswith('dense_') else 0),
                        reference_seconds=reference_s,search_seconds=map_s if arm.startswith('dense_') else 0,
                        encode_seconds=sum(r['encode_seconds_median3'] for r in arm_rows),
                        decode_seconds=sum(r['decode_seconds_median3'] for r in arm_rows)))
                status(f'p{prompt} chunk{chunk} pair{pair}: {len(rows)}/120 rows')
    if len(rows)!=120 or fingerprint()!=source:raise RuntimeError('incomplete or source changed')
    if any(not math.isfinite(float(r[k])) for r in rows for k in ('k_rel','v_rel','k_post_rope_rel')):raise RuntimeError('nonfinite metrics')
    decisions=[]
    for prompt in (0,4):
        means={arm:statistics.mean(r['error_mean'] for r in rows if r['prompt']==prompt and r['arm']==arm) for arm in ARMS}
        improvement=1-means['dense_affine']/means['affine_same']
        baseline=[r['resident_payload_bytes'] for r in timings if r['prompt']==prompt and r['arm']=='affine_same']
        dense=[r['resident_payload_bytes'] for r in timings if r['prompt']==prompt and r['arm']=='dense_affine']
        overhead=max(a/b-1 for a,b in zip(dense,baseline))
        decisions.append(dict(prompt=prompt,means=means,relative_error_reduction=improvement,
                              payload_overhead=overhead,passes=improvement>=.05 and overhead<=.01))
    OUT.mkdir(parents=True)
    for name,values in (('rows',rows),('costs',timings)):
        with (OUT/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    save(OUT/'decision.json',dict(source_sha256=source,advance=all(d['passes'] for d in decisions),decisions=decisions,
        identity_decode_checks=identity_checks,maps=map_records,fits=fit_records,captures=manifests,wall_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        scope='exploratory reused development data; current-K-selected correspondence is not physical optical flow; not full TempoKV',
        timing_scope='1 warmup + 3 repeats median per stage; reference stage includes encode+restore, codec full-model inference not measured'))
    status('COMPLETE '+('GO' if all(d['passes'] for d in decisions) else 'NO-GO'),'pass')
    print(json.dumps(decisions,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    try:main()
    except Exception as exc:status(str(exc),'failed');raise
