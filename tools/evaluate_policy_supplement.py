"""Post-hoc compressed-video diagnostics. Frozen main-study code is untouched."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'src/experiments/POLICY_STUDY_SUPPLEMENT.md'
METRICS = ('subject', 'warp_error', 'coverage', 'flow_pixels', 'raw_difference')
WINDOWS = {'whole': (0, 42), 'early': (1, 7), 'middle': (7, 21), 'late': (21, 42)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)


def report(data, complete, detail, state='running', eta=None):
    record = dict(state=state, completed=complete, total=260, detail=detail, pid=os.getpid(),
                  updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), eta_hours=eta)
    save(data/'status.json', record)
    print(time.strftime('%Y-%m-%d %H:%M:%S'), state, f'{complete}/260', detail, flush=True)


def warp(image, flow):
    n, _, h, w = image.shape
    y, x = torch.meshgrid(torch.arange(h, device=image.device),
                          torch.arange(w, device=image.device), indexing='ij')
    pos = torch.stack((x, y), dim=0)[None]+flow
    valid = (pos[:, 0]>=0)&(pos[:, 0]<=w-1)&(pos[:, 1]>=0)&(pos[:, 1]<=h-1)
    grid = torch.stack((2*pos[:, 0]/(w-1)-1, 2*pos[:, 1]/(h-1)-1), dim=-1)
    return F.grid_sample(image, grid, align_corners=True), valid


def temporal(a, b, forward, backward):
    warped, inside = warp(b, forward)
    back_at, _ = warp(backward, forward)
    consistent = (forward+back_at).square().sum(1) <= .01*(forward.square().sum(1)+back_at.square().sum(1))+.5
    mask = inside & consistent
    count = mask.sum((1, 2))
    if (count==0).any(): raise RuntimeError('empty temporal correspondence mask')
    err = ((warped-a).abs().mean(1)*mask).sum((1, 2))/count
    return dict(warp_error=err, coverage=mask.float().mean((1, 2)),
                flow_pixels=forward.square().sum(1).sqrt().mean((1, 2)),
                raw_difference=(a-b).abs().mean((1, 2, 3)))


def subject_values(features):
    f = F.normalize(features, dim=-1)
    return ((f[1:]*f[:-1]).sum(-1).clamp_min(0)+(f[1:]*f[:1]).sum(-1).clamp_min(0))/2


def self_tests():
    torch.manual_seed(872341)
    a = torch.rand(1, 3, 16, 24); b = torch.roll(a, 2, -1)
    f = torch.zeros(1, 2, 16, 24); f[:, 0] = 2
    r = temporal(a, b, f, -f)
    assert r['warp_error'].item()<1e-6, r
    assert abs(r['coverage'].item()-22/24)<1e-6
    zero = torch.zeros_like(f)
    assert temporal(a, a, zero, zero)['warp_error'].item()<1e-6
    try: temporal(a, b, f+1000, -f)
    except RuntimeError: pass
    else: raise AssertionError('empty mask accepted')
    assert torch.allclose(subject_values(torch.ones(5, 4)),torch.ones(4))
    print('PASS synthetic warp direction, identity, coverage, empty-mask rejection, subject identity', flush=True)


def inputs(base):
    plan = read(ROOT/'src/experiments/policy_study_plan.json')
    assert read(base/'queue.json')['complete']==80
    items = []
    for prompt in range(plan['prompts']):
        for seed in plan['seeds']:
            items.append(dict(key=f'p{prompt:02d}_s{seed}_bf16', prompt=prompt, seed=seed,
                              bits=0, policy='reference', arm='bf16',
                              path=f'reference/p{prompt:02d}_s{seed}/bf16.mp4'))
            for bits in plan['bits']:
                for policy in plan['policies']:
                    name = f'p{prompt:02d}_s{seed}_{policy}_int{bits}'
                    assert read(base/'cases'/name/'complete.json')['source_sha256']==read(base/'queue.json')['source_sha256']
                    for arm in ('tf', 'fr', 'noise'):
                        items.append(dict(key=f'{name}_{arm}',prompt=prompt,seed=seed,bits=bits,
                                          policy=policy,arm=arm,path=f'cases/{name}/{arm}.mp4'))
    assert len(items)==260 and len({v['path'] for v in items})==260
    return items


class Evaluator:
    def __init__(self):
        torch.set_num_threads(4); cv2.setNumThreads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.dino = torch.hub.load(str(Path.home()/'sfq_data/eval_models/dino'),
                                  'dino_vitb16', source='local').cuda().eval()
        self.raft = raft_small(weights=Raft_Small_Weights.C_T_V2).cuda().eval()
        self.resize = Resize(224, antialias=False)
        self.mean = torch.tensor([.485,.456,.406], device='cuda')[None,:,None,None]
        self.std = torch.tensor([.229,.224,.225], device='cuda')[None,:,None,None]

    @torch.inference_mode()
    def features(self, frames):
        x = torch.from_numpy(np.stack(frames)).permute(0,3,1,2).cuda().float()
        x = self.resize(x).float()/255
        return self.dino((x-self.mean)/self.std).cpu()

    @torch.inference_mode()
    def video(self, path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened(): raise RuntimeError(f'cannot open {path}')
        meta = [int(cap.get(v)) for v in (cv2.CAP_PROP_FRAME_COUNT,cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT)]
        if meta != [501,832,480] or abs(cap.get(cv2.CAP_PROP_FPS)-16)>1e-6:
            raise RuntimeError(f'unexpected video geometry {meta}')
        needed = {12*c+offset for c in range(42) for offset in (2,3,4)}
        small = {}; batch = []; feats = []; index = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok: break
                rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                batch.append(rgb)
                if index in needed:
                    small[index] = cv2.resize(rgb,(448,256),interpolation=cv2.INTER_LINEAR)
                if len(batch)==16:
                    feats.append(self.features(batch)); batch.clear()
                index+=1
            if batch: feats.append(self.features(batch))
        finally: cap.release()
        if index != 501: raise RuntimeError(f'truncated decode {index}')
        features = torch.cat(feats); subjects = subject_values(features).numpy()
        pairs = [(c,12*c+o,12*c+o+1) for c in range(42) for o in (2,3)]
        rows = []
        for start in range(0,len(pairs),4):
            group = pairs[start:start+4]
            a = torch.from_numpy(np.stack([small[i] for _,i,j in group])).permute(0,3,1,2).cuda().float()/255
            b = torch.from_numpy(np.stack([small[j] for _,i,j in group])).permute(0,3,1,2).cuda().float()/255
            n = len(group)
            flows = self.raft(torch.cat((a,b))*2-1,torch.cat((b,a))*2-1,num_flow_updates=12)[-1]
            result = temporal(a,b,flows[:n],flows[n:])
            for z,(c,i,j) in enumerate(group):
                row = dict(chunk=c,first_frame=i,last_frame=j)
                row.update({k:float(v[z]) for k,v in result.items()}); rows.append(row)
        chunks = []
        for c in range(42):
            lo = max(1,0 if c==0 else 12*c-3); hi = 12*c+9
            row = dict(chunk=c,subject=float(subjects[lo-1:hi-1].mean()),subject_transitions=hi-lo)
            for metric in METRICS[1:]:row[metric]=float(np.mean([r[metric] for r in rows if r['chunk']==c]))
            chunks.append(row)
        windows = {}
        for name,(lo,hi) in WINDOWS.items():
            window = [c for c in chunks if lo<=c['chunk']<hi]
            windows[name] = {k:float(np.average([c[k] for c in window],
                              weights=[c['subject_transitions'] if k=='subject' else 1 for c in window])) for k in METRICS}
        if not np.isfinite([v for r in windows.values() for v in r.values()]).all():
            raise RuntimeError('nonfinite video metric')
        return dict(frames=index,windows=windows,chunks=chunks,pairs=rows,
                    feature_sha256=hashlib.sha256(features.numpy().tobytes()).hexdigest())


def provenance():
    import torchvision
    hub = Path(torch.hub.get_dir())/'checkpoints'
    return dict(evaluator_sha256=digest(__file__),protocol_sha256=digest(PROTOCOL),
                dino_commit=subprocess.check_output(['git','-C',str(Path.home()/'sfq_data/eval_models/dino'),'rev-parse','HEAD'],text=True).strip(),
                checkpoints={n:digest(hub/n) for n in ('dino_vitbase16_pretrain.pth','raft_small_C_T_V2-01064c6d.pth')},
                torch=torch.__version__,torchvision=torchvision.__version__,opencv=cv2.__version__)


def analyze(data, out):
    records = [read(data/'videos'/f'{v["key"]}.json') for v in inputs(data.parent/'policy_study')]
    assert len(records)==260
    gate = read(data/'gates.json')
    for r in records:
        if r['provenance']!=gate['provenance']:raise RuntimeError('analysis source mismatch')
        if digest(data.parent/'policy_study'/r['path'])!=r['video_sha256']:raise RuntimeError('video changed')
    out.mkdir(parents=True,exist_ok=True)
    rows = []
    for r in records:
        for window,value in r['windows'].items():
            rows.append({k:r[k] for k in ('key','prompt','seed','bits','policy','arm')}|dict(window=window)|value)
    with (out/'windows.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    contrasts=[]
    for bits in (4,2):
        for arm in ('fr','tf','noise'):
            for window in WINDOWS:
                selected=[r for r in rows if r['bits']==bits and r['arm']==arm and r['window']==window]
                for metric in METRICS:
                    diffs=[]
                    for prompt in range(10):
                        byseed=[]
                        for seed in (101,202):
                            a={r['policy']:r[metric] for r in selected if r['prompt']==prompt and r['seed']==seed}
                            byseed.append(a['chunk']-a['rolling'])
                        diffs.append(float(np.mean(byseed)))
                    rng=np.random.default_rng(872341)
                    boot=np.asarray(diffs)[rng.integers(0,10,(20000,10))].mean(1)
                    lo,hi=np.quantile(boot,[.025,.975])
                    contrasts.append(dict(bits=bits,arm=arm,window=window,metric=metric,mean=float(np.mean(diffs)),
                                          ci_low=float(lo),ci_high=float(hi),n_prompts=10,prompt_differences=diffs))
    save(out/'summary.json',dict(posthoc=True,provenance=gate['provenance'],videos=260,contrasts=contrasts,
         artifacts=[dict(path=f'videos/{r["key"]}.json',sha256=digest(data/'videos'/f'{r["key"]}.json'),video=r['path'],video_sha256=r['video_sha256']) for r in records]))
    lines=['# 주체·시간 진단 추가 평가 (사후 탐색 분석)','',
      '기존 MP4 260개를 평가했다. 새 생성 실험 없이 전체 결과를 포함했다.',
      '주지표 MUSIQ 사전 등록은 유지하며, 아래 구간은 결과를 본 뒤 추가한 탐색적 95% 프롬프트 부트스트랩 구간이다.',
      '', '| 비트 | 지표 | P3−P1 FR 후반 차이 | 탐색적 95% 구간 |','|---|---|---:|---|']
    for r in contrasts:
        if r['arm']=='fr' and r['window']=='late':
            lines.append(f'| {r["bits"]} | {r["metric"]} | {r["mean"]:+.6f} | [{r["ci_low"]:+.6f}, {r["ci_high"]:+.6f}] |')
    lines += ['', 'subject는 높을수록 프레임 간 특징이 비슷하다. 전체 프레임 특징이며 객체 추적 지표가 아니다.',
      'warp_error는 낮을수록 움직임 보정 후 잔차가 작다. flow_pixels(움직임)와 coverage(유효 영역)를 함께 해석해야 한다.',
      '정지·동작 소실·부정확한 optical flow도 점수에 영향을 준다. raw_difference는 단순 픽셀 차이이며 품질 점수가 아니다.',
      'TF는 이어 붙인 반사실적 영상이므로 FR과 같은 의미의 시간 일관성으로 해석하지 않는다.',
      '본 지표는 압축 MP4를 사용하므로 원본 프레임 MUSIQ와 입력 단계가 다르다.',
      '', '[프로토콜·출처](../POLICY_STUDY_SUPPLEMENT.md) · [전체 구간](windows.csv) · [집계와 무결성](summary.json)',
      '공식 VBench 벤치마크 실행이나 새 방법의 입증을 의미하지 않는다.']
    (out/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','run','analyze'],required=True)
    ap.add_argument('--data',type=Path,default=Path.home()/'sfq_data/policy_supplement')
    ap.add_argument('--smoke-video',type=Path)
    args=ap.parse_args(); data=args.data;data.mkdir(parents=True,exist_ok=True)
    if args.mode=='analyze':analyze(data,ROOT/'src/experiments/policy_supplement_results');return
    self_tests(); evaluator=Evaluator(); prov=provenance()
    if args.mode=='smoke':
        cap=cv2.VideoCapture(str(args.smoke_video));frames=[]
        for _ in range(2):
            ok,frame=cap.read()
            if not ok:raise RuntimeError('short smoke video')
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        cap.release()
        batch=evaluator.features(frames)
        single=torch.cat([evaluator.features([frame]) for frame in frames])
        if not torch.allclose(batch,single,atol=1e-5,rtol=1e-4):raise RuntimeError('DINO batching gate failed')
        start=time.monotonic();value=evaluator.video(args.smoke_video)
        save(data/'gates.json',dict(passed=True,provenance=prov,seconds=time.monotonic()-start,
                                  dino_batch_max_abs_diff=float((batch-single).abs().max()),
                                  smoke_video_sha256=digest(args.smoke_video),windows=value['windows']))
        print('GATES PASS',read(data/'gates.json')['seconds'],flush=True);return
    gate=read(data/'gates.json')
    if not gate['passed'] or gate['provenance']!=prov:raise RuntimeError('gate mismatch')
    items=inputs(data.parent/'policy_study');elapsed=[]
    for i,item in enumerate(items):
        path=data.parent/'policy_study'/item['path']; sha=digest(path); result=data/'videos'/f'{item["key"]}.json'
        report(data,i,item['key'],eta=(260-i)*float(np.median(elapsed))/3600 if elapsed else None)
        if result.exists():
            old=read(result)
            if old['provenance']!=prov or old['video_sha256']!=sha:raise RuntimeError('resume mismatch')
            elapsed.append(old['seconds']);continue
        start=time.monotonic();value=evaluator.video(path)
        if provenance()!=prov or digest(path)!=sha:raise RuntimeError('source/input changed')
        value.update(item,provenance=prov,video_sha256=sha,seconds=time.monotonic()-start)
        save(result,value);elapsed.append(value['seconds'])
    report(data,260,'전체 분석 및 보고서 작성')
    analyze(data,ROOT/'src/experiments/policy_supplement_results')
    report(data,260,'추가 평가·분석 완료',state='pass',eta=0)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        import sys
        # Preserve traceback and a visible failure state; never silently drop a video.
        if '--data' in sys.argv: directory=Path(sys.argv[sys.argv.index('--data')+1])
        else: directory=Path.home()/'sfq_data/policy_supplement'
        count=read(directory/'status.json').get('completed',0) if (directory/'status.json').exists() else 0
        report(directory,count,f'{type(exc).__name__}: {exc}',state='failed')
        raise
