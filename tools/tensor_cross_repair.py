"""Post-screening development: preserve per-frame means, inspect SVD outputs."""
import csv,json,time
from pathlib import Path
import torch
import tensor_cross_probe as core
from postwo_round_core import output_error_metrics
ROOT=core.ROOT; OUT=ROOT/'src/experiments/tensor_cross_repair'

def main():
 core.old.free_gpu()
 if OUT.exists():raise RuntimeError('refuse overwrite')
 evidence=json.loads((core.OUT/'evidence.json').read_text())
 assert core.old.sha(core.OUT/'rows.csv')==evidence['rows_sha256']
 for a in evidence['input_artifacts']:
  assert core.old.sha(Path.home()/'sfq_data/postwo_rounding'/a['file'])==a['sha256']
 previous=list(csv.DictReader((core.OUT/'rows.csv').open()));rows=[];artifacts=[]
 for prompt in (0,4):
  for layer in (0,14,29):
   base=Path.home()/f'sfq_data/postwo_rounding/p{prompt:02d}_s101'
   p=torch.load(base/f'l{layer:02d}_c6.pt',map_location='cuda',weights_only=True)
   q=torch.load(base/f'l{layer:02d}_c7.pt',map_location='cuda',weights_only=True)['query']
   v=p['new_value'].float();x=v[0].reshape(3,1560,12,128).permute(2,0,1,3).contiguous()
   a0,a1=core.old.probabilities(q,p['key'])
   # SVD real time-domain factors of the Fourier rank truncation.
   a=torch.fft.rfft(x,dim=1);u,s,vh=torch.linalg.svd(a,full_matrices=False)
   for bits in (2,3,4):
    orig=next(r for r in previous if int(r['prompt'])==prompt and int(r['layer'])==layer and int(r['bits_budget'])==bits and r['arm']=='common_fourier')
    path=core.DATA/f'p{prompt:02d}_l{layer:02d}_b{bits}_common_fourier.pt'
    record=next(a for a in evidence['payload_artifacts'] if a['file']==path.name)
    assert core.old.sha(path)==record['sha256']
    state=torch.load(path,map_location='cuda',weights_only=True);restored=core.decode(state)
    correction=(v-restored).reshape(1,3,1560,12,128).mean(2,keepdim=True).half()
    corrected=(restored.reshape(1,3,1560,12,128)+correction.float()).reshape_as(v)
    r=int(orig['rank'])
    sf=dict(columns=torch.fft.irfft(u[...,:r]*s[...,:r].unsqueeze(-2),n=3,dim=1).half().contiguous(),coeff=torch.fft.irfft(vh[...,:r,:],n=3,dim=1).half().contiguous(),fourier=True)
    sy=core.decode(sf)
    for arm,y,size in [('mean_corrected',corrected,int(orig['payload_bytes'])+correction.numel()*2),('svd_fp16',sy,core.nbytes(sf))]:
     assert size<=int(orig['budget_bytes'])
     m=output_error_metrics(a0,a1,y[0],v[0],p['wo'].float())
     rows.append(dict(prompt=prompt,layer=layer,bits_budget=bits,arm=arm,rank=r,payload_bytes=size,budget_bytes=int(orig['budget_bytes']),D=m['mean_error_sq']/(32*1536),v_rel=float((y-v).norm()/v.norm())))
    corrpath=core.DATA/f'p{prompt:02d}_l{layer:02d}_b{bits}_mean.pt';torch.save(correction.cpu(),corrpath)
    artifacts.append(dict(file=corrpath.name,sha256=core.old.sha(corrpath)))
   core.status(f'repair {len(rows)}/36 p{prompt} layer{layer}')
 assert len(rows)==36
 OUT.mkdir(parents=True)
 with (OUT/'rows.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 core.old.save(OUT/'evidence.json',dict(source_sha256={s:core.old.sha(ROOT/s) for s in ('tools/tensor_cross_repair.py','tools/tensor_cross_probe.py','src/experiments/TENSOR_CROSS_REPAIR.md')},initial_evidence_sha256=core.old.sha(core.OUT/'evidence.json'),artifacts=artifacts,rows_sha256=core.old.sha(OUT/'rows.csv'),finite=all(torch.isfinite(torch.tensor(r['D'])) for r in rows)))
 core.status('COMPLETE method 1 mean repair; results saved')

if __name__=='__main__':main()
