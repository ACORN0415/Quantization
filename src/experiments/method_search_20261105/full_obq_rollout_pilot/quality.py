"""One VAE decode per completed trajectory; existing raw-frame metrics."""
import gc,json,sys,time,traceback
import pilot_common as p
import torch
import numpy as np
sys.path[:0]=[str(p.ROOT/'tools'),str(p.ROOT/'src'),str(p.ROOT/'src/scripts')]

def main():
 p.verify_lock();p.cuda_settings();torch.cuda.init();torch.cuda.reset_peak_memory_stats()
 from utils.wan_wrapper import WanVAEWrapper
 from mechanism_worker import auxiliary_evaluator
 from kv_channel_rollout import windows
 from torchvision.io import write_video
 import pyiqa
 target=p.HERE/'QUALITY.json';assert not target.exists()
 prior=json.loads((p.ROOT/'src/experiments/kv_channel_budget_rollout/quality.json').read_text())
 for info in prior['weights'].values():p.guard();assert p.sha(info['path'])==info['sha256']
 result={'status':'running','started_utc':time.time(),'rows':[],'weights':prior['weights'],'vae_decodes':0}
 p.save(target,result)
 try:
  vae=WanVAEWrapper().to(device='cuda',dtype=torch.bfloat16).eval();metric=pyiqa.create_metric('musiq',device='cuda');evaluator=auxiliary_evaluator();p.guard()
  for item in p.prompts():
   reference_features=None;reference_latent=None
   for arm in p.ARMS:
    p.guard();folder=p.HERE/item['id']/arm;sampling=json.loads((folder/'sampling.json').read_text())
    assert sampling['status']=='completed' and sampling['model_forwards']==210
    assert p.sha(sampling['latent']['path'])==sampling['latent']['sha256']
    latent=torch.load(sampling['latent']['path'],map_location='cpu',weights_only=True)['latent']
    if arm=='BF16':reference_latent=latent
    late=p.latent_metric(latent,reference_latent)
    assert late==sampling['late_latent']
    p.count('vae_decodes');result['vae_decodes']+=1;p.save(target,result)
    with torch.inference_mode():decoded=vae.decode_to_pixel(latent.cuda().bfloat16(),use_cache=False)
    assert decoded.shape==(1,501,3,480,832) and torch.isfinite(decoded).all()
    pixels=((decoded[0]*.5+.5).clamp(0,1)*255).byte().permute(0,2,3,1).contiguous().cpu()
    vae.model.clear_cache();del decoded;torch.cuda.empty_cache();p.guard()
    musiq=[];features=[]
    with torch.inference_mode():
     for start in range(0,501,4):
      p.guard();batch=pixels[start:start+4].permute(0,3,1,2).cuda().float()/255
      musiq.extend(metric(batch).reshape(-1).cpu().tolist())
     for start in range(0,501,16):
      p.guard();features.append(evaluator.features(list(pixels[start:start+16].numpy())))
    features=torch.cat(features);assert len(musiq)==501 and np.isfinite(musiq).all() and torch.isfinite(features).all()
    if arm=='BF16':reference_features=features.clone()
    win,similarity,adjacent=windows(musiq,features,reference_features)
    path=folder/'features.pt';assert not path.exists();torch.save({'features':features},path)
    preview=folder/'preview.mp4';assert not preview.exists()
    write_video(str(preview),pixels,fps=16,video_codec='libx264',options={'crf':'18'})
    result['rows'].append({'input_id':item['id'],'arm':arm,'windows':win,'late_latent':late,
     'musiq_per_frame':musiq,'dino_reference_per_frame':similarity,'dino_adjacent_per_frame':adjacent,
     'latent':sampling['latent'],'features':p.record(path),'preview':p.record(preview),
     'metric_input':'raw decoded RGB8 before MP4','sampling_seconds':sampling['sampling_seconds']})
    p.save(target,result);print(item['id'],arm,win['late'],flush=True)
    del latent,pixels,features,batch;gc.collect();torch.cuda.empty_cache();p.guard()
  assert result['vae_decodes']==8
  lookup={(r['input_id'],r['arm']):r for r in result['rows']};spread=[]
  for item in p.prompts():
   a=lookup[(item['id'],'R_fp32')];b=lookup[(item['id'],'R_bf16')]
   spread.append({'input_id':item['id'],'FP32_minus_BF16_RTN_latent_NSE':a['late_latent']['nse']-b['late_latent']['nse'],
    'FP32_minus_BF16_RTN_DINO':a['windows']['late']['dino_reference']-b['windows']['late']['dino_reference'],
    'FP32_minus_BF16_RTN_MUSIQ':a['windows']['late']['musiq']-b['windows']['late']['musiq'],
    'interpretation':'Different quantization arithmetic, not an instrument-noise estimate; no gate override'})
  result.update(status='completed',decision=p.decide(result['rows']),descriptive_RTN_spread=spread)
 except BaseException as exc:
  result.update(status='failed_or_incomplete',error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  result.update(finished_utc=time.time(),peak=p.peak());p.save(target,result)
 p.verify_lock();p.guard()
if __name__=='__main__':main()
