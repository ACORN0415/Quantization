"""Preflight full-size decode and raw evaluators before the long queue."""
import sys,json,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/scripts')]
os.chdir(ROOT/'src')
import torch
import pyiqa
from utils.wan_wrapper import WanVAEWrapper
from mechanism_worker import auxiliary_evaluator
from tt_video_quality import decode_pixels,score_pixels,save,provenance
from postwo_round_probe import free_gpu
free_gpu();torch.set_grad_enabled(False);torch.set_num_threads(4)
vae=WanVAEWrapper().to(device='cuda',dtype=torch.bfloat16).eval()
metric=pyiqa.create_metric('musiq',device='cuda');evaluator=auxiliary_evaluator()
torch.cuda.reset_peak_memory_stats();start=time.monotonic()
reference=Path.home()/'sfq_data/policy_study/reference/p00_s101/bf16.pt'
pixels,seconds=decode_pixels(vae,reference)
result=score_pixels(pixels,pixels,metric,evaluator,lambda msg:print(msg,flush=True))
assert len(result['per_frame_musiq'])==501 and len(result['pairs'])==125
assert all(v==0 for v in result['per_frame_pixel_mse'])
save(Path.home()/'sfq_data/tt_video_quality_preflight.json',dict(decode_seconds=seconds,total_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),windows=result['windows'],temporal_windows=result['temporal_windows'],provenance=provenance()))
print('PASS full 501-frame raw quality pipeline',flush=True)
