"""Bounded selective Q/down W8A8 rollout against the optimized packed baseline."""
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
import linear_w8a8_fused_stage1 as core
import kv_reuse_stage1 as native

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'src/experiments/linear_w8a8_rollout_stage1'
DATA=Path('/home/acorn0415/sfq_data/linear_w8a8_rollout_stage1')
START=1793803242.
DEADLINE=START+2700
SEQUENCE=[('warm_B',False),('warm_Q',True),('pair1_B',False),('pair1_Q',True),
          ('pair2_Q',True),('pair2_B',False),('pair3_B',False),('pair3_Q',True)]
core.torch=torch


def check():
    if time.time()>=DEADLINE:raise TimeoutError('registered 45-minute deadline')


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    path=OUT/'results.json'
    if path.exists():raise RuntimeError('Refusing overwrite')
    r={'status':'preparing','registered_start_utc':START,'deadline_utc':DEADLINE,
       'code_sha256':core.sha(__file__),'helper_sha256':core.sha(core.__file__),
       'native_loader_sha256':core.sha(native.__file__),
       'registration_sha256':core.sha(OUT/'REGISTRATION_KO.txt'),
       'runs':[],'model_forward_started':0,'model_forward_finished':0}
    def flush():r['updated_utc']=time.time();core.save(path,r)
    selected=[]
    try:
        check();flush()
        pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        assert all(int(p)==os.getpid() for p in pids.split())
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(18*2**30/torch.cuda.get_device_properties(0).total_memory)
        import torch._inductor.config as cfg
        for name in ('max_autotune','max_autotune_gemm','search_autotune_cache','force_fuse_int_mm_with_mul'):setattr(cfg,name,False)
        cfg.triton.cudagraphs=False
        native.OUT=OUT;native.DATA=DATA;native.setup_paths()
        frozen=json.loads((ROOT/'src/experiments/kv_reuse_stage1/manifest.json').read_text())
        r['checkpoint_hashes']={}
        for f,h in frozen['checkpoint_hashes'].items():
            actual=core.sha(ROOT/f);assert actual==h;r['checkpoint_hashes'][f]=actual
        r['upstream_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=native.SOURCE,text=True).strip()
        assert r['upstream_revision']==native.REV
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=native.SOURCE,text=True).strip()
        prompt=(ROOT/'src/experiments/policy_study_prompts.txt').read_text().splitlines()[0]
        r['prompt']=prompt;r['seed']=303;r['latent_frames']=36
        t=time.perf_counter();pipe,cond=native.build({'prompt':prompt});torch.cuda.synchronize();r['model_text_prepare_s']=time.perf_counter()-t
        import quant_videogen.uncompress as uc
        import pipeline.causal_inference as pm
        from kv_reuse_core import ReadController,make_store_class
        r['reuse_core_sha256']=core.sha(ROOT/'tools/kv_reuse_core.py')
        options={'max_autotune_gemm':True,'max_autotune_gemm_backends':'ATEN,TRITON','force_fuse_int_mm_with_mul':True,'triton.cudagraphs':False}
        compiled=torch.compile(core.w8a8_fused,backend='inductor',fullgraph=True,dynamic=False,options=options)
        r['options']=options
        t=time.perf_counter()
        for name,mod in pipe.generator.named_modules():
            if name.endswith('.self_attn.q') or name.endswith('.ffn.2'):
                assert isinstance(mod,torch.nn.Linear) and mod.out_features==1536 and mod.in_features in (1536,8960)
                qw,sw=core.quant(mod.weight);wt=qw.t().contiguous()
                def new_forward(x,wt=wt,sw=sw,bias=mod.bias):
                    assert x.ndim==3 and x.shape[0]==1 and x.shape[1]==4680 and x.dtype==torch.bfloat16
                    return compiled(x,wt,sw,bias)
                selected.append({'name':name,'module':mod,'original':mod.forward,'candidate':new_forward,'wt':wt,'sw':sw})
        assert len(selected)==60
        torch.cuda.synchronize();r['weight_prepare_s']=time.perf_counter()-t
        r['selected_modules']=[e['name'] for e in selected]
        r['extra_static_integer_weight_and_scale_bytes']=sum(e['wt'].numel()+e['sw'].numel()*4 for e in selected)
        # Compile both shapes once; weights remain dynamic tensor arguments.
        r['compile_prepare_s']={}
        for role,idx in (('q',0),('down',2)):
            cap=json.loads((ROOT/'src/experiments/linear_w8a8_stage1/results.json').read_text())['captures'][idx]
            assert core.sha(cap['path'])==cap['sha256']
            sample=torch.load(cap['path'],map_location='cpu',weights_only=True)
            x=sample['x'].cuda();e=next(e for e in selected if e['name'].endswith('.self_attn.q' if role=='q' else '.ffn.2'))
            torch.cuda.synchronize();t=time.perf_counter();y=e['candidate'](x);torch.cuda.synchronize();r['compile_prepare_s'][role]=time.perf_counter()-t
            del x,y,sample
        refs={};native_decoder=uc.triton_prq_dequantize_tensor;native_store=pm.ChunkedKVCache
        with native.Monitor() as monitor:
            for name,use_w8 in SEQUENCE:
                check();monitor.check();assert len(r['runs'])<8
                for e in selected:e['module'].forward=e['candidate'] if use_w8 else e['original']
                folder=DATA/name;folder.mkdir(exist_ok=False)
                pipe.kv_cache1=None;pipe.crossattn_cache=None;gc.collect();torch.cuda.empty_cache()
                torch.manual_seed(303);noise=torch.randn((1,36,16,60,104),device='cuda',dtype=torch.bfloat16)
                control=ReadController('B');pm.ChunkedKVCache=make_store_class(control)
                def decoder(state,block_size,num_bits,output_dtype=torch.bfloat16):
                    assert block_size==64 and num_bits==2 and output_dtype==torch.bfloat16
                    return control.decode(state)
                uc.triton_prq_dequantize_tensor=decoder
                row={'name':name,'w8a8':use_w8,'status':'running','schedule':[],'started_utc':time.time()}
                r['runs'].append(row);r['status']='running';flush();print(name,'started',flush=True)
                def before(module,args,kwargs):
                    check();monitor.check();r['model_forward_started']+=1
                    assert r['model_forward_started']<=480
                    row['schedule'].append(int(kwargs['current_start']))
                def after(module,args,out):r['model_forward_finished']+=1
                h1=pipe.generator.register_forward_pre_hook(before,with_kwargs=True);h2=pipe.generator.register_forward_hook(after)
                telemetry_start=len(monitor.rows)
                try:
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
                    _,latent=pipe.inference(noise=noise,text_prompts=[prompt],return_latents=True,profile=False,low_memory=False)
                    seconds=pipe.vae.finished-start;torch.cuda.synchronize();monitor.check()
                    row['generation_s']=seconds
                    row['peak_allocated']=torch.cuda.max_memory_allocated();row['peak_reserved']=torch.cuda.max_memory_reserved()
                    row['counts']=control.summarize()
                    assert row['schedule']==[(i//5)*4680 for i in range(60)]
                    assert control.decode_calls==1200 and control.read_calls==3660
                    cpu=latent.detach().cpu();assert torch.isfinite(cpu).all()
                    outpath=folder/'latent.pt';torch.save({'latent':cpu},outpath);row['latent']={'path':str(outpath),'sha256':core.sha(outpath)}
                    key='Q' if use_w8 else 'B'
                    if key in refs:row['repeat_bytes_equal']=core.bytes_equal(cpu,refs[key]);assert row['repeat_bytes_equal']
                    else:refs[key]=cpu;row['repeat_bytes_equal']=None
                    if use_w8:
                        d=cpu.float()-refs['B'].float();rel=float(d.double().norm()/refs['B'].double().norm())
                        row['latent_relative_l2_to_B']=rel
                        row['latent_gate']=bool(rel<=.05 and cpu.shape==refs['B'].shape and cpu.dtype==refs['B'].dtype)
                    rows=monitor.rows[telemetry_start:];core.save(folder/'telemetry.json',rows)
                    row['telemetry']={'path':str(folder/'telemetry.json'),'sha256':core.sha(folder/'telemetry.json')}
                    row['process_sample_peak']=max((t['process_bytes'] for t in rows),default=0)
                    row['status']='valid';row['finished_utc']=time.time();flush();print(name,seconds,'latent_gate',row.get('latent_gate'),flush=True)
                    if use_w8 and not row['latent_gate']:
                        r['status']='stopped_latent_gate';r['decision']='stop_A_and_move_to_B';break
                finally:
                    h1.remove();h2.remove();uc.triton_prq_dequantize_tensor=native_decoder;pm.ChunkedKVCache=native_store
                    pipe.kv_cache1=None;pipe.crossattn_cache=None
                del noise,latent
        if len(r['runs'])==8 and all(x['status']=='valid' for x in r['runs']):
            by={x['name']:x for x in r['runs']};d=[100*(1-by[f'pair{i}_Q']['generation_s']/by[f'pair{i}_B']['generation_s']) for i in (1,2,3)]
            r['paired_reduction_percent']=d;r['median_reduction_percent']=float(np.median(d))
            r['performance_passes']=bool(np.median(d)>=5 and min(d)>0)
            r['decision']='consider_separate_quality_validation' if r['performance_passes'] else 'stop_A_and_move_to_B'
            r['status']='completed'
    except BaseException as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        for e in selected:e['module'].forward=e['original']
        r['finished_utc']=time.time();flush()


if __name__=='__main__':
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(DATA/'inductor');os.environ['TRITON_CACHE_DIR']=str(DATA/'triton')
    with torch.inference_mode():main()
