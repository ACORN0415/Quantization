"""Bounded native QVG generation/read-reuse diagnostic (12 runs / 120 min)."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
DATA=Path.home()/'sfq_data/kv_reuse_stage1'
OUT=ROOT/'src/experiments/kv_reuse_stage1'
SOURCE=Path.home()/'sfq_data/qvg-reuse-reference'
DEPS=Path.home()/'sfq_data/qvg-reuse-deps'
REV='0601468f2dbba6a17ac7086faec6d41527cad188'
SEQUENCE=[('gate_U','U','verify'),('gate_B','B','verify'),('gate_R','R','verify'),
          ('profile_B','B','profile'),('warm_B','B','warmup'),('warm_R','R','warmup'),
          ('pair1_B','B','timed'),('pair1_R','R','timed'),
          ('pair2_R','R','timed'),('pair2_B','B','timed'),
          ('pair3_B','B','timed'),('pair3_R','R','timed')]
FILES=['tools/kv_reuse_core.py','tools/kv_reuse_tape.py','tools/test_kv_reuse_core.py',
       'tools/kv_reuse_stage1.py','src/experiments/kv_reuse_preflight/MEASUREMENT_PLAN_KO.md',
       'src/experiments/kv_reuse_stage1/REGISTRATION_KO.md']


def stamp():return datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(p)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()
def emit(s,**kw):
    print(stamp(),s,flush=True)
    save(DATA/'status.json',dict(updated=stamp(),message=s,**kw))


class Monitor:
    def __init__(self):
        import pynvml as nv
        self.nv=nv;nv.nvmlInit();self.handle=nv.nvmlDeviceGetHandleByIndex(0)
        self.stop=threading.Event();self.rows=[];self.error=None
        self.thread=threading.Thread(target=self.loop,daemon=True)
    def loop(self):
        while not self.stop.is_set():
            try:
                nv=self.nv;h=self.handle
                ps=nv.nvmlDeviceGetComputeRunningProcesses(h)
                own=sum(p.usedGpuMemory for p in ps if p.pid==os.getpid())
                external=[p.pid for p in ps if p.pid!=os.getpid()]
                row=dict(unix=time.time(),process_bytes=own,external_pids=external,
                         temp_c=nv.nvmlDeviceGetTemperature(h,nv.NVML_TEMPERATURE_GPU),
                         sm_mhz=nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_SM),
                         memory_mhz=nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_MEM),
                         power_mw=nv.nvmlDeviceGetPowerUsage(h))
                self.rows.append(row)
                if external:self.error=f'external GPU pids: {external}'
                if own>20*2**30:self.error=f'20GiB process cap exceeded: {own}'
                if row['temp_c']>=83:self.error=f'temperature >=83C: {row["temp_c"]}'
            except Exception as exc:self.error='NVML monitor failed: '+repr(exc)
            self.stop.wait(.5)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*_):self.stop.set();self.thread.join(timeout=2)
    def check(self):
        if self.error:raise RuntimeError(self.error)


def setup_paths():
    sys.path[:0]=[str(ROOT/'tools'),str(DEPS),str(SOURCE),str(SOURCE/'experiments/Self-Forcing')]
    work=DATA/'work';(work/'ckpts/Self-Forcing').mkdir(parents=True,exist_ok=True)
    link=work/'ckpts/Self-Forcing/Wan2.1-T2V-1.3B'
    target=Path.home()/'sfq_data/ckpt/wan_models/Wan2.1-T2V-1.3B'
    if not link.exists():link.symlink_to(target,target_is_directory=True)
    os.chdir(work)
    os.environ.update(TIME_BENCH='0',DUMP_KV_LEVEL='0',TOKENIZERS_PARALLELISM='false')


def freeze():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==REV
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip()
    own={p:sha(ROOT/p) for p in FILES}
    upstream={str(p.relative_to(SOURCE)):sha(p) for base in ('quant_videogen','experiments/Self-Forcing')
              for p in sorted((SOURCE/base).rglob('*.py'))}
    checkpoints={name:sha(ROOT/name) for name in ('src/checkpoints/self_forcing_dmd.pt',
      'src/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth',
      'src/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors')}
    prompt_file=ROOT/'src/experiments/policy_study_prompts.txt'
    return dict(created=stamp(),own_files=own,upstream_files=upstream,revision=REV,
                checkpoint_hashes=checkpoints,prompt_file=str(prompt_file),prompt_file_sha256=sha(prompt_file),
                prompt=prompt_file.read_text().splitlines()[0],seed=303,latent_frames=36,
                source=str(SOURCE),sequence=SEQUENCE,process_limit_bytes=20*2**30,
                allocator_limit_bytes=18*2**30,retention_pool_bytes=2**30)


def unchanged(manifest):
    for p,h in manifest['own_files'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('frozen measurement code changed: '+p)
    for p,h in manifest['upstream_files'].items():
        if sha(SOURCE/p)!=h:raise RuntimeError('upstream changed: '+p)


def build(manifest):
    import torch
    from omegaconf import OmegaConf
    from pipeline.causal_inference import CausalInferencePipeline
    from utils.wan_wrapper import WanDiffusionWrapper,WanTextEncoder
    from demo_utils.memory import DynamicSwapInstaller
    torch.manual_seed(303)
    config=OmegaConf.merge(OmegaConf.load(SOURCE/'experiments/Self-Forcing/configs/default_config.yaml'),
                          OmegaConf.load(SOURCE/'experiments/Self-Forcing/configs/self_forcing_dmd.yaml'))
    config.model_kwargs.local_attn_size=180
    config.quant_config=dict(quant_type='triton-nstages-kmeans-int2',quant_block_size=64,
                             num_prq_stages=1,kmeans_max_iters=2,
                             cache_num_k_centroids=256,cache_num_v_centroids=256)
    encoder=WanTextEncoder().to(dtype=torch.bfloat16)
    DynamicSwapInstaller.install_model(encoder,device=torch.device('cuda'))
    cond=encoder(text_prompts=[manifest['prompt']])
    DynamicSwapInstaller.uninstall_model(encoder)
    encoder.to('cpu');del encoder;gc.collect();torch.cuda.empty_cache()
    class FixedText(torch.nn.Module):
        def forward(self,text_prompts):return cond
    class NoVAE(torch.nn.Module):
        def __init__(self):super().__init__();self.finished=0.
        def decode_to_pixel(self,output,use_cache=False):
            torch.cuda.synchronize();self.finished=time.monotonic()
            return torch.zeros((1,),device=output.device,dtype=output.dtype)
    generator=WanDiffusionWrapper(**config.model_kwargs,is_causal=True)
    weights=torch.load(ROOT/'src/checkpoints/self_forcing_dmd.pt',map_location='cpu',weights_only=False)
    generator.load_state_dict(weights['generator_ema']);del weights
    generator=generator.to(device='cuda',dtype=torch.bfloat16).eval()
    pipe=CausalInferencePipeline(config,device=torch.device('cuda'),generator=generator,
                                 text_encoder=FixedText(),vae=NoVAE())
    # Disable the measurement-only offload/reload routine in all three arms.
    pipe._print_memory_usage=lambda *_:None
    assert pipe.num_frame_per_block==3 and pipe.frame_seq_length==1560
    assert len(pipe.denoising_step_list)==4 and not pipe.independent_first_frame
    save(OUT/'model_setup.json',dict(config=OmegaConf.to_container(config,resolve=True),
        denoising_steps=pipe.denoising_step_list.tolist(),blocks=pipe.num_transformer_blocks,
        generation_boundary='before pool/cache allocation to before VAE (synchronized)',
        common_changes=['precomputed fixed text','VAE excluded','diagnostic offload/reload disabled']))
    return pipe,cond


def run_case(pipe,manifest,name,arm,mode,monitor):
    import torch
    import quant_videogen.uncompress as uc
    import pipeline.causal_inference as pm
    import wan.modules.causal_model as cm
    from kv_reuse_core import ReadController,ReusePool,make_store_class
    from kv_reuse_tape import InputTape
    folder=DATA/name;folder.mkdir(exist_ok=True)
    tape=None
    if mode=='verify':tape=InputTape(DATA/'attention_tape','write' if arm=='U' else 'read')
    # Each run begins with empty native stores and an empty retention pool.
    pipe.kv_cache1=None;pipe.crossattn_cache=None
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(303)
    noise=torch.randn((1,36,16,60,104),device='cuda',dtype=torch.bfloat16)
    control=ReadController(arm,profile=mode=='profile')
    position={'forward':-1,'layer':None,'start':None}
    schedule=[];handles=[]
    native_attention=cm.attention
    native_decoder=uc.triton_prq_dequantize_tensor
    native_store=pm.ChunkedKVCache
    quantize=pipe.quantize_kv_cache
    def quant_wrapper(*a,**kw):
        with control.event('encode_total'):return quantize(*a,**kw)
    pipe.quantize_kv_cache=quant_wrapper
    def decoder(state,block_size,num_bits,output_dtype=torch.bfloat16):
        assert block_size==64 and num_bits==2 and output_dtype==torch.bfloat16
        return control.decode(state)
    uc.triton_prq_dequantize_tensor=decoder
    def pre_forward(module,args,kwargs):
        monitor.check();position['forward']+=1
        position['start']=int(kwargs['current_start'])
        schedule.append(dict(forward=position['forward'],start=position['start']))
        if position['forward']%5==0:
            emit(f'{name}: chunk {position["forward"]//5+1}/12',case=name,phase=mode)
    handles.append(pipe.generator.register_forward_pre_hook(pre_forward,with_kwargs=True))
    for layer,block in enumerate(pipe.generator.model.blocks):
        def before(module,args,l=layer):position['layer']=l
        handles.append(block.self_attn.register_forward_pre_hook(before))
    def attention(q,k,v,*a,**kw):
        if tape:
            for kind,value in zip(('q','k','v'),(q,k,v)):
                tape.tensor(dict(forward=position['forward'],layer=position['layer'],
                                 start=position['start'],tensor=kind),value)
        with control.event('self_attention'):
            return native_attention(q,k,v,*a,**kw)
    cm.attention=attention
    samples_start=len(monitor.rows)
    try:
        torch.cuda.synchronize();start=time.monotonic()
        if arm=='R':control.pool=ReusePool(2**30)
        pm.ChunkedKVCache=make_store_class(control)
        _,latent=pipe.inference(noise=noise,text_prompts=[manifest['prompt']],return_latents=True,
                                profile=False,low_memory=False)
        seconds=pipe.vae.finished-start
        torch.cuda.synchronize();monitor.check()
        if tape:
            tape_stats=tape.close();tape=None
            assert tape_stats['records']==60*30*3,tape_stats
        else:tape_stats=None
        assert len(schedule)==60 and all(s['start']==(i//5)*4680 for i,s in enumerate(schedule))
        expected={'U':2400,'B':1200,'R':1048}[arm]
        assert control.decode_calls==expected,(arm,control.decode_calls,expected)
        assert control.read_calls==3660,(arm,control.read_calls)
        cpu=latent.detach().cpu()
        assert torch.isfinite(cpu).all()
        torch.save({'latent':cpu},folder/'latent.pt')
        equal=None
        if name!='gate_U':
            ref=torch.load(DATA/'gate_U/latent.pt',map_location='cpu',weights_only=True)['latent']
            equal=cpu.shape==ref.shape and cpu.dtype==ref.dtype and torch.equal(
                cpu.contiguous().view(torch.uint8),ref.contiguous().view(torch.uint8))
            if not equal:raise RuntimeError('final latent bytes differ from U')
        rows=monitor.rows[samples_start:]
        result=dict(name=name,arm=arm,mode=mode,status='valid',generation_s=seconds,
            time_scope='pool/cache init through final latent, excludes model/text/VAE/file IO',
            latent_equal_to_U=equal,latent_sha256=sha(folder/'latent.pt'),tape=tape_stats,
            counts=control.summarize(),schedule=schedule,
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            process_sample_peak=max((r['process_bytes'] for r in rows),default=0),
            process_peak_note='0.5-second NVML sampled maximum, not a continuous peak',
            temp_min=min((r['temp_c'] for r in rows),default=0),temp_max=max((r['temp_c'] for r in rows),default=0),
            clock_min=min((r['sm_mhz'] for r in rows),default=0),clock_max=max((r['sm_mhz'] for r in rows),default=0),
            finished=stamp())
        save(folder/'result.json',result);save(folder/'telemetry.json',rows)
        save(OUT/f'{name}.json',result)
        emit(f'{name}: valid, generation {seconds:.3f}s, decoder {control.decode_calls}',case=name,phase='complete_case')
        return result
    finally:
        for handle in handles:handle.remove()
        cm.attention=native_attention;uc.triton_prq_dequantize_tensor=native_decoder
        pm.ChunkedKVCache=native_store;pipe.quantize_kv_cache=quantize
        if tape:
            tape.index.close();tape.blob.close();tape.previous.clear()
        pipe.kv_cache1=None;pipe.crossattn_cache=None


def summarize(budget,results,reason):
    pairs=[]
    byname={r['name']:r for r in results}
    for i in range(1,4):
        b=byname.get(f'pair{i}_B');r=byname.get(f'pair{i}_R')
        if b and r:pairs.append(dict(pair=i,B_s=b['generation_s'],R_s=r['generation_s'],
                 reduction_fraction=1-r['generation_s']/b['generation_s']))
    profile=byname.get('profile_B')
    optimistic=None
    if profile:
        # Retention can remove 152 of 1200 decodes in this fixed schedule.
        # Uniform layer cost is only an optimistic diagnostic assumption.
        events=profile['counts']['event_seconds']
        optimistic=dict(decoder_fraction=events.get('decoder',0)/profile['generation_s'],
            decode_layout_fraction=events.get('decode_and_layout',0)/profile['generation_s'],
            retained_call_fraction=152/1200,
            scaled_decode_layout_fraction=(152/1200)*events.get('decode_and_layout',0)/profile['generation_s'],
            interpretation='rough optimistic estimate under uniform decoder cost; not a rigorous upper bound')
    decision='incomplete_or_invalid'
    if len(pairs)==3 and optimistic:
        improvements=[p['reduction_fraction'] for p in pairs]
        if min(improvements)>.05 and optimistic['scaled_decode_layout_fraction']>.05:
            decision='consider_H2_separate_registration'
        elif max(improvements)<.05 and optimistic['scaled_decode_layout_fraction']<.05:
            decision='stop_performance_policy_for_this_setting'
        else:decision='inconclusive_do_not_expand'
    budget['reported']=stamp();budget['elapsed_s']=time.time()-budget['start_unix']
    save(OUT/'summary.json',dict(reason=reason,decision=decision,pairs=pairs,optimistic=optimistic,
        completed=len(results),budget=budget,results=[r['name'] for r in results],
        scope='one input, QVG materialized path; no fused comparison, no H2 fit'))


def main():
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (DATA/'budget.json').exists():raise RuntimeError('Existing stage-1 budget: refuse restart/reset')
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if pids:raise RuntimeError('GPU occupied: '+pids)
    setup_paths()
    manifest=freeze();save(DATA/'manifest.json',manifest);save(OUT/'manifest.json',manifest)
    budget=dict(start_unix=time.time(),started=stamp(),wall_limit_s=7200,max_generations=12,attempts=[])
    save(DATA/'budget.json',budget)
    def timeout(*_):raise TimeoutError('120-minute stage-1 budget reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(7200)
    results=[];reason='running'
    try:
        import torch
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(18*2**30/torch.cuda.get_device_properties(0).total_memory)
        with Monitor() as monitor:
            emit('native 24-latent-frame span validity gates',phase='unit_gate')
            import test_kv_reuse_core
            unit=test_kv_reuse_core.run();save(OUT/'unit_gates.json',unit)
            monitor.check();emit('loading native QVG Self-Forcing model',phase='loading')
            pipe,cond=build(manifest)
            for name,arm,mode in SEQUENCE:
                unchanged(manifest);monitor.check()
                budget['attempts'].append(dict(name=name,arm=arm,mode=mode,started=stamp()))
                save(DATA/'budget.json',budget)
                result=run_case(pipe,manifest,name,arm,mode,monitor)
                results.append(result);budget['attempts'][-1]['status']='valid'
                save(DATA/'budget.json',budget)
            unchanged(manifest)
            reason='complete'
    except BaseException as exc:
        reason='failed: '+repr(exc)
        save(OUT/'failure.json',dict(reason=reason,traceback=traceback.format_exc(),time=stamp()))
        if budget['attempts'] and budget['attempts'][-1].get('status')!='valid':
            budget['attempts'][-1]['status']='failed'
        emit(reason,phase='failed')
        raise
    finally:
        signal.alarm(0);budget['elapsed_s']=time.time()-budget['start_unix']
        save(DATA/'budget.json',budget);summarize(budget,results,reason)


if __name__=='__main__':main()
