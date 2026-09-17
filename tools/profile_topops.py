import sys, os, torch
sys.path.insert(0, "/home/devel/gpu/Self-Forcing")
os.chdir("/home/devel/gpu/Self-Forcing")
from torch.profiler import profile, ProfilerActivity
from profile_bottleneck import build, _run_chunks
from utils.dataset import TextDataset

torch.set_grad_enabled(False)
dev = torch.device("cuda"); fsl, nfb = 1560, 3
ds = TextDataset(prompt_path="prompts/quant3/prompts10.txt")

for tag, kvq in [("bf16", None), ("int4_kv", "RTN")]:
    pipe, _ = build("configs/gateA_A1.yaml", "checkpoints/self_forcing_dmd.pt", dev,
                    kv_quant=kvq, kv_bits=4)
    pipe.text_encoder.to(dev)
    cond = pipe.text_encoder(text_prompts=[ds[0]["prompts"]])
    pipe.text_encoder.to("cpu"); torch.cuda.empty_cache()
    noise = torch.cat([torch.randn([1,nfb,16,60,104],
                       generator=torch.Generator(device=dev).manual_seed(c),
                       device=dev, dtype=torch.bfloat16) for c in range(21)], dim=1)
    _run_chunks(pipe, cond, noise, nfb, fsl, 2); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _run_chunks(pipe, cond, noise, nfb, fsl, 3); torch.cuda.synchronize()
    print(f"\n### {tag}: 상위 12개 연산 (self device time)")
    evs = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)[:12]
    tot = sum(e.self_device_time_total for e in prof.key_averages())
    for e in evs:
        print(f"  {e.key[:44]:46s} {e.self_device_time_total/1e3:8.1f} ms  "
              f"{100*e.self_device_time_total/tot:5.1f}%  calls={e.count}")
    del pipe, noise; torch.cuda.empty_cache()
