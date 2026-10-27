"""Run a local reference checkout, capturing final latents without editing it.

Usage: python scripts/reference_run.py CHECKOUT DUMP_ROOT -- [inference args]
Only installs an output capture wrapper; model and sampling operations are the
reference checkout's own implementation. No third-party source is redistributed.
"""
import os
from pathlib import Path
import runpy
import sys

checkout, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
args = sys.argv[3:]
if args and args[0] == '--':
    args = args[1:]
os.chdir(checkout)
sys.path.insert(0, str(checkout))
from pipeline.causal_inference import CausalInferencePipeline
import torch
original = CausalInferencePipeline.inference
counter = 0


def capture(self, *args, **kwargs):
    global counter
    result = original(self, *args, **kwargs)
    if not isinstance(result, tuple):
        raise RuntimeError('reference must request return_latents=True')
    folder = output / f'prompt{counter:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({'latent': result[1].float().cpu()}, folder / 'full.pt')
    counter += 1
    return result


CausalInferencePipeline.inference = capture
sys.argv = [str(checkout / 'inference.py'), *args]
runpy.run_path(str(checkout / 'inference.py'), run_name='__main__')
