# Reproducing this environment

No conda on the session host, so this is a plain venv recipe. Ubuntu 22.04,
RTX 4090 (24 GB), driver 560.28.03, CUDA toolkit 12.6.

```bash
sudo apt-get install -y python3.10-venv          # not present by default
python3 -m venv ~/gpu/sf-venv
~/gpu/sf-venv/bin/python -m pip install --upgrade pip setuptools wheel

# torch first, from the cu124 index
~/gpu/sf-venv/bin/python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu124        # -> 2.6.0+cu124 / 0.21.0+cu124

# trimmed requirements (see note below)
~/gpu/sf-venv/bin/python -m pip install -r requirements-infer.txt
~/gpu/sf-venv/bin/python -m pip install lmdb

# flash-attn: prebuilt wheel, no source build
W=flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
curl -fL -O https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$W
~/gpu/sf-venv/bin/python -m pip install ./$W
```

Match `cxx11abiFALSE` to `torch._C._GLIBCXX_USE_CXX11_ABI` (False for the
official 2.6.0 wheels). If the wheel does not match, skip flash-attn entirely —
`wan/modules/attention.py` already falls back to
`torch.nn.functional.scaled_dot_product_attention` when the import fails.

## Why not `requirements.txt` as shipped

The repo's `requirements.txt` pulls `nvidia-pyindex`, `nvidia-tensorrt`,
`pycuda`, `onnx`, `onnxruntime`, `onnxscript`, `onnxconverter_common`,
`dashscope`, `CLIP` (from git), `open_clip_torch`, `wandb`, `flask`,
`flask-socketio`, `starlette`, `pycocotools`, `dominate` and `torchao`. None of
these are imported anywhere on the CLI inference path — they belong to the demo
and training paths. `requirements-infer.txt` is the trimmed set.

One correction to that trim: `lmdb` **is** required, because `utils/dataset.py`
imports it unconditionally.

## Checkpoints

```bash
cd Self-Forcing
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

`huggingface_hub` 1.x drops `--local-dir-use-symlinks` and renames the CLI to
`hf`; the README's `huggingface-cli` invocation still works but warns.

Sizes: base model 17 GB (of which the umT5 encoder is 11 GB), DMD checkpoint
5.3 GB (md5 `ef41ae42bfb87abe43d5f640fb634d0c`).

## Repos

- Self-Forcing at `33593df3e81fa3ec10239271dd2c100facac6de1`, plus the patches in `../patches/`
- kv-quant-longhorizon at `b4c09363735c89947222781db899862bc0dc55c5`, plus `ucsd_factory_kwargs.patch`
