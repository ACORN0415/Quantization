"""Lossless attention-input tape with unchanged-prefix delta compression.

SHA identifies the artifact; verification compares every reconstructed byte.
Only used in correctness runs, never timings.
"""
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import zstandard as zstd


class InputTape:
    def __init__(self, folder, mode):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.mode=mode;self.previous={};self.records=0;self.raw_bytes=0;self.stored_bytes=0
        self.codec=zstd.ZstdCompressor(level=3)
        self.decoder=zstd.ZstdDecompressor()
        self.index=(self.folder/'index.jsonl').open('w' if mode=='write' else 'r')
        self.blob=(self.folder/'inputs.zst').open('wb' if mode=='write' else 'rb')

    def tensor(self, key, tensor):
        cpu=tensor.detach().contiguous().cpu()
        if not bool(torch.isfinite(cpu).all()):raise RuntimeError('nonfinite attention input')
        data=cpu.view(torch.uint8).numpy().tobytes()
        del cpu
        shape=list(tensor.shape);dtype=str(tensor.dtype)
        # Independent histories per layer and K/V; Q always stored in full.
        history_key=(key['layer'],key['tensor'])
        old=self.previous.get(history_key,b'') if key['tensor']!='q' else b''
        if self.mode=='write':
            if shutil.disk_usage(self.folder).free<8*2**30:
                raise RuntimeError('attention tape disk headroom <8GiB')
            common=0
            if old:
                n=min(len(data),len(old))
                if data[:n]==old[:n]:common=n
                else:
                    # Native attention updates the last three latent frames;
                    # quantization boundaries change a prefix and fall back to full.
                    n=max(0,n-3*1560*12*128*2)
                    if data[:n]==old[:n]:common=n
            packed=self.codec.compress(data[common:])
            row=dict(key=key,shape=shape,dtype=dtype,prefix=common,
                     bytes=len(data),packed_bytes=len(packed),offset=self.blob.tell())
            self.index.write(json.dumps(row)+'\n');self.blob.write(packed)
            self.stored_bytes+=len(packed)
            if self.stored_bytes>65*2**30:raise RuntimeError('attention tape >65GiB')
        else:
            line=self.index.readline()
            if not line:raise RuntimeError('attention tape ended early')
            row=json.loads(line)
            if row['key']!=key or row['shape']!=shape or row['dtype']!=dtype:
                raise RuntimeError(f'attention schedule/shape changed: {row["key"]} vs {key}')
            packed=self.blob.read(row['packed_bytes'])
            expected=old[:row['prefix']]+self.decoder.decompress(packed)
            if len(expected)!=row['bytes'] or data!=expected:
                a=np.frombuffer(data,dtype=np.uint8);b=np.frombuffer(expected,dtype=np.uint8)
                raise RuntimeError(f'attention input bytes differ at {key}; different bytes={int(np.count_nonzero(a!=b))}')
        if key['tensor']!='q':self.previous[history_key]=data
        self.records+=1;self.raw_bytes+=len(data)

    def close(self):
        if self.mode!='write' and self.index.readline():raise RuntimeError('unconsumed attention tape records')
        self.index.close();self.blob.close();self.previous.clear()
        return dict(mode=self.mode,records=self.records,compared_or_recorded_bytes=self.raw_bytes,
                    compressed_bytes=self.stored_bytes,comparison='direct full shape/dtype/element bytes')
