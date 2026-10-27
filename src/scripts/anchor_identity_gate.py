"""Post-pilot validity gate for the exact anchor_store path; no sampling changes."""
import argparse,json,sys,time
from pathlib import Path
import torch


class IdentityState:
    def __init__(self,tensor):self.tensor=tensor.clone()
    def restore(self):return self.tensor


class IdentityQuantizer:
    def __init__(self):self.encode_calls=0
    def encode_pair(self,k,v):
        self.encode_calls+=1
        return IdentityState(k),IdentityState(v)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path.home()/'sfq_data/anchor_validity')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    # Import the unchanged implementation under review, not a rewritten helper.
    sys.path.insert(0,str(args.source.resolve()/'scripts'))
    import p3_tf_fr as pilot
    device=pilot.gpu;torch.set_grad_enabled(False);torch.set_num_threads(4)
    fingerprint=pilot.source_hash();started=time.time()
    pipe=pilot.build(0)
    baseline=Path.home()/'sfq_data/verify'
    texts=(baseline/'prompts2.txt').read_text().splitlines()
    conds=pilot.encode(pipe,texts);records=[]
    for index,cond in enumerate(conds):
        reference=pilot.bf16_cache(pipe,device)
        rx,tx=pilot.cross_cache(pipe,device),pilot.cross_cache(pipe,device)
        noise=pilot.chunk_noise(pipe,63,0,device);quantizer=IdentityQuantizer();work=None
        for chunk in range(21):
            work=None
            work=pilot.compressed_cache(pipe,quantizer,device,reference=reference)
            assert all(c['store'].__class__.__name__=='ChunkStore' for c in work)
            r=pilot.run_chunk(pipe,reference,rx,cond,noise,chunk,0)
            t=pilot.run_chunk(pipe,work,tx,cond,noise,chunk,0)
            old=torch.load(baseline/f'g3_base/latents/prompt{index:03d}/chunk_{chunk:04d}.pt',map_location='cpu',weights_only=True)['latent']
            row={'prompt':index,'chunk':chunk,'tf_bitexact':torch.equal(r,t),
                 'reference_bitexact':torch.equal(r.float().cpu(),old),'encode_calls':quantizer.encode_calls}
            records.append(row)
            if not row['tf_bitexact'] or not row['reference_bitexact']:
                (args.out/'result.json').write_text(json.dumps({'pass':False,'records':records},indent=2))
                raise RuntimeError(f'anchor identity gate failed: {index}/{chunk}')
            if chunk and quantizer.encode_calls==0:raise RuntimeError('encoded-segment path not exercised')
            if chunk%7==0 or chunk==20:print('ANCHOR_GATE',index,chunk+1,'/21 PASS calls',quantizer.encode_calls,flush=True)
        work=None;del reference,rx,tx,noise;torch.cuda.empty_cache()
    if pilot.source_hash()!=fingerprint:raise RuntimeError('tested source changed')
    result={'pass':True,'tested_source_sha256':fingerprint,'source_commit':'caf0b4b',
            'method':'compressed_cache(reference=BF16) with identity encode_pair; no plain_snapshot',
            'records':records,'seconds':time.time()-started}
    (args.out/'result.json').write_text(json.dumps(result,indent=2));print('ANCHOR GATE PASS 42/42',flush=True)

if __name__=='__main__':main()
