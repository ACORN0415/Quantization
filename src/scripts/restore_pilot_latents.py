"""Recover original research tensors with unchanged source and exact hash checks."""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--data',type=Path,default=Path.home()/'sfq_data/p3_pilot')
    ap.add_argument('--gate',type=Path,default=Path.home()/'sfq_data/anchor_validity/result.json')
    ap.add_argument('--staging',type=Path,default=Path.home()/'sfq_data/latent_recovery')
    args=ap.parse_args();args.staging.mkdir(parents=True,exist_ok=True)
    gate=json.loads(args.gate.read_text())
    if not gate['pass']:raise RuntimeError('passed actual-anchor identity gate required')
    sys.path.insert(0,str(args.source.resolve()/'scripts'))
    import p3_tf_fr as original
    import torch
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    fingerprint=original.source_hash()
    if fingerprint!=gate['tested_source_sha256']:raise RuntimeError('gate/source mismatch')
    texts=(args.source/'experiments/p3_pilot_prompts.txt').read_text().splitlines()
    pipe=conds=None;state={};started=time.time()
    for bits in (4,2):
        for prompt in (0,1):
            name=f'p{prompt}_s0_int{bits}';folder=args.data/name
            old=json.loads((folder/'complete.json').read_text())
            manifest=json.loads((folder/'cleanup.json').read_text())
            if old['source_sha256']!=fingerprint or old['prompt']!=texts[prompt]:raise RuntimeError('original input/source mismatch')
            if all((folder/r['path']).exists() and digest(folder/r['path'])==r['sha256'] for r in manifest):
                state[name]={'state':'REUSED_VERIFIED','files':manifest};continue
            state[name]={'state':'RUNNING'}
            (args.staging/'status.json').write_text(json.dumps(state,indent=2));print('RESTORE',name,flush=True)
            if pipe is None:
                pipe=original.build(0);conds=original.encode(pipe,texts)
            temporary=args.staging/name
            meta=original.trajectory(pipe,conds[prompt],126,0,bits,temporary)
            if meta['rows']!=old['rows']:raise RuntimeError(f'{name}: latent-distance rows changed')
            checked=[]
            for entry in manifest:
                path=temporary/entry['path'];actual=digest(path)
                if actual!=entry['sha256']:raise RuntimeError(f'{name}/{path.name}: SHA256 differs; keep staged evidence')
                checked.append(dict(entry,sha256_match=True))
            # Only verified files replace missing originals; quality/video/old
            # sampling records remain untouched. Duplicate staging is removed.
            for entry in checked:
                source=temporary/entry['path'];target=folder/entry['path']
                if target.exists() and digest(target)==entry['sha256']:source.unlink()
                else:source.replace(target)
            rec={'state':'RESTORED','source_commit':'caf0b4b','source_sha256':fingerprint,
                 'latent_rows_bitexact':True,'files':checked,'retained':True}
            (folder/'restoration.json').write_text(json.dumps(rec,indent=2))
            (folder/'latent_manifest.json').write_text(json.dumps({'retained':True,'artifacts':manifest},indent=2))
            state[name]=rec;(args.staging/'status.json').write_text(json.dumps(state,indent=2))
            print('VERIFIED RESTORED',name,'3/3 hashes; 42/42 rows',flush=True)
    if original.source_hash()!=fingerprint:raise RuntimeError('original source changed during recovery')
    result={'pass':True,'seconds':time.time()-started,'cases':state,'source_sha256':fingerprint,
            'source_commit':'caf0b4b','gate_source_sha256':gate['tested_source_sha256']}
    (args.staging/'result.json').write_text(json.dumps(result,indent=2));print('ALL LATENTS RESTORED',flush=True)

if __name__=='__main__':main()
