"""Archive small validation evidence in Git; large captures remain external."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--data', type=Path, default=Path.home()/'sfq_data')
a = ap.parse_args()
data = a.data.resolve()
out = data/'verify'
status = json.loads((out/'status.json').read_text())
required = ('G1','G1b','G2','G3','G4','G5','CHUNK')
if not all(status.get(g,{}).get('state') == 'PASS' for g in required):
    raise SystemExit('Required checks are incomplete; no final evidence exported')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


artifact = {'status':status,
            'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'checkpoint':json.loads((out/'checkpoint.json').read_text()),
            'commands':{},'comparisons':{},'logs':{},'source_sha256':{},'input_sha256':{}}
artifact['references'] = {}
for name in ('sf-pristine','sf-legacy','kv-reference'):
    repo = data/name
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    diff = subprocess.check_output(['git','diff','--binary','HEAD'],cwd=repo)
    artifact['references'][name] = {'commit':revision,'diff_sha256':hashlib.sha256(diff).hexdigest()}
artifact['runtime_log_sha256'] = {p.name:sha(p) for p in sorted(out.glob('*.log'))}
for p in sorted(out.glob('*.command.json')):
    artifact['commands'][p.stem] = json.loads(p.read_text())
for p in sorted(out.glob('*comparison*.json')):
    if p.name.endswith(('.done.json','.command.json')): continue
    artifact['comparisons'][p.stem] = json.loads(p.read_text())
for name in ('all_tests.log','unit_cuda.log','integration.log','memory.log','store.log'):
    path = data/name
    if path.exists(): artifact['logs'][name] = {'sha256':sha(path),'text':path.read_text()}
for folder in ('kvq','pipeline','wan','utils','configs','scripts','tests'):
    for p in sorted((ROOT/folder).rglob('*')):
        if p.is_file() and p.suffix in ('.py','.yaml'):
            artifact['source_sha256'][str(p.relative_to(ROOT))] = sha(p)
artifact['source_sha256']['inference.py'] = sha(ROOT/'inference.py')
for p in sorted(out.glob('prompts*.txt')):
    artifact['input_sha256'][p.name] = sha(p)
weights = data/'weights_manifest.json'
if weights.exists(): artifact['weights'] = json.loads(weights.read_text())
(ROOT/'docs/verification-evidence.json').write_text(json.dumps(artifact,indent=2))
commits = subprocess.check_output(['git','log','--reverse','--format=%h %s','da5438c..HEAD'],cwd=ROOT,text=True)
(ROOT/'docs/IMPLEMENTATION_COMMITS.txt').write_text(commits)
print('Exported docs/verification-evidence.json and docs/IMPLEMENTATION_COMMITS.txt')
