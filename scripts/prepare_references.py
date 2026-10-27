"""Prepare local validation checkouts; third-party source stays outside this repo.

The supplied upstream clones must already exist. Existing checkouts are checked,
not reset or replaced. This script never downloads or publishes repositories.
"""
import argparse
from pathlib import Path
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('--data', type=Path, default=Path.home()/'sfq_data')
ap.add_argument('--history', type=Path, default=Path.home()/'Quantization')
ap.add_argument('--check', action='store_true', help='Verify existing setup without modifying it')
a = ap.parse_args()
data, history = a.data.resolve(), a.history.resolve()


def git(path, *args, check=True):
    return subprocess.run(['git','-C',str(path),*args], check=check, text=True, capture_output=True)


def checkout(name, upstream, pin, patches=()):
    dest = data/name
    if not dest.exists():
        if a.check: raise RuntimeError(f'Missing {dest}')
        subprocess.run(['git','clone','--shared',str(data/'upstream'/upstream),str(dest)],check=True)
        git(dest,'checkout','--detach',pin)
    head = git(dest,'rev-parse','HEAD').stdout.strip()
    if head != pin: raise RuntimeError(f'{dest}: unexpected HEAD {head}')
    for patch in patches:
        if git(dest,'apply','--reverse','--check',str(patch),check=False).returncode == 0:
            continue
        if a.check: raise RuntimeError(f'{dest}: missing patch {patch}')
        git(dest,'apply','--check',str(patch))
        git(dest,'apply',str(patch))
    if upstream == 'Self-Forcing':
        for sub in ('wan_models','checkpoints'):
            link, target = dest/sub, data/'ckpt'/sub
            if link.is_symlink() and link.resolve() == target.resolve(): continue
            if link.exists() or link.is_symlink(): raise RuntimeError(f'{link} exists with unexpected contents')
            if a.check: raise RuntimeError(f'Missing symlink {link}')
            link.symlink_to(target)
    print(f'OK {name} {head}',flush=True)


sf = '33593df3e81fa3ec10239271dd2c100facac6de1'
kv = 'b4c09363735c89947222781db899862bc0dc55c5'
checkout('sf-pristine','Self-Forcing',sf)
checkout('sf-legacy','Self-Forcing',sf,[next((history/'patches/self-forcing').glob(n+'_*.patch')) for n in ('01','02','16')])
checkout('kv-reference','kv-quant-longhorizon',kv,[history/'patches/kv-quant-longhorizon/17_kivi_residual_window.patch'])
