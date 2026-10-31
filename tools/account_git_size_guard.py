"""User-authorized 5 GB cap for the account-level Codex snapshot repository only."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/acorn0415')
TARGET = ROOT / '.git'
STATE = ROOT / '.local/state/account-git-guard'
LIMIT = 5_000_000_000


def git(*args):
    return subprocess.run(['/usr/bin/git', '-C', str(ROOT), *args],
                          capture_output=True, text=True, timeout=20)


def in_use():
    if any(TARGET.rglob('*.lock')):
        return 'repository lock file present'
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if (proc / 'comm').read_text().strip() in ('git', 'git-repack', 'git-pack-object', 'git-index-pack'):
                return 'Git process active'
            for fd in (proc / 'fd').iterdir():
                try:
                    path = os.readlink(fd)
                except OSError:
                    continue
                if path.startswith(str(TARGET) + '/'):
                    return 'repository file open'
        except OSError:
            continue
    return None


def eligibility(size, refs, has_head, tracked):
    if size <= LIMIT:
        return 'below_limit'
    if has_head or tracked or any(not r.startswith('refs/codex/') for r in refs):
        return 'protected_non_snapshot_history'
    return 'eligible'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'guard.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if not TARGET.exists():
            return
        assert TARGET.is_dir() and not TARGET.is_symlink()
        size = sum(p.lstat().st_size for p in TARGET.rglob('*') if not p.is_dir())
        if size <= LIMIT:
            return
        if in_use():
            return
        refs = git('for-each-ref', '--format=%(refname) %(objectname)')
        tracked = git('ls-files', '-z')
        assert refs.returncode == tracked.returncode == 0
        decision = eligibility(size, [r.split()[0] for r in refs.stdout.splitlines()],
                               git('rev-parse', '--verify', 'HEAD').returncode == 0,
                               bool(tracked.stdout))
        event = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     path=str(TARGET), bytes=size, limit_bytes=LIMIT,
                     decision=decision, refs=refs.stdout, apply=args.apply)
        if args.apply and decision == 'eligible':
            if in_use() or git('for-each-ref', '--format=%(refname) %(objectname)').stdout != refs.stdout:
                return
            # Log intent first; only this exact directory may be retired.
            with (STATE / 'history.jsonl').open('a') as log:
                log.write(json.dumps(dict(event, status='deleting')) + '\n')
            quarantine = ROOT / ('.git.retired-' + str(os.getpid()))
            assert not quarantine.exists()
            TARGET.rename(quarantine)
            shutil.rmtree(quarantine)
            event['status'] = 'deleted_oversize_snapshot_repository'
        else:
            event['status'] = 'dry_run_or_protected'
        with (STATE / 'history.jsonl').open('a') as log:
            log.write(json.dumps(event) + '\n')
        print(json.dumps(event))


if __name__ == '__main__':
    main()
