"""Gates for accounting, protected RTN parity and interaction inference inputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from quantizer_policy_core import make_store, storage_bytes, interaction
from mechanism_core import make as old_make
from kvq.quant import Quantizer
from kvq.memory import memory_report
from anchor_identity_gate import IdentityQuantizer


def run():
    torch.set_num_threads(2)
    torch.manual_seed(872341)
    records = []
    for method in ('rtn', 'kivi'):
        for group in (16, 24):
            stores = {a: make_store(method, a, group, capacity=168) for a in ('L', 'C_repeat')}
            identities = {a: make_store(method, a, group, capacity=168, identity=IdentityQuantizer()) for a in stores}
            old = {a: old_make(a, group, capacity=168) for a in stores} if method == 'rtn' else {}
            for chunk in range(12):
                for call in range(5):
                    k = torch.randn(1, 24, 2, 8).bfloat16()
                    v = torch.randn_like(k)
                    got = {}; clean = {}
                    for arm, store in stores.items():
                        store.attend(k, k, v, chunk*24, lambda q, a, b, arm=arm: got.update({arm: (a.clone(), b.clone())}))
                        identities[arm].attend(k, k, v, chunk*24, lambda q, a, b, arm=arm: clean.update({arm: (a.clone(), b.clone())}))
                        state = store.state if arm == 'L' else [s['state'] for s in store.segments]
                        pending = getattr(store, 'pending', None)
                        assert storage_bytes(store)['resident_tensor_bytes'] == memory_report([state, pending])['resident_tensor_bytes']
                        assert all(torch.isfinite(x).all() for x in got[arm])
                        if arm in old:
                            ref = []
                            old[arm].attend(k, k, v, chunk*24, lambda q, a, b: ref.extend([a.clone(), b.clone()]))
                            assert all(torch.equal(a, b) for a, b in zip(got[arm], ref))
                    assert all(torch.equal(a, b) for a, b in zip(clean['L'], clean['C_repeat']))
                    if group == 24:
                        assert all(torch.equal(a, b) for a, b in zip(got['L'], got['C_repeat']))
            records.append(dict(method=method, group=group, bytes={a: storage_bytes(s) for a, s in stores.items()}))
    x = torch.randn(1, 24, 2, 8).bfloat16()
    q = Quantizer('kivi', 4, 4, 16)
    k, v = q.encode_pair(x, x)
    assert k.scale.shape == (1, 2, 1, 2, 8)
    assert v.scale.shape == (1, 2, 16, 1, 1)
    assert k.codes.dtype == v.codes.dtype == torch.uint8
    assert k.offset.dtype == v.offset.dtype == torch.float16
    sample = [dict(method=m, arm=a, late_raw_musiq=s) for m, a, s in
              [('rtn', 'L', 10), ('rtn', 'C_repeat', 13), ('kivi', 'L', 14), ('kivi', 'C_repeat', 20)]]
    assert interaction(sample) == 3
    assert interaction([dict(r, late_raw_musiq=r['late_raw_musiq']+100) for r in sample]) == 3
    for bad in (sample[:3], sample+[sample[0]], [dict(r, late_raw_musiq=float('nan')) for r in sample]):
        try: interaction(bad)
        except ValueError: pass
        else: raise AssertionError('invalid factorial sample accepted')
    print('PASS: two quantizers, partial/aligned groups, eviction/refresh, RTN parity, offset accounting, identity and interaction sign')
    return records


if __name__ == '__main__':
    run()
