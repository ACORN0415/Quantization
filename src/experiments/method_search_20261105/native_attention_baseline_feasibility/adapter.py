"""Source-only loader for the official *older* INT8-QK/FP16-PV Triton path.

No GPU launch occurs here. Importing this helper does not import Torch/Triton.
An actual benchmark needs its own reviewed runtime specification and root authority.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PINNED = {
    'quant_per_block': '990b4881fb5e8d70b59fa7b796dab20693c82761175ee51fc4c5329d51e1d377',
    'attn_qk_int8_per_block': 'f7563994203a2bb95282ac3474beb8df59391d2b1c268b8757477cc59203250b',
}


def shape_contract(kv_length):
    if kv_length not in (4680, 18720, 32760, 56160):
        raise ValueError('not a prospective registered shape')
    return dict(batch=1, heads=12, head_dim=128, q_valid=4680,
                q_padded=((4680+127)//128)*128, kv_valid=kv_length,
                kv_padded=((kv_length+63)//64)*64, causal=False)


def load_sources():
    """CPU import/decorator construction only; never call kernels in this helper."""
    modules = {}
    for short, expected in PINNED.items():
        path = HERE/'source/sageattention/triton'/(short+'.py')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('pinned source changed: '+short)
        name = '_native_baseline_'+short
        if name in sys.modules:
            raise RuntimeError('private loader name already in use: '+name)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        modules[short] = module
    return modules
