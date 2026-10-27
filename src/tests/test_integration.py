"""Exercise the actual model attention with stores and plain upstream behavior."""
import copy
import importlib.util
import os
from pathlib import Path
import unittest
import torch
from kvq import Quantizer, RollingStore
from wan.modules.causal_model import CausalWanSelfAttention, rope_params


class AttentionTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('SF_REF_REPO'), 'set SF_REF_REPO to pristine checkout')
    def test_plain_path_matches_upstream(self):
        path = Path(os.environ['SF_REF_REPO'])/'wan/modules/causal_model.py'
        spec = importlib.util.spec_from_file_location('upstream_attention', path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        from wan.modules import attention as attention_module
        attention_module.FLASH_ATTN_2_AVAILABLE = False
        attention_module.FLASH_ATTN_3_AVAILABLE = False
        for sink in (0, 1):
            torch.manual_seed(4)
            ours = CausalWanSelfAttention(8, 1, local_attn_size=4, sink_size=sink).bfloat16().eval()
            ref = reference.CausalWanSelfAttention(8, 1, local_attn_size=4, sink_size=sink).bfloat16().eval()
            ref.load_state_dict(ours.state_dict())
            def cache():
                return {'k': torch.zeros(1, 4, 1, 8, dtype=torch.bfloat16), 'v': torch.zeros(1, 4, 1, 8, dtype=torch.bfloat16),
                        'global_end_index': torch.tensor([0]), 'local_end_index': torch.tensor([0])}
            a, b = cache(), cache()
            for frame in range(8):
                for phase in range(2):
                    x = torch.randn(1, 1, 8).bfloat16()
                    args = dict(seq_lens=torch.tensor([1]), grid_sizes=torch.tensor([[1,1,1]]),
                                freqs=rope_params(64, 8), block_mask=None, current_start=frame)
                    with torch.no_grad():
                        ya = ours(x, kv_cache=a, **args)
                        yb = ref(x, kv_cache=b, **args)
                    self.assertTrue(torch.equal(ya, yb))
                    self.assertTrue(torch.equal(a['k'], b['k']))


    @unittest.skipUnless(os.environ.get('LEGACY_REF_REPO') and os.environ.get('KVQ_REF_REPO'),
                         'set LEGACY_REF_REPO and KVQ_REF_REPO')
    def test_rolling_attention_matches_legacy(self):
        import sys
        sys.path.insert(0, os.environ['KVQ_REF_REPO'])
        from kv_quant.factory import create_quantizer
        from wan.modules import attention as attention_module
        attention_module.FLASH_ATTN_2_AVAILABLE = False
        attention_module.FLASH_ATTN_3_AVAILABLE = False
        path = Path(os.environ['LEGACY_REF_REPO'])/'wan/modules/causal_model.py'
        spec = importlib.util.spec_from_file_location('legacy_attention', path)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        for method, bits in (('rtn', 2), ('rtn', 4), ('kivi', 4)):
            for sink in (0, 1):
                torch.manual_seed(7)
                ours = CausalWanSelfAttention(8, 1, local_attn_size=3, sink_size=sink).bfloat16().eval()
                ref = legacy.CausalWanSelfAttention(8, 1, local_attn_size=3, sink_size=sink).bfloat16().eval()
                ref.load_state_dict(ours.state_dict())
                def cache():
                    return {'k': torch.zeros(1, 12, 1, 8, dtype=torch.bfloat16),
                            'v': torch.zeros(1, 12, 1, 8, dtype=torch.bfloat16),
                            'global_end_index': torch.tensor([0]), 'local_end_index': torch.tensor([0])}
                a, b = cache(), cache()
                a.update(store=RollingStore(Quantizer(method, bits, bits), 12, sink*4), layer=0)
                b.update(quantizer=create_quantizer(method.upper(), bits=bits, block_size=16), quant_state=None)
                for chunk in range(8):
                    for phase in range(5):
                        x = torch.randn(1, 4, 8).bfloat16()
                        args = dict(seq_lens=torch.tensor([4]), grid_sizes=torch.tensor([[1,2,2]]),
                                    freqs=rope_params(64, 8), block_mask=None, current_start=chunk*4)
                        with torch.no_grad():
                            ya, yb = ours(x, kv_cache=a, **args), ref(x, kv_cache=b, **args)
                        self.assertTrue(torch.equal(ya, yb), (method, bits, sink, chunk, phase))
                        kr, vr = b['quantizer'].dequantize_kv(b['quant_state'])
                        k, v = a['store'].restore(x.reshape(1,4,1,8))
                        self.assertTrue(torch.equal(k, kr))
                        self.assertTrue(torch.equal(v, vr))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
