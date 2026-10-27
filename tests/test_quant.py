"""Numerical compatibility against a user-supplied, patched local reference."""
import os
import sys
import unittest
import torch
from kvq import Quantizer


class QuantTests(unittest.TestCase):
    def test_validation(self):
        for kwargs in ({"key_bits": 1}, {"value_bits": 9}, {"group": 0}, {"method": "other"}):
            with self.assertRaises(ValueError):
                Quantizer(**kwargs)

    def test_unsigned_eight_bits(self):
        x = torch.linspace(-1, 1, 16).reshape(1, 16, 1, 1).bfloat16()
        q = Quantizer("kivi", 8, 8, legacy_unsigned_int8=False).encode(x)
        self.assertEqual(q.codes.dtype, torch.uint8)
        self.assertGreater(q.restore().max().item(), 0.9)

    @unittest.skipUnless(os.environ.get("KVQ_REF_REPO"), "set KVQ_REF_REPO to patched local checkout")
    def test_reference(self):
        sys.path.insert(0, os.environ["KVQ_REF_REPO"])
        from kv_quant.factory import create_quantizer
        device = os.environ.get("TEST_DEVICE", "cpu")
        generator = torch.Generator(device=device).manual_seed(23)
        for method in ("rtn", "kivi"):
            for bits in (2, 3, 4, 8):
                ref = create_quantizer(method.upper(), bits=bits, block_size=16)
                ours = Quantizer(method, bits, bits)
                for length in (32, 4680, 32760):
                    for case in ("random", "constant", "positive", "zero"):
                        with self.subTest(method=method, bits=bits, length=length, case=case, device=device):
                            x = torch.randn((1, length, 12, 128), generator=generator, device=device, dtype=torch.bfloat16)
                            if case == "constant": x.fill_(0.25)
                            if case == "positive": x = x.abs() + 0.5
                            if case == "zero": x.zero_()
                            kr, vr = ref.dequantize_kv(ref.quantize_kv(x, x))
                            k, v = ours.encode_pair(x, x)
                            self.assertTrue(torch.equal(k.restore(), kr))
                            self.assertTrue(torch.equal(v.restore(), vr))
                # Independent K/V allocation.
                ref = create_quantizer(method.upper(), bits=bits, key_bits=bits, value_bits=2, block_size=16)
                x = torch.randn((1, 33, 12, 128), generator=generator, device=device, dtype=torch.bfloat16)
                kr, vr = ref.dequantize_kv(ref.quantize_kv(x, x))
                k, v = Quantizer(method, bits, 2).encode_pair(x, x)
                self.assertTrue(torch.equal(k.restore(), kr))
                self.assertTrue(torch.equal(v.restore(), vr))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
