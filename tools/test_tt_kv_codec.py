"""CPU tests of independent K/V packing and tensor payload ownership."""
import unittest
from unittest.mock import patch
import torch
import codec_order_core as core
from tt_kv_codec import KVCodec, pair_payloads
from change_mixed_core import encode_uniform
from tt_rank_codec import encode_payload


def tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from tensors(value)
    elif hasattr(obj, '__dict__'):
        yield from tensors(vars(obj))


class KVCodecTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(718)
        self.k = torch.randn(1, 36, 2, 8).bfloat16()
        self.v = (torch.randn_like(self.k.float()) * 0.3 + 2).bfloat16()

    def test_independent_payload_matches_single_codec(self):
        for bits in (2, 3, 4):
            for rank in (0, 8):
                kind, payloads = pair_payloads(self.k, self.v, bits, rank)
                for original, payload in zip((self.k, self.v), payloads):
                    expected = encode_uniform(original, bits=bits, group=16) if rank == 0 else encode_payload(original, bits, rank)
                    restore = (lambda x: x.restore()) if rank == 0 else core.decode
                    torch.testing.assert_close(restore(payload), restore(expected), rtol=0, atol=0)
                    self.assertEqual(restore(payload).shape, original.shape)
                    self.assertTrue(torch.isfinite(restore(payload)).all())
                # Encoded storage does not retain either original or share K/V factor buffers.
                groups = [{t.untyped_storage().data_ptr() for t in tensors(p)} for p in payloads]
                self.assertFalse(groups[0] & groups[1])
                self.assertFalse((groups[0] | groups[1]) & {self.k.untyped_storage().data_ptr(), self.v.untyped_storage().data_ptr()})
                before = [restore(p).clone() for p in payloads]
                self.k.add_(1)
                for p, ref in zip(payloads, before):
                    torch.testing.assert_close(restore(p), ref, rtol=0, atol=0)

    def test_pair_accounting_and_identity(self):
        for bits in (2, 3, 4):
            for rank in (0, 8):
                codec = KVCodec(bits, rank)
                with patch('tt_kv_codec.torch.cuda.Event'):
                    states = codec.encode_pair(self.k, self.v)
                self.assertEqual(codec.encode_calls, 1)
                self.assertEqual(len(codec.events), 1)
                self.assertEqual(codec.ranks, {rank} if rank else set())
                self.assertEqual(states[0].resident(), states[1].resident())
                if rank:
                    self.assertEqual(sum(s.resident() for s in states), 2*core.expected_bytes('tt', (2, 3, 12, 8), rank, bits))
                else:
                    self.assertEqual(sum(s.resident() for s in states), 2*((36*2*8*bits+7)//8+3*2*8*2))
        raw = KVCodec(0, 0)
        a, b = raw.encode_pair(self.k, self.v)
        self.assertIs(a.restore(), self.k)
        self.assertIs(b.restore(), self.v)
        self.assertEqual(raw.events, [])

    def test_invalid_config_and_shape(self):
        for bits, rank in ((0, 1), (1, 0), (2, -1), (2, True)):
            with self.assertRaises(ValueError):
                KVCodec(bits, rank)
        with self.assertRaises(ValueError):
            pair_payloads(self.k, self.v[:, :-1], 2, 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
