"""CPU-only fixed-rank codec accounting and reconstruction checks."""
import unittest
import torch
import codec_order_core as core
from tt_rank_codec import encode_payload, FixedRankVCodec


class FixedRankTests(unittest.TestCase):
    def test_real_shape_rank_savings(self):
        shape = (12, 3, 1560, 128)
        for bits in (2, 3, 4):
            baseline = core.expected_bytes('tt', shape, 299, bits)
            previous = 0
            for rank in (64, 128, 192):
                size = core.expected_bytes('tt', shape, rank, bits)
                self.assertGreater(size, previous)
                self.assertLess(size, baseline)
                previous = size
                codec = FixedRankVCodec(bits, rank)
                self.assertEqual(codec.rank, rank)
                self.assertEqual(codec.encode_calls, 0)

    def test_packed_roundtrip_and_actual_bytes(self):
        torch.manual_seed(719)
        v = torch.randn(1, 36, 2, 8).bfloat16()
        for bits in (2, 3, 4):
            for rank in (3, 8, 12):
                payload = encode_payload(v, bits, rank)
                self.assertEqual(payload['rank'], rank)
                self.assertEqual(core.payload_bytes(payload), core.expected_bytes('tt', (2, 3, 12, 8), rank, bits))
                out = core.decode(payload)
                self.assertEqual(out.shape, v.shape)
                self.assertTrue(torch.isfinite(out).all())
                # Independently contract unpacked factors, including the token-axis permutation.
                a, b, c = [core.unpack_factor(f) for f in payload['factors']]
                expected = torch.einsum('hsr,hrtk,hkd->htsd', a, b, c)
                expected = expected.permute(1, 2, 0, 3).reshape_as(v)
                torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
                self.assertTrue(all(f['stream'].dtype == torch.uint8 for f in payload['factors']))

    def test_zero_and_invalid_rank(self):
        v = torch.zeros(1, 36, 2, 8, dtype=torch.bfloat16)
        for bits in (2, 3, 4):
            self.assertTrue(torch.equal(core.decode(encode_payload(v, bits, 8)), v.float()))
        for rank in (0, 13, 1.5, True):
            with self.assertRaises(ValueError):
                encode_payload(v, 2, rank)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
