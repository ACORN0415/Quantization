"""CPU checks with a scalar reference independent of the vector grouping."""
import itertools
import math
import unittest

import torch

from kivi_grouped_candidate import KiviGroupedCandidate, MIN_SCALE
from quantizer_policy_core import storage_bytes
from kvq.memory import memory_report
from kvq.store import RollingStore
from kvq.quant import Quantizer


def scalar_reference(x, bits, group, value):
    bsz, length, heads, channels = x.shape
    padded_length = length if value else math.ceil(length / group) * group
    codes = torch.zeros(bsz, padded_length, heads, channels, dtype=torch.uint8)
    restored = torch.empty_like(codes, dtype=torch.bfloat16)
    scales, offsets = [], []
    for b in range(bsz):
        groups = (itertools.product(range(length), range(heads), range(channels // group))
                  if value else itertools.product(range(padded_length // group), range(heads), range(channels)))
        for a, h, c in groups:
            positions = ([(a, h, c * group + j) for j in range(group)] if value
                         else [(a * group + j, h, c) for j in range(group)])
            vals = [float(x[b, t, hh, d]) if t < length else 0.0 for t, hh, d in positions]
            low, high = min(vals), max(vals)
            step = max((high - low) / ((1 << bits) - 1), MIN_SCALE)
            saved_step = torch.tensor(step, dtype=torch.float16)
            saved_low = torch.tensor(low, dtype=torch.float16)
            scales.append(saved_step)
            offsets.append(saved_low)
            for (t, hh, d), v in zip(positions, vals):
                code = min((1 << bits) - 1, max(0, round((v - low) / step)))
                codes[b, t, hh, d] = code
                restored[b, t, hh, d] = code * float(saved_step) + float(saved_low)
    return codes, torch.stack(scales), torch.stack(offsets), restored[:, :length]


class GroupedCandidateTests(unittest.TestCase):
    def test_input_arithmetic_matches_legacy_key_and_isolated_value_groups(self):
        x = ((torch.arange(2 * 5 * 2 * 8).reshape(2, 5, 2, 8) * 7) % 37 - 18).to(torch.bfloat16)
        for bits in (2, 4):
            legacy = Quantizer("kivi", bits, bits, 4)
            candidate = KiviGroupedCandidate(bits, bits, 4, arithmetic="input")
            k, v = candidate.encode_pair(x, x)
            old_k = legacy.encode(x)
            for field in ("codes", "scale", "offset"):
                torch.testing.assert_close(getattr(k, field), getattr(old_k, field), rtol=0, atol=0)
            torch.testing.assert_close(k.restore(), old_k.restore(), rtol=0, atol=0)
            # Legacy V on one head and one D group has precisely the desired
            # local reduction. Compare each independently, trimming K-style
            # token padding imposed by the legacy container layout.
            for h, c in itertools.product(range(2), range(2)):
                isolated = legacy.encode(x[:, :, h:h+1, c*4:(c+1)*4], value=True)
                torch.testing.assert_close(v.codes[:, :, h, c],
                    isolated.codes.reshape(2, 8, 4)[:, :5], rtol=0, atol=0)
                for field in ("scale", "offset"):
                    expected = getattr(isolated, field).reshape(2, 8, 1)[:, :5]
                    torch.testing.assert_close(getattr(v, field)[:, :, h, c], expected, rtol=0, atol=0)
                torch.testing.assert_close(v.restore()[:, :, h, c*4:(c+1)*4],
                    isolated.restore()[:, :, 0], rtol=0, atol=0)
            z = torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16)
            for state in candidate.encode_pair(z, z):
                self.assertTrue((state.scale == 0).all())  # legacy FP16 underflow
                self.assertTrue(torch.isfinite(state.restore()).all())

    def test_scalar_reference_both_axes_bits_and_partial_tokens(self):
        # Integer values make FP32/scalar intermediate differences immaterial.
        x = ((torch.arange(2 * 5 * 2 * 8).reshape(2, 5, 2, 8) * 7) % 37 - 18).to(torch.bfloat16)
        for bits, value in itertools.product((2, 4), (False, True)):
            with self.subTest(bits=bits, value=value):
                q = KiviGroupedCandidate(bits, bits, 4).encode(x, value=value)
                codes, scales, offsets, restored = scalar_reference(x, bits, 4, value)
                torch.testing.assert_close(q.codes.reshape(codes.shape), codes, rtol=0, atol=0)
                torch.testing.assert_close(q.scale.flatten(), scales, rtol=0, atol=0)
                torch.testing.assert_close(q.offset.flatten(), offsets, rtol=0, atol=0)
                torch.testing.assert_close(q.restore(), restored, rtol=0, atol=0)
                self.assertEqual(q.restore().dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(q.restore()).all())

    def test_head_isolation(self):
        x = torch.arange(5 * 2 * 8).reshape(1, 5, 2, 8).to(torch.bfloat16)
        changed = x.clone()
        changed[:, :, 1] *= -17
        for value in (False, True):
            q = KiviGroupedCandidate(group=4)
            a, b = q.encode(x, value=value), q.encode(changed, value=value)
            head_axis = 2 if value else 3
            for field in ("codes", "scale", "offset"):
                torch.testing.assert_close(getattr(a, field).select(head_axis, 0),
                                           getattr(b, field).select(head_axis, 0), rtol=0, atol=0)
            torch.testing.assert_close(a.restore()[:, :, 0], b.restore()[:, :, 0], rtol=0, atol=0)

    def test_constant_zero_and_subnormal_scale(self):
        for bits, constant in itertools.product((2, 4), (0.0, -3.0, 7.0)):
            x = torch.full((1, 4, 2, 8), constant, dtype=torch.bfloat16)
            for q in KiviGroupedCandidate(bits, bits, 4).encode_pair(x, x):
                self.assertTrue((q.codes == 0).all())
                self.assertTrue((q.scale == MIN_SCALE).all())
                torch.testing.assert_close(q.restore(), x, rtol=0, atol=0)

    def test_partial_tail_zero_participates_and_is_removed(self):
        x = torch.full((1, 5, 1, 4), 6.0)
        k, v = KiviGroupedCandidate(group=4).encode_pair(x, x)
        self.assertEqual(k.codes.shape, (1, 2, 4, 1, 4))
        self.assertTrue((k.offset[:, -1] == 0).all())
        self.assertTrue((k.codes[:, -1, 1:] == 0).all())
        self.assertEqual(v.codes.numel(), x.numel())
        for q in (k, v):
            self.assertEqual(q.restore().shape, x.shape)
            torch.testing.assert_close(q.restore(), x.to(torch.bfloat16), rtol=0, atol=0)

    def test_actual_storage_including_metadata_padding(self):
        x = torch.ones(2, 5, 2, 8, dtype=torch.bfloat16)
        for bits in (2, 4):
            store = RollingStore(KiviGroupedCandidate(bits, bits, 4), capacity=5)
            store.state = store.quantizer.encode_pair(x, x)
            report = storage_bytes(store)
            # K: 256 codes + 64 scales/offsets; V: 160 + 40 each.
            self.assertEqual(report["valid_code_bytes"], 320)
            self.assertEqual(report["padding_code_bytes"], 96)
            self.assertEqual(report["scale_bytes"], 208)
            self.assertEqual(report["offset_bytes"], 208)
            self.assertEqual(report["resident_tensor_bytes"], 832)
            self.assertEqual(report["hypothetical_packed_bytes"], 416 + 416 * bits // 8)
            self.assertEqual(memory_report(store.state)["resident_tensor_bytes"], 832)
            self.assertEqual(sum(q.bytes()["resident"] for q in store.state), 832)
            self.assertEqual(memory_report([store.state, store.state])["resident_tensor_bytes"], 832)

    def test_invalid_inputs_and_metadata_overflow(self):
        q = KiviGroupedCandidate(group=4)
        for x in (torch.zeros(1, 0, 1, 4), torch.ones(1, 4),
                  torch.ones(1, 1, 1, 4, dtype=torch.int64),
                  torch.full((1, 1, 1, 4), float("nan")),
                  torch.full((1, 1, 1, 4), float("inf")),
                  torch.full((1, 1, 1, 4), 1e10)):
            with self.assertRaises(ValueError):
                q.encode(x)
        with self.assertRaises(ValueError):
            q.encode(torch.ones(1, 2, 2, 6), value=True)
        with self.assertRaises(ValueError):
            q.encode_pair(torch.ones(1, 2, 1, 4), torch.ones(1, 3, 1, 4))
        for kwargs in ({"key_bits": 3}, {"value_bits": True}, {"group": 0}):
            with self.assertRaises(ValueError):
                KiviGroupedCandidate(**kwargs)


if __name__ == "__main__":
    unittest.main()
