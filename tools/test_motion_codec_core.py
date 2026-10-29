"""CPU contracts for closed-loop residuals, escape regions, and payload rates."""
import unittest

import torch

from motion_codec_core import (decode_residual, encode_residual, make_anchor,
                               translate_reference)


class MotionCodecTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(113)
        self.reference = torch.randn(1, 8*8, 2, 4).to(torch.bfloat16)
        self.anchor = make_anchor(self.reference)
        self.reconstructed = self.anchor.restore()

    def test_lossless_closed_loop_identity_and_invalid_border(self):
        target = torch.randn_like(self.reference)
        for dy, dx in ((0, 0), (1, -2), (9, 0)):
            code = encode_residual(target, self.reconstructed, height=8, width=8,
                                   dy=dy, dx=dx, lossless=True)
            self.assertTrue(torch.equal(decode_residual(code, self.reconstructed), target))

    def test_zero_translation_matches_explicit_same_position(self):
        target = torch.randn_like(self.reference)
        code = encode_residual(target, self.reconstructed, height=8, width=8)
        explicit = make_anchor(target.float()-self.reconstructed.float())
        expected = (self.reconstructed.float()+explicit.restore()).to(target.dtype)
        self.assertTrue(torch.equal(code.residual.codes, explicit.codes))
        self.assertTrue(torch.equal(decode_residual(code, self.reconstructed), expected))

    def test_synthetic_translation_reduces_distortion(self):
        target = translate_reference(self.reconstructed, 8, 8, dy=1, dx=2)
        aligned = encode_residual(target, self.reconstructed, height=8, width=8, dy=1, dx=2, bits=2)
        same = encode_residual(target, self.reconstructed, height=8, width=8, bits=2)
        aligned_error = (decode_residual(aligned, self.reconstructed).float()-target.float()).norm()
        same_error = (decode_residual(same, self.reconstructed).float()-target.float()).norm()
        self.assertEqual(float(aligned_error), 0.)
        self.assertGreater(float(same_error), 0.)

    def test_border_is_absolute_code_without_wraparound(self):
        warped = translate_reference(self.reconstructed, 8, 8, dy=1, dx=-2).reshape(1, 8, 8, 2, 4)
        self.assertTrue(torch.equal(warped[:, 0], torch.zeros_like(warped[:, 0])))
        self.assertTrue(torch.equal(warped[:, :, -2:], torch.zeros_like(warped[:, :, -2:])))
        source = self.reconstructed.reshape(1, 8, 8, 2, 4)
        self.assertTrue(torch.equal(warped[:, 1:, :-2], source[:, :-1, 2:]))

    def test_closed_loop_reference_is_required_and_not_retained(self):
        target = self.reference + torch.tensor(.5, dtype=torch.bfloat16)
        code = encode_residual(target, self.reconstructed, height=8, width=8, lossless=True)
        self.assertTrue(torch.equal(decode_residual(code, self.reconstructed), target))
        self.assertFalse(torch.equal(decode_residual(code, self.reference), target))
        self.assertNotIn('reference', vars(code))

    def test_payload_accounting_and_fp16_scale_floor(self):
        zero = torch.zeros(1, 17, 2, 4)
        anchor = make_anchor(zero, bits=4, group=16)
        self.assertTrue(torch.isfinite(anchor.restore()).all())
        self.assertTrue(torch.equal(anchor.restore(), zero))
        self.assertEqual(anchor.bytes()['resident'], 32*2*4 + 2*2*4*2)
        self.assertEqual(anchor.bytes()['packed_estimate'], 32*2*4//2 + 2*2*4*2)
        code = encode_residual(self.reference, self.reconstructed, height=8, width=8)
        self.assertEqual(code.bytes()['map_bytes'], 4)
        self.assertEqual(code.bytes()['resident'], code.residual.bytes()['resident']+4)
        self.assertEqual(code.bytes()['reference_bytes_included'], 0)

    def test_reject_invalid_shape_nonfinite_and_metadata_overflow(self):
        with self.assertRaises(ValueError):
            make_anchor(torch.full((1, 2, 1, 1), float('nan')))
        with self.assertRaises(ValueError):
            make_anchor(torch.full((1, 2, 1, 1), 1e30))
        with self.assertRaises(ValueError):
            translate_reference(self.reference, 7, 8)


if __name__ == '__main__':
    unittest.main()
