"""Synthetic CPU contracts; no research captures or GPU are inspected."""
import unittest

import torch

from dense_motion_core import (AffineStats, choose_map, gather_reference, map_bytes)
from motion_codec_core import make_anchor


class DenseMotionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(731)
        self.height, self.width = 6, 7
        self.reference = torch.randn(2, self.height*self.width, 2, 4)

    def test_identity_and_tie_preference(self):
        for reference in (self.reference, torch.ones_like(self.reference)):
            offsets = choose_map(reference, reference, self.height, self.width)
            self.assertTrue(torch.equal(offsets, torch.zeros_like(offsets)))
            self.assertTrue(torch.equal(gather_reference(reference, offsets, self.height, self.width), reference))

    def test_nonrigid_local_correspondence(self):
        # Left/right interior halves move toward opposite neighbors; borders stay.
        offsets = torch.zeros(2, self.height*self.width, 2, dtype=torch.int8)
        grid = offsets.reshape(2, self.height, self.width, 2)
        grid[:, :, 1:3, 1] = -1
        grid[:, :, 3:-1, 1] = 1
        target = gather_reference(self.reference, offsets, self.height, self.width)
        selected = choose_map(target, self.reference, self.height, self.width)
        reconstructed = gather_reference(self.reference, selected, self.height, self.width)
        self.assertTrue(torch.equal(reconstructed, target))
        self.assertGreater(float((self.reference-target).square().sum()), 0.)
        self.assertTrue(torch.equal(selected, offsets))

    def test_border_candidates_never_wrap_or_escape(self):
        target = -self.reference
        offsets = choose_map(target, self.reference, self.height, self.width)
        y = torch.arange(self.height).repeat_interleave(self.width)[None]
        x = torch.arange(self.width).repeat(self.height)[None]
        sy, sx = y+offsets[..., 0], x+offsets[..., 1]
        self.assertTrue(bool(((sy >= 0) & (sy < self.height) & (sx >= 0) & (sx < self.width)).all()))
        gathered = gather_reference(self.reference, offsets, self.height, self.width)
        for batch in range(2):
            for index in range(self.height*self.width):
                src = int(sy[batch,index])*self.width+int(sx[batch,index])
                self.assertTrue(torch.equal(gathered[batch,index],self.reference[batch,src]))
        bad = offsets.clone(); bad[:, 0, 0] = -1
        with self.assertRaises(ValueError): gather_reference(self.reference, bad, self.height,self.width)

    def test_map_and_coefficient_payload_accounting(self):
        offsets = torch.zeros(2, self.height*self.width, 2, dtype=torch.int8)
        self.assertEqual(map_bytes(offsets), 2*2*self.height*self.width)
        stats = AffineStats().update(self.reference, 1.5*self.reference-.25)
        predictor = stats.fit(ridge=0)
        self.assertEqual(predictor.bytes()['resident'], 2*4*2*2)
        self.assertEqual(predictor.a.dtype, torch.float16)
        self.assertEqual(predictor.b.dtype, torch.float16)

    def test_streaming_train_only_fit_snapshot_and_evaluation(self):
        # Coefficients see reconstructed training references, never evaluation targets.
        reconstructed = make_anchor(self.reference).restore()
        target = 1.5*reconstructed-.25
        stats = AffineStats()
        stats.update(reconstructed[:1], target[:1]); stats.update(reconstructed[1:], target[1:])
        predictor = stats.fit(ridge=0)
        evaluation = torch.randn_like(self.reference)
        self.assertTrue(torch.equal(predictor.predict(evaluation), 1.5*evaluation-.25))
        a = predictor.a.clone(); b = predictor.b.clone()
        stats.update(evaluation, -4*evaluation+2)
        self.assertTrue(torch.equal(predictor.a,a))
        self.assertTrue(torch.equal(predictor.b,b))
        self.assertEqual(predictor.training_tokens, 2*self.height*self.width)
        self.assertFalse(torch.equal(stats.fit(ridge=0).a,a))

    def test_degenerate_features_finite_and_rejections(self):
        predictor = AffineStats().update(torch.ones_like(self.reference), 3*torch.ones_like(self.reference)).fit(ridge=0)
        self.assertTrue(torch.equal(predictor.predict(self.reference), 3*torch.ones_like(self.reference)))
        with self.assertRaises(ValueError): AffineStats().fit()
        with self.assertRaises(ValueError): choose_map(self.reference,self.reference,6,8)
        with self.assertRaises(ValueError): choose_map(self.reference,self.reference,6,7,radius=128)
        with self.assertRaises(ValueError): map_bytes(torch.zeros(2,42,2))
        with self.assertRaises(ValueError): choose_map(self.reference*float('nan'),self.reference,6,7)
        with self.assertRaises(ValueError): choose_map(torch.full_like(self.reference,1e30),self.reference,6,7)


if __name__ == '__main__': unittest.main()
