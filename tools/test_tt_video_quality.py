"""CPU checks for metric windows, empty-mask handling, JSON-safe identity PSNR."""
import json
import unittest
from tt_video_quality import window_indices,transition_pairs,summarize_frames,summarize_temporal

class QualityWindows(unittest.TestCase):
    def test_windows_and_pairs(self):
        w=window_indices()
        self.assertEqual([len(w[k]) for k in ('whole','late','boundary','late_boundary','inside','late_inside')],[501,252,41,21,460,231])
        self.assertEqual(w['late'][0],249)
        self.assertFalse(set(w['boundary'])&set(w['inside']))
        pairs=transition_pairs()
        self.assertEqual(sum(r['kind']=='boundary' for r in pairs),41)
        self.assertIn(dict(kind='boundary',chunk=21,i=248,j=249),pairs)
    def test_identity_psnr_and_transition_alignment(self):
        result=summarize_frames([70.]*501,list(range(500)),[0.]*501)
        self.assertEqual(result['late']['dino_subject'],sum(range(248,500))/252)
        self.assertEqual(result['whole']['pixel_mse'],0)
        self.assertTrue(result['whole']['pixel_psnr_infinite'])
        self.assertIsNone(result['whole']['pixel_psnr_db'])
        json.dumps(result,allow_nan=False)
    def test_undefined_not_dropped(self):
        rows=[dict(**r,warp_error=.1,coverage=1.,flow_pixels=.2,raw_difference=.3) for r in transition_pairs()]
        next(r for r in rows if r['kind']=='boundary' and r['chunk']==21).update(warp_error=None,coverage=0.)
        result=summarize_temporal(rows)
        self.assertIsNone(result['late_boundary']['warp_error'])
        self.assertEqual(result['late_boundary']['empty_mask_pairs'],1)
        self.assertEqual(result['whole_inside']['pairs'],84)
        self.assertAlmostEqual(result['late_inside']['warp_error'],.1)
if __name__=='__main__':unittest.main()
