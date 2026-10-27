import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_policy_study import paired_inference,holm


class PairedAnalysisTests(unittest.TestCase):
    def test_exact_sign_flip_and_prompt_bootstrap(self):
        result=paired_inference([1.]*10,reps=100,level=.975)
        self.assertEqual(result['mean'],1.)
        self.assertEqual(result['ci_low'],1.)
        self.assertEqual(result['ci_high'],1.)
        self.assertEqual(result['p_exact_sign_flip'],2/1024)
        self.assertEqual(paired_inference([0.]*10,reps=100)['p_exact_sign_flip'],1.)

    def test_holm_preserves_input_order_and_monotonic_adjustment(self):
        np.testing.assert_allclose(holm([.04,.01]),[.04,.02])
        np.testing.assert_allclose(holm([.03,.04]),[.06,.06])

    def test_negative_effect_and_reproducibility(self):
        a=paired_inference([-3.,-2.,-1.],reps=1000,seed=17)
        b=paired_inference([-3.,-2.,-1.],reps=1000,seed=17)
        self.assertEqual(a,b);self.assertEqual(a['mean'],-2.)
        self.assertLess(a['ci_high'],0.)
        with self.assertRaises(ValueError):paired_inference([1.,float('nan')])

if __name__=='__main__':unittest.main()
