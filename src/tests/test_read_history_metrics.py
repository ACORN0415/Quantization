import unittest
import torch
from kvq import Quantizer,RollingStore,ChunkStore
from kvq.store import relative

class HistoryMetricsTests(unittest.TestCase):
    def test_both_policies_measure_only_the_history_actually_read(self):
        for policy in (RollingStore,ChunkStore):
            torch.manual_seed(132)
            store=policy(Quantizer(key_bits=2,value_bits=2,group=4),12)
            originals=[]
            for chunk in range(5):
                for step in range(3):
                    x=torch.randn(1,4,1,2).bfloat16()
                    if step==0:originals.append(x.clone());originals=originals[-3:]
                    else:originals[-1]=x.clone()
                    record={};out=store.attend(x,x,x,chunk*4,lambda q,k,v:k.clone(),record=record)
                    expected=relative(out[:,:-4],torch.cat(originals[:-1],1)) if len(originals)>1 else 0.
                    self.assertAlmostEqual(record['k_rel'],expected,places=6)
                    self.assertAlmostEqual(record['v_rel'],expected,places=6)
                    self.assertEqual(record['read_history_tokens'],out.shape[1]-4)
                    self.assertEqual(record['kv_error_phase'],'pre_attention_history')
                    if chunk==0:self.assertEqual(record['k_rel'],0.)
            if policy is RollingStore:self.assertIn('legacy_post_k_rel',record)
            else:self.assertNotIn('legacy_post_k_rel',record)


class CurrentTokenExclusionTests(unittest.TestCase):
    def test_rolling_current_quantization_does_not_enter_read_error(self):
        torch.manual_seed(90)
        x=torch.randn(1,4,1,4).bfloat16();record={}
        store=RollingStore(Quantizer(key_bits=2,value_bits=2,group=4),12)
        store.attend(x,x,x,0,lambda q,k,v:k,record=record)
        self.assertEqual(record['k_rel'],0.)
        self.assertGreater(record['legacy_post_k_rel'],0.)
        self.assertGreater(record['source_k_rel'],0.)

if __name__=='__main__':unittest.main()
