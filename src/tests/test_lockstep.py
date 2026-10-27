import unittest
import torch
from kvq import Quantizer
from kvq.lockstep import anchor_store,plain_snapshot

class AnchoringTests(unittest.TestCase):
    def reference(self):
        x=torch.arange(24,dtype=torch.float32).reshape(1,12,1,2).bfloat16()
        return {'k':x,'v':x.clone(),'local_end_index':torch.tensor([12]),'global_end_index':torch.tensor([24])}

    def test_anchor_eviction_and_no_alias(self):
        ref=self.reference(); store=anchor_store(ref,Quantizer(group=4),12,4)
        self.assertEqual([s['start'] for s in store.segments],[12,16])
        self.assertEqual(store.pending_start,20)
        before=store.pending[0].clone();ref['k'].zero_()
        self.assertTrue(torch.equal(before,store.pending[0]))
        x=torch.ones(1,4,1,2,dtype=torch.bfloat16)
        store.attend(x,x,x,24,lambda q,k,v:k)
        self.assertEqual([s['start'] for s in store.segments],[16,20])

    def test_anchor_sink_and_snapshot(self):
        ref=self.reference(); store=anchor_store(ref,Quantizer(group=4),12,4,4)
        self.assertEqual([s['start'] for s in store.segments],[0,16])
        snapshot=plain_snapshot([ref]);ref['k'].zero_();ref['global_end_index'].zero_()
        self.assertGreater(snapshot[0]['k'].sum().item(),0)
        self.assertEqual(snapshot[0]['global_end_index'].item(),24)
        x=torch.ones(1,4,1,2,dtype=torch.bfloat16)
        store.attend(x,x,x,24,lambda q,k,v:k)
        self.assertEqual([s['start'] for s in store.segments],[0,20])

if __name__=='__main__': unittest.main()
