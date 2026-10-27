import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from anchor_identity_gate import IdentityQuantizer
from kvq.lockstep import anchor_store

class AnchorIdentityTests(unittest.TestCase):
    def test_identity_encodings_do_not_alias_reference(self):
        raw=torch.arange(24).reshape(1,12,1,2).bfloat16()
        reference={'k':raw,'v':raw.clone(),'global_end_index':torch.tensor([24]),'local_end_index':torch.tensor([12])}
        quantizer=IdentityQuantizer();store=anchor_store(reference,quantizer,12,4)
        restored=store.segments[0]['state'][0].restore().clone()
        reference['k'].zero_()
        self.assertTrue(torch.equal(restored,store.segments[0]['state'][0].restore()))
        self.assertGreater(quantizer.encode_calls,0)

if __name__=='__main__':unittest.main()
