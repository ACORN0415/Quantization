import tempfile,unittest
from pathlib import Path
import torch
from kvq import Quantizer,RollingStore,ChunkStore
from kvq.store import relative
from kvq.verification import write_progress


class ReviewRegressions(unittest.TestCase):
    def test_curated_report_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);curated=root/'docs/VERIFY.md';curated.parent.mkdir()
            curated.write_text('curated evidence')
            path=write_progress({'G1':{'state':'PASS'}},root/'data/verify')
            self.assertEqual(curated.read_text(),'curated evidence')
            self.assertEqual(path,root/'data/verify/progress.md')
            self.assertIn('PASS',path.read_text())

    def test_common_source_errors_match_actual_originals(self):
        for cls in (RollingStore,ChunkStore):
            store=cls(Quantizer(group=4,key_bits=2,value_bits=2),12)
            source=[]
            torch.manual_seed(42)
            for chunk in range(5):
                for call in range(3):
                    x=torch.randn(1,4,1,2).bfloat16();record={}
                    if call==0:
                        source.append(x.clone());source=source[-3:]
                    else:source[-1]=x.clone()
                    out=store.attend(x,x,x,chunk*4,lambda q,k,v:k.clone(),record=record)
                    if cls is RollingStore:actual=store.restore(x)[0][:,:store.local_end]
                    else:actual=out
                    expected=torch.cat(source,1)
                    self.assertAlmostEqual(record['source_k_rel'],relative(actual,expected),places=6)
                    self.assertAlmostEqual(record['source_v_rel'],relative(actual,expected),places=6)
                    self.assertIn('k_rel',record);self.assertIn('v_rel',record)
                    self.assertEqual(record['source_reference'],'insertion_bf16')
            store.reset()
            if cls is RollingStore:self.assertIsNone(store.source)

    def test_default_unsigned_and_explicit_legacy(self):
        x=torch.linspace(-1,1,16).reshape(1,16,1,1).bfloat16()
        current=Quantizer('kivi',8,8).encode(x)
        legacy=Quantizer('kivi',8,8,legacy_unsigned_int8=True).encode(x)
        self.assertEqual(current.codes.dtype,torch.uint8)
        self.assertEqual(legacy.codes.dtype,torch.int8)
        self.assertGreater(current.restore().max().item(),.9)
        self.assertFalse(torch.equal(current.restore(),legacy.restore()))


class PublicEvidenceTests(unittest.TestCase):
    def test_path_substitution_preserves_numbers_and_digests(self):
        from kvq.verification import public_snapshot
        raw={'path':'/tmp/person/project/script.py','data':'/tmp/person/data/log','score':.037,'sha256':'abc'}
        result=public_snapshot(raw,home=Path('/tmp/person'),project=Path('/tmp/person/project'),data=Path('/tmp/person/data'))
        self.assertEqual(result['path'],'${PROJECT_ROOT}/script.py')
        self.assertEqual(result['data'],'${DATA_ROOT}/log')
        self.assertEqual(result['score'],raw['score']);self.assertEqual(result['sha256'],'abc')

if __name__=='__main__':unittest.main()
