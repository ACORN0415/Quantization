"""CPU-only contracts for new rank-sweep bookkeeping."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import tt_rank_quality as q

class Contracts(unittest.TestCase):
 def test_case_grid(self):
  cases=q.cases()
  self.assertEqual(len(cases),18)
  self.assertEqual(len({c['key'] for c in cases}),18)
  self.assertEqual({c['rank'] for c in cases},{64,128,192})
  self.assertTrue(all(c['prompt']==4 for c in q.cases(4)))
 def test_reject_rank_or_latent_change(self):
  c=q.cases()[0]
  with tempfile.TemporaryDirectory() as tmp,patch.object(q,'DATA',Path(tmp)):
   folder=Path(tmp)/c['key'];folder.mkdir();(folder/'latent.pt').write_bytes(b'fake tensor bytes')
   s=dict(source_sha256='seal',latent_sha256=q.quality.digest(folder/'latent.pt'),chunks=[{}]*42,prompt=0,bits=2,ranks=[64])
   path=folder/'sampling.json';path.write_text(json.dumps(s))
   q.inputs(c,'seal')
   s['ranks']=[128];path.write_text(json.dumps(s))
   with self.assertRaisesRegex(RuntimeError,'rank'):q.inputs(c,'seal')
   s['ranks']=[64];path.write_text(json.dumps(s));(folder/'latent.pt').write_bytes(b'changed')
   with self.assertRaisesRegex(RuntimeError,'artifact'):q.inputs(c,'seal')

if __name__=='__main__':unittest.main()
