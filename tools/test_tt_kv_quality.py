"""CPU contracts: distinct direct-KV controls and TT cases, strict rank identity."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import tt_kv_quality as q

class Contracts(unittest.TestCase):
 def test_grid_has_matched_controls(self):
  cases=q.cases();self.assertEqual(len(cases),18);self.assertEqual(len({c['key'] for c in cases}),18)
  for p in (0,4):
   for b in (2,3,4):
    group=[c for c in cases if c['prompt']==p and c['bits']==b]
    self.assertEqual({c['rank'] for c in group},{0,192,299})
    self.assertEqual(sum(c['arm']=='quant' for c in group),1)
 def test_direct_and_tt_metadata_rejected_if_mixed(self):
  for c in q.cases()[:3]:
   with tempfile.TemporaryDirectory() as tmp,patch.object(q,'DATA',Path(tmp)):
    folder=Path(tmp)/c['key'];folder.mkdir();(folder/'latent.pt').write_bytes(b'fake tensor')
    s=dict(source_sha256='seal',latent_sha256=q.quality.digest(folder/'latent.pt'),chunks=[{}]*42,prompt=0,bits=2,arm=c['arm'],rank=c['rank'],ranks=[] if c['rank']==0 else [c['rank']])
    path=folder/'sampling.json';path.write_text(json.dumps(s));q.inputs(c,'seal')
    s['ranks']=[128];path.write_text(json.dumps(s))
    with self.assertRaisesRegex(RuntimeError,'rank'):q.inputs(c,'seal')
    s['ranks']=[] if c['rank']==0 else [c['rank']];s['arm']='wrong';path.write_text(json.dumps(s))
    with self.assertRaisesRegex(RuntimeError,'metadata'):q.inputs(c,'seal')
if __name__=='__main__':unittest.main()
