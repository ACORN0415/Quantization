import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import p3_tf_fr as pilot

class FakeVAE:
    model=SimpleNamespace(clear_cache=lambda:None)
    def to(self,*args,**kwargs):return self
    def decode_to_pixel(self,latent,**kwargs):return latent.float()

class RetentionTests(unittest.TestCase):
    def test_quality_preserves_exact_latent_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);before={}
            for name in ('bf16','tf','fr'):
                path=folder/f'{name}.pt'
                torch.save({'latent':torch.ones(1,3,3,2,2)},path)
                before[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(pilot,'gpu',torch.device('cpu')), \
                 patch('pyiqa.create_metric',return_value=lambda frame:torch.tensor([42.])), \
                 patch('torchvision.io.write_video',side_effect=lambda path,*args,**kwargs:Path(path).write_bytes(b'test-video')):
                pilot.quality(SimpleNamespace(vae=FakeVAE()),folder)
            for name,digest in before.items():
                self.assertEqual(hashlib.sha256((folder/f'{name}.pt').read_bytes()).hexdigest(),digest)
            manifest=json.loads((folder/'latent_manifest.json').read_text())
            self.assertTrue(manifest['retained']);self.assertEqual(len(manifest['artifacts']),3)

if __name__=='__main__':unittest.main()
