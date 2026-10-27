"""Exercise final completeness checks and pairing without generating study data."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_policy_study as report


class ReportTests(unittest.TestCase):
    def test_full_report_known_effect_and_altered_reference_rejection(self):
        plan=json.loads((report.ROOT/'experiments/policy_study_plan.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'root';data=Path(directory)/'data';out=Path(directory)/'report'
            def put(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            put(root/'experiments/policy_study_plan.json',plan)
            put(root/'experiments/policy_study_registration.json',{'source_sha256':'fixture'})
            put(data/'queue.json',{'complete':80,'total':80,'source_sha256':'fixture'})
            raw=b'synthetic tensor placeholder; never used for inference'
            digest=hashlib.sha256(raw).hexdigest()
            base=dict(source_sha256='fixture',latent_file_sha256=digest,reference_latent_sha256='reference')
            history=[dict(chunk=c,read_history_tokens=0 if c==0 else 4680,
                          read_history_k_rel=.1,read_history_v_rel=.2)
                     for c in range(42) for _ in range(150)]
            for prompt in range(10):
                for seed in plan['seeds']:
                    reference=data/'reference'/f'p{prompt:02d}_s{seed}';reference.mkdir(parents=True)
                    (reference/'bf16.pt').write_bytes(raw)
                    put(reference/'bf16_sampling.json',dict(base,latent_sha256='reference'))
                    put(reference/'bf16_quality.json',dict(base,per_chunk=[dict(chunk=c,musiq=60.) for c in range(42)]))
                    for bits in plan['bits']:
                        for policy in plan['policies']:
                            name=f'p{prompt:02d}_s{seed}_{policy}_int{bits}';folder=data/'cases'/name
                            put(folder/'complete.json',dict(base))
                            for arm in ('tf','fr','noise'):
                                (folder/f'{arm}.pt').write_bytes(raw)
                                values=[{'chunk':c,'tf_rel' if arm=='tf' else 'rel':.5} for c in range(42)]
                                put(folder/f'{arm}_sampling.json',dict(base,rows=values,history=history))
                                improvement=(2. if bits==4 else -1.) if policy=='chunk' else 0.
                                score={'tf':59.,'fr':55.+improvement,'noise':50.}[arm]
                                put(folder/f'{arm}_quality.json',dict(base,per_chunk=[dict(chunk=c,musiq=score) for c in range(42)]))
            with patch.object(report,'ROOT',root):
                result=report.analyze(data,out)
                self.assertEqual([r['mean'] for r in result['primary']],[2.,-1.])
                self.assertTrue(all(r['n_prompts']==10 for r in result['primary']))
                self.assertEqual(len((out/'windows.csv').read_text().splitlines()),241)
                self.assertTrue((out/'RESULTS.md').exists())
                first=data/'reference'/f'p00_s{plan["seeds"][0]}'/'bf16.pt'
                first.write_bytes(b'altered')
                with self.assertRaisesRegex(RuntimeError,'BF16 tensor file altered'):
                    report.analyze(data,out)

if __name__=='__main__':unittest.main()
