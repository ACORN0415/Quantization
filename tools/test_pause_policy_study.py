"""Real-process boundary test; never touches the running experiment."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('pause_control',Path(__file__).with_name('pause_policy_study.py'))
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)


class BoundaryTests(unittest.TestCase):
    def test_atomic_boundary_stops_worker_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'repo';data=Path(temp)/'data/policy_study';data.mkdir(parents=True)
            def put(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            put(root/'src/experiments/policy_study_registration.json',{'source_sha256':'fixture'})
            put(data/'queue.json',{'complete':0,'total':80})
            folder=data/'cases/case0';folder.mkdir(parents=True)
            put(folder/'complete.json',{'case':'case0','source_sha256':'fixture','noise_nonfinite':False})
            (folder/'tf.pt').write_bytes(b'fixture')
            put(folder/'latent_manifest.json',{'artifacts':[{'path':'tf.pt','bytes':7,'sha256':'fixture'}]})
            worker_path=Path(temp)/'policy_study.py'
            worker_path.write_text('''import json,time,sys
from pathlib import Path
p=Path(sys.argv[1])
time.sleep(.5)
t=p/'queue.tmp';t.write_text(json.dumps({'complete':1,'total':80}));t.replace(p/'queue.json')
time.sleep(.15)
(p/'unexpected_next_case').write_text('should never run')
time.sleep(30)
''')
            supervisor_path=Path(temp)/'continue_policy_study.py'
            supervisor_path.write_text('import time;time.sleep(30)')
            worker=subprocess.Popen([sys.executable,str(worker_path),str(data)],start_new_session=True)
            supervisor=subprocess.Popen([sys.executable,str(supervisor_path)],start_new_session=True)
            put(data/'status.json',{'study':{'pid':worker.pid,'state':'running'}})
            (data.parent/'policy_study_pipeline.pid').write_text(str(supervisor.pid))
            try:
                with patch.object(control,'REPO',root),patch.object(control.subprocess,'run'):
                    control.pause(data,1)
                worker.wait(timeout=3);supervisor.wait(timeout=3)
                self.assertFalse((data/'unexpected_next_case').exists())
                self.assertEqual(json.loads((data/'queue.json').read_text())['complete'],1)
                self.assertEqual(json.loads((data/'status.json').read_text())['study']['state'],'paused')
                self.assertEqual(json.loads((data/'pause_request.json').read_text())['remaining_cases'],79)
                self.assertTrue((folder/'tf.pt').exists())
                self.assertTrue((root/'src/experiments/policy_study_midpoint/REPORT.md').exists())
            finally:
                for process in (worker,supervisor):
                    if process.poll() is None:process.kill()
                    process.wait()

if __name__=='__main__':unittest.main()
