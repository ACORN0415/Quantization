"""Recompute frozen analysis into a temporary directory; preserve edited reports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]
DATA=Path.home()/'sfq_data/policy_supplement'
OUT=ROOT/'src/experiments/policy_supplement_results'


def snapshot():
    paths=list(OUT.rglob('*'))+[ROOT/'src/experiments/policy_study_results/REPORT_KO.md']
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def main():
    source=ROOT/'tools/evaluate_policy_supplement.py'
    spec=importlib.util.spec_from_file_location('frozen_supplement',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    before=snapshot()
    with tempfile.TemporaryDirectory(prefix='policy-reanalysis-') as folder:
        target=Path(folder);module.analyze(DATA,target)
        if json.loads((target/'summary.json').read_text())!=json.loads((OUT/'summary.json').read_text()):
            raise RuntimeError('Recomputed supplementary summary differs from publication')
        if (target/'windows.csv').read_bytes()!=(OUT/'windows.csv').read_bytes():
            raise RuntimeError('Recomputed windows differ from publication')
    if before!=snapshot():raise RuntimeError('Published artifacts changed during reanalysis')
    print('PASS all supplementary estimates and intervals reproduced; published reports and figures unchanged')


if __name__=='__main__':main()
