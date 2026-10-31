"""CPU regression checks for archival, read-only verification."""
from copy import deepcopy
from datetime import datetime, timezone
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch

import verify_moe_merge_record as verifier


class FutureClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2030, 1, 1, tzinfo=timezone.utc).astimezone(tz)


class SavedMergeAuditTests(unittest.TestCase):
    def test_audit_after_deadline_is_read_only(self):
        def snapshot():
            return {str(p.relative_to(verifier.RECORD)): verifier.digest(p)
                    for p in verifier.RECORD.rglob('*') if p.is_file()}

        before = snapshot()
        output = io.StringIO()
        with patch.object(verifier, 'datetime', FutureClock), \
             patch.object(Path, 'write_text', side_effect=AssertionError('Unexpected write')), \
             patch.object(Path, 'write_bytes', side_effect=AssertionError('Unexpected write')), \
             redirect_stdout(output):
            verifier.main()
        audit = json.loads(output.getvalue())
        self.assertEqual(audit['status'], 'PASS')
        self.assertEqual(audit['checked_at_utc'], '2030-01-01T00:00:00+00:00')
        self.assertTrue(audit['read_only'])
        self.assertTrue(audit['recorded_experiment_window']['cap_met'])
        self.assertEqual(audit['raw_files_verified'], 44)
        self.assertEqual(audit['output_arrays_verified'], 36)
        self.assertEqual(audit['agreement']['repeat_pairs_bitwise_equal'], 18)
        self.assertEqual(before, snapshot())

    def test_late_recorded_execution_is_still_rejected(self):
        def load(name):
            return json.loads((verifier.RECORD/name).read_text())

        reg = load('registration.json')
        result = load('result.json')
        completion = load('completion.json')
        for target in ('gpu_end', 'completion'):
            with self.subTest(target=target):
                bad_result, bad_completion = deepcopy(result), deepcopy(completion)
                if target == 'gpu_end':
                    bad_result['completed_at_utc'] = '2026-10-31T16:33:00+00:00'
                else:
                    bad_completion['completed_at_utc'] = '2026-10-31T16:33:00+00:00'
                with self.assertRaisesRegex(AssertionError, 'Recorded experiment'):
                    verifier.validate_recorded_window(reg, bad_result, bad_completion)


if __name__ == '__main__':
    unittest.main()
