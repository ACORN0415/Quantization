import unittest
import torch
from kvq import Quantizer
from kvq.memory import memory_report


class MemoryTests(unittest.TestCase):
    def test_views_share_storage(self):
        x = torch.zeros(16, dtype=torch.bfloat16)
        report = memory_report([x, x[3:9], {'again': x}])
        self.assertEqual(report['resident_tensor_bytes'], 32)
        self.assertEqual(report['hypothetical_packed_bytes'], 32)

    def test_packed_is_separate_from_actual(self):
        state = Quantizer(key_bits=2).encode(torch.zeros(1, 16, 1, 4).bfloat16())
        report = memory_report([state, state])
        self.assertEqual(report['resident_tensor_bytes'], 64 + 8)
        self.assertEqual(report['hypothetical_packed_bytes'], 16 + 8)


if __name__ == '__main__': unittest.main()
