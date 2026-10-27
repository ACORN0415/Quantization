import unittest
import torch
from kvq import Quantizer, RollingStore, ChunkStore


def take_keys(q, k, v):
    return k.clone()


class StoreTests(unittest.TestCase):
    def test_rolling_eviction_and_sink(self):
        store = RollingStore(Quantizer(), 12, 4)
        expected_sink = None
        for chunk in range(5):
            x = torch.full((1, 4, 1, 2), float(chunk + 1), dtype=torch.bfloat16)
            out = store.attend(x, x, x, chunk * 4, take_keys)
            self.assertEqual(store.global_end, (chunk + 1) * 4)
            self.assertEqual(store.local_end, min((chunk + 1) * 4, 12))
            if expected_sink is not None:
                self.assertTrue(torch.equal(out[:, :4], expected_sink))
            expected_sink = store.restore(x)[0][:, :4].clone()
            self.assertTrue(torch.equal(out[:, -4:], x))
        store.reset()
        self.assertIsNone(store.state)
        self.assertEqual(store.local_end, 0)

    def test_chunk_immutable_and_sink(self):
        store = ChunkStore(Quantizer(group=4), 12, 4)
        snapshots = {}
        for chunk in range(7):
            for phase in range(3):
                x = torch.randn(1, 4, 1, 2).bfloat16()
                out = store.attend(x, x, x, chunk * 4, take_keys)
                self.assertTrue(torch.equal(out[:, -4:], x))
                self.assertLessEqual(store.local_end, 12)
                for seg in store.segments:
                    codes = seg['state'][0].codes
                    if seg['start'] in snapshots:
                        self.assertTrue(torch.equal(codes, snapshots[seg['start']]))
                    else:
                        snapshots[seg['start']] = codes.clone()
            if chunk > 0:
                self.assertEqual(store.segments[0]['start'], 0)
        self.assertEqual([s['start'] for s in store.segments], [0, 20])
        store.reset()
        self.assertEqual(store.segments, [])
        self.assertIsNone(store.pending)

    def test_residual_live_prefix(self):
        store = RollingStore(Quantizer('kivi', 2, 2, group=4), 16, residual_tokens=4)
        x = torch.randn(1, 4, 2, 8).bfloat16()
        store.attend(x, x, x, 0, take_keys)
        k, v = store.restore(x)
        self.assertTrue(torch.equal(k[:, :4], x))
        self.assertTrue(torch.equal(v[:, :4], x))
        self.assertEqual(store.residual[:2], (0, 4))
        x2 = torch.randn_like(x)
        store.attend(x2, x2, x2, 4, take_keys)
        k, _ = store.restore(x)
        self.assertTrue(torch.equal(k[:, 4:8], x2))
        self.assertEqual(store.residual[:2], (4, 8))

    def test_chunk_residual(self):
        store = ChunkStore(Quantizer(group=4), 16, residual_tokens=4)
        for c in range(3):
            x = torch.randn(1, 4, 1, 2).bfloat16()
            store.attend(x, x, x, c * 4, take_keys)
        self.assertIsNotNone(store.segments[0]['state'])
        self.assertIsNotNone(store.segments[1]['raw'])

    def test_invalid_chunk_geometry(self):
        store = ChunkStore(Quantizer(), 10)
        x = torch.zeros(1, 4, 1, 2)
        with self.assertRaises(ValueError):
            store.attend(x, x, x, 0, take_keys)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
