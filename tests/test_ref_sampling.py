"""Legacy reference-donor sampling must reproduce the pre-2026-11-24 inline code draw for draw."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deconv"))
import unittest

import numpy as np

import simulator as sim


def old_inline(rng, eligible, mixture_d, matched):
    ref_draw = rng.choice(eligible, size=int(rng.integers(3, 7)), replace=True)
    if matched:
        ref_draw[0] = mixture_d
    return ref_draw


class RefSampling(unittest.TestCase):
    def test_legacy_identical(self):
        for seed in range(300):
            g = np.random.default_rng(seed)
            n = int(g.integers(1, 12)); elig = sorted(g.choice(20, n, replace=False).tolist())
            md = elig[0]; matched = bool(seed % 2)
            a, b = np.random.default_rng(seed + 1), np.random.default_rng(seed + 1)
            sim.REF_SAMPLING = 'legacy'
            np.testing.assert_array_equal(sim._draw_reference_donors(a, elig, md, matched=matched),
                                          old_inline(b, elig, md, matched))
            self.assertEqual(a.random(), b.random())  # same stream position afterwards

    def test_unique(self):
        sim.REF_SAMPLING = 'unique'
        try:
            for seed in range(300):
                g = np.random.default_rng(seed)
                n = int(g.integers(1, 12)); elig = sorted(g.choice(20, n, replace=False).tolist())
                md = elig[-1]; matched = bool(seed % 2)
                d = sim._draw_reference_donors(np.random.default_rng(seed), elig, md, matched=matched).tolist()
                self.assertEqual(len(d), len(set(d)))
                self.assertTrue(set(d) <= set(elig))
                self.assertLessEqual(len(d), min(6, len(elig)))
                self.assertGreaterEqual(len(d), min(3, len(elig)))
                if matched:
                    self.assertEqual(d[0], md)
        finally:
            sim.REF_SAMPLING = 'legacy'


if __name__ == '__main__':
    unittest.main()
