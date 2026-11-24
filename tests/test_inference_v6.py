"""Deterministic checks of v6 completeness reporting; no data are rescored."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deconv"))
import csv
import json
import math
from pathlib import Path
import unittest

import torch
from torch.distributions import Bernoulli, Beta, Dirichlet

from inference_v6 import estimate_v6
from model import Deconvolver, FractionPrediction


class InferenceTest(unittest.TestCase):
    def setUp(self):
        self.pred = FractionPrediction(Dirichlet(torch.tensor([[2., 3., 5.]])),
                                       Bernoulli(logits=torch.tensor([0.4])),
                                       Beta(torch.tensor([2.]), torch.tensor([6.])))

    def test_default_and_complete(self):
        result = estimate_v6(self.pred)
        torch.testing.assert_close(result.full_mixture, self.pred.mean)
        self.assertEqual(result.deployment_prevalence, .55)
        complete = estimate_v6(self.pred, known_complete=True)
        torch.testing.assert_close(complete.retained_full_mixture, self.pred.retained.mean)
        self.assertEqual(complete.other_full_mixture.item(), 0)
        torch.testing.assert_close(complete.full_mixture.sum(-1), torch.ones(1))

    def test_shift_and_rejection(self):
        result = estimate_v6(self.pred, deployment_prevalence=.2)
        expected = torch.sigmoid(self.pred.has_other.logits + math.log(.2/.8) - math.log(.55/.45))
        torch.testing.assert_close(result.missing_type_score, expected)
        torch.testing.assert_close(result.retained_composition, self.pred.retained.mean)
        torch.testing.assert_close(result.full_mixture.sum(-1), torch.ones(1))
        for pi in (0, 1, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                estimate_v6(self.pred, deployment_prevalence=pi)
        with self.assertRaises(ValueError):
            estimate_v6(self.pred, known_complete=True, deployment_prevalence=.2)

    def test_saved_sweep_outputs(self):
        path = Path(__file__).resolve().parents[1] / 'deconv' / 'results/prior_sweep_v6_nocohort_2026-11-18/wholeblood_per_sample.csv'
        if not path.exists():
            self.skipTest('saved sweep unavailable')
        with path.open(newline='') as f:
            row = next(csv.DictReader(f))
        comp = torch.tensor([[float(row[k]) for k in row if k.startswith('comp_')]], dtype=torch.float64)
        size = float(row['beta_mean_other'])
        pred = FractionPrediction(Dirichlet(comp), Bernoulli(logits=torch.tensor([float(row['logit_p_other_gt0'])], dtype=torch.float64)),
                                  Beta(torch.tensor([size], dtype=torch.float64), torch.tensor([1-size], dtype=torch.float64)))
        self.assertAlmostEqual(estimate_v6(pred).other_full_mixture.item(), float(row['other_model_pi055']), places=7)
        expected = 1 / (1 + math.exp(-(float(row['logit_p_other_gt0']) + math.log(.2/.8) - math.log(.55/.45)))) * size
        self.assertAlmostEqual(estimate_v6(pred, deployment_prevalence=.2).other_full_mixture.item(), expected, places=12)

    def test_checkpoint_on_synthetic_input(self):
        ckpt = Path(__file__).resolve().parents[1] / 'deconv' / 'results/gpu_v6_nocohort_seed0'
        if not (ckpt / 'model.pt').exists():
            self.skipTest('checkpoint unavailable')
        settings = json.loads((ckpt / 'summary.json').read_text())['settings']
        model = Deconvolver(width=settings['width'], heads=settings['heads'], blocks=settings['blocks'], cohort=False)
        model.load_state_dict(torch.load(ckpt / 'model.pt', map_location='cpu', weights_only=True))
        model.eval()
        torch.manual_seed(42)
        reference = torch.rand(2, 30, 3) * .8 + .1
        bulk = (reference * torch.tensor([.2, .3, .5])).sum(-1)
        with torch.inference_mode():
            prediction = model(reference, torch.full_like(reference, .02), bulk)
            torch.testing.assert_close(estimate_v6(prediction).full_mixture, prediction.mean)
            torch.testing.assert_close(estimate_v6(prediction, deployment_prevalence=.2).full_mixture.sum(-1), torch.ones(2))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
