import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deconv'))
from diagnostics import cetygo, hull_fit


def _ref(seed=0, m=200, k=4):
    return np.random.default_rng(seed).uniform(0.05, 0.95, size=(m, k))


def test_hull_fit_recovers_point_inside_hull():
    ref = _ref()
    w_true = np.array([0.5, 0.3, 0.2, 0.0])
    w, res = hull_fit(ref, ref @ w_true)
    assert np.allclose(w, w_true, atol=1e-4)
    assert np.sqrt((res ** 2).mean()) < 1e-4
    assert np.isclose(w.sum(), 1.0)


def test_hull_fit_reports_distance_outside_hull():
    ref = _ref()
    sig = np.random.default_rng(1).uniform(0.05, 0.95, size=ref.shape[0])
    w, res = hull_fit(ref, sig)
    assert (w >= 0).all() and np.isclose(w.sum(), 1.0)
    assert np.sqrt((res ** 2).mean()) > 0.1


def test_cetygo_zero_for_complete_reference_and_grows_with_omission():
    full = _ref(k=5)
    frac = np.random.default_rng(2).dirichlet(np.ones(5), size=8)
    bulk = full @ frac.T
    assert np.allclose(cetygo(full, bulk), 0, atol=1e-5)
    # Omit the last type: the error rises with its share.
    err = cetygo(full[:, :4], bulk)
    assert err.min() > 0
    assert np.corrcoef(err, frac[:, 4])[0, 1] > 0.8


def test_identical_signatures_are_undetectable():
    # The audit's counterexample: C halfway between A and B leaves no residual.
    ref = _ref(k=2)
    c = ref.mean(axis=1)
    bulk = (0.4 * ref[:, 0] + 0.4 * ref[:, 1] + 0.2 * c)[:, None]
    assert cetygo(ref, bulk)[0] < 1e-6
    assert np.sqrt((hull_fit(ref, c)[1] ** 2).mean()) < 1e-6
