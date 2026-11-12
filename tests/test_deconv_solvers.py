"""Checks for the classical deconvolution solvers and the marker picker."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deconv"))

import reference  # noqa: E402
import solvers  # noqa: E402


def _problem(seed=0, m=120, k=6, n=8, noise=0.0):
    rng = np.random.default_rng(seed)
    ref = rng.beta(0.5, 0.5, size=(m, k))
    w = rng.dirichlet(np.ones(k), size=n)
    bulk = ref @ w.T + noise * rng.standard_normal((m, n))
    return ref, bulk, w


def test_all_solvers_recover_noiseless_fractions():
    ref, bulk, w = _problem()
    for fn in (solvers.nnls, solvers.cp_inequality, solvers.cp_equality):
        assert np.allclose(fn(ref, bulk), w, atol=1e-6), fn.__name__


def test_cp_equality_is_optimal_on_the_simplex():
    ref, bulk, _ = _problem(seed=1, noise=0.05)
    est = solvers.cp_equality(ref, bulk)
    assert np.allclose(est.sum(axis=1), 1.0) and (est >= 0).all()
    rng = np.random.default_rng(2)
    for j in range(bulk.shape[1]):
        best = np.sum((ref @ est[j] - bulk[:, j]) ** 2)
        for cand in rng.dirichlet(np.ones(ref.shape[1]), size=200):
            # Any feasible point on the segment toward a random simplex point is no better.
            near = 0.99 * est[j] + 0.01 * cand
            assert np.sum((ref @ near - bulk[:, j]) ** 2) >= best - 1e-10


def test_cp_inequality_leaves_mass_unassigned_when_a_type_is_missing():
    ref, bulk, w = _problem(seed=3, noise=0.0)
    est = solvers.cp_inequality(ref[:, :-1], bulk)
    assert (est.sum(axis=1) <= 1.0 + 1e-9).all()
    assert (est >= 0).all()


def test_matches_epidish_cp():
    if not Path(solvers.RSCRIPT).exists():
        pytest.skip("Rscript not available")
    ref, bulk, _ = _problem(seed=4, noise=0.03)
    probes = [f"cg{i:08d}" for i in range(ref.shape[0])]
    ref_df = pd.DataFrame(ref, index=probes, columns=[f"T{i}" for i in range(ref.shape[1])])
    bulk_df = pd.DataFrame(bulk, index=probes, columns=[f"S{i}" for i in range(bulk.shape[1])])
    for constraint, fn in (("inequality", solvers.cp_inequality), ("equality", solvers.cp_equality)):
        theirs = solvers.epidish(ref_df, bulk_df, "CP", constraint).to_numpy()
        assert np.allclose(fn(ref, bulk), theirs, atol=1e-4), constraint


def test_paper_solvers_recover_fractions_and_respect_their_scales():
    if not Path(solvers.RSCRIPT).exists():
        pytest.skip("Rscript not available")
    ref, bulk, w = _problem(seed=6, m=150, k=5, n=6, noise=0.01)
    bulk = np.clip(bulk, 0.001, 0.999)
    probes = [f"cg{i:08d}" for i in range(ref.shape[0])]
    ref_df = pd.DataFrame(ref, index=probes, columns=[f"T{i}" for i in range(ref.shape[1])])
    bulk_df = pd.DataFrame(bulk, index=probes, columns=[f"S{i}" for i in range(bulk.shape[1])])
    out = solvers.paper_solvers(ref_df, bulk_df)
    assert not solvers.PAPER_ERRORS, solvers.PAPER_ERRORS
    assert set(out) == set(solvers.PAPER_METHODS)
    for name, est in out.items():
        assert list(est.index) == list(bulk_df.columns) and list(est.columns) == list(ref_df.columns)
        assert np.sqrt(np.mean((est.to_numpy() - w) ** 2)) < 0.02, name
    for name in ("EMeth_Laplace", "EMeth_Normal", "EMeth_Binom"):
        assert np.allclose(out[name].sum(axis=1), 1.0, atol=1e-6), name
    assert (out["ICeDT"].sum(axis=1) <= 1.0 + 1e-6).all()


def test_pick_markers_finds_planted_probes():
    rng = np.random.default_rng(5)
    labels = pd.Series(np.repeat(["A", "B", "C"], 6), index=[f"s{i}" for i in range(18)])
    x = 0.5 + 0.02 * rng.standard_normal((500, 18))
    x[:10, :6] += 0.4   # hyper in A
    x[10:20, 6:12] -= 0.4  # hypo in B
    betas = pd.DataFrame(x, index=[f"cg{i:08d}" for i in range(500)], columns=labels.index)
    markers = reference.pick_markers(betas, labels, n_per_side=10)
    assert set(betas.index[:20]) <= set(markers)
    mean, sd = reference.build_reference(betas, labels, markers)
    assert list(mean.columns) == ["A", "B", "C"] and mean.shape == sd.shape
