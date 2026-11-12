"""Check the variable-size model and the batch RPC feature path."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deconv"))

import solvers


def test_model_preserves_fraction_order_and_backpropagates():
    torch = pytest.importorskip("torch")
    from model import Deconvolver

    torch.manual_seed(123)
    model = Deconvolver(width=8, heads=2, blocks=1)
    torch.nn.init.normal_(model.type_head.weight, std=0.1)
    reference = torch.rand(2, 20, 12)
    sd = 0.05 * torch.rand_like(reference)
    bulk = torch.rand(2, 20)
    output = model(reference, sd, bulk)
    types = torch.randperm(12)
    probes = torch.randperm(20)
    shuffled = model(reference[:, probes][:, :, types], sd[:, probes][:, :, types], bulk[:, probes])

    torch.testing.assert_close(shuffled.mean[:, :-1], output.mean[:, types], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(shuffled.mean[:, -1], output.mean[:, -1], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(output.mean.sum(dim=-1), torch.ones(2))
    draws = output.sample(8)
    assert (draws >= 0).all()
    torch.testing.assert_close(draws.sum(dim=-1), torch.ones(8, 2))
    base = torch.softmax(torch.rand(2, 12), dim=-1)
    truth = torch.cat((base * torch.tensor([[1.0], [0.8]]), torch.tensor([[0.0], [0.2]])), dim=-1)
    loss = -output.log_prob(truth).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_torch_rpc_recovers_noiseless_fractions_and_uses_r_median():
    pytest.importorskip("torch")
    import rpc_torch

    rng = np.random.default_rng(4)
    ref = rng.beta(0.5, 0.5, size=(3, 60, 5))
    w = rng.dirichlet(np.ones(5), size=3)
    bulk = np.einsum("tmk,tk->tm", ref, w)
    np.testing.assert_allclose(rpc_torch.rpc(ref, bulk), w, atol=1e-8)
    import torch
    assert float(rpc_torch._median(torch.tensor([1.0, 2.0, 3.0, 10.0]))) == 2.5


def test_torch_rpc_matches_epidish():
    pytest.importorskip("torch")
    import rpc_torch

    if not Path(solvers.RSCRIPT).exists():
        pytest.skip("Rscript not available")
    rng = np.random.default_rng(321)
    ref = rng.random((4, 80, 6))
    bulk = (ref * rng.dirichlet(np.ones(6), size=4)[:, None, :]).sum(axis=2)
    bulk += 0.02 * rng.standard_normal(bulk.shape)
    bulk[:, :5] += 0.3  # outliers so the Huber weights matter
    np.testing.assert_allclose(rpc_torch.rpc(ref, bulk), solvers.rpc_batch(ref, bulk), atol=1e-8)


def test_batch_rpc_matches_single_reference_calls():
    if not Path(solvers.RSCRIPT).exists():
        pytest.skip("Rscript not available")
    rng = np.random.default_rng(123)
    ref = rng.random((2, 80, 4))
    bulk = (ref * rng.dirichlet(np.ones(4), size=2)[:, None, :]).sum(axis=2)
    bulk += 0.02 * rng.standard_normal(bulk.shape)
    probes = [f"cg{i:08d}" for i in range(ref.shape[1])]
    types = [f"type{i}" for i in range(ref.shape[2])]
    expected = np.concatenate([
        solvers.epidish(
            pd.DataFrame(r, index=probes, columns=types),
            pd.DataFrame(b, index=probes, columns=["bulk"]),
        ).to_numpy()
        for r, b in zip(ref, bulk)
    ])
    np.testing.assert_allclose(solvers.rpc_batch(ref, bulk), expected, atol=1e-10)
