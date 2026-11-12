"""v6: cross-study panels, cohort tasks and cohort model input."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deconv'))
import simulator as sim
from simulator import SortedPanel, joint_panel, simulate_cohort, simulate_task

torch = pytest.importorskip('torch')


def study(name, types, n_donors, seed, n_cpg=150, offset=0.0):
    rng = np.random.default_rng(seed)
    base = np.random.default_rng(99).uniform(.1, .9, (n_cpg, 8))  # same biology in every study
    ids = [f'{name}_{t}_{d}' for t in types for d in range(n_donors)]
    cols = [base[:, i % 8] for i, _ in enumerate(types) for _ in range(n_donors)]
    data = np.clip(np.stack(cols, axis=1) + offset + rng.normal(0, .01, (n_cpg, len(ids))), 0, 1)
    return SortedPanel(pd.DataFrame(data, index=[f'cg{i}' for i in range(n_cpg)], columns=ids),
                       pd.Series({x: x.split('_')[1] for x in ids}),
                       pd.Series({x: f"{name}:{x.split('_')[2]}" for x in ids}), source=name)


def two_studies():
    a = study('A', ['CD19+', 'CD4+', 'CD8+', 'CD14+', 'Gran', 'Eos'], 4, 1)
    b = study('B', ['B', 'CD4T', 'CD8T', 'Mono', 'Gran'], 6, 2, offset=0.05)
    return joint_panel(a, b)


def test_joint_panel_keeps_common_types_and_study_groups():
    p = two_studies()
    cube, names, donors = p.cube()
    assert sorted(names) == ['B', 'CD4T', 'CD8T', 'Gran', 'Mono']
    assert cube.shape[1:] == (5, 10)
    assert dict(zip(*np.unique(p.groups(), return_counts=True))) == {'A': 4, 'B': 6}


def test_reference_study_follows_cross_study_flag():
    p = two_studies()
    for seed in range(6):
        within = simulate_task(p, np.random.default_rng(seed), n_markers=40, n_types=4, mixture_group='B')
        cross = simulate_task(p, np.random.default_rng(seed), n_markers=40, n_types=4, mixture_group='B',
                              cross_study=True)
        assert within.mixture_donors[0].startswith('B:') and cross.mixture_donors[0].startswith('B:')
        assert all(d.startswith('B:') for d in within.reference_donors)
        assert all(d.startswith('A:') for d in cross.reference_donors)
    # The bulk carries the other study's offset (+0.05), the reference does not.
    gap = cross.bulk - cross.reference_mean @ cross.fractions[:-1]
    assert gap.mean() > 0.03


def test_sharpen_groups_limits_the_stretch_to_one_study():
    p = two_studies()
    kw = dict(n_markers=40, n_types=4, mixture_group='B', cross_study=True, noise=0.0)
    plain = simulate_task(p, np.random.default_rng(3), **kw)
    only_a = simulate_task(p, np.random.default_rng(3), sharpen=(2.0, 2.0), sharpen_groups=('A',), **kw)
    both = simulate_task(p, np.random.default_rng(3), sharpen=(2.0, 2.0), **kw)
    assert not np.allclose(only_a.reference_mean, plain.reference_mean)   # reference is study A
    assert only_a.probe_names != both.probe_names or not np.allclose(only_a.bulk, both.bulk)


def test_cohort_shares_design_and_shift_but_not_fractions():
    p = two_studies()
    kw = dict(n_markers=60, n_types=4, missing=1, augment=True, mixture_group='B', cross_study=True,
              tech_shift=True, tech_cpg_m_sd=(0.4, 0.4), noise=0.0, sample_shift=None)
    tasks = simulate_cohort(p, np.random.default_rng(5), 6, tight_prob=0.0, **kw)
    assert len(tasks) == 6
    first = tasks[0]
    for t in tasks[1:]:
        assert np.array_equal(t.reference_mean, first.reference_mean)
        assert t.probe_names == first.probe_names and t.dropped_types == first.dropped_types
        assert not np.allclose(t.fractions, first.fractions)
        assert t.mixture_donors[0].startswith('B:') and t.mixture_donors[0] not in t.reference_donors
        assert np.isclose(t.fractions.sum(), 1)
    # Same technical shift for every sample: with the shift off the bulks differ by one
    # shared per-CpG pattern (up to donor noise), so the shift is strongly correlated.
    clean = simulate_cohort(p, np.random.default_rng(5), 6, tight_prob=0.0, **{**kw, 'tech_shift': False})
    shift = np.stack([a.bulk - b.bulk for a, b in zip(tasks, clean)])
    assert np.abs(shift).mean() > 0.01
    assert np.corrcoef(shift)[np.triu_indices(6, 1)].min() > 0.8


def test_tight_cohort_is_homogeneous():
    p = two_studies()
    kw = dict(n_markers=40, n_types=5, tight_concentration=(400.0, 400.0))
    tight = simulate_cohort(p, np.random.default_rng(8), 12, tight_prob=1.0, **kw)
    wide = simulate_cohort(p, np.random.default_rng(8), 12, tight_prob=0.0, **kw)
    spread = lambda ts: np.stack([t.fractions for t in ts]).std(axis=0).max()  # noqa: E731
    assert spread(tight) < 0.05 < spread(wide)


def test_cohort_is_reproducible():
    p = two_studies()
    kw = dict(n_markers=40, n_types=4, missing=1, augment=True, tech_shift=True)
    a = simulate_cohort(p, np.random.default_rng(11), 4, **kw)
    b = simulate_cohort(p, np.random.default_rng(11), 4, **kw)
    assert all(np.array_equal(x.bulk, y.bulk) and np.array_equal(x.fractions, y.fractions) for x, y in zip(a, b))


def test_panels_round_trip(tmp_path):
    p = two_studies()
    bundle = {'train': {'joint': p}, 'validation': {'joint': p}, 'pools': {'joint': np.arange(20)}}
    sim.save_panels(bundle, tmp_path / 'p.npz')
    back = sim.load_panels(tmp_path / 'p.npz')
    q = back['train']['joint']
    assert np.allclose(q.betas.to_numpy(), p.betas.to_numpy()) and list(q.betas.index) == list(p.betas.index)
    assert q.types.to_dict() == p.types.to_dict() and q.donors.to_dict() == p.donors.to_dict()
    assert np.array_equal(back['pools']['joint'], np.arange(20)) and q.source == p.source


def test_cohort_features_and_model_input():
    from model import Deconvolver, cohort_features
    from rpc_torch import rpc_fit
    gen = torch.Generator().manual_seed(0)
    ref = torch.rand(1, 50, 4, generator=gen).expand(6, 50, 4).contiguous()
    frac = torch.distributions.Dirichlet(torch.ones(4)).sample((6,))
    offset = 0.08 * torch.randn(50, generator=gen)
    bulk = (torch.einsum('nmk,nk->nm', ref, frac) + offset + 0.002 * torch.randn(6, 50, generator=gen)).clamp(0, 1)
    fit = rpc_fit(ref, bulk)
    ids = torch.tensor([0, 0, 0, 0, 0, 0])
    cell, task = cohort_features(fit, ids, torch.float32)
    assert cell.shape == (6, 50, 3) and task.shape == (6, 2)
    assert torch.allclose(cell[0, :, 0], cell[5, :, 0])           # the median is shared
    assert task[:, 0].min() > 0.8                                 # a shared offset explains the residual
    alone_cell, alone_task = cohort_features(fit, None, torch.float32)
    assert torch.all(alone_cell[..., 1:] == 0) and torch.allclose(alone_task[:, 0], torch.ones(6))
    assert torch.all(alone_task[:, 1] == 0)

    sd = torch.full_like(ref, 0.02)
    model = Deconvolver(width=16, heads=2, blocks=1, cohort=True)
    out = model(ref, sd, bulk, fit, ids)
    assert out.mean.shape == (6, 5) and torch.allclose(out.mean.sum(-1), torch.ones(6), atol=1e-5)
    out.log_prob(torch.cat((frac, torch.zeros(6, 1)), dim=-1)).mean().backward()
    assert model.input_projection.weight.grad is not None
    # Without cohort features the layer sizes are the v2-v5 ones.
    assert Deconvolver(width=16, heads=2, blocks=1).input_projection.in_features == 9
    assert model.input_projection.in_features == 12
