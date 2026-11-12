import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deconv'))
from simulator import SortedPanel, split_donors, simulate_task


def panel():
    rng = np.random.default_rng(2)
    types = ['Neu', 'CD4', 'CD8', 'B']
    ids = [f'{t}:{d}' for t in types for d in range(6)]
    data = rng.uniform(.1, .9, (120, len(ids)))
    return SortedPanel(pd.DataFrame(data, index=[f'cg{i}' for i in range(120)], columns=ids),
                       pd.Series({x: x.split(':')[0] for x in ids}),
                       pd.Series({x: x.split(':')[1] for x in ids}))


def test_split_and_mismatch():
    train, val = split_donors(panel(), seed=1)
    assert set(train.donors).isdisjoint(set(val.donors))
    for p in [train, val]:
        t = simulate_task(p, np.random.default_rng(7), n_markers=50, n_types=3, donor_mismatch=True)
        assert set(t.reference_donors).isdisjoint(t.mixture_donors)
        assert len(t.reference_donors) <= 2
        assert t.reference_mean.shape == t.reference_sd.shape == (50, 3)
        assert t.bulk.shape == (50,)
        assert t.fractions.shape == (4,)
        assert np.isclose(t.fractions.sum(), 1)
        assert t.fractions[-1] == 0
        assert len(set(t.probe_names)) == 50


def test_missing_and_augmented():
    t = simulate_task(panel(), np.random.default_rng(9), n_markers=100, n_types=12,
                      missing=1, augment=True)
    assert t.reference_mean.shape == (100, 12)
    assert t.fractions[-1] > 0
    assert len(t.dropped_types) == 1
    assert len(t.augmented_types) == 9
    assert np.isfinite(t.reference_sd).all()
    assert np.all((t.bulk >= 0) & (t.bulk <= 1))
    with pytest.raises(ValueError):
        simulate_task(panel(), np.random.default_rng(1), n_types=12)


def test_reproducible_and_noise_free_mixture():
    p = panel()
    a = simulate_task(p, np.random.default_rng(4), n_markers=50, n_types=3, noise=0)
    b = simulate_task(p, np.random.default_rng(4), n_markers=50, n_types=3, noise=0)
    np.testing.assert_array_equal(a.bulk, b.bulk)
    np.testing.assert_array_equal(a.fractions, b.fractions)
    assert a.fractions.min() <= .05
    assert set(a.mixture_donors).issubset(a.reference_donors)
    assert a.preprocessing.endswith('not Noob')


def test_informative_markers_exclude_hidden_data():
    p = panel()
    kw = dict(n_markers=30, n_types=3, missing=1, donor_mismatch=True, marker_mode="informative")
    a = simulate_task(p, np.random.default_rng(81), **kw)
    modified = panel()
    hidden = (modified.types == a.dropped_types[0]) | modified.donors.isin(a.mixture_donors)
    modified.betas.loc[:, hidden] = 0.99
    b = simulate_task(modified, np.random.default_rng(81), **kw)
    assert a.probe_names == b.probe_names
    np.testing.assert_allclose(a.reference_mean, b.reference_mean)
    assert not np.allclose(a.bulk, b.bulk)


def test_sparse_subtypes_change_few_cpgs():
    p = panel()
    t = simulate_task(p, np.random.default_rng(3), n_markers=120, n_types=8, augment=True,
                      marker_mode="random", noise=0)
    real = [i for i, n in enumerate(t.type_names) if not n.startswith('synthetic')]
    syn = [i for i, n in enumerate(t.type_names) if n.startswith('synthetic')]
    assert syn
    for j in syn:
        gaps = [np.mean(np.abs(t.reference_mean[:, j] - t.reference_mean[:, i]) > 0.15) for i in real]
        assert min(gaps) < 0.1  # a near copy of its parent
    with pytest.raises(ValueError):
        simulate_task(p, np.random.default_rng(3), n_types=3, synthetic='other')


def test_contrast_markers_exclude_hidden_data_and_zero_type():
    p = panel()
    kw = dict(n_markers=30, n_types=3, missing=1, donor_mismatch=True, marker_mode="contrast")
    a = simulate_task(p, np.random.default_rng(82), **kw)
    modified = panel()
    hidden = (modified.types == a.dropped_types[0]) | modified.donors.isin(a.mixture_donors)
    modified.betas.loc[:, hidden] = 0.99
    b = simulate_task(modified, np.random.default_rng(82), **kw)
    assert a.probe_names == b.probe_names and len(set(a.probe_names)) == 30
    np.testing.assert_allclose(a.reference_mean, b.reference_mean)
    z = simulate_task(p, np.random.default_rng(5), n_markers=30, n_types=3, zero_type_prob=1.0)
    assert (z.fractions[:-1] == 0).sum() >= 1 and np.isclose(z.fractions.sum(), 1)


def test_tech_shift_off_is_default_and_draws_nothing_extra():
    p = panel()
    kw = dict(n_markers=40, n_types=3, missing=1, donor_mismatch=True, noise=0.03)
    default = simulate_task(p, np.random.default_rng(21), **kw)
    off = simulate_task(p, np.random.default_rng(21), tech_shift=False, **kw)
    np.testing.assert_array_equal(default.bulk, off.bulk)
    np.testing.assert_array_equal(default.reference_mean, off.reference_mean)
    np.testing.assert_array_equal(default.fractions, off.fractions)


def test_tech_shift_changes_only_the_bulk():
    p = panel()
    kw = dict(n_markers=60, n_types=3, missing=1, donor_mismatch=True, augment=True, noise=0.02)
    plain = simulate_task(p, np.random.default_rng(22), **kw)
    shifted = simulate_task(p, np.random.default_rng(22), tech_shift=True,
                            tech_global=(0.03, 0.04), tech_batch=(0.03, 0.04), **kw)
    np.testing.assert_array_equal(plain.reference_mean, shifted.reference_mean)
    np.testing.assert_array_equal(plain.reference_sd, shifted.reference_sd)
    np.testing.assert_array_equal(plain.fractions, shifted.fractions)
    assert plain.probe_names == shifted.probe_names
    assert plain.dropped_types == shifted.dropped_types
    assert not np.allclose(plain.bulk, shifted.bulk)
    assert np.all((shifted.bulk >= 0) & (shifted.bulk <= 1))


def test_tech_shift_reproducible_and_seed_dependent():
    p = panel()
    kw = dict(n_markers=60, n_types=4, tech_shift=True)
    a = simulate_task(p, np.random.default_rng(23), **kw)
    b = simulate_task(p, np.random.default_rng(23), **kw)
    c = simulate_task(p, np.random.default_rng(24), **kw)
    np.testing.assert_array_equal(a.bulk, b.bulk)
    assert not np.allclose(a.bulk, c.bulk)


def test_tech_shift_zero_ranges_is_identity():
    p = panel()
    kw = dict(n_markers=50, n_types=3, noise=0)
    plain = simulate_task(p, np.random.default_rng(25), **kw)
    shifted = simulate_task(p, np.random.default_rng(25), tech_shift=True, tech_global=(0, 0),
                            tech_batch=(0, 0), tech_warp_scale=(1, 1), tech_warp_shift_sd=0, **kw)
    np.testing.assert_allclose(plain.bulk, shifted.bulk, atol=1e-6)
    np.testing.assert_array_equal(plain.reference_mean, shifted.reference_mean)


def test_v4_pool_sharpen_and_pick_markers():
    from simulator import screen_pool
    p = panel()
    assert len(screen_pool(p, 0.3)) < len(screen_pool(p, 0.1)) <= len(p.betas)
    pool = screen_pool(p, 0.1)
    assert pool.ndim == 1 and len(pool) >= 20
    t = simulate_task(p, np.random.default_rng(11), n_markers=20, n_types=3,
                      marker_mode="pick", candidate_pool=pool)
    allowed = set(p.betas.index[pool])
    assert set(t.probe_names) <= allowed and len(set(t.probe_names)) == 20
    # sharpen off is the default and makes no draws
    kw = dict(n_markers=20, n_types=3, marker_mode="pick")
    a = simulate_task(p, np.random.default_rng(12), **kw)
    b = simulate_task(p, np.random.default_rng(12), sharpen=(1.0, 1.0), **kw)
    np.testing.assert_array_equal(a.reference_mean, b.reference_mean)
    # sharpening pushes values outward from 0.5 in reference and bulk alike
    c = simulate_task(p, np.random.default_rng(12), sharpen=(2.0, 2.0 + 1e-12), noise=0, **kw)
    d = simulate_task(p, np.random.default_rng(12), sharpen=(1.0, 1.0), noise=0, **kw)
    assert np.abs(c.reference_mean - 0.5).mean() > np.abs(d.reference_mean - 0.5).mean()
    with pytest.raises(ValueError):
        simulate_task(p, np.random.default_rng(1), n_markers=50, n_types=3, candidate_pool=pool[:10])


def test_pick_markers_exclude_hidden_data():
    p = panel()
    kw = dict(n_markers=30, n_types=3, missing=1, donor_mismatch=True, marker_mode="pick")
    a = simulate_task(p, np.random.default_rng(83), **kw)
    modified = panel()
    hidden = (modified.types == a.dropped_types[0]) | modified.donors.isin(a.mixture_donors)
    modified.betas.loc[:, hidden] = 0.99
    b = simulate_task(modified, np.random.default_rng(83), **kw)
    assert a.probe_names == b.probe_names
    np.testing.assert_allclose(a.reference_mean, b.reference_mean)


def test_v5_cpg_m_offsets_sparse_bulk_only_and_off_by_default():
    p = panel()
    kw = dict(n_markers=60, n_types=3, missing=1, noise=0, tech_shift=True,
              tech_global=(0, 0), tech_batch=(0, 0), tech_warp_scale=(1, 1), tech_warp_shift_sd=0)
    base = simulate_task(p, np.random.default_rng(31), **kw)
    # Explicit zero range is the v3/v4 default: bit-identical bulk.
    zero = simulate_task(p, np.random.default_rng(31), tech_cpg_m_sd=(0, 0), **kw)
    np.testing.assert_array_equal(base.bulk, zero.bulk)
    off = simulate_task(p, np.random.default_rng(31), tech_cpg_m_sd=(1.0, 1.0),
                        tech_cpg_m_frac=(0.2, 0.2), **kw)
    np.testing.assert_array_equal(base.reference_mean, off.reference_mean)
    np.testing.assert_array_equal(base.fractions, off.fractions)
    moved = np.abs(off.bulk - base.bulk) > 1e-9
    assert 0 < moved.mean() < 0.5
    assert np.all((off.bulk >= 0) & (off.bulk <= 1))
    # M-scale offsets move mid-range CpGs more than extreme ones on the beta scale.
    every = simulate_task(p, np.random.default_rng(31), tech_cpg_m_sd=(0.5, 0.5),
                          tech_cpg_m_frac=(1, 1), tech_cpg_m_df=1000, **kw)
    delta = np.abs(every.bulk - base.bulk)
    mid = (base.bulk > 0.3) & (base.bulk < 0.7)
    edge = (base.bulk < 0.1) | (base.bulk > 0.9)
    if mid.any() and edge.any():
        assert delta[mid].mean() > delta[edge].mean()
