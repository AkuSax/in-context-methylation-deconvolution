"""Donor-partitioned synthetic tasks from deposited sorted-cell methylation.

GSE35069 uses its deposited processing, not Noob. Synthetic augmented types
are artificial perturbations, not observed biological cell populations.

Synthetic types (v2, `synthetic="sparse"`) copy a real parent type and change a
small random subset of CpGs strongly toward the opposite methylation state.
Siblings from one parent then differ at a few percent of CpGs, like naive and
memory subsets. The v1 rule (`synthetic="dense"`) added N(0, 0.18) noise to
every CpG, which made synthetic types unlike any real subtype.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from .geo import read_meta, read_betas
except ImportError:
    from geo import read_meta, read_betas

DEFAULT_GSE35069 = Path(__file__).resolve().parent.parent / 'research-scout/GSE35069_series_matrix.txt.gz'
SORTED_TYPES = ('CD4+', 'CD8+', 'CD14+', 'CD19+', 'CD56+', 'Neu', 'Eos')
MARKER_MODES = ('random', 'informative', 'contrast', 'pick')


@dataclass
class SortedPanel:
    betas: pd.DataFrame
    types: pd.Series
    donors: pd.Series
    source: str = 'GSE35069'
    preprocessing: str = 'GSE35069 deposited processing; not Noob'
    _cube: tuple | None = field(default=None, init=False, repr=False, compare=False)

    def subset_donors(self, donors):
        keep = self.donors.isin(donors)
        cols = self.donors.index[keep]
        return SortedPanel(self.betas.loc[:, cols], self.types.loc[cols], self.donors.loc[cols], self.source, self.preprocessing)

    def groups(self):
        """Study label of each donor in `cube()` order: the donor id up to the first colon
        ('' for ids without one, so such a panel is a single study)."""
        return np.array([str(d).split(':')[0] if ':' in str(d) else '' for d in self.cube()[2]])

    def cube(self):
        """(CpG x type x donor array, type names, donor names), built once.

        Built on first use, so edit `betas` before the first simulated task.
        """
        if self._cube is None:
            names = list(pd.unique(self.types))
            donors = sorted(self.donors.unique())
            arr = np.empty((len(self.betas), len(names), len(donors)))
            x = self.betas.to_numpy(dtype=float)
            col = {c: i for i, c in enumerate(self.betas.columns)}
            for t, name in enumerate(names):
                for d, donor in enumerate(donors):
                    hit = self.types.index[(self.types == name) & (self.donors == donor)]
                    if len(hit) != 1:
                        raise ValueError('Panel must have every type for every donor')
                    arr[:, t, d] = x[:, col[hit[0]]]
            self._cube = (arr, names, donors)
        return self._cube


@dataclass
class SimulationTask:
    reference_mean: np.ndarray
    reference_sd: np.ndarray
    bulk: np.ndarray
    fractions: np.ndarray
    type_names: tuple[str, ...]
    probe_names: tuple[str, ...]
    reference_donors: tuple[str, ...]
    mixture_donors: tuple[str, ...]
    dropped_types: tuple[str, ...]
    preprocessing: str
    augmented_types: tuple[str, ...]
    # Real GSE35069 type each retained type (then each dropped type) copies.
    kept_parents: tuple[str, ...] = ()
    dropped_parents: tuple[str, ...] = ()


def load_gse35069(path=DEFAULT_GSE35069, types=SORTED_TYPES):
    """Load seven non-overlapping sorted fractions. Exclude WB, PBMC and Gran by default."""
    meta = read_meta(path)
    records = [(gsm, *title.rsplit('_', 1)) for gsm, title in zip(meta.gsm, meta.title)]
    records = [(gsm, typ, donor) for gsm, typ, donor in records if typ in types]
    betas = read_betas(path, meta).loc[:, [r[0] for r in records]]
    types = pd.Series({r[0]: r[1] for r in records})
    donors = pd.Series({r[0]: 'GSE35069:' + r[2] for r in records})
    good = betas.index.str.startswith('cg') & np.isfinite(betas.to_numpy()).all(axis=1)
    return SortedPanel(betas.loc[good], types, donors)


def split_donors(panel, seed=0, validation_fraction=0.5):
    """Return donor-disjoint train and validation panels, with at least 3 each."""
    ids = np.array(sorted(panel.donors.unique()))
    if len(ids) < 6 or not 0 < validation_fraction < 1:
        raise ValueError('Need at least six donors and a validation fraction between zero and one')
    ids = np.random.default_rng(seed).permutation(ids)
    nval = max(3, min(len(ids)-3, round(len(ids)*validation_fraction)))
    return panel.subset_donors(ids[nval:]), panel.subset_donors(ids[:nval])


def _pick_markers(mean, sd, n_ref, n_markers, t_min=5.0):
    """Like `reference.pick_markers`: per type and side, keep CpGs whose one-vs-rest
    statistic passes `t_min` (a stand-in for p < 1e-8), rank them by the size of the
    mean difference, then fall back to the statistic. Taken round-robin over types and
    sides until `n_markers` are chosen. This gives near-binary markers, as on real panels.
    """
    k = mean.shape[1]
    total = mean.sum(axis=1, keepdims=True)
    rest = (total - mean) / max(k - 1, 1)
    diff = mean - rest
    stat = diff / np.sqrt(sd ** 2 / n_ref + 1e-4)
    orders = []
    for j in range(k):
        for sign in (1, -1):
            d, t = sign * diff[:, j], sign * stat[:, j]
            passed = np.flatnonzero((t >= t_min) & (d > 0))
            first = passed[np.argsort(-d[passed])]
            rest_idx = np.argsort(-t)[np.isin(np.argsort(-t), first, invert=True)]
            orders.append(np.concatenate([first, rest_idx]))
    chosen, seen, rank = [], set(), 0
    while len(chosen) < n_markers:
        for order in orders:
            c = int(order[rank])
            if c not in seen:
                seen.add(c)
                chosen.append(c)
                if len(chosen) == n_markers:
                    break
        rank += 1
    return np.array(chosen)


def _contrast_markers(mean, sd, n_ref, n_markers):
    """One-vs-rest markers among the retained types, hyper and hypo per type.

    Mirrors `reference.pick_markers`: rank by a Welch-style statistic, then
    take CpGs round-robin over types and sides until `n_markers` are chosen.
    """
    k = mean.shape[1]
    total = mean.sum(axis=1, keepdims=True)
    rest = (total - mean) / max(k - 1, 1)
    stat = (mean - rest) / np.sqrt(sd ** 2 / n_ref + 1e-4)
    orders = []
    for j in range(k):
        orders.append(np.argsort(-stat[:, j]))
        orders.append(np.argsort(stat[:, j]))
    chosen, seen, rank = [], set(), 0
    while len(chosen) < n_markers:
        for order in orders:
            c = int(order[rank])
            if c not in seen:
                seen.add(c)
                chosen.append(c)
                if len(chosen) == n_markers:
                    break
        rank += 1
    return np.array(chosen)


def screen_pool(panel, min_range=0.3):
    """Row indices of CpGs whose cell-type means (averaged over the panel's donors)
    span at least `min_range`. Real marker selection searches ~850k probes, so its
    panels are near-binary. A random 10k pool gives much weaker markers; drawing
    candidates from this pool restores realistic marker strength.

    Compute it on the training panel and reuse it for validation, so the screen
    never sees validation donors.
    """
    cube, _, _ = panel.cube()
    means = cube.mean(axis=2)
    return np.flatnonzero(means.max(axis=1) - means.min(axis=1) >= min_range)


def _technical_shift(bulk, rng, tech_global, tech_batch, warp_scale, warp_shift_sd,
                     cpg_m_sd=(0.0, 0.0), cpg_m_df=3.0, cpg_m_frac=(1.0, 1.0)):
    """Reference-to-bulk technical difference applied to one bulk vector.

    All draws are made unconditionally, so the number of draws is fixed for a
    given setting. The v5 per-CpG M-value offsets are drawn after the v3 draws
    and only when `cpg_m_sd` is not (0, 0), so v3/v4 settings reproduce exactly.
    """
    n = len(bulk)
    a = rng.uniform(*warp_scale)
    b = rng.normal(0, warp_shift_sd)
    s_global = rng.uniform(*tech_global)
    g = rng.normal(0, s_global)
    s_batch = rng.uniform(*tech_batch)
    batch = rng.normal(0, s_batch, n)
    d = 0.0
    if tuple(cpg_m_sd) != (0.0, 0.0):
        # v5: heavy-tailed per-CpG offsets in M-value space. On the beta scale they
        # are largest at mid-range CpGs and vanish near 0 and 1, as in real
        # sorted-vs-mixture array residuals. Only a fraction ~ U(cpg_m_frac) of
        # CpGs is affected (in GSE110554 about 12% of markers carry all |z| > 3).
        s_d = rng.uniform(*cpg_m_sd)
        affected = rng.random(n) < rng.uniform(*cpg_m_frac)
        d = affected * s_d * rng.standard_t(cpg_m_df, n)
    eps = 1e-4
    m = np.log(np.clip(bulk, eps, 1 - eps) / (1 - np.clip(bulk, eps, 1 - eps)))
    warped = 1 / (1 + np.exp(-(a * m + b + d)))
    return np.clip(warped + g + batch, 0, 1)


# ---------------------------------------------------------------------------
# v6: a second sorted-cell study (GSE103541) and cross-study panels
# ---------------------------------------------------------------------------

# GSE35069 label -> common label, for the five types both studies sorted.
COMMON_TYPES = {'CD19+': 'B', 'CD4+': 'CD4T', 'CD8+': 'CD8T', 'CD14+': 'Mono', 'Gran': 'Gran'}


def load_gse103541_panel(max_missing=0.05):
    """GSE103541 (EPIC, Noob) as a SortedPanel: 27 donors x 5 types.

    CpGs masked (detection p) in more than `max_missing` of the arrays are dropped; the
    few remaining masked values are filled with that type's mean over the other donors.
    """
    try:
        from . import datasets as ds
    except ImportError:
        import datasets as ds
    previous, ds.PREPROC = ds.PREPROC, 'noob'
    try:
        data = ds.load('GSE103541')
    finally:
        ds.PREPROC = previous
    betas = data.betas.loc[data.betas.index.str.startswith('cg')]
    types = data.cell_type.astype(str).loc[betas.columns]
    x = betas.to_numpy(np.float32)
    rows = np.isnan(x).mean(axis=1) <= max_missing
    x = x[rows]
    for name in pd.unique(types):
        cols = np.flatnonzero((types == name).to_numpy())
        block = x[:, cols]
        fill = np.nanmean(np.where(np.isnan(block).all(axis=1, keepdims=True), 0.0, block), axis=1)
        x[:, cols] = np.where(np.isnan(block), fill[:, None], block)
    betas = pd.DataFrame(x, index=betas.index[rows], columns=betas.columns)
    return SortedPanel(betas, types, data.meta['donor'].astype(str), 'GSE103541',
                       'GSE103541 Noob (minfi), detection-p masked values filled by type mean')


def joint_panel(a, b, type_map=COMMON_TYPES):
    """Two studies on their common CpGs and common types. `type_map` renames a's types
    to b's labels; a's other types are left out. Donor ids keep their study prefix, so
    `groups()` tells the studies apart."""
    keep = a.types.index[a.types.isin(list(type_map))]
    common = a.betas.index.intersection(b.betas.index)
    betas = pd.concat([a.betas.loc[common, keep], b.betas.loc[common]], axis=1).astype(np.float32)
    types = pd.concat([a.types.loc[keep].map(type_map), b.types])
    donors = pd.concat([a.donors.loc[keep], b.donors])
    return SortedPanel(betas, types, donors, f'{a.source}+{b.source}',
                       f'{a.preprocessing} | {b.preprocessing}')


def build_v6_panels(seed=0, min_range=0.3, n_validation_103541=7):
    """Train/validation panels and candidate pools for the v6 prior.

    Returns {'train': {name: panel}, 'validation': {name: panel}, 'pools': {name: rows}}
    with names 'gse35069' (seven sorted types, as in v2-v5) and 'joint' (five common types
    from GSE35069 and GSE103541). All four panels share one CpG index: CpGs on both arrays
    whose type-mean range is at least `min_range` on the TRAINING donors of either panel.
    Pools index into that shared CpG list. Donors are split per study.
    """
    p35 = load_gse35069(types=SORTED_TYPES + ('Gran',))
    p103 = load_gse103541_panel()
    common = p35.betas.index.intersection(p103.betas.index)
    p35 = SortedPanel(p35.betas.loc[common], p35.types, p35.donors, p35.source, p35.preprocessing)
    p103 = SortedPanel(p103.betas.loc[common], p103.types, p103.donors, p103.source, p103.preprocessing)
    train35, val35 = split_donors(p35, seed=seed)
    ids = np.random.default_rng([seed, 103541]).permutation(np.array(sorted(p103.donors.unique())))
    val103, train103 = p103.subset_donors(ids[:n_validation_103541]), p103.subset_donors(ids[n_validation_103541:])

    def seven(p):
        cols = p.types.index[p.types.isin(SORTED_TYPES)]
        return SortedPanel(p.betas.loc[:, cols], p.types.loc[cols], p.donors.loc[cols], p.source, p.preprocessing)

    panels = {'train': {'gse35069': seven(train35), 'joint': joint_panel(train35, train103)},
              'validation': {'gse35069': seven(val35), 'joint': joint_panel(val35, val103)}}
    screens = {name: screen_pool(p, min_range) for name, p in panels['train'].items()}
    union = np.unique(np.concatenate(list(screens.values())))
    for split in panels.values():
        for name, p in split.items():
            split[name] = SortedPanel(p.betas.iloc[union].astype(np.float32), p.types, p.donors, p.source, p.preprocessing)
    pools = {name: np.searchsorted(union, rows) for name, rows in screens.items()}
    return dict(panels, pools=pools)


def save_panels(bundle, path):
    """Write `build_v6_panels` output to one .npz file, e.g. to copy to a GPU box."""
    out = {}
    for split in ('train', 'validation'):
        for name, p in bundle[split].items():
            key = f'{split}/{name}/'
            out[key + 'betas'] = p.betas.to_numpy(np.float32)
            out[key + 'probes'] = p.betas.index.to_numpy(str)
            out[key + 'samples'] = p.betas.columns.to_numpy(str)
            out[key + 'types'] = p.types.loc[p.betas.columns].to_numpy(str)
            out[key + 'donors'] = p.donors.loc[p.betas.columns].to_numpy(str)
            out[key + 'meta'] = np.array([p.source, p.preprocessing])
    for name, rows in bundle['pools'].items():
        out[f'pools/{name}'] = np.asarray(rows)
    np.savez_compressed(path, **out)


def load_panels(path):
    z = np.load(path, allow_pickle=False)
    bundle = {'train': {}, 'validation': {}, 'pools': {}}
    for key in z.files:
        parts = key.split('/')
        if parts[0] == 'pools':
            bundle['pools'][parts[1]] = z[key]
        elif parts[2] == 'betas':
            pre = f'{parts[0]}/{parts[1]}/'
            samples = z[pre + 'samples']
            betas = pd.DataFrame(z[key], index=z[pre + 'probes'], columns=samples)
            bundle[parts[0]][parts[1]] = SortedPanel(
                betas, pd.Series(z[pre + 'types'], index=samples), pd.Series(z[pre + 'donors'], index=samples),
                *z[pre + 'meta'].tolist())
    return bundle


# ---------------------------------------------------------------------------
# Task simulation
# ---------------------------------------------------------------------------

@dataclass
class _Design:
    """Everything a task or cohort shares: types, reference, markers, donor roles."""
    profiles: np.ndarray        # candidate x needed x donor
    mean: np.ndarray
    sd: np.ndarray
    selected: np.ndarray
    keep: list
    drop: set
    all_names: list
    names: list
    parents: list
    augmented: list
    donors: list
    mixture_d: int
    ref_draw: np.ndarray
    ref_donors: tuple
    probes: tuple
    needed: int


def _design(panel, rng, *, n_markers, n_types, missing, donor_mismatch, augment, marker_mode,
            synthetic, candidate_pool, sharpen, cross_study, mixture_group, sharpen_groups):
    if not isinstance(rng, np.random.Generator):
        raise TypeError('rng must be a numpy.random.Generator')
    if n_types < 2 or missing < 0 or missing > 3:
        raise ValueError('Invalid task parameters')
    if synthetic not in ('sparse', 'dense'):
        raise ValueError('synthetic must be sparse or dense')
    cube, names, donors = panel.cube()
    needed = n_types + missing
    if needed > len(names) and not augment:
        raise ValueError('Enable augment to add synthetic types')
    if needed > 15:
        raise ValueError('At most 15 total types are supported')
    if len(donors) < 2:
        raise ValueError('At least two donors required')
    if n_markers < 1 or n_markers > cube.shape[0]:
        raise ValueError('Invalid marker count')
    if marker_mode not in MARKER_MODES:
        raise ValueError(f'marker_mode must be one of {MARKER_MODES}')
    groups = panel.groups()
    chosen = [int(i) for i in rng.choice(len(names), min(needed, len(names)), replace=False)]
    augmented = [f'synthetic_{i+1}' for i in range(max(0, needed-len(names)))]
    all_names = [names[i] for i in chosen] + augmented
    drop = set(rng.choice(needed, missing, replace=False).tolist())
    keep = [i for i in range(needed) if i not in drop]
    if mixture_group is None:
        mixture_d = int(rng.integers(len(donors)))
    else:
        members = np.flatnonzero(groups == mixture_group)
        if len(members) == 0:
            raise ValueError(f'No donors in group {mixture_group}')
        mixture_d = int(rng.choice(members))
    # v6: the reference comes from the other study (cross_study) or the mixture's own.
    same = groups == groups[mixture_d]
    eligible = [d for d in range(len(donors))
                if (not same[d] if cross_study else same[d]) and (not donor_mismatch or d != mixture_d)]
    if not eligible:
        raise ValueError('No eligible reference donors')
    ref_draw = rng.choice(eligible, size=int(rng.integers(3, 7)), replace=True)
    if not donor_mismatch and not cross_study:
        ref_draw[0] = mixture_d
    ref_donors = tuple(sorted({donors[d] for d in ref_draw}))
    if candidate_pool is None:
        candidate_count = n_markers if marker_mode == 'random' else min(cube.shape[0], max(n_markers, 10000))
        idx = rng.choice(cube.shape[0], candidate_count, replace=False)
    else:
        pool = np.asarray(candidate_pool)
        if pool.ndim != 1 or len(pool) < n_markers:
            raise ValueError('candidate_pool must be a 1-D index array with at least n_markers rows')
        candidate_count = n_markers if marker_mode == 'random' else min(len(pool), max(n_markers, 10000))
        idx = pool[rng.choice(len(pool), candidate_count, replace=False)]
    sub = cube[idx]                                   # candidate x type x donor
    parents = chosen + [int(rng.choice(chosen)) for _ in augmented]
    profiles = sub[:, parents, :].astype(float)       # candidate x needed x donor (a copy)
    # Synthetic types share one perturbation across the donor samples of a task.
    for j in range(len(chosen), needed):
        parent = profiles[:, j, :]
        if synthetic == 'dense':
            profiles[:, j, :] = np.clip(parent + rng.normal(0, 0.18, candidate_count)[:, None], 0, 1)
            continue
        changed = rng.random(candidate_count) < rng.uniform(0.005, 0.04)
        strength = rng.uniform(0.3, 0.9, candidate_count)
        # Direction comes from reference donors only, so the mixture donor's
        # data never shapes the reference.
        up = parent[:, ref_draw].mean(axis=1) < 0.5
        jitter = rng.normal(0, 0.01, candidate_count)
        shifted = np.where(up[:, None], parent + strength[:, None] * (1 - parent),
                           parent - strength[:, None] * parent)
        profiles[:, j, :] = np.clip(np.where(changed[:, None], shifted, parent) + jitter[:, None], 0, 1)
    if tuple(sharpen) != (1.0, 1.0):
        # v4: GSE35069 profiles are compressed at marker CpGs (median between-type
        # range 0.75 vs 0.88 on EPIC Noob at the same CpGs). Stretch profiles,
        # reference and mixture alike, in M-value space by a ~ U(sharpen).
        # v6: `sharpen_groups` limits the stretch to donors of the named studies.
        a = rng.uniform(*sharpen)
        x = np.clip(profiles, 1e-4, 1 - 1e-4)
        stretched = 1 / (1 + np.exp(-a * np.log(x / (1 - x))))
        if sharpen_groups is None:
            profiles = stretched
        else:
            hit = np.isin(groups, list(sharpen_groups))
            profiles[:, :, hit] = stretched[:, :, hit]
    refs = profiles[:, :, ref_draw]                   # candidate x needed x draw
    mean = refs.mean(axis=2)
    sd = refs.std(axis=2, ddof=1)
    if marker_mode == 'informative':
        scores = np.var(mean[:, keep], axis=1)
        selected = np.argpartition(scores, len(scores)-n_markers)[-n_markers:]
    elif marker_mode == 'contrast':
        selected = _contrast_markers(mean[:, keep], sd[:, keep], len(ref_draw), n_markers)
    elif marker_mode == 'pick':
        selected = _pick_markers(mean[:, keep], sd[:, keep], len(ref_draw), n_markers)
    else:
        selected = np.arange(n_markers)
    probes = tuple(panel.betas.index[idx[selected]].astype(str))
    return _Design(profiles, mean, sd, selected, keep, drop, all_names, names, parents, augmented,
                   donors, mixture_d, ref_draw, ref_donors, probes, needed)


def _draw_fractions(design, rng, concentration_range, neutrophil_weight, zero_type_prob):
    """Fractions over all `needed` types. Favor neutrophils (or unsplit granulocytes)
    when present. A randomly selected type is explicitly rare."""
    needed, all_names, keep = design.needed, design.all_names, design.keep
    concentration = rng.uniform(*concentration_range)
    weights = np.ones(needed)
    for major in ('Neu', 'Gran'):
        if major in all_names:
            weights[all_names.index(major)] = neutrophil_weight
    rare = int(rng.integers(needed))
    share = np.exp(rng.uniform(np.log(0.001), np.log(0.05)))
    others = np.arange(needed) != rare
    fractions = np.zeros(needed)
    fractions[others] = (1-share) * rng.dirichlet(concentration * weights[others] / weights[others].sum())
    fractions[rare] = share
    if zero_type_prob > 0 and rng.random() < zero_type_prob:
        absent = keep[int(rng.integers(len(keep)))]
        fractions[absent] = 0.0
        fractions /= fractions.sum()
    return fractions


def _mix(design, rng, fractions, mixture_d, noise):
    mixture = design.profiles[:, :, mixture_d] @ fractions
    mixture = np.clip(mixture + rng.normal(0, noise * np.sqrt(mixture*(1-mixture))), 0, 1)
    return mixture[design.selected]


def _task(design, panel, bulk, fractions, mixture_d):
    keep, drop, names, parents = design.keep, design.drop, design.names, design.parents
    return SimulationTask(design.mean[design.selected][:, keep], design.sd[design.selected][:, keep], bulk,
                          np.r_[fractions[keep], fractions[list(drop)].sum()],
                          tuple(design.all_names[i] for i in keep), design.probes, design.ref_donors,
                          (design.donors[mixture_d],), tuple(design.all_names[i] for i in sorted(drop)),
                          panel.preprocessing, tuple(design.augmented),
                          tuple(names[parents[i]] for i in keep),
                          tuple(names[parents[i]] for i in sorted(drop)))


def simulate_task(panel, rng, *, n_markers=300, n_types=6, missing=0,
                  donor_mismatch=False, augment=False, noise=0.02,
                  marker_mode="random", synthetic="sparse", zero_type_prob=0.0,
                  concentration_range=(8, 30), neutrophil_weight=4.0,
                  tech_shift=False, tech_global=(0.0, 0.04), tech_batch=(0.0, 0.04),
                  tech_warp_scale=(0.9, 1.1), tech_warp_shift_sd=0.1,
                  tech_cpg_m_sd=(0.0, 0.0), tech_cpg_m_df=3.0, tech_cpg_m_frac=(1.0, 1.0),
                  candidate_pool=None, sharpen=(1.0, 1.0),
                  cross_study=False, mixture_group=None, sharpen_groups=None):
    """Draw one task. n_types counts retained types. fractions ends with other.

    Technical shift (`tech_shift=True`, off by default). Real arrays differ from
    their sorted reference by a platform, batch and processing offset. The bulk
    (and only the bulk, after mixing and mixture noise) gets, per task:
      - a warp in M-value space, m' = a*m + b, with a ~ U(tech_warp_scale) and
        b ~ N(0, tech_warp_shift_sd), then back to beta;
      - a global offset g ~ N(0, s_g), with s_g ~ U(tech_global);
      - per-CpG batch offsets c_j ~ N(0, s_c), with s_c ~ U(tech_batch);
      - (v5) per-CpG M-value offsets d_j = s_d * t(tech_cpg_m_df), with
        s_d ~ U(tech_cpg_m_sd), on a random fraction ~ U(tech_cpg_m_frac) of
        CpGs, added inside the warp (off by default).
    The result is clipped to [0, 1]. The reference mean and SD are never changed.
    Shift draws come from a child generator seeded by one draw made after all
    other draws in the task, so reference, markers, fractions and missing-type
    choices are identical with the shift on or off. Every shift draw is made
    whether or not a type is missing, so the shift does not depend on the
    missing-type label. With tech_shift=False no extra draws are made.

    Bootstrap 3–6 reference samples from the eligible donors. The donor IDs in
    reference_donors record distinct biological donors, not bootstrap draws.
    A rare type has a log-uniform fraction from 0.1% to 5% before omission.
    Informative and contrast markers use only retained reference means from a
    random candidate pool. Random markers provide a harder stress test.
    `sharpen` (v4) stretches every profile by a ~ U(sharpen) in M-value space
    before the reference and mixture are built (default (1, 1) = off, no draw).
    `candidate_pool` (v4) restricts candidate CpGs to given rows, e.g. from
    `screen_pool`, so contrast markers are as strong as real ones.
    With probability `zero_type_prob` one retained type is absent from the
    mixture (an over-complete reference).

    v6, for panels holding more than one study (`joint_panel`): reference donors
    come from the mixture donor's own study, or with `cross_study=True` from the
    other studies, so the reference-to-bulk difference is a measured cross-study
    one. `mixture_group` fixes the mixture's study. `sharpen_groups` limits the
    v4 stretch to the named studies. On a single-study panel the defaults leave
    every draw as before.
    """
    if noise < 0:
        raise ValueError('Invalid task parameters')
    design = _design(panel, rng, n_markers=n_markers, n_types=n_types, missing=missing,
                     donor_mismatch=donor_mismatch, augment=augment, marker_mode=marker_mode,
                     synthetic=synthetic, candidate_pool=candidate_pool, sharpen=sharpen,
                     cross_study=cross_study, mixture_group=mixture_group, sharpen_groups=sharpen_groups)
    fractions = _draw_fractions(design, rng, concentration_range, neutrophil_weight, zero_type_prob)
    bulk = _mix(design, rng, fractions, design.mixture_d, noise)
    if tech_shift:
        # One draw from rng, made last and unconditionally, seeds the shift.
        shift_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        bulk = _technical_shift(bulk, shift_rng, tech_global, tech_batch,
                                tech_warp_scale, tech_warp_shift_sd,
                                tech_cpg_m_sd, tech_cpg_m_df, tech_cpg_m_frac)
    return _task(design, panel, bulk, fractions, design.mixture_d)


SAMPLE_SHIFT = dict(tech_global=(0.0, 0.01), tech_batch=(0.0, 0.015),
                    warp_scale=(0.97, 1.03), warp_shift_sd=0.03)


def simulate_cohort(panel, rng, n_samples, *, n_markers=300, n_types=6, missing=0,
                    donor_mismatch=False, augment=False, noise=0.02,
                    marker_mode="random", synthetic="sparse", zero_type_prob=0.0,
                    concentration_range=(8, 30), neutrophil_weight=4.0,
                    tech_shift=False, tech_global=(0.0, 0.04), tech_batch=(0.0, 0.04),
                    tech_warp_scale=(0.9, 1.1), tech_warp_shift_sd=0.1,
                    tech_cpg_m_sd=(0.0, 0.0), tech_cpg_m_df=3.0, tech_cpg_m_frac=(1.0, 1.0),
                    candidate_pool=None, sharpen=(1.0, 1.0),
                    cross_study=False, mixture_group=None, sharpen_groups=None,
                    tight_prob=0.5, tight_concentration=(20.0, 500.0), sample_shift=SAMPLE_SHIFT):
    """v6: `n_samples` bulks measured against one reference, as one batch of arrays.

    Shared by the cohort: the retained and dropped types, the reference, the markers, the
    mixture study, and the technical shift (one draw of `_technical_shift` applied to every
    sample). Per sample: the mixture donor (from the mixture study; never a reference donor
    under `donor_mismatch` or `cross_study`), the fractions, the measurement noise, and a
    small extra shift with the `sample_shift` ranges.

    With probability `tight_prob` the cohort is homogeneous, like a whole-blood cohort:
    fractions are Dirichlet(c * centre) around one centre drawn as in `simulate_task`, with
    c log-uniform in `tight_concentration`. Otherwise every sample is an independent draw,
    like a set of constructed mixtures. In a tight cohort a missing type contributes almost
    the same residual to every sample, so it is hard to tell from a batch offset by design.

    Returns a list of SimulationTask with identical reference arrays.
    """
    if noise < 0 or n_samples < 1:
        raise ValueError('Invalid cohort parameters')
    design = _design(panel, rng, n_markers=n_markers, n_types=n_types, missing=missing,
                     donor_mismatch=donor_mismatch, augment=augment, marker_mode=marker_mode,
                     synthetic=synthetic, candidate_pool=candidate_pool, sharpen=sharpen,
                     cross_study=cross_study, mixture_group=mixture_group, sharpen_groups=sharpen_groups)
    groups = panel.groups()
    used = set(design.ref_draw.tolist()) if (donor_mismatch or cross_study) else set()
    members = [int(d) for d in np.flatnonzero(groups == groups[design.mixture_d]) if d not in used]
    shared_seed = int(rng.integers(0, 2**63 - 1))
    tight = bool(rng.random() < tight_prob)
    centre = _draw_fractions(design, rng, concentration_range, neutrophil_weight, zero_type_prob)
    scale = float(np.exp(rng.uniform(*np.log(tight_concentration))))
    tasks = []
    for i in range(n_samples):
        d = design.mixture_d if i == 0 else int(rng.choice(members))
        if tight:
            fractions = np.zeros(design.needed)
            positive = centre > 0
            fractions[positive] = rng.dirichlet(scale * centre[positive] + 1e-3)
        elif i == 0:
            fractions = centre
        else:
            fractions = _draw_fractions(design, rng, concentration_range, neutrophil_weight, zero_type_prob)
        bulk = _mix(design, rng, fractions, d, noise)
        if tech_shift:
            bulk = _technical_shift(bulk, np.random.default_rng(shared_seed), tech_global, tech_batch,
                                    tech_warp_scale, tech_warp_shift_sd,
                                    tech_cpg_m_sd, tech_cpg_m_df, tech_cpg_m_frac)
            if sample_shift:
                own = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                bulk = _technical_shift(bulk, own, sample_shift['tech_global'], sample_shift['tech_batch'],
                                        sample_shift['warp_scale'], sample_shift['warp_shift_sd'])
        tasks.append(_task(design, panel, bulk, fractions, d))
    return tasks
