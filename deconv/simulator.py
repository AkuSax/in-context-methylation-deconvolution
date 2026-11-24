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
import os
import numpy as np
import pandas as pd
try:
    from .geo import read_meta, read_betas
except ImportError:
    from geo import read_meta, read_betas

DEFAULT_GSE35069 = (Path(os.environ.get('DECONV_RAW', Path(__file__).resolve().parent.parent / 'data' / 'raw'))
                    / 'GSE35069' / 'GSE35069_series_matrix.txt.gz')
SORTED_TYPES = ('CD4+', 'CD8+', 'CD14+', 'CD19+', 'CD56+', 'Neu', 'Eos')
MARKER_MODES = ('random', 'informative', 'contrast', 'pick')
# v7 adds 'real': the ranking rule of reference.pick_markers (see `_real_markers`).
MARKER_MODES_V7 = MARKER_MODES + ('real',)


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

    def donor_arrays(self):
        """Per cube donor, the set of (type, source array) it is built from, in `cube()` donor order.

        Pseudo-donor panels name their samples `<array>@<pseudo-donor>` (`pair_pseudo_donors`,
        `_swap_b_arrays`), so one measured array can sit in several donors. Panels without that
        naming give every donor its own arrays."""
        donors = self.cube()[2]
        out = {d: set() for d in donors}
        for col, typ, donor in zip(self.betas.columns, self.types.loc[self.betas.columns], self.donors.loc[self.betas.columns]):
            out[donor].add((str(typ), str(col).split('@')[0]))
        return [frozenset(out[d]) for d in donors]

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


def welch_one_vs_rest(mean, sd, n_ref):
    """Welch t-test of each type against all samples of the other types, from summary
    statistics: the mean difference `diff` and the two-sided p value, both CpG x type.

    Every type is assumed to hold `n_ref` samples with mean `mean` and SD `sd` (ddof=1), so the
    pooled "rest" group has (k-1)*n_ref samples whose mean and variance follow exactly from the
    per-type moments. This equals `scipy.stats.ttest_ind(a, b, equal_var=False)` on the
    samples, which is what `reference.pick_markers` runs (NaN, e.g. zero variance on both sides,
    becomes p = 1 as there).
    """
    from scipy.special import stdtr
    mean, sd = np.asarray(mean, float), np.asarray(sd, float)
    k = mean.shape[1]
    na, nb = n_ref, (k - 1) * n_ref
    total = mean.sum(axis=1, keepdims=True)
    rest_mean = (total - mean) / (k - 1)
    ssq_within = ((n_ref - 1) * sd ** 2).sum(axis=1, keepdims=True) - (n_ref - 1) * sd ** 2
    ssq_between = n_ref * (((mean ** 2).sum(axis=1, keepdims=True) - mean ** 2) - (k - 1) * rest_mean ** 2)
    rest_var = np.maximum(ssq_within + ssq_between, 0) / (nb - 1)
    va, vb = sd ** 2 / na, rest_var / nb
    diff = mean - rest_mean
    se2 = va + vb
    with np.errstate(divide='ignore', invalid='ignore'):
        t = diff / np.sqrt(se2)
        df = se2 ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
        p = 2 * stdtr(df, -np.abs(t))
    return diff, np.nan_to_num(p, nan=1.0)


def real_marker_lists(mean, sd, n_ref, n_per_side=50, p_max=1e-8):
    """For every (type, side) the CpG order `reference.pick_markers` ranks: Welch p < p_max and
    a positive signed difference, by decreasing difference; if fewer than `n_per_side` pass, the
    5 * n_per_side best p values among the positive differences, again by difference.
    Returns a list of 2k index arrays (type 0 hyper, type 0 hypo, type 1 hyper, ...). Its first
    `n_per_side` entries are what `pick_markers` takes."""
    diff, p = welch_one_vs_rest(mean, sd, n_ref)
    lists = []
    for j in range(mean.shape[1]):
        for sign in (1, -1):
            d = sign * diff[:, j]
            pool = np.flatnonzero((p[:, j] < p_max) & (d > 0))
            if len(pool) < n_per_side:
                pool = np.flatnonzero(d > 0)
                pool = pool[np.argsort(p[pool, j])[: 5 * n_per_side]]
            lists.append(pool[np.argsort(-d[pool])])
    return lists


def real_marker_union(mean, sd, n_ref, n_per_side=50, p_max=1e-8):
    """Sorted unique CpG rows, exactly the set `reference.pick_markers` returns."""
    return np.unique(np.concatenate([l[:n_per_side] for l in real_marker_lists(mean, sd, n_ref, n_per_side, p_max)]))


def _real_markers(mean, sd, n_ref, n_markers, p_max=1e-8):
    """`marker_mode='real'`: the `reference.pick_markers` rule with a fixed marker count.

    `reference.pick_markers` takes n per type and side and returns however many unique CpGs
    that gives (about 245-286 for n = 25). A task needs `n_markers` exactly, so n_per_side =
    ceil(n_markers / 2k) and the (type, side) lists are read in rank order, round-robin, skipping
    CpGs already taken, until `n_markers` are chosen. Whenever no trimming or extension is needed
    the chosen set is identical to `reference.pick_markers`' (`real_marker_union`)."""
    k = mean.shape[1]
    n_per_side = int(np.ceil(n_markers / (2 * k)))
    lists = real_marker_lists(mean, sd, n_ref, n_per_side, p_max)
    chosen, seen, rank = [], set(), 0
    longest = max(len(l) for l in lists)
    while len(chosen) < n_markers and rank < longest:
        for l in lists:
            if rank < len(l) and int(l[rank]) not in seen:
                seen.add(int(l[rank]))
                chosen.append(int(l[rank]))
                if len(chosen) == n_markers:
                    break
        rank += 1
    if len(chosen) < n_markers:          # lists exhausted: fill with the largest |t| overall
        diff, p = welch_one_vs_rest(mean, sd, n_ref)
        score = np.abs(diff).max(axis=1)
        for c in np.argsort(-score):
            if int(c) not in seen:
                seen.add(int(c)); chosen.append(int(c))
                if len(chosen) == n_markers:
                    break
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



# ---------------------------------------------------------------------------
# v7: training-donor-only masking and fill, subtype and B-cell sources
# ---------------------------------------------------------------------------

def load_gse103541_panel_raw():
    """GSE103541 as a SortedPanel with the detection-p masked values kept as NaN.

    Same arrays, labels and probes as `load_gse103541_panel` before it drops and fills; use
    `fit_mask_and_fill` on the training donors, then apply it to every split."""
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
    return SortedPanel(betas.astype(np.float32), types, data.meta['donor'].astype(str), 'GSE103541',
                       'GSE103541 Noob (minfi), masked values kept as NaN')


@dataclass
class MaskFill:
    """A probe filter and per-type fill values fitted on training samples only."""
    probes: pd.Index              # probes kept (masked in at most `max_missing` of the training samples)
    fill: pd.DataFrame            # probe x type: mean over the training samples of that type
    max_missing: float

    def apply(self, panel, probes=None):
        """Panel restricted to the kept probes (or `probes`, a subset of them), NaN replaced by
        the training mean of that type. A type absent from the fitting panel raises."""
        probes = self.probes if probes is None else probes
        x = panel.betas.loc[probes].to_numpy(np.float32).copy()
        fill = self.fill.loc[probes]
        for name in pd.unique(panel.types):
            if name not in fill.columns:
                raise ValueError(f'type {name} was not in the fitting panel')
            cols = np.flatnonzero((panel.types.loc[panel.betas.columns] == name).to_numpy())
            block = x[:, cols]
            x[:, cols] = np.where(np.isnan(block), fill[name].to_numpy(np.float32)[:, None], block)
        betas = pd.DataFrame(x, index=probes, columns=panel.betas.columns)
        return SortedPanel(betas, panel.types, panel.donors, panel.source, panel.preprocessing)


def fit_mask_and_fill(train_panel, max_missing=0.05):
    """Decide the probe filter and the fill values from the TRAINING panel alone.

    A probe is kept when at most `max_missing` of the training arrays are NaN. The fill of a
    masked value is the mean of that probe over the training arrays of the same type; if every
    training array of that type is masked at the probe it falls back to the probe's mean over
    all training arrays (0 if that is empty too). Validation donors never enter, unlike
    `load_gse103541_panel`, which decides both on all donors."""
    x = train_panel.betas.to_numpy(np.float32)
    keep = np.isnan(x).mean(axis=1) <= max_missing
    x = x[keep]
    overall = np.nanmean(np.where(np.isnan(x).all(axis=1, keepdims=True), 0.0, x), axis=1)
    cols = {}
    for name in pd.unique(train_panel.types):
        block = x[:, np.flatnonzero((train_panel.types.loc[train_panel.betas.columns] == name).to_numpy())]
        empty = np.isnan(block).all(axis=1)
        cols[name] = np.where(empty, overall, np.nanmean(np.where(empty[:, None], 0.0, block), axis=1))
    index = train_panel.betas.index[keep]
    return MaskFill(index, pd.DataFrame(cols, index=index).astype(np.float32), max_missing)


def pair_pseudo_donors(panel, n_donors, rng, prefix, types):
    """Panel whose donors are made by drawing one array of every type from `panel`.

    `panel` holds independent arrays. Pseudo-donor i takes, for each type, the i-th entry of a
    sequence built from shuffled copies of that type's arrays, so every array is used about the
    same number of times and an array is reused in other pseudo-donors once `n_donors` exceeds the
    type's array count. Arrays of different types in one pseudo-donor come from different real
    donors. Sample names are `<array>@<pseudo-donor>`; the draws are fixed by `rng`."""
    picks = {}
    for t in types:
        arrays = list(panel.types.index[panel.types == t])
        if not arrays:
            raise ValueError(f'No arrays of type {t}')
        seq = []
        while len(seq) < n_donors:
            seq.extend(rng.permutation(arrays).tolist())
        picks[t] = seq[:n_donors]
    cols, names, ts, donors = [], [], [], []
    for i in range(n_donors):
        for t in types:
            cols.append(picks[t][i]); names.append(f'{picks[t][i]}@{prefix}{i:02d}')
            ts.append(t); donors.append(f'{prefix}{i:02d}')
    betas = pd.DataFrame(panel.betas.loc[:, cols].to_numpy(np.float32), index=panel.betas.index, columns=names)
    return SortedPanel(betas, pd.Series(ts, index=names), pd.Series(donors, index=names), panel.source,
                       panel.preprocessing)


LYMPH_TYPES = ('CD4CM', 'CD4EM', 'CD8nv', 'CD8CM', 'CD8EM', 'CD8EMRA')
LYMPH_UNUSED = ('Bmem', 'Bnv', 'CD4nv', 'CD8nv_frozen')   # one array each after the overlap drop


def split_lymph_arrays(meta, seed, min_train=3):
    """Validation arrays of GSE174666 chosen by donor key (`datasets.load_gse174666` meta).

    Donor keys (a leukoreduction-filter number shared by several subsets) are shuffled and moved
    to validation one at a time while every used subtype keeps at least `min_train` training
    arrays, until every subtype has at least one validation array. Returns the validation GSMs."""
    meta = meta[meta['subtype'].isin(LYMPH_TYPES)]
    keys = np.random.default_rng([seed, 174666]).permutation(np.array(sorted(meta['donor_key'].unique())))
    count = meta['subtype'].value_counts().to_dict()
    val_n = {t: 0 for t in LYMPH_TYPES}
    val = set()
    for key in keys:
        if all(val_n[t] >= 1 for t in LYMPH_TYPES):
            break
        rows = meta[meta['donor_key'] == key]
        add = rows['subtype'].value_counts().to_dict()
        if any(count[t] - val_n[t] - n < min_train for t, n in add.items()):
            continue
        if all(val_n[t] >= 1 for t in add):          # nothing new needed from this key
            continue
        val.update(rows.index)
        for t, n in add.items():
            val_n[t] += n
    if any(val_n[t] < 1 for t in LYMPH_TYPES):
        raise ValueError(f'Could not give every subtype a validation array: {val_n}')
    return val


def _swap_b_arrays(base, b_arrays, rng):
    """`base` (a GSE103541 split) with each donor's B array replaced by a distinct array of
    `b_arrays` (a one-type panel). Donor ids become `BC:<patient>+<array>`."""
    donors = sorted(base.donors.unique())
    chosen = rng.choice(np.array(b_arrays.betas.columns), len(donors), replace=False)
    cols, names, ts, ids = [], [], [], []
    for donor, arr in zip(donors, chosen):
        new_id = 'BC:' + donor.split(':', 1)[1] + '+' + arr
        for t in pd.unique(base.types):
            if t == 'B':
                vec, src = b_arrays.betas[arr], arr
            else:
                src = base.types.index[(base.types == t) & (base.donors == donor)][0]
                vec = base.betas[src]
            cols.append(vec.to_numpy(np.float32)); names.append(f'{src}@{new_id}'); ts.append(t); ids.append(new_id)
    return SortedPanel(pd.DataFrame(np.stack(cols, axis=1), index=base.betas.index, columns=names),
                       pd.Series(ts, index=names), pd.Series(ids, index=names), 'GSE103541+GSE137594',
                       'GSE103541 Noob, B arrays replaced by GSE137594 B arrays (pseudo-donors)')


def build_v7_panels(seed=0, min_range=0.3, n_validation_103541=7, lymph_min_range=0.15,
                    n_lymph=(24, 6), n_validation_bcell=24, max_missing=0.05,
                    sources=('gse35069', 'joint', 'lymph', 'bcell')):
    """v6's two panels plus 'lymph' (GSE174666 subtype pseudo-donors) and 'bcell' (GSE137594 B arrays).

    Differences from `build_v6_panels`, all new here:
    - GSE103541 masking: the > `max_missing` probe filter and the type-mean fill are fitted on the
      training donors only (`fit_mask_and_fill`) and applied to the validation donors. The donor
      split equals v6's for the same seed, so the two routes can be compared probe by probe.
    - 'lymph': 6 real lymphocyte subtypes (`LYMPH_TYPES`) from the 36 GSE174666 arrays left after
      the SNP-overlap drop with the reserved GSE167998. Arrays are split into training and
      validation BEFORE pairing (`split_lymph_arrays`, by donor key); `n_lymph` pseudo-donors
      (training, validation) are then drawn inside each split (`pair_pseudo_donors`). The four
      subtypes with a single kept array (`LYMPH_UNUSED`) are left out.
    - 'bcell': GSE103541's five types, each donor's B array swapped for a distinct GSE137594 B array
      of the same split (`_swap_b_arrays`). One pseudo-donor per GSE103541 donor, so no donor's
      other-type profiles appear twice. The 119 B arrays are split `n_validation_bcell` / rest first.
    All panels share one CpG index: probes finite in GSE35069 and kept by every loaded source's
    training-only mask, then those whose type-mean range reaches `min_range` (`lymph_min_range`
    for 'lymph') on that panel's TRAINING data. Returns the v6 layout plus 'info'."""
    try:
        from . import datasets as ds
    except ImportError:
        import datasets as ds
    p35 = load_gse35069(types=SORTED_TYPES + ('Gran',))
    p103 = load_gse103541_panel_raw()
    common = p35.betas.index.intersection(p103.betas.index)
    p103 = SortedPanel(p103.betas.loc[common], p103.types, p103.donors, p103.source, p103.preprocessing)
    ids = np.random.default_rng([seed, 103541]).permutation(np.array(sorted(p103.donors.unique())))
    train103, val103 = p103.subset_donors(ids[n_validation_103541:]), p103.subset_donors(ids[:n_validation_103541])
    mf103 = fit_mask_and_fill(train103, max_missing)
    keep = mf103.probes
    info = {'gse103541_probes_after_training_mask': int(len(keep))}
    extra = {}
    previous, ds.PREPROC = ds.PREPROC, 'noob'
    try:
        if 'lymph' in sources:
            d = ds.load_gse174666()
            used = d.meta.index[d.meta['subtype'].isin(LYMPH_TYPES)]
            cg = d.betas.index.str.startswith('cg')
            ly = SortedPanel(d.betas.loc[cg].reindex(common)[used].astype(np.float32),
                             d.meta.loc[used, 'subtype'].astype(str), pd.Series(used, index=used).astype(str),
                             'GSE174666', 'GSE174666 Noob, SNP-overlap arrays dropped, subtype pseudo-donors')
            val_gsm = split_lymph_arrays(d.meta, seed)
            ly_val = ly.subset_donors([g for g in used if g in val_gsm])
            ly_train = ly.subset_donors([g for g in used if g not in val_gsm])
            mf = fit_mask_and_fill(ly_train, max_missing)
            keep = keep.intersection(mf.probes)
            extra['lymph'] = (ly_train, ly_val, mf)
        if 'bcell' in sources:
            d = ds.load_gse137594()
            cg = d.betas.index.str.startswith('cg')
            bb = SortedPanel(d.betas.loc[cg].reindex(common).astype(np.float32), d.cell_type.astype(str),
                             pd.Series(d.betas.columns, index=d.betas.columns).astype(str), 'GSE137594',
                             'GSE137594 Noob, B-cell arrays of arthritis patients')
            perm = np.random.default_rng([seed, 137594]).permutation(np.array(sorted(bb.donors.unique())))
            b_val, b_train = bb.subset_donors(perm[:n_validation_bcell]), bb.subset_donors(perm[n_validation_bcell:])
            mf = fit_mask_and_fill(b_train, max_missing)
            keep = keep.intersection(mf.probes)
            extra['bcell'] = (b_train, b_val, mf)
    finally:
        ds.PREPROC = previous
    keep = keep.sort_values()
    info['shared_probes_before_range_screen'] = int(len(keep))
    p35 = SortedPanel(p35.betas.loc[keep], p35.types, p35.donors, p35.source, p35.preprocessing)
    train35, val35 = split_donors(p35, seed=seed)
    tr103, va103 = mf103.apply(train103, keep), mf103.apply(val103, keep)

    def seven(p):
        cols = p.types.index[p.types.isin(SORTED_TYPES)]
        return SortedPanel(p.betas.loc[:, cols], p.types.loc[cols], p.donors.loc[cols], p.source, p.preprocessing)

    panels, ranges = {'train': {}, 'validation': {}}, {}
    if 'gse35069' in sources:
        panels['train']['gse35069'], panels['validation']['gse35069'] = seven(train35), seven(val35)
    if 'joint' in sources:
        panels['train']['joint'], panels['validation']['joint'] = joint_panel(train35, tr103), joint_panel(val35, va103)
    if 'lymph' in sources:
        ly_train, ly_val, mf = extra['lymph']
        rng = np.random.default_rng([seed, 174667])
        panels['train']['lymph'] = pair_pseudo_donors(mf.apply(ly_train, keep), n_lymph[0], rng, 'LY', LYMPH_TYPES)
        panels['validation']['lymph'] = pair_pseudo_donors(mf.apply(ly_val, keep), n_lymph[1], rng, 'LYV', LYMPH_TYPES)
        ranges['lymph'] = lymph_min_range
        info['lymph_arrays'] = dict(train_by_type=ly_train.types.value_counts().to_dict(),
                                    validation_by_type=ly_val.types.value_counts().to_dict(),
                                    unused_subtypes=list(LYMPH_UNUSED))
    if 'bcell' in sources:
        b_train, b_val, mf = extra['bcell']
        rng = np.random.default_rng([seed, 137595])
        panels['train']['bcell'] = _swap_b_arrays(tr103, mf.apply(b_train, keep), rng)
        panels['validation']['bcell'] = _swap_b_arrays(va103, mf.apply(b_val, keep), rng)
        info['bcell_arrays'] = dict(train=int(b_train.betas.shape[1]), validation=int(b_val.betas.shape[1]))
        # The donor-separated B-array bank: one 'B' sample per GSE137594 patient (one array each), split
        # train/validation by patient as above. Episodes draw their B arrays from it (`b_bank`).
        panels['train']['bcell_bank'], panels['validation']['bcell_bank'] = mf.apply(b_train, keep), mf.apply(b_val, keep)
    screens = {name: screen_pool(p, ranges.get(name, min_range)) for name, p in panels['train'].items() if name != 'bcell_bank'}
    union = np.unique(np.concatenate(list(screens.values())))
    for split in panels.values():
        for name, p in split.items():
            split[name] = SortedPanel(p.betas.iloc[union].astype(np.float32), p.types, p.donors, p.source, p.preprocessing)
    pools = {name: np.searchsorted(union, rows) for name, rows in screens.items()}
    info['final_index_size'] = int(len(union))
    info['pool_sizes'] = {n: int(len(r)) for n, r in pools.items()}
    info['donors'] = {f'{sp}/{n}': int(p.donors.nunique()) for sp in ('train', 'validation') for n, p in panels[sp].items()}
    return dict(panels, pools=pools, info=info)


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
    arrays: list | None = None      # v7: per cube donor, the set of (type, source array) it is built from
    b_roles: dict | None = None     # v7: how the B column was drawn for this episode (bank mode only)


OMISSION_MODES = ('legacy', 'retained_first')


# Reference-donor sampling (added 2026-11-24). 'legacy' is the v6/v7 behaviour and
# stays the default so frozen runs reproduce: 3-6 draws WITH replacement from the eligible pool, then
# the first draw overwritten by the mixture donor in the matched case. That repeats donors (a
# repeated donor counts twice in the reference mean and SD, so the SD is biased low), can overwrite
# the only other draw of a donor, and asks for more draws than a small pool holds. 'unique' reserves
# the matched donor first, caps the draw at the eligible pool, and never repeats a donor.
REF_SAMPLING_MODES = ('legacy', 'unique')
# Env override so DataLoader workers (fork or spawn) see the same mode as the parent.
REF_SAMPLING = os.environ.get('DECONV_REF_SAMPLING', 'legacy')


def _draw_reference_donors(rng, eligible, mixture_d, *, matched):
    """Indices of the reference donors of one episode (see REF_SAMPLING)."""
    if REF_SAMPLING not in REF_SAMPLING_MODES:
        raise ValueError(f'REF_SAMPLING must be one of {REF_SAMPLING_MODES}')
    size = int(rng.integers(3, 7))
    if REF_SAMPLING == 'legacy':
        draw = rng.choice(eligible, size=size, replace=True)
        if matched:
            draw[0] = mixture_d
        return draw
    pool = [d for d in eligible if not (matched and d == mixture_d)]
    head = [mixture_d] if matched else []
    take = min(size - len(head), len(pool))
    rest = rng.choice(pool, size=take, replace=False).tolist() if take > 0 else []
    return np.asarray(head + rest, dtype=int)


class NoEligibleDonors(ValueError):
    """No donor may serve as reference (e.g. every donor shares an array with the mixture donor)."""


def _b_needs_mismatch(b_bank, b_ref, b_query):
    """Reference and query B arrays from different sources only make sense for disjoint donors."""
    return b_bank is not None and b_ref != b_query
B_SOURCES = ('orig', 'bank')


def _check_bank(panel, bank):
    """The B-array bank must be a one-type ('B') panel on the same CpG rows as `panel`."""
    key = (id(panel.betas), id(bank.betas))
    if key not in _BANK_OK:
        if not panel.betas.index.equals(bank.betas.index):
            raise ValueError('b_bank must share the CpG index of the panel')
        if list(pd.unique(bank.types)) != ['B']:
            raise ValueError("b_bank must hold B arrays only (type 'B')")
        _BANK_OK.add(key)


_BANK_OK: set = set()


def _design(panel, rng, *, n_markers, n_types, missing, donor_mismatch, augment, marker_mode,
            synthetic, candidate_pool, sharpen, cross_study, mixture_group, sharpen_groups,
            omission_mode='legacy', b_bank=None, b_ref='bank', b_query='bank',
            exclude_shared_arrays=False):
    if not isinstance(rng, np.random.Generator):
        raise TypeError('rng must be a numpy.random.Generator')
    if n_types < 2 or missing < 0 or missing > 3:
        raise ValueError('Invalid task parameters')
    if synthetic not in ('sparse', 'dense'):
        raise ValueError('synthetic must be sparse or dense')
    if omission_mode not in OMISSION_MODES:
        raise ValueError(f'omission_mode must be one of {OMISSION_MODES}')
    if b_ref not in B_SOURCES or b_query not in B_SOURCES:
        raise ValueError(f'b_ref and b_query must be one of {B_SOURCES}')
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
    if marker_mode not in MARKER_MODES_V7:
        raise ValueError(f'marker_mode must be one of {MARKER_MODES_V7}')
    groups = panel.groups()
    chosen = [int(i) for i in rng.choice(len(names), min(needed, len(names)), replace=False)]
    augmented = [f'synthetic_{i+1}' for i in range(max(0, needed-len(names)))]
    all_names = [names[i] for i in chosen] + augmented
    if omission_mode == 'legacy':
        drop = set(rng.choice(needed, missing, replace=False).tolist())
        keep = [i for i in range(needed) if i not in drop]
    else:
        # v7 'retained_first'. The retained reference is built by the rule for `n_types` columns
        # alone: min(n_types, #measured types) measured columns, the rest synthetic siblings. The
        # `missing` omitted columns are then ADDED (measured types not yet used first, siblings
        # after), so the retained columns have the same law whether or not anything is omitted. In
        # 'legacy' the total n_types+missing is built first and columns are dropped at random,
        # which puts more siblings (and sibling/parent pairs) into the retained reference of an
        # incomplete task than of a complete one. `chosen` is a random permutation, so the first
        # columns are a uniform random subset of the measured types.
        n_real_total = min(needed, len(names))
        n_real_keep = min(n_types, len(names))
        n_synth_keep = n_types - n_real_keep
        keep = list(range(n_real_keep)) + list(range(n_real_total, n_real_total + n_synth_keep))
        drop = set(range(needed)) - set(keep)
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
    arrays = None
    if exclude_shared_arrays:
        # v7: on pseudo-donor panels (lymph, legacy bcell) one measured array is part of several
        # donors. Under donor mismatch the reference must not contain any array the mixture donor
        # is built from, not only the mixture donor's own label.
        arrays = panel.donor_arrays()
        if donor_mismatch or cross_study:
            eligible = [d for d in eligible if not (arrays[d] & arrays[mixture_d])]
    if not eligible:
        raise NoEligibleDonors('No eligible reference donors')
    ref_draw = _draw_reference_donors(rng, eligible, mixture_d,
                                      matched=not donor_mismatch and not cross_study)
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
    b_roles = None
    if b_bank is not None and 'B' in names:
        # v7: the B profile of each donor of this episode comes from the B-array bank (a different
        # array for every donor, redrawn every episode) or stays the panel's own. Reference-role
        # donors (`ref_draw`) take `b_ref`, all others (the mixture donor and cohort members) take
        # `b_query`. With b_ref != b_query a donor must not be both, so callers force donor_mismatch.
        _check_bank(panel, b_bank)
        bcube = b_bank.cube()[0]
        role_ref = np.zeros(len(donors), bool)
        role_ref[ref_draw] = True
        use_bank = np.where(role_ref, b_ref == 'bank', b_query == 'bank')
        need = int(use_bank.sum())
        if need > bcube.shape[2]:
            raise ValueError('b_bank has fewer arrays than donors that need one')
        if need:
            pick = rng.permutation(bcube.shape[2])[:need]
            sub[:, names.index('B'), np.flatnonzero(use_bank)] = bcube[idx][:, 0, :][:, pick]
        bank_cols = b_bank.cube()[2]            # bank donor ids = B array ids, in cube order
        b_roles = dict(ref=b_ref, query=b_query, n_bank_donors=need,
                       bank_arrays=tuple(bank_cols[i] for i in pick) if need else (),
                       bank_donor_slots=tuple(int(i) for i in np.flatnonzero(use_bank)))
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
    sd = refs.std(axis=2, ddof=1) if refs.shape[2] > 1 else np.zeros_like(mean)
    if marker_mode == 'informative':
        scores = np.var(mean[:, keep], axis=1)
        selected = np.argpartition(scores, len(scores)-n_markers)[-n_markers:]
    elif marker_mode == 'contrast':
        selected = _contrast_markers(mean[:, keep], sd[:, keep], len(ref_draw), n_markers)
    elif marker_mode == 'pick':
        selected = _pick_markers(mean[:, keep], sd[:, keep], len(ref_draw), n_markers)
    elif marker_mode == 'real':
        selected = _real_markers(mean[:, keep], sd[:, keep], len(ref_draw), n_markers)
    else:
        selected = np.arange(n_markers)
    probes = tuple(panel.betas.index[idx[selected]].astype(str))
    return _Design(profiles, mean, sd, selected, keep, drop, all_names, names, parents, augmented,
                   donors, mixture_d, ref_draw, ref_donors, probes, needed, arrays, b_roles)


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
                  cross_study=False, mixture_group=None, sharpen_groups=None,
                  omission_mode='legacy', b_bank=None, b_ref='bank', b_query='bank', exclude_shared_arrays=False):
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
    donor_mismatch = donor_mismatch or _b_needs_mismatch(b_bank, b_ref, b_query)
    design = _design(panel, rng, n_markers=n_markers, n_types=n_types, missing=missing,
                     donor_mismatch=donor_mismatch, augment=augment, marker_mode=marker_mode,
                     synthetic=synthetic, candidate_pool=candidate_pool, sharpen=sharpen,
                     cross_study=cross_study, mixture_group=mixture_group, sharpen_groups=sharpen_groups,
                     omission_mode=omission_mode, b_bank=b_bank, b_ref=b_ref, b_query=b_query,
                     exclude_shared_arrays=exclude_shared_arrays)
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
                    tight_prob=0.5, tight_concentration=(20.0, 500.0), sample_shift=SAMPLE_SHIFT,
                    omission_mode='legacy', b_bank=None, b_ref='bank', b_query='bank', exclude_shared_arrays=False):
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
    donor_mismatch = donor_mismatch or _b_needs_mismatch(b_bank, b_ref, b_query)
    design = _design(panel, rng, n_markers=n_markers, n_types=n_types, missing=missing,
                     donor_mismatch=donor_mismatch, augment=augment, marker_mode=marker_mode,
                     synthetic=synthetic, candidate_pool=candidate_pool, sharpen=sharpen,
                     cross_study=cross_study, mixture_group=mixture_group, sharpen_groups=sharpen_groups,
                     omission_mode=omission_mode, b_bank=b_bank, b_ref=b_ref, b_query=b_query,
                     exclude_shared_arrays=exclude_shared_arrays)
    groups = panel.groups()
    used = set(design.ref_draw.tolist()) if (donor_mismatch or cross_study) else set()
    members = [int(d) for d in np.flatnonzero(groups == groups[design.mixture_d]) if d not in used]
    if design.arrays is not None and (donor_mismatch or cross_study):
        ref_arrays = set().union(*(design.arrays[d] for d in design.ref_draw))
        members = [d for d in members if not (design.arrays[d] & ref_arrays)] or [design.mixture_d]
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


def simulate_omission_pair(panel, rng, masses=(0.0, 0.1, 0.3), *, n_markers=300, n_types=6, missing=1,
                           donor_mismatch=False, augment=True, noise=0.02, marker_mode="real",
                           synthetic="sparse", concentration_range=(8, 30), neutrophil_weight=4.0,
                           tech_shift=False, tech_global=(0.0, 0.04), tech_batch=(0.0, 0.04),
                           tech_warp_scale=(0.9, 1.1), tech_warp_shift_sd=0.1,
                           tech_cpg_m_sd=(0.0, 0.0), tech_cpg_m_df=3.0, tech_cpg_m_frac=(1.0, 1.0),
                           candidate_pool=None, sharpen=(1.0, 1.0), cross_study=False, mixture_group=None,
                           sharpen_groups=None, b_bank=None, b_ref='bank', b_query='bank',
                           exclude_shared_arrays=False):
    """v7: bulks that differ ONLY in the omitted mass, measured against one fixed reference.

    One design (`omission_mode='retained_first'`, `missing` >= 1): the same retained reference,
    reference donors, markers, mixture donor and omitted columns for every entry of `masses`. A
    base composition is drawn once; the retained types keep their relative proportions and the
    omitted columns keep theirs, and only the total omitted mass w changes: retained = (1-w) *
    relative_retained, omitted = w * relative_omitted. The measurement noise draws and the technical
    shift (one seed, drawn once) are identical across the pair. `masses[0]` = 0.0 is the complete
    bulk of the same panel. Every task carries the same reference arrays, so a score that reads only
    the reference ties within each pair by construction, and the label `fractions[-1]` equals the
    mass. Which column is omitted (a near-sibling of a retained type, or a well-separated one) is
    the design's draw; read `dropped_parents` and `augmented_types` of the tasks to stratify.
    Returns a list of SimulationTask in the order of `masses`.
    """
    if missing < 1 or noise < 0 or not all(0 <= w < 1 for w in masses):
        raise ValueError('Invalid pair parameters: need missing >= 1 and masses in [0, 1)')
    donor_mismatch = donor_mismatch or _b_needs_mismatch(b_bank, b_ref, b_query)
    design = _design(panel, rng, n_markers=n_markers, n_types=n_types, missing=missing,
                     donor_mismatch=donor_mismatch, augment=augment, marker_mode=marker_mode,
                     synthetic=synthetic, candidate_pool=candidate_pool, sharpen=sharpen,
                     cross_study=cross_study, mixture_group=mixture_group, sharpen_groups=sharpen_groups,
                     omission_mode='retained_first', b_bank=b_bank, b_ref=b_ref, b_query=b_query,
                     exclude_shared_arrays=exclude_shared_arrays)
    base = _draw_fractions(design, rng, concentration_range, neutrophil_weight, 0.0)
    keep, drop = design.keep, sorted(design.drop)
    rel_keep = base[keep] / base[keep].sum()
    rel_drop = base[drop] / base[drop].sum() if base[drop].sum() > 0 else np.full(len(drop), 1 / len(drop))
    noise_seed = int(rng.integers(0, 2**63 - 1))
    shift_seed = int(rng.integers(0, 2**63 - 1))
    tasks = []
    for w in masses:
        fractions = np.zeros(design.needed)
        fractions[keep] = (1 - w) * rel_keep
        fractions[drop] = w * rel_drop
        bulk = _mix(design, np.random.default_rng(noise_seed), fractions, design.mixture_d, noise)
        if tech_shift:
            bulk = _technical_shift(bulk, np.random.default_rng(shift_seed), tech_global, tech_batch,
                                    tech_warp_scale, tech_warp_shift_sd,
                                    tech_cpg_m_sd, tech_cpg_m_df, tech_cpg_m_frac)
        tasks.append(_task(design, panel, bulk, fractions, design.mixture_d))
    return tasks
