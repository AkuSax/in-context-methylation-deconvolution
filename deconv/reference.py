"""Build a reference panel (marker CpGs + per-type mean and SD) from purified samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def usable_probes(*tables: pd.DataFrame) -> pd.Index:
    """CpG probes present and fully observed in every table."""
    keep = None
    for t in tables:
        ok = t.index[t.notna().all(axis=1).to_numpy()]
        keep = ok if keep is None else keep.intersection(ok)
    return keep[keep.str.startswith("cg")]


def pick_markers(sorted_betas: pd.DataFrame, labels: pd.Series, n_per_side: int = 50,
                 p_max: float = 1e-8) -> pd.Index:
    """One-vs-rest marker selection in the style of minfi's pickCompProbes('both').

    For each type: Welch t-test against all other purified samples, keep probes
    with p < p_max (falling back to the best-ranked if too few pass), then take
    the n_per_side most hypermethylated and most hypomethylated by mean difference.
    """
    x = sorted_betas.to_numpy(dtype=np.float64)
    lab = labels.loc[sorted_betas.columns].to_numpy()
    chosen: list[np.ndarray] = []
    for t in pd.unique(lab):
        a, b = x[:, lab == t], x[:, lab != t]
        diff = a.mean(axis=1) - b.mean(axis=1)
        _, p = stats.ttest_ind(a, b, axis=1, equal_var=False)
        p = np.nan_to_num(p, nan=1.0)
        for sign in (1, -1):
            d = sign * diff
            pool = np.flatnonzero((p < p_max) & (d > 0))
            if len(pool) < n_per_side:
                pool = np.flatnonzero(d > 0)
                pool = pool[np.argsort(p[pool])[: 5 * n_per_side]]
            chosen.append(pool[np.argsort(-d[pool])[:n_per_side]])
    idx = np.unique(np.concatenate(chosen))
    return sorted_betas.index[idx]


def build_reference(sorted_betas: pd.DataFrame, labels: pd.Series, markers: pd.Index,
                    types: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-type mean and SD at the marker CpGs (markers x types)."""
    lab = labels.loc[sorted_betas.columns]
    sub = sorted_betas.loc[markers]
    grouped = sub.T.groupby(lab.to_numpy(), sort=False)
    mean, sd = grouped.mean().T, grouped.std(ddof=1).T
    types = types or list(mean.columns)
    return mean[types], sd[types]
