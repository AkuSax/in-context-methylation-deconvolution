"""Reference-level identifiability and sample-level fit diagnostics.

`hull_fit` asks how close one cell type's profile is to a mixture of the other types in a
reference panel. If a type sits on that mixture, leaving it out of the reference produces
bulks that a complete-but-different mixture explains exactly, so no method can recover the
missing share from the bulk and reference alone.

`cetygo` is the sample-level error of Hannon et al. (2021, CETYGO): the root mean squared
difference between the observed bulk and the bulk reconstructed from the reference and the
Houseman constrained-projection fractions, over the deconvolution CpGs. It is a fit
diagnostic, not a missing-fraction estimate.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls as _nnls

import solvers


def hull_fit(ref: np.ndarray, sig: np.ndarray, lam: float = 1e3) -> tuple[np.ndarray, np.ndarray]:
    """Closest convex combination of the columns of `ref` (markers x types) to `sig`.

    Returns (weights summing to one, residual sig - ref @ weights). Uses NNLS with a heavily
    weighted sum-to-one row, then renormalises.
    """
    k = ref.shape[1]
    w, _ = _nnls(np.vstack([ref, lam * np.ones(k)]), np.r_[sig, lam])
    w = w / w.sum() if w.sum() > 0 else np.full(k, 1.0 / k)
    return w, sig - ref @ w


def cetygo(ref: np.ndarray, bulk: np.ndarray, fractions: np.ndarray | None = None) -> np.ndarray:
    """CETYGO per sample. `ref` is markers x types, `bulk` markers x samples.

    `fractions` (samples x types) defaults to constrained projection with sum <= 1, the
    Houseman/minfi estimator the CETYGO package uses.
    """
    if fractions is None:
        fractions = solvers.cp_inequality(ref, bulk)
    recon = ref @ fractions.T
    return np.sqrt(((bulk - recon) ** 2).mean(axis=0))
