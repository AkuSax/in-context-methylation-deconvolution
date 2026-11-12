"""Batched EpiDISH RPC in PyTorch.

EpiDISH RPC (`DoRPC`) fits `MASS::rlm(bulk ~ ref, maxit = 50)` per sample: a
Huber M-estimator (k = 1.345) with an intercept, least-squares start, MAD scale
re-estimated each iteration and IRLS until the relative residual change is at
most 1e-4. Negative coefficients are set to zero and the rest are normalised
to sum to one. This module repeats those steps for a batch of tasks, each with
its own reference, so training can compute RPC on fresh tasks without R.

Besides the published fractions it returns the fit itself (raw coefficients,
intercept, robust scale, residuals and final IRLS weights). Those are inputs
for the model: a missing cell type leaves a raw coefficient sum below one and
down-weighted outlier CpGs where its markers sit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

HUBER_K = 1.345
MAD_CONSTANT = 0.6745


@dataclass
class RPCFit:
    fractions: Tensor      # task x type, EpiDISH output (clamped, sum one)
    coefficients: Tensor   # task x type, raw rlm slopes
    intercept: Tensor      # task
    scale: Tensor          # task, final MAD scale used for the weights
    residuals: Tensor      # task x marker, bulk - intercept - ref @ coefficients
    weights: Tensor        # task x marker, final Huber IRLS weights
    iterations: Tensor     # task, IRLS iterations used


def _median(x: Tensor) -> Tensor:
    """R's median: the mean of the two middle values for an even count."""
    s = x.sort(dim=-1).values
    n = s.shape[-1]
    return s[..., n // 2] if n % 2 else 0.5 * (s[..., n // 2 - 1] + s[..., n // 2])


def _wls(design: Tensor, y: Tensor, w: Tensor) -> Tensor:
    root = w.sqrt()
    return torch.linalg.lstsq(design * root.unsqueeze(-1), (y * root).unsqueeze(-1)).solution.squeeze(-1)


def rpc_fit(reference, bulk, maxit: int = 50, acc: float = 1e-4) -> RPCFit:
    """reference: task x marker x type, bulk: task x marker. Computed in float64."""
    ref = torch.as_tensor(reference, dtype=torch.float64)
    y = torch.as_tensor(bulk, dtype=torch.float64)
    if ref.ndim != 3 or ref.shape[:2] != y.shape:
        raise ValueError("Expected task x marker x type references and task x marker bulks")
    tasks, markers, _ = ref.shape
    design = torch.cat((torch.ones(tasks, markers, 1, dtype=ref.dtype), ref), dim=-1)
    coef = _wls(design, y, torch.ones_like(y))
    resid = y - (design @ coef.unsqueeze(-1)).squeeze(-1)
    weights = torch.ones_like(y)
    scale = _median(resid.abs()) / MAD_CONSTANT
    active = torch.ones(tasks, dtype=torch.bool)
    iterations = torch.zeros(tasks, dtype=torch.long)
    for _ in range(maxit):
        if not active.any():
            break
        old = resid
        scale_now = _median(resid.abs()) / MAD_CONSTANT
        zero = scale_now == 0
        # rlm stops before reweighting when the scale is zero.
        active = active & ~zero
        if not active.any():
            break
        u = resid / scale_now.clamp_min(1e-300).unsqueeze(-1)
        w_new = torch.clamp(HUBER_K / u.abs().clamp_min(1e-300), max=1.0)
        coef_new = _wls(design, y, w_new)
        resid_new = y - (design @ coef_new.unsqueeze(-1)).squeeze(-1)
        delta = ((old - resid_new) ** 2).sum(-1).div(torch.clamp((old ** 2).sum(-1), min=1e-20)).sqrt()
        a = active.unsqueeze(-1)
        coef = torch.where(a, coef_new, coef)
        resid = torch.where(a, resid_new, resid)
        weights = torch.where(a, w_new, weights)
        scale = torch.where(active, scale_now, scale)
        iterations = iterations + active.long()
        active = active & (delta > acc)
    slopes = coef[:, 1:]
    clamped = slopes.clamp_min(0)
    total = clamped.sum(-1, keepdim=True)
    fractions = torch.where(total > 0, clamped / total.clamp_min(1e-300), torch.full_like(clamped, 1 / clamped.shape[-1]))
    return RPCFit(fractions, slopes, coef[:, 0], scale, resid, weights, iterations)


def rpc(reference, bulk) -> np.ndarray:
    """EpiDISH RPC fractions as a task x type NumPy array."""
    return rpc_fit(reference, bulk).fractions.numpy()
