"""Explicit completeness assumptions for v6 fraction predictions.

This operates on the prediction returned by Deconvolver; it does not rerun RPC
or the network. See V6_INFERENCE.md for assumptions and an example.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class V6Estimate:
    # Composition among retained types, conditional on retained mass being positive.
    retained_composition: Tensor
    # Fractions of the whole mixture, in reference order.
    retained_full_mixture: Tensor
    other_full_mixture: Tensor
    # A model score under the stated assumptions, not a calibrated probability.
    missing_type_score: Tensor
    assumption: str
    deployment_prevalence: float | None

    @property
    def full_mixture(self) -> Tensor:
        return torch.cat((self.retained_full_mixture, self.other_full_mixture.unsqueeze(-1)), dim=-1)


def estimate_v6(prediction, *, known_complete: bool = False,
                deployment_prevalence: float | None = None) -> V6Estimate:
    """Return retained composition and estimated full-mixture fractions.

    Default uncertain-panel inference reproduces the v6 model mean, using its
    synthetic training missing-type prevalence of 0.55. Set a prevalence in
    (0, 1) for label-shift sensitivity; known_complete=True instead sets other
    to zero. Prevalence and known_complete are mutually exclusive. This is for
    v6 checkpoints trained with that prior, not an automatic prior estimator.
    """
    if known_complete and deployment_prevalence is not None:
        raise ValueError("known_complete and deployment_prevalence are mutually exclusive")
    composition = prediction.retained.mean
    if known_complete:
        score = torch.zeros_like(prediction.has_other.probs)
        other = torch.zeros_like(score)
        return V6Estimate(composition, composition, other, score, "known_complete", None)

    pi = 0.55 if deployment_prevalence is None else float(deployment_prevalence)
    if not math.isfinite(pi) or not 0 < pi < 1:
        raise ValueError("deployment_prevalence must be finite and strictly between 0 and 1; use known_complete for a known complete panel")
    if pi == 0.55:
        score = prediction.has_other.probs
    else:
        shift = math.log(pi / (1 - pi)) - math.log(0.55 / 0.45)
        score = torch.sigmoid(prediction.has_other.logits + shift)
    other = score * prediction.positive_other.mean
    return V6Estimate(composition, composition * (1 - other.unsqueeze(-1)), other,
                      score, "uncertain_panel", pi)
