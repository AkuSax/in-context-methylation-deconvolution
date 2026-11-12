"""Reference-conditioned fraction estimates and prediction distributions.

v2 inputs: besides the reference, the bulk and the RPC fractions, the model
sees the RPC robust fit itself. Per CpG it gets the normalised residual and the
final Huber weight; per type the raw slope; per task the raw slope sum,
intercept, robust scale and the share of down-weighted CpGs. A missing type
shows up there: the slope sum drops below one and its marker CpGs become
outliers. v1 saw only the residual of the normalised RPC solution and no
task-level summary, and learned a constant `other`.

v6 (`Deconvolver(cohort=True)`): cohort inputs. Samples measured against one
reference in one batch of arrays share the reference-to-bulk batch offset, while
a missing type contributes in proportion to each sample's fraction. Per CpG the
model also gets the cohort median of the RPC residual, its spread over the
cohort, and the sample's own deviation from the median; per task the share of
the sample's residual energy explained by the cohort median and the cohort
size. Subtracting the median outright halves the false `other` on real data
but also removes a type missing from the whole cohort
(results/cohort_center_gpu_v5), so it is an input, not a correction.
Models built with `cohort=False` are unchanged and load v2-v5 checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Beta, Bernoulli, Dirichlet
from torch.nn import functional as F

try:
    from .rpc_torch import RPCFit, rpc_fit
except ImportError:
    from rpc_torch import RPCFit, rpc_fit

CELL_FEATURES = 9
GLOBAL_FEATURES = 9
COHORT_CELL_FEATURES = 3
COHORT_GLOBAL_FEATURES = 2


@dataclass
class FractionPrediction:
    """Dirichlet for retained types and a zero-inflated beta for other."""

    retained: Dirichlet
    has_other: Bernoulli
    positive_other: Beta

    @property
    def mean(self) -> Tensor:
        other = (self.has_other.probs * self.positive_other.mean).unsqueeze(-1)
        return torch.cat((self.retained.mean * (1 - other), other), dim=-1)

    def sample(self, count: int) -> Tensor:
        shape = torch.Size((count,))
        other = (self.has_other.sample(shape) * self.positive_other.sample(shape)).unsqueeze(-1)
        return torch.cat((self.retained.sample(shape) * (1 - other), other), dim=-1)

    def log_prob(self, fractions: Tensor) -> Tensor:
        other = fractions[..., -1]
        kept = fractions[..., :-1]
        positive = other > 0
        # The Dirichlet density is defined in the interior of the simplex.
        kept = kept.clamp_min(1e-5)
        composition = kept / kept.sum(dim=-1, keepdim=True)
        safe_other = other.clamp(1e-5, 1 - 1e-5)
        # Account for the scaling from retained composition to total fractions.
        log_scale = (kept.shape[-1] - 1) * torch.log1p(-safe_other)
        other_log_prob = self.positive_other.log_prob(safe_other) - log_scale
        return (
            self.retained.log_prob(composition)
            + self.has_other.log_prob(positive.to(other.dtype))
            + torch.where(positive, other_log_prob, 0.0)
        )


class AxialBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        options = dict(
            d_model=width, nhead=heads, dim_feedforward=2 * width,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.probes = nn.TransformerEncoderLayer(**options)
        self.types = nn.TransformerEncoderLayer(**options)

    def forward(self, grid: Tensor) -> Tensor:
        batch, markers, types, width = grid.shape
        by_type = grid.transpose(1, 2).reshape(batch * types, markers, width)
        by_type = self.probes(by_type).reshape(batch, types, markers, width)
        by_probe = by_type.transpose(1, 2).reshape(batch * markers, types, width)
        return self.types(by_probe).reshape(batch, markers, types, width)


def fit_features(reference_mean: Tensor, reference_sd: Tensor, bulk: Tensor, fit: RPCFit) -> tuple[Tensor, Tensor]:
    """Cell features (task x CpG x type x 9) and task features (task x 9)."""
    dtype = reference_mean.dtype
    rpc = fit.fractions.to(dtype)
    slopes = fit.coefficients.to(dtype)
    scale = fit.scale.to(dtype).clamp_min(1e-6)
    resid = torch.asinh(fit.residuals.to(dtype) / scale.unsqueeze(-1))
    weight = fit.weights.to(dtype)
    shape = reference_mean.shape
    cell = torch.stack((
        reference_mean,
        reference_sd,
        bulk.unsqueeze(-1).expand(shape),
        reference_mean - bulk.unsqueeze(-1),
        rpc.unsqueeze(1).expand(shape),
        slopes.unsqueeze(1).expand(shape),
        reference_mean * rpc.unsqueeze(1),
        resid.unsqueeze(-1).expand(shape),
        weight.unsqueeze(-1).expand(shape),
    ), dim=-1)
    positive_sum = slopes.clamp_min(0).sum(-1)
    task = torch.stack((
        positive_sum - 1,
        slopes.sum(-1) - 1,
        fit.intercept.to(dtype),
        scale.log() / 4,
        1 - weight.mean(-1),
        (weight < 1).to(dtype).mean(-1),
        resid.abs().mean(-1),
        torch.full_like(positive_sum, shape[-1]).log() / 3,
        torch.full_like(positive_sum, shape[1]).log() / 6,
    ), dim=-1)
    return cell, task


def cohort_features(fit: RPCFit, cohort: Tensor | None, dtype) -> tuple[Tensor, Tensor]:
    """Cohort features: per CpG (task x CpG x 3) and per task (task x 2).

    `cohort` holds one integer id per task; tasks with the same id were measured
    against the same reference in the same batch. None means every task is alone.
    A lone task has median = its own residual, zero spread and zero deviation.
    """
    resid = fit.residuals.to(dtype)
    scale = fit.scale.to(dtype).clamp_min(1e-6)
    n = len(resid)
    if cohort is None:
        cohort = torch.arange(n, device=resid.device)
    cell = resid.new_zeros((*resid.shape, COHORT_CELL_FEATURES))
    task = resid.new_zeros((n, COHORT_GLOBAL_FEATURES))
    for group in torch.unique(cohort):
        rows = (cohort == group).nonzero().squeeze(-1)
        r = resid[rows]
        median = r.quantile(0.5, dim=0)
        deviation = r - median
        spread = deviation.abs().quantile(0.5, dim=0)
        typical = scale[rows].quantile(0.5)
        cell[rows, :, 0] = torch.asinh(median / typical).expand_as(r)
        cell[rows, :, 1] = torch.asinh(spread / typical).expand_as(r)
        cell[rows, :, 2] = torch.asinh(deviation / scale[rows].unsqueeze(-1))
        energy = r.pow(2).mean(-1).clamp_min(1e-12)
        task[rows, 0] = (1 - deviation.pow(2).mean(-1) / energy).clamp(-1, 1)
        task[rows, 1] = torch.log(torch.tensor(float(len(rows)), dtype=dtype, device=resid.device)) / 3
    return cell, task


class Deconvolver(nn.Module):
    """Take reference mean and SD and bulk beta values; RPC is computed inside
    unless a precomputed `RPCFit` is passed.

    Each call takes a batch with one marker count and one type count.
    These counts can change between calls. The last output is always other.

    With `cohort=True` (v6) the model also takes the cohort features of
    `cohort_features`; pass `cohort` ids to `forward`, or leave them out to
    treat every task as its own cohort.
    """

    def __init__(self, width: int = 32, heads: int = 4, blocks: int = 2, cohort: bool = False) -> None:
        super().__init__()
        self.cohort = cohort
        extra_cell, extra_task = (COHORT_CELL_FEATURES, COHORT_GLOBAL_FEATURES) if cohort else (0, 0)
        self.input_projection = nn.Linear(CELL_FEATURES + extra_cell, width)
        self.task_projection = nn.Sequential(nn.Linear(GLOBAL_FEATURES + extra_task, width), nn.GELU(), nn.Linear(width, width))
        self.blocks = nn.Sequential(*(AxialBlock(width, heads) for _ in range(blocks)))
        self.final_norm = nn.LayerNorm(width)
        self.type_head = nn.Linear(width, 1)
        self.concentration = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))
        self.other_head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 3))
        nn.init.zeros_(self.type_head.weight)
        nn.init.zeros_(self.type_head.bias)

    def forward(self, reference_mean: Tensor, reference_sd: Tensor, bulk: Tensor,
                fit: RPCFit | None = None, cohort: Tensor | None = None) -> FractionPrediction:
        if fit is None:
            with torch.no_grad():
                fit = rpc_fit(reference_mean.detach(), bulk.detach())
        cell, task = fit_features(reference_mean, reference_sd, bulk, fit)
        if self.cohort:
            extra_cell, extra_task = cohort_features(fit, cohort, reference_mean.dtype)
            cell = torch.cat((cell, extra_cell.unsqueeze(2).expand(*cell.shape[:3], -1)), dim=-1)
            task = torch.cat((task, extra_task), dim=-1)
        task_embedding = self.task_projection(task)
        grid = self.input_projection(cell) + task_embedding[:, None, None, :]
        grid = self.final_norm(self.blocks(grid))
        per_type = grid.mean(dim=1)
        pooled = torch.cat((per_type.mean(dim=1), task_embedding), dim=-1)

        rpc = fit.fractions.to(reference_mean.dtype)
        logits = rpc.clamp_min(0.005).log() + self.type_head(per_type).squeeze(-1)
        proportion = logits.softmax(dim=-1)
        concentration = 2 + F.softplus(self.concentration(pooled)).squeeze(-1)
        retained = Dirichlet(proportion * concentration.unsqueeze(-1) + 1e-4)

        other_logit, other_mean, other_concentration = self.other_head(pooled).unbind(dim=-1)
        other_mean = other_mean.sigmoid().clamp(1e-4, 1 - 1e-4)
        other_concentration = 2 + F.softplus(other_concentration)
        return FractionPrediction(
            retained,
            Bernoulli(logits=other_logit),
            Beta(other_mean * other_concentration, (1 - other_mean) * other_concentration),
        )
