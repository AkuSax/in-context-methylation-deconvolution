"""Cohort-offset centering test on a frozen checkpoint (post hoc, development use).

Idea: a reference-to-bulk batch offset is the same for every sample of a cohort, while a
missing cell type contributes in proportion to its fraction. So estimate a per-CpG offset
as the cohort median of the RPC residual (label-free), subtract it from the bulk, and rerun
the model. Two variants:

  resid      subtract the per-CpG median residual
  resid_int  also subtract the cohort-median RPC intercept (the global part of the offset)

Each is iterated PASSES times (refit RPC on the centred bulk, subtract again).

Part 1 (no labels): the four unlabelled whole-blood cohorts of label_free_null.py, plus
random subsets of 12 samples to see whether a small cohort is enough.
Part 2 (labels used for scoring only): GSE110554 clean and strict drop-one panels at 25 per
side. Here the dropped type is absent from every mixture, so centring can absorb it; that
is the result to watch.

Writes results/cohort_center_gpu_v5/{null_summary.csv, null_estimates.csv, dev_long.csv,
dev_summary.csv}. Run from the deconv folder:  py -3.11 -B cohort_center.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import eval_real_dev as ev  # sets DECONV_PREPROC=noob before datasets is imported
import datasets as ds
import label_free_null as lfn
import reference
from model import Deconvolver
from rpc_torch import rpc_fit

CKPT = Path("results/gpu_v5_seed0")
OUT = Path("results/cohort_center_gpu_v5")
PER_SIDE = 25
PASSES = 3
VARIANTS = ("none", "resid", "resid_int")
SUBSET_N, SUBSET_DRAWS = 12, 20


def center(mean: pd.DataFrame, bulk: pd.DataFrame, variant: str, passes: int = PASSES) -> pd.DataFrame:
    """Bulk with the cohort's per-CpG median RPC residual removed. Uses no labels."""
    if variant == "none":
        return bulk
    ref = mean.to_numpy(float)
    B = bulk.to_numpy(float).T.copy()  # sample x marker
    for _ in range(passes):
        fit = rpc_fit(np.repeat(ref[None], len(B), axis=0), B)
        off = np.median(fit.residuals.numpy(), axis=0)
        if variant == "resid_int":
            off = off + float(np.median(fit.intercept.numpy()))
        B = np.clip(B - off[None], 0.0, 1.0)
    return pd.DataFrame(B.T, index=bulk.index, columns=bulk.columns)


def rawsum_other(mean, bulk) -> np.ndarray:
    fit = rpc_fit(np.repeat(mean.to_numpy(float)[None], bulk.shape[1], axis=0), bulk.to_numpy(float).T)
    return np.clip(1 - fit.coefficients.clamp_min(0).sum(-1).numpy(), 0, 1)


def null_part(model, src):
    rows, est = [], []
    rng = np.random.default_rng(20261112)
    for name, loader in lfn.LOADERS.items():
        print(f"[{name}] loading", flush=True)
        df, _, _ = loader()
        shared = reference.usable_probes(src.betas).intersection(pd.Index(df.index))
        sub = df.loc[shared]
        sub = sub[sub.notna().all(axis=1)].astype(np.float64)
        mean, sd = ev.panel(src, SimpleNamespace(betas=sub), PER_SIDE, None)
        bulk = sub.loc[mean.index]
        subsets = [rng.choice(bulk.shape[1], SUBSET_N, replace=False) for _ in range(SUBSET_DRAWS)]
        for variant in VARIANTS:
            b = center(mean, bulk, variant)
            methods, _ = ev.run_methods(model, mean, sd, b)
            other, raw = methods["model"][1], rawsum_other(mean, b)
            # small-cohort version: offset estimated from 12 samples, applied to those 12
            small = []
            for idx in subsets:
                bs = center(mean, bulk.iloc[:, idx], variant)
                small.append(ev.run_methods(model, mean, sd, bs)[0]["model"][1])
            small = 100 * np.concatenate(small)
            rows.append(dict(cohort=name, variant=variant, n=bulk.shape[1], n_markers=len(mean),
                             model_other_mean=100 * other.mean(), model_other_median=100 * np.median(other),
                             model_other_sd=100 * other.std(), model_frac_gt5=float((other > 0.05).mean()),
                             rpc_rawsum_mean=100 * raw.mean(),
                             rpc_shift_pts=100 * float(np.abs(methods["RPC"][0] - base_rpc).mean()) if variant != "none" else 0.0,
                             n12_model_other_mean=small.mean(), n12_model_frac_gt5=float((small > 5).mean())))
            if variant == "none":
                base_rpc = methods["RPC"][0]
                rows[-1]["rpc_shift_pts"] = 0.0
            est.append(pd.DataFrame(dict(cohort=name, variant=variant, sample_id=bulk.columns,
                                         model_other=other, rpc_rawsum_other=raw)))
            print(f"[{name}] {variant}: model other {100 * other.mean():.1f}, rawsum {100 * raw.mean():.1f}", flush=True)
    return pd.DataFrame(rows), pd.concat(est)


def dev_part(model):
    data = ds.load("GSE110554")
    src, target = data.sorted, data.mixtures
    truth = target.truth
    long = []
    for drop in [None] + list(pd.unique(src.cell_type)):
        mean, sd = ev.panel(src, target, PER_SIDE, drop)
        kept = list(mean.columns)
        bulk = target.betas.loc[mean.index]
        t_kept = truth.loc[bulk.columns, kept].to_numpy(float)
        t_other = truth.loc[bulk.columns, drop].to_numpy(float) if drop else np.zeros(bulk.shape[1])
        for variant in VARIANTS:
            methods, _ = ev.run_methods(model, mean, sd, center(mean, bulk, variant))
            for method in ("model", "RPC"):
                k, o = methods[method]
                for i, sample in enumerate(bulk.columns):
                    long.append(dict(dropped=drop or "none", variant=variant, method=method, sample=sample,
                                     sq_err_sum=float(((k[i] - t_kept[i]) ** 2).sum()), n_cells=len(kept),
                                     other_est=float(o[i]), other_true=float(t_other[i])))
        print(f"dev drop {drop or 'none'} done", flush=True)
    df = pd.DataFrame(long)

    def score(g):
        return pd.Series(dict(kept_rmse_pts=100 * np.sqrt(g.sq_err_sum.sum() / g.n_cells.sum()),
                              other_est_mean_pts=100 * g.other_est.mean(),
                              other_true_mean_pts=100 * g.other_true.mean(),
                              other_mae_pts=100 * (g.other_est - g.other_true).abs().mean(), n=len(g)))

    df["regime"] = np.where(df.dropped == "none", "clean", "drop_one")
    parts = [df.groupby(["regime", "variant", "method"]).apply(score).reset_index().assign(dropped="all"),
             df[(df.regime == "drop_one") & (df.dropped != "Neu")].groupby(["variant", "method"]).apply(score)
             .reset_index().assign(regime="drop_one_no_Neu", dropped="all"),
             df[df.regime == "drop_one"].groupby(["dropped", "variant", "method"]).apply(score)
             .reset_index().assign(regime="per_drop")]
    summ = pd.concat(parts, ignore_index=True)
    m = df[df.method == "model"]
    for variant, g in m.groupby("variant"):
        neg = g[g.dropped == "none"].other_est
        pos = g[(g.dropped != "none") & (g.other_true >= 0.05)].other_est
        summ = pd.concat([summ, pd.DataFrame([dict(regime="detection", variant=variant, method="model",
                                                   dropped="all", auroc=ev.auroc(neg, pos))])], ignore_index=True)
    return df, summ


def main():
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = json.loads((CKPT / "summary.json").read_text())
    s = ckpt["settings"]
    model = Deconvolver(width=s["width"], heads=s["heads"], blocks=s["blocks"])
    model.load_state_dict(torch.load(CKPT / "model.pt", map_location="cpu"))
    model.eval()
    pd.set_option("display.width", 250)

    long, summ = dev_part(model)
    long.to_csv(OUT / "dev_long.csv", index=False)
    summ.to_csv(OUT / "dev_summary.csv", index=False)
    print(summ.round(2).to_string(), flush=True)

    if "--dev-only" not in sys.argv:
        src = ds.load("GSE110554").sorted
        rows, est = null_part(model, src)
        rows.to_csv(OUT / "null_summary.csv", index=False)
        est.to_csv(OUT / "null_estimates.csv", index=False)
        print(rows.round(2).to_string(), flush=True)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
