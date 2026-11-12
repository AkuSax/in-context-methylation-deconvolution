"""Development check of a frozen model on GSE110554 (see results/dev_gse110554_gpu_v2/PROTOCOL.md).

Clean six-type panel and strict drop-one panels (the dropped type is removed
before probe filtering and marker selection). Every method uses the same CpGs.

The RPC-shrunk baseline's `other` constant is read from the checkpoint's
summary.json (`best.prior_mean_other`). Older checkpoints without that field
fall back to the v2 value 0.0909 used in the first look.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["DECONV_PREPROC"] = "noob"

import numpy as np
import pandas as pd
import torch

import datasets as ds
import reference
import solvers
from model import Deconvolver
from rpc_torch import rpc_fit

PRIOR_OTHER = 0.0909  # fallback: training-prior mean of `other` for gpu_v2_seed0 (set from the checkpoint in main)


def auroc(negative, positive):
    neg, pos = np.asarray(negative, float), np.asarray(positive, float)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def panel(src, target, n_per_side, drop=None):
    kept = src if drop is None else src.subset(src.cell_type.ne(drop))
    probes = reference.usable_probes(kept.betas, target.betas)
    markers = reference.pick_markers(kept.betas.loc[probes], kept.cell_type, n_per_side)
    mean, sd = reference.build_reference(kept.betas, kept.cell_type, markers)
    return mean, sd


def run_methods(model, mean, sd, bulk):
    """Return {method: (kept n x K, other n)} plus model intervals."""
    ref = mean.to_numpy(float)
    n = bulk.shape[1]
    R = np.repeat(ref[None], n, axis=0)
    S = np.repeat(sd.to_numpy(float)[None], n, axis=0)
    B = bulk.to_numpy(float).T
    fit = rpc_fit(R, B)
    as_t = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
    with torch.inference_mode():
        torch.manual_seed(0)
        # A v6 model treats the samples passed in one call as one cohort.
        cohort = torch.zeros(n, dtype=torch.long) if getattr(model, "cohort", False) else None
        p = model(as_t(R), as_t(S), as_t(B), fit, cohort)
        est = p.mean.numpy().astype(float)
        q = p.sample(400).quantile(torch.tensor([0.05, 0.95]), dim=0).numpy().astype(float)
    rpc = fit.fractions.numpy()
    nnls = solvers.nnls(ref, B.T)
    cp = solvers.cp_inequality(ref, B.T)
    methods = {
        "model": (est[:, :-1], est[:, -1]),
        "RPC": (rpc, np.zeros(n)),
        "RPC_shrunk_const": (rpc * (1 - PRIOR_OTHER), np.full(n, PRIOR_OTHER)),
        "NNLS": (nnls, np.clip(1 - nnls.sum(1), 0, 1)),
        "CP_ineq": (cp, np.clip(1 - cp.sum(1), 0, 1)),
    }
    return methods, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("results/gpu_v2_seed0"))
    ap.add_argument("--out", type=Path, default=Path("results/dev_gse110554_gpu_v2"))
    ap.add_argument("--per-side", type=int, nargs="+", default=[25, 12, 50])
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    global PRIOR_OTHER
    ckpt = json.loads((args.checkpoint / "summary.json").read_text())
    settings = ckpt["settings"]
    PRIOR_OTHER = float(ckpt.get("best", {}).get("prior_mean_other", PRIOR_OTHER))
    print(f"RPC-shrunk prior other = {PRIOR_OTHER:.5f}", flush=True)
    model = Deconvolver(width=settings["width"], heads=settings["heads"], blocks=settings["blocks"],
                        cohort=settings.get("cohort", False))
    model.load_state_dict(torch.load(args.checkpoint / "model.pt", map_location="cpu"))
    model.eval()
    torch.set_num_threads(8)

    data = ds.load("GSE110554")
    src, target = data.sorted, data.mixtures
    truth = target.truth
    types = list(pd.unique(src.cell_type))
    rows, long = [], []
    for nps in args.per_side:
        for drop in [None] + types:
            mean, sd = panel(src, target, nps, drop)
            kept = list(mean.columns)
            bulk = target.betas.loc[mean.index]
            methods, (low, high) = run_methods(model, mean, sd, bulk)
            t_kept = truth.loc[bulk.columns, kept].to_numpy(float)
            t_other = truth.loc[bulk.columns, drop].to_numpy(float) if drop else np.zeros(len(bulk.columns))
            for method, (k, o) in methods.items():
                for i, sample in enumerate(bulk.columns):
                    rec = dict(per_side=nps, n_markers=len(mean), dropped=drop or "none", method=method,
                               sample=sample, sq_err_sum=float(((k[i] - t_kept[i]) ** 2).sum()),
                               n_cells=len(kept), other_est=float(o[i]), other_true=float(t_other[i]))
                    if method == "model":
                        full_t = np.r_[t_kept[i], t_other[i]]
                        cov = (full_t >= low[i]) & (full_t <= high[i])
                        rec.update(covered=int(cov.sum()), n_int=len(cov), width=float((high[i] - low[i]).mean()))
                    long.append(rec)
            print(f"per_side {nps} drop {drop or 'none'}: {len(mean)} markers", flush=True)
    df = pd.DataFrame(long)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "estimates_long.csv", index=False)

    rng = np.random.default_rng(20261111)
    samples = sorted(df["sample"].unique())
    boot_idx = rng.integers(0, len(samples), size=(args.boot, len(samples)))
    for nps, g in df.groupby("per_side"):
        for regime, sub in [("clean", g[g.dropped == "none"]), ("drop_one", g[g.dropped != "none"])]:
            abundant_mask = sub.other_true >= 0.05
            for subset, s2 in [("all", sub), ("abundant", sub[abundant_mask] if regime == "drop_one" else None)]:
                if s2 is None or s2.empty:
                    continue
                for method, m in s2.groupby("method"):
                    row = dict(per_side=nps, n_markers_mean=float(m.n_markers.mean()), regime=regime, subset=subset,
                               method=method, n_rows=len(m),
                               kept_rmse_pts=100 * float(np.sqrt(m.sq_err_sum.sum() / m.n_cells.sum())),
                               other_mae_pts=100 * float((m.other_est - m.other_true).abs().mean()),
                               other_bias_pts=100 * float((m.other_est - m.other_true).mean()))
                    if method == "model":
                        row.update(coverage=float(m.covered.sum() / m.n_int.sum()), width_pts=100 * float(m.width.mean()))
                    rows.append(row)
                # paired bootstrap of model - RPC kept RMSE over mixture rows
                mm = s2[s2.method == "model"].groupby("sample")[["sq_err_sum", "n_cells"]].sum()
                rr = s2[s2.method == "RPC"].groupby("sample")[["sq_err_sum", "n_cells"]].sum()
                mm, rr = mm.reindex(samples).fillna(0), rr.reindex(samples).fillna(0)
                def rmse(t, idx):
                    return 100 * np.sqrt(t.sq_err_sum.to_numpy()[idx].sum(1) / t.n_cells.to_numpy()[idx].sum(1))
                diff = rmse(mm, boot_idx) - rmse(rr, boot_idx)
                rows.append(dict(per_side=nps, regime=regime, subset=subset, method="model_minus_RPC_boot",
                                 diff_lo_pts=float(np.percentile(diff, 2.5)), diff_hi_pts=float(np.percentile(diff, 97.5))))
        clean = g[(g.dropped == "none")]
        drops = g[(g.dropped != "none") & (g.other_true >= 0.05)]
        for method in ("model", "NNLS", "CP_ineq"):
            rows.append(dict(per_side=nps, regime="detection", subset="clean_vs_abundant", method=method,
                             auroc=auroc(clean[clean.method == method].other_est, drops[drops.method == method].other_est),
                             n_neg=int((clean.method == method).sum()), n_pos=int((drops.method == method).sum())))
        for drop, d in g[g.dropped != "none"].groupby("dropped"):
            for method, m in d.groupby("method"):
                rows.append(dict(per_side=nps, regime="per_drop", subset=drop, method=method,
                                 kept_rmse_pts=100 * float(np.sqrt(m.sq_err_sum.sum() / m.n_cells.sum())),
                                 other_mae_pts=100 * float((m.other_est - m.other_true).abs().mean()),
                                 other_true_mean_pts=100 * float(m.other_true.mean()),
                                 other_est_mean_pts=100 * float(m.other_est.mean())))
    pd.DataFrame(rows).to_csv(args.out / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
