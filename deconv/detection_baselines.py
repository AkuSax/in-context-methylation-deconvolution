"""Can a plain fit diagnostic flag an incomplete reference as well as the model's `other`?

Rebuilds the GSE110554 development panels exactly as `eval_real_dev.py` (clean six-type and
strict drop-one; markers re-picked after the drop) and scores, per mixture and panel:

  CETYGO          RMS of bulk - reference @ constrained-projection (sum <= 1) fractions
  CETYGO_eq       the same with fractions forced to sum to one
  RPC_resid_rms   RMS of the RPC (Huber rlm) residuals
  RPC_unexplained 1 - sum of the clamped raw RPC slopes (what EpiDISH renormalises away)
  NNLS_unexplained, CP_unexplained  1 - sum of fractions
  model_<tag>     the frozen checkpoint's `other`, read from its saved development estimates

Detection AUROC: clean panel rows (negatives) against drop-one rows whose omitted share is at
least 5 points (positives), as in `eval_real_dev.py`; also per dropped type. Rows share 12
mixtures, so AUROCs are descriptive. No model is run; GSE110554 labels were already used for
development.

    python detection_baselines.py          # writes results/detection_baselines/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["DECONV_PREPROC"] = "noob"

import numpy as np
import pandas as pd

import datasets as ds
import solvers
from diagnostics import cetygo
from eval_real_dev import auroc, panel
from rpc_torch import rpc_fit

MODELS = {
    "v5": Path("results/dev_gse110554_gpu_v5/estimates_long.csv"),
    "v6_nocohort": Path("results/real_check_gpu_v6/dev_nocohort/estimates_long.csv"),
    "v6_cohort": Path("results/real_check_gpu_v6/dev_cohort/estimates_long.csv"),
}
ABUNDANT = 0.05


def scores(mean, bulk):
    ref, B = mean.to_numpy(float), bulk.to_numpy(float)
    cp = solvers.cp_inequality(ref, B)
    nn = solvers.nnls(ref, B)
    eq = solvers.cp_equality(ref, B)
    fit = rpc_fit(np.repeat(ref[None], B.shape[1], axis=0), B.T)
    slopes = fit.coefficients.clamp_min(0).sum(-1).numpy()
    return {
        "CETYGO": cetygo(ref, B, cp),
        "CETYGO_eq": cetygo(ref, B, eq),
        "RPC_resid_rms": np.sqrt((fit.residuals.numpy() ** 2).mean(-1)),
        "RPC_unexplained": 1 - slopes,
        "NNLS_unexplained": np.clip(1 - nn.sum(1), 0, 1),
        "CP_unexplained": np.clip(1 - cp.sum(1), 0, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-side", type=int, nargs="+", default=[25, 12, 50])
    ap.add_argument("--out", type=Path, default=Path("results/detection_baselines"))
    args = ap.parse_args()

    data = ds.load("GSE110554")
    src, target = data.sorted, data.mixtures
    truth = target.truth
    types = list(pd.unique(src.cell_type))
    rows = []
    for nps in args.per_side:
        for drop in [None] + types:
            mean, _ = panel(src, target, nps, drop)
            bulk = target.betas.loc[mean.index]
            t_other = truth.loc[bulk.columns, drop].to_numpy(float) if drop else np.zeros(bulk.shape[1])
            for name, s in scores(mean, bulk).items():
                rows += [dict(per_side=nps, dropped=drop or "none", sample=c, score=name, value=float(v),
                              other_true=float(t)) for c, v, t in zip(bulk.columns, s, t_other)]
            print(f"per_side {nps} drop {drop or 'none'}", flush=True)
    df = pd.DataFrame(rows)
    for tag, path in MODELS.items():
        if path.exists():
            e = pd.read_csv(path)
            e = e[(e.method == "model") & e.per_side.isin(args.per_side)]
            df = pd.concat([df, pd.DataFrame(dict(per_side=e.per_side, dropped=e.dropped, sample=e["sample"],
                                                  score=f"model_{tag}", value=e.other_est, other_true=e.other_true))])
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "scores_long.csv", index=False)

    out = []
    for (nps, score), g in df.groupby(["per_side", "score"]):
        neg = g[g.dropped == "none"].value
        pos = g[(g.dropped != "none") & (g.other_true >= ABUNDANT)]
        rec = dict(per_side=nps, score=score, auroc_abundant=auroc(neg, pos.value),
                   n_neg=len(neg), n_pos=len(pos),
                   pos_excl_neu_auroc=auroc(neg, pos[pos.dropped != "Neu"].value))
        for d, p in pos.groupby("dropped"):
            rec[f"auroc_{d}"] = auroc(neg, p.value)
        out.append(rec)
    summary = pd.DataFrame(out)
    summary.to_csv(args.out / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
