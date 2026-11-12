"""Post hoc, label-using diagnostic on GSE110554 (development set). Changes no model.

Where does a checkpoint's drop-one gain come from: the `other` scalar, or
better proportions among the kept types? Also reports P(other > 0) on clean
and drop-one rows, and a thresholded point estimate (`other` = 0 when
P(other > 0) < 0.5).

Columns of the printed table, all RMSE in points against whole-mixture truth
unless noted:
  model               model as reported (predictive mean)
  rpc                 RPC, fractions sum to one
  rpc_x_model_other   RPC proportions x (1 - model `other`)
  rpc_x_true_other    RPC proportions x (1 - true omitted fraction)   [oracle scalar]
  comp_x_true_other   model proportions x (1 - true omitted fraction) [oracle scalar]
  model_thresh        model proportions x (1 - thresholded `other`)
  prop_model/prop_rpc proportions against truth renormalised to the kept types

Run from deconv/:
    python decompose_dev.py --checkpoint results/gpu_v5_seed0 --out results/audit_gse110554_gpu_v5
"""
import argparse
import json
import os
from pathlib import Path

os.environ["DECONV_PREPROC"] = "noob"
import numpy as np
import pandas as pd
import torch

import datasets as ds
from eval_real_dev import panel
from model import Deconvolver
from rpc_torch import rpc_fit

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", type=Path, default=Path("results/gpu_v5_seed0"))
ap.add_argument("--out", type=Path, default=Path("results/audit_gse110554_gpu_v5"))
ap.add_argument("--per-side", type=int, nargs="+", default=[25, 12, 50])
args = ap.parse_args()

s = json.loads((args.checkpoint / "summary.json").read_text())["settings"]
model = Deconvolver(width=s["width"], heads=s["heads"], blocks=s["blocks"])
model.load_state_dict(torch.load(args.checkpoint / "model.pt", map_location="cpu"))
model.eval()
torch.set_num_threads(8)

data = ds.load("GSE110554")
src, target = data.sorted, data.mixtures
truth = target.truth
types = list(pd.unique(src.cell_type))
rows = []
for nps in args.per_side:
    for drop in [None] + types:
        mean, sd = panel(src, target, nps, drop)
        kept = list(mean.columns)
        bulk = target.betas.loc[mean.index]
        n = bulk.shape[1]
        R = np.repeat(mean.to_numpy(float)[None], n, 0)
        S = np.repeat(sd.to_numpy(float)[None], n, 0)
        B = bulk.to_numpy(float).T
        fit = rpc_fit(R, B)
        t = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
        with torch.inference_mode():
            p = model(t(R), t(S), t(B), fit)
            est = p.mean.numpy().astype(float)
            prob = p.has_other.probs.numpy().astype(float)
            pos_mean = p.positive_other.mean.numpy().astype(float)
            comp = p.retained.mean.numpy().astype(float)
        rpc = fit.fractions.numpy()
        tk = truth.loc[bulk.columns, kept].to_numpy(float)
        to = truth.loc[bulk.columns, drop].to_numpy(float) if drop else np.zeros(n)
        for i in range(n):
            if to[i] >= 0.999:
                continue
            tprop = tk[i] / tk[i].sum()
            thresh = pos_mean[i] if prob[i] >= 0.5 else 0.0
            rows.append(dict(
                per_side=nps, drop=drop or "none", sample=bulk.columns[i], true_other=to[i],
                model_other=est[i, -1], p_other=prob[i], pos_mean=pos_mean[i], thresh_other=thresh,
                se_model=((est[i, :-1] - tk[i]) ** 2).sum(),
                se_rpc=((rpc[i] - tk[i]) ** 2).sum(),
                se_rpc_x_model_other=((rpc[i] * (1 - est[i, -1]) - tk[i]) ** 2).sum(),
                se_rpc_x_true_other=((rpc[i] * (1 - to[i]) - tk[i]) ** 2).sum(),
                se_comp_x_true_other=((comp[i] * (1 - to[i]) - tk[i]) ** 2).sum(),
                se_model_thresh=((comp[i] * (1 - thresh) - tk[i]) ** 2).sum(),
                se_prop_model=((comp[i] - tprop) ** 2).sum(),
                se_prop_rpc=((rpc[i] - tprop) ** 2).sum(),
                k=len(kept)))
df = pd.DataFrame(rows)
args.out.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out / "decompose_long.csv", index=False)

cols = [c for c in df.columns if c.startswith("se_")]
pd.set_option("display.width", 250)
summary = []
for nps, g in df.groupby("per_side"):
    regimes = [("clean", g[g["drop"] == "none"]), ("drop all", g[g["drop"] != "none"]),
               ("drop no Neu", g[(g["drop"] != "none") & (g["drop"] != "Neu")])]
    regimes += [(f"drop {d}", g[g["drop"] == d]) for d in types]
    for label, sub in regimes:
        r = dict(per_side=nps, regime=label, n=len(sub))
        r.update({c[3:]: 100 * np.sqrt(sub[c].sum() / sub.k.sum()) for c in cols})
        r.update(p_other=sub.p_other.mean(), frac_p_ge_half=(sub.p_other >= 0.5).mean(),
                 other_mae_mean=100 * (sub.model_other - sub.true_other).abs().mean(),
                 other_mae_thresh=100 * (sub.thresh_other - sub.true_other).abs().mean())
        summary.append(r)
    c = g[g["drop"] == "none"]
    print(f"per_side {nps} clean P(other > 0) per sample:", np.round(np.sort(c.p_other.to_numpy()), 2))
out = pd.DataFrame(summary)
out.to_csv(args.out / "decompose_summary.csv", index=False)
print(out.round(2).to_string(index=False))
