"""Checks on GSE103541 (sorted EPIC, Noob) before it is used as a simulator source.

No truth labels and no benchmark series are involved. Three checks:
1. Design and array quality: donors x types, detection-p failures.
2. Purity estimate per array: NNLS of each array on the other donors' type medians at
   the 3,000 CpGs with the widest type-median range. `own` is the share assigned to the
   array's own label. This runs on every donor with data, including the excluded ones.
3. Cross-study comparison with GSE35069 (450K, deposited betas) for the five shared types:
   marker sharpness and per-CpG offsets between the two studies' type profiles.

Run from deconv/:
    python check_gse103541.py --out results/data_gse103541
"""
import argparse
import os
from pathlib import Path

os.environ["DECONV_PREPROC"] = "noob"
import numpy as np
import pandas as pd
from scipy.optimize import nnls

import datasets as ds
import simulator

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path("results/data_gse103541"))
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)

# --- 1. design and array quality, all 145 arrays -------------------------------------
_, _, full = ds._series("GSE103541")
qc = pd.read_csv(ds.CACHE / "GSE103541_noob.qc.csv")
qc["gsm"] = qc["sample"].str.split("_").str[0]
qc["title"] = qc.gsm.map(full["title"])
print("arrays", len(qc), "array type", qc["array"].iloc[0])
print(f"detection-p failures per array: median {qc.frac_detp_fail.median():.4f} max {qc.frac_detp_fail.max():.4f}")
print(qc.sort_values("frac_detp_fail", ascending=False)[["title", "frac_detp_fail"]].head(5).to_string(index=False))

d = ds.load("GSE103541")
tab = pd.crosstab(d.meta["donor"], d.cell_type)
print(f"\nloader: {d.betas.shape[1]} arrays, {len(tab)} donors, complete donors {int((tab == 1).all(axis=1).sum())},"
      f" types {d.cell_type.value_counts().to_dict()}")

# --- 2. purity estimate, every donor except the submitter's QC failure ---------------
full = full[full["patient number"].ne("Patient0")]
betas = ds._noob_betas("GSE103541", full)
lab = full["cell type"].map({"B-cells": "B", "CD4 T-cells": "CD4T", "CD8 T-cells": "CD8T",
                             "Granulocytes": "Gran", "Monocytes": "Mono"}).to_numpy()
don = full["patient number"].to_numpy()
x = betas.to_numpy()
ok = np.isfinite(x).all(axis=1) & betas.index.str.startswith("cg")
x, probes = x[ok], betas.index[ok]
types = sorted(set(lab))
trusted = ~np.isin(don, ds.GSE103541_EXCLUDED)
med = np.stack([np.median(x[:, trusted & (lab == t)], 1) for t in types], 1)
top = np.argsort(-(med.max(1) - med.min(1)))[:3000]
xt = x[top]
rows = []
for j in range(xt.shape[1]):
    others = trusted & (don != don[j])
    cen = np.stack([np.median(xt[:, others & (lab == t)], 1) for t in types], 1)
    w, _ = nnls(cen, xt[:, j])
    w = w / w.sum()
    rows.append(dict(gsm=full.index[j], donor=don[j], type=lab[j], own=w[types.index(lab[j])],
                     **{t: w[k] for k, t in enumerate(types)}))
purity = pd.DataFrame(rows)
purity.to_csv(args.out / "purity_estimate.csv", index=False)
print("\nown-type share by type (all donors with data):")
print(purity.groupby("type").own.describe()[["min", "25%", "50%", "75%", "max"]].round(3))
print("arrays with own-type share < 0.90:")
print(purity[purity.own < 0.90].sort_values("own").drop(columns="gsm").round(3).to_string(index=False))
kept = purity[~purity.donor.isin(ds.GSE103541_EXCLUDED)]
print(f"kept arrays: min own-type share {kept.own.min():.3f}; below 0.95 by type"
      f" {kept[kept.own < 0.95].groupby('type').size().to_dict()}")

# --- 3. cross-study comparison with GSE35069 ------------------------------------------
p = simulator.load_gse35069()
common = p.betas.index.intersection(d.betas.index)
pairs = {"CD4+": "CD4T", "CD8+": "CD8T", "CD19+": "B", "CD14+": "Mono", "Neu": "Gran"}
A = pd.DataFrame({b: p.betas.loc[common, p.types.index[p.types == a]].mean(1) for a, b in pairs.items()})
B = pd.DataFrame({b: d.betas.loc[common, d.betas.columns[(d.cell_type == b).to_numpy()]].median(1)
                  for b in pairs.values()})
ok = np.isfinite(B.to_numpy()).all(1)
A, B = A[ok], B[ok]
rng_a, rng_b = A.max(1) - A.min(1), B.max(1) - B.min(1)
print(f"\ncommon complete CpGs {len(A)} (GSE35069 granulocyte proxy is its neutrophil fraction)")
print(f"CpGs with type-mean range >= 0.8 over the same five types: GSE35069 {int((rng_a >= 0.8).sum())},"
      f" GSE103541 {int((rng_b >= 0.8).sum())}")
top = rng_b.nlargest(2000).index
print(f"median range at GSE103541's 2,000 widest CpGs: GSE103541 {rng_b[top].median():.2f}, GSE35069 {rng_a[top].median():.2f}")
sel = (rng_a >= 0.3) | (rng_b >= 0.3)


def logit2(v):
    v = np.clip(v, 1e-3, 1 - 1e-3)
    return np.log2(v / (1 - v))


D = (B - A)[sel]
DM = (logit2(B) - logit2(A))[sel]
print(f"offsets GSE103541 - GSE35069 at the {int(sel.sum())} CpGs with range >= 0.3 in either study:")
print(pd.DataFrame({"beta mean": D.mean(), "beta SD": D.std(), "M median (log2)": DM.median(),
                    "M SD (log2)": DM.std(), "share positive": (DM > 0).mean()}).round(3))
print("correlation of the per-CpG beta offsets between types:")
print(D.corr().round(2))
shared = D.mean(1)
print(f"offset shared by the five types: SD {shared.std():.3f}; type-specific remainder: SD"
      f" {D.sub(shared, axis=0).to_numpy().std():.3f}")
