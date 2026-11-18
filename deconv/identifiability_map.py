"""Which omitted cell types could a bulk + reference reveal at all?

For each sorted-cell source and each type d: pick markers on the retained types exactly as
the strict drop-one protocol does (`reference.pick_markers` after removing d), then find the
convex mixture of the retained type means closest to d's mean profile at those markers.
A small residual means omitting d is (nearly) unidentifiable: a bulk with missing share f
of d looks like a complete bulk with shifted retained fractions, up to f x residual.

Uses sorted profiles only; no mixture labels. GSE167998 (final-test reference) is excluded
by design. For GSE110554 the per-drop `other` tracking of frozen checkpoints is joined from
their existing development estimates (labels already seen; nothing is re-run).

    python identifiability_map.py            # writes results/identifiability_map/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("DECONV_PREPROC", "noob")

import numpy as np
import pandas as pd
from scipy import stats

import datasets as ds
import reference
import simulator as sim
from diagnostics import hull_fit

TRACKED = {
    "v5": Path("results/dev_gse110554_gpu_v5/estimates_long.csv"),
    "v6_nocohort": Path("results/real_check_gpu_v6/dev_nocohort/estimates_long.csv"),
    "v6_cohort": Path("results/real_check_gpu_v6/dev_cohort/estimates_long.csv"),
}


def sources():
    d = ds.load("GSE110554").sorted
    yield "GSE110554", d.betas, d.cell_type.loc[d.betas.columns].astype(str)
    d = ds.load("GSE103541")
    yield "GSE103541", d.betas, d.cell_type.loc[d.betas.columns].astype(str)
    p = sim.load_gse35069()
    yield "GSE35069", p.betas, p.types.loc[p.betas.columns].astype(str)


def map_source(name, betas, labels, per_side):
    betas = betas.loc[reference.usable_probes(betas)]
    rows = []
    for d in pd.unique(labels):
        keep = labels.ne(d).to_numpy()
        kb, kl = betas.loc[:, keep], labels[keep]
        markers = reference.pick_markers(kb, kl, per_side)
        mean, sd = reference.build_reference(kb, kl, markers)
        sig = betas.loc[markers, ~keep].mean(axis=1).to_numpy()
        w, res = hull_fit(mean.to_numpy(), sig)
        noise = float(np.median(sd.to_numpy()))
        order = np.argsort(-w)
        rows.append(dict(
            source=name, per_side=per_side, dropped=d, n_markers=len(markers), n_dropped_arrays=int((~keep).sum()),
            hull_resid_rms=float(np.sqrt((res ** 2).mean())),
            hull_resid_over_sd=float(np.sqrt((res ** 2).mean()) / noise),
            frac_markers_resid_gt_0_2=float((np.abs(res) > 0.2).mean()),
            nearest_mix=" + ".join(f"{mean.columns[i]}:{w[i]:.2f}" for i in order[:3] if w[i] >= 0.01),
        ))
    return rows


def tracking(per_side):
    """Per drop, from saved estimates: OLS slope (with intercept) and Pearson r of the `other`
    estimate on the true omitted share across the 12 mixtures. An intercept is fitted so a
    constant false `other` does not pass for tracking."""
    out = {}
    for tag, path in TRACKED.items():
        if not path.exists():
            continue
        e = pd.read_csv(path)
        e = e[(e.per_side == per_side) & (e.method == "model") & (e.dropped != "none")]
        for d, s in e.groupby("dropped"):
            fit = stats.linregress(s.other_true, s.other_est)
            out.setdefault(f"other_slope_{tag}", {})[d] = float(fit.slope)
            out.setdefault(f"other_r_{tag}", {})[d] = float(fit.rvalue)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-side", type=int, nargs="+", default=[25, 12, 50])
    ap.add_argument("--out", type=Path, default=Path("results/identifiability_map"))
    args = ap.parse_args()
    rows = []
    for name, betas, labels in sources():
        for nps in args.per_side:
            rows += map_source(name, betas, labels, nps)
            print(f"{name} per_side {nps} done", flush=True)
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "hull_map.csv", index=False)

    joined = []
    for nps in args.per_side:
        g = df[(df.source == "GSE110554") & (df.per_side == nps)].set_index("dropped")
        t = tracking(nps)
        j = g.join(t, how="left").reset_index()
        for col in t.columns:
            rho = stats.spearmanr(j.hull_resid_rms, j[col]).statistic
            j[f"spearman_{col}"] = rho
        joined.append(j)
    pd.concat(joined).to_csv(args.out / "gse110554_vs_tracking.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[df.per_side == 25].sort_values(["source", "hull_resid_rms"]).round(3).to_string(index=False))
        print(joined[0].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
