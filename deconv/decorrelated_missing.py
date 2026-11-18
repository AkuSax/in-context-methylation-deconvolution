"""Does any score track omitted-type mass once composition is no longer a confound?

GSE110554 cannot answer this: its 12 mixtures trade neutrophils off against every other type,
so any score that rises when neutrophils are few looks like it tracks the omitted fraction
(results/missing_mass_bounds/). Here the omitted fraction is drawn independently of the rest.

Design (in silico, linear in beta, plus N(0, NOISE) per CpG, clipped to [0, 1]):
- Profiles: the 7 GSE103541 donors held out of v6 training (its `validation_donors`).
  Each mixture uses one donor's five sorted profiles (B, CD4T, CD8T, Mono, Gran).
- For omitted type d: u ~ U(0, U_MAX); the other four share 1 - u with conditional
  proportions ~ Dirichlet(ALPHA * blood-like mean), independent of u.
- Each mixture set is scored twice on the same samples: with the reduced reference
  (d removed before marker selection, strict protocol) and the FULL reference. A score that
  tracks missing mass must follow u under the reduced reference and not under the full one.
  The full-reference run is also the complete-but-shifted control in the cross-study scenario.
- Scenarios: `within` = reference from the 20 GSE103541 v6-training donors (same study,
  same array, same processing); `cross` = reference from all six GSE35069 donors
  (450K, deposited processing; Gran is GSE35069's granulocyte fraction).

Scores: per-sample LP lower bound (z = 0, 2; missing_mass_bounds.py), cohort residual PC1
(sign fixed by the bound), CETYGO, 1 - sum of NNLS and CP fractions, and the frozen
v6 no-cohort model's `other`. The model is in-distribution here: v6 trained on these two
sources (other donors, same studies), and these donors were its checkpoint-selection
validation set. This is a test of whether its `other` follows missing mass without the
confound, not a transfer test. No real mixture labels are used; nothing touches GSE110554
or the reserved test series.

    python decorrelated_missing.py      # writes results/decorrelated_missing/
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

import eval_real_dev
import reference
import simulator as sim
import solvers
from diagnostics import cetygo
from eval_real_dev import auroc, run_methods
from missing_mass_bounds import FLOOR, cohort, lower_bound
from model import Deconvolver

TYPES = ["Gran", "CD4T", "CD8T", "B", "Mono"]
BASE = {"Gran": 0.62, "CD4T": 0.15, "CD8T": 0.08, "B": 0.06, "Mono": 0.09}
ALPHA, U_MAX, NOISE, N_MIX = 20.0, 0.3, 0.01, 60
CKPT = Path("results/gpu_v6_nocohort_seed0")


def load_model():
    s = json.loads((CKPT / "summary.json").read_text())
    st = s["settings"]
    m = Deconvolver(width=st["width"], heads=st["heads"], blocks=st["blocks"], cohort=st.get("cohort", False))
    m.load_state_dict(torch.load(CKPT / "model.pt", map_location="cpu"))
    m.eval()
    eval_real_dev.PRIOR_OTHER = float(s.get("best", {}).get("prior_mean_other", eval_real_dev.PRIOR_OTHER))
    return m, s


def mixtures(profiles, donors, drop, rng):
    """profiles: {(donor, type): beta vector}. Returns (CpG x N bulk, u, conditional props)."""
    kept = [t for t in TYPES if t != drop]
    a = ALPHA * np.array([BASE[t] for t in kept]) / sum(BASE[t] for t in kept)
    u = rng.uniform(0, U_MAX, N_MIX)
    cond = rng.dirichlet(a, N_MIX)
    who = rng.choice(donors, N_MIX)
    cols = []
    for n in range(N_MIX):
        f = dict(zip(kept, (1 - u[n]) * cond[n]), **{drop: u[n]})
        cols.append(sum(f[t] * profiles[(who[n], t)] for t in TYPES))
    B = np.clip(np.stack(cols, 1) + rng.normal(0, NOISE, (len(cols[0]), N_MIX)), 0, 1)
    return B, u, pd.DataFrame(cond, columns=kept)


def score_panel(model, mean, sd, bulk):
    R, S, B = mean.to_numpy(float), sd.to_numpy(float), bulk.to_numpy(float)
    out = {}
    for z in (0.0, 2.0):
        eps = z * S.max(1) + (FLOOR if z > 0 else 0.0)
        out[f"lower_bound_z{z:g}"] = np.array([lower_bound(R, B[:, n], eps) for n in range(B.shape[1])])
    out["cohort_pc1"] = cohort(R, B, out["lower_bound_z2"], 2.0 * S.max(1) + FLOOR)["scores"]
    out["CETYGO"] = cetygo(R, B)
    out["NNLS_unexplained"] = np.clip(1 - solvers.nnls(R, B).sum(1), 0, 1)
    out["CP_unexplained"] = np.clip(1 - solvers.cp_inequality(R, B).sum(1), 0, 1)
    methods, _ = run_methods(model, mean, sd, bulk)
    out["model_v6_nocohort"] = methods["model"][1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--per-side", type=int, default=25)
    ap.add_argument("--out", type=Path, default=Path("results/decorrelated_missing"))
    args = ap.parse_args()
    torch.set_num_threads(8)
    model, summ = load_model()
    held = [d for d in summ["validation_donors"] if d.startswith("GSE103541:")]
    trained = [d for d in summ["train_donors"] if d.startswith("GSE103541:")]

    p35 = sim.load_gse35069(types=sim.SORTED_TYPES + ("Gran",))
    p103 = sim.load_gse103541_panel()
    joint = sim.joint_panel(p35, p103)
    betas, types, donors = joint.betas, joint.types, joint.donors
    betas = betas.loc[np.isfinite(betas.to_numpy()).all(1)]
    mix_cols = donors.index[donors.isin(held)]
    profiles = {(donors[c], types[c]): betas[c].to_numpy(float) for c in mix_cols}
    refs = {"within": donors.index[donors.isin(trained)],
            "cross": donors.index[donors.str.startswith("GSE35069:") & types.isin(TYPES)]}
    print(f"{len(betas)} common CpGs; mixture donors {held}", flush=True)

    rows = []
    for scen, cols in refs.items():
        rb, rl = betas[cols], types[cols]
        panels = {}
        for drop in TYPES:
            for kind in ("reduced", "full"):
                keep = rl.ne(drop) if kind == "reduced" else rl.notna()
                kb, kl = rb.loc[:, keep], rl[keep]
                mk = reference.pick_markers(kb, kl, args.per_side)
                panels[(drop, kind)] = reference.build_reference(kb, kl, mk)
        for seed in range(args.seeds):
            for drop in TYPES:
                rng = np.random.default_rng([seed, TYPES.index(drop), 20261112])
                B, u, cond = mixtures(profiles, held, drop, rng)
                bulk_all = pd.DataFrame(B, index=betas.index, columns=[f"m{n}" for n in range(N_MIX)])
                for kind in ("reduced", "full"):
                    mean, sd = panels[(drop, kind)]
                    sc = score_panel(model, mean, sd, bulk_all.loc[mean.index])
                    for name, v in sc.items():
                        rows += [dict(scenario=scen, seed=seed, dropped=drop, reference=kind, score=name,
                                      sample=n, value=float(v[n]), u_true=float(u[n]), n_markers=len(mean))
                                 for n in range(N_MIX)]
                print(f"{scen} seed {seed} drop {drop}", flush=True)

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "scores_long.csv.gz", index=False)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan

    per = []
    for (scen, seed, drop, name), g in df.groupby(["scenario", "seed", "dropped", "score"]):
        r, f = g[g.reference == "reduced"], g[g.reference == "full"]
        rec = dict(scenario=scen, seed=seed, dropped=drop, score=name,
                   r_reduced=corr(r.value, r.u_true), r_full=corr(f.value, f.u_true),
                   auroc=auroc(f.value, r[r.u_true >= 0.05].value),
                   mean_reduced=r.value.mean(), mean_full=f.value.mean(),
                   slope_reduced=float(np.polyfit(r.u_true, r.value, 1)[0]))
        if name.startswith(("lower_bound", "model", "NNLS", "CP")):
            rec.update(mae_reduced=float((r.value - r.u_true).abs().mean()),
                       covered=float((r.value <= r.u_true + 1e-6).mean()))
        per.append(rec)
    per = pd.DataFrame(per)
    per.to_csv(args.out / "per_seed.csv", index=False)
    agg = per.groupby(["scenario", "dropped", "score"]).agg(["mean", "std"]).drop(columns="seed")
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(args.out / "summary.csv", index=False)
    const = dict(mae_oracle_constant=U_MAX / 4, mae_prior_constant=float(
        np.mean(np.abs(np.random.default_rng(0).uniform(0, U_MAX, 100000) - eval_real_dev.PRIOR_OTHER))),
        prior_mean_other=eval_real_dev.PRIOR_OTHER)
    (args.out / "run.json").write_text(json.dumps(dict(
        seeds=args.seeds, per_side=args.per_side, n_mix=N_MIX, alpha=ALPHA, u_max=U_MAX, noise=NOISE,
        base=BASE, mixture_donors=held, n_within_ref_donors=len(trained), checkpoint=str(CKPT),
        n_cpgs=int(len(betas)), **const), indent=2))
    show = ["scenario", "dropped", "score", "r_reduced_mean", "r_full_mean", "auroc_mean",
            "mean_reduced_mean", "mean_full_mean", "slope_reduced_mean", "mae_reduced_mean", "covered_mean"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 500):
        print(agg[show].round(3).to_string(index=False))
        print(const)


if __name__ == "__main__":
    main()
