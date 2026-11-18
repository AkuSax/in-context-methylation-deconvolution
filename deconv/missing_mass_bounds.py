"""How much omitted-type mass can the data support, without assuming the omitted profile?

Two prototypes, both training-free, on the GSE110554 strict drop-one development panels.

1. Per-sample lower bound (partial identification). Model b = R v + u q with v >= 0,
   sum(v) + u = 1 and an unknown omitted profile q in [0, 1]^p. Any u above the smallest
   feasible one is also feasible (blend q towards the retained mixture), so the identified
   set is [u_lb, 1]. u_lb is one linear program:

       minimise u  s.t.  -eps <= b - R v <= u + eps,  v >= 0,  sum(v) + u = 1

   eps_j = z * max_k sd_jk + floor (donor SD of the retained types at CpG j) relaxes the
   box for noise; z = 0 is the noise-free bound. With eps > 0 the bound can be invalid
   (exceed the truth); coverage on the clean and drop-one panels measures that.

2. Cohort residual (shared omitted profile). Projecting each bulk onto the orthogonal
   complement of span(R) leaves u_n * P_perp q: rank one across samples if the omitted type
   is shared and its fraction varies. Reference error Delta w_n is also low rank, so the
   clean panel is the control. Reported: PC1 share of residual variance, and correlation of
   PC1 scores with the true omitted fraction. The sign is fixed without labels by making
   the scores correlate positively with the per-sample lower bounds.
   Anchored scale: write q = R a + lam d with d the unit PC1 direction. The box constraint
   on q caps lam (an LP in a, lam); with scores t_n = d . E_n, u_n >= t_n / lam_max. This
   assumes no shared offset in the residuals; the clean panel shows how badly that fails.

CpG sets: `panel` = markers re-picked on the retained types (strict protocol);
`panel+var` = panel plus the top-N CpGs by bulk variance across the 12 mixtures (bulk only,
no labels). GSE110554 labels are used only to score; this is another development look. No model is run.

    python missing_mass_bounds.py        # writes results/missing_mass_bounds/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["DECONV_PREPROC"] = "noob"

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import datasets as ds
import reference
from eval_real_dev import panel

ZS = [0.0, 1.0, 2.0, 3.0]
FLOOR = 0.01
SCORES = Path("results/detection_baselines/scores_long.csv")


def lower_bound(R: np.ndarray, b: np.ndarray, eps: np.ndarray) -> float:
    p, k = R.shape
    c = np.r_[np.zeros(k), 1.0]
    # R v <= b + eps ;  -R v - u <= eps - b
    A = np.block([[R, np.zeros((p, 1))], [-R, -np.ones((p, 1))]])
    ub = np.r_[b + eps, eps - b]
    res = linprog(c, A_ub=A, b_ub=ub, A_eq=np.ones((1, k + 1)), b_eq=[1.0],
                  bounds=[(0, None)] * k + [(0, 1)], method="highs")
    return float(res.x[-1]) if res.status == 0 else np.nan


def lam_max(R: np.ndarray, d: np.ndarray, eps: np.ndarray) -> float:
    """Largest lam with -eps <= R a + lam d <= 1 + eps for some real a."""
    p, k = R.shape
    c = np.r_[np.zeros(k), -1.0]
    M = np.c_[R, d]
    A = np.vstack([M, -M])
    ub = np.r_[1 + eps, eps]
    res = linprog(c, A_ub=A, b_ub=ub, bounds=[(None, None)] * k + [(0, None)], method="highs")
    return float(res.x[-1]) if res.status == 0 else np.nan


def cohort(R: np.ndarray, B: np.ndarray, lb: np.ndarray, eps: np.ndarray):
    Q, _ = np.linalg.qr(R)
    E = B - Q @ (Q.T @ B)                       # p x N residuals
    Ec = E - E.mean(1, keepdims=True)
    U, s, Vt = np.linalg.svd(Ec, full_matrices=False)
    d = U[:, 0]
    t = d @ E                                    # uncentered scores along PC1
    flip = np.corrcoef(t, lb)[0, 1] < 0 if np.std(lb) > 0 else t.mean() < 0
    if flip:
        d, t = -d, -t
    lm = lam_max(R, d, eps)
    return dict(pc1_share=float(s[0] ** 2 / (s ** 2).sum()), s1_s2=float(s[0] / s[1]),
                scores=t, anchored=np.clip(t / lm, 0, 1) if lm and lm > 0 else np.full_like(t, np.nan),
                lam_max=lm)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-side", type=int, nargs="+", default=[25, 50])
    ap.add_argument("--n-var", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("results/missing_mass_bounds"))
    args = ap.parse_args()

    data = ds.load("GSE110554")
    src, target = data.sorted, data.mixtures
    truth = target.truth
    types = list(pd.unique(src.cell_type))
    usable = reference.usable_probes(src.betas, target.betas)
    bulk_all = target.betas.loc[usable].dropna()
    var_cpgs = bulk_all.var(axis=1).sort_values(ascending=False).index[: args.n_var]

    rows, coh = [], []
    for nps in args.per_side:
        for drop in [None] + types:
            mean, _ = panel(src, target, nps, drop)
            kept = src if drop is None else src.subset(src.cell_type.ne(drop))
            u_true = truth.loc[target.betas.columns, drop].to_numpy(float) if drop else np.zeros(target.betas.shape[1])
            for cset in ["panel", "panel+var"]:
                cpgs = mean.index if cset == "panel" else mean.index.union(var_cpgs)
                m, sd = reference.build_reference(kept.betas, kept.cell_type, cpgs)
                B = target.betas.loc[cpgs]
                ok = ~(B.isna().any(axis=1) | m.isna().any(axis=1) | sd.isna().any(axis=1))
                R, S, Bm = m[ok].to_numpy(float), sd[ok].to_numpy(float), B[ok].to_numpy(float)
                for z in ZS:
                    eps = z * S.max(1) + (FLOOR if z > 0 else 0.0)
                    lb = np.array([lower_bound(R, Bm[:, n], eps) for n in range(Bm.shape[1])])
                    c = cohort(R, Bm, lb, eps)
                    for n, col in enumerate(B.columns):
                        rows.append(dict(per_side=nps, dropped=drop or "none", cpgs=cset, z=z, sample=col,
                                         n_cpg=int(ok.sum()), u_true=u_true[n], u_lb=lb[n],
                                         pc1_score=c["scores"][n], anchored=c["anchored"][n]))
                    coh.append(dict(per_side=nps, dropped=drop or "none", cpgs=cset, z=z,
                                    pc1_share=c["pc1_share"], s1_s2=c["s1_s2"], lam_max=c["lam_max"]))
            print(f"per_side {nps} drop {drop or 'none'}", flush=True)

    df, ch = pd.DataFrame(rows), pd.DataFrame(coh)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "per_sample.csv", index=False)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan

    summ = []
    for key, g in df.groupby(["per_side", "dropped", "cpgs", "z"]):
        summ.append(dict(zip(["per_side", "dropped", "cpgs", "z"], key),
                         lb_mean=g.u_lb.mean(), truth_mean=g.u_true.mean(),
                         lb_valid=float((g.u_lb <= g.u_true + 1e-6).mean()),
                         lb_over_truth=float(g.u_lb.sum() / g.u_true.sum()) if g.u_true.sum() > 0 else np.nan,
                         r_lb=corr(g.u_lb, g.u_true), r_pc1=corr(g.pc1_score, g.u_true),
                         anchored_valid=float((g.anchored <= g.u_true + 1e-6).mean()),
                         anchored_mean=g.anchored.mean()))
    summ = pd.DataFrame(summ).merge(ch, on=["per_side", "dropped", "cpgs", "z"])

    # Existing per-sample scores on the same panels, for correlation with the true omitted fraction.
    if SCORES.exists():
        sc = pd.read_csv(SCORES)
        sc = sc[sc.dropped != "none"]
        ref_r = (sc.groupby(["per_side", "dropped", "score"])
                 .apply(lambda g: corr(g.value, g.other_true), include_groups=False)
                 .rename("r").reset_index()
                 .pivot_table(index=["per_side", "dropped"], columns="score", values="r").reset_index())
        ref_r.to_csv(args.out / "existing_scores_r.csv", index=False)
    summ.to_csv(args.out / "summary.csv", index=False)

    # Composition-confound control. The 12 mixtures trade neutrophils against everything else,
    # so a score that tracks "few neutrophils" also tracks most omitted fractions. For each
    # score: r with the omitted fraction on its drop panel, the same r after partialling out
    # the neutrophil fraction, and r of the CLEAN-panel score with that type's fraction (no
    # mass is missing there, so a high value means the score follows composition).
    def partial(x, y, z):
        X = np.c_[np.ones_like(z), z]
        res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]  # noqa: E731
        return corr(res(x), res(y))

    T = truth.loc[target.betas.columns]
    srcs = {}
    for nps in args.per_side:
        base = df[(df.per_side == nps) & (df.cpgs == "panel") & (df.z == 2.0)]
        srcs[(nps, "lower_bound_z2")] = base.rename(columns={"u_lb": "value"})
        srcs[(nps, "cohort_pc1")] = base.rename(columns={"pc1_score": "value"})
        if SCORES.exists():
            prior = pd.read_csv(SCORES)
            for s in ["CETYGO", "model_v5"]:
                srcs[(nps, s)] = prior[(prior.per_side == nps) & (prior.score == s)]
    ctrl = []
    for (nps, name), d in srcs.items():
        clean = d[d.dropped == "none"].set_index("sample").value
        for drop in [t for t in types if t != "Neu"]:
            g = d[d.dropped == drop].set_index("sample").value
            t = T.loc[g.index]
            ctrl.append(dict(per_side=nps, score=name, dropped=drop, r_drop=corr(g.values, t[drop].values),
                             r_drop_given_neu=partial(g.values, t[drop].values, t.Neu.values),
                             r_clean_same_type=corr(clean.loc[t.index].values, t[drop].values)))
    ctrl = pd.DataFrame(ctrl)
    ctrl.to_csv(args.out / "confound_control.csv", index=False)
    T.corr().to_csv(args.out / "truth_correlation.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 500):
        print(summ.round(3).to_string(index=False))
        if SCORES.exists():
            print(ref_r.round(2).to_string(index=False))
    (args.out / "run.json").write_text(json.dumps(dict(per_side=args.per_side, n_var=args.n_var, zs=ZS,
                                                        floor=FLOOR, n_samples=int(target.betas.shape[1])),
                                                   indent=2))


if __name__ == "__main__":
    main()
