"""BLEND-M against RPC, CP and NNLS on the same markers.

Part 1 (`synthetic`) reuses the decorrelated_missing.py design exactly: same seeds, mixtures,
held-out GSE103541 mixture donors and `within` / `cross` reference donors. No real mixture
labels. Part 2 (`dev`) is GSE110554 clean + strict drop-one (classical baselines only).

Two marker sets per panel, both chosen from the kept reference only:
  project = reference.pick_markers, 25 per side (EVALUATION.md shared-CpG rule)
  blendm  = BLEND-M's select_marker_cpg rule (blendm.select_markers)
Every method in a panel uses the same CpGs. BLEND-M receives individual reference samples;
the others receive the per-type mean.

    python blendm_baseline.py            # writes results/blendm_baseline/
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ["DECONV_PREPROC"] = "noob"

import numpy as np
import pandas as pd
import torch

import blendm
import datasets as ds
import decorrelated_missing as dm
import eval_real_dev
import reference
import simulator as sim
import solvers
from rpc_torch import rpc

OUT = Path("results/blendm_baseline")


def classical(mean: pd.DataFrame, bulk: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{method: (kept fractions n x K, other n)} for mean-reference solvers."""
    R = mean.to_numpy(float)
    B = bulk.to_numpy(float)
    n = B.shape[1]
    fr = rpc(np.repeat(R[None], n, 0), B.T)
    nn = solvers.nnls(R, B)
    cp = solvers.cp_inequality(R, B)
    return {"RPC": (fr, np.zeros(n)),
            "NNLS": (nn, np.clip(1 - nn.sum(1), 0, 1)),
            "CP_ineq": (cp, np.clip(1 - cp.sum(1), 0, 1))}


def blendm_methods(res: dict, kept: list[str], n: int) -> tuple[dict, dict]:
    out, fails = {}, {}
    for v, name in (("unknown", "BLENDM"), ("nounknown", "BLENDM_nounknown")):
        r = res[v]
        if isinstance(r, str):
            fails[name] = r
            continue
        other = r["unknown"].to_numpy(float) if "unknown" in r.columns else np.zeros(n)
        out[name] = (r[kept].to_numpy(float), other)
    return out, fails


def panels_for(betas: pd.DataFrame, labels: pd.Series, per_side: int, probes: pd.Index):
    """{marker set: (markers, mean)} from purified samples."""
    b = betas.loc[probes]
    pm = reference.pick_markers(b, labels, per_side)
    bm = blendm.select_markers(b, labels)
    out = {}
    for name, mk in (("project", pm), ("blendm", bm)):
        mean, _ = reference.build_reference(betas, labels, mk)
        out[name] = (mk, mean)
    return out


def synthetic(args, failures):
    torch.set_num_threads(8)
    model, summ = dm.load_model()
    held = [d for d in summ["validation_donors"] if d.startswith("GSE103541:")]
    trained = [d for d in summ["train_donors"] if d.startswith("GSE103541:")]
    gse35069 = sim.DEFAULT_GSE35069
    if not gse35069.exists():  # research-scout/ is untracked; a worktree uses the main checkout's copy
        gse35069 = Path(os.environ.get("DECONV_GSE35069", Path(__file__).resolve().parents[4]
                                       / "research-scout" / "GSE35069_series_matrix.txt.gz"))
    p35 = sim.load_gse35069(gse35069, types=sim.SORTED_TYPES + ("Gran",))
    p103 = sim.load_gse103541_panel()
    joint = sim.joint_panel(p35, p103)
    betas, types, donors = joint.betas, joint.types, joint.donors
    betas = betas.loc[np.isfinite(betas.to_numpy()).all(1)]
    mix_cols = donors.index[donors.isin(held)]
    profiles = {(donors[c], types[c]): betas[c].to_numpy(float) for c in mix_cols}
    refs = {"within": donors.index[donors.isin(trained)],
            "cross": donors.index[donors.str.startswith("GSE35069:") & types.isin(dm.TYPES)]}
    print(f"synthetic: {len(betas)} common CpGs", flush=True)

    rows, meta = [], []
    for scen, cols in refs.items():
        rb, rl = betas[cols], types[cols]
        panels = {}
        for drop in dm.TYPES:
            for kind in ("reduced", "full"):
                keep = rl.ne(drop) if kind == "reduced" else rl.notna()
                kb, kl = rb.loc[:, keep], rl[keep]
                panels[(drop, kind)] = (kb, kl, panels_for(kb, kl, args.per_side, betas.index))
                for ms, (mk, _) in panels[(drop, kind)][2].items():
                    meta.append(dict(part="synthetic", scenario=scen, dropped=drop, reference=kind,
                                     markers=ms, n_markers=len(mk), n_ref_samples=kb.shape[1]))
        print(f"{scen}: panels built", flush=True)
        for seed in range(args.seeds):
            jobs, ctx = [], []
            for drop in dm.TYPES:
                rng = np.random.default_rng([seed, dm.TYPES.index(drop), 20261112])  # same as decorrelated_missing
                B, u, cond = dm.mixtures(profiles, held, drop, rng)
                bulk_all = pd.DataFrame(B, index=betas.index, columns=[f"m{n}" for n in range(dm.N_MIX)])
                kept4 = [t for t in dm.TYPES if t != drop]
                truth = pd.DataFrame((1 - u)[:, None] * cond[kept4].to_numpy(), columns=kept4)
                truth[drop] = u
                for kind in ("reduced", "full"):
                    kb, kl, pan = panels[(drop, kind)]
                    for ms, (mk, mean) in pan.items():
                        bulk = bulk_all.loc[mk]
                        meth = classical(mean, bulk)
                        if ms == "project":
                            sd = reference.build_reference(kb, kl, mk)[1]
                            m, _ = eval_real_dev.run_methods(model, mean, sd, bulk)
                            meth["model_v6_nocohort"] = m["model"]
                        jobs.append((kb.loc[mk], kl, bulk))
                        ctx.append((drop, kind, ms, list(mean.columns), truth, meth))
            t0 = time.time()
            results = blendm.run(jobs)
            for (drop, kind, ms, kept, truth, meth), res in zip(ctx, results):
                bm, fails = blendm_methods(res, kept, dm.N_MIX)
                for k, msg in fails.items():
                    failures.append(dict(part="synthetic", scenario=scen, seed=seed, dropped=drop,
                                         reference=kind, markers=ms, method=k, error=msg))
                meth.update(bm)
                kept4 = [t for t in dm.TYPES if t != drop]
                u = truth[drop].to_numpy()
                for name, (k, o) in meth.items():
                    est = pd.DataFrame(k, columns=kept)
                    e4 = est[kept4].to_numpy() - truth[kept4].to_numpy()
                    rec = dict(part="synthetic", scenario=scen, seed=seed, dropped=drop, reference=kind,
                               markers=ms, method=name,
                               kept4_rmse_pts=100 * float(np.sqrt((e4 ** 2).mean())),
                               other_mean=float(o.mean()),
                               other_mae=float(np.abs(o - (u if kind == "reduced" else 0)).mean()),
                               other_r_u=float(np.corrcoef(o, u)[0, 1]) if np.std(o) > 0 else np.nan)
                    if kind == "full":
                        e5 = est[dm.TYPES].to_numpy() - truth[dm.TYPES].to_numpy()
                        rec["all5_rmse_pts"] = 100 * float(np.sqrt((e5 ** 2).mean()))
                    rows.append(rec)
            print(f"{scen} seed {seed}: BLEND-M {len(jobs)} tasks in {time.time() - t0:.0f} s", flush=True)
    return rows, meta


def dev(args, failures):
    data = ds.load("GSE110554")
    src, target = data.sorted, data.mixtures
    truth = target.truth
    types = list(pd.unique(src.cell_type))
    long, meta, jobs, ctx = [], [], [], []
    for drop in [None] + types:
        kept_src = src if drop is None else src.subset(src.cell_type.ne(drop))
        probes = reference.usable_probes(kept_src.betas, target.betas)
        pan = panels_for(kept_src.betas, kept_src.cell_type, args.per_side, probes)
        for ms, (mk, mean) in pan.items():
            bulk = target.betas.loc[mk]
            meta.append(dict(part="dev", dropped=drop or "none", markers=ms, n_markers=len(mk),
                             n_ref_samples=kept_src.betas.shape[1]))
            jobs.append((kept_src.betas.loc[mk], kept_src.cell_type, bulk))
            ctx.append((drop, ms, list(mean.columns), bulk.columns, classical(mean, bulk)))
    results = blendm.run(jobs)
    for (drop, ms, kept, samples, meth), res in zip(ctx, results):
        bm, fails = blendm_methods(res, kept, len(samples))
        for k, msg in fails.items():
            failures.append(dict(part="dev", dropped=drop or "none", markers=ms, method=k, error=msg))
        meth.update(bm)
        t_kept = truth.loc[samples, kept].to_numpy(float)
        t_other = truth.loc[samples, drop].to_numpy(float) if drop else np.zeros(len(samples))
        for name, (k, o) in meth.items():
            for i, s in enumerate(samples):
                long.append(dict(dropped=drop or "none", markers=ms, method=name, sample=s,
                                 sq_err_sum=float(((k[i] - t_kept[i]) ** 2).sum()), n_cells=len(kept),
                                 other_est=float(o[i]), other_true=float(t_other[i])))
    df = pd.DataFrame(long)
    summ = []
    for (ms, method), g in df.groupby(["markers", "method"]):
        for regime, sub in (("clean", g[g.dropped == "none"]),
                            ("drop_one_all", g[g.dropped != "none"]),
                            ("drop_one_abundant", g[(g.dropped != "none") & (g.other_true >= 0.05)]),
                            ("drop_one_abundant_no_Neu", g[(g.dropped != "none") & (g.dropped != "Neu")
                                                            & (g.other_true >= 0.05)])):
            summ.append(dict(markers=ms, method=method, regime=regime, n_rows=len(sub),
                             kept_rmse_pts=100 * float(np.sqrt(sub.sq_err_sum.sum() / sub.n_cells.sum())),
                             other_mae_pts=100 * float((sub.other_est - sub.other_true).abs().mean()),
                             other_bias_pts=100 * float((sub.other_est - sub.other_true).mean())))
    return df, pd.DataFrame(summ), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--per-side", type=int, default=25)
    ap.add_argument("--parts", nargs="+", default=["dev", "synthetic"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    failures, meta = [], []
    if "dev" in args.parts:
        df, summ, m = dev(args, failures)
        meta += m
        df.to_csv(OUT / "dev_estimates_long.csv", index=False)
        summ.to_csv(OUT / "dev_summary.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(summ.round(3).to_string(index=False), flush=True)
    if "synthetic" in args.parts:
        rows, m = synthetic(args, failures)
        meta += m
        per = pd.DataFrame(rows)
        per.to_csv(OUT / "synthetic_per_seed.csv", index=False)
        keys = ["scenario", "reference", "markers", "method", "dropped"]
        agg = per.drop(columns=["part", "seed"]).groupby(keys).agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg.reset_index().to_csv(OUT / "synthetic_summary.csv", index=False)
    tag = "_".join(args.parts)
    pd.DataFrame(meta).to_csv(OUT / f"panels_{tag}.csv", index=False)
    pd.DataFrame(failures, columns=["part", "error"] if not failures else None).to_csv(
        OUT / f"failures_{tag}.csv", index=False)
    (OUT / f"run_{tag}.json").write_text(json.dumps(dict(
        seeds=args.seeds, per_side=args.per_side, parts=args.parts,
        blendm_commit="90546bc3e75bf4fb3be50c1b0336b7650241f074", n_failures=len(failures)), indent=2))
    print(f"{len(failures)} BLEND-M failures", flush=True)


if __name__ == "__main__":
    main()
