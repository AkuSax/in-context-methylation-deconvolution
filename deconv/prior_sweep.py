"""Missing-type prior sweep for the frozen v6 no-cohort checkpoint (CPU, no training).

New script; imports model.py, eval_real_dev.py, label_free_null.py and train_cpu.py unchanged.
Design: MODEL_DESIGN_REVIEW_2026-11-18.md section 4.1.

Label-shift correction. The Bernoulli head has logit z = logit P(other > 0) under the
training prior pi0 = 0.55 (train_cpu.MISSING_PROBS: 45% of tasks have no missing type).
For an assumed rate pi: p(pi) = sigmoid(z + logit(pi) - logit(pi0)). The Beta size head is
left at its training prior. Posterior mean of other:  o(pi) = p(pi) * BetaMean.
Retained fractions:  kept(pi) = Dirichlet-mean composition (sums to 1) * (1 - o(pi)),
which is exactly how model.FractionPrediction.mean combines them, so kept + o sums to 1
and pi = 0.55 reproduces the model's own mean.

Commands (from deconv/, `python prior_sweep.py <cmd>`):
  wholeblood   four unlabelled whole-blood cohorts (no labels)
  dev          GSE110554 complete and drop-one mixtures (USES LABELS: another look at them)
  synth        synthetic validation sanity check of the correction (no real labels)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import label_free_null as lf
from label_free_null import ev, ds, reference
from model import Deconvolver
from rpc_torch import rpc_fit

OUT = Path("results/prior_sweep_v6_nocohort_2026-11-18")
CKPT = Path("results/gpu_v6_nocohort_seed0")
PI0 = 0.55
PIS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
CHUNK = 64


def logit(p):
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_model():
    s = json.loads((CKPT / "summary.json").read_text())["settings"]
    m = Deconvolver(width=s["width"], heads=s["heads"], blocks=s["blocks"], cohort=s.get("cohort", False))
    m.load_state_dict(torch.load(CKPT / "model.pt", map_location="cpu"))
    return m.eval()


def take(fit, rows):
    n = fit.fractions.shape[0]
    return type(fit)(**{k: (v[rows] if torch.is_tensor(v) and v.shape[:1] == (n,) else v) for k, v in vars(fit).items()})


def extract(model, R, S, B, fit):
    """Per sample: Dirichlet-mean composition, Bernoulli logit, P(other>0), Beta mean, model mean."""
    n = len(B)
    comp, z, p, bm, mean = [], [], [], [], []
    with torch.inference_mode():
        torch.manual_seed(0)
        for i in range(0, n, CHUNK):
            r = torch.arange(i, min(i + CHUNK, n))
            q = model(R[r], S[r], B[r], take(fit, r), None)
            comp.append(q.retained.mean.numpy()); z.append(q.has_other.logits.numpy())
            p.append(q.has_other.probs.numpy()); bm.append(q.positive_other.mean.numpy()); mean.append(q.mean.numpy())
    c = np.concatenate
    return dict(comp=c(comp).astype(float), logit=c(z).astype(float), p=c(p).astype(float),
                beta_mean=c(bm).astype(float), mean=c(mean).astype(float))


def shifted(e, pi):
    """kept (n x K) and other (n) at assumed missing rate pi."""
    p = sigmoid(e["logit"] + logit(pi) - logit(PI0))
    other = p * e["beta_mean"]
    return e["comp"] * (1 - other)[:, None], other, p


def tensors(mean, sd, bulk):
    n = bulk.shape[1]
    as_t = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
    Rn = np.repeat(mean.to_numpy(float)[None], n, axis=0)
    R = as_t(Rn)
    S = as_t(np.repeat(sd.to_numpy(float)[None], n, axis=0))
    Bn = bulk.to_numpy(float).T
    return R, S, as_t(Bn), rpc_fit(Rn, Bn)


def wholeblood():
    model = load_model()
    src = ds.load("GSE110554").sorted
    old = pd.read_csv("results/real_check_gpu_v6/wholeblood_estimates.csv")
    old = old[old.config == "v6_nocohort"]
    per_sample, rows = [], []
    for name, loader in lf.LOADERS.items():
        print(f"[{name}] loading", flush=True)
        df, _, _ = loader()
        shared = reference.usable_probes(src.betas).intersection(pd.Index(df.index))
        sub = df.loc[shared]
        sub = sub[sub.notna().all(axis=1)].astype(np.float64)
        mean, sd = ev.panel(src, SimpleNamespace(betas=sub), lf.PER_SIDE, None)
        bulk = sub.loc[mean.index]
        R, S, B, fit = tensors(mean, sd, bulk)
        e = extract(model, R, S, B, fit)
        o_model = e["mean"][:, -1]
        ref = old[old.cohort == name].set_index("sample_id")["other"].reindex(bulk.columns).to_numpy()
        print(f"[{name}] n={len(B)} markers={len(mean)} max|other - saved null_v6 other| = {np.abs(o_model - ref).max():.2e}", flush=True)
        rec = pd.DataFrame(dict(cohort=name, sample_id=bulk.columns, logit_p_other_gt0=e["logit"], p_other_gt0=e["p"],
                                beta_mean_other=e["beta_mean"], other_model_pi055=o_model))
        for j, t in enumerate(mean.columns):
            rec[f"comp_{t}"] = e["comp"][:, j]
        per_sample.append(rec)
        for pi in PIS:
            _, o, p = shifted(e, pi)
            rows.append(dict(cohort=name, pi=pi, n=len(o), other_mean_pts=100 * o.mean(), other_median_pts=100 * np.median(o),
                             frac_other_gt5=float((o > lf.THRESH).mean()), mean_p_other_gt0=float(p.mean()),
                             max_abs_diff_vs_saved_at_055=float(np.abs(o_model - ref).max()) if pi == PI0 else np.nan))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(per_sample).to_csv(OUT / "wholeblood_per_sample.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "wholeblood_sweep.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)


def dev():
    """GSE110554 labels are used here: scoring only, as in eval_real_dev.py (another look)."""
    model = load_model()
    data = ds.load("GSE110554")
    src, target = data.sorted, data.mixtures
    truth = target.truth
    types = list(pd.unique(src.cell_type))
    recs, ps = [], []
    for nps in [25, 12, 50]:
        for drop in [None] + types:
            mean, sd = ev.panel(src, target, nps, drop)
            kept = list(mean.columns)
            bulk = target.betas.loc[mean.index]
            R, S, B, fit = tensors(mean, sd, bulk)
            e = extract(model, R, S, B, fit)
            t_kept = truth.loc[bulk.columns, kept].to_numpy(float)
            t_other = truth.loc[bulk.columns, drop].to_numpy(float) if drop else np.zeros(len(bulk.columns))
            ps.append(pd.DataFrame(dict(per_side=nps, dropped=drop or "none", sample=bulk.columns, n_markers=len(mean),
                                        logit_p_other_gt0=e["logit"], p_other_gt0=e["p"], beta_mean_other=e["beta_mean"],
                                        other_true=t_other)))
            for pi in PIS:
                k, o, _ = shifted(e, pi)
                for i, s in enumerate(bulk.columns):
                    recs.append(dict(per_side=nps, pi=pi, dropped=drop or "none", method="model", sample=s, n_markers=len(mean),
                                     sq_err_sum=float(((k[i] - t_kept[i]) ** 2).sum()), n_cells=len(kept),
                                     other_est=float(o[i]), other_true=float(t_other[i])))
            rpc = fit.fractions.numpy()
            for i, s in enumerate(bulk.columns):
                recs.append(dict(per_side=nps, pi=np.nan, dropped=drop or "none", method="RPC", sample=s, n_markers=len(mean),
                                 sq_err_sum=float(((rpc[i] - t_kept[i]) ** 2).sum()), n_cells=len(kept), other_est=0.0,
                                 other_true=float(t_other[i])))
            print(f"per_side {nps} drop {drop or 'none'}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(ps).to_csv(OUT / "gse110554_per_sample_outputs.csv", index=False)
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "gse110554_estimates_long.csv", index=False)
    summarize_dev(df)


def auroc(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def stats(m):
    return dict(n_rows=len(m), kept_rmse_pts=100 * float(np.sqrt(m.sq_err_sum.sum() / m.n_cells.sum())),
                other_mae_pts=100 * float((m.other_est - m.other_true).abs().mean()),
                other_bias_pts=100 * float((m.other_est - m.other_true).mean()),
                other_est_mean_pts=100 * float(m.other_est.mean()))


def summarize_dev(df=None):
    if df is None:
        df = pd.read_csv(OUT / "gse110554_estimates_long.csv")
    rows = []
    for (nps, pi), g in df[df.method == "model"].groupby(["per_side", "pi"]):
        clean = g[g.dropped == "none"]
        drops = g[g.dropped != "none"]
        subsets = {"complete": clean, "one_dropped_all": drops,
                   "one_dropped_abundant(other_true>=5)": drops[drops.other_true >= 0.05],
                   "one_dropped_no_neutrophil": drops[drops.dropped != "Neu"],
                   "one_dropped_no_neutrophil_abundant": drops[(drops.dropped != "Neu") & (drops.other_true >= 0.05)]}
        for name, m in subsets.items():
            rows.append(dict(per_side=nps, pi=pi, regime=name, **stats(m)))
        ab = drops[drops.other_true >= 0.05]
        rows.append(dict(per_side=nps, pi=pi, regime="auroc_clean_vs_abundant", auroc=auroc(clean.other_est, ab.other_est)))
        rows.append(dict(per_side=nps, pi=pi, regime="auroc_clean_vs_abundant_no_neutrophil",
                         auroc=auroc(clean.other_est, ab[ab.dropped != "Neu"].other_est)))
        for d, m in drops.groupby("dropped"):
            rows.append(dict(per_side=nps, pi=pi, regime=f"per_drop:{d}", **stats(m)))
    for nps, g in df[df.method == "RPC"].groupby("per_side"):
        clean, drops = g[g.dropped == "none"], g[g.dropped != "none"]
        for name, m in {"complete": clean, "one_dropped_all": drops, "one_dropped_no_neutrophil": drops[drops.dropped != "Neu"]}.items():
            rows.append(dict(per_side=nps, pi=np.nan, regime="RPC:" + name, **stats(m)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "gse110554_sweep.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[out.per_side == 25].round(2).to_string(), flush=True)


def synth():
    """Check the correction on synthetic validation tasks with a known missing-type rate.

    Pool the checkpoint's validation regimes (same 3 validation sources, held-out donors, seed 0, the
    simulator's `tech_shift` on). Complete tasks: clean12, mismatch12, mismatch6. Missing tasks: the
    missing1_* regimes. Then reweight complete/missing tasks to a TEST rate pi_test (importance weights)
    and compare the corrected P(other>0) averaged over the reweighted tasks with pi_test, for assumed
    pi = pi_test and for the uncorrected 0.55 model.
    """
    import train_cpu as tc
    import simulator
    s = json.loads((CKPT / "summary.json").read_text())["settings"]
    model = load_model()
    t0 = __import__("time").time()
    bundle = simulator.build_v6_panels(seed=s["seed"], min_range=tc.POOL_MIN_RANGE)
    print(f"panels built in {time_str(t0)}", flush=True)
    for p in (*bundle["train"].values(), *bundle["validation"].values()):
        p.cube()
    val = tc.validation_set(bundle["validation"], bundle["pools"], s["seed"], s["val_per_regime"], s["val_markers"])
    recs = []
    for name, batch in val.items():
        truth = batch["truth"].numpy().astype(float)
        fit = batch["fit"]
        n = len(truth)
        e = extract(model, batch["reference"], batch["sd"], batch["bulk"], fit)
        for pi in PIS:
            k, o, p = shifted(e, pi)
            recs.append(pd.DataFrame(dict(regime=name, pi=pi, p=p, other_est=o, other_true=truth[:, -1],
                                          missing=(truth[:, -1] > 0).astype(int),
                                          sq_err_sum=((k - truth[:, :-1]) ** 2).sum(1), n_cells=truth.shape[1] - 1)))
        print(f"{name}: {n} tasks, missing share {np.mean(truth[:, -1] > 0):.2f}", flush=True)
    df = pd.concat(recs)
    df.to_csv(OUT / "synthetic_per_task.csv", index=False)
    rows = []
    for pi_test in [0.10, 0.20, 0.30, 0.55]:
        for pi in PIS:
            g = df[df.pi == pi]
            w = np.where(g.missing == 1, pi_test / g.missing.mean(), (1 - pi_test) / (1 - g.missing.mean()))
            wsum = w.sum()
            rows.append(dict(pi_test=pi_test, pi_assumed=pi, mean_corrected_p=float((w * g.p).sum() / wsum),
                             mean_other_est_pts=100 * float((w * g.other_est).sum() / wsum),
                             mean_other_true_pts=100 * float((w * g.other_true).sum() / wsum),
                             other_mae_pts=100 * float((w * (g.other_est - g.other_true).abs()).sum() / wsum),
                             kept_rmse_pts=100 * float(np.sqrt((w * g.sq_err_sum).sum() / (w * g.n_cells).sum()))))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "synthetic_check.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(), flush=True)


def time_str(t0):
    return f"{__import__('time').time() - t0:.0f}s"


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    {"wholeblood": wholeblood, "dev": dev, "synth": synth, "summarize_dev": summarize_dev}[sys.argv[1]]()
