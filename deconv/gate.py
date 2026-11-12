"""Hybrid RPC/model gate (results/dev_gse110554_gate_v5/PROTOCOL.md).

Report RPC's fractions (and `other` = 0) unless the model's `other` is at or
above tau(K, M), the q-quantile of the model's `other` on simulated CLEAN tasks
with the panel's shape (K retained types, M markers). The threshold uses no
labels or bulk data from the target series, only the panel shape.

    py -3.11 -B gate.py --checkpoint results/gpu_v5_seed0 \
        --estimates results/dev_gse110554_gpu_v5/estimates_long.csv \
        --out results/dev_gse110554_gate_v5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import simulator
import train_cpu
from model import Deconvolver

QUANTILES = (0.80, 0.90, 0.95)
PRIMARY_Q = 0.90


def load_model(checkpoint: Path) -> Deconvolver:
    s = json.loads((checkpoint / "summary.json").read_text())["settings"]
    m = Deconvolver(width=s["width"], heads=s["heads"], blocks=s["blocks"])
    m.load_state_dict(torch.load(checkpoint / "model.pt", map_location="cpu"))
    m.eval()
    return m


def simulate(panel, pool, k, m, n, seed, missing):
    rng = np.random.default_rng(seed)
    return [simulator.simulate_task(panel, rng, n_markers=m, n_types=k, missing=missing,
                                    donor_mismatch=bool(rng.random() < 0.5), augment=True,
                                    marker_mode="pick", candidate_pool=pool, sharpen=train_cpu.SHARPEN,
                                    noise=0.02, tech_shift=True, **train_cpu.TECH_SHIFT_RANGES)
            for _ in range(n)]


def predict(model, tasks, chunk=64):
    """Return (model mean incl. other, RPC fractions, truth) as numpy arrays."""
    means, rpcs, truths = [], [], []
    for i in range(0, len(tasks), chunk):
        b = train_cpu.collate(tasks[i:i + chunk])
        with torch.inference_mode():
            means.append(model(b["reference"], b["sd"], b["bulk"], b["fit"]).mean.numpy().astype(float))
        rpcs.append(b["fit"].fractions.numpy().astype(float))
        truths.append(b["truth"].numpy().astype(float))
    return np.concatenate(means), np.concatenate(rpcs), np.concatenate(truths)


def gated(model_mean, rpc, tau):
    use_model = model_mean[:, -1] >= tau
    kept = np.where(use_model[:, None], model_mean[:, :-1], rpc)
    other = np.where(use_model, model_mean[:, -1], 0.0)
    return kept, other, use_model


def rmse(est, truth):
    return 100 * float(np.sqrt(((est - truth) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("results/gpu_v5_seed0"))
    ap.add_argument("--estimates", type=Path, default=Path("results/dev_gse110554_gpu_v5/estimates_long.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/dev_gse110554_gate_v5"))
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    torch.set_num_threads(8)
    model = load_model(args.checkpoint)
    full = simulator.load_gse35069()
    train_panel, val_panel = simulator.split_donors(full, seed=0)
    pool = simulator.screen_pool(train_panel, train_cpu.POOL_MIN_RANGE)

    est = pd.read_csv(args.estimates)
    shapes = sorted({(int(k), int(m)) for k, m in est[["n_cells", "n_markers"]].drop_duplicates().itertuples(index=False)})
    thresholds, synth = [], []
    for k, m in shapes:
        calib, _, _ = predict(model, simulate(val_panel, pool, k, m, args.n_calib, [20261111, k, m], 0))
        taus = {q: float(np.quantile(calib[:, -1], q)) for q in QUANTILES}
        thresholds.append(dict(n_cells=k, n_markers=m, **{f"tau_q{int(q*100)}": t for q, t in taus.items()},
                               calib_other_mean=float(calib[:, -1].mean())))
        for missing in (0, 1):
            mm, rr, tt = predict(model, simulate(val_panel, pool, k, m, args.n_test, [20261112, k, m, missing], missing))
            row = dict(n_cells=k, n_markers=m, regime="clean" if missing == 0 else "missing1",
                       model=rmse(mm[:, :-1], tt[:, :-1]), RPC=rmse(rr, tt[:, :-1]))
            for q, t in taus.items():
                kept, _, use = gated(mm, rr, t)
                row[f"gate_q{int(q*100)}"] = rmse(kept, tt[:, :-1])
                row[f"to_rpc_q{int(q*100)}"] = float(1 - use.mean())
            synth.append(row)
        print(f"K={k} M={m}: tau90={taus[0.9]:.4f}", flush=True)
    th = pd.DataFrame(thresholds)
    args.out.mkdir(parents=True, exist_ok=True)
    th.to_csv(args.out / "thresholds.csv", index=False)
    pd.DataFrame(synth).to_csv(args.out / "synthetic_check.csv", index=False)

    # Real data: gate rows of the frozen v5 estimates.
    est = est.merge(th, on=["n_cells", "n_markers"], how="left")
    key = ["per_side", "dropped", "sample"]
    mod = est[est.method == "model"].set_index(key)
    rpc = est[est.method == "RPC"].set_index(key)
    rows = [est[est.method.isin(["model", "RPC"])][key + ["method", "sq_err_sum", "n_cells", "other_est", "other_true"]]]
    for q in QUANTILES:
        tau = mod[f"tau_q{int(q*100)}"]
        use = mod.other_est >= tau
        g = pd.DataFrame(dict(sq_err_sum=np.where(use, mod.sq_err_sum, rpc.sq_err_sum.reindex(mod.index)),
                              n_cells=mod.n_cells, other_est=np.where(use, mod.other_est, 0.0),
                              other_true=mod.other_true, used_model=use.astype(int),
                              covered=mod.covered.where(use), n_int=mod.n_int.where(use)), index=mod.index).reset_index()
        g["method"] = f"gate_q{int(q*100)}"
        rows.append(g)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(args.out / "estimates_long.csv", index=False)

    rng = np.random.default_rng(20261111)
    samples = sorted(df["sample"].unique())
    boot_idx = rng.integers(0, len(samples), size=(args.boot, len(samples)))
    out = []
    for nps, g in df.groupby("per_side"):
        for regime, sub in [("clean", g[g.dropped == "none"]), ("drop_one", g[g.dropped != "none"])]:
            for subset, s2 in [("all", sub), ("abundant", sub[sub.other_true >= 0.05] if regime == "drop_one" else None)]:
                if s2 is None or s2.empty:
                    continue
                per = {}
                for method, mth in s2.groupby("method"):
                    row = dict(per_side=nps, regime=regime, subset=subset, method=method, n_rows=len(mth),
                               kept_rmse_pts=100 * float(np.sqrt(mth.sq_err_sum.sum() / mth.n_cells.sum())),
                               other_mae_pts=100 * float((mth.other_est - mth.other_true).abs().mean()),
                               other_bias_pts=100 * float((mth.other_est - mth.other_true).mean()))
                    if method.startswith("gate"):
                        row["to_rpc_frac"] = float(1 - mth.used_model.mean())
                        if mth.n_int.sum() > 0:  # interval coverage on rows left with the model
                            row["coverage_model_rows"] = float(mth.covered.sum() / mth.n_int.sum())
                    out.append(row)
                    per[method] = mth.groupby("sample")[["sq_err_sum", "n_cells"]].sum().reindex(samples).fillna(0)
                def boot(t):
                    return 100 * np.sqrt(t.sq_err_sum.to_numpy()[boot_idx].sum(1) / t.n_cells.to_numpy()[boot_idx].sum(1))
                for method in [m for m in per if m.startswith("gate")]:
                    d = boot(per[method]) - boot(per["RPC"])
                    out.append(dict(per_side=nps, regime=regime, subset=subset, method=f"{method}_minus_RPC_boot",
                                    diff_lo_pts=float(np.percentile(d, 2.5)), diff_hi_pts=float(np.percentile(d, 97.5))))
        for drop, d in g[g.dropped != "none"].groupby("dropped"):
            for method, mth in d.groupby("method"):
                out.append(dict(per_side=nps, regime="per_drop", subset=drop, method=method,
                                kept_rmse_pts=100 * float(np.sqrt(mth.sq_err_sum.sum() / mth.n_cells.sum())),
                                other_mae_pts=100 * float((mth.other_est - mth.other_true).abs().mean())))
    res = pd.DataFrame(out)
    res.to_csv(args.out / "summary.csv", index=False)
    print(res.to_string())


if __name__ == "__main__":
    main()
