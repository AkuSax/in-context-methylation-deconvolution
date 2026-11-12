"""Re-read a development check's saved per-sample estimates (no model run).

Post hoc audit of `results/dev_gse110554_*/estimates_long.csv`. It asks three
things the summary tables do not show:
- how much of the pooled drop-one result is the neutrophil drop;
- whether the model's `other` tracks the true omitted fraction within a drop,
  and whether it beats a constant guess;
- how detection depends on the size of the omitted fraction.

Uses truth labels. Run from deconv/:
    python audit_dev_estimates.py results/dev_gse110554_gpu_v5/estimates_long.csv
"""
import sys

import numpy as np
import pandas as pd

PRIOR_OTHER_PTS = 9.06  # training-prior mean of `other` for gpu_v5_seed0, in points

df = pd.read_csv(sys.argv[1])
pd.set_option("display.width", 200)


def rmse(m):
    return 100 * np.sqrt(m.sq_err_sum.sum() / m.n_cells.sum())


for nps, g in df.groupby("per_side"):
    print(f"\n=== per_side {nps} ===")
    d = g[g.dropped != "none"]
    for label, sub in [("all drops", d), ("without Neu", d[d.dropped != "Neu"]), ("Neu only", d[d.dropped == "Neu"])]:
        r = {m: rmse(x) for m, x in sub.groupby("method")}
        print(f"{label:12s} kept RMSE  model {r['model']:.2f}  RPC {r['RPC']:.2f}  shrunk {r['RPC_shrunk_const']:.2f}"
              f"  change vs RPC {100 * (r['model'] / r['RPC'] - 1):+.0f}%")
    # Paired bootstrap over mixture rows, without the neutrophil drop.
    rng = np.random.default_rng(20261111)
    samples = sorted(df["sample"].unique())
    idx = rng.integers(0, len(samples), size=(10000, len(samples)))
    sub = d[d.dropped != "Neu"]

    def agg(method):
        t = sub[sub.method == method].groupby("sample")[["sq_err_sum", "n_cells"]].sum().reindex(samples)
        return t.sq_err_sum.to_numpy(), t.n_cells.to_numpy()

    ms, mc = agg("model")
    rs, rc = agg("RPC")
    bm = 100 * np.sqrt(ms[idx].sum(1) / mc[idx].sum(1))
    br = 100 * np.sqrt(rs[idx].sum(1) / rc[idx].sum(1))
    print(f"  without Neu, boot model-RPC: {np.percentile(bm - br, 2.5):+.2f} to {np.percentile(bm - br, 97.5):+.2f};"
          f" ratio {np.percentile(bm / br, 2.5):.2f} to {np.percentile(bm / br, 97.5):.2f}")

    m = d[d.method == "model"]
    clean = g[(g.dropped == "none") & (g.method == "model")]
    print(f"clean other: mean {100 * clean.other_est.mean():.1f} min {100 * clean.other_est.min():.1f}"
          f" max {100 * clean.other_est.max():.1f}")
    # `other` estimate against truth, with constant-guess baselines.
    for label, sub in [("all drops", m), ("without Neu", m[m.dropped != "Neu"])]:
        e, t = 100 * sub.other_est, 100 * sub.other_true
        slope, icpt = np.polyfit(t, e, 1)
        print(f"other {label:12s} n {len(sub)} r {np.corrcoef(e, t)[0, 1]:.2f} slope {slope:.2f} intercept {icpt:.1f}"
              f" model MAE {(e - t).abs().mean():.2f}"
              f"  median-of-truth const MAE {(t - np.median(t)).abs().mean():.2f} (const {np.median(t):.1f})"
              f"  prior const ({PRIOR_OTHER_PTS}) MAE {(t - PRIOR_OTHER_PTS).abs().mean():.2f}  zero MAE {t.abs().mean():.2f}")
    # Within a drop: does `other` follow the omitted fraction across the 12 mixtures?
    rows = []
    for drop, sub in m.groupby("dropped"):
        e, t = 100 * sub.other_est, 100 * sub.other_true
        rows.append(dict(drop=drop, true_mean=t.mean(), true_min=t.min(), true_max=t.max(), est_mean=e.mean(),
                         est_min=e.min(), est_max=e.max(),
                         r_within=np.corrcoef(e, t)[0, 1] if t.std() > 0 else np.nan, mae=(e - t).abs().mean()))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    small = m[m.other_true < 0.05]
    print(f"drop rows with true other < 5 pts: n {len(small)}, est mean {100 * small.other_est.mean():.1f},"
          f" true mean {100 * small.other_true.mean():.1f}")
    neg = clean.other_est.to_numpy()
    pos = m[(m.other_true >= 0.05) & (m.dropped != "Neu")].other_est.to_numpy()
    au = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    print(f"AUROC clean vs abundant drops excluding Neu: {au:.2f} (n_pos {len(pos)})")
    for lo, hi in [(5, 10), (10, 20), (20, 100)]:
        s = m[(m.other_true * 100 >= lo) & (m.other_true * 100 < hi)]
        if len(s):
            p = s.other_est.to_numpy()
            print(f"  true other {lo}-{hi}: n {len(s)} est mean {100 * p.mean():.1f} true mean {100 * s.other_true.mean():.1f}"
                  f" AUROC vs clean {(p[:, None] > neg[None, :]).mean():.2f}")
