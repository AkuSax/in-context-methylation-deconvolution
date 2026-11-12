"""Accuracy summaries for estimated versus true cell fractions.

Errors are reported in percentage points of the whole sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_type(est: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """RMSE, bias (both in points) and Pearson r for each shared cell type."""
    cols = [c for c in truth.columns if c in est.columns]
    rows = []
    for c in cols:
        e, t = est[c].to_numpy(float), truth[c].to_numpy(float)
        ok = ~np.isnan(t)
        e, t = e[ok], t[ok]
        r = np.corrcoef(e, t)[0, 1] if len(t) > 2 and t.std() > 0 and e.std() > 0 else np.nan
        rows.append({"type": c, "n": int(ok.sum()), "rmse_pts": 100 * np.sqrt(np.mean((e - t) ** 2)),
                     "bias_pts": 100 * np.mean(e - t), "pearson_r": r, "true_mean_pts": 100 * t.mean()})
    return pd.DataFrame(rows)


def overall(est: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    """Pooled RMSE over all sample-type pairs, mean per-type RMSE, worst type."""
    pt = per_type(est, truth)
    cols = list(pt["type"])
    d = (est[cols] - truth[cols]).to_numpy(float)
    worst = pt.loc[pt["rmse_pts"].idxmax()]
    return {
        "rmse_pts": 100 * float(np.sqrt(np.nanmean(d ** 2))),
        "mean_type_rmse_pts": float(pt["rmse_pts"].mean()),
        "worst_type": str(worst["type"]),
        "worst_type_rmse_pts": float(worst["rmse_pts"]),
        "mean_pearson_r": float(pt["pearson_r"].mean()),
        "n_types": len(cols),
        "n_samples": int(len(truth)),
    }


def strict_bootstrap(rows: pd.DataFrame, n_boot: int = 10000, seed: int = 20261111) -> pd.DataFrame:
    """Mixture-row percentile intervals; pair methods and repeated drops by sample.

    RMSE is pooled over retained types and drops. Differences are method minus RPC.
    The same mixture indices are used for all drops and methods within each replicate.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    samples = sorted(rows["sample"].unique())
    drops = sorted(rows["dropped"].unique())
    methods = sorted(rows["method"].unique())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(samples), size=(n_boot, len(samples)))
    full_index = pd.MultiIndex.from_product([drops, samples], names=["dropped", "sample"])
    arrays = {}
    for method in methods:
        r = rows[rows.method == method].set_index(["dropped", "sample"]).reindex(full_index)
        if r[["kept_squared_error_mean", "true_omitted", "estimated_omitted"]].isna().any().any():
            raise ValueError("Incomplete or duplicate sample/drop/method results")
        arrays[method] = {c: r[c].to_numpy().reshape(len(drops), len(samples))
                          for c in ["kept_squared_error_mean", "true_omitted", "estimated_omitted"]}
    output = []
    selections = [(d, [i], False) for i, d in enumerate(drops)]
    selections += [("ALL_DROPS", list(range(len(drops))), False),
                   ("ABUNDANT_DROPS_GE_5_PTS", list(range(len(drops))), True)]
    for selection, indices, abundant in selections:
        for method in methods:
            a = arrays[method]
            mask = a["true_omitted"][indices] >= 0.05 if abundant else np.ones_like(a["true_omitted"][indices], dtype=bool)
            mse = np.where(mask, a["kept_squared_error_mean"][indices], np.nan)
            omitted_error = np.where(mask, (a["estimated_omitted"] - a["true_omitted"])[indices], np.nan)
            if not mask.any():
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                boot_rmse = 100 * np.sqrt(np.nanmean(mse[:, draws], axis=(0, 2)))
                boot_other = 100 * np.sqrt(np.nanmean(omitted_error[:, draws] ** 2, axis=(0, 2)))
                boot_mae = 100 * np.nanmean(np.abs(omitted_error[:, draws]), axis=(0, 2))
            metrics = {"rmse_kept_pts": (100 * np.sqrt(np.nanmean(mse)), boot_rmse),
                       "omitted_rmse_pts": (100 * np.sqrt(np.nanmean(omitted_error ** 2)), boot_other),
                       "omitted_mae_pts": (100 * np.nanmean(np.abs(omitted_error)), boot_mae)}
            if "EpiDISH_RPC" in arrays:
                rpc_mse = np.where(mask, arrays["EpiDISH_RPC"]["kept_squared_error_mean"][indices], np.nan)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rpc_boot = 100 * np.sqrt(np.nanmean(rpc_mse[:, draws], axis=(0, 2)))
                metrics["rmse_kept_difference_vs_rpc_pts"] = (100 * (np.sqrt(np.nanmean(mse)) - np.sqrt(np.nanmean(rpc_mse))), boot_rmse - rpc_boot)
            for metric, (value, boot) in metrics.items():
                lo, hi = np.nanquantile(boot, [0.025, 0.975])
                output.append({"dropped": selection, "method": method, "metric": metric,
                               "estimate_pts": value, "ci_low_pts": lo, "ci_high_pts": hi,
                               "resampling_unit": "mixture row; independence unresolved", "n_mixtures": len(samples),
                               "n_drops": len(indices), "n_sample_drops": int(mask.sum()),
                               "n_mixtures_contributing": int(mask.any(axis=0).sum()),
                               "n_boot": n_boot, "seed": seed})
    return pd.DataFrame(output)
