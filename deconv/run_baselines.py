"""Classical-solver baselines on every blood dataset with known fractions.

Three reference panels are used:
  ref6   GSE110554 purified samples, six types, markers picked here
  ref12  GSE167998 purified samples, twelve types, markers picked here
  dhs7   EpiDISH's shipped centDHSbloodDMC panel (450K-era, seven types)

Markers are chosen among probes observed in both the reference and the target,
so each reference/target pair gets its own marker set. Estimates and truth are
summed to the finest level they share before scoring.

Outputs (in --out): summary.csv, per_type.csv, estimates.csv, leave_one_out.csv.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import datasets as ds
import metrics
import reference
import solvers

N_PER_SIDE = 50
JOBS = 1  # set by --jobs; panels are fitted in separate processes when > 1


def _fit(job):
    ref, bulk, kw = job
    return solvers.run_all(ref, bulk, **kw), dict(solvers.PAPER_ERRORS)


def fit_many(panels, **kw):
    """run_all on each (ref, bulk) pair, in parallel worker processes when JOBS > 1.
    Results are identical to serial runs; only marker selection stays in this process."""
    jobs = [(ref, bulk, kw) for ref, bulk in panels]
    if JOBS <= 1 or len(jobs) <= 1:
        results = [_fit(j) for j in jobs]
    else:
        with ProcessPoolExecutor(min(JOBS, len(jobs))) as pool:
            results = list(pool.map(_fit, jobs))
    for _, errors in results:
        if errors:
            print("  paper solver errors:", errors, flush=True)
    return [fits for fits, _ in results]


def dhs7_reference() -> pd.DataFrame:
    path = ds.CACHE / "centDHSbloodDMC.csv"
    if not path.exists():
        ds.CACHE.mkdir(parents=True, exist_ok=True)
        code = f'suppressPackageStartupMessages(library(EpiDISH)); data(centDHSbloodDMC.m); write.csv(centDHSbloodDMC.m, "{path.as_posix()}")'
        subprocess.run([solvers.RSCRIPT, "--vanilla", "-e", code], check=True, capture_output=True)
    ref = pd.read_csv(path, index_col=0)
    return ref.rename(columns={"Neutro": "Neu", "Eosino": "Eos"})


def shared_level(est: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum both tables to the finest labels they have in common."""
    if set(truth.columns) <= set(est.columns):
        return est[list(truth.columns)], truth
    # Fall back to the coarse six types; anything outside them (nRBC) stays as is.
    e, t = ds.collapse(est, ds.TO_COARSE), ds.collapse(truth, ds.TO_COARSE)
    cols = [c for c in t.columns if c in e.columns]
    return e[cols], t[cols]


def uncovered(truth: pd.DataFrame, ref_types) -> list[str]:
    """Truth types present in the samples that the reference cannot represent:
    not in the reference itself, nor as its parent, nor as one of its subtypes."""
    ref_types = set(ref_types)
    parents = {ds.TO_COARSE.get(c) for c in ref_types}
    return sorted(c for c in truth.columns
                  if truth[c].max() > 0 and c not in ref_types
                  and ds.TO_COARSE.get(c) not in ref_types and c not in parents)


def panel_for(ref_name: str, refs: dict, target: ds.Dataset):
    """Reference means restricted to markers usable on this target."""
    if ref_name == "dhs7":
        full = refs["dhs7"]
        ok = reference.usable_probes(target.betas.loc[target.betas.index.intersection(full.index)])
        return full.loc[full.index.intersection(ok)], len(full)
    src = refs[ref_name]
    probes = reference.usable_probes(src.betas, target.betas)
    markers = reference.pick_markers(src.betas.loc[probes], src.cell_type, N_PER_SIDE)
    mean, _ = reference.build_reference(src.betas, src.cell_type, markers)
    return mean, len(markers)


def strict_panel(src: ds.Dataset, target: ds.Dataset, drop: str):
    """Remove all dropped-type samples before probe filtering and marker selection."""
    keep = src.cell_type.ne(drop)
    if not (~keep).any():
        raise ValueError(f"Unknown dropped type: {drop}")
    kept = src.subset(keep)
    probes = reference.usable_probes(kept.betas, target.betas)
    markers = reference.pick_markers(kept.betas.loc[probes], kept.cell_type, N_PER_SIDE)
    mean, _ = reference.build_reference(kept.betas, kept.cell_type, markers)
    return mean, len(probes), kept


def strict_missing_type(out: Path, with_epidish: bool, n_boot: int = 10000, with_paper: bool = True) -> None:
    """Score each missing type and save the exact panel used by every solver."""
    src = ds.load("GSE167998").sorted
    target = ds.load("GSE182379").mixtures
    truth = target.truth
    rows, marker_rows, estimates, panels = [], [], [], []
    drops = list(pd.unique(src.cell_type))
    built = [strict_panel(src, target, drop) for drop in drops]
    all_fits = fit_many([(ref, target.betas.loc[ref.index]) for ref, _, _ in built],
                        with_epidish=with_epidish, with_paper=with_paper)
    for drop, (ref, n_usable, kept_src), fits in zip(drops, built, all_fits):
        marker_rows.extend({"dropped": drop, "probe": p, "order": i} for i, p in enumerate(ref.index))
        panels.append({"dropped": drop, "n_usable": n_usable, "n_markers": len(ref),
                       "retained_types": list(ref.columns), "reference_samples": list(kept_src.betas.columns)})
        for method, est in fits.items():
            kept = list(ref.columns)
            error = est[kept].to_numpy() - truth.loc[est.index, kept].to_numpy()
            other = 1.0 - est.sum(axis=1).to_numpy()
            t_other = truth.loc[est.index, drop].to_numpy()
            rows.append({"dropped": drop, "method": method, "n_markers": len(ref),
                         "n_usable": n_usable, "n_samples": len(est),
                         "rmse_kept_pts": 100 * np.sqrt(np.mean(error ** 2)),
                         "true_omitted_mean_pts": 100 * np.mean(t_other),
                         "estimated_omitted_mean_pts": 100 * np.mean(other),
                         "omitted_rmse_pts": 100 * np.sqrt(np.mean((other - t_other) ** 2)),
                         "omitted_mae_pts": 100 * np.mean(np.abs(other - t_other)),
                         "omitted_bias_pts": 100 * np.mean(other - t_other)})
            for i, sample in enumerate(est.index):
                estimates.append({"dropped": drop, "method": method, "sample": sample,
                                  "kept_squared_error_mean": float(np.mean(error[i] ** 2)),
                                  "true_omitted": t_other[i], "estimated_omitted": other[i],
                                  **{f"est_{c}": est.loc[sample, c] for c in kept},
                                  **{f"true_{c}": truth.loc[sample, c] for c in kept}})
        print(f"done strict leave-out {drop}: {len(ref)} markers", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "leave_one_out.csv", index=False)
    pd.DataFrame(marker_rows).to_csv(out / "markers.csv", index=False)
    long = pd.DataFrame(estimates)
    long.to_csv(out / "estimates.csv", index=False)
    bootstrap = metrics.strict_bootstrap(long, n_boot=n_boot)
    bootstrap.to_csv(out / "bootstrap.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({
        "preprocessing": ds.PREPROC, "reference": "GSE167998 sorted", "target": "GSE182379 mixtures",
        "marker_selection": "Welch one-vs-rest; 50 per side per retained type; dropped samples excluded before probe filtering",
        "omitted_estimate": "1 - sum of retained estimates; signed and not clipped; not a fitted other component",
        "bootstrap": "percentile 95%; paired resampling of 12 mixture rows; repeated drops resampled jointly; mixture independence unresolved",
        "bootstrap_seed": 20261111, "bootstrap_replicates": n_boot,
        "same_cpg_rule": "Use markers.csv for all strict comparators and exclude dropped reference samples.",
        "panels": panels}, indent=2))
    print("wrote", out, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc", choices=["geo", "noob"], default=ds.PREPROC,
                    help="betas as deposited on GEO, or rebuilt from IDATs by reprocess.py")
    ap.add_argument("--out", help="default: results/baselines for geo, results/baselines_noob for noob")
    ap.add_argument("--no-epidish", action="store_true")
    ap.add_argument("--no-paper", action="store_true",
                    help="skip the De Ridder et al. 2024 solvers (EMeth, ICeDT, FARDEEP, LTS)")
    ap.add_argument("--strict-only", action="store_true", help="run only the strict missing-type benchmark")
    ap.add_argument("--bootstrap-replicates", type=int, default=10000)
    ap.add_argument("--jobs", type=int, default=1, help="worker processes for solver fits")
    args = ap.parse_args()
    global JOBS
    JOBS = args.jobs
    ds.PREPROC = args.preproc
    if args.strict_only:
        out = Path(args.out or Path(__file__).with_name("results") / f"strict_missing_{args.preproc}")
        strict_missing_type(out, not args.no_epidish, args.bootstrap_replicates, not args.no_paper)
        return
    suffix = "" if args.preproc == "geo" else f"_{args.preproc}"
    out = Path(args.out or Path(__file__).with_name("results") / f"baselines{suffix}")
    out.mkdir(parents=True, exist_ok=True)

    data = {n: ds.load(n) for n in ds.LOADERS}
    refs = {"ref6": data["GSE110554"].sorted, "ref12": data["GSE167998"].sorted, "dhs7": dhs7_reference()}
    ref_source = {"ref6": "GSE110554", "ref12": "GSE167998", "dhs7": "EpiDISH"}

    kw = {"with_epidish": not args.no_epidish, "with_paper": not args.no_paper}
    pairs = [(ref_name, tname, full) for ref_name in refs for tname, full in data.items()
             if full.mixtures.betas.shape[1] > 0]
    built = [panel_for(ref_name, refs, full.mixtures) for ref_name, _, full in pairs]
    all_fits = fit_many([(ref, full.mixtures.betas.loc[ref.index]) for (_, _, full), (ref, _) in zip(pairs, built)], **kw)

    summary, per_type, estimates = [], [], []
    for (ref_name, tname, full), (ref, n_requested), est_all in zip(pairs, built, all_fits):
        mix = full.mixtures
        for kind in pd.unique(mix.truth_kind):
            ids = mix.truth_kind.index[mix.truth_kind == kind]
            truth = mix.truth.loc[ids].dropna(axis=1, how="all")
            for method, est in est_all.items():
                e, t = shared_level(est.loc[ids], truth)
                tag = {"ref": ref_name, "ref_source": ref_source[ref_name], "target": tname,
                       "platform": full.platform, "truth": kind, "method": method,
                       "same_series": ref_source[ref_name] == tname,
                       "n_markers": len(ref), "n_markers_requested": n_requested,
                       "types_missing_from_ref": ",".join(uncovered(truth, ref.columns)),
                       "mean_unassigned_pts": 100 * float((1 - est.loc[ids].sum(axis=1)).mean())}
                summary.append({**tag, **metrics.overall(e, t)})
                per_type.append(metrics.per_type(e, t).assign(**{k: tag[k] for k in ("ref", "target", "truth", "method")}))
                long = e.rename_axis("sample").reset_index().melt("sample", var_name="type", value_name="est")
                long["true"] = t.rename_axis("sample").reset_index().melt("sample")["value"].to_numpy()
                estimates.append(long.assign(**{k: tag[k] for k in ("ref", "target", "truth", "method")}))
        print(f"done {ref_name} -> {tname} ({len(ref)} markers)", flush=True)

    # Leave one cell type out of the twelve-type reference, scored on GSE182379.
    target = data["GSE182379"].mixtures
    ref_full, _ = panel_for("ref12", refs, target)
    bulk = target.betas.loc[ref_full.index]
    truth = target.truth
    loo_types = list(ref_full.columns)
    loo_fits = fit_many([(ref_full, bulk)] + [(ref_full[[c for c in loo_types if c != d]], bulk) for d in loo_types], **kw)
    full_est = loo_fits[0]
    loo = []
    for drop, est_drop in zip(loo_types, loo_fits[1:]):
        kept = [c for c in ref_full.columns if c != drop]
        for method, est in est_drop.items():
            err_drop = (est[kept] - truth[kept]).to_numpy()
            err_full = (full_est[method][kept] - truth[kept]).to_numpy()
            worst = np.sqrt((err_drop ** 2).mean(axis=0))
            loo.append({
                "dropped": drop, "method": method,
                "true_dropped_mean_pts": 100 * truth[drop].mean(),
                "rmse_kept_full_ref_pts": 100 * np.sqrt((err_full ** 2).mean()),
                "rmse_kept_dropped_ref_pts": 100 * np.sqrt((err_drop ** 2).mean()),
                "most_inflated_type": kept[int(np.argmax(err_drop.mean(axis=0)))],
                "most_inflated_bias_pts": 100 * float(err_drop.mean(axis=0).max()),
                "worst_kept_type": kept[int(np.argmax(worst))],
                "worst_kept_rmse_pts": 100 * float(worst.max()),
                "unassigned_mean_pts": 100 * float((1 - est.sum(axis=1)).mean()),
                "unassigned_vs_true_r": float(np.corrcoef(1 - est.sum(axis=1), truth[drop])[0, 1])
                if (1 - est.sum(axis=1)).std() > 1e-9 else np.nan,
            })
        print(f"done leave-out {drop}", flush=True)

    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.concat(per_type).to_csv(out / "per_type.csv", index=False)
    pd.concat(estimates).to_csv(out / "estimates.csv", index=False)
    pd.DataFrame(loo).to_csv(out / "leave_one_out.csv", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
