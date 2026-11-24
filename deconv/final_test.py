"""Locked final test: GSE167998 sorted reference -> GSE182379 mixtures (12 types).

Protocol: FINAL_TEST_PREDECLARATION_2026-11-19.md (committed before this script ran on the
reserved data). The v6 run on the reserved pair has been done once:

    python final_test.py --checkpoint results/gpu_v6_nocohort_seed0 --out results/final_test_gse167998_gse182379

IMPORTANT: the locked pair has already been scored once (for v6). A run of ANY other checkpoint (v7, a
retrained v6, a different selection) on GSE167998 -> GSE182379 is a second look at reserved data and needs
its OWN new pre-declaration, committed before the run, that states the v6 result is known. This script
only enforces the mechanics: --checkpoint and --out are both required for the non-dry run, --out must not
exist or must be empty, and the checkpoint SHA-256 is written to summary.json.

Mechanics are dry-run on GSE110554 only, which never touches the reserved pair (--checkpoint and --out are
optional there; the defaults reproduce the earlier v6 dry run):

    python final_test.py --dry-run [--checkpoint DIR --out DIR]

RSCRIPT = R 4.6.2, R_LIBS_USER = its library. The checkpoint must be a train_cpu.py output directory
(model.pt + summary.json) of a model WITHOUT cohort features (singleton inference; --no-cohort).

The script changes nothing in the model, markers or thresholds between the two modes. Markers
are 50 per side per retained type (run_baselines.N_PER_SIDE, as in the strict benchmark),
selected from reference labels only; a dropped type is removed before probe filtering and
marker selection. Model = results/gpu_v6_nocohort_seed0 (singleton inference). Primary prior
pi = 0.55 (as trained); pi = 0.10 is a declared secondary readout. Comparator = EpiDISH RPC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

os.environ["DECONV_PREPROC"] = "noob"

import numpy as np
import pandas as pd
import torch

import datasets as ds
import metrics  # noqa: F401  (imported so a broken environment fails before the run)
import prior_sweep as ps
import reference
import solvers
from model import Deconvolver
from run_baselines import N_PER_SIDE

SEED = 20261119
BOOT = 10_000
PIS = (0.55, 0.10)
COMPARATOR = "EpiDISH_RPC"
CLASSICAL = ("EpiDISH_RPC", "NNLS_norm", "CP_eq", "CP_ineq", "EpiDISH_CBS", "EpiDISH_CP")
ABUNDANT = 0.05


DRY_DEFAULT_CKPT = Path("results/gpu_v6_nocohort_seed0")
DRY_DEFAULT_OUT = Path("results/final_test_dryrun_gse110554")


def parse_args(argv=None):
    """Argument checks live here so they can be tested without loading any data."""
    ap = argparse.ArgumentParser(description="Locked final test (see module docstring)")
    ap.add_argument("--dry-run", action="store_true", help="GSE110554 only; never loads the reserved series")
    ap.add_argument("--checkpoint", type=Path, default=None, help="train_cpu.py output dir (model.pt + summary.json)")
    ap.add_argument("--out", type=Path, default=None, help="result directory; must not exist or be empty")
    args = ap.parse_args(argv)
    if not args.dry_run and (args.checkpoint is None or args.out is None):
        ap.error("the non-dry run requires both --checkpoint and --out (no defaults on the reserved pair)")
    if args.dry_run:
        args.checkpoint = args.checkpoint or DRY_DEFAULT_CKPT
        args.out = args.out or DRY_DEFAULT_OUT
    if not (args.checkpoint / "model.pt").is_file() or not (args.checkpoint / "summary.json").is_file():
        ap.error(f"--checkpoint {args.checkpoint} needs model.pt and summary.json")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        ap.error(f"--out {args.out} exists and is not empty; refusing to overwrite results")
    return args


def load_checkpoint(ckpt: Path):
    """Model from a train_cpu.py directory (same construction as prior_sweep.load_model, which is tied to CKPT)."""
    s = json.loads((ckpt / "summary.json").read_text())["settings"]
    if s.get("cohort", False):
        raise SystemExit(f"{ckpt} has cohort features on; this script scores singleton (no-cohort) models only")
    m = Deconvolver(width=s["width"], heads=s["heads"], blocks=s["blocks"], cohort=False)
    m.load_state_dict(torch.load(ckpt / "model.pt", map_location="cpu"))
    return m.eval()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def panel(src: ds.Dataset, target: ds.Dataset, drop: str | None):
    kept = src if drop is None else src.subset(src.cell_type.ne(drop))
    probes = reference.usable_probes(kept.betas, target.betas)
    markers = reference.pick_markers(kept.betas.loc[probes], kept.cell_type, N_PER_SIDE)
    mean, sd = reference.build_reference(kept.betas, kept.cell_type, markers)
    return mean, sd, len(probes), list(kept.betas.columns)


def score_panel(model, mean, sd, bulk):
    """Every method on one panel: {method: (kept n x K, other n)}."""
    ref_df = mean
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fits = solvers.run_all(ref_df, bulk, with_epidish=True, with_paper=False)
    for name, est in fits.items():
        k = est[list(mean.columns)].to_numpy(float)
        out[name] = (k, 1.0 - k.sum(axis=1))  # classical `other` = 1 - sum (diagnostic only)
    R, S, B, fit = ps.tensors(mean, sd, bulk)
    e = ps.extract(model, R, S, B, fit)
    for pi in PIS:
        kept, other, _ = ps.shifted(e, pi)
        out[f"model_pi{pi:.2f}"] = (kept, other)
    out["model_other0"] = (e["comp"], np.zeros(len(e["comp"])))  # retained composition, other forced to 0
    return out


def rmse_cells(err: np.ndarray) -> float:
    return 100 * float(np.sqrt(np.mean(err ** 2)))


def boot_ci(stat, n_rows: int, rng_seed: int = SEED):
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n_rows, size=(BOOT, n_rows))
    return np.array([stat(i) for i in idx])


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)

    if args.dry_run:
        src, target = ds.load("GSE110554").sorted, ds.load("GSE110554").mixtures
        tag = "DRY RUN on GSE110554 (not the reserved test)"
    else:
        src, target = ds.load("GSE167998").sorted, ds.load("GSE182379").mixtures
        tag = "LOCKED FINAL TEST GSE167998 -> GSE182379"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    truth = target.truth
    types = list(pd.unique(src.cell_type))
    samples = list(target.betas.columns)
    print(tag, "| types", len(types), "| mixtures", len(samples), flush=True)

    model = load_checkpoint(args.checkpoint)
    rows = []  # per (drop, method, sample)
    panels = {}
    for drop in [None] + types:
        mean, sd, n_usable, ref_samples = panel(src, target, drop)
        bulk = target.betas.loc[mean.index]
        res = score_panel(model, mean, sd, bulk)
        kept = list(mean.columns)
        panels[drop or "none"] = dict(n_usable=n_usable, n_markers=len(mean), retained=kept, ref_samples=ref_samples)
        t_kept = truth.loc[samples, kept].to_numpy(float)
        t_other = truth.loc[samples, drop].to_numpy(float) if drop else np.zeros(len(samples))
        for method, (k, o) in res.items():
            for i, s in enumerate(samples):
                rows.append(dict(dropped=drop or "none", method=method, sample=s,
                                 n_markers=len(mean), other_est=float(o[i]), other_true=float(t_other[i]),
                                 **{f"est_{c}": float(k[i, j]) for j, c in enumerate(kept)},
                                 **{f"true_{c}": float(t_kept[i, j]) for j, c in enumerate(kept)}))
        print(f"done {drop or 'clean'}: {len(mean)} markers", flush=True)
    long = pd.DataFrame(rows)
    long.to_csv(out / "estimates_long.csv", index=False)

    def kept_err(sub: pd.DataFrame, kept_cols):
        e = np.stack([sub[f"est_{c}"].to_numpy() - sub[f"true_{c}"].to_numpy() for c in kept_cols], axis=1)
        return e

    summary = {"tag": tag, "n_per_side": N_PER_SIDE, "seed": SEED, "boot": BOOT, "pis": PIS,
               "comparator": COMPARATOR, "types": types, "n_mixtures": len(samples), "panels": panels}
    methods = [m for m in long.method.unique()]
    method_order = [m for m in CLASSICAL if m in methods] + [m for m in methods if m.startswith("model")]

    # ---- primary 1: clean pooled RMSE (full-mixture fractions), 12 mixtures x K types
    clean = long[long.dropped == "none"]
    kept_all = panels["none"]["retained"]
    cell = {}
    for m in method_order:
        sub = clean[clean.method == m].set_index("sample").loc[samples].reset_index()
        cell[m] = kept_err(sub, kept_all)  # n x K
    clean_res = {m: rmse_cells(cell[m]) for m in method_order}
    D_obs, D_boot = {}, {}
    for m in method_order:
        if m == COMPARATOR:
            continue
        D_obs[m] = rmse_cells(cell[m]) - 1.10 * rmse_cells(cell[COMPARATOR])
        d = boot_ci(lambda i, m=m: rmse_cells(cell[m][i]) - 1.10 * rmse_cells(cell[COMPARATOR][i]), len(samples))
        D_boot[m] = d
    summary["clean"] = {
        "rmse_pts": clean_res,
        "ratio_to_RPC": {m: clean_res[m] / clean_res[COMPARATOR] for m in method_order},
        "D_obs_pts": D_obs,
        "D_upper95_one_sided_pts": {m: float(np.percentile(D_boot[m], 95)) for m in D_boot},
        "D_two_sided95_pts": {m: [float(np.percentile(D_boot[m], 2.5)), float(np.percentile(D_boot[m], 97.5))] for m in D_boot},
        "goal_met_within_1p10x": {m: bool(np.percentile(D_boot[m], 95) < 0) for m in D_boot},
    }
    # per-sample and per-type clean tables
    per_type = pd.DataFrame({m: 100 * np.sqrt((cell[m] ** 2).mean(axis=0)) for m in method_order}, index=kept_all)
    per_type.to_csv(out / "clean_rmse_per_type.csv")
    bias = pd.DataFrame({m: 100 * cell[m].mean(axis=0) for m in method_order}, index=kept_all)
    bias.to_csv(out / "clean_signed_bias_per_type.csv")

    # ---- secondary: parent totals (clean)
    parents = {}
    for m in method_order:
        sub = clean[clean.method == m].set_index("sample").loc[samples]
        est = pd.DataFrame({c: sub[f"est_{c}"] for c in kept_all})
        tru = pd.DataFrame({c: sub[f"true_{c}"] for c in kept_all})
        e, t = ds.collapse(est, ds.TO_COARSE), ds.collapse(tru, ds.TO_COARSE)
        parents[m] = {"pooled_rmse_pts": rmse_cells((e - t).to_numpy()),
                      "per_parent_rmse_pts": {c: 100 * float(np.sqrt(((e[c] - t[c]) ** 2).mean())) for c in e.columns},
                      "per_parent_bias_pts": {c: 100 * float((e[c] - t[c]).mean()) for c in e.columns}}
    summary["parent_totals_clean"] = parents

    # ---- primary 2/3: missing types
    dr = long[long.dropped != "none"].copy()
    elig = {(d, s) for d in types for s in samples if truth.loc[s, d] >= ABUNDANT}
    summary["eligible_pairs"] = {"total": len(elig), "per_type": {d: sum(1 for p in elig if p[0] == d) for d in types}}

    def mt_stats(drops, sample_idx=None):
        sidx = range(len(samples)) if sample_idx is None else sample_idx
        out_m = {}
        for m in method_order:
            sq, n_cells, oerr = [], 0, []
            for d in drops:
                sub = dr[(dr.dropped == d) & (dr.method == m)].set_index("sample").loc[samples]
                kept = panels[d]["retained"]
                e = kept_err(sub.reset_index(), kept)
                o = (sub.other_est - sub.other_true).to_numpy()
                for i in sidx:
                    if (d, samples[i]) in elig:
                        sq.append((e[i] ** 2).sum()); n_cells += len(kept); oerr.append(o[i])
            out_m[m] = (100 * float(np.sqrt(sum(sq) / n_cells)) if n_cells else float("nan"),
                        100 * float(np.mean(np.abs(oerr))) if oerr else float("nan"),
                        100 * float(np.mean(oerr)) if oerr else float("nan"))
        return out_m

    def mt_block(drops):
        obs = mt_stats(drops)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(samples), size=(BOOT, len(samples)))
        # paired bootstrap of the RMSE ratio / reduction vs RPC over resampled mixture rows
        boots = {m: np.empty(BOOT) for m in method_order}
        other_boot = {m: np.empty(BOOT) for m in method_order}
        # precompute per (drop, method, sample) squared-error sums and omitted errors for speed
        cache = {}
        for m in method_order:
            for d in drops:
                sub = dr[(dr.dropped == d) & (dr.method == m)].set_index("sample").loc[samples]
                e = kept_err(sub.reset_index(), panels[d]["retained"])
                mask = np.array([(d, s) in elig for s in samples])
                cache[(m, d)] = ((e ** 2).sum(axis=1) * mask, mask * len(panels[d]["retained"]),
                                 np.abs(sub.other_est.to_numpy() - sub.other_true.to_numpy()) * mask, mask)
        for m in method_order:
            sq = sum(cache[(m, d)][0] for d in drops); nc = sum(cache[(m, d)][1] for d in drops)
            ab = sum(cache[(m, d)][2] for d in drops); nm = sum(cache[(m, d)][3] for d in drops)
            boots[m] = 100 * np.sqrt(sq[idx].sum(axis=1) / np.maximum(nc[idx].sum(axis=1), 1))
            other_boot[m] = 100 * ab[idx].sum(axis=1) / np.maximum(nm[idx].sum(axis=1), 1)
        res = {}
        for m in method_order:
            r = {"kept_rmse_pts": obs[m][0], "other_mae_pts": obs[m][1], "other_bias_pts": obs[m][2],
                 "other_mae_ci95": [float(np.percentile(other_boot[m], 2.5)), float(np.percentile(other_boot[m], 97.5))]}
            if m != COMPARATOR:
                red = 1 - boots[m] / boots[COMPARATOR]
                r.update(reduction_vs_RPC=1 - obs[m][0] / obs[COMPARATOR][0],
                         reduction_ci95=[float(np.percentile(red, 2.5)), float(np.percentile(red, 97.5))],
                         reduction_one_sided_lower95=float(np.percentile(red, 5)),
                         goal_30pct_reduction_met=bool(np.percentile(red, 5) > 0.30))
            res[m] = r
        return res

    summary["missing_all_drops"] = mt_block(types)
    summary["missing_no_neutrophil"] = mt_block([d for d in types if d != "Neu"]) if "Neu" in types else None
    per_drop = []
    for d in types:
        o = mt_stats([d])
        for m in method_order:
            per_drop.append(dict(dropped=d, method=m, n_pairs=sum(1 for p in elig if p[0] == d),
                                 kept_rmse_pts=o[m][0], other_mae_pts=o[m][1], other_bias_pts=o[m][2]))
    pd.DataFrame(per_drop).to_csv(out / "missing_per_drop.csv", index=False)

    # parent totals on drops: only parents that do not contain the dropped type
    pt = []
    for m in method_order:
        sq, n = 0.0, 0
        for d in types:
            sub = dr[(dr.dropped == d) & (dr.method == m)].set_index("sample").loc[samples]
            kept = panels[d]["retained"]
            est = pd.DataFrame({c: sub[f"est_{c}"] for c in kept}); tru = pd.DataFrame({c: sub[f"true_{c}"] for c in kept})
            e, t = ds.collapse(est, ds.TO_COARSE), ds.collapse(tru, ds.TO_COARSE)
            ok = [c for c in e.columns if c != ds.TO_COARSE.get(d, d)]
            for s in samples:
                if (d, s) in elig:
                    sq += float(((e.loc[s, ok] - t.loc[s, ok]) ** 2).sum()); n += len(ok)
        pt.append((m, 100 * float(np.sqrt(sq / n)) if n else float("nan")))
    summary["parent_totals_unaffected_parents_on_eligible_drops_pts"] = dict(pt)

    # ---- provenance
    summary["provenance"] = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint / "model.pt"),
        "checkpoint_settings": json.loads((args.checkpoint / "summary.json").read_text())["settings"],
        "script_sha256": sha256(Path(__file__)), "prior_sweep_sha256": sha256(Path("prior_sweep.py")),
        "python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
        "rscript": solvers.RSCRIPT, "predeclaration": "FINAL_TEST_PREDECLARATION_2026-11-19.md",
        "target_cache_files": {p.name: sha256(p) for p in sorted(ds.CACHE.glob("GSE182379_noob.f32.bin"))
                               } if not args.dry_run else {},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("clean", "eligible_pairs")}, indent=2, default=float)[:3500])
    print("missing_all_drops:", json.dumps({m: summary["missing_all_drops"][m] for m in method_order
                                            if m in (COMPARATOR, "model_pi0.55", "model_pi0.10")}, indent=2, default=float)[:3500])
    print("wrote", out)


if __name__ == "__main__":
    main()
