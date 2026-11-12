"""Label-free null check: the v5 checkpoint on unlabelled whole-blood cohorts.

Whole blood has essentially no `other` (cells outside the six sorted types), so any
large `other` on these cohorts is false. No truth labels are read anywhere here.

Reference: six-type sorted panel from GSE110554 (`datasets.load("GSE110554").sorted`),
built with the same panel and method functions as eval_real_dev.py (25 markers per side,
no dropped type). eval_real_dev sets DECONV_PREPROC=noob, so the reference uses Noob betas.

Cohorts use the processed matrices as deposited (NOT Noob). Probes: the reference's fully
observed CpGs intersected with each cohort. Any probe with a missing beta in the cohort is
dropped; samples are kept. (Dropping any sample with a missing beta would remove every sample
in GSE87571, whose deposited betas have scattered NA values; the number of samples the
sample-level rule would have removed is reported as `n_samples_literal_drop`.)

Writes to results/null_whole_blood_gpu_v5/: estimates.csv (one row per sample) and
summary.csv (one row per cohort, with a status column so failures are visible).
Run from the deconv folder:  py -3.11 -B label_free_null.py
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import eval_real_dev as ev  # sets DECONV_PREPROC=noob before datasets is imported
import datasets as ds
import reference
from model import Deconvolver
from rpc_torch import rpc_fit

CKPT = Path("results/gpu_v5_seed0")
OUT = Path("results/null_whole_blood_gpu_v5")
DATA = Path(os.environ.get("COMPGEN_DATA", str(Path.home() / "data" / "comp-genomics")))
PER_SIDE = 25
THRESH = 0.05  # 5 points

COHORT_NOTES = {
    "GSE40279": "450K; deposited average beta table (GSE40279_average_beta.txt.gz)",
    "GSE42861": "450K; deposited processed methylation matrix",
    "GSE87571": "450K; deposited matrix1of2 + matrix2of2, beta columns only (detection-p columns dropped)",
    "GSE318173": "EPIC; deposited betas in discovery/r3 (all 559 arrays, PD and control whole blood)",
}


def read_matrix(path: Path, keep=lambda c: True) -> pd.DataFrame:
    """Probe x sample float32 table from a tab-separated matrix with ID_REF in column 1."""
    cols = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    use = [cols[0]] + [c for c in cols[1:] if keep(c)]
    df = pd.read_csv(path, sep="\t", usecols=use, index_col=0,
                     dtype={c: np.float32 for c in use[1:]}, na_values=["NA", "null", ""])
    df.index = df.index.astype(str)
    return df


def series_fields(path: Path) -> dict[str, list[str]]:
    """Per-sample fields from a GEO series matrix. Characteristic rows are keyed by their name."""
    out: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            key, _, rest = line.rstrip("\n").partition("\t")
            vals = [v.strip('"') for v in rest.split("\t")]
            if key == "!Sample_characteristics_ch1":
                out[vals[0].split(":", 1)[0]] = [v.split(":", 1)[1].strip() for v in vals]
            else:
                out[key[len("!Sample_"):]] = vals
    return out


def load_gse40279():
    base = DATA / "discovery" / "publication_legacy450k" / "raw" / "GSE40279"
    df = read_matrix(base / "GSE40279_average_beta.txt.gz")
    key = pd.read_csv(base / "GSE40279_sample_key.txt.gz", sep="\t", header=None,
                      names=["idx", "x_id", "barcode"], dtype=str)
    barcode = {f"X{x}": b for x, b in zip(key.x_id, key.barcode)}
    ids = list(df.columns)
    extra = pd.Series([barcode.get(c, "") for c in ids], index=ids)
    return df, extra, "GSE40279 sample key gives array barcodes (no GSM mapping in the local files)"


def load_gse42861():
    df = read_matrix(DATA / "raw" / "GSE42861_processed_methylation_matrix.txt.gz")
    return df, pd.Series("", index=df.columns), ""


def load_gse87571():
    base = DATA / "discovery" / "publication_legacy450k" / "raw" / "GSE87571"
    parts = [read_matrix(base / f"GSE87571_matrix{p}of2.txt.gz", keep=lambda c: not c.endswith(".1"))
             for p in (1, 2)]
    df = pd.concat(parts, axis=1, join="inner")
    fields = series_fields(base / "GSE87571_series_matrix.txt.gz")
    to_gsm = {t.split()[0]: g for t, g in zip(fields["title"], fields["geo_accession"])}
    missing = [c for c in df.columns if c not in to_gsm]
    if missing:
        raise ValueError(f"GSE87571: {len(missing)} columns without a GSM title, e.g. {missing[:3]}")
    df = df.rename(columns=to_gsm)
    return df, pd.Series("", index=df.columns), "beta columns are the un-suffixed ones; .1 columns are detection p"


def load_gse318173():
    base = DATA / "discovery" / "r3"
    vals = np.load(base / "betas_gse318173_all559_f32.npy", mmap_mode="r")
    probes = (base / "betas_gse318173_probe_ids.txt").read_text().split()
    names = (base / "betas_gse318173_sample_names.txt").read_text().split()
    if vals.shape != (len(probes), len(names)):
        raise ValueError(f"GSE318173 shape {vals.shape} vs {len(probes)} probes x {len(names)} names")
    fields = series_fields(base / "raw" / "gse318173_series" / "GSE318173_series_matrix.txt.gz")
    barcode_to_gsm = {t.replace("whole_blood_", ""): g
                      for t, g in zip(fields["title"], fields["geo_accession"])}
    status = dict(zip(barcode_to_gsm.values(), fields["disease_status"]))
    gsm = [barcode_to_gsm.get(n) for n in names]
    if any(g is None for g in gsm):
        raise ValueError("GSE318173: sample names without a GEO title")
    df = pd.DataFrame(np.asarray(vals, dtype=np.float32), index=pd.Index(probes, name="ID_REF"), columns=gsm)
    extra = pd.Series([status[g] for g in gsm], index=gsm)
    return df, extra, "GSE318173 disease_status in the series matrix: " + str(extra.value_counts().to_dict())


LOADERS = {
    "GSE40279": load_gse40279,
    "GSE42861": load_gse42861,
    "GSE87571": load_gse87571,
    "GSE318173": load_gse318173,
}


def model_p_other_positive(model, mean, sd, bulk) -> np.ndarray:
    """P(other > 0) from the model's zero-inflated head, same inputs as ev.run_methods."""
    ref = mean.to_numpy(float)
    n = bulk.shape[1]
    R = np.repeat(ref[None], n, axis=0)
    S = np.repeat(sd.to_numpy(float)[None], n, axis=0)
    B = bulk.to_numpy(float).T
    fit = rpc_fit(R, B)
    as_t = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
    with torch.inference_mode():
        cohort = torch.zeros(n, dtype=torch.long) if getattr(model, "cohort", False) else None
        p = model(as_t(R), as_t(S), as_t(B), fit, cohort)
        return p.has_other.probs.numpy().astype(float)


def rpc_rawsum_other(mean, sd, bulk) -> np.ndarray:
    """RPC's own unnormalised fit: other = 1 - sum of clamped slopes (as in train_cpu.py)."""
    ref = mean.to_numpy(float)
    n = bulk.shape[1]
    R = np.repeat(ref[None], n, axis=0)
    B = bulk.to_numpy(float).T
    fit = rpc_fit(R, B)
    raw_sum = fit.coefficients.clamp_min(0).sum(-1).numpy()
    return np.clip(1 - raw_sum, 0, 1)


def run_cohort(name, model, src, rows, summary):
    print(f"[{name}] loading", flush=True)
    df, extra, note = LOADERS[name]()
    n_in = df.shape[1]
    ref_ok = reference.usable_probes(src.betas)
    shared = ref_ok.intersection(pd.Index(df.index))
    sub = df.loc[shared]
    literal_drop = int(sub.isna().any(axis=0).sum())
    sub = sub[sub.notna().all(axis=1)]  # probe-level: drop probes with any missing beta
    sub = sub.astype(np.float64)
    print(f"[{name}] {n_in} samples, {len(ref_ok)} reference probes, {len(shared)} shared, "
          f"{len(sub)} used; sample-level rule would drop {literal_drop}", flush=True)
    cohort = SimpleNamespace(betas=sub)
    mean, sd = ev.panel(src, cohort, PER_SIDE, None)
    bulk = sub.loc[mean.index]
    methods, _ = ev.run_methods(model, mean, sd, bulk)
    other = methods["model"][1]
    p_pos = model_p_other_positive(model, mean, sd, bulk)
    rpc = methods["RPC"][0]
    raw_other = rpc_rawsum_other(mean, sd, bulk)
    for i, sample in enumerate(bulk.columns):
        rec = dict(cohort=name, sample_id=sample, extra=extra.get(sample, ""),
                   model_other=float(other[i]), model_p_other_gt0=float(p_pos[i]),
                   rpc_rawsum_other=float(raw_other[i]))
        rec.update({f"RPC_{t}": float(rpc[i, j]) for j, t in enumerate(mean.columns)})
        rows.append(rec)
    o, r = 100 * other, 100 * raw_other
    summary.append(dict(
        cohort=name, status="ok", reason="", platform=("EPIC" if name == "GSE318173" else "450K"),
        preprocessing="deposited (not Noob)", note=COHORT_NOTES[name] + ("; " + note if note else ""),
        n_samples_in_file=n_in, n_samples_used=int(bulk.shape[1]),
        n_samples_literal_drop=literal_drop, n_reference_probes=int(len(ref_ok)),
        n_shared_probes=int(len(shared)), n_probes_used=int(len(sub)), n_markers=int(len(mean)),
        model_mean_other_pts=float(o.mean()), model_median_other_pts=float(np.median(o)),
        model_frac_other_gt5=float((other > THRESH).mean()), model_mean_p_other_gt0=float(p_pos.mean()),
        rpc_rawsum_mean_other_pts=float(r.mean()), rpc_rawsum_median_other_pts=float(np.median(r)),
        rpc_rawsum_frac_other_gt5=float((raw_other > THRESH).mean()),
    ))


def main():
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = json.loads((CKPT / "summary.json").read_text())
    ev.PRIOR_OTHER = float(ckpt.get("best", {}).get("prior_mean_other", ev.PRIOR_OTHER))
    s = ckpt["settings"]
    model = Deconvolver(width=s["width"], heads=s["heads"], blocks=s["blocks"], cohort=s.get("cohort", False))
    model.load_state_dict(torch.load(CKPT / "model.pt", map_location="cpu"))
    model.eval()

    src = ds.load("GSE110554").sorted
    types = list(pd.unique(src.cell_type))
    print("reference types:", types, "samples:", src.betas.shape[1], flush=True)
    rows, summary = [], []
    for name in LOADERS:
        try:
            run_cohort(name, model, src, rows, summary)
        except Exception as exc:  # keep going; record why
            traceback.print_exc()
            summary.append(dict(cohort=name, status="failed", reason=f"{type(exc).__name__}: {exc}"))
            print(f"[{name}] FAILED: {exc}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "estimates.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    print(pd.DataFrame(summary).T.to_string(), flush=True)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
