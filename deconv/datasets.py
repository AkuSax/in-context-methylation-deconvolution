"""Loaders for the blood methylation datasets with known cell fractions.

Every loader returns a Dataset whose `truth` is a sample x cell-type table of
fractions in [0, 1] (NaN where a sample has no truth) and whose `cell_type`
labels purified samples (None for mixtures and whole blood).

Raw files live outside the repo; set DECONV_RAW to override the location.

DECONV_PREPROC picks which betas the loaders return:
  geo   as deposited on GEO, each series with its own submitter's preprocessing
  noob  every series rebuilt from IDATs by reprocess.py (one recipe for all)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import geo

RAW = Path(os.environ.get("DECONV_RAW", str(Path.home() / "data" / "comp-genomics" / "deconv" / "raw")))
CACHE = RAW.parent / "cache"
PREPROC = os.environ.get("DECONV_PREPROC", "geo")

FINE12 = ["Neu", "Eos", "Bas", "Mono", "Bnv", "Bmem", "CD4nv", "CD4mem", "Treg", "CD8nv", "CD8mem", "NK"]
COARSE6 = ["Gran", "Mono", "B", "CD4T", "CD8T", "NK"]

# Any label -> its coarse six-type parent.
TO_COARSE = {
    "Neu": "Gran", "Eos": "Gran", "Bas": "Gran", "Gran": "Gran",
    "Mono": "Mono",
    "Bnv": "B", "Bmem": "B", "B": "B",
    "CD4nv": "CD4T", "CD4mem": "CD4T", "Treg": "CD4T", "CD4T": "CD4T",
    "CD8nv": "CD8T", "CD8mem": "CD8T", "CD8T": "CD8T",
    "NK": "NK",
}


def collapse(frac: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Sum fraction columns into parent categories; unmapped columns are kept."""
    parents = [mapping.get(c, c) for c in frac.columns]
    out = frac.T.groupby(parents, sort=False).sum(min_count=1).T
    return out


@dataclass
class Dataset:
    name: str
    platform: str
    betas: pd.DataFrame  # probe x sample
    cell_type: pd.Series  # sample -> purified label, or None
    truth: pd.DataFrame  # sample x type, fractions
    truth_kind: pd.Series  # sample -> "constructed" | "flow" | None
    meta: pd.DataFrame

    def subset(self, mask) -> "Dataset":
        ids = self.betas.columns[np.asarray(mask)]
        return Dataset(self.name, self.platform, self.betas[ids], self.cell_type[ids],
                       self.truth.loc[ids], self.truth_kind[ids], self.meta.loc[ids])

    @property
    def sorted(self) -> "Dataset":
        return self.subset(self.cell_type.notna())

    @property
    def mixtures(self) -> "Dataset":
        return self.subset(self.truth_kind.notna())


def _series(gse: str):
    path = RAW / gse / f"{gse}_series_matrix.txt.gz"
    meta = geo.read_meta(path)
    return path, meta, meta.frame()


def _noob_betas(gse: str, df: pd.DataFrame) -> pd.DataFrame:
    """Betas written by noob_idats.R, columns renamed from IDAT basenames to GSM ids."""
    vals, rows, cols = (CACHE / f"{gse}_noob.{s}" for s in ("f32.bin", "probes.txt", "samples.txt"))
    if not vals.exists():
        raise FileNotFoundError(f"{vals} is missing; run `python reprocess.py all {gse}` first")
    probes = rows.read_text().split()
    gsm = [c.split("_", 1)[0] for c in cols.read_text().split()]
    x = np.fromfile(vals, dtype=np.float32).reshape((len(probes), len(gsm)), order="F")
    betas = pd.DataFrame(x, index=pd.Index(probes, name="probe"), columns=gsm)
    missing = [g for g in df.index if g not in betas.columns]
    if missing:
        raise ValueError(f"{gse}: no IDAT for {missing[:5]}")
    return betas[list(df.index)]


def _betas(gse: str, df: pd.DataFrame, deposited) -> pd.DataFrame:
    """The series' betas under the selected preprocessing; `deposited` loads the GEO ones."""
    if PREPROC == "geo":
        return deposited()
    if PREPROC == "noob":
        return _noob_betas(gse, df)
    raise ValueError(f"DECONV_PREPROC must be 'geo' or 'noob', not {PREPROC!r}")


def _num(df: pd.DataFrame, cols: dict[str, str], scale: float) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for src, dst in cols.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce") * scale
    return out


def load_gse110554() -> Dataset:
    """EPIC. 37 purified samples of six types plus 12 constructed mixtures."""
    path, _, df = _series("GSE110554")
    betas = _betas("GSE110554", df, lambda: geo.cached_betas(path, CACHE))
    lab = df["cell type"].map({"Neu": "Neu", "Mono": "Mono", "Bcell": "B", "CD4T": "CD4T",
                               "CD8T": "CD8T", "NK": "NK"})
    is_mix = df["cell type"].eq("MIX")
    truth = _num(df, {"neu": "Neu", "mono": "Mono", "bcell": "B", "cd4t": "CD4T",
                      "cd8t": "CD8T", "nk": "NK"}, 0.01)
    truth[~is_mix] = np.nan
    kind = pd.Series(np.where(is_mix, "constructed", None), index=df.index, dtype=object)
    return Dataset("GSE110554", "EPIC", betas, lab.where(~is_mix, None), truth, kind, df)


_SRC12 = [
    ("CD4+ CD45RA+", "CD4nv"), ("CD4+ CD45RO+", "CD4mem"), ("Basophils", "Bas"),
    ("CD19+ CD27+", "Bmem"), ("CD19+ CD27-", "Bnv"), ("T regulatory", "Treg"),
    ("Effector Memory", "CD8mem"), ("CD8+ CD45RA+", "CD8nv"), ("Eosinophils", "Eos"),
    ("Natural killers", "NK"), ("neutrophils", "Neu"), ("Monocytes", "Mono"),
]


def load_gse167998() -> Dataset:
    """EPIC. 56 purified samples of twelve types plus 12 constructed mixtures.

    The mixture annotation gives CD8 T cells as one total and granulocytes as a
    total, so mixture truth has eleven columns: CD8T is unsplit and basophils
    are granulocytes minus neutrophils minus eosinophils.
    """
    _, _, df = _series("GSE167998")
    betas = _betas("GSE167998", df, lambda: _gse167998_deposited(df))
    is_mix = df["source"].str.contains("artificial mixture")
    lab = pd.Series(None, index=df.index, dtype=object)
    for key, name in _SRC12:
        lab[df["source"].str.contains(key, regex=False) & ~is_mix] = name
    if lab[~is_mix].isna().any():
        raise ValueError("GSE167998: purified sample with unrecognised source")
    t = _num(df, {"neu": "Neu", "eos": "Eos", "gran": "Gran", "mono": "Mono", "bnv": "Bnv",
                  "bmem": "Bmem", "cd4nv": "CD4nv", "cd4mem": "CD4mem", "treg": "Treg",
                  "cd8t": "CD8T", "nk": "NK"}, 1.0)
    t["Bas"] = (t["Gran"] - t["Neu"] - t["Eos"]).clip(lower=0)
    truth = t[["Neu", "Eos", "Bas", "Mono", "Bnv", "Bmem", "CD4nv", "CD4mem", "Treg", "CD8T", "NK"]].copy()
    truth[~is_mix] = np.nan
    kind = pd.Series(np.where(is_mix, "constructed", None), index=df.index, dtype=object)
    return Dataset("GSE167998", "EPIC", betas, lab, truth, kind, df)


def _gse167998_deposited(df: pd.DataFrame) -> pd.DataFrame:
    """The submitter's SeSAMe betas, which GEO holds as a supplementary table."""
    vals, rows, cols = (CACHE / f"GSE167998_matrix_processed.{s}" for s in ("f32.npy", "probes.txt", "samples.txt"))
    sentrix = df["basename"].str.rsplit("/", n=1).str[-1]
    if vals.exists():
        betas = pd.DataFrame(np.load(vals), index=pd.Index(rows.read_text().split("\n"), name="probe"),
                             columns=cols.read_text().split("\n"))
    else:
        proc = RAW / "GSE167998" / "GSE167998_matrix_processed.txt.gz"
        header = pd.read_csv(proc, sep="\t", nrows=0).columns
        keep = [c for c in header if "Detection" not in c]
        raw = pd.read_csv(proc, sep="\t", usecols=keep, index_col=0, na_values=["", "NA", "null"],
                          dtype={c: np.float32 for c in keep[1:]})
        raw.index = raw.index.astype(str)
        raw.index.name = "probe"
        back = dict(zip(sentrix, df.index))
        missing = [c for c in raw.columns if c not in back]
        if missing:
            raise ValueError(f"GSE167998: array ids without a GSM: {missing[:5]}")
        betas = raw.rename(columns=back)[list(df.index)]
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(vals, betas.to_numpy(dtype=np.float32))
        rows.write_text("\n".join(betas.index))
        cols.write_text("\n".join(betas.columns))
    return betas


def load_gse182379() -> Dataset:
    """EPIC. 12 constructed mixtures of twelve types (validation set)."""
    path, _, df = _series("GSE182379")
    betas = _betas("GSE182379", df, lambda: geo.cached_betas(path, CACHE))
    truth = _num(df, {"neu": "Neu", "eos": "Eos", "bas": "Bas", "mono": "Mono", "bnv": "Bnv",
                      "bmem": "Bmem", "cd4nv": "CD4nv", "cd4mem": "CD4mem", "treg": "Treg",
                      "cd8nv": "CD8nv", "cd8mem": "CD8mem", "nk": "NK"}, 0.01)
    kind = pd.Series("constructed", index=df.index, dtype=object)
    return Dataset("GSE182379", "EPIC", betas, pd.Series(None, index=df.index, dtype=object), truth, kind, df)


def load_gse112618() -> Dataset:
    """EPIC. Six adult whole-blood samples with flow-cytometry proportions."""
    path, _, df = _series("GSE112618")
    betas = _betas("GSE112618", df, lambda: geo.cached_betas(path, CACHE))
    truth = _num(df, {"granulocytes proportion": "Gran", "monocytes proportion": "Mono",
                      "bcell proportion": "B", "cd4t proportion": "CD4T",
                      "cd8t proportion": "CD8T", "nk proportion": "NK"}, 1.0)
    kind = pd.Series("flow", index=df.index, dtype=object)
    return Dataset("GSE112618", "EPIC", betas, pd.Series(None, index=df.index, dtype=object), truth, kind, df)


def load_gse77797() -> Dataset:
    """450K. 12 reconstructed mixtures and 6 whole-blood samples with flow."""
    path, _, df = _series("GSE77797")
    betas = _betas("GSE77797", df, lambda: geo.cached_betas(path, CACHE))
    truth = _num(df, {"granulocyte (%)": "Gran", "monocyte (%)": "Mono", "b cell (%)": "B",
                      "cd4+ t cell (%)": "CD4T", "cd8+ t cell (%)": "CD8T",
                      "natural killer cell (%)": "NK"}, 0.01)
    kind = pd.Series(np.where(df["source"].str.contains("Reconstructed"), "constructed", "flow"),
                     index=df.index, dtype=object)
    return Dataset("GSE77797", "450K", betas, pd.Series(None, index=df.index, dtype=object), truth, kind, df)


def load_gse127824() -> Dataset:
    """450K. 24 cord-blood samples with FACS/CBC proportions including nRBC."""
    path, _, df = _series("GSE127824")
    betas = _betas("GSE127824", df, lambda: geo.cached_betas(path, CACHE))
    truth = _num(df, {"granulocytes": "Gran", "monocytes": "Mono", "b cells": "B",
                      "cd4t cells": "CD4T", "cd8t cells": "CD8T", "nk cells": "NK",
                      "nrbcs": "nRBC"}, 0.01)
    kind = pd.Series("flow", index=df.index, dtype=object)
    return Dataset("GSE127824", "450K", betas, pd.Series(None, index=df.index, dtype=object), truth, kind, df)


GSE103541_EXCLUDED = ("Patient0", "Patient25")


def load_gse103541() -> Dataset:
    """EPIC. 135 purified samples: five types from each of 27 donors. No mixtures, no truth.

    Granulocytes are labelled `Gran` (not split into neutrophils and eosinophils); there
    are no NK cells. `meta["donor"]` holds the patient id. Only Noob betas are available
    locally; the submitter's normalised table was not downloaded.

    Two donors are dropped (`GSE103541_EXCLUDED`):
    - `Patient0`: the submitter's quality control failed all five arrays.
    - `Patient25`: its B, CD4 T, CD8 T and monocyte arrays are 33-60% granulocyte by
      `check_gse103541.py` (own-type share 0.22-0.46; every other array is at least 0.85),
      and the submitter left those four arrays out of the later GSE166844 deposit.

    141 of the 145 arrays were deposited again in GSE166844, so the two series are not
    independent sources.
    """
    _, _, df = _series("GSE103541")
    df = df[~df["patient number"].isin(GSE103541_EXCLUDED)].copy()
    df["donor"] = "GSE103541:" + df["patient number"]
    lab = df["cell type"].map({"B-cells": "B", "CD4 T-cells": "CD4T", "CD8 T-cells": "CD8T",
                               "Granulocytes": "Gran", "Monocytes": "Mono"})
    if lab.isna().any():
        raise ValueError("GSE103541: unrecognised cell type")
    if PREPROC != "noob":
        raise ValueError("GSE103541 has Noob betas only; set DECONV_PREPROC=noob")
    betas = _noob_betas("GSE103541", df)
    truth = pd.DataFrame(np.nan, index=df.index, columns=["Gran", "Mono", "B", "CD4T", "CD8T"])
    kind = pd.Series(None, index=df.index, dtype=object)
    return Dataset("GSE103541", "EPIC", betas, lab.astype(object), truth, kind, df)


# Sorted-cell series with no fraction truth: sources for the simulator, not benchmarks.
# Kept out of LOADERS so the baseline and reprocessing defaults do not pick them up.
SORTED_LOADERS = {
    "GSE103541": load_gse103541,
}

LOADERS = {
    "GSE110554": load_gse110554,
    "GSE167998": load_gse167998,
    "GSE182379": load_gse182379,
    "GSE112618": load_gse112618,
    "GSE77797": load_gse77797,
    "GSE127824": load_gse127824,
}


def load(name: str) -> Dataset:
    return (LOADERS.get(name) or SORTED_LOADERS[name])()
