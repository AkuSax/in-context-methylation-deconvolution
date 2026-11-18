"""BLEND-M (Huang, Peters & McKennan 2026) as a comparator.

`select_markers` is a vectorized port of the package's `select_marker_cpg`. It is checked
against the R function by `python blendm.py --check`. `run` calls the vendored R
implementation (vendor/BLENDM, commit 90546bc) through blendm_solve.R. All tasks go
through one Rscript session.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from solvers import RSCRIPT

_HELPER = Path(__file__).with_name("blendm_solve.R")


def _anova(x: np.ndarray, codes: np.ndarray, k: int):
    """Row-wise one-way ANOVA. Returns (p-value, residual SD, group means G x k)."""
    n = x.shape[1]
    onehot = np.zeros((n, k))
    onehot[np.arange(n), codes] = 1.0
    cnt = onehot.sum(0)
    means = (x @ onehot) / cnt
    grand = x.mean(1, keepdims=True)
    ssb = ((means - grand) ** 2 * cnt).sum(1)
    ssw = ((x - means[:, codes]) ** 2).sum(1)
    msw = ssw / (n - k)
    f = (ssb / (k - 1)) / msw
    p = stats.f.sf(f, k - 1, n - k)
    return p, np.sqrt(msw), means


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment, same as R p.adjust(method = "BH")."""
    n = len(p)
    order = np.argsort(p)[::-1]
    adj = np.minimum.accumulate(p[order] * n / np.arange(n, 0, -1))
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def select_markers(betas: pd.DataFrame, labels: pd.Series, pval_cutoff: float = 0.05,
                   diff_cutoff: float = 10.0, chunk: int = 100_000) -> pd.Index:
    """BLEND-M two-step marker rule on purified samples (CpG x sample betas)."""
    lab = labels.loc[betas.columns].to_numpy()
    cats, codes = np.unique(lab, return_inverse=True)
    k = len(cats)
    p_all, diff_all = [], []
    for s in range(0, len(betas), chunk):
        x = np.clip(betas.iloc[s:s + chunk].to_numpy(np.float64), 1e-6, 1 - 1e-6)
        _, se, means = _anova(x, codes, k)
        norm = means / se[:, None]
        diff_all.append(norm.max(1) - norm.min(1))
        p, _, _ = _anova(np.log2(x / (1 - x)), codes, k)
        p_all.append(p)
    p = np.concatenate(p_all)
    diff = np.concatenate(diff_all)
    p = np.where(np.isnan(p), 1.0, p)  # R's p.adjust would keep NA; NA never passes the cutoff
    keep = (_bh(p) < pval_cutoff) & (diff > diff_cutoff)
    return betas.index[keep]


def run(tasks: list[tuple[pd.DataFrame, pd.Series, pd.DataFrame]]) -> list[dict[str, pd.DataFrame | str]]:
    """tasks: (reference samples markers x samples, sample -> type, bulk markers x samples).

    Returns, per task, {"unknown": samples x types (+ 'unknown'), "nounknown": samples x types};
    a failed variant holds its error message instead.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        names = []
        for i, (ref, lab, bulk) in enumerate(tasks):
            d = root / f"t{i}"
            d.mkdir()
            if not ref.index.equals(bulk.index):
                raise ValueError("reference and bulk CpGs differ")
            ref.to_csv(d / "ref.csv")
            pd.DataFrame({"sample": ref.columns, "type": lab.loc[ref.columns].to_numpy()}).to_csv(
                d / "labels.csv", index=False)
            bulk.to_csv(d / "bulk.csv")
            names.append(d.name)
        (root / "tasks.txt").write_text("\n".join(names) + "\n")
        proc = subprocess.run([RSCRIPT, "--vanilla", str(_HELPER), str(root)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"blendm_solve.R failed:\n{proc.stderr[-2000:]}")
        out = []
        for i, (_, _, bulk) in enumerate(tasks):
            d, res = root / f"t{i}", {}
            for v in ("unknown", "nounknown"):
                if (d / f"blendm_{v}.err").exists():
                    res[v] = (d / f"blendm_{v}.err").read_text().strip()
                else:
                    res[v] = pd.read_csv(d / f"blendm_{v}.csv", index_col=0).loc[bulk.columns]
            out.append(res)
    return out


def _check(n_probes: int = 3000):
    """Compare select_markers with the R select_marker_cpg on a random GSE110554 subset."""
    import os
    os.environ["DECONV_PREPROC"] = "noob"
    import datasets as ds
    src = ds.load("GSE110554").sorted
    b = src.betas.dropna()
    b = b.iloc[np.random.default_rng(0).choice(len(b), n_probes, replace=False)]
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        b.to_csv(t / "b.csv")
        pd.Series(src.cell_type.loc[b.columns].to_numpy()).to_csv(t / "l.csv", index=False)
        code = f"""
source('{(_HELPER.parent / 'vendor' / 'BLENDM' / 'BLENDM.R').as_posix()}')
b <- as.matrix(read.csv('{(t / 'b.csv').as_posix()}', row.names = 1, check.names = FALSE))
l <- read.csv('{(t / 'l.csv').as_posix()}')[[1]]
writeLines(select_marker_cpg(b, l), '{(t / 'out.txt').as_posix()}')
"""
        (t / "c.R").write_text(code)
        subprocess.run([RSCRIPT, "--vanilla", str(t / "c.R")], check=True, capture_output=True)
        r_sel = set((t / "out.txt").read_text().split())
    py_sel = set(select_markers(b, src.cell_type))
    print(f"{n_probes} probes: R {len(r_sel)}, Python {len(py_sel)}, identical {r_sel == py_sel}")
    return r_sel == py_sel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        raise SystemExit(0 if _check() else 1)
