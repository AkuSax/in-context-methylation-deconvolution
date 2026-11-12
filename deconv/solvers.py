"""Classical reference-based solvers for bulk = reference @ fractions.

`ref` is markers x types, `bulk` is markers x samples. Every solver returns a
samples x types array. EpiDISH's RPC/CBS/CP are called through Rscript so the
comparison uses the published implementation, as are the extra solvers from the
De Ridder et al. 2024 benchmark (EMeth, ICeDT, FARDEEP, LTS; `paper_solvers`).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize
from scipy.optimize import nnls as _nnls

RSCRIPT = os.environ.get("RSCRIPT", r"C:\Program Files\R\R-4.2.2\bin\Rscript.exe")
_R_HELPER = Path(__file__).with_name("epidish_solve.R")
_RPC_BATCH_HELPER = Path(__file__).with_name("epidish_rpc_batch.R")


def nnls(ref: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    """Non-negative least squares, no sum constraint."""
    return np.stack([_nnls(ref, bulk[:, j])[0] for j in range(bulk.shape[1])])


def nnls_intercept(ref: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    """NNLS with a free intercept, which absorbs a constant offset between the
    reference and the bulk (for example from different array preprocessing)."""
    design = np.column_stack([ref, np.ones(len(ref))])
    lo = np.append(np.zeros(ref.shape[1]), -np.inf)
    return np.stack([lsq_linear(design, bulk[:, j], bounds=(lo, np.inf)).x[:-1]
                     for j in range(bulk.shape[1])])


def _simplex_ls(ref: np.ndarray, b: np.ndarray) -> np.ndarray:
    """min ||ref w - b||^2 subject to w >= 0, sum(w) = 1."""
    k = ref.shape[1]
    gram, rhs = ref.T @ ref, ref.T @ b
    start = _nnls(ref, b)[0]
    start = start / start.sum() if start.sum() > 0 else np.full(k, 1.0 / k)
    res = minimize(
        lambda w: 0.5 * w @ gram @ w - rhs @ w,
        start,
        jac=lambda w: gram @ w - rhs,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(k)}],
        options={"ftol": 1e-14, "maxiter": 500},
    )
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def cp_equality(ref: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    """Constrained projection with fractions summing to exactly one."""
    return np.stack([_simplex_ls(ref, bulk[:, j]) for j in range(bulk.shape[1])])


def cp_inequality(ref: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    """Constrained projection with sum <= 1 (Houseman's default).

    The problem is convex, so if the NNLS optimum already satisfies the sum
    bound it is the answer; otherwise the bound is active and the equality
    solution is.
    """
    out = nnls(ref, bulk)
    over = out.sum(axis=1) > 1.0
    if over.any():
        out[over] = cp_equality(ref, bulk[:, over])
    return out


def epidish(ref: pd.DataFrame, bulk: pd.DataFrame, method: str = "RPC",
            constraint: str = "inequality") -> pd.DataFrame:
    """EpiDISH `epidish()` fractions (samples x types). method: RPC | CBS | CP."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref.to_csv(tmp / "ref.csv")
        bulk.to_csv(tmp / "bulk.csv")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(_R_HELPER), str(tmp / "ref.csv"), str(tmp / "bulk.csv"),
             method, constraint, str(tmp / "out.csv")],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"EpiDISH {method} failed:\n{proc.stderr[-2000:]}")
        out = pd.read_csv(tmp / "out.csv", index_col=0)
    return out.loc[bulk.columns, ref.columns]


PAPER_METHODS = ("EMeth_Laplace", "EMeth_Normal", "EMeth_Binom", "ICeDT", "FARDEEP", "LTS")


def paper_solvers(ref: pd.DataFrame, bulk: pd.DataFrame,
                  methods: tuple[str, ...] = PAPER_METHODS) -> dict[str, pd.DataFrame]:
    """Solvers from De Ridder et al. 2024 with that paper's settings (see paper_solvers.R).

    One Rscript session runs every method. Returned tables are samples x types and keep
    each method's own scale: EMeth sums to 1, ICeDT to at most 1, FARDEEP and LTS are
    unconstrained. A method that errors is left out and reported in `PAPER_ERRORS`.
    """
    PAPER_ERRORS.clear()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref.to_csv(tmp / "ref.csv")
        bulk.to_csv(tmp / "bulk.csv")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(_PAPER_HELPER), str(tmp / "ref.csv"), str(tmp / "bulk.csv"),
             str(tmp), ",".join(methods)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"paper_solvers.R failed:\n{proc.stderr[-2000:]}")
        out = {}
        for m in methods:
            if (tmp / f"{m}.err").exists():
                PAPER_ERRORS[m] = (tmp / f"{m}.err").read_text().strip()
                continue
            out[m] = pd.read_csv(tmp / f"{m}.csv", index_col=0).loc[bulk.columns, ref.columns]
    return out


PAPER_ERRORS: dict[str, str] = {}
_PAPER_HELPER = Path(__file__).with_name("paper_solvers.R")


def rpc_batch(ref: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    """Run published EpiDISH RPC on tasks that each have their own reference."""
    if ref.ndim != 3 or ref.shape[:2] != bulk.shape or min(ref.shape) == 0:
        raise ValueError("Expected nonempty task × probe × type references and task × probe bulks")
    tasks, markers, types = ref.shape
    data = np.concatenate((ref, bulk[..., None]), axis=-1).astype("<f8")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        source, result = path / "inputs.bin", path / "fractions.bin"
        data.tofile(source)
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(_RPC_BATCH_HELPER), str(source), str(result),
             str(tasks), str(markers), str(types)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"EpiDISH batch RPC failed:\n{proc.stderr[-2000:]}")
        return np.fromfile(result, dtype="<f8").reshape(tasks, types)


def run_all(ref: pd.DataFrame, bulk: pd.DataFrame, with_epidish: bool = True,
            with_paper: bool = False) -> dict[str, pd.DataFrame]:
    """All baselines on one reference/bulk pair, as samples x types tables."""
    r, b = ref.to_numpy(dtype=float), bulk.to_numpy(dtype=float)
    wrap = lambda a: pd.DataFrame(a, index=bulk.columns, columns=ref.columns)  # noqa: E731
    raw = nnls(r, b)
    icpt = nnls_intercept(r, b)
    out = {
        "NNLS": wrap(raw),
        "NNLS_norm": wrap(raw / np.clip(raw.sum(axis=1, keepdims=True), 1e-12, None)),
        "NNLS_icpt_norm": wrap(icpt / np.clip(icpt.sum(axis=1, keepdims=True), 1e-12, None)),
        "CP_ineq": wrap(cp_inequality(r, b)),
        "CP_eq": wrap(cp_equality(r, b)),
    }
    if with_epidish:
        out["EpiDISH_RPC"] = epidish(ref, bulk, "RPC")
        out["EpiDISH_CBS"] = epidish(ref, bulk, "CBS")
        out["EpiDISH_CP"] = epidish(ref, bulk, "CP", "inequality")
    if with_paper:
        out.update(paper_solvers(ref, bulk))
    return out
