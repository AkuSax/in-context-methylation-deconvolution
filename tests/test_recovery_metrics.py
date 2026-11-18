"""Unit checks for pipeline/recovery_metrics.py (plain python, no pytest).

Run: python tests/test_recovery_metrics.py
No data files are read. No writes. Exits nonzero on failure.
Bytecode writing for the project module is disabled before the import so no
__pycache__ appears anywhere outside SWARM_OUTPUT_DIR (static-preflight
finding 3; same rule as the pilot script).
"""

import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import recovery_metrics as rmet  # noqa: E402


def main():
    rng = np.random.default_rng(7)
    d, p, c = 30, 6, 3
    zt = rng.uniform(0.1, 0.9, size=(d, p, c))
    zh = zt + rng.normal(0, 0.02, size=(d, p, c))

    # rmse/bias/r on known values
    err = zh - zt
    assert abs(rmet.rmse(zt, zh) - float(np.sqrt((err ** 2).mean()))) < 1e-12
    assert abs(rmet.bias(zt, zh) - float(err.mean())) < 1e-12

    rows = rmet.per_cpg_cell_metrics(zt, zh)
    assert len(rows) == p * c
    assert all(r["n_donors"] == d for r in rows)
    assert all(r["r_reason"] == "" for r in rows)
    assert all(-1.0 <= r["pearson_r"] <= 1.0 for r in rows)

    # zero-variance truth -> explicit reason, r NaN, rmse/bias still defined
    zt0 = np.broadcast_to(zt[:1, :, :], (d, p, c)).copy()
    rows0 = rmet.per_cpg_cell_metrics(zt0, zh)
    assert all(r["r_reason"] == "truth_zero_variance" and np.isnan(r["pearson_r"])
               for r in rows0)
    assert all(np.isfinite(r["rmse"]) and np.isfinite(r["bias"]) for r in rows0)

    # donor mask is respected
    mask = np.zeros(d, dtype=bool)
    mask[:10] = True
    rows1 = rmet.per_cpg_cell_metrics(zt, zh, donor_mask=mask)
    assert all(r["n_donors"] == 10 for r in rows1)

    # cell summary counts undefined r
    summ = rmet.summarize_by_cell(rows0, c, ["a", "b", "c"])
    assert all(s["n_undefined_r"] == p for s in summ)
    assert len(summ) == c

    oor = rmet.out_of_range_count(zh, donor_mask=mask)
    assert oor["n_scored"] == 10 * p * c

    w = rng.dirichlet(np.ones(c), size=d)
    rr = rmet.reconstruction_residuals(zh, w, np.einsum("dc,dpc->dp", w, zh))
    assert rr["max"] < 1e-12 and len(rr["per_donor_max"]) == d

    # shape guards raise
    for fn, args in ((rmet.rmse, (zt, zh[:, :, 0])),
                     (rmet.out_of_range_count, (zh[:, :, 0],)),
                     (rmet.reconstruction_residuals, (zh, w[:, :2], zh[:, :, 0]))):
        try:
            fn(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} should have raised ValueError")

    print("TEST PASS")


if __name__ == "__main__":
    raise SystemExit(main())
