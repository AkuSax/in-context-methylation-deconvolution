import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deconv"))
import datasets as ds
import run_baselines as run
import metrics


def test_strict_panel_removes_dropped_samples_before_probe_filter(monkeypatch):
    betas = pd.DataFrame([[0.1, 0.9, np.nan], [0.2, 0.8, 0.5]],
                         index=["cg1", "cg2"], columns=["a", "b", "d"])
    def dataset(b, labels):
        ids = b.columns
        return ds.Dataset("x", "EPIC", b, pd.Series(labels, index=ids),
                          pd.DataFrame(index=ids), pd.Series(None, index=ids), pd.DataFrame(index=ids))
    src = dataset(betas, ["A", "B", "DROP"])
    target = dataset(pd.DataFrame([[0.3], [0.4]], index=betas.index, columns=["m"]), [None])
    def pick(b, labels, n):
        assert list(b.columns) == ["a", "b"]
        assert list(b.index) == ["cg1", "cg2"]
        assert "DROP" not in labels.values
        return b.index
    monkeypatch.setattr(run.reference, "pick_markers", pick)
    panel, n, kept = run.strict_panel(src, target, "DROP")
    assert n == 2 and list(panel.columns) == ["A", "B"]
    assert list(kept.betas.columns) == ["a", "b"]


def test_bootstrap_pairs_methods_and_repeated_drops():
    rows = []
    for drop in ["A", "B"]:
        for i in range(4):
            for method in ["EpiDISH_RPC", "copy"]:
                rows.append(dict(dropped=drop, sample=str(i), method=method,
                                 kept_squared_error_mean=(i + 1) / 100,
                                 true_omitted=0.02 if i == 0 else 0.2, estimated_omitted=i / 10))
    result = metrics.strict_bootstrap(pd.DataFrame(rows), n_boot=100)
    delta = result.query("method == 'copy' and metric == 'rmse_kept_difference_vs_rpc_pts'")
    assert len(delta) == 4
    assert np.allclose(delta[["estimate_pts", "ci_low_pts", "ci_high_pts"]], 0)
    assert result.query("dropped == 'ALL_DROPS'").n_mixtures.eq(4).all()


    abundant = result.query("dropped == 'ABUNDANT_DROPS_GE_5_PTS'")
    assert abundant.n_sample_drops.eq(6).all()
