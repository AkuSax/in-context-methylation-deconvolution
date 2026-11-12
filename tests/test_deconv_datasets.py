"""Checks for reading the betas that noob_idats.R writes."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deconv"))

import datasets as ds  # noqa: E402


def _write(prefix: Path, beta: np.ndarray, probes, samples):
    # Same layout as the R script: one float32 column after another.
    beta.astype(np.float32).T.tofile(f"{prefix}.f32.bin")
    Path(f"{prefix}.probes.txt").write_text("\n".join(probes) + "\n")
    Path(f"{prefix}.samples.txt").write_text("\n".join(samples) + "\n")


def test_noob_betas_follow_series_order_and_keep_missing_values(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "CACHE", tmp_path)
    beta = np.array([[0.1, 0.4], [0.2, 0.5], [np.nan, 0.6]])
    _write(tmp_path / "GSE1_noob", beta, ["cg1", "cg2", "cg3"], ["GSM2_9_R02C01", "GSM1_9_R01C01"])
    got = ds._noob_betas("GSE1", pd.DataFrame(index=["GSM1", "GSM2"]))
    assert list(got.columns) == ["GSM1", "GSM2"] and list(got.index) == ["cg1", "cg2", "cg3"]
    assert np.allclose(got["GSM1"], [0.4, 0.5, 0.6])
    assert np.allclose(got["GSM2"][:2], [0.1, 0.2]) and np.isnan(got["GSM2"].iloc[2])


def test_noob_betas_reject_a_sample_without_an_idat(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "CACHE", tmp_path)
    _write(tmp_path / "GSE1_noob", np.zeros((2, 1)), ["cg1", "cg2"], ["GSM1_9_R01C01"])
    with pytest.raises(ValueError, match="GSM7"):
        ds._noob_betas("GSE1", pd.DataFrame(index=["GSM1", "GSM7"]))
