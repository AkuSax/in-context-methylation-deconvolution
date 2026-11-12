"""Checks for the chunked Noob run in reprocess.py (R is replaced by a stub)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deconv"))

import datasets as ds  # noqa: E402
import reprocess as rp  # noqa: E402

PROBES = ["cg1", "cg2", "cg3"]


def _fake_noob(seen):
    """Stand-in for noob_idats.R: one beta column per array, filled with its GSM number."""
    def run(gse, src, prefix, detp_max):
        names = sorted(p.name.replace("_Grn.idat", "") for p in src.glob("*_Grn.idat"))
        assert all((src / f"{n}_Red.idat").exists() for n in names)
        seen.append(names)
        beta = np.array([[float(n[3:].split("_")[0])] * len(PROBES) for n in names], dtype=np.float32)
        beta.tofile(f"{prefix}.f32.bin")  # rows are arrays here, i.e. column-major probes x arrays
        Path(f"{prefix}.probes.txt").write_text("\n".join(PROBES) + "\n")
        Path(f"{prefix}.samples.txt").write_text("\n".join(names) + "\n")
        pd.DataFrame(dict(sample=names, array="EPIC", frac_detp_fail=0.0)).to_csv(f"{prefix}.qc.csv", index=False)
    return run


@pytest.fixture
def series(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "RAW", tmp_path / "raw")
    monkeypatch.setattr(ds, "CACHE", tmp_path / "cache")
    idat = tmp_path / "raw" / "GSE1" / "idat"
    idat.mkdir(parents=True)
    for i in range(1, 6):
        for colour in ("Grn", "Red"):
            (idat / f"GSM{i}_9_R0{i}C01_{colour}.idat").write_bytes(b"x")
    seen = []
    monkeypatch.setattr(rp, "_run_noob", _fake_noob(seen))
    return tmp_path, seen


def test_chunked_noob_joins_every_array_once(series):
    tmp_path, seen = series
    rp.noob("GSE1", 0.01, chunk=2)
    assert [len(s) for s in seen] == [2, 2, 1]
    got = ds._noob_betas("GSE1", pd.DataFrame(index=[f"GSM{i}" for i in (3, 1, 5, 2, 4)]))
    assert list(got.index) == PROBES
    assert all(np.allclose(got[f"GSM{i}"], i) for i in range(1, 6))
    assert len(pd.read_csv(tmp_path / "cache" / "GSE1_noob.qc.csv")) == 5
    assert not list((tmp_path / "cache").glob("*.part*"))
    assert not list((tmp_path / "raw" / "GSE1").glob("idat_chunk*"))


def test_small_series_runs_in_one_session(series):
    _, seen = series
    rp.noob("GSE1", 0.01, chunk=80)
    assert [len(s) for s in seen] == [5]
