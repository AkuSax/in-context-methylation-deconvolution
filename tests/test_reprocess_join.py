"""Regression tests for reprocess._noob_chunked's join of per-chunk outputs.

noob_idats.R writes float32, probes x samples, column-major (so one sample's probes are
contiguous). Betas here are a pure function of (probe id, sample), so a join that puts one
probe's values under another probe's name is detectable. Run: python -m pytest tests/test_reprocess_join.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deconv"))
import datasets as ds  # noqa: E402
import reprocess  # noqa: E402

N_PER_CHUNK = 2


def beta(probe: str, sample: int) -> float:
    v = (int(probe[2:]) * 37 + sample * 11) % 997 / 997.0
    return float("nan") if (int(probe[2:]) + sample) % 13 == 0 else float(np.float32(v))


def run_join(tmp_path, monkeypatch, chunk_lists, source=None):
    mod = source or reprocess
    monkeypatch.setattr(ds, "RAW", tmp_path / "raw")
    monkeypatch.setattr(ds, "CACHE", tmp_path / "cache")
    idat = ds.RAW / "GSEFAKE" / "idat"
    idat.mkdir(parents=True)
    ds.CACHE.mkdir(parents=True)
    n = N_PER_CHUNK * len(chunk_lists)
    grn = []
    for s in range(n):
        g = idat / f"GSM{s:03d}_x_Grn.idat"
        g.write_bytes(b"g")
        (idat / f"GSM{s:03d}_x_Red.idat").write_bytes(b"r")
        grn.append(g)
    calls = iter(range(len(chunk_lists)))

    def fake_run(gse, src, prefix, detp_max):
        c = next(calls)
        names = sorted(p.name for p in Path(src).glob("*_Grn.idat"))
        samples = [int(x[3:6]) for x in names]
        probes = chunk_lists[c]
        cols = np.array([[beta(p, s) for p in probes] for s in samples], dtype=np.float32)
        cols.tofile(f"{prefix}.f32.bin")  # (samples, probes) C order == probes x samples column-major
        Path(f"{prefix}.probes.txt").write_text("\n".join(probes) + "\n")
        Path(f"{prefix}.samples.txt").write_text("".join(f"GSM{s:03d}_x\n" for s in samples))
        pd.DataFrame({"array": [f"GSM{s:03d}_x" for s in samples], "frac_detp_fail": 0.0}).to_csv(
            f"{prefix}.qc.csv", index=False)

    monkeypatch.setattr(mod, "_run_noob", fake_run)
    prefix = ds.CACHE / "GSEFAKE_noob"
    mod._noob_chunked("GSEFAKE", grn, prefix, 0.01, N_PER_CHUNK)
    probes = Path(f"{prefix}.probes.txt").read_text().split()
    samples = Path(f"{prefix}.samples.txt").read_text().split()
    mat = np.fromfile(f"{prefix}.f32.bin", dtype=np.float32).reshape((len(probes), len(samples)), order="F")
    return probes, samples, mat, prefix


def check(probes, samples, mat):
    assert len(samples) == N_PER_CHUNK * 3 and mat.shape == (len(probes), len(samples))
    assert len(set(probes)) == len(probes)
    for j, s in enumerate(samples):
        sid = int(s[3:6])
        want = np.array([beta(p, sid) for p in probes], dtype=np.float32)
        np.testing.assert_array_equal(np.isnan(mat[:, j]), np.isnan(want))
        np.testing.assert_array_equal(np.nan_to_num(mat[:, j], nan=-1), np.nan_to_num(want, nan=-1))


P = [f"cg{i:04d}" for i in range(1, 41)]


def test_identical_lists_fast_path(tmp_path, monkeypatch):
    probes, samples, mat, prefix = run_join(tmp_path, monkeypatch, [P, P, P])
    assert probes == P and not Path(f"{prefix}.dropped_probes.txt").exists()
    check(probes, samples, mat)


def test_subset_lists_same_order(tmp_path, monkeypatch):
    lists = [P, [p for p in P if p not in ("cg0003", "cg0010")], [p for p in P if p != "cg0020"]]
    probes, samples, mat, prefix = run_join(tmp_path, monkeypatch, lists)
    assert probes == [p for p in P if p not in ("cg0003", "cg0010", "cg0020")]
    assert Path(f"{prefix}.dropped_probes.txt").read_text().split() == ["cg0003", "cg0010", "cg0020"]
    check(probes, samples, mat)


def test_permuted_same_set(tmp_path, monkeypatch):
    lists = [P, P[::-1], P[7:] + P[:7]]
    probes, samples, mat, prefix = run_join(tmp_path, monkeypatch, lists)
    assert probes == P and Path(f"{prefix}.dropped_probes.txt").read_text().strip() == ""
    check(probes, samples, mat)


def test_permuted_and_different_sets(tmp_path, monkeypatch):
    lists = [P, P[::-1][3:], P[11:] + P[:11]]
    lists[2] = [p for p in lists[2] if p != "cg0005"]
    probes, samples, mat, prefix = run_join(tmp_path, monkeypatch, lists)
    keep = set(lists[0]) & set(lists[1]) & set(lists[2])
    assert probes == [p for p in P if p in keep]  # chunk 0's order
    assert len(probes) < len(P)
    check(probes, samples, mat)


def test_duplicate_probe_ids_rejected(tmp_path, monkeypatch):
    lists = [P, P[:-1] + [P[0]], P]
    with pytest.raises(ValueError):
        run_join(tmp_path, monkeypatch, lists)
