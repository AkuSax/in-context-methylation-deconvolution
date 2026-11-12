"""Read GEO series-matrix files: per-sample metadata plus the beta table.

GEO pads `!Sample_characteristics_ch1` rows unevenly when a sample lacks a
field, so a row is not a variable. Each sample's `key: value` strings are
therefore collected into its own dict.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SeriesMeta:
    gsm: list[str]
    title: list[str]
    source: list[str]
    chars: list[dict[str, str]]  # one dict per sample
    table_start: int | None  # 0-based line index of the ID_REF header row

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.chars, index=self.gsm)
        df.insert(0, "title", self.title)
        df.insert(1, "source", self.source)
        return df


def _cells(line: str) -> list[str]:
    return [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]


def read_meta(path: str | Path) -> SeriesMeta:
    gsm: list[str] = []
    title: list[str] = []
    source: list[str] = []
    char_rows: list[list[str]] = []
    table_start = None
    with gzip.open(path, "rt", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("!Sample_geo_accession"):
                gsm = _cells(line)
            elif line.startswith("!Sample_title"):
                title = _cells(line)
            elif line.startswith("!Sample_source_name_ch1"):
                source = _cells(line)
            elif line.startswith("!Sample_characteristics_ch1"):
                char_rows.append(_cells(line))
            elif line.startswith("!series_matrix_table_begin"):
                table_start = i + 1
                break
    n = len(gsm)
    chars: list[dict[str, str]] = [{} for _ in range(n)]
    for row in char_rows:
        for j, cell in enumerate(row[:n]):
            if ":" not in cell:
                continue
            k, v = cell.split(":", 1)
            chars[j][k.strip().lower()] = v.strip()
    if not source:
        source = [""] * n
    return SeriesMeta(gsm, title, source, chars, table_start)


def read_betas(path: str | Path, meta: SeriesMeta | None = None) -> pd.DataFrame:
    """Probe x sample table from a series matrix, float32, columns = GSM ids."""
    meta = meta or read_meta(path)
    if meta.table_start is None:
        raise ValueError(f"{path}: no data table")
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=meta.table_start,
        index_col=0,
        comment="!",
        na_values=["", "NA", "null", "NaN"],
        dtype={"ID_REF": str},
        low_memory=False,
    )
    df.index = df.index.astype(str)
    df.index.name = "probe"
    return df.astype(np.float32)


def cached_betas(path: str | Path, cache_dir: str | Path) -> pd.DataFrame:
    """read_betas with an on-disk cache (values .npy + probe/sample id lists)."""
    path, cache_dir = Path(path), Path(cache_dir)
    stem = path.name.replace(".txt.gz", "")
    vals, rows, cols = (cache_dir / f"{stem}.{s}" for s in ("f32.npy", "probes.txt", "samples.txt"))
    if vals.exists() and rows.exists() and cols.exists():
        return pd.DataFrame(
            np.load(vals),
            index=pd.Index(rows.read_text().split("\n"), name="probe"),
            columns=cols.read_text().split("\n"),
        )
    df = read_betas(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(vals, df.to_numpy(dtype=np.float32))
    rows.write_text("\n".join(df.index))
    cols.write_text("\n".join(df.columns))
    return df
