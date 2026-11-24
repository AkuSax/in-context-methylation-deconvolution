"""Download GEO series matrices and write one tidy label table per series.

    python fetch_labels.py GSE174666 GSE130029 ...

Writes <RAW>/<GSE>/labels.tsv with one row per GSM: gsm, title, source_name, every
characteristics_ch1 field split into its own column ("key: value"), and idat_prefix (the
GSM_sentrix_position basename that reprocess.py unpacks, when the supplementary-file
column names one). Read-only with respect to the models: nothing here assigns a role.
"""

from __future__ import annotations

import gzip
import io
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

import datasets as ds


def matrix_urls(gse: str) -> list[str]:
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse[:-3]}nnn/{gse}/matrix/"
    req = urllib.request.Request(base, headers={"User-Agent": "python-urllib"})
    html = urllib.request.urlopen(req).read().decode()
    names = sorted(set(re.findall(r'href="(GSE\d+(?:-GPL\d+)?_series_matrix\.txt\.gz)"', html)))
    return [base + n for n in names]


def parse_matrix(text: str) -> pd.DataFrame:
    rows: dict[str, list[list[str]]] = {}
    for line in text.splitlines():
        if line.startswith("!series_matrix_table_begin"):
            break
        if not line.startswith("!Sample_"):
            continue
        key, *vals = [v.strip().strip('"') for v in line.rstrip("\n").split("\t")]
        rows.setdefault(key[len("!Sample_"):], []).append(vals)
    n = len(rows["geo_accession"][0])
    out = pd.DataFrame({"gsm": rows["geo_accession"][0]})
    out["title"] = rows.get("title", [[""] * n])[0]
    out["source_name"] = rows.get("source_name_ch1", [[""] * n])[0]
    for fields in rows.get("characteristics_ch1", []):
        for i, v in enumerate(fields):
            key, _, val = v.partition(": ")
            if not val:
                key, val = "characteristic", v
            out.loc[i, key.strip().lower()] = val.strip()
    sup = rows.get("supplementary_file", [])
    if sup:
        files = [v for v in sup[0]]
        out["idat_prefix"] = [re.sub(r"_(Grn|Red)\.idat(\.gz)?$", "", f.rsplit("/", 1)[-1])
                              if "idat" in f.lower() else "" for f in files]
    return out


def main() -> None:
    for gse in sys.argv[1:]:
        frames = []
        for url in matrix_urls(gse):
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            raw = urllib.request.urlopen(req).read()
            text = gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode("utf-8", "replace")
            df = parse_matrix(text)
            df["matrix_file"] = url.rsplit("/", 1)[-1]
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        dest = ds.RAW / gse
        dest.mkdir(parents=True, exist_ok=True)
        out.to_csv(dest / "labels.tsv", sep="\t", index=False)
        print(f"{gse}: {len(out)} samples, columns {list(out.columns)}", flush=True)


if __name__ == "__main__":
    main()
