"""Reprocess every series from raw IDATs with one recipe (see noob_idats.R).

    python reprocess.py fetch [GSE ...]   download GSE*_RAW.tar from GEO, unpack the IDATs
    python reprocess.py noob  [GSE ...]   run noob_idats.R, write <cache>/<GSE>_noob.*
    python reprocess.py all   [GSE ...]   both

`fetch` records each archive's size and SHA-256 in data_manifest.json. The
loaders in datasets.py read the result when DECONV_PREPROC=noob.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

import datasets as ds
import solvers

MANIFEST = Path(__file__).with_name("data_manifest.json")
R_HELPER = Path(__file__).with_name("noob_idats.R")
IDAT_NAME = re.compile(r"^GSM\d+_\w+_(Grn|Red)\.idat(\.gz)?$")


def raw_tar_url(gse: str) -> str:
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse[:-3]}nnn/{gse}/suppl/{gse}_RAW.tar"


def idat_dir(gse: str) -> Path:
    return ds.RAW / gse / "idat"


def _record(entry: dict) -> None:
    rows = json.loads(MANIFEST.read_text())
    rows = [r for r in rows if (r["series"], r["file"]) != (entry["series"], entry["file"])]
    rows.append(entry)
    rows.sort(key=lambda r: r["series"])  # stable: order within a series is kept
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n")


def fetch(gse: str) -> None:
    url = raw_tar_url(gse)
    tar_path = ds.RAW / gse / f"{gse}_RAW.tar"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    if not tar_path.exists():
        part = tar_path.with_name(tar_path.name + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req) as resp, open(part, "wb") as f:
            expected = int(resp.headers.get("Content-Length", 0))
            while chunk := resp.read(1 << 22):
                f.write(chunk)
        if expected and part.stat().st_size != expected:
            raise IOError(f"{gse}: got {part.stat().st_size} bytes, expected {expected}")
        part.rename(tar_path)
    digest = hashlib.sha256()
    with open(tar_path, "rb") as f:
        while chunk := f.read(1 << 22):
            digest.update(chunk)
    _record({"series": gse, "file": tar_path.name, "url": url, "bytes": tar_path.stat().st_size,
             "sha256": digest.hexdigest(), "downloaded": datetime.date.today().isoformat()})

    # Unpack only IDATs, by basename, so nothing in the archive can choose its own path.
    out = idat_dir(gse)
    out.mkdir(exist_ok=True)
    skipped = []
    with tarfile.open(tar_path) as tar:
        for member in tar:
            name = member.name.rsplit("/", 1)[-1]
            if not (member.isfile() and IDAT_NAME.match(name)):
                skipped.append(member.name)
                continue
            dest = out / name
            if not dest.exists() or dest.stat().st_size != member.size:
                with tar.extractfile(member) as src, open(dest, "wb") as dst:
                    while chunk := src.read(1 << 22):
                        dst.write(chunk)
    n = len(list(out.glob("*_Grn.idat*")))
    print(f"{gse}: {tar_path.stat().st_size / 1e6:.0f} MB, {n} arrays unpacked"
          + (f", {len(skipped)} non-IDAT members skipped" if skipped else ""), flush=True)


def _run_noob(gse: str, src: Path, prefix: Path, detp_max: float) -> None:
    done = subprocess.run([solvers.RSCRIPT, "--vanilla", str(R_HELPER), src.as_posix(),
                           prefix.as_posix(), str(detp_max)], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{gse}: noob_idats.R failed\n{done.stderr[-3000:]}")
    print(f"{gse}: {done.stdout.strip().splitlines()[-1]}", flush=True)


def _noob_chunked(gse: str, grn: list[Path], prefix: Path, detp_max: float, chunk: int) -> None:
    """Run noob_idats.R on `chunk` arrays at a time and join the outputs.

    The recipe treats every array on its own, so the betas do not depend on the
    chunking; this only bounds R's memory (all 145 arrays of GSE103541 at once do
    not fit in 16 GB). Chunk folders hold hard links, not copies.
    """
    parts = []
    for i in range(0, len(grn), chunk):
        part_dir = ds.RAW / gse / f"idat_chunk{i // chunk:02d}"
        if part_dir.exists():
            shutil.rmtree(part_dir)
        part_dir.mkdir()
        for g in grn[i:i + chunk]:
            for f in (g, g.with_name(g.name.replace("_Grn.idat", "_Red.idat"))):
                os.link(f, part_dir / f.name)
        part = prefix.with_name(f"{prefix.name}.part{i // chunk:02d}")
        _run_noob(gse, part_dir, part, detp_max)
        shutil.rmtree(part_dir)
        parts.append(part)
    probes = Path(f"{parts[0]}.probes.txt").read_text()
    if any(Path(f"{p}.probes.txt").read_text() != probes for p in parts[1:]):
        raise RuntimeError(f"{gse}: chunks disagree on the probe list")
    with open(f"{prefix}.f32.bin", "wb") as out:  # column-major, so columns concatenate
        for p in parts:
            with open(f"{p}.f32.bin", "rb") as src:
                shutil.copyfileobj(src, out, 1 << 24)
    Path(f"{prefix}.probes.txt").write_text(probes)
    Path(f"{prefix}.samples.txt").write_text("".join(Path(f"{p}.samples.txt").read_text() for p in parts))
    pd.concat([pd.read_csv(f"{p}.qc.csv") for p in parts]).to_csv(f"{prefix}.qc.csv", index=False)
    for p in parts:
        for suffix in ("f32.bin", "probes.txt", "samples.txt", "qc.csv"):
            Path(f"{p}.{suffix}").unlink()


def noob(gse: str, detp_max: float, chunk: int = 80) -> None:
    ds.CACHE.mkdir(parents=True, exist_ok=True)
    prefix = ds.CACHE / f"{gse}_noob"
    grn = sorted(idat_dir(gse).glob("*_Grn.idat*"))
    if len(grn) > chunk:
        _noob_chunked(gse, grn, prefix, detp_max, chunk)
    else:
        _run_noob(gse, idat_dir(gse), prefix, detp_max)
    qc = pd.read_csv(f"{prefix}.qc.csv")
    print(f"{gse}: {qc['array'].iloc[0]}, detection-p failures per sample "
          f"median {100 * qc['frac_detp_fail'].median():.2f}% max {100 * qc['frac_detp_fail'].max():.2f}%",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "noob", "all"])
    ap.add_argument("series", nargs="*", default=list(ds.LOADERS))
    ap.add_argument("--detp-max", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=80,
                    help="arrays per R session; larger series are processed in chunks and joined")
    args = ap.parse_args()
    for gse in args.series:
        if args.step in ("fetch", "all"):
            fetch(gse)
        if args.step in ("noob", "all"):
            noob(gse, args.detp_max, args.chunk)


if __name__ == "__main__":
    main()
