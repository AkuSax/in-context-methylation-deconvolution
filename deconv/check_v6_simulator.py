"""Realism check of the v6 simulator on held-out donors, with the frozen v5 checkpoint.

For each source, clean five-type cohorts of 12 samples with `pick` markers (250 = 25 per
side), with the synthetic technical shift off (measured study difference only) and on
(the training prior). Reports what RPC and v5 see, to set against real data:

  real, in batch (GSE110554 clean mixtures):   RPC kept RMSE 1.72, RPC unexplained mass 0.7,
                                               v5 `other` 6.4, median marker range 0.88
  real, out of batch (four whole-blood cohorts): RPC unexplained mass 9.5-16.6, v5 `other` 8-19

Also the drop-one case (one of the five types removed from the reference).
Uses no reserved data. Writes results/check_v6_simulator/summary.csv.
Run from the deconv folder:  py -3.11 -B check_v6_simulator.py [panels.npz]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import simulator
import train_cpu as tr
from model import Deconvolver

CKPT = Path("results/gpu_v5_seed0")
OUT = Path("results/check_v6_simulator")
SOURCES = {
    "GSE35069 alone (v5 source, sharpened)": ("gse35069", dict(), dict(sharpen=tr.SHARPEN)),
    "GSE103541 alone": ("joint", dict(mixture_group="GSE103541"), {}),
    "ref GSE35069 -> bulk GSE103541": ("joint", dict(mixture_group="GSE103541", cross_study=True), {}),
    "ref GSE35069 sharpened -> bulk GSE103541": ("joint", dict(mixture_group="GSE103541", cross_study=True),
                                                 dict(sharpen=tr.SHARPEN, sharpen_groups=("GSE35069",))),
    "ref GSE103541 -> bulk GSE35069": ("joint", dict(mixture_group="GSE35069", cross_study=True), {}),
}
N_COHORTS, SIZE, MARKERS, TYPES = 12, 12, 250, 5


def main():
    torch.set_num_threads(8)
    bundle = simulator.load_panels(sys.argv[1]) if len(sys.argv) > 1 else simulator.build_v6_panels(seed=0)
    panels, pools = bundle["validation"], bundle["pools"]
    s = json.loads((CKPT / "summary.json").read_text())["settings"]
    v5 = Deconvolver(width=s["width"], heads=s["heads"], blocks=s["blocks"])
    v5.load_state_dict(torch.load(CKPT / "model.pt", map_location="cpu"))
    v5.eval()
    rows = []
    for label, (panel, extra, sharpen) in SOURCES.items():
        for shift in (False, True):
            for missing in (0, 1):
                rng = np.random.default_rng([20261112, missing])
                cohorts = [simulator.simulate_cohort(
                    panels[panel], rng, SIZE, n_markers=MARKERS, n_types=TYPES - missing, missing=missing,
                    donor_mismatch=True, marker_mode="pick", candidate_pool=pools[panel],
                    tech_shift=shift, **tr.TECH_SHIFT_RANGES, **extra, **sharpen) for _ in range(N_COHORTS)]
                batch = tr.collate(cohorts)
                fit, truth = batch["fit"], batch["truth"].numpy().astype(float)
                ref = batch["reference"].numpy()
                rpc = fit.fractions.numpy()
                raw_other = np.clip(1 - fit.coefficients.clamp_min(0).sum(-1).numpy(), 0, 1)
                with torch.inference_mode():
                    est = v5(batch["reference"], batch["sd"], batch["bulk"], fit).mean.numpy().astype(float)
                rmse = lambda e: 100 * float(np.sqrt(np.mean(e ** 2)))  # noqa: E731
                rows.append(dict(
                    source=label, synthetic_shift=shift, missing=missing, n=len(truth),
                    marker_range_median=float(np.median(ref.max(-1) - ref.min(-1))),
                    rpc_kept_rmse=rmse(rpc - truth[:, :-1]), v5_kept_rmse=rmse(est[:, :-1] - truth[:, :-1]),
                    true_other=100 * truth[:, -1].mean(), v5_other=100 * est[:, -1].mean(),
                    v5_other_sd=100 * est[:, -1].std(), rpc_unexplained=100 * raw_other.mean(),
                    rpc_intercept=float(fit.intercept.mean()), rpc_scale=float(fit.scale.mean())))
                print(rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
