"""v5 and both v6 checkpoints on the four unlabelled whole-blood cohorts.

Same reference, probes and markers as label_free_null.py (GSE110554 sorted, Noob, 25 per
side; cohorts as deposited). These cohorts have no known `other` truth, so this is a
reference-compatibility stress test, not a null. Each cohort is loaded and fitted once;
every configuration then runs on the same RPC fit.

Cohort-model contexts: `all` (the whole series is one cohort), `groups12` (random groups
of 12, seed 0) and `single` (every sample alone).

Design was fixed before the run.
Run from the deconv folder:  py -3.11 -B null_v6.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import label_free_null as lf
from label_free_null import ev, ds, reference
from model import Deconvolver
from rpc_torch import rpc_fit

OUT = Path("results/real_check_gpu_v6")
CONFIGS = [  # name, checkpoint, cohort context
    ("v5", "results/gpu_v5_seed0", None),
    ("v6_nocohort", "results/gpu_v6_nocohort_seed0", None),
    ("v6_cohort_all", "results/gpu_v6_seed0", "all"),
    ("v6_cohort_groups12", "results/gpu_v6_seed0", "groups12"),
    ("v6_cohort_single", "results/gpu_v6_seed0", "single"),
]
CHUNK = 64  # tasks per forward call


def load_model(path):
    s = json.loads((Path(path) / "summary.json").read_text())["settings"]
    model = Deconvolver(width=s["width"], heads=s["heads"], blocks=s["blocks"], cohort=s.get("cohort", False))
    model.load_state_dict(torch.load(Path(path) / "model.pt", map_location="cpu"))
    return model.eval()


def cohort_ids(mode, n):
    if mode == "all":
        return torch.zeros(n, dtype=torch.long)
    if mode == "single":
        return torch.arange(n)
    order = np.random.default_rng(0).permutation(n)
    ids = np.empty(n, dtype=np.int64)
    ids[order] = np.arange(n) // 12
    return torch.as_tensor(ids)


def take(fit, rows):
    """Rows of an RPCFit (every per-task tensor field)."""
    return type(fit)(**{k: (v[rows] if torch.is_tensor(v) and v.shape[:1] == (fit.fractions.shape[0],) else v)
                        for k, v in vars(fit).items()})


def predict(model, R, S, B, fit, ids):
    """Mean fractions and P(other > 0). Groups are never split across forward calls."""
    n = len(B)
    kept = np.zeros((n, R.shape[-1] + 1))
    p_pos = np.zeros(n)
    if ids is None:
        batches = [np.arange(i, min(i + CHUNK, n)) for i in range(0, n, CHUNK)]
    else:
        groups = [np.flatnonzero(ids.numpy() == g) for g in np.unique(ids.numpy())]
        batches, cur = [], []
        for g in groups:
            if cur and sum(map(len, cur)) + len(g) > CHUNK:
                batches.append(np.concatenate(cur)); cur = []
            cur.append(g)
        batches.append(np.concatenate(cur))
    with torch.inference_mode():
        for rows in batches:
            r = torch.as_tensor(rows)
            p = model(R[r], S[r], B[r], take(fit, r), None if ids is None else ids[r])
            kept[rows] = p.mean.numpy()
            p_pos[rows] = p.has_other.probs.numpy()
    return kept, p_pos


def main():
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    models = {path: load_model(path) for _, path, _ in CONFIGS}
    src = ds.load("GSE110554").sorted
    rows, summary = [], []
    for name, loader in lf.LOADERS.items():
        try:
            print(f"[{name}] loading", flush=True)
            df, _, _ = loader()
            shared = reference.usable_probes(src.betas).intersection(pd.Index(df.index))
            sub = df.loc[shared]
            sub = sub[sub.notna().all(axis=1)].astype(np.float64)
            mean, sd = ev.panel(src, SimpleNamespace(betas=sub), lf.PER_SIDE, None)
            bulk = sub.loc[mean.index]
            n = bulk.shape[1]
            as_t = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
            R = as_t(np.repeat(mean.to_numpy(float)[None], n, axis=0))
            S = as_t(np.repeat(sd.to_numpy(float)[None], n, axis=0))
            B = as_t(bulk.to_numpy(float).T)
            fit = rpc_fit(np.repeat(mean.to_numpy(float)[None], n, axis=0), bulk.to_numpy(float).T)
            rpc = fit.fractions.numpy()
            raw = np.clip(1 - fit.coefficients.clamp_min(0).sum(-1).numpy(), 0, 1)
            print(f"[{name}] {n} samples, {len(mean)} markers, RPC unexplained {100 * raw.mean():.1f}", flush=True)
            for config, path, mode in CONFIGS:
                ids = cohort_ids(mode, n) if mode else None
                est, p_pos = predict(models[path], R, S, B, fit, ids)
                other = est[:, -1]
                renorm = est[:, :-1] / est[:, :-1].sum(1, keepdims=True)
                summary.append(dict(
                    cohort=name, config=config, n=n, n_markers=len(mean),
                    other_mean_pts=100 * other.mean(), other_median_pts=100 * np.median(other),
                    other_sd_pts=100 * other.std(), frac_other_gt5=float((other > lf.THRESH).mean()),
                    mean_p_other_gt0=float(p_pos.mean()),
                    kept_vs_rpc_rms_pts=100 * float(np.sqrt(np.mean((renorm - rpc) ** 2))),
                    rpc_unexplained_pts=100 * raw.mean()))
                print(summary[-1], flush=True)
                for i, sample in enumerate(bulk.columns):
                    rec = dict(cohort=name, config=config, sample_id=sample, other=float(other[i]),
                               p_other_gt0=float(p_pos[i]), rpc_unexplained=float(raw[i]))
                    rec.update({f"est_{t}": float(est[i, j]) for j, t in enumerate(mean.columns)})
                    rows.append(rec)
        except Exception as exc:
            traceback.print_exc()
            summary.append(dict(cohort=name, config="FAILED", reason=f"{type(exc).__name__}: {exc}"))
        pd.DataFrame(summary).to_csv(OUT / "wholeblood_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "wholeblood_estimates.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summary).round(2).to_string(), flush=True)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
