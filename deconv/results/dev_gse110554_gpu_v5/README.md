# GSE110554 development check of gpu_v5: result (fourth development look)

Protocol: [PROTOCOL.md](PROTOCOL.md), frozen before scoring. Script: `deconv/eval_real_dev.py`.
Files: `summary.csv`, `estimates_long.csv`, `run.log`. Model SHA-256 `a8c9a9fe…a6342dab3`, best
step 30,000.

> **Read with the later audit:** [../audit_gse110554_gpu_v5/README.md](../audit_gse110554_gpu_v5/README.md).
> The 63% figure is driven by the neutrophil drop (30% without it), "all six drops" holds at
> 25 per side only, and `other` does not track T or NK drops. The text below is unchanged.

## Verdict: missing-type error is 63% below RPC; clean is close but not at parity

| Primary panel (25 per side, 286 CpGs) | v5 | v4 | v3 | RPC | CP |
|---|---|---|---|---|---|
| Clean kept RMSE | **2.84** | 5.87 | 9.54 | 1.72 | 1.55 |
| Clean `other` (truth 0) | **6.4** | 18.9 | 38.6 | 0 | 0 |
| Drop-one kept RMSE (72 pairs) | **4.00** | 5.26 | 8.43 | 10.74 | 9.97 |
| Drop-one kept RMSE, abundant drops | **4.02** | 5.23 | 8.28 | 11.80 | 10.96 |
| Drop-one `other` MAE / bias | **6.8 / +1.2** | 11.3 / +10.8 | 26.4 / +26.4 | 16.7 / −16.7 | 16.7 / −16.7 |
| Detection AUROC, clean vs abundant drops | **0.79** | 0.70 | 0.69 | — | 0.52 |
| 90% coverage (clean / drop), width | 0.96 / 0.96, 15 pts | 0.89 / 0.91, 22 | 0.74 / 0.82, 23 | | |

Paired bootstrap, model − RPC kept RMSE (95% interval):
- clean **+0.56 to +1.59** (still worse);
- drop-one **−8.2 to −5.1** (better).

Sensitivity panels:

| Per side | Clean kept | Clean `other` | Drop-one kept | AUROC |
|---|---|---|---|---|
| 12 | 4.01 | 10.9 | 5.35 | 0.67 |
| 50 | 2.95 | 6.1 | 4.28 | 0.79 |

The 12-per-side panel (about 150 CpGs) is the weakest. It sits at the bottom of the training
marker range.

Per drop (25 per side), kept RMSE, model vs RPC:

| Dropped | RPC | v5 | v4 | v5 `other` (truth) |
|---|---|---|---|---|
| B | 4.36 | **3.97** | 5.16 | 12.0 (9.3) |
| CD4T | 6.55 | **4.68** | 7.02 | 14.1 (12.5) |
| CD8T | 6.59 | **5.54** | 5.60 | 11.1 (13.4) |
| Mono | 6.80 | **2.71** | 4.65 | 15.9 (12.7) |
| NK | 4.88 | **3.35** | 4.92 | 9.7 (9.0) |
| Neu | 22.74 | **3.00** | 3.59 | 44.2 (43.1) |

- v5 beats RPC on all six drops.
- Its `other` estimates track the true omitted fraction within about 3 points.
- Intervals now slightly over-cover (0.96 for a nominal 0.90).

Read against the frozen targets (EVALUATION.md), with the caveat that those targets are for
GSE182379, not this development set:
- missing-type kept RMSE at least 30% below RPC: here 63% below, **met in development**;
- `other` MAE ≤ 5 points: here 6.8, **not met**;
- clean RMSE ≤ 1.10 × RPC: here 1.65 ×, **not met**.

Synthetic validation (`results/gpu_v5_seed0/summary.json`): v5 beats RPC in every regime, by
5–34% on kept types. That includes both donor-mismatch regimes, which v4 lost (−8% and −21%).

## Diagnostic: what is left of the false `other`

Same decomposition as for v4. It uses the real reference and builds bulks from the truth
fractions, so it is a label-using diagnostic and changes no model.

| Residual added to reference × truth | v5 `other` | v4 `other` |
|---|---|---|
| Real residual | 6.4 | 18.9 |
| Per-CpG offset shared by the 12 mixtures | 6.1 | 17.1 |
| Sample-specific part | 3.8 | 3.9 |
| Inside the reference span | 0.0 | 0.0 |

- The v5 offsets absorb most of the shared batch effect, but not all of it.
- On simulated clean 6-type tasks at the matched offset setting, v5 reports 2.0–3.6 points of
  `other` and ties RPC (kept RMSE 2.10 vs 2.12), against 6.4 on real clean mixtures. The
  remaining gap is now small.

## Implication

- The missing-type advantage is large and robust across panels and drops.
- The clean-mixture penalty, about +1.1 points RMSE against RPC, is the open problem.
- Candidate fixes, none applied yet:
  - **Hybrid gate:** report RPC's fractions when the model's `other` is below a threshold
    chosen on *synthetic* validation, and the model's otherwise. This is cheap and CPU-only.
  - More capacity or steps. Several seeds would show seed-to-seed variance.
  - Post-hoc `other` recalibration on GSE110554. EVALUATION.md allows it as calibration
    development, but GSE110554 would then stop being a check.
- GSE110554 has now been used four times and shaped two priors. It is a development set only.
