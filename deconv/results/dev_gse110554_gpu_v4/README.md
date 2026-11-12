# GSE110554 development check of gpu_v4: result (third development look)

Protocol: [PROTOCOL.md](PROTOCOL.md), frozen before scoring. Script: `deconv/eval_real_dev.py`.
Files: `summary.csv`, `estimates_long.csv`, `run.log`. Model SHA-256 `2113a683…eb03559`, best
step 28,000.

## Verdict: half the false `other` is gone, and missing-type error is half of RPC's; clean still fails

| Primary panel (25 per side, 286 CpGs) | v4 | v3 | v2 | RPC | CP |
|---|---|---|---|---|---|
| Clean kept RMSE | 5.87 | 9.54 | 8.25 | **1.72** | 1.55 |
| Clean `other` (truth 0) | **18.9** | 38.6 | 34.0 | 0 | 0 |
| Drop-one kept RMSE (72 pairs) | **5.26** | 8.43 | 7.54 | 10.74 | 9.97 |
| Drop-one kept RMSE, abundant drops | **5.23** | | | 11.80 | 10.96 |
| Drop-one `other` MAE / bias | 11.3 / +10.8 | | | 16.7 / −16.7 | 16.7 / −16.7 |
| Detection AUROC, clean vs abundant drops | 0.70 | 0.69 | 0.80 | — | 0.52 |
| 90% coverage, width | 0.89 / 0.91, 22 pts | 0.74 / 0.82 | 0.81 / 0.88 | | |

Paired bootstrap, model − RPC kept RMSE (95% interval): clean **+3.3 to +5.0** (worse),
drop-one **−6.6 to −4.2** (better), abundant drops −8.8 to −4.6. The sensitivity panels (12 and
50 per side) show the same pattern: clean 5.6 and 6.2, drop-one 5.5 and 5.6.

Per drop (25 per side), kept RMSE, model vs RPC:

| Dropped | RPC | v4 | v3 | v4 `other` (truth) |
|---|---|---|---|---|
| B | 4.36 | 5.16 | 8.10 | 18.8 (9.3) |
| CD4T | 6.55 | 7.02 | 9.12 | 24.2 (12.5) |
| CD8T | 6.59 | **5.60** | 8.75 | 22.2 (13.4) |
| Mono | 6.80 | **4.65** | 9.64 | 28.2 (12.7) |
| NK | 4.88 | 4.92 | 8.41 | 16.7 (9.0) |
| Neu | 22.74 | **3.59** | 6.07 | 54.7 (43.1) |

v4 now matches or beats RPC on four of the six drops, where v3 beat it only on neutrophils.
The kept-type gain is real, but `other` is still about 10 points too high everywhere: the
same offset as on clean mixtures.

On synthetic validation, v4 beats RPC by 12–36% on kept types except under donor mismatch
(mismatch12 −8%, mismatch6 −21%). See `results/gpu_v4_seed0/summary.json`.

## Diagnosis: the reference is fixed; the residual shape is not

This ablation uses the real reference (286 CpGs) and builds bulks from the truth fractions. It
is a diagnostic that uses labels and changes no model. Script:
`scratchpad/ablate_v4.py` (not in the repo).

| Bulk given to the model | v4 `other` | v3 `other` |
|---|---|---|
| Real mixtures | 18.9 | 38.6 |
| reference × truth, no noise | 0.0 | 0.0 |
| reference × truth + N(0, 0.02) | **1.6** | 15.1 |
| reference × truth + N(0, 0.04) (same robust scale as real, 0.039) | **3.2** | 23.1 |
| reference × truth + constant +0.027 (the real mean offset) | 0.0 | 0.0 |
| reference × truth + real residuals, demeaned per sample | 19.2 | 38.9 |
| reference × truth + real residuals shuffled across CpGs | 14.7 | 27.1 |
| (Simulated clean 6-type tasks, v4 prior, same CpG count) | 0.9 | 13.8 |

- The v4 marker fix worked. With Gaussian noise on the real reference, `other` stays near
  zero even at the real residual scale.
- The offset is harmless: a constant shift of +0.027 gives 0.
- The remaining false `other` comes from the **shape** of the real residuals. With the same
  robust scale (0.039), real residuals give 15–19 points of `other` and Gaussian noise gives 3.
- Standardised RPC residuals |r|/scale:

  | | P > 2 | P > 3 | P > 5 | kurtosis |
  |---|---|---|---|---|
  | Real | 0.104 | 0.038 | 0.004 | 7.4 |
  | Simulated v4 | 0.069 | 0.017 | 0.003 | 37 |
  | Gaussian | 0.045 | 0.003 | 0.000 | 3 |

  Real residuals have a broad band of moderate (2–5σ) outliers. The simulator has fewer of
  those and a handful of extreme ones.
- The moderate outliers sit at CpGs with mid-range bulk beta: 5–6% of CpGs with bulk beta
  0.1–0.7, against 1–2% above 0.7 and 0 below 0.1. They are spread across all marker types,
  and 35 of 286 CpGs account for all of them. This fits a per-CpG systematic offset between the
  sorted arrays and the mixture arrays on the M-value scale. On the beta scale, such an offset
  is largest at mid-range values. Its spread is wider than the simulator's.
- Each such CpG looks to the model like a marker of an absent type, so `other` rises.

## Implication for v5

- Draw per-CpG batch offsets on the M-value scale, from a heavier-tailed distribution (for
  example a Student-t, or a mixture where 10–15% of CpGs get a large offset). Then check that
  the standardised RPC residuals reach P(|z| > 2) ≈ 0.10 and P(|z| > 3) ≈ 0.04 on simulated
  clean tasks.
- Separately, the mismatch regimes on synthetic data now favour RPC. Training should keep
  enough mismatch-only tasks with `other` = 0.
- GSE110554 has now been used three times in development, and its residual statistics would
  shape v5. Report it as a development set only. A post-hoc recalibration of `other` on
  GSE110554 is allowed under EVALUATION.md (calibration development), but it would then no
  longer be a check.
