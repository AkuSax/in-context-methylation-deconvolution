# Cohort-offset centering test, v5 checkpoint (2026-11-12)

Script: `deconv/cohort_center.py` (run from `deconv/` with `py -3.11 -B cohort_center.py`; about
75 min on this machine, `--dev-only` about 4 min). Files: `dev_summary.csv`, `dev_long.csv`,
`null_summary.csv`, `null_estimates.csv`, `run.log`.

Post hoc, no protocol frozen first. The GSE110554 part uses truth labels for scoring only and is
the eighth development look at that series. The centering step itself uses no labels.

## Method

- Model `results/gpu_v5_seed0/model.pt`, unchanged. Reference: GSE110554 sorted samples (Noob),
  25 markers per side, same panel code as `eval_real_dev.py`.
- Centering: fit RPC to every sample of a cohort, take the per-CpG median residual over the
  cohort, subtract it from the bulk, clip to [0, 1]; three passes. Variant `resid_int` also
  subtracts the cohort-median RPC intercept. The two variants agree to within 0.3 points
  everywhere, so only `resid` is shown.
- Whole-blood cohorts are the four of `label_free_null.py` (deposited betas, not Noob).
- "n = 12" rows: offset estimated from 12 random samples and applied to those 12, 20 draws.

## GSE110554 (12 mixtures, points)

| | v5 as is | v5 centred | RPC as is | RPC centred |
|---|---|---|---|---|
| Clean kept RMSE | 2.84 | 2.24 | 1.72 | 1.75 |
| Clean false `other` | 6.4 | 3.3 | 0 | 0 |
| Drop-one kept RMSE | 4.00 | 6.18 | 10.74 | 10.70 |
| Drop-one kept RMSE, no neutrophil drop | 4.17 | 4.88 | 5.92 | 5.94 |
| Drop-one `other` mean (truth 16.7) | 17.8 | 8.2 | 0 | 0 |
| Detection AUROC, clean vs abundant drops | 0.79 | 0.64 | — | — |

Per drop, v5 kept RMSE as is → centred (RPC): B 3.97 → 3.79 (4.36); CD4T 4.68 → 6.39 (6.55);
CD8T 5.54 → 5.75 (6.59); Mono 2.71 → 3.61 (6.80); NK 3.35 → 4.21 (4.88); Neu 3.00 → 10.49 (22.74).

## Whole-blood cohorts (no labels; `other` in points)

| Cohort | n | v5 mean, as is → centred | Frac > 5, as is → centred | Within-cohort SD, as is → centred | n = 12 centred mean | RPC unexplained mass, as is → centred |
|---|---|---|---|---|---|---|
| GSE40279 | 656 | 10.7 → 6.0 | 0.92 → 0.41 | 5.5 → 5.4 | 5.8 | 9.8 → 10.1 |
| GSE42861 | 690 | 14.0 → 9.0 | 0.99 → 0.65 | 6.8 → 7.7 | 7.3 | 16.6 → 16.8 |
| GSE87571 | 732 | 19.3 → 4.8 | 1.00 → 0.30 | 5.2 → 4.3 | 4.0 | 11.9 → 12.1 |
| GSE318173 | 559 | 8.2 → 4.8 | 0.76 → 0.26 | 5.4 → 4.8 | 4.8 | 9.5 → 9.8 |

Centering moves RPC's own fractions by 0.3–0.5 points (mean absolute change).

## Reading

- **A cohort-shared per-CpG offset accounts for about half of v5's false `other`.** Clean
  GSE110554 falls from 6.4 to 3.3 and the four cohorts fall by 3.5–14.5 points, to 4.8–9.0.
  The between-cohort spread narrows from 8–19 to 5–9. Whole blood holds a few points of
  eosinophils and basophils outside this reference, so 4.8 is near what a correct model
  could report; 6.0 and 9.0 are not.
- **Twelve samples estimate the offset as well as several hundred.** The n = 12 means are within
  2 points of the full-cohort ones.
- **The per-sample scatter is untouched.** Within-cohort SD stays at 4–8 points, so the model
  still cannot say whether an individual sample has 0 or 10 points of `other`.
- **Centering by subtraction removes a type that is missing from every sample.** On drop-one,
  mean `other` falls from 17.8 to 8.2 (truth 16.7), kept RMSE rises from 4.00 to 6.18 and
  AUROC falls from 0.79 to 0.64. The gain over RPC without the neutrophil drop shrinks from
  30% to 18%. In practice a missing type is missing from the whole cohort, so this is the
  realistic case, not a corner case.
- **RPC's unexplained mass does not move** (10–17 points before and after). That part of the
  cohort mismatch is not a per-CpG additive offset; it behaves like a compression of the
  cohort's beta range against the reference, which a cohort median cannot see.
- **Conclusion.** Cohort-level evidence carries real information about the batch offset, but
  it must not be applied as a preprocessing subtraction. The offset and a cohort-wide missing
  type differ in shape, not in being shared: a missing type's shared residual scales with the
  per-sample fraction and lies along a cell-type-like profile. That argues for giving the
  model the cohort residual summary as an input and training it on simulated cohorts where
  offsets and missing types co-occur, instead of subtracting.

## Limits

One checkpoint, one seed, one reference. The cohorts are deposited betas against a Noob
reference. GSE110554's drop-one design removes a type from all 12 mixtures and its mixtures
vary widely in composition, which real cohorts do not.
