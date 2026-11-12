# Audit of the gpu_v5 development result on GSE110554 (sixth development look)

Done 2026-11-11, after the v5 check and the gate check. **Post hoc: no protocol was frozen
first, and both scripts use truth labels.** No model was trained or changed. The purpose was to
test how the v5 result is being read, not to produce a new score.

- `audit_dev_estimates.py` re-reads `../dev_gse110554_gpu_v5/estimates_long.csv` (no model
  run). Output: `estimates_audit.log`.
- `decompose_dev.py` runs the frozen `gpu_v5_seed0` checkpoint on the same panels. Output:
  `decompose.log`, `decompose_summary.csv`, `decompose_long.csv`.

Reproduced first: clean kept RMSE 2.84 vs RPC 1.72, drop-one 4.00 vs 10.74, and the clean
proportions result 1.60 vs 1.72. The 26 deconvolution tests pass.

All numbers are in percentage points of the whole mixture, primary panel (25 per side) unless
stated.

## 1. The pooled drop-one result is dominated by the neutrophil drop

| Drop-one kept RMSE | v5 | RPC | Change |
|---|---|---|---|
| All six drops (72 pairs) | 4.00 | 10.74 | −63% |
| Neutrophil drop only | 3.00 | 22.74 | −87% |
| Without the neutrophil drop (60 pairs) | 4.17 | 5.92 | −30% |

- Without neutrophils, the paired bootstrap of model − RPC is −2.53 to −0.82 and the ratio is
  0.62 to 0.84.
- Sensitivity panels without neutrophils: −14% at 12 per side (ratio 0.78 to 0.98) and −24% at
  50 per side.
- "Beats RPC on all six drops" holds at 25 per side only. v5 loses the B and NK drops at 12
  per side and the B drop at 50 per side (see `../dev_gse110554_gpu_v5/summary.csv`).

Neutrophils are 43% of these mixtures on average, and a blood reference without them is not a
realistic case. The figure to carry forward is about 30%.

## 2. `other` tracks myeloid and B-cell drops, not T or NK drops

Correlation of the model's `other` with the true omitted fraction across the 12 mixtures,
within each drop:

| Dropped | True mean (range) | v5 mean | r within drop | v5 MAE | Constant 9.06 MAE |
|---|---|---|---|---|---|
| Neu | 43.1 (6–75) | 44.2 | 1.00 | 5.1 | 34.5 |
| Mono | 12.7 (4–23) | 15.9 | 0.87 | 5.3 | 6.7 |
| B | 9.3 (1–26) | 12.0 | 0.67 | 6.3 | 7.2 |
| CD8T | 13.4 (2–33) | 11.1 | 0.18 | 9.7 | 7.7 |
| CD4T | 12.5 (6–18) | 14.1 | 0.16 | 7.1 | 4.3 |
| NK | 9.0 (0–22) | 9.7 | 0.04 | 7.4 | 6.0 |

- The constant is the training-prior mean of `other` (9.06 points), a label-free guess.
- Without the neutrophil drop, v5's `other` MAE is 7.16 against 6.38 for that constant. Across
  those 60 pairs the correlation with truth is 0.46 and the slope is 0.49.
- On the 13 drop rows with a true omitted fraction under 5 points (mean 2.4), v5 reports 9.7.
- Detection of abundant drops against the 12 clean rows, by size of the true omitted
  fraction: AUROC 0.72 for 5–10 points, 0.68 for 10–20 and 0.99 above 20. Excluding
  neutrophils the overall AUROC is 0.75 (0.79 with them).

So "`other` tracks the true omitted fraction within about 3 points" is true of the per-drop
means only.

## 3. The model reads clean real mixtures as missing something

| Mean P(`other` > 0) | Clean | One type missing |
|---|---|---|
| Real GSE110554 | 0.76 | 0.86 |
| Synthetic validation, 12 types | 0.37 | 0.64 |

- P(`other` > 0) is at least 0.5 for 10 of the 12 clean real mixtures.
- Setting `other` to zero when that probability is under 0.5 therefore does not help: clean
  kept RMSE 3.23 (as reported 2.84), drop-one 4.21 (4.00).

## 4. The drop-one gain is in the proportions, not only in `other`

| Drop-one kept RMSE | All drops | Without neutrophils |
|---|---|---|
| RPC | 10.74 | 5.92 |
| RPC proportions × (1 − model `other`) | 6.08 | 5.76 |
| RPC proportions × (1 − true omitted fraction) | 5.49 | 4.84 |
| v5 as reported | 4.00 | 4.17 |
| v5 proportions × (1 − true omitted fraction) | 3.08 | 3.17 |

- v5 beats RPC even when RPC is given the true omitted fraction. Against truth renormalised to
  the kept types, proportion RMSE is 4.00 for v5 and 10.11 for RPC (3.87 and 5.81 without
  neutrophils).
- The `other` estimate is the weak part. A perfect `other` would take v5 from 4.00 to 3.08 on
  drop-one and from 2.84 to 1.60 on clean.
- The gain is not uniform: on the B drop v5's proportions are worse than RPC's at 12 and 50
  per side.

## 5. Synthetic validation at 12 types is already below the approved target

From `../gpu_v5_seed0/summary.json`, kept RMSE, model vs RPC:

| Regime | v5 | RPC | Change | v2 simulator, gpu_v2 |
|---|---|---|---|---|
| One missing, 12 types | 2.63 | 3.65 | −28% | −37% |
| One missing + donor mismatch, 12 types | 3.39 | 3.87 | −12% | −29% |

The approved missing-type target is a 30% reduction on GSE182379, which has 12 types and drops
of about 8 points. The simulator's own closest regime does not reach it.

## What this changes

- State the development missing-type gain as about 30% (14–30% across panels), not 63%.
- The defensible claim is robust proportions under an incomplete reference. Quantifying the
  missing mass from one sample is not established for T and NK cells.
- The clean penalty is an identifiability problem between batch offsets and a missing type,
  and it has been examined on one batch only.
- Each further look at GSE110554 lowers its value. This was the sixth.
