# Label-free whole-blood null check, v5 checkpoint

Script: `deconv/label_free_null.py` (run from `deconv/` with `py -3.11 -B label_free_null.py`).
Outputs: `estimates.csv` (one row per sample) and `summary.csv` (one row per cohort).

## Method

- Model: `results/gpu_v5_seed0/model.pt`, `Deconvolver(width=96, heads=4, blocks=4)`, eval mode, CPU.
  Fallback prior for the RPC-shrunk baseline read from `best.prior_mean_other` (0.0906).
- Reference: six sorted types from GSE110554 (`datasets.load("GSE110554").sorted`, 37 samples:
  Neu, NK, B, CD4T, CD8T, Mono). Built with `eval_real_dev.panel` at 25 markers per side, no dropped
  type, and run with `eval_real_dev.run_methods`. The target truth labels are not used.
- Reference preprocessing is Noob, because `eval_real_dev` sets `DECONV_PREPROC=noob`.
- Cohorts use the **deposited processing, not Noob**. The v5 checkpoint was trained on deposited
  GSE35069 betas, so this is the closest match, but the reference and the cohorts were processed
  differently.
- Model `other` is the posterior mean. P(other > 0) is the model's zero-inflation head
  (`has_other.probs`). RPC-raw other is `1 - sum(clamp_min(0, RPC slopes))`, as in `train_cpu.py`
  (plain RPC fractions sum to one, so their implied other is zero by construction).

## Probes and samples

| Cohort | Platform | Samples used | Reference probes | Shared with cohort | Probes used | Markers | Sample-level rule would drop |
|---|---|---|---|---|---|---|---|
| GSE40279 | 450K | 656 | 860,656 | 437,801 | 437,801 | 272 | 0 |
| GSE42861 | 450K | 690 | 860,656 | 448,475 | 448,475 | 272 | 0 |
| GSE87571 | 450K | 732 | 860,656 | 448,475 | 416,604 | 270 | 732 |
| GSE318173 | EPIC | 559 | 860,656 | 500,549 | 500,536 | 275 | 4 |

- **Probe rule (deviation):** the spec said to drop samples with any missing beta. In GSE87571 that
  would remove every sample, because the deposited betas have scattered NA values (detection
  filter). I therefore dropped probes with any missing value in the cohort and kept all samples.
  The sample-level counts are in the table for transparency.
- GSE318173 contains **both PD (281) and control (278)** whole-blood arrays. It is not a controls-only
  subset. All 559 were used.
- GSE40279 and GSE87571 sample IDs are array barcodes or GSM IDs taken from the local sample key or
  series matrix. GSE42861 IDs are array barcodes (no series matrix locally).
- GSE42861 has 690 samples in the file, against the "about 689" in the brief.

## Results (points; threshold 5)

| Cohort | Model mean other | Model median | Frac > 5 | Mean P(other>0) | RPC-raw mean | RPC-raw median | RPC-raw frac > 5 |
|---|---|---|---|---|---|---|---|
| GSE40279 | 10.7 | 9.6 | 0.92 | 0.74 | 9.8 | 9.8 | 0.96 |
| GSE42861 | 14.0 | 12.3 | 0.99 | 0.85 | 16.6 | 16.3 | 1.00 |
| GSE87571 | 19.3 | 19.2 | 1.00 | 0.92 | 11.9 | 11.9 | 0.97 |
| GSE318173 | 8.2 | 6.9 | 0.76 | 0.68 | 9.5 | 9.4 | 1.00 |

## Reading

Whole blood should have essentially zero true `other`, yet the model reports 8 to 19 points of
`other` on average, with 76% to 100% of samples above the 5-point threshold. The false `other`
varies by cohort: GSE87571 is highest (19.3), GSE318173 lowest (8.2). The RPC-raw implied residual
is also large (9.5 to 16.6), so part of the excess is not specific to the model. It reflects the
reference-to-cohort mismatch (Noob reference against deposited cohorts, and an unconstrained sum of
slopes). These numbers come from one mismatched preprocessing setup and do not separate that cause
from the model's own behaviour. The mean false `other` is 1.6 to 3.9 times the 5-point threshold across
cohorts, so the v5 `other` estimate is not usable as a null check on these data without
re-processing the cohorts to match the reference.
