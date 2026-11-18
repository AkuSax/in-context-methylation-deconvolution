# Detection baselines on GSE110554, 12 November 2026

`python detection_baselines.py` (from `deconv/`; a few minutes on CPU). The panels are rebuilt
exactly as in `eval_real_dev.py`. Model scores are read from the saved development estimates;
no model was run. This is another look at the GSE110554 labels, for scoring only.

Task: separate the 12 clean-panel mixtures (negatives) from drop-one rows whose omitted share
is at least 5 points (59 positives). The rows reuse 12 mixtures, so the AUROCs are descriptive.

`CETYGO` follows the definition of Hannon et al. (2021): the RMS of bulk minus
reference × Houseman constrained-projection fractions, over the deconvolution CpGs. It is
implemented in `diagnostics.cetygo` and is not the R package itself. `CETYGO_eq` is identical
here because NNLS fractions sum to at least one on every row. For the same reason
`NNLS_unexplained` is constant (AUROC 0.5).

| Score | AUROC, 12 per side | 25 | 50 | 25, excluding the Neu drop |
|---|---|---|---|---|
| **CETYGO** | **0.81** | **0.82** | 0.74 | **0.77** |
| RPC residual RMS | 0.77 | 0.79 | 0.70 | 0.73 |
| RPC unexplained (1 − sum of raw slopes) | 0.63 | 0.65 | 0.65 | 0.56 |
| CP unexplained (1 − sum) | 0.49 | 0.52 | 0.44 | 0.51 |
| model `other`, v5 | 0.67 | 0.79 | **0.79** | 0.75 |
| model `other`, v6 no cohort | 0.69 | 0.78 | 0.78 | 0.73 |
| model `other`, v6 cohort | 0.65 | 0.72 | 0.75 | 0.65 |

Per omitted type at 25 per side (AUROC, CETYGO / v6 no cohort): B 0.96 / 0.91, CD4T 0.65 / 0.69,
CD8T 0.68 / 0.71, Mono 0.78 / 0.63, NK 0.92 / 0.79, Neu 1.00 / 1.00. Both scores fail on the T-cell
drops, which `results/identifiability_map/` shows are the near-hull types.

## Reading

- **A free reconstruction-error diagnostic detects incomplete references as well as the
  model's `other`, or better.** CETYGO needs no training. It matches or beats v6 at 12 and 25
  markers per side and loses only at 50. The model's `other` therefore has no demonstrated
  detection advantage. It does give a share, which CETYGO does not, but that share's accuracy
  is the separate, unproven claim.
- The model is worth more for its retained-type fractions than as a detector. Any detection
  or reliability claim must report CETYGO alongside it.
- Neither CETYGO nor `other` can separate an omitted type from a complete-but-shifted
  reference. Both rise under either. These rows contain omissions only.
