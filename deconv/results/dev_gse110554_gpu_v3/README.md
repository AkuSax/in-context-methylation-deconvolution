# GSE110554 development check of gpu_v3: result (second development look)

Protocol: [PROTOCOL.md](PROTOCOL.md), frozen before scoring. Script: `deconv/eval_real_dev.py`.
Files: `summary.csv`, `estimates_long.csv`, `run.log`.

## Verdict: the technical-shift fix did not remove the false `other`

| Primary panel (25 per side, ~286 CpGs) | v3 | v2 | RPC | CP |
|---|---|---|---|---|
| Clean kept RMSE | 9.54 | 8.25 | **1.72** | 1.55 |
| Clean `other` (truth 0) | **38.6** | 34.0 | 0 | 0 |
| Drop-one kept RMSE (72 pairs) | 8.43 | 7.54 | 10.74 | 9.97 |
| Detection AUROC, clean vs abundant drops | 0.69 | 0.80 | — | 0.52 |
| 90% coverage, width | 0.74 / 0.82, 23–25 pts | 0.81 / 0.88, 27 pts | | |

Per drop, v3 is worse than RPC for every type except neutrophils, the same pattern as v2.
On synthetic validation, which includes the shift, v3 beats RPC by 14–27% on kept types,
with detection AUROC 0.61–0.79 (`results/gpu_v3_seed0/summary.json`; best step 14,000).

## Diagnosis: the problem is the reference markers, not only the noise

This ablation uses the real GSE110554 reference (286 CpGs) and builds bulks from the truth
fractions. It is a diagnostic that uses labels and changes no model.

| Bulk given to the model | v2 `other` | v3 `other` |
|---|---|---|
| Real mixtures | 34.0 | 38.6 |
| reference × truth, no noise | **0.0** | **0.0** |
| reference × truth + N(0, 0.02) | 9.2 | 13.8 |
| reference × truth + real residuals shuffled across CpGs | 29.2 | 27.0 |
| (Simulated clean 6-type tasks, same CpG count) | 1.3 | 3.2 |

- The model is not broken on real panels: with a noise-free bulk it reports zero `other`.
- With a real reference, though, ordinary noise inflates `other` far more than in
  simulation. N(0, 0.02) gives 9–14 points, against 1–3 points on simulated tasks with
  similar noise.
- The real reference is out of distribution. Real markers, picked by `pick_markers` from
  about 850k EPIC probes, are near-binary: the median range across types is 0.88, 93% of
  markers have a type below 0.1 and 75% a type above 0.9. Simulated contrast markers come
  from a random 10,000-CpG candidate pool and are much weaker: median range 0.53, with 24%
  and 17%.
- Real mixture residuals against their own reference are also large: mean +0.027 and SD
  0.052, against at most about 0.04 batch SD in the v3 prior.

## Implication for v4

- Make simulated marker panels as extreme as real ones. Draw the candidate pool from the
  whole array, or from CpGs pre-screened for between-type range, so that contrast markers
  reach a median range of about 0.85. Check the marker statistics against real panels
  before training.
- Consider widening the per-CpG batch SD up to about 0.06.
- GSE110554 has now been used twice in development and its statistics shaped the prior.
  Report it as a development set only.
