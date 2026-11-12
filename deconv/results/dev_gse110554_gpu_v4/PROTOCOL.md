# Development check on GSE110554 (v4): protocol, frozen before scoring

Written 2026-11-11T16:25Z, before any v4 output on real data was seen. **Third development look** at
GSE110554 for this model family. v2 and v3 were scored on it, and GSE110554 statistics shaped
the v4 prior: the target median marker range of about 0.88, the sharpening range and the
pool screen were all chosen to match real GSE110554 panels. This is a **development** check,
not the primary test. EVALUATION.md assigns GSE110554 to calibration development and a
descriptive six-type test. GSE167998 and GSE182379 are not touched.

## Fixed inputs

- Model: `results/gpu_v4_seed0/model.pt`, SHA-256
  `2113a68383bf964f8c40f0929f1f136a7982c502e14a358db326c7845eb03559` (step 28,000; width 96,
  4 blocks, 4 heads). It was trained on synthetic GSE35069 tasks only (450K, deposited
  processing). No retraining or tuning on these data.
- v4 training prior: everything in v3, plus candidate CpGs drawn from a pool of 33,385
  GSE35069 CpGs whose type-mean range is at least 0.3 (training donors only), M-value
  sharpening of every profile with a ~ U(1.3, 2.2), and marker modes pick 0.6 / contrast 0.2 /
  informative 0.1 / random 0.1. Per-CpG batch SD up to 0.06.
- Data: GSE110554 Noob betas with detection-p masking (`DECONV_PREPROC=noob`).
  - Reference: all 37 sorted arrays (6 types, including the known technical replicate, as in
    the baseline harness).
  - Targets: the 12 constructed mixtures.
- Markers: `reference.usable_probes(reference, targets)` (label-free availability), then
  `reference.pick_markers` on reference labels only.
  - **Primary:** `n_per_side = 25` (about 300 CpGs, the top of the training range).
  - **Sensitivity:** 12 (about 150) and 50 (about 600; the baseline harness default, outside
    the training range).
- Reference mean and SD (ddof = 1) per type from `reference.build_reference`.
- Every method uses the same CpGs within each panel.

## Methods

- Model: point estimate = predictive mean; 90% intervals = 5th/95th percentiles of 400
  draws (torch seed 0).
- RPC (`rpc_torch`, equal to EpiDISH RPC), NNLS, CP with sum ≤ 1 (`solvers`).
- RPC shrunk by the training-prior mean of `other` (read from the checkpoint summary, 0.0907); its `other` estimate is that
  constant.
- `other` for NNLS and CP is max(0, 1 − sum). For RPC it is 0 by definition.

## Regimes

1. **Clean:** all 6 types, 12 mixtures.
2. **Strict drop-one:** for each of the 6 types, remove its sorted samples before probe
   filtering and marker selection, then rescore all 12 mixtures (72 sample–drop pairs).
   Truth is not renormalised; the dropped type's truth is the `other` target.
   - Abundant pairs: dropped truth ≥ 5 points, fixed from truth before scoring.

## Metrics

- Pooled kept-type RMSE (points).
- `other` MAE and signed bias, over all pairs and abundant pairs.
- Per-drop kept RMSE.
- Detection AUROC: model `other`, and NNLS/CP residual mass, on 12 clean rows vs abundant
  drop rows.
- 90% interval marginal coverage and mean width.
- Paired bootstrap of model − RPC kept RMSE: 10,000 replicates, seed 20261111, resampling
  the 12 mixture rows and keeping all drops of a row together.

## Interpretation limits

- 12 mixtures from the same study as the reference, so donor and DNA independence is not
  established.
- Six types, while training used 4–12.
- Cross-platform and cross-processing: the model was trained on 450K deposited betas and is
  applied to EPIC Noob.
- All results are reported whatever they show. A model change prompted by these results
  must be stated as such, and GSE110554 then no longer counts as unseen for that model.
