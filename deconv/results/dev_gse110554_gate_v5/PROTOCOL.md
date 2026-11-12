# Hybrid gate on gpu_v5, GSE110554 development check: protocol, frozen before scoring

Written 2026-11-11, before any gated output on real data was computed. **Fifth development
look** at GSE110554. The idea for a gate came from the v5 result on these data (clean kept
RMSE 2.84 vs RPC 1.72), so this is a model change prompted by GSE110554 and is reported as
such. The threshold rule below uses **no GSE110554 labels or bulk data**: only each panel's
shape (number of retained types K and number of markers M), which is known at test time.
GSE167998 and GSE182379 are not touched.

## Fixed inputs

- Model: `results/gpu_v5_seed0/model.pt`, SHA-256
  `a8c9a9fe0199cd9a51e8b78d216f40890513aa0486b6f7a52210cf0a6342dab3`, unchanged.
- Real-data model and RPC estimates: `results/dev_gse110554_gpu_v5/estimates_long.csv`,
  produced under the frozen v5 protocol. The gate is computed from those rows, so the
  model and RPC numbers are identical to the v5 check.

## Gate (decision rule)

For a sample with panel shape (K, M):

1. Simulate 256 **clean** tasks (no missing type) with K retained types and M markers. Use
   the v5 validation prior: validation donors of GSE35069 (seed-0 donor split), `pick`
   markers, training-donor candidate pool, sharpening, the v5 technical shift
   (`train_cpu.TECH_SHIFT_RANGES`), noise 0.02, donor mismatch with probability 0.5 and
   synthetic augmentation. RNG seed `[20261111, K, M]`.
2. τ(K, M) = the **90th percentile** of the model's predicted `other` on those tasks.
3. If the model's `other` < τ: report the **RPC** fractions for the retained types and
   `other` = 0. Otherwise report the model's fractions and `other`.

Primary quantile q = 0.90. Sensitivity: q = 0.80 and 0.95 (same simulated tasks).

## Synthetic check of the gate (no real data)

At each panel shape, a separate synthetic test set uses seed `[20261112, K, M]`, the same
prior and 128 tasks per regime: clean, and one missing type, each with donor mismatch at 0.5.
Report kept RMSE for the model, RPC and the gate, plus the gate's fraction routed to RPC.

## Real-data metrics

The same metrics, panels and bootstrap as the v5 protocol, for method `gate_q90` (and the
sensitivity gates):
- clean and drop-one kept RMSE and `other` MAE;
- paired bootstrap of gate − RPC (10,000 replicates, seed 20261111);
- fraction of rows routed to RPC.

Detection AUROC is unchanged by the gate for the model score and is not recomputed.
Interval coverage is reported only for rows the gate leaves with the model.

## Interpretation limits

As in the v5 protocol. In addition, the gate was motivated by these data. Its real-data
numbers are development evidence, and only the GSE182379 test can confirm it.
