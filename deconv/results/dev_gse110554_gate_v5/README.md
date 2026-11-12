# Hybrid gate on gpu_v5, GSE110554: result (fifth development look)

Protocol: [PROTOCOL.md](PROTOCOL.md), frozen before the full run. Script: `deconv/gate.py`.
Files: `thresholds.csv`, `synthetic_check.csv`, `summary.csv`, `estimates_long.csv`, `run.log`.

Disclosure: a smoke run of the script (16 calibration tasks per shape, against 256) printed
some gated real-data rows before the frozen run. The protocol was already written and was not
changed.

## Verdict: the gate does not fix clean mixtures

| 25 per side | Model | Gate q90 (primary) | Gate q95 | Gate q80 | RPC |
|---|---|---|---|---|---|
| Clean kept RMSE | 2.84 | 2.82 | 2.41 | 2.84 | **1.72** |
| Clean `other` MAE | 6.4 | 4.4 | 2.6 | 5.3 | 0 |
| Clean rows routed to RPC | — | 50% | 75% | 33% | |
| Drop-one kept RMSE | **4.00** | 4.43 | 4.47 | 4.10 | 10.74 |
| Drop-one `other` MAE | **6.8** | 8.2 | 8.1 | 7.2 | 16.7 |

- Bootstrap, gate q90 − RPC: clean +0.53 to +1.58, drop-one −8.1 to −4.4. Other panels show
  the same pattern.
- Why it fails:
  - The gate sends to RPC the clean samples where the model's `other` is already small, and
    there the model and RPC agree.
  - The samples with large false `other` (up to 11–14 points) sit above the threshold, so
    they keep the model's estimate.
  - Meanwhile, drop-one samples with a small omitted type lose the model's advantage.
- The synthetic check (`synthetic_check.csv`) predicted this. On simulated clean tasks of
  the same shapes, v5 is already within about ±0.3 points of RPC, so the gate gains nothing
  there. On missing-type tasks it costs 0.2–1 point. A threshold tuned on synthetic data
  cannot see a gap that exists only on real data.
- The gate is **not adopted**. `gate.py` is kept because its per-shape simulated thresholds
  (`thresholds.csv`) are a useful label-free null for `other`.

## Diagnostic: v5's clean error is all in `other`, not in the kept proportions

This uses truth labels and is post hoc. It is not an estimator, because always rescaling
would erase the missing-type gain.

| Clean mixtures | 12 per side | 25 per side | 50 per side |
|---|---|---|---|
| Model as reported | 4.01 | 2.84 | 2.95 |
| Model kept / (1 − `other`) | 2.93 | **1.60** | 1.68 |
| RPC | 2.33 | 1.72 | 1.64 |

- With the false `other` mass removed, v5's proportions among the present types are as good
  as RPC's, or slightly better at the primary panel. Per-type biases are similar to RPC's
  (both overestimate Mono and underestimate Neu).
- The only clean-mixture problem is therefore **`other` calibration on real data**: 2–14
  points per sample, mean 6.4.

## Implication

- Work on the `other` head, not the type head. Options:
  - A v6 prior with a larger share of clean tasks that carry strong per-CpG offsets, so false
    `other` is penalised more on exactly this nuisance.
  - A larger model or more steps, which may improve clean/missing separation (AUROC 0.79
    now).
  - Monotone recalibration of `other` against the simulated null. That is the
    `thresholds.csv` idea used as a soft shrink rather than a hard switch, and it can be set
    on synthetic data only.
- GSE110554 has now been looked at five times. Every further choice made on it lowers its
  value as evidence. GSE182379 remains untouched.
