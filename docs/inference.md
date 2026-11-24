# v6 completeness assumptions at inference

`inference_v6.estimate_v6` turns a v6 `FractionPrediction` into two explicitly
different outputs. It consumes an existing prediction, so the normal RPC fit,
model call and cohort construction remain unchanged. The helper is for v6
checkpoints trained with missing-type prevalence 0.55; do not silently apply
that prior to checkpoints trained under different priors.

```python
from inference_v6 import estimate_v6

# Given a loaded v6 model and tensors for a singleton or no-cohort prediction.
# Shapes are batch x markers x types, batch x markers x types, and batch x markers.
prediction = model(reference_mean, reference_sd, bulk)
estimate = estimate_v6(prediction)  # uncertain panel, training prevalence 0.55
composition = estimate.retained_composition     # sums to 1 across retained types
fractions = estimate.retained_full_mixture       # fractions of the full mixture
other = estimate.other_full_mixture              # fraction of the full mixture
all_fractions = estimate.full_mixture            # retained types, then other

# If panel completeness is known, force other to zero:
complete = estimate_v6(prediction, known_complete=True)
# Otherwise specify an assumed deployment prevalence, e.g. for sensitivity:
uncertain = estimate_v6(prediction, deployment_prevalence=0.10)
```

Retained composition is the model's estimated relative allocation among
retained types, conditional on retained mass being positive; it is not a
full-mixture fraction and is not guaranteed to equal truth when the panel is
incomplete or shifted. Full-mixture retained estimates equal that composition
times (1 - estimated other). The known-complete option imposes zero missing
mass; it does not establish that the panel is complete. `known_complete` and
`deployment_prevalence` cannot be combined. The prevalence must be strictly
between zero and one; zero for a panel known complete uses `known_complete`.
Default output equals the existing `prediction.mean`; existing benchmarks are
unchanged. `missing_type_score` is a model score, not a calibrated probability.

For an uncertain panel, the helper shifts the Bernoulli logit by
`logit(deployment_prevalence) - logit(0.55)` and multiplies the resulting score
by the unchanged positive-other Beta mean. This is a label-shift assumption:
the conditional feature distributions and positive-mass distribution must
transfer. Deployment prevalence is the fraction of inference cases with a
missing type, not the amount of missing mass in a sample. It is supplied by the
user, not estimated from these data. Synthetic training frequency does not
calibrate real-world prevalence or the output scores, and arbitrary missing mass
is not identifiable without assumptions. These are estimates, not calibrated
probabilities or prediction intervals.

For a cohort-enabled v6 model, define cohorts by assay provenance, same
reference and batch, without outcome labels. Predictions depend on cohort
composition and size; `cohort=None` makes each sample a singleton. Do not chunk
one cohort across model calls unless its features are recomputed consistently.

## Existing sensitivity, not a new evaluation

The prior sweep (`deconv/prior_sweep.py`) uses the no-cohort seed-0
v6 checkpoint and repeatedly examined GSE110554 development labels. At 25
markers per side, increasing assumed prevalence from 0.05 to 0.55 changed
full-mixture retained-type RMSE on complete physical mixtures from 1.90 to
2.29 percentage points; for all six strict one-type drops it changed from 4.37
to 4.03, and excluding neutrophil drops from 4.66 to 4.31. `other` MAE changed
from 1.06 to 5.87 on complete mixtures, and from 7.51 to 5.16 on all drops
(8.28 to 5.47 excluding neutrophil). RPC retained-type RMSE was 1.72, 10.74,
and 5.92, respectively, and its defined other estimate is zero. Abundant-drop
retained RMSE changed from 4.73 to 4.28; abundant drops excluding neutrophil
changed from 5.16 to 4.67. See the sweep for per-drop and all-prevalence
results, controls and sample counts. This trade-off does not select a
deployment prevalence: these are repeatedly used development data and not an
independent calibration or final test. Four unlabelled blood cohorts showed
lower reported other under lower assumed prevalence, but lack verified
zero-other truth.

The previously reported locked-test aggregates give a second view of this
choice. All entries below are errors in percentage points. The last column
applies the zero-missing-mass assumption even to genuinely incomplete panels
to measure its cost.

| Locked-test metric | Default prevalence 0.55 | Assumed prevalence 0.10 | Force `other=0` |
|---|---:|---:|---:|
| Complete panel, retained-type RMSE | 2.032 | 1.919 | 1.898 |
| Omitted type, retained-type RMSE | 2.650 | 2.768 | 2.876 |
| Omitted type, missing-mass MAE | 2.88 | 6.12 | 9.25 |

Choose the assumption from panel provenance and deployment knowledge before
scoring. For a documented complete panel, RPC is also a direct comparator
(clean RMSE 1.806). The v6 known-complete option uses the frozen model's
retained allocation.

`prior_sweep.py` writes aggregate and per-sample results to
`deconv/results/prior_sweep_v6_nocohort_2026-11-18/`.
Validation: `python -m pytest tests/test_inference_v6.py` checks deterministic algebra,
the default, explicit complete case, invalid priors, a saved output and a CPU
checkpoint on synthetic inputs. It does not score locked data.
