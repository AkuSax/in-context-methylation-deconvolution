# Missing mass with the composition confound removed

Run 2026-11-12, `python decorrelated_missing.py` (SHA-256 `1f1fb0ba…6958e`, on top of
`b387fae`), 5 seeds, about 8 minutes on CPU. This follows
[missing_mass_bounds](../missing_mass_bounds/README.md). There, GSE110554 could not show
whether any score tracks omitted mass, because its omitted fractions are collinear with the
neutrophil fraction. **No real mixture labels are used here**; nothing touches GSE110554 or
the reserved test series.

## Design

- **Profiles:** the 7 GSE103541 donors held out of v6 training. They were v6's
  checkpoint-selection validation donors. Five types: B, CD4T, CD8T, Mono, Gran.
- **Mixtures:** each uses one donor's sorted profiles, mixed linearly in beta, plus
  N(0, 0.01) noise per CpG. The omitted type's fraction `u` ~ U(0, 0.3), *independent* of
  the retained types' conditional proportions, which follow a Dirichlet(20 × blood-like
  mean). 60 mixtures per omitted type and seed; 5 seeds.
- **Paired control:** every mixture set is scored with the reduced reference (omitted type
  removed before marker selection) and with the full reference. A score that tracks missing
  mass must follow `u` under the reduced reference and not under the full one.
- **Scenarios:** `within` uses a reference from the 20 GSE103541 training donors (same study,
  array and processing). `cross` uses all six GSE35069 donors (450K, deposited processing).
  The cross-study full reference is the complete-but-shifted case.
- **Scores:** per-sample LP lower bound (z = 2), cohort residual PC1, CETYGO, 1 − ΣCP, and
  the frozen v6 no-cohort `other`. 25 markers per side.
- **Constant controls for MAE:** 0.075 for the oracle constant (the true mean, 0.15) and
  0.087 for v6's training-prior mean (0.091).

## Results (mean over 5 seeds)

**The control works.** Under the full reference, most correlations with `u` are within ±0.2
(`r_full` in `summary.csv`). The confound that dominated GSE110554 is gone. Exceptions are
cross-study CETYGO (−0.60 for CD4T, −0.40 for CD8T) and the lower bound (−0.32/−0.34), which
respond to composition-dependent mismatch, and v6 Gran (+0.28/+0.32).

**Correlation with the omitted fraction, reduced reference:**

| | B | CD4T | CD8T | Gran | Mono |
|---|---:|---:|---:|---:|---:|
| within: lower bound z=2 | .96 | .55 | .52 | .94 | .68 |
| within: CETYGO | .98 | .70 | .78 | .96 | .96 |
| within: cohort PC1 | 1.00 | .02 | .31 | .97 | .97 |
| within: v6 `other` | .87 | .67 | .63 | .93 | .94 |
| cross: lower bound z=2 | .32 | .02 | .09 | .32 | .36 |
| cross: CETYGO | .90 | −.04 | .23 | .45 | .86 |
| cross: cohort PC1 | .82 | .15 | .09 | −.06 | .91 |
| cross: v6 `other` | .29 | .25 | −.06 | .51 | .64 |

**Absolute error of the missing fraction (MAE; constants 0.075 oracle, 0.087 prior):**

| | B | CD4T | CD8T | Gran | Mono |
|---|---:|---:|---:|---:|---:|
| within: lower bound z=2 | **.05** | .08 | .09 | **.04** | **.06** |
| within: v6 `other` | .08 | .08 | .10 | .08 | .11 |
| cross: lower bound z=2 | .08 | .14 | .11 | .09 | .08 |
| cross: v6 `other` | .08 | .09 | .12 | **.06** | **.07** |
| either: 1 − ΣCP | .15 | .15 | .15 | .15 | .16 |

Seed SD of MAE ≤ 0.02. Bold marks values below the oracle constant.

**Share of samples where the lower bound is at or below the truth:** 63–93% within-study,
29–46% cross-study. v6 overshoots on Gran and Mono within-study (14% and 28% of samples at
or below the truth; slopes 1.4 and 1.9).

**Detection AUROC** (reduced-reference rows with `u` ≥ 5% against the same mixtures under
the full reference): within-study, v6 is highest (0.76–0.99), with CETYGO close behind
(0.78–0.97). Cross-study, the lower bound is the most consistent (0.72–0.89) and CETYGO is
best for B (0.97).

## Reading

1. **Without the confound, missing mass is partly recoverable when reference and target
   match technically.** For distinct types (B, Gran, Mono), the assumption-light LP bound
   correlates 0.68–0.96 with the truth and beats both constant controls on MAE. It also beats
   the learned v6 `other` on MAE, although v6 ranks samples about as well and detects best.
2. **Cross-study mismatch, not the geometry of the omitted profile, is what removes most of
   the absolute level.** The bound stops covering the truth, and v6 is near the constants
   except for Gran and Mono. Rankings partly survive for B and Mono (CETYGO, PC1). This is
   the audit's `delta` term in measured form: the same mixtures lose the signal when only
   the reference study changes.
3. **T-cell subsets are hard in both scenarios.** CD4T and CD8T have the lowest correlations
   and MAE near the constants, consistent with their small hull distances in
   `results/identifiability_map/`. Here the geometry also limits recovery.
4. **1 − Σ fractions from NNLS/CP carries no missing-mass signal** (always near 0).
   Unconstrained solvers fill the gap with retained types.

## Limits

Mixtures are synthetic: linear in beta, one donor per mixture, light Gaussian noise. The
within-study reference shares study, array and processing with the mixture donors, which is
the most favourable case. v6 is in distribution: it trained on other donors of these two
studies, and these donors selected its checkpoint. `u` is uniform on 0–30%, so MAE depends
on that range; compare against the constants, not across designs. Cross-study Gran is
GSE35069's granulocyte fraction against GSE103541 Gran, which adds a definitional mismatch.
None of this is real-data evidence; it says which signals exist when the confound is absent.

## Files

`scores_long.csv.gz` (every score, sample, reference, seed), `per_seed.csv`, `summary.csv`
(means and SDs over seeds), `run.json` (design, donors, constants), `run.log`.
