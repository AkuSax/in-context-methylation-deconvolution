# Identifiability map of omitted types, 12 November 2026

`python identifiability_map.py` (from `deconv/`; about 3 min on CPU). Sorted profiles only; no
mixture labels are used to build the map. GSE167998, the final-test reference, is left out by
design: compute its map only inside the frozen final-test protocol.

For each source and each type d, the markers are picked on the retained types as in the strict
drop-one protocol. The table reports the closest convex mixture of the retained type means to
d's mean profile. `hull_resid_rms` is the RMS distance in beta units, and `hull_resid_over_sd`
divides it by the median within-type SD. A bulk with omitted share f differs from the best
complete-reference fit by at most about f × that residual, so a type near the hull cannot be
recovered from the bulk and reference alone. That limit applies to any method.

## 25 markers per side

| Source | Omitted type, closest to farthest from the hull (residual RMS, nearest mixture) |
|---|---|
| GSE110554 (6 types, EPIC) | CD4T 0.093 (CD8T 0.94) · CD8T 0.102 (CD4T 0.79 + NK 0.21) · Neu 0.174 (Mono 0.88) · Mono 0.178 (Neu 0.84) · NK 0.220 (CD8T 0.63) · B 0.263 |
| GSE103541 (5 types, EPIC) | CD4T 0.065 (CD8T 0.98) · CD8T 0.083 (CD4T 0.95) · Mono 0.173 · Gran 0.197 · B 0.294 |
| GSE35069 (7 types, 450K) | CD8+ 0.099 · **Neu 0.099 (Eos 0.66)** · CD4+ 0.111 · **Eos 0.114 (Neu 0.98)** · CD56+ 0.122 · CD14+ 0.170 · CD19+ 0.266 |

The pattern holds across three studies and both array types:

- Missing T-cell subsets are the least identifiable. Each lies almost on the other.
- **Eosinophils lie almost on neutrophils.** Whole blood scored against a reference without
  eosinophils (GSE110554) loses about 2–4 points of eosinophils. Most of that share is expected
  to be absorbed into neutrophils rather than reported as missing. This is the main
  whole-blood case.
- B cells, and to a lesser degree NK cells and monocytes, leave a large footprint when omitted.

## Does the map predict the model's `other` tracking on GSE110554?

The columns below are the OLS slope (with intercept) and Pearson r of `other` against the true
omitted share across the 12 development mixtures (`gse110554_vs_tracking.csv`). They are read
from the saved estimates; no model was run.

| Omitted type | Residual | v5 slope / r | v6 no cohort slope / r |
|---|---|---|---|
| CD4T | 0.093 | 0.36 / 0.16 | **−0.54 / −0.79** |
| CD8T | 0.102 | 0.11 / 0.18 | 0.32 / 0.76 |
| Neu | 0.174 | 1.18 / 1.00 | 0.86 / 1.00 |
| Mono | 0.178 | 1.32 / 0.87 | 1.88 / 0.95 |
| NK | 0.220 | 0.03 / 0.04 | 0.43 / 0.94 |
| B | 0.263 | 0.60 / 0.67 | 0.86 / 0.91 |

The Spearman correlation between residual and v6 no-cohort slope is 0.71 at 25 and 50 markers
per side, and 0.43 at 12. For v5 it is 0.09–0.49. With six types this is descriptive, not a
test.

- v6's tracking is worst exactly where the map says the information is absent. When CD4T is
  omitted, v6's `other` falls as the omitted CD4T share rises.
- NK is the main exception. It is well separated but only partly tracked (r 0.94, slope 0.43).
- Monocyte omission is over-reported (slope 1.9).

Consequence for claims: report missing-mass performance per omitted type, stratified by this
map. Do not pool it. A failure on near-hull types is the expected limit, not a model defect to
train away.
