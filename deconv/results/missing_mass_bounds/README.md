# Missing-mass lower bound and cohort residual: GSE110554 drop-one prototype

Run 2026-11-12 from `deconv/missing_mass_bounds.py` (SHA-256 `895fa278…3dae0e`, on top of
commit `b387fae`), `python missing_mass_bounds.py`, about 100 s on CPU. No model was
trained or run. GSE110554 labels were used only to score results. This is another
development look at that series.

## Question

Without assuming what the omitted cell type looks like, how much of its fraction can the bulk
and the reference reveal? Two training-free estimators:

1. **Per-sample lower bound.** In `b = R v + u q` with an unknown omitted profile `q` in
   `[0,1]^p`, the set of `u` values consistent with the data is `[u_lb, 1]`. `u_lb` is one
   linear program. The relaxation `eps_j = z · max_k sd_jk + 0.01` allows for noise.
2. **Cohort residual.** Remove span(R) from every bulk sample. If one omitted type with a
   varying fraction is shared, the residuals are rank one, and the PC1 scores track the
   omitted fraction up to an unknown scale. An "anchored" scale uses the 0–1 bound on `q`.

CpG sets: `panel` (markers re-picked on the retained types, strict protocol) and
`panel+var` (plus the 2000 CpGs with the highest bulk variance, selected without labels).

## Results

**1. The noise-free bound is not valid on real data, and the relaxed bound is barely
informative.** Mean `u_lb` on the 25-per-side panel (true missing fraction in brackets):

| z | clean (0) | NK (.09) | B (.09) | CD4T (.13) | CD8T (.13) | Mono (.13) | Neu (.43) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .44 | .43 | .45 | .48 | .40 | .46 | .61 |
| 2 | .16 | .21 | .20 | .15 | .18 | .26 | .44 |
| 3 | .09 | .15 | .15 | .10 | .12 | .23 | .39 |

At z = 0, a clean sample "needs" 44% unknown material. Mismatch between the sorted-cell
reference and the mixtures pushes some CpGs outside every retained type's range. The bound
takes the maximum over CpGs, so it is set by the worst one. With the `panel+var` set it gets
worse (clean 0.45 at z = 3): more CpGs give more chances for a violation. Above the clean
level, the drop panels add only 2–10 points. The bound almost never covers the truth for
the small types (`lb_valid` mostly 0 in `summary.csv`). The mathematics holds; the
assumption it rests on, that the retained profiles describe the target exactly up to noise,
does not. That is the audit's `delta` ambiguity in concrete form.

**2. Cohort PC1 does not separate clean from incomplete panels.** The PC1 share of residual
variance is 0.44 for the clean panel and 0.46–0.60 for the NK/B/T/Mono drops. Only the
neutrophil drop stands out (0.84). The anchored scale fails, as expected: per-CpG offsets
shared by all samples enter the uncentered scores, and `anchored_valid` sits near 0.5.

**3. High within-drop correlations are mostly composition, not missing mass.** The 12
mixtures trade neutrophils off against everything else (Neu vs NK r = −0.91, Mono −0.87,
B −0.83, CD8T −0.77; `truth_correlation.csv`). Any score that grows when neutrophils are few
therefore "tracks" those omitted fractions. In `confound_control.csv`, `r_clean_same_type`
is the score's correlation with the type's fraction on the **clean** panel, where nothing is
missing:

| Score (25/side) | NK: r drop / r given Neu / r clean | B | CD8T | Mono |
|---|---|---|---|---|
| Cohort PC1 | .97 / .91 / .94 | .96 / .97 / .85 | .79 / .30 / .76 | .91 / .71 / .84 |
| CETYGO | .95 / .73 / .77 | .97 / .91 / .57 | .89 / .72 / .79 | .91 / .56 / .53 |
| Lower bound, z = 2 | .74 / .27 / .15 | .82 / .47 / .13 | .67 / .55 / .49 | .90 / .53 / .28 |
| v5 `other` | .04 / .73 / −.66 | .67 / .67 / −.70 | .18 / .52 / −.45 | .87 / .81 / −.61 |

Cohort PC1 and CETYGO reach clean-panel correlations similar to their drop-panel ones.
These correlations therefore cannot show missing-mass tracking. The lower bound is the
least confounded (clean r 0.13–0.28 for NK, B and Mono), but its partial correlations are
modest. v5 `other` is confounded in the opposite direction: high on neutrophil-rich clean
samples. After adjusting for neutrophils, it tracks NK and B better than its raw
correlation suggests. With 12 samples (9 degrees of freedom), every partial r here has a
wide spread. CD4T is uninformative throughout, consistent with its small hull distance in
`results/identifiability_map/`.

## Reading

- Without assumptions about `q`, the box-constraint bound is the only honest per-sample
  statement. On these arrays, reference/target mismatch dominates it. A usable version first
  needs an explicit mismatch model. A per-CpG offset estimated from the cohort is one option,
  but it absorbs any nearly constant omitted fraction (the audit's caveat). A trimmed bound
  that lets a fixed share of CpGs violate the box is another.
- **GSE110554 cannot answer whether a score tracks missing mass**, for any score including
  the learned models. Its design makes the omitted fractions collinear with the neutrophil
  fraction. Earlier within-drop correlations (the audit's 0.16/0.18/0.04 for v5, and the
  CETYGO/RPC values in `existing_scores_r.csv`) should be read with this in mind.
- A decisive test needs omitted fractions that vary independently of the rest of the
  composition. Candidates are in-silico mixtures with decorrelated designs, built from
  held-out sorted donors (GSE103541 profiles against a GSE35069 reference, for cross-study
  mismatch), or a physical mixture series designed that way.

## Files

| File | Contents |
|---|---|
| `per_sample.csv` | Per panel × CpG set × z × sample: true omitted fraction, `u_lb`, PC1 score, anchored estimate |
| `summary.csv` | Per panel: mean bound, coverage, correlations, PC1 share, `s1/s2`, `lam_max` |
| `confound_control.csv` | The composition control above, for both marker settings |
| `existing_scores_r.csv` | Within-drop r of the earlier detection scores (from `results/detection_baselines/`) |
| `truth_correlation.csv` | Correlations among the 12 mixtures' true fractions |
| `run.json`, `run.log` | Settings and console output |
