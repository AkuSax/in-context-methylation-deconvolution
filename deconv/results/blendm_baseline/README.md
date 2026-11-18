# BLEND-M baseline

Run 2026-11-12/13 with `python blendm_baseline.py` on CPU. The development part took
5 min and the synthetic part about 2 h. The design was fixed before the run.

**Code.**
- BLEND-M is `Penghuihuang2000/BLENDM` at commit `90546bc`, vendored in
  `../../vendor/BLENDM/`. The only edit replaces the `rlist` calls.
- It is called through `blendm_solve.R` and `blendm.py`.
- The Python port of BLEND-M's marker rule picks the same 184 of 3,000 GSE110554 probes as
  the R function (`python blendm.py --check`).
- The RPC numbers reproduce `real_check_gpu_v6`: clean 1.72, drop-one 10.74, and every
  per-drop value.

**Common settings.**
- The unit is RMSE in percentage points of the full mixture, over retained types.
- Every method in a row uses the same CpGs.
- BLEND-M gets individual reference samples; the other methods get the per-type mean. This
  is an unequal-input advantage by design.
- "project" markers are 25 per side from `reference.pick_markers`. "blendm" markers come
  from BLEND-M's ANOVA plus effect-size rule.

## Part 1: synthetic, same mixtures as `decorrelated_missing` (no real labels)

**Design.**
- Mixtures are drawn from the 7 held-out GSE103541 donors: 60 per omitted type, 5 seeds.
- `within`: the reference comes from 20 other GSE103541 donors.
- `cross`: the reference comes from the 6 GSE35069 donors.
- `full` means a complete reference. In `cross` it is the complete-but-shifted case.
- `reduced` means the omitted type is removed before marker selection.
- RMSE is over the four types that are always retained. Values are means over seeds and
  drops; seed SD of the paired differences is ≤ 0.1 points except against v6 (≤ 0.34).

| Scenario, reference | BLEND-M | BLEND-M, no `unknown` | RPC | CP (sum ≤ 1) | NNLS | v6 no-cohort |
|---|---:|---:|---:|---:|---:|---:|
| within, full | **0.56** | 0.55 | 0.82 | 0.79 | 0.79 | 1.23 |
| within, reduced | 7.72 | 7.87 | 7.80 | 7.44 | 7.81 | **5.51** |
| cross, full | 2.08 | **1.85** | 4.02 | 2.96 | 3.03 | 6.36 |
| cross, reduced | 6.77 | 7.26 | 7.76 | 7.44 | 8.03 | **6.35** |

Project markers. With BLEND-M's own markers (10–15k CpGs):

| Scenario, reference | BLEND-M | RPC | CP |
|---|---:|---:|---:|
| within, full | 0.43 | 0.79 | 0.68 |
| within, reduced | 7.64 | 7.55 | 7.40 |
| cross, full | 1.93 | 2.95 | 4.05 |
| cross, reduced | 7.72 | 7.54 | 8.43 |

**Cross-study, reduced, project markers, by omitted type:**

| Omitted type | BLEND-M | RPC | v6 |
|---|---:|---:|---:|
| B | 4.11 | 5.61 | 6.57 |
| CD4T | 7.16 | 7.83 | 6.93 |
| CD8T | 7.58 | 8.35 | 7.91 |
| Gran | 7.72 | 8.54 | 5.56 |
| Mono | 7.27 | 8.49 | 4.78 |

**Missing mass.**
- BLEND-M's built-in `unknown` component (an all-ones reference column) is not a
  missing-mass estimate.
- MAE against the omitted fraction:
  - cross, project markers: 0.09–0.15;
  - within: 0.14–0.16;
  - oracle constant: 0.075;
  - v6: 0.065–0.115.
- On BLEND-M's markers, `unknown` is about 0.
- On the complete-but-shifted reference, `unknown` averages 3.3% where the truth is 0. v6
  averages 8.8%.

## Part 2: GSE110554 real development (12 mixtures)

| Method, markers | Clean | Drop-one, all | Drop-one, no Neu | Drop-one `other` MAE |
|---|---:|---:|---:|---:|
| BLEND-M, project | 1.87 | 9.77 | 5.61 | 13.1 |
| BLEND-M no `unknown`, project | 1.80 | 10.02 | 5.91 | 16.7 |
| RPC, project | 1.72 | 10.74 | 5.92 | 16.7 |
| CP, project | 1.55 | 9.97 | 5.86 | 16.7 |
| BLEND-M, blendm markers | 1.76 | 9.24 | 4.97 | 14.2 |
| RPC, blendm markers | **1.04** | 10.00 | 5.39 | 16.7 |
| v6 no-cohort (`real_check_gpu_v6`) | 2.29 | **4.03** | **4.31** | **5.2** |

Per-drop values are in `dev_card_view.csv`.

**Limits.**
- N = 12 mixtures is below the authors' N ≥ √(markers) guideline for estimating dispersion.
- The reference and the mixtures come from one study.
- GSE110554 has now been looked at twelve times.

## Reading

1. **BLEND-M solves the complete-but-shifted case better than every comparator here.**
   - Cross-study full reference: half of RPC's error (2.08 against 4.02 points, 25/25
     seed-drop cells).
   - It also beats RPC within a study.
   - v6 is worst in this case (6.36), because it puts 8.8% into `other` when nothing is
     missing, against 1.9% within a study and 5.9% on clean GSE110554. This is the same
     clean penalty, and it is largest across studies.
2. **Neither BLEND-M nor any classical solver handles a truly missing type.**
   - v6's drop-one advantage survives BLEND-M on real development data (4.03 against
     9.77) and within-study synthetic data (5.51 against 7.72).
   - Across studies v6 is only marginally ahead (6.35 against 6.77). It wins on Gran and
     Mono and loses on B and CD8T.
   - v6 is in-distribution here: it trained on these two studies, and the mixture donors
     were its checkpoint-selection set.
3. **The learned model's defensible niche is missing types, not mismatch robustness in
   general.** A model that cannot match BLEND-M when nothing is missing will lose overall
   on references that are usually complete. This is evidence for path D: lower the
   training rate of missing types, or condition on it.
4. **Marker choice matters as much as the solver.** RPC on BLEND-M's markers is much better
   than RPC on the project markers:
   - real clean data: 1.04 against 1.72;
   - synthetic cross-study full reference: 2.95 against 4.02.

   This has not been tested under a frozen rule. GSE110554 cannot be used to choose it. It
   would need a dated protocol amendment, like the audit's panel-quality sensitivity
   analysis, before any final scoring.

## Not run here

GSE182379 (primary final) and GSE127824 (cord stress test) were not scored. The card
declares BLEND-M, with `unknown.include` TRUE and FALSE, as a secondary comparator for the
locked evaluation on both, using the same reference donors and markers as RPC.
