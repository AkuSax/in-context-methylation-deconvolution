# Evaluation protocol

Status: the user approved the three primary accuracy targets on 2026-11-11
before model training. The secondary targets remain exploratory.
The existing baseline numbers in `HANDOFF.md` are exploratory. They do not make
the final set unseen by the project. Do not tune the model to those numbers.

## Data roles and provenance

| Data | Fixed role | Independence and limits |
|---|---|---|
| GSE35069 and synthetic types | Simulator training and development | GSE35069 has six donors with seven non-overlapping sorted types. The simulator keeps three donors in each partition. It uses the deposited betas and labels artificial types. |
| GSE103541 | Simulator training and development (added 2026-11-11, approved by the user as a training source) | 27 usable donors with five sorted types each, EPIC, Noob. It has no fraction truth. Never use it as a test set. 141 of its arrays are also in GSE166844, so treat GSE166844 as the same source. |
| GSE131989 | Candidate simulator source | Donor identity across sorted types needs verification. The proposed title-based join gives conflicting ages for matched samples. |
| GSE110554 | Real calibration development; six-type secondary descriptive test only before calibration use | Exclude its entire series from simulator and model training. Its 37 sorted arrays include one technical replicate. It has 12 mixtures. The same study supplies reference and mixtures, and does not establish independent donors or DNA for this use. Do not call its calibration an independent final test. |
| GSE167998 sorted samples + GSE182379 mixtures | Primary final test, 12 types | Exclude both entire series from simulator training, validation, marker tuning, model choice and calibration. GSE167998 supplies the reference at inference only. GSE182379 supplies all 12 final mixtures. |
| GSE167998 mixtures | Exploratory only | Reference and mixtures are from the same source study. Their DNA or donor independence from sorted reference samples is not established. Truth has coarser CD8 labels. Never merge these mixtures into the primary test. |
| GSE112618 and GSE77797 whole blood | Paired adult platform stress test | Six matched flow records, each within 0.005 percentage points across series. Treat each match as one paired unit. Flow agreement alone does not prove same donor or same DNA. Verify sample links before calling these same-donor or same-DNA assays. |
| GSE77797 constructed mixtures | Secondary 450K test | Keep its 12 mixtures separate from its six whole-blood records. Audit source DNA and donor overlap before an independence claim. |
| GSE127824 | Secondary cord-blood stress test | Keep all 24 samples and labels out of training and calibration. Adult reference mismatch and missing nRBC occur together. This is not an isolated test of either effect. |

The [2022 source study](https://www.nature.com/articles/s41467-021-27864-7)
names GSE182379 as its independent validation series. It states that its 12
validation mixtures came from a set of 12 isolated cell types not used in its
training or testing mixtures. It does not establish that every reference donor
differs from every validation donor. The 12 mixtures are not proven to use 12
independent DNA pools or 12 independent donors. GSE167998 and GSE182379 belong
to the same research program. Do not call this an independent-study or proven
independent-donor test. The [GEO reference record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE167998)
also lists the within-series mixtures. The [GEO validation record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE182379)
lists 12 validation mixtures.

The [2018 source study](https://link.springer.com/article/10.1186/s13059-018-1448-7)
describes GSE112618 and GSE77797 as separate sets of six flow samples. The local
matching flow values are evidence of a link, not proof of sample identity. Save
the six-row sample-ID and flow-value match table before evaluation. Report both
the published description and the local match. Do not infer a pure platform
effect from this pairing until source DNA identity and other assay conditions
are verified. The [GSE110554 record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE110554)
describes its sorted samples and two sets of artificial mixtures. It does not
prove independent donor or DNA sources for the intended split.

## Lock before scoring

1. Freeze the model, seed, preprocessing, type map, sample exclusions, marker
   algorithm, evaluation script and all comparison rules. Record versions and
   input hashes. Log every use of real labels. Do not select a model, marker
   count or solver from final outcomes.
2. Use Noob and detection-p masking for all six truth series. State that Noob
   does not remove Infinium type bias. No cross-reactive or SNP mask is applied
   in the current baseline. Freeze any added mask before scoring.
3. Use the same reference, target samples, type map and CpG list for all methods
   within each comparison. Select markers from reference labels only. A frozen
   label-free detection mask may use target availability. Record that
   transductive use and the number of retained CpGs. Never use target truth to
   select markers. Use a fixed tie rule and seed.
4. For each dropped type, remove it before marker selection and reference
   construction. Select a new common CpG list from the remaining reference.
   Do not retain markers chosen with the dropped type present. Freeze the
   selection rule, including how a method that cannot use a CpG is handled.
5. Score RPC, NNLS, constrained projection and other available classical
   methods. The primary comparator is RPC fixed in advance. Do not choose the
   best comparator after seeing the final results. Show every method failure
   and its reason. Do not silently change the evaluated sample set.

## Primary clean test

Use all 12 GSE182379 mixtures and all 12 truth types. Let `e[i,k]` be
100 times the estimated minus true fraction. The unit is a percentage point of
the full mixture. Define pooled RMSE as `sqrt(mean(i,k, e[i,k]^2))`, with equal
weight for each of the 144 sample-type cells. Fix missing predictions as
failures, not excluded cells. Report each sample and type count.

The approved margin is a 1.10 RMSE ratio to RPC. The saved RPC result of 1.806
points gives a target of about 1.99 points on the same CpGs. Let
`D = RMSE(model) - 1.10 * RMSE(RPC)`. The model meets this goal only if the
upper one-sided 95% paired bootstrap bound for D is below zero. Resample the 12
mixture rows as complete vectors. Recompute both RMSE values in each of 10,000
seeded bootstrap replicates. Use the 95th percentile of D as the one-sided
bound. Report the observed RMSEs, ratio, D and paired 95% two-sided effect
interval as well. A zero RPC RMSE makes the ratio undefined; report D and do
not claim ratio non-inferiority. This row bootstrap assumes mixtures are
independent sampling units. Shared DNA pools or composition design can violate
that assumption. Label the interval conditional and exploratory until that
structure is verified. It does not establish population-level non-inferiority.

## Missing types and other stress tests

For each of 12 types, remove that type from the reference and rescore all 12
mixtures. Predefine an abundant drop as truth at least 5 percentage points in
that mixture. Freeze the eligible sample-drop pairs
from truth before model results are inspected. Report their count and each
type's count. Include only the 11 kept types in the primary missing-type RMSE:
`sqrt(mean(i,d,k != d, e[i,d,k]^2))` over eligible sample-drop pairs. Give each
eligible pair and each kept type equal weight. Do not include the dropped type
or `other` in that RMSE. Compare model and RPC on exactly these same pairs and
types. Report each drop separately and the all-drop result. The approved goal
is a 30% or greater reduction versus RPC, with paired effect intervals. The
strict Noob baseline gives RPC 3.205 points over 121 eligible pairs. The approved
30% reduction corresponds to an RMSE of at most 2.244 points.
[Strict benchmark](results/strict_missing_noob/README.md)

For `other`, its truth is the dropped type's original fraction. Report its
signed bias, MAE, RMSE, and sample-level errors. The approved goal is MAE at
most 5 points. Do not infer `other` from `1 - sum` for a method that enforces
sum = 1 and then claim it detects missing mass. Report zero residual mass for
such a method as its defined output. Include residual mass for sum-at-most-one
methods, labelled as a diagnostic rather than a verified missing-type estimate.
No absent or unmeasured type has known truth unless the dataset supplies it.

For platform stress, compare EPIC and 450K target profiles within each of the
six matched flow records. Use one fixed reference, type map and common CpG list
across both platforms and all methods. Report the paired change in error and
the individual pairs. Resample pairs, not 12 platform records. The provisional
20% gain goal applies to a predeclared pooled RMSE comparison on the 450K
target against RPC. It is not evidence of a platform-specific causal effect.
For cord blood, report kept-type RMSE and nRBC as `other` only when its truth
and denominator match the output. The provisional goal is 30% lower kept-type
RMSE than RPC and nRBC recovery. Do not attribute a gain only to nRBC.

For every regime also report MAE, signed bias, per-type RMSE, worst-type RMSE,
and Pearson r only where both variables vary. Report rare-type results and
sample-level errors. Correlation does not replace error. Map types and truth
denominators before scoring. Do not renormalize truth to kept types for a
missing-type test. Report partial or incompatible flow truth separately.

For repeated drops, retain all drops and types for a sampled mixture in each
bootstrap replicate. If source DNA groups are identified, resample those groups
instead. For six platform pairs resample whole pairs. Report 95% paired
percentile intervals for RMSE differences and percentage changes, with 10,000
seeded replicates. Do not interpret sample-type cells or drop scenarios as
independent replicates. State the number of distinct clusters. With few or
unknown clusters, intervals can be unstable or too narrow. Show raw results and
do not make formal superiority claims. All stress tests are secondary. Do not
pool their results into a single confirmatory claim.

## Prediction intervals and calibration

Benchmark effect intervals above describe uncertainty in a dataset-level
comparison. Per-sample prediction intervals describe uncertainty about a
sample's true fraction. They are different quantities.

The primary report uses the synthetic-only trained model and its uncalibrated
real-data predictions. Also report a distinct real-recalibrated model if
calibration is feasible. Fit a single frozen split-conformal interval rule to
GSE110554 constructed mixture residuals only, after model and score are fixed.
Use each entire mixture as the unit. A proposed score is the maximum normalized
absolute residual over the six types. Set a positive scale floor before seeing
calibration labels. For 90% nominal coverage with 12 calibration rows, use
rank `ceil((12 + 1) * 0.90) = 12`, the largest score. If the rank exceeds the
available count, use an infinite bound. Clamp fraction intervals to [0, 1].
This gives a simultaneous six-type target under exchangeability of independent
calibration and future rows. Shared DNA and reference, unknown donor overlap,
and transfer from six-type GSE110554 to 12-type GSE182379 break or leave that
assumption unverified. Thus this is an empirical transfer experiment, not a
90% guarantee for the primary test. Never split mixtures that share source DNA
across calibration and test and claim independent calibration. No final labels
may set the scale, score or quantile.

Report marginal coverage, simultaneous per-mixture coverage, mean interval
width, and counts by type and regime before and after recalibration. Report
coverage against constructed DNA proportions and flow measures separately.
Flow measurement error is part of the observed disagreement. Report binomial
intervals only as descriptive if independence is justified; otherwise report
counts and dependence limits. At 12 mixtures, one simultaneous failure changes
coverage by 8.33 points. Even 11/12 = 91.7% cannot establish that population
coverage lies between the provisional 85% and 95% limits. Compare width with
the Houseman bootstrap only at the same nominal level, types and sample set.
The provisional goal is 90% nominal intervals with observed coverage of
85–95% and lower mean width than that bootstrap. This goal is descriptive,
not evidence of calibrated population coverage. If a bootstrap method is not
available, mark the width comparison unavailable.

## Decision and reporting

Use the three approved accuracy targets for the first real-data test. Record
any change with its date and reason. Report all results
even if a goal fails. Include the sample and reference IDs, hashes, CpG counts,
seeds, solver failures, per-sample estimates and uncertainty intervals. State
which analyses used real labels for calibration. A specialist control uses a
fixed reference and synthetic training only; compare it on the same test rows,
CpGs and outputs. Its provisional goal is model RMSE within 10% of that
control. If it requires the final reference in training, label it a separate
reference-specific comparator and do not claim a shared held-out-reference
training condition.
