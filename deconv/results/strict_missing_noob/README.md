# Strict missing-type benchmark

The reference uses GSE167998 sorted samples. The target uses GSE182379 mixtures.
The run uses Noob betas. Each drop removes every sorted sample of that type
before it filters probes and selects markers. All solvers for that drop use the
same markers. `markers.csv` lists them. `metadata.json` lists reference sample
IDs and marker counts. `estimates.csv` gives sample-level estimates and truth.

Retained-type RMSE uses fractions of the whole mixture. The omitted estimate is
`1 - sum(retained estimates)`. It is signed and not clipped. It is not an
explicitly fitted other component. It can be negative for unconstrained NNLS.
Omitted-fraction RMSE and bias compare it with the true omitted fraction.

The bootstrap resamples the 12 mixture rows. It pairs methods and all drops on
sample ID. It reports 95% percentile intervals for retained-type RMSE, omitted
fraction RMSE, and retained RMSE differences from RPC. These are mixture-row
intervals. The local series metadata does not establish independent donor or
DNA sources for these mixtures. The intervals do not establish donor-level
uncertainty. They do not include reference or marker-selection uncertainty.

The published 12-type IDOL panel is not a strict missing-type panel. It was
optimized with all 12 types. The local source
`research-scout/literature-audit/sources/Guidelines2026.clean.txt`, paragraphs
at lines 123–143, identifies GSE167998 mixtures for training and GSE182379 for
validation. IDOL requires truth-labelled training mixtures. The local loader
combines CD8 subtype truth in GSE167998 mixtures. A strict 12-type IDOL training
run therefore needs the separate subtype truth or original supplemental data,
and must exclude the dropped type from the training design. The installed user
R library did not show an IDOL package. No IDOL run was performed. A future
accuracy comparison must use the same CpGs and exclude GSE182379 truth from
training and tuning. The earlier `baselines_noob/leave_one_out.csv` selected
markers before dropping the type and is not the strict benchmark.

The `ABUNDANT_DROPS_GE_5_PTS` rows include sample/drop pairs where the true
omitted fraction is at least 5 percentage points. The bootstrap resamples all
mixture rows first and then applies this fixed truth-based mask. It preserves
all included drops for each sampled mixture. The table records the number of
included sample/drop pairs. The omitted MAE is the mean absolute difference
between the residual estimate and the true omitted fraction.
