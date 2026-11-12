# GSE110554 development check of gpu_v2: result

Protocol: [PROTOCOL.md](PROTOCOL.md), frozen before scoring. Script: `deconv/eval_real_dev.py`.
Run 2026-11-11 (4.5 min on CPU). Files: `summary.csv`, `estimates_long.csv`.

## Verdict: the model does not transfer to real data yet

The failure is specific. On clean real mixtures the model assigns about **34 points to
`other`** (truth 0), shrinking every kept type. Kept-type RMSE is 8.3 points vs RPC 1.7.

| Primary panel, 25 per side (~286 CpGs) | Model | RPC | NNLS | CP | RPC shrunk |
|---|---|---|---|---|---|
| Clean kept RMSE | 8.25 | **1.72** | 1.83 | 1.55 | 3.10 |
| Clean `other` (truth 0) | 34.0 | 0 | 0 | 0 | 9.1 |
| Drop-one kept RMSE (72 pairs) | 7.54 | 10.74 | 10.45 | 9.97 | 9.76 |
| Drop-one `other` MAE | 25.1 | 16.7 | 16.7 | 16.7 | 11.1 |
| Detection AUROC, clean vs abundant drops | **0.80** | — | 0.50 | 0.52 | — |
| 90% coverage, width | 0.81 / 0.88, 27 pts | | | | |

The 150- and 600-CpG panels give the same picture (clean kept 8.6 / 8.3; clean `other`
32.8 / 35.4).

The drop-one kept-RMSE "win" (bootstrap model − RPC [−4.8, −1.6]) is **not a real gain**.
Per drop, the model is worse than RPC for B, CD4T, CD8T, Mono and NK. It wins only when
neutrophils are dropped (43 points of truth), where its constant high `other` happens to be
right.

The ranking does transfer: AUROC 0.80, where the classical leftover-mass detectors score
0.50. The level does not.

## Cause (input-feature check)

The RPC-fit inputs on the real clean panel against 64 simulated tasks at the same K and M:

| Feature | Real clean | Sim clean | Sim mismatch | Sim missing + mismatch |
|---|---|---|---|---|
| RPC intercept | **0.028** | 0.001 ± 0.008 | 0.003 ± 0.014 | 0.004 ± 0.018 |
| log(robust scale)/4 | **−0.815** (scale 0.038) | −1.03 ± 0.07 (0.016) | −0.90 ± 0.05 (0.027) | −0.89 ± 0.04 (0.029) |
| raw slope sum − 1 | −0.002 | −0.001 ± 0.012 | −0.001 ± 0.017 | 0.002 ± 0.021 |
| share of down-weighted CpGs | 0.22 | 0.20 | 0.21 | 0.22 |

Real mixtures differ from their own sorted reference by a constant offset (intercept 0.028,
about 2–3 SD beyond simulation) and by more residual noise (scale 0.038, above every
simulated regime). In simulation, larger residuals mean a missing type, so the model reads
real technical noise as missing mass. The simulator's bulk noise (0.5–4% × √(β(1−β)), at
most about 0.02 SD) and its lack of any reference-to-bulk offset are too clean.

## Implication

The v3 simulator needs reference-to-bulk technical shift: noise up to the real scale or
beyond, a global offset, and per-CpG batch offsets or warps (the platform-shift prior in
BACKGROUND.md §5). Then large residuals alone stop meaning a missing type. GSE110554 has now
been looked at once for this model family; a v3 recheck here is a second development look
and must be labelled so.
