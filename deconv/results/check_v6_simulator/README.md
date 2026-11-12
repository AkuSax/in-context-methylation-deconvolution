# v6 simulator realism check (2026-11-12)

Script: `deconv/check_v6_simulator.py` (run from `deconv/`; about 2 min with a saved panel
file). Output: `summary.csv`. Held-out donors only (3 of GSE35069, 7 of GSE103541), frozen v5
checkpoint, no reserved data.

Each row: 12 cohorts × 12 samples, five types, 250 `pick` markers, reference donors never in
the mixtures. "Shift off" is the measured study difference alone; "shift on" adds the v5
synthetic shift, as in training. Points, except the marker range.

## Clean (no type missing)

| Source | Shift | Marker range | RPC kept RMSE | RPC unexplained mass | v5 false `other` |
|---|---|---|---|---|---|
| GSE35069 alone, sharpened (the v5 source) | off | 0.90 | 3.1 | 0.7 | 2.6 |
| | on | 0.91 | 4.1 | 3.4 | 4.2 |
| GSE103541 alone | off | 0.86 | 1.2 | 0.6 | 2.4 |
| | on | 0.87 | 1.6 | 1.0 | 3.1 |
| Reference GSE35069 → bulk GSE103541 | off | 0.77 | 3.9 | 2.6 | 4.5 |
| | on | 0.77 | 3.5 | 3.7 | 3.6 |
| Reference GSE35069 sharpened → bulk GSE103541 | off | 0.92 | 1.9 | 15.3 | 21.3 |
| | on | 0.90 | 2.3 | 15.0 | 18.1 |
| Reference GSE103541 → bulk GSE35069 | off | 0.86 | 3.1 | 4.3 | 4.6 |
| | on | 0.87 | 4.4 | 4.6 | 7.4 |
| **Real, in batch** (GSE110554 clean mixtures) | | 0.88 | 1.7 | 0.7 | 6.4 |
| **Real, out of batch** (four whole-blood cohorts) | | | | 9.5–16.6 | 8–19 |

## One of five types missing

v5 kept RMSE is 3.2–6.7 against RPC 9.3–18.1 in every source except the sharpened
cross-study one (8.1–10.8 against 12.7–13.8), where v5's `other` overshoots the truth by 6–8
points.

## Reading

- **GSE103541 alone is the closest match to real in-batch data** on RPC error (1.2–1.6 vs 1.7),
  marker range (0.86 vs 0.88) and unexplained mass (0.6–1.0 vs 0.7). v5 already behaves there
  (2.4–3.1 false `other`), better than on real GSE110554 (6.4).
- **A reference more extreme than the bulk reproduces the whole-blood failure.** Stretching
  only the GSE35069 reference leaves the GSE103541 bulk compressed against it: RPC's
  unexplained mass is 15 and v5 reports 18–21, the range seen on the four cohorts (10–17 and
  8–19). This supports the reading that the cohort mismatch is a range compression between a
  Noob reference and deposited betas, and it is the one v6 source that puts that failure in
  the training prior. It is a stand-in built from a stretch, not a measured second pipeline.
- **The plain measured cross-study difference is mild for v5** (false `other` 3.6–7.4). The
  type-specific offset seen in `data_gse103541` raises RPC error to 3–4 points but does not
  by itself make v5 report much missing mass.
- The synthetic shift on top changes little, so keeping it costs nothing.

## Limits

One seed and 144 samples per row; cohort-level effects dominate, so differences under about
1 point are not meaningful. The five-type drop is dominated by granulocytes (true `other`
14–28 points on average). The GSE35069 side is deposited 450K data from three validation
donors.
