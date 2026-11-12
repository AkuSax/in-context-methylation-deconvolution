# Does the sorted-vs-mixture CpG offset follow probe annotation? (2026-11-11)

Post hoc diagnostic, **seventh development look at GSE110554**. It uses the mixture truth
labels, and no protocol was frozen first. No model or simulator was changed. Script:
`deconv/offset_annotation.py`. Files: `run.log`, `summary.json`, `offset_by_annotation.csv`,
`panel_markers.csv`.

Question: the v5 clean penalty is false `other` driven by a per-CpG offset between sorted
arrays and mixture arrays (HANDOFF step 5). If the offset could be predicted from fixed probe
properties (design type, colour channel, CpG-island context, SNP and Zhou masks), those
properties could become CpG features for the model and could shape the simulator's offsets.
These are option 1 of the external-data review.

## Verdict: no useful edge from probe annotation

- **Within GSE110554 the offset is real.** The offset is mixture − reference × truth, using
  all 841,502 autosomal complete CpGs (Noob). Split half and half, with disjoint sorted
  arrays and disjoint mixtures, the signed M-scale offset correlates at Pearson 0.64.
  Its magnitude replicates less well (Spearman 0.32), so most CpGs carry only a small offset.
- **Annotation adds almost nothing beyond what the model already sees.** The model already
  gets the bulk beta level, the reference SD and the between-type range. Fits are
  cross-validated by chromosome parity:

  | Target | Features the model sees | + annotation |
  |---|---|---|
  | \|M offset\|: R², AUROC for the top 5% | 0.345, 0.890 | 0.352, 0.893 |
  | Signed M offset: R², AUROC | 0.160, 0.742 | 0.166, 0.757 |

  Beta level does most of the work. Mid-beta CpGs carry the largest offsets. The sign also
  follows beta level: CpGs with bulk beta ≤ 0.1 (mostly islands) shift down, with only 26%
  positive, while the rest shift up, 59–65% positive. That pattern fits a monotone warp
  between the two sets of arrays.
- **Zhou-masked probes do carry larger offsets.** Their top-5% rate is 0.148 for
  `M_general` against 0.045 for the rest. Masked probes are rare among markers, though:
  20 of the 285 primary-panel CpGs, and only 1 of the 26 recurrent ones below.
- **The primary 25-per-side panel (285 CpGs).** 26 CpGs have |z| > 3 in at least 3 of the 12
  mixtures, and they hold 146 of the 172 such cells. The v4 README counted 35 under a
  different definition. Of those 26:
  - 22 have mid-range beta (0.3–0.7);
  - type I probes are enriched: 10 of 56 type I against 16 of 229 type II (Fisher OR 2.9,
    p 0.018);
  - shores are over-represented: 6 of 20.

  Beta level is already visible to the model. The type I flag is the one fixed annotation
  with a signal here, and it is weak.
- **The offset barely transfers to another study.** As a label-free check, GSE35069's
  whole-blood and PBMC arrays were fitted against its own sorted fractions by NNLS
  (450K, deposited processing). Against GSE110554 at 418,311 shared CpGs the offsets correlate
  at Pearson 0.21 (signed) and Spearman 0.06 (magnitude), or 0.05 within bulk-beta bins. A
  fixed table of which probes are "batch-prone" would not carry from one batch to the next.
  - The exception is the worst CpGs. The 8 recurrent GSE110554 panel CpGs that are also on
    450K sit at the 95th percentile of GSE35069's offset magnitude. That is n = 8, so treat it
    as a hint.
  - The design-type effect depends on preprocessing. In GSE35069 (deposited), type I probes
    carry twice the type II offset (median 0.13 vs 0.06 log2). In GSE110554 (Noob) the two
    are the same (0.081).

## What this means for the model

- Don't add annotation-derived CpG features or a per-probe reliability table. The one
  possible exception is a type I/II flag, which BACKGROUND.md §5 already lists as a token
  feature and which costs nothing.
- The per-CpG offset is batch-specific. A model therefore can't tell it apart from a missing
  type using fixed per-CpG knowledge. It needs evidence from the batch itself. One kind is
  cohort-level input: a batch offset repeats in every sample of a batch, while a missing type
  scales with its fraction. The other is real cross-study pairs in the simulator (HANDOFF
  proposed steps 3 and 4). This result pushes priority toward those.
- The simulator can tie offset size and sign to beta level, as observed here (larger at
  mid-beta, negative below 0.1). Whether v5's M-value warp already covers this was not
  checked.

## Limits

- One batch with labels (GSE110554). The GSE35069 comparison mixes platform, processing,
  a label-free fit, and whole-blood content outside the sorted set (basophils, for example).
- GSE110554's mixtures may share DNA with its sorted arrays (EVALUATION.md). The split-half
  test separates arrays, not DNA sources.
- Published replicate-based probe-reliability tables (ICC, array bias) were not downloaded.
  Given how weakly the offset transfers across studies, they are unlikely to change the
  verdict. That has not been tested.
