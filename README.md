# In-context methylation deconvolution

Reference-based deconvolution of blood DNA methylation works well when the reference panel contains every cell type in the sample, and badly when a type is missing: classical solvers push the missing type's signal onto the types that remain. This repository contains a reference-conditioned network that takes an arbitrary reference panel (per-type mean and SD over marker CpGs) and a bulk beta-value profile, and returns fractions for each reference type plus an explicit `other` fraction for mass the reference cannot explain. It is trained only on simulated mixtures of public sorted-cell arrays and starts from the robust partial correlation (RPC) fit used by EpiDISH.

> **Status: research prototype with mixed results.** On the held-out test the model does **not** match RPC when the reference is complete, and its improvement when a type is missing is smaller than the pre-declared goal. Read [Results](#results) and [Limitations](#limitations) before using it.

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/AkuSax/in-context-methylation-deconvolution.git
cd in-context-methylation-deconvolution
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests
```

The classical baselines and the Noob preprocessing need R (tested with 4.6.2) with Bioconductor `minfi`, the 450K and EPIC annotation packages, and `EpiDISH`; `paper_solvers.R` also uses EMeth, ICeDT and FARDEEP. Set `RSCRIPT` if `Rscript` is not on your `PATH`. Tests that need R or a trained checkpoint are skipped when those are absent.

## Data

No data or checkpoints are bundled. All inputs are public GEO series; [`deconv/data_manifest.json`](deconv/data_manifest.json) lists the archive URLs, sizes and SHA-256 hashes used.

Files are read from `data/raw/<GSE>/` (override with `DECONV_RAW`), and the Noob cache is written next to it in `data/cache/`.

| Role | Series |
|---|---|
| Simulator training (v6) | GSE35069, GSE103541 |
| Development checks | GSE110554 |
| Held-out test | GSE167998 sorted cells as reference, GSE182379 mixtures |
| Other baselines and stress tests | GSE112618, GSE77797, GSE127824 |

Each loader reads the GEO series matrix for sample labels and, with `DECONV_PREPROC=noob`, the Noob betas built from the raw IDATs:

```bash
for gse in GSE35069 GSE103541 GSE110554 GSE167998 GSE182379; do
  mkdir -p data/raw/$gse
  curl -L -o data/raw/$gse/${gse}_series_matrix.txt.gz \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/${gse:0:-3}nnn/$gse/matrix/${gse}_series_matrix.txt.gz"
done
cd deconv
python reprocess.py all GSE103541 GSE110554 GSE167998 GSE182379   # download IDATs, run Noob
```

The IDAT archives are several GB in total. The role of every dataset, and the rules for keeping test data out of training, are in [docs/evaluation_protocol.md](docs/evaluation_protocol.md).

## Usage

Given a trained checkpoint directory (`model.pt` + `summary.json` from `train_cpu.py`):

```python
import json, torch
from model import Deconvolver
from inference_v6 import estimate_v6

s = json.load(open("results/gpu_v6_nocohort_seed0/summary.json"))["settings"]
model = Deconvolver(width=s["width"], heads=s["heads"], blocks=s["blocks"], cohort=False)
model.load_state_dict(torch.load("results/gpu_v6_nocohort_seed0/model.pt", map_location="cpu"))
model.eval()

# ref_mean, ref_sd: batch x markers x types; bulk: batch x markers (beta values)
pred = model(ref_mean, ref_sd, bulk)
est = estimate_v6(pred)                        # trained missing-type prior (0.55)
est = estimate_v6(pred, known_complete=True)   # if the panel is known to be complete
est.retained_full_mixture, est.other_full_mixture
```

Marker selection (`reference.pick_markers`) and the reference construction used in the paper-style evaluations are in `reference.py` and `eval_real_dev.py`. [docs/inference.md](docs/inference.md) explains the outputs and the `other` prior.

## Reproducing the main results

Run from `deconv/`.

```bash
# 1. Classical baselines, including the strict missing-type benchmark
python run_baselines.py --preproc noob

# 2. Train v6 (no cohort features); about 3 h on one A100
python train_cpu.py --out results/gpu_v6_nocohort_seed0 --seed 0 --steps 40000 \
    --width 96 --heads 4 --blocks 4 --val-per-regime 192 --eval-every 2000 \
    --threads 2 --device cuda --no-cohort

# 3. Development check on GSE110554 (never touches the held-out pair)
python final_test.py --dry-run --checkpoint results/gpu_v6_nocohort_seed0

# 4. Held-out test: GSE167998 reference -> GSE182379 mixtures
python final_test.py --checkpoint results/gpu_v6_nocohort_seed0 --out results/final_test

# Optional: sensitivity of `other` to the assumed missing-type rate
python prior_sweep.py dev
```

The v7 training prior (wider marker range, B-cell and lymphocyte-subtype sources) is behind `--prior v7`; see `python train_cpu.py --help`. Training uses `DECONV_REF_SAMPLING=legacy` by default, which is what the reported checkpoints used.

## Results

Held-out test, run once with the targets fixed in advance: 12-type GSE167998 reference, 12 GSE182379 mixtures, Noob, 50 markers per side, v6 no-cohort checkpoint. RMSE in percentage points on retained types.

| | RPC | v6 (prior 0.55) | v6 (prior 0.10) | v6 (`other` = 0) | Goal | Met |
|---|---:|---:|---:|---:|---|---|
| Complete reference | **1.81** | 2.03 | 1.92 | 1.90 | within 1.10 x RPC | No (1.125 x) |
| One type missing (121 pairs) | 3.21 | **2.65** | 2.77 | 2.88 | 30% below RPC | No (17% below) |
| `other` MAE when a type is missing | 9.25 | **2.88** | 6.12 | 9.25 | at most 5 | Yes |

Intervals are bootstraps over 12 mixtures from one study. The test panels had about 1,000 markers, above the 150 to 600 used in training; this was noticed after the run.

Other findings:

- **v7** (wider marker range and extra training sources) did not improve on a matched v6 control; the difference is within one-seed noise.
- **Detecting an incomplete reference:** on development data `other` separates complete from incomplete references no better than cheap diagnostics (AUROC 0.78, against 0.79 for RPC's residual and 0.82 for CETYGO), and that comparison is confounded by panel size.
- **Clean-reference gap:** most of v6's clean-data loss comes from false `other` induced by the 55% missing-type training prior. Lowering the assumed prior to 10% cuts false `other` on whole blood from about 6 to 2 points but worsens error when a type really is missing. This is a trade-off, not a fix.
- **Real omissions in patient blood (GSE180683, 49 patients):** `other` responds when all CD4 T cells are dropped, but not when a single T-cell subtype is dropped, where it does no better than a constant.
- **Epigenome-wide association (GSE42861, rheumatoid arthritis):** with a cell type removed from the reference, v6 tracks the full-reference association results worse than RPC when neutrophils or CD4 T cells are dropped, and ties when monocytes are dropped. There is no demonstrated application benefit.

## Limitations

- RPC remains the better choice when the reference is known to be complete.
- `other` reflects large omissions only. Closely related types (T-cell subsets, eosinophils versus neutrophils) are nearly collinear in methylation space, so omitting one is largely unrecoverable from the bulk profile.
- `other` depends on the assumed missing-type rate and is not calibrated.
- Evaluation sets are small (12 mixtures per study), donor and DNA independence across series is not fully established, and GSE110554 was examined repeatedly during development.
- Noob does not remove Infinium probe-type bias, and no cross-reactive or SNP probe mask is applied.

## Repository layout

```
deconv/
  datasets.py geo.py fetch_labels.py reprocess.py noob_idats.R   GEO loading and Noob preprocessing
  reference.py solvers.py metrics.py diagnostics.py              marker selection, classical solvers, scoring
  run_baselines.py epidish_*.R paper_solvers.R                   classical and published baselines
  simulator.py rpc_torch.py model.py train_cpu.py                mixture simulator, network, training
  inference_v6.py                                                outputs under an assumed missing-type rate
  eval_real_dev.py label_free_null.py prior_sweep.py final_test.py   evaluation entry points
docs/       evaluation protocol and inference notes
tests/      unit tests on small synthetic inputs
```

Scripts are run from `deconv/` and write to `deconv/results/`, which is not tracked.

## Citation

There is no paper yet. If you use this code, please cite the repository and the methods it builds on:

- Teschendorff et al. (2017). A comparison of reference-based algorithms for correcting cell-type heterogeneity in Epigenome-Wide Association Studies. *BMC Bioinformatics* 18:105 (EpiDISH, RPC).
- The GEO series listed under [Data](#data).

## License

MIT; see [LICENSE](LICENSE).
