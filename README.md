# In-context methylation deconvolution

Can a model estimate the cell types in a DNA methylation mixture when it is given a reference panel it has never seen before? This project explores that question for blood methylation arrays, especially when a cell type is missing from the reference.

The model takes a reference panel and a bulk methylation profile and predicts cell-type fractions plus an unexplained “other” fraction. It is trained on simulated mixtures of sorted-cell profiles and receives the output and residuals of robust partial correlation (RPC), a strong classical method. The goal is to handle reference mismatch while remaining competitive on clean mixtures.

This is ongoing course research, not a validated analysis tool. The code is experimental, and its uncertainty estimates have not yet been calibrated on independent real data.

## Where things stand

We have built a benchmark on public, truth-labelled blood datasets and compared the model with RPC and other classical methods. In a **development** experiment on GSE110554, the v5 model reduced retained-type RMSE from 10.74 to 4.00 percentage points when one reference type was removed. On clean mixtures it scored 2.84 versus RPC's 1.72, and assigned 6.4 percentage points to “other” on average when no type was omitted. The missing-type gain is smaller when the neutrophil drop is excluded, and “other” does not reliably track every omitted type. These results are exploratory, from a small set of constructed mixtures, and do not establish generalization to new studies.

A newer simulator adds a second sorted-cell source and cohort-level information; it has been checked locally but has not yet completed the planned training and final evaluation. The primary 12-type evaluation, calibration, and broader stress tests remain future work. See [EVALUATION.md](deconv/EVALUATION.md) for dataset roles, controls, and limitations; the [development audit](deconv/results/audit_gse110554_gpu_v5/README.md) and [strict missing-type benchmark](deconv/results/strict_missing_noob/README.md) give more detail.

## What's here

- `deconv/datasets.py`, `geo.py`, `reprocess.py`, and `noob_idats.R`: GEO loading and Noob preprocessing.
- `deconv/reference.py`, `solvers.py`, and `run_baselines.py`: reference construction and classical benchmarks. Optional R integrations add published solvers.
- `deconv/simulator.py`, `rpc_torch.py`, `model.py`, and `train_cpu.py`: simulated tasks and the reference-conditioned model.
- `deconv/eval_real_dev.py` and related scripts: development checks and audits.
- `deconv/results/`: saved summaries, estimates, and experimental logs. Raw data, cached matrices, and model checkpoints are not included.
- `tests/`: unit tests for data handling, solvers, simulator, and model.

## Getting started

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
# Activate .venv using your platform's usual command
pip install -r requirements.txt
python -m pytest -q tests
```

The tests use small synthetic inputs; full benchmarks need public datasets and, for some methods, R. Dependencies are not locked, so exact replication requires recording the versions used.

### Data and benchmarks

Data are not bundled. `deconv/data_manifest.json` records GEO URLs and hashes for the original downloads. Download the series matrices listed there to `<data>/raw/<GSE>/<GSE>_series_matrix.txt.gz`, and set `DECONV_RAW` to `<data>/raw`. The Noob cache is written to its sibling `<data>/cache`. The manifest also lists the GSE167998 processed matrix needed for deposited-beta runs. The simulator expects the GSE35069 series matrix at `research-scout/GSE35069_series_matrix.txt.gz` relative to the repository root.

For uniform Noob preprocessing, install R with Bioconductor `minfi` and the appropriate 450K and EPIC manifest and annotation packages. Then, from `deconv/`, run:

```bash
python reprocess.py all
python run_baselines.py --preproc noob --no-epidish --no-paper
```

`reprocess.py all` downloads and extracts large raw IDAT archives and can require substantial disk space. It does not download the series matrices; obtain those separately first. The command above runs the Python baselines only. Omit `--no-epidish` to include EpiDISH, and omit `--no-paper` to include the additional R solvers; their package requirements and settings are in the R scripts. Set `RSCRIPT` if `Rscript` is not on your PATH. The simulator and exploratory analyses have additional data dependencies; inspect their scripts before running them. The default preprocessing is the deposited GEO data, so pass `--preproc noob` explicitly for the uniform comparison.

Checkpoints are not included, so checkpoint-dependent evaluations require training first. The newer simulator also needs a separately built panel cache; see `simulator.build_v6_panels` and `train_cpu.py --help` for the current experimental interfaces.

## Data and interpretation

The simulator uses sorted blood profiles from GSE35069 and GSE103541, while GSE110554 is used for real-data development. The planned primary test uses GSE167998 as a reference and GSE182379 as the mixture set. Small sample sizes, incomplete evidence about donor and DNA independence, platform and preprocessing differences, and the difficulty of identifying a missing type that resembles included types all limit interpretation. Noob does not remove Infinium probe-type bias, and the current benchmark does not apply a cross-reactive or SNP probe mask.

For background on the comparison methods, see [EpiDISH](https://bioconductor.org/packages/EpiDISH/) and the [DNA methylation deconvolution benchmark](https://doi.org/10.1038/s41467-024-48466-z). Dataset sources are recorded in the manifest and evaluation protocol.

Some integration tests require Rscript and the EpiDISH, EMeth, ICeDT, FARDEEP, and other R packages named in the solver scripts. If Rscript exists without those packages, these tests fail; they are not automatically skipped. Model tests require PyTorch.
