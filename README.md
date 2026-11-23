# OCAP

Official implementation of **OCAP: Online Collaborative Adapter Personalization for Mental-Health Monitoring**, which was accepted at **KDD 2027**.

This repository includes the OCAP framework, dataset preparation scripts, and commands for running the main experiments on CES and GLOBEM.

Both datasets link individual participants’ sensing records to mental-health assessments. To protect participant privacy, we provide scripts to prepare the data locally and direct readers to the original releases for access. Please follow the [dataset preparation instructions](#dataset-preparation) below.

## Repository structure

```text
configs/                 Dataset-specific OCAP configurations
data/raw/                Please download CES and GLOBEM and place the raw files here.
data/processed/          The preparation scripts save the processed data here.
process/run_ocap.py       Main OCAP experiment runner
scripts/                 CES and GLOBEM preparation utilities
ocap/                    OCAP implementation
```

## Installation

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Dataset preparation

### CES

Please download the **College Experience Dataset (CES)** from its
[official Kaggle page](https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset).
After extracting the archive, place the release files so that the following
two paths exist:

```text
data/raw/ces/official_v5/EMA/general_ema.csv
data/raw/ces/official_v5/Sensing/sensing.csv
```

Prepare and validate the chronological CES stream:

```bash
python scripts/prepare_ces_historical.py \
  --source-root data/raw/ces/official_v5 \
  --output-csv data/processed/ces_historical_usernorm.csv

python scripts/validate_stream.py \
  --dataset ces \
  --input data/processed/ces_historical_usernorm.csv
```

The preparation script joins the daily PHQ-4 assessments with sensing
features, defines the risk label as `PHQ-4 >= 4`, performs the preprocessing
used in the experiments, and orders the resulting data by collection time.

### GLOBEM

PhysioNet limits access to **GLOBEM v1.1** to credentialed users. Please sign in to PhysioNet, complete its credentialing and training requirements, and sign the data use agreement before downloading the files from the [official dataset page](https://physionet.org/content/globem/1.1/).

Place the four extracted cohort directories as follows:

```text
data/raw/globem/physionet-1.1/INS-W_1/
data/raw/globem/physionet-1.1/INS-W_2/
data/raw/globem/physionet-1.1/INS-W_3/
data/raw/globem/physionet-1.1/INS-W_4/
```

Prepare and validate the chronological GLOBEM stream:

```bash
python scripts/align_globem_official.py \
  --root data/raw/globem/physionet-1.1 \
  --out-dir data/processed/globem_stage1

python scripts/prepare_globem_weekly.py \
  --input-csv data/processed/globem_stage1/globem_weekly_clean_all_cohorts.csv.gz \
  --output-arff data/processed/globem_top128.arff \
  --top-k-features 128

python scripts/convert_legacy_arff.py \
  --input-arff data/processed/globem_top128.arff \
  --output-csv data/processed/globem_full.csv

python scripts/validate_stream.py \
  --dataset globem \
  --input data/processed/globem_full.csv
```

The pipeline joins weekly depression labels with Bluetooth, call, location,
screen, sleep, step, and Wi-Fi features. It then selects 128 behavioral
features, retains the two platform indicators, and preserves chronological
order.

The final files used by the experiment runners are:

```text
data/processed/ces_historical_usernorm.csv
data/processed/globem_full.csv
```

## Running OCAP

The runner standardizes each feature using statistics from earlier triplets,
then updates those statistics. Each triplet is evaluated in predict-then-update order. The runner saves eight metrics per seed in `<dataset>_main_seed<seed>/metrics.json` and prints their means across the requested seeds.
Table 2 uses F1, Accuracy, Recall, and ROC AUC over the full stream. AUC uses the positive-class probability used for prediction.
Table 3 uses F1@5, F1@10, R@5, and R@10. At each cutoff, scores are computed on each eligible user's first records and averaged equally across users with at least that many records.

```bash
python process/run_ocap.py \
  --dataset ces \
  --data data/processed/ces_historical_usernorm.csv \
  --output results/ocap_main \
  --seeds 42,43,44,45,46 \
  --device cuda \
  --torch-threads 1

python process/run_ocap.py \
  --dataset globem \
  --data data/processed/globem_full.csv \
  --output results/ocap_main \
  --seeds 42,43,44,45,46 \
  --device cuda \
  --torch-threads 1
```

The main experiments use the hyperparameters in `configs/ces.yaml` and `configs/globem.yaml`. The
commands above require no manual hyperparameter selection.

## Data policy

The `.gitignore` rules prevent raw data, processed participant records,
predictions, and checkpoints from being committed. Before publishing a fork,
please verify that `git status` contains no files under `data/raw/`,
`data/processed/`, or `results/`.
