# Cloud Workload Forecasting

A reproducible time-series forecasting pipeline for aggregate cloud workload demand, developed for the Atomity AI/ML Engineering take-home challenge using the Microsoft Azure Public Dataset V2.

## Overview

This project analyzes virtual machine CPU telemetry and forecasts two aggregate workload indicators over a 24-step horizon:

- `active_vm_count`: number of distinct virtual machines with valid CPU observations at each timestamp.
- `aggregate_cpu_utilization`: mean CPU utilization across the available VM observations at each timestamp.

The dataset contains 10,000 virtual machines, 134 measurements per VM, and 1,340,000 CPU observations sampled at five-minute intervals.

The final forecasting approach uses:

- A constant forecast for `active_vm_count`, based on its observed stability in the training period.
- ARIMA(2,0,0) with a constant trend for `aggregate_cpu_utilization`.

The implementation includes data validation, time-aware model evaluation, automated prediction generation, and tests.

## Repository structure

```text
.
├── challenge_data/          # Local dataset (not committed)
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory analysis
│   ├── 01_eda.py            # Reproducible analysis script
│   └── figures/             # Generated EDA figures
├── reports/                 # Evaluation results and experiment reports
├── predictions.csv          # Final 24-step forecast
├── src/cwfc/
│   ├── data.py              # Data loading and validation
│   ├── baselines.py         # Baseline forecasting methods
│   ├── models.py            # Forecasting models
│   ├── validation.py        # Time-series cross-validation
│   ├── evaluate.py          # Model evaluation
│   └── predict.py           # Final prediction pipeline
├── tests/                   # Automated tests
├── pyproject.toml
└── README.md
```

## Dataset setup

The challenge dataset is distributed through the original repository's fixed `data-v1` GitHub Release.

Download `data-v1.tar.gz` from:

[Azure Dataset v1 Release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)

Extract the archive into the project root so that the following files are available:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

On Linux or macOS:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
cd challenge_data
sha256sum -c SHA256SUMS
cd ..
```

On Windows PowerShell, verify the downloaded files against the hashes listed in `challenge_data/SHA256SUMS` using `Get-FileHash`.

The dataset and archive are intentionally excluded from Git. The pipeline reads the dataset from the supplied `challenge_data` directory.

## Environment setup

Python 3.11 or newer is required.

Create and activate a virtual environment.

**Windows PowerShell**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project and test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For exploratory analysis:

```bash
python -m pip install -e ".[analysis]"
```

For the optional HistGradientBoosting experiment:

```bash
python -m pip install -e ".[boosting]"
```

## Reproducing the final predictions

From the repository root, run:

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

The command generates `predictions.csv` using the default final model, ARIMA(2,0,0).

The output must contain exactly three columns:

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

It contains exactly 24 rows, corresponding to the final two-hour forecasting horizon. The pipeline validates the output schema, timestamp ordering, uniqueness, finite values, and target constraints before writing the file.

To inspect the generated predictions:

```bash
python -c "import pandas as pd; d=pd.read_csv('predictions.csv'); print(d.to_string(index=False))"
```

## Methodology

### Exploratory data analysis

The exploratory analysis examines:

- Dataset integrity, timestamp cadence, and VM coverage.
- Aggregate CPU utilization distribution and variability.
- Autocorrelation and short-term temporal behavior.
- Differences in VM-level CPU characteristics.
- Workload concentration across virtual machines.

The analysis distinguishes full-release data quality checks from target-related analysis. Statistics that could influence model selection are restricted to the training period.

Run the analysis script with:

```bash
python notebooks/01_eda.py
```

### Leakage prevention and validation

The forecasting setup uses:

- 110 observations for training.
- 24 final observations reserved as the holdout.
- A five-minute sampling interval.
- A 24-step forecasting horizon.

The final 24 target observations are excluded from model fitting, feature selection, hyperparameter tuning, and model selection.

Model comparison uses expanding-window time-series cross-validation with:

- Initial training window: 48 observations.
- Forecasting horizon: 24 observations.
- Step size: 1 observation.
- Total: 39 folds.

The folds overlap, so their prediction errors are not independent. The evaluation also covers a single aggregate time series; results should therefore not be interpreted as evidence of generalization to unrelated workloads or datasets.

### Model selection

Several forecasting approaches were compared, including statistical baselines, exponential smoothing, moving averages, temporal regression, ARIMA configurations, and an optional HistGradientBoosting model.

The selected model was ARIMA(2,0,0) with a constant trend.

Cross-validation results for aggregate CPU utilization:

| Model                |    MAE |   RMSE | sMAPE (%) |
| -------------------- | -----: | -----: | --------: |
| ARIMA(2,0,0)         | 0.1926 | 0.2317 |    2.0343 |
| Temporal regression  | 0.1961 | 0.2350 |    2.0716 |
| Mean baseline        | 0.2050 | 0.2420 |    2.1656 |
| HistGradientBoosting | 0.2244 | 0.2802 |    2.3592 |

ARIMA(2,0,0) achieved the lowest MAE among the evaluated candidates. Its improvement over temporal regression was relatively small, so the results support a preference for the simpler selected statistical model within this validation setup rather than a broad claim of model superiority.

The active VM count remained constant at 10,000 throughout the training observations. A constant forecast was therefore used for this target.

Detailed results are available in the `reports/` directory, including fold-level metrics, horizon-level metrics, and the HistGradientBoosting experiment.

## Evaluation

Run the automated test suite:

```bash
python -m pytest -q
```

Run the optional candidate comparison:

```bash
python -m pip install -e ".[boosting]"
python -m cwfc.evaluate --data challenge_data --candidates --detailed
```

The evaluation command compares the forecasting candidates using the time-aware validation procedure. The optional HistGradientBoosting model is experimental and is not used by the default final prediction pipeline.

## Final submission artifact

The final deliverable is `predictions.csv`.

It contains 24 chronological predictions, with no missing values or duplicate timestamps, and satisfies the required target constraints:

- `active_vm_count` is non-negative.
- `aggregate_cpu_utilization` is between 0 and 100.
- All predictions are finite numeric values.

The dataset, final holdout targets, and local environment files are not included in the repository.

## Data source

Microsoft Azure Public Dataset V2:

https://github.com/Azure/AzurePublicDataset

The challenge uses the fixed curated `data-v1` release supplied by Atomity, rather than the full upstream trace.

## Reproducibility

The project is designed to run from a clean checkout when the required dataset release is supplied. Dependencies are declared in `pyproject.toml`, and the prediction pipeline can be executed without manually editing generated files.
