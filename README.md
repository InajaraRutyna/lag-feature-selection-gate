# Gated Lag and Feature Selection for SCADA-Based Wind and Solar Power Forecasting

This repository contains the Python implementation associated with the Ph.D. thesis:

**Wind and Solar Power Forecasting for Short- and Medium-Term Horizons Using Machine Learning and Optimisation Techniques**  
Inajara da Silva Freitas Rutyna  
Warsaw University of Technology, 2026

The code implements a SCADA-only time-series forecasting pipeline for wind and solar power generation. The pipeline is designed for cases where only on-site operational measurements are available. It includes data loading, missing-data treatment, output normalisation, lagged sequence construction, lag-feature selection, model training, and deterministic error evaluation.

The project is intended to be run through the Dash interface defined in `app.py`.

## Research context

Wind and solar forecasting models are often developed with a combination of SCADA measurements and Numerical Weather Prediction (NWP) data. In many operational settings, NWP inputs are not available. The forecasting model must then use only historical measurements collected at the installation site.

This repository focuses on the SCADA-only case. The lagged input representation becomes the main source of predictive information. Missing values, sensor noise, inconsistent sampling, and redundant lag-feature combinations must be treated before model training.

The core methodological component is a gated lag-feature selection mechanism. It selects temporal lags and measured signals from SCADA windows during model training. The selected representation is then passed to machine-learning or deep-learning forecasting models.

## Repository structure

```text
lag-feature-selection-gate/
│
├── app.py                  # Dash interface used to run the project
├── main.py                 # Core forecasting pipeline called by the interface
├── run_all.py              # Experiment runner used internally by the pipeline workflow
├── requirements.txt        # Python dependencies
│
├── code_ai/                # Data treatment, feature engineering, models, metrics, plots
├── code_visual/            # Dash layouts and callbacks
├── config/                 # YAML configuration files
├── assets/                 # CSS files used by the Dash interface
├── fig/                    # Interface figures
├── tests/                  # Output directory for experiment results
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/InajaraRutyna/lag-feature-selection-gate.git
cd lag-feature-selection-gate
git checkout main_test
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the project

Run the Dash application:

```bash
python app.py
```

After starting the app, open the local address printed in the terminal. In most local runs, this is:

```text
http://127.0.0.1:8050
```

The interface is the entry point for the project. It is used to select the renewable-energy case, choose the dataset, configure the model and filtering method, and start the forecasting workflow.

## Configuration files

The configuration files are stored in:

```text
config/
```

These files define the dataset paths, result paths, model lists, feature-filtering options, forecasting setup, and hyperparameter search space.

There are two main configuration files:

```text
config/config.yaml
config/model_config.yaml
```

### Dataset and output paths

Dataset and result locations are configured in `config/config.yaml`.

The dataset path points to the directory where the prepared input datasets are stored. The pipeline expects prepared `.pkl` files in this directory.

Example:

```yaml
datasets:
  inserted_datasets_dir: "../data/inserted_datasets"
```

This path is relative to the repository root. With the example above, the expected directory structure is:

```text
parent-folder/
│
├── data/
│   └── inserted_datasets/
│       ├── dataset_1.pkl
│       ├── dataset_2.pkl
│       └── ...
│
└── lag-feature-selection-gate/
    ├── app.py
    ├── config/
    └── ...
```

If the datasets are stored inside the repository, the path can be changed to:

```yaml
datasets:
  inserted_datasets_dir: "data/inserted_datasets"
```

Then the expected structure becomes:

```text
lag-feature-selection-gate/
│
├── data/
│   └── inserted_datasets/
│       ├── dataset_1.pkl
│       ├── dataset_2.pkl
│       └── ...
│
├── app.py
├── config/
└── ...
```

The results are saved under the `tests/` directory. This directory is used as the experiment-output location, not as a unit-test folder.

Example output path:

```text
tests/
└── metrics_results_ore_lags_168_ahead_144_10_min_dl_gate_ml_all/
    └── metrics_results_ore_lags_168_ahead_144_10_min_dl_gate_ml_all.xlsx
```

The main result file is an Excel workbook. It stores dataset-level errors, model-level errors, selected hyperparameters, and hyperparameter-search summaries.

Before running an experiment, check that:

```text
1. the dataset directory exists;
2. the selected .pkl files are inside the dataset directory;
3. the output directory tests/ exists or can be created by Python;
4. the repository has write permission for the tests/ directory.
```

### `config/config.yaml`

This file defines available pipeline choices, dataset paths, filtering methods, and model lists.

The feature-filtering options are:

```yaml
feature_filtering:
  - 'Full'
  - 'Gate'
  - 'Pearson'
  - 'MI'
  - 'CCF'
```

The model list is:

```yaml
model:
  - 'naive'
  - 'linear_regression'
  - 'random_forest'
  - 'elm'
  - 'xgboost'
  - 'lightgbm'
  - 'catboost'
  - 'gradient_boosted_decision_trees'
  - 'version_extreme_random_forest' 
```

The deep-learning model list is:

```yaml
deep_model:
  - 'lstm'
  - 'cnn'
  - 'rnn'
  - 'cnn_rnn'
  - 'cnn_lstm'
  - 'gru'
  - 'tcn'
  - 'ffnn'
```

### `config/model_config.yaml`

This file defines the forecasting setup and hyperparameter search space.

The forecasting setup is controlled by:

```yaml
prediction_config:
  timesteps: 168
  steps_ahead: 144
```

For 10-minute data, this corresponds to 168 input time steps and 144 forecast steps ahead.

The same file also defines:

```text
train, validation, trial, and test sizes
number of Optuna trials
trial and final training epochs
early-stopping patience
batch-size options
random seed
lag-gate hyperparameters
model-size ranges
optimiser settings
learning-rate schedule
Optuna pruning settings
```

To change the forecast horizon, edit:

```yaml
prediction_config:
  timesteps: ...
  steps_ahead: ...
```

To change the number of hyperparameter trials, edit:

```yaml
training:
  n_trials: ...
```

For a quick test run, reduce the training values:

```yaml
training:
  n_trials: 5
  trial_epochs: 2
  final_epochs: 5
  trial_patience: 2
  final_patience: 3
```

The default configuration may be computationally heavy. Full experiments should use the intended thesis-scale configuration. Quick tests should use smaller values.

## Pipeline steps

The main pipeline is implemented in `main.py` and called from the Dash interface.

The training procedure follows these steps:

1. Load configuration files.
2. Load a prepared `.pkl` dataset.
3. Read metadata for input features and output variable.
4. Remove duplicated timestamps by averaging records with the same index.
5. Detect missing and outlier values in the input time series.
6. Impute missing input values.
7. Estimate a capacity signal from the output time series.
8. Normalise the output by the estimated capacity.
9. Construct lagged input-output sequences.
10. Scale input and output arrays.
11. Apply lag-feature filtering.
12. Split the data into training, validation, and test subsets.
13. Train the selected model.
14. Produce direct multi-step forecasts.
15. Inverse-transform predictions.
16. Compute per-step and averaged deterministic error metrics.
17. Save metrics, selected features, and hyperparameters in the output directory.

## Lag-feature selection

The repository supports several input-selection modes.

`Full` keeps the complete lag-feature representation.

`Gate` applies the gated lag-feature selection mechanism.

`Pearson` applies correlation-based filtering.

`MI` applies mutual-information-based filtering.

`CCF` applies cross-correlation-based filtering.

The selected lag-feature mask is applied before model training. For non-sequential models, the retained lag-feature tensor is flattened. For sequence models, the lag-feature structure is preserved.

## Models

The code supports classical machine-learning models and deep-learning sequence models.

The active model list is controlled through:

```text
config/config.yaml
```

Classical models include:

```text
naive
linear_regression
random_forest
elm
xgboost
```

Deep-learning models include:

```text
lstm
cnn
rnn
cnn_rnn
cnn_lstm
gru
tcn
ffnn
```

The `naive` model is used as a baseline.

## Evaluation metrics

The pipeline computes deterministic forecast errors per forecast step and in aggregated form.

The reported metrics include:

```text
RMSE
MAE
MBE
nRMSE
nMAE
nMBE
```

The test predictions are inverse-transformed before final dataset-level evaluation.

## Output files

Results are saved in:

```text
tests/
```

The main output file is an Excel workbook with model metrics and hyperparameter summaries.

The workbook includes:

```text
Dataset Error
Model Error
Hyperparameters
```

Each experiment writes its results to a subfolder inside `tests/`.

## Reproducibility notes

This repository contains the implementation of the forecasting pipeline. Reproducing the thesis experiments also depends on access to the prepared wind and solar datasets, matching input metadata, selected configuration files, the Python environment, package versions, and local CPU or GPU availability.

Some datasets or trained outputs may be absent from the public repository because of data-access restrictions or file size.

## Citation

If this repository is used in academic work, cite the associated thesis:

```bibtex
@phdthesis{rutyna2026scadaforecasting,
  author = {Rutyna, Inajara da Silva Freitas},
  title = {Wind and Solar Power Forecasting for Short- and Medium-Term Horizons Using Machine Learning and Optimisation Techniques},
  school = {Warsaw University of Technology},
  year = {2026}
}
```

Related publication:

```bibtex
@article{rutyna2025gated,
  author = {Rutyna, Inajara},
  title = {Gated Lag and Feature Selection for Day-Ahead Wind Power Forecasting Using On-Site SCADA Data},
  journal = {Wind},
  volume = {5},
  number = {4},
  pages = {28},
  year = {2025},
  doi = {10.3390/wind5040028}
}
```

## Author

Inajara da Silva Freitas Rutyna  
Warsaw University of Technology