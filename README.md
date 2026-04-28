# Gated Lag and Feature Selection for SCADA-Based Wind and Solar Power Forecasting

This repository contains the Python implementation associated with the Ph.D. thesis:

**Wind and Solar Power Forecasting for Short- and Medium-Term Horizons Using Machine Learning and Optimisation Techniques**  
Inajara da Silva Freitas Rutyna  
Warsaw University of Technology, 2026

The code implements a SCADA-only time-series forecasting pipeline for wind and solar power generation. The pipeline is designed for cases where only on-site operational measurements are available. It includes data loading, missing-data treatment, output normalisation, lagged sequence construction, lag-feature selection, model training, and deterministic error evaluation.

The main way to run the project is through the Dash interface defined in `app.py`.

## Research context

Wind and solar forecasting models are often developed with a combination of SCADA measurements and Numerical Weather Prediction (NWP) data. In many operational settings, NWP inputs are not available. The forecasting model must then use only historical measurements collected at the installation site.

This repository focuses on the SCADA-only case. The lagged input representation becomes the main source of predictive information. Missing values, sensor noise, inconsistent sampling, and redundant lag-feature combinations must be treated before model training.

The core methodological component is a gated lag-feature selection mechanism. It selects temporal lags and measured signals from SCADA windows during model training. The selected representation is then passed to machine-learning or deep-learning forecasting models.

## Repository structure

```text
lag-feature-selection-gate/
│
├── app.py                  # Main Dash interface for running the pipeline
├── main.py                 # Core forecasting pipeline
├── run_all.py              # Batch execution over configured models
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

## Running the interface

The intended way to run the project is through the Dash application:

```bash
python app.py
```

After starting the app, open the local address printed in the terminal. In most local runs, this is:

```text
http://127.0.0.1:8050
```

## Running all configured models

To run all models listed in the configuration file:

```bash
python run_all.py
```

This script loops over the models defined in `config/config.yaml`. It runs the forecasting pipeline for each model and saves metrics and hyperparameter summaries to an Excel file.

The output directory is created under:

```text
tests/
```

The generated folder name includes the number of input lags, forecast steps ahead, data resolution, and selected filtering setup. The Excel file is saved inside that folder.

Example output structure:

```text
tests/
└── metrics_results_ore_lags_168_ahead_144_10_min_dl_gate_ml_all/
    └── metrics_results_ore_lags_168_ahead_144_10_min_dl_gate_ml_all.xlsx
```

The Excel workbook contains sheets for dataset error metrics, model error metrics, selected hyperparameters, and hyperparameter search summaries.

## Configuration files

The configuration files are stored in:

```text
config/
```

These files control the hyperparameter configuration, model lists, dataset paths, filtering options, and forecasting setup.

### `config/config.yaml`

This file defines available pipeline choices, including datasets, preprocessing methods, feature-filtering methods, and model lists.

The feature-filtering options are:

```yaml
feature_filtering:
  - 'Full'
  - 'Gate'
  - 'Pearson'
  - 'MI'
  - 'CCF'
```

The machine-learning model list is:

```yaml
ml_model:
  - 'naive'
  - 'linear_regression'
  - 'random_forest'
  - 'elm'
  - 'xgboost'
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

Some additional models may be present in the file but commented out. They can be reactivated after checking package compatibility and runtime requirements.

### `config/model_config.yaml`

This file defines the forecasting setup and hyperparameter search space.

The forecasting setup is controlled by:

```yaml
prediction_config:
  timesteps: 168
  steps_ahead: 144
```

For 10-minute data, this corresponds to 168 input time steps and 144 forecast steps ahead.

The same file also defines train, validation, trial, and test sizes; number of Optuna trials; trial and final training epochs; early-stopping patience; batch-size options; random seed; lag-gate hyperparameters; model-size ranges; optimiser settings; learning-rate schedule; and Optuna pruning settings.

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

## Input data format

The pipeline expects a prepared pickle file with a dictionary structure:

```python
{
    "data": pandas.DataFrame,
    "metadata": {
        "input_features": [...],
        "output_variable": "..."
    }
}
```

The DataFrame index must be datetime-like. The input features must be columns in the DataFrame. The output variable must also be a column in the DataFrame.

Example:

```python
metadata = {
    "input_features": ["wind_speed", "wind_direction", "temperature"],
    "output_variable": "active_power"
}
```

For a univariate setup, the input feature list can contain only the generated power or energy signal:

```python
metadata = {
    "input_features": ["active_power"],
    "output_variable": "active_power"
}
```

The code adds the output variable to the input feature list when it is missing. This keeps the historical output signal available for lagged forecasting.

## Pipeline steps

The main pipeline is implemented in `main.py`.

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
17. Save or return metrics, selected features, and hyperparameters.

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

Results from batch execution are saved in:

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

Each model is appended to the same workbook during `run_all.py`.

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