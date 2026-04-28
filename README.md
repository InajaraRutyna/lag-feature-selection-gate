# Gated Lag and Feature Selection for SCADA-Based Wind and Solar Power Forecasting

This repository contains the Python implementation associated with the Ph.D. thesis:

**Wind and Solar Power Forecasting for Short- and Medium-Term Horizons Using Machine Learning and Optimisation Techniques**  
Inajara da Silva Freitas Rutyna  
Warsaw University of Technology, 2026

The code implements a SCADA-only time-series forecasting pipeline for wind and solar power generation. The pipeline is designed for cases where only on-site operational measurements are available. It includes data loading, missing-data treatment, output normalization, lagged sequence construction, lag-feature selection, model training, deterministic error evaluation, and result visualization.

The project is intended to be run through the Dash interface defined in `app.py`.

## Research context

Wind and solar forecasting models are often developed with a combination of SCADA measurements and Numerical Weather Prediction (NWP) data. In many operational settings, NWP inputs are not available. The forecasting model must then use only historical measurements collected at the installation site.

This repository focuses on the SCADA-only case. The lagged input representation becomes the main source of predictive information. Missing values, sensor noise, inconsistent sampling, and redundant lag-feature combinations must be treated before model training.

The core methodological component is a gated lag-feature selection mechanism. It selects temporal lags and measured signals from SCADA windows during model training. The selected representation is then passed to machine-learning or deep-learning forecasting models.

## Repository structure

```text
lag-feature-selection-gate/
|
├── app.py                  # Dash interface used to run the project
├── main.py                 # Core forecasting pipeline called by the interface
├── run_all.py              # Result writer and batch-run helper used by the workflow
├── requirements.txt        # Python dependencies
|
├── code_ai/                # Data treatment, imputation, feature filtering, models, metrics, plots
├── code_visual/            # Dash layouts, callbacks, data selection, and run controls
├── config/                 # YAML configuration files
├── assets/                 # CSS files used by the Dash interface
├── fig/                    # Static figures used by the Dash interface
├── tests/                  # Experiment-output directory
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

The Dash interface is the entry point for the project. It is used to select the renewable-energy case, choose or prepare the dataset, configure the feature-filtering method, select forecasting models, start the training workflow, and save the generated results.

The code is not intended to be operated primarily from `main.py`. The `main.py` file contains the core forecasting pipeline, but the user-facing workflow is controlled by `app.py`.

## Configuration files

The configuration files are stored in:

```text
config/
```

The project uses two main configuration files:

```text
config/config.yaml
config/model_config.yaml
```

`config/config.yaml` defines dataset paths, interface-level choices, categorical treatment methods, feature-filtering methods, and model lists.

`config/model_config.yaml` defines the forecasting setup and training configuration. It includes the number of input lags, number of forecast steps ahead, train-validation-test split sizes, Optuna settings, model hyperparameter ranges, and lag-gate settings.

## Dataset paths

Dataset paths are defined in `config/config.yaml`.

```yaml
datasets:
  raw_datasets_dir: "../data/raw_datasets"
  inserted_datasets_dir: "../data/inserted_datasets"
  plots_dir: "graphs/"
```

`raw_datasets_dir` stores raw datasets before they are processed through the Dash interface.

`inserted_datasets_dir` stores prepared `.pkl` datasets. These files are created after the user selects variables, handles categorical columns, defines input and output variables, and saves the processed dataset through the interface.

`plots_dir` is the configured plot path. During the experiment workflow, generated plots are saved in the active experiment-output directory.

With the default paths, the expected structure is:

```text
parent-folder/
|
├── data/
|   ├── raw_datasets/
|   |   ├── raw_dataset_1.csv
|   |   └── ...
|   |
|   └── inserted_datasets/
|       ├── processed_dataset_1.pkl
|       ├── processed_dataset_2.pkl
|       └── ...
|
└── lag-feature-selection-gate/
    ├── app.py
    ├── config/
    └── ...
```

If the datasets are stored inside the repository, change the paths to:

```yaml
datasets:
  raw_datasets_dir: "data/raw_datasets"
  inserted_datasets_dir: "data/inserted_datasets"
  plots_dir: "graphs/"
```

Then the expected structure becomes:

```text
lag-feature-selection-gate/
|
├── data/
|   ├── raw_datasets/
|   |   ├── raw_dataset_1.csv
|   |   └── ...
|   |
|   └── inserted_datasets/
|       ├── processed_dataset_1.pkl
|       ├── processed_dataset_2.pkl
|       └── ...
|
├── app.py
├── config/
└── ...
```

Before running an experiment, check that:

```text
1. the raw dataset directory exists if raw files will be processed;
2. the inserted dataset directory exists if prepared .pkl files will be used;
3. the selected .pkl files are inside the inserted dataset directory;
4. the repository has write permission for the tests/ output directory.
```

## Result paths and generated figures

Experiment results are saved under:

```text
tests/
```

This directory is used for experiment outputs. It is not a unit-test folder.

Each run creates an experiment subfolder inside `tests/`. The folder name is generated from the experiment setup. It includes the dataset label, number of input lags, number of forecast steps ahead, data resolution, model group, and filtering setup.

Example:

```text
tests/
└── metrics_results_ore_lags_168_ahead_144_10_min_dl_gate_ml_all/
    ├── metrics_results_ore_lags_168_ahead_144_10_min_dl_gate_ml_all.xlsx
    ├── capacity.png
    ├── naive_predictions.png
    ├── linear_regression_predictions.png
    ├── random_forest_predictions.png
    ├── xgboost_predictions.png
    ├── lstm_predictions.png
    ├── logits.png
    ├── logits_1.png
```

The exact files depend on the selected models and filtering method.

The Excel workbook stores numerical results. It contains:

```text
Dataset Error
Model Error
Hyperparameters
```

`Dataset Error` stores test-set errors per forecast step and averaged across the forecast horizon.

`Model Error` stores model training and validation errors.

`Hyperparameters` stores selected hyperparameters and parameter-search summaries when hyperparameter tuning is used.

The generated figures are saved in the same experiment subfolder.

`capacity.png` shows the estimated capacity signal used for output normalization.

`<model_name>_predictions.png` compares observed and predicted multi-step trajectories for selected test samples. The plotting routine selects one low-error sample, one high-error sample, and several randomly selected non-zero samples.

Examples:

```text
naive_predictions.png
xgboost_predictions.png
lstm_predictions.png
```

Filtering-related figures are generated when the selected filtering method computes a lag-feature score or mask.

Typical filtering figures include:

```text
logits.png
logits_1.png
```

`logits.png` is used for binary lag-feature masks.

`logits_1.png` is used for continuous lag-feature score maps.


Not every experiment creates all filtering figures. A run with `Full` filtering may produce prediction figures but no lag-feature filtering map.


These figures are loaded by `app.py` and displayed in the interface.

## `config/config.yaml`

This file defines available pipeline choices, dataset paths, categorical treatment methods, imputation methods, feature-creation options, feature-filtering methods, and model lists.

The categorical treatment options are:

```yaml
categorical_methods:
  - 'onehot'
  - 'label'
  - 'ordinal'
  - 'binary'
  - 'frequency'
  - 'exclude_all'
```

The imputation options are:

```yaml
imputation_methods:
  - "CSDI"
  - "SAITS"
```


The feature-filtering options are:

```yaml
feature_filtering:
  - 'Full'
  - 'Gate'
  - 'Pearson'
  - 'MI'
  - 'CCF'
```

The active machine-learning model list is:

```yaml
ml_model:
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

The active deep-learning model list is:

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


The Dash interface combines the machine-learning and deep-learning model lists into the visible model-selection workflow. The separate lists are used to identify models that depend on sequential data and models that do not.

## `config/model_config.yaml`

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
number of retest trials
trial and final training epochs
early-stopping patience
batch-size options
random seed
lag-gate hyperparameters
model-size ranges
optimizer settings
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

The project is run from `app.py`. The Dash interface controls the full workflow. It loads data, collects user selections, saves processed datasets, starts model execution, streams logs, and writes results.

The pipeline has three stages.

### 1. Dataset preparation in the Dash interface

The interface first loads either a raw dataset or an existing processed `.pkl` file.

During this stage, the interface:

1. Reads data from the configured raw-data or inserted-data directory.
2. Displays dataframe information, column names, and preview rows.
3. Detects object-type columns and empty columns.
4. Lets the user categorize or exclude non-numeric columns.
5. Lets the user select input features.
6. Lets the user select the output variable.
7. Lets the user select the feature-filtering method.
8. Lets the user select one or more forecasting models.
9. Saves the processed dataset and metadata as a `.pkl` file.

The saved `.pkl` file is stored in the directory configured by:

```yaml
datasets:
  inserted_datasets_dir: "../data/inserted_datasets"
```

### 2. Model execution

After the user starts the run from the interface, the selected models are executed one by one.

For each selected model, the pipeline:

1. Loads `config/config.yaml` and `config/model_config.yaml`.
2. Reads the processed dataframe and metadata from the interface state.
3. Adds the output variable to the input-feature list when needed.
4. Creates an internal output column with the prefix `output_`.
5. Converts full zero-valued weeks in the output signal to missing values.
6. Averages duplicated timestamps.
7. Detects missing values and outliers in the selected input data.
8. Imputes missing input values.
9. Estimates the capacity signal from the output series.
10. Normalizes the output signal by the estimated capacity.
11. Constructs supervised lagged input-output sequences.
12. Scales the multi-step target array.
13. Splits the target array into training, validation, and test subsets.
14. Loads an existing lag-feature mask when available.
15. Scales the lagged input tensor.
16. Applies the selected filtering method: `Full`, `Gate`, `Pearson`, `MI`, or `CCF`.
17. Reduces the lag-feature tensor according to the selected mask.
18. Splits the filtered input tensor into training, validation, and test subsets.
19. Flattens the input tensor for non-sequential models.
20. Trains the selected model or computes the naive baseline.
21. Generates direct multi-step forecasts.
22. Inverse-transforms predictions to the normalized output scale.
23. Computes model-level training and validation errors.
24. Computes dataset-level test errors for each forecast step.
25. Converts predictions and observations back to the original capacity-scaled units.
26. Saves generated figures in the experiment-output folder.
27. Returns metrics, selected lag-feature information, best hyperparameters, and hyperparameter-search summaries.

### 3. Result saving

After each model finishes, the interface writes the returned results to an Excel workbook in the experiment-output folder.

Each experiment writes outputs to a subfolder inside:

```text
tests/
```

A typical experiment folder contains:

```text
experiment-output-folder/
├── metrics_results_*.xlsx
├── capacity.png
├── <model_name>_predictions.png
├── logits.png
├── logits_1.png

```

The Excel workbook stores numerical metrics and hyperparameter information. The `.png` files store the capacity plot, prediction plots, and lag-feature filtering maps generated during the run.

Not every run creates every figure. The saved figures depend on the selected model, selected filtering method, and whether the corresponding plotting function is called.

## Lag-feature selection

The repository supports several input-selection modes.

`Full` keeps the complete lag-feature representation.

`Gate` applies the gated lag-feature selection mechanism.

`Pearson` applies correlation-based filtering.

`MI` applies mutual-information-based filtering.

`CCF` applies cross-correlation-based filtering.

The selected lag-feature mask is applied before model training. For non-sequential models, the retained lag-feature tensor is flattened. For sequence models, the lag-feature structure is preserved.


## Evaluation metrics

The pipeline reports deterministic errors on the normalized forecasting target:

```text
nRMSE
nMAE
nMBE
```

Errors are computed per forecast step and as averages across the full forecast horizon.


## Reproducibility notes

This repository contains the implementation of the forecasting pipeline. Reproducing the thesis experiments also depends on access to the prepared wind and solar datasets, matching input metadata, selected configuration files, the Python environment, package versions, and local CPU or GPU availability.

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