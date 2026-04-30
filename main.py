import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
# Import blocks
from code_ai.feature_engineering import *
from code_ai.models_op import *
from code_ai.data_imputation import *
from code_ai.data_treatment import *
from code_ai.plots import *
from permetrics.regression import RegressionMetric
from config.load import *
import os
import torch
# (optional) Pick CPU only
torch.set_default_device("cpu")


def fixed_variables():
    sv.DECAY = False
    MODEL_CONFIG = load_model_config()
    sv.CACHE_PATH = ('cache.pth')


    sv.TIMESTEPS = MODEL_CONFIG['prediction_config']['timesteps']
    sv.STEPS_AHEAD = MODEL_CONFIG['prediction_config']['steps_ahead']

    sv.TRIAL_SPLIT = MODEL_CONFIG['data_split']['trial']
    sv.VALIDATION_SPLIT = MODEL_CONFIG['data_split']['validation']
    sv.TEST_SPLIT = MODEL_CONFIG['data_split']['test']

    sv.N_TRIALS = MODEL_CONFIG['training']['n_trials']
    sv.TRIAL_EPOCHS = MODEL_CONFIG['training']['trial_epochs']
    sv.RETEST_TRIALS = MODEL_CONFIG['training']['retest_trials']
    sv.FINAL_EPOCHS = MODEL_CONFIG['training']['final_epochs']
    sv.TRIAL_PATIENCE = MODEL_CONFIG['training']['trial_patience']
    sv.FINAL_PATIENCE = MODEL_CONFIG['training']['final_patience']
    sv.BATCH_SIZE_OPTIONS = MODEL_CONFIG['training']['batch_size_options']
    sv.RANDOM_SEED = MODEL_CONFIG['training']['random_seed']

    sv.KEEP_FRAC = MODEL_CONFIG['hyperparameter_lag']['keep_frac']
    sv.DECAY_LAG = MODEL_CONFIG['hyperparameter_lag']['decay_lag']
    sv.INCENTIVE_FEAT = MODEL_CONFIG['hyperparameter_lag']['incentive_feat']
    sv.SPAN_FRAC = MODEL_CONFIG['hyperparameter_lag']['span_frac']

    sv.UNITS_MAP = MODEL_CONFIG['model_sizes']
    sv.OPTIMIZER_CONFIG = MODEL_CONFIG['optimizer']
    sv.HYPERPARAMETER_SPACE = MODEL_CONFIG['hyperparameter_space']
    sv.CPUS = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    print(f"CPUs: {sv.CPUS}")

    sv.LR_SCHEDULE = MODEL_CONFIG['lr_schedule']
    sv.OPTUNA_CONFIG = MODEL_CONFIG['optuna']["pruner"]
    sv.GENERIC_FEATURES = False


def zero_weeks_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Index must be datetime")

    # Iterate weekly and set NaN where all are zero
    for week_start, week_data in df.resample("W"):
        if (week_data == 0).all().all():  # All columns, all rows zero
            df.loc[week_data.index] = np.nan

    return df

def get_data(config):
    inserted_datasets_dir = config.get('datasets', {}).get('inserted_datasets_dir', '.')
    if not os.path.exists(inserted_datasets_dir):
        print(f"Directory '{inserted_datasets_dir}' does not exist.")
        return
    root = tk.Tk()
    root.withdraw()
    pickle_file = filedialog.askopenfilename(
        title="Select a pickle file",
        filetypes=[("Pickle files", "*.pkl")],
        initialdir=inserted_datasets_dir
    )
    if not pickle_file:
        print("No file selected. Exiting.")
        return
    with open(pickle_file, 'rb') as f:
        saved_data = pd.read_pickle(f)
    data = saved_data.get('data', None)
    sv.metadata = saved_data.get('metadata', {})
    if data is None or sv.metadata is None:
        print("Failed to load data or metadata from the pickle file.")
        return

    input_features = sv.metadata.get("input_features", [])
    output_variable = sv.metadata.get("output_variable", None)

    if not input_features or output_variable is None:
        print("Input features or output variable not specified in metadata.")
        return

    if output_variable not in input_features:
        input_features.append(output_variable)

    data_input = data[input_features]
    new_output = f"output_{output_variable}"
    data_input[new_output] = zero_weeks_to_nan(data_input[output_variable])

    data_input = data_input.groupby(level=0).mean() # guarantee unique indices

    sv.metadata["input_features"] = input_features
    sv.metadata["output_variable"] = new_output
    sv.data_input = data_input.astype('float32')
    return

def compute_model_metrics_per_step(train_error, validation_error, class_, train_error_all, validation_error_all):
    train_error['class'] = class_
    validation_error['class'] = class_
    train_error_all = pd.concat([train_error_all, train_error])
    validation_error_all = pd.concat([validation_error_all, validation_error])

    return train_error_all, validation_error_all


def compute_dataset_metrics_per_step(y_true, y_pred, average_dataset_error, dataset_metrics, cl):
    """
    Compute dataset error metrics for each forecast step.

    Parameters:
    - y_true: true values with shape [samples, steps_ahead]
    - y_pred: predicted values with shape [samples, steps_ahead]
    - average_dataset_error: dataframe storing averaged errors
    - dataset_metrics: dataframe storing per-step errors
    - cl: class/model label used in the result tables
    """

    rows = []

    for step in range(sv.STEPS_AHEAD):
        y_true_step = y_true[:, step]
        y_pred_step = y_pred[:, step]

        evaluator = RegressionMetric(y_true_step, y_pred_step)

        nrmse = evaluator.RMSE()
        nmae = evaluator.MAE()
        nmbe = evaluator.MBE()

        rows.append({
            'class': cl,
            'step': step + 1,
            'nrmse': nrmse,
            'nmae': nmae,
            'nmbe': nmbe
        })

    new_metrics = pd.DataFrame(rows)

    dataset_metrics = pd.concat(
        [dataset_metrics, new_metrics],
        ignore_index=True
    )

    average_row = pd.DataFrame([{
        'class': cl,
        'Avg NRMSE': new_metrics['nrmse'].mean(),
        'Avg NMAE': new_metrics['nmae'].mean(),
        'Avg NMBE': new_metrics['nmbe'].mean()
    }])

    average_dataset_error = pd.concat(
        [average_dataset_error, average_row],
        ignore_index=True
    )

    return average_dataset_error, dataset_metrics

def compute_capacity(output_series, window='30D', threshold=0.50):
    """
    Compute the capacity series from the smoothed output variable using fixed rolling windows.

    Parameters:
    - output_series (pd.Series): The smoothed output variable series.
    - window (str or int): The fixed window size (e.g., '30D' for 30 days).
    - threshold (float): The threshold to use for capacity estimation (optional).

    Returns:
    - pd.Series: Capacity series aligned with output_series.
    """

    # Step 1: Assign each date to a fixed window group
    window_size_cap = pd.to_timedelta(window) if isinstance(window, str) else pd.Timedelta(days=window)
    start = output_series.index.min()
    group_numbers = ((output_series.index - start) // window_size_cap).astype(int)

    # Create a DataFrame for grouping
    df = output_series.to_frame(name='value')
    df['group'] = group_numbers

    # Step 2: Compute group-wise max and min
    grouped = df.groupby('group')
    df['group_max'] = grouped['value'].transform('max')
    df['group_min'] = grouped['value'].transform('min')

    # Step 3: Calculate range per group
    df['range'] = df['group_max'] - df['group_min']

    # Step 4: (Optional) Identify significant changes between groups
    # For demonstration, we'll compute the difference in range between consecutive groups
    group_range = grouped['range'].first()
    group_diff = group_range.diff().abs()

    # Define the threshold based on the maximum range
    overall_max_range = group_range.max()
    significant_changes = group_diff > (overall_max_range * threshold)

    # Step 5: Assign new group numbers based on significant changes
    # This allows merging groups where changes are not significant
    new_group_numbers = significant_changes.cumsum()
    df['new_group'] = df['group'].map(lambda x: new_group_numbers[x])
    # Step 6: Compute adjusted capacity based on new groups
    df['capacity'] = (df.groupby('new_group'))['range'].transform('max')

    return df['capacity']

#############################
# MAIN FUNCTION
#############################
def main(**kwargs):
    """
    Main function to run the entire time-series pipeline.
    Parameters:
        data: pd.DataFrame
            The data provided from the Dash app or loaded from a pickle file
        metadata: dict
            Dictionary containing the methods and parameters for each step
    """
    config = load_config('config/config.yaml')
    fixed_variables()
    if sv.data_input is None:
        get_data(config)

    # Extract methods from metadata
    sv.outliers_method = kwargs.get("outlier_method", sv.metadata.get('outlier_method', 'percentile'))
    sv.model_type = kwargs.get("model", sv.metadata.get("model", "knn"))
    sv.input_features = sv.metadata.get("input_features", [])
    sv.output_variable = sv.metadata.get("output_variable",None)
    sv.feature_filtering_method = kwargs.get("feature_filtering_method", sv.metadata.get("feature_filtering_method", "correlation"))
    sv.size_data = kwargs.get("size_data",'All')
    sv.scaler_type="RobustMinMax" #MinMaxScaler, Log
    sv.SOLAR = bool(getattr(sv, "SOLAR", False))

    print(f"Outlier method: {sv.outliers_method}")
    print(f"Model type: {sv.model_type}")
    print(f"Feature filtering method: {sv.feature_filtering_method}")
    print(f"Size of data: {sv.size_data}")
    print(f"Scaler type: {sv.scaler_type}")
    print(f"Input features: {sv.input_features}")
    print(f"Output variable: {sv.output_variable}")
    print(f"Steps ahead: {sv.STEPS_AHEAD}")
    print(f"SOLAR: {sv.SOLAR}")


    predictions_full = np.empty((0, sv.STEPS_AHEAD))
    y_test_full = np.empty((0, sv.STEPS_AHEAD))
    param_summary_all, best_params_all = {}, {}
    sv.is_sequence = (sv.model_type in config['deep_model'] and sv.model_type != 'ffnn')

    # Step 1: Prepare the data
    ("Step 1: Clean and reshape dataset...")
    if sv.data_out is None:
        sv.data_out = detect_data_outliers_missing_input(sv.data_input, method=sv.outliers_method)
        print(f"Outliers removed using method: {sv.outliers_method}")
        # plot_dataframes_comparison(sv.data_out, sv.data_input, "Cleaned", "Input")                                                 #
        if sv.size_data != 'All':
            sv.data_out = sv.data_out.last(sv.size_data)  # Last 2 years

        print("Step 2: Handling missing data...")
        sv.data_out[sv.input_features] = data_imputation(sv.data_out[sv.input_features])  # imputation_method)
        print(f"Data Imputation Completed")
        # # plot_dataframes_comparison(data_imputed, data, "Imputed", "Cleaned")

    data = sv.data_out.copy()
    capacity = compute_capacity(data[sv.output_variable])
    # plot_capacity(data[output_variable], capacity)
    data[sv.output_variable] = data[sv.output_variable]/capacity
    data[data.columns[-2]] = data[data.columns[-2]]/capacity

    # Step 3: Feature Engineering (on features only)
    print("Step 3: Feature engineering...")
    print("data.shape", data.shape)

    # Step 5: Feature Filtering (input  features and output)
    X_Raw, y, y_index = create_sequences(data[sv.input_features], data[sv.output_variable], fe_lookahead=(False))

    X_filtered = X_Raw
    best_params_all["Total variables"] =  X_filtered.shape[2]
    best_params_all["New variables - Decomposition"] = X_filtered.shape[2] - len(sv.input_features)

    sv.y_scaler = data_normalization(y, sv.scaler_type)
    y_scaled = sv.y_scaler.transform(y)
    sv.DATA_Y["y_train"], sv.DATA_Y["y_val"], sv.DATA_Y["y_test"] = split_time_series(y_scaled)
    _, _, sv.DATA_Y["y_test_ind"] = split_time_series(y_index)
    _, _, sv.DATA_Y["y_test_orig_f"] = split_time_series(y)

    print("Step 4: Feature filtering...")
    load_save_filter()

    X_scaler_p = data_normalization(X_filtered, sv.scaler_type)
    X_filtered_scaled = X_scaler_p.transform(X_filtered.reshape(-1, X_filtered.shape[2])).reshape(X_filtered.shape)


    if sv.seq_reduction is None and sv.feature_filtering_method != 'Full':
        sv.seq_reduction = data_filter(X_filtered_scaled, sequence=sv.is_sequence)
        print(f"Feature filtering done using method: {sv.feature_filtering_method}")

        X_filtered_scaled = reduce_lags_features(X_filtered_scaled, sv.seq_reduction, False)
        print(f"Mask of retained lags and features applied \n{mask_compact(sv.seq_reduction)}")
        print(f"{X_filtered_scaled.shape[1]} lags | {X_filtered_scaled.shape[2]} features retained")

    print(X_filtered_scaled.shape)


    sv.DATA_p["X_train"], sv.DATA_p["X_val"], sv.DATA_p["X_test"] = split_time_series(X_filtered_scaled)
    load_save_filter()

    if (not sv.is_sequence or (sv.model_type == 'ffnn')) and sv.model_type != 'naive':
        _, L, F = X_filtered_scaled.shape
        X_filtered_scaled = X_filtered_scaled.reshape(X_filtered_scaled.shape[0], L * F).astype(np.float32)
        print(f"Data reduced to {X_filtered_scaled.shape[1]} features, for non sequential algorithm")
        X_lag = X_filtered_scaled.copy()
        X_filtered_scaled = pd.DataFrame(X_lag)
        sv.DATA_p["X_train"], sv.DATA_p["X_val"], sv.DATA_p["X_test"] = split_time_series(X_filtered_scaled)

    del  X_filtered_scaled, X_Raw, y, y_index
    print(f"Step 9: Model training and multi-step forecasting  {sv.model_type} ...")
    n_tr, n_v = len(sv.DATA_Y["y_train"]), len(sv.DATA_Y["y_val"])

    capacity_test = np.array([capacity.get(x, x) for x in sv.DATA_Y["y_test_ind"][:, 0]])

    sv.DATA.update(dict(X_train=sv.DATA_p["X_train"], X_val=sv.DATA_p["X_val"], X_test=sv.DATA_p["X_test"],
                        y_train=sv.DATA_Y["y_train"], y_val=sv.DATA_Y["y_val"], y_test=sv.DATA_Y["y_test"],
                        y_test_orig = sv.DATA_Y["y_test_orig_f"]))

    if sv.model_type == "naive":
        idx = sv.DATA_Y["y_test_ind"][:, 0]  # Test index positions
        if sv.SOLAR:
            # y_test_ind contains the horizon timestamps for each sample: shape (n_samples, H)
            idx_mat = sv.DATA_Y["y_test_ind"]  # (n_test, H)

            y_series = data[sv.output_variable]

            src_idx = pd.DatetimeIndex(idx_mat.reshape(-1)) - pd.Timedelta(days=1)

            # tolerance = half typical step (robust to missing points)
            di = pd.DatetimeIndex(y_series.index).sort_values().to_series().diff().dropna()
            freq = di.median() if len(di) else pd.Timedelta(minutes=10)
            tol = freq / 2

            vals = y_series.reindex(src_idx, method="nearest", tolerance=tol).to_numpy(dtype=np.float32)
            predictions = vals.reshape(idx_mat.shape)

            # fill if any NaNs (start-of-series / gaps)
            if np.isnan(predictions).any():
                predictions = (
                    pd.DataFrame(predictions)
                    .ffill(axis=1)
                    .bfill(axis=1)
                    .to_numpy(dtype=np.float32)
                )

            predictions_scaled = sv.y_scaler.transform(predictions)
            best_params = param_summary = None
            train_error = validation_error = pd.DataFrame()

        else:
            s = data[sv.output_variable].shift(1).ffill()  # Series length N
            v = s.reindex(idx).to_numpy(dtype=np.float32)  # Only test rows
            predictions = np.repeat(v[:, None], sv.STEPS_AHEAD, axis=1)  # (n_test, H)
            predictions_scaled = sv.y_scaler.transform(predictions)
            best_params = param_summary = None
            train_error = validation_error = pd.DataFrame()
    elif sv.is_sequence or (sv.model_type == 'ffnn'):
        predictions_scaled, best_params, param_summary, train_error, validation_error, model1 = machine_learning_models_sequence(sv.DATA, method=sv.model_type)
    else:
        predictions_scaled, best_params, param_summary, train_error, validation_error, model1 = machine_learning_models_ml(sv.DATA, method=sv.model_type)

    print(f"Model training done using {sv.model_type}, forecasting {sv.STEPS_AHEAD} steps ahead.")

    # Step 10: Model Evaluation Metrics (RMSE, MAE, MBE)
    print("Step 10: Computing error metrics for the model...")

    if 'average_dataset_error' not in locals():
        # Initialize empty DataFrames to store metrics
        average_dataset_error, dataset_metrics_all = pd.DataFrame(), pd.DataFrame()
        validation_error_all, train_error_all = pd.DataFrame(), pd.DataFrame()

    # Compute error metrics per step
    class_ = str(len(sv.DATA['y_val']))
    train_error_all, validation_error_all = compute_model_metrics_per_step(train_error, validation_error, class_, train_error_all, validation_error_all)

    # Step 11: Dataset Evaluation Metrics (NRMSE, NMAE, NMBE)
    print("Step 11: Computing dataset performance metrics (NRMSE, NMAE, NMBE)...")
    y_test_full = np.concatenate((y_test_full, sv.DATA["y_test_orig"]), axis=0)
    predictions = sv.y_scaler.inverse_transform(predictions_scaled)
    predictions_full = np.concatenate((predictions_full, predictions), axis=0)
    average_dataset_error, dataset_metrics_all = compute_dataset_metrics_per_step(sv.DATA["y_test_orig"], predictions,  average_dataset_error, dataset_metrics_all, class_)

    ###### test
    print("⇢  scaled RMSE", np.sqrt(np.mean((predictions_scaled - sv.DATA["y_test"]) ** 2)))
    print("⇢  real-space RMSE", np.sqrt(np.mean((predictions - sv.DATA["y_test_orig"]) ** 2)))
    ######

    if sv.model_type != "naive":
        param_summary_all.update(param_summary)
        best_params_all.update(best_params)

    print("Metrics for class collected and stored.")
    # Multiply adjusted predictions and true values by capacity to get back to original scale
    y_true = sv.DATA["y_test_orig"] * capacity_test[:,np.newaxis]
    y_pred = predictions * capacity_test[:,np.newaxis]
    plot_predictions(y_true, y_pred, sv.model_type) # Call the plotting function


    # Compute error metrics for the entire dataset
    average_dataset_error, dataset_metrics_all = compute_dataset_metrics_per_step(y_test_full, predictions_full,
                                                                                      average_dataset_error,
                                                                                      dataset_metrics_all, 'AVG Total')
    if sv.model_type != "naive":
        if sv.seq_reduction is not None:
            best_params_all["Best variables"] = mask_compact(sv.seq_reduction)
        if not sv.is_sequence:
            best_params_all["Best features ML"] = sv.DATA["X_train"].shape[1]
    print("Paused here, press Enter to continue...")
    # input()
    return average_dataset_error, train_error_all, dataset_metrics_all,validation_error_all, best_params_all, param_summary_all

if __name__ == "__main__":
    average_dataset_error, train_error_all, dataset_metrics_all,validation_error_all, best_params, param_summary = main()

    print("\nTrain Error Metrics:")
    print(train_error_all)

    print("\nValidation Error Metrics:")
    print(validation_error_all)

    print("\nPer-Step Dataset Error Metrics:")
    print(average_dataset_error)

    print("\nAverage Dataset Error Metrics Across All Steps:")
    print(dataset_metrics_all)

    print(f"Best hyperparameters for model: {best_params}")
