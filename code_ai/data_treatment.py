# data_treatment.py
import os
import chardet
from category_encoders import BinaryEncoder, OrdinalEncoder  # Additional encoders
import logging
import numpy as np
import pandas as pd
import optuna
import code_ai.shared_variables as sv
import torch
from code_ai.models_op import *
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read(10000)  # Read a sample of the file (first 10,000 bytes)
        result = chardet.detect(raw_data)
        return result['encoding']

# Function to load data from files
def load_data(uploaded_files):
    logger.info("Step 1: Loading data...")
    r_data = None  # Initialize r_data

    for n, f in enumerate(uploaded_files):
        try:
            file_name = os.path.basename(f)
            _, file_extension = os.path.splitext(file_name)
            file_extension = file_extension.lower().strip('.')
            encoding = detect_encoding(f)

            if file_extension == "csv":
                data = pd.read_csv(f, encoding=encoding, index_col=0, header=0, skip_blank_lines=True)
            elif file_extension in ["xls", "xlsx"]:
                data = pd.read_excel(f, sheet_name=0, index_col=0, header=0)
            elif file_extension == "dat":
                data = pd.read_csv(f, delim_whitespace=True, encoding=encoding, index_col=0, header=0, skip_blank_lines=True)
            else:
                logger.warning(f"Unsupported file extension: {file_extension}")
                continue

            # Ensure that the index is datetime
            data.index = parse_dates_with_order_inference(data.index)
            if data.index.hasnans:
                raise ValueError("Index contains NaT after conversion to datetime.")

            if r_data is None:
                r_data = data
            else:
                r_data = pd.concat([r_data, data], axis=0)
        except Exception as e:
            logger.error(f"Error loading file {f}: {e}")
            continue  # Continue processing other files

    if r_data is not None:
        r_data = convert_columns_to_numeric(r_data) # Convert columns to numeric where appropriate
        try:  # Sort the DataFrame by its datetime index
            r_data = r_data.sort_index()
            logger.info("Data sorted by datetime index successfully.")
        except Exception as e:
            logger.error(f"Error sorting data by index: {e}")

        logger.info("Data loaded and processed successfully!")
        logger.info(f"Data types after loading:\n{r_data.dtypes}")
    else:
        logger.warning("No data loaded from the provided files.")
    return r_data

def parse_dates_with_order_inference(date_index):
    dates1 = pd.to_datetime(date_index, errors='coerce', dayfirst=False) # Attempt parsing with dayfirst=False
    is_increasing1 = dates1.is_monotonic_increasing # Check if dates are increasing
    dates2 = pd.to_datetime(date_index, errors='coerce', dayfirst=True) # Attempt parsing with dayfirst=True
    is_increasing2 = dates2.is_monotonic_increasing # Check if dates are increasing

    if is_increasing1 and not is_increasing2: # Decide which dates to use
        logger.info("Using date parsing with dayfirst=False")
        return dates1
    elif is_increasing2 and not is_increasing1:
        logger.info("Using date parsing with dayfirst=True")
        return dates2
    else:
        # If both or neither are increasing, choose the one with fewer NaT
        num_nans1 = dates1.isna().sum()
        num_nans2 = dates2.isna().sum()
        if num_nans1 < num_nans2:
            logger.info("Using date parsing with dayfirst=False (fewer NaT values)")
            return dates1
        else:
            logger.info("Using date parsing with dayfirst=True (fewer NaT values)")
            return dates2

# Function to convert columns to numeric where possible
def convert_columns_to_numeric(df, threshold=0.3):
    for col in df.columns:
        # Skip columns that are already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Attempt to convert to numeric, coercing errors to NaN
        converted_series = pd.to_numeric(df[col], errors='coerce')
        num_numeric = converted_series.notna().sum()
        total_values = len(df[col])

        if num_numeric / total_values >= threshold:
            # Replace the column with the converted numeric values
            df[col] = converted_series
            logger.info(f"Column '{col}' converted to numeric.")
        else:
            # Leave the column as is
            logger.info(f"Column '{col}' remains as object (non-numeric data).")
    return df

# Function to handle categorical columns based on user choices
def handle_categorical_columns(r_data, user_choices, categorical_method):
    # Create a copy of the DataFrame to avoid modifying the original
    r_data_copy = r_data.copy()

    # Collect columns to exclude and categorize
    columns_to_exclude = []
    columns_to_categorize = []

    # If 'exclude_all' is selected, exclude all object columns
    if categorical_method == 'exclude_all':
        columns_to_exclude = r_data_copy.select_dtypes(include=['object']).columns.tolist()
        print("Excluding all object columns:", columns_to_exclude)
    else:
        for column, action in user_choices.items():
            print(f"Column: {column}, Action: {action}")
            if action == "exclude":
                columns_to_exclude.append(column)
            elif action == "categorize":
                columns_to_categorize.append(column)
                # Handle missing values
                r_data_copy[column] = r_data_copy[column].fillna('Missing')

    print("Columns to exclude:", columns_to_exclude)
    print("Columns to categorize:", columns_to_categorize)

    # Drop columns to exclude
    if columns_to_exclude:
        r_data_copy = r_data_copy.drop(columns=columns_to_exclude, errors='ignore')

    # Apply encoding to columns to categorize
    if columns_to_categorize:
        if categorical_method == "onehot":
            # One-hot encoding for specified columns
            data_to_encode = r_data_copy[columns_to_categorize] # Separate the columns to encode
            r_data_copy = r_data_copy.drop(columns=columns_to_categorize) # Preserve the rest of the DataFrame
            encoded_data = pd.get_dummies(data_to_encode, drop_first=True, dtype='int64')
            # Convert bool columns to int64
            bool_cols = encoded_data.select_dtypes(include=['bool']).columns
            if len(bool_cols) > 0:
                encoded_data[bool_cols] = encoded_data[bool_cols].astype('int64')
            # Concatenate the encoded columns back to the DataFrame
            r_data_copy = pd.concat([r_data_copy, encoded_data], axis=1)
        elif categorical_method == "label":
            # Label encoding
            for column in columns_to_categorize:
                r_data_copy[column] = r_data_copy[column].astype('category').cat.codes
        elif categorical_method == "ordinal":
            # Ordinal encoding
            encoder = OrdinalEncoder(cols=columns_to_categorize)
            r_data_copy = encoder.fit_transform(r_data_copy)
        elif categorical_method == "binary":
            # Binary encoding
            encoder = BinaryEncoder(cols=columns_to_categorize, drop_invariant=True)
            r_data_copy = encoder.fit_transform(r_data_copy)
        elif categorical_method == "frequency":
            # Frequency encoding
            for column in columns_to_categorize:
                freq = r_data_copy[column].value_counts() / len(r_data_copy)
                r_data_copy[column] = r_data_copy[column].map(freq)
    # Convert any remaining bool columns to int64
    bool_cols = r_data_copy.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        r_data_copy[bool_cols] = r_data_copy[bool_cols].astype('int64')

    return r_data_copy, columns_to_categorize, columns_to_exclude


def detect_data_outliers_missing_input(
    df,
    method='iqr',
    threshold=3,
    factor=1.5,
    lower_percentile=0.5,
    upper_percentile=99.5,
    window='30D',               # <-- new: size of the outlier windows
    ):
    """
    Detect outliers per fixed window (default 30 days) and replace them with NaN.

    See original docstring for parameter meanings; only the grouping logic changed.
    """

    # ------------------------------------------------------------------
    # 1. build the fixed-length window index
    # ------------------------------------------------------------------
    df = df.copy()
    window_size = pd.to_timedelta(window) if isinstance(window, str) else pd.Timedelta(days=window)
    start = df.index.min()
    df['group'] = ((df.index - start) // window_size).astype(int)

    # ------------------------------------------------------------------
    # 2. outlier detection, per column and per group
    # ------------------------------------------------------------------
    total_outliers_per_column = {}

    for col in df.columns:
        if col == 'group':
            continue

        total_outliers = 0          # counter for this column

        if method == 'zscore':
            def zscore_remove(x):
                nonlocal total_outliers
                std = x.std()
                if std == 0:
                    return x
                z = (x - x.mean()) / std
                mask = z.abs() > threshold
                total_outliers += mask.sum()
                x = x.copy()
                x[mask] = np.nan
                return x

            df[col] = df.groupby('group')[col].transform(zscore_remove)

        elif method == 'iqr':
            def iqr_remove(x):
                nonlocal total_outliers
                q1, q3 = x.quantile([0.25, 0.75])
                iqr = q3 - q1
                mask = (x < q1 - factor * iqr) | (x > q3 + factor * iqr)
                total_outliers += mask.sum()
                x = x.copy()
                x[mask] = np.nan
                return x

            df[col] = df.groupby('group')[col].transform(iqr_remove)

        elif method == 'percentile':
            def pct_remove(x):
                nonlocal total_outliers
                lo = x.quantile(lower_percentile / 100)
                hi = x.quantile(upper_percentile / 100)
                mask = (x < lo) | (x > hi)
                total_outliers += mask.sum()
                x = x.copy()
                x[mask] = np.nan
                return x

            df[col] = df.groupby('group')[col].transform(pct_remove)

        elif method == 'None':
            pass
        else:
            raise ValueError(
                f"Unsupported method '{method}'. Choose 'zscore', 'iqr', 'percentile', or 'None'."
            )

        total_outliers_per_column[col] = total_outliers
        print(f"{method.upper()} method: {col}: {total_outliers} outliers replaced by NaN.")

    # ------------------------------------------------------------------
    # 3. clean-up, gap filling, output
    # ------------------------------------------------------------------
    df.drop(columns='group', inplace=True)

    return df

def data_normalization(X_full, scaler_type="RobustMinMax"):
    """
    Normalize data using specified scaler.

    Parameters:
    - X_full, X_test: Input features.
    - scaler_type (str): Type of scaler to use ('QuantileTransformer', 'MinMaxScaler', 'StandardScaler', 'Log').

    Returns:
    - scaler adjusted to the data.
    """
    X, _, _ = split_time_series(X_full)
    if len(X.shape) == 3:
        n_feat = X.shape[2]
        X_r = X.reshape(-1, n_feat)
    else:
        X_r = X
    # Initialize scalers based on scaler_type
    if scaler_type == "RobustMinMax":
        X_scaler = Pipeline([("robust", RobustScaler()), ("minmax", MinMaxScaler(feature_range=(0, 1)))])
    elif scaler_type == "MinMaxScaler":
        X_scaler = MinMaxScaler()
    elif scaler_type == "StandardScaler":
        X_scaler = StandardScaler()
    elif scaler_type == "Log":
        X_scaler = FunctionTransformer(
            np.log1p, inverse_func=np.expm1, validate=False, accept_sparse=False
        )
    else:
        raise ValueError(f"Unsupported scaler_type: {scaler_type}")
    X_scaler.fit(X_r)
    return X_scaler

def split_time_series(X=None):
    """
        Split time series data into training and testing sets.

        Parameters:
        - X (array-like): Input features.
        - y (array-like): Target variable.
        - train_ratio (float): Proportion of data to use for training.

        Returns:
        - X_train, X_test, y_train, y_test
        """
    N = len(X)
    if N <= sv.VALIDATION_SPLIT + sv.TEST_SPLIT:
        raise ValueError(f"Not enough samples ({N}) for {sv.VALIDATION_SPLIT}+{sv.TEST_SPLIT} validation/test split.")
    train_end = N - (sv.VALIDATION_SPLIT + sv.TEST_SPLIT)
    val_end   = N - sv.TEST_SPLIT
    X_tr, X_val, X_te = X[:train_end], X[train_end:val_end], X[val_end:]
    return X_tr, X_val, X_te

def transform_and_split(X_l, X_f):
    X_comb = np.concatenate([X_l, X_f], axis=2)  # (N, T, Fp+Fs)
    flat = X_comb.reshape(-1, X_comb.shape[2])  # (N*T, F_std+F_priv)

    X_scaler = data_normalization(flat, sv.scaler_type)
    flat_norm = X_scaler.transform(flat)  # (N*T, F_std+F_priv)
    norm = flat_norm.reshape(X_l.shape[0], X_l.shape[1], -1)  # (N, T, F_std+F_priv)

    # 3) slice back into student vs. privileged
    F_std = X_l.shape[2]
    # first F_std channels are the privileged, last F_priv channels the student‐side
    return norm[:, :, :F_std], norm[:, :, F_std:]

def summarize_params(study):
    # Get all parameter names from trials
    all_params = [t.params for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not all_params:
        return {}, study.trials  # No completed trials
    param_summary = {}
    # Get parameter names from completed trials
    param_names = set()
    for p in all_params:
        param_names.update(p.keys())
    # Store only the parameters that were actually tuned
    for name in param_names:
        values = [p[name] for p in all_params if name in p]
        # Ensure consistent format for all parameters
        if isinstance(values[0], str):
            # For categorical parameters
            param_summary[name] = {
                'min': sorted(list(set(values))),  # List of unique values
                'max': None,
                'type': 'categorical'
            }
        else:
            # For numerical parameters, store min and max
            param_summary[name] = {
                'min': min(values),
                'max': max(values),
                'type': 'numerical'
            }
    # Add training configuration info with consistent format
    training_info = {
        'best_value': study.best_trial.value,
        'best_trial_number': study.best_trial.number,
        'Number of Trials': len(study.trials),
        'trial_epochs': sv.TRIAL_EPOCHS,
        'final_epochs': sv.FINAL_EPOCHS,
        'trial_patience': sv.TRIAL_PATIENCE,
        'final_patience': sv.FINAL_PATIENCE,
        'batch_sizes': sv.BATCH_SIZE_OPTIONS,
    }

    # Convert training info to the same format as other parameters
    for key, value in training_info.items():
        if isinstance(value, dict):
            # For nested dictionaries (like learning_rate_schedule)
            param_summary[key] = {
                'min': str(value),  # Convert dict to string
                'max': "",
                'type': 'info'
            }
        else:
            param_summary[key] = {
                'min': value,
                'max': "",
                'type': 'info'
            }
    return param_summary

def load_save_filter():
    # load phase
    cache_path = os.path.join(sv.OUTPUT_DIR, sv.CACHE_PATH)
    if sv.seq_reduction is None:
        if not os.path.exists(cache_path):
            print("No cache found.")
            return
        print("Loading teacher cache…")
        ckpt = torch.load(cache_path, map_location='cpu', weights_only=False)

        # # existing
        sv.seq_reduction  = ckpt.get('seq_reduction', None)

        print("Teacher cache loaded.")
    else:
        # save phase: everything in one go
        ckpt = {
            'seq_reduction':  sv.seq_reduction,
        }
        torch.save(ckpt, cache_path)
        print(f"Saved teacher cache to {cache_path}")


def create_sequences(data, target_variable, fe_lookahead: bool = False):
    data = data.dropna(axis=1, how="all")
    num_records = len(data)
    data_values = data.to_numpy(dtype=np.float32, copy=False)
    target_values = target_variable.to_numpy(dtype=np.float32, copy=False).squeeze()
    index_values = target_variable.index.to_numpy(copy=False)

    T = sv.TIMESTEPS
    H = sv.STEPS_AHEAD
    T_x = T + H if fe_lookahead else T

    max_sequences = num_records - T - H + 1
    X = np.empty((max_sequences, T_x, data_values.shape[1]), dtype=np.float32)
    y = np.empty((max_sequences, H), dtype=np.float32)
    y_index = np.empty((max_sequences, H), dtype=index_values.dtype)

    valid_count = 0
    for i in range(max_sequences):
        X_seq = data_values[i:i + T_x]
        y_seq = target_values[i + T:i + T + H]
        idx_seq = index_values[i + T:i + T + H]

        if np.isnan(X_seq).any() or np.isnan(y_seq).any() or np.isinf(X_seq).any() or np.isinf(y_seq).any():
            continue

        X[valid_count] = X_seq
        y[valid_count] = y_seq
        y_index[valid_count] = idx_seq
        valid_count += 1

    # Truncate to actual valid size
    return X[:valid_count], y[:valid_count], y_index[:valid_count]

def get_data_subset_continuous(total_train, total_val):
    """
        Returns a contiguous, stratified subset of the data starting from a random index.
        Added stratification to ensure better representation of the data patterns.
    """
    t_start = np.random.randint(0, total_train - sv.TRIAL_SPLIT + 1)
    t_end   = t_start + sv.TRIAL_SPLIT
    v_len   = max(1, int(round(sv.TRIAL_SPLIT * total_val / total_train)))
    v_len   = min(v_len, total_val)
    v_start = np.random.randint(0, total_val - v_len + 1)
    v_end   = v_start + v_len
    return t_start, t_end, v_start, v_end

# def get_config(config, name, *suggest_args, default=None,
#                 pop_type=None, **suggest_kwargs):
#     """
#     Behaviour identical to TF version, agnostic to backend.
#     """
#     kwargs = dict(suggest_kwargs) if suggest_kwargs else {}
#     if isinstance(config, optuna.trial.Trial):
#         if pop_type == "categorical" or isinstance(suggest_args[0], (list, tuple)):
#             fn = config.suggest_categorical
#         elif isinstance(suggest_args[0], float):
#             fn = config.suggest_float
#             kwargs.pop("step", None)
#         else:
#             fn = config.suggest_int
#         if fn is not config.suggest_float:
#             kwargs.pop("log", None)
#         return fn(name, *suggest_args, **kwargs)
#     return config.get(name, default)
def get_config(config, name, *suggest_args, default=None,
                pop_type=None, **suggest_kwargs):
    """
    Behaviour identical to TF version, agnostic to backend.
    """
    kwargs = dict(suggest_kwargs) if suggest_kwargs else {}

    if isinstance(config, optuna.trial.Trial):
        # Categorical: don't pass unrelated kwargs
        if pop_type == "categorical" or (suggest_args and isinstance(suggest_args[0], (list, tuple))):
            return config.suggest_categorical(name, *suggest_args)

        # Float: map any extras to keyword-only params
        if suggest_args and isinstance(suggest_args[0], float):
            low, high, *rest = suggest_args
            if rest:
                if len(rest) >= 1: kwargs.setdefault("step", rest[0])
                if len(rest) >= 2: kwargs.setdefault("log", rest[1])
            return config.suggest_float(name, low, high, **kwargs)

        # Int: map any extras to keyword-only params (fixes your warning)
        low, high, *rest = suggest_args
        if rest:
            if len(rest) >= 1: kwargs.setdefault("step", rest[0])
            if len(rest) >= 2: kwargs.setdefault("log", rest[1])
        return config.suggest_int(name, low, high, **kwargs)

    return config.get(name, default)

from typing import Sequence, Dict, List, Tuple
import numpy as np

def _band_names(method: str, B: int) -> List[str]:
    m = method.lower()
    if m == "wavelet":
        # RAW=0, then A, D1..D(B-1)  (since B = 1+levels)
        return ["RAW", "A"] + [f"D{k}" for k in range(1, B)]
    elif m == "stft":
        return ["RAW"] + [f"B{i}" for i in range(1, B+1)]
    elif m == "vmd":
        return ["RAW"] + [f"K{i}" for i in range(1, B+1)]
    else:
        return ["RAW"] + [f"M{i}" for i in range(1, B+1)]

def build_masks_by_feat(feats_kept, levels, input_features, method="wavelet"):
    """
    Return masks_by_feat: list of boolean masks over feats_kept,
    one mask per raw feature (0..F0-1).

    feats_kept      : 1D int indices into the *original* channel axis
    levels          : sv.DECOMPOSITION_LEVELS
    input_features  : sv.input_features (list/seq of raw feature names)
    method          : "wavelet" | "stft" | "vmd"
    """
    F0 = len(input_features)
    B = int(levels)   # A + D1..D_L

    feats_kept = np.asarray(feats_kept, dtype=int)

    # map each kept channel -> owning raw feature id (0..F0-1)
    feat_ids = np.where(feats_kept < F0, feats_kept, (feats_kept - F0) // B)

    # safety: ensure all engineered channels map inside 0..F0-1
    if (feat_ids < 0).any() or (feat_ids >= F0).any():
        raise ValueError("feats_kept contains channel(s) outside expected layout.")

    # boolean masks per feature
    masks_by_feat = [(feat_ids == f) for f in range(F0)]
    return masks_by_feat


