import gc
import traceback
import math
import torch.nn as nn
import torch.nn.functional as F
import os, tempfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from functools import partial
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from hpelm import ELM
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import optuna
from optuna.pruners import MedianPruner
from permetrics.regression import RegressionMetric
from tqdm import trange
import code_ai.shared_variables as sv
from code_ai.plots import history_train
from code_ai.data_treatment import *

# ---------------------------------------------------------------------
class _Shim:               # minimal object with .history
    def __init__(self, d): self.history = d

def _np(arr):
    """Ensure NumPy array (handles DataFrame/Series)."""
    return arr.to_numpy(copy=False) if isinstance(arr, (pd.DataFrame, pd.Series)) else arr

def _cleanup() -> None:          # name kept for API-compat
    """Free GPU/CPU memory (PyTorch version)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ╔════════════════════════════════════════════════════════════════════╗
# ║                   Classical ML  (unchanged)                        ║
# ╚════════════════════════════════════════════════════════════════════╝
class SklearnELM(BaseEstimator, RegressorMixin):
    # <unchanged — depends on hpelm, not TF>
    def __init__(self, n_input=None, n_output=None, classification='r', n_hidden=100, activation_function='sigm'):
        self.n_input  = n_input
        self.n_output = n_output
        self.classification = classification
        self.n_hidden = n_hidden
        self.activation_function = activation_function
        self.elm_model = None

    def fit(self, X, y):
        X_np = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        y_np = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y
        if self.n_input  is None: self.n_input  = X_np.shape[1]
        if self.n_output is None: self.n_output = y_np.shape[1] if y_np.ndim > 1 else 1
        self.elm_model = ELM(self.n_input, self.n_output, classification=self.classification)
        self.elm_model.add_neurons(self.n_hidden, self.activation_function)
        self.elm_model.train(X_np, y_np)
        return self

    def predict(self, X):
        X_np = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        return self.elm_model.predict(X_np)

    def get_params(self, deep=True):
        return dict(n_input=self.n_input, n_output=self.n_output, classification=self.classification,
                    n_hidden=self.n_hidden, activation_function=self.activation_function)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

def create_ml_model(trial, method):
    if method == "linear_regression":
        model = LinearRegression()
    elif method == "random_forest":
        n_estimators = trial.suggest_int('n_estimators', 5, 100, step=5)
        max_depth = trial.suggest_int('max_depth', 2, 12)
        min_split = trial.suggest_int('min_samples_split', 5, 40, step=5)
        min_leaf = trial.suggest_int('min_samples_leaf', 1, 21, step=5)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_split=min_split, min_samples_leaf=min_leaf,
                                      random_state=42)
    elif method == "elm":
        n_hidden = trial.suggest_int('n_hidden', 5, 100, step=5)
        activation_function = trial.suggest_categorical('activation_function',
                                             ["lin", "sigm", "tanh", "rbf_l1", "rbf_l2", "rbf_linf"])
        model = SklearnELM(n_hidden=n_hidden, activation_function=activation_function)
    elif method == "xgboost":
        n_estimators = trial.suggest_int('n_estimators', 5, 100, step=5)
        max_depth    = trial.suggest_int('max_depth', 1, 6)
        lr           = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        subsample    = trial.suggest_float('subsample', 0.5, 1.0)
        colsample    = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                             learning_rate=lr, subsample=subsample, tree_method="hist",
                             colsample_bytree=colsample, random_state=42)
    elif method == "lightgbm":
        n_estimators = trial.suggest_int('n_estimators', 5, 100, step=5)
        leaves       = trial.suggest_int('num_leaves', 31, 128)
        lr           = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        max_depth    = trial.suggest_int('max_depth', 3, 8)
        model = LGBMRegressor(n_estimators=n_estimators, num_leaves=leaves,
                              learning_rate=lr, max_depth=max_depth,
                              random_state=42, verbose=-1)
    elif method == "catboost":
        # iterations = trial.suggest_int('iterations', 5, 100, step=5)
        # depth      = trial.suggest_int('depth', 4, 10)
        iterations = trial.suggest_int('iterations', 5, 50, step=5)
        depth = trial.suggest_int('depth', 2, 5)
        lr         = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        l2_reg     = trial.suggest_float('l2_leaf_reg', 1e-4, 1e2, log=True)
        trial_dir = tempfile.mkdtemp(prefix=f"cb_trial_{trial.number}_", dir="/tmp")

        model = CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=lr,
            l2_leaf_reg=l2_reg, random_state=42,  verbose=0,
            thread_count=1, loss_function="MultiRMSE",
          allow_writing_files=False, train_dir=trial_dir,
        )
    elif method == "gradient_boosted_decision_trees":
        # n_estimators = trial.suggest_int('n_estimators', 5, 100, step=5)
        # max_depth    = trial.suggest_int('max_depth', 3, 10)
        n_estimators = trial.suggest_int('n_estimators', 5, 50, step=5)
        max_depth = trial.suggest_int('max_depth', 3, 5)
        lr           = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        subsample    = trial.suggest_float('subsample', 0.5, 1.0)
        model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                          learning_rate=lr, subsample=subsample,
                                          random_state=42)
    elif method == "version_extreme_random_forest":
        n_estimators = trial.suggest_int('n_estimators', 5, 100, step=5)
        max_depth    = trial.suggest_int('max_depth', 5, 30, step=5)
        min_split    = trial.suggest_int('min_samples_split', 2, 10)
        min_leaf     = trial.suggest_int('min_samples_leaf', 1, 4)
        model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                    min_samples_split=min_split, min_samples_leaf=min_leaf,
                                    random_state=42)
    else:
        raise ValueError(f"Unsupported machine learning model: {method}")

    if method in ["lightgbm", "bayesian_ridge", "adaboost", "gradient_boosted_decision_trees"]:
        model = MultiOutputRegressor(model)
    return model

# ╔════════════════════════════════════════════════════════════════════╗
# ║                   Sequence-model builder (torch)                   ║
# ╚════════════════════════════════════════════════════════════════════╝
class SequenceModel(nn.Module):
    """
    Unified torch model covering LSTM, GRU, RNN, CNN, CNN_LSTM, CNN_RNN,
    TCN, FFNN and a *minimal* TFT-like hybrid.  Returns both the forecast
    and an intermediate ‘embedding’ (Dense bottleneck).
    """

    def __init__(self, method, num_features, timesteps, cfg, output_dim, trial_id="0"):
        super().__init__()
        self.method      = method
        self.output_dim  = output_dim
        self.embed_size  = cfg['latent_dim']

        dr               = cfg['dropout_rate']
        units            = cfg.get('units', 32)
        filters          = cfg.get('filters', 32)
        kernel_size      = cfg.get('kernel_size', 2)
        cfg.setdefault('original_num_features', num_features)

        if method in {"lstm", "gru", "rnn"}:
            Cell = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[method]
            self.rnn = Cell(num_features, units, batch_first=True)
            self.layer_norm = nn.LayerNorm(units)
            backbone_out = units

        elif method == "cnn":
            self.backbone = nn.Sequential(nn.Conv1d(num_features, filters, kernel_size, padding='same'),
                nn.ReLU(), nn.Dropout(dr), nn.Conv1d(filters, filters, 1),
                nn.ReLU(), nn.Dropout(dr), nn.AdaptiveMaxPool1d(1),
                nn.Flatten())
            backbone_out = filters
        elif method == "cnn_lstm":
            self.conv = nn.Sequential(nn.Conv1d(num_features, filters, kernel_size,padding='same'), nn.ReLU(),
                                      nn.Dropout(dr), nn.AdaptiveMaxPool1d(timesteps // 2))
            self.rnn  = nn.LSTM(filters, units, batch_first=True)
            backbone_out = units

        elif method == "cnn_rnn":
            self.conv = nn.Sequential(nn.Conv1d(num_features, filters, kernel_size, padding='same'), nn.ReLU(),
                                      nn.Dropout(dr), nn.AdaptiveMaxPool1d(timesteps // 2))
            self.rnn  = nn.RNN(filters, units, batch_first=True)
            backbone_out = units

        elif method == "tcn":
            dilation = 2
            self.backbone = nn.Sequential(nn.Conv1d(num_features, filters, kernel_size, padding='same', dilation=dilation),
                nn.ReLU(), nn.Dropout(dr), nn.Conv1d(filters, filters, 1),
                nn.ReLU(), nn.AdaptiveMaxPool1d(1), nn.Flatten())
            backbone_out = filters

        elif method == "ffnn":
            self.backbone = nn.Sequential(nn.Linear(num_features, units), nn.ReLU(), nn.Dropout(dr),
                nn.Linear(units, units // 2), nn.ReLU(), nn.Dropout(dr))
            backbone_out = units // 2
        else:
            raise ValueError(f"Unknown method {method}")
        # Embedding + output head
        self.embed   = nn.Sequential(nn.Linear(backbone_out, self.embed_size), nn.ReLU())
        self.out_lin = nn.Linear(self.embed_size, output_dim)
        self.backbone_dim = backbone_out

    def _extract_backbone(self, x):
        if self.method in {"lstm", "gru", "rnn"}:
            out, _ = self.rnn(x)  # out: (B, T, units)
            out = self.layer_norm(out)  # LayerNorm gets a tensor, not a tuple
            backbone = out[:, -1]
        elif self.method in {"cnn", "tcn"}:
            x = x.transpose(1, 2)  # (B, F, T)
            z = self.backbone(x)  # already ends with Flatten
            return z

        elif self.method == "cnn_lstm":
            x = x.transpose(1, 2)  # (B, F, T)  ← move this line up
            x = self.conv(x)  # Conv1d now sees correct C_in
            x = x.transpose(1, 2)  # (B, T_new, C_out)
            _, (h, _) = self.rnn(x)
            backbone = h[-1]  # last layer hidden
        elif self.method == "cnn_rnn":
            x = x.transpose(1, 2)  # (B, F, T)
            x = self.conv(x)
            x = x.transpose(1, 2)  # (B, T_new, C_out)
            _, h = self.rnn(x)
            backbone = h[-1]
        elif self.method == "ffnn":
            return self.backbone(x) # x: (B, F)
        else:
            raise ValueError(f"Unknown method {self.method}")
        return backbone

    def forward(self, x):
        """
        x shape:
          * sequence models       : (B, T, F)
          * ffnn: (B, F)
        """
        backbone = self._extract_backbone(x)
        self._backbone = backbone
        emb_final = self.embed(backbone)
        out = self.out_lin(emb_final)
        return out, emb_final, backbone


def create_sequence_model(config, method, num_features, timesteps,
                          n_samples, epochs, ramp_epochs, warmup_epochs, trial_id="", add_embed=True,
                          full_model=True):
    """
    Torch replacement of the original TF builder.
    Returns: (model, batch_size)
    """
    from code_ai.data_treatment import get_config
    # ─── collect hyper-params (unchanged logic) ───────────────────────
    dropout_rate = get_config(config, "dropout_rate",
                               sv.HYPERPARAMETER_SPACE['dropout_rate']['min'],
                               sv.HYPERPARAMETER_SPACE['dropout_rate']['max'],
                               default=0.1,
                               step=sv.HYPERPARAMETER_SPACE['dropout_rate']['step'])
    weight_decay = get_config(config, "weight_decay",
                               sv.HYPERPARAMETER_SPACE['weight_decay']['min'],
                               sv.HYPERPARAMETER_SPACE['weight_decay']['max'],
                               default=0.0, log=True)
    initial_lr   = get_config(config, "initial_learning_rate",
                               *sv.HYPERPARAMETER_SPACE['ini_learning_rate'],
                               default=1e-3, log=True)
    batch_size   = get_config(config, "batch_size",
                               sv.BATCH_SIZE_OPTIONS,
                               default=sv.BATCH_SIZE_OPTIONS[0],
                               pop_type="categorical")
    latent_dim = get_config(config, "latent_dim", *sv.HYPERPARAMETER_SPACE['latent_dim_s'])
    model_size   = get_config(config, 'model_size',
                               sv.HYPERPARAMETER_SPACE['model_sizes'])
    min_u, max_u, step = sv.UNITS_MAP[model_size]['units']
    if method in ["lstm", "cnn", "gru", "rnn", "ffnn", "cnn_lstm", "cnn_rnn"]:
        units = get_config(config, 'units', min_u, max_u, step=step)
    if method in ["cnn", "cnn_lstm", "cnn_rnn", "tcn"]:
        filters = get_config(config, 'filters', min_u, max_u, step=step)
        kernel_size = get_config(config, "kernel_size", sv.HYPERPARAMETER_SPACE['kernel_sizes'],
                                  default=sv.HYPERPARAMETER_SPACE['kernel_sizes'][0],
                                  pop_type = "categorical" )

    else:
        filters = kernel_size = None

    cfg = dict(dropout_rate=dropout_rate, weight_decay=weight_decay,
               latent_dim=latent_dim, units=units if 'units' in locals() else None,
               filters=filters, kernel_size=kernel_size)

    output_dim = sv.STEPS_AHEAD

    model = SequenceModel(method, num_features, timesteps, cfg, output_dim, trial_id)

    param_groups = [{'params': model.parameters(), 'weight_decay': weight_decay}]
    optimizer = Adam(param_groups, lr=initial_lr, betas=(sv.OPTIMIZER_CONFIG['beta_1'], sv.OPTIMIZER_CONFIG['beta_2']),
                     eps=sv.OPTIMIZER_CONFIG['epsilon'])
    steps_per_epoch = math.ceil(n_samples / batch_size)
    total_steps = epochs * steps_per_epoch
    # T_0 = ramp_epochs * steps_per_epoch
    # warm_restart_sched = CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=T_0,  # period (in batches) before restart
    #     T_mult=1,  # keep the same period each cycle
    #     eta_min=initial_lr * 1e-3  # floor LR
    # )
    # model.lr_sched = warm_restart_sched
    safe_warm = min(warmup_epochs, max(1, epochs - 1))  # never > epochs‑1
    pct = safe_warm / epochs
    lr_sched = OneCycleLR(optimizer,
                            max_lr=initial_lr,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            pct_start=pct,
                            anneal_strategy=sv.LR_SCHEDULE['anneal_strategy'],
                            div_factor=25.0,
                            final_div_factor=1e3,
                            )
    model.lr_sched = lr_sched
    # attach for outside access
    model.optimizer  = optimizer

    return model, batch_size


# ╔════════════════════════════════════════════════════════════════════╗
# ║                   Forecast helper (unchanged)                      ║
# ╚════════════════════════════════════════════════════════════════════╝

def compute_model_metrics_train_validation(y_train_split, y_train_pred,
                                           y_val_split,   y_val_pred):
    """
    Return two DataFrames: train_error, validation_error
    (schema identical to the original code).
    """
    # --- train ---
    ev_tr = RegressionMetric(y_train_split, y_train_pred)
    train_error = pd.DataFrame([{
        'class': 0,
        'step' : 'Train',
        'rmse' : np.average(ev_tr.RMSE()),
        'mae'  : np.average(ev_tr.MAE()),
        'mbe'  : np.average(ev_tr.MBE())
    }])

    # --- validation ---
    ev_val = RegressionMetric(y_val_split, y_val_pred)
    validation_error = pd.DataFrame([{
        'class': 0,
        'step' : 'Validation',
        'rmse' : np.average(ev_val.RMSE()),
        'mae'  : np.average(ev_val.MAE()),
        'mbe'  : np.average(ev_val.MBE())
    }])

    return train_error, validation_error


# ╔════════════════════════════════════════════════════════════════════╗
# ║      machine_learning_models_ml  (identical, uses create_ml_model) ║
# ╚════════════════════════════════════════════════════════════════════╝
def machine_learning_models_ml(data, method):
    from code_ai.data_treatment import get_data_subset_continuous, summarize_params
    def retrial_top_k_ml(trials, data, method, k=5, re_runs=3, n_jobs=sv.CPUS):
        """
        Retest the best-k ML trials in parallel, hiding Optuna INFO logs and
        printing each trial’s summary in a single block using original trial numbers.
        Returns (best_params, best_original_trial).
        """
        # 1) select top-k by original MSE
        top = sorted(trials, key=lambda t: t.value)[:k]
        # store (orig_trial, orig_num, orig_mse, params)
        orig_entries = [(t, t.number, t.value, t.params.copy()) for t in top]

        # 2) silence Optuna INFO logs
        prev_level = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        sampler = optuna.samplers.TPESampler(seed=sv.RANDOM_SEED, warn_independent_sampling=False,
                                             consider_endpoints=True)
        study = optuna.create_study(direction='minimize', sampler=sampler, pruner = pruner)
        for _, _, _, params in orig_entries:
            study.enqueue_trial(params)

        # 4) objective that buffers prints
        def _objective(trial: optuna.trial.Trial):
            _cleanup()
            idx = trial.number  # matches order we enqueued
            orig_trial, orig_num, orig_mse, params = orig_entries[idx]
            lines = [f"▶️ Trial #{orig_num} (orig MSE={orig_mse:.5f})"]

            errors = [orig_mse]
            for i in range(re_runs):
                try:
                    err = objective(trial, data, method)
                    errors.append(err)
                    lines.append(f"   rerun {i + 1}/{re_runs}: MSE={err:.5f}")
                except Exception:
                    lines.append(f"   rerun {i + 1}: failed")

            mean_e = float(np.mean(errors))
            std_e = float(np.std(errors))
            lines.append(f"   → mean={mean_e:.5f}, σ={std_e:.5f}\n")

            print("\n".join(lines))
            # store σ for tie-breaks
            trial.set_user_attr("std", std_e)
            return mean_e

        # 5) run the k enqueued trials in parallel
        k = min(k, len(orig_entries))
        study.optimize(_objective, n_trials=k, n_jobs=n_jobs)
        optuna.logging.set_verbosity(prev_level)
        # 6) pick best by (mean, σ)
        best_trial = min(study.trials, key=lambda t: (t.value, t.user_attrs.get("std", 0.0)))
        best_idx = best_trial.number
        orig_trial, orig_num, _, params = orig_entries[best_idx]
        mean_best = best_trial.value
        std_best = best_trial.user_attrs["std"]

        print(f"✅ Selected trial #{orig_num}: mean MSE={mean_best:.5f} ±{std_best:.5f}")
        return params.copy(), orig_trial
    def objective(trial=None, data=None, method=None):
        _cleanup()
        from code_ai.data_treatment import get_data_subset_continuous
        t_start, t_end, v_start, v_end = get_data_subset_continuous(len(data["X_train"]), len(data["X_val"]))
        model = create_ml_model(trial, method)

        y_train = np.asarray(data["y_train"][t_start:t_end].copy(),  dtype=np.float32)
        y_val = np.asarray(data["y_val"][v_start:v_end].copy(),  dtype=np.float32)

        X_train = np.asarray(data["X_train"][t_start:t_end].copy(),  dtype=np.float32)
        X_val = np.asarray(data["X_val"][v_start:v_end].copy(),  dtype=np.float32)
        history = model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        mse = mean_squared_error(y_val, predictions)
        _cleanup()
        return mse

    # Set random seed for reproducibility
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    sampler = optuna.samplers.TPESampler(seed=sv.RANDOM_SEED, warn_independent_sampling=False, consider_endpoints=True)
    trials = sv.N_TRIALS if method != "linear_regression" else 1
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner = pruner)
    study.optimize(partial(objective, data=data, method=method), n_trials=trials, n_jobs=sv.CPUS, gc_after_trial=False)

    # Retrieve best hyperparameters
    best_params, best_trial = retrial_top_k_ml(study.trials, data=data, method=method, k=sv.RETEST_TRIALS)
    param_summary = summarize_params(study)

    print(f"Best hyperparameters for {method}: {best_params}")

    # Train the model with the best hyperparameters
    best_model = create_ml_model(best_trial, method)

    X_tr = np.ascontiguousarray(data["X_train"])
    y_tr = np.ascontiguousarray(data["y_train"])

    best_model.fit(X_tr, y_tr)
    y_train_pred = best_model.predict(data["X_train"])
    y_val_pred = best_model.predict(data["X_val"])
    predictions = best_model.predict(data["X_test"])

    train_error, validation_error = compute_model_metrics_train_validation(data["y_train"], y_train_pred, data["y_val"], y_val_pred)

    return predictions, best_params, param_summary, train_error, validation_error, best_model


# ╔════════════════════════════════════════════════════════════════════╗
# ║        machine_learning_models_sequence (torch training loop)      ║
# ╚════════════════════════════════════════════════════════════════════╝
class DistillDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return ( self.X[i],
                 self.y[i])


def predict_all(model, X):
    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():  # ← turn off autograd
        x_t = torch.from_numpy(_np(X)).float().to(device)
        y_pred, _, _ = model(x_t)  # student‑only
    return y_pred.cpu().numpy()

def machine_learning_models_sequence(data, method, data_p=None):
    """
    Hyper-parameter tuning (Optuna) + retrial + final training
    for all deep-learning sequence models, implemented in PyTorch.
    Returns:
        predictions, best_params, param_summary,
        train_error_df, val_error_df, trained_model
    """
    from code_ai.data_treatment import get_data_subset_continuous, summarize_params
    # 1.  Helpers
    def objective(trial_or_params, data, method, num_features, timesteps,
                  final: bool = False, n_samples=None):
        """One Optuna trial (surrogate training run)."""

        _cleanup()

        # history containers
        history = {"loss": [], "val_loss": []}

        warmup_epochs = sv.OPTUNA_CONFIG["warmup_steps"]
        # determine data slice and epoch counts
        if not final:
            t0, t1, v0, v1 = get_data_subset_continuous(len(data["X_train"]), len(data["X_val"]))
            epochs, patience = sv.TRIAL_EPOCHS, sv.TRIAL_PATIENCE
            config = trial_or_params
        else:
            t0, t1, v0, v1 = 0, len(data["X_train"]), 0, len(data["X_val"])
            epochs, patience = sv.FINAL_EPOCHS, sv.FINAL_PATIENCE #*3 if use_teacher_ctx else sv.FINAL_PATIENCE
            config = trial_or_params  # dict of best_params
            n_samples = n_samples or len(data["X_train"])
        ramp_epochs = int(0.6 * epochs)  # Longer ramp for better knowledge transfer

        # build model, optimizer, scheduler
        model, batch_size = create_sequence_model(config, method, num_features, timesteps,
                                                  n_samples or sv.TRIAL_SPLIT, epochs, ramp_epochs, warmup_epochs,
                                                  trial_id=str(getattr(trial_or_params, "number", "final")),
                                                  add_embed=final, full_model=final)
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        device = torch.device("cpu")
        model.to(device)
        sched  =  model.lr_sched
        main = model.optimizer

        # prepare data loaders
        def to_tensor(arr):
            return torch.from_numpy(_np(arr)).float().to(device)

        Xtr = to_tensor(data["X_train"][t0:t1]); Ytr = to_tensor(data["y_train"][t0:t1])
        Xva = to_tensor(data["X_val"][v0:v1]); Yva = to_tensor(data["y_val"][v0:v1])

        train_ds = DistillDataset(Xtr, Ytr)
        va_ds   = DistillDataset(Xva, Yva)
        tr_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

        best_val, wait = float("inf"), 0
        bar = trange(epochs, desc=f"{method}  final-fit", dynamic_ncols=True) if final else range(sv.TRIAL_EPOCHS)
        for epoch in bar:  # bar = tqdm / range
            # ─── TRAIN ─────────────────────────────────────────────────────────
            model.train()
            tr_loss_acc = []  # blended loss
            tr_direct_acc, tr_pred, tr_hint = [], [], []
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                y_p, _, _ = model(xb)
                loss = F.mse_loss(y_p, yb)

                main.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                main.step()
                sched.step()

                tr_loss_acc.append(loss.item())
                tr_direct_acc.append(loss.item())

            # log epoch averages
            history["loss"].append(np.mean(tr_loss_acc))
            history.setdefault("direct_loss", []).append(np.mean(tr_direct_acc))

            # ─── VALIDATION ────────────────────────────────────────────────────
            model.eval()
            val_loss_acc = []
            with torch.no_grad():
                for xb,  yb in va_dl:
                    xb,  yb = xb.to(device),  yb.to(device)
                    s_pred, _, _ = model(xb)
                    v_loss = F.mse_loss(s_pred, yb)
                    val_loss_acc.append(v_loss.item())

            history["val_loss"].append(np.mean(val_loss_acc))
            history.setdefault("val_direct", []).append(np.mean(val_loss_acc))

            # ─── EARLY-STOP / PROGRESS BAR ────────────────────────────────────
            val_metric = history["val_loss"][-1]  # or val_direct

            if hasattr(bar, "set_postfix"):  # tqdm bars have this
                bar.set_postfix(loss=f"{history['loss'][-1]:.4f}", val=f"{val_metric:.4f}")
            if val_metric < best_val - 1.0e-4:
                best_val, wait = val_metric, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            if wait >= patience:
                break

            if isinstance(trial_or_params, optuna.trial.Trial):
                trial_or_params.report(val_metric, epoch)
                if trial_or_params.should_prune():
                    raise optuna.TrialPruned()

        _cleanup()
        if not final:
            return best_val
        model.load_state_dict(best_state)

        y_tr_pred = predict_all(model, data["X_train"])
        y_va_pred = predict_all(model, data["X_val"])
        train_error, val_error = compute_model_metrics_train_validation(data["y_train"], y_tr_pred,
                                                                        data["y_val"], y_va_pred)

        return model, history, train_error, val_error, batch_size

    def retrial_top_k_dl(trials, data, method, num_features, timesteps, k=5, re_runs=2, n_jobs=sv.CPUS):
        """
        Re-evaluate the top-k completed trials `re_runs` times each and
        pick the one with the best (mean, std) MSE.
        """
        completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            raise RuntimeError("No completed trials to retrial.")

        top = sorted(completed, key=lambda t: t.value)[:k]
        entries = [(t.number, t.params) for t in top]
        prev_lvl = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # new study with fixed params
        study = optuna.create_study(direction="minimize")
        for _, params in entries:
            study.enqueue_trial(params.copy())

        def _fixed_objective(inner_trial):
            # index in the list we enqueued
            idx = inner_trial.number
            orig_num, params = entries[idx]
            orig_mse = trials[orig_num].value  # original study score
            rank = idx  # 0-based rank in top-k

            lines = [f"▶️ Trial #{orig_num} (rank {rank}) " f"original MSE={orig_mse:.5f}"]
            results = [orig_mse]  # first element is the original
            for r in range(re_runs):
                try:
                    mse = objective(params.copy(), data, method, num_features, timesteps)
                    results.append(mse)
                    lines.append(f"   rerun {r + 1}/{re_runs}: MSE={mse:.5f}")
                except Exception as e:
                    tb_last = traceback.format_exception_only(type(e), e)[-1].strip()
                    lines.append(f"   rerun {r + 1}: failed → {tb_last}")

            mean_mse, std_mse = float(np.mean(results)), float(np.std(results))
            lines.append(f"   → mean={mean_mse:.5f}, σ={std_mse:.5f}\n")
            # flush everything at once
            print("\n".join(lines))

            inner_trial.set_user_attr("std", std_mse)
            return mean_mse

        study.optimize(_fixed_objective, n_trials=len(entries), n_jobs=n_jobs)
        optuna.logging.set_verbosity(prev_lvl)

        # pick by (mean, std)
        best = min(study.trials, key=lambda t: (t.value, t.user_attrs.get("std", 0.0)))
        best_idx = best.number
        orig_num, best_params = entries[best_idx]
        mean_best = best.value
        std_best = best.user_attrs["std"]
        # 6) summary printout
        print(f"✅ Selected trial #{orig_num}: mean MSE={mean_best:.5f} ±{std_best:.5f}")
        return best_params

    # 2.  Optuna search
    num_features = data["X_train"].shape[2] if method != "ffnn" else data["X_train"].shape[1]
    timesteps    = data["X_train"].shape[1] if method != "ffnn" else data["X_train"].shape[0]

    pruner = MedianPruner(n_startup_trials=sv.OPTUNA_CONFIG["startup_trials"], n_warmup_steps=sv.OPTUNA_CONFIG["warmup_steps"])
    sampler = optuna.samplers.TPESampler(seed=sv.RANDOM_SEED, warn_independent_sampling=False, consider_endpoints=True,
                                         n_startup_trials=10)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    study.optimize(lambda tr: objective(tr, data, method, num_features, timesteps), n_trials=sv.N_TRIALS,
                   n_jobs=sv.CPUS, gc_after_trial=False)

    # 3.  Retrial of best-k (same logic as original)
    best_params = retrial_top_k_dl(study.trials, data, method, num_features, timesteps, k=sv.RETEST_TRIALS)
    param_summary = summarize_params(study)
    # 4.  Final full-data training with best params
    model, history, train_error, val_error, batch_size = objective(best_params, data, method, num_features, timesteps,
                                                                    final=True, n_samples=data["X_train"].shape[0])
    if data_p is not None:
        model_p, history, train_error, val_error, batch_size = objective(best_params, data_p, method, num_features,
                                                                       timesteps,
                                                                       final=True, n_samples=data["X_train"].shape[0])
        return model, model_p
    else:
        history_train(_Shim(history), method)
        _cleanup()
        predictions = predict_all(model, data["X_test"])
        return predictions, best_params, param_summary, train_error, val_error, model