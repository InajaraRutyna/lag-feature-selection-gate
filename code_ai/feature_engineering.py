# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import joblib
from scipy.signal import istft
from config.load import *
import pandas as pd
from scipy.signal import stft
from numpy.fft import rfft, irfft
from scipy.signal import get_window
import pywt
from joblib import Parallel, delayed
from vmdpy import VMD
import optuna
import optuna.logging
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from typing import Optional, List, Tuple
import gc
import traceback
import warnings
from code_ai.gate_torch import *
from code_ai.data_treatment import *
from code_ai.plots import plot_correlation_map_from_sv
import code_ai.shared_variables as sv

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

warnings.filterwarnings("ignore",
    message=r"The reported value is ignored because this `step` \d+ is already reported."
)

FLOATX = np.float32          # change to np.float64 if you need full precision
N_JOBS = sv.CPUS

os.environ["TORCH_COMPILE"] = "0"
# keep_frac.py  (revision: 10 % sampling) ─────────────────────────────────────
# ------------------------------ #
def _keep_filter(
    vec: np.ndarray,
    frac: float | str,
) -> list[int]:
    """
    vec         — 1D importance scores
    frac        — float in (0,1]
    """
    vec = vec / (vec.max() + 1e-9)
    vec = np.asarray(vec, dtype=float)

    L = vec.size
    order = np.argsort(vec)[::-1]
    k = min(max(int(np.ceil(frac * L)) - 1, 0), L - 1)
    keep = order[: k + 1].tolist()
    return sorted(keep)

def compute_mi_score(X, y, n_neighbors=3, block_feats=8):
    """
    X: (N, L, F), y: (N, H)
    Returns MI per (lag, feature): (L, F)
    Processes features in blocks to avoid a giant (N, L·F) and avoid per-column jobs.
    """
    X = np.asarray(X, dtype=np.float32, order="C")
    y = np.asarray(y, dtype=np.float32, order="C")
    if y.ndim == 2:         # your code uses mean over H
        ym = y.mean(axis=1)
    else:
        ym = y

    N, L, F = X.shape
    out = np.empty((L, F), dtype=np.float32)

    B = max(1, int(block_feats))
    for f0 in range(0, F, B):
        f1 = min(f0 + B, F)
        # This reshape is a view; no big allocation.
        X2d = X[:, :, f0:f1].reshape(N, (f1 - f0) * L)
        mi = mutual_info_regression(X2d, ym, n_neighbors=n_neighbors)
        out[:, f0:f1] = mi.reshape(L, f1 - f0)

    return out

def compute_pearson_score(X: np.ndarray, y: np.ndarray, block_feats: int = 16) -> np.ndarray:
    """
    Mean absolute Pearson correlation between each (lag, feature) in X (N,L,F)
    and each target dim in y (N,H). Returns (L,F).
    RAM use ~ O(L*block_feats + N*H). No (L*F,·) allocations.

    block_feats: process this many features at a time (1 = minimal RAM).
    """
    eps = 1e-12
    X = np.asarray(X, dtype=np.float32, order="C")
    y = np.asarray(y, dtype=np.float32, order="C")
    if y.ndim == 1:
        y = y[:, None]

    N, L, F = X.shape
    H = y.shape[1]

    # center y once, and precompute its norm per dim
    y0 = y - y.mean(axis=0, keepdims=True)          # (N,H)
    sy = np.sqrt((y0 * y0).sum(axis=0)) + eps       # (H,)

    out = np.empty((L, F), dtype=np.float32)

    B = max(1, int(block_feats))
    for f0 in range(0, F, B):
        f1 = min(f0 + B, F)
        Xb = X[:, :, f0:f1]                         # view (N,L,B)

        # per-(lag,feat) stats without building xc
        meanx = Xb.mean(axis=0)                     # (L,B)
        sumsq = (Xb * Xb).sum(axis=0)               # (L,B)
        sx = np.sqrt(np.maximum(sumsq - N * (meanx ** 2), 0.0)) + eps  # (L,B)

        # accumulate over H to avoid (L,B,H) tensor
        acc = np.zeros((L, f1 - f0), dtype=np.float32)
        for h in range(H):
            # num = sum_n Xb[n,l,b] * y0[n,h]  ->  (L,B)
            num = np.einsum('nlb,n->lb', Xb, y0[:, h], optimize=True)
            acc += np.abs(num) / (sx * sy[h])

        out[:, f0:f1] = acc / H

    return out

def compute_ccf_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    For each feature f in X and each lag l in [0..L-1], compute the
    Pearson correlation between X[:, l, f] and y (averaged over multi-output).
    Returns a (L, F) array of absolute correlations.
    """
    N, L, F = X.shape
    y_flat = y.mean(axis=1)

    mat = np.zeros((L, F), dtype=np.float32)
    # zero-mean and std once
    y0 = y_flat - y_flat.mean()
    sy = np.sqrt((y0**2).sum()) + 1e-12

    for f in range(F):
        xf = X[:, :, f]  # (N, L)
        # for each lag l compute corr(xf[:,l], y_flat)
        # vectorized over l if you like, but simple loop is fine for moderate F
        for l in range(L):
            x0 = xf[:, l] - xf[:, l].mean()
            sx = np.sqrt((x0**2).sum()) + 1e-12
            mat[l, f] = np.abs((x0 * y0).sum() / (sx * sy))

    return mat

def filter_methodology(score, decay_lag, incentive_feat, frac: float | str, sequence=True):
    """
       score:  (L, F) raw importance
       decay:  (L,1) time decay weights
       frac:   single float or (lag_frac, feat_frac) or "auto"
       order candidates with both decays applied,
        count their contribution with only the lag decay.
       """
    # 1) apply decay & normalize per-feature to [0,1]

    if sequence:
        # unpack your two fractions
        L, F = score.shape
        lag_frac, feat_frac = (frac, frac) if not isinstance(frac, tuple) else frac
        feat_score = score * incentive_feat
        kept_feats = _keep_filter(feat_score.sum(axis=0), feat_frac)  # which cols to consider

        if sv.DECAY:
            feat_score = feat_score * decay_lag
        lf_mask = np.zeros((L, F), dtype=bool)
        for f in kept_feats:
            col = feat_score[:, f]
            lags = _keep_filter(col, lag_frac)  # top-k by score
            lf_mask[lags, f] = True
        return lf_mask

    else:
        # flat case: just pick a fraction of the flattened, normalized matrix
        idx_score = score * decay_lag * incentive_feat
        flat_mass = idx_score.ravel()
        keep_idx = _keep_filter(flat_mass, frac)#, z_thresh=-0.5)
        return keep_idx

def compute_score(X,y, feat=""):
    if sv.feature_filtering_method == "Pearson" or sv.feature_filtering_method == "gate":
        mat= compute_pearson_score(X,y)
    elif sv.feature_filtering_method == "CCF":
        mat = compute_ccf_score(X, y)
    elif sv.feature_filtering_method == "MI":
        mat = compute_mi_score(X, y)
    else:
        raise ValueError(f"Unknown method '{sv.feature_filtering_method}'")

    # ── 2. lag-side decay ───────────────────────────────
    score = mat / (mat.max(axis=0, keepdims=True) + 1e-9)
    # ── 2. lag-side decay ───────────────────────────────
    L, F = mat.shape  # number of lags
    decay_lag = (np.linspace(0, 1, L))[:, None]
    # ── 3. feature-side decay (volatility) ──────────────
    feat_var = X.var(axis=(0, 1))
    if feat == "volatile":
        sigma = feat_var / (feat_var.max()+ 1e-9)
        incentive_feat = np.exp(sv.INCENTIVE_FEAT * sigma)[None, :]
    # ── 3. feature-side decay (smotnes) ──────────────
    elif feat == "smotnes":
        mu = X.mean(axis=(0, 1))
        cv = np.sqrt(feat_var) / (mu + 1e-9)
        cv /= cv.max()
        incentive_feat = np.exp(-sv.INCENTIVE_FEAT * (1 - cv)) #coefficient of variation so a large mean + large var doesn’t dominate.
    else:
        incentive_feat = np.ones((1, F))
    del mat
    return score, decay_lag, incentive_feat


def dl_filter(X_t: np.ndarray, y_t: np.ndarray, X_v: np.ndarray, y_v: np.ndarray, best_params: dict,
    *, cpt_error=False, simple=False):
    """
    Given the “raw” corr‐based score, the decay/incentive,
    and Optuna’s best_params = {
      "g_lag": …,
      "g_feat": …,
    }
    retrain the tiny proxy on the full pre‐filtered data,
    blend scores, and return:
      - X_final: the filtered X_full,
      - keep_lags, keep_feats
    """



    # 2) retrain proxy on full coarse data
    if simple:
        g_lag, g_feat = best_params["g_lag"], best_params["g_feat"]
        # g_lag, g_feat = 1.0, best_params["g_feat"]
        score, decay_lag, incentive_feat = sv.score_data
        lf_mask = filter_methodology(score=score, decay_lag=decay_lag, incentive_feat=incentive_feat,
                                                   frac=(g_lag, g_feat), sequence=True)
        error = None
        if cpt_error:
            X_t_f = X_t * lf_mask[None, :, :]
            X_v_f = X_v * lf_mask[None, :, :]

            X_t_f = X_t_f.reshape(X_t_f.shape[0], X_t_f.shape[1] * X_t_f.shape[2])
            X_v_f = X_v_f.reshape(X_v_f.shape[0], X_v_f.shape[1] * X_v_f.shape[2])
            model = ExtraTreesRegressor(n_estimators=40,  # small forest → fast
                                        max_depth=4,  # shallow trees keep training O(n)
                                        min_samples_leaf=8,
                                        random_state=42,
                                        n_jobs=-1).fit(X_t_f, y_t)
            prediction = model.predict(X_v_f)
            error = mean_squared_error(prediction, y_v)
    else:
        win, feats = X_t.shape[1], X_t.shape[2]

        num_epochs = sv.TRIAL_EPOCHS if cpt_error else sv.FINAL_EPOCHS
        model, error = gate_train(X_t, y_t, X_v, y_v, win, feats, best_params, num_epochs, use_bar=(not cpt_error))
        if not cpt_error:
            lf_mask = compare_gates_torch(model)
        else:
            lf_mask = None

    return lf_mask, error

def apply_best_ml(X: np.ndarray,            # shape (N, D=L*F)
    y: np.ndarray,                 # shape (N, H)
    X_val: np.ndarray,
    y_val: np.ndarray,
    score: np.ndarray,             # shape (L, F)
    decay_lag: np.ndarray,         # shape (L, 1)
    incentive_feat: np.ndarray,    # shape (1, F)
    params: dict,
    *,
    cpt_error=False,) -> Tuple[List[int], float]:
    q_frac = params["q_frac"]

    # 6) final flat‐filter
    final_keep = filter_methodology(score, decay_lag, incentive_feat, q_frac, sequence=False)
    error = None
    if cpt_error:
        if len(final_keep) == 0:
            error =  np.inf
        else:
            X_tr = X[:, final_keep]
            X_vl = X_val[:, final_keep]
            mdl = MultiOutputRegressor(LinearRegression()).fit(X_tr, y)
            prediction = mdl.predict(X_vl)
            error = mean_squared_error(prediction, y_val)
    return final_keep, error
# __________________________________________________________________

class NoImprovementStop:
    """Stop after `patience` consecutive trials without a better value."""
    def __init__(self, patience: int = 50, tol: float = 1e-4):
        self.patience = patience
        self.tol      = tol
        self.best     = float("inf")
        self.wait     = 0

    def __call__(self, study: optuna.study.Study,
                       trial:  optuna.trial.FrozenTrial):
        if trial.value < self.best - self.tol:
            self.best = trial.value
            self.wait = 0                    # reset counter
        else:
            self.wait += 1
        if self.wait >= self.patience:       # no progress for `patience`
            study.stop()

def _cleanup():
     torch.cuda.empty_cache()
     gc.collect()

def mask_compact(mask, threshold=0.0,):
    """
    mask: (L,F) bool/float. Cells > threshold are kept.
    Prints: '2[0,3,7], 4[1,5], 10[0,2,6]' (only features with any lags).
    """
    M = np.asarray(mask)
    assert M.ndim == 2, "mask must be (L,F)"
    L, F = M.shape
    parts = []
    for f in range(F):
        lags = np.where(M[:, f] > threshold)[0]
        if lags.size:
            parts.append(f"{f}[{','.join(map(str, lags))}]")
    s = ", \n".join(parts)
    return s

def reduce_lags_features(
    X: np.ndarray,
    mask: np.ndarray,
    add_lag_index: bool = False,
    add_full_lag: bool = False,
):
    """
    X:    (N, L, F)
    mask: (L, F) boolean

    Returns:
      if add_full_lag=False:
          (N, K, Fnew) or (N, K, 2*Fnew)
      if add_full_lag=True:
          (N, L, Fnew) or (N, L, 2*Fnew)

    Notes:
      - Requires equal #lags per kept feature when add_full_lag=False (compressed).
      - When add_full_lag=True, equal-K is NOT required (works with any mask).
      - lag_norm is in [0,1], with denom = max(1, L-1).
    """
    assert X.ndim == 3, "X must be (N,L,F)"
    N, L, F = X.shape
    assert mask.shape == (L, F), "mask must be (L,F)"
    assert mask.dtype == bool, "mask must be boolean"

    fidx = np.flatnonzero(mask.any(axis=0))
    if fidx.size == 0:
        raise ValueError("mask selects no features")

    denom = float(max(1, L - 1))

    # ─────────────────────────────────────────────────────────────
    # Mode 1: aligned full timeline (N, L, Fnew), zeros where masked
    # ─────────────────────────────────────────────────────────────
    if add_full_lag:
        # keep only selected features, then zero out masked lags
        Xf = X[:, :, fidx].astype(np.float32, copy=False)          # (N,L,Fnew)
        mf = mask[:, fidx].astype(bool, copy=False)               # (L,Fnew)
        Xr = Xf * mf[None, :, :]                                  # broadcast mask

        if not add_lag_index:
            return Xr

        # lag channel: same for all features (but we duplicate per feature to keep shape consistent)
        lag_norm_1d = (np.arange(L, dtype=np.float32) / denom)    # (L,)
        lag_norm = np.broadcast_to(lag_norm_1d[None, :, None], (N, L, fidx.size))
        return np.concatenate([Xr, lag_norm], axis=2)             # (N,L,2*Fnew)

    # ─────────────────────────────────────────────────────────────
    # Mode 2: compressed (N, K, Fnew), requires equal K per kept feature
    # ─────────────────────────────────────────────────────────────
    K_per_feat = mask[:, fidx].sum(axis=0)
    K = int(K_per_feat[0])
    if not np.all(K_per_feat == K):
        raise ValueError(
            f"mask must select the SAME # of lags per kept feature for compressed mode; got {K_per_feat.tolist()}"
        )

    # lag indices per kept feature: (K, Fnew)
    Ilags = np.column_stack([np.flatnonzero(mask[:, j]) for j in fidx]).astype(np.int64)

    # gather values -> (N, K, Fnew)
    Xr = X[np.arange(N)[:, None, None], Ilags[None, :, :], fidx[None, None, :]].astype(np.float32, copy=False)

    if not add_lag_index:
        return Xr

    # per-feature lag index channel -> (N, K, Fnew)
    lag_norm = (Ilags.astype(np.float32) / denom)[None, :, :]     # (1,K,Fnew)
    lag_norm = np.broadcast_to(lag_norm, (N, K, fidx.size))
    return np.concatenate([Xr, lag_norm], axis=2)                 # (N,K,2*Fnew)


def data_filter(
    X_scaled: np.ndarray,
    sequence: bool = True,
    simple: bool = False) -> tuple[tuple[list[int], list[int]], list[int] | None]:
    """
             Determine optimal keep fractions for lag and feature selection.
    @@
            best_seq : tuple of two lists of ints
                (lag indices, feature indices) referring back to the *original* X_tr
            best_ml : list of ints or None
                Flat feature indices (on the post-sequence flatten) or None if is_sequence=True
         """
    def retrial_top_k(trials, k=5, re_runs=2, sequence=True, n_jobs=sv.CPUS):
        """
        Retest the best-k Optuna trials in parallel, hiding Optuna INFO logs and
        printing each trial’s full summary in one block using the original trial numbers.
        """
        top = sorted(trials, key=lambda t: t.value)[:k] # 1) pick top-k by original MSE
        orig_info = [(t.number, t.value, t.params.copy()) for t in top] # store (orig_num, orig_mse, params) for each
        prev_level = optuna.logging.get_verbosity() # 2) silence Optuna INFO lines
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize') # 3) enqueue them into a fresh study
        for _, _, params in orig_info:
            study.enqueue_trial(params)

        def _objective(trial: optuna.trial.Trial):
            idx = trial.number  # matches order of enqueue_trial
            orig_num, orig_mse, params = orig_info[idx]
            lines = []
            lines.append(f"▶️ Trial #{orig_num} (orig MSE={orig_mse:.5f})")
            errors = [orig_mse]
            for i in range(re_runs):
                try:
                    if sequence:
                        e = objective_DL(fixed_params=params.copy())
                    else:
                        e = objective_ML(fixed_params=params.copy())
                    errors.append(e)
                    lines.append(f"   rerun {i + 1}/{re_runs}: MSE={e:.5f}")
                except Exception:
                    lines.append(f"   rerun {i + 1}: failed")

            mean_e = float(np.mean(errors))
            std_e = float(np.std(errors))
            lines.append(f"   → mean={mean_e:.5f}, σ={std_e:.5f}\n")

            # now print the entire block at once
            print("\n".join(lines))
            trial.set_user_attr("std", std_e)
            return mean_e

        # 5) run the k trials in parallel
        study.optimize(_objective, n_trials=k, n_jobs=n_jobs)
        optuna.logging.set_verbosity(prev_level)
        # 6) pick best by (mean, σ)
        best_trial = min(study.trials, key=lambda t: (t.value, t.user_attrs.get("std", 0.0)))
        best_idx = best_trial.number
        orig_num, _, best_params = orig_info[best_idx]
        mean_best = best_trial.value
        std_best = best_trial.user_attrs["std"]

        print(f"✅ Selected trial #{orig_num}: mean MSE={mean_best:.5f} ±{std_best:.5f}")
        return best_params

    # ── small random subsample ────────────────────────────────────────────
    def subsample(X_tr, y_tr, X_vl, y_vl, val_frac=0.2):
        n_val = int(val_frac * sv.TRIAL_SPLIT)
        n_train = sv.TRIAL_SPLIT - n_val
        assert X_tr.shape[0] >= n_train, "not enough train samples"
        assert X_vl.shape[0] >= n_val, "not enough val samples"

        tr_idx = np.random.choice(X_tr.shape[0], n_train, replace=False)
        vl_idx = np.random.choice(X_vl.shape[0], n_val, replace=False)

        X_tr_s, y_tr_s = X_tr[tr_idx], y_tr[tr_idx]
        X_vl_s, y_vl_s = X_vl[vl_idx], y_vl[vl_idx]

        # ---- clean NaN / ±Inf ----------------------------------
        X_tr_s = np.nan_to_num(X_tr_s, nan=0.0, posinf=0.0, neginf=0.0)
        y_tr_s = np.nan_to_num(y_tr_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_vl_s = np.nan_to_num(X_vl_s, nan=0.0, posinf=0.0, neginf=0.0)
        y_vl_s = np.nan_to_num(y_vl_s, nan=0.0, posinf=0.0, neginf=0.0)
        # ---------------------------------------------------------

        return X_tr_s, y_tr_s, X_vl_s, y_vl_s

    def objective_DL(trial=None, fixed_params=None):
        try:
            _cleanup()
            # ── 1) trial hyperparameters ──────────────────────────
            if fixed_params is not None:
                params = fixed_params
            else:
                params = {}
                if sv.feature_filtering_method != "Gate":
                    params["g_lag"] = trial.suggest_float("g_lag", sv.KEEP_FRAC[0], sv.KEEP_FRAC[1], step=sv.KEEP_FRAC[2])
                    params["g_feat"] = trial.suggest_float("g_feat", sv.KEEP_FRAC[0], sv.KEEP_FRAC[1], step=sv.KEEP_FRAC[2])
                # gate
                else:
                    params["init_logit"] = trial.suggest_float("init_logit", -1.5, -0.5)
                    params["temp_start"] = trial.suggest_float("temp_start", 0.35, 0.60)  # colder early
                    params["temp_end"] = trial.suggest_float("temp_end", 0.02, 0.08)  # sharper by mid-training
                    params["sparsity_warmup_frac"] = trial.suggest_float("sparsity_warmup_frac", 0.03, 0.06)
                    params["auto_eps"] = trial.suggest_float("auto_eps", 0.025, 0.060)  # accept more cuts
                    params["prune_frac_k"] = trial.suggest_float("prune_frac_k", 0.30, 0.45)  # ✔ good
                    params["prune_frac_M"] = trial.suggest_float("prune_frac_M", 0.15, 0.35)  # allow faster M shrink
                    params["auto_prune_every"] = trial.suggest_int("auto_prune_every", 1, 2)
                    params["dropout"] = trial.suggest_float("dropout", 0.02, 0.06)

                # ── 2) train tiny proxy to get W_proxy ──────────────
            X_tr_s, y_tr_s, X_vl_s, y_vl_s = subsample(X_tr, y_tr_f, X_val, y_val_f)

            _, error = dl_filter(X_tr_s, y_tr_s, X_vl_s, y_vl_s, params, cpt_error=True, simple=simple)
            _cleanup()
            return error

        except optuna.TrialPruned:
            # Let Optuna handle pruning from its own callbacks
            raise
        except Exception as e:
            print("[objective_sequence] 🎯 Trial failed with exception:")
            traceback.print_exc()  # ← full stack trace!
            # builder or data errors → count as “bad” trial, but don’t prune
            print(f"[objective_sequence] non-pruning error: {e}")
            _cleanup()
            return np.inf

    # ── DL search ────────────────────────────────────────────────────
    if sv.feature_filtering_method == "Full":
        # X_scaled: (N, L, F)
        N, L, F = X_scaled.shape
        lf_mask = np.ones((L, F), dtype=bool)  # keep everything
        best_ml = list(range(L * F))                     # keep all flattened indices
        return lf_mask, best_ml
    X_tr, X_val, _ = split_time_series(X_scaled)
    y_tr_f, y_val_f = sv.DATA_Y["y_train"], sv.DATA_Y["y_val"]
    if sv.feature_filtering_method == "Gate":
        simple = False
    else:
        simple = True
        sv.score_data = compute_score(X_tr, y_tr_f)
        print("score computed")
        plot_correlation_map_from_sv()

    sampler = TPESampler(seed=sv.RANDOM_SEED, n_startup_trials=20)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    stop_cb_d = NoImprovementStop(patience=50, tol=1e-5)
    study_seq = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    study_seq.optimize(objective_DL, n_trials=sv.N_TRIALS, callbacks=[stop_cb_d], n_jobs=sv.CPUS, gc_after_trial=True)

    best_params = retrial_top_k(study_seq.trials, k=sv.RETEST_TRIALS, sequence=True)
    lf_mask, _ = dl_filter(X_tr, y_tr_f, X_val, y_val_f, best_params, simple=simple)
    plot_correlation_map_from_sv(lf_mask)
    print(f"Best dl sections:\n {mask_compact(lf_mask)}")

    return lf_mask