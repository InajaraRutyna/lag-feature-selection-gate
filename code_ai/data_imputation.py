# ─────────────────────────────────────────────────────────────────────────────
# Calendar-window DL imputation, memory-aware multiprocessing
# ─────────────────────────────────────────────────────────────────────────────
import os, io, contextlib, warnings, logging, sys, types, multiprocessing as mp
from functools import partial, lru_cache
import numpy as np, pandas as pd
from tqdm.auto import tqdm
import psutil
from loguru import logger
logger.disable("pypots")

os.environ.setdefault("OMP_NUM_THREADS", "1")       # keep every worker single-threaded
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)

# ── 1. ALWAYS provide a stub for `ai4ts`  (parent & children) ──────────────
def _inject_ai4ts_stub():
    if "ai4ts" in sys.modules:            # already there → leave it
        return
    ai4ts_stub        = types.ModuleType("ai4ts")
    ai4ts_client_stub = types.ModuleType("ai4ts.client")
    ai4ts_client_stub.TimeSeriesAI = object
    sys.modules.update({
        "ai4ts": ai4ts_stub,
        "ai4ts.client": ai4ts_client_stub,
    })

_inject_ai4ts_stub()

# ── 2. global env & device ─────────────────────────────────────────────────
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["AI4TS_DISABLE_BANNER"]   = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"

# ── 3. tiny model builders (unchanged decision logic, leaner nets) ─────────
def _tiny_saits(steps: int, feats: int):
    with contextlib.redirect_stdout(io.StringIO()):
        from pypots.imputation import SAITS
    return SAITS(
        steps, feats,
        2, 24, 1, 24, 24, 48,
        dropout=0.05,
        batch_size=32,
        epochs=15,
        patience=2,
        device="cpu",
        verbose=False,
    )

def _tiny_csdi(steps: int, feats: int):
    with contextlib.redirect_stdout(io.StringIO()):
        from pypots.imputation import CSDI
    return CSDI(
        steps, feats,
        2, 1, 24, 24, 24, 24,
        n_diffusion_steps=20,
        batch_size=32,
        epochs=6,
        patience=2,
        device="cpu",
        verbose=False,
    )

# ── 4. helpers ─────────────────────────────────────
def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return (edges[1::2] - edges[::2]).max(initial=0)

def _align_hat(hat: np.ndarray, target: tuple) -> np.ndarray:
    hat = np.squeeze(hat)
    if hat.shape == target:
        return hat
    if hat.T.shape == target:
        return hat.T
    raise RuntimeError(f"Cannot align imputed shape {hat.shape} to {target}")

# ── 5. model cache (build once per (builder, steps, feats)) ────────────────
@lru_cache(maxsize=64)
def _get_model(key: tuple):
    builder_name, steps, feats = key
    builder = _tiny_saits if builder_name == "saits" else _tiny_csdi
    return builder(steps, feats)

# ── 6. choose MP context per device ────────────────────────────────────────
def _get_ctx():
    # CUDA needs 'spawn', pure-CPU runs on Linux are faster with 'fork'
    return mp.get_context( "fork")

# ── 7. worker initialiser  (only for 'spawn') ──────────────────────────────
def _worker_init():
    from loguru import logger
    logger.disable("pypots")
    if mp.get_start_method() != "spawn":
        return              # 'fork' workers inherit everything, skip heavy init

    # a) stdout always has .encoding
    if sys.stdout is None:
        sys.stdout = io.TextIOWrapper(open(os.devnull, "wb"))
    if sys.stdout.encoding is None:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                                      encoding="utf-8",
                                      write_through=True)

    # b) same env vars
    os.environ.update({
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "AI4TS_DISABLE_BANNER": "1",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })

    # c) silence spam
    import absl.logging as absl_logger
    absl_logger.set_verbosity(absl_logger.ERROR)
    absl_logger.set_stderrthreshold("fatal")
    from loguru import logger
    logger.disable("pypots")
    warnings.filterwarnings("ignore")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    # d) ensure stub exists in the child
    _inject_ai4ts_stub()

# ── 8. core per-window function (logic untouched) ──────────────────────────
def _impute_one(block: pd.DataFrame,
                big_gap_ratio: float,
                gap_rows_thr: int) -> np.ndarray:
    x    = block.to_numpy(np.float32, copy=True)
    miss = np.isnan(x)
    if not miss.any():
        return x

    ratio   = miss.mean()
    longest = _longest_run(miss.all(axis=1))
    builder_name = "csdi" if (ratio > big_gap_ratio or longest >= gap_rows_thr) else "saits"

    model = _get_model((builder_name, len(x), x.shape[1]))

    x_z = x.copy()
    model.fit({"X": x_z[None], "missing_mask": miss[None]})
    hat = model.impute({"X": x_z[None]})
    hat = hat["imputation"][0] if isinstance(hat, dict) else hat[0]
    hat = _align_hat(hat, x.shape)

    x[miss] = hat[miss]
    return x

# ── 9. choose pool size by *free* RAM ──────────────────────────────────────
def _auto_n_jobs(gb_per_worker: float = 1.5):
    proc_gb   = psutil.Process().memory_info().rss / (1024 ** 3)          # this script’s resident set
    avail_gb  = psutil.virtual_memory().available / (1024 ** 3) - proc_gb # truly free
    max_by_mem = max(1, int(avail_gb // gb_per_worker))
    return min(max_by_mem, os.cpu_count() or 1)

# ── 10. public API ─────────────────────────────────────────────────────────
def data_imputation(df: pd.DataFrame,
                    *,
                    big_gap_days: int = 2,
                    big_gap_ratio: float = 0.20,
                    n_jobs: int | None = None,
                    subhour_threshold_sec: int = 3600,
                    weekly_freq: str = "W-SUN",      # anchored for determinism
                    monthly_freq: str = "MS",
                    gb_per_worker: float = 1.5):
    """
    Impute df in parallel.

    n_jobs=None  →  min( free_RAM/gb_per_worker , CPU count )
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="raise")

    # window granularity
    dt_sec = df.index.to_series().diff().dropna().dt.total_seconds()
    sample_sec = dt_sec.median() if len(dt_sec) else 60.0          # pandas median → no NumPy cast
    window_freq = weekly_freq if sample_sec < subhour_threshold_sec else monthly_freq

    rows_day = int(round(86_400 / sample_sec))
    gap_rows = big_gap_days * rows_day

    if n_jobs is None:
        n_jobs = _auto_n_jobs(gb_per_worker)

    pieces = [m for _, m in df.groupby(pd.Grouper(freq=window_freq))]

    ctx  = _get_ctx()
    func = partial(_impute_one,
                   big_gap_ratio=big_gap_ratio,
                   gap_rows_thr=gap_rows)

    with ctx.Pool(processes=n_jobs, initializer=_worker_init) as pool:
        filled_list = [r for r in tqdm(pool.imap(func, pieces),
                                       total=len(pieces),
                                       desc="imputing windows",
                                       unit="window")]

    filled = np.vstack(filled_list)
    return pd.DataFrame(filled, index=df.index, columns=df.columns)