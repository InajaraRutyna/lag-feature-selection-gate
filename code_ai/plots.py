import os
os.environ["MPLBACKEND"] = "Agg"  # must be set before importing pyplot
import matplotlib
matplotlib.use("Agg")             # belt-and-suspenders
import matplotlib.pyplot as plt
import code_ai.shared_variables as sv
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
# Turn interactive mode on

# plt.ion()

def plot_dataframes_comparison(data_after, data_before, after, before):
    """
    Plots each common column of two DataFrames on three subplots:
    - Before data on the top subplot.
    - After data on the middle subplot.
    - Both data series on the bottom subplot for comparison.

    Parameters:
    data_after (pd.DataFrame): DataFrame with after data.
    data_before (pd.DataFrame): DataFrame with before data.

    The function aligns both DataFrames on their indices, and for each column that exists in both DataFrames,
    it creates a figure with three subplots as described above.
    """

    # Ensure both inputs are pandas DataFrames
    if not isinstance(data_after, pd.DataFrame):
        raise TypeError("data_after must be a pandas DataFrame.")
    if not isinstance(data_before, pd.DataFrame):
        raise TypeError("data_before must be a pandas DataFrame.")

    # Ensure the index is a DatetimeIndex
    if not isinstance(data_after.index, pd.DatetimeIndex):
        raise TypeError("The index of data_after must be a pandas DatetimeIndex.")
    if not isinstance(data_before.index, pd.DatetimeIndex):
        raise TypeError("The index of data_before must be a pandas DatetimeIndex.")

    # Get the list of common columns
    common_columns = data_after.columns.intersection(data_before.columns)
    if common_columns.empty:
        raise ValueError("No common columns between data_after and data_before.")

    # Align the DataFrames on their indices
    data_after_aligned, data_before_aligned = data_after.align(data_before, join='inner', axis=0)

    # Plot each common column
    for col in common_columns:
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 12))

        # Plot before data on the top subplot
        axes[0].plot(data_before_aligned.index, data_before_aligned[col], color='red')
        axes[0].set_title(f"{before} Data for '{col}'")
        axes[0].set_ylabel(col)
        axes[0].grid(True)

        # Plot after data on the middle subplot
        axes[1].plot(data_after_aligned.index, data_after_aligned[col], color='blue')
        axes[1].set_title(f"{after} Data for '{col}'")
        axes[1].set_ylabel(col)
        axes[1].grid(True)

        # Plot both data series on the bottom subplot
        axes[2].plot(data_before_aligned.index, data_before_aligned[col], label=f'{before}', color='red', alpha=0.7)
        axes[2].plot(data_after_aligned.index, data_after_aligned[col], label=f'{after}', color='blue')
        axes[2].set_title(f"Comparison of '{col}' Over Time")
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel(col)
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

def plot_capacity(output_series, capacity_series):
    # Ensure that both series have the same index
    assert output_series.index.equals(capacity_series.index), "Indices of output_series and capacity_series do not match."

    # ---------------------------
    # 1. Increase default font sizes
    # ---------------------------
    plt.rcParams['font.size'] = 16  # base font size
    plt.rcParams['axes.labelsize'] = 16  # x/y labels
    plt.rcParams['axes.titlesize'] = 16  # figure title
    plt.rcParams['xtick.labelsize'] = 16  # x tick labels
    plt.rcParams['ytick.labelsize'] = 16  # y tick labels
    plt.rcParams['legend.fontsize'] = 16  # legend text
    # Create the plot
    plt.figure(figsize=(15, 7))

    plt.plot(output_series.index, output_series.values, label='Smoothed Output Variable', alpha=0.7)
    plt.plot(capacity_series.index, capacity_series.values, label='Estimated Capacity', color='red', linewidth=2)

    # Format the x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()  # Rotate date labels

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Energy Output')
    # plt.title('Comparison of Smoothed Output Variable and Estimated Capacity')
    plt.legend()

    # Add grid and layout adjustments
    plt.grid(True)
    plt.tight_layout()
    out_dir = getattr(sv, "OUTPUT_DIR", ".")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "capacity.png"), dpi=150, bbox_inches='tight')
    plt.close("all")  # instead of plt.show()

def plot_predictions(y_true, y_pred, model_type):
    import numpy as np
    import matplotlib.pyplot as plt
    import random

    # Convert y_true and y_pred to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure y_true and y_pred are 2D arrays
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Compute errors for each sample (row)
    errors = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))

    # Identify samples where y_true and y_pred are not all zeros
    non_zero_indices = [i for i in range(len(y_true))
                        if not np.all(y_true[i] == 0) and not np.all(y_pred[i] == 0)]

    # If no non-zero samples, print a message and exit
    if len(non_zero_indices) == 0:
        print("All samples have zero values. No plots to display.")
        return

    # Extract errors for non-zero samples
    errors_non_zero = errors[non_zero_indices]

    # Find indices of samples with largest and smallest errors among non-zero samples
    idx_max_error = non_zero_indices[np.argmax(errors_non_zero)]
    idx_min_error = non_zero_indices[np.argmin(errors_non_zero)]

    # Get indices of up to 5 random samples, excluding the ones already selected
    available_indices = list(set(non_zero_indices) - {idx_max_error, idx_min_error})
    num_random_samples = min(5, len(available_indices))
    idx_random = random.sample(available_indices, num_random_samples)

    # Combine indices for plotting
    idx_list = [idx_min_error] + idx_random + [idx_max_error]

    n_plots = len(idx_list)

    # Create subplots
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(10, n_plots * 3))

    # Handle the case when there's only one subplot
    if n_plots == 1:
        axes = [axes]

    for i, idx in enumerate(idx_list):
        axes[i].plot(y_true[idx], label='True')
        axes[i].plot(y_pred[idx], label='Predicted')
        axes[i].set_title(f'Sample {idx}: Error={errors[idx]:.4f}, Model: {model_type}')
        axes[i].set_xlabel('Step Ahead')
        axes[i].set_ylabel('Value')
        axes[i].legend()

    plt.tight_layout()
    out_dir = getattr(sv, "OUTPUT_DIR", ".")
    fname = f"{model_type}_predictions.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close('all')

def history_train(history, method):
    df = pd.DataFrame(history.history)

    # ── 1. plot direct MSE only ────────────────────────────────────
    direct_cols = [c for c in ('direct_loss', 'val_direct') if c in df]
    if direct_cols:
        (df[direct_cols]
         .rename(columns={'direct_loss':'train', 'val_direct':'validation'})
         .plot(title=f'{method}: direct MSE', figsize=(9,4))
         .set(xlabel='epoch', ylabel='mse'))

    # ── 2. plot each distillation term on its own axis (log-scale) ─
    distill_groups = (
        ('pred_loss','val_pred','crd MSE'),
        ('hint_loss','val_hint','adv MSE'),
    )
    for train_c,val_c,label in distill_groups:
        if train_c in df:
            (df[[train_c,val_c]]
             .rename(columns={train_c:'train', val_c:'validation'})
             .plot(title=f'{method}: {label}', figsize=(9,3), logy=True)
             .set(xlabel='epoch', ylabel='loss'))

    plt.tight_layout()
    out_dir = getattr(sv, "OUTPUT_DIR", ".")
    fname = f"{method}.png"  # was sv.method
    plt.savefig(os.path.join(out_dir, fname))
    plt.close('all')


def plot_correlation_map_from_sv(P=None, vmin=0.0, vmax=1.0, feat_names=None, title=None):
    # pick the matrix
    if P is not None:
        M = P
    else:
        score, decay_lag, _ = sv.score_data
        if sv.DECAY:
            score = score * decay_lag
        M = score

    L, F = M.shape

    # wide landscape figure + auto layout
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    im = ax.imshow(M, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_xlabel('feature')
    ax.set_ylabel('lag')
    ax.set_title(title or 'Correlation map')

    # small, tight colorbar
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('normalized correlation')

    # optional ticks
    if feat_names is not None and len(feat_names) == F and F <= 100:
        ax.set_xticks(np.arange(F))
        ax.set_xticklabels(feat_names, rotation=90, fontsize=8)
    if L > 50:
        ax.set_yticks(np.linspace(0, L-1, 11, dtype=int))
    if F > 50:
        ax.set_xticks(np.linspace(0, F-1, 11, dtype=int))

    out_dir = getattr(sv, "OUTPUT_DIR", ".")
    uniq = np.unique(M[~np.isnan(M)]) if np.isnan(M).any() else np.unique(M)
    is_binary = set(uniq.tolist()) <= {0, 1}
    save_name = "logits.png" if is_binary else ("logits_1.png" if np.issubdtype(M.dtype, np.floating) else "correlation_map.png")
    fig.savefig(os.path.join(out_dir, save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)
