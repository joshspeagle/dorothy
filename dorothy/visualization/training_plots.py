"""
Training visualization and report generation for DOROTHY.

Generates comprehensive plots after training including:
- Loss curves (train/val)
- Grokking metrics (gradient norms, weight updates)
- Per-parameter performance metrics
- Z-score calibration histograms
- Prediction vs truth scatter plots
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from dorothy.training.trainer import TrainingHistory


# Standard parameter names for labeling
PARAM_DISPLAY_NAMES = [
    r"$T_{\rm eff}$",
    r"$\log g$",
    "[Fe/H]",
    "[Mg/Fe]",
    "[C/Fe]",
    "[Si/Fe]",
    "[Ni/Fe]",
    "[Al/Fe]",
    "[Ca/Fe]",
    "[N/Fe]",
    "[Mn/Fe]",
]


def _simplify_layer_name(name: str) -> str:
    """
    Simplify layer names for plot legends.

    Examples:
        "layers.1.weight" -> "L1"
        "layers.4.bias" -> "L4.b"
        "encoder.fc.weight" -> "enc.fc"
        "trunk.layers.2.weight" -> "trunk.L2"
    """
    import re

    # Handle common patterns
    # Pattern: "layers.N.weight/bias" -> "LN" or "LN.b"
    match = re.match(r"(?:.*\.)?layers\.(\d+)\.(weight|bias)", name)
    if match:
        layer_num = match.group(1)
        param_type = match.group(2)
        suffix = "" if param_type == "weight" else ".b"
        return f"L{layer_num}{suffix}"

    # Pattern: "encoders.survey.layers.N.weight" -> "survey.LN"
    match = re.match(r"encoders\.(\w+)\.layers\.(\d+)\.(weight|bias)", name)
    if match:
        survey = match.group(1)[:4]  # First 4 chars
        layer_num = match.group(2)
        return f"{survey}.L{layer_num}"

    # Pattern: "trunk.layers.N.weight" -> "trunk.LN"
    match = re.match(r"(\w+)\.layers\.(\d+)\.(weight|bias)", name)
    if match:
        prefix = match.group(1)[:5]  # First 5 chars
        layer_num = match.group(2)
        return f"{prefix}.L{layer_num}"

    # Fallback: take last 2 parts
    parts = name.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return name


def _safe_legend(ax) -> None:
    """Add legend to axis only if there are labeled artists.

    This avoids the matplotlib warning:
    "No artists with labels found to put in legend."
    """
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def _try_import_matplotlib():
    """Try to import matplotlib, return None if not available."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for saving
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def plot_loss_curves(
    history: "TrainingHistory",
    output_path: Path,
    title: str = "Training Progress",
) -> Path | None:
    """
    Plot training and validation loss curves.

    Args:
        history: TrainingHistory object with loss data.
        output_path: Directory to save the plot.
        title: Plot title.

    Returns:
        Path to saved plot, or None if matplotlib unavailable.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    if not history.train_losses:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    n_epochs = len(history.train_losses)
    epochs = np.arange(1, n_epochs + 1)

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, history.train_losses, label="Train", alpha=0.8)
    if len(history.val_losses) == n_epochs:
        ax.plot(epochs, history.val_losses, label="Validation", alpha=0.8)

    # Mark best epoch (best_epoch is 0-indexed, display as 1-indexed)
    if history.best_epoch is not None and history.best_epoch < n_epochs:
        best_epoch_display = history.best_epoch + 1
        ax.axvline(
            best_epoch_display,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Best (epoch {best_epoch_display})",
        )
        ax.scatter(
            [best_epoch_display],
            [history.best_val_loss],
            color="green",
            s=100,
            zorder=5,
            marker="*",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3)

    # Learning rate - may be per-step or per-epoch
    ax = axes[1]
    if history.learning_rates:
        n_lr = len(history.learning_rates)
        if n_lr == n_epochs:
            # Per-epoch learning rates
            ax.plot(epochs, history.learning_rates, color="orange")
            ax.set_xlabel("Epoch")
        else:
            # Per-step learning rates - plot against step number
            steps = np.arange(1, n_lr + 1)
            ax.plot(steps, history.learning_rates, color="orange", alpha=0.8)
            ax.set_xlabel("Training Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    save_path = output_path / "loss_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_loss_components(
    history: "TrainingHistory",
    output_path: Path,
) -> Path | None:
    """
    Plot per-parameter loss breakdown (mean vs scatter components).

    Args:
        history: TrainingHistory object with loss breakdown.
        output_path: Directory to save the plot.

    Returns:
        Path to saved plot, or None if data unavailable.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    # Check if we have breakdown data
    if not history.val_loss_breakdown.get("mean_component"):
        return None

    mean_comp = np.array(history.val_loss_breakdown["mean_component"])
    scatter_comp = np.array(history.val_loss_breakdown["scatter_component"])

    if mean_comp.ndim != 2:
        return None

    n_epochs, n_params = mean_comp.shape
    epochs = np.arange(1, n_epochs + 1)

    # Use subset of params for readability
    n_show = min(n_params, 6)
    param_names = PARAM_DISPLAY_NAMES[:n_show]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Mean component per parameter
    ax = axes[0]
    for i in range(n_show):
        ax.plot(epochs, mean_comp[:, i], label=param_names[i], alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Component")
    ax.set_title("Mean Loss Component (per parameter)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Scatter component per parameter
    ax = axes[1]
    for i in range(n_show):
        ax.plot(epochs, scatter_comp[:, i], label=param_names[i], alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Scatter Component")
    ax.set_title("Scatter Loss Component (per parameter)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = output_path / "loss_components.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def _categorize_layer_names(layer_names: list[str]) -> dict[str, list[str]]:
    """
    Categorize layer names by component for grouped plotting.

    Returns dict mapping component name to list of layer names.
    Components: encoders (by survey), trunk, output_head(s), other.
    """
    import re

    categories: dict[str, list[str]] = {}

    for name in layer_names:
        # Check for encoder pattern: encoders.survey_name.*
        match = re.match(r"encoders\.(\w+)\.", name)
        if match:
            survey = match.group(1)
            cat = f"encoder_{survey}"
            categories.setdefault(cat, []).append(name)
            continue

        # Check for trunk pattern
        if name.startswith("trunk."):
            categories.setdefault("trunk", []).append(name)
            continue

        # Check for output head pattern (single or multi)
        if name.startswith("output_head.") or name.startswith("output_heads."):
            match = re.match(r"output_heads\.(\w+)\.", name)
            if match:
                source = match.group(1)
                cat = f"output_{source}"
            else:
                cat = "output_head"
            categories.setdefault(cat, []).append(name)
            continue

        # Standard MLP layers pattern
        if name.startswith("layers."):
            categories.setdefault("layers", []).append(name)
            continue

        # Fallback
        categories.setdefault("other", []).append(name)

    return categories


def plot_grokking_metrics(
    history: "TrainingHistory",
    output_path: Path,
) -> Path | None:
    """
    Plot grokking-related metrics: gradient norms, weight norms, weight updates.

    For multi-head models, weight norms and updates are grouped by component
    (encoders, trunk, output heads) for better interpretability.

    Args:
        history: TrainingHistory object.
        output_path: Directory to save the plot.

    Returns:
        Path to saved plot, or None if data unavailable.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    if not history.grad_norms:
        return None

    epochs = np.arange(1, len(history.grad_norms) + 1)

    # Determine number of subplots based on available data
    has_weight_norms = bool(history.weight_norms)
    has_weight_updates = bool(history.weight_updates)

    n_plots = 1 + has_weight_norms + has_weight_updates
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Gradient norms
    ax = axes[plot_idx]
    ax.plot(epochs, history.grad_norms, color="blue", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm Over Training")
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # Categorize layers for grouped plotting
    if has_weight_norms:
        categories = _categorize_layer_names(list(history.weight_norms.keys()))
    else:
        categories = {}

    # Color palette for different components
    component_colors = {
        "layers": "blue",
        "trunk": "green",
        "output_head": "red",
    }
    encoder_colors = ["steelblue", "coral", "purple", "olive"]
    output_colors = ["crimson", "darkred", "indianred", "salmon"]

    # Weight norms per layer (grouped by component)
    if has_weight_norms:
        ax = axes[plot_idx]

        # Assign colors to categories
        encoder_idx = 0
        output_idx = 0
        for cat_name, layer_names in sorted(categories.items()):
            if cat_name.startswith("encoder_"):
                color = encoder_colors[encoder_idx % len(encoder_colors)]
                encoder_idx += 1
            elif cat_name.startswith("output_"):
                color = output_colors[output_idx % len(output_colors)]
                output_idx += 1
            else:
                color = component_colors.get(cat_name, "gray")

            # Plot aggregate (mean) norm for this category
            cat_norms = []
            for layer_name in layer_names:
                norms = history.weight_norms.get(layer_name, [])
                if norms:
                    cat_norms.append(np.array(norms))

            if cat_norms:
                # Stack and compute mean across layers in category
                stacked = np.stack(cat_norms, axis=0)
                mean_norm = np.mean(stacked, axis=0)
                # Prettify category name for legend
                display_name = cat_name.replace("_", " ").title()
                ax.plot(
                    epochs[: len(mean_norm)],
                    mean_norm,
                    label=display_name,
                    color=color,
                    alpha=0.8,
                    linewidth=2,
                )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Weight Norm")
        ax.set_title("Weight Norms by Component")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Weight updates per layer (grouped by component)
    if has_weight_updates:
        ax = axes[plot_idx]
        categories = _categorize_layer_names(list(history.weight_updates.keys()))

        encoder_idx = 0
        output_idx = 0
        for cat_name, layer_names in sorted(categories.items()):
            if cat_name.startswith("encoder_"):
                color = encoder_colors[encoder_idx % len(encoder_colors)]
                encoder_idx += 1
            elif cat_name.startswith("output_"):
                color = output_colors[output_idx % len(output_colors)]
                output_idx += 1
            else:
                color = component_colors.get(cat_name, "gray")

            # Plot aggregate (mean) update for this category
            cat_updates = []
            for layer_name in layer_names:
                updates = history.weight_updates.get(layer_name, [])
                if updates:
                    cat_updates.append(np.array(updates))

            if cat_updates:
                stacked = np.stack(cat_updates, axis=0)
                mean_update = np.mean(stacked, axis=0)
                display_name = cat_name.replace("_", " ").title()
                ax.plot(
                    epochs[: len(mean_update)],
                    mean_update,
                    label=display_name,
                    color=color,
                    alpha=0.8,
                    linewidth=2,
                )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Weight Update")
        ax.set_title("Weight Updates by Component")
        ax.legend(fontsize=8, loc="best")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = output_path / "grokking_metrics.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_val_metrics(
    history: "TrainingHistory",
    output_path: Path,
) -> Path | None:
    """
    Plot validation metrics per parameter (RMSE, Robust SD, Bias, MAE).

    All metrics are in normalized space:
    - Labels are standardized: (value - median) / IQR
    - Teff is log10-transformed before standardization

    Args:
        history: TrainingHistory object with val_metrics.
        output_path: Directory to save the plot.

    Returns:
        Path to saved plot, or None if data unavailable.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    if not history.val_metrics:
        return None

    # val_metrics is dict[str, list[np.ndarray]] - get final epoch values
    rmse_list = history.val_metrics.get("rmse", [])
    mae_list = history.val_metrics.get("mae", [])
    bias_list = history.val_metrics.get("bias", [])
    sd_list = history.val_metrics.get("sd", [])  # Standard deviation
    robust_scatter_list = history.val_metrics.get("robust_scatter", [])

    if not rmse_list:
        return None

    # Get final epoch metrics
    rmse = rmse_list[-1]
    mae = mae_list[-1] if mae_list else np.zeros_like(rmse)
    bias = bias_list[-1] if bias_list else np.zeros_like(rmse)
    # Use robust_scatter if available, fall back to sd
    scatter = (
        robust_scatter_list[-1]
        if robust_scatter_list
        else (sd_list[-1] if sd_list else np.zeros_like(rmse))
    )

    n_params = len(rmse)
    if n_params == 0:
        return None

    param_names = PARAM_DISPLAY_NAMES[:n_params]
    x = np.arange(n_params)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RMSE bar chart
    ax = axes[0, 0]
    bars = ax.bar(x, rmse, width=0.6, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RMSE (normalized)")
    ax.set_title("RMSE by Parameter")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, rmse, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # Robust Scatter (SD) bar chart - shows accuracy/precision
    ax = axes[0, 1]
    bars = ax.bar(x, scatter, width=0.6, color="purple", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Robust Scatter (normalized)")
    ax.set_title("Scatter/Precision by Parameter")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, scatter, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # Bias bar chart
    ax = axes[1, 0]
    colors = [
        "green" if abs(b) < 0.1 else "orange" if abs(b) < 0.3 else "red" for b in bias
    ]
    bars = ax.bar(x, bias, width=0.6, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.1, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(-0.1, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Bias (normalized)")
    ax.set_title("Bias by Parameter (pred - true)")
    ax.grid(True, alpha=0.3, axis="y")

    # MAE bar chart
    ax = axes[1, 1]
    bars = ax.bar(x, mae, width=0.6, color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("MAE (normalized)")
    ax.set_title("MAE by Parameter")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, mae, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.suptitle(
        "Validation Metrics (normalized space: median/IQR standardized, Teff in log₁₀)",
        fontsize=12,
    )
    plt.tight_layout()

    save_path = output_path / "val_metrics.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_zscore_calibration(
    history: "TrainingHistory",
    output_path: Path,
) -> Path | None:
    """
    Plot z-score distribution to check uncertainty calibration.

    Well-calibrated uncertainties should give z-scores ~ N(0, 1).

    Args:
        history: TrainingHistory object with val_metrics containing z-scores.
        output_path: Directory to save the plot.

    Returns:
        Path to saved plot, or None if data unavailable.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    if not history.val_metrics:
        return None

    # val_metrics is dict[str, list[np.ndarray]] - get final epoch values
    # z_median approximates z-score mean, z_robust_scatter approximates z-score std
    z_median_list = history.val_metrics.get("z_median", [])
    z_scatter_list = history.val_metrics.get("z_robust_scatter", [])

    if not z_median_list or not z_scatter_list:
        return None

    zscore_mean = z_median_list[-1]
    zscore_std = z_scatter_list[-1]

    n_params = len(zscore_mean)
    param_names = PARAM_DISPLAY_NAMES[:n_params]
    x = np.arange(n_params)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Z-score median (should be ~0)
    ax = axes[0]
    colors = [
        "green" if abs(m) < 0.5 else "orange" if abs(m) < 1 else "red"
        for m in zscore_mean
    ]
    ax.bar(x, zscore_mean, width=0.6, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(-0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Z-score Median")
    ax.set_title("Z-score Median (should be ~0)")
    ax.set_ylim(-2, 2)
    ax.grid(True, alpha=0.3, axis="y")

    # Z-score robust scatter (should be ~1)
    ax = axes[1]
    colors = [
        "green" if abs(s - 1) < 0.2 else "orange" if abs(s - 1) < 0.5 else "red"
        for s in zscore_std
    ]
    ax.bar(x, zscore_std, width=0.6, color=colors, alpha=0.8)
    ax.axhline(1, color="black", linewidth=1)
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(1.2, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Z-score Robust Scatter")
    ax.set_title("Z-score Scatter (should be ~1)")
    ax.set_ylim(0, 2.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Uncertainty Calibration (z-score diagnostics)", fontsize=12)
    plt.tight_layout()

    save_path = output_path / "zscore_calibration.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_metrics_evolution(
    history: "TrainingHistory",
    output_path: Path,
) -> Path | None:
    """
    Plot evolution of key metrics over training epochs.

    Args:
        history: TrainingHistory object.
        output_path: Directory to save the plot.

    Returns:
        Path to saved plot, or None if data unavailable.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    if not history.val_metrics:
        return None

    # val_metrics is dict[str, list[np.ndarray]]
    rmse_list = history.val_metrics.get("rmse", [])
    z_scatter_list = history.val_metrics.get("z_robust_scatter", [])

    if len(rmse_list) < 2:
        return None

    n_epochs = len(rmse_list)
    epochs = np.arange(1, n_epochs + 1)

    # Stack into arrays: (n_epochs, n_params)
    rmse_over_time = np.array(rmse_list)
    n_params = rmse_over_time.shape[1]
    if n_params == 0:
        return None

    n_show = min(3, n_params)
    param_names = PARAM_DISPLAY_NAMES[:n_show]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # RMSE evolution
    ax = axes[0]
    for i in range(n_show):
        ax.plot(epochs, rmse_over_time[:, i], label=param_names[i], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE Evolution During Training")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3)

    # Z-score scatter evolution (if available)
    ax = axes[1]
    if z_scatter_list and len(z_scatter_list) == n_epochs:
        zscore_std_over_time = np.array(z_scatter_list)
        for i in range(n_show):
            ax.plot(epochs, zscore_std_over_time[:, i], label=param_names[i], alpha=0.8)
        ax.axhline(1, color="black", linestyle="--", alpha=0.5, label="Ideal")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Z-score Robust Scatter")
        ax.set_title("Uncertainty Calibration Evolution")
        _safe_legend(ax)
        ax.grid(True, alpha=0.3)
    else:
        # Fall back to validation loss if z-scores unavailable
        if len(history.val_losses) == n_epochs:
            ax.plot(epochs, history.val_losses, color="blue", alpha=0.8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation Loss")
            ax.set_title("Validation Loss Evolution")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = output_path / "metrics_evolution.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_per_survey_loss_curves(
    history: "TrainingHistory",
    output_path: Path,
) -> Path | None:
    """
    Plot per-survey validation loss curves and relative performance.

    For multi-survey training, shows:
    - Left: Per-survey loss curves with overall loss for reference
    - Right: Relative loss (survey loss / overall loss) to show survey difficulty

    Args:
        history: TrainingHistory object with per_survey_val_losses.
        output_path: Directory to save the plot.

    Returns:
        Path to saved plot, or None if data unavailable.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    # Check for per-survey data
    if not history.survey_names or not history.per_survey_val_losses:
        return None

    # Check we have data for at least one survey
    has_data = any(len(losses) > 0 for losses in history.per_survey_val_losses.values())
    if not has_data:
        return None

    n_epochs = len(history.val_losses)
    epochs = np.arange(1, n_epochs + 1)

    # Survey colors
    colors = ["steelblue", "coral", "green", "purple"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Per-survey validation loss with overall
    ax = axes[0]
    ax.plot(
        epochs,
        history.val_losses,
        label="Overall",
        color="black",
        linewidth=2,
        alpha=0.9,
    )
    for i, survey_name in enumerate(history.survey_names):
        losses = history.per_survey_val_losses.get(survey_name, [])
        if losses and len(losses) == n_epochs:
            ax.plot(
                epochs,
                losses,
                label=survey_name.upper(),
                color=colors[i % len(colors)],
                alpha=0.8,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Per-Survey Validation Loss")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3)

    # Right: Relative loss (survey / overall) - shows which surveys are harder
    ax = axes[1]
    overall_loss = np.array(history.val_losses)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    for i, survey_name in enumerate(history.survey_names):
        losses = history.per_survey_val_losses.get(survey_name, [])
        if losses and len(losses) == n_epochs:
            relative_loss = np.array(losses) / np.maximum(overall_loss, 1e-8)
            ax.plot(
                epochs,
                relative_loss,
                label=survey_name.upper(),
                color=colors[i % len(colors)],
                alpha=0.8,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative Loss (Survey / Overall)")
    ax.set_title("Survey Difficulty (>1 = harder)")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Survey Training Progress", fontsize=12)
    plt.tight_layout()

    save_path = output_path / "per_survey_loss_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_per_survey_metrics(
    history: "TrainingHistory",
    output_path: Path,
) -> Path | None:
    """
    Plot final validation metrics grouped by survey.

    Creates a 2x2 grid of grouped bar charts comparing RMSE, Scatter, Bias, and MAE
    across surveys for each parameter (matching the layout of val_metrics).

    Args:
        history: TrainingHistory object with per_survey_val_metrics.
        output_path: Directory to save the plot.

    Returns:
        Path to saved plot, or None if data unavailable.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    # Check for per-survey data
    if not history.survey_names or not history.per_survey_val_metrics:
        return None

    # Verify we have metrics data
    first_survey = history.survey_names[0]
    if first_survey not in history.per_survey_val_metrics:
        return None
    survey_metrics = history.per_survey_val_metrics[first_survey]
    if not survey_metrics.get("rmse"):
        return None

    n_surveys = len(history.survey_names)
    n_params = len(survey_metrics["rmse"][-1])

    if n_params == 0:
        return None

    # Show up to 6 parameters for readability
    n_show = min(n_params, 6)
    param_names = PARAM_DISPLAY_NAMES[:n_show]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(n_show)
    width = 0.35 if n_surveys == 2 else 0.25

    # Color palette for surveys
    colors = ["steelblue", "coral", "green", "purple"][:n_surveys]

    # RMSE by survey (top-left)
    ax = axes[0, 0]
    for i, survey_name in enumerate(history.survey_names):
        rmse_list = history.per_survey_val_metrics[survey_name].get("rmse", [])
        if rmse_list:
            rmse = rmse_list[-1][:n_show]
            offset = (i - n_surveys / 2 + 0.5) * width
            ax.bar(
                x + offset,
                rmse,
                width,
                label=survey_name.upper(),
                color=colors[i],
                alpha=0.8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RMSE (normalized)")
    ax.set_title("RMSE by Survey")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3, axis="y")

    # Robust scatter by survey (top-right)
    ax = axes[0, 1]
    for i, survey_name in enumerate(history.survey_names):
        scatter_list = history.per_survey_val_metrics[survey_name].get(
            "robust_scatter", []
        )
        if scatter_list:
            scatter = scatter_list[-1][:n_show]
            offset = (i - n_surveys / 2 + 0.5) * width
            ax.bar(
                x + offset,
                scatter,
                width,
                label=survey_name.upper(),
                color=colors[i],
                alpha=0.8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Robust Scatter (normalized)")
    ax.set_title("Scatter/Precision by Survey")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3, axis="y")

    # Bias by survey (bottom-left)
    ax = axes[1, 0]
    for i, survey_name in enumerate(history.survey_names):
        bias_list = history.per_survey_val_metrics[survey_name].get("bias", [])
        if bias_list:
            bias = bias_list[-1][:n_show]
            offset = (i - n_surveys / 2 + 0.5) * width
            ax.bar(
                x + offset,
                bias,
                width,
                label=survey_name.upper(),
                color=colors[i],
                alpha=0.8,
            )
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.1, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(-0.1, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Bias (normalized)")
    ax.set_title("Bias by Survey (pred - true)")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3, axis="y")

    # MAE by survey (bottom-right)
    ax = axes[1, 1]
    for i, survey_name in enumerate(history.survey_names):
        mae_list = history.per_survey_val_metrics[survey_name].get("mae", [])
        if mae_list:
            mae = mae_list[-1][:n_show]
            offset = (i - n_surveys / 2 + 0.5) * width
            ax.bar(
                x + offset,
                mae,
                width,
                label=survey_name.upper(),
                color=colors[i],
                alpha=0.8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("MAE (normalized)")
    ax.set_title("MAE by Survey")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Per-Survey Validation Metrics (normalized space, Final Epoch)", fontsize=12
    )
    plt.tight_layout()

    save_path = output_path / "per_survey_metrics.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_per_survey_metrics_evolution(
    history: "TrainingHistory",
    output_path: Path,
) -> Path | None:
    """
    Plot evolution of per-survey RMSE over training epochs for key parameters.

    Creates subplots for a few key parameters (Teff, logg, [Fe/H]) showing
    how each survey's RMSE evolves over training.

    Args:
        history: TrainingHistory object with per_survey_val_metrics.
        output_path: Directory to save the plot.

    Returns:
        Path to saved plot, or None if data unavailable.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    # Check for per-survey data
    if not history.survey_names or not history.per_survey_val_metrics:
        return None

    # Verify we have metrics data
    first_survey = history.survey_names[0]
    if first_survey not in history.per_survey_val_metrics:
        return None
    survey_metrics = history.per_survey_val_metrics[first_survey]
    rmse_list = survey_metrics.get("rmse", [])
    if not rmse_list or len(rmse_list) < 2:
        return None

    n_epochs = len(rmse_list)
    epochs = np.arange(1, n_epochs + 1)
    n_params = len(rmse_list[0])

    # Show up to 3 key parameters
    n_show = min(3, n_params)
    param_names = PARAM_DISPLAY_NAMES[:n_show]

    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 4))
    if n_show == 1:
        axes = [axes]

    # Line styles for surveys
    styles = ["-", "--", "-.", ":"]
    colors = ["steelblue", "coral", "green", "purple"]

    for param_idx in range(n_show):
        ax = axes[param_idx]

        for i, survey_name in enumerate(history.survey_names):
            rmse_list = history.per_survey_val_metrics[survey_name].get("rmse", [])
            if rmse_list and len(rmse_list) == n_epochs:
                # Extract this parameter's RMSE over epochs
                rmse_over_time = np.array([r[param_idx] for r in rmse_list])
                ax.plot(
                    epochs,
                    rmse_over_time,
                    label=survey_name.upper(),
                    color=colors[i % len(colors)],
                    linestyle=styles[i % len(styles)],
                    alpha=0.8,
                )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE (normalized)")
        ax.set_title(f"{param_names[param_idx]} RMSE Evolution")
        _safe_legend(ax)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Survey RMSE Evolution by Parameter", fontsize=12)
    plt.tight_layout()

    save_path = output_path / "per_survey_metrics_evolution.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_per_labelset_loss_curves(
    history: "TrainingHistory",
    output_path: Path,
) -> Path | None:
    """
    Plot per-labelset validation loss curves and relative performance.

    For multi-labelset training (multiple output heads), shows:
    - Left: Per-labelset loss curves with overall loss for reference
    - Right: Relative loss (labelset loss / overall loss) to show head difficulty

    Args:
        history: TrainingHistory object with per_labelset_val_losses.
        output_path: Directory to save the plot.

    Returns:
        Path to saved plot, or None if data unavailable.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    # Check for per-labelset data
    if not history.label_source_names or not history.per_labelset_val_losses:
        return None

    # Check we have data for at least one labelset
    has_data = any(
        len(losses) > 0 for losses in history.per_labelset_val_losses.values()
    )
    if not has_data:
        return None

    n_epochs = len(history.val_losses)
    epochs = np.arange(1, n_epochs + 1)

    # Labelset colors
    colors = ["steelblue", "coral", "green", "purple"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Per-labelset validation loss with overall
    ax = axes[0]
    ax.plot(
        epochs,
        history.val_losses,
        label="Overall",
        color="black",
        linewidth=2,
        alpha=0.9,
    )
    for i, source_name in enumerate(history.label_source_names):
        losses = history.per_labelset_val_losses.get(source_name, [])
        if losses and len(losses) == n_epochs:
            ax.plot(
                epochs,
                losses,
                label=source_name.upper(),
                color=colors[i % len(colors)],
                alpha=0.8,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Per-Labelset Validation Loss")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3)

    # Right: Relative loss (labelset / overall)
    ax = axes[1]
    overall_loss = np.array(history.val_losses)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    for i, source_name in enumerate(history.label_source_names):
        losses = history.per_labelset_val_losses.get(source_name, [])
        if losses and len(losses) == n_epochs:
            relative_loss = np.array(losses) / np.maximum(overall_loss, 1e-8)
            ax.plot(
                epochs,
                relative_loss,
                label=source_name.upper(),
                color=colors[i % len(colors)],
                alpha=0.8,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative Loss (Labelset / Overall)")
    ax.set_title("Labelset Difficulty (>1 = harder)")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Labelset (Output Head) Training Progress", fontsize=12)
    plt.tight_layout()

    save_path = output_path / "per_labelset_loss_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_per_labelset_metrics(
    history: "TrainingHistory",
    output_path: Path,
) -> Path | None:
    """
    Plot final validation metrics grouped by labelset (output head).

    Creates a 2x2 grid of grouped bar charts comparing RMSE, Scatter, Bias, and MAE
    across labelsets for each parameter (matching the layout of val_metrics).

    Args:
        history: TrainingHistory object with per_labelset_val_metrics.
        output_path: Directory to save the plot.

    Returns:
        Path to saved plot, or None if data unavailable.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    # Check for per-labelset data
    if not history.label_source_names or not history.per_labelset_val_metrics:
        return None

    # Verify we have metrics data
    first_source = history.label_source_names[0]
    if first_source not in history.per_labelset_val_metrics:
        return None
    source_metrics = history.per_labelset_val_metrics[first_source]
    if not source_metrics.get("rmse"):
        return None

    n_labelsets = len(history.label_source_names)
    n_params = len(source_metrics["rmse"][-1])

    if n_params == 0:
        return None

    # Show up to 6 parameters for readability
    n_show = min(n_params, 6)
    param_names = PARAM_DISPLAY_NAMES[:n_show]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(n_show)
    width = 0.35 if n_labelsets == 2 else 0.25

    # Color palette for labelsets
    colors = ["steelblue", "coral", "green", "purple"][:n_labelsets]

    # RMSE by labelset (top-left)
    ax = axes[0, 0]
    for i, source_name in enumerate(history.label_source_names):
        rmse_list = history.per_labelset_val_metrics[source_name].get("rmse", [])
        if rmse_list:
            rmse = rmse_list[-1][:n_show]
            offset = (i - n_labelsets / 2 + 0.5) * width
            ax.bar(
                x + offset,
                rmse,
                width,
                label=source_name.upper(),
                color=colors[i],
                alpha=0.8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RMSE (normalized)")
    ax.set_title("RMSE by Labelset")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3, axis="y")

    # Robust scatter by labelset (top-right)
    ax = axes[0, 1]
    for i, source_name in enumerate(history.label_source_names):
        scatter_list = history.per_labelset_val_metrics[source_name].get(
            "robust_scatter", []
        )
        if scatter_list:
            scatter = scatter_list[-1][:n_show]
            offset = (i - n_labelsets / 2 + 0.5) * width
            ax.bar(
                x + offset,
                scatter,
                width,
                label=source_name.upper(),
                color=colors[i],
                alpha=0.8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Robust Scatter (normalized)")
    ax.set_title("Scatter/Precision by Labelset")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3, axis="y")

    # Bias by labelset (bottom-left)
    ax = axes[1, 0]
    for i, source_name in enumerate(history.label_source_names):
        bias_list = history.per_labelset_val_metrics[source_name].get("bias", [])
        if bias_list:
            bias = bias_list[-1][:n_show]
            offset = (i - n_labelsets / 2 + 0.5) * width
            ax.bar(
                x + offset,
                bias,
                width,
                label=source_name.upper(),
                color=colors[i],
                alpha=0.8,
            )
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.1, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(-0.1, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Bias (normalized)")
    ax.set_title("Bias by Labelset (pred - true)")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3, axis="y")

    # MAE by labelset (bottom-right)
    ax = axes[1, 1]
    for i, source_name in enumerate(history.label_source_names):
        mae_list = history.per_labelset_val_metrics[source_name].get("mae", [])
        if mae_list:
            mae = mae_list[-1][:n_show]
            offset = (i - n_labelsets / 2 + 0.5) * width
            ax.bar(
                x + offset,
                mae,
                width,
                label=source_name.upper(),
                color=colors[i],
                alpha=0.8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("MAE (normalized)")
    ax.set_title("MAE by Labelset")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Per-Labelset Validation Metrics (normalized space, Final Epoch)", fontsize=12
    )
    plt.tight_layout()

    save_path = output_path / "per_labelset_metrics.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_per_labelset_metrics_evolution(
    history: "TrainingHistory",
    output_path: Path,
) -> Path | None:
    """
    Plot evolution of per-labelset RMSE over training epochs for key parameters.

    Creates subplots for a few key parameters (Teff, logg, [Fe/H]) showing
    how each labelset's RMSE evolves over training.

    Args:
        history: TrainingHistory object with per_labelset_val_metrics.
        output_path: Directory to save the plot.

    Returns:
        Path to saved plot, or None if data unavailable.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    # Check for per-labelset data
    if not history.label_source_names or not history.per_labelset_val_metrics:
        return None

    # Verify we have metrics data
    first_source = history.label_source_names[0]
    if first_source not in history.per_labelset_val_metrics:
        return None
    source_metrics = history.per_labelset_val_metrics[first_source]
    rmse_list = source_metrics.get("rmse", [])
    if not rmse_list or len(rmse_list) < 2:
        return None

    n_epochs = len(rmse_list)
    epochs = np.arange(1, n_epochs + 1)
    n_params = len(rmse_list[0])

    # Show up to 3 key parameters
    n_show = min(3, n_params)
    param_names = PARAM_DISPLAY_NAMES[:n_show]

    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 4))
    if n_show == 1:
        axes = [axes]

    # Line styles for labelsets
    styles = ["-", "--", "-.", ":"]
    colors = ["steelblue", "coral", "green", "purple"]

    for param_idx in range(n_show):
        ax = axes[param_idx]

        for i, source_name in enumerate(history.label_source_names):
            rmse_list = history.per_labelset_val_metrics[source_name].get("rmse", [])
            if rmse_list and len(rmse_list) == n_epochs:
                # Extract this parameter's RMSE over epochs
                rmse_over_time = np.array([r[param_idx] for r in rmse_list])
                ax.plot(
                    epochs,
                    rmse_over_time,
                    label=source_name.upper(),
                    color=colors[i % len(colors)],
                    linestyle=styles[i % len(styles)],
                    alpha=0.8,
                )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE (normalized)")
        ax.set_title(f"{param_names[param_idx]} RMSE Evolution")
        _safe_legend(ax)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Labelset (Output Head) RMSE Evolution by Parameter", fontsize=12)
    plt.tight_layout()

    save_path = output_path / "per_labelset_metrics_evolution.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def generate_training_report(
    history: "TrainingHistory",
    output_dir: Path,
    experiment_name: str = "training",
) -> list[Path]:
    """
    Generate comprehensive training report with multiple plots.

    Args:
        history: TrainingHistory object from training.
        output_dir: Directory to save plots.
        experiment_name: Name for the report title.

    Returns:
        List of paths to generated plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = []

    # Generate each plot
    plot_funcs = [
        (plot_loss_curves, {"title": f"{experiment_name} Training"}),
        (plot_loss_components, {}),
        (plot_grokking_metrics, {}),
        (plot_val_metrics, {}),
        (plot_zscore_calibration, {}),
        (plot_metrics_evolution, {}),
    ]

    # Add per-survey plots if multi-survey data is available
    if history.survey_names and history.per_survey_val_losses:
        plot_funcs.extend(
            [
                (plot_per_survey_loss_curves, {}),
                (plot_per_survey_metrics, {}),
                (plot_per_survey_metrics_evolution, {}),
            ]
        )

    # Add per-labelset plots if multi-labelset data is available
    if history.label_source_names and history.per_labelset_val_losses:
        plot_funcs.extend(
            [
                (plot_per_labelset_loss_curves, {}),
                (plot_per_labelset_metrics, {}),
                (plot_per_labelset_metrics_evolution, {}),
            ]
        )

    for func, kwargs in plot_funcs:
        try:
            path = func(history, output_dir, **kwargs)
            if path is not None:
                plots.append(path)
        except Exception as e:
            # Don't fail training if plotting fails
            print(f"  Warning: Could not generate {func.__name__}: {e}")

    return plots
