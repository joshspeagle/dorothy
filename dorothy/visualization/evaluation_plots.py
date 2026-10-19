"""Test-set evaluation plots.

All functions take arrays in physical units, shaped (n_samples, n_params):
predictions, true labels, predicted model scatter, label errors and a validity
mask (1 = label valid). Grids are sized to the number of parameters, and axis
ranges default to percentiles of the data. A plot function
writes a PNG into a directory and returns its path, or None when matplotlib is
not installed.

Adapted from evaluation plots by Diego (d5martin).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dorothy.config.schema import parameter_display_name
from dorothy.visualization.training_plots import _try_import_matplotlib

# Sequential colour maps used by the scatter and density panels
_SCATTER_COLOURS = [
    (0.8, 1.0, 0.8),
    (0.2, 0.85, 0.5),
    (0.0, 0.6, 0.8),
    (0.0314, 0.4078, 0.6745),
    (0.0, 0.1490, 0.3176),
    (0.4, 0.0, 0.4),
]
_DENSITY_COLOURS = [
    (1.0, 1.0, 1.0),
    (0.2, 0.85, 0.5),
    (0.0, 0.6, 0.8),
    (0.0314, 0.4078, 0.6745),
    (0.0, 0.1490, 0.3176),
]

Range = tuple[float, float]


def _grid(plt, n_panels: int, panel_size: tuple[float, float] = (4.0, 3.5)):
    """Create a near-square grid of axes; unused axes are removed later."""
    n_cols = min(5, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
        squeeze=False,
    )
    axes = axes.flatten()
    for ax in axes[n_panels:]:
        fig.delaxes(ax)
    return fig, axes[:n_panels]


def _robust_range(*arrays: NDArray, lo: float = 0.5, hi: float = 99.5) -> Range:
    """Percentile range covering all finite values in the given arrays."""
    values = np.concatenate([np.ravel(a) for a in arrays])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (0.0, 1.0)
    vmin, vmax = np.percentile(values, [lo, hi])
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    pad = 0.02 * (vmax - vmin)
    return (float(vmin - pad), float(vmax + pad))


def _symmetric_range(values: NDArray, pct: float = 99.0) -> Range:
    """Symmetric range about zero covering the given percentile of |values|."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (-1.0, 1.0)
    half = float(np.percentile(np.abs(values), pct)) or 1.0
    return (-half, half)


def _valid(mask: NDArray, i: int, *arrays: NDArray) -> NDArray:
    """Boolean selector for samples with a valid, finite label for parameter i."""
    keep = mask[:, i].astype(bool)
    for a in arrays:
        keep &= np.isfinite(a[:, i])
    return keep


def _label_text(label_source: str | None) -> str:
    return label_source.upper() if label_source else "Label"


def _check_shapes(*arrays: NDArray) -> None:
    shapes = {np.shape(a) for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"All arrays must have the same shape, got {shapes}")


def plot_pred_vs_truth(
    output_path: str | Path,
    preds: NDArray,
    true: NDArray,
    pred_unc: NDArray,
    mask: NDArray,
    parameter_names: list[str],
    label_source: str | None = None,
    ranges: dict[str, Range] | None = None,
    filename: str = "pred_vs_truth.png",
) -> Path | None:
    """Predicted vs. true value per parameter, coloured by predicted uncertainty.

    Args:
        output_path: Directory to save the plot.
        preds: Predicted values, shape (n_samples, n_params).
        true: True label values, same shape.
        pred_unc: Predicted uncertainties, same shape.
        mask: Label validity mask (1 = valid), same shape.
        parameter_names: Parameter names (length n_params).
        label_source: Label source name, used in axis labels and titles.
        ranges: Optional per-parameter (min, max) axis ranges, keyed by name.
        filename: Output file name.

    Returns:
        Path of the plot, or None if matplotlib is missing.

    Raises:
        ValueError: If the arrays differ in shape.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None
    from matplotlib.colors import LinearSegmentedColormap

    _check_shapes(preds, true, pred_unc, mask)
    cmap = LinearSegmentedColormap.from_list("dorothy_scatter", _SCATTER_COLOURS)
    ranges = ranges or {}
    fig, axes = _grid(plt, len(parameter_names))

    for i, (ax, name) in enumerate(zip(axes, parameter_names, strict=True)):
        keep = _valid(mask, i, preds, true)
        x, y, c = true[keep, i], preds[keep, i], pred_unc[keep, i]
        lim = ranges.get(name) or _robust_range(x, y)
        if len(x) > 0:
            vmin, vmax = (
                np.nanpercentile(c, [1, 99]) if np.isfinite(c).any() else (0, 1)
            )
            sc = ax.scatter(x, y, s=4, alpha=0.6, c=c, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, label="Pred. uncertainty")
        ax.plot(lim, lim, color="black", lw=1, ls="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"{_label_text(label_source)} {parameter_display_name(name)}")
        ax.set_ylabel(f"Predicted {parameter_display_name(name)}")
        ax.set_title(f"{parameter_display_name(name)} (N={len(x):,})", fontsize=10)
        ax.grid(True, ls=":")

    fig.suptitle(f"Predicted vs. true ({_label_text(label_source)})")
    fig.tight_layout()
    save_path = Path(output_path) / filename
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_residuals(
    output_path: str | Path,
    preds: NDArray,
    true: NDArray,
    mask: NDArray,
    parameter_names: list[str],
    label_source: str | None = None,
    ranges: dict[str, Range] | None = None,
    n_bins: int = 50,
    filename: str = "residuals.png",
) -> Path | None:
    """Residual (true - predicted) density vs. true value with running quartiles.

    Args:
        output_path: Directory to save the plot.
        preds: Predicted values, shape (n_samples, n_params).
        true: True label values, same shape.
        mask: Label validity mask (1 = valid), same shape.
        parameter_names: Parameter names (length n_params).
        label_source: Label source name, used in axis labels and titles.
        ranges: Optional per-parameter (min, max) ranges for the true value.
        n_bins: Number of bins along each axis.
        filename: Output file name.

    Returns:
        Path of the plot, or None if matplotlib is missing.

    Raises:
        ValueError: If the arrays differ in shape.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None
    from matplotlib.colors import LinearSegmentedColormap

    _check_shapes(preds, true, mask)
    cmap = LinearSegmentedColormap.from_list("dorothy_density", _DENSITY_COLOURS)
    ranges = ranges or {}
    fig, axes = _grid(plt, len(parameter_names))

    for i, (ax, name) in enumerate(zip(axes, parameter_names, strict=True)):
        keep = _valid(mask, i, preds, true)
        x = true[keep, i]
        res = x - preds[keep, i]
        if len(x) == 0:
            ax.set_title(f"{parameter_display_name(name)} (no data)", fontsize=10)
            continue
        x_lim = ranges.get(name) or _robust_range(x)
        y_lim = _symmetric_range(res)
        x_edges = np.linspace(*x_lim, n_bins + 1)
        y_edges = np.linspace(*y_lim, n_bins + 1)
        counts, _, _ = np.histogram2d(x, res, bins=[x_edges, y_edges])
        pcm = ax.pcolormesh(x_edges, y_edges, counts.T, cmap=cmap, shading="auto")
        fig.colorbar(pcm, ax=ax, label="Number of stars")

        # Running quartiles in bins with data
        centres, quartiles = [], []
        bin_idx = np.digitize(x, x_edges) - 1
        for b in range(n_bins):
            in_bin = res[bin_idx == b]
            if len(in_bin) > 0:
                centres.append(0.5 * (x_edges[b] + x_edges[b + 1]))
                quartiles.append(np.percentile(in_bin, [25, 50, 75]))
        if centres:
            quartiles = np.array(quartiles)
            ax.plot(centres, quartiles[:, 0], "k--", lw=1, label="25th pct")
            ax.plot(centres, quartiles[:, 1], "k-", lw=1.2, label="Median")
            ax.plot(centres, quartiles[:, 2], "k:", lw=1, label="75th pct")
            ax.legend(fontsize=7)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel(f"{_label_text(label_source)} {parameter_display_name(name)}")
        ax.set_ylabel("True - predicted")
        ax.set_title(parameter_display_name(name), fontsize=10)
        ax.grid(True, ls="--", alpha=0.5)

    fig.suptitle(f"Residuals ({_label_text(label_source)})")
    fig.tight_layout()
    save_path = Path(output_path) / filename
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_histograms(
    output_path: str | Path,
    preds: NDArray,
    true: NDArray,
    mask: NDArray,
    parameter_names: list[str],
    label_source: str | None = None,
    ranges: dict[str, Range] | None = None,
    n_bins: int = 60,
    filename: str = "histograms.png",
) -> Path | None:
    """Overlaid histograms of true and predicted values per parameter.

    Args:
        output_path: Directory to save the plot.
        preds: Predicted values, shape (n_samples, n_params).
        true: True label values, same shape.
        mask: Label validity mask (1 = valid), same shape.
        parameter_names: Parameter names (length n_params).
        label_source: Label source name, used in the legend.
        ranges: Optional per-parameter (min, max) histogram ranges.
        n_bins: Number of histogram bins.
        filename: Output file name.

    Returns:
        Path of the plot, or None if matplotlib is missing.

    Raises:
        ValueError: If the arrays differ in shape.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    _check_shapes(preds, true, mask)
    ranges = ranges or {}
    fig, axes = _grid(plt, len(parameter_names))

    for i, (ax, name) in enumerate(zip(axes, parameter_names, strict=True)):
        keep = _valid(mask, i, preds, true)
        x, y = true[keep, i], preds[keep, i]
        lim = ranges.get(name) or _robust_range(x, y)
        bins = np.linspace(*lim, n_bins + 1)
        ax.hist(x, bins=bins, alpha=0.6, label=_label_text(label_source))
        ax.hist(y, bins=bins, alpha=0.6, label="DOROTHY")
        ax.set_xlabel(parameter_display_name(name))
        ax.set_ylabel("Number of stars")
        ax.legend(fontsize=7)

    fig.tight_layout()
    save_path = Path(output_path) / filename
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_zscore_histograms(
    output_path: str | Path,
    preds: NDArray,
    true: NDArray,
    pred_unc: NDArray,
    label_err: NDArray | None,
    mask: NDArray,
    parameter_names: list[str],
    label_source: str | None = None,
    z_max: float = 5.0,
    filename: str = "zscore_histograms.png",
) -> Path | None:
    """Per-parameter z-score histograms against a unit Gaussian.

    z = (predicted - true) / sqrt(pred_unc^2 + label_err^2), as in
    ``Evaluator``. Each panel reports the median and the robust scatter
    (P84 - P16) / 2. A calibrated parameter has median 0 and scatter 1.
    Scatter above 1 means overconfident, below 1 underconfident. Values
    outside [-z_max, z_max] fall in the end bins.

    Args:
        output_path: Directory to save the plot.
        preds: Predicted values, shape (n_samples, n_params).
        true: True label values, same shape.
        pred_unc: Predicted model scatter, same shape.
        label_err: Label errors, same shape, or None to use pred_unc only.
        mask: Label validity mask (1 = valid), same shape.
        parameter_names: Parameter names (length n_params).
        label_source: Label source name, used in the title.
        z_max: Histogram range is [-z_max, z_max].
        filename: Output file name.

    Returns:
        Path of the plot, or None if matplotlib is missing.

    Raises:
        ValueError: If the arrays differ in shape.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    if label_err is None:
        label_err = np.zeros_like(pred_unc)
    _check_shapes(preds, true, pred_unc, label_err, mask)
    fig, axes = _grid(plt, len(parameter_names), panel_size=(3.6, 3.0))
    grid = np.linspace(-z_max, z_max, 200)
    gaussian = np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)

    for i, (ax, name) in enumerate(zip(axes, parameter_names, strict=True)):
        keep = _valid(mask, i, preds, true, pred_unc)
        total = np.sqrt(pred_unc[keep, i] ** 2 + label_err[keep, i] ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (preds[keep, i] - true[keep, i]) / total
        z = z[np.isfinite(z)]
        if len(z) > 0:
            p16, p50, p84 = np.percentile(z, [16, 50, 84])
            ax.hist(
                np.clip(z, -z_max, z_max),
                bins=np.linspace(-z_max, z_max, 41),
                density=True,
                alpha=0.7,
            )
            ax.text(
                0.03,
                0.95,
                f"med={p50:+.2f}\nscatter={(p84 - p16) / 2:.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
        ax.plot(grid, gaussian, "k--", lw=1)
        ax.set_xlim(-z_max, z_max)
        ax.set_title(f"{parameter_display_name(name)} (N={len(z):,})", fontsize=10)
        ax.set_xlabel("z-score")

    fig.suptitle(f"Uncertainty calibration ({_label_text(label_source)})")
    fig.tight_layout()
    save_path = Path(output_path) / filename
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def generate_evaluation_plots(
    output_dir: str | Path,
    preds: NDArray,
    true: NDArray,
    pred_unc: NDArray,
    label_err: NDArray | None,
    mask: NDArray,
    parameter_names: list[str],
    label_source: str | None = None,
    ranges: dict[str, Range] | None = None,
) -> list[Path]:
    """Write the four evaluation plots for one label source.

    The files are ``pred_vs_truth.png``, ``residuals.png``,
    ``histograms.png`` and ``zscore_histograms.png``, each prefixed with
    ``<label_source>_`` when a label source is given.

    Args:
        output_dir: Directory of the plots, created if missing.
        preds: Predicted values, shape (n_samples, n_params).
        true: True label values, same shape.
        pred_unc: Predicted model scatter, same shape.
        label_err: Label errors, same shape, or None to use pred_unc only.
        mask: Label validity mask (1 = valid), same shape.
        parameter_names: Parameter names (length n_params).
        label_source: Label source name, used in titles and file names.
        ranges: Optional per-parameter (min, max) ranges for the first three
            plots, keyed by name.

    Returns:
        Paths of the plots written; empty if matplotlib is missing.

    Raises:
        ValueError: If the arrays differ in shape.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{label_source}_" if label_source else ""
    common = {"parameter_names": parameter_names, "label_source": label_source}
    paths = [
        plot_pred_vs_truth(
            output_dir,
            preds,
            true,
            pred_unc,
            mask,
            ranges=ranges,
            filename=f"{prefix}pred_vs_truth.png",
            **common,
        ),
        plot_residuals(
            output_dir,
            preds,
            true,
            mask,
            ranges=ranges,
            filename=f"{prefix}residuals.png",
            **common,
        ),
        plot_histograms(
            output_dir,
            preds,
            true,
            mask,
            ranges=ranges,
            filename=f"{prefix}histograms.png",
            **common,
        ),
        plot_zscore_histograms(
            output_dir,
            preds,
            true,
            pred_unc,
            label_err,
            mask,
            filename=f"{prefix}zscore_histograms.png",
            **common,
        ),
    ]
    return [p for p in paths if p is not None]
