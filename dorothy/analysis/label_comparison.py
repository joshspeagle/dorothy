"""Compare the labels of the stars that two label sources both measure.

If APOGEE and GALAH report correct errors, the difference of their labels for
a star has the spread that the two errors predict. This module compares the
measured and the predicted spread for each parameter, and plots the
differences against Teff and [Fe/H]. It reads only the catalogue and needs no
trained model.

Examples:
    >>> from dorothy.analysis import compare_label_sources, plot_label_differences
    >>> comparison = compare_label_sources("data/super_catalogue.h5")
    >>> print(comparison.summary())
    >>> fig = plot_label_differences("data/super_catalogue.h5")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dorothy.config.schema import parameter_display_name
from dorothy.data.catalogue_loader import CatalogueLoader
from dorothy.inference.evaluator import Evaluator

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass
class LabelComparison:
    """Differences between the labels of two sources for the stars they share.

    For one parameter and one shared star, ``d = y_A - y_B`` is the difference
    of the labels of sources A and B. The spread that their errors predict is
    ``sigma_d = sqrt(sigma_A^2 + sigma_B^2)``. Each array has one entry per
    parameter, in the order of ``parameter_names``, and is in the physical
    units of the catalogue (K for Teff, dex for the others), except
    ``n_shared`` and ``ratio``. A parameter without a shared star has
    ``n_shared`` 0 and NaN statistics.

    Attributes:
        parameter_names: Parameter names of the catalogue.
        source_a: Name of label source A.
        source_b: Name of label source B.
        n_shared: Number of stars with a valid label of the parameter in both
            sources, shape (n_parameters,).
        median_offset: Median of d, shape (n_parameters,).
        robust_spread: (P84 - P16) / 2 of d, shape (n_parameters,).
        predicted_spread: Median of sigma_d, shape (n_parameters,).
        ratio: (P84 - P16) / 2 of ``d / sigma_d``, shape (n_parameters,). It is
            about 1 when both sources report correct errors. When all stars
            have the same errors, it equals ``robust_spread / predicted_spread``.
    """

    parameter_names: list[str]
    source_a: str
    source_b: str
    n_shared: NDArray[np.int64]
    median_offset: NDArray[np.float64]
    robust_spread: NDArray[np.float64]
    predicted_spread: NDArray[np.float64]
    ratio: NDArray[np.float64]

    def summary(self) -> str:
        """Format the statistics as a text table with one row per parameter."""
        width = max(len(name) for name in ["Parameter", *self.parameter_names]) + 2
        lines = [
            f"Label differences d = {self.source_a} - {self.source_b}",
            f"{'Parameter':<{width}}{'n_shared':>10}{'median_offset':>15}"
            f"{'robust_spread':>15}{'predicted_spread':>18}{'ratio':>8}",
        ]
        for i, name in enumerate(self.parameter_names):
            lines.append(
                f"{name:<{width}}{self.n_shared[i]:>10d}"
                f"{self.median_offset[i]:>15.4g}{self.robust_spread[i]:>15.4g}"
                f"{self.predicted_spread[i]:>18.4g}{self.ratio[i]:>8.3f}"
            )
        return "\n".join(lines)


def _shared_labels(
    loader: CatalogueLoader, source_a: str, source_b: str, max_flag_bits: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Labels of both sources for the stars they share for any parameter.

    Returns ``(a, b, both)``: the (n_stars, 3, n_parameters) labels [values,
    errors, mask] of the two sources, row i of both for the same star, and
    the (n_stars, n_parameters) mask of the labels valid in both.
    """
    if source_a == source_b:
        raise ValueError(f"source_a and source_b must differ, got '{source_a}' twice")
    a, b = (loader.load_labels(s, max_flag_bits) for s in (source_a, source_b))
    both = (a[:, 2] > 0) & (b[:, 2] > 0)
    # Only the shared stars are kept and widened to float64
    shared = both.any(axis=1)
    return a[shared].astype(np.float64), b[shared].astype(np.float64), both[shared]


def compare_label_sources(
    catalogue_path: str | Path,
    source_a: str = "apogee",
    source_b: str = "galah",
    max_flag_bits: int = 0,
) -> LabelComparison:
    """Compare two label sources on the stars that both measure.

    For each parameter, the function compares the difference ``d = y_A - y_B``
    of the two labels of a star with ``sigma_d = sqrt(sigma_A^2 + sigma_B^2)``,
    the spread that the reported errors predict. If both sources report
    correct errors and agree on average, ``z = d / sigma_d`` has a standard
    normal distribution.

    A star is shared for a parameter when both sources have a valid label of
    it. :meth:`~dorothy.data.catalogue_loader.CatalogueLoader.load_labels`
    reads each source, applies the validity rule and matches the labels to
    stars by Gaia ID. The function reads the two label groups and no
    spectra.

    Args:
        catalogue_path: Path to the HDF5 super-catalogue.
        source_a: Name of label source A.
        source_b: Name of label source B.
        max_flag_bits: Largest number of set flag bits, summed over the
            parameters of a star, for its labels to count; -1 sets no limit
            (see ``CatalogueLoader.load_labels``).

    Returns:
        The statistics of each parameter, in physical units (K for Teff, dex
        for the others).

    Raises:
        FileNotFoundError: If the catalogue does not exist.
        ValueError: If ``source_a`` and ``source_b`` are equal, a source is not
            in the catalogue, or the catalogue or a label group has no Gaia
            IDs.
    """
    loader = CatalogueLoader(catalogue_path)
    names = loader.get_parameter_names()
    a, b, both = _shared_labels(loader, source_a, source_b, max_flag_bits)
    # The evaluator's residual is a - b, and with sigma_d as the predicted
    # scatter its z-score is d / sigma_d
    result = Evaluator(names, teff_in_log=False).evaluate(
        a[:, 0], b[:, 0], pred_scatter=np.hypot(a[:, 1], b[:, 1]), mask=both
    )
    metrics = [result[name] for name in names]
    return LabelComparison(
        parameter_names=names,
        source_a=source_a,
        source_b=source_b,
        n_shared=np.array([m.n_samples for m in metrics]),
        median_offset=np.array([m.median_offset for m in metrics]),
        robust_spread=np.array([m.robust_scatter for m in metrics]),
        predicted_spread=np.array([m.pred_unc_p50 for m in metrics]),
        ratio=np.array([m.z_robust_scatter for m in metrics]),
    )


def plot_label_differences(
    catalogue_path: str | Path,
    source_a: str = "apogee",
    source_b: str = "galah",
    max_flag_bits: int = 0,
) -> Figure:
    """Plot the label differences of the shared stars against Teff and [Fe/H].

    The figure has one row per parameter and two columns. A panel plots
    ``d = y_A - y_B`` against the mean of the two sources' Teff (left) or
    [Fe/H] (right). It shows the stars that have a valid label of the
    parameter, and of Teff or [Fe/H], in both sources. A trend of d with Teff
    or [Fe/H] is a systematic difference between the sources, and it raises
    the ``ratio`` of ``compare_label_sources``.

    Args:
        catalogue_path: Path to the HDF5 super-catalogue.
        source_a: Name of label source A.
        source_b: Name of label source B.
        max_flag_bits: As in ``compare_label_sources``.

    Returns:
        The Matplotlib figure, with d in the physical units of the catalogue
        (K for Teff, dex for the others).

    Raises:
        ValueError: If the catalogue has no parameter ``teff`` or ``fe_h``, or
            for a reason given at ``compare_label_sources``.
    """
    loader = CatalogueLoader(catalogue_path)
    names = loader.get_parameter_names()
    missing = [name for name in ("teff", "fe_h") if name not in names]
    if missing:
        raise ValueError(
            f"The catalogue has no parameter {', '.join(map(repr, missing))}, "
            f"which the plot needs for its x axes; it has {names}"
        )
    a, b, both = _shared_labels(loader, source_a, source_b, max_flag_bits)
    import matplotlib.pyplot as plt

    from dorothy.visualization.evaluation_plots import _robust_range, _symmetric_range

    fig, axes = plt.subplots(
        len(names),
        2,
        figsize=(9, 1.8 * len(names)),
        sharex="col",
        layout="constrained",
    )
    # The axis limits are percentiles, so that a few outliers do not hide the
    # trends
    for column, x_name in enumerate(("teff", "fe_h")):
        x = names.index(x_name)
        for row in range(len(names)):
            shared = both[:, row] & both[:, x]
            d = a[shared, 0, row] - b[shared, 0, row]
            ax = axes[row, column]
            ax.scatter((a[shared, 0, x] + b[shared, 0, x]) / 2, d, s=2, alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylim(_symmetric_range(d))
        valid = both[:, x]
        axes[-1, column].set_xlim(_robust_range((a[valid, 0, x] + b[valid, 0, x]) / 2))
        axes[-1, column].set_xlabel(f"mean {parameter_display_name(x_name)}")
    for row, name in enumerate(names):
        axes[row, 0].set_ylabel(
            f"{parameter_display_name(name)}: {source_a} - {source_b}"
        )
    return fig
