"""Tests for the comparison of the labels of two label sources.

tests/synthetic.py draws each label as the truth plus Gaussian noise whose
standard deviation is the label's reported error, and both sources share the
truth. So the difference d of the two labels of a star has the standard
deviation sigma_d = sqrt(sigma_A^2 + sigma_B^2), and d / sigma_d is standard
normal.
"""

from __future__ import annotations

import shutil

import h5py
import numpy as np
import pytest

from dorothy.analysis import compare_label_sources, plot_label_differences
from tests.synthetic import PARAMS, build_catalogue

SOURCES = ("apogee", "galah")
STATISTICS = ("n_shared", "median_offset", "robust_spread", "predicted_spread", "ratio")


@pytest.fixture(scope="module")
def catalogue(tmp_path_factory):
    """A catalogue whose two sources share 7,588 stars per parameter."""
    path = tmp_path_factory.mktemp("label_comparison") / "catalogue.h5"
    return build_catalogue(path, n_stars=20_000)


# Tolerances are 5 standard errors. For n normal values with standard
# deviation s, the standard error of the median is 1.25 s / sqrt(n) and that
# of (P84 - P16) / 2 is 0.96 s / sqrt(n), from the asymptotic variance of
# sample quantiles. For the 7,588 shared stars of the catalogue above,
# 5 * 0.96 / sqrt(7588) = 0.055 and 5 * 1.25 / sqrt(7588) = 0.072. (P84 - P16)
# / 2 is also 0.5% below s for normal values, which is small against these.


def _rtol(n):
    """Five standard errors of (P84 - P16) / 2, relative to its expected value."""
    return 5 * 0.96 / np.sqrt(n)


def _noise(path, source):
    """The noise of each parameter's labels, which is also their reported error."""
    with h5py.File(path) as f:
        return f[f"labels/{source}/errors"][:].max(axis=0).astype(np.float64)


def test_correct_errors_give_a_ratio_of_one_and_no_median_offset(catalogue):
    result = compare_label_sources(catalogue)
    n = int(result.n_shared.min())
    spread = np.hypot(_noise(catalogue, "apogee"), _noise(catalogue, "galah"))

    np.testing.assert_allclose(result.predicted_spread, spread, rtol=1e-6)
    np.testing.assert_allclose(result.robust_spread, spread, rtol=_rtol(n))
    np.testing.assert_allclose(result.ratio, 1, rtol=_rtol(n))
    assert (np.abs(result.median_offset) < 5 * 1.25 * spread / np.sqrt(n)).all()
    # Every star has the same errors, so the ratio is the ratio of the spreads
    np.testing.assert_allclose(
        result.ratio, result.robust_spread / result.predicted_spread, rtol=1e-9
    )


@pytest.mark.parametrize("max_flag_bits", [-1, 0])
def test_only_labels_valid_in_both_sources_are_counted(tmp_path, max_flag_bits):
    path = build_catalogue(tmp_path / "catalogue.h5", n_stars=500, seed=2)
    rng = np.random.default_rng(1)
    valid = []
    with h5py.File(path, "r+") as f:
        for source in SOURCES:
            flags, errors = f[f"labels/{source}/flags"], f[f"labels/{source}/errors"]
            # A flag on 5% of the labels, and an error of 0 on another 5%
            flags[...] = rng.uniform(size=flags.shape) < 0.05
            errors[...] = np.where(rng.uniform(size=flags.shape) < 0.05, 0, errors[...])
            # The rows of both sources are the same stars
            ok = (flags[...] == 0) & (errors[...] > 0)
            if max_flag_bits == 0:
                ok &= (flags[...] == 0).all(axis=1)[:, None]
            valid.append(ok)

    result = compare_label_sources(path, max_flag_bits=max_flag_bits)

    np.testing.assert_array_equal(result.n_shared, (valid[0] & valid[1]).sum(axis=0))


def test_statistics_match_a_direct_computation(catalogue, tmp_path):
    # Errors that vary from star to star separate the median, P16 and P84 of
    # sigma_d, so a statistic read from the wrong metric fails
    path = shutil.copy(catalogue, tmp_path / "varied.h5")
    rng = np.random.default_rng(5)
    with h5py.File(path, "r+") as f:
        stored = f["labels/galah/errors"]
        stored[...] = stored[...] * rng.uniform(0.5, 2.0, stored.shape)
        values, errors = (
            [f[f"labels/{source}/{name}"][:].astype(np.float64) for source in SOURCES]
            for name in ("values", "errors")
        )
    shared = (errors[0] > 0) & (errors[1] > 0)  # all flags are 0

    result = compare_label_sources(path)

    def spread(x):
        p16, p84 = np.percentile(x, [16, 84])
        return (p84 - p16) / 2

    for i in range(len(PARAMS)):
        rows = shared[:, i]
        d = values[0][rows, i] - values[1][rows, i]
        sigma_d = np.hypot(errors[0][rows, i], errors[1][rows, i])
        assert result.n_shared[i] == rows.sum()
        np.testing.assert_allclose(result.median_offset[i], np.median(d))
        np.testing.assert_allclose(result.robust_spread[i], spread(d))
        np.testing.assert_allclose(result.predicted_spread[i], np.median(sigma_d))
        np.testing.assert_allclose(result.ratio[i], spread(d / sigma_d))


def test_summary_lists_the_sources_and_each_parameter_with_its_statistics(catalogue):
    result = compare_label_sources(catalogue)

    lines = result.summary().splitlines()
    rows = [line.split() for line in lines[2:]]

    assert lines[0] == "Label differences d = apogee - galah"
    assert [row[0] for row in rows] == PARAMS
    assert [int(row[1]) for row in rows] == result.n_shared.tolist()
    np.testing.assert_allclose([float(row[5]) for row in rows], result.ratio, atol=5e-4)


def test_parameter_without_shared_stars_has_zero_count_and_nan_statistics(tmp_path):
    path = build_catalogue(tmp_path / "catalogue.h5", n_stars=500, seed=6)
    empty = PARAMS.index("logg")
    with h5py.File(path, "r+") as f:
        f["labels/galah/errors"][:, empty] = 0

    result = compare_label_sources(path)

    others = [i for i in range(len(PARAMS)) if i != empty]
    assert result.n_shared[empty] == 0
    assert (result.n_shared[others] > 0).all()
    for name in STATISTICS[1:]:
        statistic = getattr(result, name)
        assert np.isnan(statistic[empty])
        assert np.isfinite(statistic[others]).all()


def test_equal_label_sources_are_rejected(tmp_path):
    path = build_catalogue(tmp_path / "catalogue.h5", n_stars=50)
    with pytest.raises(ValueError, match="source_a and source_b must differ"):
        compare_label_sources(path, source_b="apogee")


def test_plot_has_a_panel_per_parameter_and_axis_and_limits_without_outliers(
    catalogue, tmp_path
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = shutil.copy(catalogue, tmp_path / "outlier.h5")
    with h5py.File(path, "r+") as f:
        # One shared star whose GALAH labels are 1e5 above its APOGEE labels
        star = np.flatnonzero(
            (f["labels/apogee/errors"][:] > 0).all(axis=1)
            & (f["labels/galah/errors"][:] > 0).all(axis=1)
        )[0]
        f["labels/galah/values"][star] += 1e5
    fig = plot_label_differences(path)
    try:
        assert len(fig.axes) == 2 * len(PARAMS)
        for ax in fig.axes:
            assert np.abs([*ax.get_xlim(), *ax.get_ylim()]).max() < 1e4
    finally:
        plt.close(fig)


@pytest.mark.parametrize("missing", ["teff", "fe_h"])
def test_plot_without_teff_or_fe_h_names_the_missing_parameter(tmp_path, missing):
    path = build_catalogue(tmp_path / "catalogue.h5", n_stars=50)
    with h5py.File(path, "r+") as f:
        f.attrs["parameter_names"] = [
            name.replace(missing, "renamed") for name in PARAMS
        ]
    with pytest.raises(ValueError, match=f"no parameter '{missing}'"):
        plot_label_differences(path)
