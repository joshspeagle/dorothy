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

from dorothy.analysis import (
    LabelComparison,
    compare_label_sources,
    plot_label_differences,
)
from dorothy.inference.evaluator import evaluate_predictions
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


def _copy(path, tmp_path):
    return shutil.copy(path, tmp_path / "modified.h5")


def _keep_rows(path, source, rows):
    """Replace a label group by the given rows of itself, in the given order."""
    with h5py.File(path, "r+") as f:
        group = f[f"labels/{source}"]
        for name in ("values", "errors", "flags", "gaia_id"):
            data = group[name][...][rows]
            del group[name]
            group.create_dataset(name, data=data)


def _assert_same(result: LabelComparison, expected: LabelComparison) -> None:
    assert result.parameter_names == expected.parameter_names
    for name in STATISTICS:
        np.testing.assert_array_equal(
            getattr(result, name), getattr(expected, name), err_msg=name
        )


def test_correct_errors_give_a_ratio_of_one_and_no_median_offset(catalogue):
    result = compare_label_sources(catalogue)
    n = int(result.n_shared.min())
    spread = np.hypot(_noise(catalogue, "apogee"), _noise(catalogue, "galah"))

    assert result.parameter_names == PARAMS
    assert (result.source_a, result.source_b) == SOURCES
    assert result.ratio.shape == (len(PARAMS),)
    np.testing.assert_allclose(result.predicted_spread, spread, rtol=1e-6)
    np.testing.assert_allclose(result.robust_spread, spread, rtol=_rtol(n))
    np.testing.assert_allclose(result.ratio, 1, rtol=_rtol(n))
    assert (np.abs(result.median_offset) < 5 * 1.25 * spread / np.sqrt(n)).all()
    # Every star has the same errors, so the ratio is the ratio of the spreads
    np.testing.assert_allclose(
        result.ratio, result.robust_spread / result.predicted_spread, rtol=1e-9
    )


def test_overestimated_errors_lower_the_ratio_to_the_value_they_predict(
    catalogue, tmp_path
):
    path = _copy(catalogue, tmp_path)
    with h5py.File(path, "r+") as f:
        errors = f["labels/galah/errors"]
        errors[...] = 2 * errors[...]
    before = compare_label_sources(catalogue)
    result = compare_label_sources(path)
    sigma_a, sigma_b = _noise(catalogue, "apogee"), _noise(catalogue, "galah")
    predicted = np.hypot(sigma_a, 2 * sigma_b)

    # The labels are unchanged, so their differences and spread are too
    np.testing.assert_array_equal(result.n_shared, before.n_shared)
    np.testing.assert_array_equal(result.median_offset, before.median_offset)
    np.testing.assert_array_equal(result.robust_spread, before.robust_spread)
    np.testing.assert_allclose(result.predicted_spread, predicted, rtol=1e-6)
    np.testing.assert_allclose(
        result.ratio,
        np.hypot(sigma_a, sigma_b) / predicted,
        rtol=_rtol(result.n_shared.min()),
    )


def test_ratio_is_one_for_correct_errors_that_vary_from_star_to_star(
    catalogue, tmp_path
):
    # Add noise to three stars in ten, so that their GALAH noise is 5 times the
    # old error, and report the larger error. The errors now vary from star to
    # star and are correct for every star.
    path = _copy(catalogue, tmp_path)
    rng = np.random.default_rng(3)
    with h5py.File(path, "r+") as f:
        values, errors = f["labels/galah/values"], f["labels/galah/errors"]
        v, e = values[...], errors[...]
        wide = (np.arange(len(v)) % 10 < 3)[:, None]
        v += np.where(wide, rng.normal(size=v.shape) * e * np.sqrt(5**2 - 1), 0)
        values[...] = v
        errors[...] = np.where(wide, 5 * e, e)
    result = compare_label_sources(path)

    np.testing.assert_allclose(result.ratio, 1, rtol=_rtol(result.n_shared.min()))
    # The ratio of the two spreads would flag these correct errors
    assert (result.robust_spread / result.predicted_spread > 1.2).all()


@pytest.mark.parametrize("max_flag_bits", [-1, 0, 1, 3])
def test_only_labels_valid_in_both_sources_are_counted(tmp_path, max_flag_bits):
    path = build_catalogue(tmp_path / "catalogue.h5", n_stars=500, seed=2)
    rng = np.random.default_rng(1)
    valid = []
    with h5py.File(path, "r+") as f:
        for source in SOURCES:
            flags, errors = f[f"labels/{source}/flags"], f[f"labels/{source}/errors"]
            # Flags of 1, 1, 2 and 8 set bits on 5% of the labels, and an error
            # of 0 on another 5%
            hit = rng.uniform(size=flags.shape) < 0.05
            flags[...] = np.where(hit, rng.choice([1, 2, 3, 255], flags.shape), 0)
            errors[...] = np.where(rng.uniform(size=flags.shape) < 0.05, 0, errors[...])
            # Counted independently: the rows of both sources are the same stars
            ok = (flags[...] == 0) & (errors[...] > 0)
            if max_flag_bits >= 0:
                set_bits = np.unpackbits(flags[...], axis=1).sum(axis=1)
                ok &= (set_bits <= max_flag_bits)[:, None]
            valid.append(ok)

    result = compare_label_sources(path, max_flag_bits=max_flag_bits)

    np.testing.assert_array_equal(result.n_shared, (valid[0] & valid[1]).sum(axis=0))


def test_labels_stored_in_another_row_order_are_matched_by_gaia_id(tmp_path):
    path = build_catalogue(tmp_path / "catalogue.h5", n_stars=2000, seed=4)
    expected = compare_label_sources(path)
    rng = np.random.default_rng(7)
    for source in SOURCES:
        _keep_rows(path, source, rng.permutation(2000))

    result = compare_label_sources(path)

    assert expected.n_shared.min() > 0
    _assert_same(result, expected)


def test_stars_that_a_source_does_not_list_are_skipped(tmp_path):
    path = build_catalogue(tmp_path / "catalogue.h5", n_stars=2000, seed=4)
    expected = compare_label_sources(path)
    for source in SOURCES:
        with h5py.File(path) as f:
            labelled = np.flatnonzero(f[f"labels/{source}/errors"][...].any(axis=1))
        _keep_rows(path, source, labelled)

    _assert_same(compare_label_sources(path), expected)


def test_label_group_without_gaia_id_is_rejected(tmp_path):
    path = build_catalogue(tmp_path / "catalogue.h5", n_stars=50)
    with h5py.File(path, "r+") as f:
        del f["labels/galah/gaia_id"]
    with pytest.raises(ValueError, match="labels/galah has no gaia_id"):
        compare_label_sources(path)


def test_statistics_agree_with_the_evaluator(catalogue):
    # The evaluator's residual is a - b, and its z-score divides it by
    # sqrt(label_errors^2 + pred_scatter^2), so these arguments give d and z
    with h5py.File(catalogue) as f:
        values, errors = (
            [f[f"labels/{source}/{name}"][:].astype(np.float64) for source in SOURCES]
            for name in ("values", "errors")
        )
    shared = (errors[0] > 0) & (errors[1] > 0)  # all flags are 0
    reference = evaluate_predictions(
        values[0],
        values[1],
        pred_scatter=errors[0],
        label_errors=errors[1],
        parameter_names=PARAMS,
        mask=shared.astype(np.float32),
    )

    result = compare_label_sources(catalogue)

    for i, name in enumerate(PARAMS):
        metrics = reference[name]
        assert result.n_shared[i] == metrics.n_samples
        np.testing.assert_allclose(result.median_offset[i], metrics.median_offset)
        np.testing.assert_allclose(result.robust_spread[i], metrics.robust_scatter)
        np.testing.assert_allclose(result.ratio[i], metrics.z_robust_scatter)
        sigma_d = np.hypot(errors[0][shared[:, i], i], errors[1][shared[:, i], i])
        np.testing.assert_allclose(result.predicted_spread[i], np.median(sigma_d))


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
    # One NaN in each of the four statistic columns of the empty row
    assert result.summary().count("nan") == 4


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"source_b": "apogee"}, "source_a and source_b must differ"),
        ({"source_a": "lamost"}, "source_a: label source 'lamost'"),
        ({"source_b": "lamost"}, "source_b: label source 'lamost'"),
    ],
)
def test_invalid_label_sources_are_rejected(tmp_path, kwargs, message):
    path = build_catalogue(tmp_path / "catalogue.h5", n_stars=50)
    with pytest.raises(ValueError, match=message):
        compare_label_sources(path, **kwargs)


def test_plot_returns_a_figure_with_a_panel_per_parameter_and_axis(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    path = build_catalogue(tmp_path / "catalogue.h5", n_stars=300)
    fig = plot_label_differences(path)
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2 * len(PARAMS)
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
