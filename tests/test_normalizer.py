"""
Tests for the label normalization module.

These tests verify:
1. Correct median/IQR normalization
2. Special log10 handling for Teff
3. Error propagation through transforms
4. Roundtrip consistency (fit -> transform -> inverse_transform)
5. Save/load functionality
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dorothy.data.normalizer import LabelNormalizer, ParameterStats


class TestParameterStats:
    """Tests for ParameterStats dataclass."""

    def test_basic_stats(self):
        """Test creating basic parameter stats."""
        stats = ParameterStats(name="feh", median=-0.5, iqr=0.4)

        assert stats.name == "feh"
        assert stats.median == -0.5
        assert stats.iqr == 0.4
        assert stats.use_log is False

    def test_log_stats_validation(self):
        """Test that log stats are required when use_log=True."""
        with pytest.raises(ValueError, match="log_median and log_iqr required"):
            ParameterStats(name="teff", median=5000, iqr=1000, use_log=True)

    def test_valid_log_stats(self):
        """Test creating stats with log transformation."""
        stats = ParameterStats(
            name="teff",
            median=5000,
            iqr=1000,
            use_log=True,
            log_median=3.7,
            log_iqr=0.1,
        )

        assert stats.use_log is True
        assert stats.log_median == 3.7
        assert stats.log_iqr == 0.1


class TestLabelNormalizerInit:
    """Tests for LabelNormalizer initialization."""

    def test_default_parameters(self):
        """Test default parameters include all stellar parameters."""
        normalizer = LabelNormalizer()

        assert normalizer.n_parameters == 11
        assert "teff" in normalizer.parameters
        assert "feh" in normalizer.parameters
        assert normalizer.is_fitted is False

    def test_custom_parameters(self):
        """Test custom parameter list."""
        normalizer = LabelNormalizer(parameters=["teff", "logg", "feh"])

        assert normalizer.n_parameters == 3
        assert normalizer.parameters == ["teff", "logg", "feh"]

    def test_empty_parameters_rejected(self):
        """Test that empty parameters list is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            LabelNormalizer(parameters=[])


class TestLabelNormalizerFit:
    """Tests for fitting the normalizer."""

    def test_fit_simple(self):
        """Test fitting on simple data."""
        normalizer = LabelNormalizer(parameters=["param1", "param2"])

        # Create data with known statistics
        y = np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
            ]
        )

        normalizer.fit(y)

        assert normalizer.is_fitted
        assert "param1" in normalizer.stats
        assert "param2" in normalizer.stats

    def test_fit_computes_correct_median(self):
        """Test that fit computes correct median values."""
        normalizer = LabelNormalizer(parameters=["test"])

        y = np.array([[1], [2], [3], [4], [5]])  # Median = 3
        normalizer.fit(y)

        assert normalizer.stats["test"].median == 3.0

    def test_fit_computes_correct_iqr(self):
        """Test that fit computes correct IQR values."""
        normalizer = LabelNormalizer(parameters=["test"])

        # Values: 1, 2, 3, 4, 5
        # Q1 (25th) = 1.5, Q3 (75th) = 4.5, IQR = 3.0
        y = np.array([[1], [2], [3], [4], [5]])
        normalizer.fit(y)

        # NumPy percentile uses linear interpolation by default
        expected_iqr = np.percentile([1, 2, 3, 4, 5], 75) - np.percentile(
            [1, 2, 3, 4, 5], 25
        )
        assert normalizer.stats["test"].iqr == expected_iqr

    def test_fit_teff_uses_log(self):
        """Test that Teff parameter uses log10 transformation."""
        normalizer = LabelNormalizer(parameters=["teff", "logg"])

        y = np.array(
            [
                [4000, 2.0],
                [5000, 3.0],
                [6000, 4.0],
            ]
        )
        normalizer.fit(y)

        assert normalizer.stats["teff"].use_log is True
        assert normalizer.stats["teff"].log_median is not None
        assert normalizer.stats["logg"].use_log is False

    def test_fit_rejects_nan(self):
        """Test that NaN values are rejected."""
        normalizer = LabelNormalizer(parameters=["test"])

        y = np.array([[1], [np.nan], [3]])

        with pytest.raises(ValueError, match="NaN values"):
            normalizer.fit(y)

    def test_fit_rejects_inf(self):
        """Test that infinite values are rejected."""
        normalizer = LabelNormalizer(parameters=["test"])

        y = np.array([[1], [np.inf], [3]])

        with pytest.raises(ValueError, match="Inf values"):
            normalizer.fit(y)

    def test_fit_rejects_non_positive_teff(self):
        """Test that non-positive Teff values are rejected."""
        normalizer = LabelNormalizer(parameters=["teff"])

        y = np.array([[5000], [0], [6000]])  # Zero is invalid for log

        with pytest.raises(ValueError, match="Non-positive values"):
            normalizer.fit(y)

    def test_fit_rejects_zero_iqr(self):
        """Test that constant values (zero IQR) are rejected."""
        normalizer = LabelNormalizer(parameters=["test"])

        y = np.array([[1], [1], [1]])  # All same value

        with pytest.raises(ValueError, match="Zero IQR"):
            normalizer.fit(y)


class TestLabelNormalizerTransform:
    """Tests for transform functionality."""

    @pytest.fixture
    def fitted_normalizer(self):
        """Create a fitted normalizer for testing."""
        normalizer = LabelNormalizer(parameters=["param"])
        # Data: median=3, IQR=2
        y = np.array([[1], [2], [3], [4], [5]])
        normalizer.fit(y)
        return normalizer

    def test_transform_requires_fit(self):
        """Test that transform fails without fitting."""
        normalizer = LabelNormalizer(parameters=["test"])
        y = np.array([[1], [2]])

        with pytest.raises(RuntimeError, match="has not been fitted"):
            normalizer.transform(y)

    def test_transform_centers_median(self, fitted_normalizer):
        """Test that transform centers values around zero."""
        median = fitted_normalizer.stats["param"].median
        y = np.array([[median]])

        y_norm = fitted_normalizer.transform(y)

        assert abs(y_norm[0, 0]) < 1e-10

    def test_transform_scales_by_iqr(self, fitted_normalizer):
        """Test that transform scales by IQR."""
        stats = fitted_normalizer.stats["param"]
        # Value at median + 1 IQR should transform to 1.0
        y = np.array([[stats.median + stats.iqr]])

        y_norm = fitted_normalizer.transform(y)

        assert abs(y_norm[0, 0] - 1.0) < 1e-10

    def test_transform_with_errors(self, fitted_normalizer):
        """Test that errors are scaled correctly."""
        stats = fitted_normalizer.stats["param"]
        y = np.array([[stats.median]])
        errors = np.array([[stats.iqr]])  # Error equal to IQR

        y_norm, errors_norm = fitted_normalizer.transform(y, errors)

        # Error should be scaled to 1.0
        assert abs(errors_norm[0, 0] - 1.0) < 1e-10

    def test_transform_teff_log_space(self):
        """Test that Teff transform uses log10 space."""
        normalizer = LabelNormalizer(parameters=["teff"])

        # Create training data
        y_train = np.array([[4000], [5000], [6000], [7000], [8000]])
        normalizer.fit(y_train)

        # Transform should use log10
        y = np.array([[5000]])
        y_norm = normalizer.transform(y)

        # Manually compute expected value
        log_y = np.log10(5000)
        expected = (log_y - normalizer.stats["teff"].log_median) / normalizer.stats[
            "teff"
        ].log_iqr

        assert abs(y_norm[0, 0] - expected) < 1e-10


class TestLabelNormalizerInverseTransform:
    """Tests for inverse transform functionality."""

    def test_inverse_transform_requires_fit(self):
        """Test that inverse_transform fails without fitting."""
        normalizer = LabelNormalizer(parameters=["test"])
        y_norm = np.array([[0.0]])

        with pytest.raises(RuntimeError, match="has not been fitted"):
            normalizer.inverse_transform(y_norm)

    def test_inverse_transform_roundtrip(self):
        """Test that transform -> inverse_transform recovers original values."""
        normalizer = LabelNormalizer(parameters=["param1", "param2"])

        y_train = np.random.randn(100, 2) * 2 + 5
        normalizer.fit(y_train)

        y_test = np.random.randn(20, 2) * 2 + 5
        y_norm = normalizer.transform(y_test)
        y_recovered = normalizer.inverse_transform(y_norm)

        np.testing.assert_array_almost_equal(y_test, y_recovered)

    def test_inverse_transform_roundtrip_with_errors(self):
        """Test roundtrip with errors."""
        normalizer = LabelNormalizer(parameters=["param1", "param2"])

        y_train = np.random.randn(100, 2) * 2 + 5
        normalizer.fit(y_train)

        y_test = np.random.randn(20, 2) * 2 + 5
        errors = np.abs(np.random.randn(20, 2) * 0.5)

        y_norm, errors_norm = normalizer.transform(y_test, errors)
        y_recovered, errors_recovered = normalizer.inverse_transform(
            y_norm, errors_norm
        )

        np.testing.assert_array_almost_equal(y_test, y_recovered)
        np.testing.assert_array_almost_equal(errors, errors_recovered)

    def test_inverse_transform_teff_roundtrip(self):
        """Test Teff roundtrip through log space."""
        normalizer = LabelNormalizer(parameters=["teff"])

        y_train = np.array([[4000], [5000], [6000], [7000], [8000]])
        normalizer.fit(y_train)

        y_test = np.array([[4500], [5500], [6500]])
        y_norm = normalizer.transform(y_test)
        y_recovered = normalizer.inverse_transform(y_norm)

        np.testing.assert_array_almost_equal(y_test, y_recovered)


class TestLabelNormalizerSaveLoad:
    """Tests for save/load functionality."""

    def test_save_and_load(self):
        """Test saving and loading normalizer."""
        normalizer = LabelNormalizer(parameters=["param1", "param2"])
        y = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
        normalizer.fit(y)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            normalizer.save(path)
            loaded = LabelNormalizer.load(path, parameters=["param1", "param2"])

            assert loaded.is_fitted
            assert loaded.stats["param1"].median == normalizer.stats["param1"].median
            assert loaded.stats["param1"].iqr == normalizer.stats["param1"].iqr
        finally:
            path.unlink()

    def test_save_requires_fit(self):
        """Test that save fails without fitting."""
        normalizer = LabelNormalizer(parameters=["test"])

        with pytest.raises(RuntimeError, match="has not been fitted"):
            normalizer.save("test.pkl")

    def test_load_teff_with_log_stats(self):
        """Test loading normalizer with Teff log stats."""
        normalizer = LabelNormalizer(parameters=["teff", "logg"])
        y = np.array([[4000, 2.0], [5000, 3.0], [6000, 4.0], [7000, 5.0]])
        normalizer.fit(y)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            normalizer.save(path)
            loaded = LabelNormalizer.load(path, parameters=["teff", "logg"])

            assert loaded.stats["teff"].use_log is True
            assert (
                loaded.stats["teff"].log_median == normalizer.stats["teff"].log_median
            )
            assert loaded.stats["teff"].log_iqr == normalizer.stats["teff"].log_iqr
        finally:
            path.unlink()

    def test_roundtrip_after_load(self):
        """Test that loaded normalizer produces same results."""
        normalizer = LabelNormalizer(parameters=["teff", "feh"])
        y_train = np.column_stack(
            [
                np.random.uniform(4000, 8000, 100),
                np.random.uniform(-2, 0.5, 100),
            ]
        )
        normalizer.fit(y_train)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            normalizer.save(path)
            loaded = LabelNormalizer.load(path, parameters=["teff", "feh"])

            y_test = np.array([[5000, -0.5], [6000, 0.0]])
            y_norm_orig = normalizer.transform(y_test)
            y_norm_loaded = loaded.transform(y_test)

            np.testing.assert_array_almost_equal(y_norm_orig, y_norm_loaded)
        finally:
            path.unlink()


class TestLabelNormalizerGetParamsDict:
    """Tests for get_params_dict method."""

    def test_params_dict_format(self):
        """Test that params dict has expected format."""
        normalizer = LabelNormalizer(parameters=["feh", "logg"])
        y = np.array([[0.1, 2.0], [0.0, 3.0], [-0.1, 4.0]])
        normalizer.fit(y)

        params = normalizer.get_params_dict()

        assert "feh" in params
        assert "logg" in params
        assert "median" in params["feh"]
        assert "IQR" in params["feh"]

    def test_params_dict_teff_includes_log(self):
        """Test that Teff params include log stats."""
        normalizer = LabelNormalizer(parameters=["teff"])
        y = np.array([[4000], [5000], [6000]])
        normalizer.fit(y)

        params = normalizer.get_params_dict()

        assert "log_median" in params["teff"]
        assert "log_IQR" in params["teff"]


class TestLabelNormalizerMasking:
    """Tests for mask-aware normalization."""

    def test_fit_without_mask_unchanged(self):
        """Test that fit without mask works as before."""
        normalizer = LabelNormalizer(parameters=["feh", "logg"])
        y = np.array([[0.1, 2.0], [0.0, 3.0], [-0.1, 4.0]])

        normalizer.fit(y)
        normalizer_with_none = LabelNormalizer(parameters=["feh", "logg"])
        normalizer_with_none.fit(y, mask=None)

        assert (
            normalizer.stats["feh"].median == normalizer_with_none.stats["feh"].median
        )
        assert normalizer.stats["logg"].iqr == normalizer_with_none.stats["logg"].iqr

    def test_fit_with_all_ones_mask(self):
        """Test that all-ones mask gives same result as no mask."""
        normalizer = LabelNormalizer(parameters=["feh", "logg"])
        y = np.array([[0.1, 2.0], [0.0, 3.0], [-0.1, 4.0]])
        mask = np.ones_like(y)

        normalizer.fit(y, mask=mask)

        # Should match no-mask stats
        normalizer_no_mask = LabelNormalizer(parameters=["feh", "logg"])
        normalizer_no_mask.fit(y)

        assert normalizer.stats["feh"].median == normalizer_no_mask.stats["feh"].median

    def test_fit_mask_excludes_samples(self):
        """Test that masked samples don't contribute to stats."""
        normalizer = LabelNormalizer(parameters=["feh"])

        # First sample is an outlier that would affect stats
        y = np.array([[100.0], [0.0], [0.1], [0.2]])  # 100 is outlier
        mask = np.array([[0], [1], [1], [1]])  # Mask out the outlier

        normalizer.fit(y, mask=mask)

        # Stats should be computed from [0.0, 0.1, 0.2] only
        expected_median = 0.1
        assert normalizer.stats["feh"].median == pytest.approx(expected_median)

    def test_fit_mask_per_parameter(self):
        """Test that mask works independently per parameter."""
        normalizer = LabelNormalizer(parameters=["feh", "logg"])

        y = np.array(
            [
                [100.0, 2.0],  # feh outlier
                [0.0, 100.0],  # logg outlier
                [0.1, 3.0],
                [0.2, 4.0],
            ]
        )
        mask = np.array(
            [
                [0, 1],  # Mask feh outlier
                [1, 0],  # Mask logg outlier
                [1, 1],
                [1, 1],
            ]
        )

        normalizer.fit(y, mask=mask)

        # feh computed from [0.0, 0.1, 0.2]
        assert normalizer.stats["feh"].median == pytest.approx(0.1)
        # logg computed from [2.0, 3.0, 4.0]
        assert normalizer.stats["logg"].median == pytest.approx(3.0)

    def test_fit_all_masked_raises(self):
        """Test that fully masked parameter raises error."""
        normalizer = LabelNormalizer(parameters=["feh", "logg"])
        y = np.random.randn(10, 2)
        mask = np.ones_like(y)
        mask[:, 0] = 0  # Mask all feh values

        with pytest.raises(ValueError, match="All values masked"):
            normalizer.fit(y, mask=mask)

    def test_fit_mask_shape_mismatch_raises(self):
        """Test that mask shape mismatch raises error."""
        normalizer = LabelNormalizer(parameters=["feh", "logg"])
        y = np.random.randn(10, 2)
        mask = np.ones((5, 2))  # Wrong shape

        with pytest.raises(ValueError, match="mask shape"):
            normalizer.fit(y, mask=mask)

    def test_transform_with_mask_shape_validation(self):
        """Test that transform validates mask shape."""
        normalizer = LabelNormalizer(parameters=["feh", "logg"])
        y = np.array([[0.1, 2.0], [0.0, 3.0], [-0.1, 4.0]])
        normalizer.fit(y)

        y_new = np.random.randn(5, 2)
        mask = np.ones((3, 2))  # Wrong shape

        with pytest.raises(ValueError, match="mask shape"):
            normalizer.transform(y_new, mask=mask)

    def test_transform_with_mask_works(self):
        """Test that transform with mask parameter works."""
        normalizer = LabelNormalizer(parameters=["feh", "logg"])
        y = np.array([[0.1, 2.0], [0.0, 3.0], [-0.1, 4.0]])
        normalizer.fit(y)

        mask = np.ones_like(y)
        y_norm = normalizer.transform(y, mask=mask)

        # Result should be same as without mask
        y_norm_no_mask = normalizer.transform(y)
        assert np.allclose(y_norm, y_norm_no_mask)

    def test_fit_with_mask_and_teff(self):
        """Test mask-aware fitting with log-space Teff."""
        normalizer = LabelNormalizer(parameters=["teff", "logg"])

        y = np.array(
            [
                [10000.0, 2.0],  # Outlier teff
                [4000.0, 3.0],
                [5000.0, 4.0],
                [6000.0, 5.0],
            ]
        )
        mask = np.array(
            [
                [0, 1],  # Mask teff outlier
                [1, 1],
                [1, 1],
                [1, 1],
            ]
        )

        normalizer.fit(y, mask=mask)

        # Teff stats should be from [4000, 5000, 6000] in log space
        expected_log_median = np.median(np.log10([4000, 5000, 6000]))
        assert normalizer.stats["teff"].log_median == pytest.approx(
            expected_log_median, rel=0.01
        )
