"""
Tests for the Evaluator class.

These tests verify:
1. Metric computation (RMSE, bias, SD, MAE, median offset, robust scatter)
2. Per-parameter evaluation
3. Special handling for Teff in log space
4. Z-score calibration metrics
5. Summary output formatting
"""

import numpy as np
import pytest

from dorothy.inference.evaluator import (
    EvaluationResult,
    Evaluator,
    ParameterMetrics,
    evaluate_predictions,
)


class TestParameterMetrics:
    """Tests for ParameterMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ParameterMetrics(
            name="teff",
            n_samples=100,
            rmse=75.0,
            bias=-10.0,
            sd=70.0,
            mae=50.0,
            median_offset=-8.0,
            robust_scatter=65.0,
        )

        d = metrics.to_dict()
        assert d["name"] == "teff"
        assert d["rmse"] == 75.0
        assert d["bias"] == -10.0
        assert d["sd"] == 70.0
        assert d["mae"] == 50.0
        assert d["median_offset"] == -8.0
        assert d["robust_scatter"] == 65.0
        assert d["n_samples"] == 100

    def test_to_dict_with_zscore_metrics(self):
        """Test conversion with z-score metrics included."""
        metrics = ParameterMetrics(
            name="logg",
            n_samples=50,
            rmse=0.15,
            bias=0.02,
            sd=0.14,
            mae=0.1,
            median_offset=0.01,
            robust_scatter=0.12,
            z_median=0.05,
            z_robust_scatter=1.02,
            pred_unc_p16=0.08,
            pred_unc_p50=0.12,
            pred_unc_p84=0.18,
        )

        d = metrics.to_dict()
        assert d["z_median"] == 0.05
        assert d["z_robust_scatter"] == 1.02
        assert d["pred_unc_p16"] == 0.08
        assert d["pred_unc_p50"] == 0.12
        assert d["pred_unc_p84"] == 0.18


class TestEvaluationResult:
    """Tests for EvaluationResult container."""

    def test_getitem(self):
        """Test dictionary-style access."""
        metrics = ParameterMetrics(
            name="logg",
            n_samples=50,
            rmse=0.15,
            bias=0.02,
            sd=0.14,
            mae=0.1,
            median_offset=0.01,
            robust_scatter=0.12,
        )
        result = EvaluationResult(
            metrics={"logg": metrics},
            parameter_names=["logg"],
        )

        assert result["logg"] == metrics

    def test_summary_text(self):
        """Test text summary generation."""
        metrics = ParameterMetrics(
            name="teff",
            n_samples=100,
            rmse=75.0,
            bias=-10.0,
            sd=70.0,
            mae=50.0,
            median_offset=-8.0,
            robust_scatter=65.0,
        )
        result = EvaluationResult(
            metrics={"teff": metrics},
            parameter_names=["teff"],
        )

        summary = result.summary(format="text")
        assert "teff" in summary
        assert "RMSE" in summary
        assert "100" in summary

    def test_summary_markdown(self):
        """Test markdown summary generation."""
        metrics = ParameterMetrics(
            name="feh",
            n_samples=200,
            rmse=0.08,
            bias=0.01,
            sd=0.07,
            mae=0.05,
            median_offset=0.005,
            robust_scatter=0.06,
        )
        result = EvaluationResult(
            metrics={"feh": metrics},
            parameter_names=["feh"],
        )

        summary = result.summary(format="markdown")
        assert "| feh |" in summary
        assert "|---|" in summary or "|-----------|" in summary

    def test_to_dict(self):
        """Test conversion to nested dictionary."""
        metrics = ParameterMetrics(
            name="logg",
            n_samples=50,
            rmse=0.15,
            bias=0.02,
            sd=0.14,
            mae=0.1,
            median_offset=0.01,
            robust_scatter=0.12,
        )
        result = EvaluationResult(
            metrics={"logg": metrics},
            parameter_names=["logg"],
        )

        d = result.to_dict()
        assert "logg" in d
        assert d["logg"]["mae"] == 0.1


class TestEvaluator:
    """Tests for the Evaluator class."""

    def test_init_default_params(self):
        """Test default initialization."""
        evaluator = Evaluator()
        assert evaluator.parameter_names == ["teff", "logg", "feh"]
        assert evaluator.teff_in_log is True

    def test_init_custom_params(self):
        """Test custom parameter initialization."""
        evaluator = Evaluator(
            parameter_names=["teff", "logg", "feh", "mgfe"],
            teff_in_log=False,
        )
        assert len(evaluator.parameter_names) == 4
        assert evaluator.teff_in_log is False

    def test_evaluate_basic(self):
        """Test basic evaluation."""
        evaluator = Evaluator(parameter_names=["teff", "logg"], teff_in_log=False)

        n_samples = 100
        y_true = np.column_stack(
            [
                np.full(n_samples, 5500.0),  # teff
                np.full(n_samples, 4.0),  # logg
            ]
        ).astype(np.float32)

        # Predictions with known errors
        y_pred = np.column_stack(
            [
                y_true[:, 0] + 50.0,  # teff: +50K bias
                y_true[:, 1] + 0.1,  # logg: +0.1 bias
            ]
        ).astype(np.float32)

        result = evaluator.evaluate(y_pred, y_true)

        assert "teff" in result.metrics
        assert "logg" in result.metrics

        # Check teff metrics
        teff_metrics = result["teff"]
        assert teff_metrics.mae == pytest.approx(50.0, abs=0.1)
        assert teff_metrics.bias == pytest.approx(50.0, abs=0.1)
        assert teff_metrics.median_offset == pytest.approx(50.0, abs=0.1)
        assert teff_metrics.n_samples == n_samples

        # Check logg metrics
        logg_metrics = result["logg"]
        assert logg_metrics.mae == pytest.approx(0.1, abs=0.001)
        assert logg_metrics.bias == pytest.approx(0.1, abs=0.001)

    def test_evaluate_with_scatter(self):
        """Test evaluation with random scatter."""
        evaluator = Evaluator(parameter_names=["teff"], teff_in_log=False)

        np.random.seed(42)
        n_samples = 1000
        true_scatter = 100.0

        y_true = np.full((n_samples, 1), 5500.0, dtype=np.float32)
        y_pred = (y_true + np.random.randn(n_samples, 1) * true_scatter).astype(
            np.float32
        )

        result = evaluator.evaluate(y_pred, y_true)

        # SD should be approximately the true scatter
        assert result["teff"].sd == pytest.approx(true_scatter, rel=0.1)
        # Robust scatter should also be close
        assert result["teff"].robust_scatter == pytest.approx(true_scatter, rel=0.15)
        # Bias should be approximately 0
        assert result["teff"].bias == pytest.approx(0.0, abs=10.0)
        assert result["teff"].median_offset == pytest.approx(0.0, abs=10.0)

    def test_evaluate_log_teff(self):
        """Test Teff evaluation in log space."""
        evaluator = Evaluator(parameter_names=["teff", "logg"], teff_in_log=True)

        n_samples = 100
        y_true = np.column_stack(
            [
                np.linspace(4500, 6500, n_samples),
                np.linspace(2.0, 5.0, n_samples),
            ]
        ).astype(np.float32)

        # Add small perturbations
        y_pred = y_true.copy()
        y_pred[:, 0] += 50.0

        result = evaluator.evaluate(y_pred, y_true)

        # Should have log_teff in metrics
        assert "log_teff" in result.metrics
        assert "log_teff" in result.parameter_names

        # log_teff bias should be small positive (since we added to linear scale)
        assert result["log_teff"].bias > 0

    def test_evaluate_with_uncertainties(self):
        """Test evaluation with uncertainty metrics."""
        evaluator = Evaluator(parameter_names=["teff"], teff_in_log=False)

        np.random.seed(42)
        n_samples = 1000

        y_true = np.full((n_samples, 1), 5500.0, dtype=np.float32)
        pred_scatter = np.full((n_samples, 1), 100.0, dtype=np.float32)
        label_errors = np.full((n_samples, 1), 50.0, dtype=np.float32)

        # Generate predictions with scatter matching total uncertainty
        total_std = np.sqrt(100**2 + 50**2 + 0.01**2)
        y_pred = (y_true + np.random.randn(n_samples, 1) * total_std).astype(np.float32)

        result = evaluator.evaluate(y_pred, y_true, pred_scatter, label_errors)

        # Z-score metrics should be available
        assert result["teff"].z_median is not None
        assert result["teff"].z_robust_scatter is not None

        # Z-scores should be approximately standard normal
        assert result["teff"].z_median == pytest.approx(0.0, abs=0.1)
        assert result["teff"].z_robust_scatter == pytest.approx(1.0, rel=0.15)

        # Predicted uncertainty stats
        assert result["teff"].pred_unc_p50 == pytest.approx(100.0, abs=0.1)

    def test_evaluate_shape_mismatch(self):
        """Test that shape mismatch raises error."""
        evaluator = Evaluator(parameter_names=["teff"])

        y_pred = np.random.randn(100, 1).astype(np.float32)
        y_true = np.random.randn(50, 1).astype(np.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            evaluator.evaluate(y_pred, y_true)

    def test_evaluate_wrong_n_params(self):
        """Test that wrong number of parameters raises error."""
        evaluator = Evaluator(parameter_names=["teff", "logg", "feh"])

        y_pred = np.random.randn(100, 2).astype(np.float32)  # Only 2 params
        y_true = np.random.randn(100, 2).astype(np.float32)

        with pytest.raises(ValueError, match="doesn't match"):
            evaluator.evaluate(y_pred, y_true)


class TestEvaluatePredictions:
    """Tests for the convenience function."""

    def test_basic_usage(self):
        """Test basic usage of convenience function."""
        y_pred = np.array([[5500, 4.0, -0.1]], dtype=np.float32)
        y_true = np.array([[5450, 4.1, -0.15]], dtype=np.float32)

        result = evaluate_predictions(y_pred, y_true)

        assert "teff" in result.metrics
        assert "logg" in result.metrics
        assert "feh" in result.metrics

    def test_custom_params(self):
        """Test with custom parameter names."""
        y_pred = np.random.randn(10, 2).astype(np.float32)
        y_true = np.random.randn(10, 2).astype(np.float32)

        result = evaluate_predictions(
            y_pred,
            y_true,
            parameter_names=["param1", "param2"],
        )

        assert "param1" in result.metrics
        assert "param2" in result.metrics


class TestMetricComputations:
    """Tests for specific metric calculations."""

    def test_mae_computation(self):
        """Test MAE is computed correctly."""
        evaluator = Evaluator(parameter_names=["x"], teff_in_log=False)

        y_pred = np.array([[10], [20], [30]], dtype=np.float32)
        y_true = np.array([[12], [18], [33]], dtype=np.float32)

        # Residuals: -2, 2, -3
        # MAE = (2 + 2 + 3) / 3 = 2.333...
        result = evaluator.evaluate(y_pred, y_true)
        assert result["x"].mae == pytest.approx(7 / 3, abs=0.001)

    def test_rmse_computation(self):
        """Test RMSE is computed correctly."""
        evaluator = Evaluator(parameter_names=["x"], teff_in_log=False)

        y_pred = np.array([[10], [20], [30]], dtype=np.float32)
        y_true = np.array([[12], [18], [33]], dtype=np.float32)

        # Residuals: -2, 2, -3
        # RMSE = sqrt((4 + 4 + 9) / 3) = sqrt(17/3)
        result = evaluator.evaluate(y_pred, y_true)
        assert result["x"].rmse == pytest.approx(np.sqrt(17 / 3), abs=0.001)

    def test_bias_computation(self):
        """Test bias is computed correctly."""
        evaluator = Evaluator(parameter_names=["x"], teff_in_log=False)

        # All predictions 10 higher than truth
        y_pred = np.array([[110], [120], [130]], dtype=np.float32)
        y_true = np.array([[100], [110], [120]], dtype=np.float32)

        result = evaluator.evaluate(y_pred, y_true)
        assert result["x"].bias == pytest.approx(10.0, abs=0.001)

    def test_median_offset_computation(self):
        """Test median offset is computed correctly."""
        evaluator = Evaluator(parameter_names=["x"], teff_in_log=False)

        # Residuals: -1, -1, 2, -3, 100 (outlier)
        y_pred = np.array([[10], [20], [32], [37], [150]], dtype=np.float32)
        y_true = np.array([[11], [21], [30], [40], [50]], dtype=np.float32)

        # Sorted residuals: -3, -1, -1, 2, 100
        # Median: -1
        result = evaluator.evaluate(y_pred, y_true)
        assert result["x"].median_offset == pytest.approx(-1.0, abs=0.001)

    def test_robust_scatter_computation(self):
        """Test robust scatter (P84-P16)/2 is computed correctly."""
        evaluator = Evaluator(parameter_names=["x"], teff_in_log=False)

        np.random.seed(42)
        # Generate Gaussian residuals with known sigma
        n = 10000
        sigma = 5.0
        residuals = np.random.randn(n) * sigma
        y_true = np.zeros((n, 1), dtype=np.float32)
        y_pred = (y_true + residuals.reshape(-1, 1)).astype(np.float32)

        result = evaluator.evaluate(y_pred, y_true)
        # For Gaussian, (P84-P16)/2 ≈ sigma
        assert result["x"].robust_scatter == pytest.approx(sigma, rel=0.05)

    def test_sd_computation(self):
        """Test SD is computed correctly."""
        evaluator = Evaluator(parameter_names=["x"], teff_in_log=False)

        y_pred = np.array([[10], [20], [30], [40], [50]], dtype=np.float32)
        y_true = np.array([[11], [21], [28], [43], [55]], dtype=np.float32)

        # Residuals: -1, -1, 2, -3, -5
        # Mean: -1.6
        # SD = sqrt(mean((residuals - mean)^2))
        residuals = np.array([-1, -1, 2, -3, -5])
        expected_sd = np.std(residuals)

        result = evaluator.evaluate(y_pred, y_true)
        assert result["x"].sd == pytest.approx(expected_sd, abs=0.001)


class TestZScoreMetrics:
    """Tests for z-score calibration metrics."""

    def test_zscore_with_perfect_calibration(self):
        """Test z-scores when uncertainties perfectly match observed scatter."""
        evaluator = Evaluator(parameter_names=["x"], teff_in_log=False)

        np.random.seed(42)
        n = 5000
        pred_unc = 10.0

        y_true = np.zeros((n, 1), dtype=np.float32)
        pred_scatter = np.full((n, 1), pred_unc, dtype=np.float32)

        # Generate predictions with scatter exactly matching pred_unc
        y_pred = (y_true + np.random.randn(n, 1) * pred_unc).astype(np.float32)

        result = evaluator.evaluate(y_pred, y_true, pred_scatter)

        # Z-scores should be standard normal
        assert result["x"].z_median == pytest.approx(0.0, abs=0.1)
        assert result["x"].z_robust_scatter == pytest.approx(1.0, rel=0.1)

    def test_zscore_with_underestimated_uncertainty(self):
        """Test z-scores when uncertainties are underestimated."""
        evaluator = Evaluator(parameter_names=["x"], teff_in_log=False)

        np.random.seed(42)
        n = 5000
        true_scatter = 20.0
        reported_unc = 10.0  # Half of true scatter

        y_true = np.zeros((n, 1), dtype=np.float32)
        pred_scatter = np.full((n, 1), reported_unc, dtype=np.float32)
        y_pred = (y_true + np.random.randn(n, 1) * true_scatter).astype(np.float32)

        result = evaluator.evaluate(y_pred, y_true, pred_scatter)

        # Z-scores should be wider than 1 (underconfident)
        assert result["x"].z_robust_scatter > 1.5

    def test_pred_uncertainty_percentiles(self):
        """Test predicted uncertainty percentile statistics."""
        evaluator = Evaluator(parameter_names=["x"], teff_in_log=False)

        n = 1000
        y_true = np.zeros((n, 1), dtype=np.float32)
        y_pred = y_true.copy()

        # Create varying uncertainties
        pred_scatter = np.linspace(5, 15, n).reshape(-1, 1).astype(np.float32)

        result = evaluator.evaluate(y_pred, y_true, pred_scatter)

        # Check percentiles
        assert result["x"].pred_unc_p16 == pytest.approx(
            np.percentile(pred_scatter, 16), rel=0.01
        )
        assert result["x"].pred_unc_p50 == pytest.approx(
            np.percentile(pred_scatter, 50), rel=0.01
        )
        assert result["x"].pred_unc_p84 == pytest.approx(
            np.percentile(pred_scatter, 84), rel=0.01
        )


class TestEvaluatorMasking:
    """Tests for mask parameter in Evaluator."""

    def test_evaluate_without_mask_unchanged(self):
        """Test that evaluation without mask works as before."""
        evaluator = Evaluator(parameter_names=["x", "y"], teff_in_log=False)

        y_pred = np.random.randn(100, 2).astype(np.float32)
        y_true = np.random.randn(100, 2).astype(np.float32)

        result_no_mask = evaluator.evaluate(y_pred, y_true)
        result_none_mask = evaluator.evaluate(y_pred, y_true, mask=None)

        assert result_no_mask["x"].rmse == result_none_mask["x"].rmse
        assert result_no_mask["y"].bias == result_none_mask["y"].bias

    def test_evaluate_with_all_ones_mask(self):
        """Test that all-ones mask gives same result as no mask."""
        evaluator = Evaluator(parameter_names=["x", "y"], teff_in_log=False)

        y_pred = np.random.randn(100, 2).astype(np.float32)
        y_true = np.random.randn(100, 2).astype(np.float32)
        mask = np.ones((100, 2), dtype=np.float32)

        result_no_mask = evaluator.evaluate(y_pred, y_true)
        result_with_mask = evaluator.evaluate(y_pred, y_true, mask=mask)

        assert result_no_mask["x"].rmse == pytest.approx(result_with_mask["x"].rmse)
        assert result_no_mask["y"].mae == pytest.approx(result_with_mask["y"].mae)

    def test_mask_excludes_samples(self):
        """Test that masked samples don't contribute to metrics."""
        evaluator = Evaluator(parameter_names=["x"], teff_in_log=False)

        # Create predictions with a big outlier in first half
        y_true = np.zeros((100, 1), dtype=np.float32)
        y_pred = np.zeros((100, 1), dtype=np.float32)
        y_pred[:50] = 100.0  # First 50 samples are way off

        # Mask out the outliers
        mask = np.ones((100, 1), dtype=np.float32)
        mask[:50] = 0

        result = evaluator.evaluate(y_pred, y_true, mask=mask)

        # Without outliers, RMSE should be ~0
        assert result["x"].rmse == pytest.approx(0.0, abs=0.001)
        assert result["x"].n_samples == 50

    def test_mask_per_parameter(self):
        """Test that masking works independently per parameter."""
        evaluator = Evaluator(parameter_names=["x", "y"], teff_in_log=False)

        y_true = np.zeros((100, 2), dtype=np.float32)
        y_pred = np.zeros((100, 2), dtype=np.float32)

        # Add bias to first 50 samples of parameter x
        y_pred[:50, 0] = 10.0

        # Mask out those samples for x, but keep all for y
        mask = np.ones((100, 2), dtype=np.float32)
        mask[:50, 0] = 0

        result = evaluator.evaluate(y_pred, y_true, mask=mask)

        # x should have n_samples=50 and bias~0
        assert result["x"].n_samples == 50
        assert result["x"].bias == pytest.approx(0.0, abs=0.001)

        # y should have n_samples=100 and bias=0
        assert result["y"].n_samples == 100
        assert result["y"].bias == pytest.approx(0.0, abs=0.001)

    def test_all_masked_returns_nan(self):
        """Test that fully masked parameter returns NaN metrics."""
        evaluator = Evaluator(parameter_names=["x", "y"], teff_in_log=False)

        y_true = np.random.randn(100, 2).astype(np.float32)
        y_pred = np.random.randn(100, 2).astype(np.float32)

        # Mask out all samples for x
        mask = np.ones((100, 2), dtype=np.float32)
        mask[:, 0] = 0

        result = evaluator.evaluate(y_pred, y_true, mask=mask)

        # x should have NaN metrics
        assert result["x"].n_samples == 0
        assert np.isnan(result["x"].rmse)
        assert np.isnan(result["x"].bias)
        assert np.isnan(result["x"].mae)

        # y should have valid metrics
        assert result["y"].n_samples == 100
        assert not np.isnan(result["y"].rmse)

    def test_mask_with_uncertainties(self):
        """Test masking with uncertainty metrics."""
        evaluator = Evaluator(parameter_names=["x"], teff_in_log=False)

        np.random.seed(42)
        n = 200
        y_true = np.zeros((n, 1), dtype=np.float32)
        y_pred = np.random.randn(n, 1).astype(np.float32) * 10
        pred_scatter = np.full((n, 1), 10.0, dtype=np.float32)

        # Mask first half
        mask = np.ones((n, 1), dtype=np.float32)
        mask[:100] = 0

        result = evaluator.evaluate(y_pred, y_true, pred_scatter, mask=mask)

        assert result["x"].n_samples == 100
        assert result["x"].z_median is not None
        assert result["x"].z_robust_scatter is not None

    def test_mask_shape_mismatch_raises(self):
        """Test that mask shape mismatch raises error."""
        evaluator = Evaluator(parameter_names=["x", "y"], teff_in_log=False)

        y_pred = np.random.randn(100, 2).astype(np.float32)
        y_true = np.random.randn(100, 2).astype(np.float32)
        mask = np.ones((50, 2), dtype=np.float32)  # Wrong shape

        with pytest.raises(ValueError, match="mask shape"):
            evaluator.evaluate(y_pred, y_true, mask=mask)

    def test_evaluate_predictions_with_mask(self):
        """Test convenience function with mask parameter."""
        y_pred = np.array([[5500, 4.0, -0.1], [5600, 4.2, 0.0]], dtype=np.float32)
        y_true = np.array([[5450, 4.1, -0.15], [5650, 4.1, 0.1]], dtype=np.float32)
        mask = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.float32)

        result = evaluate_predictions(y_pred, y_true, mask=mask)

        # teff should use both samples
        assert result["teff"].n_samples == 2
        # logg should use only second sample
        assert result["logg"].n_samples == 1
        # feh should use only first sample
        assert result["feh"].n_samples == 1
