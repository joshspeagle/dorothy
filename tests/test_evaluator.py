"""
Tests for the Evaluator class.

These tests verify:
1. Metric computation (MAE, RMSE, bias, scatter, MAD)
2. Per-parameter evaluation
3. Special handling for Teff in log space
4. Summary output formatting
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
            mae=50.0,
            rmse=75.0,
            bias=-10.0,
            scatter=70.0,
            mad=40.0,
            n_samples=100,
        )

        d = metrics.to_dict()
        assert d["name"] == "teff"
        assert d["mae"] == 50.0
        assert d["rmse"] == 75.0
        assert d["bias"] == -10.0
        assert d["scatter"] == 70.0
        assert d["mad"] == 40.0
        assert d["n_samples"] == 100


class TestEvaluationResult:
    """Tests for EvaluationResult container."""

    def test_getitem(self):
        """Test dictionary-style access."""
        metrics = ParameterMetrics(
            name="logg",
            mae=0.1,
            rmse=0.15,
            bias=0.02,
            scatter=0.14,
            mad=0.08,
            n_samples=50,
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
            mae=50.0,
            rmse=75.0,
            bias=-10.0,
            scatter=70.0,
            mad=40.0,
            n_samples=100,
        )
        result = EvaluationResult(
            metrics={"teff": metrics},
            parameter_names=["teff"],
        )

        summary = result.summary(format="text")
        assert "teff" in summary
        assert "50.0000" in summary
        assert "100" in summary

    def test_summary_markdown(self):
        """Test markdown summary generation."""
        metrics = ParameterMetrics(
            name="feh",
            mae=0.05,
            rmse=0.08,
            bias=0.01,
            scatter=0.07,
            mad=0.04,
            n_samples=200,
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
            mae=0.1,
            rmse=0.15,
            bias=0.02,
            scatter=0.14,
            mad=0.08,
            n_samples=50,
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

        # Scatter should be approximately the true scatter
        assert result["teff"].scatter == pytest.approx(true_scatter, rel=0.1)
        # Bias should be approximately 0
        assert result["teff"].bias == pytest.approx(0.0, abs=10.0)

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

    def test_mad_computation(self):
        """Test MAD is computed correctly."""
        evaluator = Evaluator(parameter_names=["x"], teff_in_log=False)

        y_pred = np.array([[10], [20], [30], [40], [50]], dtype=np.float32)
        y_true = np.array([[11], [21], [28], [43], [55]], dtype=np.float32)

        # Residuals: -1, -1, 2, -3, -5
        # Abs residuals: 1, 1, 2, 3, 5
        # Median: 2
        result = evaluator.evaluate(y_pred, y_true)
        assert result["x"].mad == pytest.approx(2.0, abs=0.001)
