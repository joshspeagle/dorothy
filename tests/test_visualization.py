"""
Tests for the visualization module (training_plots.py).

These tests verify that plotting functions:
- Handle missing data gracefully (return None)
- Generate expected output files when data is available
- Handle matplotlib import failures gracefully
"""

from unittest.mock import patch

import numpy as np
import pytest

from dorothy.training.trainer import TrainingHistory
from dorothy.visualization.training_plots import (
    PARAM_DISPLAY_NAMES,
    _simplify_layer_name,
    _try_import_matplotlib,
    generate_training_report,
    plot_grokking_metrics,
    plot_loss_components,
    plot_loss_curves,
    plot_metrics_evolution,
    plot_per_survey_loss_curves,
    plot_per_survey_metrics,
    plot_per_survey_metrics_evolution,
    plot_val_metrics,
    plot_zscore_calibration,
)


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestSimplifyLayerName:
    """Tests for _simplify_layer_name helper function."""

    def test_simple_layer_weight(self):
        """Test simplification of simple layer weight names."""
        assert _simplify_layer_name("layers.1.weight") == "L1"
        assert _simplify_layer_name("layers.4.weight") == "L4"
        assert _simplify_layer_name("layers.10.weight") == "L10"

    def test_simple_layer_bias(self):
        """Test simplification of simple layer bias names."""
        assert _simplify_layer_name("layers.1.bias") == "L1.b"
        assert _simplify_layer_name("layers.3.bias") == "L3.b"

    def test_encoder_layer(self):
        """Test simplification of encoder layer names."""
        result = _simplify_layer_name("encoders.boss.layers.0.weight")
        assert "boss" in result or "L0" in result

    def test_trunk_layer(self):
        """Test simplification of trunk layer names."""
        result = _simplify_layer_name("trunk.layers.2.weight")
        assert "trunk" in result or "L2" in result

    def test_fallback_for_unknown_pattern(self):
        """Test fallback behavior for unknown patterns."""
        # Should return last two parts
        result = _simplify_layer_name("some.unknown.pattern")
        assert "unknown" in result or "pattern" in result

    def test_single_part_name(self):
        """Test handling of single-part names."""
        assert _simplify_layer_name("weights") == "weights"


class TestTryImportMatplotlib:
    """Tests for _try_import_matplotlib function."""

    def test_matplotlib_available(self):
        """Test when matplotlib is available."""
        plt = _try_import_matplotlib()
        # If matplotlib is installed, we get a module
        # If not installed, we get None
        assert plt is None or hasattr(plt, "subplots")

    def test_matplotlib_import_error(self):
        """Test handling of matplotlib import failure."""
        with (
            patch.dict("sys.modules", {"matplotlib": None}),
            patch("builtins.__import__", side_effect=ImportError),
        ):
            result = _try_import_matplotlib()
            # Should return None when import fails
            assert result is None


# =============================================================================
# Fixtures for TrainingHistory
# =============================================================================


@pytest.fixture
def empty_history():
    """Create an empty TrainingHistory for testing edge cases."""
    return TrainingHistory(parameter_names=["teff", "logg", "fe_h"])


@pytest.fixture
def basic_history():
    """Create a basic TrainingHistory with loss data."""
    history = TrainingHistory(parameter_names=["teff", "logg", "fe_h"])
    history.train_losses = [1.0, 0.8, 0.6, 0.5, 0.4]
    history.val_losses = [1.1, 0.85, 0.65, 0.55, 0.45]
    history.best_val_loss = 0.45
    history.best_epoch = 4
    history.learning_rates = [0.001, 0.0008, 0.0006, 0.0004, 0.0002]
    history.grad_norms = [10.0, 8.0, 5.0, 3.0, 2.0]
    return history


@pytest.fixture
def full_history():
    """Create a comprehensive TrainingHistory with all metrics."""
    history = TrainingHistory(
        parameter_names=["teff", "logg", "fe_h"],
        val_loss_breakdown={"mean_component": [], "scatter_component": []},
        val_metrics={
            "rmse": [],
            "bias": [],
            "sd": [],
            "mae": [],
            "robust_scatter": [],
            "z_median": [],
            "z_robust_scatter": [],
        },
    )

    n_epochs = 5
    n_params = 3

    history.train_losses = [1.0, 0.8, 0.6, 0.5, 0.4]
    history.val_losses = [1.1, 0.85, 0.65, 0.55, 0.45]
    history.best_val_loss = 0.45
    history.best_epoch = 4
    history.learning_rates = [0.001, 0.0008, 0.0006, 0.0004, 0.0002]
    history.grad_norms = [10.0, 8.0, 5.0, 3.0, 2.0]

    # Add loss breakdown per epoch
    for _ in range(n_epochs):
        history.val_loss_breakdown["mean_component"].append(
            np.random.rand(n_params).astype(np.float32)
        )
        history.val_loss_breakdown["scatter_component"].append(
            np.random.rand(n_params).astype(np.float32)
        )

    # Add validation metrics per epoch
    for _ in range(n_epochs):
        history.val_metrics["rmse"].append(
            np.random.rand(n_params).astype(np.float32) * 0.5
        )
        history.val_metrics["bias"].append(
            (np.random.rand(n_params).astype(np.float32) - 0.5) * 0.2
        )
        history.val_metrics["sd"].append(
            np.random.rand(n_params).astype(np.float32) * 0.3
        )
        history.val_metrics["mae"].append(
            np.random.rand(n_params).astype(np.float32) * 0.4
        )
        history.val_metrics["robust_scatter"].append(
            np.random.rand(n_params).astype(np.float32) * 0.3
        )
        history.val_metrics["z_median"].append(
            (np.random.rand(n_params).astype(np.float32) - 0.5) * 0.5
        )
        history.val_metrics["z_robust_scatter"].append(
            np.random.rand(n_params).astype(np.float32) * 0.3 + 0.85
        )

    # Add weight tracking
    history.weight_norms = {
        "layers.0.weight": [1.0, 1.1, 1.2, 1.3, 1.4],
        "layers.1.weight": [0.8, 0.85, 0.9, 0.95, 1.0],
    }
    history.weight_updates = {
        "layers.0.weight": [0.1, 0.08, 0.05, 0.03, 0.02],
        "layers.1.weight": [0.05, 0.04, 0.03, 0.02, 0.01],
    }

    return history


@pytest.fixture
def multi_survey_history(full_history):
    """Create a TrainingHistory with per-survey metrics."""
    history = full_history
    history.survey_names = ["boss", "desi"]
    history.per_survey_val_losses = {
        "boss": [1.2, 0.9, 0.7, 0.6, 0.5],
        "desi": [1.0, 0.8, 0.65, 0.55, 0.48],
    }
    history.per_survey_val_metrics = {
        "boss": {
            "rmse": [np.array([0.3, 0.25, 0.2]) for _ in range(5)],
            "bias": [np.array([0.05, -0.02, 0.01]) for _ in range(5)],
            "robust_scatter": [np.array([0.2, 0.18, 0.15]) for _ in range(5)],
        },
        "desi": {
            "rmse": [np.array([0.28, 0.22, 0.18]) for _ in range(5)],
            "bias": [np.array([0.03, -0.01, 0.02]) for _ in range(5)],
            "robust_scatter": [np.array([0.18, 0.16, 0.14]) for _ in range(5)],
        },
    }
    return history


# =============================================================================
# Plot Function Tests - Empty/Missing Data
# =============================================================================


class TestPlotFunctionsEmptyData:
    """Test that plot functions handle empty/missing data gracefully."""

    def test_plot_loss_curves_empty_history(self, empty_history, tmp_path):
        """Test plot_loss_curves returns None with empty history."""
        result = plot_loss_curves(empty_history, tmp_path)
        assert result is None

    def test_plot_loss_components_no_breakdown(self, basic_history, tmp_path):
        """Test plot_loss_components returns None without breakdown data."""
        result = plot_loss_components(basic_history, tmp_path)
        assert result is None

    def test_plot_grokking_metrics_no_grad_norms(self, empty_history, tmp_path):
        """Test plot_grokking_metrics returns None without grad_norms."""
        result = plot_grokking_metrics(empty_history, tmp_path)
        assert result is None

    def test_plot_val_metrics_no_metrics(self, basic_history, tmp_path):
        """Test plot_val_metrics returns None without val_metrics."""
        basic_history.val_metrics = {}
        result = plot_val_metrics(basic_history, tmp_path)
        assert result is None

    def test_plot_zscore_calibration_no_zscores(self, basic_history, tmp_path):
        """Test plot_zscore_calibration returns None without z-scores."""
        basic_history.val_metrics = {"rmse": [np.array([0.1, 0.2, 0.3])]}
        result = plot_zscore_calibration(basic_history, tmp_path)
        assert result is None

    def test_plot_metrics_evolution_insufficient_epochs(self, tmp_path):
        """Test plot_metrics_evolution returns None with < 2 epochs."""
        history = TrainingHistory(parameter_names=["teff"])
        history.val_metrics = {"rmse": [np.array([0.1])]}  # Only 1 epoch
        result = plot_metrics_evolution(history, tmp_path)
        assert result is None


class TestPerSurveyPlotsMissingData:
    """Test per-survey plots with missing data."""

    def test_per_survey_loss_curves_no_survey_names(self, basic_history, tmp_path):
        """Test returns None without survey_names."""
        result = plot_per_survey_loss_curves(basic_history, tmp_path)
        assert result is None

    def test_per_survey_loss_curves_no_losses(self, tmp_path):
        """Test returns None without per_survey_val_losses."""
        history = TrainingHistory(parameter_names=["teff"])
        history.survey_names = ["boss"]
        history.per_survey_val_losses = {}
        result = plot_per_survey_loss_curves(history, tmp_path)
        assert result is None

    def test_per_survey_metrics_no_metrics(self, tmp_path):
        """Test returns None without per_survey_val_metrics."""
        history = TrainingHistory(parameter_names=["teff"])
        history.survey_names = ["boss"]
        history.per_survey_val_metrics = {}
        result = plot_per_survey_metrics(history, tmp_path)
        assert result is None


# =============================================================================
# Plot Function Tests - With Data (Integration)
# =============================================================================


@pytest.mark.skipif(_try_import_matplotlib() is None, reason="matplotlib not available")
class TestPlotFunctionsWithData:
    """Test that plot functions generate files when data is available."""

    def test_plot_loss_curves_generates_file(self, basic_history, tmp_path):
        """Test plot_loss_curves generates a file."""
        result = plot_loss_curves(basic_history, tmp_path)
        assert result is not None
        assert result.exists()
        assert result.name == "loss_curves.png"

    def test_plot_loss_components_generates_file(self, full_history, tmp_path):
        """Test plot_loss_components generates a file."""
        result = plot_loss_components(full_history, tmp_path)
        assert result is not None
        assert result.exists()
        assert result.name == "loss_components.png"

    def test_plot_grokking_metrics_generates_file(self, basic_history, tmp_path):
        """Test plot_grokking_metrics generates a file."""
        result = plot_grokking_metrics(basic_history, tmp_path)
        assert result is not None
        assert result.exists()
        assert result.name == "grokking_metrics.png"

    def test_plot_grokking_with_weight_norms(self, full_history, tmp_path):
        """Test plot_grokking_metrics with weight tracking."""
        result = plot_grokking_metrics(full_history, tmp_path)
        assert result is not None
        assert result.exists()

    def test_plot_val_metrics_generates_file(self, full_history, tmp_path):
        """Test plot_val_metrics generates a file."""
        result = plot_val_metrics(full_history, tmp_path)
        assert result is not None
        assert result.exists()
        assert result.name == "val_metrics.png"

    def test_plot_zscore_calibration_generates_file(self, full_history, tmp_path):
        """Test plot_zscore_calibration generates a file."""
        result = plot_zscore_calibration(full_history, tmp_path)
        assert result is not None
        assert result.exists()
        assert result.name == "zscore_calibration.png"

    def test_plot_metrics_evolution_generates_file(self, full_history, tmp_path):
        """Test plot_metrics_evolution generates a file."""
        result = plot_metrics_evolution(full_history, tmp_path)
        assert result is not None
        assert result.exists()
        assert result.name == "metrics_evolution.png"


@pytest.mark.skipif(_try_import_matplotlib() is None, reason="matplotlib not available")
class TestPerSurveyPlotsWithData:
    """Test per-survey plot functions with actual data."""

    def test_per_survey_loss_curves_generates_file(
        self, multi_survey_history, tmp_path
    ):
        """Test plot_per_survey_loss_curves generates a file."""
        result = plot_per_survey_loss_curves(multi_survey_history, tmp_path)
        assert result is not None
        assert result.exists()
        assert result.name == "per_survey_loss_curves.png"

    def test_per_survey_metrics_generates_file(self, multi_survey_history, tmp_path):
        """Test plot_per_survey_metrics generates a file."""
        result = plot_per_survey_metrics(multi_survey_history, tmp_path)
        assert result is not None
        assert result.exists()
        assert result.name == "per_survey_metrics.png"

    def test_per_survey_metrics_evolution_generates_file(
        self, multi_survey_history, tmp_path
    ):
        """Test plot_per_survey_metrics_evolution generates a file."""
        result = plot_per_survey_metrics_evolution(multi_survey_history, tmp_path)
        assert result is not None
        assert result.exists()
        assert result.name == "per_survey_metrics_evolution.png"


# =============================================================================
# Report Generation Tests
# =============================================================================


@pytest.mark.skipif(_try_import_matplotlib() is None, reason="matplotlib not available")
class TestGenerateTrainingReport:
    """Tests for generate_training_report function."""

    def test_generates_multiple_plots(self, full_history, tmp_path):
        """Test that report generates multiple plot files."""
        plots = generate_training_report(full_history, tmp_path, "test_experiment")
        assert len(plots) > 0
        assert all(p.exists() for p in plots)

    def test_creates_output_directory(self, full_history, tmp_path):
        """Test that report creates output directory if needed."""
        output_dir = tmp_path / "new_dir" / "plots"
        generate_training_report(full_history, output_dir, "test_experiment")
        assert output_dir.exists()

    def test_report_with_multi_survey_data(self, multi_survey_history, tmp_path):
        """Test report includes per-survey plots when data available."""
        plots = generate_training_report(
            multi_survey_history, tmp_path, "multi_survey_test"
        )
        plot_names = [p.name for p in plots]

        # Should include per-survey plots
        assert "per_survey_loss_curves.png" in plot_names
        assert "per_survey_metrics.png" in plot_names

    def test_report_handles_plot_errors(self, basic_history, tmp_path, capsys):
        """Test that report continues even if individual plots fail."""
        # Add incomplete data that might cause some plots to fail
        basic_history.val_metrics = {"rmse": "invalid_data"}  # Invalid data type

        # Should not raise exception
        plots = generate_training_report(basic_history, tmp_path)
        # Should still generate some plots
        assert isinstance(plots, list)

    def test_report_with_empty_history(self, empty_history, tmp_path):
        """Test report handles empty history gracefully."""
        plots = generate_training_report(empty_history, tmp_path)
        # Should return empty list, not raise exception
        assert isinstance(plots, list)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_param_display_names_count(self):
        """Test that PARAM_DISPLAY_NAMES has expected count."""
        assert len(PARAM_DISPLAY_NAMES) == 11

    def test_param_display_names_format(self):
        """Test that PARAM_DISPLAY_NAMES uses LaTeX formatting."""
        assert r"$T_{\rm eff}$" in PARAM_DISPLAY_NAMES
        assert r"$\log g$" in PARAM_DISPLAY_NAMES
        assert "[Fe/H]" in PARAM_DISPLAY_NAMES


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.skipif(
        _try_import_matplotlib() is None, reason="matplotlib not available"
    )
    def test_single_epoch_history(self, tmp_path):
        """Test handling of single-epoch training history."""
        history = TrainingHistory(parameter_names=["teff", "logg", "fe_h"])
        history.train_losses = [0.5]
        history.val_losses = [0.6]
        history.best_val_loss = 0.6
        history.best_epoch = 0
        history.learning_rates = [0.001]
        history.grad_norms = [5.0]

        # Should generate loss curves for single epoch
        result = plot_loss_curves(history, tmp_path)
        assert result is not None

    @pytest.mark.skipif(
        _try_import_matplotlib() is None, reason="matplotlib not available"
    )
    def test_many_parameters(self, tmp_path):
        """Test handling of many parameters (more than 6)."""
        params = [f"param_{i}" for i in range(11)]
        history = TrainingHistory(
            parameter_names=params,
            val_metrics={"rmse": [], "bias": [], "robust_scatter": [], "mae": []},
        )
        history.train_losses = [1.0, 0.5]
        history.val_losses = [1.1, 0.6]
        history.best_val_loss = 0.6
        history.best_epoch = 1

        # Add metrics for all 11 parameters
        for _ in range(2):
            history.val_metrics["rmse"].append(np.random.rand(11).astype(np.float32))
            history.val_metrics["bias"].append(np.random.rand(11).astype(np.float32))
            history.val_metrics["robust_scatter"].append(
                np.random.rand(11).astype(np.float32)
            )
            history.val_metrics["mae"].append(np.random.rand(11).astype(np.float32))

        # Should handle > 6 parameters (shows subset)
        result = plot_val_metrics(history, tmp_path)
        assert result is not None

    def test_none_best_epoch(self, tmp_path):
        """Test handling when best_epoch is None."""
        history = TrainingHistory(parameter_names=["teff"])
        history.train_losses = [1.0, 0.5]
        history.val_losses = [1.1, 0.6]
        history.best_epoch = None  # Not set
        history.best_val_loss = float("inf")

        plt = _try_import_matplotlib()
        if plt is not None:
            # Should not crash when best_epoch is None
            result = plot_loss_curves(history, tmp_path)
            assert result is not None
