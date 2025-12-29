"""
Tests for the saliency analysis module.

These tests verify:
1. SaliencyResult dataclass
2. SaliencyAnalyzer Jacobian computation
3. Fisher importance aggregation
4. Visualization functions
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from dorothy.analysis.saliency import (
    AblationSaliencyAnalyzer,
    AblationSaliencyResult,
    SaliencyAnalyzer,
    SaliencyResult,
    _find_masked_regions,
    plot_ablation_parameter_saliency,
    plot_ablation_saliency_heatmap,
    plot_parameter_saliency,
    plot_saliency_heatmap,
)


class SimpleLinearModel(nn.Module):
    """A simple linear model for testing with known Jacobian."""

    def __init__(self, n_wavelengths: int = 100, n_params: int = 11):
        super().__init__()
        self.n_params = n_params
        # Linear layer from flattened 3*n_wavelengths to 2*n_params
        self.linear = nn.Linear(3 * n_wavelengths, 2 * n_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, 3, N) -> (batch, 2, n_params)."""
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # (batch, 3*N)
        out = self.linear(x_flat)  # (batch, 2*n_params)
        return out.view(batch_size, 2, self.n_params)


class TestSaliencyResult:
    """Tests for the SaliencyResult dataclass."""

    def test_basic_creation(self):
        """Test that SaliencyResult can be created."""
        n_wavelengths = 100
        n_params = 11

        result = SaliencyResult(
            survey="boss",
            wavelength=np.linspace(3600, 9000, n_wavelengths).astype(np.float32),
            spectrum=np.random.randn(n_wavelengths).astype(np.float32),
            spectrum_error=np.abs(np.random.randn(n_wavelengths)).astype(np.float32),
            mask=np.ones(n_wavelengths, dtype=bool),
            jacobian_mean_flux=np.random.randn(n_params, n_wavelengths).astype(
                np.float32
            ),
            jacobian_mean_error=np.random.randn(n_params, n_wavelengths).astype(
                np.float32
            ),
            jacobian_lns_flux=np.random.randn(n_params, n_wavelengths).astype(
                np.float32
            ),
            jacobian_lns_error=np.random.randn(n_params, n_wavelengths).astype(
                np.float32
            ),
            predictions=np.random.randn(n_params).astype(np.float32),
            uncertainties=np.abs(np.random.randn(n_params)).astype(np.float32) + 0.01,
            fisher_importance_per_param=np.random.randn(n_params, n_wavelengths).astype(
                np.float32
            ),
            fisher_importance=np.random.randn(n_wavelengths).astype(np.float32),
            parameter_names=[
                "teff",
                "logg",
                "fe_h",
                "mg_fe",
                "c_fe",
                "si_fe",
                "ni_fe",
                "al_fe",
                "ca_fe",
                "n_fe",
                "mn_fe",
            ],
        )

        assert result.survey == "boss"
        assert len(result.wavelength) == n_wavelengths
        assert result.jacobian_mean_flux.shape == (n_params, n_wavelengths)

    def test_get_param_index(self):
        """Test parameter index lookup."""
        n = 100
        result = SaliencyResult(
            survey="boss",
            wavelength=np.zeros(n, dtype=np.float32),
            spectrum=np.zeros(n, dtype=np.float32),
            spectrum_error=np.zeros(n, dtype=np.float32),
            mask=np.ones(n, dtype=bool),
            jacobian_mean_flux=np.zeros((11, n), dtype=np.float32),
            jacobian_mean_error=np.zeros((11, n), dtype=np.float32),
            jacobian_lns_flux=np.zeros((11, n), dtype=np.float32),
            jacobian_lns_error=np.zeros((11, n), dtype=np.float32),
            predictions=np.zeros(11, dtype=np.float32),
            uncertainties=np.ones(11, dtype=np.float32),
            fisher_importance_per_param=np.zeros((11, n), dtype=np.float32),
            fisher_importance=np.zeros(n, dtype=np.float32),
            parameter_names=[
                "teff",
                "logg",
                "fe_h",
                "mg_fe",
                "c_fe",
                "si_fe",
                "ni_fe",
                "al_fe",
                "ca_fe",
                "n_fe",
                "mn_fe",
            ],
        )

        assert result.get_param_index("teff") == 0
        assert result.get_param_index("fe_h") == 2
        assert result.get_param_index("mn_fe") == 10

    def test_get_param_index_unknown(self):
        """Test error for unknown parameter."""
        n = 100
        result = SaliencyResult(
            survey="boss",
            wavelength=np.zeros(n, dtype=np.float32),
            spectrum=np.zeros(n, dtype=np.float32),
            spectrum_error=np.zeros(n, dtype=np.float32),
            mask=np.ones(n, dtype=bool),
            jacobian_mean_flux=np.zeros((11, n), dtype=np.float32),
            jacobian_mean_error=np.zeros((11, n), dtype=np.float32),
            jacobian_lns_flux=np.zeros((11, n), dtype=np.float32),
            jacobian_lns_error=np.zeros((11, n), dtype=np.float32),
            predictions=np.zeros(11, dtype=np.float32),
            uncertainties=np.ones(11, dtype=np.float32),
            fisher_importance_per_param=np.zeros((11, n), dtype=np.float32),
            fisher_importance=np.zeros(n, dtype=np.float32),
            parameter_names=["teff", "logg", "fe_h"] + [f"p{i}" for i in range(8)],
        )

        with pytest.raises(ValueError, match="Unknown parameter"):
            result.get_param_index("unknown_param")


class TestSaliencyAnalyzer:
    """Tests for SaliencyAnalyzer."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple linear model for testing."""
        return SimpleLinearModel(n_wavelengths=100, n_params=11)

    @pytest.fixture
    def analyzer(self, simple_model):
        """Create an analyzer with the simple model."""
        return SaliencyAnalyzer(simple_model, device="cpu")

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.device == torch.device("cpu")
        assert len(analyzer.parameter_names) == 11
        assert analyzer.scatter_floor == 0.01

    def test_initialization_custom_params(self, simple_model):
        """Test analyzer with custom parameters."""
        analyzer = SaliencyAnalyzer(
            simple_model,
            device="cpu",
            scatter_floor=0.05,
            parameter_names=["a", "b", "c"],
        )
        assert analyzer.scatter_floor == 0.05
        assert analyzer.parameter_names == ["a", "b", "c"]

    def test_compute_saliency_shape(self, analyzer):
        """Test that compute_saliency returns correct shapes."""
        n_wavelengths = 100
        X = torch.randn(3, n_wavelengths)
        wavelength = np.linspace(3600, 9000, n_wavelengths).astype(np.float32)

        result = analyzer.compute_saliency(X, wavelength, survey="test")

        assert result.jacobian_mean_flux.shape == (11, n_wavelengths)
        assert result.jacobian_mean_error.shape == (11, n_wavelengths)
        assert result.jacobian_lns_flux.shape == (11, n_wavelengths)
        assert result.jacobian_lns_error.shape == (11, n_wavelengths)
        assert result.predictions.shape == (11,)
        assert result.uncertainties.shape == (11,)
        assert result.fisher_importance_per_param.shape == (11, n_wavelengths)
        assert result.fisher_importance.shape == (n_wavelengths,)

    def test_jacobian_nonzero_for_linear_model(self, analyzer):
        """Test that Jacobian is non-zero for a linear model."""
        n_wavelengths = 100
        X = torch.randn(3, n_wavelengths)
        wavelength = np.linspace(3600, 9000, n_wavelengths).astype(np.float32)

        result = analyzer.compute_saliency(X, wavelength, survey="test")

        # For a linear model, Jacobian should be non-zero
        assert np.abs(result.jacobian_mean_flux).sum() > 0
        assert np.abs(result.jacobian_mean_error).sum() > 0

    def test_fisher_importance_nonnegative(self, analyzer):
        """Test that Fisher importance is non-negative."""
        n_wavelengths = 100
        X = torch.randn(3, n_wavelengths)
        wavelength = np.linspace(3600, 9000, n_wavelengths).astype(np.float32)

        result = analyzer.compute_saliency(X, wavelength, survey="test")

        # Fisher importance should be non-negative (sum of squares)
        assert np.all(result.fisher_importance >= 0)
        assert np.all(result.fisher_importance_per_param >= 0)

    def test_mask_extraction(self, analyzer):
        """Test that mask is correctly extracted."""
        n_wavelengths = 100
        X = torch.zeros(3, n_wavelengths)
        X[2, :50] = 1.0  # First half valid
        X[2, 50:] = 0.0  # Second half masked
        wavelength = np.linspace(3600, 9000, n_wavelengths).astype(np.float32)

        result = analyzer.compute_saliency(X, wavelength, survey="test")

        assert result.mask[:50].all()
        assert not result.mask[50:].any()


class TestFisherImportance:
    """Tests for Fisher importance computation."""

    def test_fisher_aggregation(self):
        """Test Fisher importance aggregation formula."""
        n_params = 3
        n_wavelengths = 10

        # Create analyzer with simple model
        model = SimpleLinearModel(n_wavelengths=n_wavelengths, n_params=n_params)
        analyzer = SaliencyAnalyzer(
            model, device="cpu", parameter_names=["a", "b", "c"]
        )

        # Manually test the Fisher computation
        jac_mean_flux = torch.randn(n_params, n_wavelengths)
        jac_mean_error = torch.randn(n_params, n_wavelengths)
        jac_lns_flux = torch.randn(n_params, n_wavelengths)
        jac_lns_error = torch.randn(n_params, n_wavelengths)
        sigma = torch.ones(n_params) * 0.5

        per_param, total = analyzer._compute_fisher_importance(
            jac_mean_flux, jac_mean_error, jac_lns_flux, jac_lns_error, sigma
        )

        # Check shapes
        assert per_param.shape == (n_params, n_wavelengths)
        assert total.shape == (n_wavelengths,)

        # Check that total is sum of per_param
        np.testing.assert_allclose(
            total.numpy(), per_param.sum(dim=0).numpy(), rtol=1e-5
        )

        # Check non-negativity
        assert (per_param >= 0).all()
        assert (total >= 0).all()


class TestMaskedRegions:
    """Tests for the _find_masked_regions helper function."""

    def test_no_masked_regions(self):
        """Test with all valid data."""
        mask = np.ones(100, dtype=bool)
        regions = _find_masked_regions(mask)
        assert regions == []

    def test_single_masked_region(self):
        """Test with a single masked region."""
        mask = np.ones(100, dtype=bool)
        mask[30:50] = False
        regions = _find_masked_regions(mask)
        assert regions == [(30, 50)]

    def test_multiple_masked_regions(self):
        """Test with multiple masked regions."""
        mask = np.ones(100, dtype=bool)
        mask[10:20] = False
        mask[50:60] = False
        mask[80:90] = False
        regions = _find_masked_regions(mask)
        assert regions == [(10, 20), (50, 60), (80, 90)]

    def test_masked_at_start(self):
        """Test with masked region at the start."""
        mask = np.ones(100, dtype=bool)
        mask[:20] = False
        regions = _find_masked_regions(mask)
        assert regions == [(0, 20)]

    def test_masked_at_end(self):
        """Test with masked region at the end."""
        mask = np.ones(100, dtype=bool)
        mask[80:] = False
        regions = _find_masked_regions(mask)
        assert regions == [(80, 100)]

    def test_all_masked(self):
        """Test with all data masked."""
        mask = np.zeros(100, dtype=bool)
        regions = _find_masked_regions(mask)
        assert regions == [(0, 100)]


class TestVisualization:
    """Tests for visualization functions."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample SaliencyResult for testing."""
        n_wavelengths = 100
        n_params = 11

        return SaliencyResult(
            survey="boss",
            wavelength=np.linspace(3600, 9000, n_wavelengths).astype(np.float32),
            spectrum=np.random.randn(n_wavelengths).astype(np.float32) + 1.0,
            spectrum_error=np.abs(np.random.randn(n_wavelengths)).astype(np.float32)
            * 0.1
            + 0.01,
            mask=np.ones(n_wavelengths, dtype=bool),
            jacobian_mean_flux=np.random.randn(n_params, n_wavelengths).astype(
                np.float32
            ),
            jacobian_mean_error=np.random.randn(n_params, n_wavelengths).astype(
                np.float32
            )
            * 0.1,
            jacobian_lns_flux=np.random.randn(n_params, n_wavelengths).astype(
                np.float32
            )
            * 0.5,
            jacobian_lns_error=np.random.randn(n_params, n_wavelengths).astype(
                np.float32
            )
            * 0.1,
            predictions=np.array(
                [5000, 4.0, -0.5, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05],
                dtype=np.float32,
            ),
            uncertainties=np.array(
                [50, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                dtype=np.float32,
            ),
            fisher_importance_per_param=np.abs(
                np.random.randn(n_params, n_wavelengths)
            ).astype(np.float32),
            fisher_importance=np.abs(np.random.randn(n_wavelengths)).astype(np.float32),
            parameter_names=[
                "teff",
                "logg",
                "fe_h",
                "mg_fe",
                "c_fe",
                "si_fe",
                "ni_fe",
                "al_fe",
                "ca_fe",
                "n_fe",
                "mn_fe",
            ],
        )

    def test_plot_parameter_saliency_runs(self, sample_result):
        """Test that plot_parameter_saliency runs without error."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig = plot_parameter_saliency(sample_result, "teff")
        assert fig is not None
        plt.close(fig)

    def test_plot_parameter_saliency_saves(self, sample_result):
        """Test that plot_parameter_saliency saves to file."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_saliency.png"
            fig = plot_parameter_saliency(
                sample_result, "fe_h", output_path=output_path
            )
            assert output_path.exists()
            plt.close(fig)

    def test_plot_saliency_heatmap_runs(self, sample_result):
        """Test that plot_saliency_heatmap runs without error."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plot_saliency_heatmap(sample_result)
        assert fig is not None
        plt.close(fig)

    def test_plot_saliency_heatmap_saves(self, sample_result):
        """Test that plot_saliency_heatmap saves to file."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_heatmap.png"
            fig = plot_saliency_heatmap(sample_result, output_path=output_path)
            assert output_path.exists()
            plt.close(fig)

    def test_plot_with_masked_regions(self, sample_result):
        """Test plotting with masked regions."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Add some masked regions
        sample_result.mask[30:40] = False
        sample_result.mask[70:80] = False

        fig = plot_parameter_saliency(sample_result, "teff")
        assert fig is not None
        plt.close(fig)

        fig = plot_saliency_heatmap(sample_result)
        assert fig is not None
        plt.close(fig)


class TestWithMultiHeadMLP:
    """Tests with the actual MultiHeadMLP model."""

    @pytest.fixture
    def multi_head_model(self):
        """Create a MultiHeadMLP model for testing."""
        from dorothy.models import MultiHeadMLP

        return MultiHeadMLP(
            survey_configs={"boss": 100, "desi": 150},
            n_parameters=11,
            latent_dim=32,
            encoder_hidden=[64],
            trunk_hidden=[32],
            output_hidden=[16],
        )

    def test_saliency_with_multihead(self, multi_head_model):
        """Test saliency computation with MultiHeadMLP."""
        analyzer = SaliencyAnalyzer(multi_head_model, device="cpu")

        # Test with BOSS survey
        X_boss = torch.randn(3, 100)
        wavelength = np.linspace(3600, 9000, 100).astype(np.float32)

        result = analyzer.compute_saliency(X_boss, wavelength, survey="boss")

        assert result.survey == "boss"
        assert result.jacobian_mean_flux.shape == (11, 100)
        assert np.isfinite(result.jacobian_mean_flux).all()

    def test_saliency_with_different_surveys(self, multi_head_model):
        """Test that different surveys produce different saliency maps."""
        analyzer = SaliencyAnalyzer(multi_head_model, device="cpu")

        # Same random input for both surveys (different wavelength grids)
        X_boss = torch.randn(3, 100)
        X_desi = torch.randn(3, 150)
        wavelength_boss = np.linspace(3600, 9000, 100).astype(np.float32)
        wavelength_desi = np.linspace(3600, 9800, 150).astype(np.float32)

        result_boss = analyzer.compute_saliency(X_boss, wavelength_boss, survey="boss")
        result_desi = analyzer.compute_saliency(X_desi, wavelength_desi, survey="desi")

        assert result_boss.survey == "boss"
        assert result_desi.survey == "desi"
        assert result_boss.jacobian_mean_flux.shape[1] == 100
        assert result_desi.jacobian_mean_flux.shape[1] == 150


# =============================================================================
# Ablation Saliency Tests
# =============================================================================


class TestAblationSaliencyResult:
    """Tests for the AblationSaliencyResult dataclass."""

    def test_basic_creation(self):
        """Test that AblationSaliencyResult can be created."""
        n_wavelengths = 100
        n_params = 11

        result = AblationSaliencyResult(
            survey="boss",
            wavelength=np.linspace(3600, 9000, n_wavelengths).astype(np.float32),
            spectrum=np.random.randn(n_wavelengths).astype(np.float32),
            spectrum_error=np.abs(np.random.randn(n_wavelengths)).astype(np.float32),
            mask=np.ones(n_wavelengths, dtype=bool),
            block_size=50,
            delta_mu=np.abs(np.random.randn(n_params, n_wavelengths)).astype(
                np.float32
            ),
            delta_mu_total=np.abs(np.random.randn(n_wavelengths)).astype(np.float32),
            delta_sigma=np.abs(np.random.randn(n_params, n_wavelengths)).astype(
                np.float32
            ),
            delta_sigma_total=np.abs(np.random.randn(n_wavelengths)).astype(np.float32),
            fisher_weighted=np.abs(np.random.randn(n_params, n_wavelengths)).astype(
                np.float32
            ),
            fisher_weighted_total=np.abs(np.random.randn(n_wavelengths)).astype(
                np.float32
            ),
            predictions=np.random.randn(n_params).astype(np.float32),
            uncertainties=np.abs(np.random.randn(n_params)).astype(np.float32) + 0.01,
            parameter_names=[
                "teff",
                "logg",
                "fe_h",
                "mg_fe",
                "c_fe",
                "si_fe",
                "ni_fe",
                "al_fe",
                "ca_fe",
                "n_fe",
                "mn_fe",
            ],
        )

        assert result.survey == "boss"
        assert len(result.wavelength) == n_wavelengths
        assert result.delta_mu.shape == (n_params, n_wavelengths)
        assert result.delta_sigma.shape == (n_params, n_wavelengths)
        assert result.block_size == 50

    def test_get_param_index(self):
        """Test parameter index lookup."""
        n = 100
        result = AblationSaliencyResult(
            survey="boss",
            wavelength=np.zeros(n, dtype=np.float32),
            spectrum=np.zeros(n, dtype=np.float32),
            spectrum_error=np.zeros(n, dtype=np.float32),
            mask=np.ones(n, dtype=bool),
            block_size=50,
            delta_mu=np.zeros((11, n), dtype=np.float32),
            delta_mu_total=np.zeros(n, dtype=np.float32),
            delta_sigma=np.zeros((11, n), dtype=np.float32),
            delta_sigma_total=np.zeros(n, dtype=np.float32),
            fisher_weighted=np.zeros((11, n), dtype=np.float32),
            fisher_weighted_total=np.zeros(n, dtype=np.float32),
            predictions=np.zeros(11, dtype=np.float32),
            uncertainties=np.ones(11, dtype=np.float32),
            parameter_names=[
                "teff",
                "logg",
                "fe_h",
                "mg_fe",
                "c_fe",
                "si_fe",
                "ni_fe",
                "al_fe",
                "ca_fe",
                "n_fe",
                "mn_fe",
            ],
        )

        assert result.get_param_index("teff") == 0
        assert result.get_param_index("fe_h") == 2
        assert result.get_param_index("mn_fe") == 10

    def test_block_size_list(self):
        """Test that block_size can be a list (for training distribution mode)."""
        n = 100
        result = AblationSaliencyResult(
            survey="boss",
            wavelength=np.zeros(n, dtype=np.float32),
            spectrum=np.zeros(n, dtype=np.float32),
            spectrum_error=np.zeros(n, dtype=np.float32),
            mask=np.ones(n, dtype=bool),
            block_size=[10, 25, 50],  # Multiple block sizes
            delta_mu=np.zeros((11, n), dtype=np.float32),
            delta_mu_total=np.zeros(n, dtype=np.float32),
            delta_sigma=np.zeros((11, n), dtype=np.float32),
            delta_sigma_total=np.zeros(n, dtype=np.float32),
            fisher_weighted=np.zeros((11, n), dtype=np.float32),
            fisher_weighted_total=np.zeros(n, dtype=np.float32),
            predictions=np.zeros(11, dtype=np.float32),
            uncertainties=np.ones(11, dtype=np.float32),
            parameter_names=["p" + str(i) for i in range(11)],
        )

        assert result.block_size == [10, 25, 50]


class TestAblationSaliencyAnalyzer:
    """Tests for AblationSaliencyAnalyzer."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple linear model for testing."""
        return SimpleLinearModel(n_wavelengths=100, n_params=11)

    @pytest.fixture
    def analyzer(self, simple_model):
        """Create an analyzer with the simple model."""
        return AblationSaliencyAnalyzer(simple_model, device="cpu", batch_size=16)

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.device == torch.device("cpu")
        assert len(analyzer.parameter_names) == 11
        assert analyzer.scatter_floor == 0.01
        assert analyzer.batch_size == 16

    def test_compute_saliency_shape(self, analyzer):
        """Test that compute_saliency returns correct shapes."""
        n_wavelengths = 100
        X = torch.randn(3, n_wavelengths)
        X[2, :] = 1.0  # All valid
        wavelength = np.linspace(3600, 9000, n_wavelengths).astype(np.float32)

        result = analyzer.compute_saliency(X, wavelength, survey="test", block_size=10)

        assert result.delta_mu.shape == (11, n_wavelengths)
        assert result.delta_mu_total.shape == (n_wavelengths,)
        assert result.fisher_weighted.shape == (11, n_wavelengths)
        assert result.fisher_weighted_total.shape == (n_wavelengths,)
        assert result.predictions.shape == (11,)
        assert result.uncertainties.shape == (11,)

    def test_fisher_nonnegative_deltas_signed(self, analyzer):
        """Test that Fisher is non-negative while deltas are signed."""
        n_wavelengths = 100
        X = torch.randn(3, n_wavelengths)
        X[2, :] = 1.0  # All valid
        wavelength = np.linspace(3600, 9000, n_wavelengths).astype(np.float32)

        result = analyzer.compute_saliency(X, wavelength, survey="test", block_size=10)

        # Fisher importance should always be non-negative (squared terms)
        assert np.all(result.fisher_weighted >= 0)
        assert np.all(result.fisher_weighted_total >= 0)

        # Delta-mu and delta-sigma are signed (can be positive or negative)
        # Just check they have valid shape and are finite
        assert result.delta_mu.shape == (11, n_wavelengths)
        assert result.delta_sigma.shape == (11, n_wavelengths)
        assert np.all(np.isfinite(result.delta_mu))
        assert np.all(np.isfinite(result.delta_sigma))

    def test_masked_pixels_zero_importance(self, analyzer):
        """Test that naturally masked pixels have zero importance."""
        n_wavelengths = 100
        X = torch.randn(3, n_wavelengths)
        X[2, :] = 1.0  # Start with all valid
        X[2, 40:60] = 0.0  # Mask a region
        wavelength = np.linspace(3600, 9000, n_wavelengths).astype(np.float32)

        result = analyzer.compute_saliency(X, wavelength, survey="test", block_size=10)

        # Masked pixels should have zero importance
        assert np.all(result.delta_mu[:, 40:60] == 0)
        assert np.all(result.fisher_weighted[:, 40:60] == 0)

    def test_block_size_affects_results(self, analyzer):
        """Test that different block sizes produce different results."""
        n_wavelengths = 100
        X = torch.randn(3, n_wavelengths)
        X[2, :] = 1.0
        wavelength = np.linspace(3600, 9000, n_wavelengths).astype(np.float32)

        result_small = analyzer.compute_saliency(
            X, wavelength, survey="test", block_size=5
        )
        result_large = analyzer.compute_saliency(
            X, wavelength, survey="test", block_size=25
        )

        # Results should differ (not identical)
        assert not np.allclose(result_small.delta_mu, result_large.delta_mu)

    def test_training_distribution_mode(self, analyzer):
        """Test averaging over training block size distribution."""
        n_wavelengths = 100
        X = torch.randn(3, n_wavelengths)
        X[2, :] = 1.0
        wavelength = np.linspace(3600, 9000, n_wavelengths).astype(np.float32)

        result = analyzer.compute_saliency(
            X,
            wavelength,
            survey="test",
            use_training_distribution=True,
            n_block_sizes=5,  # Use fewer for speed
            f_max=0.3,
        )

        # block_size should be a list when using training distribution
        assert isinstance(result.block_size, list)
        assert len(result.block_size) > 1

        # Results should still be valid (deltas are signed, fisher is non-negative)
        assert result.delta_mu.shape == (11, n_wavelengths)
        assert np.all(np.isfinite(result.delta_mu))
        assert np.all(result.fisher_weighted >= 0)

    def test_large_block_size(self, analyzer):
        """Test with block size larger than spectrum."""
        n_wavelengths = 100
        X = torch.randn(3, n_wavelengths)
        X[2, :] = 1.0
        wavelength = np.linspace(3600, 9000, n_wavelengths).astype(np.float32)

        # Block size >= spectrum length should return zeros
        result = analyzer.compute_saliency(X, wavelength, survey="test", block_size=150)

        assert np.all(result.delta_mu == 0)
        assert np.all(result.fisher_weighted == 0)


class TestAblationWithMultiHeadMLP:
    """Tests with the actual MultiHeadMLP model."""

    @pytest.fixture
    def multi_head_model(self):
        """Create a MultiHeadMLP model for testing."""
        from dorothy.models import MultiHeadMLP

        return MultiHeadMLP(
            survey_configs={"boss": 100, "desi": 150},
            n_parameters=11,
            latent_dim=32,
            encoder_hidden=[64],
            trunk_hidden=[32],
            output_hidden=[16],
        )

    def test_ablation_with_multihead(self, multi_head_model):
        """Test ablation saliency with MultiHeadMLP."""
        analyzer = AblationSaliencyAnalyzer(
            multi_head_model, device="cpu", batch_size=16
        )

        X_boss = torch.randn(3, 100)
        X_boss[2, :] = 1.0
        wavelength = np.linspace(3600, 9000, 100).astype(np.float32)

        result = analyzer.compute_saliency(
            X_boss, wavelength, survey="boss", block_size=10
        )

        assert result.survey == "boss"
        assert result.delta_mu.shape == (11, 100)
        assert np.isfinite(result.delta_mu).all()

    def test_ablation_different_surveys(self, multi_head_model):
        """Test that different surveys work correctly."""
        analyzer = AblationSaliencyAnalyzer(
            multi_head_model, device="cpu", batch_size=16
        )

        X_boss = torch.randn(3, 100)
        X_boss[2, :] = 1.0
        X_desi = torch.randn(3, 150)
        X_desi[2, :] = 1.0
        wavelength_boss = np.linspace(3600, 9000, 100).astype(np.float32)
        wavelength_desi = np.linspace(3600, 9800, 150).astype(np.float32)

        result_boss = analyzer.compute_saliency(
            X_boss, wavelength_boss, survey="boss", block_size=10
        )
        result_desi = analyzer.compute_saliency(
            X_desi, wavelength_desi, survey="desi", block_size=10
        )

        assert result_boss.survey == "boss"
        assert result_desi.survey == "desi"
        assert result_boss.delta_mu.shape[1] == 100
        assert result_desi.delta_mu.shape[1] == 150


class TestAblationVisualization:
    """Tests for ablation visualization functions."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample AblationSaliencyResult for testing."""
        n_wavelengths = 100
        n_params = 11

        return AblationSaliencyResult(
            survey="boss",
            wavelength=np.linspace(3600, 9000, n_wavelengths).astype(np.float32),
            spectrum=np.random.randn(n_wavelengths).astype(np.float32) + 1.0,
            spectrum_error=np.abs(np.random.randn(n_wavelengths)).astype(np.float32)
            * 0.1
            + 0.01,
            mask=np.ones(n_wavelengths, dtype=bool),
            block_size=50,
            delta_mu=np.abs(np.random.randn(n_params, n_wavelengths)).astype(
                np.float32
            ),
            delta_mu_total=np.abs(np.random.randn(n_wavelengths)).astype(np.float32),
            delta_sigma=np.abs(np.random.randn(n_params, n_wavelengths)).astype(
                np.float32
            ),
            delta_sigma_total=np.abs(np.random.randn(n_wavelengths)).astype(np.float32),
            fisher_weighted=np.abs(np.random.randn(n_params, n_wavelengths)).astype(
                np.float32
            ),
            fisher_weighted_total=np.abs(np.random.randn(n_wavelengths)).astype(
                np.float32
            ),
            predictions=np.array(
                [5000, 4.0, -0.5, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05],
                dtype=np.float32,
            ),
            uncertainties=np.array(
                [50, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                dtype=np.float32,
            ),
            parameter_names=[
                "teff",
                "logg",
                "fe_h",
                "mg_fe",
                "c_fe",
                "si_fe",
                "ni_fe",
                "al_fe",
                "ca_fe",
                "n_fe",
                "mn_fe",
            ],
        )

    def test_plot_ablation_parameter_saliency_runs(self, sample_result):
        """Test that plot_ablation_parameter_saliency runs without error."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plot_ablation_parameter_saliency(sample_result, "teff")
        assert fig is not None
        plt.close(fig)

    def test_plot_ablation_parameter_saliency_saves(self, sample_result):
        """Test that plot saves to file."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_ablation.png"
            fig = plot_ablation_parameter_saliency(
                sample_result, "fe_h", output_path=output_path
            )
            assert output_path.exists()
            plt.close(fig)

    def test_plot_with_masked_regions(self, sample_result):
        """Test plotting with masked regions."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample_result.mask[30:40] = False
        sample_result.mask[70:80] = False

        fig = plot_ablation_parameter_saliency(sample_result, "teff")
        assert fig is not None
        plt.close(fig)

    def test_plot_with_list_block_size(self):
        """Test plotting when block_size is a list."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_wavelengths = 100
        n_params = 11

        result = AblationSaliencyResult(
            survey="boss",
            wavelength=np.linspace(3600, 9000, n_wavelengths).astype(np.float32),
            spectrum=np.random.randn(n_wavelengths).astype(np.float32) + 1.0,
            spectrum_error=np.abs(np.random.randn(n_wavelengths)).astype(np.float32)
            * 0.1
            + 0.01,
            mask=np.ones(n_wavelengths, dtype=bool),
            block_size=[10, 25, 50],  # List of block sizes
            delta_mu=np.abs(np.random.randn(n_params, n_wavelengths)).astype(
                np.float32
            ),
            delta_mu_total=np.abs(np.random.randn(n_wavelengths)).astype(np.float32),
            delta_sigma=np.abs(np.random.randn(n_params, n_wavelengths)).astype(
                np.float32
            ),
            delta_sigma_total=np.abs(np.random.randn(n_wavelengths)).astype(np.float32),
            fisher_weighted=np.abs(np.random.randn(n_params, n_wavelengths)).astype(
                np.float32
            ),
            fisher_weighted_total=np.abs(np.random.randn(n_wavelengths)).astype(
                np.float32
            ),
            predictions=np.zeros(n_params, dtype=np.float32),
            uncertainties=np.ones(n_params, dtype=np.float32),
            parameter_names=["p" + str(i) for i in range(n_params)],
        )

        fig = plot_ablation_parameter_saliency(result, "p0")
        assert fig is not None
        plt.close(fig)

    def test_plot_ablation_saliency_heatmap_runs(self, sample_result):
        """Test that plot_ablation_saliency_heatmap runs without error."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plot_ablation_saliency_heatmap(sample_result)
        assert fig is not None
        plt.close(fig)

    def test_plot_ablation_saliency_heatmap_saves(self, sample_result):
        """Test that heatmap saves to file."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_ablation_heatmap.png"
            fig = plot_ablation_saliency_heatmap(sample_result, output_path=output_path)
            assert output_path.exists()
            plt.close(fig)
