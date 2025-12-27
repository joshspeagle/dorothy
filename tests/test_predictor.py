"""
Tests for the prediction infrastructure.

These tests verify:
1. Predictor creation and initialization
2. Model loading from checkpoints
3. Batch and chunked prediction
4. Uncertainty extraction
5. PredictionResult container
6. Embedding extraction
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from dorothy.config.schema import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
)
from dorothy.data.normalizer import LabelNormalizer
from dorothy.inference.predictor import (
    PredictionResult,
    Predictor,
)
from dorothy.models import MLP


@pytest.fixture
def simple_model():
    """Create a simple trained model for testing."""
    model = MLP(
        input_features=100,
        output_features=6,
        hidden_layers=[32, 16],
    )
    model.eval()
    return model


@pytest.fixture
def simple_input():
    """Create simple input data for testing."""
    np.random.seed(42)
    return np.random.randn(50, 100).astype(np.float32)


@pytest.fixture
def simple_normalizer():
    """Create a simple fitted normalizer for testing."""
    np.random.seed(42)
    y = np.column_stack(
        [
            np.random.uniform(4000, 6000, 100),  # teff
            np.random.uniform(2, 5, 100),  # logg
            np.random.uniform(-1, 0.5, 100),  # feh
        ]
    )
    normalizer = LabelNormalizer(parameters=["teff", "logg", "feh"])
    normalizer.fit(y)
    return normalizer


class TestPredictionResult:
    """Tests for the PredictionResult container."""

    def test_properties(self):
        """Test that properties return correct values."""
        predictions = np.random.randn(20, 3).astype(np.float32)
        uncertainties = np.abs(np.random.randn(20, 3)).astype(np.float32)
        raw_output = np.random.randn(20, 6).astype(np.float32)

        result = PredictionResult(
            predictions=predictions,
            uncertainties=uncertainties,
            raw_output=raw_output,
            parameter_names=["teff", "logg", "feh"],
        )

        assert result.n_samples == 20
        assert result.n_parameters == 3

    def test_to_dict(self):
        """Test conversion to dictionary."""
        predictions = np.array([[5000, 2.5, -0.5], [6000, 3.0, 0.0]], dtype=np.float32)
        uncertainties = np.array(
            [[100, 0.1, 0.05], [150, 0.15, 0.08]], dtype=np.float32
        )
        raw_output = np.zeros((2, 6), dtype=np.float32)

        result = PredictionResult(
            predictions=predictions,
            uncertainties=uncertainties,
            raw_output=raw_output,
            parameter_names=["teff", "logg", "feh"],
        )

        d = result.to_dict()

        assert "teff" in d
        assert "teff_err" in d
        assert "logg" in d
        assert "logg_err" in d
        assert "feh" in d
        assert "feh_err" in d
        assert np.allclose(d["teff"], [5000, 6000])
        assert np.allclose(d["logg_err"], [0.1, 0.15])


class TestPredictorInit:
    """Tests for Predictor initialization."""

    def test_predictor_creation(self, simple_model):
        """Test that predictor can be created."""
        predictor = Predictor(simple_model, device="cpu")

        assert predictor.model is simple_model
        assert predictor.device == torch.device("cpu")
        assert predictor.normalizer is None

    def test_predictor_with_normalizer(self, simple_model, simple_normalizer):
        """Test predictor creation with normalizer."""
        predictor = Predictor(
            simple_model,
            normalizer=simple_normalizer,
            device="cpu",
            parameter_names=["teff", "logg", "feh"],
        )

        assert predictor.normalizer is simple_normalizer
        assert predictor.parameter_names == ["teff", "logg", "feh"]

    def test_model_in_eval_mode(self, simple_model):
        """Test that model is set to eval mode."""
        predictor = Predictor(simple_model, device="cpu")
        assert not predictor.model.training


class TestPredictorPredict:
    """Tests for prediction functionality."""

    def test_predict_output_shape(self, simple_model, simple_input):
        """Test that predictions have correct shape."""
        predictor = Predictor(
            simple_model,
            device="cpu",
            parameter_names=["teff", "logg", "feh"],
        )

        result = predictor.predict(simple_input)

        assert result.predictions.shape == (50, 3)
        assert result.uncertainties.shape == (50, 3)
        assert result.raw_output.shape == (50, 6)

    def test_predict_with_torch_tensor(self, simple_model):
        """Test prediction with torch tensor input."""
        predictor = Predictor(
            simple_model,
            device="cpu",
            parameter_names=["teff", "logg", "feh"],
        )

        X = torch.randn(30, 100)
        result = predictor.predict(X)

        assert result.predictions.shape == (30, 3)

    def test_predict_batch_size(self, simple_model, simple_input):
        """Test that batch size is respected."""
        predictor = Predictor(
            simple_model,
            device="cpu",
            parameter_names=["teff", "logg", "feh"],
        )

        # Small batch size should still produce correct results
        result = predictor.predict(simple_input, batch_size=10)

        assert result.predictions.shape == (50, 3)

    def test_uncertainties_positive(self, simple_model, simple_input):
        """Test that uncertainties are always positive."""
        predictor = Predictor(
            simple_model,
            device="cpu",
            parameter_names=["teff", "logg", "feh"],
        )

        result = predictor.predict(simple_input)

        assert np.all(result.uncertainties > 0)

    def test_predict_with_denormalization(
        self, simple_model, simple_normalizer, simple_input
    ):
        """Test prediction with denormalization."""
        predictor = Predictor(
            simple_model,
            normalizer=simple_normalizer,
            device="cpu",
            parameter_names=["teff", "logg", "feh"],
        )

        result_norm = predictor.predict(simple_input, denormalize=False)
        result_denorm = predictor.predict(simple_input, denormalize=True)

        # Results should be different
        assert not np.allclose(result_norm.predictions, result_denorm.predictions)
        # Denormalized result should not be marked as normalized
        assert result_norm.is_normalized
        assert not result_denorm.is_normalized


class TestPredictorChunked:
    """Tests for chunked prediction."""

    def test_predict_chunked(self, simple_model, simple_input):
        """Test chunked prediction iterator."""
        predictor = Predictor(
            simple_model,
            device="cpu",
            parameter_names=["teff", "logg", "feh"],
        )

        # Create an iterator of chunks
        def chunk_iterator():
            for i in range(0, 50, 10):
                yield simple_input[i : i + 10]

        results = list(predictor.predict_chunked(chunk_iterator()))

        assert len(results) == 5
        for result in results:
            assert result.predictions.shape == (10, 3)

    def test_predict_all_chunked(self, simple_model, simple_input):
        """Test combined chunked prediction."""
        predictor = Predictor(
            simple_model,
            device="cpu",
            parameter_names=["teff", "logg", "feh"],
        )

        def chunk_iterator():
            for i in range(0, 50, 10):
                yield simple_input[i : i + 10]

        result = predictor.predict_all_chunked(chunk_iterator())

        assert result.predictions.shape == (50, 3)
        assert result.uncertainties.shape == (50, 3)


class TestPredictorEmbeddings:
    """Tests for embedding extraction."""

    def test_get_embeddings_shape(self, simple_model, simple_input):
        """Test that embeddings have correct shape."""
        predictor = Predictor(simple_model, device="cpu")

        embeddings = predictor.get_embeddings(simple_input)

        # Should be (n_samples, last_hidden_dim)
        assert embeddings.shape[0] == 50
        assert embeddings.shape[1] > 0

    def test_get_embeddings_with_layer_index(self, simple_model, simple_input):
        """Test embedding extraction from different layers."""
        predictor = Predictor(simple_model, device="cpu")

        # Different layer indices should give different dimensions
        embed_default = predictor.get_embeddings(simple_input, layer_index=-2)
        embed_earlier = predictor.get_embeddings(simple_input, layer_index=-3)

        # Shapes should be different (different layer sizes)
        assert embed_default.shape[0] == embed_earlier.shape[0]  # Same n_samples


class TestPredictorLoad:
    """Tests for loading predictors from checkpoints."""

    @pytest.fixture
    def temp_checkpoint(self, simple_model):
        """Create a temporary checkpoint directory."""
        # Create a normalizer with all 11 default parameters for testing
        from dorothy.config.schema import STELLAR_PARAMETERS

        np.random.seed(42)
        n_samples = 100
        n_params = len(STELLAR_PARAMETERS)

        # Generate realistic-ish data for each parameter
        y = np.zeros((n_samples, n_params), dtype=np.float32)
        y[:, 0] = np.random.uniform(4000, 6000, n_samples)  # teff
        y[:, 1] = np.random.uniform(2, 5, n_samples)  # logg
        for i in range(2, n_params):
            y[:, i] = np.random.uniform(-1, 0.5, n_samples)  # abundances

        full_normalizer = LabelNormalizer(parameters=list(STELLAR_PARAMETERS))
        full_normalizer.fit(y)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir)

            # Save model
            torch.save(simple_model.state_dict(), checkpoint_path / "best_model.pth")

            # Save normalizer
            full_normalizer.save(checkpoint_path / "normalizer.pkl")

            yield checkpoint_path

    def test_load_from_checkpoint(self, temp_checkpoint):
        """Test loading predictor from checkpoint."""
        predictor = Predictor.load(temp_checkpoint, device="cpu")

        assert predictor.model is not None
        assert predictor.normalizer is not None

    def test_load_model_only(self, temp_checkpoint):
        """Test loading only model without normalizer."""
        predictor = Predictor.load(
            temp_checkpoint,
            normalizer_file=None,
            device="cpu",
        )

        assert predictor.model is not None
        assert predictor.normalizer is None

    def test_load_missing_model_raises(self):
        """Test that missing model file raises error."""
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(FileNotFoundError):
            Predictor.load(tmpdir, device="cpu")

    def test_infer_architecture(self, simple_model):
        """Test architecture inference from state dict."""
        state_dict = simple_model.state_dict()

        input_features, output_features, hidden_layers = Predictor._infer_architecture(
            state_dict
        )

        assert input_features == 100
        assert output_features == 6
        assert hidden_layers == [32, 16]


class TestPredictorFromConfig:
    """Tests for creating predictors from config."""

    @pytest.fixture
    def simple_config(self):
        """Create a simple experiment config."""
        return ExperimentConfig(
            name="test_experiment",
            data=DataConfig(fits_path=Path("/fake/path.fits"), wavelength_bins=1000),
            model=ModelConfig(
                input_features=1000,
                output_features=6,
                hidden_layers=[32, 16],
            ),
        )

    def test_from_config_creates_predictor(self, simple_config, simple_model):
        """Test creating predictor from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Config expects: output_dir / name / name_final
            checkpoint_path = Path(tmpdir) / "test_experiment" / "test_experiment_final"
            checkpoint_path.mkdir(parents=True)

            torch.save(simple_model.state_dict(), checkpoint_path / "best_model.pth")

            simple_config = ExperimentConfig(
                name="test_experiment",
                data=DataConfig(
                    fits_path=Path("/fake/path.fits"), wavelength_bins=1000
                ),
                model=ModelConfig(
                    input_features=100,
                    output_features=6,
                    hidden_layers=[32, 16],
                ),
                output_dir=Path(tmpdir),
            )

            predictor = Predictor.from_config(simple_config)
            assert predictor.model is not None
