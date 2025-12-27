"""
Integration tests for DOROTHY end-to-end workflows.

These tests verify that all components work together correctly:
1. Config creation and validation
2. Model training workflow
3. Prediction workflow
4. Anomaly detection workflow

Note: These tests use synthetic data and small models for speed.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dorothy.analysis import AnomalyDetector
from dorothy.config.schema import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)
from dorothy.data import SpectralData, split_data
from dorothy.data.normalizer import LabelNormalizer
from dorothy.inference import PredictionResult, Predictor
from dorothy.models import MLP
from dorothy.training import Trainer


@pytest.fixture
def synthetic_spectral_data():
    """Create synthetic spectral data for testing.

    Uses 1000 wavelength bins to match minimum valid config.
    """
    np.random.seed(42)
    n_samples = 200
    n_wavelengths = 1000  # Match minimum valid wavelength_bins
    n_params = 3

    # Synthetic spectra (normalized flux + ivar)
    flux = np.random.randn(n_samples, n_wavelengths).astype(np.float32)
    ivar = np.abs(np.random.randn(n_samples, n_wavelengths)).astype(np.float32) + 0.1
    wavelength = np.linspace(3800, 9000, n_wavelengths).astype(np.float64)

    # Synthetic labels (3 parameters for small test)
    labels = np.column_stack(
        [
            np.random.uniform(4500, 6500, n_samples),  # teff
            np.random.uniform(2.0, 5.0, n_samples),  # logg
            np.random.uniform(-1.0, 0.5, n_samples),  # feh
        ]
    ).astype(np.float32)

    # Synthetic errors
    errors = (
        np.abs(np.random.randn(n_samples, n_params)).astype(np.float32) * 0.1 + 0.01
    )

    return SpectralData(
        flux=flux,
        ivar=ivar,
        wavelength=wavelength,
        labels=labels,
        errors=errors,
        ids=np.array([str(i) for i in range(n_samples)]),
        quality_mask=np.ones(n_samples, dtype=bool),
        parameter_names=["teff", "logg", "feh"],
    )


def make_test_config(tmpdir=None, epochs=5):
    """Create a test config for integration tests.

    Uses 1000 wavelength_bins * 2 channels = 2000 input_features.
    """
    return ExperimentConfig(
        name="integration_test",
        data=DataConfig(
            fits_path=Path("/fake/path.fits"),
            wavelength_bins=1000,
            input_channels=2,
        ),
        model=ModelConfig(
            # input_features will be auto-synced to 2000 (1000 * 2)
            output_features=6,  # 3 params * 2 (mean + uncertainty)
            hidden_layers=[64, 32],
        ),
        training=TrainingConfig(
            epochs=epochs,
            batch_size=32,
            learning_rate=0.001,
        ),
        output_dir=Path(tmpdir) if tmpdir else Path("./outputs"),
        seed=42,
        device="cpu",
    )


class TestConfigWorkflow:
    """Tests for configuration workflows."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = ExperimentConfig(
            name="test",
            data=DataConfig(fits_path=Path("/test.fits")),
        )
        assert config.model.hidden_layers == [5000, 2000, 1000, 500, 200, 100]
        assert config.training.epochs == 300
        assert config.training.loss.value == "heteroscedastic"

    def test_config_sync_input_features(self):
        """Test that input features are synced from data config."""
        config = ExperimentConfig(
            name="test",
            data=DataConfig(
                fits_path=Path("/test.fits"),
                wavelength_bins=1000,
                input_channels=2,
            ),
        )
        assert config.model.input_features == 2000  # 2 * 1000


class TestTrainingWorkflow:
    """Tests for the training workflow."""

    def test_trainer_creation(self):
        """Test trainer can be created from config."""
        config = make_test_config()
        trainer = Trainer(config)
        assert trainer.model is not None
        assert trainer.config is config

    def test_trainer_fit(self, synthetic_spectral_data):
        """Test complete training cycle."""
        (X_train, y_train), (X_val, y_val), _ = split_data(
            synthetic_spectral_data,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42,
        )

        config = make_test_config(epochs=5)
        trainer = Trainer(config)
        history = trainer.fit(X_train, y_train, X_val, y_val)

        assert history is not None
        assert len(history.train_losses) == 5
        assert len(history.val_losses) == 5
        assert history.best_epoch >= 0

    def test_trainer_save_and_load(self, synthetic_spectral_data):
        """Test checkpoint saving and loading."""
        (X_train, y_train), (X_val, y_val), _ = split_data(
            synthetic_spectral_data,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_test_config(tmpdir=tmpdir, epochs=3)

            trainer = Trainer(config)
            trainer.fit(X_train, y_train, X_val, y_val)
            checkpoint_path = trainer.save_checkpoint()

            assert checkpoint_path.exists()
            assert (checkpoint_path / "best_model.pth").exists()


class TestPredictionWorkflow:
    """Tests for the prediction workflow."""

    def test_predictor_from_model(self, synthetic_spectral_data):
        """Test creating predictor from trained model."""
        model = MLP(
            input_features=2000,
            output_features=6,
            hidden_layers=[64, 32],
        )

        predictor = Predictor(
            model,
            device="cpu",
            parameter_names=["teff", "logg", "feh"],
        )

        X = synthetic_spectral_data.get_model_input()
        result = predictor.predict(X)

        assert isinstance(result, PredictionResult)
        assert result.predictions.shape == (200, 3)
        assert result.uncertainties.shape == (200, 3)
        assert np.all(result.uncertainties > 0)

    def test_predictor_load_from_checkpoint(self, synthetic_spectral_data):
        """Test loading predictor from saved checkpoint."""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
            synthetic_spectral_data,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_test_config(tmpdir=tmpdir, epochs=2)

            # Train and save
            trainer = Trainer(config)
            trainer.fit(X_train, y_train, X_val, y_val)
            checkpoint_path = trainer.save_checkpoint()

            # Load and predict
            predictor = Predictor.load(checkpoint_path, device="cpu")
            result = predictor.predict(X_test)

            assert result.predictions.shape[0] == X_test.shape[0]
            assert result.predictions.shape[1] == 3  # 3 parameters

    def test_prediction_with_normalizer(self, synthetic_spectral_data):
        """Test prediction with denormalization."""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
            synthetic_spectral_data,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42,
        )

        # Extract just labels (first 3 columns) from combined labels+errors
        n_params = 3
        y_train_labels = y_train[:, :n_params]

        # Create and fit normalizer
        normalizer = LabelNormalizer(parameters=["teff", "logg", "feh"])
        normalizer.fit(y_train_labels)

        model = MLP(
            input_features=2000,
            output_features=6,
            hidden_layers=[64, 32],
        )

        predictor = Predictor(
            model,
            normalizer=normalizer,
            device="cpu",
            parameter_names=["teff", "logg", "feh"],
        )

        result_norm = predictor.predict(X_test, denormalize=False)
        result_denorm = predictor.predict(X_test, denormalize=True)

        assert result_norm.is_normalized
        assert not result_denorm.is_normalized
        # Values should differ after denormalization
        assert not np.allclose(result_norm.predictions, result_denorm.predictions)


class TestAnomalyDetectionWorkflow:
    """Tests for the anomaly detection workflow."""

    def test_detector_from_predictor(self, synthetic_spectral_data):
        """Test creating anomaly detector from predictor."""
        model = MLP(
            input_features=2000,
            output_features=6,
            hidden_layers=[64, 32],
        )
        predictor = Predictor(model, device="cpu")

        detector = AnomalyDetector.from_predictor(predictor)
        assert detector.predictor is predictor

    def test_detector_fit_and_detect(self, synthetic_spectral_data):
        """Test fitting detector and detecting anomalies."""
        model = MLP(
            input_features=2000,
            output_features=6,
            hidden_layers=[64, 32],
        )
        predictor = Predictor(model, device="cpu")

        X = synthetic_spectral_data.get_model_input()
        X_train = X[:150]
        X_test = X[150:]

        detector = AnomalyDetector.from_predictor(predictor)
        detector.fit(X_train, k=5, percentile=95.0)

        result = detector.detect(X_test)

        assert result.n_samples == 50
        assert result.threshold > 0
        assert result.k == 5

    def test_detector_saves_with_checkpoint(self, synthetic_spectral_data):
        """Test saving anomaly detector alongside model checkpoint."""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
            synthetic_spectral_data,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_test_config(tmpdir=tmpdir, epochs=2)

            # Train model
            trainer = Trainer(config)
            trainer.fit(X_train, y_train, X_val, y_val)
            checkpoint_path = trainer.save_checkpoint()

            # Create predictor and fit anomaly detector
            predictor = Predictor.load(checkpoint_path, device="cpu")
            detector = AnomalyDetector.from_predictor(predictor)
            detector.fit(X_train, k=5, percentile=95.0)

            # Save detector
            detector_path = checkpoint_path / "anomaly_detector.pkl"
            detector.save(detector_path)
            assert detector_path.exists()

            # Load and verify
            new_predictor = Predictor.load(checkpoint_path, device="cpu")
            new_detector = AnomalyDetector.from_predictor(new_predictor)
            new_detector.load_state(detector_path)

            # Should produce same results
            original_result = detector.detect(X_test)
            loaded_result = new_detector.detect(X_test)

            np.testing.assert_array_almost_equal(
                original_result.distances, loaded_result.distances
            )


class TestEndToEndWorkflow:
    """Complete end-to-end integration tests."""

    def test_full_pipeline(self, synthetic_spectral_data):
        """Test complete pipeline: config -> train -> save -> load -> predict -> anomaly."""
        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
            synthetic_spectral_data,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create config
            config = make_test_config(tmpdir=tmpdir, epochs=3)

            # 2. Train model
            trainer = Trainer(config)
            history = trainer.fit(X_train, y_train, X_val, y_val)
            assert history.best_val_loss <= min(history.val_losses)  # Best is tracked

            # 3. Save checkpoint
            checkpoint_path = trainer.save_checkpoint()
            assert checkpoint_path.exists()

            # 4. Load predictor
            predictor = Predictor.load(checkpoint_path, device="cpu")
            assert predictor.model is not None

            # 5. Make predictions
            result = predictor.predict(X_test)
            assert result.predictions.shape == (X_test.shape[0], 3)

            # 6. Fit anomaly detector
            detector = AnomalyDetector.from_predictor(predictor)
            detector.fit(X_train, k=5)

            # 7. Detect anomalies
            anomaly_result = detector.detect(X_test)
            assert anomaly_result.n_samples == X_test.shape[0]

            # Verify the pipeline produces valid outputs
            assert not np.any(np.isnan(result.predictions))
            assert not np.any(np.isnan(result.uncertainties))
            assert not np.any(np.isnan(anomaly_result.distances))

    def test_reproducibility(self, synthetic_spectral_data):
        """Test that training is reproducible with same seed."""
        (X_train, y_train), (X_val, y_val), _ = split_data(
            synthetic_spectral_data,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_test_config(tmpdir=tmpdir, epochs=2)

            # First training run
            trainer1 = Trainer(config)
            history1 = trainer1.fit(X_train, y_train, X_val, y_val)

            # Reset seed and train again
            trainer2 = Trainer(config)
            history2 = trainer2.fit(X_train, y_train, X_val, y_val)

            # Losses should be identical
            np.testing.assert_allclose(
                history1.train_losses, history2.train_losses, rtol=1e-5
            )
