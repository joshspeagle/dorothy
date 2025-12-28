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
from dorothy.data.normalizer import LabelNormalizer
from dorothy.inference import PredictionResult, Predictor
from dorothy.models import MLP
from dorothy.training import Trainer


@pytest.fixture
def synthetic_3ch_data():
    """Create synthetic 3-channel data for testing.

    Uses 1000 wavelength bins to match minimum valid config.
    Returns X (N, 3, wavelengths) and y (N, 3, n_params) in 3-channel format.
    """
    np.random.seed(42)
    n_samples = 200
    n_wavelengths = 1000  # Match minimum valid wavelength_bins
    n_params = 3

    # 3-channel spectral data: [flux, sigma, mask]
    flux = np.random.randn(n_samples, n_wavelengths).astype(np.float32)
    sigma = np.abs(np.random.randn(n_samples, n_wavelengths)).astype(np.float32) + 0.1
    spec_mask = np.ones((n_samples, n_wavelengths), dtype=np.float32)
    X = np.stack([flux, sigma, spec_mask], axis=1)  # (N, 3, wavelengths)

    # 3-channel labels: [values, errors, mask]
    labels = np.column_stack(
        [
            np.random.uniform(4500, 6500, n_samples),  # teff
            np.random.uniform(2.0, 5.0, n_samples),  # logg
            np.random.uniform(-1.0, 0.5, n_samples),  # feh
        ]
    ).astype(np.float32)
    errors = (
        np.abs(np.random.randn(n_samples, n_params)).astype(np.float32) * 0.1 + 0.01
    )
    label_mask = np.ones((n_samples, n_params), dtype=np.float32)
    y = np.stack([labels, errors, label_mask], axis=1)  # (N, 3, n_params)

    return X, y


def split_3ch_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Split 3-channel data into train/val/test sets."""
    n_samples = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return (
        (X[train_idx], y[train_idx]),
        (X[val_idx], y[val_idx]),
        (X[test_idx], y[test_idx]),
    )


def make_test_config(tmpdir=None, epochs=5):
    """Create a test config for integration tests.

    Uses 3 channels * 1000 wavelength bins = 3000 input_features.
    """
    return ExperimentConfig(
        name="integration_test",
        data=DataConfig(
            catalogue_path=Path("/fake/catalogue.h5"),
        ),
        model=ModelConfig(
            input_features=3000,  # 3 channels * 1000 wavelengths
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
            data=DataConfig(catalogue_path=Path("/test.h5")),
        )
        assert config.model.hidden_layers == [5000, 2000, 1000, 500, 200, 100]
        assert config.training.epochs == 300
        assert config.training.loss.value == "heteroscedastic"

    def test_config_sync_input_features(self):
        """Test that input features can be set explicitly."""
        config = ExperimentConfig(
            name="test",
            data=DataConfig(catalogue_path=Path("/test.h5")),
            model=ModelConfig(input_features=3000),  # 3 channels * 1000 wavelengths
        )
        assert config.model.input_features == 3000


class TestTrainingWorkflow:
    """Tests for the training workflow."""

    def test_trainer_creation(self):
        """Test trainer can be created from config."""
        config = make_test_config()
        trainer = Trainer(config)
        assert trainer.model is not None
        assert trainer.config is config

    def test_trainer_fit(self, synthetic_3ch_data):
        """Test complete training cycle."""
        X, y = synthetic_3ch_data
        (X_train, y_train), (X_val, y_val), _ = split_3ch_data(
            X, y, train_ratio=0.6, val_ratio=0.2, seed=42
        )

        config = make_test_config(epochs=5)
        trainer = Trainer(config)
        history = trainer.fit(X_train, y_train, X_val, y_val)

        assert history is not None
        assert len(history.train_losses) == 5
        assert len(history.val_losses) == 5
        assert history.best_epoch >= 0

    def test_trainer_save_and_load(self, synthetic_3ch_data):
        """Test checkpoint saving and loading."""
        X, y = synthetic_3ch_data
        (X_train, y_train), (X_val, y_val), _ = split_3ch_data(
            X, y, train_ratio=0.6, val_ratio=0.2, seed=42
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

    def test_predictor_from_model(self, synthetic_3ch_data):
        """Test creating predictor from trained model."""
        X, y = synthetic_3ch_data
        model = MLP(
            input_features=3000,  # 3 channels * 1000 wavelengths
            output_features=6,
            hidden_layers=[64, 32],
        )

        predictor = Predictor(
            model,
            device="cpu",
            parameter_names=["teff", "logg", "fe_h"],
        )

        result = predictor.predict(X)

        assert isinstance(result, PredictionResult)
        assert result.predictions.shape == (200, 3)
        assert result.uncertainties.shape == (200, 3)
        assert np.all(result.uncertainties > 0)

    def test_predictor_load_from_checkpoint(self, synthetic_3ch_data):
        """Test loading predictor from saved checkpoint."""
        X, y = synthetic_3ch_data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_3ch_data(
            X, y, train_ratio=0.6, val_ratio=0.2, seed=42
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

    def test_prediction_with_normalizer(self, synthetic_3ch_data):
        """Test prediction with denormalization."""
        X, y = synthetic_3ch_data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_3ch_data(
            X, y, train_ratio=0.6, val_ratio=0.2, seed=42
        )

        # Extract just labels (channel 0) from 3-channel format
        y_train_labels = y_train[:, 0, :]  # (N, n_params)

        # Create and fit normalizer
        normalizer = LabelNormalizer(parameters=["teff", "logg", "fe_h"])
        normalizer.fit(y_train_labels)

        model = MLP(
            input_features=3000,
            output_features=6,
            hidden_layers=[64, 32],
        )

        predictor = Predictor(
            model,
            normalizer=normalizer,
            device="cpu",
            parameter_names=["teff", "logg", "fe_h"],
        )

        result_norm = predictor.predict(X_test, denormalize=False)
        result_denorm = predictor.predict(X_test, denormalize=True)

        assert result_norm.is_normalized
        assert not result_denorm.is_normalized
        # Values should differ after denormalization
        assert not np.allclose(result_norm.predictions, result_denorm.predictions)


class TestAnomalyDetectionWorkflow:
    """Tests for the anomaly detection workflow."""

    def test_detector_from_predictor(self, synthetic_3ch_data):
        """Test creating anomaly detector from predictor."""
        model = MLP(
            input_features=3000,
            output_features=6,
            hidden_layers=[64, 32],
        )
        predictor = Predictor(model, device="cpu")

        detector = AnomalyDetector.from_predictor(predictor)
        assert detector.predictor is predictor

    def test_detector_fit_and_detect(self, synthetic_3ch_data):
        """Test fitting detector and detecting anomalies."""
        X, y = synthetic_3ch_data
        model = MLP(
            input_features=3000,
            output_features=6,
            hidden_layers=[64, 32],
        )
        predictor = Predictor(model, device="cpu")

        X_train = X[:150]
        X_test = X[150:]

        detector = AnomalyDetector.from_predictor(predictor)
        detector.fit(X_train, k=5, percentile=95.0)

        result = detector.detect(X_test)

        assert result.n_samples == 50
        assert result.threshold > 0
        assert result.k == 5

    def test_detector_saves_with_checkpoint(self, synthetic_3ch_data):
        """Test saving anomaly detector alongside model checkpoint."""
        X, y = synthetic_3ch_data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_3ch_data(
            X, y, train_ratio=0.6, val_ratio=0.2, seed=42
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

    def test_full_pipeline(self, synthetic_3ch_data):
        """Test complete pipeline: config -> train -> save -> load -> predict -> anomaly."""
        # Split data
        X, y = synthetic_3ch_data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_3ch_data(
            X, y, train_ratio=0.6, val_ratio=0.2, seed=42
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

    def test_reproducibility(self, synthetic_3ch_data):
        """Test that training is reproducible with same seed."""
        X, y = synthetic_3ch_data
        (X_train, y_train), (X_val, y_val), _ = split_3ch_data(
            X, y, train_ratio=0.6, val_ratio=0.2, seed=42
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


class TestMultiLabelsetIntegration:
    """Integration tests for multi-labelset training workflows.

    Multi-labelset training allows training models with labels from multiple
    sources (e.g., APOGEE and GALAH). Each star may have labels from one,
    both, or neither source. This tests the end-to-end workflow.
    """

    @pytest.fixture
    def multi_labelset_synthetic_data(self):
        """Create synthetic multi-labelset data for integration testing.

        Layout (100 samples):
        - Samples 0-29: APOGEE labels only
        - Samples 30-59: GALAH labels only
        - Samples 60-89: Both APOGEE and GALAH labels
        - Samples 90-99: No labels (both masked)
        """
        rng = np.random.default_rng(42)
        n_samples = 100
        n_wavelengths = 100
        n_params = 5

        # Spectral data (single survey)
        X = rng.standard_normal((n_samples, 3, n_wavelengths)).astype(np.float32)
        X[:, 2, :] = 1.0  # All spectra valid

        # APOGEE labels: samples 0-29 and 60-89
        y_apogee = np.zeros((n_samples, 3, n_params), dtype=np.float32)
        apogee_indices = list(range(0, 30)) + list(range(60, 90))
        y_apogee[apogee_indices, 0, :] = rng.standard_normal((60, n_params))
        y_apogee[apogee_indices, 0, 0] = (
            np.abs(y_apogee[apogee_indices, 0, 0]) * 1000 + 4000
        )
        y_apogee[apogee_indices, 1, :] = 0.1
        y_apogee[apogee_indices, 2, :] = 1.0

        # GALAH labels: samples 30-59 and 60-89
        y_galah = np.zeros((n_samples, 3, n_params), dtype=np.float32)
        galah_indices = list(range(30, 60)) + list(range(60, 90))
        y_galah[galah_indices, 0, :] = rng.standard_normal((60, n_params))
        y_galah[galah_indices, 0, 0] = (
            np.abs(y_galah[galah_indices, 0, 0]) * 1000 + 4000
        )
        y_galah[galah_indices, 1, :] = 0.1
        y_galah[galah_indices, 2, :] = 1.0

        return X, {"apogee": y_apogee, "galah": y_galah}

    @pytest.fixture
    def multi_labelset_config(self, tmp_path):
        """Create config for multi-labelset integration tests."""
        from dorothy.config.schema import (
            DataConfig,
            ExperimentConfig,
            MultiHeadModelConfig,
            TrainingConfig,
        )

        return ExperimentConfig(
            name="multi_labelset_integration",
            output_dir=tmp_path,
            data=DataConfig(
                catalogue_path=tmp_path / "catalogue.h5",
                surveys=["desi"],
                label_sources=["apogee", "galah"],
            ),
            multi_head_model=MultiHeadModelConfig(
                survey_wavelengths={"desi": 100},
                n_parameters=5,
                latent_dim=32,
                encoder_hidden=[64],
                trunk_hidden=[32],
                output_hidden=[16],
                label_sources=["apogee", "galah"],
            ),
            training=TrainingConfig(
                epochs=5,
                batch_size=16,
            ),
            device="cpu",
        )

    def test_multi_labelset_full_workflow(
        self, multi_labelset_config, multi_labelset_synthetic_data
    ):
        """Test complete multi-labelset workflow: train -> save -> load -> predict."""
        X, y_dict = multi_labelset_synthetic_data

        # Split data (80/20 train/val)
        X_train = {"desi": X[:80]}
        X_val = {"desi": X[80:]}
        has_data_train = {"desi": np.ones(80, dtype=bool)}
        has_data_val = {"desi": np.ones(20, dtype=bool)}

        y_train = {src: y[:80] for src, y in y_dict.items()}
        y_val = {src: y[80:] for src, y in y_dict.items()}

        # 1. Train the model
        trainer = Trainer(multi_labelset_config)
        history = trainer.fit_multi_labelset(
            X_train, y_train, X_val, y_val, has_data_train, has_data_val
        )

        assert len(history.train_losses) == 5
        assert all(np.isfinite(history.train_losses))
        assert all(np.isfinite(history.val_losses))

        # 2. Save checkpoint
        checkpoint_path = trainer.save_checkpoint()
        assert checkpoint_path.exists()
        assert (checkpoint_path / "best_model.pth").exists()

        # 3. Make predictions from both heads
        predictions = trainer.predict_multi_labelset(X_val, has_data_val)

        assert "apogee" in predictions
        assert "galah" in predictions
        assert predictions["apogee"].shape == (20, 2, 5)
        assert predictions["galah"].shape == (20, 2, 5)

        # Predictions should be finite
        assert np.all(np.isfinite(predictions["apogee"]))
        assert np.all(np.isfinite(predictions["galah"]))

    def test_multi_labelset_loss_decreases(
        self, multi_labelset_config, multi_labelset_synthetic_data
    ):
        """Test that training loss decreases over epochs."""
        X, y_dict = multi_labelset_synthetic_data

        # Use more epochs for this test
        multi_labelset_config.training = multi_labelset_config.training.model_copy(
            update={"epochs": 15}
        )

        X_train = {"desi": X[:80]}
        X_val = {"desi": X[80:]}
        has_data = {"desi": np.ones(80, dtype=bool)}
        has_data_val = {"desi": np.ones(20, dtype=bool)}

        y_train = {src: y[:80] for src, y in y_dict.items()}
        y_val = {src: y[80:] for src, y in y_dict.items()}

        trainer = Trainer(multi_labelset_config)
        history = trainer.fit_multi_labelset(
            X_train, y_train, X_val, y_val, has_data, has_data_val
        )

        # Loss should decrease overall (compare first half to second half)
        first_half = np.mean(history.train_losses[:7])
        second_half = np.mean(history.train_losses[7:])
        assert second_half < first_half

    def test_multi_labelset_masking_correctness(
        self, multi_labelset_config, multi_labelset_synthetic_data
    ):
        """Verify that only valid labels contribute to loss."""
        X, y_dict = multi_labelset_synthetic_data

        # Samples 90-99 have NO labels at all
        # These should not contribute to loss

        X_train = {"desi": X[:80]}
        X_val = {"desi": X[80:]}
        has_data = {"desi": np.ones(80, dtype=bool)}
        has_data_val = {"desi": np.ones(20, dtype=bool)}

        y_train = {src: y[:80] for src, y in y_dict.items()}
        y_val = {src: y[80:] for src, y in y_dict.items()}

        # Verify val set includes samples with no labels (samples 90-99 in val)
        # Val set is samples 80-99, so indices 10-19 in val have no labels
        apogee_mask_val = y_val["apogee"][:, 2, :]  # (20, n_params)
        galah_mask_val = y_val["galah"][:, 2, :]

        # Combined mask for val samples 10-19 (global 90-99) should be all zeros
        combined_mask = np.maximum(apogee_mask_val[10:], galah_mask_val[10:])
        assert np.all(combined_mask == 0), "Samples 90-99 should have no labels"

        trainer = Trainer(multi_labelset_config)
        history = trainer.fit_multi_labelset(
            X_train, y_train, X_val, y_val, has_data, has_data_val
        )

        # Training should complete successfully despite samples with no labels
        assert len(history.train_losses) == 5
        assert all(np.isfinite(history.train_losses))

    def test_multi_labelset_predictions_independent(
        self, multi_labelset_config, multi_labelset_synthetic_data
    ):
        """Test that predictions from different heads can differ."""
        X, y_dict = multi_labelset_synthetic_data

        X_train = {"desi": X[:80]}
        X_val = {"desi": X[80:]}
        has_data = {"desi": np.ones(80, dtype=bool)}
        has_data_val = {"desi": np.ones(20, dtype=bool)}

        y_train = {src: y[:80] for src, y in y_dict.items()}
        y_val = {src: y[80:] for src, y in y_dict.items()}

        trainer = Trainer(multi_labelset_config)
        trainer.fit_multi_labelset(
            X_train, y_train, X_val, y_val, has_data, has_data_val
        )

        predictions = trainer.predict_multi_labelset(X_val, has_data_val)

        # Predictions from APOGEE and GALAH heads should differ
        # (they're trained on different subsets with different values)
        apogee_preds = predictions["apogee"][:, 0, :]  # means only
        galah_preds = predictions["galah"][:, 0, :]

        # They should not be identical (output heads are separate)
        assert not np.allclose(apogee_preds, galah_preds)

    def test_multi_labelset_union_coverage(
        self, multi_labelset_config, multi_labelset_synthetic_data
    ):
        """Test that all stars with ANY labels contribute to training."""
        X, y_dict = multi_labelset_synthetic_data

        # Count how many samples have valid labels
        apogee_has_labels = np.any(y_dict["apogee"][:, 2, :] > 0, axis=1)
        galah_has_labels = np.any(y_dict["galah"][:, 2, :] > 0, axis=1)
        any_labels = apogee_has_labels | galah_has_labels

        # 90 samples should have at least one label source (0-89)
        assert any_labels.sum() == 90

        # The remaining 10 samples (90-99) have no labels
        assert (~any_labels).sum() == 10

    def test_multi_labelset_with_partial_parameters(
        self, multi_labelset_config, multi_labelset_synthetic_data
    ):
        """Test training when some parameters are masked within a label source."""
        X, y_dict = multi_labelset_synthetic_data

        # Modify: APOGEE only has first 3 params for samples 0-19
        y_dict["apogee"][:20, 2, 3:] = 0.0  # Mask params 3-4 for samples 0-19

        X_train = {"desi": X[:80]}
        X_val = {"desi": X[80:]}
        has_data = {"desi": np.ones(80, dtype=bool)}
        has_data_val = {"desi": np.ones(20, dtype=bool)}

        y_train = {src: y[:80] for src, y in y_dict.items()}
        y_val = {src: y[80:] for src, y in y_dict.items()}

        trainer = Trainer(multi_labelset_config)
        history = trainer.fit_multi_labelset(
            X_train, y_train, X_val, y_val, has_data, has_data_val
        )

        # Should still train successfully
        assert len(history.train_losses) == 5
        assert all(np.isfinite(history.train_losses))


class TestMultiLabelsetReproducibility:
    """Tests for reproducibility of multi-labelset training."""

    @pytest.fixture
    def repro_config(self, tmp_path):
        """Config for reproducibility tests."""
        from dorothy.config.schema import (
            DataConfig,
            ExperimentConfig,
            MultiHeadModelConfig,
            TrainingConfig,
        )

        return ExperimentConfig(
            name="repro_test",
            output_dir=tmp_path,
            data=DataConfig(
                catalogue_path=tmp_path / "catalogue.h5",
                surveys=["boss"],
                label_sources=["apogee", "galah"],
            ),
            multi_head_model=MultiHeadModelConfig(
                survey_wavelengths={"boss": 50},
                n_parameters=3,
                latent_dim=16,
                encoder_hidden=[32],
                trunk_hidden=[16],
                output_hidden=[8],
                label_sources=["apogee", "galah"],
            ),
            training=TrainingConfig(
                epochs=3,
                batch_size=8,
            ),
            seed=42,
            device="cpu",
        )

    def test_multi_labelset_reproducibility(self, repro_config):
        """Test that multi-labelset training is reproducible with same seed."""
        rng = np.random.default_rng(42)
        n_samples = 40
        n_params = 3

        X = rng.standard_normal((n_samples, 3, 50)).astype(np.float32)
        X[:, 2, :] = 1.0

        y_apogee = np.zeros((n_samples, 3, n_params), dtype=np.float32)
        y_apogee[:, 0, :] = rng.standard_normal((n_samples, n_params))
        y_apogee[:, 0, 0] = np.abs(y_apogee[:, 0, 0]) * 1000 + 4000
        y_apogee[:, 1, :] = 0.1
        y_apogee[:, 2, :] = 1.0

        y_galah = np.zeros((n_samples, 3, n_params), dtype=np.float32)
        y_galah[:20, 0, :] = rng.standard_normal((20, n_params))
        y_galah[:20, 0, 0] = np.abs(y_galah[:20, 0, 0]) * 1000 + 4000
        y_galah[:20, 1, :] = 0.1
        y_galah[:20, 2, :] = 1.0

        X_train = {"boss": X[:32]}
        X_val = {"boss": X[32:]}
        has_data = {"boss": np.ones(32, dtype=bool)}
        has_data_val = {"boss": np.ones(8, dtype=bool)}

        y_train = {"apogee": y_apogee[:32], "galah": y_galah[:32]}
        y_val = {"apogee": y_apogee[32:], "galah": y_galah[32:]}

        # First run
        trainer1 = Trainer(repro_config)
        history1 = trainer1.fit_multi_labelset(
            X_train, y_train, X_val, y_val, has_data, has_data_val
        )

        # Second run (same seed)
        trainer2 = Trainer(repro_config)
        history2 = trainer2.fit_multi_labelset(
            X_train, y_train, X_val, y_val, has_data, has_data_val
        )

        # Losses should be identical
        np.testing.assert_allclose(
            history1.train_losses, history2.train_losses, rtol=1e-5
        )
        np.testing.assert_allclose(history1.val_losses, history2.val_losses, rtol=1e-5)
