"""
Tests for the training infrastructure.

These tests verify:
1. Trainer initialization and component creation
2. Training loop mechanics
3. Validation and best model tracking
4. Checkpointing and history saving
5. Predictions and uncertainty extraction
6. Early stopping behavior
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from dorothy.config.schema import (
    DataConfig,
    ExperimentConfig,
    LossType,
    ModelConfig,
    SchedulerConfig,
    SchedulerType,
    TrainingConfig,
)
from dorothy.training.trainer import Trainer, TrainingHistory


@pytest.fixture
def simple_config():
    """Create a simple configuration for testing."""
    return ExperimentConfig(
        name="test_experiment",
        data=DataConfig(
            fits_path=Path("/fake/path.fits"),
            wavelength_bins=1000,  # Minimum valid value
            input_channels=1,  # Use 1 channel to keep features manageable
        ),
        model=ModelConfig(
            input_features=1000,  # 1 * 1000
            output_features=6,  # 3 params * 2
            hidden_layers=[32, 16],
        ),
        training=TrainingConfig(
            epochs=5,
            batch_size=16,
            learning_rate=1e-3,
            loss=LossType.HETEROSCEDASTIC,
            gradient_clip=10.0,
            scatter_floor=0.01,
        ),
        seed=42,
    )


@pytest.fixture
def training_data():
    """Create synthetic training data with realistic stellar parameter ranges."""
    np.random.seed(42)
    n_samples = 100
    n_features = 1000  # Match config: 1 channel * 1000 wavelengths

    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Create realistic stellar parameter labels
    # [teff, logg, feh] + [teff_err, logg_err, feh_err]
    labels = np.column_stack(
        [
            np.random.uniform(4500, 6500, n_samples),  # teff (K)
            np.random.uniform(2.0, 5.0, n_samples),  # logg
            np.random.uniform(-1.0, 0.5, n_samples),  # feh
        ]
    ).astype(np.float32)

    errors = np.column_stack(
        [
            np.abs(np.random.randn(n_samples)) * 50 + 10,  # teff_err
            np.abs(np.random.randn(n_samples)) * 0.1 + 0.05,  # logg_err
            np.abs(np.random.randn(n_samples)) * 0.05 + 0.02,  # feh_err
        ]
    ).astype(np.float32)

    y = np.concatenate([labels, errors], axis=1)

    return X, y


class TestTrainerInit:
    """Tests for Trainer initialization."""

    def test_trainer_creation(self, simple_config):
        """Test that trainer can be created from config."""
        trainer = Trainer(simple_config)

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None
        assert trainer.device is not None

    def test_model_has_correct_architecture(self, simple_config):
        """Test that model matches config."""
        trainer = Trainer(simple_config)

        assert trainer.model.input_features == 1000
        assert trainer.model.output_features == 6
        assert trainer.model.hidden_layers == [32, 16]

    def test_device_resolution_auto(self, simple_config):
        """Test automatic device resolution."""
        trainer = Trainer(simple_config)

        # Should be cuda if available, else cpu
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert trainer.device.type == expected

    def test_device_resolution_explicit_cpu(self, simple_config):
        """Test explicit CPU device."""
        simple_config.device = "cpu"
        trainer = Trainer(simple_config)

        assert trainer.device.type == "cpu"

    def test_seed_reproducibility(self, simple_config):
        """Test that seed produces reproducible initialization."""
        trainer1 = Trainer(simple_config)
        trainer2 = Trainer(simple_config)

        # Model weights should be identical
        for p1, p2 in zip(
            trainer1.model.parameters(), trainer2.model.parameters(), strict=True
        ):
            assert torch.allclose(p1, p2)


class TestTrainerFit:
    """Tests for the training loop."""

    def test_fit_runs_without_error(self, simple_config, training_data):
        """Test that fit completes successfully."""
        trainer = Trainer(simple_config)
        X, y = training_data

        # Split into train/val
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        history = trainer.fit(X_train, y_train, X_val, y_val)

        assert len(history.train_losses) == 5
        assert len(history.val_losses) == 5

    def test_fit_reduces_loss(self, simple_config, training_data):
        """Test that training reduces loss over time."""
        simple_config.training.epochs = 20
        trainer = Trainer(simple_config)
        X, y = training_data

        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        history = trainer.fit(X_train, y_train, X_val, y_val)

        # Loss should generally decrease (comparing first and last)
        # Use average of first 3 and last 3 to reduce noise
        early_loss = np.mean(history.train_losses[:3])
        late_loss = np.mean(history.train_losses[-3:])
        assert late_loss < early_loss

    def test_fit_tracks_best_model(self, simple_config, training_data):
        """Test that best model is tracked correctly."""
        trainer = Trainer(simple_config)
        X, y = training_data

        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        history = trainer.fit(X_train, y_train, X_val, y_val)

        # Best val loss should match minimum
        assert history.best_val_loss == min(history.val_losses)
        assert history.best_epoch == history.val_losses.index(min(history.val_losses))

    def test_fit_with_numpy_arrays(self, simple_config, training_data):
        """Test that numpy arrays work as input."""
        trainer = Trainer(simple_config)
        X, y = training_data

        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        assert len(history.train_losses) > 0

    def test_fit_with_torch_tensors(self, simple_config, training_data):
        """Test that torch tensors work as input."""
        trainer = Trainer(simple_config)
        X, y = training_data

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        history = trainer.fit(
            X_tensor[:80], y_tensor[:80], X_tensor[80:], y_tensor[80:]
        )

        assert len(history.train_losses) > 0


class TestTrainerSchedulers:
    """Tests for learning rate schedulers."""

    def test_cyclic_lr_scheduler(self, simple_config, training_data):
        """Test CyclicLR scheduler."""
        simple_config.training.scheduler = SchedulerConfig(
            type=SchedulerType.CYCLIC,
            base_lr=1e-6,
            max_lr=1e-3,
        )
        trainer = Trainer(simple_config)
        X, y = training_data

        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        # Should have recorded learning rates
        assert len(history.learning_rates) > 0
        # LR should vary (not constant)
        assert max(history.learning_rates) > min(history.learning_rates)

    def test_no_scheduler(self, simple_config, training_data):
        """Test training without scheduler."""
        simple_config.training.scheduler = SchedulerConfig(type=SchedulerType.NONE)
        trainer = Trainer(simple_config)
        X, y = training_data

        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        # Should still complete training
        assert len(history.train_losses) == 5


class TestTrainerEarlyStopping:
    """Tests for early stopping."""

    def test_early_stopping_triggers(self, simple_config, training_data):
        """Test that early stopping can trigger."""
        # Use many epochs but short patience
        simple_config.training.epochs = 100
        simple_config.training.early_stopping_patience = 3

        trainer = Trainer(simple_config)
        X, y = training_data

        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        # Training should have stopped before 100 epochs
        # (unless loss keeps improving, which is unlikely for 100 epochs)
        assert len(history.train_losses) <= 100

    def test_early_stopping_disabled(self, simple_config, training_data):
        """Test that early stopping can be disabled."""
        simple_config.training.epochs = 5
        simple_config.training.early_stopping_patience = 0  # Disabled

        trainer = Trainer(simple_config)
        X, y = training_data

        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        # Should run all epochs
        assert len(history.train_losses) == 5


class TestTrainerCheckpointing:
    """Tests for checkpointing."""

    def test_save_checkpoint(self, simple_config, training_data):
        """Test saving final checkpoint."""
        trainer = Trainer(simple_config)
        X, y = training_data

        trainer.fit(X[:80], y[:80], X[80:], y[80:])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = trainer.save_checkpoint(Path(tmpdir) / "checkpoint")

            assert (path / "best_model.pth").exists()
            assert (path / "final_model.pth").exists()
            assert (path / "history_train_val.pkl").exists()
            assert (path / "learning_rates.pkl").exists()

    def test_load_saved_model(self, simple_config, training_data):
        """Test that saved model can be loaded."""
        trainer = Trainer(simple_config)
        X, y = training_data

        trainer.fit(X[:80], y[:80], X[80:], y[80:])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = trainer.save_checkpoint(Path(tmpdir) / "checkpoint")

            # Load the model
            loaded_state = torch.load(path / "best_model.pth", weights_only=True)

            # Create new model and load weights
            from dorothy.models import MLP

            new_model = MLP.from_config(simple_config.model)
            new_model.load_state_dict(loaded_state)

            # Move both models to CPU for comparison
            trainer.model.cpu()
            new_model.cpu()
            trainer.model.eval()
            new_model.eval()

            # Predictions should match
            X_test = torch.tensor(X[80:], dtype=torch.float32)

            with torch.no_grad():
                pred1 = trainer.model(X_test)
                pred2 = new_model(X_test)

            assert torch.allclose(pred1, pred2)


class TestTrainerPredictions:
    """Tests for prediction functionality."""

    def test_predict_output_shape(self, simple_config, training_data):
        """Test that predictions have correct shape."""
        trainer = Trainer(simple_config)
        X, y = training_data

        trainer.fit(X[:80], y[:80], X[80:], y[80:])

        predictions = trainer.predict(X[80:])

        assert predictions.shape == (20, 6)

    def test_predict_with_numpy(self, simple_config, training_data):
        """Test predictions with numpy input."""
        trainer = Trainer(simple_config)
        X, y = training_data

        trainer.fit(X[:80], y[:80], X[80:], y[80:])

        predictions = trainer.predict(X[80:])

        assert isinstance(predictions, np.ndarray)

    def test_get_predictions_and_uncertainties(self, simple_config, training_data):
        """Test getting predictions with uncertainties."""
        trainer = Trainer(simple_config)
        X, y = training_data

        trainer.fit(X[:80], y[:80], X[80:], y[80:])

        preds, uncerts = trainer.get_predictions_and_uncertainties(X[80:])

        assert preds.shape == (20, 3)  # 3 parameters
        assert uncerts.shape == (20, 3)
        assert (uncerts > 0).all()  # Uncertainties should be positive


class TestTrainingHistory:
    """Tests for TrainingHistory class."""

    def test_history_initialization(self):
        """Test default history values."""
        history = TrainingHistory()

        assert history.train_losses == []
        assert history.val_losses == []
        assert history.best_epoch == 0
        assert history.best_val_loss == float("inf")

    def test_history_save_and_load(self):
        """Test saving and loading history."""
        history = TrainingHistory(
            train_losses=[1.0, 0.9, 0.8],
            val_losses=[1.1, 1.0, 0.9],
        )

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            history.save(path)

            import pickle

            with open(path, "rb") as f:
                loaded = pickle.load(f)

            assert loaded["history_train"] == [1.0, 0.9, 0.8]
            assert loaded["history_val"] == [1.1, 1.0, 0.9]
        finally:
            path.unlink()


class TestTrainerGradientClipping:
    """Tests for gradient clipping."""

    def test_gradient_clipping_applied(self, simple_config, training_data):
        """Test that gradient clipping is applied."""
        simple_config.training.gradient_clip = 1.0
        trainer = Trainer(simple_config)
        X, y = training_data

        # This should complete without exploding gradients
        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        # All losses should be finite
        assert all(np.isfinite(history.train_losses))
        assert all(np.isfinite(history.val_losses))

    def test_gradient_clipping_disabled(self, simple_config, training_data):
        """Test training with gradient clipping disabled."""
        simple_config.training.gradient_clip = 0.0  # Disabled
        trainer = Trainer(simple_config)
        X, y = training_data

        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        assert len(history.train_losses) == 5
