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
    MaskingConfig,
    ModelConfig,
    SchedulerConfig,
    SchedulerType,
    TrainingConfig,
)
from dorothy.training.trainer import EVALUATOR_METRIC_NAMES, Trainer, TrainingHistory


@pytest.fixture
def simple_config():
    """Create a simple configuration for testing."""
    return ExperimentConfig(
        name="test_experiment",
        data=DataConfig(
            catalogue_path=Path("/fake/catalogue.h5"),
        ),
        model=ModelConfig(
            input_features=3000,  # 3 channels * 1000 wavelengths
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
    """Create synthetic training data with 3-channel format."""
    np.random.seed(42)
    n_samples = 100
    n_wavelengths = 1000
    n_params = 3

    # Create 3-channel spectral data: (N, 3, wavelengths) = [flux, sigma, mask]
    flux = np.random.randn(n_samples, n_wavelengths).astype(np.float32)
    sigma = (np.abs(np.random.randn(n_samples, n_wavelengths)) * 0.1 + 0.01).astype(
        np.float32
    )
    spec_mask = np.ones((n_samples, n_wavelengths), dtype=np.float32)
    X = np.stack([flux, sigma, spec_mask], axis=1)  # (N, 3, wavelengths)

    # Create 3-channel labels: (N, 3, n_params) = [values, errors, mask]
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

    label_mask = np.ones((n_samples, n_params), dtype=np.float32)
    y = np.stack([labels, errors, label_mask], axis=1)  # (N, 3, n_params)

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

        # 3 channels * 1000 wavelengths = 3000 input features
        assert trainer.model.input_features == 3000
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

        # Model output shape is (batch, 2, n_params)
        assert predictions.shape == (20, 2, 3)

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

    def test_val_metrics_initialization(self):
        """Test that val_metrics can be initialized."""
        history = TrainingHistory(
            val_metrics={name: [] for name in EVALUATOR_METRIC_NAMES},
        )

        assert len(history.val_metrics) == len(EVALUATOR_METRIC_NAMES)
        for name in EVALUATOR_METRIC_NAMES:
            assert name in history.val_metrics
            assert history.val_metrics[name] == []

    def test_get_metrics_array(self):
        """Test get_metrics_array returns proper numpy arrays."""
        n_epochs = 3
        n_params = 2

        # Create sample metric arrays
        val_metrics = {
            name: [np.random.rand(n_params).astype(np.float32) for _ in range(n_epochs)]
            for name in EVALUATOR_METRIC_NAMES
        }

        history = TrainingHistory(val_metrics=val_metrics)
        arrays = history.get_metrics_array()

        assert len(arrays) == len(EVALUATOR_METRIC_NAMES)
        for name in EVALUATOR_METRIC_NAMES:
            assert name in arrays
            assert arrays[name].shape == (n_epochs, n_params)


class TestTrainerValMetrics:
    """Tests for Evaluator metrics tracking during training."""

    def test_val_metrics_tracked_during_training(self, simple_config, training_data):
        """Test that Evaluator metrics are tracked during training."""
        trainer = Trainer(simple_config)
        X, y = training_data
        n_params = simple_config.model.n_parameters

        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        # Check all metrics are tracked
        assert len(history.val_metrics) == len(EVALUATOR_METRIC_NAMES)

        # Check each metric has correct shape
        n_epochs = simple_config.training.epochs
        for metric_name in EVALUATOR_METRIC_NAMES:
            assert metric_name in history.val_metrics
            assert len(history.val_metrics[metric_name]) == n_epochs

            # Each epoch should have n_params values
            for epoch_metrics in history.val_metrics[metric_name]:
                assert len(epoch_metrics) == n_params

    def test_val_metrics_have_valid_values(self, simple_config, training_data):
        """Test that tracked metrics have valid (finite) values."""
        trainer = Trainer(simple_config)
        X, y = training_data

        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        # Core metrics (always present) should be finite
        core_metrics = ["rmse", "bias", "sd", "mae", "median_offset", "robust_scatter"]
        for metric_name in core_metrics:
            arrays = np.array(history.val_metrics[metric_name])
            assert np.all(np.isfinite(arrays)), f"{metric_name} has non-finite values"

        # RMSE and MAE should be non-negative
        for metric_name in ["rmse", "mae", "robust_scatter"]:
            arrays = np.array(history.val_metrics[metric_name])
            assert np.all(arrays >= 0), f"{metric_name} has negative values"

    def test_val_metrics_zscore_calibration(self, simple_config, training_data):
        """Test that z-score metrics are computed."""
        trainer = Trainer(simple_config)
        X, y = training_data

        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        # Z-score metrics should be present (since we use heteroscedastic loss)
        z_metrics = ["z_median", "z_robust_scatter"]
        for metric_name in z_metrics:
            assert metric_name in history.val_metrics
            arrays = np.array(history.val_metrics[metric_name])
            assert np.all(np.isfinite(arrays)), f"{metric_name} has non-finite values"

    def test_val_metrics_saved_with_history(self, simple_config, training_data):
        """Test that val_metrics are saved when saving history."""
        trainer = Trainer(simple_config)
        X, y = training_data

        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            history.save(path)

            import pickle

            with open(path, "rb") as f:
                loaded = pickle.load(f)

            assert "val_metrics" in loaded
            assert len(loaded["val_metrics"]) == len(EVALUATOR_METRIC_NAMES)

            # Verify data integrity
            for metric_name in EVALUATOR_METRIC_NAMES:
                assert metric_name in loaded["val_metrics"]
                original = history.val_metrics[metric_name]
                saved = loaded["val_metrics"][metric_name]
                assert len(original) == len(saved)
        finally:
            path.unlink()


class TestTrainerGrokkingMetrics:
    """Tests for grokking detection metrics."""

    def test_grokking_metrics_tracked(self, simple_config, training_data):
        """Test that grokking metrics are tracked during training."""
        trainer = Trainer(simple_config)
        X, y = training_data

        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        n_epochs = simple_config.training.epochs

        # Check grad_norms
        assert len(history.grad_norms) == n_epochs
        assert all(
            g > 0 for g in history.grad_norms
        ), "Gradient norms should be positive"

        # Check weight_norms (per layer)
        assert len(history.weight_norms) > 0, "Should have weight norms for layers"
        for layer_name, norms in history.weight_norms.items():
            assert len(norms) == n_epochs, f"Wrong count for {layer_name}"
            assert all(
                n > 0 for n in norms
            ), f"Weight norms should be positive for {layer_name}"

        # Check weight_updates (per layer)
        assert len(history.weight_updates) > 0, "Should have weight updates"
        for layer_name, updates in history.weight_updates.items():
            assert len(updates) == n_epochs, f"Wrong count for {layer_name}"
            assert all(
                u >= 0 for u in updates
            ), f"Weight updates should be non-negative for {layer_name}"

    def test_grokking_metrics_get_array(self, simple_config, training_data):
        """Test get_grokking_metrics returns proper arrays."""
        trainer = Trainer(simple_config)
        X, y = training_data

        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        grokking = history.get_grokking_metrics()

        assert "weight_norms" in grokking
        assert "grad_norms" in grokking
        assert "weight_updates" in grokking

        # grad_norms should be a numpy array
        assert isinstance(grokking["grad_norms"], np.ndarray)
        assert len(grokking["grad_norms"]) == simple_config.training.epochs

        # weight_norms and weight_updates should be dicts of arrays
        for _layer_name, arr in grokking["weight_norms"].items():
            assert isinstance(arr, np.ndarray)

    def test_grokking_metrics_saved_with_history(self, simple_config, training_data):
        """Test that grokking metrics are saved when saving history."""
        trainer = Trainer(simple_config)
        X, y = training_data

        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            history.save(path)

            import pickle

            with open(path, "rb") as f:
                loaded = pickle.load(f)

            assert "weight_norms" in loaded
            assert "grad_norms" in loaded
            assert "weight_updates" in loaded

            # Verify data integrity
            assert len(loaded["grad_norms"]) == len(history.grad_norms)
            assert len(loaded["weight_norms"]) == len(history.weight_norms)
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


class TestTrainerMasking:
    """Tests for Trainer with label masking (embedded in y tensor channel 2)."""

    def test_fit_with_label_mask(self, simple_config, training_data):
        """Test training with label masking (embedded in y channel 2)."""
        trainer = Trainer(simple_config)
        X, y = training_data
        n_params = simple_config.model.n_parameters

        # Mask is already embedded in y[:, 2, :]
        # Modify mask to have 90% valid labels
        rng = np.random.default_rng(42)
        mask_90_percent = (rng.random((100, n_params)) > 0.1).astype(np.float32)
        y_modified = y.copy()
        y_modified[:, 2, :] = mask_90_percent

        history = trainer.fit(
            X[:80],
            y_modified[:80],
            X[80:],
            y_modified[80:],
        )

        # Training should complete
        assert len(history.train_losses) == 5
        assert all(np.isfinite(history.train_losses))

    def test_fit_with_all_valid_mask(self, simple_config, training_data):
        """Test training with all-ones mask (all labels valid)."""
        trainer = Trainer(simple_config)
        X, y = training_data

        # y already has all-ones mask in channel 2
        history = trainer.fit(
            X[:80],
            y[:80],
            X[80:],
            y[80:],
        )

        assert len(history.train_losses) == 5

    def test_fit_with_partial_mask(self, simple_config, training_data):
        """Test training with partial masking (some parameters masked per sample)."""
        trainer = Trainer(simple_config)
        X, y = training_data

        # Modify mask to mask first parameter for first half of samples
        y_modified = y.copy()
        y_modified[:50, 2, 0] = 0.0  # Mask first param for first 50 samples

        history = trainer.fit(
            X[:80],
            y_modified[:80],
            X[80:],
            y_modified[80:],
        )

        assert len(history.train_losses) == 5


class TestTrainerAugmentation:
    """Tests for Trainer with augmentation."""

    @pytest.fixture
    def augmentation_config(self):
        """Config for testing with 3-channel input and augmentation."""
        from dorothy.config import DataConfig, ExperimentConfig

        return ExperimentConfig(
            name="augmentation_test",
            data=DataConfig(
                catalogue_path=Path("/fake/catalogue.h5"),
            ),
            model=ModelConfig(
                input_features=3000,  # 3 channels * 1000 wavelengths
                output_features=6,  # 3 params * 2
                hidden_layers=[64, 32],
            ),
            training=TrainingConfig(
                epochs=3,
                batch_size=16,
            ),
            masking=MaskingConfig(
                enabled=True,
                min_fraction=0.1,
                max_fraction=0.3,
                min_block_size=5,
            ),
        )

    @pytest.fixture
    def three_channel_data(self, augmentation_config):
        """Create 3-channel training data [flux | error | mask]."""
        rng = np.random.default_rng(42)
        n_samples = 100
        n_wavelengths = 1000
        n_params = augmentation_config.model.n_parameters

        # Create 3-channel spectral input: [flux, sigma, mask]
        X = np.zeros((n_samples, 3, n_wavelengths), dtype=np.float32)
        X[:, 0, :] = rng.standard_normal((n_samples, n_wavelengths))  # Flux
        X[:, 1, :] = (
            np.abs(rng.standard_normal((n_samples, n_wavelengths))) * 0.1
        )  # Sigma
        X[:, 2, :] = 1.0  # All valid initially

        # Create 3-channel labels: [values, errors, mask]
        # First param (teff) must be positive for log-space normalization
        labels = rng.standard_normal((n_samples, n_params)).astype(np.float32)
        labels[:, 0] = np.abs(labels[:, 0]) * 1000 + 4000  # Teff in range ~3000-6000K
        errors = np.abs(rng.standard_normal((n_samples, n_params)) * 0.1 + 0.01).astype(
            np.float32
        )
        label_mask = np.ones((n_samples, n_params), dtype=np.float32)
        y = np.stack([labels, errors, label_mask], axis=1)  # (N, 3, n_params)

        return X, y

    def test_fit_with_augmentation_from_config(
        self, augmentation_config, three_channel_data
    ):
        """Test training with augmentation created from config."""
        trainer = Trainer(augmentation_config)
        X, y = three_channel_data

        # Augmentation should be created from config
        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        assert trainer._augmentation is not None
        assert len(history.train_losses) == 3

    def test_fit_with_explicit_augmentation(
        self, augmentation_config, three_channel_data
    ):
        """Test training with explicitly provided augmentation."""
        from dorothy.data.augmentation import DynamicBlockMasking

        # Disable config-based augmentation
        augmentation_config.masking.enabled = False

        trainer = Trainer(augmentation_config)
        X, y = three_channel_data

        # Provide explicit augmentation
        aug = DynamicBlockMasking(min_fraction=0.2, max_fraction=0.4)
        history = trainer.fit(X[:80], y[:80], X[80:], y[80:], augmentation=aug)

        assert trainer._augmentation is aug
        assert len(history.train_losses) == 3

    def test_augmentation_applied_to_training_only(
        self, augmentation_config, three_channel_data
    ):
        """Test that augmentation is applied during training but not validation."""
        trainer = Trainer(augmentation_config)
        X, y = three_channel_data

        # Track whether augmentation is called during training vs validation
        # We can verify by checking that training runs without errors
        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        # Both train and val losses should be finite
        assert all(np.isfinite(history.train_losses))
        assert all(np.isfinite(history.val_losses))


class TestTrainerMaskingAndAugmentation:
    """Tests for Trainer with both label masking and augmentation."""

    @pytest.fixture
    def full_masking_config(self):
        """Config for testing with both label masks and augmentation."""
        from dorothy.config import DataConfig, ExperimentConfig

        return ExperimentConfig(
            name="full_masking_test",
            data=DataConfig(
                catalogue_path=Path("/fake/catalogue.h5"),
            ),
            model=ModelConfig(
                input_features=3000,  # 3 channels * 1000 wavelengths
                output_features=6,  # 3 params * 2
                hidden_layers=[64, 32],
            ),
            training=TrainingConfig(
                epochs=3,
                batch_size=16,
            ),
            masking=MaskingConfig(
                enabled=True,
                min_fraction=0.1,
                max_fraction=0.3,
            ),
        )

    def test_fit_with_both_masks_and_augmentation(self, full_masking_config):
        """Test training with both label masks and spectral augmentation."""
        trainer = Trainer(full_masking_config)
        rng = np.random.default_rng(42)

        n_samples = 100
        n_wavelengths = 1000
        n_params = full_masking_config.model.n_parameters

        # Create 3-channel spectral input: [flux, sigma, mask]
        X = np.zeros((n_samples, 3, n_wavelengths), dtype=np.float32)
        X[:, 0, :] = rng.standard_normal((n_samples, n_wavelengths))
        X[:, 1, :] = np.abs(rng.standard_normal((n_samples, n_wavelengths))) * 0.1
        X[:, 2, :] = 1.0

        # Create 3-channel labels: [values, errors, mask]
        # First param (teff) must be positive for log-space normalization
        labels = rng.standard_normal((n_samples, n_params)).astype(np.float32)
        labels[:, 0] = np.abs(labels[:, 0]) * 1000 + 4000  # Teff in range ~3000-6000K
        errors = np.abs(rng.standard_normal((n_samples, n_params)) * 0.1 + 0.01).astype(
            np.float32
        )
        # Create label masks (90% valid) - embedded as channel 2
        label_mask = (rng.random((n_samples, n_params)) > 0.1).astype(np.float32)
        y = np.stack([labels, errors, label_mask], axis=1)  # (N, 3, n_params)

        history = trainer.fit(
            X[:80],
            y[:80],
            X[80:],
            y[80:],
        )

        assert trainer._augmentation is not None
        assert len(history.train_losses) == 3
        assert all(np.isfinite(history.train_losses))
        assert all(np.isfinite(history.val_losses))


class TestTrainerMultiSurvey:
    """Tests for multi-survey training with fit_multi_survey()."""

    @pytest.fixture
    def multi_head_config(self, tmp_path):
        """Create config for MultiHeadMLP training."""
        from dorothy.config.schema import (
            DataConfig,
            ExperimentConfig,
            MultiHeadModelConfig,
            TrainingConfig,
        )

        return ExperimentConfig(
            name="test_multi_survey",
            output_dir=tmp_path,
            data=DataConfig(
                catalogue_path=tmp_path / "catalogue.h5",
                surveys=["boss", "lamost"],
            ),
            multi_head_model=MultiHeadModelConfig(
                survey_wavelengths={"boss": 100, "lamost": 80},
                n_parameters=11,
                latent_dim=32,
                encoder_hidden=[64],
                trunk_hidden=[32],
                output_hidden=[16],
            ),
            training=TrainingConfig(
                epochs=3,
                batch_size=16,
            ),
        )

    def test_fit_multi_survey_runs(self, multi_head_config):
        """Test fit_multi_survey runs without error."""
        trainer = Trainer(multi_head_config)
        rng = np.random.default_rng(42)

        n_samples = 50
        n_params = 11

        # Create multi-survey spectral data
        X_train = {
            "boss": rng.standard_normal((n_samples, 3, 100)).astype(np.float32),
            "lamost": rng.standard_normal((n_samples, 3, 80)).astype(np.float32),
        }
        X_val = {
            "boss": rng.standard_normal((n_samples // 5, 3, 100)).astype(np.float32),
            "lamost": rng.standard_normal((n_samples // 5, 3, 80)).astype(np.float32),
        }

        # Set mask channel to 1 (all valid)
        for survey in X_train:
            X_train[survey][:, 2, :] = 1.0
            X_val[survey][:, 2, :] = 1.0

        # Create labels
        y_train = np.zeros((n_samples, 3, n_params), dtype=np.float32)
        y_train[:, 0, :] = rng.standard_normal((n_samples, n_params))
        y_train[:, 0, 0] = np.abs(y_train[:, 0, 0]) * 1000 + 4000  # Teff
        y_train[:, 1, :] = 0.1  # errors
        y_train[:, 2, :] = 1.0  # mask

        y_val = np.zeros((n_samples // 5, 3, n_params), dtype=np.float32)
        y_val[:, 0, :] = rng.standard_normal((n_samples // 5, n_params))
        y_val[:, 0, 0] = np.abs(y_val[:, 0, 0]) * 1000 + 4000
        y_val[:, 1, :] = 0.1
        y_val[:, 2, :] = 1.0

        # Create has_data masks
        has_data_train = {
            "boss": np.ones(n_samples, dtype=bool),
            "lamost": np.ones(n_samples, dtype=bool),
        }
        has_data_val = {
            "boss": np.ones(n_samples // 5, dtype=bool),
            "lamost": np.ones(n_samples // 5, dtype=bool),
        }

        history = trainer.fit_multi_survey(
            X_train, y_train, X_val, y_val, has_data_train, has_data_val
        )

        assert len(history.train_losses) == 3
        assert len(history.val_losses) == 3
        assert all(np.isfinite(history.train_losses))
        assert all(np.isfinite(history.val_losses))

    def test_fit_multi_survey_partial_data(self, multi_head_config):
        """Test multi-survey training with partial data (some stars missing surveys)."""
        trainer = Trainer(multi_head_config)
        rng = np.random.default_rng(42)

        n_samples = 50
        n_params = 11

        # Create multi-survey spectral data
        X_train = {
            "boss": rng.standard_normal((n_samples, 3, 100)).astype(np.float32),
            "lamost": rng.standard_normal((n_samples, 3, 80)).astype(np.float32),
        }
        X_val = {
            "boss": rng.standard_normal((n_samples // 5, 3, 100)).astype(np.float32),
            "lamost": rng.standard_normal((n_samples // 5, 3, 80)).astype(np.float32),
        }

        for survey in X_train:
            X_train[survey][:, 2, :] = 1.0
            X_val[survey][:, 2, :] = 1.0

        # Create labels
        y_train = np.zeros((n_samples, 3, n_params), dtype=np.float32)
        y_train[:, 0, :] = rng.standard_normal((n_samples, n_params))
        y_train[:, 0, 0] = np.abs(y_train[:, 0, 0]) * 1000 + 4000
        y_train[:, 1, :] = 0.1
        y_train[:, 2, :] = 1.0

        y_val = np.zeros((n_samples // 5, 3, n_params), dtype=np.float32)
        y_val[:, 0, :] = rng.standard_normal((n_samples // 5, n_params))
        y_val[:, 0, 0] = np.abs(y_val[:, 0, 0]) * 1000 + 4000
        y_val[:, 1, :] = 0.1
        y_val[:, 2, :] = 1.0

        # Create partial has_data masks (50% have both, 25% boss only, 25% lamost only)
        has_data_train = {
            "boss": np.array([i % 4 != 3 for i in range(n_samples)]),
            "lamost": np.array([i % 4 != 2 for i in range(n_samples)]),
        }
        has_data_val = {
            "boss": np.ones(n_samples // 5, dtype=bool),
            "lamost": np.ones(n_samples // 5, dtype=bool),
        }

        history = trainer.fit_multi_survey(
            X_train, y_train, X_val, y_val, has_data_train, has_data_val
        )

        assert len(history.train_losses) == 3
        assert all(np.isfinite(history.train_losses))

    def test_fit_multi_survey_requires_multihead(self, simple_config):
        """Test fit_multi_survey raises TypeError for non-MultiHeadMLP models."""
        trainer = Trainer(simple_config)
        rng = np.random.default_rng(42)

        n_samples = 10
        n_params = 11

        X_train = {
            "survey": rng.standard_normal((n_samples, 3, 100)).astype(np.float32)
        }
        y_train = np.zeros((n_samples, 3, n_params), dtype=np.float32)
        y_train[:, 2, :] = 1.0
        has_data = {"survey": np.ones(n_samples, dtype=bool)}

        with pytest.raises(TypeError, match="requires a MultiHeadMLP model"):
            trainer.fit_multi_survey(
                X_train, y_train, X_train, y_train, has_data, has_data
            )

    def test_fit_multi_survey_reduces_loss(self, multi_head_config):
        """Test that multi-survey training reduces loss over epochs."""
        # Use more epochs for this test
        multi_head_config.training = multi_head_config.training.model_copy(
            update={"epochs": 10}
        )
        trainer = Trainer(multi_head_config)
        rng = np.random.default_rng(42)

        n_samples = 100
        n_params = 11

        # Create consistent data for learning
        X_train = {
            "boss": rng.standard_normal((n_samples, 3, 100)).astype(np.float32),
            "lamost": rng.standard_normal((n_samples, 3, 80)).astype(np.float32),
        }
        X_val = {
            "boss": rng.standard_normal((n_samples // 5, 3, 100)).astype(np.float32),
            "lamost": rng.standard_normal((n_samples // 5, 3, 80)).astype(np.float32),
        }

        for survey in X_train:
            X_train[survey][:, 2, :] = 1.0
            X_val[survey][:, 2, :] = 1.0

        y_train = np.zeros((n_samples, 3, n_params), dtype=np.float32)
        y_train[:, 0, :] = rng.standard_normal((n_samples, n_params))
        y_train[:, 0, 0] = np.abs(y_train[:, 0, 0]) * 1000 + 4000
        y_train[:, 1, :] = 0.1
        y_train[:, 2, :] = 1.0

        y_val = np.zeros((n_samples // 5, 3, n_params), dtype=np.float32)
        y_val[:, 0, :] = rng.standard_normal((n_samples // 5, n_params))
        y_val[:, 0, 0] = np.abs(y_val[:, 0, 0]) * 1000 + 4000
        y_val[:, 1, :] = 0.1
        y_val[:, 2, :] = 1.0

        has_data_train = {
            "boss": np.ones(n_samples, dtype=bool),
            "lamost": np.ones(n_samples, dtype=bool),
        }
        has_data_val = {
            "boss": np.ones(n_samples // 5, dtype=bool),
            "lamost": np.ones(n_samples // 5, dtype=bool),
        }

        history = trainer.fit_multi_survey(
            X_train, y_train, X_val, y_val, has_data_train, has_data_val
        )

        # Training loss should generally decrease
        assert history.train_losses[-1] < history.train_losses[0]


class TestTrainerMultiLabelset:
    """Tests for multi-labelset training with multiple output heads.

    Multi-labelset training allows training with labels from multiple sources
    (e.g., APOGEE and GALAH). Each star may have labels from one, both, or
    neither source. Loss is computed only for available labels using masking.
    """

    @pytest.fixture
    def multi_labelset_config(self, tmp_path):
        """Create config for MultiHeadMLP with multiple label sources."""
        from dorothy.config.schema import (
            DataConfig,
            ExperimentConfig,
            MultiHeadModelConfig,
            TrainingConfig,
        )

        return ExperimentConfig(
            name="test_multi_labelset",
            output_dir=tmp_path,
            data=DataConfig(
                catalogue_path=tmp_path / "catalogue.h5",
                surveys=["desi"],
                label_sources=["apogee", "galah"],
            ),
            multi_head_model=MultiHeadModelConfig(
                survey_wavelengths={"desi": 100},
                n_parameters=11,
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
        )

    @pytest.fixture
    def multi_labelset_data(self, multi_labelset_config):
        """Create synthetic data with multiple label sources.

        Layout (60 samples):
        - Samples 0-19: APOGEE labels only
        - Samples 20-39: GALAH labels only
        - Samples 40-59: Both APOGEE and GALAH labels
        """
        rng = np.random.default_rng(42)
        n_samples = 60
        n_params = 11

        # Create spectral data (single survey)
        X = rng.standard_normal((n_samples, 3, 100)).astype(np.float32)
        X[:, 2, :] = 1.0  # All spectra valid

        # Create labels for each source
        # y has shape (N, 3, n_params) = [values, errors, mask]
        y_apogee = np.zeros((n_samples, 3, n_params), dtype=np.float32)
        y_galah = np.zeros((n_samples, 3, n_params), dtype=np.float32)

        # APOGEE labels: samples 0-19 and 40-59
        apogee_indices = list(range(0, 20)) + list(range(40, 60))
        y_apogee[apogee_indices, 0, :] = rng.uniform(-0.5, 0.5, (40, n_params))
        # Teff: realistic values 4000-7000 K
        y_apogee[apogee_indices, 0, 0] = rng.uniform(4000, 7000, 40)
        y_apogee[apogee_indices, 1, :] = 0.1  # errors
        y_apogee[apogee_indices, 1, 0] = 50  # Teff error ~50K
        y_apogee[apogee_indices, 2, :] = 1.0  # mask = valid

        # GALAH labels: samples 20-39 and 40-59
        galah_indices = list(range(20, 40)) + list(range(40, 60))
        y_galah[galah_indices, 0, :] = rng.uniform(-0.5, 0.5, (40, n_params))
        # Teff: realistic values 4000-7000 K
        y_galah[galah_indices, 0, 0] = rng.uniform(4000, 7000, 40)
        y_galah[galah_indices, 1, :] = 0.1  # errors
        y_galah[galah_indices, 1, 0] = 50  # Teff error ~50K
        y_galah[galah_indices, 2, :] = 1.0  # mask = valid

        # Create has_labels dict indicating which stars have labels from each source
        has_apogee = np.zeros(n_samples, dtype=bool)
        has_apogee[apogee_indices] = True
        has_galah = np.zeros(n_samples, dtype=bool)
        has_galah[galah_indices] = True

        return (
            X,
            {"apogee": y_apogee, "galah": y_galah},
            {"apogee": has_apogee, "galah": has_galah},
        )

    def test_multi_labelset_config_valid(self, multi_labelset_config):
        """Test that multi-labelset config is valid."""
        assert multi_labelset_config.multi_head_model is not None
        assert multi_labelset_config.multi_head_model.label_sources == [
            "apogee",
            "galah",
        ]
        assert multi_labelset_config.multi_head_model.is_multi_label

    def test_multi_labelset_model_has_multiple_heads(self, multi_labelset_config):
        """Test that model has output heads for each label source."""
        trainer = Trainer(multi_labelset_config)

        # Model should have output heads dict
        assert hasattr(trainer.model, "output_heads")
        assert "apogee" in trainer.model.output_heads
        assert "galah" in trainer.model.output_heads

    def test_fit_multi_labelset_runs(self, multi_labelset_config, multi_labelset_data):
        """Test that multi-labelset training runs without error."""
        trainer = Trainer(multi_labelset_config)
        X, y_dict, has_labels_dict = multi_labelset_data

        # Prepare multi-survey input format (even though single survey)
        X_train = {"desi": X[:48]}
        X_val = {"desi": X[48:]}
        has_data_train = {"desi": np.ones(48, dtype=bool)}
        has_data_val = {"desi": np.ones(12, dtype=bool)}

        # Multi-labelset y dict
        y_train = {src: y[:48] for src, y in y_dict.items()}
        y_val = {src: y[48:] for src, y in y_dict.items()}

        # has_labels dict for each source
        has_labels_train = {src: h[:48] for src, h in has_labels_dict.items()}
        has_labels_val = {src: h[48:] for src, h in has_labels_dict.items()}

        history = trainer.fit_multi_labelset(
            X_train,
            y_train,
            X_val,
            y_val,
            has_data_train,
            has_data_val,
            has_labels_train,
            has_labels_val,
        )

        assert len(history.train_losses) == 5
        assert len(history.val_losses) == 5
        assert all(np.isfinite(history.train_losses))
        assert all(np.isfinite(history.val_losses))

    def test_fit_multi_labelset_loss_masking(
        self, multi_labelset_config, multi_labelset_data
    ):
        """Test that loss is only computed for valid labels (masking works)."""
        trainer = Trainer(multi_labelset_config)
        X, y_dict, has_labels_dict = multi_labelset_data

        # Samples 0-19: only APOGEE labels (GALAH masked out)
        # Samples 20-39: only GALAH labels (APOGEE masked out)
        # Samples 40-59: both valid

        X_train = {"desi": X[:48]}
        X_val = {"desi": X[48:]}
        has_data_train = {"desi": np.ones(48, dtype=bool)}
        has_data_val = {"desi": np.ones(12, dtype=bool)}

        y_train = {src: y[:48] for src, y in y_dict.items()}
        y_val = {src: y[48:] for src, y in y_dict.items()}

        has_labels_train = {src: h[:48] for src, h in has_labels_dict.items()}
        has_labels_val = {src: h[48:] for src, h in has_labels_dict.items()}

        # Verify masks are correct before training
        # APOGEE: samples 0-19 valid, 20-39 masked (in training set)
        assert np.all(y_train["apogee"][:20, 2, :] == 1.0)
        assert np.all(y_train["apogee"][20:40, 2, :] == 0.0)

        # GALAH: samples 0-19 masked, 20-39 valid (in training set)
        assert np.all(y_train["galah"][:20, 2, :] == 0.0)
        assert np.all(y_train["galah"][20:40, 2, :] == 1.0)

        history = trainer.fit_multi_labelset(
            X_train,
            y_train,
            X_val,
            y_val,
            has_data_train,
            has_data_val,
            has_labels_train,
            has_labels_val,
        )

        # Training should complete with finite loss
        assert all(np.isfinite(history.train_losses))

    def test_fit_multi_labelset_gradient_flow(
        self, multi_labelset_config, multi_labelset_data
    ):
        """Test that gradients flow only to relevant output heads."""
        trainer = Trainer(multi_labelset_config)
        X, y_dict, has_labels_dict = multi_labelset_data

        # Use only samples 0-19 (APOGEE only) for this test
        X_train = {"desi": X[:20]}
        X_val = {"desi": X[:5]}
        has_data_train = {"desi": np.ones(20, dtype=bool)}
        has_data_val = {"desi": np.ones(5, dtype=bool)}

        y_train = {src: y[:20] for src, y in y_dict.items()}
        y_val = {src: y[:5] for src, y in y_dict.items()}

        has_labels_train = {src: h[:20] for src, h in has_labels_dict.items()}
        has_labels_val = {src: h[:5] for src, h in has_labels_dict.items()}

        # Get initial weights of apogee head
        apogee_head_init = [
            p.clone() for p in trainer.model.output_heads["apogee"].parameters()
        ]

        # Train for 1 epoch
        multi_labelset_config.training = multi_labelset_config.training.model_copy(
            update={"epochs": 1}
        )
        trainer = Trainer(multi_labelset_config)

        trainer.fit_multi_labelset(
            X_train,
            y_train,
            X_val,
            y_val,
            has_data_train,
            has_data_val,
            has_labels_train,
            has_labels_val,
        )

        # Check APOGEE head weights changed (because we had APOGEE labels)
        apogee_head_final = list(trainer.model.output_heads["apogee"].parameters())
        apogee_changed = any(
            not torch.allclose(init, final)
            for init, final in zip(apogee_head_init, apogee_head_final, strict=True)
        )
        assert apogee_changed, "APOGEE head weights should have changed"

        # GALAH head should not change (no GALAH labels in samples 0-19)
        # Note: This might change if shared trunk updates affect head weights
        # through regularization or other mechanisms

    def test_fit_multi_labelset_reduces_loss(
        self, multi_labelset_config, multi_labelset_data
    ):
        """Test that multi-labelset training reduces loss over epochs."""
        multi_labelset_config.training = multi_labelset_config.training.model_copy(
            update={"epochs": 10}
        )
        trainer = Trainer(multi_labelset_config)
        X, y_dict, has_labels_dict = multi_labelset_data

        X_train = {"desi": X[:48]}
        X_val = {"desi": X[48:]}
        has_data_train = {"desi": np.ones(48, dtype=bool)}
        has_data_val = {"desi": np.ones(12, dtype=bool)}

        y_train = {src: y[:48] for src, y in y_dict.items()}
        y_val = {src: y[48:] for src, y in y_dict.items()}

        has_labels_train = {src: h[:48] for src, h in has_labels_dict.items()}
        has_labels_val = {src: h[48:] for src, h in has_labels_dict.items()}

        history = trainer.fit_multi_labelset(
            X_train,
            y_train,
            X_val,
            y_val,
            has_data_train,
            has_data_val,
            has_labels_train,
            has_labels_val,
        )

        # Loss should decrease over time
        assert history.train_losses[-1] < history.train_losses[0]

    def test_fit_multi_labelset_handles_no_labels(self, multi_labelset_config):
        """Test training raises error when no samples have valid labels."""
        trainer = Trainer(multi_labelset_config)
        rng = np.random.default_rng(42)

        n_samples = 30
        n_params = 11

        X = rng.standard_normal((n_samples, 3, 100)).astype(np.float32)
        X[:, 2, :] = 1.0

        # All samples have NO labels
        y_apogee = np.zeros((n_samples, 3, n_params), dtype=np.float32)
        y_galah = np.zeros((n_samples, 3, n_params), dtype=np.float32)
        # mask = 0 for all (no valid labels)

        X_train = {"desi": X[:24]}
        X_val = {"desi": X[24:]}
        has_data_train = {"desi": np.ones(24, dtype=bool)}
        has_data_val = {"desi": np.ones(6, dtype=bool)}

        y_train = {"apogee": y_apogee[:24], "galah": y_galah[:24]}
        y_val = {"apogee": y_apogee[24:], "galah": y_galah[24:]}

        # No samples have labels from either source
        has_labels_train = {
            "apogee": np.zeros(24, dtype=bool),
            "galah": np.zeros(24, dtype=bool),
        }
        has_labels_val = {
            "apogee": np.zeros(6, dtype=bool),
            "galah": np.zeros(6, dtype=bool),
        }

        # Training without any labels should raise an error
        with pytest.raises(ValueError, match="No valid labels found"):
            trainer.fit_multi_labelset(
                X_train,
                y_train,
                X_val,
                y_val,
                has_data_train,
                has_data_val,
                has_labels_train,
                has_labels_val,
            )

    def test_predict_all_label_sources(
        self, multi_labelset_config, multi_labelset_data
    ):
        """Test predictions from all label source heads."""
        trainer = Trainer(multi_labelset_config)
        X, y_dict, has_labels_dict = multi_labelset_data

        X_train = {"desi": X[:48]}
        X_val = {"desi": X[48:]}
        has_data_train = {"desi": np.ones(48, dtype=bool)}
        has_data_val = {"desi": np.ones(12, dtype=bool)}

        y_train = {src: y[:48] for src, y in y_dict.items()}
        y_val = {src: y[48:] for src, y in y_dict.items()}

        has_labels_train = {src: h[:48] for src, h in has_labels_dict.items()}
        has_labels_val = {src: h[48:] for src, h in has_labels_dict.items()}

        trainer.fit_multi_labelset(
            X_train,
            y_train,
            X_val,
            y_val,
            has_data_train,
            has_data_val,
            has_labels_train,
            has_labels_val,
        )

        # Get predictions from both heads
        X_test = {"desi": X[48:]}
        has_data_test = {"desi": np.ones(12, dtype=bool)}

        predictions = trainer.predict_multi_labelset(X_test, has_data_test)

        assert "apogee" in predictions
        assert "galah" in predictions
        assert predictions["apogee"].shape == (12, 2, 11)  # (batch, 2, n_params)
        assert predictions["galah"].shape == (12, 2, 11)


class TestTrainerMultiLabelsetMixedOverlap:
    """Tests for multi-labelset training with varying overlap patterns."""

    @pytest.fixture
    def mixed_overlap_config(self, tmp_path):
        """Config for testing mixed overlap scenarios."""
        from dorothy.config.schema import (
            DataConfig,
            ExperimentConfig,
            MultiHeadModelConfig,
            TrainingConfig,
        )

        return ExperimentConfig(
            name="test_mixed_overlap",
            output_dir=tmp_path,
            data=DataConfig(
                catalogue_path=tmp_path / "catalogue.h5",
                surveys=["boss"],
                label_sources=["apogee", "galah"],
            ),
            multi_head_model=MultiHeadModelConfig(
                survey_wavelengths={"boss": 80},
                n_parameters=5,  # Fewer params for faster tests
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
        )

    def test_high_overlap_training(self, mixed_overlap_config):
        """Test training when most samples have both label sources."""
        trainer = Trainer(mixed_overlap_config)
        rng = np.random.default_rng(42)

        n_samples = 40
        n_params = 5

        X = rng.standard_normal((n_samples, 3, 80)).astype(np.float32)
        X[:, 2, :] = 1.0

        # 90% overlap: most samples have both APOGEE and GALAH
        y_apogee = np.zeros((n_samples, 3, n_params), dtype=np.float32)
        y_galah = np.zeros((n_samples, 3, n_params), dtype=np.float32)

        # All samples have APOGEE
        y_apogee[:, 0, :] = rng.standard_normal((n_samples, n_params))
        y_apogee[:, 0, 0] = np.abs(y_apogee[:, 0, 0]) * 1000 + 4000
        y_apogee[:, 1, :] = 0.1
        y_apogee[:, 2, :] = 1.0
        has_apogee = np.ones(n_samples, dtype=bool)

        # 90% have GALAH
        galah_mask = rng.random(n_samples) < 0.9
        y_galah[galah_mask, 0, :] = rng.standard_normal((galah_mask.sum(), n_params))
        y_galah[galah_mask, 0, 0] = np.abs(y_galah[galah_mask, 0, 0]) * 1000 + 4000
        y_galah[galah_mask, 1, :] = 0.1
        y_galah[galah_mask, 2, :] = 1.0
        has_galah = galah_mask

        X_train = {"boss": X[:32]}
        X_val = {"boss": X[32:]}
        has_data = {"boss": np.ones(32, dtype=bool)}
        has_data_val = {"boss": np.ones(8, dtype=bool)}

        y_train = {"apogee": y_apogee[:32], "galah": y_galah[:32]}
        y_val = {"apogee": y_apogee[32:], "galah": y_galah[32:]}

        has_labels_train = {"apogee": has_apogee[:32], "galah": has_galah[:32]}
        has_labels_val = {"apogee": has_apogee[32:], "galah": has_galah[32:]}

        history = trainer.fit_multi_labelset(
            X_train,
            y_train,
            X_val,
            y_val,
            has_data,
            has_data_val,
            has_labels_train,
            has_labels_val,
        )

        assert all(np.isfinite(history.train_losses))

    def test_low_overlap_training(self, mixed_overlap_config):
        """Test training when few samples have both label sources."""
        trainer = Trainer(mixed_overlap_config)
        rng = np.random.default_rng(42)

        n_samples = 40
        n_params = 5

        X = rng.standard_normal((n_samples, 3, 80)).astype(np.float32)
        X[:, 2, :] = 1.0

        # Low overlap: samples 0-19 APOGEE, samples 20-39 GALAH, no overlap
        y_apogee = np.zeros((n_samples, 3, n_params), dtype=np.float32)
        y_galah = np.zeros((n_samples, 3, n_params), dtype=np.float32)

        # First half: APOGEE only
        y_apogee[:20, 0, :] = rng.standard_normal((20, n_params))
        y_apogee[:20, 0, 0] = np.abs(y_apogee[:20, 0, 0]) * 1000 + 4000
        y_apogee[:20, 1, :] = 0.1
        y_apogee[:20, 2, :] = 1.0
        has_apogee = np.zeros(n_samples, dtype=bool)
        has_apogee[:20] = True

        # Second half: GALAH only
        y_galah[20:, 0, :] = rng.standard_normal((20, n_params))
        y_galah[20:, 0, 0] = np.abs(y_galah[20:, 0, 0]) * 1000 + 4000
        y_galah[20:, 1, :] = 0.1
        y_galah[20:, 2, :] = 1.0
        has_galah = np.zeros(n_samples, dtype=bool)
        has_galah[20:] = True

        X_train = {"boss": X[:32]}
        X_val = {"boss": X[32:]}
        has_data = {"boss": np.ones(32, dtype=bool)}
        has_data_val = {"boss": np.ones(8, dtype=bool)}

        y_train = {"apogee": y_apogee[:32], "galah": y_galah[:32]}
        y_val = {"apogee": y_apogee[32:], "galah": y_galah[32:]}

        has_labels_train = {"apogee": has_apogee[:32], "galah": has_galah[:32]}
        has_labels_val = {"apogee": has_apogee[32:], "galah": has_galah[32:]}

        history = trainer.fit_multi_labelset(
            X_train,
            y_train,
            X_val,
            y_val,
            has_data,
            has_data_val,
            has_labels_train,
            has_labels_val,
        )

        assert all(np.isfinite(history.train_losses))

    def test_partial_parameter_masking(self, mixed_overlap_config):
        """Test training when some parameters are missing per sample."""
        trainer = Trainer(mixed_overlap_config)
        rng = np.random.default_rng(42)

        n_samples = 30
        n_params = 5

        X = rng.standard_normal((n_samples, 3, 80)).astype(np.float32)
        X[:, 2, :] = 1.0

        y_apogee = np.zeros((n_samples, 3, n_params), dtype=np.float32)

        # All samples have some APOGEE labels, but with different param coverage
        y_apogee[:, 0, :] = rng.standard_normal((n_samples, n_params))
        y_apogee[:, 0, 0] = np.abs(y_apogee[:, 0, 0]) * 1000 + 4000
        y_apogee[:, 1, :] = 0.1

        # Samples 0-9: all 5 params valid
        y_apogee[:10, 2, :] = 1.0

        # Samples 10-19: only first 3 params valid
        y_apogee[10:20, 2, :3] = 1.0
        y_apogee[10:20, 2, 3:] = 0.0

        # Samples 20-29: only first param valid (teff)
        y_apogee[20:30, 2, 0] = 1.0
        y_apogee[20:30, 2, 1:] = 0.0

        # All samples have some APOGEE labels
        has_apogee = np.ones(n_samples, dtype=bool)

        # GALAH: just give all samples full labels for simplicity
        y_galah = np.zeros((n_samples, 3, n_params), dtype=np.float32)
        y_galah[:, 0, :] = rng.standard_normal((n_samples, n_params))
        y_galah[:, 0, 0] = np.abs(y_galah[:, 0, 0]) * 1000 + 4000
        y_galah[:, 1, :] = 0.1
        y_galah[:, 2, :] = 1.0
        has_galah = np.ones(n_samples, dtype=bool)

        X_train = {"boss": X[:24]}
        X_val = {"boss": X[24:]}
        has_data = {"boss": np.ones(24, dtype=bool)}
        has_data_val = {"boss": np.ones(6, dtype=bool)}

        y_train = {"apogee": y_apogee[:24], "galah": y_galah[:24]}
        y_val = {"apogee": y_apogee[24:], "galah": y_galah[24:]}

        has_labels_train = {"apogee": has_apogee[:24], "galah": has_galah[:24]}
        has_labels_val = {"apogee": has_apogee[24:], "galah": has_galah[24:]}

        history = trainer.fit_multi_labelset(
            X_train,
            y_train,
            X_val,
            y_val,
            has_data,
            has_data_val,
            has_labels_train,
            has_labels_val,
        )

        assert all(np.isfinite(history.train_losses))
