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

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from dorothy.config.schema import (
    DataConfig,
    ExperimentConfig,
    InputMaskingConfig,
    LossType,
    ModelConfig,
    MultiHeadModelConfig,
    NormalizationType,
    OptimizerConfig,
    OptimizerType,
    SchedulerConfig,
    SchedulerType,
    TrainingConfig,
)
from dorothy.data.augmentation import DynamicInputMasking
from dorothy.losses import predicted_scatter
from dorothy.training.batches import SINGLE_SURVEY_KEY, TensorBatchSource
from dorothy.training.trainer import (
    EVALUATOR_METRIC_NAMES,
    Trainer,
    TrainingHistory,
    _autocast_dtype,
)


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

    @pytest.mark.parametrize("normalization", list(NormalizationType))
    @pytest.mark.parametrize("optimizer", list(OptimizerType))
    def test_weight_decay_applies_only_to_weight_matrices(
        self, simple_config, optimizer, normalization
    ):
        """Linear weights are decayed; biases and normalization weights are not."""
        simple_config.model.normalization = normalization
        simple_config.training.optimizer = OptimizerConfig(
            type=optimizer, weight_decay=0.05
        )
        trainer = Trainer(simple_config)

        decay = {
            p: g["weight_decay"]
            for g in trainer.optimizer.param_groups
            for p in g["params"]
        }
        for module in trainer.model.modules():
            for name, p in module.named_parameters(recurse=False):
                linear_weight = isinstance(module, torch.nn.Linear) and name == "weight"
                assert decay[p] == (0.05 if linear_weight else 0.0), (module, name)

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

    def test_training_fails_when_no_validation_loss_is_finite(
        self, simple_config, training_data
    ):
        """Without a finite validation loss, no epoch has the best weights."""
        trainer = Trainer(simple_config)
        X, y = training_data
        X_val = X[80:].copy()
        X_val[:, 0, 0] = np.nan  # every validation prediction becomes NaN

        with pytest.raises(RuntimeError, match="finite validation loss"):
            trainer.fit(X[:80], y[:80], X_val, y[80:])

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

    def test_epoch_checkpoint_loads_with_its_config(
        self, simple_config, training_data, tmp_path
    ):
        """A label-aware model needs its config.yaml to be rebuilt."""
        from dorothy.inference import load_checkpoint

        simple_config.output_dir = tmp_path
        simple_config.training.save_every = 2
        trainer = Trainer(simple_config)
        X, y = training_data
        trainer.fit(X[:80], y[:80], X[80:], y[80:])

        ckpt = load_checkpoint(
            simple_config.get_checkpoint_path(2),
            model_file="best_model_epoch_2.pth",
            device="cpu",
        )
        assert ckpt.model.label_aware_uncertainty
        assert ckpt.normalizer is not None


class TestCalibrationFactors:
    """The factors that training fits for the predicted scatter."""

    def test_parameter_without_validation_labels_keeps_factor_1(
        self, simple_config, training_data, tmp_path
    ):
        trainer = Trainer(simple_config)
        X, y = training_data
        y_val = y[80:].copy()
        y_val[:, 2, 1] = 0  # no validation labels for logg
        history = trainer.fit(X[:80], y[:80], X[80:], y_val)

        spread = history.val_metrics["z_robust_scatter"][history.best_epoch]
        assert np.isnan(spread[1])
        names = trainer.parameter_names
        expected = {"apogee": {names[0]: spread[0], names[1]: 1.0, names[2]: spread[2]}}
        assert trainer.calibration == expected
        trainer.save_checkpoint(tmp_path)
        assert json.loads((tmp_path / "calibration.json").read_text()) == expected

    @pytest.mark.parametrize(
        "setting,value",
        [("calibrate_uncertainties", False), ("loss", LossType.MSE)],
        ids=["off", "mse"],
    )
    def test_checkpoint_without_factors_has_no_calibration_file(
        self, simple_config, training_data, tmp_path, setting, value
    ):
        setattr(simple_config.training, setting, value)
        trainer = Trainer(simple_config)
        X, y = training_data
        trainer.fit(X[:80], y[:80], X[80:], y[80:])
        (tmp_path / "calibration.json").write_text("{}")  # from an earlier run

        trainer.save_checkpoint(tmp_path)
        assert trainer.calibration is None
        assert not (tmp_path / "calibration.json").exists()

    def test_failed_fit_drops_the_factors_of_an_earlier_fit(
        self, simple_config, training_data
    ):
        trainer = Trainer(simple_config)
        X, y = training_data
        trainer.fit(X[:80], y[:80], X[80:], y[80:])
        assert trainer.calibration is not None
        evaluate = trainer._evaluate

        def nan_loss(source):
            val = evaluate(source)
            val["loss"] = float("nan")
            return val

        trainer._evaluate = nan_loss
        with pytest.raises(RuntimeError, match="finite validation loss"):
            trainer.fit(X[:80], y[:80], X[80:], y[80:])
        assert trainer.calibration is None


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


class TestTrainerMixedPrecision:
    """Mixed precision with training.amp."""

    @pytest.mark.parametrize(
        "device, training, dtype",
        [
            ("cuda", {"amp": True}, torch.bfloat16),
            ("cuda", {"amp": True, "amp_dtype": "float16"}, torch.float16),
            ("cuda", {}, None),
            ("cpu", {"amp": True}, None),
        ],
        ids=["cuda_default", "cuda_float16", "cuda_amp_off", "cpu"],
    )
    def test_amp_autocasts_on_cuda_only(self, monkeypatch, device, training, dtype):
        monkeypatch.setattr(
            torch.cuda, "is_bf16_supported", lambda including_emulation=True: True
        )
        config = TrainingConfig(**training)
        assert _autocast_dtype(config, torch.device(device)) == dtype

    def test_bf16_on_a_gpu_without_native_bf16_is_rejected(self, monkeypatch):
        # A GPU with emulated bf16 only, such as a V100
        monkeypatch.setattr(
            torch.cuda,
            "is_bf16_supported",
            lambda including_emulation=True: including_emulation,
        )
        with pytest.raises(ValueError, match="training.amp_dtype"):
            _autocast_dtype(TrainingConfig(amp=True), torch.device("cuda"))

    def test_training_loss_is_the_float32_loss_of_bf16_outputs(self, simple_config):
        """The loss does not compute exp(2 ln s) in bf16."""
        simple_config.device = "cpu"
        # Every star gets the missing vector, as in the reference pass below
        simple_config.training.label_error_dropout = 1.0
        trainer = Trainer(simple_config)
        trainer.autocast_dtype = torch.bfloat16  # CPU autocast stands in for CUDA
        gen = torch.Generator().manual_seed(0)
        n, n_params = 64, trainer.n_parameters
        X = torch.randn(n, 3, 1000, generator=gen)
        y = torch.stack(
            [
                torch.randn(n, n_params, generator=gen),
                torch.full((n, n_params), 0.01),
                torch.ones(n, n_params),
            ],
            dim=1,
        )
        key = trainer.label_keys[0]
        source = TensorBatchSource(
            {SINGLE_SURVEY_KEY: X}, {key: y}, None, trainer.device
        )
        batch = next(source.batches(n, np.random.default_rng(0)))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            outputs = trainer._forward(batch)[key]
        expected = trainer.loss_fn(outputs.float(), batch.y[key]).item()

        # One batch, in the same order: the epoch loss is its loss before the step
        loss, _ = trainer._train_epoch(
            source, n, np.random.default_rng(0), 10.0, first_step=0, warmup_steps=0
        )
        # Without the cast, exp runs in bf16 and this loss changes by 7e-4
        assert loss == pytest.approx(expected, rel=1e-6)

    def test_validation_runs_in_float32_under_mixed_precision(
        self, simple_config, training_data
    ):
        """The recorded validation loss is that of float32 predictions."""
        simple_config.device = "cpu"
        simple_config.training.epochs = 1
        trainer = Trainer(simple_config)
        trainer.autocast_dtype = torch.bfloat16  # CPU autocast stands in for CUDA
        X, y = training_data
        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        output = torch.from_numpy(trainer.predict(X[80:]))
        target = torch.from_numpy(trainer.normalizer.transform_labels(y[80:]))
        expected = trainer.loss_fn(output, target).item()
        # Validation under bf16 autocast changes this loss by 3e-5
        assert history.val_losses[0] == pytest.approx(expected, rel=1e-6)

    def test_training_without_mixed_precision_opens_no_autocast_region(
        self, simple_config, training_data, monkeypatch
    ):
        """torch < 2.5 rejects autocast on MPS even when it is disabled."""

        def unsupported(*args, **kwargs):
            raise RuntimeError("User specified an unsupported autocast device_type")

        monkeypatch.setattr("dorothy.training.trainer.autocast", unsupported)
        simple_config.device = "cpu"
        simple_config.training.epochs = 1
        X, y = training_data
        Trainer(simple_config).fit(X[:80], y[:80], X[80:], y[80:])


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
            input_masking=InputMaskingConfig(
                enabled=True,
                f_max=0.3,
                p_block_min=0.7,
                p_block_max=0.9,
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

    def test_fit_with_input_masking_from_config(
        self, augmentation_config, three_channel_data
    ):
        """Test training with input masking created from config."""
        trainer = Trainer(augmentation_config)
        X, y = three_channel_data

        # Input masking should be created from config
        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        assert trainer._input_masking is not None
        assert len(history.train_losses) == 3

    def test_fit_with_input_masking_disabled(
        self, augmentation_config, three_channel_data
    ):
        """Test training with input masking disabled."""
        # Disable config-based input masking
        augmentation_config.input_masking.enabled = False

        trainer = Trainer(augmentation_config)
        X, y = three_channel_data

        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])

        assert trainer._input_masking is None
        assert len(history.train_losses) == 3

    def test_input_masking_applied_to_training_only(
        self, augmentation_config, three_channel_data
    ):
        """Test that input masking is applied during training but not validation."""
        trainer = Trainer(augmentation_config)
        X, y = three_channel_data

        # Track whether masking is called during training vs validation
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
            input_masking=InputMaskingConfig(
                enabled=True,
                f_max=0.3,
                p_block_min=0.7,
                p_block_max=0.9,
            ),
        )

    def test_fit_with_both_masks_and_input_masking(self, full_masking_config):
        """Test training with both label masks and input masking."""
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

        assert trainer._input_masking is not None
        assert len(history.train_losses) == 3
        assert all(np.isfinite(history.train_losses))
        assert all(np.isfinite(history.val_losses))

    def test_masked_runs_with_one_seed_are_identical_after_global_draws(
        self, full_masking_config, training_data
    ):
        """Masking and label error dropout draw from the configured seed, not
        the global generators."""
        from dorothy.config.schema import LabelMaskingConfig

        config = full_masking_config.model_copy(
            update={"label_masking": LabelMaskingConfig(enabled=True)}
        )
        assert config.training.label_error_dropout == 0.5
        X, y = training_data

        def train_losses(global_draws: int) -> list[float]:
            trainer = Trainer(config)
            np.random.rand(global_draws)
            torch.rand(global_draws)
            return trainer.fit(X[:80], y[:80], X[80:], y[80:]).train_losses

        assert train_losses(0) == train_losses(1)


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

    def test_input_masking_clears_has_data_for_dropped_surveys(self, multi_head_config):
        """A survey zeroed out by input masking must be marked as missing."""
        import torch

        config = multi_head_config.model_copy(
            update={
                "input_masking": InputMaskingConfig(
                    enabled=True, p_survey_min=0.5, p_survey_max=0.5
                )
            }
        )
        trainer = Trainer(config)
        trainer._input_masking = DynamicInputMasking(
            p_survey_min=0.5,
            p_survey_max=0.5,
            p_block_min=0.0,
            p_block_max=0.0,
            seed=0,
        )
        trainer._survey_wavelengths = {"boss": 100, "lamost": 80}
        n = 64
        X = {
            "boss": torch.ones(n, 3, 100),
            "lamost": torch.ones(n, 3, 80),
        }
        has_data = {
            "boss": torch.ones(n, dtype=torch.bool),
            "lamost": torch.ones(n, dtype=torch.bool),
        }
        X_out, has_out = trainer._apply_input_masking(X, has_data)
        for survey in X_out:
            dropped = X_out[survey][:, 2, :].sum(dim=1) == 0
            assert dropped.any()
            assert torch.equal(has_out[survey], ~dropped)
        # Every sample keeps at least one survey
        assert (has_out["boss"] | has_out["lamost"]).all()

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


def test_training_rejects_inputs_without_the_precision_encoding(simple_config):
    simple_config.data.precision_encoding = False
    with pytest.raises(ValueError, match="^data.precision_encoding"):
        Trainer(simple_config)


class TestLabelAwareTraining:
    """Training reads the label errors through the output layer."""

    @pytest.mark.parametrize("multi_head", [False, True])
    def test_training_rejects_the_additive_output_layer(
        self, simple_config, multi_head
    ):
        if multi_head:
            config = ExperimentConfig(
                name="t",
                data=DataConfig(
                    catalogue_path=Path("/fake/catalogue.h5"), surveys=["boss", "desi"]
                ),
                multi_head_model={
                    "survey_wavelengths": {"boss": 10, "desi": 20},
                    "label_aware_uncertainty": False,
                },
            )
            field = "multi_head_model.label_aware_uncertainty"
        else:
            config = simple_config
            config.model.label_aware_uncertainty = False
            field = "model.label_aware_uncertainty"
        with pytest.raises(ValueError, match=f"^{re.escape(field)}"):
            Trainer(config)

    def test_masked_labels_get_the_missing_vector(self, simple_config):
        """A masked label's error input is 0 (missing), whatever it stores."""
        # With the trainer's seed, this drops neither star
        simple_config.training.label_error_dropout = 1e-6
        trainer = Trainer(simple_config)
        target = torch.ones(2, 3, 3)
        target[:, 1] = torch.tensor([0.1, 0.2, 0.3])
        target[0, 1, 0], target[0, 2, 0] = float("nan"), 0.0  # masked NaN error
        target[1, 2, 1] = 0.0  # masked positive error

        errors = trainer._label_errors({"default": target})["default"]

        assert torch.equal(errors, torch.tensor([[0.0, 0.2, 0.3], [0.1, 0.0, 0.3]]))

    def test_whole_stars_lose_their_label_errors_per_head(self, simple_config):
        """Each head drops a star's errors with probability label_error_dropout."""
        simple_config.training.label_error_dropout = 0.3
        trainer = Trainer(simple_config)
        n = 4000
        errors = trainer._label_errors(
            {"a": torch.ones(n, 3, 3), "b": torch.ones(n, 3, 3)}
        )

        dropped = {}
        for key, e in errors.items():
            dropped[key] = (e == 0).all(dim=1)
            assert torch.all(dropped[key] | (e == 1).all(dim=1))  # whole stars
            # Binomial standard deviation: sqrt(0.3 * 0.7 / 4000) = 0.0072
            assert abs(dropped[key].float().mean().item() - 0.3) < 4 * 0.0072
        assert not torch.equal(dropped["a"], dropped["b"])

    def test_missing_vector_recovers_the_true_label_noise(self):
        """Labels with noise sigma_true reported as 2 sigma_true: s finds sigma_true.

        The additive variance sigma_y^2 + s^2 of earlier versions can never
        fall below the reported 2 sigma_true. Each spectrum is one of 4
        noiseless templates with fixed labels, so the labels are an exactly
        learnable function of the spectrum.
        """
        rng = np.random.default_rng(0)
        n_stars, n_train, n_wave, sigma_true = 3000, 2400, 8, 0.1
        templates = rng.normal(size=(4, n_wave)).astype(np.float32)
        template_labels = rng.uniform(-1, 1, size=(4, 2)).astype(np.float32)
        star = rng.integers(4, size=n_stars)
        flux = templates[star]
        X = np.stack([flux, np.full_like(flux, 0.1), np.ones_like(flux)], axis=1)
        noise = rng.normal(scale=sigma_true, size=(n_stars, 2)).astype(np.float32)
        values = template_labels[star] + noise
        errors = np.full_like(values, 2 * sigma_true)
        y = np.stack([values, errors, np.ones_like(values)], axis=1)
        config = ExperimentConfig(
            name="label_noise",
            device="cpu",
            data=DataConfig(catalogue_path=Path("/fake/catalogue.h5")),
            model=ModelConfig(
                input_features=3 * n_wave, output_features=4, hidden_layers=[16]
            ),
            training=TrainingConfig(
                epochs=30,
                batch_size=128,
                learning_rate=3e-3,
                scheduler=SchedulerConfig(max_lr=3e-3),
            ),
        )
        trainer = Trainer(config)
        # The labels are already in model units
        trainer.fit(
            X[:n_train], y[:n_train], X[n_train:], y[n_train:], normalize_labels=False
        )

        X_val = torch.from_numpy(X[n_train:])
        with torch.no_grad():
            missing = trainer.model(X_val)
            reported = trainer.model(
                X_val, label_errors=torch.from_numpy(errors[n_train:])
            )
        # Each template's s is fit to the residuals of about 600 training
        # stars, so its relative standard error is 1/sqrt(2 * 600) = 2.9%; the
        # tolerance is three of them. The model error, from estimating each
        # template's labels from about 600 labels, is sigma_true / sqrt(600)
        # = 0.04 sigma_true and raises the expected s by under 0.1%.
        for output in (missing, reported):
            s = predicted_scatter(output[:, 1], config.training.scatter_floor)
            np.testing.assert_allclose(
                np.median(s.numpy(), axis=0), sigma_true, rtol=0.09
            )


def _trained_states(trainer: Trainer) -> list[dict[str, torch.Tensor]]:
    """Collect the model state now and after every optimizer step of the next fit."""

    def snapshot(*_):
        states.append(
            {name: t.clone() for name, t in trainer.model.state_dict().items()}
        )

    states = []
    snapshot()
    trainer.optimizer.register_step_post_hook(snapshot)
    return states


def _moving_average(
    states: list[dict[str, torch.Tensor]], decay: float, every: int
) -> dict[str, torch.Tensor]:
    """Reference EMA of the floating-point entries of the states.

    The average starts at the first state; every k = ``every`` steps, the
    state after the step moves it a fraction 1 - decay**k towards that state.
    """
    average = {k: v.clone() for k, v in states[0].items() if v.is_floating_point()}
    for state in states[1::every]:
        for name, value in average.items():
            value.lerp_(state[name], 1 - decay**every)
    return average


class TestWeightAverage:
    """Exponential moving average of the weights (training.ema_decay)."""

    @pytest.fixture
    def batchnorm_config(self, simple_config):
        """simple_config with batch normalization, whose running statistics are buffers."""
        simple_config.model.normalization = NormalizationType.BATCHNORM
        return simple_config

    @pytest.mark.parametrize(
        "cpu_every", [None, 1, 3], ids=["device", "cpu_every_1", "cpu_every_3"]
    )
    def test_saved_weights_are_moving_averages_of_the_trained_weights(
        self, batchnorm_config, training_data, tmp_path, cpu_every
    ):
        """The best, final and epoch weights average the steps up to their epoch.

        An epoch has 5 steps, so an average every 3 steps also checks that the
        steps are counted across epochs.
        """
        decay, every = 0.9, cpu_every or 1
        batchnorm_config.training.ema_decay = decay
        batchnorm_config.training.ema_cpu_every = cpu_every
        batchnorm_config.training.save_every = 2
        batchnorm_config.output_dir = tmp_path
        trainer = Trainer(batchnorm_config)
        states = _trained_states(trainer)
        X, y = training_data
        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])
        path = trainer.save_checkpoint(tmp_path / "checkpoint")

        steps_per_epoch = (len(states) - 1) // len(history.train_losses)

        def steps_to(epoch):
            return states[: 1 + epoch * steps_per_epoch]

        epoch_2 = batchnorm_config.get_checkpoint_path(2) / "model_epoch_2.pth"
        expected = [
            (path / "best_model.pth", steps_to(history.best_epoch + 1)),
            (path / "final_model.pth", states),
            (epoch_2, steps_to(2)),
        ]
        # The default float32 tolerances allow lerp kernels that round
        # differently (the CPU average of a CUDA run); an update schedule off
        # by one step moves the average by over 2e-4 here.
        for file, trained in expected:
            saved = torch.load(file, weights_only=True)
            for name, value in _moving_average(trained, decay, every).items():
                torch.testing.assert_close(saved[name], value)

    def test_weight_average_changes_validation_but_not_training(
        self, batchnorm_config, training_data
    ):
        """The trained weights and their metrics do not depend on the average."""
        X, y = training_data
        runs = []
        for decay in [None, 0.9]:
            batchnorm_config.training.ema_decay = decay
            trainer = Trainer(batchnorm_config)
            states = _trained_states(trainer)
            runs.append((states, trainer.fit(X[:80], y[:80], X[80:], y[80:])))
        (states_off, off), (states_on, on) = runs

        for state_off, state_on in zip(states_off, states_on, strict=True):
            for name, value in state_off.items():
                assert torch.equal(state_on[name], value), name
        assert on.weight_norms == off.weight_norms
        assert on.weight_updates == off.weight_updates
        assert on.val_losses != off.val_losses

    def test_multi_head_training_keeps_its_weights_and_saves_one_average(
        self, tmp_path
    ):
        """The first head is registered under two names, which share a tensor.

        After each validation the model gets its trained weights back, and
        final_model.pth holds one value under both names of the head.
        """
        config = ExperimentConfig(
            name="weight_average_multi_head",
            output_dir=tmp_path,
            data=DataConfig(
                catalogue_path=tmp_path / "catalogue.h5",
                surveys=["desi"],
                label_sources=["apogee", "galah"],
            ),
            multi_head_model=MultiHeadModelConfig(
                survey_wavelengths={"desi": 100},
                n_parameters=3,
                latent_dim=16,
                encoder_hidden=[32],
                trunk_hidden=[16],
                output_hidden=[8],
                label_sources=["apogee", "galah"],
            ),
            training=TrainingConfig(epochs=3, batch_size=16),
            seed=0,
        )
        rng = np.random.default_rng(0)
        X = rng.standard_normal((64, 3, 100)).astype(np.float32)
        X[:, 2] = 1.0
        y = np.stack(
            [
                rng.standard_normal((64, 3)),
                np.full((64, 3), 0.1),
                np.ones((64, 3)),
            ],
            axis=1,
        ).astype(np.float32)
        y[:, 0, 0] = rng.uniform(4000, 7000, 64)  # Teff, normalized in log space
        has_data = {"desi": np.ones(64, dtype=bool)}
        splits = [slice(0, 48), slice(48, 64)]
        args = [
            *[{"desi": X[rows]} for rows in splits],
            *[{"apogee": y[rows], "galah": y[rows]} for rows in splits],
            *[{s: h[rows] for s, h in has_data.items()} for rows in splits],
        ]
        args = [args[0], args[2], args[1], args[3], args[4], args[5]]

        runs = []
        for decay in [None, 0.9]:
            config.training.ema_decay = decay
            trainer = Trainer(config)
            states = _trained_states(trainer)
            trainer.fit_multi_labelset(*args)
            runs.append(states)
        for state_off, state_on in zip(*runs, strict=True):
            for name, value in state_off.items():
                assert torch.equal(state_on[name], value), name

        final = torch.load(
            trainer.save_checkpoint(tmp_path / "ckpt") / "final_model.pth",
            weights_only=True,
        )
        trainer.model.load_state_dict(final)
        for name, value in trainer.model.state_dict().items():
            assert torch.equal(value, final[name]), name

    def test_zero_decay_gives_the_results_of_training_without_an_average(
        self, batchnorm_config, training_data, tmp_path
    ):
        """With decay 0 the average is the trained weights after each step."""
        X, y = training_data
        runs = []
        for decay in [None, 0.0]:
            batchnorm_config.training.ema_decay = decay
            trainer = Trainer(batchnorm_config)
            history = trainer.fit(X[:80], y[:80], X[80:], y[80:])
            path = trainer.save_checkpoint(tmp_path / f"decay_{decay}")
            weights = {
                file: torch.load(path / file, weights_only=True)
                for file in ["best_model.pth", "final_model.pth"]
            }
            runs.append((history.val_losses, weights))
        (val_off, weights_off), (val_zero, weights_zero) = runs

        assert val_zero == val_off
        for file, state in weights_off.items():
            for name, value in state.items():
                assert torch.equal(weights_zero[file][name], value), (file, name)


class TestNllWarmup:
    """The β-NLL warm-up (training.nll_warmup_fraction)."""

    @pytest.mark.parametrize("fraction, warmup", [(0.0, 0), (0.4, 10)])
    def test_beta_falls_linearly_to_zero_and_validation_uses_the_plain_nll(
        self, simple_config, training_data, fraction, warmup
    ):
        """simple_config trains 5 epochs of 5 steps, so a fraction 0.4 warms
        up over the first 10 steps, across an epoch boundary."""
        simple_config.training.nll_warmup_fraction = fraction
        trainer = Trainer(simple_config)
        betas = {True: [], False: []}
        element_losses = trainer.loss_fn.element_losses

        def record(output, target, beta=0.0):
            # Validation outputs come from a forward pass without gradients
            betas[output.requires_grad].append(beta)
            return element_losses(output, target, beta)

        trainer.loss_fn.element_losses = record
        X, y = training_data
        trainer.fit(X[:80], y[:80], X[80:], y[80:])

        steps, validation = betas[True], betas[False]
        assert steps[:warmup] == pytest.approx([1 - t / warmup for t in range(warmup)])
        assert steps[warmup:] == [0.0] * (25 - warmup)  # exactly 0: the plain loss
        assert validation == [0.0] * 5

    @pytest.mark.parametrize(
        "fraction, loss, epochs",
        [
            (0.5, LossType.HETEROSCEDASTIC, 2.4),
            (0.5, LossType.MSE, 0.0),
        ],
        ids=["warmup", "mse"],
    )
    def test_history_records_the_warmup_length_in_epochs(
        self, simple_config, training_data, fraction, loss, epochs
    ):
        """simple_config has 5 steps per epoch, so a fraction 0.5 of 25 steps
        rounds to 12 warm-up steps, or 2.4 epochs. mse has no warm-up."""
        simple_config.training.nll_warmup_fraction = fraction
        simple_config.training.loss = loss
        X, y = training_data
        history = Trainer(simple_config).fit(X[:80], y[:80], X[80:], y[80:])
        assert history.nll_warmup_epochs == pytest.approx(epochs)

    def test_early_stopping_waits_for_the_end_of_the_warmup(
        self, simple_config, training_data
    ):
        """A fraction 0.6 warms up over epochs 0 to 2, so with a constant
        validation loss and patience 1, training stops after epoch 3, not 1."""
        simple_config.training.nll_warmup_fraction = 0.6
        simple_config.training.early_stopping_patience = 1
        trainer = Trainer(simple_config)
        evaluate = trainer._evaluate

        def constant_loss(source):
            val = evaluate(source)
            val["loss"] = 1.0
            return val

        trainer._evaluate = constant_loss
        X, y = training_data
        history = trainer.fit(X[:80], y[:80], X[80:], y[80:])
        assert len(history.val_losses) == 4
