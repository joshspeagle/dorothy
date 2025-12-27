"""
Training infrastructure for DOROTHY models.

This module implements the training loop for stellar parameter prediction,
including:
- Model and optimizer setup from configuration
- Batch-wise training with gradient clipping
- Learning rate scheduling (CyclicLR, ReduceOnPlateau, etc.)
- Validation evaluation and best model tracking
- Checkpointing and training history
- Early stopping support

The trainer is designed to reproduce the training behavior from the
original DOROTHY notebooks while providing a cleaner, more modular interface.
"""

from __future__ import annotations

import copy
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, ReduceLROnPlateau

from dorothy.config.schema import STELLAR_PARAMETERS, LossType, SchedulerType
from dorothy.data.normalizer import LabelNormalizer
from dorothy.losses.heteroscedastic import HeteroscedasticLoss
from dorothy.models.mlp import MLP

if TYPE_CHECKING:
    from dorothy.config.schema import ExperimentConfig

logger = logging.getLogger(__name__)


def _has_tqdm() -> bool:
    """Check if tqdm is available."""
    try:
        import tqdm  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class TrainingHistory:
    """
    Container for training metrics and history.

    Attributes:
        train_losses: Per-epoch training loss values.
        val_losses: Per-epoch validation loss values.
        learning_rates: Learning rate at each training step.
        best_epoch: Epoch with the best validation loss.
        best_val_loss: Best validation loss achieved.
        total_time: Total training time in seconds.
        parameter_names: Names of stellar parameters being tracked.
        val_loss_breakdown: Per-epoch loss breakdown with keys:
            - 'mean_component': (n_epochs, n_params) mean loss per param
            - 'scatter_component': (n_epochs, n_params) scatter loss per param
            - 'per_param_loss': (n_epochs, n_params) total loss per param
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    total_time: float = 0.0

    # Detailed per-parameter loss tracking
    parameter_names: list[str] = field(default_factory=list)
    val_loss_breakdown: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save training history to a pickle file."""
        data = {
            "history_train": self.train_losses,
            "history_val": self.val_losses,
            "parameter_names": self.parameter_names,
            "val_loss_breakdown": self.val_loss_breakdown,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def save_learning_rates(self, path: str | Path) -> None:
        """Save learning rates to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self.learning_rates, f)

    def get_loss_breakdown_array(self) -> dict[str, np.ndarray]:
        """Get loss breakdown as numpy arrays.

        Returns:
            Dictionary with keys 'mean_component', 'scatter_component', 'per_param_loss',
            each with shape (n_epochs, n_params).
        """
        return {
            key: np.array(values) for key, values in self.val_loss_breakdown.items()
        }


class Trainer:
    """
    Trainer for DOROTHY stellar parameter models.

    Handles the complete training pipeline including model creation,
    optimization, scheduling, validation, checkpointing, and label normalization.

    Attributes:
        config: Experiment configuration.
        model: The neural network model.
        optimizer: The optimizer (Adam).
        scheduler: Learning rate scheduler.
        loss_fn: Loss function.
        device: Compute device (cuda/cpu).
        history: Training history and metrics.
        normalizer: Label normalizer for converting to/from normalized space.
        parameter_names: Names of stellar parameters being predicted.

    Example:
        >>> from dorothy.config import ExperimentConfig, DataConfig
        >>> config = ExperimentConfig(
        ...     name="test",
        ...     data=DataConfig(fits_path=Path("/data/train.fits")),
        ... )
        >>> trainer = Trainer(config)
        >>> # y_train format: [labels, errors] concatenated
        >>> trainer.fit(X_train, y_train, X_val, y_val)
        >>> trainer.save_checkpoint()  # Saves model + normalizer
    """

    def __init__(
        self,
        config: ExperimentConfig,
        parameter_names: list[str] | None = None,
    ) -> None:
        """
        Initialize the trainer from configuration.

        Args:
            config: Complete experiment configuration.
            parameter_names: Names of stellar parameters being predicted.
                If None, uses the first n parameters from STELLAR_PARAMETERS
                where n = config.model.n_parameters.
        """
        self.config = config
        self.history = TrainingHistory()

        # Determine device
        self.device = self._resolve_device(config.device)
        logger.info(f"Using device: {self.device}")

        # Set random seeds for reproducibility
        self._set_seeds(config.seed)

        # Create model
        self.model = self._create_model()
        self.model.to(self.device)
        logger.info(f"Model created with {self.model.count_parameters():,} parameters")

        # Create loss function
        self.loss_fn = self._create_loss_fn()

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler will be created when we know the number of batches
        self.scheduler: CyclicLR | ReduceLROnPlateau | CosineAnnealingLR | None = None

        # Best model weights
        self._best_weights: dict | None = None

        # Set up parameter names for normalization
        n_params = config.model.n_parameters
        if parameter_names is not None:
            if len(parameter_names) != n_params:
                raise ValueError(
                    f"parameter_names has {len(parameter_names)} entries but "
                    f"model expects {n_params} parameters"
                )
            self.parameter_names = parameter_names
        else:
            # Use first n parameters from standard list
            self.parameter_names = list(STELLAR_PARAMETERS[:n_params])

        # Normalizer will be fitted during training
        self.normalizer: LabelNormalizer | None = None

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve the device string to a torch.device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _create_model(self) -> MLP:
        """Create the model from configuration."""
        return MLP.from_config(self.config.model)

    def _create_loss_fn(self) -> nn.Module:
        """Create the loss function from configuration."""
        training_config = self.config.training

        if training_config.loss == LossType.HETEROSCEDASTIC:
            return HeteroscedasticLoss(
                scatter_floor=training_config.scatter_floor,
                n_parameters=self.config.model.n_parameters,
            )
        elif training_config.loss == LossType.MSE:
            return nn.MSELoss()
        else:
            raise NotImplementedError(
                f"Loss type {training_config.loss} not yet implemented"
            )

    def _create_optimizer(self) -> torch.optim.Adam:
        """Create the optimizer from configuration."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
        )

    def _create_scheduler(
        self, steps_per_epoch: int
    ) -> CyclicLR | ReduceLROnPlateau | CosineAnnealingLR | None:
        """Create the learning rate scheduler from configuration."""
        scheduler_config = self.config.training.scheduler
        training_config = self.config.training

        if scheduler_config.type == SchedulerType.NONE:
            return None

        elif scheduler_config.type == SchedulerType.CYCLIC:
            return CyclicLR(
                optimizer=self.optimizer,
                base_lr=scheduler_config.base_lr,
                max_lr=scheduler_config.max_lr,
                step_size_up=scheduler_config.step_size_up,
                step_size_down=scheduler_config.step_size_down,
                mode="exp_range",
                gamma=scheduler_config.gamma,
                cycle_momentum=False,
            )

        elif scheduler_config.type == SchedulerType.REDUCE_ON_PLATEAU:
            return ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode="min",
                factor=scheduler_config.factor,
                patience=scheduler_config.patience,
            )

        elif scheduler_config.type == SchedulerType.COSINE_ANNEALING:
            return CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=training_config.epochs * steps_per_epoch,
            )

        else:
            raise NotImplementedError(
                f"Scheduler {scheduler_config.type} not yet implemented"
            )

    def fit(
        self,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        X_val: np.ndarray | torch.Tensor,
        y_val: np.ndarray | torch.Tensor,
        normalize_labels: bool = True,
    ) -> TrainingHistory:
        """
        Train the model.

        Args:
            X_train: Training features of shape (n_samples, ...).
            y_train: Training labels of shape (n_samples, 2*n_params) where
                first n_params columns are labels and last n_params are errors.
            X_val: Validation features.
            y_val: Validation labels (same format as y_train).
            normalize_labels: Whether to fit and apply label normalization.
                If True, creates a LabelNormalizer, fits on training labels,
                and transforms both train and validation labels to normalized space.

        Returns:
            TrainingHistory with loss curves and metrics.
        """
        # Convert to numpy for normalization if needed
        y_train_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
        y_val_np = y_val.numpy() if isinstance(y_val, torch.Tensor) else y_val

        n_params = self.config.model.n_parameters

        # Apply label normalization if requested
        if normalize_labels:
            # Extract labels and errors
            train_labels = y_train_np[:, :n_params]
            train_errors = y_train_np[:, n_params:]
            val_labels = y_val_np[:, :n_params]
            val_errors = y_val_np[:, n_params:]

            # Create and fit normalizer
            self.normalizer = LabelNormalizer(parameters=self.parameter_names)
            self.normalizer.fit(train_labels)
            logger.info(f"Fitted label normalizer on {len(train_labels)} samples")

            # Transform labels and errors
            train_labels_norm, train_errors_norm = self.normalizer.transform(
                train_labels, train_errors
            )
            val_labels_norm, val_errors_norm = self.normalizer.transform(
                val_labels, val_errors
            )

            # Recombine into expected format
            y_train_np = np.concatenate([train_labels_norm, train_errors_norm], axis=1)
            y_val_np = np.concatenate([val_labels_norm, val_errors_norm], axis=1)

        # Convert to tensors
        X_train = self._to_tensor(X_train)
        y_train = self._to_tensor(y_train_np)
        X_val = self._to_tensor(X_val)
        y_val = self._to_tensor(y_val_np)

        # Move validation data to device
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        # Training parameters
        training_config = self.config.training
        batch_size = training_config.batch_size
        n_epochs = training_config.epochs
        n_samples = len(X_train)
        steps_per_epoch = (n_samples + batch_size - 1) // batch_size

        # Create scheduler now that we know steps_per_epoch
        self.scheduler = self._create_scheduler(steps_per_epoch)

        # Initialize tracking
        self._best_weights = copy.deepcopy(self.model.state_dict())
        self.history = TrainingHistory(
            parameter_names=self.parameter_names,
            val_loss_breakdown={
                "mean_component": [],
                "scatter_component": [],
                "per_param_loss": [],
            },
        )
        patience_counter = 0
        rng = np.random.default_rng(self.config.seed)

        start_time = time.time()
        logger.info(f"Starting training for {n_epochs} epochs")

        # Set up progress bar if tqdm is available
        epoch_iter = range(n_epochs)
        use_tqdm = _has_tqdm()
        if use_tqdm:
            from tqdm import tqdm

            epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

        for epoch in epoch_iter:
            # Training phase
            epoch_train_loss = self._train_epoch(
                X_train, y_train, batch_size, rng, training_config.gradient_clip
            )
            self.history.train_losses.append(epoch_train_loss)

            # Validation phase with detailed metrics
            val_details = self._validate_detailed(X_val, y_val)
            val_loss = val_details["loss"]
            self.history.val_losses.append(val_loss)

            # Store per-parameter loss breakdown
            self.history.val_loss_breakdown["mean_component"].append(
                val_details["mean_component"]
            )
            self.history.val_loss_breakdown["scatter_component"].append(
                val_details["scatter_component"]
            )
            self.history.val_loss_breakdown["per_param_loss"].append(
                val_details["per_param_loss"]
            )

            # Track best model
            if val_loss < self.history.best_val_loss:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch
                self._best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # ReduceOnPlateau scheduler step (per epoch)
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)

            # Update progress bar or log
            current_lr = self.optimizer.param_groups[0]["lr"]
            if use_tqdm:
                epoch_iter.set_postfix(
                    train=f"{epoch_train_loss:.4f}",
                    val=f"{val_loss:.4f}",
                    best=f"{self.history.best_val_loss:.4f}",
                    lr=f"{current_lr:.1e}",
                )
            elif (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{n_epochs} - "
                    f"Train: {epoch_train_loss:.6f}, Val: {val_loss:.6f}, "
                    f"LR: {current_lr:.2e}"
                )

            # Save checkpoint if configured
            if (
                training_config.save_every > 0
                and (epoch + 1) % training_config.save_every == 0
            ):
                self._save_epoch_checkpoint(epoch + 1)

            # Early stopping
            if (
                training_config.early_stopping_patience > 0
                and patience_counter >= training_config.early_stopping_patience
            ):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        self.history.total_time = time.time() - start_time

        # Restore best weights
        self.model.load_state_dict(self._best_weights)
        logger.info(
            f"Training complete. Best val loss: {self.history.best_val_loss:.6f} "
            f"at epoch {self.history.best_epoch + 1}"
        )

        return self.history

    def _train_epoch(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        rng: np.random.Generator,
        gradient_clip: float,
    ) -> float:
        """Run one training epoch."""
        self.model.train()
        n_samples = len(X)

        # Shuffle data
        indices = rng.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuffled[start:end].to(self.device)
            y_batch = y_shuffled[start:end].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(X_batch)
            loss = self.loss_fn(output, y_batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

            self.optimizer.step()

            # Step scheduler (for per-step schedulers like CyclicLR)
            if self.scheduler is not None and not isinstance(
                self.scheduler, ReduceLROnPlateau
            ):
                self.scheduler.step()
                self.history.learning_rates.append(self.optimizer.param_groups[0]["lr"])

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate(self, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_val)
            loss = self.loss_fn(output, y_val)
        return loss.item()

    def _validate_detailed(
        self, X_val: torch.Tensor, y_val: torch.Tensor
    ) -> dict[str, np.ndarray | float]:
        """Run validation with detailed per-parameter loss breakdown.

        Returns:
            Dictionary containing:
                - 'loss': Total validation loss (scalar)
                - 'mean_component': Per-parameter mean loss component
                - 'scatter_component': Per-parameter scatter loss component
                - 'per_param_loss': Per-parameter total loss
        """
        self.model.eval()
        n_params = self.config.model.n_parameters

        with torch.no_grad():
            output = self.model(X_val)

            # Get detailed loss breakdown
            if isinstance(self.loss_fn, HeteroscedasticLoss):
                detailed = self.loss_fn.forward_detailed(output, y_val)
                loss = detailed["loss"].item()
                mean_component = detailed["mean_component"].cpu().numpy()
                scatter_component = detailed["scatter_component"].cpu().numpy()
                per_param_loss = detailed["per_param_loss"].cpu().numpy()
            else:
                # Fallback for non-heteroscedastic losses
                loss = self.loss_fn(output, y_val).item()
                mean_component = np.zeros(n_params)
                scatter_component = np.zeros(n_params)
                per_param_loss = np.zeros(n_params)

        return {
            "loss": loss,
            "mean_component": mean_component,
            "scatter_component": scatter_component,
            "per_param_loss": per_param_loss,
        }

    def _to_tensor(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert numpy array to tensor if needed."""
        if isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=torch.float32)
        return data.float()

    def _save_epoch_checkpoint(self, epoch: int) -> None:
        """Save checkpoint for a specific epoch."""
        checkpoint_dir = self.config.get_checkpoint_path(epoch)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save current model
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / f"model_epoch_{epoch}.pth",
        )

        # Save best model so far
        torch.save(
            self._best_weights,
            checkpoint_dir / f"best_model_epoch_{epoch}.pth",
        )

        # Save history
        self.history.save(checkpoint_dir / "history_train_val.pkl")
        self.history.save_learning_rates(checkpoint_dir / "learning_rates.pkl")

        logger.debug(f"Saved checkpoint at epoch {epoch}")

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """
        Save the final model and training artifacts.

        Saves:
            - best_model.pth: Best model weights
            - final_model.pth: Final model weights
            - normalizer.pkl: Label normalizer (if used)
            - history_train_val.pkl: Training/validation losses
            - learning_rates.pkl: Learning rate history
            - config.yaml: Experiment configuration

        Args:
            path: Directory to save to. If None, uses config output path.

        Returns:
            Path to the saved checkpoint directory.
        """
        path = self.config.get_checkpoint_path() if path is None else Path(path)

        path.mkdir(parents=True, exist_ok=True)

        # Save best model
        torch.save(self._best_weights, path / "best_model.pth")

        # Save final model (current state)
        torch.save(self.model.state_dict(), path / "final_model.pth")

        # Save normalizer if it was fitted
        if self.normalizer is not None:
            self.normalizer.save(path / "normalizer.pkl")
            logger.info("Saved label normalizer")

        # Save training history
        self.history.save(path / "history_train_val.pkl")
        self.history.save_learning_rates(path / "learning_rates.pkl")

        # Save config for reproducibility
        self.config.save_yaml(path / "config.yaml")

        logger.info(f"Saved final checkpoint to {path}")
        return path

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Make predictions with the trained model.

        Args:
            X: Input features.

        Returns:
            Predictions as numpy array.
        """
        self.model.eval()
        X_tensor = self._to_tensor(X).to(self.device)

        with torch.no_grad():
            output = self.model(X_tensor)

        return output.cpu().numpy()

    def get_predictions_and_uncertainties(
        self, X: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and uncertainty estimates.

        Only works with HeteroscedasticLoss.

        Args:
            X: Input features.

        Returns:
            Tuple of (predictions, uncertainties) as numpy arrays.
        """
        output = self.predict(X)
        n_params = self.config.model.n_parameters

        predictions = output[:, :n_params]
        log_scatter = output[:, n_params:]

        # Convert log-scatter to scatter (with floor)
        scatter_floor = self.config.training.scatter_floor
        uncertainties = np.sqrt(np.exp(2 * log_scatter) + scatter_floor**2)

        return predictions, uncertainties
