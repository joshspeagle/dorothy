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
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CyclicLR,
    OneCycleLR,
    ReduceLROnPlateau,
)

from dorothy.config.schema import (
    STELLAR_PARAMETERS,
    LossType,
    OptimizerType,
    SchedulerType,
)
from dorothy.data.augmentation import DynamicBlockMasking
from dorothy.data.normalizer import LabelNormalizer
from dorothy.inference.evaluator import Evaluator
from dorothy.losses.heteroscedastic import HeteroscedasticLoss
from dorothy.models.mlp import MLP
from dorothy.models.multi_head_mlp import MultiHeadMLP


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


# Metric names tracked by the Evaluator (in order)
EVALUATOR_METRIC_NAMES = [
    "rmse",
    "bias",
    "sd",
    "mae",
    "median_offset",
    "robust_scatter",
    "z_median",
    "z_robust_scatter",
    "pred_unc_p16",
    "pred_unc_p50",
    "pred_unc_p84",
]


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
            - 'mean_component': (n_epochs, n_params) weighted squared error per param
            - 'scatter_component': (n_epochs, n_params) log-variance penalty per param
        val_metrics: Per-epoch Evaluator metrics as dict of lists of arrays.
            Keys are metric names (rmse, bias, sd, mae, median_offset, robust_scatter,
            z_median, z_robust_scatter, pred_unc_p16, pred_unc_p50, pred_unc_p84).
            Each value is a list of arrays with shape (n_params,) per epoch.
        weight_norms: Per-layer L2 weight norms per epoch for grokking detection.
            Keys are layer names, values are lists of norms per epoch.
        grad_norms: Total gradient norm per epoch (average across batches).
        weight_updates: Per-layer weight update magnitudes per epoch.
            Keys are layer names, values are lists of ||W_new - W_old|| per epoch.
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

    # Evaluator metrics per epoch (k metrics x p parameters)
    val_metrics: dict[str, list[np.ndarray]] = field(default_factory=dict)

    # Grokking detection metrics
    weight_norms: dict[str, list[float]] = field(default_factory=dict)
    grad_norms: list[float] = field(default_factory=list)
    weight_updates: dict[str, list[float]] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save training history to a pickle file."""
        data = {
            "history_train": self.train_losses,
            "history_val": self.val_losses,
            "parameter_names": self.parameter_names,
            "val_loss_breakdown": self.val_loss_breakdown,
            "val_metrics": self.val_metrics,
            # Grokking detection metrics
            "weight_norms": self.weight_norms,
            "grad_norms": self.grad_norms,
            "weight_updates": self.weight_updates,
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
            Dictionary with keys 'mean_component' and 'scatter_component',
            each with shape (n_epochs, n_params).
        """
        return {
            key: np.array(values) for key, values in self.val_loss_breakdown.items()
        }

    def get_metrics_array(self) -> dict[str, np.ndarray]:
        """Get Evaluator metrics as numpy arrays.

        Returns:
            Dictionary with metric names as keys (rmse, bias, sd, mae, etc.),
            each with shape (n_epochs, n_params).
        """
        return {key: np.array(values) for key, values in self.val_metrics.items()}

    def get_grokking_metrics(self) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """Get grokking detection metrics as numpy arrays.

        Returns:
            Dictionary containing:
                - 'weight_norms': dict mapping layer names to (n_epochs,) arrays
                - 'grad_norms': (n_epochs,) array of average gradient norms
                - 'weight_updates': dict mapping layer names to (n_epochs,) arrays
        """
        return {
            "weight_norms": {
                name: np.array(values) for name, values in self.weight_norms.items()
            },
            "grad_norms": np.array(self.grad_norms),
            "weight_updates": {
                name: np.array(values) for name, values in self.weight_updates.items()
            },
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
        ...     data=DataConfig(catalogue_path=Path("/data/catalogue.h5")),
        ... )
        >>> trainer = Trainer(config)
        >>> # X_train shape: (n_samples, 3, n_wavelengths) - [flux, sigma, mask]
        >>> # y_train shape: (n_samples, 3, n_params) - [values, errors, mask]
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
                where n is from config.model or config.multi_head_model.
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
        self.scheduler: (
            OneCycleLR | CyclicLR | ReduceLROnPlateau | CosineAnnealingLR | None
        ) = None

        # Best model weights
        self._best_weights: dict | None = None

        # Set up parameter names for normalization
        # Get n_params from appropriate config (model or multi_head_model)
        if config.is_multi_head:
            n_params = config.multi_head_model.n_parameters
        else:
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

        # Augmentation will be created during fit() if configured
        self._augmentation: DynamicBlockMasking | None = None

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

    @property
    def n_parameters(self) -> int:
        """Get number of stellar parameters from config."""
        if self.config.is_multi_head:
            return self.config.multi_head_model.n_parameters
        return self.config.model.n_parameters

    def _create_model(self) -> MLP | MultiHeadMLP:
        """Create the model from configuration.

        Returns MLP for single-survey config, MultiHeadMLP for multi-survey.
        """
        if self.config.is_multi_head:
            mh_config = self.config.multi_head_model
            return MultiHeadMLP(
                survey_configs=mh_config.survey_wavelengths,
                n_parameters=mh_config.n_parameters,
                latent_dim=mh_config.latent_dim,
                encoder_hidden=mh_config.encoder_hidden,
                trunk_hidden=mh_config.trunk_hidden,
                output_hidden=mh_config.output_hidden,
                combination_mode=mh_config.combination_mode.value,
                normalization=mh_config.normalization.value,
                activation=mh_config.activation.value,
                dropout=mh_config.dropout,
                label_sources=mh_config.label_sources,
            )
        return MLP.from_config(self.config.model)

    def _create_loss_fn(self) -> nn.Module:
        """Create the loss function from configuration."""
        training_config = self.config.training

        if training_config.loss == LossType.HETEROSCEDASTIC:
            return HeteroscedasticLoss(
                scatter_floor=training_config.scatter_floor,
                n_parameters=self.n_parameters,
            )
        elif training_config.loss == LossType.MSE:
            return nn.MSELoss()
        else:
            raise NotImplementedError(
                f"Loss type {training_config.loss} not yet implemented"
            )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer from configuration."""
        training_config = self.config.training
        optimizer_config = training_config.optimizer

        if optimizer_config.type == OptimizerType.ADAMW:
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=training_config.learning_rate,
                betas=optimizer_config.betas,
                weight_decay=optimizer_config.weight_decay,
            )

        elif optimizer_config.type == OptimizerType.ADAM:
            return torch.optim.Adam(
                self.model.parameters(),
                lr=training_config.learning_rate,
                betas=optimizer_config.betas,
                weight_decay=optimizer_config.weight_decay,
            )

        elif optimizer_config.type == OptimizerType.SGD:
            return torch.optim.SGD(
                self.model.parameters(),
                lr=training_config.learning_rate,
                momentum=optimizer_config.momentum,
                weight_decay=optimizer_config.weight_decay,
                nesterov=optimizer_config.nesterov,
            )

        else:
            raise NotImplementedError(
                f"Optimizer {optimizer_config.type} not yet implemented"
            )

    def _create_scheduler(
        self, steps_per_epoch: int
    ) -> OneCycleLR | CyclicLR | ReduceLROnPlateau | CosineAnnealingLR | None:
        """Create the learning rate scheduler from configuration."""
        scheduler_config = self.config.training.scheduler
        training_config = self.config.training

        if scheduler_config.type == SchedulerType.NONE:
            return None

        elif scheduler_config.type == SchedulerType.ONE_CYCLE:
            total_steps = training_config.epochs * steps_per_epoch
            return OneCycleLR(
                optimizer=self.optimizer,
                max_lr=scheduler_config.max_lr,
                total_steps=total_steps,
                pct_start=scheduler_config.pct_start,
                div_factor=scheduler_config.div_factor,
                final_div_factor=scheduler_config.final_div_factor,
                anneal_strategy=scheduler_config.anneal_strategy,
            )

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
        augmentation: DynamicBlockMasking | None = None,
    ) -> TrainingHistory:
        """
        Train the model.

        Args:
            X_train: Training features of shape (n_samples, 3, n_wavelengths)
                with 3 channels: [flux, sigma, mask].
            y_train: Training labels of shape (n_samples, 3, n_params) where:
                - Channel 0: label values
                - Channel 1: label errors (uncertainties)
                - Channel 2: mask (1=valid, 0=masked)
            X_val: Validation features (same format as X_train).
            y_val: Validation labels (same format as y_train).
            normalize_labels: Whether to fit and apply label normalization.
                If True, creates a LabelNormalizer, fits on training labels,
                and transforms both train and validation labels to normalized space.
                Normalization is applied to channels 0-1 (values and errors),
                channel 2 (mask) is preserved unchanged.
            augmentation: Optional DynamicBlockMasking augmentation to apply during
                training. If None but config.masking.enabled is True, creates one
                from config.

        Returns:
            TrainingHistory with loss curves and metrics.
        """
        # Convert to numpy for normalization if needed
        y_train_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
        y_val_np = y_val.numpy() if isinstance(y_val, torch.Tensor) else y_val

        # Validate 3-channel y format: (n_samples, 3, n_params)
        if y_train_np.ndim != 3 or y_train_np.shape[1] != 3:
            raise ValueError(
                f"y_train must have shape (n_samples, 3, n_params), got {y_train_np.shape}"
            )
        if y_val_np.ndim != 3 or y_val_np.shape[1] != 3:
            raise ValueError(
                f"y_val must have shape (n_samples, 3, n_params), got {y_val_np.shape}"
            )

        # Extract mask from channel 2 for normalization
        y_train_mask_np = y_train_np[:, 2, :]  # (n_samples, n_params)

        # Apply label normalization if requested
        if normalize_labels:
            # Extract labels and errors from 3-channel format
            train_labels = y_train_np[:, 0, :]  # (n_samples, n_params)
            train_errors = y_train_np[:, 1, :]
            val_labels = y_val_np[:, 0, :]
            val_errors = y_val_np[:, 1, :]

            # Create and fit normalizer (mask-aware)
            self.normalizer = LabelNormalizer(parameters=self.parameter_names)
            self.normalizer.fit(train_labels, mask=y_train_mask_np)
            logger.info(f"Fitted label normalizer on {len(train_labels)} samples")

            # Transform labels and errors
            train_labels_norm, train_errors_norm = self.normalizer.transform(
                train_labels, train_errors
            )
            val_labels_norm, val_errors_norm = self.normalizer.transform(
                val_labels, val_errors
            )

            # Reassemble 3-channel y with normalized values/errors, original mask
            y_train_np = np.stack(
                [train_labels_norm, train_errors_norm, y_train_np[:, 2, :]], axis=1
            )
            y_val_np = np.stack(
                [val_labels_norm, val_errors_norm, y_val_np[:, 2, :]], axis=1
            )

        # Create augmentation from config if not provided
        if augmentation is None and self.config.masking.enabled:
            augmentation = DynamicBlockMasking(
                min_fraction=self.config.masking.min_fraction,
                max_fraction=self.config.masking.max_fraction,
                fraction_choices=self.config.masking.fraction_choices,
                min_block_size=self.config.masking.min_block_size,
                max_block_size=self.config.masking.max_block_size,
            )
            logger.info(f"Created augmentation from config: {augmentation}")

        # Store augmentation for use in training
        self._augmentation = augmentation

        # Convert to tensors (y is now 3-channel: [values, errors, mask])
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
            },
            val_metrics={name: [] for name in EVALUATOR_METRIC_NAMES},
        )

        # Create Evaluator for validation metrics
        self._evaluator = Evaluator(
            parameter_names=self.parameter_names,
            teff_in_log=False,  # Don't add extra log_teff param during training
            scatter_floor=self.config.training.scatter_floor,
        )
        patience_counter = 0
        rng = np.random.default_rng(self.config.seed)

        start_time = time.time()
        logger.info(f"Starting training for {n_epochs} epochs")
        if self._augmentation is not None:
            logger.info("Dynamic block masking augmentation enabled for training")

        # Initialize grokking metric tracking
        initial_weight_norms = self._compute_weight_norms()
        for layer_name in initial_weight_norms:
            self.history.weight_norms[layer_name] = []
            self.history.weight_updates[layer_name] = []

        # Set up progress bar if tqdm is available
        epoch_iter = range(n_epochs)
        use_tqdm = _has_tqdm()
        if use_tqdm:
            from tqdm import tqdm

            epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

        for epoch in epoch_iter:
            # Snapshot weights before training for update magnitude tracking
            prev_weights = self._get_weight_snapshot()

            # Training phase (with augmentation applied per-batch)
            # y_train is 3-channel: [values, errors, mask]
            epoch_train_loss, epoch_grad_norm = self._train_epoch(
                X_train,
                y_train,
                batch_size,
                rng,
                training_config.gradient_clip,
            )
            self.history.train_losses.append(epoch_train_loss)

            # Store grokking detection metrics
            self.history.grad_norms.append(epoch_grad_norm)
            weight_norms = self._compute_weight_norms()
            weight_updates = self._compute_weight_updates(prev_weights)
            for layer_name in weight_norms:
                self.history.weight_norms[layer_name].append(weight_norms[layer_name])
                self.history.weight_updates[layer_name].append(
                    weight_updates[layer_name]
                )

            # Validation phase with detailed metrics (no augmentation)
            # y_val is 3-channel: [values, errors, mask]
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

            # Compute and store Evaluator metrics
            # Extract from 3-channel y_val: [values, errors, mask]
            y_true = y_val[:, 0, :].cpu().numpy()  # Channel 0: values
            label_errors = y_val[:, 1, :].cpu().numpy()  # Channel 1: errors
            val_mask_np = y_val[:, 2, :].cpu().numpy()  # Channel 2: mask
            eval_result = self._evaluator.evaluate(
                y_pred=val_details["y_pred"],
                y_true=y_true,
                pred_scatter=val_details["pred_scatter"],
                label_errors=label_errors,
                mask=val_mask_np,
            )

            # Store each metric as an array of shape (n_params,)
            for metric_name in EVALUATOR_METRIC_NAMES:
                metric_values = np.array(
                    [
                        getattr(eval_result.metrics[p], metric_name)
                        for p in self.parameter_names
                    ],
                    dtype=np.float32,
                )
                self.history.val_metrics[metric_name].append(metric_values)

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
    ) -> tuple[float, float]:
        """Run one training epoch.

        Args:
            X: Training features tensor of shape (n_samples, 3, n_wavelengths).
            y: Training labels tensor of shape (n_samples, 3, n_params) where:
                - Channel 0: label values
                - Channel 1: label errors
                - Channel 2: mask (1=valid, 0=masked)
            batch_size: Batch size for training.
            rng: Random number generator for shuffling.
            gradient_clip: Maximum gradient norm for clipping.

        Returns:
            Tuple of (average_loss, average_gradient_norm).
        """
        self.model.train()
        n_samples = len(X)

        # Shuffle data
        indices = rng.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuffled[start:end].to(self.device)
            y_batch = y_shuffled[start:end].to(self.device)

            # Apply augmentation if available (training only)
            if self._augmentation is not None:
                X_batch = self._augmentation(X_batch)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(X_batch)

            # Compute loss - y_batch is 3-channel, loss extracts mask from channel 2
            loss = self.loss_fn(output, y_batch)

            # Backward pass
            loss.backward()

            # Compute gradient norm before clipping (for grokking detection)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float("inf"),  # Don't actually clip, just compute
            ).item()
            total_grad_norm += grad_norm

            # Gradient clipping (apply the actual clip)
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

        return total_loss / n_batches, total_grad_norm / n_batches

    def _validate(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> float:
        """Run validation and return average loss.

        Args:
            X_val: Validation features tensor of shape (n_samples, 3, n_wavelengths).
            y_val: Validation labels tensor of shape (n_samples, 3, n_params).

        Returns:
            Average validation loss.
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_val)
            loss = self.loss_fn(output, y_val)
        return loss.item()

    def _validate_detailed(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> dict[str, np.ndarray | float]:
        """Run validation with detailed per-parameter loss breakdown.

        Args:
            X_val: Validation features tensor of shape (n_samples, 3, n_wavelengths).
            y_val: Validation labels tensor of shape (n_samples, 3, n_params) where:
                - Channel 0: label values
                - Channel 1: label errors
                - Channel 2: mask (1=valid, 0=masked)

        Returns:
            Dictionary containing:
                - 'loss': Total validation loss (scalar)
                - 'mean_component': Per-parameter mean loss component
                - 'scatter_component': Per-parameter scatter loss component
                - 'y_pred': Predicted means, shape (n_samples, n_params)
                - 'pred_scatter': Predicted scatter, shape (n_samples, n_params)
        """
        self.model.eval()
        n_params = self.n_parameters

        with torch.no_grad():
            output = self.model(X_val)

            # Get detailed loss breakdown
            # y_val is 3-channel, loss function extracts mask from channel 2
            if isinstance(self.loss_fn, HeteroscedasticLoss):
                detailed = self.loss_fn.forward_detailed(output, y_val)
                loss = detailed["loss"].item()
                mean_component = detailed["mean_component"].cpu().numpy()
                scatter_component = detailed["scatter_component"].cpu().numpy()

                # Extract predictions and scatter for Evaluator
                # Output shape is (batch, 2, n_params): [means, log_scatter]
                y_pred = output[:, 0, :].cpu().numpy()
                pred_scatter = self.loss_fn.get_predicted_scatter(output).cpu().numpy()
            else:
                # Fallback for non-heteroscedastic losses
                loss = self.loss_fn(output, y_val).item()
                mean_component = np.zeros(n_params)
                scatter_component = np.zeros(n_params)
                # Output shape is (batch, 2, n_params): [means, log_scatter]
                y_pred = output[:, 0, :].cpu().numpy()
                pred_scatter = None

        return {
            "loss": loss,
            "mean_component": mean_component,
            "scatter_component": scatter_component,
            "y_pred": y_pred,
            "pred_scatter": pred_scatter,
        }

    def _compute_weight_norms(self) -> dict[str, float]:
        """Compute L2 norms of weight matrices for each layer.

        Returns:
            Dictionary mapping layer names to their L2 norms.
        """
        norms = {}
        for name, param in self.model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                # Only track weight matrices (not biases or 1D params)
                norms[name] = torch.norm(param).item()
        return norms

    def _get_weight_snapshot(self) -> dict[str, torch.Tensor]:
        """Get a copy of current weight matrices.

        Returns:
            Dictionary mapping layer names to detached weight tensors.
        """
        snapshot = {}
        for name, param in self.model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                snapshot[name] = param.detach().clone()
        return snapshot

    def _compute_weight_updates(
        self, prev_weights: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """Compute magnitude of weight updates since last snapshot.

        Args:
            prev_weights: Previous weight snapshot from _get_weight_snapshot.

        Returns:
            Dictionary mapping layer names to ||W_new - W_old||.
        """
        updates = {}
        for name, param in self.model.named_parameters():
            if name in prev_weights:
                delta = param.detach() - prev_weights[name].to(param.device)
                updates[name] = torch.norm(delta).item()
        return updates

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
            predictions has shape (batch, n_params).
            uncertainties has shape (batch, n_params).
        """
        output = self.predict(X)
        # Output shape is (batch, 2, n_params): [means, log_scatter]
        predictions = output[:, 0, :]
        log_scatter = output[:, 1, :]

        # Convert log-scatter to scatter (with floor)
        scatter_floor = self.config.training.scatter_floor
        uncertainties = np.sqrt(np.exp(2 * log_scatter) + scatter_floor**2)

        return predictions, uncertainties

    def fit_multi_survey(
        self,
        X_train: dict[str, np.ndarray | torch.Tensor],
        y_train: np.ndarray | torch.Tensor,
        X_val: dict[str, np.ndarray | torch.Tensor],
        y_val: np.ndarray | torch.Tensor,
        has_data_train: dict[str, np.ndarray | torch.Tensor],
        has_data_val: dict[str, np.ndarray | torch.Tensor],
        normalize_labels: bool = True,
    ) -> TrainingHistory:
        """
        Train the model with multi-survey data.

        This method handles training with multiple spectral surveys where each
        star may have data from one or more surveys. Uses MultiHeadMLP with
        survey-specific encoders.

        Args:
            X_train: Dict mapping survey names to training spectral data arrays.
                Each array has shape (n_samples, 3, n_wavelengths) with 3 channels:
                [flux, sigma, mask]. Missing data is indicated by mask=0.
            y_train: Training labels of shape (n_samples, 3, n_params) where:
                - Channel 0: label values
                - Channel 1: label errors (uncertainties)
                - Channel 2: mask (1=valid, 0=masked)
            X_val: Dict mapping survey names to validation spectral data.
            y_val: Validation labels (same format as y_train).
            has_data_train: Dict mapping survey names to boolean arrays indicating
                which samples have data from that survey.
            has_data_val: Same for validation set.
            normalize_labels: Whether to fit and apply label normalization.

        Returns:
            TrainingHistory with loss curves and metrics.

        Raises:
            TypeError: If model is not a MultiHeadMLP.
        """
        if not isinstance(self.model, MultiHeadMLP):
            raise TypeError(
                "fit_multi_survey() requires a MultiHeadMLP model. "
                "Use fit() for single-survey training."
            )

        # Convert y to numpy for normalization
        y_train_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
        y_val_np = y_val.numpy() if isinstance(y_val, torch.Tensor) else y_val

        # Validate 3-channel y format
        if y_train_np.ndim != 3 or y_train_np.shape[1] != 3:
            raise ValueError(
                f"y_train must have shape (n_samples, 3, n_params), got {y_train_np.shape}"
            )

        # Apply label normalization if requested
        y_train_mask_np = y_train_np[:, 2, :]
        if normalize_labels:
            train_labels = y_train_np[:, 0, :]
            train_errors = y_train_np[:, 1, :]
            val_labels = y_val_np[:, 0, :]
            val_errors = y_val_np[:, 1, :]

            self.normalizer = LabelNormalizer(parameters=self.parameter_names)
            self.normalizer.fit(train_labels, mask=y_train_mask_np)
            logger.info(f"Fitted label normalizer on {len(train_labels)} samples")

            train_labels_norm, train_errors_norm = self.normalizer.transform(
                train_labels, train_errors
            )
            val_labels_norm, val_errors_norm = self.normalizer.transform(
                val_labels, val_errors
            )

            y_train_np = np.stack(
                [train_labels_norm, train_errors_norm, y_train_np[:, 2, :]], axis=1
            )
            y_val_np = np.stack(
                [val_labels_norm, val_errors_norm, y_val_np[:, 2, :]], axis=1
            )

        # Convert to tensors
        X_train_tensors = {
            survey: self._to_tensor(arr) for survey, arr in X_train.items()
        }
        X_val_tensors = {
            survey: self._to_tensor(arr).to(self.device)
            for survey, arr in X_val.items()
        }
        y_train_tensor = self._to_tensor(y_train_np)
        y_val_tensor = self._to_tensor(y_val_np).to(self.device)

        # Convert has_data to tensors
        has_data_train_tensors = {
            survey: torch.as_tensor(arr, dtype=torch.bool)
            for survey, arr in has_data_train.items()
        }
        has_data_val_tensors = {
            survey: torch.as_tensor(arr, dtype=torch.bool).to(self.device)
            for survey, arr in has_data_val.items()
        }

        # Training parameters
        training_config = self.config.training
        batch_size = training_config.batch_size
        n_epochs = training_config.epochs
        n_samples = len(y_train_tensor)
        steps_per_epoch = (n_samples + batch_size - 1) // batch_size

        # Create scheduler
        self.scheduler = self._create_scheduler(steps_per_epoch)

        # Initialize tracking
        self._best_weights = copy.deepcopy(self.model.state_dict())
        self.history = TrainingHistory(
            parameter_names=self.parameter_names,
            val_loss_breakdown={"mean_component": [], "scatter_component": []},
            val_metrics={name: [] for name in EVALUATOR_METRIC_NAMES},
        )

        self._evaluator = Evaluator(
            parameter_names=self.parameter_names,
            teff_in_log=False,
            scatter_floor=self.config.training.scatter_floor,
        )
        patience_counter = 0
        rng = np.random.default_rng(self.config.seed)

        start_time = time.time()
        logger.info(f"Starting multi-survey training for {n_epochs} epochs")
        logger.info(f"Surveys: {list(X_train.keys())}")

        # Initialize grokking metrics
        initial_weight_norms = self._compute_weight_norms()
        for layer_name in initial_weight_norms:
            self.history.weight_norms[layer_name] = []
            self.history.weight_updates[layer_name] = []

        # Set up progress bar
        epoch_iter = range(n_epochs)
        use_tqdm = _has_tqdm()
        if use_tqdm:
            from tqdm import tqdm

            epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

        for epoch in epoch_iter:
            prev_weights = self._get_weight_snapshot()

            # Training epoch
            epoch_train_loss, epoch_grad_norm = self._train_epoch_multi_survey(
                X_train_tensors,
                y_train_tensor,
                has_data_train_tensors,
                batch_size,
                rng,
                training_config.gradient_clip,
            )
            self.history.train_losses.append(epoch_train_loss)

            # Grokking metrics
            self.history.grad_norms.append(epoch_grad_norm)
            weight_norms = self._compute_weight_norms()
            weight_updates = self._compute_weight_updates(prev_weights)
            for layer_name in weight_norms:
                self.history.weight_norms[layer_name].append(weight_norms[layer_name])
                self.history.weight_updates[layer_name].append(
                    weight_updates[layer_name]
                )

            # Validation
            val_details = self._validate_detailed_multi_survey(
                X_val_tensors, y_val_tensor, has_data_val_tensors
            )
            val_loss = val_details["loss"]
            self.history.val_losses.append(val_loss)

            self.history.val_loss_breakdown["mean_component"].append(
                val_details["mean_component"]
            )
            self.history.val_loss_breakdown["scatter_component"].append(
                val_details["scatter_component"]
            )

            # Evaluator metrics
            y_true = y_val_tensor[:, 0, :].cpu().numpy()
            y_err = y_val_tensor[:, 1, :].cpu().numpy()
            y_mask = y_val_tensor[:, 2, :].cpu().numpy()

            self.model.eval()
            with torch.no_grad():
                output = self.model.forward(
                    X_val_tensors, has_data=has_data_val_tensors
                )
            y_pred = output[:, 0, :].cpu().numpy()
            log_scatter = output[:, 1, :].cpu().numpy()
            pred_scatter = np.sqrt(
                np.exp(2 * log_scatter) + self.config.training.scatter_floor**2
            )

            eval_result = self._evaluator.evaluate(
                y_pred=y_pred,
                y_true=y_true,
                pred_scatter=pred_scatter,
                label_errors=y_err,
                mask=y_mask,
            )

            # Store each metric as an array of shape (n_params,)
            for metric_name in EVALUATOR_METRIC_NAMES:
                metric_values = np.array(
                    [
                        getattr(eval_result.metrics[p], metric_name)
                        for p in self.parameter_names
                    ],
                    dtype=np.float32,
                )
                self.history.val_metrics[metric_name].append(metric_values)

            # Best model tracking
            if val_loss < self.history.best_val_loss:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch
                self._best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # Store learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history.learning_rates.append(current_lr)

            # Progress logging
            if use_tqdm:
                epoch_iter.set_postfix(
                    {
                        "train": f"{epoch_train_loss:.4f}",
                        "val": f"{val_loss:.4f}",
                        "best": f"{self.history.best_val_loss:.4f}",
                    }
                )

            # Periodic checkpoints
            if (
                training_config.save_every > 0
                and (epoch + 1) % training_config.save_every == 0
            ):
                self._save_epoch_checkpoint(epoch + 1)

            # Early stopping
            patience = training_config.early_stopping_patience
            if patience > 0 and patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best weights
        self.model.load_state_dict(self._best_weights)
        self.history.total_time = time.time() - start_time

        logger.info(
            f"Training complete. Best val loss: {self.history.best_val_loss:.6f} "
            f"at epoch {self.history.best_epoch + 1}"
        )
        return self.history

    def _train_epoch_multi_survey(
        self,
        X: dict[str, torch.Tensor],
        y: torch.Tensor,
        has_data: dict[str, torch.Tensor],
        batch_size: int,
        rng: np.random.Generator,
        gradient_clip: float,
    ) -> tuple[float, float]:
        """Run one training epoch with multi-survey data."""
        self.model.train()
        n_samples = len(y)

        # Shuffle indices
        indices = rng.permutation(n_samples)

        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            # Extract batch for each survey
            X_batch = {
                survey: arr[batch_idx].to(self.device) for survey, arr in X.items()
            }
            has_data_batch = {
                survey: arr[batch_idx].to(self.device)
                for survey, arr in has_data.items()
            }
            y_batch = y[batch_idx].to(self.device)

            # Forward pass through multi-survey model
            self.optimizer.zero_grad()
            output = self.model.forward(X_batch, has_data=has_data_batch)
            loss = self.loss_fn(output, y_batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), gradient_clip
                )
            else:
                grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm**0.5

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            total_grad_norm += grad_norm
            n_batches += 1

        return total_loss / n_batches, total_grad_norm / n_batches

    def _validate_detailed_multi_survey(
        self,
        X: dict[str, torch.Tensor],
        y: torch.Tensor,
        has_data: dict[str, torch.Tensor],
    ) -> dict:
        """Compute validation loss with per-parameter breakdown for multi-survey."""
        self.model.eval()

        with torch.no_grad():
            output = self.model.forward(X, has_data=has_data)

            # Compute loss
            loss = self.loss_fn(output, y)

            # Get per-parameter breakdown if available
            if hasattr(self.loss_fn, "forward_detailed"):
                details = self.loss_fn.forward_detailed(output, y)
                mean_component = details["mean_component"].cpu().numpy()
                scatter_component = details["scatter_component"].cpu().numpy()
            else:
                n_params = y.shape[2]
                mean_component = np.zeros(n_params)
                scatter_component = np.zeros(n_params)

        return {
            "loss": loss.item(),
            "mean_component": mean_component,
            "scatter_component": scatter_component,
        }

    def fit_multi_labelset(
        self,
        X_train: dict[str, np.ndarray | torch.Tensor],
        y_train: dict[str, np.ndarray | torch.Tensor],
        X_val: dict[str, np.ndarray | torch.Tensor],
        y_val: dict[str, np.ndarray | torch.Tensor],
        has_data_train: dict[str, np.ndarray | torch.Tensor],
        has_data_val: dict[str, np.ndarray | torch.Tensor],
        has_labels_train: dict[str, np.ndarray | torch.Tensor] | None = None,
        has_labels_val: dict[str, np.ndarray | torch.Tensor] | None = None,
        normalize_labels: bool = True,
    ) -> TrainingHistory:
        """
        Train the model with multi-survey inputs and multi-labelset outputs.

        This method handles training where:
        - Different surveys provide spectral data (multi-survey inputs)
        - Different label sources provide targets (multi-labelset outputs)
        - Stars may have data from any subset of surveys and label sources

        The approach uses union-style masking: if a star has labels from APOGEE
        but not GALAH, the loss for GALAH predictions is masked for that star.
        This is analogous to how multi-survey input masking already works.

        Args:
            X_train: Dict mapping survey names to training spectral data arrays.
                Each array has shape (n_samples, 3, n_wavelengths) with 3 channels:
                [flux, sigma, mask]. Missing data is indicated by mask=0.
            y_train: Dict mapping label source names to training label arrays.
                Each array has shape (n_samples, 3, n_params) where:
                - Channel 0: label values
                - Channel 1: label errors (uncertainties)
                - Channel 2: mask (1=valid, 0=masked)
            X_val: Dict mapping survey names to validation spectral data.
            y_val: Dict mapping label source names to validation label arrays.
            has_data_train: Dict mapping survey names to boolean arrays indicating
                which samples have data from that survey.
            has_data_val: Same for validation set.
            has_labels_train: Dict mapping label source names to boolean arrays
                indicating which samples have labels from that source.
                If None, derived from y_train mask channels (any valid label).
            has_labels_val: Same for validation set.
                If None, derived from y_val mask channels.
            normalize_labels: Whether to fit and apply label normalization.
                Uses only the first label source for fitting the normalizer.

        Returns:
            TrainingHistory with loss curves and metrics.

        Raises:
            TypeError: If model is not a MultiHeadMLP with multi-label support.
        """
        if not isinstance(self.model, MultiHeadMLP):
            raise TypeError(
                "fit_multi_labelset() requires a MultiHeadMLP model. "
                "Use fit() for single-survey training."
            )

        if not self.model.is_multi_label:
            raise TypeError(
                "fit_multi_labelset() requires a MultiHeadMLP with multiple label_sources. "
                f"Current model has label_sources: {self.model.label_sources}"
            )

        label_sources = list(y_train.keys())

        # Derive has_labels from y_train/y_val mask channels if not provided
        # A star has labels from a source if any of its mask values are non-zero
        if has_labels_train is None:
            has_labels_train = {}
            for source, y_source in y_train.items():
                if isinstance(y_source, torch.Tensor):
                    mask = (
                        y_source[:, 2, :].numpy()
                        if y_source.ndim == 3
                        else y_source.numpy()
                    )
                else:
                    mask = y_source[:, 2, :] if y_source.ndim == 3 else y_source
                has_labels_train[source] = np.any(mask > 0, axis=1)

        if has_labels_val is None:
            has_labels_val = {}
            for source, y_source in y_val.items():
                if isinstance(y_source, torch.Tensor):
                    mask = (
                        y_source[:, 2, :].numpy()
                        if y_source.ndim == 3
                        else y_source.numpy()
                    )
                else:
                    mask = y_source[:, 2, :] if y_source.ndim == 3 else y_source
                has_labels_val[source] = np.any(mask > 0, axis=1)

        # Convert y_train and y_val to numpy for normalization
        y_train_np = {
            source: arr.numpy() if isinstance(arr, torch.Tensor) else arr
            for source, arr in y_train.items()
        }
        y_val_np = {
            source: arr.numpy() if isinstance(arr, torch.Tensor) else arr
            for source, arr in y_val.items()
        }

        # Get sample count from first label source
        first_source = label_sources[0]
        n_samples = y_train_np[first_source].shape[0]
        n_params = self.n_parameters

        # Apply label normalization if requested
        # Use first label source for fitting normalizer (they should have same scale)
        if normalize_labels:
            # Combine labels from all sources for fitting normalizer
            # Use union of all available labels
            combined_labels = np.zeros((n_samples, n_params), dtype=np.float32)
            combined_mask = np.zeros((n_samples, n_params), dtype=np.float32)

            for source in label_sources:
                source_labels = y_train_np[source][:, 0, :]
                source_mask = y_train_np[source][:, 2, :]
                # Use this source's value where mask is valid and we don't already have a value
                use_source = (source_mask > 0) & (combined_mask == 0)
                combined_labels[use_source] = source_labels[use_source]
                combined_mask[use_source] = 1.0

            self.normalizer = LabelNormalizer(parameters=self.parameter_names)
            # Only fit on valid samples (mask > 0)
            valid_rows = np.any(combined_mask > 0, axis=1)
            if valid_rows.sum() == 0:
                raise ValueError("No valid labels found for normalization")
            self.normalizer.fit(
                combined_labels[valid_rows], mask=combined_mask[valid_rows]
            )
            logger.info(f"Fitted label normalizer using union of {label_sources}")

            # Transform all label sources
            # Replace masked (invalid) values with safe defaults before transform
            # This prevents NaN from log10(0) for Teff
            for source in label_sources:
                labels = y_train_np[source][:, 0, :].copy()
                errors = y_train_np[source][:, 1, :].copy()
                mask = y_train_np[source][:, 2, :]

                # For masked values, use median from combined data to avoid NaN
                for i in range(n_params):
                    invalid = mask[:, i] == 0
                    if invalid.any() and combined_mask[:, i].any():
                        # Use median of valid combined values
                        valid_combined = combined_labels[combined_mask[:, i] > 0, i]
                        labels[invalid, i] = np.median(valid_combined)
                        errors[invalid, i] = 0.1  # Safe default error

                labels_norm, errors_norm = self.normalizer.transform(labels, errors)
                y_train_np[source] = np.stack(
                    [labels_norm, errors_norm, y_train_np[source][:, 2, :]], axis=1
                )

                # Same for validation
                labels_val = y_val_np[source][:, 0, :].copy()
                errors_val = y_val_np[source][:, 1, :].copy()
                mask_val = y_val_np[source][:, 2, :]

                for i in range(n_params):
                    invalid = mask_val[:, i] == 0
                    if invalid.any() and combined_mask[:, i].any():
                        valid_combined = combined_labels[combined_mask[:, i] > 0, i]
                        labels_val[invalid, i] = np.median(valid_combined)
                        errors_val[invalid, i] = 0.1

                labels_val_norm, errors_val_norm = self.normalizer.transform(
                    labels_val, errors_val
                )
                y_val_np[source] = np.stack(
                    [labels_val_norm, errors_val_norm, y_val_np[source][:, 2, :]],
                    axis=1,
                )

        # Convert to tensors
        X_train_tensors = {
            survey: self._to_tensor(arr) for survey, arr in X_train.items()
        }
        X_val_tensors = {
            survey: self._to_tensor(arr).to(self.device)
            for survey, arr in X_val.items()
        }
        y_train_tensors = {
            source: self._to_tensor(arr) for source, arr in y_train_np.items()
        }
        y_val_tensors = {
            source: self._to_tensor(arr).to(self.device)
            for source, arr in y_val_np.items()
        }

        # Convert has_data to tensors
        has_data_train_tensors = {
            survey: torch.as_tensor(arr, dtype=torch.bool)
            for survey, arr in has_data_train.items()
        }
        has_data_val_tensors = {
            survey: torch.as_tensor(arr, dtype=torch.bool).to(self.device)
            for survey, arr in has_data_val.items()
        }
        has_labels_train_tensors = {
            source: torch.as_tensor(arr, dtype=torch.bool)
            for source, arr in has_labels_train.items()
        }
        has_labels_val_tensors = {
            source: torch.as_tensor(arr, dtype=torch.bool).to(self.device)
            for source, arr in has_labels_val.items()
        }

        # Training parameters
        training_config = self.config.training
        batch_size = training_config.batch_size
        n_epochs = training_config.epochs
        steps_per_epoch = (n_samples + batch_size - 1) // batch_size

        # Create scheduler
        self.scheduler = self._create_scheduler(steps_per_epoch)

        # Initialize tracking
        self._best_weights = copy.deepcopy(self.model.state_dict())
        self.history = TrainingHistory(
            parameter_names=self.parameter_names,
            val_loss_breakdown={"mean_component": [], "scatter_component": []},
            val_metrics={name: [] for name in EVALUATOR_METRIC_NAMES},
        )

        self._evaluator = Evaluator(
            parameter_names=self.parameter_names,
            teff_in_log=False,
            scatter_floor=self.config.training.scatter_floor,
        )
        patience_counter = 0
        rng = np.random.default_rng(self.config.seed)

        start_time = time.time()
        logger.info(f"Starting multi-labelset training for {n_epochs} epochs")
        logger.info(f"Surveys: {list(X_train.keys())}")
        logger.info(f"Label sources: {label_sources}")

        # Initialize grokking metrics
        initial_weight_norms = self._compute_weight_norms()
        for layer_name in initial_weight_norms:
            self.history.weight_norms[layer_name] = []
            self.history.weight_updates[layer_name] = []

        # Set up progress bar
        epoch_iter = range(n_epochs)
        use_tqdm = _has_tqdm()
        if use_tqdm:
            from tqdm import tqdm

            epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

        for epoch in epoch_iter:
            prev_weights = self._get_weight_snapshot()

            # Training epoch with multi-labelset
            epoch_train_loss, epoch_grad_norm = self._train_epoch_multi_labelset(
                X_train_tensors,
                y_train_tensors,
                has_data_train_tensors,
                has_labels_train_tensors,
                batch_size,
                rng,
                training_config.gradient_clip,
            )
            self.history.train_losses.append(epoch_train_loss)

            # Grokking metrics
            self.history.grad_norms.append(epoch_grad_norm)
            weight_norms = self._compute_weight_norms()
            weight_updates = self._compute_weight_updates(prev_weights)
            for layer_name in weight_norms:
                self.history.weight_norms[layer_name].append(weight_norms[layer_name])
                self.history.weight_updates[layer_name].append(
                    weight_updates[layer_name]
                )

            # Validation with multi-labelset
            val_details = self._validate_detailed_multi_labelset(
                X_val_tensors,
                y_val_tensors,
                has_data_val_tensors,
                has_labels_val_tensors,
            )
            val_loss = val_details["loss"]
            self.history.val_losses.append(val_loss)

            self.history.val_loss_breakdown["mean_component"].append(
                val_details["mean_component"]
            )
            self.history.val_loss_breakdown["scatter_component"].append(
                val_details["scatter_component"]
            )

            # Compute Evaluator metrics using first label source for simplicity
            # (Could aggregate across sources, but first source is typically primary)
            y_true = y_val_tensors[first_source][:, 0, :].cpu().numpy()
            y_err = y_val_tensors[first_source][:, 1, :].cpu().numpy()
            y_mask = y_val_tensors[first_source][:, 2, :].cpu().numpy()

            self.model.eval()
            with torch.no_grad():
                output = self.model.forward_for_label_source(
                    X_val_tensors, first_source, has_data=has_data_val_tensors
                )
            y_pred = output[:, 0, :].cpu().numpy()
            log_scatter = output[:, 1, :].cpu().numpy()
            pred_scatter = np.sqrt(
                np.exp(2 * log_scatter) + self.config.training.scatter_floor**2
            )

            eval_result = self._evaluator.evaluate(
                y_pred=y_pred,
                y_true=y_true,
                pred_scatter=pred_scatter,
                label_errors=y_err,
                mask=y_mask,
            )

            # Store each metric
            for metric_name in EVALUATOR_METRIC_NAMES:
                metric_values = np.array(
                    [
                        getattr(eval_result.metrics[p], metric_name)
                        for p in self.parameter_names
                    ],
                    dtype=np.float32,
                )
                self.history.val_metrics[metric_name].append(metric_values)

            # Best model tracking
            if val_loss < self.history.best_val_loss:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch
                self._best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # Store learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history.learning_rates.append(current_lr)

            # Progress logging
            if use_tqdm:
                epoch_iter.set_postfix(
                    {
                        "train": f"{epoch_train_loss:.4f}",
                        "val": f"{val_loss:.4f}",
                        "best": f"{self.history.best_val_loss:.4f}",
                    }
                )

            # Periodic checkpoints
            if (
                training_config.save_every > 0
                and (epoch + 1) % training_config.save_every == 0
            ):
                self._save_epoch_checkpoint(epoch + 1)

            # Early stopping
            patience = training_config.early_stopping_patience
            if patience > 0 and patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best weights
        self.model.load_state_dict(self._best_weights)
        self.history.total_time = time.time() - start_time

        logger.info(
            f"Training complete. Best val loss: {self.history.best_val_loss:.6f} "
            f"at epoch {self.history.best_epoch + 1}"
        )
        return self.history

    def _train_epoch_multi_labelset(
        self,
        X: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        has_data: dict[str, torch.Tensor],
        has_labels: dict[str, torch.Tensor],
        batch_size: int,
        rng: np.random.Generator,
        gradient_clip: float,
    ) -> tuple[float, float]:
        """Run one training epoch with multi-survey input and multi-labelset output.

        The loss is computed for each label source separately, with masking applied
        to stars that don't have labels from that source. The total loss is the
        average across all label sources (weighted by the number of valid samples).
        """
        self.model.train()

        # Get sample count from first label source
        label_sources = list(y.keys())
        first_source = label_sources[0]
        n_samples = len(y[first_source])

        # Shuffle indices
        indices = rng.permutation(n_samples)

        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            # Extract batch for each survey
            X_batch = {
                survey: arr[batch_idx].to(self.device) for survey, arr in X.items()
            }
            has_data_batch = {
                survey: arr[batch_idx].to(self.device)
                for survey, arr in has_data.items()
            }

            # Compute loss for each label source
            self.optimizer.zero_grad()

            batch_losses = []
            batch_weights = []

            for source in label_sources:
                y_batch = y[source][batch_idx].to(self.device)
                has_labels_batch = has_labels[source][batch_idx].to(self.device)

                # Get predictions for this label source's output head
                output = self.model.forward_for_label_source(
                    X_batch, source, has_data=has_data_batch
                )

                # Apply label source mask to y
                # Stars without labels from this source should not contribute to loss
                # We do this by zeroing out the mask channel for those stars
                y_masked = y_batch.clone()
                y_masked[:, 2, :] = y_masked[
                    :, 2, :
                ] * has_labels_batch.float().unsqueeze(1)

                # Compute loss (only valid samples contribute due to masking in loss fn)
                loss = self.loss_fn(output, y_masked)
                n_valid = has_labels_batch.sum().item()

                if n_valid > 0:
                    batch_losses.append(loss)
                    batch_weights.append(n_valid)

            if len(batch_losses) > 0:
                # Weighted average of losses across label sources
                total_weight = sum(batch_weights)
                combined_loss = sum(
                    loss * (w / total_weight)
                    for loss, w in zip(batch_losses, batch_weights, strict=False)
                )

                # Backward pass
                combined_loss.backward()

                # Gradient clipping
                if gradient_clip > 0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), gradient_clip
                    )
                else:
                    grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm**0.5

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                total_loss += combined_loss.item()
                total_grad_norm += grad_norm
                n_batches += 1

        if n_batches == 0:
            return 0.0, 0.0
        return total_loss / n_batches, total_grad_norm / n_batches

    def _validate_detailed_multi_labelset(
        self,
        X: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        has_data: dict[str, torch.Tensor],
        has_labels: dict[str, torch.Tensor],
    ) -> dict:
        """Compute validation loss with per-parameter breakdown for multi-labelset."""
        self.model.eval()
        label_sources = list(y.keys())
        n_params = self.n_parameters

        total_loss = 0.0
        total_weight = 0
        mean_component = np.zeros(n_params, dtype=np.float32)
        scatter_component = np.zeros(n_params, dtype=np.float32)

        with torch.no_grad():
            for source in label_sources:
                y_source = y[source]
                has_labels_source = has_labels[source]

                # Get predictions for this label source
                output = self.model.forward_for_label_source(
                    X, source, has_data=has_data
                )

                # Apply label source mask
                y_masked = y_source.clone()
                y_masked[:, 2, :] = y_masked[
                    :, 2, :
                ] * has_labels_source.float().unsqueeze(1)

                # Compute loss
                loss = self.loss_fn(output, y_masked)
                n_valid = has_labels_source.sum().item()

                if n_valid > 0:
                    total_loss += loss.item() * n_valid
                    total_weight += n_valid

                    # Get per-parameter breakdown if available
                    if hasattr(self.loss_fn, "forward_detailed"):
                        details = self.loss_fn.forward_detailed(output, y_masked)
                        mean_component += (
                            details["mean_component"].cpu().numpy() * n_valid
                        )
                        scatter_component += (
                            details["scatter_component"].cpu().numpy() * n_valid
                        )

        if total_weight > 0:
            total_loss /= total_weight
            mean_component /= total_weight
            scatter_component /= total_weight

        return {
            "loss": total_loss,
            "mean_component": mean_component,
            "scatter_component": scatter_component,
        }

    def predict_multi_labelset(
        self,
        X: dict[str, np.ndarray | torch.Tensor],
        has_data: dict[str, np.ndarray | torch.Tensor],
    ) -> dict[str, np.ndarray]:
        """
        Make predictions from all label source output heads.

        This method is used for inference after training with fit_multi_labelset().
        It returns predictions from all label source heads.

        Args:
            X: Dict mapping survey names to spectral data arrays.
                Each array has shape (n_samples, 3, n_wavelengths).
            has_data: Dict mapping survey names to boolean arrays indicating
                which samples have data from that survey.

        Returns:
            Dict mapping label source names to prediction arrays.
            Each array has shape (n_samples, 2, n_params) where:
                - [:, 0, :] contains predicted means
                - [:, 1, :] contains predicted log-scatter

        Raises:
            TypeError: If model is not a MultiHeadMLP with multi-label support.
        """
        if not isinstance(self.model, MultiHeadMLP):
            raise TypeError("predict_multi_labelset() requires a MultiHeadMLP model.")

        if not self.model.is_multi_label:
            raise TypeError(
                "predict_multi_labelset() requires a MultiHeadMLP with multiple label_sources."
            )

        self.model.eval()

        # Convert to tensors and move to device
        X_tensors = {
            survey: self._to_tensor(arr).to(self.device) for survey, arr in X.items()
        }
        has_data_tensors = {
            survey: torch.as_tensor(arr, dtype=torch.bool).to(self.device)
            for survey, arr in has_data.items()
        }

        with torch.no_grad():
            predictions = self.model.forward_all_label_sources(
                X_tensors, has_data=has_data_tensors
            )

        # Convert to numpy
        return {source: output.cpu().numpy() for source, output in predictions.items()}
