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
from torch.amp import GradScaler, autocast
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
from dorothy.data.augmentation import (
    DynamicInputMasking,
    DynamicLabelMasking,
)
from dorothy.data.catalogue_loader import SparseMergedData
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

    # Per-survey metrics for multi-survey training
    survey_names: list[str] = field(default_factory=list)
    per_survey_val_losses: dict[str, list[float]] = field(default_factory=dict)
    per_survey_val_metrics: dict[str, dict[str, list[np.ndarray]]] = field(
        default_factory=dict
    )

    # Per-labelset metrics for multi-labelset training (multiple output heads)
    label_source_names: list[str] = field(default_factory=list)
    per_labelset_val_losses: dict[str, list[float]] = field(default_factory=dict)
    per_labelset_val_metrics: dict[str, dict[str, list[np.ndarray]]] = field(
        default_factory=dict
    )

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
            # Per-survey metrics
            "survey_names": self.survey_names,
            "per_survey_val_losses": self.per_survey_val_losses,
            "per_survey_val_metrics": self.per_survey_val_metrics,
            # Per-labelset metrics
            "label_source_names": self.label_source_names,
            "per_labelset_val_losses": self.per_labelset_val_losses,
            "per_labelset_val_metrics": self.per_labelset_val_metrics,
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
    """Trainer for DOROTHY stellar parameter models.

    Handles the complete training pipeline including model creation,
    optimization, scheduling, validation, checkpointing, and label normalization.

    Training Modes:
        The trainer supports four training modes for different data configurations:

        1. **Single-Survey** (fit): Standard MLP with one spectral survey.
           Use when training on a single survey (e.g., BOSS only).

        2. **Multi-Survey** (fit_multi_survey): MultiHeadMLP with multiple surveys
           sharing a common label source. Use when combining spectra from
           different surveys (e.g., BOSS + DESI) with labels from one source.

        3. **Multi-Survey Sparse** (fit_multi_survey_sparse): Same as multi-survey
           but uses memory-efficient sparse storage. Use for large datasets.

        4. **Multi-Labelset** (fit_multi_labelset): MultiHeadMLP with multiple
           surveys AND multiple label sources. Use when different stars have
           labels from different sources (e.g., APOGEE and GALAH).

    Training Flow:
        1. Initialize Trainer with ExperimentConfig
        2. Trainer creates model, loss function, optimizer
        3. Call fit*() with training and validation data
        4. fit() creates scheduler, normalizer, runs training loop
        5. Each epoch: train_epoch() -> validate() -> update history
        6. Best model weights saved based on validation loss
        7. After training: save_checkpoint() to persist model + normalizer

    Attributes:
        config: Experiment configuration.
        model: The neural network model (MLP or MultiHeadMLP).
        optimizer: The optimizer (AdamW by default).
        scheduler: Learning rate scheduler (OneCycleLR by default).
        loss_fn: Loss function (HeteroscedasticLoss).
        device: Compute device (cuda/cpu).
        history: TrainingHistory with loss curves and metrics.
        normalizer: LabelNormalizer for converting to/from normalized space.
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

        # Mixed precision training (AMP) - disabled by default due to numerical
        # instability with large encoder inputs (causes NaN in multi-survey training)
        # Can be enabled via config.training.amp = True if needed
        self.scaler: GradScaler | None = None
        use_amp = getattr(config.training, "amp", False)
        if use_amp and self.device.type == "cuda":
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled (AMP)")

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

        # Dynamic masking for hierarchical label/input augmentation
        self._label_masking: DynamicLabelMasking | None = None
        self._input_masking: DynamicInputMasking | None = None
        self._survey_wavelengths: dict[str, int] | None = None

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

        # Check if fused optimizers are available (CUDA only, faster by 10-20%)
        use_fused = torch.cuda.is_available() and self.device.type == "cuda"

        if optimizer_config.type == OptimizerType.ADAMW:
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=training_config.learning_rate,
                betas=optimizer_config.betas,
                weight_decay=optimizer_config.weight_decay,
                fused=use_fused,
            )

        elif optimizer_config.type == OptimizerType.ADAM:
            return torch.optim.Adam(
                self.model.parameters(),
                lr=training_config.learning_rate,
                betas=optimizer_config.betas,
                weight_decay=optimizer_config.weight_decay,
                fused=use_fused,
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

        # Create input masking from config if enabled
        # For single-survey, we wrap tensor as dict and use DynamicInputMasking
        if self.config.input_masking.enabled:
            # Get n_wavelengths from input shape
            n_wavelengths = (
                X_train.shape[2]
                if isinstance(X_train, (np.ndarray, torch.Tensor))
                else X_train.shape[2]
            )
            self._survey_wavelengths = {"default": n_wavelengths}
            self._input_masking = DynamicInputMasking(
                p_survey_min=1.0,  # Always keep single survey
                p_survey_max=1.0,
                f_min_override=self.config.input_masking.f_min_override,
                f_max=self.config.input_masking.f_max,
                p_block_min=self.config.input_masking.p_block_min,
                p_block_max=self.config.input_masking.p_block_max,
            )
            logger.info(f"Created input masking: {self._input_masking}")

        # Create label masking from config if enabled
        if self.config.label_masking.enabled:
            self._label_masking = DynamicLabelMasking(
                p_labelset_min=self.config.label_masking.p_labelset_min,
                p_labelset_max=self.config.label_masking.p_labelset_max,
                p_label_min=self.config.label_masking.p_label_min,
                p_label_max=self.config.label_masking.p_label_max,
            )
            logger.info(f"Created label masking: {self._label_masking}")

        # Convert to tensors (y is now 3-channel: [values, errors, mask])
        X_train = self._to_tensor(X_train)
        y_train = self._to_tensor(y_train_np)
        X_val = self._to_tensor(X_val)
        y_val = self._to_tensor(y_val_np)

        # Keep validation data on CPU - _validate_detailed streams batches to device
        # This reduces GPU memory usage by ~90% for large validation sets

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
        if self._input_masking is not None:
            logger.info("Dynamic input masking enabled for training")

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

    def _backward_and_step(
        self,
        loss: torch.Tensor,
        gradient_clip: float,
    ) -> float:
        """Backward pass, gradient clipping, and optimizer step.

        This unified helper handles both AMP (mixed precision) and standard
        training paths, avoiding code duplication across epoch methods.

        Args:
            loss: The computed loss tensor to backpropagate.
            gradient_clip: Max gradient norm for clipping (0 or negative = no clip).

        Returns:
            The gradient norm (before clipping if clipping is applied).
        """
        # Backward pass with gradient scaling for AMP
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            # Unscale gradients before clipping (required for accurate norm)
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        # Gradient clipping: use inf when not clipping to still compute norm
        clip_value = gradient_clip if gradient_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=clip_value,
        ).item()

        # Optimizer step with scaler if using AMP
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        return grad_norm

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

            # Apply input masking if available (training only)
            if self._input_masking is not None:
                X_batch_dict = {"default": X_batch}
                X_batch_dict = self._input_masking(
                    X_batch_dict, self._survey_wavelengths
                )
                X_batch = X_batch_dict["default"]

            # Apply label masking if available (training only)
            if self._label_masking is not None:
                y_batch = self._label_masking(y_batch)

            # Forward pass with mixed precision (AMP)
            self.optimizer.zero_grad()

            # Use autocast for forward pass and loss on CUDA (1.5-2x speedup)
            use_amp = self.scaler is not None
            with autocast(device_type=self.device.type, enabled=use_amp):
                output = self.model(X_batch)
                # Compute loss - y_batch is 3-channel, loss extracts mask from channel 2
                loss = self.loss_fn(output, y_batch)

            # Backward pass, gradient clipping, and optimizer step
            grad_norm = self._backward_and_step(loss, gradient_clip)
            total_grad_norm += grad_norm

            # Step scheduler (for per-step schedulers like CyclicLR)
            if self.scheduler is not None and not isinstance(
                self.scheduler, ReduceLROnPlateau
            ):
                self.scheduler.step()
                self.history.learning_rates.append(self.optimizer.param_groups[0]["lr"])

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches, total_grad_norm / n_batches

    def _validate_detailed(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        batch_size: int = 2048,
    ) -> dict[str, np.ndarray | float]:
        """Run validation with detailed per-parameter loss breakdown.

        Uses batched streaming to reduce GPU memory usage by ~90%.

        Args:
            X_val: Validation features tensor of shape (n_samples, 3, n_wavelengths).
                Can be on CPU; batches are streamed to device.
            y_val: Validation labels tensor of shape (n_samples, 3, n_params) where:
                - Channel 0: label values
                - Channel 1: label errors
                - Channel 2: mask (1=valid, 0=masked)
            batch_size: Batch size for streaming validation (default 2048).

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
        n_samples = len(X_val)

        # Accumulators for batched results
        y_pred_list = []
        pred_scatter_list = []
        total_loss = 0.0
        total_mean_comp = np.zeros(n_params)
        total_scatter_comp = np.zeros(n_params)
        n_batches = 0

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_i = min(i + batch_size, n_samples)
                X_batch = X_val[i:end_i].to(self.device)
                y_batch = y_val[i:end_i].to(self.device)

                output = self.model(X_batch)

                # Get detailed loss breakdown
                if isinstance(self.loss_fn, HeteroscedasticLoss):
                    detailed = self.loss_fn.forward_detailed(output, y_batch)
                    total_loss += detailed["loss"].item()
                    total_mean_comp += detailed["mean_component"].cpu().numpy()
                    total_scatter_comp += detailed["scatter_component"].cpu().numpy()

                    # Output shape is (batch, 2, n_params): [means, log_scatter]
                    y_pred_list.append(output[:, 0, :].cpu().numpy())
                    pred_scatter_list.append(
                        self.loss_fn.get_predicted_scatter(output).cpu().numpy()
                    )
                else:
                    total_loss += self.loss_fn(output, y_batch).item()
                    y_pred_list.append(output[:, 0, :].cpu().numpy())

                n_batches += 1

        # Compute averages
        loss = total_loss / n_batches
        mean_component = total_mean_comp / n_batches
        scatter_component = total_scatter_comp / n_batches

        # Concatenate predictions
        y_pred = np.concatenate(y_pred_list, axis=0)
        pred_scatter = (
            np.concatenate(pred_scatter_list, axis=0) if pred_scatter_list else None
        )

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
        survey_names = list(X_train.keys())
        self._best_weights = copy.deepcopy(self.model.state_dict())
        self.history = TrainingHistory(
            parameter_names=self.parameter_names,
            val_loss_breakdown={"mean_component": [], "scatter_component": []},
            val_metrics={name: [] for name in EVALUATOR_METRIC_NAMES},
            # Initialize per-survey tracking
            survey_names=survey_names,
            per_survey_val_losses={name: [] for name in survey_names},
            per_survey_val_metrics={
                name: {metric: [] for metric in EVALUATOR_METRIC_NAMES}
                for name in survey_names
            },
        )

        self._evaluator = Evaluator(
            parameter_names=self.parameter_names,
            teff_in_log=False,
            scatter_floor=self.config.training.scatter_floor,
        )
        patience_counter = 0
        rng = np.random.default_rng(self.config.seed)

        # Create label masking from config if enabled
        if self.config.label_masking.enabled:
            self._label_masking = DynamicLabelMasking(
                p_labelset_min=self.config.label_masking.p_labelset_min,
                p_labelset_max=self.config.label_masking.p_labelset_max,
                p_label_min=self.config.label_masking.p_label_min,
                p_label_max=self.config.label_masking.p_label_max,
            )
            logger.info(f"Created label masking: {self._label_masking}")

        # Create input masking from config if enabled
        # Build survey wavelengths dict from X_train tensors
        survey_wavelengths = {
            survey: X_train_tensors[survey].shape[2] for survey in survey_names
        }
        if self.config.input_masking.enabled:
            self._input_masking = DynamicInputMasking(
                p_survey_min=self.config.input_masking.p_survey_min,
                p_survey_max=self.config.input_masking.p_survey_max,
                f_min_override=self.config.input_masking.f_min_override,
                f_max=self.config.input_masking.f_max,
                p_block_min=self.config.input_masking.p_block_min,
                p_block_max=self.config.input_masking.p_block_max,
            )
            # Store survey wavelengths for use in training loop
            self._survey_wavelengths = survey_wavelengths
            logger.info(f"Created input masking: {self._input_masking}")
            logger.info(f"Survey wavelengths: {survey_wavelengths}")

        start_time = time.time()
        logger.info(f"Starting multi-survey training for {n_epochs} epochs")
        logger.info(f"Surveys: {survey_names}")

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

            # Per-survey evaluation using has_data (non-exclusive mode)
            has_data_val_np = {
                survey: arr.cpu().numpy()
                for survey, arr in has_data_val_tensors.items()
            }
            survey_eval_result = self._evaluator.evaluate_by_survey(
                y_pred=y_pred,
                y_true=y_true,
                pred_scatter=pred_scatter,
                label_errors=y_err,
                mask=y_mask,
                has_data=has_data_val_np,
            )

            # Store per-survey metrics and compute actual per-survey losses
            for survey_name in survey_names:
                survey_metrics = survey_eval_result.by_survey[survey_name]
                survey_mask = has_data_val_tensors[survey_name]
                n_survey_samples = survey_mask.sum().item()

                # Compute actual per-survey heteroscedastic loss
                if n_survey_samples > 0:
                    # Get outputs and targets for this survey's samples
                    survey_output = output[survey_mask]
                    survey_target = y_val_tensor[survey_mask]
                    # Compute loss for this survey subset
                    survey_loss = self.loss_fn(survey_output, survey_target).item()
                else:
                    survey_loss = float("nan")
                self.history.per_survey_val_losses[survey_name].append(survey_loss)

                # Store each metric for this survey
                for metric_name in EVALUATOR_METRIC_NAMES:
                    metric_values = np.array(
                        [
                            getattr(survey_metrics.metrics[p], metric_name)
                            for p in self.parameter_names
                        ],
                        dtype=np.float32,
                    )
                    self.history.per_survey_val_metrics[survey_name][
                        metric_name
                    ].append(metric_values)

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
        """Run one training epoch with multi-survey data.

        Performs batched training where each batch contains samples from all surveys.
        Uses MultiHeadMLP's forward pass which internally handles missing survey data
        via the has_data masks.

        Args:
            X: Dict mapping survey names to spectral tensors (n_samples, 3, n_wave).
            y: Label tensor of shape (n_samples, 3, n_params) in 3-channel format.
            has_data: Dict mapping survey names to boolean tensors (n_samples,)
                indicating which samples have data from that survey.
            batch_size: Number of samples per batch.
            rng: Random number generator for shuffling.
            gradient_clip: Maximum gradient norm for clipping.

        Returns:
            Tuple of (average_loss, average_gradient_norm) for the epoch.
        """
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

            # Apply input masking if enabled (training only)
            if self._input_masking is not None:
                X_batch = self._input_masking(X_batch, self._survey_wavelengths)

            # Apply label masking if enabled (training only)
            if self._label_masking is not None:
                y_batch = self._label_masking(y_batch)

            # Forward pass through multi-survey model
            self.optimizer.zero_grad()
            output = self.model.forward(X_batch, has_data=has_data_batch)
            loss = self.loss_fn(output, y_batch)

            # Backward pass, gradient clipping, and optimizer step
            grad_norm = self._backward_and_step(loss, gradient_clip)

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
        """Compute validation loss with per-parameter breakdown for multi-survey.

        Runs inference on the validation set and computes detailed metrics including
        per-parameter loss components and Evaluator metrics.

        Args:
            X: Dict mapping survey names to spectral tensors (n_samples, 3, n_wave).
            y: Label tensor of shape (n_samples, 3, n_params) in 3-channel format.
            has_data: Dict mapping survey names to boolean tensors (n_samples,).

        Returns:
            Dict with keys:
            - 'total_loss': Overall validation loss (float).
            - 'mean_component': Per-param weighted squared error, shape (n_params,).
            - 'scatter_component': Per-param log-variance penalty, shape (n_params,).
            - 'evaluator_metrics': Dict of metric_name -> array(n_params).
        """
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

    def _init_sparse_batch_buffers(
        self,
        data: SparseMergedData,
        batch_size: int,
    ) -> None:
        """
        Initialize pinned memory buffers for efficient batch building.

        Pre-allocates pinned (page-locked) memory buffers for each survey's
        flux and ivar data, enabling asynchronous CPU→GPU transfers. The sigma
        computation is done on GPU for maximum throughput.

        This optimization provides ~7x speedup over the baseline approach:
        - Pinned memory enables async transfers (non_blocking=True)
        - GPU sigma computation leverages GPU parallelism
        - Buffer reuse avoids repeated allocation overhead

        Args:
            data: SparseMergedData to get survey shapes from.
            batch_size: Maximum batch size to allocate for.
        """
        use_pinned = self.device.type == "cuda"

        self._sparse_batch_buffers = {}
        for survey in data.surveys:
            n_wave = data.wavelengths[survey].shape[0]
            self._sparse_batch_buffers[survey] = {
                "flux": torch.zeros((batch_size, n_wave), pin_memory=use_pinned),
                "ivar": torch.zeros((batch_size, n_wave), pin_memory=use_pinned),
                "has_data": torch.zeros(
                    batch_size, dtype=torch.bool, pin_memory=use_pinned
                ),
            }
        self._sparse_batch_buffers["labels"] = torch.zeros(
            (batch_size, 3, data.n_params), pin_memory=use_pinned
        )

        logger.debug(
            f"Initialized sparse batch buffers for {len(data.surveys)} surveys, "
            f"batch_size={batch_size}, pinned={use_pinned}"
        )

    def _build_batch_from_sparse(
        self,
        data: SparseMergedData,
        global_indices: np.ndarray,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor | dict[str, torch.Tensor],
    ]:
        """
        Build dense batch tensors from sparse survey data.

        This method constructs on-the-fly dense batches from memory-efficient
        sparse storage. Uses pinned memory and GPU sigma computation for
        ~7x speedup over naive CPU-based approach.

        Args:
            data: SparseMergedData with sparse survey arrays.
            global_indices: Which stars to include in batch (global indices).

        Returns:
            X_batch: Dict[survey -> (batch_size, 3, n_wave)] dense tensor on device.
            has_data_batch: Dict[survey -> (batch_size,)] bool tensor on device.
            y_batch: Either (batch_size, 3, n_params) labels tensor for single-label,
                or Dict[source -> (batch_size, 3, n_params)] for multi-label.
        """
        batch_size = len(global_indices)
        X_batch = {}
        has_data_batch = {}

        # Check if we have pre-allocated buffers (optimized path)
        use_optimized = hasattr(self, "_sparse_batch_buffers")

        for survey in data.surveys:
            n_wave = data.wavelengths[survey].shape[0]

            # Map global indices to local (survey-specific) indices
            local_idx = data.global_to_local[survey][global_indices]
            has_data = local_idx >= 0  # (batch_size,) bool

            if use_optimized:
                # Optimized path: pinned memory + GPU sigma
                buf = self._sparse_batch_buffers[survey]

                # Zero the pinned buffers
                buf["flux"][:batch_size].zero_()
                buf["ivar"][:batch_size].zero_()

                # Fill pinned memory directly via numpy view
                if has_data.any():
                    valid_batch_idx = np.where(has_data)[0]
                    valid_local = local_idx[has_data]
                    buf["flux"][:batch_size].numpy()[valid_batch_idx] = data.flux[
                        survey
                    ][valid_local]
                    buf["ivar"][:batch_size].numpy()[valid_batch_idx] = data.ivar[
                        survey
                    ][valid_local]

                # Async transfer to GPU
                flux_gpu = buf["flux"][:batch_size].to(self.device, non_blocking=True)
                ivar_gpu = buf["ivar"][:batch_size].to(self.device, non_blocking=True)

                # Compute sigma on GPU (much faster than CPU)
                sigma_gpu = torch.zeros_like(ivar_gpu)
                valid_ivar = ivar_gpu > 0
                sigma_gpu[valid_ivar] = 1.0 / torch.sqrt(ivar_gpu[valid_ivar])
                mask_gpu = valid_ivar.float()

                # Stack on GPU
                X_batch[survey] = torch.stack([flux_gpu, sigma_gpu, mask_gpu], dim=1)

                # Transfer has_data
                buf["has_data"][:batch_size] = torch.from_numpy(has_data)
                # On CPU, .to() returns the same tensor (not a copy), causing buffer
                # aliasing issues when collecting batches in validation. Clone to fix.
                if self.device.type == "cpu":
                    has_data_batch[survey] = buf["has_data"][:batch_size].clone()
                else:
                    has_data_batch[survey] = buf["has_data"][:batch_size].to(
                        self.device, non_blocking=True
                    )
            else:
                # Fallback path: original CPU-based approach
                flux_batch = np.zeros((batch_size, n_wave), dtype=np.float32)
                ivar_batch = np.zeros((batch_size, n_wave), dtype=np.float32)

                if has_data.any():
                    valid_batch_idx = np.where(has_data)[0]
                    valid_local = local_idx[has_data]
                    flux_batch[valid_batch_idx] = data.flux[survey][valid_local]
                    ivar_batch[valid_batch_idx] = data.ivar[survey][valid_local]

                sigma_batch = np.zeros_like(ivar_batch)
                valid_ivar = ivar_batch > 0
                sigma_batch[valid_ivar] = 1.0 / np.sqrt(ivar_batch[valid_ivar])

                mask = valid_ivar.astype(np.float32)
                X_survey = np.stack([flux_batch, sigma_batch, mask], axis=1)

                X_batch[survey] = torch.from_numpy(X_survey).to(self.device)
                has_data_batch[survey] = torch.from_numpy(has_data).to(self.device)

        # Labels - handle both single-label and multi-label modes
        if data.labels_dict is not None and data.label_sources is not None:
            # Multi-label mode: return per-source labels dict
            y_batch = {}
            for source in data.label_sources:
                source_labels = data.labels_dict[source][global_indices]
                y_batch[source] = torch.from_numpy(source_labels).to(self.device)
        else:
            # Single-label mode: return single tensor
            if use_optimized:
                label_buf = self._sparse_batch_buffers["labels"]
                label_buf[:batch_size] = torch.from_numpy(data.labels[global_indices])
                # On CPU, .to() returns the same tensor (not a copy), causing buffer
                # aliasing issues when collecting batches in validation. Clone to fix.
                if self.device.type == "cpu":
                    y_batch = label_buf[:batch_size].clone()
                else:
                    y_batch = label_buf[:batch_size].to(self.device, non_blocking=True)
            else:
                y_batch = torch.from_numpy(data.labels[global_indices]).to(self.device)

        return X_batch, has_data_batch, y_batch

    def fit_multi_survey_sparse(
        self,
        data: SparseMergedData,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        normalize_labels: bool = True,
    ) -> TrainingHistory:
        """
        Train the model with memory-efficient sparse multi-survey data.

        This method is designed for large multi-survey datasets where storing
        dense arrays for all surveys would exceed available memory. It builds
        dense batches on-the-fly from sparse storage, reducing memory usage
        by 60-80% compared to fit_multi_survey().

        Memory comparison for 155K stars, 4 surveys:
        - Dense (fit_multi_survey): ~28 GB for spectra
        - Sparse (this method): ~7 GB for spectra + ~50 MB per batch

        Args:
            data: SparseMergedData from CatalogueLoader.load_merged_sparse().
                Contains sparse per-survey spectra and dense labels.
            train_indices: Global indices for training set.
            val_indices: Global indices for validation set.
            normalize_labels: Whether to fit and apply label normalization.

        Returns:
            TrainingHistory with loss curves and metrics.

        Raises:
            TypeError: If model is not a MultiHeadMLP.

        Example:
            >>> loader = CatalogueLoader("super_catalogue.h5")
            >>> data = loader.load_merged_sparse(["boss", "desi", "lamost_lrs"])
            >>> train_idx, val_idx, test_idx = split_indices(data.n_total)
            >>> history = trainer.fit_multi_survey_sparse(data, train_idx, val_idx)
        """
        if not isinstance(self.model, MultiHeadMLP):
            raise TypeError(
                "fit_multi_survey_sparse() requires a MultiHeadMLP model. "
                "Use fit() for single-survey training."
            )

        # Apply label normalization if requested
        y_labels = data.labels.copy()  # Make a copy to avoid modifying original
        y_train_mask = y_labels[train_indices, 2, :]

        # Also copy labels_dict if multi-label mode
        y_labels_dict = None
        if data.labels_dict is not None:
            y_labels_dict = {
                source: labels.copy() for source, labels in data.labels_dict.items()
            }

        if normalize_labels:
            train_labels = y_labels[train_indices, 0, :]

            self.normalizer = LabelNormalizer(parameters=self.parameter_names)
            self.normalizer.fit(train_labels, mask=y_train_mask)
            logger.info(f"Fitted label normalizer on {len(train_labels)} samples")

            # Normalize primary labels (train and val)
            all_labels = y_labels[:, 0, :]
            all_errors = y_labels[:, 1, :]
            all_labels_norm, all_errors_norm = self.normalizer.transform(
                all_labels, all_errors
            )
            y_labels[:, 0, :] = all_labels_norm
            y_labels[:, 1, :] = all_errors_norm

            # Also normalize all label sources in labels_dict
            if y_labels_dict is not None:
                for source in y_labels_dict:
                    source_labels = y_labels_dict[source][:, 0, :]
                    source_errors = y_labels_dict[source][:, 1, :]
                    norm_labels, norm_errors = self.normalizer.transform(
                        source_labels, source_errors
                    )
                    y_labels_dict[source][:, 0, :] = norm_labels
                    y_labels_dict[source][:, 1, :] = norm_errors

        # Create a modified SparseMergedData with normalized labels
        # (Use object.__setattr__ since dataclass may be frozen)
        data_normalized = SparseMergedData(
            flux=data.flux,
            ivar=data.ivar,
            wavelengths=data.wavelengths,
            snr=data.snr,
            global_to_local=data.global_to_local,
            local_to_global=data.local_to_global,
            labels=y_labels,
            gaia_ids=data.gaia_ids,
            ra=data.ra,
            dec=data.dec,
            surveys=data.surveys,
            n_total=data.n_total,
            n_params=data.n_params,
            # Multi-label support
            labels_dict=y_labels_dict,
            has_labels_dict=data.has_labels_dict,
            label_sources=data.label_sources,
        )

        # Training parameters
        training_config = self.config.training
        batch_size = training_config.batch_size
        n_epochs = training_config.epochs
        n_train = len(train_indices)
        steps_per_epoch = (n_train + batch_size - 1) // batch_size

        # Create scheduler
        self.scheduler = self._create_scheduler(steps_per_epoch)

        # Initialize optimized batch buffers (pinned memory for GPU)
        self._init_sparse_batch_buffers(data_normalized, batch_size)

        # Initialize tracking
        survey_names = data.surveys
        self._best_weights = copy.deepcopy(self.model.state_dict())

        # Check if model has multiple label sources (output heads)
        label_sources = getattr(self.model, "label_sources", ["default"])
        is_multi_label = len(label_sources) > 1

        self.history = TrainingHistory(
            parameter_names=self.parameter_names,
            val_loss_breakdown={"mean_component": [], "scatter_component": []},
            val_metrics={name: [] for name in EVALUATOR_METRIC_NAMES},
            survey_names=survey_names,
            per_survey_val_losses={name: [] for name in survey_names},
            per_survey_val_metrics={
                name: {metric: [] for metric in EVALUATOR_METRIC_NAMES}
                for name in survey_names
            },
            # Add labelset tracking if model has multiple output heads
            label_source_names=label_sources if is_multi_label else [],
            per_labelset_val_losses=(
                {name: [] for name in label_sources} if is_multi_label else {}
            ),
            per_labelset_val_metrics=(
                {
                    name: {metric: [] for metric in EVALUATOR_METRIC_NAMES}
                    for name in label_sources
                }
                if is_multi_label
                else {}
            ),
        )

        self._evaluator = Evaluator(
            parameter_names=self.parameter_names,
            teff_in_log=False,
            scatter_floor=self.config.training.scatter_floor,
        )
        patience_counter = 0
        rng = np.random.default_rng(self.config.seed)

        # Create label masking from config if enabled
        if self.config.label_masking.enabled:
            self._label_masking = DynamicLabelMasking(
                p_labelset_min=self.config.label_masking.p_labelset_min,
                p_labelset_max=self.config.label_masking.p_labelset_max,
                p_label_min=self.config.label_masking.p_label_min,
                p_label_max=self.config.label_masking.p_label_max,
            )
            logger.info(f"Created label masking: {self._label_masking}")

        # Create input masking from config if enabled
        # Build survey wavelengths dict from data
        survey_wavelengths = {
            survey: len(data.wavelengths[survey]) for survey in data.surveys
        }
        if self.config.input_masking.enabled:
            self._input_masking = DynamicInputMasking(
                p_survey_min=self.config.input_masking.p_survey_min,
                p_survey_max=self.config.input_masking.p_survey_max,
                f_min_override=self.config.input_masking.f_min_override,
                f_max=self.config.input_masking.f_max,
                p_block_min=self.config.input_masking.p_block_min,
                p_block_max=self.config.input_masking.p_block_max,
            )
            # Store survey wavelengths for use in training loop
            self._survey_wavelengths = survey_wavelengths
            logger.info(f"Created input masking: {self._input_masking}")
            logger.info(f"Survey wavelengths: {survey_wavelengths}")

        start_time = time.time()
        logger.info(f"Starting sparse multi-survey training for {n_epochs} epochs")
        logger.info(f"Surveys: {survey_names}")
        logger.info(
            f"Training samples: {n_train}, Validation samples: {len(val_indices)}"
        )
        logger.info(f"Memory usage: {data.memory_usage_mb()['total']:.1f} MB")

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
            epoch_train_loss, epoch_grad_norm = self._train_epoch_multi_survey_sparse(
                data_normalized,
                train_indices,
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
            val_details = self._validate_detailed_multi_survey_sparse(
                data_normalized, val_indices, batch_size
            )
            val_loss = val_details["loss"]
            self.history.val_losses.append(val_loss)

            self.history.val_loss_breakdown["mean_component"].append(
                val_details["mean_component"]
            )
            self.history.val_loss_breakdown["scatter_component"].append(
                val_details["scatter_component"]
            )

            # Store evaluator metrics
            for metric_name in EVALUATOR_METRIC_NAMES:
                self.history.val_metrics[metric_name].append(
                    val_details["eval_metrics"][metric_name]
                )

            # Store per-survey metrics
            for survey_name in survey_names:
                self.history.per_survey_val_losses[survey_name].append(
                    val_details["per_survey_losses"][survey_name]
                )
                for metric_name in EVALUATOR_METRIC_NAMES:
                    self.history.per_survey_val_metrics[survey_name][
                        metric_name
                    ].append(
                        val_details["per_survey_metrics"][survey_name][metric_name]
                    )

            # Store per-labelset metrics (if multi-label)
            if is_multi_label and val_details.get("per_labelset_losses"):
                for source in label_sources:
                    self.history.per_labelset_val_losses[source].append(
                        val_details["per_labelset_losses"][source]
                    )
                    for metric_name in EVALUATOR_METRIC_NAMES:
                        self.history.per_labelset_val_metrics[source][
                            metric_name
                        ].append(
                            val_details["per_labelset_metrics"][source][metric_name]
                        )

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
            if (
                training_config.early_stopping_patience > 0
                and patience_counter >= training_config.early_stopping_patience
            ):
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(patience={training_config.early_stopping_patience})"
                )
                break

        # Finalize
        elapsed_time = time.time() - start_time
        self.history.total_time = elapsed_time
        self.model.load_state_dict(self._best_weights)

        logger.info(
            f"Training completed in {elapsed_time:.1f}s. "
            f"Best val loss: {self.history.best_val_loss:.4f} at epoch {self.history.best_epoch}"
        )

        return self.history

    def _train_epoch_multi_survey_sparse(
        self,
        data: SparseMergedData,
        train_indices: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
        gradient_clip: float,
    ) -> tuple[float, float]:
        """Run one training epoch with sparse multi-survey data.

        Uses memory-efficient batch construction from SparseMergedData, which stores
        only stars with actual data per survey (reducing memory by 60-80%).

        Args:
            data: SparseMergedData containing spectra, labels, and index mappings.
            train_indices: Global indices of training samples in data.
            batch_size: Number of samples per batch.
            rng: Random number generator for shuffling.
            gradient_clip: Maximum gradient norm for clipping.

        Returns:
            Tuple of (average_loss, average_gradient_norm) for the epoch.
        """
        self.model.train()
        n_samples = len(train_indices)

        # Shuffle training indices
        perm = rng.permutation(n_samples)

        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0

        # Check if multi-label mode (y_batch will be a dict)
        is_multi_label = data.labels_dict is not None

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_perm = perm[start:end]
            global_indices = train_indices[batch_perm]

            # Build dense batch from sparse storage (on-the-fly)
            X_batch, has_data_batch, y_batch = self._build_batch_from_sparse(
                data, global_indices
            )

            # Apply input masking if enabled (training only)
            if self._input_masking is not None:
                X_batch = self._input_masking(X_batch, self._survey_wavelengths)

            # Apply label masking if enabled (training only)
            if self._label_masking is not None:
                y_batch = self._label_masking(y_batch)

            # Forward pass
            self.optimizer.zero_grad()
            use_amp = self.scaler is not None
            with autocast(device_type=self.device.type, enabled=use_amp):
                if is_multi_label and isinstance(y_batch, dict):
                    # Multi-label mode: forward through all output heads
                    per_head_outputs = self.model.forward_all_label_sources(
                        X_batch, has_data=has_data_batch
                    )
                    # Compute loss for each label source and average
                    total_source_loss = 0.0
                    n_sources_with_labels = 0
                    for source, source_output in per_head_outputs.items():
                        source_labels = y_batch[source]
                        # Compute loss (mask is built into the loss function)
                        source_loss = self.loss_fn(source_output, source_labels)
                        total_source_loss += source_loss
                        n_sources_with_labels += 1
                    loss = total_source_loss / max(n_sources_with_labels, 1)
                else:
                    # Single-label mode: standard forward pass
                    output = self.model.forward(X_batch, has_data=has_data_batch)
                    loss = self.loss_fn(output, y_batch)

            # Backward pass, gradient clipping, and optimizer step
            grad_norm = self._backward_and_step(loss, gradient_clip)

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            total_grad_norm += grad_norm
            n_batches += 1

        return total_loss / n_batches, total_grad_norm / n_batches

    def _validate_detailed_multi_survey_sparse(
        self,
        data: SparseMergedData,
        val_indices: np.ndarray,
        batch_size: int,
    ) -> dict:
        """Compute validation loss with per-parameter breakdown for sparse data.

        Streams validation data in batches to avoid memory issues. Uses the same
        sparse batch construction as training.

        Args:
            data: SparseMergedData containing spectra, labels, and index mappings.
            val_indices: Global indices of validation samples in data.
            batch_size: Number of samples per batch for streaming.

        Returns:
            Dict with keys:
            - 'total_loss': Overall validation loss (float).
            - 'mean_component': Per-param weighted squared error, shape (n_params,).
            - 'scatter_component': Per-param log-variance penalty, shape (n_params,).
            - 'evaluator_metrics': Dict of metric_name -> array(n_params).
        """
        self.model.eval()

        # Determine multi-label mode early to handle y_batch correctly
        label_sources = getattr(self.model, "label_sources", ["default"])
        is_multi_label = len(label_sources) > 1
        primary_label_source = label_sources[0] if is_multi_label else None

        all_outputs = []
        all_targets = []
        all_has_data = {survey: [] for survey in data.surveys}

        # Stream validation in batches
        with torch.no_grad():
            for start in range(0, len(val_indices), batch_size):
                end = min(start + batch_size, len(val_indices))
                global_indices = val_indices[start:end]

                X_batch, has_data_batch, y_batch = self._build_batch_from_sparse(
                    data, global_indices
                )

                use_amp = self.scaler is not None
                with autocast(device_type=self.device.type, enabled=use_amp):
                    output = self.model.forward(X_batch, has_data=has_data_batch)

                all_outputs.append(output.cpu())
                # Handle multi-label mode where y_batch is a dict
                if is_multi_label and isinstance(y_batch, dict):
                    # Use primary label source for overall metrics
                    all_targets.append(y_batch[primary_label_source].cpu())
                else:
                    all_targets.append(y_batch.cpu())
                for survey in data.surveys:
                    all_has_data[survey].append(has_data_batch[survey].cpu())

        # Concatenate all batches
        outputs = torch.cat(all_outputs)
        targets = torch.cat(all_targets)
        has_data_all = {
            survey: torch.cat(arrs) for survey, arrs in all_has_data.items()
        }

        # Compute overall loss
        loss = self.loss_fn(outputs, targets)

        # Get per-parameter breakdown
        if hasattr(self.loss_fn, "forward_detailed"):
            details = self.loss_fn.forward_detailed(outputs, targets)
            mean_component = details["mean_component"].numpy()
            scatter_component = details["scatter_component"].numpy()
        else:
            n_params = targets.shape[2]
            mean_component = np.zeros(n_params)
            scatter_component = np.zeros(n_params)

        # Evaluator metrics
        y_true = targets[:, 0, :].numpy()
        y_err = targets[:, 1, :].numpy()
        y_mask = targets[:, 2, :].numpy()
        y_pred = outputs[:, 0, :].numpy()
        log_scatter = outputs[:, 1, :].numpy()
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

        eval_metrics = {}
        for metric_name in EVALUATOR_METRIC_NAMES:
            metric_values = np.array(
                [
                    getattr(eval_result.metrics[p], metric_name)
                    for p in self.parameter_names
                ],
                dtype=np.float32,
            )
            eval_metrics[metric_name] = metric_values

        # Per-survey evaluation
        has_data_np = {survey: arr.numpy() for survey, arr in has_data_all.items()}
        survey_eval_result = self._evaluator.evaluate_by_survey(
            y_pred=y_pred,
            y_true=y_true,
            pred_scatter=pred_scatter,
            label_errors=y_err,
            mask=y_mask,
            has_data=has_data_np,
        )

        per_survey_losses = {}
        per_survey_metrics = {}
        for survey_name in data.surveys:
            survey_mask = has_data_all[survey_name]
            n_survey_samples = survey_mask.sum().item()

            if n_survey_samples > 0:
                survey_output = outputs[survey_mask]
                survey_target = targets[survey_mask]
                survey_loss = self.loss_fn(survey_output, survey_target).item()
            else:
                survey_loss = float("nan")
            per_survey_losses[survey_name] = survey_loss

            survey_metrics = survey_eval_result.by_survey[survey_name]
            per_survey_metrics[survey_name] = {}
            for metric_name in EVALUATOR_METRIC_NAMES:
                metric_values = np.array(
                    [
                        getattr(survey_metrics.metrics[p], metric_name)
                        for p in self.parameter_names
                    ],
                    dtype=np.float32,
                )
                per_survey_metrics[survey_name][metric_name] = metric_values

        # Per-labelset evaluation (if model has multiple output heads)
        per_labelset_losses = {}
        per_labelset_metrics = {}
        # Note: label_sources and is_multi_label already defined at method start

        if is_multi_label:
            # Check if we have per-source labels in data
            has_labels_dict = data.labels_dict is not None

            # Re-run validation to get per-labelset outputs and targets
            all_labelset_outputs = {source: [] for source in label_sources}
            all_labelset_targets = {source: [] for source in label_sources}

            self.model.eval()
            with torch.no_grad():
                for start in range(0, len(val_indices), batch_size):
                    end = min(start + batch_size, len(val_indices))
                    global_indices = val_indices[start:end]
                    X_batch, has_data_batch, y_batch = self._build_batch_from_sparse(
                        data, global_indices
                    )

                    use_amp = self.scaler is not None
                    with autocast(device_type=self.device.type, enabled=use_amp):
                        per_head_outputs = self.model.forward_all_label_sources(
                            X_batch, has_data=has_data_batch
                        )

                    for source in label_sources:
                        all_labelset_outputs[source].append(
                            per_head_outputs[source].cpu()
                        )
                        # Collect per-source targets
                        if has_labels_dict and isinstance(y_batch, dict):
                            all_labelset_targets[source].append(y_batch[source].cpu())
                        else:
                            # Fallback: use primary targets for all sources
                            if isinstance(y_batch, dict):
                                # Should not happen, but handle gracefully
                                fallback_labels = next(iter(y_batch.values()))
                                all_labelset_targets[source].append(
                                    fallback_labels.cpu()
                                )
                            else:
                                all_labelset_targets[source].append(y_batch.cpu())

            # Compute losses and metrics per labelset using per-source targets
            for source in label_sources:
                source_outputs = torch.cat(all_labelset_outputs[source])
                source_targets = torch.cat(all_labelset_targets[source])
                source_loss = self.loss_fn(source_outputs, source_targets).item()
                per_labelset_losses[source] = source_loss

                # Compute metrics for this labelset using per-source targets
                source_pred = source_outputs[:, 0, :].numpy()
                source_log_scatter = source_outputs[:, 1, :].numpy()
                source_scatter = np.sqrt(
                    np.exp(2 * source_log_scatter)
                    + self.config.training.scatter_floor**2
                )

                # Extract per-source ground truth
                source_y_true = source_targets[:, 0, :].numpy()
                source_y_err = source_targets[:, 1, :].numpy()
                source_y_mask = source_targets[:, 2, :].numpy()

                source_eval = self._evaluator.evaluate(
                    y_pred=source_pred,
                    y_true=source_y_true,
                    pred_scatter=source_scatter,
                    label_errors=source_y_err,
                    mask=source_y_mask,
                )

                per_labelset_metrics[source] = {}
                for metric_name in EVALUATOR_METRIC_NAMES:
                    metric_values = np.array(
                        [
                            getattr(source_eval.metrics[p], metric_name)
                            for p in self.parameter_names
                        ],
                        dtype=np.float32,
                    )
                    per_labelset_metrics[source][metric_name] = metric_values

        return {
            "loss": loss.item(),
            "mean_component": mean_component,
            "scatter_component": scatter_component,
            "eval_metrics": eval_metrics,
            "per_survey_losses": per_survey_losses,
            "per_survey_metrics": per_survey_metrics,
            "per_labelset_losses": per_labelset_losses,
            "per_labelset_metrics": per_labelset_metrics,
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
        survey_names = list(X_train.keys())
        self.history = TrainingHistory(
            parameter_names=self.parameter_names,
            val_loss_breakdown={"mean_component": [], "scatter_component": []},
            val_metrics={name: [] for name in EVALUATOR_METRIC_NAMES},
            # Initialize per-survey tracking
            survey_names=survey_names,
            per_survey_val_losses={name: [] for name in survey_names},
            per_survey_val_metrics={
                name: {metric: [] for metric in EVALUATOR_METRIC_NAMES}
                for name in survey_names
            },
            # Initialize per-labelset tracking
            label_source_names=label_sources,
            per_labelset_val_losses={name: [] for name in label_sources},
            per_labelset_val_metrics={
                name: {metric: [] for metric in EVALUATOR_METRIC_NAMES}
                for name in label_sources
            },
        )

        self._evaluator = Evaluator(
            parameter_names=self.parameter_names,
            teff_in_log=False,
            scatter_floor=self.config.training.scatter_floor,
        )
        patience_counter = 0
        rng = np.random.default_rng(self.config.seed)

        # Create label masking from config if enabled
        if self.config.label_masking.enabled:
            self._label_masking = DynamicLabelMasking(
                p_labelset_min=self.config.label_masking.p_labelset_min,
                p_labelset_max=self.config.label_masking.p_labelset_max,
                p_label_min=self.config.label_masking.p_label_min,
                p_label_max=self.config.label_masking.p_label_max,
            )
            logger.info(f"Created label masking: {self._label_masking}")

        # Create input masking from config if enabled
        # Build survey wavelengths dict from X_train tensors
        survey_wavelengths = {
            survey: X_train_tensors[survey].shape[2] for survey in survey_names
        }
        if self.config.input_masking.enabled:
            self._input_masking = DynamicInputMasking(
                p_survey_min=self.config.input_masking.p_survey_min,
                p_survey_max=self.config.input_masking.p_survey_max,
                f_min_override=self.config.input_masking.f_min_override,
                f_max=self.config.input_masking.f_max,
                p_block_min=self.config.input_masking.p_block_min,
                p_block_max=self.config.input_masking.p_block_max,
            )
            # Store survey wavelengths for use in training loop
            self._survey_wavelengths = survey_wavelengths
            logger.info(f"Created input masking: {self._input_masking}")
            logger.info(f"Survey wavelengths: {survey_wavelengths}")

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

            # Per-labelset evaluation
            for source in label_sources:
                # Store per-labelset loss
                source_loss = val_details["per_labelset_losses"].get(
                    source, float("nan")
                )
                self.history.per_labelset_val_losses[source].append(source_loss)

                # Compute per-labelset metrics
                y_source = y_val_tensors[source]
                has_labels_source = has_labels_val_tensors[source]

                y_true_source = y_source[:, 0, :].cpu().numpy()
                y_err_source = y_source[:, 1, :].cpu().numpy()
                y_mask_source = y_source[:, 2, :].cpu().numpy()

                # Get predictions for this label source
                self.model.eval()
                with torch.no_grad():
                    output_source = self.model.forward_for_label_source(
                        X_val_tensors, source, has_data=has_data_val_tensors
                    )
                y_pred_source = output_source[:, 0, :].cpu().numpy()
                log_scatter_source = output_source[:, 1, :].cpu().numpy()
                pred_scatter_source = np.sqrt(
                    np.exp(2 * log_scatter_source)
                    + self.config.training.scatter_floor**2
                )

                # Apply has_labels mask to parameter mask
                y_mask_source = y_mask_source * has_labels_source.cpu().numpy().reshape(
                    -1, 1
                )

                # Evaluate
                source_eval_result = self._evaluator.evaluate(
                    y_pred=y_pred_source,
                    y_true=y_true_source,
                    pred_scatter=pred_scatter_source,
                    label_errors=y_err_source,
                    mask=y_mask_source,
                )

                # Store each metric for this labelset
                for metric_name in EVALUATOR_METRIC_NAMES:
                    metric_values = np.array(
                        [
                            getattr(source_eval_result.metrics[p], metric_name)
                            for p in self.parameter_names
                        ],
                        dtype=np.float32,
                    )
                    self.history.per_labelset_val_metrics[source][metric_name].append(
                        metric_values
                    )

            # Per-survey evaluation (from validation details)
            per_survey_losses = val_details.get("per_survey_losses", {})
            per_survey_metrics = val_details.get("per_survey_metrics", {})
            for survey_name in survey_names:
                survey_loss = per_survey_losses.get(survey_name, float("nan"))
                self.history.per_survey_val_losses[survey_name].append(survey_loss)

                survey_mets = per_survey_metrics.get(survey_name, {})
                for metric_name in EVALUATOR_METRIC_NAMES:
                    metric_values = survey_mets.get(
                        metric_name,
                        np.full(len(self.parameter_names), float("nan")),
                    )
                    self.history.per_survey_val_metrics[survey_name][
                        metric_name
                    ].append(metric_values)

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

        Args:
            X: Dict mapping survey names to spectral tensors (n_samples, 3, n_wave).
            y: Dict mapping label source names to label tensors (n_samples, 3, n_params).
            has_data: Dict mapping survey names to boolean tensors (n_samples,).
            has_labels: Dict mapping label source names to boolean tensors (n_samples,).
            batch_size: Number of samples per batch.
            rng: Random number generator for shuffling.
            gradient_clip: Maximum gradient norm for clipping.

        Returns:
            Tuple of (average_loss, average_gradient_norm) for the epoch.
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

            # Apply input masking if enabled (training only)
            if self._input_masking is not None:
                X_batch = self._input_masking(X_batch, self._survey_wavelengths)

            # Extract all label batches and apply label masking
            y_batch_dict = {
                source: y[source][batch_idx].to(self.device) for source in label_sources
            }
            if self._label_masking is not None:
                y_batch_dict = self._label_masking(y_batch_dict)

            # Compute loss for each label source
            self.optimizer.zero_grad()

            batch_losses = []
            batch_weights = []

            for source in label_sources:
                y_batch = y_batch_dict[source]
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

                # Backward pass, gradient clipping, and optimizer step
                grad_norm = self._backward_and_step(combined_loss, gradient_clip)

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
        """Compute validation loss with per-parameter breakdown for multi-labelset.

        Evaluates each label source separately and averages results weighted by
        the number of valid samples per source. Also computes per-survey metrics.

        Args:
            X: Dict mapping survey names to spectral tensors (n_samples, 3, n_wave).
            y: Dict mapping label source names to label tensors (n_samples, 3, n_params).
            has_data: Dict mapping survey names to boolean tensors (n_samples,).
            has_labels: Dict mapping label source names to boolean tensors (n_samples,).

        Returns:
            Dict with keys:
            - 'loss': Overall validation loss (float).
            - 'mean_component': Per-param weighted squared error, shape (n_params,).
            - 'scatter_component': Per-param log-variance penalty, shape (n_params,).
            - 'per_labelset_losses': Dict of source -> loss.
            - 'per_survey_losses': Dict of survey -> loss.
            - 'per_survey_metrics': Dict of survey -> {metric -> array}.
        """
        self.model.eval()
        label_sources = list(y.keys())
        survey_names = list(X.keys())
        first_source = label_sources[0]
        n_params = self.n_parameters

        total_loss = 0.0
        total_weight = 0
        mean_component = np.zeros(n_params, dtype=np.float32)
        scatter_component = np.zeros(n_params, dtype=np.float32)

        # Per-labelset losses
        per_labelset_losses: dict[str, float] = {}

        # Store outputs and targets for per-survey evaluation
        all_outputs = []
        all_targets = []

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

                # Store per-labelset loss
                per_labelset_losses[source] = (
                    loss.item() if n_valid > 0 else float("nan")
                )

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

                # Store first source outputs for per-survey evaluation
                if source == first_source:
                    all_outputs.append(output)
                    all_targets.append(y_masked)

        if total_weight > 0:
            total_loss /= total_weight
            mean_component /= total_weight
            scatter_component /= total_weight

        # Per-survey evaluation using first label source
        per_survey_losses: dict[str, float] = {}
        per_survey_metrics: dict[str, dict[str, np.ndarray]] = {}

        if all_outputs:
            outputs = all_outputs[0]  # (n_samples, 2, n_params)
            targets = all_targets[0]  # (n_samples, 3, n_params)

            # Extract for evaluator
            y_pred = outputs[:, 0, :].cpu().numpy()
            log_scatter = outputs[:, 1, :].cpu().numpy()
            pred_scatter = np.sqrt(
                np.exp(2 * log_scatter) + self.config.training.scatter_floor**2
            )
            y_true = targets[:, 0, :].cpu().numpy()
            y_err = targets[:, 1, :].cpu().numpy()
            y_mask = targets[:, 2, :].cpu().numpy()

            # Compute per-survey metrics using evaluate_by_survey
            has_data_np = {survey: hd.cpu().numpy() for survey, hd in has_data.items()}

            survey_eval_result = self._evaluator.evaluate_by_survey(
                y_pred=y_pred,
                y_true=y_true,
                pred_scatter=pred_scatter,
                label_errors=y_err,
                mask=y_mask,
                has_data=has_data_np,
            )

            for survey_name in survey_names:
                survey_mask = has_data[survey_name]
                n_survey_samples = survey_mask.sum().item()

                if n_survey_samples > 0:
                    survey_output = outputs[survey_mask]
                    survey_target = targets[survey_mask]
                    survey_loss = self.loss_fn(survey_output, survey_target).item()
                else:
                    survey_loss = float("nan")
                per_survey_losses[survey_name] = survey_loss

                survey_metrics = survey_eval_result.by_survey[survey_name]
                per_survey_metrics[survey_name] = {}
                for metric_name in EVALUATOR_METRIC_NAMES:
                    metric_values = np.array(
                        [
                            getattr(survey_metrics.metrics[p], metric_name)
                            for p in self.parameter_names
                        ],
                        dtype=np.float32,
                    )
                    per_survey_metrics[survey_name][metric_name] = metric_values

        return {
            "loss": total_loss,
            "mean_component": mean_component,
            "scatter_component": scatter_component,
            "per_labelset_losses": per_labelset_losses,
            "per_survey_losses": per_survey_losses,
            "per_survey_metrics": per_survey_metrics,
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
