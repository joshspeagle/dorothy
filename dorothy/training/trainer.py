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

All training modes (single-survey, dense multi-survey, sparse multi-survey and
multi-labelset) share one pipeline: inputs are wrapped in a
:mod:`~dorothy.training.batches` source, labels are normalized by one routine,
and a single epoch loop handles training, validation, scheduling, best-weight
tracking, early stopping and history.
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
from dorothy.losses.heteroscedastic import (
    HeteroscedasticLoss,
    MaskedMSELoss,
    predicted_scatter,
)
from dorothy.models.mlp import MLP
from dorothy.models.multi_head_mlp import MultiHeadMLP
from dorothy.training.batches import (
    SINGLE_SURVEY_KEY,
    Batch,
    SparseBatchSource,
    TensorBatchSource,
    forward_label_sources,
)
from dorothy.utils import resolve_device

if TYPE_CHECKING:
    from dorothy.config.schema import ExperimentConfig, TrainingConfig
    from dorothy.inference.evaluator import EvaluationResult

logger = logging.getLogger(__name__)

# Default batch size for Trainer.predict()
PREDICT_BATCH_SIZE = 2048


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
        val_losses: Per-epoch validation loss values. With several label
            sources this combines all heads (see
            ``training.label_source_weighting``) and selects the best epoch.
        learning_rates: Learning rate after each optimizer step.
        best_epoch: Epoch with the best validation loss (0-based).
        best_val_loss: Best validation loss achieved.
        total_time: Total training time in seconds.
        parameter_names: Names of stellar parameters being tracked.
        val_loss_breakdown: Per-epoch loss breakdown (primary label source) with keys:
            - 'mean_component': (n_epochs, n_params) weighted squared error per param
            - 'scatter_component': (n_epochs, n_params) log-variance penalty per param
        val_metrics: Per-epoch Evaluator metrics (primary label source) as dict
            of lists of arrays. Keys are metric names (rmse, bias, sd, mae,
            median_offset, robust_scatter, z_median, z_robust_scatter,
            pred_unc_p16, pred_unc_p50, pred_unc_p84). Each value is a list of
            arrays with shape (n_params,) per epoch.
        weight_norms: Per-layer L2 weight norms per epoch for grokking detection.
            Keys are layer names, values are lists of norms per epoch.
        grad_norms: Total gradient norm per epoch (average across batches).
        weight_updates: Per-layer weight update magnitudes per epoch.
            Keys are layer names, values are lists of ||W_new - W_old|| per epoch.
        survey_names / per_survey_val_losses / per_survey_val_metrics:
            Per-survey validation loss and metrics (primary label source).
        label_source_names / per_labelset_val_losses / per_labelset_val_metrics:
            Per-head validation loss and metrics for multi-labelset training.
        per_labelset_best_epoch / per_labelset_best_val_loss: Epoch at which
            each head's own validation loss was lowest, and that loss.
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
    per_labelset_best_epoch: dict[str, int] = field(default_factory=dict)
    per_labelset_best_val_loss: dict[str, float] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save training history to a pickle file."""
        data = {
            "history_train": self.train_losses,
            "history_val": self.val_losses,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_time": self.total_time,
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
            "per_labelset_best_epoch": self.per_labelset_best_epoch,
            "per_labelset_best_val_loss": self.per_labelset_best_val_loss,
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


def _is_weight_matrix(param: torch.Tensor) -> bool:
    """Whether weight decay and the grokking metrics apply to a parameter.

    Biases and normalization weights (1-D) set offsets and scales, which
    weight decay does not regularize.
    """
    return param.ndim > 1


def _as_numpy(arr: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return a numpy view/copy of a tensor or array."""
    return arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else arr


def _check_label_array(name: str, arr: np.ndarray) -> None:
    """Validate the 3-channel (n_samples, 3, n_params) label format."""
    if arr.ndim != 3 or arr.shape[1] != 3:
        raise ValueError(
            f"{name} must have shape (n_samples, 3, n_params), got {arr.shape}"
        )


def _mixed_precision(
    training: TrainingConfig, device: torch.device
) -> tuple[torch.dtype | None, GradScaler | None]:
    """The autocast dtype of the forward pass and the gradient scaler.

    Mixed precision runs on CUDA only; elsewhere both are None. bfloat16 has
    the exponent range of float32, so only float16 needs a scaler.

    Raises:
        ValueError: If ``training.amp_dtype`` is bfloat16 and the CUDA device
            does not support bfloat16.
    """
    if not training.amp or device.type != "cuda":
        return None, None
    if training.amp_dtype == "float16":
        return torch.float16, GradScaler()
    if not torch.cuda.is_bf16_supported():
        raise ValueError(
            "training.amp_dtype is 'bfloat16', which this CUDA device does not "
            "support; set training.amp_dtype to 'float16'"
        )
    return torch.bfloat16, None


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
           but uses memory-efficient sparse storage, with one or several label
           sources. This is the path used by ``dorothy train``.

        4. **Multi-Labelset** (fit_multi_labelset): MultiHeadMLP with multiple
           surveys AND multiple label sources from dense arrays.

        All modes run the same pipeline, so they normalize labels, weight
        label sources, validate and select the best epoch identically.

    Training Flow:
        1. Initialize Trainer with ExperimentConfig
        2. Trainer creates model, loss function, optimizer
        3. Call fit*() with training and validation data
        4. fit*() normalizes labels, creates the scheduler and runs the loop
        5. Each epoch: train epoch -> validate -> update history
        6. Best model weights saved based on validation loss
        7. After training: save_checkpoint() to persist model + normalizer

    Attributes:
        config: Experiment configuration.
        model: The neural network model (MLP or MultiHeadMLP).
        optimizer: The optimizer (AdamW by default).
        scheduler: Learning rate scheduler (OneCycleLR by default).
        loss_fn: Loss function (HeteroscedasticLoss).
        device: Compute device (cuda/mps/cpu).
        autocast_dtype: Dtype of the forward pass with mixed precision
            (``training.amp`` on CUDA), or None for float32.
        scaler: Gradient scaler for float16 mixed precision, or None.
        history: TrainingHistory with loss curves and metrics.
        normalizer: LabelNormalizer for converting to/from normalized space.
        parameter_names: Names of stellar parameters being predicted.

    Examples:
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
                If None, uses ``config.parameter_names`` (``data.parameter_names``
                when set, otherwise the first n STELLAR_PARAMETERS).
        """
        self.config = config
        self.history = TrainingHistory()

        self.device = resolve_device(config.device)
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

        self.autocast_dtype, self.scaler = _mixed_precision(
            config.training, self.device
        )
        if self.autocast_dtype is not None:
            logger.info(f"Mixed precision training enabled ({self.autocast_dtype})")

        # Scheduler will be created when we know the number of batches
        self.scheduler: (
            OneCycleLR | CyclicLR | ReduceLROnPlateau | CosineAnnealingLR | None
        ) = None

        # Best model weights, and the weights after the last epoch
        self._best_weights: dict | None = None
        self._final_weights: dict | None = None

        # Set up parameter names for normalization
        n_params = self.n_parameters
        if parameter_names is not None:
            if len(parameter_names) != n_params:
                raise ValueError(
                    f"parameter_names has {len(parameter_names)} entries but "
                    f"model expects {n_params} parameters"
                )
            self.parameter_names = list(parameter_names)
        else:
            self.parameter_names = config.parameter_names

        # Normalizer will be fitted during training
        self.normalizer: LabelNormalizer | None = None

        # Dynamic masking for hierarchical label/input augmentation
        self._label_masking: DynamicLabelMasking | None = None
        self._input_masking: DynamicInputMasking | None = None
        self._survey_wavelengths: dict[str, int] | None = None
        self._evaluator: Evaluator | None = None

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

    @property
    def label_keys(self) -> list[str]:
        """Label source names, in model output-head order.

        A plain MLP (or a single-head MultiHeadMLP without named sources) has
        the single key ``"default"``.
        """
        if isinstance(self.model, MultiHeadMLP):
            return list(self.model.label_sources)
        return ["default"]

    def _create_model(self) -> MLP | MultiHeadMLP:
        """Create the model from configuration.

        Returns MLP for single-survey config, MultiHeadMLP for multi-survey.
        """
        if self.config.is_multi_head:
            return MultiHeadMLP.from_config(self.config.multi_head_model)
        return MLP.from_config(self.config.model)

    def _create_loss_fn(self) -> nn.Module:
        """Create the loss function from configuration."""
        training_config = self.config.training

        if training_config.loss == LossType.MSE:
            return MaskedMSELoss(n_parameters=self.n_parameters)
        return HeteroscedasticLoss(
            scatter_floor=training_config.scatter_floor,
            n_parameters=self.n_parameters,
        )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer from configuration."""
        training_config = self.config.training
        optimizer_config = training_config.optimizer

        # Check if fused optimizers are available (CUDA only, faster by 10-20%)
        use_fused = torch.cuda.is_available() and self.device.type == "cuda"

        params = list(self.model.parameters())
        groups = [
            {"params": [p for p in params if _is_weight_matrix(p)]},
            {
                "params": [p for p in params if not _is_weight_matrix(p)],
                "weight_decay": 0.0,
            },
        ]

        if optimizer_config.type == OptimizerType.ADAMW:
            return torch.optim.AdamW(
                groups,
                lr=training_config.learning_rate,
                betas=optimizer_config.betas,
                weight_decay=optimizer_config.weight_decay,
                fused=use_fused,
            )

        elif optimizer_config.type == OptimizerType.ADAM:
            return torch.optim.Adam(
                groups,
                lr=training_config.learning_rate,
                betas=optimizer_config.betas,
                weight_decay=optimizer_config.weight_decay,
                fused=use_fused,
            )

        elif optimizer_config.type == OptimizerType.SGD:
            return torch.optim.SGD(
                groups,
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

    # ------------------------------------------------------------------
    # Public training API: thin adapters onto the shared pipeline
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        X_val: np.ndarray | torch.Tensor,
        y_val: np.ndarray | torch.Tensor,
        normalize_labels: bool = True,
    ) -> TrainingHistory:
        """
        Train the model on a single survey.

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

        Raises:
            RuntimeError: If no epoch has a finite validation loss.
        """
        key = self.label_keys[0]
        y_train_np, y_val_np = _as_numpy(y_train), _as_numpy(y_val)
        _check_label_array("y_train", y_train_np)
        _check_label_array("y_val", y_val_np)

        y_tr, (y_va,) = self._prepare_labels(
            {key: y_train_np}, [{key: y_val_np}], normalize_labels
        )
        survey_key = SINGLE_SURVEY_KEY
        if isinstance(self.model, MultiHeadMLP):
            if len(self.model.survey_names) != 1:
                raise ValueError(
                    "fit() takes single-survey data; use fit_multi_survey() or "
                    "fit_multi_survey_sparse() for a multi-survey model"
                )
            survey_key = self.model.survey_names[0]
        X_tr = {survey_key: self._to_tensor(X_train)}
        X_va = {survey_key: self._to_tensor(X_val)}
        self._setup_masking({survey_key: X_tr[survey_key].shape[2]})

        train_source = TensorBatchSource(X_tr, self._tensors(y_tr), None, self.device)
        val_source = TensorBatchSource(X_va, self._tensors(y_va), None, self.device)
        return self._run(train_source, val_source, survey_names=[])

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
            RuntimeError: If no epoch has a finite validation loss.
        """
        self._require_multi_head("fit_multi_survey")
        key = self.label_keys[0]
        y_train_np, y_val_np = _as_numpy(y_train), _as_numpy(y_val)
        _check_label_array("y_train", y_train_np)
        _check_label_array("y_val", y_val_np)

        y_tr, (y_va,) = self._prepare_labels(
            {key: y_train_np}, [{key: y_val_np}], normalize_labels
        )
        return self._fit_dense(X_train, X_val, y_tr, y_va, has_data_train, has_data_val)

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

        For a model with several output heads, the labels of each source come
        from ``data.labels_dict``; otherwise ``data.labels`` is used.

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
            ValueError: If the model's label sources are not in the data.
            RuntimeError: If no epoch has a finite validation loss.

        Examples:
            >>> loader = CatalogueLoader("super_catalogue.h5")
            >>> data = loader.load_merged_sparse(["boss", "desi", "lamost_lrs"])
            >>> train_idx, val_idx, test_idx = split_indices(data.n_total)
            >>> history = trainer.fit_multi_survey_sparse(data, train_idx, val_idx)
        """
        self._require_multi_head("fit_multi_survey_sparse")
        train_indices = np.asarray(train_indices)
        val_indices = np.asarray(val_indices)

        keys = self.label_keys
        if self.model.is_multi_label:
            available = set(data.labels_dict or {})
            missing = [k for k in keys if k not in available]
            if missing:
                raise ValueError(
                    f"Model label sources {keys} not all present in the data's "
                    f"label sources {sorted(available)} (missing {missing})"
                )
            labels = {k: data.labels_dict[k] for k in keys}
        else:
            labels = {keys[0]: data.labels}

        if normalize_labels:
            self.normalizer = self._fit_normalizer(
                {k: v[train_indices] for k, v in labels.items()}
            )
            labels = self._normalize_labels(labels)
        else:
            labels = {k: v.astype(np.float32, copy=False) for k, v in labels.items()}

        self._setup_masking(
            {s: len(data.wavelengths[s]) for s in data.surveys}, multi_survey=True
        )
        logger.info(f"Memory usage: {data.memory_usage_mb()['total']:.1f} MB")
        train_source = SparseBatchSource(data, train_indices, labels, self.device)
        val_source = SparseBatchSource(data, val_indices, labels, self.device)
        return self._run(train_source, val_source, survey_names=list(data.surveys))

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
                The normalizer is fit on the union of all label sources (the
                first source's value is used where several have one).

        Returns:
            TrainingHistory with loss curves and metrics.

        Raises:
            TypeError: If model is not a MultiHeadMLP with multi-label support.
            ValueError: If the label sources don't match the model's heads.
            RuntimeError: If no epoch has a finite validation loss.
        """
        self._require_multi_head("fit_multi_labelset")
        if not self.model.is_multi_label:
            raise TypeError(
                "fit_multi_labelset() requires a MultiHeadMLP with multiple label_sources. "
                f"Current model has label_sources: {self.model.label_sources}"
            )
        keys = self.label_keys
        for name, y in [("y_train", y_train), ("y_val", y_val)]:
            if set(y) != set(keys):
                raise ValueError(
                    f"{name} label sources {sorted(y)} do not match the model's "
                    f"label sources {keys}"
                )

        def fold(y_dict, has_labels):
            """Mask out stars a source has no labels for."""
            out = {}
            for k in keys:
                arr = _as_numpy(y_dict[k]).astype(np.float32, copy=True)
                _check_label_array(k, arr)
                if has_labels is not None:
                    arr[:, 2, :] *= _as_numpy(has_labels[k]).astype(bool)[:, None]
                out[k] = arr
            return out

        y_tr, (y_va,) = self._prepare_labels(
            fold(y_train, has_labels_train),
            [fold(y_val, has_labels_val)],
            normalize_labels,
        )
        return self._fit_dense(X_train, X_val, y_tr, y_va, has_data_train, has_data_val)

    # ------------------------------------------------------------------
    # Shared pipeline
    # ------------------------------------------------------------------

    def _require_multi_head(self, method: str) -> None:
        if not isinstance(self.model, MultiHeadMLP):
            raise TypeError(
                f"{method}() requires a MultiHeadMLP model. "
                "Use fit() for single-survey training."
            )

    def _fit_dense(
        self,
        X_train: dict[str, np.ndarray | torch.Tensor],
        X_val: dict[str, np.ndarray | torch.Tensor],
        y_train: dict[str, np.ndarray],
        y_val: dict[str, np.ndarray],
        has_data_train: dict[str, np.ndarray | torch.Tensor],
        has_data_val: dict[str, np.ndarray | torch.Tensor],
    ) -> TrainingHistory:
        """Run the pipeline on dense multi-survey dict inputs."""
        X_tr = {s: self._to_tensor(x) for s, x in X_train.items()}
        X_va = {s: self._to_tensor(x) for s, x in X_val.items()}
        self._setup_masking({s: x.shape[2] for s, x in X_tr.items()}, multi_survey=True)
        train_source = TensorBatchSource(
            X_tr,
            self._tensors(y_train),
            self._bool_tensors(has_data_train),
            self.device,
        )
        val_source = TensorBatchSource(
            X_va, self._tensors(y_val), self._bool_tensors(has_data_val), self.device
        )
        return self._run(train_source, val_source, survey_names=list(X_tr))

    def _fit_normalizer(self, y_train: dict[str, np.ndarray]) -> LabelNormalizer:
        """Fit the label normalizer on training labels from all label sources.

        Where several sources have a valid value for the same star and
        parameter, the first source's value is used.
        """
        arrays = list(y_train.values())
        combined = np.zeros(arrays[0][:, 0, :].shape, dtype=np.float64)
        combined_mask = np.zeros(combined.shape, dtype=bool)
        for arr in arrays:
            use = (arr[:, 2, :] > 0) & ~combined_mask
            combined[use] = arr[:, 0, :][use]
            combined_mask |= use

        rows = combined_mask.any(axis=1)
        if not rows.any():
            raise ValueError("No valid labels found for normalization")
        normalizer = LabelNormalizer(parameters=self.parameter_names)
        normalizer.fit(combined[rows], mask=combined_mask[rows].astype(np.float32))
        logger.info(
            f"Fitted label normalizer on {rows.sum():,} stars "
            f"(label sources: {list(y_train)})"
        )
        return normalizer

    def _normalize_labels(self, y: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Normalize each label source with the fitted normalizer."""
        return {key: self.normalizer.transform_labels(arr) for key, arr in y.items()}

    def _prepare_labels(
        self,
        y_train: dict[str, np.ndarray],
        others: list[dict[str, np.ndarray]],
        normalize: bool,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]]]:
        """Fit the normalizer on y_train and normalize y_train and others."""
        if not normalize:
            return y_train, others
        self.normalizer = self._fit_normalizer(y_train)
        return self._normalize_labels(y_train), [
            self._normalize_labels(y) for y in others
        ]

    def _setup_masking(
        self, survey_wavelengths: dict[str, int], multi_survey: bool = False
    ) -> None:
        """Create the label/input masking augmentations from the config."""
        lm = self.config.label_masking
        self._label_masking = (
            DynamicLabelMasking(
                p_labelset_min=lm.p_labelset_min,
                p_labelset_max=lm.p_labelset_max,
                p_label_min=lm.p_label_min,
                p_label_max=lm.p_label_max,
            )
            if lm.enabled
            else None
        )
        im = self.config.input_masking
        self._input_masking = None
        if im.enabled:
            # A single survey is always kept; only wavelength blocks are masked
            p_survey = (
                (im.p_survey_min, im.p_survey_max) if multi_survey else (1.0, 1.0)
            )
            self._input_masking = DynamicInputMasking(
                p_survey_min=p_survey[0],
                p_survey_max=p_survey[1],
                f_min_override=im.f_min_override,
                f_max=im.f_max,
                p_block_min=im.p_block_min,
                p_block_max=im.p_block_max,
            )
            self._survey_wavelengths = dict(survey_wavelengths)
            logger.info(f"Created input masking: {self._input_masking}")
        if self._label_masking is not None:
            logger.info(f"Created label masking: {self._label_masking}")

    def _apply_input_masking(
        self,
        X_batch: dict[str, torch.Tensor],
        has_data_batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Apply input masking and drop surveys it fully masked from has_data.

        DynamicInputMasking zeroes out whole surveys (including the mask
        channel). The model must then treat that survey as missing, exactly as
        at inference time, rather than encoding an all-zero spectrum.

        The batch tensors are freshly built for this step, so they are masked
        in place.
        """
        X_batch = self._input_masking(X_batch, self._survey_wavelengths, inplace=True)
        has_data_batch = {
            survey: has_data_batch[survey] & (X_batch[survey][:, 2, :] > 0).any(dim=1)
            for survey in has_data_batch
        }
        return X_batch, has_data_batch

    def _forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Model outputs (batch, 2, n_params) for every label source key.

        With mixed precision the model runs under autocast, and its outputs
        are cast to float32, so the loss never runs in reduced precision.
        """
        with autocast(
            self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        ):
            outputs = forward_label_sources(self.model, batch, self.label_keys)
        return {k: v.float() for k, v in outputs.items()}

    def _combine_losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Combine per-label-source losses into one scalar.

        "per_label" pools all valid (star, parameter) labels across sources,
        so every label counts equally. "per_source" averages each source's
        mean loss over the sources with any valid label. Both apply the
        optional ``training.label_source_weights``; with a single source they
        reduce to the loss function's masked mean.
        """
        training = self.config.training
        sums, counts, weights = [], [], []
        for key, target in targets.items():
            sums.append(self.loss_fn.element_losses(outputs[key], target).sum())
            counts.append(target[:, 2, :].sum())
            weights.append(training.label_source_weights.get(key, 1.0))

        if training.label_source_weighting == "per_label":
            numerator = sum(w * s for w, s in zip(weights, sums, strict=True))
            denominator = sum(w * c for w, c in zip(weights, counts, strict=True))
            return torch.where(
                denominator > 0,
                numerator / torch.clamp(denominator, min=1e-12),
                numerator * 0.0,
            )

        # Which sources have any valid label, in one host transfer
        has_labels = torch.stack(counts).gt(0).tolist()
        terms = [
            (w, s / c)
            for w, s, c, ok in zip(weights, sums, counts, has_labels, strict=True)
            if ok
        ]
        if not terms or sum(w for w, _ in terms) == 0:
            return sums[0] * 0.0
        return sum(w * m for w, m in terms) / sum(w for w, _ in terms)

    def _predicted_scatter(self, output: torch.Tensor) -> np.ndarray | None:
        """Predicted scatter for metrics (None if the loss has no scatter)."""
        if isinstance(self.loss_fn, HeteroscedasticLoss):
            return self.loss_fn.get_predicted_scatter(output).numpy()
        return None

    def _metric_arrays(self, result: EvaluationResult) -> dict[str, np.ndarray]:
        """Evaluator metrics as metric name -> (n_params,) arrays."""
        return {
            name: np.array(
                [getattr(result.metrics[p], name) for p in self.parameter_names],
                dtype=np.float32,
            )
            for name in EVALUATOR_METRIC_NAMES
        }

    def _head_inputs(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> dict[str, np.ndarray | None]:
        """Evaluator inputs for one head, as column-major arrays.

        The Evaluator selects each parameter's valid entries column by
        column, which is much faster on column-major copies. The values (and
        their order) are unchanged.
        """
        scatter = self._predicted_scatter(output)
        return {
            "y_pred": np.asfortranarray(output[:, 0, :].numpy()),
            "y_true": np.asfortranarray(target[:, 0, :].numpy()),
            "pred_scatter": None if scatter is None else np.asfortranarray(scatter),
            "label_errors": np.asfortranarray(target[:, 1, :].numpy()),
            "mask": np.asfortranarray(target[:, 2, :].numpy()),
        }

    def _head_metrics(
        self,
        inputs: dict[str, np.ndarray | None],
        rows: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Evaluator metrics for one head, optionally on a subset of rows.

        A subset without rows gives NaN metrics, as
        ``Evaluator.evaluate_by_survey`` does for a survey without samples.
        """
        if rows is not None:
            if not rows.any():
                return {
                    name: np.full(len(self.parameter_names), np.nan, dtype=np.float32)
                    for name in EVALUATOR_METRIC_NAMES
                }
            inputs = {
                k: None if v is None else np.asfortranarray(v[rows])
                for k, v in inputs.items()
            }
        return self._metric_arrays(self._evaluator.evaluate(**inputs))

    def _evaluate(self, source: TensorBatchSource | SparseBatchSource) -> dict:
        """Run validation over a batch source and compute all metrics.

        Returns a dict with the combined loss ('loss', used for model
        selection), the primary head's loss breakdown and metrics, per-survey
        losses/metrics (primary head) and per-head losses/metrics.
        """
        self.model.eval()
        keys = source.label_keys
        outputs = {k: [] for k in keys}
        targets = {k: [] for k in keys}
        has_data: dict[str, list[torch.Tensor]] = {}
        batch_size = self.config.training.batch_size

        with torch.no_grad():
            for batch in source.batches(batch_size):
                batch_out = self._forward(batch)
                for k in keys:
                    outputs[k].append(batch_out[k].cpu())
                    targets[k].append(batch.y[k].cpu())
                for s, h in batch.has_data.items():
                    has_data.setdefault(s, []).append(h.cpu())

        outputs = {k: torch.cat(v) for k, v in outputs.items()}
        targets = {k: torch.cat(v) for k, v in targets.items()}
        has_data_all = {s: torch.cat(v) for s, v in has_data.items()}
        return self._validation_results(outputs, targets, has_data_all)

    def _validation_results(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        has_data: dict[str, torch.Tensor],
    ) -> dict:
        """Losses and Evaluator metrics from concatenated validation outputs.

        Each (head, row subset) is evaluated once: the primary head's overall
        metrics also serve as its per-labelset metrics, and per-survey metrics
        evaluate only that survey's rows.
        """
        keys = list(outputs)
        primary = keys[0]
        details = self.loss_fn.forward_detailed(outputs[primary], targets[primary])
        inputs = self._head_inputs(outputs[primary], targets[primary])
        metrics = self._head_metrics(inputs)

        survey_losses, survey_metrics = {}, {}
        if isinstance(self.model, MultiHeadMLP):
            output, target = outputs[primary], targets[primary]
            for survey, mask in has_data.items():
                survey_losses[survey] = (
                    self.loss_fn(output[mask], target[mask]).item()
                    if mask.any()
                    else float("nan")
                )
                survey_metrics[survey] = self._head_metrics(inputs, mask.numpy())

        labelset_losses, labelset_metrics = {}, {}
        if len(keys) > 1:
            for k in keys:
                has_labels = targets[k][:, 2, :].sum() > 0
                labelset_losses[k] = (
                    self.loss_fn(outputs[k], targets[k]).item()
                    if has_labels
                    else float("nan")
                )
                labelset_metrics[k] = (
                    {name: arr.copy() for name, arr in metrics.items()}
                    if k == primary
                    else self._head_metrics(self._head_inputs(outputs[k], targets[k]))
                )

        return {
            "loss": self._combine_losses(outputs, targets).item(),
            "mean_component": details["mean_component"].numpy(),
            "scatter_component": details["scatter_component"].numpy(),
            "metrics": metrics,
            "per_survey_losses": survey_losses,
            "per_survey_metrics": survey_metrics,
            "per_labelset_losses": labelset_losses,
            "per_labelset_metrics": labelset_metrics,
        }

    def _train_epoch(
        self,
        source: TensorBatchSource | SparseBatchSource,
        batch_size: int,
        rng: np.random.Generator,
        gradient_clip: float,
    ) -> tuple[float, float]:
        """Run one training epoch.

        Args:
            source: Training batch source.
            batch_size: Batch size for training.
            rng: Random number generator for shuffling.
            gradient_clip: Maximum gradient norm for clipping.

        Returns:
            Tuple of (average_loss, average_gradient_norm).
        """
        self.model.train()
        step_scheduler = self.scheduler is not None and not isinstance(
            self.scheduler, ReduceLROnPlateau
        )
        # Summed on the device in float64 (as the Python floats were) and read
        # once per epoch, so steps do not wait for the GPU. MPS has no float64.
        acc_device = self.device if self.device.type != "mps" else torch.device("cpu")
        total_loss = torch.zeros((), dtype=torch.float64, device=acc_device)
        total_grad_norm = torch.zeros((), dtype=torch.float64, device=acc_device)
        n_batches = 0

        for batch in source.batches(batch_size, rng):
            # Augmentations (training only)
            if self._input_masking is not None:
                batch.X, batch.has_data = self._apply_input_masking(
                    batch.X, batch.has_data
                )
            if self._label_masking is not None:
                batch.y = self._label_masking(batch.y)

            self.optimizer.zero_grad()
            loss = self._combine_losses(self._forward(batch), batch.y)

            grad_norm = self._backward_and_step(loss, gradient_clip)
            total_grad_norm += grad_norm.to(acc_device).double()
            if step_scheduler:
                self.scheduler.step()
            self.history.learning_rates.append(self.optimizer.param_groups[0]["lr"])

            total_loss += loss.detach().to(acc_device).double()
            n_batches += 1

        n = max(n_batches, 1)
        return total_loss.item() / n, total_grad_norm.item() / n

    def _run(
        self,
        train_source: TensorBatchSource | SparseBatchSource,
        val_source: TensorBatchSource | SparseBatchSource,
        survey_names: list[str],
    ) -> TrainingHistory:
        """The shared epoch loop used by every fit*() method."""
        training_config = self.config.training
        batch_size = training_config.batch_size
        n_epochs = training_config.epochs
        steps_per_epoch = (train_source.n_samples + batch_size - 1) // batch_size
        self.scheduler = self._create_scheduler(steps_per_epoch)

        # Train/validate the label sources the data provides (a multi-head
        # model fit on one label set trains only that head)
        keys = train_source.label_keys
        multi_label = len(keys) > 1
        labelsets = keys if multi_label else []
        self.history = TrainingHistory(
            parameter_names=self.parameter_names,
            val_loss_breakdown={"mean_component": [], "scatter_component": []},
            val_metrics={name: [] for name in EVALUATOR_METRIC_NAMES},
            survey_names=survey_names,
            per_survey_val_losses={s: [] for s in survey_names},
            per_survey_val_metrics={
                s: {m: [] for m in EVALUATOR_METRIC_NAMES} for s in survey_names
            },
            label_source_names=labelsets,
            per_labelset_val_losses={k: [] for k in labelsets},
            per_labelset_val_metrics={
                k: {m: [] for m in EVALUATOR_METRIC_NAMES} for k in labelsets
            },
        )
        self._evaluator = Evaluator(
            parameter_names=self.parameter_names,
            teff_in_log=False,  # Don't add extra log_teff param during training
            scatter_floor=training_config.scatter_floor,
        )
        self._best_weights = copy.deepcopy(self.model.state_dict())
        patience_counter = 0
        rng = np.random.default_rng(self.config.seed)

        for layer_name in self._compute_weight_norms():
            self.history.weight_norms[layer_name] = []
            self.history.weight_updates[layer_name] = []

        logger.info(
            f"Starting training for {n_epochs} epochs: "
            f"{train_source.n_samples:,} train / {val_source.n_samples:,} val samples, "
            f"surveys={survey_names or ['single']}, label sources={keys}"
        )
        start_time = time.time()

        epoch_iter = range(n_epochs)
        use_tqdm = _has_tqdm()
        if use_tqdm:
            from tqdm import tqdm

            epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

        for epoch in epoch_iter:
            prev_weights = self._get_weight_snapshot()

            train_loss, grad_norm = self._train_epoch(
                train_source, batch_size, rng, training_config.gradient_clip
            )
            self.history.train_losses.append(train_loss)
            self.history.grad_norms.append(grad_norm)
            weight_updates = self._compute_weight_updates(prev_weights)
            for layer_name, norm in self._compute_weight_norms().items():
                self.history.weight_norms[layer_name].append(norm)
                self.history.weight_updates[layer_name].append(
                    weight_updates[layer_name]
                )

            val = self._evaluate(val_source)
            val_loss = val["loss"]
            self._record_validation(epoch, val)

            if val_loss < self.history.best_val_loss:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch
                self._store_best_weights()
                patience_counter = 0
            else:
                patience_counter += 1

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if use_tqdm:
                epoch_iter.set_postfix(
                    train=f"{train_loss:.4f}",
                    val=f"{val_loss:.4f}",
                    best=f"{self.history.best_val_loss:.4f}",
                    lr=f"{current_lr:.1e}",
                )
            elif (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{n_epochs} - Train: {train_loss:.6f}, "
                    f"Val: {val_loss:.6f}, LR: {current_lr:.2e}"
                )

            if (
                training_config.save_every > 0
                and (epoch + 1) % training_config.save_every == 0
            ):
                self._save_epoch_checkpoint(epoch + 1)

            patience = training_config.early_stopping_patience
            if patience > 0 and patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        self.history.total_time = time.time() - start_time
        if not np.isfinite(self.history.best_val_loss):
            raise RuntimeError(
                "No epoch had a finite validation loss, so there is no best "
                "epoch. Check the validation data for NaN or infinite values "
                "and the training losses for divergence."
            )
        self._final_weights = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self._best_weights)
        logger.info(
            f"Training complete in {self.history.total_time:.1f}s. Best val loss: "
            f"{self.history.best_val_loss:.6f} at epoch {self.history.best_epoch + 1}"
        )
        if multi_label:
            for k in keys:
                logger.info(
                    f"  {k}: best val loss "
                    f"{self.history.per_labelset_best_val_loss.get(k, float('nan')):.6f} "
                    f"at epoch {self.history.per_labelset_best_epoch.get(k, -1) + 1}"
                )
        return self.history

    def _record_validation(self, epoch: int, val: dict) -> None:
        """Append one epoch's validation results to the history."""
        history = self.history
        history.val_losses.append(val["loss"])
        history.val_loss_breakdown["mean_component"].append(val["mean_component"])
        history.val_loss_breakdown["scatter_component"].append(val["scatter_component"])
        for name in EVALUATOR_METRIC_NAMES:
            history.val_metrics[name].append(val["metrics"][name])
        for survey in history.survey_names:
            history.per_survey_val_losses[survey].append(
                val["per_survey_losses"][survey]
            )
            for name in EVALUATOR_METRIC_NAMES:
                history.per_survey_val_metrics[survey][name].append(
                    val["per_survey_metrics"][survey][name]
                )
        for k in history.label_source_names:
            loss = val["per_labelset_losses"][k]
            history.per_labelset_val_losses[k].append(loss)
            for name in EVALUATOR_METRIC_NAMES:
                history.per_labelset_val_metrics[k][name].append(
                    val["per_labelset_metrics"][k][name]
                )
            if np.isfinite(loss) and loss < history.per_labelset_best_val_loss.get(
                k, float("inf")
            ):
                history.per_labelset_best_val_loss[k] = loss
                history.per_labelset_best_epoch[k] = epoch

    def _backward_and_step(
        self,
        loss: torch.Tensor,
        gradient_clip: float,
    ) -> torch.Tensor:
        """Backward pass, gradient clipping, and optimizer step.

        With float16 mixed precision, the loss and the gradients go through
        the gradient scaler.

        Args:
            loss: The computed loss tensor to backpropagate.
            gradient_clip: Max gradient norm for clipping (0 or negative = no clip).

        Returns:
            The gradient norm (before clipping if clipping is applied), as a
            0-dim tensor on the device (not read back, to avoid a host sync).
        """
        # Backward pass with gradient scaling for float16
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
        ).detach()

        # Optimizer step through the scaler for float16
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        return grad_norm

    def _store_best_weights(self) -> None:
        """Copy the current weights into the best-weights state dict.

        Copies in place into the state dict made at the start of training,
        instead of deep-copying a new one on every improvement.
        """
        with torch.no_grad():
            for name, tensor in self.model.state_dict().items():
                self._best_weights[name].copy_(tensor)

    def _weight_matrices(self) -> dict[str, torch.Tensor]:
        """Weight matrices tracked for grokking detection (not biases/1D)."""
        return {
            name: param.detach()
            for name, param in self.model.named_parameters()
            if _is_weight_matrix(param)
        }

    @staticmethod
    def _norms(tensors: dict[str, torch.Tensor]) -> dict[str, float]:
        """L2 norm of each tensor, read back in one host transfer."""
        if not tensors:
            return {}
        norms = torch.stack([torch.norm(t) for t in tensors.values()]).tolist()
        return dict(zip(tensors, norms, strict=True))

    def _compute_weight_norms(self) -> dict[str, float]:
        """Compute L2 norms of weight matrices for each layer.

        Returns:
            Dictionary mapping layer names to their L2 norms.
        """
        return self._norms(self._weight_matrices())

    def _get_weight_snapshot(self) -> dict[str, torch.Tensor]:
        """Get a copy of current weight matrices.

        Returns:
            Dictionary mapping layer names to detached weight tensors.
        """
        return {name: w.clone() for name, w in self._weight_matrices().items()}

    def _compute_weight_updates(
        self, prev_weights: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """Compute magnitude of weight updates since last snapshot.

        The snapshot is overwritten with W_old - W_new (whose norm equals
        ||W_new - W_old|| exactly) to avoid allocating the differences.

        Args:
            prev_weights: Previous weight snapshot from _get_weight_snapshot
                (modified in place).

        Returns:
            Dictionary mapping layer names to ||W_new - W_old||.
        """
        weights = self._weight_matrices()
        deltas = {
            name: prev.sub_(weights[name].to(prev.device))
            for name, prev in prev_weights.items()
            if name in weights
        }
        return self._norms(deltas)

    def _to_tensor(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert numpy array to tensor if needed."""
        if isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=torch.float32)
        return data.float()

    def _tensors(self, arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {k: self._to_tensor(v) for k, v in arrays.items()}

    @staticmethod
    def _bool_tensors(
        arrays: dict[str, np.ndarray | torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {k: torch.as_tensor(v, dtype=torch.bool) for k, v in arrays.items()}

    # ------------------------------------------------------------------
    # Checkpoints and prediction
    # ------------------------------------------------------------------

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
            - best_model.pth: Best model weights (also loaded in self.model)
            - final_model.pth: Weights after the last training epoch
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

        # Save final model (last epoch; the model itself holds the best weights)
        final_weights = (
            self._final_weights
            if self._final_weights is not None
            else self.model.state_dict()
        )
        torch.save(final_weights, path / "final_model.pth")

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

    def predict(
        self, X: np.ndarray | torch.Tensor, batch_size: int = PREDICT_BATCH_SIZE
    ) -> np.ndarray:
        """
        Make predictions with the trained model.

        Args:
            X: Input features.
            batch_size: Number of samples per forward pass.

        Returns:
            Predictions as numpy array of shape (n_samples, 2, n_params).
        """
        self.model.eval()
        X_tensor = self._to_tensor(X)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X_tensor), batch_size):
                batch = X_tensor[start : start + batch_size].to(self.device)
                outputs.append(self.model(batch).cpu())
        return torch.cat(outputs).numpy()

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
        predictions = output[:, 0, :]
        uncertainties = predicted_scatter(
            torch.from_numpy(output[:, 1, :]), self.config.training.scatter_floor
        ).numpy()
        return predictions, uncertainties

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
            Dict mapping label source names to prediction arrays of shape
            (n_samples, 2, n_params): predicted means in ``[:, 0, :]`` and
            predicted log-scatter in ``[:, 1, :]``.

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
