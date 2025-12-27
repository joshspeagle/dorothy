"""
Pydantic configuration schemas for DOROTHY experiments.

This module defines the configuration structure for all aspects of a DOROTHY
experiment, including data loading, model architecture, training parameters,
and optional masking strategies.

All configurations use Pydantic v2 for validation and serialization.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SurveyType(str, Enum):
    """Supported astronomical survey types."""

    DESI = "desi"
    BOSS = "boss"
    LAMOST = "lamost"


class NormalizationType(str, Enum):
    """Supported normalization layer types."""

    BATCHNORM = "batchnorm"
    LAYERNORM = "layernorm"


class ActivationType(str, Enum):
    """Supported activation function types."""

    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"


class LossType(str, Enum):
    """Supported loss function types."""

    MSE = "mse"
    HETEROSCEDASTIC = "heteroscedastic"
    PENALTY = "penalty"


class OptimizerType(str, Enum):
    """Supported optimizer types."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class SchedulerType(str, Enum):
    """Supported learning rate scheduler types."""

    NONE = "none"
    ONE_CYCLE = "one_cycle"
    CYCLIC = "cyclic"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    COSINE_ANNEALING = "cosine_annealing"


# The 11 stellar parameters predicted by DOROTHY models
STELLAR_PARAMETERS = [
    "teff",  # Effective temperature (K), normalized in log10 space
    "logg",  # Surface gravity (log g)
    "feh",  # Iron abundance [Fe/H]
    "mgfe",  # Magnesium abundance [Mg/Fe]
    "cfe",  # Carbon abundance [C/Fe]
    "sife",  # Silicon abundance [Si/Fe]
    "nife",  # Nickel abundance [Ni/Fe]
    "alfe",  # Aluminum abundance [Al/Fe]
    "cafe",  # Calcium abundance [Ca/Fe]
    "nfe",  # Nitrogen abundance [N/Fe]
    "mnfe",  # Manganese abundance [Mn/Fe]
]


class DataConfig(BaseModel):
    """Configuration for data loading and preprocessing.

    Attributes:
        fits_path: Path to the input FITS file containing spectra and labels.
        survey: The astronomical survey type (affects wavelength handling).
        input_channels: Number of input channels (typically 2: flux + ivar).
        wavelength_bins: Number of wavelength bins in the spectra.
        train_ratio: Fraction of data for training (default 0.7).
        val_ratio: Fraction of data for validation (default 0.2).
        quality_filter: Whether to apply APOGEE quality flag filtering.
    """

    fits_path: Path = Field(description="Path to input FITS file")
    survey: SurveyType = Field(default=SurveyType.DESI, description="Survey type")
    input_channels: int = Field(
        default=2, ge=1, le=3, description="Number of input channels"
    )
    wavelength_bins: int = Field(
        default=7650,
        ge=1000,
        le=20000,
        description="Number of wavelength bins",
    )
    train_ratio: float = Field(
        default=0.7, gt=0.0, lt=1.0, description="Training data fraction"
    )
    val_ratio: float = Field(
        default=0.2, gt=0.0, lt=1.0, description="Validation data fraction"
    )
    quality_filter: bool = Field(
        default=True, description="Apply APOGEE quality filtering"
    )

    @property
    def test_ratio(self) -> float:
        """Compute test ratio from train and validation ratios."""
        return 1.0 - self.train_ratio - self.val_ratio

    @model_validator(mode="after")
    def validate_ratios(self) -> DataConfig:
        """Ensure train + val ratios don't exceed 1.0."""
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError(
                f"train_ratio ({self.train_ratio}) + val_ratio ({self.val_ratio}) "
                "must be less than 1.0 to leave room for test set"
            )
        return self


class ModelConfig(BaseModel):
    """Configuration for the neural network architecture.

    The default architecture follows the standard DOROTHY MLP:
    15300 -> 5000 -> 2000 -> 1000 -> 500 -> 200 -> 100 -> 22

    Attributes:
        hidden_layers: List of hidden layer sizes.
        normalization: Type of normalization layer (batchnorm or layernorm).
        activation: Activation function type.
        dropout: Dropout probability (0.0 = no dropout).
        input_features: Number of input features (computed from data config).
        output_features: Number of output features (11 params + 11 uncertainties = 22).
    """

    hidden_layers: list[int] = Field(
        default=[5000, 2000, 1000, 500, 200, 100],
        min_length=1,
        description="Hidden layer sizes",
    )
    normalization: NormalizationType = Field(
        default=NormalizationType.LAYERNORM,
        description="Normalization layer type",
    )
    activation: ActivationType = Field(
        default=ActivationType.GELU,
        description="Activation function",
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout probability",
    )
    input_features: int = Field(
        default=15300,
        ge=1,
        description="Number of input features (channels * wavelength_bins)",
    )
    output_features: int = Field(
        default=22,
        ge=2,
        description="Number of outputs (n_params * 2 for mean + uncertainty)",
    )

    @field_validator("hidden_layers")
    @classmethod
    def validate_hidden_layers(cls, v: list[int]) -> list[int]:
        """Ensure all hidden layer sizes are positive."""
        for i, size in enumerate(v):
            if size <= 0:
                raise ValueError(
                    f"Hidden layer {i} has invalid size {size}; must be positive"
                )
        return v

    @property
    def n_parameters(self) -> int:
        """Number of stellar parameters being predicted."""
        return self.output_features // 2


class SchedulerConfig(BaseModel):
    """Configuration for learning rate schedulers.

    Attributes:
        type: The scheduler type to use.
        base_lr: Minimum learning rate (for cyclic schedulers).
        max_lr: Maximum learning rate (for cyclic/one_cycle schedulers).
        step_size_up: Steps to increase from base_lr to max_lr (cyclic).
        step_size_down: Steps to decrease from max_lr to base_lr (cyclic).
        gamma: Multiplicative factor for exp_range mode (cyclic).
        patience: Epochs to wait before reducing LR (for ReduceOnPlateau).
        factor: Factor to reduce LR by (for ReduceOnPlateau).
        pct_start: Fraction of cycle spent increasing LR (one_cycle).
        div_factor: Initial LR = max_lr / div_factor (one_cycle).
        final_div_factor: Final LR = initial_lr / final_div_factor (one_cycle).
        anneal_strategy: Annealing strategy for one_cycle ('cos' or 'linear').
    """

    type: SchedulerType = Field(
        default=SchedulerType.ONE_CYCLE, description="Scheduler type"
    )
    base_lr: float = Field(default=1e-6, gt=0, description="Base learning rate")
    max_lr: float = Field(default=1e-3, gt=0, description="Maximum learning rate")
    step_size_up: int = Field(default=250, ge=1, description="Steps to reach max_lr")
    step_size_down: int = Field(
        default=650, ge=1, description="Steps to descend to base_lr"
    )
    gamma: float = Field(
        default=0.9995, gt=0, le=1.0, description="Exp range decay factor"
    )
    patience: int = Field(default=10, ge=1, description="Patience for ReduceOnPlateau")
    factor: float = Field(default=0.5, gt=0, lt=1.0, description="LR reduction factor")
    # OneCycleLR parameters
    pct_start: float = Field(
        default=0.3, gt=0, lt=1.0, description="Fraction of cycle spent increasing LR"
    )
    div_factor: float = Field(
        default=25.0, ge=1.0, description="Initial LR = max_lr / div_factor"
    )
    final_div_factor: float = Field(
        default=1e4, ge=1.0, description="Final LR = initial_lr / final_div_factor"
    )
    anneal_strategy: Literal["cos", "linear"] = Field(
        default="cos", description="Annealing strategy for one_cycle"
    )

    @model_validator(mode="after")
    def validate_lr_range(self) -> SchedulerConfig:
        """Ensure base_lr <= max_lr."""
        if self.base_lr > self.max_lr:
            raise ValueError(
                f"base_lr ({self.base_lr}) must be <= max_lr ({self.max_lr})"
            )
        return self


class OptimizerConfig(BaseModel):
    """Configuration for optimizers.

    Attributes:
        type: The optimizer type to use.
        weight_decay: L2 regularization coefficient.
        betas: Coefficients for computing running averages (Adam/AdamW).
        momentum: Momentum factor (SGD only).
        nesterov: Whether to use Nesterov momentum (SGD only).
    """

    type: OptimizerType = Field(
        default=OptimizerType.ADAMW, description="Optimizer type"
    )
    weight_decay: float = Field(
        default=0.01, ge=0, description="Weight decay (L2 regularization)"
    )
    betas: tuple[float, float] = Field(
        default=(0.9, 0.999),
        description="Coefficients for running averages (Adam/AdamW)",
    )
    momentum: float = Field(
        default=0.9, ge=0, le=1.0, description="Momentum factor (SGD only)"
    )
    nesterov: bool = Field(
        default=False, description="Use Nesterov momentum (SGD only)"
    )


class TrainingConfig(BaseModel):
    """Configuration for the training process.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate for optimizer.
        loss: Loss function type.
        optimizer: Optimizer configuration.
        scheduler: Learning rate scheduler configuration.
        gradient_clip: Maximum gradient norm for clipping (0 = no clipping).
        save_every: Save checkpoint every N epochs (0 = only save final).
        early_stopping_patience: Epochs without improvement before stopping (0 = disabled).
        scatter_floor: Minimum scatter floor (s_0) for heteroscedastic loss.
    """

    epochs: int = Field(default=300, ge=1, le=10000, description="Number of epochs")
    batch_size: int = Field(default=1024, ge=1, le=65536, description="Batch size")
    learning_rate: float = Field(
        default=1e-3, gt=0, le=1.0, description="Initial learning rate"
    )
    loss: LossType = Field(
        default=LossType.HETEROSCEDASTIC, description="Loss function"
    )
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig, description="Optimizer configuration"
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig, description="LR scheduler"
    )
    gradient_clip: float = Field(
        default=10.0, ge=0, description="Max gradient norm (0=disabled)"
    )
    save_every: int = Field(
        default=0, ge=0, description="Save checkpoint every N epochs"
    )
    early_stopping_patience: int = Field(
        default=0, ge=0, description="Early stopping patience"
    )
    scatter_floor: float = Field(
        default=0.01,
        ge=0,
        le=1.0,
        description="Minimum scatter floor for heteroscedastic loss",
    )


class MaskingConfig(BaseModel):
    """Configuration for dynamic block masking augmentation.

    Masking is a training-time augmentation that randomly masks contiguous
    blocks of the input spectrum to improve model robustness to missing data.
    This simulates real-world scenarios where portions of spectra may be
    unavailable due to bad pixels, cosmic rays, or atmospheric absorption.

    The augmentation works on 3-channel input [flux | error | mask] and updates
    the mask channel by combining the original mask with random block masks.

    Attributes:
        enabled: Whether to apply block masking during training.
        min_fraction: Minimum fraction of wavelengths to mask (0.0 to 1.0).
        max_fraction: Maximum fraction of wavelengths to mask (0.0 to 1.0).
        fraction_choices: Optional list of specific fractions to choose from.
            If provided, min_fraction and max_fraction are ignored.
        min_block_size: Minimum size of each masked block.
        max_block_size: Maximum size of each masked block. If None, defaults
            to n_wavelengths // 2 at runtime.
    """

    enabled: bool = Field(
        default=False, description="Enable block masking augmentation"
    )
    min_fraction: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of wavelengths to mask",
    )
    max_fraction: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of wavelengths to mask",
    )
    fraction_choices: list[float] | None = Field(
        default=None,
        description="Optional list of specific fractions to choose from",
    )
    min_block_size: int = Field(
        default=5, ge=1, description="Minimum size of each masked block"
    )
    max_block_size: int | None = Field(
        default=None, ge=1, description="Maximum size of each masked block"
    )

    @model_validator(mode="after")
    def validate_fractions(self) -> MaskingConfig:
        """Validate fraction parameters."""
        if self.fraction_choices is not None:
            if not all(0 <= f <= 1 for f in self.fraction_choices):
                raise ValueError("All fraction_choices must be between 0 and 1")
        elif self.min_fraction > self.max_fraction:
            raise ValueError(
                f"min_fraction ({self.min_fraction}) must be <= "
                f"max_fraction ({self.max_fraction})"
            )
        return self


class ExperimentConfig(BaseModel):
    """Top-level configuration for a DOROTHY experiment.

    This is the main configuration class that combines all sub-configurations
    and provides the complete specification for training a model.

    Attributes:
        name: Unique name for this experiment.
        description: Human-readable description of the experiment.
        data: Data loading and preprocessing configuration.
        model: Neural network architecture configuration.
        training: Training process configuration.
        masking: Optional masking configuration for robustness.
        output_dir: Directory for saving outputs (checkpoints, logs).
        seed: Random seed for reproducibility.
        device: Device to use for training ("cuda", "cpu", or "auto").
    """

    name: str = Field(min_length=1, max_length=100, description="Experiment name")
    description: str = Field(
        default="", max_length=500, description="Experiment description"
    )
    data: DataConfig = Field(description="Data configuration")
    model: ModelConfig = Field(
        default_factory=ModelConfig, description="Model configuration"
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training configuration",
    )
    masking: MaskingConfig = Field(
        default_factory=MaskingConfig,
        description="Masking configuration",
    )
    output_dir: Path = Field(default=Path("./outputs"), description="Output directory")
    seed: int = Field(default=42, ge=0, description="Random seed")
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto", description="Compute device"
    )

    @model_validator(mode="after")
    def sync_model_input_features(self) -> ExperimentConfig:
        """Synchronize model input features with data configuration."""
        expected_features = self.data.input_channels * self.data.wavelength_bins
        if self.model.input_features != expected_features:
            # Update the model config to match data config
            self.model = self.model.model_copy(
                update={"input_features": expected_features}
            )
        return self

    def get_output_path(self) -> Path:
        """Get the full output path for this experiment."""
        return self.output_dir / self.name

    def get_checkpoint_path(self, epoch: int | None = None) -> Path:
        """Get path for a checkpoint file.

        Args:
            epoch: Epoch number, or None for final/best model.

        Returns:
            Path to the checkpoint file.
        """
        base = self.get_output_path()
        if epoch is None:
            return base / f"{self.name}_final"
        return base / f"epoch_{epoch}"

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the YAML file.
        """
        import yaml

        path = Path(path)

        # Convert to dict with proper serialization
        def serialize(obj):
            """Recursively serialize objects for YAML."""
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, Enum):
                return obj.value
            return obj

        data = serialize(self.model_dump())

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            ExperimentConfig instance.
        """
        import yaml

        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        # Convert fits_path back to Path
        if "data" in data and "fits_path" in data["data"]:
            data["data"]["fits_path"] = Path(data["data"]["fits_path"])
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])

        return cls(**data)
