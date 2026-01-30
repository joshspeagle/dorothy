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
    "fe_h",  # Iron abundance [Fe/H]
    "mg_fe",  # Magnesium abundance [Mg/Fe]
    "c_fe",  # Carbon abundance [C/Fe]
    "si_fe",  # Silicon abundance [Si/Fe]
    "ni_fe",  # Nickel abundance [Ni/Fe]
    "al_fe",  # Aluminum abundance [Al/Fe]
    "ca_fe",  # Calcium abundance [Ca/Fe]
    "n_fe",  # Nitrogen abundance [N/Fe]
    "mn_fe",  # Manganese abundance [Mn/Fe]
]


class DataConfig(BaseModel):
    """Configuration for data loading and preprocessing.

    Loads data from the super-catalogue HDF5 file which contains multiple
    surveys with cross-matched labels from APOGEE and/or GALAH.

    Attributes:
        catalogue_path: Path to HDF5 super-catalogue.
        surveys: List of input surveys to load (e.g., ['boss', 'lamost_lrs']).
            If a single survey, loads that survey only.
            If multiple surveys, creates merged dataset with outer join.
        label_sources: List of label sources to use for training targets.
            e.g., ['apogee'] for single-source, ['apogee', 'galah'] for multi-target.
        train_ratio: Fraction of data for training (default 0.7).
        val_ratio: Fraction of data for validation (default 0.2).
        max_flag_bits: Maximum allowed flag bits (0=highest quality).
        duplicate_labels: Dict mapping target label source to source label source.
            Used for testing multi-labelset training when some label sources don't exist.
            e.g., {'galah': 'apogee'} copies APOGEE labels to create "fake" GALAH labels.
    """

    catalogue_path: Path = Field(description="Path to HDF5 super-catalogue")
    surveys: list[str] = Field(
        default=["boss"],
        min_length=1,
        description="List of input surveys to load (e.g., ['boss', 'lamost_lrs'])",
    )
    label_sources: list[str] = Field(
        default=["apogee"],
        min_length=1,
        description="List of label sources for training targets (e.g., ['apogee', 'galah'])",
    )
    train_ratio: float = Field(
        default=0.7, gt=0.0, lt=1.0, description="Training data fraction"
    )
    val_ratio: float = Field(
        default=0.2, gt=0.0, lt=1.0, description="Validation data fraction"
    )
    max_flag_bits: int = Field(
        default=0,
        ge=0,
        description="Maximum allowed flag bits (0=highest quality)",
    )
    duplicate_labels: dict[str, str] | None = Field(
        default=None,
        description="Map target label source to source (e.g., {'galah': 'apogee'})",
    )
    use_dense_loading: bool = Field(
        default=False,
        description="Use dense loading for spectra (higher memory, ~40GB vs ~7GB sparse)",
    )

    @property
    def test_ratio(self) -> float:
        """Compute test ratio from train and validation ratios."""
        return 1.0 - self.train_ratio - self.val_ratio

    @property
    def is_multi_survey(self) -> bool:
        """Whether loading from multiple input surveys."""
        return len(self.surveys) > 1

    @property
    def is_multi_label(self) -> bool:
        """Whether training with multiple label sources."""
        return len(self.label_sources) > 1

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


class CombinationMode(str, Enum):
    """How to combine multi-survey embeddings."""

    MEAN = "mean"
    CONCAT = "concat"


class MultiHeadModelConfig(BaseModel):
    """Configuration for multi-head MLP architecture.

    Used when training on multiple surveys with different wavelength grids.
    Each survey has its own encoder, feeding into a shared trunk.
    Optionally supports multiple output heads for different label sources.

    Attributes:
        survey_wavelengths: Dict mapping survey names to wavelength counts.
        n_parameters: Number of stellar parameters to predict.
        latent_dim: Output dimension of each survey encoder.
        encoder_hidden: Hidden layer sizes for encoders.
        trunk_hidden: Hidden layer sizes for shared trunk.
        output_hidden: Hidden layer sizes for output head(s).
        combination_mode: How to combine multi-survey embeddings.
        label_sources: List of label sources for multi-output heads.
        normalization: Type of normalization layer.
        activation: Activation function type.
        dropout: Dropout probability.

    Example:
        >>> config = MultiHeadModelConfig(
        ...     survey_wavelengths={"boss": 4506, "lamost_lrs": 3700},
        ...     n_parameters=11,
        ...     latent_dim=256,
        ...     label_sources=["apogee", "galah"],  # Multi-output heads
        ... )
    """

    survey_wavelengths: dict[str, int] = Field(
        description="Dict mapping survey names to wavelength bin counts",
    )
    n_parameters: int = Field(
        default=11,
        ge=1,
        description="Number of stellar parameters to predict",
    )
    latent_dim: int = Field(
        default=256,
        ge=16,
        description="Output dimension of each survey encoder",
    )
    encoder_hidden: list[int] = Field(
        default=[1024, 512],
        min_length=1,
        description="Hidden layer sizes for survey encoders",
    )
    trunk_hidden: list[int] = Field(
        default=[512, 256],
        min_length=1,
        description="Hidden layer sizes for shared trunk",
    )
    output_hidden: list[int] = Field(
        default=[64],
        min_length=1,
        description="Hidden layer sizes for output head(s)",
    )
    combination_mode: CombinationMode = Field(
        default=CombinationMode.CONCAT,
        description="How to combine multi-survey embeddings",
    )
    label_sources: list[str] | None = Field(
        default=None,
        description="Label sources for multi-output heads (e.g., ['apogee', 'galah']). "
        "If None, creates single output head.",
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

    @property
    def is_multi_label(self) -> bool:
        """Whether using multiple output heads for different label sources."""
        return self.label_sources is not None and len(self.label_sources) > 1

    @field_validator("survey_wavelengths")
    @classmethod
    def validate_survey_wavelengths(cls, v: dict[str, int]) -> dict[str, int]:
        """Ensure survey wavelengths are valid."""
        if not v:
            raise ValueError("survey_wavelengths cannot be empty")
        for name, n_wave in v.items():
            if n_wave <= 0:
                raise ValueError(
                    f"Survey '{name}' has invalid wavelength count {n_wave}"
                )
        return v

    @property
    def output_features(self) -> int:
        """Number of output features (2 * n_parameters)."""
        return 2 * self.n_parameters

    @property
    def survey_names(self) -> list[str]:
        """List of survey names."""
        return list(self.survey_wavelengths.keys())


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
    amp: bool = Field(
        default=False,
        description="Enable mixed precision training (AMP). Disabled by default "
        "due to numerical instability with large encoder inputs.",
    )


class LabelMaskingConfig(BaseModel):
    """Configuration for dynamic label masking during training.

    Label masking is a training-time augmentation that randomly masks labels
    to improve model robustness when not all labels are available. This uses
    a hierarchical scheme:

    1. Labelset level: Which label sources (apogee, galah, etc.) to include
    2. Label level: Which parameters (teff, logg, feh, etc.) within each set

    All probabilities are sampled fresh per batch from uniform ranges to ensure
    the model sees diverse masking conditions during training.

    A guaranteed keeper mechanism ensures at least one labelset and one label
    per labelset are always kept, preventing complete masking.

    Attributes:
        enabled: Whether to apply label masking during training.
        p_labelset_min: Minimum probability of keeping each labelset.
        p_labelset_max: Maximum probability of keeping each labelset.
        p_label_min: Minimum probability of keeping each label within kept sets.
        p_label_max: Maximum probability of keeping each label within kept sets.
    """

    enabled: bool = Field(
        default=False, description="Enable label masking augmentation"
    )
    p_labelset_min: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum probability of keeping each labelset (sampled per batch)",
    )
    p_labelset_max: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum probability of keeping each labelset (sampled per batch)",
    )
    p_label_min: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum probability of keeping each label (sampled per batch)",
    )
    p_label_max: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum probability of keeping each label (sampled per batch)",
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> LabelMaskingConfig:
        """Validate probability ranges."""
        if self.p_labelset_min > self.p_labelset_max:
            raise ValueError(
                f"p_labelset_min ({self.p_labelset_min}) must be <= "
                f"p_labelset_max ({self.p_labelset_max})"
            )
        if self.p_label_min > self.p_label_max:
            raise ValueError(
                f"p_label_min ({self.p_label_min}) must be <= "
                f"p_label_max ({self.p_label_max})"
            )
        return self


class InputMaskingConfig(BaseModel):
    """Configuration for dynamic input (spectrum) masking during training.

    Input masking is a training-time augmentation that randomly masks portions
    of input spectra to improve model robustness. This uses a hierarchical scheme:

    1. Survey level: Which surveys to include (for multi-survey training)
    2. Block level: Which wavelength blocks within each kept survey

    Block sizes are sampled log-uniformly to explore all scales from single
    pixels up to large spectral regions. All probabilities are sampled fresh
    per batch from uniform ranges.

    A guaranteed keeper mechanism ensures at least one survey and one block
    per survey are always kept, preventing complete masking.

    Attributes:
        enabled: Whether to apply input masking during training.
        p_survey_min: Minimum probability of keeping each survey.
        p_survey_max: Maximum probability of keeping each survey.
        f_min_override: Optional override for minimum block size fraction.
            If None, defaults to 1/N_wavelengths (single pixel).
        f_max: Maximum block size as fraction of spectrum (default 0.5).
        p_block_min: Minimum probability of keeping each block.
        p_block_max: Maximum probability of keeping each block.
    """

    enabled: bool = Field(
        default=False, description="Enable input masking augmentation"
    )
    p_survey_min: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum probability of keeping each survey (sampled per batch)",
    )
    p_survey_max: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum probability of keeping each survey (sampled per batch)",
    )
    f_min_override: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override for minimum block size fraction. "
        "If None, defaults to 1/N_wavelengths (single pixel).",
    )
    f_max: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Maximum block size as fraction of spectrum",
    )
    p_block_min: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum probability of keeping each block (sampled per batch)",
    )
    p_block_max: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum probability of keeping each block (sampled per batch)",
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> InputMaskingConfig:
        """Validate probability and fraction ranges."""
        if self.p_survey_min > self.p_survey_max:
            raise ValueError(
                f"p_survey_min ({self.p_survey_min}) must be <= "
                f"p_survey_max ({self.p_survey_max})"
            )
        if self.p_block_min > self.p_block_max:
            raise ValueError(
                f"p_block_min ({self.p_block_min}) must be <= "
                f"p_block_max ({self.p_block_max})"
            )
        if self.f_min_override is not None and self.f_min_override > self.f_max:
            raise ValueError(
                f"f_min_override ({self.f_min_override}) must be <= "
                f"f_max ({self.f_max})"
            )
        return self


class ExperimentConfig(BaseModel):
    """Top-level configuration for a DOROTHY experiment.

    This is the main configuration class that combines all sub-configurations
    and provides the complete specification for training a model.

    For single-survey training, use the `model` config (standard MLP).
    For multi-survey training, use `multi_head_model` config (MultiHeadMLP).

    Attributes:
        name: Unique name for this experiment.
        description: Human-readable description of the experiment.
        data: Data loading and preprocessing configuration.
        model: Standard MLP configuration (for single-survey training).
        multi_head_model: Multi-head MLP configuration (for multi-survey training).
        training: Training process configuration.
        label_masking: Dynamic label masking configuration for robustness.
        input_masking: Dynamic input masking configuration for robustness.
        output_dir: Directory for saving outputs (checkpoints, logs).
        seed: Random seed for reproducibility.
        device: Device to use for training ("cuda", "cpu", or "auto").
    """

    name: str = Field(min_length=1, max_length=100, description="Experiment name")
    description: str = Field(
        default="", max_length=500, description="Experiment description"
    )
    data: DataConfig = Field(description="Data configuration")
    model: ModelConfig | None = Field(
        default=None, description="Standard MLP configuration (single-survey)"
    )
    multi_head_model: MultiHeadModelConfig | None = Field(
        default=None, description="Multi-head MLP configuration (multi-survey)"
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training configuration",
    )
    label_masking: LabelMaskingConfig = Field(
        default_factory=LabelMaskingConfig,
        description="Dynamic label masking configuration",
    )
    input_masking: InputMaskingConfig = Field(
        default_factory=InputMaskingConfig,
        description="Dynamic input masking configuration",
    )
    output_dir: Path = Field(default=Path("./outputs"), description="Output directory")
    seed: int = Field(default=42, ge=0, description="Random seed")
    device: Literal["cuda", "cpu", "mps", "auto"] = Field(
        default="auto", description="Compute device (mps for Apple Silicon)"
    )

    @property
    def is_multi_head(self) -> bool:
        """Whether using multi-head architecture."""
        return self.multi_head_model is not None

    @model_validator(mode="after")
    def validate_model_config(self) -> ExperimentConfig:
        """Ensure model configuration is consistent with data configuration."""
        # If neither model is specified, create default based on data config
        if self.model is None and self.multi_head_model is None:
            if self.data.is_multi_survey:
                raise ValueError(
                    "Multi-survey training requires 'multi_head_model' configuration. "
                    f"Surveys: {self.data.surveys}"
                )
            # Default to standard MLP for single-survey
            object.__setattr__(self, "model", ModelConfig())

        # Warn if using standard MLP with multi-survey data
        if self.model is not None and self.data.is_multi_survey:
            import warnings

            warnings.warn(
                f"Using standard MLP with multi-survey data ({self.data.surveys}). "
                "Consider using 'multi_head_model' for better performance.",
                UserWarning,
                stacklevel=2,
            )

        # Ensure only one model config is specified
        if self.model is not None and self.multi_head_model is not None:
            raise ValueError(
                "Cannot specify both 'model' and 'multi_head_model'. "
                "Use 'model' for single-survey or 'multi_head_model' for multi-survey."
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

        # Convert path fields back to Path objects
        if "data" in data and "catalogue_path" in data["data"]:
            data["data"]["catalogue_path"] = Path(data["data"]["catalogue_path"])
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])

        return cls(**data)
