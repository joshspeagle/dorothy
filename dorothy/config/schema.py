"""
Pydantic configuration schemas for DOROTHY experiments.

This module defines the configuration structure for all aspects of a DOROTHY
experiment, including data loading, model architecture, training parameters,
and optional masking strategies.

All configurations use Pydantic v2 for validation and serialization.
"""

from __future__ import annotations

import typing
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class _ConfigModel(BaseModel):
    """Base for config sections: unknown keys are errors, not silently ignored."""

    model_config = ConfigDict(extra="forbid")


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

# Display names for parameters that don't follow the "x_h" / "x_fe" pattern
_SPECIAL_DISPLAY_NAMES = {
    "teff": r"$T_{\rm eff}$",
    "logg": r"$\log g$",
}


def parameter_display_name(name: str) -> str:
    """Return a plot label for a parameter name.

    Abundance names of the form ``x_h`` / ``x_fe`` become ``[X/H]`` / ``[X/Fe]``
    (e.g. ``mg_fe`` -> ``[Mg/Fe]``, ``fe_h`` -> ``[Fe/H]``). Unrecognized names
    are returned unchanged.

    Args:
        name: Parameter name as stored in the catalogue.

    Returns:
        Human-readable (matplotlib mathtext) label.
    """
    if name in _SPECIAL_DISPLAY_NAMES:
        return _SPECIAL_DISPLAY_NAMES[name]
    parts = name.split("_")
    if len(parts) == 2 and parts[1] in ("h", "fe") and parts[0].isalpha():
        return f"[{parts[0].capitalize()}/{parts[1].capitalize()}]"
    return name


class DataConfig(_ConfigModel):
    """Configuration for data loading and preprocessing.

    Loads data from the super-catalogue HDF5 file which contains multiple
    surveys with cross-matched labels from APOGEE and/or GALAH.

    """

    catalogue_path: Path = Field(description="Path to HDF5 super-catalogue")
    surveys: list[str] = Field(
        default=["boss"],
        min_length=1,
        description="List of input surveys to load (e.g., ['boss', 'lamost_lrs']). "
        "Several surveys give a merged dataset (outer join).",
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
        ge=-1,
        description="Maximum allowed flag bits (0=highest quality, -1=allow all with per-param masking)",
    )
    duplicate_labels: dict[str, str] | None = Field(
        default=None,
        description="Map target label source to source, to test multi-labelset "
        "training when a label source does not exist (e.g., {'galah': 'apogee'} "
        "copies the APOGEE labels as GALAH labels)",
    )
    use_dense_loading: bool = Field(
        default=False,
        description="Use dense loading for spectra (more memory than sparse loading)",
    )
    parameter_names: list[str] | None = Field(
        default=None,
        min_length=1,
        description="Label parameter names in catalogue order. dorothy train fills "
        "them from the catalogue; when None, the first n STELLAR_PARAMETERS "
        "(n from the model section) are used, then param_<i>.",
    )
    normalize_spectra: bool = Field(
        default=True,
        description="Per-spectrum median/IQR flux normalization, with ivar rescaled "
        "to match. Training and inference must agree; the checkpoint config "
        "stores it.",
    )
    precision_encoding: bool = Field(
        default=True,
        description="Replace the sigma channel of the model inputs with "
        "log(1 + 1/sigma) at valid pixels and 0 at masked pixels, after "
        "normalization (encode_precision). False keeps sigma and exists only to "
        "load checkpoints trained with it; training rejects it.",
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

    @model_validator(mode="before")
    @classmethod
    def accept_singular_keys(cls, values: Any) -> Any:
        """Accept ``survey`` / ``label_source`` as aliases of the list fields.

        A string becomes a one-element list.

        Raises:
            ValueError: If a singular key and its plural are both given.
        """
        if not isinstance(values, dict):
            return values
        values = dict(values)
        for single, plural in (
            ("survey", "surveys"),
            ("label_source", "label_sources"),
        ):
            if single in values:
                if plural in values:
                    raise ValueError(f"Use either '{single}' or '{plural}', not both")
                value = values.pop(single)
                values[plural] = [value] if isinstance(value, str) else list(value)
        return values

    @field_validator("parameter_names")
    @classmethod
    def validate_parameter_names(cls, v: list[str] | None) -> list[str] | None:
        """Ensure parameter names are unique."""
        if v is not None and len(set(v)) != len(v):
            raise ValueError(f"parameter_names contains duplicates: {v}")
        return v

    @model_validator(mode="after")
    def validate_ratios(self) -> DataConfig:
        """Ensure train + val ratios don't exceed 1.0."""
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError(
                f"train_ratio ({self.train_ratio}) + val_ratio ({self.val_ratio}) "
                "must be less than 1.0 to leave room for test set"
            )
        return self


_LABEL_AWARE_DESCRIPTION = (
    "Predict ln s with an error decoder, parallel to the label decoder, from "
    "the head's input and the reported label errors, or a learned missing "
    "vector, so that s includes the label noise (see LabelAwareOutput). False "
    "builds the output layers of checkpoints trained without it and exists "
    "only to load them; training rejects it."
)


class ModelConfig(_ConfigModel):
    """Configuration for the neural network architecture.

    The default architecture follows the standard DOROTHY MLP:
    input_features -> 5000 -> 2000 -> 1000 -> 500 -> 200 -> 100 -> output_features
    (15300 and 22 by default).
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
    label_aware_uncertainty: bool = Field(
        default=True,
        description=_LABEL_AWARE_DESCRIPTION,
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


class MultiHeadModelConfig(_ConfigModel):
    """Configuration for multi-head MLP architecture.

    Used when training on multiple surveys with different wavelength grids.
    Each survey has its own encoder, feeding into a shared trunk.
    Optionally supports multiple output heads for different label sources.

    Examples:
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
        description="Hidden layer sizes of each output head's label and error "
        "decoders",
    )
    combination_mode: CombinationMode = Field(
        default=CombinationMode.CONCAT,
        description="How to combine multi-survey embeddings",
    )
    label_sources: list[str] | None = Field(
        default=None,
        description="Label sources of the output heads, one head each (e.g., "
        "['apogee', 'galah']). None or an empty list gives one head, named after "
        "the first of data.label_sources.",
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
    head_dropout: dict[str, float] = Field(
        default_factory=dict,
        description='Output head dropout per label source, e.g. {"galah": 0.2}; '
        "sources not listed use dropout",
    )
    label_aware_uncertainty: bool = Field(
        default=True,
        description=_LABEL_AWARE_DESCRIPTION,
    )
    residual_stream: bool = Field(
        default=True,
        description="Build the trunk and the output heads' hidden layers as a "
        "residual stream of width latent_dim, with one residual block per "
        "trunk_hidden and output_hidden size; False builds plain blocks. "
        "Checkpoints saved before this option load with False.",
    )

    @field_validator("head_dropout")
    @classmethod
    def validate_head_dropout(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure per-head dropout probabilities are in [0, 1)."""
        for source, p in v.items():
            if not 0.0 <= p < 1.0:
                raise ValueError(
                    f"head_dropout for '{source}' must be in [0, 1), got {p}"
                )
        return v

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


class SchedulerConfig(_ConfigModel):
    """Configuration for learning rate schedulers."""

    type: SchedulerType = Field(
        default=SchedulerType.ONE_CYCLE, description="Scheduler type"
    )
    base_lr: float = Field(
        default=1e-6, gt=0, description="Minimum learning rate (cyclic)"
    )
    max_lr: float = Field(
        default=1e-3, gt=0, description="Maximum learning rate (cyclic, one_cycle)"
    )
    step_size_up: int = Field(
        default=250, ge=1, description="Steps to rise from base_lr to max_lr (cyclic)"
    )
    step_size_down: int = Field(
        default=650, ge=1, description="Steps to fall from max_lr to base_lr (cyclic)"
    )
    gamma: float = Field(
        default=0.9995,
        gt=0,
        le=1.0,
        description="Multiplicative factor of the exp_range mode (cyclic)",
    )
    patience: int = Field(
        default=10,
        ge=1,
        description="Epochs to wait before reducing the LR (reduce_on_plateau)",
    )
    factor: float = Field(
        default=0.5,
        gt=0,
        lt=1.0,
        description="Factor to reduce the LR by (reduce_on_plateau)",
    )
    # OneCycleLR parameters
    pct_start: float = Field(
        default=0.3,
        gt=0,
        lt=1.0,
        description="Fraction of the cycle spent increasing the LR (one_cycle)",
    )
    div_factor: float = Field(
        default=25.0, ge=1.0, description="Initial LR = max_lr / div_factor (one_cycle)"
    )
    final_div_factor: float = Field(
        default=1e4,
        ge=1.0,
        description="Final LR = initial LR / final_div_factor (one_cycle)",
    )
    anneal_strategy: Literal["cos", "linear"] = Field(
        default="cos", description="Annealing strategy (one_cycle)"
    )

    @model_validator(mode="after")
    def validate_lr_range(self) -> SchedulerConfig:
        """Ensure base_lr <= max_lr for the cyclic schedule, the one that uses both."""
        if self.type == SchedulerType.CYCLIC and self.base_lr > self.max_lr:
            raise ValueError(
                f"base_lr ({self.base_lr}) must be <= max_lr ({self.max_lr})"
            )
        return self


class OptimizerConfig(_ConfigModel):
    """Configuration for optimizers."""

    type: OptimizerType = Field(
        default=OptimizerType.ADAMW, description="Optimizer type"
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0,
        description=(
            "Weight decay of the weight matrices. Biases and normalization "
            "weights are not decayed."
        ),
    )
    betas: tuple[float, float] = Field(
        default=(0.9, 0.999),
        description="Coefficients for running averages (Adam/AdamW). The "
        "one_cycle scheduler replaces the first: it cycles it between 0.85 and "
        "0.95.",
    )
    momentum: float = Field(
        default=0.9,
        ge=0,
        le=1.0,
        description="Momentum factor (SGD only). The one_cycle scheduler "
        "replaces it: it cycles it between 0.85 and 0.95.",
    )
    nesterov: bool = Field(
        default=False, description="Use Nesterov momentum (SGD only)"
    )


class TrainingConfig(_ConfigModel):
    """Configuration for the training process."""

    epochs: int = Field(default=300, ge=1, le=10000, description="Number of epochs")
    batch_size: int = Field(default=1024, ge=1, le=65536, description="Batch size")
    learning_rate: float = Field(
        default=1e-3,
        gt=0,
        le=1.0,
        description="Initial learning rate of the none, reduce_on_plateau and "
        "cosine_annealing schedulers. one_cycle and cyclic set the learning "
        "rate from their own fields.",
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
        default=0,
        ge=0,
        description="Save a checkpoint every N epochs (0 = only the final one)",
    )
    early_stopping_patience: int = Field(
        default=0,
        ge=0,
        description="Epochs without a lower validation loss before training "
        "stops, counted after the β-NLL warm-up; 0 turns early stopping off",
    )
    scatter_floor: float = Field(
        default=0.01,
        gt=0,
        le=1.0,
        description="Minimum scatter floor for heteroscedastic loss",
    )
    nll_warmup_fraction: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of the optimizer steps of the run over which β of "
        "the β-NLL loss falls linearly from 1 (mean-squared-error gradients for "
        "the means) to 0 (the plain negative log-likelihood); 0 turns this "
        "warm-up off. Applies to the heteroscedastic loss only.",
    )
    label_error_dropout: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Fraction of training stars whose label errors are replaced "
        "by the learned missing vector, drawn per batch and output head. "
        "Validation, evaluation and prediction always use the missing vector, "
        "so the fraction must be above 0.",
    )
    calibrate_uncertainties: bool = Field(
        default=True,
        description="After training, fit one factor per output head and "
        "parameter that scales the predicted scatter so that the validation "
        "z-scores of the best epoch have unit robust spread, and save the "
        "factors in calibration.json. loss: mse has no scatter, so it saves no "
        "factors.",
    )
    label_source_weighting: Literal["per_label", "per_source"] = Field(
        default="per_label",
        description="How the losses of the label sources (output heads) combine, "
        "for training and for selecting the best epoch. per_label gives every "
        "valid label equal weight; per_source averages each head's mean loss, "
        "so a smaller label source gets as much total weight as a larger one.",
    )
    label_source_weights: dict[str, float] = Field(
        default_factory=dict,
        description='Relative loss weight per label source, e.g. {"galah": 0.5}; '
        "sources not listed have weight 1.0",
    )
    amp: bool = Field(
        default=False,
        description="Run the training forward pass in amp_dtype on CUDA. CPU "
        "and MPS train in float32.",
    )
    amp_dtype: Literal["bfloat16", "float16"] = Field(
        default="bfloat16",
        description="Precision of the forward pass with amp. bfloat16 needs a "
        "GPU with compute capability 8.0 or higher; float16 uses a gradient "
        "scaler.",
    )
    ema_decay: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description="Decay per optimizer step of the exponential moving average "
        "of the weights, which validation, the best epoch and the saved weights "
        "then use (null: no average)",
    )
    ema_cpu_every: int | None = Field(
        default=None,
        ge=1,
        description="Keep the weight average in CPU memory and update it every "
        "this many optimizer steps (null: on the training device, every step)",
    )

    @field_validator("label_source_weights")
    @classmethod
    def validate_label_source_weights(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure label source weights are non-negative."""
        for source, w in v.items():
            if w < 0:
                raise ValueError(f"label_source_weights['{source}'] must be >= 0")
        return v

    @model_validator(mode="after")
    def validate_ema(self) -> TrainingConfig:
        """Reject a weight-average option without a weight average."""
        if self.ema_cpu_every is not None and self.ema_decay is None:
            raise ValueError("ema_cpu_every requires ema_decay to be set")
        return self


class LabelMaskingConfig(_ConfigModel):
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
        description="Minimum probability of keeping each label within kept labelsets "
        "(sampled per batch)",
    )
    p_label_max: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum probability of keeping each label within kept labelsets "
        "(sampled per batch)",
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


class InputMaskingConfig(_ConfigModel):
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


def _config_section_model(annotation: Any) -> type[BaseModel] | None:
    """The config model class in a field annotation (e.g. ``X | None``)."""
    candidates = typing.get_args(annotation) or (annotation,)
    for candidate in candidates:
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def _drop_unknown_keys(
    model: type[BaseModel], values: dict, prefix: str, dropped: list[str]
) -> dict:
    """Copy of ``values`` without keys ``model`` (or its sections) don't define.

    Used for checkpoint configs, which may contain keys from older versions.
    """
    allowed = set(model.model_fields)
    if model is DataConfig:
        allowed |= {"survey", "label_source"}  # accepted aliases
    out = {}
    for key, value in values.items():
        if key not in allowed:
            dropped.append(f"{prefix}{key}")
            continue
        field = model.model_fields.get(key)
        section = _config_section_model(field.annotation) if field else None
        if section is not None and isinstance(value, dict):
            value = _drop_unknown_keys(section, value, f"{prefix}{key}.", dropped)
        out[key] = value
    return out


class ExperimentConfig(_ConfigModel):
    """Top-level configuration for a DOROTHY experiment.

    This is the main configuration class that combines all sub-configurations
    and provides the complete specification for training a model.

    For single-survey training, use the ``model`` config (MLP). For
    multi-survey training, use the ``multi_head_model`` config (MultiHeadMLP).
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
    output_dir: Path = Field(
        default=Path("./outputs"),
        description="Directory for outputs (checkpoints, logs)",
    )
    seed: int = Field(default=42, ge=0, description="Random seed")
    device: Literal["cuda", "cpu", "mps", "auto"] = Field(
        default="auto", description="Compute device (mps for Apple Silicon)"
    )

    @property
    def is_multi_head(self) -> bool:
        """Whether using multi-head architecture."""
        return self.multi_head_model is not None

    @property
    def parameter_names(self) -> list[str]:
        """Names of the predicted parameters, in model output order.

        Uses ``data.parameter_names`` when set (populated from the catalogue),
        otherwise the first n ``STELLAR_PARAMETERS``, followed by ``param_<i>``
        for each output beyond that list.
        """
        if self.data.parameter_names is not None:
            return list(self.data.parameter_names)
        if self.multi_head_model is not None:
            n_params = self.multi_head_model.n_parameters
        else:
            n_params = self.model.n_parameters
        names = list(STELLAR_PARAMETERS[:n_params])
        # Generic names for any outputs beyond the standard list
        names += [f"param_{i}" for i in range(len(names), n_params)]
        return names

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

        # Per-label-source settings must name configured label sources
        sources = set(self.data.label_sources)
        unknown = set(self.training.label_source_weights) - sources
        if unknown:
            raise ValueError(
                f"training.label_source_weights has unknown label sources "
                f"{sorted(unknown)}; configured: {self.data.label_sources}"
            )
        if self.multi_head_model is not None:
            if not self.multi_head_model.label_sources:
                self.multi_head_model = self.multi_head_model.model_copy(
                    update={"label_sources": self.data.label_sources[:1]}
                )
            heads = set(self.multi_head_model.label_sources)
            unknown = set(self.multi_head_model.head_dropout) - heads
            if unknown:
                raise ValueError(
                    f"multi_head_model.head_dropout has unknown label sources "
                    f"{sorted(unknown)}; heads: {sorted(heads)}"
                )

        # Keep the model's output size in sync with the catalogue's parameters
        if self.data.parameter_names is not None:
            n_params = len(self.data.parameter_names)
            if self.multi_head_model is not None:
                mh = self.multi_head_model
                if (
                    "n_parameters" in mh.model_fields_set
                    and mh.n_parameters != n_params
                ):
                    raise ValueError(
                        f"multi_head_model.n_parameters={mh.n_parameters} does not "
                        f"match {n_params} data.parameter_names"
                    )
                mh.n_parameters = n_params
            else:
                model = self.model
                if (
                    "output_features" in model.model_fields_set
                    and model.output_features != 2 * n_params
                ):
                    raise ValueError(
                        f"model.output_features={model.output_features} does not "
                        f"match 2 x {n_params} data.parameter_names"
                    )
                model.output_features = 2 * n_params

        return self

    def get_output_path(self) -> Path:
        """Get the output directory of this experiment.

        Returns:
            ``output_dir / name``.
        """
        return self.output_dir / self.name

    def get_checkpoint_path(self, epoch: int | None = None) -> Path:
        """Get the directory of a checkpoint.

        Args:
            epoch: Epoch number, or None for the final checkpoint.

        Returns:
            ``<output path>/epoch_<epoch>``, or ``<output path>/<name>_final``
            when epoch is None.
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
    def load_checkpoint_config(cls, path: str | Path) -> ExperimentConfig:
        """Load the ``config.yaml`` saved alongside a trained checkpoint.

        Unlike :meth:`load_yaml`, this accepts the Python-specific tags (e.g.
        tuples) written by :meth:`save_yaml`, and fills in defaults that keep
        older checkpoints reproducible. A config without
        ``data.normalize_spectra`` was trained on un-normalized catalogue
        spectra, so the flag defaults to False, except for configs from the
        ``dorothy_update`` branch, recognizable by a
        ``multi_head_model.head_dropout`` entry or 20 predicted parameters,
        whose sparse and single-survey loaders already normalized spectra. A
        warning is issued either way. A model section without
        ``label_aware_uncertainty``, a multi-head model section without
        ``residual_stream`` and a data section without ``precision_encoding``
        get False for that flag. Settings this version no longer has are
        ignored with a warning.

        Args:
            path: Path to the checkpoint's ``config.yaml``.

        Returns:
            ExperimentConfig instance.

        Raises:
            pydantic.ValidationError: If the settings are invalid.
        """
        import yaml

        with open(path) as f:
            data = yaml.full_load(f)

        # Keys from older versions (since removed) are ignored, not errors
        dropped: list[str] = []
        data = _drop_unknown_keys(cls, data, "", dropped)
        if dropped:
            warnings.warn(
                f"Ignoring settings in {path} that this version doesn't use: "
                f"{', '.join(dropped)}",
                UserWarning,
                stacklevel=2,
            )

        data_section = data.setdefault("data", {})
        if "normalize_spectra" not in data_section:
            # dorothy_update configs: head_dropout (multi-head) or 20 outputs
            # (its only parameter set; main supported only 11 at the time)
            mh_section = data.get("multi_head_model") or {}
            model_section = data.get("model") or {}
            n_params = mh_section.get(
                "n_parameters", model_section.get("output_features", 22) // 2
            )
            from_update_branch = "head_dropout" in mh_section or n_params == 20
            normalized = from_update_branch and not data_section.get(
                "use_dense_loading", False
            )
            data_section["normalize_spectra"] = normalized
            warnings.warn(
                f"Checkpoint config {path} predates data.normalize_spectra; "
                f"assuming normalize_spectra={normalized}. Add it to the "
                "config's data section if this is wrong.",
                UserWarning,
                stacklevel=2,
            )
        data_section.setdefault("precision_encoding", False)
        for key in ("model", "multi_head_model"):
            if data.get(key) is not None:
                data[key].setdefault("label_aware_uncertainty", False)
        if data.get("multi_head_model") is not None:
            data["multi_head_model"].setdefault("residual_stream", False)
        # Dropout is irrelevant once trained: drop head_dropout entries for
        # heads the model doesn't have rather than refusing to load
        mh_section = data.get("multi_head_model") or {}
        if mh_section.get("head_dropout"):
            heads = set(
                mh_section.get("label_sources")
                or data_section.get("label_sources", ["apogee"])[:1]
            )
            unknown = set(mh_section["head_dropout"]) - heads
            if unknown:
                warnings.warn(
                    f"Ignoring head_dropout for unknown heads {sorted(unknown)}",
                    UserWarning,
                    stacklevel=2,
                )
                for key in unknown:
                    del mh_section["head_dropout"][key]

        if "catalogue_path" in data_section:
            data_section["catalogue_path"] = Path(data_section["catalogue_path"])
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])

        return cls(**data)

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
