"""
Multi-head MLP architecture for multi-survey stellar parameter inference.

This module implements a multi-head architecture where each survey has its own
encoder network, feeding into a shared trunk that produces stellar parameter
predictions. This allows handling surveys with different wavelength grids
while learning shared representations.

Architecture:
    Survey Encoders (one per survey):
        Input: (batch, 3, n_wavelengths_i) -> Flatten -> Hidden layers -> (batch, latent_dim)

    Shared Trunk:
        Input: (batch, latent_dim) or (batch, n_surveys * latent_dim) for combined
        Hidden layers -> (batch, trunk_output_dim)

    Output Head (a LabelAwareOutput per label source):
        Label decoder: the trunk output -> Hidden layers -> mu
        Error decoder: the trunk output and the log label errors -> Hidden
        layers of the same sizes -> ln s
        Output: (batch, 2, n_params) = [mu, ln s]

    With residual_stream (the default), the trunk and the decoders' hidden
    layers are residual blocks of width latent_dim (see SharedTrunk and
    LabelAwareOutput).

For multi-survey training:
    - Each sample may have data from one or more surveys
    - Encoders only process surveys with valid data (indicated by has_data mask)
    - Encoded representations are combined (averaged or concatenated) before trunk
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from dorothy.models.output import output_modules
from dorothy.models.utils import (
    get_activation,
    get_normalization,
    hidden_blocks,
    residual_blocks,
)

if TYPE_CHECKING:
    from dorothy.config.schema import MultiHeadModelConfig

# Layers whose output for a row depends only on that row, in any mode (see
# MultiHeadMLP._rows_are_independent)
_ROWWISE_MODULES: tuple[type[nn.Module], ...] = (
    nn.Sequential,
    nn.Flatten,
    nn.Linear,
    nn.LayerNorm,
    nn.GELU,
    nn.ReLU,
    nn.SiLU,
    nn.Identity,
)


class SurveyEncoder(nn.Module):
    """Encoder network for a single survey's spectral data.

    Takes 3-channel spectral input (flux, rho, mask) and produces a
    fixed-size latent representation.

    Attributes:
        survey_name: Name of the survey this encoder handles.
        n_wavelengths: Number of wavelength bins for this survey.
        latent_dim: Output dimension of the encoder.
    """

    def __init__(
        self,
        survey_name: str,
        n_wavelengths: int,
        latent_dim: int = 256,
        hidden_layers: list[int] | None = None,
        normalization: str = "layernorm",
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        """Initialize the survey encoder.

        Args:
            survey_name: Name identifier for this survey.
            n_wavelengths: Number of wavelength bins in input spectra.
            latent_dim: Output dimension of the encoder.
            hidden_layers: Hidden layer sizes. Default: [1024, 512].
            normalization: Normalization type ("layernorm" or "batchnorm").
            activation: Activation function ("gelu", "relu", or "silu").
            dropout: Dropout probability of the hidden layers; the layer to the
                latent has none.

        Raises:
            ValueError: If the normalization or activation is not recognized.
        """
        super().__init__()

        self.survey_name = survey_name
        self.n_wavelengths = n_wavelengths
        self.latent_dim = latent_dim

        if hidden_layers is None:
            hidden_layers = [1024, 512]

        # The projection to the latent has no dropout
        sizes = [3 * n_wavelengths, *hidden_layers]
        self.encoder = nn.Sequential(
            nn.Flatten(),
            *hidden_blocks(sizes[0], hidden_layers, normalization, activation, dropout),
            *hidden_blocks(sizes[-1], [latent_dim], normalization, activation, 0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode spectral data to latent representation.

        Args:
            x: Input tensor of shape (batch, 3, n_wavelengths).

        Returns:
            Latent tensor of shape (batch, latent_dim).
        """
        return self.encoder(x)


class SharedTrunk(nn.Module):
    """Shared trunk network that processes combined survey embeddings.

    Takes latent representations from survey encoders and produces
    features for the output head. The plain trunk is one hidden block
    (Linear, normalization, activation) per hidden layer, then a block to
    output_dim. The residual trunk is a stream of width output_dim: a linear
    map from input_dim when the sizes differ, one pre-norm ResidualBlock per
    hidden layer, whose hidden size it is, then a normalization.

    Attributes:
        input_dim: Input dimension (latent_dim for single survey or
            n_surveys * latent_dim for concatenation).
        output_dim: Output dimension for the output head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_layers: list[int] | None = None,
        normalization: str = "layernorm",
        activation: str = "gelu",
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        """Initialize the shared trunk.

        Args:
            input_dim: Input dimension from encoder(s).
            output_dim: Output dimension for output head, and the stream
                width of a residual trunk.
            hidden_layers: Hidden layer sizes. Default: [512, 256].
            normalization: Normalization type.
            activation: Activation function.
            dropout: Dropout probability.
            residual: Build the residual trunk.

        Raises:
            ValueError: If the normalization or activation is not recognized.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if hidden_layers is None:
            hidden_layers = [512, 256]

        if residual:
            projection = (
                [nn.Linear(input_dim, output_dim)] if input_dim != output_dim else []
            )
            self.trunk = nn.Sequential(
                *projection,
                *residual_blocks(
                    output_dim, hidden_layers, normalization, activation, dropout
                ),
            )
        else:
            width = hidden_layers[-1] if hidden_layers else input_dim
            self.trunk = nn.Sequential(
                *hidden_blocks(
                    input_dim, hidden_layers, normalization, activation, dropout
                ),
                nn.Linear(width, output_dim),
                get_normalization(normalization)(output_dim),
                get_activation(activation)(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process combined embeddings.

        Args:
            x: Combined latent tensor of shape (batch, input_dim).

        Returns:
            Features tensor of shape (batch, output_dim).
        """
        return self.trunk(x)


class OutputHead(nn.Module):
    """Output head that produces stellar parameter predictions.

    Takes features from the shared trunk and produces predictions in the
    standard (batch, 2, n_params) format. By default the head is a
    LabelAwareOutput: its label decoder maps the features through hidden_layers to mu, and its
    error decoder maps the features and the label errors through layers of
    the same sizes to ln s.

    Attributes:
        n_parameters: Number of stellar parameters to predict.
    """

    def __init__(
        self,
        input_dim: int,
        n_parameters: int = 11,
        hidden_layers: list[int] | None = None,
        normalization: str = "layernorm",
        activation: str = "gelu",
        dropout: float = 0.0,
        label_aware_uncertainty: bool = True,
        residual: bool = False,
    ) -> None:
        """Initialize the output head.

        Args:
            input_dim: Input dimension from shared trunk.
            n_parameters: Number of stellar parameters to predict.
            hidden_layers: Hidden layer sizes of each decoder. Default: [64].
            normalization: Normalization type.
            activation: Activation function.
            dropout: Dropout probability.
            label_aware_uncertainty: Make the head a LabelAwareOutput (True)
                or end it in an AdditiveOutput (False).
            residual: Make the hidden layers residual blocks, for a residual
                stream input (see LabelAwareOutput).

        Raises:
            ValueError: If label_aware_uncertainty is True and hidden_layers
                is empty.
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_parameters = n_parameters
        self.output_features = 2 * n_parameters

        if hidden_layers is None:
            hidden_layers = [64]

        self.head = nn.Sequential(
            *output_modules(
                input_dim,
                n_parameters,
                hidden_layers,
                normalization,
                activation,
                dropout,
                label_aware_uncertainty,
                residual,
            )
        )

    def forward(
        self, x: torch.Tensor, label_errors: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Produce parameter predictions.

        Args:
            x: Features tensor of shape (batch, input_dim).
            label_errors: Reported label errors of shape (batch, n_parameters),
                in normalized units, for the log-scatter (see
                LabelAwareOutput). None uses the learned missing vector for
                every entry.

        Returns:
            Predictions of shape (batch, 2, n_parameters) where:
                - output[:, 0, :] contains predicted means
                - output[:, 1, :] contains predicted log-scatter

        Raises:
            ValueError: If label_errors is given to a head built with
                label_aware_uncertainty=False, or has the wrong shape.
        """
        return self.head[-1](self.head[:-1](x), label_errors)


def _rename_default_head(
    module: MultiHeadMLP, state_dict: dict, prefix: str, *args
) -> None:
    # Single-head models saved before heads were named after their label
    # source store the alias output_heads.<source> as output_heads.default
    old = f"{prefix}output_heads.default."
    new = f"{prefix}output_heads.{module.label_sources[0]}."
    for key in [k for k in state_dict if k.startswith(old)]:
        state_dict[new + key[len(old) :]] = state_dict.pop(key)


class MultiHeadMLP(nn.Module):
    """Multi-head MLP for multi-survey stellar parameter prediction.

    This architecture has separate encoder networks for each survey,
    a shared trunk that combines the encodings, and one output head per
    label source.

    The model can handle:
    - Single survey input (standard case)
    - Multiple surveys with per-sample availability masks

    Attributes:
        survey_names: List of survey names in order.
        n_parameters: Number of stellar parameters being predicted.
        combination_mode: How to combine multi-survey embeddings ("mean" or "concat").
        encoders: ModuleDict of survey encoders.
        trunk: Shared trunk network.
        latent_dim: Output dimension of each survey encoder.
        label_sources: Label source of each output head; ["default"] when
            none were given.
        output_heads: ModuleDict mapping each label source to its output head.
        output_head: The output head of the first label source.
        label_aware_uncertainty: Whether the heads are LabelAwareOutputs
            (True) or end in an AdditiveOutput.
        residual_stream: Whether the trunk and the heads' decoders form a
            residual stream of width latent_dim (True) or are plain blocks.
        encode_present_rows_only: With multi-survey input, run each encoder
            only on the rows that have that survey's data (default True).
            The rows without data are discarded either way, so this changes
            results only by float rounding (a different GEMM size); False
            encodes every row.

    Examples:
        >>> model = MultiHeadMLP(
        ...     survey_configs={"boss": 4506, "lamost": 3700},
        ...     n_parameters=11,
        ... )
        >>> # Single survey forward
        >>> x_boss = torch.randn(32, 3, 4506)
        >>> output = model.forward_single(x_boss, "boss")
        >>>
        >>> # Multi-survey forward
        >>> inputs = {"boss": x_boss, "lamost": torch.randn(32, 3, 3700)}
        >>> has_data = {"boss": torch.ones(32, dtype=torch.bool),
        ...             "lamost": torch.ones(32, dtype=torch.bool)}
        >>> output = model.forward_multi(inputs, has_data)
    """

    encode_present_rows_only: bool = True

    def __init__(
        self,
        survey_configs: dict[str, int],
        n_parameters: int = 11,
        latent_dim: int = 256,
        encoder_hidden: list[int] | None = None,
        trunk_hidden: list[int] | None = None,
        output_hidden: list[int] | None = None,
        combination_mode: str = "concat",
        normalization: str = "layernorm",
        activation: str = "gelu",
        dropout: float = 0.0,
        label_sources: list[str] | None = None,
        head_dropout: dict[str, float] | None = None,
        label_aware_uncertainty: bool = True,
        residual_stream: bool = True,
    ) -> None:
        """Initialize the multi-head MLP.

        Args:
            survey_configs: Dict mapping survey names to their wavelength counts.
                Example: {"boss": 4506, "lamost_lrs": 3700}
            n_parameters: Number of stellar parameters to predict.
            latent_dim: Output dimension of each survey encoder.
            encoder_hidden: Hidden layers for encoders. Default: [1024, 512].
            trunk_hidden: Hidden layers for shared trunk. Default: [512, 256].
            output_hidden: Hidden layers of each output head's decoders.
                Default: [64].
            combination_mode: How to combine multi-survey embeddings. Default: "concat".
                - "concat": Concatenate embeddings (trunk input = n_surveys * latent_dim).
                    Missing surveys are zero-padded.
                - "mean": Average embeddings from available surveys (excludes missing)
            normalization: Normalization type for all components.
            activation: Activation function for all components.
            dropout: Dropout probability for all components.
            label_sources: Label sources of the output heads, one head each
                (e.g., ["apogee", "galah"]). Default: None, one head named
                "default".
            head_dropout: Optional per-label-source dropout for the output heads,
                e.g. {"galah": 0.2}. Sources not listed use ``dropout``.
            label_aware_uncertainty: Make every head a LabelAwareOutput (True)
                or end each in an AdditiveOutput (False).
            residual_stream: Build the trunk and the heads' hidden layers as
                a residual stream of width latent_dim (default; see
                SharedTrunk and LabelAwareOutput); False builds plain blocks.

        Raises:
            ValueError: If survey_configs is empty.
            ValueError: If combination_mode is invalid.
        """
        super().__init__()

        if not survey_configs:
            raise ValueError("survey_configs cannot be empty")
        if combination_mode not in ("mean", "concat"):
            raise ValueError(
                f"combination_mode must be 'mean' or 'concat', got '{combination_mode}'"
            )

        self.survey_names = list(survey_configs.keys())
        self.n_surveys = len(self.survey_names)
        self.n_parameters = n_parameters
        self.latent_dim = latent_dim
        self.combination_mode = combination_mode
        self.label_aware_uncertainty = label_aware_uncertainty
        self.residual_stream = residual_stream

        # Label sources for multi-output heads
        self.label_sources = label_sources if label_sources else ["default"]
        self.n_label_sources = len(self.label_sources)
        self.is_multi_label = self.n_label_sources > 1

        # Create survey encoders
        self.encoders = nn.ModuleDict()
        for survey_name, n_wavelengths in survey_configs.items():
            self.encoders[survey_name] = SurveyEncoder(
                survey_name=survey_name,
                n_wavelengths=n_wavelengths,
                latent_dim=latent_dim,
                hidden_layers=encoder_hidden,
                normalization=normalization,
                activation=activation,
                dropout=dropout,
            )

        # Trunk input dimension depends on combination mode
        if combination_mode == "concat":
            trunk_input_dim = self.n_surveys * latent_dim
        else:  # mean
            trunk_input_dim = latent_dim

        # Create shared trunk
        self.trunk = SharedTrunk(
            input_dim=trunk_input_dim,
            output_dim=latent_dim if residual_stream else 128,
            hidden_layers=trunk_hidden,
            normalization=normalization,
            activation=activation,
            dropout=dropout,
            residual=residual_stream,
        )

        # Create output head(s) - one per label source
        head_dropout = head_dropout or {}
        self.head_dropout = dict(head_dropout)
        if self.is_multi_label:
            self.output_heads = nn.ModuleDict()
            for source in self.label_sources:
                self.output_heads[source] = OutputHead(
                    input_dim=self.trunk.output_dim,
                    n_parameters=n_parameters,
                    hidden_layers=output_hidden,
                    normalization=normalization,
                    activation=activation,
                    dropout=head_dropout.get(source, dropout),
                    label_aware_uncertainty=label_aware_uncertainty,
                    residual=residual_stream,
                )
            # Backward compatibility: also set output_head to first source
            self.output_head = self.output_heads[self.label_sources[0]]
        else:
            self.output_head = OutputHead(
                input_dim=self.trunk.output_dim,
                n_parameters=n_parameters,
                hidden_layers=output_hidden,
                normalization=normalization,
                activation=activation,
                dropout=head_dropout.get(self.label_sources[0], dropout),
                label_aware_uncertainty=label_aware_uncertainty,
                residual=residual_stream,
            )
            self.output_heads = nn.ModuleDict({self.label_sources[0]: self.output_head})
            self.register_load_state_dict_pre_hook(_rename_default_head)

    @classmethod
    def from_config(cls, config: MultiHeadModelConfig) -> MultiHeadMLP:
        """Create a MultiHeadMLP from a MultiHeadModelConfig.

        Args:
            config: Multi-head model configuration.

        Returns:
            Configured MultiHeadMLP instance.
        """
        return cls(
            survey_configs=dict(config.survey_wavelengths),
            n_parameters=config.n_parameters,
            latent_dim=config.latent_dim,
            encoder_hidden=list(config.encoder_hidden),
            trunk_hidden=list(config.trunk_hidden),
            output_hidden=list(config.output_hidden),
            combination_mode=config.combination_mode.value,
            normalization=config.normalization.value,
            activation=config.activation.value,
            dropout=config.dropout,
            label_sources=config.label_sources,
            head_dropout=config.head_dropout,
            label_aware_uncertainty=config.label_aware_uncertainty,
            residual_stream=config.residual_stream,
        )

    def forward_single(
        self,
        x: torch.Tensor,
        survey: str,
    ) -> torch.Tensor:
        """Forward pass for single-survey input.

        This is the standard case where all samples come from one survey.

        Args:
            x: Input tensor of shape (batch, 3, n_wavelengths).
            survey: Name of the survey.

        Returns:
            Predictions of shape (batch, 2, n_parameters).

        Raises:
            KeyError: If survey is not in the model's encoders.
        """
        return self.output_head(self._forward_single_to_trunk(x, survey))

    def forward_multi(
        self,
        inputs: dict[str, torch.Tensor],
        has_data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for multi-survey input.

        Handles samples that may have data from different subsets of surveys.

        Args:
            inputs: Dict mapping survey names to input tensors.
                Each tensor has shape (batch, 3, n_wavelengths_survey).
            has_data: Dict mapping survey names to boolean masks.
                Each mask has shape (batch,) indicating which samples
                have valid data from that survey.

        Returns:
            Predictions of shape (batch, 2, n_parameters) from the first
            head.

        Raises:
            ValueError: If none of the keys of inputs is a survey of the model.

        Note:
            A sample without data from any survey enters the trunk as a
            zero vector (concat) or a zero mean (mean), so its output carries
            no information.
        """
        return self.output_head(self._forward_to_trunk(inputs, has_data))

    def forward(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        survey: str | None = None,
        has_data: dict[str, torch.Tensor] | None = None,
        label_errors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Unified forward pass.

        Dispatches to forward_single or forward_multi based on input type.

        Args:
            x: Either a single tensor (with survey name) or dict of tensors.
            survey: Survey name if x is a single tensor.
            has_data: Data availability masks if x is a dict.
            label_errors: Reported label errors (batch, n_parameters) for the
                first head's log-scatter (see LabelAwareOutput). None uses
                the learned missing vector.

        Returns:
            Predictions of shape (batch, 2, n_parameters).

        Raises:
            KeyError: If survey is not in the model's encoders.
            ValueError: If x is a single tensor without survey on a
                multi-survey model, or a dict without any of the model's
                surveys; or if label_errors is given to a model built with
                label_aware_uncertainty=False, or has the wrong shape.
        """
        return self.output_head(self._trunk_features(x, survey, has_data), label_errors)

    def forward_for_label_source(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        label_source: str,
        survey: str | None = None,
        has_data: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass using a specific label source's output head.

        Use this for multi-label training where different samples may have
        labels from different sources.

        Args:
            x: Input tensor or dict of tensors (same as forward()).
            label_source: Which label source's output head to use.
            survey: Survey name if x is single tensor.
            has_data: Data availability masks if x is dict.

        Returns:
            Predictions of shape (batch, 2, n_parameters).

        Raises:
            KeyError: If label_source is not in the model's output heads.
        """
        if label_source not in self.output_heads:
            raise KeyError(
                f"Unknown label source '{label_source}'. "
                f"Available: {self.label_sources}"
            )

        return self.output_heads[label_source](
            self._trunk_features(x, survey, has_data)
        )

    def forward_all_label_sources(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        survey: str | None = None,
        has_data: dict[str, torch.Tensor] | None = None,
        label_errors: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through all label source output heads.

        Useful for inference when you want predictions from all label sources.

        Args:
            x: Input tensor or dict of tensors.
            survey: Survey name if x is single tensor.
            has_data: Data availability masks if x is dict.
            label_errors: Label source name -> reported label errors
                (batch, n_parameters) for that head's log-scatter (see
                LabelAwareOutput). Heads without an entry, and all heads when
                None, use their learned missing vector.

        Returns:
            Dict mapping label source names to predictions.
            Each prediction has shape (batch, 2, n_parameters).

        Raises:
            KeyError: If label_errors names a label source the model has no
                head for, or survey is not in the model's encoders.
            ValueError: If x is a single tensor without survey on a
                multi-survey model, or a dict without any of the model's
                surveys; or if label_errors is given to a model built with
                label_aware_uncertainty=False, or has the wrong shape.
        """
        label_errors = label_errors or {}
        unknown = set(label_errors) - set(self.output_heads)
        if unknown:
            raise KeyError(
                f"label_errors has unknown label sources {sorted(unknown)}. "
                f"Available: {list(self.output_heads)}"
            )
        features = self._trunk_features(x, survey, has_data)
        return {
            source: head(features, label_errors.get(source))
            for source, head in self.output_heads.items()
        }

    def _trunk_features(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        survey: str | None,
        has_data: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Shared trunk features for single-tensor or dict (multi-survey) input."""
        if isinstance(x, dict):
            if has_data is None:
                # Assume all samples have all provided surveys
                batch_size = next(iter(x.values())).shape[0]
                device = next(iter(x.values())).device
                has_data = {
                    name: torch.ones(batch_size, dtype=torch.bool, device=device)
                    for name in x
                }
            return self._forward_to_trunk(x, has_data)
        if survey is None:
            if len(self.survey_names) != 1:
                raise ValueError(
                    "Must specify survey for single-tensor input with multiple surveys"
                )
            survey = self.survey_names[0]
        return self._forward_single_to_trunk(x, survey)

    def _forward_single_to_trunk(self, x: torch.Tensor, survey: str) -> torch.Tensor:
        """Forward through encoder and trunk for single survey."""
        if survey not in self.encoders:
            raise KeyError(f"Unknown survey '{survey}'. Available: {self.survey_names}")
        latent = self.encoders[survey](x)

        if self.combination_mode == "concat":
            batch_size = latent.shape[0]
            survey_idx = self.survey_names.index(survey)

            full_latent = torch.zeros(
                batch_size,
                self.n_surveys * self.latent_dim,
                device=latent.device,
                dtype=latent.dtype,
            )
            start_idx = survey_idx * self.latent_dim
            end_idx = start_idx + self.latent_dim
            full_latent[:, start_idx:end_idx] = latent
            latent = full_latent

        return self.trunk(latent)

    def _forward_to_trunk(
        self,
        inputs: dict[str, torch.Tensor],
        has_data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward through encoders and trunk for multi-survey."""
        if not any(name in self.encoders for name in inputs):
            raise ValueError(
                f"None of the input surveys {list(inputs)} match the model's "
                f"surveys {self.survey_names}"
            )
        return self.trunk(self._combine_encodings(inputs, has_data))

    @staticmethod
    def _rows_are_independent(encoder: nn.Module) -> bool:
        """Whether the encoder maps each row independently of the others.

        True when every layer is row-wise (Linear, LayerNorm, activations),
        or is a BatchNorm using running statistics or a Dropout in eval mode.
        In training mode BatchNorm uses batch statistics and Dropout draws
        random numbers per batch element, so the whole batch must be encoded
        as before.
        """
        for module in encoder.modules():
            if isinstance(module, (SurveyEncoder, *_ROWWISE_MODULES)):
                continue
            if isinstance(module, nn.Dropout) and not module.training:
                continue
            if (
                isinstance(module, nn.BatchNorm1d)
                and not module.training
                and module.track_running_stats
            ):
                continue
            return False
        return True

    def _combine_encodings(
        self,
        inputs: dict[str, torch.Tensor],
        has_data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode each survey and combine the encodings (concat or mean).

        Rows without a survey's data contribute nothing (zeros in concat
        mode, excluded from the mean). When the encoder maps rows
        independently, only the rows that have data are encoded, which skips
        most of the encoder work for low-coverage surveys; the result is the
        same as encoding every row and discarding the rows without data.
        When every row is encoded, the rows without data enter the encoder
        as zeros.
        """
        first = next(iter(inputs.values()))
        batch_size, device, dtype = first.shape[0], first.device, first.dtype

        # Surveys present in the inputs, with their availability masks
        surveys = []
        for i, survey_name in enumerate(self.survey_names):
            if survey_name in inputs:
                mask = has_data.get(survey_name)
                if mask is None:
                    mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                surveys.append((i, survey_name, mask))
        # Number of rows with data per survey, in one host transfer
        n_rows = (
            torch.stack([(mask != 0).sum() for _, _, mask in surveys]).tolist()
            if surveys
            else []
        )

        # (survey index, latent, rows, mask): latent has every row when rows
        # is None, else only the given rows. With a mask (whole batch encoded
        # because rows are not independent) only the masked rows count.
        encoded: list[tuple[int, torch.Tensor, torch.Tensor | None, torch.Tensor]] = []
        for (i, survey_name, mask), n in zip(surveys, n_rows, strict=True):
            if n == 0:
                continue
            x = inputs[survey_name]
            encoder = self.encoders[survey_name]
            if (
                n == batch_size
                or not self.encode_present_rows_only
                or mask.dtype != torch.bool
                or not self._rows_are_independent(encoder)
            ):
                if n < batch_size:
                    # Zero the rows without data, as in the trainer's batches:
                    # a NaN there would reach BatchNorm's batch statistics and,
                    # through the discarded latents, the weight gradients
                    present = (mask != 0).view(-1, *[1] * (x.dim() - 1))
                    x = torch.where(present, x, 0.0)
                encoded.append((i, encoder(x), None, mask))
            else:
                # Indices of the rows with data, in order (a stable sort puts
                # them first), without another host sync
                rows = torch.argsort((~mask).to(torch.uint8), stable=True)[:n]
                encoded.append((i, encoder(x.index_select(0, rows)), rows, mask))

        if self.combination_mode == "concat":
            blocks = [
                torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)
                for _ in self.survey_names
            ]
            for i, latent, rows, mask in encoded:
                if rows is not None:
                    blocks[i] = blocks[i].index_copy(0, rows, latent.to(dtype))
                else:
                    # Missing rows stay zero
                    mask_expanded = mask.unsqueeze(1).expand(-1, self.latent_dim)
                    blocks[i] = torch.where(mask_expanded, latent, blocks[i]).to(dtype)
            return torch.cat(blocks, dim=1)

        # Mean combination: accumulate the sum and count of available surveys
        combined = torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        for _, latent, rows, mask in encoded:
            mask_float = mask.float().unsqueeze(1)
            if rows is not None:
                combined = combined.index_add(0, rows, latent.to(dtype))
            else:
                combined = combined + torch.where(mask_float > 0, latent, 0.0)
            count = count + mask_float

        # Average (avoid division by zero)
        count = count.clamp(min=1)
        return combined / count

    def get_embeddings(
        self,
        x: torch.Tensor,
        survey: str,
        level: str = "encoder",
    ) -> torch.Tensor:
        """Get latent embeddings for a survey's input.

        Useful for anomaly detection and analysis.

        Args:
            x: Input tensor of shape (batch, 3, n_wavelengths).
            survey: Name of the survey.
            level: "encoder" for the survey encoder's latent vector, or
                "trunk" for the shared trunk features (comparable across
                surveys, and the input to every output head).

        Returns:
            Embeddings of shape (batch, latent_dim) for "encoder" or
            (batch, trunk output dim) for "trunk": 128, or latent_dim with
            residual_stream.

        Raises:
            KeyError: If survey is not in the model's encoders.
            ValueError: If level is not "encoder" or "trunk".
        """
        if level == "trunk":
            return self._forward_single_to_trunk(x, survey)
        if level != "encoder":
            raise ValueError(f"level must be 'encoder' or 'trunk', got {level!r}")
        return self.encoders[survey](x)

    def count_parameters(self) -> int:
        """Count the trainable parameters.

        Returns:
            Number of weights and biases that require gradients.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_component(self) -> dict[str, int]:
        """Count the trainable parameters by component.

        Returns:
            Dict with the keys "encoder_<survey>", "trunk", "output_head_<source>"
            (one per head with several label sources, else "output_head") and
            "total", each the number of trainable parameters.
        """
        counts = {}
        for name, encoder in self.encoders.items():
            counts[f"encoder_{name}"] = sum(
                p.numel() for p in encoder.parameters() if p.requires_grad
            )
        counts["trunk"] = sum(
            p.numel() for p in self.trunk.parameters() if p.requires_grad
        )
        # Count output heads
        if self.is_multi_label:
            for source, head in self.output_heads.items():
                counts[f"output_head_{source}"] = sum(
                    p.numel() for p in head.parameters() if p.requires_grad
                )
        else:
            counts["output_head"] = sum(
                p.numel() for p in self.output_head.parameters() if p.requires_grad
            )
        counts["total"] = self.count_parameters()
        return counts

    def extra_repr(self) -> str:
        """Return string representation of model configuration."""
        repr_str = (
            f"surveys={self.survey_names}, "
            f"n_parameters={self.n_parameters}, "
            f"latent_dim={self.latent_dim}, "
            f"combination_mode='{self.combination_mode}'"
        )
        if self.is_multi_label:
            repr_str += f", label_sources={self.label_sources}"
        if self.residual_stream:
            repr_str += ", residual_stream=True"
        return repr_str
