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

    Output Head:
        Input: (batch, trunk_output_dim) -> (batch, 2, n_params)

For multi-survey training:
    - Each sample may have data from one or more surveys
    - Encoders only process surveys with valid data (indicated by has_data mask)
    - Encoded representations are combined (averaged or concatenated) before trunk
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SurveyEncoder(nn.Module):
    """Encoder network for a single survey's spectral data.

    Takes 3-channel spectral input (flux, sigma, mask) and produces a
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
            dropout: Dropout probability.
        """
        super().__init__()

        self.survey_name = survey_name
        self.n_wavelengths = n_wavelengths
        self.latent_dim = latent_dim

        if hidden_layers is None:
            hidden_layers = [1024, 512]

        # Input size: 3 channels * n_wavelengths
        input_features = 3 * n_wavelengths

        # Build encoder layers
        activation_fn = self._get_activation(activation)
        norm_class = self._get_normalization(normalization)

        layers: list[nn.Module] = [nn.Flatten()]

        prev_size = input_features
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(norm_class(hidden_size))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_size = hidden_size

        # Final projection to latent space
        layers.append(nn.Linear(prev_size, latent_dim))
        layers.append(norm_class(latent_dim))
        layers.append(activation_fn())

        self.encoder = nn.Sequential(*layers)

    @staticmethod
    def _get_activation(activation: str) -> type[nn.Module]:
        """Get the activation function class."""
        activations = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
        if activation.lower() not in activations:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Valid options: {list(activations.keys())}"
            )
        return activations[activation.lower()]

    @staticmethod
    def _get_normalization(normalization: str) -> type[nn.Module]:
        """Get the normalization layer class."""
        normalizations = {"batchnorm": nn.BatchNorm1d, "layernorm": nn.LayerNorm}
        if normalization.lower() not in normalizations:
            raise ValueError(
                f"Unknown normalization '{normalization}'. "
                f"Valid options: {list(normalizations.keys())}"
            )
        return normalizations[normalization.lower()]

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
    features for the output head.

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
    ) -> None:
        """Initialize the shared trunk.

        Args:
            input_dim: Input dimension from encoder(s).
            output_dim: Output dimension for output head.
            hidden_layers: Hidden layer sizes. Default: [512, 256].
            normalization: Normalization type.
            activation: Activation function.
            dropout: Dropout probability.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if hidden_layers is None:
            hidden_layers = [512, 256]

        activation_fn = SurveyEncoder._get_activation(activation)
        norm_class = SurveyEncoder._get_normalization(normalization)

        layers: list[nn.Module] = []
        prev_size = input_dim

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(norm_class(hidden_size))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_size = hidden_size

        # Final layer to output_dim
        layers.append(nn.Linear(prev_size, output_dim))
        layers.append(norm_class(output_dim))
        layers.append(activation_fn())

        self.trunk = nn.Sequential(*layers)

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

    Takes features from the shared trunk and produces predictions
    in the standard (batch, 2, n_params) format.

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
    ) -> None:
        """Initialize the output head.

        Args:
            input_dim: Input dimension from shared trunk.
            n_parameters: Number of stellar parameters to predict.
            hidden_layers: Hidden layer sizes. Default: [64].
            normalization: Normalization type.
            activation: Activation function.
            dropout: Dropout probability.
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_parameters = n_parameters
        self.output_features = 2 * n_parameters

        if hidden_layers is None:
            hidden_layers = [64]

        activation_fn = SurveyEncoder._get_activation(activation)
        norm_class = SurveyEncoder._get_normalization(normalization)

        layers: list[nn.Module] = []
        prev_size = input_dim

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(norm_class(hidden_size))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_size = hidden_size

        # Final output layer (no norm or activation)
        layers.append(nn.Linear(prev_size, self.output_features))

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce parameter predictions.

        Args:
            x: Features tensor of shape (batch, input_dim).

        Returns:
            Predictions of shape (batch, 2, n_parameters) where:
                - output[:, 0, :] contains predicted means
                - output[:, 1, :] contains predicted log-scatter
        """
        flat_output = self.head(x)  # (batch, 2 * n_params)
        batch_size = flat_output.shape[0]
        return flat_output.view(batch_size, 2, self.n_parameters)


class MultiHeadMLP(nn.Module):
    """Multi-head MLP for multi-survey stellar parameter prediction.

    This architecture has separate encoder networks for each survey,
    a shared trunk that combines the encodings, and an output head
    that produces predictions.

    The model can handle:
    - Single survey input (standard case)
    - Multiple surveys with per-sample availability masks

    Attributes:
        survey_names: List of survey names in order.
        n_parameters: Number of stellar parameters being predicted.
        combination_mode: How to combine multi-survey embeddings ("mean" or "concat").
        encoders: ModuleDict of survey encoders.
        trunk: Shared trunk network.
        output_head: Output prediction head.

    Example:
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
    ) -> None:
        """Initialize the multi-head MLP.

        Args:
            survey_configs: Dict mapping survey names to their wavelength counts.
                Example: {"boss": 4506, "lamost_lrs": 3700}
            n_parameters: Number of stellar parameters to predict.
            latent_dim: Output dimension of each survey encoder.
            encoder_hidden: Hidden layers for encoders. Default: [1024, 512].
            trunk_hidden: Hidden layers for shared trunk. Default: [512, 256].
            output_hidden: Hidden layers for output head. Default: [64].
            combination_mode: How to combine multi-survey embeddings. Default: "concat".
                - "concat": Concatenate embeddings (trunk input = n_surveys * latent_dim).
                    Missing surveys are zero-padded.
                - "mean": Average embeddings from available surveys (excludes missing)
            normalization: Normalization type for all components.
            activation: Activation function for all components.
            dropout: Dropout probability for all components.
            label_sources: List of label sources for multi-output heads. Default: None
                (single output head). If multiple sources specified, creates one
                output head per source (e.g., ["apogee", "galah"]).

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
            output_dim=128,  # Fixed trunk output
            hidden_layers=trunk_hidden,
            normalization=normalization,
            activation=activation,
            dropout=dropout,
        )

        # Create output head(s) - one per label source
        if self.is_multi_label:
            self.output_heads = nn.ModuleDict()
            for source in self.label_sources:
                self.output_heads[source] = OutputHead(
                    input_dim=self.trunk.output_dim,
                    n_parameters=n_parameters,
                    hidden_layers=output_hidden,
                    normalization=normalization,
                    activation=activation,
                    dropout=dropout,
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
                dropout=dropout,
            )
            # For consistency
            self.output_heads = nn.ModuleDict({"default": self.output_head})

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
        if survey not in self.encoders:
            raise KeyError(f"Unknown survey '{survey}'. Available: {self.survey_names}")

        # Encode
        latent = self.encoders[survey](x)  # (batch, latent_dim)

        # For concat mode with single survey, pad with zeros for other surveys
        if self.combination_mode == "concat":
            batch_size = latent.shape[0]
            survey_idx = self.survey_names.index(survey)

            # Create full latent with zeros
            full_latent = torch.zeros(
                batch_size,
                self.n_surveys * self.latent_dim,
                device=latent.device,
                dtype=latent.dtype,
            )
            # Insert this survey's latent at the right position
            start_idx = survey_idx * self.latent_dim
            end_idx = start_idx + self.latent_dim
            full_latent[:, start_idx:end_idx] = latent

            latent = full_latent

        # Through trunk and output
        features = self.trunk(latent)
        return self.output_head(features)

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
            Predictions of shape (batch, 2, n_parameters).

        Note:
            Samples must have data from at least one survey.
            If a sample has no data from any survey, the output will be zeros.
        """
        batch_size = next(iter(inputs.values())).shape[0]
        device = next(iter(inputs.values())).device
        dtype = next(iter(inputs.values())).dtype

        if self.combination_mode == "concat":
            # Initialize full concatenated latent
            combined = torch.zeros(
                batch_size, self.n_surveys * self.latent_dim, device=device, dtype=dtype
            )

            for i, survey_name in enumerate(self.survey_names):
                if survey_name not in inputs:
                    continue

                x = inputs[survey_name]
                mask = has_data.get(
                    survey_name, torch.ones(batch_size, dtype=torch.bool, device=device)
                )

                if mask.any():
                    # Encode samples that have this survey's data
                    latent = self.encoders[survey_name](x)  # (batch, latent_dim)

                    # Insert at correct position
                    start_idx = i * self.latent_dim
                    end_idx = start_idx + self.latent_dim

                    # Use mask to only update relevant samples
                    mask_expanded = mask.unsqueeze(1).expand(-1, self.latent_dim)
                    combined[:, start_idx:end_idx] = torch.where(
                        mask_expanded, latent, combined[:, start_idx:end_idx]
                    )

        else:  # mean combination
            # Accumulate weighted sum
            combined = torch.zeros(
                batch_size, self.latent_dim, device=device, dtype=dtype
            )
            count = torch.zeros(batch_size, 1, device=device, dtype=dtype)

            for survey_name in self.survey_names:
                if survey_name not in inputs:
                    continue

                x = inputs[survey_name]
                mask = has_data.get(
                    survey_name, torch.ones(batch_size, dtype=torch.bool, device=device)
                )

                if mask.any():
                    latent = self.encoders[survey_name](x)  # (batch, latent_dim)

                    # Add to sum for samples with this survey
                    mask_float = mask.float().unsqueeze(1)
                    combined = combined + latent * mask_float
                    count = count + mask_float

            # Average (avoid division by zero)
            count = count.clamp(min=1)
            combined = combined / count

        # Through trunk and output
        features = self.trunk(combined)
        return self.output_head(features)

    def forward(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        survey: str | None = None,
        has_data: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Unified forward pass.

        Dispatches to forward_single or forward_multi based on input type.

        Args:
            x: Either a single tensor (with survey name) or dict of tensors.
            survey: Survey name if x is a single tensor.
            has_data: Data availability masks if x is a dict.

        Returns:
            Predictions of shape (batch, 2, n_parameters).
        """
        if isinstance(x, dict):
            # Multi-survey mode
            if has_data is None:
                # Assume all samples have all surveys
                batch_size = next(iter(x.values())).shape[0]
                device = next(iter(x.values())).device
                has_data = {
                    name: torch.ones(batch_size, dtype=torch.bool, device=device)
                    for name in x
                }
            return self.forward_multi(x, has_data)
        else:
            # Single survey mode
            if survey is None:
                if len(self.survey_names) == 1:
                    survey = self.survey_names[0]
                else:
                    raise ValueError(
                        "Must specify survey for single-tensor input with multiple surveys"
                    )
            return self.forward_single(x, survey)

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

        # Get trunk features using existing forward logic
        if isinstance(x, dict):
            # Multi-survey mode
            if has_data is None:
                batch_size = next(iter(x.values())).shape[0]
                device = next(iter(x.values())).device
                has_data = {
                    name: torch.ones(batch_size, dtype=torch.bool, device=device)
                    for name in x
                }
            features = self._forward_to_trunk(x, has_data)
        else:
            # Single survey mode
            if survey is None:
                if len(self.survey_names) == 1:
                    survey = self.survey_names[0]
                else:
                    raise ValueError("Must specify survey for single-tensor input")
            features = self._forward_single_to_trunk(x, survey)

        return self.output_heads[label_source](features)

    def forward_all_label_sources(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        survey: str | None = None,
        has_data: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through all label source output heads.

        Useful for inference when you want predictions from all label sources.

        Args:
            x: Input tensor or dict of tensors.
            survey: Survey name if x is single tensor.
            has_data: Data availability masks if x is dict.

        Returns:
            Dict mapping label source names to predictions.
            Each prediction has shape (batch, 2, n_parameters).
        """
        # Get trunk features
        if isinstance(x, dict):
            if has_data is None:
                batch_size = next(iter(x.values())).shape[0]
                device = next(iter(x.values())).device
                has_data = {
                    name: torch.ones(batch_size, dtype=torch.bool, device=device)
                    for name in x
                }
            features = self._forward_to_trunk(x, has_data)
        else:
            if survey is None:
                if len(self.survey_names) == 1:
                    survey = self.survey_names[0]
                else:
                    raise ValueError("Must specify survey for single-tensor input")
            features = self._forward_single_to_trunk(x, survey)

        return {source: head(features) for source, head in self.output_heads.items()}

    def _forward_single_to_trunk(self, x: torch.Tensor, survey: str) -> torch.Tensor:
        """Forward through encoder and trunk for single survey."""
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
        batch_size = next(iter(inputs.values())).shape[0]
        device = next(iter(inputs.values())).device
        dtype = next(iter(inputs.values())).dtype

        if self.combination_mode == "concat":
            combined = torch.zeros(
                batch_size, self.n_surveys * self.latent_dim, device=device, dtype=dtype
            )

            for i, survey_name in enumerate(self.survey_names):
                if survey_name not in inputs:
                    continue

                x = inputs[survey_name]
                mask = has_data.get(
                    survey_name, torch.ones(batch_size, dtype=torch.bool, device=device)
                )

                if mask.any():
                    latent = self.encoders[survey_name](x)
                    start_idx = i * self.latent_dim
                    end_idx = start_idx + self.latent_dim
                    mask_expanded = mask.unsqueeze(1).expand(-1, self.latent_dim)
                    combined[:, start_idx:end_idx] = torch.where(
                        mask_expanded, latent, combined[:, start_idx:end_idx]
                    )

        else:  # mean combination
            combined = torch.zeros(
                batch_size, self.latent_dim, device=device, dtype=dtype
            )
            count = torch.zeros(batch_size, 1, device=device, dtype=dtype)

            for survey_name in self.survey_names:
                if survey_name not in inputs:
                    continue

                x = inputs[survey_name]
                mask = has_data.get(
                    survey_name, torch.ones(batch_size, dtype=torch.bool, device=device)
                )

                if mask.any():
                    latent = self.encoders[survey_name](x)
                    mask_float = mask.float().unsqueeze(1)
                    combined = combined + latent * mask_float
                    count = count + mask_float

            count = count.clamp(min=1)
            combined = combined / count

        return self.trunk(combined)

    def get_embeddings(
        self,
        x: torch.Tensor,
        survey: str,
    ) -> torch.Tensor:
        """Get latent embeddings for a survey's input.

        Useful for anomaly detection and analysis.

        Args:
            x: Input tensor of shape (batch, 3, n_wavelengths).
            survey: Name of the survey.

        Returns:
            Latent embeddings of shape (batch, latent_dim).
        """
        return self.encoders[survey](x)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_component(self) -> dict[str, int]:
        """Count trainable parameters by component."""
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
        return repr_str
