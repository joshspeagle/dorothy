"""
Multi-layer perceptron (MLP) architecture for stellar parameter inference.

This module implements the configurable MLP architecture used by DOROTHY models.
The network takes 3-channel spectral data (flux, sigma, mask) as input and predicts
stellar parameters along with their uncertainties.

Default architecture (DOROTHY standard):
    Input: (batch, 3, wavelengths) -> Flatten -> (batch, 3*wavelengths)
    Hidden: 3*wavelengths -> 5000 -> 2000 -> 1000 -> 500 -> 200 -> 100
    Output: LabelAwareOutput, 100 -> (batch, 2, 11)

Input tensor has shape (batch, 3, wavelengths) where:
    - input[:, 0, :] contains the flux values
    - input[:, 1, :] contains the sigma (uncertainty) values
    - input[:, 2, :] contains the mask (1=valid, 0=invalid)

Output tensor has shape (batch, 2, n_params) where:
    - output[:, 0, :] contains the predicted means
    - output[:, 1, :] contains the predicted log-scatter values
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from dorothy.models.output import output_modules
from dorothy.models.utils import hidden_blocks

if TYPE_CHECKING:
    from dorothy.config.schema import ModelConfig


class MLP(nn.Module):
    """
    Configurable multi-layer perceptron for stellar parameter prediction.

    This network follows the DOROTHY architecture with configurable:
    - Hidden layer sizes
    - Normalization type (BatchNorm or LayerNorm)
    - Activation function (GELU, ReLU, or SiLU)
    - Dropout probability

    The network flattens 3D input tensors (batch, 3, wavelengths) to 2D
    before processing through the hidden layers. The last module, a
    LabelAwareOutput, reads the last hidden layer h and returns
    (batch, 2, n_params): the means, linear in h, and the log-scatter, from
    an error decoder with one hidden layer of h's size on h and the label
    errors.

    Input format (3-channel):
        - Channel 0: flux values
        - Channel 1: sigma (uncertainty) values
        - Channel 2: mask (1=valid, 0=invalid)

    Attributes:
        input_features: Number of input features after flattening (3 * wavelengths).
        output_features: Number of output features (2 * n_parameters).
        n_parameters: Number of stellar parameters being predicted.
        hidden_layers: List of hidden layer sizes.
        label_aware_uncertainty: Whether the last layer is a LabelAwareOutput
            (True) or an AdditiveOutput.
        layers: The sequential container of all network layers.

    Examples:
        >>> from dorothy.config import ModelConfig
        >>> config = ModelConfig(hidden_layers=[1000, 500, 100])
        >>> model = MLP.from_config(config)
        >>> x = torch.randn(32, 3, 4506)  # batch of 3-channel spectra
        >>> output = model(x)  # shape: (32, 2, 11)
        >>> means = output[:, 0, :]  # predicted means
        >>> log_scatter = output[:, 1, :]  # predicted log-scatter
    """

    def __init__(
        self,
        input_features: int = 15300,
        output_features: int = 22,
        hidden_layers: list[int] | None = None,
        normalization: str = "layernorm",
        activation: str = "gelu",
        dropout: float = 0.0,
        label_aware_uncertainty: bool = True,
    ) -> None:
        """
        Initialize the MLP.

        Args:
            input_features: Number of input features (channels * wavelength_bins).
                Default is 15300 (2 channels * 7650 wavelengths).
            output_features: Number of output features. Should be 2 * n_parameters
                to output both predictions and uncertainties. Default is 22.
            hidden_layers: List of hidden layer sizes. Default is the DOROTHY
                standard: [5000, 2000, 1000, 500, 200, 100].
            normalization: Type of normalization layer. One of:
                - "layernorm": Layer normalization (default, better for small batches)
                - "batchnorm": Batch normalization (best for large batches)
            activation: Activation function type. One of:
                - "gelu": Gaussian Error Linear Unit (default)
                - "relu": Rectified Linear Unit
                - "silu": Sigmoid Linear Unit (Swish)
            dropout: Dropout probability. Default is 0.0 (no dropout).
            label_aware_uncertainty: End in a LabelAwareOutput (default);
                False ends in the AdditiveOutput of old checkpoints.

        Raises:
            ValueError: If normalization or activation type is invalid.
            ValueError: If hidden_layers is empty.
        """
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [5000, 2000, 1000, 500, 200, 100]

        if len(hidden_layers) == 0:
            raise ValueError("hidden_layers cannot be empty")

        self.input_features = input_features
        self.output_features = output_features
        self.n_parameters = output_features // 2
        self.hidden_layers = hidden_layers
        self.label_aware_uncertainty = label_aware_uncertainty

        self.layers = nn.Sequential(
            nn.Flatten(),
            *hidden_blocks(
                input_features, hidden_layers, normalization, activation, dropout
            ),
            *output_modules(
                hidden_layers[-1],
                self.n_parameters,
                [],
                normalization,
                activation,
                dropout,
                label_aware_uncertainty,
            ),
        )

    def forward(
        self, x: torch.Tensor, label_errors: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, wavelength_bins), or
                pre-flattened (batch_size, input_features), where:

                - x[:, 0, :] contains the flux values
                - x[:, 1, :] contains the sigma (uncertainty) values
                - x[:, 2, :] contains the mask (1=valid, 0=invalid)
            label_errors: Reported label errors of shape
                (batch_size, n_parameters), in normalized units, for the
                log-scatter (see LabelAwareOutput). None uses the learned
                missing vector for every entry.

        Returns:
            Output tensor of shape (batch_size, 2, n_parameters) where:
                - output[:, 0, :] contains the predicted means
                - output[:, 1, :] contains the predicted log-scatter values

        Raises:
            ValueError: If label_errors is given to a model built with
                label_aware_uncertainty=False, or has the wrong shape.
        """
        return self.layers[-1](self.layers[:-1](x), label_errors)

    def get_embeddings(self, x: torch.Tensor, layer_index: int = -2) -> torch.Tensor:
        """
        Extract intermediate embeddings from the network.

        This is useful for k-NN anomaly detection and other analyses that
        require the learned feature representation.

        Args:
            x: Input tensor of shape (batch_size, channels, wavelength_bins).
            layer_index: Index of the layer to extract embeddings from.
                Default is -2 (second-to-last layer, before output).

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        # Get the layer to stop at (skip the output layer)
        # Need to account for: Flatten + (Linear + Norm + Activation + [Dropout]) * N + output
        # We want to get output after the last activation/dropout, before the output layer

        # Find the actual layer index
        layers_list = list(self.layers.children())
        if layer_index < 0:
            layer_index = len(layers_list) + layer_index

        # Run through layers up to the specified index
        result = x
        for i, layer in enumerate(layers_list):
            result = layer(result)
            if i == layer_index:
                break

        return result

    @classmethod
    def from_config(cls, config: ModelConfig) -> MLP:
        """
        Create an MLP from a ModelConfig.

        Args:
            config: Model configuration specifying architecture parameters.

        Returns:
            Configured MLP instance.
        """
        return cls(
            input_features=config.input_features,
            output_features=config.output_features,
            hidden_layers=list(config.hidden_layers),
            normalization=config.normalization.value,
            activation=config.activation.value,
            dropout=config.dropout,
            label_aware_uncertainty=config.label_aware_uncertainty,
        )

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        """Return a string representation of the model configuration."""
        return (
            f"input_features={self.input_features}, "
            f"output_features={self.output_features}, "
            f"hidden_layers={self.hidden_layers}"
        )
