"""
Multi-layer perceptron (MLP) architecture for stellar parameter inference.

This module implements the configurable MLP architecture used by DOROTHY models.
The network takes spectral data (flux and inverse variance) as input and predicts
stellar parameters along with their uncertainties.

Default architecture (DOROTHY standard):
    Input: (batch, 2, 7650) -> Flatten -> (batch, 15300)
    Hidden: 15300 -> 5000 -> 2000 -> 1000 -> 500 -> 200 -> 100
    Output: 100 -> 22 (11 parameters + 11 uncertainties)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

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

    The network flattens 3D input tensors (batch, channels, wavelengths) to 2D
    before processing through the hidden layers.

    Attributes:
        input_features: Number of input features after flattening.
        output_features: Number of output features (2 * n_parameters).
        hidden_layers: List of hidden layer sizes.
        layers: The sequential container of all network layers.

    Example:
        >>> from dorothy.config import ModelConfig
        >>> config = ModelConfig(hidden_layers=[1000, 500, 100])
        >>> model = MLP.from_config(config)
        >>> x = torch.randn(32, 2, 7650)  # batch of spectra
        >>> output = model(x)  # shape: (32, 22)
    """

    def __init__(
        self,
        input_features: int = 15300,
        output_features: int = 22,
        hidden_layers: list[int] | None = None,
        normalization: str = "batchnorm",
        activation: str = "gelu",
        dropout: float = 0.0,
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
                - "batchnorm": Batch normalization (default, best for large batches)
                - "layernorm": Layer normalization (better for small batches)
            activation: Activation function type. One of:
                - "gelu": Gaussian Error Linear Unit (default)
                - "relu": Rectified Linear Unit
                - "silu": Sigmoid Linear Unit (Swish)
            dropout: Dropout probability. Default is 0.0 (no dropout).

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
        self.hidden_layers = hidden_layers

        # Select activation function
        activation_fn = self._get_activation(activation)

        # Select normalization class
        norm_class = self._get_normalization(normalization)

        # Build the network layers
        layers: list[nn.Module] = [nn.Flatten()]

        # Input layer
        prev_size = input_features
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(norm_class(hidden_size))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_size = hidden_size

        # Output layer (no normalization or activation)
        layers.append(nn.Linear(prev_size, output_features))

        self.layers = nn.Sequential(*layers)

    @staticmethod
    def _get_activation(activation: str) -> type[nn.Module]:
        """Get the activation function class."""
        activations = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        if activation.lower() not in activations:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Supported: {list(activations.keys())}"
            )
        return activations[activation.lower()]

    @staticmethod
    def _get_normalization(normalization: str) -> type[nn.Module]:
        """Get the normalization layer class."""
        normalizations = {
            "batchnorm": nn.BatchNorm1d,
            "layernorm": nn.LayerNorm,
        }
        if normalization.lower() not in normalizations:
            raise ValueError(
                f"Unknown normalization '{normalization}'. "
                f"Supported: {list(normalizations.keys())}"
            )
        return normalizations[normalization.lower()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, wavelength_bins) or
               (batch_size, input_features). If 3D, the tensor is flattened.

        Returns:
            Output tensor of shape (batch_size, output_features) containing
            predicted parameters and their uncertainties.
        """
        return self.layers(x)

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
        # Get the layer to stop at (skip final linear layer)
        # Need to account for: Flatten + (Linear + Norm + Activation + [Dropout]) * N + Linear
        # We want to get output after the last activation/dropout, before final linear

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
