"""Shared utilities for neural network model building.

This module provides common functions used across different model architectures,
including activation and normalization layer factories and the residual
block of the multi-head model's residual stream.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def get_activation(activation: str) -> type[nn.Module]:
    """Get the activation function class by name.

    Args:
        activation: Name of activation function ("gelu", "relu", "silu").

    Returns:
        The activation module class.

    Raises:
        ValueError: If activation name is not recognized.

    Examples:
        >>> act_class = get_activation("gelu")
        >>> layer = act_class()
    """
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


def get_normalization(normalization: str) -> type[nn.Module]:
    """Get the normalization layer class by name.

    Args:
        normalization: Name of normalization ("batchnorm", "layernorm").

    Returns:
        The normalization module class.

    Raises:
        ValueError: If normalization name is not recognized.

    Examples:
        >>> norm_class = get_normalization("layernorm")
        >>> layer = norm_class(256)  # 256 features
    """
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


def hidden_blocks(
    input_dim: int,
    hidden_layers: list[int],
    normalization: str,
    activation: str,
    dropout: float,
) -> list[nn.Module]:
    """Build the hidden layers of a network, one block per layer.

    Each block is Linear, normalization, activation and, when dropout > 0,
    Dropout.

    Args:
        input_dim: Size of the input.
        hidden_layers: Size of each hidden layer, in order.
        normalization: Normalization type ("batchnorm" or "layernorm").
        activation: Activation function ("gelu", "relu" or "silu").
        dropout: Dropout probability.

    Returns:
        The modules of all blocks, in order.

    Raises:
        ValueError: If the normalization or activation is not recognized.
    """
    activation_fn = get_activation(activation)
    norm_class = get_normalization(normalization)
    modules: list[nn.Module] = []
    for size in hidden_layers:
        modules += [nn.Linear(input_dim, size), norm_class(size), activation_fn()]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))
        input_dim = size
    return modules


class ResidualBlock(nn.Module):
    """A pre-norm residual block of constant width.

    For an input x of size width::

        x <- x + W_2 act(W_1 norm(x))

    W_1 maps width to hidden and W_2 maps hidden back to width, both with a
    bias. When dropout > 0, Dropout follows the activation.

    Attributes:
        update: The layers that compute the update added to x.
    """

    def __init__(
        self,
        width: int,
        hidden: int,
        normalization: str,
        activation: str,
        dropout: float,
    ) -> None:
        """Initialize the block.

        Args:
            width: Size of the input and output (the stream width).
            hidden: Size of the hidden layer.
            normalization: Normalization type ("batchnorm" or "layernorm").
            activation: Activation function ("gelu", "relu" or "silu").
            dropout: Dropout probability.

        Raises:
            ValueError: If the normalization or activation is not recognized.
        """
        super().__init__()
        layers = [
            get_normalization(normalization)(width),
            nn.Linear(width, hidden),
            get_activation(activation)(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden, width))
        self.update = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the block's update to x.

        Args:
            x: Input of shape (batch, width).

        Returns:
            Output of shape (batch, width).
        """
        return x + self.update(x)
