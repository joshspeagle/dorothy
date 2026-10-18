"""Shared utilities for neural network model building.

This module provides common functions used across different model architectures,
including activation and normalization layer factories.
"""

from __future__ import annotations

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
