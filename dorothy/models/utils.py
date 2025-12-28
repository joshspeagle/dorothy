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

    Example:
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

    Example:
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
