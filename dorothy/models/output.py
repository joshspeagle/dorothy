"""The last layer of every output head, which predicts the mean and the log-scatter."""

from __future__ import annotations

import torch
import torch.nn as nn

from dorothy.losses.heteroscedastic import LN_S_MAX, LN_S_MIN
from dorothy.models.utils import get_activation


class LabelAwareOutput(nn.Module):
    """Predict the mean from a head's hidden vector, and ln s also from label errors.

    For the hidden vector h and the reported label errors sigma (normalized
    units), per parameter p::

        mu = W h + b
        u_p = clamp(ln sigma_p, LN_S_MIN, LN_S_MAX) if sigma_p > 0, else m_p
        ln_s = g([h, u])

    where m is the learned missing vector and g is Linear, activation, Linear.
    The mean never sees the label errors.

    Attributes:
        mean: The linear map from h to mu.
        missing_log_error: The missing vector m, shape (n_parameters,), in the
            units of u.
        scatter: The network g.
    """

    def __init__(self, input_dim: int, n_parameters: int, activation: str) -> None:
        """Initialize the layer.

        Args:
            input_dim: Size of the hidden vector h.
            n_parameters: Number of predicted parameters.
            activation: Activation of g ("gelu", "relu" or "silu").

        Raises:
            ValueError: If the activation is not recognized.
        """
        super().__init__()
        self.mean = nn.Linear(input_dim, n_parameters)
        # 0 is a label error of one normalized unit (the training labels'
        # interquartile range), far above real errors, so the missing state
        # starts distinct from them
        self.missing_log_error = nn.Parameter(torch.zeros(n_parameters))
        self.scatter = nn.Sequential(
            nn.Linear(input_dim + n_parameters, input_dim),
            get_activation(activation)(),
            nn.Linear(input_dim, n_parameters),
        )

    def forward(
        self, h: torch.Tensor, label_errors: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict the mean and log-scatter of every parameter.

        Args:
            h: Hidden vector, shape (batch, input_dim).
            label_errors: Reported label errors in normalized units, shape
                (batch, n_parameters). An entry > 0 is a reported error; any
                other value (0, negative, NaN) is missing. None means every
                entry is missing.

        Returns:
            Tensor of shape (batch, 2, n_parameters) = [mu, ln_s], in
            normalized units.

        Raises:
            ValueError: If label_errors does not have shape
                (batch, n_parameters).
        """
        u = self.missing_log_error.expand(h.shape[0], -1)
        if label_errors is not None:
            if label_errors.shape != u.shape:
                raise ValueError(
                    f"label_errors must have shape (batch, n_parameters) = "
                    f"{tuple(u.shape)}, got {tuple(label_errors.shape)}"
                )
            reported = label_errors > 0
            # Missing entries become 1 before the log, so no log is taken of
            # 0, a negative value or NaN
            log_errors = torch.log(torch.where(reported, label_errors, 1.0))
            u = torch.where(reported, log_errors.clamp(LN_S_MIN, LN_S_MAX), u)
        # u is float32; matching h's dtype avoids two full casts under AMP
        ln_s = self.scatter(torch.cat([h, u.to(h.dtype)], dim=1))
        return torch.stack([self.mean(h), ln_s], dim=1)


class AdditiveOutput(nn.Linear):
    """The last layer of checkpoints saved before label-aware uncertainty.

    One linear map from h to [mu, ln s]. These models add the label errors
    to s outside the network, as sqrt(sigma_y^2 + s^2).
    """

    def __init__(self, input_dim: int, n_parameters: int) -> None:
        """Initialize the layer.

        Args:
            input_dim: Size of the hidden vector h.
            n_parameters: Number of predicted parameters.
        """
        super().__init__(input_dim, 2 * n_parameters)

    def forward(
        self, h: torch.Tensor, label_errors: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict the mean and log-scatter of every parameter.

        Args:
            h: Hidden vector, shape (batch, input_dim).
            label_errors: Must be None, since the layer reads no label errors.

        Returns:
            Tensor of shape (batch, 2, n_parameters) = [mu, ln_s], in
            normalized units.

        Raises:
            ValueError: If label_errors is given.
        """
        if label_errors is not None:
            raise ValueError(
                "label_errors needs a model built with label_aware_uncertainty=True"
            )
        return super().forward(h).view(h.shape[0], 2, -1)


def output_layer(
    input_dim: int, n_parameters: int, activation: str, label_aware: bool
) -> LabelAwareOutput | AdditiveOutput:
    """Build the last layer of an output head.

    Args:
        input_dim: Size of the hidden vector h.
        n_parameters: Number of predicted parameters.
        activation: Activation of the LabelAwareOutput's scatter network.
        label_aware: Build a LabelAwareOutput; False builds an AdditiveOutput.

    Returns:
        The layer, which maps (h, label_errors) to (batch, 2, n_parameters).
    """
    if label_aware:
        return LabelAwareOutput(input_dim, n_parameters, activation)
    return AdditiveOutput(input_dim, n_parameters)
