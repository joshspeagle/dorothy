"""The layers that map an output head's input to the mean and log-scatter."""

from __future__ import annotations

import torch
import torch.nn as nn

from dorothy.losses.heteroscedastic import LN_S_MAX, LN_S_MIN
from dorothy.models.utils import hidden_blocks, residual_blocks


def _decoder(
    input_dim: int,
    n_parameters: int,
    hidden_layers: list[int],
    normalization: str,
    activation: str,
    dropout: float,
    residual: bool,
) -> nn.Sequential:
    """Hidden layers of hidden_layers, then a linear map to n_parameters.

    The hidden layers are plain hidden blocks, or with residual one
    ResidualBlock of width input_dim per hidden size and a closing
    normalization.
    """
    if residual:
        return nn.Sequential(
            *residual_blocks(
                input_dim, hidden_layers, normalization, activation, dropout
            ),
            nn.Linear(input_dim, n_parameters),
        )
    return nn.Sequential(
        *hidden_blocks(input_dim, hidden_layers, normalization, activation, dropout),
        nn.Linear(hidden_layers[-1], n_parameters),
    )


class LabelAwareOutput(nn.Module):
    """Predict the mean, and ln s also from label errors, with parallel decoders.

    Two decoders read the head input x. For the reported label errors sigma
    (normalized units), per parameter p::

        mu = D_mu(x)
        u_p = clamp(ln sigma_p, LN_S_MIN, LN_S_MAX) if sigma_p > 0, else m_p
        ln_s = D_s([x, u])

    where m is the learned missing vector. Each decoder is the hidden blocks
    of hidden_layers (Linear, normalization, activation, dropout), then a
    linear map; u enters the first layer of D_s. At least one hidden layer
    makes ln s a nonlinear function of u. No layer that feeds mu reads u.

    With residual, x is a residual stream of width input_dim. Each decoder
    is one ResidualBlock per hidden size, a normalization and a linear map,
    and u is added to a copy of x instead of concatenated::

        ln_s = D_s(x + W_u u)

    Attributes:
        mean: The label decoder D_mu.
        missing_log_error: The missing vector m, shape (n_parameters,), in the
            units of u.
        scatter: The error decoder D_s.
        error_projection: The map W_u from u to the stream, without bias;
            None without residual.
    """

    def __init__(
        self,
        input_dim: int,
        n_parameters: int,
        hidden_layers: list[int],
        normalization: str,
        activation: str,
        dropout: float,
        residual: bool,
    ) -> None:
        """Initialize the decoders.

        Args:
            input_dim: Size of the head input x.
            n_parameters: Number of predicted parameters.
            hidden_layers: Hidden layer sizes of both decoders, at least one.
            normalization: Normalization type ("batchnorm" or "layernorm").
            activation: Activation function ("gelu", "relu" or "silu").
            dropout: Dropout probability of the hidden blocks.
            residual: Build residual decoders that read u through W_u.

        Raises:
            ValueError: If hidden_layers is empty, or the normalization or
                activation is not recognized.
        """
        if not hidden_layers:
            raise ValueError(
                "hidden_layers must have at least one size, so that ln s is a "
                "nonlinear function of the label errors"
            )
        super().__init__()
        self.mean = _decoder(
            input_dim,
            n_parameters,
            hidden_layers,
            normalization,
            activation,
            dropout,
            residual,
        )
        # 0 is a label error of one normalized unit (the training labels'
        # interquartile range), far above real errors, so the missing state
        # starts distinct from them
        self.missing_log_error = nn.Parameter(torch.zeros(n_parameters))
        self.error_projection = (
            nn.Linear(n_parameters, input_dim, bias=False) if residual else None
        )
        self.scatter = _decoder(
            input_dim if residual else input_dim + n_parameters,
            n_parameters,
            hidden_layers,
            normalization,
            activation,
            dropout,
            residual,
        )

    def forward(
        self, x: torch.Tensor, label_errors: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict the mean and log-scatter of every parameter.

        Args:
            x: Head input, shape (batch, input_dim).
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
        u = self.missing_log_error.expand(x.shape[0], -1)
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
        # u is float32; matching x's dtype avoids two full casts under AMP
        u = u.to(x.dtype)
        if self.error_projection is None:
            ln_s = self.scatter(torch.cat([x, u], dim=1))
        else:
            ln_s = self.scatter(x + self.error_projection(u))
        return torch.stack([self.mean(x), ln_s], dim=1)


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


def output_modules(
    input_dim: int,
    n_parameters: int,
    hidden_layers: list[int],
    normalization: str,
    activation: str,
    dropout: float,
    label_aware: bool,
    residual: bool = False,
) -> list[nn.Module]:
    """Build the modules of an output head, from its input to [mu, ln s].

    Args:
        input_dim: Size of the head input.
        n_parameters: Number of predicted parameters.
        hidden_layers: Hidden layer sizes between the input and the means.
        normalization: Normalization type ("batchnorm" or "layernorm").
        activation: Activation function ("gelu", "relu" or "silu").
        dropout: Dropout probability of the hidden blocks.
        label_aware: Build a LabelAwareOutput; False builds the hidden blocks
            and the AdditiveOutput of checkpoints saved before label-aware
            uncertainty.
        residual: Read a residual stream: the hidden layers are residual
            blocks of width input_dim (see LabelAwareOutput).

    Returns:
        The modules in order. The last one maps (features, label_errors) to
        (batch, 2, n_parameters); the ones before it map the input to the
        features.

    Raises:
        ValueError: If label_aware and hidden_layers is empty, or the
            normalization or activation is not recognized.
    """
    if label_aware:
        return [
            LabelAwareOutput(
                input_dim,
                n_parameters,
                hidden_layers,
                normalization,
                activation,
                dropout,
                residual,
            )
        ]
    if residual:
        return [
            *residual_blocks(
                input_dim, hidden_layers, normalization, activation, dropout
            ),
            AdditiveOutput(input_dim, n_parameters),
        ]
    width = hidden_layers[-1] if hidden_layers else input_dim
    return [
        *hidden_blocks(input_dim, hidden_layers, normalization, activation, dropout),
        AdditiveOutput(width, n_parameters),
    ]
