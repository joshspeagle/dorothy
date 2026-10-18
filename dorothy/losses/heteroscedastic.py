"""
Heteroscedastic loss function for stellar parameter inference.

The loss is the Gaussian negative log-likelihood of each label under the
prediction, up to a constant and a factor of 2::

    s = sqrt(exp(2 * ln_s) + s_0^2)
    loss = (mu - y)^2 / s^2 + log(s^2)

Where:
    - mu: predicted stellar parameter value
    - ln_s: predicted log-scatter. The model computes it from the reported
      label error or a learned missing vector (see
      :class:`~dorothy.models.output.LabelAwareOutput`), so s is the whole
      expected scatter between mu and the label, label noise included.
    - y: label value
    - s_0: scatter floor to prevent numerical instability

A ``beta`` above 0 weights the loss of each label as the β-NLL (see
:meth:`~dorothy.losses.heteroscedastic.HeteroscedasticLoss.element_losses`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ln_s is clamped before exponentiating: s stays in [4.5e-5, 148] (normalized units)
LN_S_MIN = -10.0
LN_S_MAX = 5.0


def _safe_targets(
    mu: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Labels and mask of a (batch, 3, P) target, neutralizing masked entries.

    Masked labels are replaced by the (detached) prediction before any
    arithmetic, so non-finite values stored in masked entries (e.g. log10(0)
    for a masked Teff) can't produce NaN losses or gradients.
    """
    mask = target[:, 2, :]
    y = torch.where(mask > 0, target[:, 0, :], mu.detach())
    return y, mask


def predicted_scatter(ln_s: torch.Tensor, scatter_floor: float) -> torch.Tensor:
    """Convert predicted log-scatter to scatter: sqrt(exp(2 ln_s) + s_0^2).

    This is the single definition used by the loss, training validation,
    evaluation and inference, so predicted uncertainties agree everywhere.

    Args:
        ln_s: Predicted log-scatter (any shape).
        scatter_floor: Scatter floor s_0.

    Returns:
        Predicted scatter, same shape as ln_s.
    """
    ln_s = torch.clamp(ln_s, min=LN_S_MIN, max=LN_S_MAX)
    return torch.sqrt(torch.exp(2 * ln_s) + scatter_floor**2)


class HeteroscedasticLoss(nn.Module):
    """
    Heteroscedastic loss for regression with predicted uncertainties.

    loss = (mu - y)^2 / s^2 + log(s^2), where s is the predicted scatter
    (:func:`~dorothy.losses.heteroscedastic.predicted_scatter`). The squared
    error is weighted by the predicted variance, and the log-variance term
    penalizes underconfidence.

    The model output has shape (batch, 2, n_parameters):
    - output[:, 0, :]: predicted means (mu)
    - output[:, 1, :]: predicted log-scatter (ln_s)

    The target has 3-channel format (batch, 3, n_parameters):
    - target[:, 0, :]: true labels (y)
    - target[:, 1, :]: label errors, not used here (the model reads them)
    - target[:, 2, :]: mask (1=valid, 0=masked) - used for loss computation

    Attributes:
        scatter_floor: Minimum scatter value (s_0) to prevent division by zero.
        n_parameters: Number of stellar parameters being predicted.
        reduction: How to reduce the loss ('mean', 'sum', or 'none').

    Examples:
        >>> loss_fn = HeteroscedasticLoss(scatter_floor=0.01, n_parameters=11)
        >>> output = model(x)  # Shape: (batch, 2, 11)
        >>> target = labels    # Shape: (batch, 3, 11) - [y, sigma, mask]
        >>> loss = loss_fn(output, target)
    """

    def __init__(
        self,
        scatter_floor: float = 0.01,
        n_parameters: int = 11,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize the heteroscedastic loss.

        Args:
            scatter_floor: Minimum scatter value (s_0) to ensure numerical stability.
                This prevents the model from predicting arbitrarily small uncertainties.
                Default is 0.01, matching the original DOROTHY implementation.
            n_parameters: Number of stellar parameters being predicted.
                The model output should have 2 * n_parameters values.
            reduction: Specifies the reduction to apply to the output:
                'none': no reduction, returns loss for each element
                'mean': returns the mean of all losses (default)
                'sum': returns the sum of all losses

        Raises:
            ValueError: If scatter_floor is not positive.
            ValueError: If n_parameters is not positive.
            ValueError: If reduction is not one of 'none', 'mean', 'sum'.
        """
        super().__init__()

        if scatter_floor <= 0:
            raise ValueError(f"scatter_floor must be positive, got {scatter_floor}")
        if n_parameters <= 0:
            raise ValueError(f"n_parameters must be positive, got {n_parameters}")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'"
            )

        self.scatter_floor = scatter_floor
        self.n_parameters = n_parameters
        self.reduction = reduction

    def _check_shapes(self, output: torch.Tensor, target: torch.Tensor) -> None:
        if output.shape[1:] != (2, self.n_parameters):
            raise ValueError(
                f"Output shape should be (batch, 2, {self.n_parameters}), "
                f"got {output.shape}"
            )
        if target.shape[1:] != (3, self.n_parameters):
            raise ValueError(
                f"Target shape should be (batch, 3, {self.n_parameters}), "
                f"got {target.shape}"
            )

    def element_losses(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        beta: float = 0.0,
    ) -> torch.Tensor:
        """Unreduced loss per (sample, parameter), zero where masked.

        loss = sg(s^2)^beta * ((mu - y)^2 / s^2 + log(s^2)), the β-NLL of
        Seitzer et al. (2022), where sg passes its value but no gradient.
        With beta = 1 the gradient for mu is 2 (mu - y), that of the squared
        error, and beta = 0 gives the plain loss.

        Masked entries are zeroed with ``torch.where`` (not multiplied by the
        mask), so non-finite values in masked entries cannot leak NaN into
        the loss.

        Args:
            output: Model predictions of shape (batch_size, 2, n_parameters).
            target: Targets of shape (batch_size, 3, n_parameters) holding
                [labels, label errors, mask].
            beta: Exponent of the variance weight, in [0, 1].

        Returns:
            Tensor of shape (batch_size, n_parameters).

        Raises:
            ValueError: If output or target shapes don't match expectations,
                or beta is outside [0, 1].
        """
        self._check_shapes(output, target)
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")

        mu = output[:, 0, :]
        y, mask = _safe_targets(mu, target)
        variance = self.get_predicted_scatter(output) ** 2

        loss = (mu - y) ** 2 / variance + torch.log(variance)
        if beta > 0:
            loss = loss * variance.detach() ** beta
        return torch.where(mask > 0, loss * mask, torch.zeros_like(loss))

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        beta: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute the heteroscedastic loss.

        Args:
            output: Model predictions of shape (batch_size, 2, n_parameters).
                output[:, 0, :] contains predicted means (mu).
                output[:, 1, :] contains predicted log-scatter (ln_s).
            target: Target values of shape (batch_size, 3, n_parameters).
                target[:, 0, :] contains true labels (y).
                target[:, 1, :] contains label errors (not used).
                target[:, 2, :] contains mask (1=valid, 0=masked).
            beta: Exponent of the β-NLL variance weight, in [0, 1] (see
                :meth:`element_losses`).

        Returns:
            The computed loss, reduced according to self.reduction.

        Raises:
            ValueError: If output or target shapes don't match expectations,
                or beta is outside [0, 1].
        """
        loss_per_element = self.element_losses(output, target, beta)

        if self.reduction == "none":
            return loss_per_element
        elif self.reduction == "sum":
            return loss_per_element.sum()
        else:  # mean over valid elements only
            return loss_per_element.sum() / target[:, 2, :].sum().clamp(min=1)

    def get_predicted_scatter(self, output: torch.Tensor) -> torch.Tensor:
        """
        Extract the predicted scatter (standard deviation) from model output.

        This is useful for post-processing predictions to get uncertainty estimates.

        Args:
            output: Model predictions of shape (batch_size, 2, n_parameters).

        Returns:
            Predicted scatter of shape (batch_size, n_parameters), computed as
            sqrt(exp(2 * ln_s) + s_0^2).
        """
        return predicted_scatter(output[:, 1, :], self.scatter_floor)

    def forward_detailed(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss with detailed per-parameter and per-component breakdown.

        Returns both the total loss and a breakdown into:
        - Mean component per parameter: (mu - y)² / s²
        - Scatter component per parameter: log(s²)

        Args:
            output: Model predictions of shape (batch_size, 2, n_parameters).
            target: Target values of shape (batch_size, 3, n_parameters).
                target[:, 0, :] contains true labels (y).
                target[:, 1, :] contains label errors (not used).
                target[:, 2, :] contains mask (1=valid, 0=masked).

        Returns:
            Dictionary containing:
                - 'loss': Total scalar loss (reduced)
                - 'mean_component': Per-parameter mean loss, shape (n_parameters,)
                - 'scatter_component': Per-parameter scatter loss, shape (n_parameters,)
                - 'residuals': Raw residuals (mu - y), shape (batch, n_parameters)
                - 'y_pred': Predicted means, shape (batch, n_parameters)
                - 'pred_scatter': Predicted scatter, shape (batch, n_parameters)
        """
        self._check_shapes(output, target)
        mu = output[:, 0, :]
        y, mask = _safe_targets(mu, target)

        pred_scatter = predicted_scatter(output[:, 1, :], self.scatter_floor)
        variance = pred_scatter**2

        residuals = mu - y
        mean_component = residuals**2 / variance  # (batch, n_params)
        scatter_component = torch.log(variance)  # (batch, n_params)

        # Mask with torch.where so non-finite masked entries cannot leak NaN
        valid = mask > 0
        zeros = torch.zeros_like(mean_component)
        masked_mean = torch.where(valid, mean_component * mask, zeros)
        masked_scatter = torch.where(valid, scatter_component * mask, zeros)
        loss_per_element = masked_mean + masked_scatter

        # Average over batch for each parameter, counting only valid samples
        valid_counts = mask.sum(dim=0).clamp(min=1)  # (n_params,)
        mean_component_avg = masked_mean.sum(dim=0) / valid_counts
        scatter_component_avg = masked_scatter.sum(dim=0) / valid_counts

        if self.reduction == "none":
            total_loss = loss_per_element
        elif self.reduction == "sum":
            total_loss = loss_per_element.sum()
        else:  # mean
            total_loss = loss_per_element.sum() / mask.sum().clamp(min=1)

        return {
            "loss": total_loss,
            "mean_component": mean_component_avg,
            "scatter_component": scatter_component_avg,
            "residuals": residuals,
            "y_pred": mu,
            "pred_scatter": pred_scatter,
        }

    def extra_repr(self) -> str:
        """Return a string representation of the loss configuration."""
        return (
            f"scatter_floor={self.scatter_floor}, "
            f"n_parameters={self.n_parameters}, "
            f"reduction='{self.reduction}'"
        )


class MaskedMSELoss(nn.Module):
    """Masked mean squared error on predicted means, for ``loss: mse``.

    Uses the same (batch, 2, n_params) output / (batch, 3, n_params) target
    format as :class:`HeteroscedasticLoss` and ignores label uncertainties and
    the predicted log-scatter channel (which receives no gradient).
    """

    def __init__(self, n_parameters: int = 11, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'"
            )
        self.n_parameters = n_parameters
        self.reduction = reduction

    def element_losses(
        self, output: torch.Tensor, target: torch.Tensor, beta: float = 0.0
    ) -> torch.Tensor:
        """Squared error per (sample, parameter), zero where masked.

        ``beta`` has no effect: the squared error is the negative
        log-likelihood with unit variance, whose β-NLL weight is 1.
        """
        if output.shape[1:] != (2, self.n_parameters) or target.shape[1:] != (
            3,
            self.n_parameters,
        ):
            raise ValueError(
                f"Expected output (batch, 2, {self.n_parameters}) and target "
                f"(batch, 3, {self.n_parameters}), got {output.shape} and {target.shape}"
            )
        y, mask = _safe_targets(output[:, 0, :], target)
        sq = (output[:, 0, :] - y) ** 2
        return torch.where(mask > 0, sq * mask, torch.zeros_like(sq))

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.element_losses(output, target)
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.sum() / target[:, 2, :].sum().clamp(min=1)

    def forward_detailed(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Per-parameter MSE as the mean component (scatter component is 0)."""
        loss = self.element_losses(output, target)
        valid_counts = target[:, 2, :].sum(dim=0).clamp(min=1)
        mean_component = loss.sum(dim=0) / valid_counts
        return {
            "loss": self.forward(output, target),
            "mean_component": mean_component,
            "scatter_component": torch.zeros_like(mean_component),
            "residuals": output[:, 0, :] - target[:, 0, :],
            "y_pred": output[:, 0, :],
        }
