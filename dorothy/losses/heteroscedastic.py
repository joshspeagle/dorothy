"""
Heteroscedastic loss function for stellar parameter inference.

This module implements the uncertainty-aware loss function used by DOROTHY models.
The loss accounts for both label uncertainties (from APOGEE measurements) and
model-predicted uncertainties, enabling the network to learn when it is confident
or uncertain about its predictions.

Mathematical formulation:
    s = sqrt(exp(2 * ln_s) + s_0^2)
    loss = (mu - y)^2 / (sigma_label^2 + s^2) + log(sigma_label^2 + s^2)

Where:
    - mu: predicted stellar parameter value
    - ln_s: predicted log scatter (uncertainty)
    - y: true label value
    - sigma_label: measurement uncertainty from APOGEE
    - s_0: scatter floor to prevent numerical instability
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HeteroscedasticLoss(nn.Module):
    """
    Heteroscedastic loss for regression with predicted uncertainties.

    This loss function combines the measurement uncertainty from the labels
    with the model's predicted uncertainty to weight the squared error.
    It also includes a log-variance term that penalizes overconfidence.

    The model output has shape (batch, 2, n_parameters):
    - output[:, 0, :]: predicted means (mu)
    - output[:, 1, :]: predicted log-scatter (ln_s)

    The target has 3-channel format (batch, 3, n_parameters):
    - target[:, 0, :]: true labels (y)
    - target[:, 1, :]: label uncertainties (sigma_label)
    - target[:, 2, :]: mask (1=valid, 0=masked) - used for loss computation

    Attributes:
        scatter_floor: Minimum scatter value (s_0) to prevent division by zero.
        n_parameters: Number of stellar parameters being predicted.
        reduction: How to reduce the loss ('mean', 'sum', or 'none').

    Example:
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

        # Pre-compute squared scatter floor for efficiency
        self.register_buffer(
            "scatter_floor_sq",
            torch.tensor(scatter_floor**2, dtype=torch.float32),
        )

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the heteroscedastic loss.

        Args:
            output: Model predictions of shape (batch_size, 2, n_parameters).
                output[:, 0, :] contains predicted means (mu).
                output[:, 1, :] contains predicted log-scatter (ln_s).
            target: Target values of shape (batch_size, 3, n_parameters).
                target[:, 0, :] contains true labels (y).
                target[:, 1, :] contains label uncertainties (sigma_label).
                target[:, 2, :] contains mask (1=valid, 0=masked).

        Returns:
            The computed loss, reduced according to self.reduction.

        Raises:
            ValueError: If output or target shapes don't match expectations.
        """
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

        # Extract predictions and log-uncertainties from 3D output
        mu = output[:, 0, :]  # (batch, n_params)
        ln_s = output[:, 1, :]  # (batch, n_params)

        # Extract labels, uncertainties, and mask from 3-channel target
        y = target[:, 0, :]  # (batch, n_params)
        sigma_label = target[:, 1, :]  # (batch, n_params)
        mask = target[:, 2, :]  # (batch, n_params)

        # Compute model's predicted scatter with floor
        # s = sqrt(exp(2 * ln_s) + s_0^2)
        # Using exp(2 * ln_s) = exp(ln_s)^2 for numerical stability
        s_squared = torch.exp(2 * ln_s) + self.scatter_floor_sq

        # Total variance is sum of label variance and model variance
        # variance = sigma_label^2 + s^2
        sigma_label_sq = sigma_label**2
        total_variance = sigma_label_sq + s_squared

        # Compute weighted squared error and log-variance penalty
        # loss = (mu - y)^2 / variance + log(variance)
        squared_error = (mu - y) ** 2
        loss_per_element = squared_error / total_variance + torch.log(total_variance)

        # Apply mask from target channel 2
        loss_per_element = loss_per_element * mask

        # Apply reduction
        if self.reduction == "none":
            return loss_per_element
        elif self.reduction == "sum":
            return loss_per_element.sum()
        else:  # mean
            # Average over valid elements only
            return loss_per_element.sum() / mask.sum().clamp(min=1)

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
        ln_s = output[:, 1, :]  # (batch, n_params)
        s_squared = torch.exp(2 * ln_s) + self.scatter_floor_sq
        return torch.sqrt(s_squared)

    def forward_detailed(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss with detailed per-parameter and per-component breakdown.

        Returns both the total loss and a breakdown into:
        - Mean component per parameter: (mu - y)² / variance
        - Scatter component per parameter: log(variance)

        Args:
            output: Model predictions of shape (batch_size, 2, n_parameters).
            target: Target values of shape (batch_size, 3, n_parameters).
                target[:, 0, :] contains true labels (y).
                target[:, 1, :] contains label uncertainties (sigma_label).
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
        # Extract predictions and log-uncertainties from 3D output
        mu = output[:, 0, :]  # (batch, n_params)
        ln_s = output[:, 1, :]  # (batch, n_params)

        # Extract labels, uncertainties, and mask from 3-channel target
        y = target[:, 0, :]  # (batch, n_params)
        sigma_label = target[:, 1, :]  # (batch, n_params)
        mask = target[:, 2, :]  # (batch, n_params)

        # Compute model's predicted scatter with floor
        s_squared = torch.exp(2 * ln_s) + self.scatter_floor_sq
        pred_scatter = torch.sqrt(s_squared)

        # Total variance
        sigma_label_sq = sigma_label**2
        total_variance = sigma_label_sq + s_squared

        # Compute the two components separately
        residuals = mu - y
        squared_error = residuals**2
        mean_component = squared_error / total_variance  # (batch, n_params)
        scatter_component = torch.log(total_variance)  # (batch, n_params)

        # Per-element loss
        loss_per_element = mean_component + scatter_component

        # Apply mask from target channel 2
        loss_per_element = loss_per_element * mask
        # Average over batch for each parameter, counting only valid samples
        valid_counts = mask.sum(dim=0).clamp(min=1)  # (n_params,)
        mean_component_avg = (mean_component * mask).sum(dim=0) / valid_counts
        scatter_component_avg = (scatter_component * mask).sum(dim=0) / valid_counts

        # Total loss
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
