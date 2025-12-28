"""
Gradient-based saliency analysis for stellar parameter predictions.

This module provides tools for computing and visualizing how sensitive
model predictions are to changes in the input spectrum. It uses the
Jacobian of the model outputs with respect to inputs and aggregates
gradients using Fisher information weighting.

Example:
    >>> from dorothy.inference import Predictor
    >>> from dorothy.analysis import SaliencyAnalyzer, plot_parameter_saliency
    >>>
    >>> predictor = Predictor.load("outputs/variant5/checkpoint")
    >>> analyzer = SaliencyAnalyzer(predictor.model, device="cuda")
    >>>
    >>> result = analyzer.compute_saliency(X, wavelength, survey="boss")
    >>> fig = plot_parameter_saliency(result, "teff", output_path="teff_saliency.png")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from dorothy.data.normalizer import LabelNormalizer


# Default parameter names (same as catalogue_loader)
DEFAULT_PARAMETER_NAMES = [
    "teff",
    "logg",
    "fe_h",
    "mg_fe",
    "c_fe",
    "si_fe",
    "ni_fe",
    "al_fe",
    "ca_fe",
    "n_fe",
    "mn_fe",
]


@dataclass
class SaliencyResult:
    """Container for saliency analysis results.

    Attributes:
        survey: Name of the survey this spectrum is from.
        wavelength: Wavelength grid of shape (N,).
        spectrum: Flux values of shape (N,).
        spectrum_error: Flux error values of shape (N,).
        mask: Boolean mask of shape (N,) where True = valid, False = masked.
        jacobian_mean_flux: Gradient of means w.r.t. flux, shape (n_params, N).
        jacobian_mean_error: Gradient of means w.r.t. error, shape (n_params, N).
        jacobian_lns_flux: Gradient of log_scatter w.r.t. flux, shape (n_params, N).
        jacobian_lns_error: Gradient of log_scatter w.r.t. error, shape (n_params, N).
        predictions: Predicted means of shape (n_params,).
        uncertainties: Predicted uncertainties (sigma) of shape (n_params,).
        fisher_importance_per_param: Per-parameter Fisher importance, shape (n_params, N).
        fisher_importance: Total Fisher importance, shape (N,).
        parameter_names: List of parameter names.
        gaia_id: Optional Gaia DR3 source ID.
        ra: Optional right ascension in degrees.
        dec: Optional declination in degrees.
    """

    survey: str
    wavelength: NDArray[np.float32]
    spectrum: NDArray[np.float32]
    spectrum_error: NDArray[np.float32]
    mask: NDArray[np.bool_]

    # Raw Jacobians
    jacobian_mean_flux: NDArray[np.float32]
    jacobian_mean_error: NDArray[np.float32]
    jacobian_lns_flux: NDArray[np.float32]
    jacobian_lns_error: NDArray[np.float32]

    # Predictions
    predictions: NDArray[np.float32]
    uncertainties: NDArray[np.float32]

    # Fisher importance
    fisher_importance_per_param: NDArray[np.float32]
    fisher_importance: NDArray[np.float32]

    parameter_names: list[str]

    # Optional metadata
    gaia_id: int | None = None
    ra: float | None = None
    dec: float | None = None

    def get_param_index(self, param_name: str) -> int:
        """Get the index of a parameter by name.

        Args:
            param_name: Parameter name (e.g., "teff", "fe_h").

        Returns:
            Index into the parameter arrays.

        Raises:
            ValueError: If parameter name is not found.
        """
        try:
            return self.parameter_names.index(param_name)
        except ValueError:
            raise ValueError(
                f"Unknown parameter '{param_name}'. "
                f"Available: {self.parameter_names}"
            ) from None


class SaliencyAnalyzer:
    """Compute gradient-based saliency maps for stellar parameter predictions.

    This analyzer computes the Jacobian of model outputs with respect to
    input spectra, showing how sensitive predictions are to each wavelength.

    The Fisher information metric is used to aggregate gradients across
    output dimensions, weighting by prediction uncertainty.

    Attributes:
        model: The neural network model.
        device: Device for computation.
        scatter_floor: Floor for uncertainty computation.
        parameter_names: Names of the stellar parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        scatter_floor: float = 0.01,
        parameter_names: list[str] | None = None,
    ) -> None:
        """Initialize the saliency analyzer.

        Args:
            model: Trained MultiHeadMLP or MLP model.
            device: Device for computation ("auto", "cuda", "cpu").
            scatter_floor: Floor for uncertainty computation (default 0.01).
            parameter_names: Names for the stellar parameters.
                Default: standard 11 APOGEE parameters.
        """
        self.model = model
        self.scatter_floor = scatter_floor
        self.parameter_names = parameter_names or DEFAULT_PARAMETER_NAMES.copy()

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def compute_saliency(
        self,
        X: torch.Tensor,
        wavelength: NDArray[np.float32],
        survey: str | None = None,
        gaia_id: int | None = None,
        ra: float | None = None,
        dec: float | None = None,
    ) -> SaliencyResult:
        """Compute saliency map for a single sample.

        Args:
            X: Input spectrum of shape (3, N_wavelengths) - [flux, error, mask].
            wavelength: Wavelength array of shape (N,).
            survey: Survey name (required for MultiHeadMLP).
            gaia_id: Optional Gaia DR3 source ID for metadata.
            ra: Optional right ascension in degrees.
            dec: Optional declination in degrees.

        Returns:
            SaliencyResult with Jacobians and Fisher importance.
        """
        # Ensure X is on device
        X = X.to(self.device)

        # Extract spectrum components (raw, before masking)
        spectrum_raw = X[0].detach().cpu().numpy()
        spectrum_error_raw = X[1].detach().cpu().numpy()
        mask_channel = X[2].detach().cpu().numpy()
        mask = mask_channel > 0.5  # Convert to boolean

        # Apply mask to input exactly as the model sees it: (mask*flux, mask*err, mask)
        # This ensures gradients are computed for the actual input the model receives
        X_masked = X.clone()
        X_masked[0] = X[0] * X[2]  # mask * flux
        X_masked[1] = X[1] * X[2]  # mask * error
        # X_masked[2] is already the mask

        # Store the raw spectrum for visualization (so we can see the actual data)
        spectrum = spectrum_raw
        spectrum_error = spectrum_error_raw

        # Compute Jacobian on the properly masked input
        (
            jac_mean_flux,
            jac_mean_error,
            jac_lns_flux,
            jac_lns_error,
            predictions,
            log_scatter,
        ) = self._compute_jacobian(X_masked, survey)

        # Compute sigma from log_scatter
        sigma = torch.sqrt(torch.exp(2 * log_scatter) + self.scatter_floor**2)

        # Compute Fisher importance
        fisher_per_param, fisher_total = self._compute_fisher_importance(
            jac_mean_flux,
            jac_mean_error,
            jac_lns_flux,
            jac_lns_error,
            sigma,
        )

        # Convert to numpy and apply mask to zero out invalid pixels
        jac_mean_flux_np = jac_mean_flux.cpu().numpy()
        jac_mean_error_np = jac_mean_error.cpu().numpy()
        jac_lns_flux_np = jac_lns_flux.cpu().numpy()
        jac_lns_error_np = jac_lns_error.cpu().numpy()
        fisher_per_param_np = fisher_per_param.cpu().numpy()
        fisher_total_np = fisher_total.cpu().numpy()

        # Zero out gradients for masked (invalid) pixels
        # This ensures bad pixels don't show spurious saliency values
        jac_mean_flux_np[:, ~mask] = 0.0
        jac_mean_error_np[:, ~mask] = 0.0
        jac_lns_flux_np[:, ~mask] = 0.0
        jac_lns_error_np[:, ~mask] = 0.0
        fisher_per_param_np[:, ~mask] = 0.0
        fisher_total_np[~mask] = 0.0

        return SaliencyResult(
            survey=survey or "unknown",
            wavelength=wavelength,
            spectrum=spectrum,
            spectrum_error=spectrum_error,
            mask=mask,
            jacobian_mean_flux=jac_mean_flux_np,
            jacobian_mean_error=jac_mean_error_np,
            jacobian_lns_flux=jac_lns_flux_np,
            jacobian_lns_error=jac_lns_error_np,
            predictions=predictions.cpu().numpy(),
            uncertainties=sigma.cpu().numpy(),
            fisher_importance_per_param=fisher_per_param_np,
            fisher_importance=fisher_total_np,
            parameter_names=self.parameter_names,
            gaia_id=gaia_id,
            ra=ra,
            dec=dec,
        )

    def _compute_jacobian(
        self,
        X: torch.Tensor,
        survey: str | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute Jacobian using torch.autograd.

        Args:
            X: Input tensor of shape (3, N) - [flux, error, mask].
            survey: Survey name for MultiHeadMLP.

        Returns:
            Tuple of (jac_mean_flux, jac_mean_error, jac_lns_flux, jac_lns_error,
                      predictions, log_scatter).
            Jacobians have shape (n_params, N).
        """
        X = X.clone().detach().requires_grad_(True)

        # Add batch dimension
        X_batch = X.unsqueeze(0)  # (1, 3, N)

        # Forward pass
        if hasattr(self.model, "forward_single") and survey is not None:
            output = self.model.forward_single(X_batch, survey)
        else:
            output = self.model(X_batch)

        # output shape: (1, 2, n_params)
        means = output[0, 0, :]  # (n_params,)
        log_scatter = output[0, 1, :]  # (n_params,)

        n_params = means.shape[0]
        N = X.shape[1]  # Number of wavelengths

        # Initialize Jacobian tensors
        jac_mean_flux = torch.zeros(n_params, N, device=self.device)
        jac_mean_error = torch.zeros(n_params, N, device=self.device)
        jac_lns_flux = torch.zeros(n_params, N, device=self.device)
        jac_lns_error = torch.zeros(n_params, N, device=self.device)

        # Compute gradients for each output parameter
        for i in range(n_params):
            # Gradient of mean w.r.t. input
            if X.grad is not None:
                X.grad.zero_()
            grad_mean = torch.autograd.grad(
                means[i], X, retain_graph=True, create_graph=False
            )[
                0
            ]  # (3, N)
            jac_mean_flux[i] = grad_mean[0, :]
            jac_mean_error[i] = grad_mean[1, :]

            # Gradient of log_scatter w.r.t. input
            grad_lns = torch.autograd.grad(
                log_scatter[i], X, retain_graph=True, create_graph=False
            )[
                0
            ]  # (3, N)
            jac_lns_flux[i] = grad_lns[0, :]
            jac_lns_error[i] = grad_lns[1, :]

        return (
            jac_mean_flux,
            jac_mean_error,
            jac_lns_flux,
            jac_lns_error,
            means.detach(),
            log_scatter.detach(),
        )

    def _compute_fisher_importance(
        self,
        jac_mean_flux: torch.Tensor,
        jac_mean_error: torch.Tensor,
        jac_lns_flux: torch.Tensor,
        jac_lns_error: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate Jacobians into Fisher-weighted importance.

        The Fisher information for a Gaussian N(μ, σ²) is:
            I_μμ = 1/σ², I_σσ = 2/σ²

        For each parameter p and wavelength i:
            I_p(i) = (∂μ_p/∂flux_i)²/σ_p² + (∂μ_p/∂err_i)²/σ_p²
                   + 2·(∂σ_p/∂flux_i)²/σ_p² + 2·(∂σ_p/∂err_i)²/σ_p²

        Note: We convert ∂ln_s/∂x to ∂σ/∂x using chain rule:
            ∂σ/∂x = σ · ∂ln_s/∂x (for the exponential part)

        Args:
            jac_mean_flux: Gradient of means w.r.t. flux, shape (n_params, N).
            jac_mean_error: Gradient of means w.r.t. error, shape (n_params, N).
            jac_lns_flux: Gradient of log_scatter w.r.t. flux, shape (n_params, N).
            jac_lns_error: Gradient of log_scatter w.r.t. error, shape (n_params, N).
            sigma: Predicted uncertainties, shape (n_params,).

        Returns:
            Tuple of (per_param_importance, total_importance).
            - per_param_importance: (n_params, N) importance per parameter
            - total_importance: (N,) sum over all parameters
        """
        # sigma shape: (n_params,) -> (n_params, 1) for broadcasting
        sigma_sq = (sigma**2).unsqueeze(1)  # (n_params, 1)

        # Mean gradients contribution: (∂μ/∂x)² / σ²
        mean_contrib = (jac_mean_flux**2 + jac_mean_error**2) / sigma_sq

        # For scatter gradients, we use the Fisher metric for σ: 2/σ²
        # The gradient ∂ln_s/∂x relates to ∂σ/∂x ≈ σ · ∂ln_s/∂x
        # So (∂σ/∂x)² / σ² ≈ (∂ln_s/∂x)²
        # And the Fisher weight is 2/σ², so contribution is 2·(∂ln_s/∂x)²
        scatter_contrib = 2 * (jac_lns_flux**2 + jac_lns_error**2)

        # Per-parameter importance
        per_param_importance = mean_contrib + scatter_contrib

        # Total importance (sum over parameters)
        total_importance = per_param_importance.sum(dim=0)

        return per_param_importance, total_importance


def plot_parameter_saliency(
    result: SaliencyResult,
    param_name: str,
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (14, 14),
    cmap: str = "RdBu",
    cmap_fisher: str = "Purples",
    normalizer: LabelNormalizer | None = None,
    label_values: NDArray[np.float32] | None = None,
    label_errors: NDArray[np.float32] | None = None,
) -> Figure:
    """Plot saliency map for a single stellar parameter.

    Creates a 3-panel figure (vertically stacked), each panel containing:
    - Main subplot: flux spectrum colored by gradient
    - Smaller subplot below: error spectrum colored by gradient
    - Masked wavelengths are grayed out

    Panels:
    1. Mean sensitivity (∂μ/∂x)
    2. Scatter sensitivity (∂σ/∂x) - converted from log-scatter using chain rule
    3. Fisher importance for this parameter

    Args:
        result: SaliencyResult from compute_saliency().
        param_name: Which parameter to visualize (e.g., "teff", "fe_h").
        output_path: If provided, save figure to this path.
        figsize: Figure size (width, height).
        cmap: Colormap for gradient panels (diverging, red=negative, blue=positive).
        cmap_fisher: Colormap for Fisher importance (sequential).
        normalizer: Optional LabelNormalizer to show denormalized predictions.
        label_values: Optional ground truth label values (physical units), shape (n_params,).
        label_errors: Optional ground truth label errors (physical units), shape (n_params,).

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec

    param_idx = result.get_param_index(param_name)

    # Get data
    wavelength = result.wavelength
    flux = result.spectrum
    error = result.spectrum_error
    mask = result.mask
    sigma = result.uncertainties[param_idx]

    # Get gradients for this parameter
    grad_mean_flux = result.jacobian_mean_flux[param_idx]
    grad_mean_error = result.jacobian_mean_error[param_idx]
    # Convert dln_s to dσ using chain rule: ∂σ/∂x = σ · ∂ln_s/∂x
    grad_sigma_flux = sigma * result.jacobian_lns_flux[param_idx]
    grad_sigma_error = sigma * result.jacobian_lns_error[param_idx]
    fisher_importance = result.fisher_importance_per_param[param_idx]

    # Create figure with nested GridSpec for proper spacing
    # Outer grid: 3 panels with space between them
    # Inner grids: flux (3/4) and error (1/4) tightly coupled
    fig = plt.figure(figsize=figsize)
    outer_gs = GridSpec(
        3,
        1,
        height_ratios=[1, 1, 1],
        hspace=0.25,
        figure=fig,
        left=0.08,
        right=0.88,
        top=0.90,
        bottom=0.06,
    )

    # Panel data: (title, flux_gradient, error_gradient, is_fisher, cbar_label)
    panel_data = [
        (
            "Mean Sensitivity ∂μ/∂x",
            grad_mean_flux,
            grad_mean_error,
            False,
            "Sensitivity (normalized)",
        ),
        (
            "Scatter Sensitivity ∂σ/∂x",
            grad_sigma_flux,
            grad_sigma_error,
            False,
            "Sensitivity (normalized)",
        ),
        (
            "Fisher Importance",
            fisher_importance,
            fisher_importance,
            True,
            "Information (normalized)",
        ),
    ]

    # Store axes for shared x-axis
    all_axes = []
    colorbars = []

    for panel_idx, (title, flux_grad, error_grad, is_fisher, cbar_label) in enumerate(
        panel_data
    ):
        # Create inner GridSpec for this panel (flux 3/4, error 1/4, tightly coupled)
        inner_gs = outer_gs[panel_idx].subgridspec(
            2, 1, height_ratios=[3, 1], hspace=0.05
        )
        ax_flux = fig.add_subplot(inner_gs[0])
        ax_error = fig.add_subplot(inner_gs[1], sharex=ax_flux)
        all_axes.extend([ax_flux, ax_error])

        # Choose colormap
        panel_cmap = cmap_fisher if is_fisher else cmap

        # Determine color normalization
        if is_fisher:
            # Fisher importance - non-negative
            norm = Normalize(vmin=0, vmax=np.percentile(fisher_importance[mask], 99))
        else:
            # Gradient panels - symmetric around zero
            max_val = max(
                np.percentile(np.abs(flux_grad[mask]), 99),
                np.percentile(np.abs(error_grad[mask]), 99),
            )
            norm = Normalize(vmin=-max_val, vmax=max_val)

        # --- Flux subplot ---
        # Create colored line collection
        points = np.array([wavelength, flux]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = flux_grad[:-1]

        lc_flux = LineCollection(
            segments, cmap=panel_cmap, norm=norm, linewidth=1.5, alpha=0.9
        )
        lc_flux.set_array(colors)
        ax_flux.add_collection(lc_flux)

        # Overplot thin black trace for visibility
        ax_flux.plot(wavelength, flux, "k-", linewidth=0.3, alpha=0.5)

        # Set axis limits
        flux_valid = flux[mask]
        ax_flux.set_ylim(flux_valid.min() - 0.1, flux_valid.max() + 0.1)
        ax_flux.set_xlim(wavelength.min(), wavelength.max())
        ax_flux.set_ylabel("Flux", fontsize=9)
        ax_flux.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=6)

        # Hide x-axis labels for flux (shared with error below)
        plt.setp(ax_flux.get_xticklabels(), visible=False)
        ax_flux.tick_params(axis="x", length=0)

        # --- Error subplot ---
        points_err = np.array([wavelength, error]).T.reshape(-1, 1, 2)
        segments_err = np.concatenate([points_err[:-1], points_err[1:]], axis=1)
        colors_err = error_grad[:-1]

        lc_error = LineCollection(
            segments_err, cmap=panel_cmap, norm=norm, linewidth=1.2, alpha=0.9
        )
        lc_error.set_array(colors_err)
        ax_error.add_collection(lc_error)

        # Overplot thin black trace
        ax_error.plot(wavelength, error, "k-", linewidth=0.3, alpha=0.5)

        # Set error axis limits
        error_valid = error[mask]
        if error_valid.max() > 0:
            ax_error.set_ylim(0, error_valid.max() * 1.2)
        else:
            ax_error.set_ylim(0, 1)
        ax_error.set_ylabel("Err", fontsize=8)
        ax_error.tick_params(axis="y", labelsize=7)

        # Gray out masked regions on both subplots
        masked_regions = _find_masked_regions(mask)
        for start, end in masked_regions:
            wl_start = wavelength[start]
            wl_end = wavelength[min(end, len(wavelength) - 1)]
            ax_flux.axvspan(wl_start, wl_end, alpha=0.4, color="gray", zorder=10)
            ax_error.axvspan(wl_start, wl_end, alpha=0.4, color="gray", zorder=10)

        # Add colorbar in dedicated space on right
        cbar_ax = fig.add_axes(
            [
                0.90,
                ax_error.get_position().y0,
                0.02,
                ax_flux.get_position().y1 - ax_error.get_position().y0,
            ]
        )
        cbar = fig.colorbar(lc_flux, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(cbar_label, fontsize=8)
        colorbars.append(cbar)

    # X-axis label on bottom
    all_axes[-1].set_xlabel("Wavelength (Å)", fontsize=11)

    # Build title: line 1 = survey + Gaia ID, line 2 = truth (Pred: prediction)
    # Format: PARAM = truth ± err (Pred: pred ± pred_err)

    # Get prediction values (denormalized if normalizer provided)
    if normalizer is not None:
        pred_denorm, unc_denorm = normalizer.inverse_transform(
            result.predictions.reshape(1, -1),
            result.uncertainties.reshape(1, -1),
        )
        pred_val = pred_denorm[0, param_idx]
        pred_unc = unc_denorm[0, param_idx]
    else:
        pred_val = result.predictions[param_idx]
        pred_unc = result.uncertainties[param_idx]

    # Format prediction string
    if param_name == "teff":
        pred_str = f"{pred_val:.0f} ± {pred_unc:.0f} K"
    else:
        pred_str = f"{pred_val:.3f} ± {pred_unc:.3f}"

    # Build parameter string with truth and prediction
    if label_values is not None:
        # Format truth string
        truth_val = label_values[param_idx]
        truth_err = label_errors[param_idx] if label_errors is not None else 0.0
        if param_name == "teff":
            truth_str = f"{truth_val:.0f} ± {truth_err:.0f} K"
        else:
            truth_str = f"{truth_val:.3f} ± {truth_err:.3f}"
        param_str = f"{param_name.upper()} = {truth_str}  (Pred: {pred_str})"
    else:
        # No truth available, just show prediction
        param_str = f"{param_name.upper()} = {pred_str}"
        if normalizer is None:
            param_str += " (normalized)"

    # Build two-line title
    title_line1 = f"Saliency Map ({result.survey})"
    if result.gaia_id is not None:
        title_line1 += f"    Gaia DR3 {result.gaia_id}"

    fig.suptitle(
        f"{title_line1}\n{param_str}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_saliency_heatmap(
    result: SaliencyResult,
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (16, 14),
    cmap_gradient: str = "RdBu",
    cmap_fisher: str = "Purples",
) -> Figure:
    """Plot heatmap of gradients: wavelength × parameter.

    Creates a 3-panel figure:
    1. Mean sensitivity heatmap (∂μ/∂x) - normalized per component
    2. Scatter sensitivity heatmap (∂σ/∂x) - normalized per component
    3. Fisher importance heatmap - sequential colormap

    Each heatmap has:
    - Y-axis: parameters × 2 rows (flux gradient row, error gradient row)
    - X-axis: wavelength
    - Masked wavelengths are blocked out (grayed columns)
    - Horizontal black lines between each parameter group

    For gradient panels, each component (2 rows: flux + error) is normalized
    to [-1, 1] to show relative importance within each parameter.

    Args:
        result: SaliencyResult from compute_saliency().
        output_path: If provided, save figure to this path.
        figsize: Figure size (width, height).
        cmap_gradient: Colormap for gradient panels (diverging, red=negative, blue=positive).
        cmap_fisher: Colormap for Fisher importance (sequential).

    Returns:
        Matplotlib Figure with 3 heatmap panels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    n_params = len(result.parameter_names)
    n_wavelengths = len(result.wavelength)

    # Build combined arrays with interleaved flux/error rows
    # Shape: (n_params * 2, n_wavelengths)
    mean_combined = np.zeros((n_params * 2, n_wavelengths), dtype=np.float32)
    scatter_combined = np.zeros((n_params * 2, n_wavelengths), dtype=np.float32)
    fisher_combined = np.zeros((n_params * 2, n_wavelengths), dtype=np.float32)

    for i in range(n_params):
        mean_combined[2 * i] = result.jacobian_mean_flux[i]
        mean_combined[2 * i + 1] = result.jacobian_mean_error[i]
        # Convert dln_s to dσ using chain rule: ∂σ/∂x = σ · ∂ln_s/∂x
        sigma = result.uncertainties[i]
        scatter_combined[2 * i] = sigma * result.jacobian_lns_flux[i]
        scatter_combined[2 * i + 1] = sigma * result.jacobian_lns_error[i]
        # For Fisher, compute separate flux/error contributions
        fisher_flux = (
            result.jacobian_mean_flux[i] ** 2 / sigma**2
            + 2 * result.jacobian_lns_flux[i] ** 2
        )
        fisher_error = (
            result.jacobian_mean_error[i] ** 2 / sigma**2
            + 2 * result.jacobian_lns_error[i] ** 2
        )
        fisher_combined[2 * i] = fisher_flux
        fisher_combined[2 * i + 1] = fisher_error

    # Normalize gradient arrays per-component (2 rows at a time) to [-1, 1]
    def normalize_per_component(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Normalize each pair of rows (flux, error) to [-1, 1]."""
        normalized = np.zeros_like(data)
        for i in range(n_params):
            # Get both rows for this parameter
            rows = data[2 * i : 2 * i + 2, :]
            # Compute max abs value over valid (masked) wavelengths
            valid_data = rows[:, mask]
            max_val = np.abs(valid_data).max()
            if max_val > 0:
                normalized[2 * i : 2 * i + 2, :] = rows / max_val
            else:
                normalized[2 * i : 2 * i + 2, :] = rows
        return normalized

    mean_normalized = normalize_per_component(mean_combined, result.mask)
    scatter_normalized = normalize_per_component(scatter_combined, result.mask)

    # For Fisher, normalize per-component to [0, 1]
    fisher_normalized = np.zeros_like(fisher_combined)
    for i in range(n_params):
        rows = fisher_combined[2 * i : 2 * i + 2, :]
        valid_data = rows[:, result.mask]
        max_val = valid_data.max()
        if max_val > 0:
            fisher_normalized[2 * i : 2 * i + 2, :] = rows / max_val

    # Create row labels
    row_labels = []
    for name in result.parameter_names:
        row_labels.append(f"{name} (flux)")
        row_labels.append(f"{name} (err)")

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Panel data: (title, data, cmap, vmin, vmax)
    panel_data = [
        ("Mean Sensitivity ∂μ/∂x (relative)", mean_normalized, cmap_gradient, -1, 1),
        (
            "Scatter Sensitivity ∂σ/∂x (relative)",
            scatter_normalized,
            cmap_gradient,
            -1,
            1,
        ),
        ("Fisher Importance (relative)", fisher_normalized, cmap_fisher, 0, 1),
    ]

    for ax, (title, data, cmap, vmin, vmax) in zip(axes, panel_data, strict=False):
        # Apply mask - set masked columns to NaN for visualization
        data_masked = data.copy()
        data_masked[:, ~result.mask] = np.nan

        norm = Normalize(vmin=vmin, vmax=vmax)

        # Plot heatmap
        im = ax.imshow(
            data_masked,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=[
                result.wavelength.min(),
                result.wavelength.max(),
                n_params * 2 - 0.5,
                -0.5,
            ],
            interpolation="nearest",
        )

        # Add horizontal black lines between each parameter group (every 2 rows)
        for i in range(1, n_params):
            ax.axhline(y=2 * i - 0.5, color="black", linewidth=0.8, zorder=5)

        # Add masked region overlays
        for start, end in _find_masked_regions(result.mask):
            ax.axvspan(
                result.wavelength[start],
                result.wavelength[min(end, n_wavelengths - 1)],
                alpha=0.5,
                color="gray",
                zorder=10,
            )

        # Y-axis labels
        ax.set_yticks(np.arange(n_params * 2))
        ax.set_yticklabels(row_labels, fontsize=7)

        # Title
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Relative importance", fontsize=8)

    # X-axis label on bottom plot
    axes[-1].set_xlabel("Wavelength (Å)", fontsize=11)

    # Overall title with metadata if available
    title_line = f"Saliency Heatmaps ({result.survey})"
    if result.gaia_id is not None:
        title_line += f"    Gaia DR3 {result.gaia_id}"

    fig.suptitle(
        title_line,
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def _find_masked_regions(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Find contiguous masked (False) regions in a boolean mask.

    Args:
        mask: Boolean mask where True = valid, False = masked.

    Returns:
        List of (start, end) index tuples for masked regions.
    """
    regions = []
    in_masked = False
    start = 0

    for i, valid in enumerate(mask):
        if not valid and not in_masked:
            # Start of masked region
            start = i
            in_masked = True
        elif valid and in_masked:
            # End of masked region
            regions.append((start, i))
            in_masked = False

    # Handle trailing masked region
    if in_masked:
        regions.append((start, len(mask)))

    return regions
