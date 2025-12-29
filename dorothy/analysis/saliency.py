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


@dataclass
class AblationSaliencyResult:
    """Container for ablation-based saliency analysis results.

    Ablation saliency computes importance by masking contiguous blocks
    of the spectrum and measuring the change in predictions. Unlike
    gradient-based saliency, ablation affects both flux and error
    channels identically.

    Sign convention (matches gradient saliency): delta_mu and delta_sigma are SIGNED.
    - Positive delta_mu = ablating INCREASED the prediction (region was suppressing it)
    - Negative delta_mu = ablating DECREASED the prediction (region was boosting it)
    This matches gradient saliency where positive gradient means the input
    positively correlates with the output.

    Attributes:
        survey: Name of the survey this spectrum is from.
        wavelength: Wavelength grid of shape (N,).
        spectrum: Flux values of shape (N,).
        spectrum_error: Flux error values of shape (N,).
        mask: Boolean mask of shape (N,) where True = valid, False = masked.
        block_size: Block size(s) used for ablation.
        delta_mu: Per-parameter signed change in mean (μ_ablated - μ₀), shape (n_params, N).
            Positive = region suppressed prediction; negative = region boosted it.
        delta_mu_total: Sum of signed changes across parameters, shape (N,).
        delta_sigma: Per-parameter signed change in scatter (σ_ablated - σ₀), shape (n_params, N).
            Positive = region decreased uncertainty; negative = region increased it.
        delta_sigma_total: Sum of signed scatter changes across parameters, shape (N,).
        fisher_weighted: Per-parameter Fisher metric (unsigned), shape (n_params, N).
        fisher_weighted_total: Total Fisher metric, shape (N,).
        predictions: Baseline predicted means of shape (n_params,).
        uncertainties: Baseline predicted uncertainties (sigma) of shape (n_params,).
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

    # Block size used
    block_size: int | list[int]

    # Delta-mu metrics: mu_baseline - mu_ablated (signed)
    delta_mu: NDArray[np.float32]
    delta_mu_total: NDArray[np.float32]

    # Delta-sigma metrics: sigma_baseline - sigma_ablated (signed)
    delta_sigma: NDArray[np.float32]
    delta_sigma_total: NDArray[np.float32]

    # Fisher-weighted metrics: (delta_mu)^2/sigma^2 + 2*(delta_ln_s)^2 (unsigned)
    fisher_weighted: NDArray[np.float32]
    fisher_weighted_total: NDArray[np.float32]

    # Baseline predictions
    predictions: NDArray[np.float32]
    uncertainties: NDArray[np.float32]

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


class AblationSaliencyAnalyzer:
    """Compute ablation-based saliency maps for stellar parameter predictions.

    This analyzer uses sliding window ablation - masking a contiguous block
    and shifting one pixel at a time to compute per-pixel importance scores.

    Two metrics are computed:
    1. Delta-mu: Absolute change in predicted parameters when region is ablated
    2. Fisher-weighted: (delta_mu)^2/sigma^2 + 2*(delta_ln_s)^2

    Attributes:
        model: The neural network model.
        device: Device for computation.
        scatter_floor: Floor for uncertainty computation.
        parameter_names: Names of the stellar parameters.
        batch_size: Number of ablated versions to process at once.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        scatter_floor: float = 0.01,
        parameter_names: list[str] | None = None,
        batch_size: int = 64,
    ) -> None:
        """Initialize the ablation saliency analyzer.

        Args:
            model: Trained MultiHeadMLP or MLP model.
            device: Device for computation ("auto", "cuda", "cpu").
            scatter_floor: Floor for uncertainty computation (default 0.01).
            parameter_names: Names for the stellar parameters.
            batch_size: Number of ablated versions to process at once for GPU efficiency.
        """
        self.model = model
        self.scatter_floor = scatter_floor
        self.parameter_names = parameter_names or DEFAULT_PARAMETER_NAMES.copy()
        self.batch_size = batch_size

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
        block_size: int | None = None,
        use_training_distribution: bool = False,
        n_block_sizes: int = 21,
        f_min: float | None = None,
        f_max: float = 0.5,
        gaia_id: int | None = None,
        ra: float | None = None,
        dec: float | None = None,
    ) -> AblationSaliencyResult:
        """Compute ablation-based saliency map for a single sample.

        Args:
            X: Input spectrum of shape (3, N_wavelengths) - [flux, error, mask].
            wavelength: Wavelength array of shape (N,).
            survey: Survey name (required for MultiHeadMLP).
            block_size: Fixed block size for ablation. If None, uses 50 pixels.
            use_training_distribution: If True, average over block sizes sampled
                from the log-uniform distribution used during training.
            n_block_sizes: Number of block sizes when using training distribution.
            f_min: Minimum block fraction (default: 1/N_wavelengths).
            f_max: Maximum block fraction when using training distribution.
            gaia_id: Optional Gaia DR3 source ID for metadata.
            ra: Optional right ascension in degrees.
            dec: Optional declination in degrees.

        Returns:
            AblationSaliencyResult with per-pixel importance scores.
        """
        # Ensure X is on device
        X = X.to(self.device)
        N = X.shape[1]

        # Extract spectrum components
        spectrum_raw = X[0].detach().cpu().numpy()
        spectrum_error_raw = X[1].detach().cpu().numpy()
        mask_channel = X[2].detach().cpu().numpy()
        natural_mask = mask_channel > 0.5  # Boolean mask

        # Apply mask to input (as model sees it)
        X_masked = X.clone()
        X_masked[0] = X[0] * X[2]
        X_masked[1] = X[1] * X[2]

        # Get baseline predictions
        mu_0, ln_s_0 = self._forward_model(X_masked.unsqueeze(0), survey)
        mu_0 = mu_0.squeeze(0)  # (n_params,)
        ln_s_0 = ln_s_0.squeeze(0)  # (n_params,)
        sigma_0 = torch.sqrt(torch.exp(2 * ln_s_0) + self.scatter_floor**2)

        if use_training_distribution:
            # Compute block sizes evenly spaced in log-space
            f_min_actual = f_min if f_min is not None else (1.0 / N)
            log_f = np.linspace(np.log(f_min_actual), np.log(f_max), n_block_sizes)
            fractions = np.exp(log_f)
            block_sizes_list = np.unique(
                [max(1, int(np.ceil(f * N))) for f in fractions]
            )

            # Weights: log-uniform means P(b) ∝ 1/b
            weights = 1.0 / block_sizes_list.astype(np.float32)
            weights /= weights.sum()

            # Compute importance for each block size and accumulate weighted average
            delta_mu_accum = np.zeros((len(self.parameter_names), N), dtype=np.float32)
            delta_sigma_accum = np.zeros(
                (len(self.parameter_names), N), dtype=np.float32
            )
            fisher_accum = np.zeros((len(self.parameter_names), N), dtype=np.float32)

            for block_sz, weight in zip(block_sizes_list, weights, strict=False):
                delta_mu_single, delta_sigma_single, fisher_single = (
                    self._compute_ablation_importance(
                        X_masked,
                        survey,
                        int(block_sz),
                        mu_0,
                        ln_s_0,
                        sigma_0,
                        natural_mask,
                    )
                )
                delta_mu_accum += weight * delta_mu_single
                delta_sigma_accum += weight * delta_sigma_single
                fisher_accum += weight * fisher_single

            delta_mu_np = delta_mu_accum
            delta_sigma_np = delta_sigma_accum
            fisher_np = fisher_accum
            block_size_used: int | list[int] = block_sizes_list.tolist()
        else:
            # Single block size mode
            if block_size is None:
                block_size = 5
            delta_mu_np, delta_sigma_np, fisher_np = self._compute_ablation_importance(
                X_masked, survey, block_size, mu_0, ln_s_0, sigma_0, natural_mask
            )
            block_size_used = block_size

        # Compute totals (sum across parameters)
        delta_mu_total = delta_mu_np.sum(axis=0)
        delta_sigma_total = delta_sigma_np.sum(axis=0)
        fisher_total = fisher_np.sum(axis=0)

        return AblationSaliencyResult(
            survey=survey or "unknown",
            wavelength=wavelength,
            spectrum=spectrum_raw,
            spectrum_error=spectrum_error_raw,
            mask=natural_mask,
            block_size=block_size_used,
            delta_mu=delta_mu_np,
            delta_mu_total=delta_mu_total,
            delta_sigma=delta_sigma_np,
            delta_sigma_total=delta_sigma_total,
            fisher_weighted=fisher_np,
            fisher_weighted_total=fisher_total,
            predictions=mu_0.cpu().numpy(),
            uncertainties=sigma_0.cpu().numpy(),
            parameter_names=self.parameter_names,
            gaia_id=gaia_id,
            ra=ra,
            dec=dec,
        )

    def _compute_ablation_importance(
        self,
        X: torch.Tensor,
        survey: str | None,
        block_size: int,
        mu_0: torch.Tensor,
        ln_s_0: torch.Tensor,
        sigma_0: torch.Tensor,
        natural_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Compute ablation importance for a single block size.

        Uses sliding window ablation with stride 1.

        Args:
            X: Masked input tensor of shape (3, N).
            survey: Survey name for MultiHeadMLP.
            block_size: Size of ablation block.
            mu_0: Baseline means, shape (n_params,).
            ln_s_0: Baseline log_scatter, shape (n_params,).
            sigma_0: Baseline sigma, shape (n_params,).
            natural_mask: Boolean mask of valid pixels.

        Returns:
            Tuple of (delta_mu, delta_sigma, fisher_weighted), each shape (n_params, N).
        """
        N = X.shape[1]
        n_params = len(self.parameter_names)
        n_positions = N - block_size + 1

        if n_positions <= 0:
            # Block size >= spectrum length, return zeros
            return (
                np.zeros((n_params, N), dtype=np.float32),
                np.zeros((n_params, N), dtype=np.float32),
                np.zeros((n_params, N), dtype=np.float32),
            )

        # Compute metrics for all ablation positions in batches
        all_delta_mu = []  # List of (batch, n_params) tensors
        all_delta_sigma = []  # Absolute change in sigma
        all_delta_ln_s = []  # For Fisher metric computation

        positions = list(range(n_positions))
        for batch_start in range(0, n_positions, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_positions)
            batch_positions = positions[batch_start:batch_end]

            # Create batch of ablated inputs
            X_ablated = self._create_ablated_batch(X, batch_positions, block_size)

            # Forward pass
            mu_ablated, ln_s_ablated = self._forward_model(X_ablated, survey)

            # Compute sigma from log_scatter
            sigma_ablated = torch.sqrt(
                torch.exp(2 * ln_s_ablated) + self.scatter_floor**2
            )

            # Compute deltas (sign matches gradient saliency convention)
            # mu_ablated - mu_0: positive means ablating increased prediction (region suppressed it)
            # This matches gradient saliency where positive = increasing input increases output
            delta_mu = mu_ablated - mu_0.unsqueeze(0)  # (batch, n_params)
            delta_sigma = sigma_ablated - sigma_0.unsqueeze(0)  # (batch, n_params)
            delta_ln_s = ln_s_ablated - ln_s_0.unsqueeze(0)  # (batch, n_params)

            all_delta_mu.append(delta_mu.cpu())
            all_delta_sigma.append(delta_sigma.cpu())
            all_delta_ln_s.append(delta_ln_s.cpu())

        # Concatenate all batches
        all_delta_mu_tensor = torch.cat(all_delta_mu, dim=0)  # (n_positions, n_params)
        all_delta_sigma_tensor = torch.cat(
            all_delta_sigma, dim=0
        )  # (n_positions, n_params)
        all_delta_ln_s_tensor = torch.cat(
            all_delta_ln_s, dim=0
        )  # (n_positions, n_params)

        # Compute Fisher-weighted metric per position
        sigma_0_cpu = sigma_0.cpu()
        fisher_per_pos = (
            all_delta_mu_tensor**2 / (sigma_0_cpu**2).unsqueeze(0)
            + 2 * all_delta_ln_s_tensor**2
        )  # (n_positions, n_params)

        # Aggregate to per-pixel importance
        # Each pixel i is covered by windows starting at max(0, i-B+1) to min(i, n_positions-1)
        delta_mu_np = self._aggregate_to_pixels(
            all_delta_mu_tensor.numpy().T, block_size, N, natural_mask
        )
        delta_sigma_np = self._aggregate_to_pixels(
            all_delta_sigma_tensor.numpy().T, block_size, N, natural_mask
        )
        fisher_np = self._aggregate_to_pixels(
            fisher_per_pos.numpy().T, block_size, N, natural_mask
        )

        return delta_mu_np, delta_sigma_np, fisher_np

    def _forward_model(
        self,
        X: torch.Tensor,
        survey: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning means and log_scatter.

        Args:
            X: Input tensor of shape (batch, 3, N).
            survey: Survey name for MultiHeadMLP.

        Returns:
            Tuple of (means, log_scatter), each shape (batch, n_params).
        """
        with torch.no_grad():
            if hasattr(self.model, "forward_single") and survey is not None:
                output = self.model.forward_single(X, survey)
            else:
                output = self.model(X)

        # output shape: (batch, 2, n_params)
        means = output[:, 0, :]  # (batch, n_params)
        log_scatter = output[:, 1, :]  # (batch, n_params)
        return means, log_scatter

    def _create_ablated_batch(
        self,
        X: torch.Tensor,
        positions: list[int],
        block_size: int,
    ) -> torch.Tensor:
        """Create batch of ablated inputs for given positions.

        Args:
            X: Original input of shape (3, N).
            positions: List of ablation start positions.
            block_size: Size of ablation block.

        Returns:
            Tensor of shape (len(positions), 3, N) with ablations applied.
        """
        batch_size = len(positions)

        # Replicate X for each position
        X_batch = X.unsqueeze(0).expand(batch_size, -1, -1).clone()

        # Apply ablation for each position
        for i, pos in enumerate(positions):
            end_pos = pos + block_size
            # Zero out flux, error, and mask in the ablation window
            X_batch[i, 0, pos:end_pos] = 0.0  # flux
            X_batch[i, 1, pos:end_pos] = 0.0  # error
            X_batch[i, 2, pos:end_pos] = 0.0  # mask

        return X_batch

    def _aggregate_to_pixels(
        self,
        per_position_metrics: NDArray[np.float32],
        block_size: int,
        n_wavelengths: int,
        natural_mask: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        """Aggregate per-position metrics to per-pixel importance.

        Each pixel may be covered by multiple ablation windows. This aggregates
        the metrics using mean across all windows that cover each pixel.

        Args:
            per_position_metrics: Metrics per position, shape (n_params, n_positions).
            block_size: Size of ablation block.
            n_wavelengths: Total number of wavelengths.
            natural_mask: Boolean mask of valid pixels.

        Returns:
            Per-pixel importance, shape (n_params, n_wavelengths).
        """
        n_params = per_position_metrics.shape[0]
        n_positions = per_position_metrics.shape[1]

        importance = np.zeros((n_params, n_wavelengths), dtype=np.float32)

        for i in range(n_wavelengths):
            if not natural_mask[i]:
                # Already masked pixel - zero importance
                continue

            # Windows that cover this pixel start at positions max(0, i-B+1) to min(i, n_pos-1)
            start_pos = max(0, i - block_size + 1)
            end_pos = min(i + 1, n_positions)

            if end_pos > start_pos:
                # Average the metrics from all covering windows
                importance[:, i] = per_position_metrics[:, start_pos:end_pos].mean(
                    axis=1
                )

        return importance


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

    # Compute wavelength tick positions and labels
    # Use np.interp to map between pixel indices and wavelengths
    wavelength = result.wavelength
    pixel_indices = np.arange(len(wavelength))
    wl_min, wl_max = wavelength.min(), wavelength.max()

    # Choose nice round tick values (multiples of 500 Å)
    tick_spacing = 500
    first_tick = int(np.ceil(wl_min / tick_spacing) * tick_spacing)
    last_tick = int(np.floor(wl_max / tick_spacing) * tick_spacing)
    tick_wavelengths = np.arange(first_tick, last_tick + 1, tick_spacing)

    # Use np.interp to find pixel positions for these wavelengths
    tick_positions = np.interp(tick_wavelengths, wavelength, pixel_indices)
    tick_labels = [f"{int(wl)}" for wl in tick_wavelengths]

    for ax, (title, data, cmap, vmin, vmax) in zip(axes, panel_data, strict=False):
        # Apply mask - set masked columns to NaN for visualization
        data_masked = data.copy()
        data_masked[:, ~result.mask] = np.nan

        norm = Normalize(vmin=vmin, vmax=vmax)

        # Plot heatmap using pixel indices (no extent - use pixel coordinates)
        im = ax.imshow(
            data_masked,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )

        # Set y-axis limits to match previous behavior
        ax.set_ylim(n_params * 2 - 0.5, -0.5)

        # Set x-axis ticks using actual wavelength values
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(0, n_wavelengths - 1)

        # Add horizontal black lines between each parameter group (every 2 rows)
        for i in range(1, n_params):
            ax.axhline(y=2 * i - 0.5, color="black", linewidth=0.8, zorder=5)

        # Add masked region overlays (using pixel coordinates)
        for start, end in _find_masked_regions(result.mask):
            ax.axvspan(
                start,
                min(end, n_wavelengths - 1),
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


def plot_ablation_parameter_saliency(
    result: AblationSaliencyResult,
    param_name: str,
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (14, 14),
    cmap: str = "RdBu",
    cmap_fisher: str = "Purples",
    normalizer: LabelNormalizer | None = None,
    label_values: NDArray[np.float32] | None = None,
    label_errors: NDArray[np.float32] | None = None,
) -> Figure:
    """Plot ablation-based saliency map for a single stellar parameter.

    Creates a 3-panel figure matching the gradient saliency layout. Since ablation
    affects both flux and error channels identically, both spectrum lines within
    each panel show the same importance coloring.

    Panels (matching gradient saliency structure):
    1. Mean Sensitivity Δμ - signed change in mean prediction (positive = boosted)
    2. Scatter Sensitivity Δσ - signed change in scatter prediction
    3. Fisher Importance - uncertainty-weighted metric for this parameter

    Args:
        result: AblationSaliencyResult from compute_saliency().
        param_name: Which parameter to visualize (e.g., "teff", "fe_h").
        output_path: If provided, save figure to this path.
        figsize: Figure size (width, height).
        cmap: Colormap for signed panels (diverging, default "RdBu").
        cmap_fisher: Colormap for Fisher importance (sequential, default "Purples").
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

    # Get importance metrics for this parameter
    delta_mu_param = result.delta_mu[param_idx]
    delta_sigma_param = result.delta_sigma[param_idx]
    fisher_param = result.fisher_weighted[param_idx]

    # Create figure with nested GridSpec for proper spacing
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

    # Panel data matching gradient saliency structure:
    # (title, flux_importance, error_importance, is_fisher, cbar_label)
    # For ablation, flux and error importance are identical (masked together)
    panel_data = [
        (
            "Mean Sensitivity Δμ",
            delta_mu_param,
            delta_mu_param,
            False,
            "Sensitivity (normalized)",
        ),
        (
            "Scatter Sensitivity Δσ",
            delta_sigma_param,
            delta_sigma_param,
            False,
            "Sensitivity (normalized)",
        ),
        (
            "Fisher Importance",
            fisher_param,
            fisher_param,
            True,
            "Information (normalized)",
        ),
    ]

    # Store axes for shared x-axis
    all_axes = []
    colorbars = []

    for panel_idx, (title, flux_imp, error_imp, is_fisher, cbar_label) in enumerate(
        panel_data
    ):
        # Create inner GridSpec for this panel (flux 3/4, error 1/4, tightly coupled)
        inner_gs = outer_gs[panel_idx].subgridspec(
            2, 1, height_ratios=[3, 1], hspace=0.05
        )
        ax_flux = fig.add_subplot(inner_gs[0])
        ax_error = fig.add_subplot(inner_gs[1], sharex=ax_flux)
        all_axes.extend([ax_flux, ax_error])

        # Choose colormap based on panel type
        panel_cmap = cmap_fisher if is_fisher else cmap

        # Determine normalization based on panel type
        valid_vals = flux_imp[mask]
        if is_fisher:
            # Fisher importance is non-negative
            if len(valid_vals) > 0 and valid_vals.max() > 0:
                norm = Normalize(vmin=0, vmax=np.percentile(valid_vals, 99))
            else:
                norm = Normalize(vmin=0, vmax=1)
        else:
            # Signed values - symmetric around zero
            if len(valid_vals) > 0:
                max_abs = np.percentile(np.abs(valid_vals), 99)
                if max_abs > 0:
                    norm = Normalize(vmin=-max_abs, vmax=max_abs)
                else:
                    norm = Normalize(vmin=-1, vmax=1)
            else:
                norm = Normalize(vmin=-1, vmax=1)

        # --- Flux subplot ---
        points = np.array([wavelength, flux]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = flux_imp[:-1]

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
        colors_err = error_imp[:-1]

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
        truth_val = label_values[param_idx]
        truth_err = label_errors[param_idx] if label_errors is not None else 0.0
        if param_name == "teff":
            truth_str = f"{truth_val:.0f} ± {truth_err:.0f} K"
        else:
            truth_str = f"{truth_val:.3f} ± {truth_err:.3f}"
        param_str = f"{param_name.upper()} = {truth_str}  (Pred: {pred_str})"
    else:
        param_str = f"{param_name.upper()} = {pred_str}"
        if normalizer is None:
            param_str += " (normalized)"

    # Build two-line title with block size info
    block_info = (
        f"block={result.block_size}"
        if isinstance(result.block_size, int)
        else f"blocks={len(result.block_size)}"
    )
    title_line1 = f"Ablation Saliency Map ({result.survey}, {block_info})"
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


def plot_ablation_saliency_heatmap(
    result: AblationSaliencyResult,
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (16, 14),
    cmap_signed: str = "RdBu",
    cmap_fisher: str = "Purples",
) -> Figure:
    """Plot heatmap of ablation importance: wavelength × parameter.

    Creates a 3-panel figure matching the gradient saliency heatmap structure:
    1. Mean sensitivity heatmap Δμ - signed, normalized per parameter
    2. Scatter sensitivity heatmap Δσ - signed, normalized per parameter
    3. Fisher importance heatmap - non-negative, normalized per parameter

    Each heatmap has:
    - Y-axis: parameters × 2 rows (flux row, error row) - identical since ablation affects both
    - X-axis: wavelength
    - Masked wavelengths are blocked out (grayed columns)
    - Horizontal black lines between each parameter group

    Note: Unlike gradient saliency, ablation affects flux and error identically,
    so both rows within each parameter show the same importance values.

    Args:
        result: AblationSaliencyResult from compute_saliency().
        output_path: If provided, save figure to this path.
        figsize: Figure size (width, height).
        cmap_signed: Colormap for signed panels (diverging, default "RdBu").
        cmap_fisher: Colormap for Fisher importance (sequential, default "Purples").

    Returns:
        Matplotlib Figure with 3 heatmap panels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    n_params = len(result.parameter_names)
    n_wavelengths = len(result.wavelength)

    # Build combined arrays with interleaved flux/error rows
    # Shape: (n_params * 2, n_wavelengths)
    # For ablation, flux and error rows are identical
    delta_mu_combined = np.zeros((n_params * 2, n_wavelengths), dtype=np.float32)
    delta_sigma_combined = np.zeros((n_params * 2, n_wavelengths), dtype=np.float32)
    fisher_combined = np.zeros((n_params * 2, n_wavelengths), dtype=np.float32)

    for i in range(n_params):
        # Both flux and error rows get the same importance (ablation affects both)
        delta_mu_combined[2 * i] = result.delta_mu[i]
        delta_mu_combined[2 * i + 1] = result.delta_mu[i]
        delta_sigma_combined[2 * i] = result.delta_sigma[i]
        delta_sigma_combined[2 * i + 1] = result.delta_sigma[i]
        fisher_combined[2 * i] = result.fisher_weighted[i]
        fisher_combined[2 * i + 1] = result.fisher_weighted[i]

    # Normalize signed arrays per-component to [-1, 1] (symmetric around zero)
    def normalize_signed_per_component(
        data: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Normalize each pair of rows (flux, error) to [-1, 1]."""
        normalized = np.zeros_like(data)
        for i in range(n_params):
            # Get both rows for this parameter
            rows = data[2 * i : 2 * i + 2, :]
            # Compute max abs value over valid (masked) wavelengths
            valid_data = rows[:, mask]
            max_abs = np.abs(valid_data).max() if valid_data.size > 0 else 0
            if max_abs > 0:
                normalized[2 * i : 2 * i + 2, :] = rows / max_abs
            else:
                normalized[2 * i : 2 * i + 2, :] = rows
        return normalized

    # Normalize non-negative arrays per-component to [0, 1]
    def normalize_unsigned_per_component(
        data: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Normalize each pair of rows (flux, error) to [0, 1]."""
        normalized = np.zeros_like(data)
        for i in range(n_params):
            # Get both rows for this parameter
            rows = data[2 * i : 2 * i + 2, :]
            # Compute max value over valid (masked) wavelengths
            valid_data = rows[:, mask]
            max_val = valid_data.max() if valid_data.size > 0 else 0
            if max_val > 0:
                normalized[2 * i : 2 * i + 2, :] = rows / max_val
            else:
                normalized[2 * i : 2 * i + 2, :] = rows
        return normalized

    delta_mu_normalized = normalize_signed_per_component(delta_mu_combined, result.mask)
    delta_sigma_normalized = normalize_signed_per_component(
        delta_sigma_combined, result.mask
    )
    fisher_normalized = normalize_unsigned_per_component(fisher_combined, result.mask)

    # Create row labels
    row_labels = []
    for name in result.parameter_names:
        row_labels.append(f"{name} (flux)")
        row_labels.append(f"{name} (err)")

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Panel data: (title, data, cmap, vmin, vmax)
    panel_data = [
        ("Mean Sensitivity Δμ (relative)", delta_mu_normalized, cmap_signed, -1, 1),
        (
            "Scatter Sensitivity Δσ (relative)",
            delta_sigma_normalized,
            cmap_signed,
            -1,
            1,
        ),
        ("Fisher Importance (relative)", fisher_normalized, cmap_fisher, 0, 1),
    ]

    # Compute wavelength tick positions and labels
    wavelength = result.wavelength
    pixel_indices = np.arange(len(wavelength))
    wl_min, wl_max = wavelength.min(), wavelength.max()

    # Choose nice round tick values (multiples of 500 Å)
    tick_spacing = 500
    first_tick = int(np.ceil(wl_min / tick_spacing) * tick_spacing)
    last_tick = int(np.floor(wl_max / tick_spacing) * tick_spacing)
    tick_wavelengths = np.arange(first_tick, last_tick + 1, tick_spacing)

    # Use np.interp to find pixel positions for these wavelengths
    tick_positions = np.interp(tick_wavelengths, wavelength, pixel_indices)
    tick_labels = [f"{int(wl)}" for wl in tick_wavelengths]

    for ax, (title, data, panel_cmap, vmin, vmax) in zip(
        axes, panel_data, strict=False
    ):
        # Apply mask - set masked columns to NaN for visualization
        data_masked = data.copy()
        data_masked[:, ~result.mask] = np.nan

        norm = Normalize(vmin=vmin, vmax=vmax)

        # Plot heatmap using pixel indices
        im = ax.imshow(
            data_masked,
            aspect="auto",
            cmap=panel_cmap,
            norm=norm,
            interpolation="nearest",
        )

        # Set y-axis limits
        ax.set_ylim(n_params * 2 - 0.5, -0.5)

        # Set x-axis ticks using actual wavelength values
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(0, n_wavelengths - 1)

        # Add horizontal black lines between each parameter group (every 2 rows)
        for i in range(1, n_params):
            ax.axhline(y=2 * i - 0.5, color="black", linewidth=0.8, zorder=5)

        # Add masked region overlays (using pixel coordinates)
        for start, end in _find_masked_regions(result.mask):
            ax.axvspan(
                start,
                min(end, n_wavelengths - 1),
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

    # Overall title with metadata
    block_info = (
        f"block={result.block_size}"
        if isinstance(result.block_size, int)
        else f"blocks={len(result.block_size)}"
    )
    title_line = f"Ablation Saliency Heatmaps ({result.survey}, {block_info})"
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
