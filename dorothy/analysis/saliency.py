"""Gradient and ablation saliency of the predictions to the input spectrum.

``SaliencyAnalyzer`` takes the Jacobian of the predicted means and scatters
with respect to the input and weights it by the Fisher information.
``AblationSaliencyAnalyzer`` masks blocks of the spectrum and measures the
change of the predictions. Both analyse one spectrum, as the model sees it
(preprocessed, in normalized units), with the first output head of a
``MultiHeadMLP`` and no label errors.

Examples:
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

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from dorothy.config.schema import STELLAR_PARAMETERS
from dorothy.losses import predicted_scatter
from dorothy.utils import resolve_device

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from dorothy.data.normalizer import LabelNormalizer


# Standard parameter names; prefer passing the catalogue's names
DEFAULT_PARAMETER_NAMES = list(STELLAR_PARAMETERS)


def _default_parameter_names(model: nn.Module) -> list[str]:
    """Standard parameter names sized to the model's number of outputs."""
    n_params = getattr(model, "n_parameters", len(DEFAULT_PARAMETER_NAMES))
    names = DEFAULT_PARAMETER_NAMES[:n_params]
    return names + [f"param_{i}" for i in range(len(names), n_params)]


@dataclass
class SaliencyResult:
    """Result of ``SaliencyAnalyzer.compute_saliency``.

    Predictions, scatters and Jacobians are in normalized units, with the
    input as the model sees it. The Jacobians and the Fisher importance are
    zero at masked pixels.

    Attributes:
        survey: Survey of the spectrum, "unknown" if not given.
        wavelength: Wavelength grid in Å, shape (N,).
        spectrum: Flux channel of the input, shape (N,).
        spectrum_error: Error channel of the input, shape (N,): the precision
            encoding ρ when ``data.precision_encoding`` is set, else σ.
        mask: True where the pixel is valid, shape (N,).
        jacobian_mean_flux: Gradient of the predicted means w.r.t. flux, shape
            (n_params, N).
        jacobian_mean_error: Gradient of the means w.r.t. the error channel,
            shape (n_params, N).
        jacobian_sigma_flux: Gradient of the predicted scatter σ w.r.t. flux,
            shape (n_params, N). σ is ``losses.predicted_scatter``, so the
            gradient is zero where ln s is clamped.
        jacobian_sigma_error: Gradient of σ w.r.t. the error channel, shape
            (n_params, N).
        predictions: Predicted means, shape (n_params,).
        uncertainties: Predicted scatters σ, shape (n_params,).
        fisher_importance_per_param: Fisher importance of each parameter,
            shape (n_params, N).
        fisher_importance: Sum of ``fisher_importance_per_param`` over the
            parameters, shape (N,).
        parameter_names: Parameter names, length n_params.
        gaia_id: Gaia DR3 source ID, if given.
        ra: Right ascension in degrees, if given.
        dec: Declination in degrees, if given.
    """

    survey: str
    wavelength: NDArray[np.float32]
    spectrum: NDArray[np.float32]
    spectrum_error: NDArray[np.float32]
    mask: NDArray[np.bool_]

    # Raw Jacobians
    jacobian_mean_flux: NDArray[np.float32]
    jacobian_mean_error: NDArray[np.float32]
    jacobian_sigma_flux: NDArray[np.float32]
    jacobian_sigma_error: NDArray[np.float32]

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
        """Index of a parameter in the parameter arrays.

        Args:
            param_name: Parameter name, such as "teff".

        Returns:
            The index.

        Raises:
            ValueError: If ``param_name`` is not in ``parameter_names``.
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
    """Result of ``AblationSaliencyAnalyzer.compute_saliency``.

    A block of the spectrum is ablated by setting its pixels to zero in the
    flux, error and mask channels. The value at a pixel is the mean over the
    ablated blocks that cover it, and zero at masked pixels. Ablation changes
    flux and error together, so the result has no separate maps for them.

    The changes are signed like gradients. A positive ``delta_mu`` means that
    ablating the region raised the prediction, so the region was lowering it.
    Predictions and scatters are in normalized units.

    Attributes:
        survey: Survey of the spectrum, "unknown" if not given.
        wavelength: Wavelength grid in Å, shape (N,).
        spectrum: Flux channel of the input, shape (N,).
        spectrum_error: Error channel of the input, shape (N,): the precision
            encoding ρ when ``data.precision_encoding`` is set, else σ.
        mask: True where the pixel is valid, shape (N,).
        block_size: Block size in pixels. With the training distribution, the
            list of block sizes, one per grid point and with repeats; the maps
            are the mean over them.
        delta_mu: Change of the predicted mean, μ_ablated - μ₀, shape
            (n_params, N).
        delta_mu_total: Sum of ``delta_mu`` over the parameters, shape (N,).
        delta_sigma: Change of the predicted scatter, σ_ablated - σ₀, shape
            (n_params, N). Positive means that the region lowered σ.
        delta_sigma_total: Sum of ``delta_sigma`` over the parameters, shape
            (N,).
        fisher_weighted: ((Δμ)² + 2 (Δσ)²) / σ₀², unsigned, shape
            (n_params, N).
        fisher_weighted_total: Sum of ``fisher_weighted`` over the parameters,
            shape (N,).
        predictions: Baseline predicted means μ₀, shape (n_params,).
        uncertainties: Baseline predicted scatters σ₀, shape (n_params,).
        parameter_names: Parameter names, length n_params.
        gaia_id: Gaia DR3 source ID, if given.
        ra: Right ascension in degrees, if given.
        dec: Declination in degrees, if given.
    """

    survey: str
    wavelength: NDArray[np.float32]
    spectrum: NDArray[np.float32]
    spectrum_error: NDArray[np.float32]
    mask: NDArray[np.bool_]

    # Block size used
    block_size: int | list[int]

    # Delta-mu metrics: mu_ablated - mu_baseline (signed)
    delta_mu: NDArray[np.float32]
    delta_mu_total: NDArray[np.float32]

    # Delta-sigma metrics: sigma_ablated - sigma_baseline (signed)
    delta_sigma: NDArray[np.float32]
    delta_sigma_total: NDArray[np.float32]

    # Fisher-weighted metrics: ((Δμ)² + 2 (Δσ)²) / σ₀² (unsigned)
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
        """Index of a parameter in the parameter arrays.

        Args:
            param_name: Parameter name, such as "teff".

        Returns:
            The index.

        Raises:
            ValueError: If ``param_name`` is not in ``parameter_names``.
        """
        try:
            return self.parameter_names.index(param_name)
        except ValueError:
            raise ValueError(
                f"Unknown parameter '{param_name}'. "
                f"Available: {self.parameter_names}"
            ) from None


class SaliencyAnalyzer:
    """Gradient saliency of the predictions to the input spectrum.

    The Jacobian holds the derivatives of the predicted means μ_p and
    scatters σ_p with respect to the flux and error channels. The Fisher
    information of N(μ, σ²) is 1/σ² for μ and 2/σ² for σ, so the importance
    of pixel i for parameter p is

        I_p(i) = [(∂μ_p/∂flux_i)² + (∂μ_p/∂err_i)²
                  + 2 (∂σ_p/∂flux_i)² + 2 (∂σ_p/∂err_i)²] / σ_p².

    The model uses the first output head of a ``MultiHeadMLP``, without label
    errors.

    Attributes:
        model: The model, on ``device`` and in eval mode.
        device: Device of the computation.
        scatter_floor: Scatter floor s_0 of ``losses.predicted_scatter``.
        parameter_names: Parameter names.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        scatter_floor: float = 0.01,
        parameter_names: list[str] | None = None,
    ) -> None:
        """Move the model to the device and set it to eval mode.

        Args:
            model: Trained ``MLP`` or ``MultiHeadMLP``. It is moved in place.
            device: Device string as in ``dorothy.utils.resolve_device``.
            scatter_floor: Scatter floor s_0 of the model's training
                (``training.scatter_floor``).
            parameter_names: Parameter names, one per model output. The
                default is the standard names, sized to the model's outputs.
        """
        self.model = model
        self.scatter_floor = scatter_floor
        self.parameter_names = parameter_names or _default_parameter_names(model)

        self.device = resolve_device(device)

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
        """Compute the saliency of one spectrum.

        One forward pass and one batched backward pass over the 2 * n_params
        outputs.

        Args:
            X: Model input, shape (3, N): [flux, error, mask], with the
                preprocessing of training. A pixel is valid where the mask is
                above 0.5; masked pixels may hold any value, including NaN.
            wavelength: Wavelength in Å, shape (N,).
            survey: Survey of the spectrum, required for a ``MultiHeadMLP``
                with several surveys.
            gaia_id: Gaia DR3 source ID, stored in the result.
            ra: Right ascension in degrees, stored in the result.
            dec: Declination in degrees, stored in the result.

        Returns:
            The Jacobians, predictions and Fisher importance.
        """
        # Ensure X is on device
        X = X.to(self.device)

        # Extract spectrum components (raw, before masking)
        spectrum_raw = X[0].detach().cpu().numpy()
        spectrum_error_raw = X[1].detach().cpu().numpy()
        valid = X[2] > 0.5
        mask = valid.cpu().numpy()

        # The model sees masked pixels as zeros in every channel. Select them
        # rather than multiply by the mask: masked entries may hold NaN or inf.
        X_masked = torch.where(valid, X, 0.0)

        # Store the raw spectrum for visualization (so we can see the actual data)
        spectrum = spectrum_raw
        spectrum_error = spectrum_error_raw

        # Compute Jacobian on the properly masked input
        (
            jac_mean_flux,
            jac_mean_error,
            jac_sigma_flux,
            jac_sigma_error,
            predictions,
            sigma,
        ) = self._compute_jacobian(X_masked, survey)

        # Compute Fisher importance
        fisher_per_param, fisher_total = self._compute_fisher_importance(
            jac_mean_flux,
            jac_mean_error,
            jac_sigma_flux,
            jac_sigma_error,
            sigma,
        )

        # Convert to numpy and apply mask to zero out invalid pixels
        jac_mean_flux_np = jac_mean_flux.cpu().numpy()
        jac_mean_error_np = jac_mean_error.cpu().numpy()
        jac_sigma_flux_np = jac_sigma_flux.cpu().numpy()
        jac_sigma_error_np = jac_sigma_error.cpu().numpy()
        fisher_per_param_np = fisher_per_param.cpu().numpy()
        fisher_total_np = fisher_total.cpu().numpy()

        # Zero out gradients for masked (invalid) pixels
        # This ensures bad pixels don't show spurious saliency values
        jac_mean_flux_np[:, ~mask] = 0.0
        jac_mean_error_np[:, ~mask] = 0.0
        jac_sigma_flux_np[:, ~mask] = 0.0
        jac_sigma_error_np[:, ~mask] = 0.0
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
            jacobian_sigma_flux=jac_sigma_flux_np,
            jacobian_sigma_error=jac_sigma_error_np,
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
        """Compute the Jacobians of the means and scatters with torch.autograd.

        The scatter is σ = ``predicted_scatter(ln s)``, so autograd gives its
        exact derivative, dσ/dx = (exp(2 ln s) / σ) d ln s/dx, which is zero
        where ln s is clamped. All 2 * n_params rows are computed in one batched
        backward pass (``is_grads_batched``), which turns the per-row
        matrix-vector products into matrix-matrix products. Models whose
        backward cannot be batched fall back to one backward pass per row.

        Args:
            X: Input tensor of shape (3, N) - [flux, error, mask].
            survey: Survey name for MultiHeadMLP.

        Returns:
            Tuple of (jac_mean_flux, jac_mean_error, jac_sigma_flux,
            jac_sigma_error, predictions, sigma). Jacobians have shape
            (n_params, N); predictions and sigma have shape (n_params,).
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
        sigma = predicted_scatter(output[0, 1, :], self.scatter_floor)

        n_params = means.shape[0]
        N = X.shape[1]  # Number of wavelengths

        # Initialize Jacobian tensors
        jac_mean_flux = torch.zeros(n_params, N, device=self.device)
        jac_mean_error = torch.zeros(n_params, N, device=self.device)
        jac_sigma_flux = torch.zeros(n_params, N, device=self.device)
        jac_sigma_error = torch.zeros(n_params, N, device=self.device)

        # Row k of the identity selects output k, so grads[k] = d output_k / dX
        # for all outputs [means..., sigma...] in a single backward pass.
        flat_output = torch.cat([means, sigma])  # (2 * n_params,)
        try:
            (grads,) = torch.autograd.grad(
                flat_output,
                X,
                grad_outputs=torch.eye(
                    flat_output.numel(),
                    device=flat_output.device,
                    dtype=flat_output.dtype,
                ),
                retain_graph=True,
                is_grads_batched=True,
            )
        except RuntimeError:
            grads = None  # Backward not batchable for this model: loop below

        if grads is not None:
            grads = grads.reshape(2, n_params, *X.shape)  # (2, n_params, 3, N)
            jac_mean_flux.copy_(grads[0, :, 0, :])
            jac_mean_error.copy_(grads[0, :, 1, :])
            jac_sigma_flux.copy_(grads[1, :, 0, :])
            jac_sigma_error.copy_(grads[1, :, 1, :])
            return (
                jac_mean_flux,
                jac_mean_error,
                jac_sigma_flux,
                jac_sigma_error,
                means.detach(),
                sigma.detach(),
            )

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

            # Gradient of the predicted scatter w.r.t. input
            grad_sigma = torch.autograd.grad(
                sigma[i], X, retain_graph=True, create_graph=False
            )[
                0
            ]  # (3, N)
            jac_sigma_flux[i] = grad_sigma[0, :]
            jac_sigma_error[i] = grad_sigma[1, :]

        return (
            jac_mean_flux,
            jac_mean_error,
            jac_sigma_flux,
            jac_sigma_error,
            means.detach(),
            sigma.detach(),
        )

    def _compute_fisher_importance(
        self,
        jac_mean_flux: torch.Tensor,
        jac_mean_error: torch.Tensor,
        jac_sigma_flux: torch.Tensor,
        jac_sigma_error: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate Jacobians into Fisher-weighted importance.

        The Fisher information for a Gaussian N(μ, σ²) is:
            I_μμ = 1/σ², I_σσ = 2/σ²

        For each parameter p and wavelength i:
            I_p(i) = (∂μ_p/∂flux_i)²/σ_p² + (∂μ_p/∂err_i)²/σ_p²
                   + 2·(∂σ_p/∂flux_i)²/σ_p² + 2·(∂σ_p/∂err_i)²/σ_p²

        Args:
            jac_mean_flux: Gradient of means w.r.t. flux, shape (n_params, N).
            jac_mean_error: Gradient of means w.r.t. error, shape (n_params, N).
            jac_sigma_flux: Gradient of σ w.r.t. flux, shape (n_params, N).
            jac_sigma_error: Gradient of σ w.r.t. error, shape (n_params, N).
            sigma: Predicted uncertainties, shape (n_params,).

        Returns:
            Tuple of (per_param_importance, total_importance).
            - per_param_importance: (n_params, N) importance per parameter
            - total_importance: (N,) sum over all parameters
        """
        # sigma shape: (n_params,) -> (n_params, 1) for broadcasting
        sigma_sq = (sigma**2).unsqueeze(1)  # (n_params, 1)
        per_param_importance = (
            jac_mean_flux**2
            + jac_mean_error**2
            + 2 * (jac_sigma_flux**2 + jac_sigma_error**2)
        ) / sigma_sq

        # Total importance (sum over parameters)
        total_importance = per_param_importance.sum(dim=0)

        return per_param_importance, total_importance


@contextmanager
def _pin_output(module: nn.Module, output: torch.Tensor) -> Iterator[None]:
    """Temporarily make ``module`` return ``output`` instead of computing it."""

    def forward(inp: torch.Tensor) -> torch.Tensor:
        if inp.shape[0] != output.shape[0]:
            raise RuntimeError("Pinned output does not match the input batch size")
        return output

    module.__dict__["forward"] = forward  # the instance attribute shadows the method
    try:
        yield
    finally:
        del module.__dict__["forward"]


class _InputLayerShortcut:
    """Evaluate ablated spectra without recomputing the model's input layer.

    Ablation zeroes pixels [p, p + B) in all three channels. When the spectrum
    reaches the model output only through one nn.Linear applied to the
    flattened input x (3 * N features), that layer's output for the ablated
    input is

        h(p) = b + sum of C[i] over pixels i outside [p, p + B),
        C[i] = sum_c W[:, c * N + i] * x[c, i].

    With float64 prefix sums of C over pixels, each ablated input costs
    O(hidden) for this layer instead of O(3 * N * hidden). The layer is pinned
    to return h(p) and the rest of the model runs as usual. h(p) is rounded
    once from float64, so it is at least as accurate as the float32 matrix
    product it replaces; results agree with the direct computation up to
    floating-point rounding.
    """

    def __init__(self, linear: nn.Module, X: torch.Tensor, output: torch.Tensor):
        """Precompute prefix sums of per-pixel contributions.

        Args:
            linear: The model's input layer (nn.Linear on the flattened input).
            X: Masked input tensor of shape (3, N).
            output: The layer's baseline output, shape (1, hidden).
        """
        # float64 arithmetic runs on the CPU (MPS has no float64)
        n_channels, n_pixels = X.shape
        weight = linear.weight.detach()
        x = X.detach().cpu().double()
        contrib = torch.zeros(n_pixels, weight.shape[0], dtype=torch.float64)
        for c in range(n_channels):
            w_c = weight[:, c * n_pixels : (c + 1) * n_pixels].cpu().double()
            contrib += x[c].unsqueeze(1) * w_c.T  # (N, hidden)

        # prefix[k] = sum of C[i] for i < k
        self.prefix = torch.zeros(n_pixels + 1, weight.shape[0], dtype=torch.float64)
        torch.cumsum(contrib, dim=0, out=self.prefix[1:])
        self.full = self.prefix[-1].clone()  # b + sum of all C[i]
        if linear.bias is not None:
            self.full += linear.bias.detach().cpu().double()

        self.linear = linear
        self.dtype = output.dtype
        self.device = output.device

    @classmethod
    def create(
        cls,
        calls: list[tuple[nn.Module, tuple, torch.Tensor]],
        X: torch.Tensor,
        mu_0: torch.Tensor,
        ln_s_0: torch.Tensor,
        forward: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> _InputLayerShortcut | None:
        """Return a shortcut if the baseline pass allows one, else None.

        Conditions: the first nn.Linear called uses the standard Linear
        forward, receives exactly the flattened (finite) masked spectrum and
        is called only once; with that layer pinned to its baseline output, a
        different input leaves the model output unchanged (the input reaches
        the output only through it); and the precomputed output for the
        unablated spectrum matches the layer's actual output.

        Args:
            calls: (module, args, output) of each nn.Linear call in the
                baseline pass, in call order.
            X: Masked input tensor of shape (3, N).
            mu_0: Baseline means, shape (1, n_params).
            ln_s_0: Baseline log_scatter, shape (1, n_params).
            forward: Model forward pass returning (means, log_scatter).
        """
        if not calls:
            return None
        linear, args, output = calls[0]
        flat = X.reshape(1, -1)
        if (
            type(linear).forward is not nn.Linear.forward
            or "forward" in linear.__dict__
            or len(args) != 1
            or sum(module is linear for module, _, _ in calls) != 1
            or args[0].shape != flat.shape
            or args[0].dtype != flat.dtype
            or not torch.equal(args[0], flat)
            or not torch.isfinite(flat).all()
        ):
            return None

        with _pin_output(linear, output.clone()):
            mu, ln_s = forward((X + 1).unsqueeze(0))
        if not (torch.equal(mu, mu_0) and torch.equal(ln_s, ln_s_0)):
            return None

        shortcut = cls(linear, X, output)
        # Self-check: float32 rounding of the layer is far below this tolerance
        expected = output[0].detach().cpu().double()
        if (shortcut.full - expected).abs().max() > 1e-4 * expected.abs().max():
            return None
        return shortcut

    def layer_outputs(self, n_positions: int, block_size: int) -> torch.Tensor:
        """Input-layer outputs for windows starting at 0..n_positions-1.

        Returns:
            Tensor of shape (n_positions, hidden) on the model's device.
        """
        window = (
            self.prefix[block_size : block_size + n_positions]
            - self.prefix[:n_positions]
        )
        return (self.full - window).to(self.dtype).to(self.device)

    def pinned(self, layer_output: torch.Tensor):
        """Context in which the input layer returns ``layer_output``."""
        return _pin_output(self.linear, layer_output)


class AblationSaliencyAnalyzer:
    """Ablation saliency of the predictions to the input spectrum.

    A window of ``block_size`` pixels slides over the spectrum in steps of one
    pixel. Each window is zeroed in the flux, error and mask channels. Three
    maps follow:

    1. Delta-mu: the signed change of the predicted means.
    2. Delta-sigma: the signed change of the predicted scatters.
    3. Fisher-weighted: ((delta_mu)^2 + 2 (delta_sigma)^2) / sigma_0^2, the
       Fisher quadratic form of N(mu, sigma^2) that gradient saliency uses.

    The model uses the first output head of a ``MultiHeadMLP``, without label
    errors.

    When the spectrum reaches the output only through the model's first
    nn.Linear layer, the analyzer temporarily replaces that layer's forward
    method during a run. Do not call the model from another thread meanwhile.

    Attributes:
        model: The model, on ``device`` and in eval mode.
        device: Device of the computation.
        scatter_floor: Scatter floor s_0 of ``losses.predicted_scatter``.
        parameter_names: Parameter names.
        batch_size: Number of ablated spectra per forward pass.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        scatter_floor: float = 0.01,
        parameter_names: list[str] | None = None,
        batch_size: int = 64,
    ) -> None:
        """Move the model to the device and set it to eval mode.

        Args:
            model: Trained ``MLP`` or ``MultiHeadMLP``. It is moved in place.
            device: Device string as in ``dorothy.utils.resolve_device``.
            scatter_floor: Scatter floor s_0 of the model's training
                (``training.scatter_floor``).
            parameter_names: Parameter names, one per model output. The
                default is the standard names, sized to the model's outputs.
            batch_size: Number of ablated spectra per forward pass.
        """
        self.model = model
        self.scatter_floor = scatter_floor
        self.parameter_names = parameter_names or _default_parameter_names(model)
        self.batch_size = batch_size

        self.device = resolve_device(device)

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
        """Compute the ablation saliency of one spectrum.

        Each block size needs N - block_size + 1 forward passes, in batches
        of ``batch_size``.

        Args:
            X: Model input, shape (3, N): [flux, error, mask], with the
                preprocessing of training. A pixel is valid where the mask is
                above 0.5; masked pixels may hold any value, including NaN.
            wavelength: Wavelength in Å, shape (N,).
            survey: Survey of the spectrum, required for a ``MultiHeadMLP``
                with several surveys.
            block_size: Block size in pixels; None means 5. A block longer
                than N gives zero maps.
            use_training_distribution: If True, ignore ``block_size`` and
                average the maps of ``n_block_sizes`` block sizes ceil(N * f),
                with f evenly spaced in log f from ``f_min`` to ``f_max``
                (the log-uniform sizes of the training input masking), each
                with equal weight.
            n_block_sizes: Number of block sizes with the training
                distribution.
            f_min: Smallest block fraction f with the training distribution;
                None means 1/N.
            f_max: Largest block fraction f with the training distribution.
            gaia_id: Gaia DR3 source ID, stored in the result.
            ra: Right ascension in degrees, stored in the result.
            dec: Declination in degrees, stored in the result.

        Returns:
            The per-pixel changes of the predictions and the Fisher-weighted
            importance.
        """
        # Ensure X is on device
        X = X.to(self.device)
        N = X.shape[1]

        # Extract spectrum components
        spectrum_raw = X[0].detach().cpu().numpy()
        spectrum_error_raw = X[1].detach().cpu().numpy()
        valid = X[2] > 0.5
        natural_mask = valid.cpu().numpy()

        # Input as the model sees it: masked pixels are zeros in every channel,
        # selected rather than multiplied, as they may hold NaN or inf
        X_masked = torch.where(valid, X, 0.0)

        # Get baseline predictions, and a shortcut that evaluates ablated inputs
        # without recomputing the model's input layer (None if not applicable)
        mu_0, ln_s_0, shortcut = self._baseline_forward(X_masked, survey)
        mu_0 = mu_0.squeeze(0)  # (n_params,)
        ln_s_0 = ln_s_0.squeeze(0)  # (n_params,)
        sigma_0 = predicted_scatter(ln_s_0, self.scatter_floor)

        if use_training_distribution:
            # Fractions evenly spaced in log f sample the log-uniform block
            # sizes of training's input masking, so every grid point has equal
            # weight; a size that several fractions round to counts once each
            f_min_actual = f_min if f_min is not None else (1.0 / N)
            log_f = np.linspace(np.log(f_min_actual), np.log(f_max), n_block_sizes)
            block_sizes = [max(1, int(np.ceil(f * N))) for f in np.exp(log_f)]
            sizes, counts = np.unique(block_sizes, return_counts=True)

            maps = np.zeros((3, len(self.parameter_names), N))
            for size, count in zip(sizes, counts, strict=True):
                maps += count * np.stack(
                    self._compute_ablation_importance(
                        X_masked,
                        survey,
                        int(size),
                        mu_0,
                        sigma_0,
                        natural_mask,
                        shortcut=shortcut,
                    )
                )
            delta_mu_np, delta_sigma_np, fisher_np = (maps / n_block_sizes).astype(
                np.float32
            )
            block_size_used: int | list[int] = block_sizes
        else:
            # Single block size mode
            if block_size is None:
                block_size = 5
            delta_mu_np, delta_sigma_np, fisher_np = self._compute_ablation_importance(
                X_masked,
                survey,
                block_size,
                mu_0,
                sigma_0,
                natural_mask,
                shortcut=shortcut,
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
        sigma_0: torch.Tensor,
        natural_mask: NDArray[np.bool_],
        shortcut: _InputLayerShortcut | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Compute ablation importance for a single block size.

        Uses sliding window ablation with stride 1. For each window,
        fisher_weighted = ((Δμ)² + 2 (Δσ)²) / σ₀², where σ is
        ``losses.predicted_scatter``.

        Args:
            X: Masked input tensor of shape (3, N).
            survey: Survey name for MultiHeadMLP.
            block_size: Size of ablation block.
            mu_0: Baseline means, shape (n_params,).
            sigma_0: Baseline predicted scatter σ₀, shape (n_params,).
            natural_mask: Boolean mask of valid pixels.
            shortcut: Optional input-layer shortcut from _baseline_forward().
                If None, each ablated input is built and run through the model.

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
        all_delta_sigma = []  # Signed change in sigma

        if shortcut is not None:
            layer_outputs = shortcut.layer_outputs(n_positions, block_size)

        positions = list(range(n_positions))
        for batch_start in range(0, n_positions, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_positions)
            batch_positions = positions[batch_start:batch_end]

            if shortcut is not None:
                # Forward pass with the input layer's outputs computed directly
                with shortcut.pinned(layer_outputs[batch_start:batch_end]):
                    mu_ablated, ln_s_ablated = self._forward_model(
                        X.unsqueeze(0).expand(len(batch_positions), -1, -1), survey
                    )
            else:
                # Create batch of ablated inputs
                X_ablated = self._create_ablated_batch(X, batch_positions, block_size)

                # Forward pass
                mu_ablated, ln_s_ablated = self._forward_model(X_ablated, survey)

            sigma_ablated = predicted_scatter(ln_s_ablated, self.scatter_floor)

            # Compute deltas (sign matches gradient saliency convention)
            # mu_ablated - mu_0: positive means ablating increased prediction (region suppressed it)
            # This matches gradient saliency where positive = increasing input increases output
            delta_mu = mu_ablated - mu_0.unsqueeze(0)  # (batch, n_params)
            delta_sigma = sigma_ablated - sigma_0.unsqueeze(0)  # (batch, n_params)

            all_delta_mu.append(delta_mu.cpu())
            all_delta_sigma.append(delta_sigma.cpu())

        # Concatenate all batches
        all_delta_mu_tensor = torch.cat(all_delta_mu, dim=0)  # (n_positions, n_params)
        all_delta_sigma_tensor = torch.cat(
            all_delta_sigma, dim=0
        )  # (n_positions, n_params)

        # Fisher quadratic form of N(mu, sigma^2) for the finite changes,
        # shape (n_positions, n_params)
        sigma_0_sq = (sigma_0.cpu() ** 2).unsqueeze(0)
        fisher_per_pos = (
            all_delta_mu_tensor**2 + 2 * all_delta_sigma_tensor**2
        ) / sigma_0_sq

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

    def _baseline_forward(
        self,
        X: torch.Tensor,
        survey: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor, _InputLayerShortcut | None]:
        """Baseline forward pass that also looks for an input-layer shortcut.

        The nn.Linear layers called during the pass are recorded (hooks only
        observe, so the baseline is unchanged) and passed to
        _InputLayerShortcut.create(), which keeps a shortcut only if the model
        satisfies its conditions.

        Args:
            X: Masked input tensor of shape (3, N).
            survey: Survey name for MultiHeadMLP.

        Returns:
            Tuple of (means, log_scatter, shortcut); means and log_scatter have
            shape (1, n_params), shortcut is None when not applicable.
        """
        calls: list[tuple[nn.Module, tuple, torch.Tensor]] = []

        def record(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
            # Copy: later in-place ops (e.g. ReLU(inplace=True)) may modify it
            calls.append((module, args, output.detach().clone()))

        handles = [
            module.register_forward_hook(record)
            for module in self.model.modules()
            if isinstance(module, nn.Linear)
        ]
        try:
            mu_0, ln_s_0 = self._forward_model(X.unsqueeze(0), survey)
        finally:
            for handle in handles:
                handle.remove()

        shortcut = _InputLayerShortcut.create(
            calls, X, mu_0, ln_s_0, lambda x: self._forward_model(x, survey)
        )
        return mu_0, ln_s_0, shortcut

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
        # Window mask (batch, N): True inside [pos, pos + block_size)
        starts = torch.as_tensor(positions, device=X.device).unsqueeze(1)
        pixel = torch.arange(X.shape[1], device=X.device)
        in_window = (pixel >= starts) & (pixel < starts + block_size)

        # Zero out flux, error, and mask in the ablation window
        return torch.where(in_window.unsqueeze(1), X.new_zeros(()), X)

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

        # Windows that cover pixel i start at positions max(0, i-B+1) to
        # min(i, n_pos-1). Window sums come from float64 prefix sums over
        # positions, so each pixel's mean costs O(1) instead of O(B).
        pixels = np.flatnonzero(natural_mask)  # masked pixels keep zero importance
        start_pos = np.maximum(0, pixels - block_size + 1)
        end_pos = np.minimum(pixels + 1, n_positions)
        keep = end_pos > start_pos
        pixels, start_pos, end_pos = pixels[keep], start_pos[keep], end_pos[keep]

        prefix = np.zeros((n_params, n_positions + 1), dtype=np.float64)
        np.cumsum(per_position_metrics, axis=1, dtype=np.float64, out=prefix[:, 1:])

        # Average the metrics from all covering windows
        importance[:, pixels] = (prefix[:, end_pos] - prefix[:, start_pos]) / (
            end_pos - start_pos
        )

        return importance


def _require_valid_pixels(result: SaliencyResult | AblationSaliencyResult) -> None:
    if not result.mask.any():
        raise ValueError(
            "result.mask has no valid pixel: a fully masked spectrum has no "
            "saliency to plot"
        )


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
    """Plot the gradient saliency of one parameter along the spectrum.

    Three stacked panels show the sensitivity of the mean (∂μ/∂x), the
    sensitivity of the scatter (∂σ/∂x) and the Fisher importance of the
    parameter. Each panel has the flux spectrum above the error spectrum,
    coloured by the panel's quantity at each pixel. Masked wavelengths are
    grey. The colour scales span the 99th percentile of the valid pixels.

    Args:
        result: Result of ``SaliencyAnalyzer.compute_saliency``.
        param_name: Parameter to plot, such as "teff".
        output_path: File to save the figure to; None saves nothing.
        figsize: Figure size (width, height) in inches.
        cmap: Colormap of the sensitivity panels (red is negative, blue
            positive for the default).
        cmap_fisher: Colormap of the Fisher panel.
        normalizer: If given, the title shows the prediction in physical
            units; otherwise in normalized units.
        label_values: Catalogue labels in physical units, shape (n_params,),
            shown in the title.
        label_errors: Label errors in physical units, shape (n_params,), shown
            with ``label_values``.

    Returns:
        The Matplotlib figure. The caller closes it.

    Raises:
        ValueError: If param_name is unknown or result.mask has no valid pixel.
    """
    _require_valid_pixels(result)
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

    # Get gradients for this parameter
    grad_mean_flux = result.jacobian_mean_flux[param_idx]
    grad_mean_error = result.jacobian_mean_error[param_idx]
    grad_sigma_flux = result.jacobian_sigma_flux[param_idx]
    grad_sigma_error = result.jacobian_sigma_error[param_idx]
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
    """Plot the gradient saliency of all parameters as heatmaps.

    Three stacked panels show ∂μ/∂x, ∂σ/∂x and the Fisher importance. Each has
    two rows per parameter (flux and error channel) and one column per pixel.
    Masked wavelengths are grey. Each parameter's two rows are divided by their
    largest absolute value over the valid pixels: the sensitivity panels
    span [-1, 1] and the Fisher panel [0, 1]. The Fisher rows of a parameter
    are the flux and error terms of its importance.

    Args:
        result: Result of ``SaliencyAnalyzer.compute_saliency``.
        output_path: File to save the figure to; None saves nothing.
        figsize: Figure size (width, height) in inches.
        cmap_gradient: Colormap of the sensitivity panels (red is negative,
            blue positive for the default).
        cmap_fisher: Colormap of the Fisher panel.

    Returns:
        The Matplotlib figure. The caller closes it.

    Raises:
        ValueError: If result.mask has no valid pixel.
    """
    _require_valid_pixels(result)
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
        scatter_combined[2 * i] = result.jacobian_sigma_flux[i]
        scatter_combined[2 * i + 1] = result.jacobian_sigma_error[i]
        # For Fisher, compute separate flux/error contributions
        sigma_sq = result.uncertainties[i] ** 2
        fisher_combined[2 * i] = (
            result.jacobian_mean_flux[i] ** 2 + 2 * result.jacobian_sigma_flux[i] ** 2
        ) / sigma_sq
        fisher_combined[2 * i + 1] = (
            result.jacobian_mean_error[i] ** 2 + 2 * result.jacobian_sigma_error[i] ** 2
        ) / sigma_sq

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
    """Plot the ablation saliency of one parameter along the spectrum.

    The layout is that of ``plot_parameter_saliency``. The three panels show
    Δμ, Δσ and the Fisher-weighted change of the parameter. Ablation changes
    flux and error together, so the flux and error spectra of a panel have the
    same colours. A positive Δμ means that ablating the region raised the
    prediction.

    Args:
        result: Result of ``AblationSaliencyAnalyzer.compute_saliency``.
        param_name: Parameter to plot, such as "teff".
        output_path: File to save the figure to; None saves nothing.
        figsize: Figure size (width, height) in inches.
        cmap: Colormap of the signed panels.
        cmap_fisher: Colormap of the Fisher panel.
        normalizer: If given, the title shows the prediction in physical
            units; otherwise in normalized units.
        label_values: Catalogue labels in physical units, shape (n_params,),
            shown in the title.
        label_errors: Label errors in physical units, shape (n_params,), shown
            with ``label_values``.

    Returns:
        The Matplotlib figure. The caller closes it.

    Raises:
        ValueError: If param_name is unknown or result.mask has no valid pixel.
    """
    _require_valid_pixels(result)
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
            if valid_vals.max() > 0:
                norm = Normalize(vmin=0, vmax=np.percentile(valid_vals, 99))
            else:
                norm = Normalize(vmin=0, vmax=1)
        else:
            # Signed values - symmetric around zero
            max_abs = np.percentile(np.abs(valid_vals), 99)
            if max_abs > 0:
                norm = Normalize(vmin=-max_abs, vmax=max_abs)
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
    """Plot the ablation saliency of all parameters as heatmaps.

    The layout is that of ``plot_saliency_heatmap``. The three panels show Δμ,
    Δσ and the Fisher-weighted change. Ablation changes flux and error
    together, so the two rows of a parameter are equal. Each parameter is
    divided by its largest absolute value over the valid pixels: the signed
    panels span [-1, 1] and the Fisher panel [0, 1].

    Args:
        result: Result of ``AblationSaliencyAnalyzer.compute_saliency``.
        output_path: File to save the figure to; None saves nothing.
        figsize: Figure size (width, height) in inches.
        cmap_signed: Colormap of the signed panels.
        cmap_fisher: Colormap of the Fisher panel.

    Returns:
        The Matplotlib figure. The caller closes it.

    Raises:
        ValueError: If result.mask has no valid pixel.
    """
    _require_valid_pixels(result)
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
            max_abs = np.abs(valid_data).max()
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
            max_val = valid_data.max()
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
