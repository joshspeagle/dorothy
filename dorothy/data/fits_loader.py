"""Load stellar spectra and labels from FITS files, and preprocess spectra.

:class:`FITSLoader` reads FITS files with DESI, BOSS or LAMOST spectra
cross-matched with APOGEE stellar parameters. The preprocessing functions
(:func:`zero_invalid_pixels`, :func:`normalize_spectra`,
:func:`encode_precision`) are shared with the catalogue loaders.

Expected FITS file structure:
    - APOGEE extension: Table with stellar parameters and quality flags
    - SPECTRA extension: Table with flux arrays for each star
    - SPEC_IVAR extension: Table with inverse variance arrays
    - WAVELENGTH extension: Table with wavelength grid

The loader normalizes spectra using median/IQR normalization per-star
and creates the 2-channel input format (flux, scaled ivar) expected by
DOROTHY models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dorothy.config.schema import DataConfig


# Column name mappings for APOGEE data
# Maps internal parameter names to FITS column names
APOGEE_COLUMN_MAP = {
    "teff": ("TEFF", "TEFF_ERR"),
    "logg": ("LOGG", "LOGG_ERR"),
    "fe_h": ("FE_H", "FE_H_ERR"),
    "mg_fe": ("MG_FE", "MG_FE_ERR"),
    "c_fe": ("C_FE", "C_FE_ERR"),
    "si_fe": ("SI_FE", "SI_FE_ERR"),
    "ni_fe": ("NI_FE", "NI_FE_ERR"),
    "al_fe": ("AL_FE", "AL_FE_ERR"),
    "ca_fe": ("CA_FE", "CA_FE_ERR"),
    "n_fe": ("N_FE", "N_FE_ERR"),
    "mn_fe": ("MN_FE", "MN_FE_ERR"),
}

# Physical bounds for quality filtering
PARAMETER_BOUNDS = {
    "teff": (2500, 10000),  # Kelvin
    "logg": (-1.0, 6.0),  # log cm/s^2
    "fe_h": (-5.0, 1.0),  # dex
    "mg_fe": (-2.0, 2.0),  # dex
    "c_fe": (-2.0, 2.0),
    "si_fe": (-2.0, 2.0),
    "ni_fe": (-2.0, 2.0),
    "al_fe": (-2.0, 2.0),
    "ca_fe": (-2.0, 2.0),
    "n_fe": (-2.0, 2.0),
    "mn_fe": (-2.0, 2.0),
}


@dataclass
class SpectralData:
    """
    Container for loaded spectral data.

    Attributes:
        flux: Flux of shape (n_samples, n_wavelengths), normalized if
            requested, and 0 at invalid pixels (:func:`zero_invalid_pixels`).
        ivar: Inverse variance with the same shape, scaled with the flux, and
            0 at invalid pixels.
        wavelength: Wavelength grid of shape (n_wavelengths,).
        labels: Stellar parameter labels of shape (n_samples, n_parameters).
        errors: Label uncertainties of shape (n_samples, n_parameters).
        ids: Star identifiers (e.g., APOGEE_ID) of length n_samples.
        quality_mask: Boolean mask where True = good quality (for samples).
        parameter_names: List of parameter names in order.
        spectral_mask: Binary mask where 1=valid, 0=masked (per wavelength).
            Shape is (n_samples, n_wavelengths). Derived from ivar > 0.
        spectral_error: Error array (sigma) converted from ivar.
            Shape is (n_samples, n_wavelengths). Computed as 1/sqrt(ivar).
    """

    flux: NDArray[np.float32]
    ivar: NDArray[np.float32]
    wavelength: NDArray[np.float64]
    labels: NDArray[np.float32]
    errors: NDArray[np.float32]
    ids: NDArray[np.str_]
    quality_mask: NDArray[np.bool_]
    parameter_names: list[str] = field(default_factory=list)
    spectral_mask: NDArray[np.float32] | None = None
    spectral_error: NDArray[np.float32] | None = None

    @property
    def n_samples(self) -> int:
        """Number of spectra in the dataset."""
        return self.flux.shape[0]

    @property
    def n_wavelengths(self) -> int:
        """Number of wavelength bins."""
        return self.flux.shape[1]

    @property
    def n_parameters(self) -> int:
        """Number of stellar parameters."""
        return self.labels.shape[1]

    def get_model_input(self, apply_quality_mask: bool = True) -> NDArray[np.float32]:
        """
        Get combined flux/ivar array for model input (2-channel format).

        Args:
            apply_quality_mask: Whether to filter by quality mask.

        Returns:
            Array of shape (n_samples, 2, n_wavelengths) containing
            [flux, scaled_ivar] as the two channels.
        """
        if apply_quality_mask:
            flux = self.flux[self.quality_mask]
            ivar = self.ivar[self.quality_mask]
        else:
            flux = self.flux
            ivar = self.ivar

        return np.stack([flux, ivar], axis=1).astype(np.float32)

    def get_model_input_3channel(
        self, apply_quality_mask: bool = True
    ) -> NDArray[np.float32]:
        """
        Get combined flux/error/mask array for model input (3-channel format).

        This format is consistent with label format [values | errors] and
        includes an explicit mask channel for handling missing data and
        dynamic block masking augmentation.

        Args:
            apply_quality_mask: Whether to filter by sample quality mask.

        Returns:
            Array of shape (n_samples, 3, n_wavelengths) containing:
            - Channel 0: Normalized flux (masked wavelengths → 0)
            - Channel 1: Error (sigma, converted from ivar, masked → 0)
            - Channel 2: Mask (1=valid, 0=masked)

        Raises:
            ValueError: If spectral_mask or spectral_error are not available.
        """
        if self.spectral_mask is None or self.spectral_error is None:
            raise ValueError(
                "3-channel format requires spectral_mask and spectral_error. "
                "Reload with compute_spectral_mask=True."
            )

        if apply_quality_mask:
            flux = self.flux[self.quality_mask]
            error = self.spectral_error[self.quality_mask]
            mask = self.spectral_mask[self.quality_mask]
        else:
            flux = self.flux
            error = self.spectral_error
            mask = self.spectral_mask

        return np.stack([flux, error, mask], axis=1).astype(np.float32)

    def get_labels(self, apply_quality_mask: bool = True) -> NDArray[np.float32]:
        """Get labels, optionally filtered by quality mask.

        Args:
            apply_quality_mask: Whether to keep only quality_mask stars.

        Returns:
            Array of shape (n_samples, n_parameters).
        """
        if apply_quality_mask:
            return self.labels[self.quality_mask]
        return self.labels

    def get_errors(self, apply_quality_mask: bool = True) -> NDArray[np.float32]:
        """Get label errors, optionally filtered by quality mask.

        Args:
            apply_quality_mask: Whether to keep only quality_mask stars.

        Returns:
            Array of shape (n_samples, n_parameters).
        """
        if apply_quality_mask:
            return self.errors[self.quality_mask]
        return self.errors

    def get_combined_labels(
        self, apply_quality_mask: bool = True
    ) -> NDArray[np.float32]:
        """
        Get labels and errors combined for training.

        Args:
            apply_quality_mask: Whether to keep only quality_mask stars.

        Returns:
            Array of shape (n_samples, 2 * n_parameters) where the first half
            contains labels and the second half contains errors.
        """
        labels = self.get_labels(apply_quality_mask)
        errors = self.get_errors(apply_quality_mask)
        return np.concatenate([labels, errors], axis=1).astype(np.float32)


# torch.nanquantile rejects inputs above 2**24 elements in some torch versions
_MAX_QUANTILE_ELEMENTS = 2**24

# Rows per block in zero_invalid_pixels; bounds its boolean temporaries
_BLOCK_ROWS = 256


def zero_invalid_pixels(
    flux: NDArray,
    ivar: NDArray,
    inplace: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Set the flux and ivar of invalid pixels to 0.

    A pixel is valid when its flux is finite and its ivar is finite and
    positive. Every loading path applies this rule before it builds the
    model inputs (:func:`normalize_spectra` applies it first), so the mask
    ivar > 0 marks exactly the valid pixels and a masked pixel enters the
    model as zeros.

    Args:
        flux: Flux array of shape (n_spectra, n_wavelengths) or (n_wavelengths,).
        ivar: Inverse variance array with the same shape as flux.
        inplace: Write the results into ``flux`` and ``ivar`` (must be float32
            arrays) instead of allocating new ones.

    Returns:
        Tuple of (flux, ivar), float32, same shape as input.

    Raises:
        ValueError: If flux and ivar differ in shape, or ``inplace`` is set
            and they are not float32.
    """
    flux = np.asarray(flux)
    ivar = np.asarray(ivar)
    if flux.shape != ivar.shape:
        raise ValueError(f"flux shape {flux.shape} != ivar shape {ivar.shape}")
    if not inplace:
        flux = flux.astype(np.float32)
        ivar = ivar.astype(np.float32)
    elif flux.dtype != np.float32 or ivar.dtype != np.float32:
        raise ValueError("inplace requires float32 flux and ivar arrays")

    flux_2d = flux[None] if flux.ndim == 1 else flux
    ivar_2d = ivar[None] if ivar.ndim == 1 else ivar
    for start in range(0, len(flux_2d), _BLOCK_ROWS):
        f = flux_2d[start : start + _BLOCK_ROWS]
        iv = ivar_2d[start : start + _BLOCK_ROWS]
        invalid = ~(np.isfinite(f) & np.isfinite(iv) & (iv > 0))
        np.copyto(f, 0, where=invalid)
        np.copyto(iv, 0, where=invalid)
    return flux, ivar


def normalize_spectra(
    flux: NDArray,
    ivar: NDArray,
    chunk_size: int = 4096,
    inplace: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Per-spectrum median/IQR normalization over valid pixels.

    For each spectrum (row)::

        normalized_flux = (flux - median) / IQR
        scaled_ivar = IQR^2 * ivar

    where the median and IQR are computed over valid pixels only (finite flux,
    finite and positive ivar). Invalid pixels are returned with flux = 0 and
    ivar = 0 (:func:`zero_invalid_pixels`), so downstream code that derives
    the pixel mask from ivar > 0 treats them as masked. So are pixels whose
    normalized flux or scaled ivar overflows to infinity. A spectrum with no
    valid pixels uses median = 0 and IQR = 1; one with zero IQR uses IQR = 1.

    Args:
        flux: Flux array of shape (n_spectra, n_wavelengths) or (n_wavelengths,).
        ivar: Inverse variance array with the same shape as flux.
        chunk_size: Number of spectra processed at once (bounds peak memory).
        inplace: Write the results into ``flux`` and ``ivar`` (must be float32
            arrays) instead of allocating new ones.

    Returns:
        Tuple of (normalized_flux, scaled_ivar), float32, same shape as input.

    Raises:
        ValueError: If flux and ivar differ in shape, or ``inplace`` is set
            and they are not float32.
    """
    flux, ivar = zero_invalid_pixels(flux, ivar, inplace)
    flux_2d = flux[None] if flux.ndim == 1 else flux
    ivar_2d = ivar[None] if ivar.ndim == 1 else ivar

    # Torch is used for speed: its nan-quantiles and elementwise ops are
    # vectorized and multithreaded (numpy's nan-quantiles loop over rows).
    import torch

    quantiles = torch.tensor([0.25, 0.5, 0.75])
    nan = torch.tensor(float("nan"))
    zero = torch.tensor(0.0)
    n_cols = max(flux_2d.shape[1], 1)
    rows_per_chunk = max(1, min(chunk_size, _MAX_QUANTILE_ELEMENTS // n_cols))
    for start in range(0, flux_2d.shape[0], rows_per_chunk):
        sl = slice(start, start + rows_per_chunk)
        f = torch.from_numpy(flux_2d[sl])
        iv = torch.from_numpy(ivar_2d[sl])
        valid = iv > 0
        q25, med, q75 = torch.nanquantile(
            torch.where(valid, f, nan), quantiles, dim=1, keepdim=True
        )
        iqr = q75 - q25
        iqr = torch.where(torch.isfinite(iqr) & (iqr > 0), iqr, torch.ones_like(iqr))
        med = torch.nan_to_num(med, nan=0.0)
        # Compute both outputs before writing (f and iv are views of them).
        # A finite ivar can overflow when scaled; such a pixel is invalid too.
        scaled_flux, scaled_ivar = (f - med) / iqr, iv * iqr**2
        keep = valid & torch.isfinite(scaled_flux) & torch.isfinite(scaled_ivar)
        new_flux = torch.where(keep, scaled_flux, zero)
        new_ivar = torch.where(keep, scaled_ivar, zero)
        flux_2d[sl] = new_flux.numpy()
        ivar_2d[sl] = new_ivar.numpy()
    return flux, ivar


def _encode_sigma(sigma: NDArray, valid: NDArray) -> None:
    """Replace ``sigma`` in place with log(1 + 1/sigma) where ``valid``, else 0.

    Shared by :func:`encode_precision` and the sparse loader, which stores
    sigma without a mask channel.
    """
    # 1/inf = 0, so a masked pixel becomes log1p(0) = 0 whatever it held
    np.copyto(sigma, np.inf, where=~valid)
    np.reciprocal(sigma, out=sigma)
    np.log1p(sigma, out=sigma)


def encode_precision(spectra: NDArray, inplace: bool = False) -> NDArray[np.float32]:
    """Replace the sigma channel of model inputs with the precision encoding.

    For each pixel with uncertainty sigma and mask m::

        rho = log(1 + 1/sigma) if m > 0, else 0

    Flux and mask are unchanged. rho falls to 0 as sigma grows, so a pixel
    without information reads like a masked one. Apply it after
    :func:`zero_invalid_pixels` and :func:`normalize_spectra`, which rescales
    sigma. A valid pixel needs sigma > 0; a masked pixel may hold any value,
    including NaN.

    Args:
        spectra: Model inputs of shape (N, 3, n_wavelengths) = [flux, sigma,
            mask].
        inplace: Write the result into ``spectra`` (must be float32) instead
            of a new array.

    Returns:
        Float32 array of shape (N, 3, n_wavelengths) = [flux, rho, mask].

    Raises:
        ValueError: If ``spectra`` does not have shape (N, 3, n_wavelengths),
            has sigma <= 0 or NaN at a valid pixel, or ``inplace`` is set
            and it is not float32.
    """
    spectra = np.asarray(spectra)
    if spectra.ndim != 3 or spectra.shape[1] != 3:
        raise ValueError(
            f"spectra must have shape (N, 3, n_wavelengths), got {spectra.shape}"
        )
    for start in range(0, len(spectra), _BLOCK_ROWS):
        block = spectra[start : start + _BLOCK_ROWS]
        if not (block[:, 1][block[:, 2] > 0] > 0).all():
            raise ValueError("spectra must have sigma > 0 at every valid pixel")
    if not inplace:
        spectra = spectra.astype(np.float32)
    elif spectra.dtype != np.float32:
        raise ValueError("inplace requires a float32 spectra array")
    for start in range(0, len(spectra), _BLOCK_ROWS):
        block = spectra[start : start + _BLOCK_ROWS]
        _encode_sigma(block[:, 1], block[:, 2] > 0)
    return spectra


def normalize_spectrum(flux: NDArray, ivar: NDArray) -> tuple[NDArray, NDArray]:
    """
    Normalize a single spectrum using median/IQR scaling.

    Calls :func:`normalize_spectra` on one spectrum.

    Args:
        flux: Flux array of shape (n_wavelengths,).
        ivar: Inverse variance array of shape (n_wavelengths,).

    Returns:
        Tuple of (normalized_flux, scaled_ivar), float32.

    Raises:
        ValueError: If flux and ivar differ in shape.
    """
    return normalize_spectra(flux, ivar)


def ivar_to_error_and_mask(
    ivar: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Convert inverse variance to error (sigma) and mask.

    Args:
        ivar: Inverse variance array of shape (n_wavelengths,) or (n_samples, n_wavelengths).
            Can contain NaN, Inf, or <= 0 values indicating invalid data.

    Returns:
        Tuple of (error, mask) where:
        - error: Standard deviation (sigma = 1/sqrt(ivar)) where valid, else 0
        - mask: Binary mask (1=valid, 0=masked)
    """
    # Create mask: valid where ivar is finite and positive
    valid = np.isfinite(ivar) & (ivar > 0)
    mask = valid.astype(np.float32)

    # Compute error where valid
    error = np.zeros_like(ivar, dtype=np.float32)
    error[valid] = 1.0 / np.sqrt(ivar[valid])

    return error, mask


def apply_quality_filter(
    labels: NDArray,
    errors: NDArray,
    parameter_names: list[str],
) -> NDArray[np.bool_]:
    """
    Create a quality mask based on physical bounds and valid values.

    A star passes if, for every parameter, the value and error are finite,
    the error is positive, the value lies within ``PARAMETER_BOUNDS`` (if the
    parameter has bounds) and the value is nonzero (except ``fe_h``).

    Args:
        labels: Array of shape (n_samples, n_parameters).
        errors: Array of shape (n_samples, n_parameters).
        parameter_names: Parameter name of each column.

    Returns:
        Boolean mask of shape (n_samples,) where True = good quality.
    """
    n_samples = labels.shape[0]
    mask = np.ones(n_samples, dtype=bool)

    for i, param_name in enumerate(parameter_names):
        values = labels[:, i]
        errs = errors[:, i]

        # Check for NaN/Inf in values and errors
        mask &= np.isfinite(values)
        mask &= np.isfinite(errs)
        mask &= errs > 0

        # Check physical bounds
        if param_name in PARAMETER_BOUNDS:
            low, high = PARAMETER_BOUNDS[param_name]
            mask &= (values >= low) & (values <= high)

        # Special check for zero values (often indicates missing data)
        if param_name != "fe_h":  # [Fe/H] can legitimately be 0
            mask &= values != 0

    return mask


class FITSLoader:
    """
    Loader for FITS files containing stellar spectra and parameters.

    Reads FITS files in the DOROTHY format, normalizes spectra and computes a
    quality mask from the label values and the APOGEE ``ASPCAPFLAG`` bits.

    Attributes:
        fits_path: Path to the FITS file.
        parameters: List of stellar parameter names to load.

    Examples:
        >>> loader = FITSLoader(Path("data/loa_trainingset.fits"))
        >>> data = loader.load()
        >>> X = data.get_model_input()  # Shape: (n_good_stars, 2, 7650)
        >>> y = data.get_combined_labels()  # Shape: (n_good_stars, 22)
    """

    def __init__(
        self,
        fits_path: Path | str,
        parameters: list[str] | None = None,
    ) -> None:
        """
        Initialize the FITS loader.

        Args:
            fits_path: Path to the FITS file.
            parameters: Parameter names to load, keys of
                ``APOGEE_COLUMN_MAP``. If None, loads all of them in the order
                of the map.
        """
        self.fits_path = Path(fits_path)

        if parameters is None:
            parameters = list(APOGEE_COLUMN_MAP.keys())
        self.parameters = parameters

    @classmethod
    def from_config(cls, config: DataConfig) -> FITSLoader:
        """
        Create a loader from a DataConfig.

        Args:
            config: Data configuration object.

        Returns:
            Configured FITSLoader instance.
        """
        return cls(fits_path=config.fits_path)

    def load(
        self,
        apply_normalization: bool = True,
        apply_quality_filter: bool = True,
        compute_spectral_mask: bool = False,
        require_labels: bool = True,
    ) -> SpectralData:
        """
        Load spectral data from the FITS file.

        Args:
            apply_normalization: Whether to normalize spectra (recommended).
                Either way, invalid pixels get flux 0 and ivar 0.
            apply_quality_filter: Whether to compute quality mask.
            compute_spectral_mask: Whether to compute spectral_mask and
                spectral_error from ivar. Required for 3-channel format.
            require_labels: Require an APOGEE extension with all label
                columns. When False (e.g. for prediction), missing labels are
                returned as NaN and the quality mask keeps every spectrum.

        Returns:
            SpectralData object containing all loaded data. ``ids`` come from
            the APOGEE_ID or TARGET_ID column; without an APOGEE extension,
            from the first column of a GAIA_ID extension; else they are the
            row numbers. ``wavelength`` is
            ``arange(n_wavelengths)`` without a WAVELENGTH extension.

        Raises:
            FileNotFoundError: If FITS file doesn't exist.
            ValueError: If a required extension is missing, or, with
                require_labels, a parameter is not in ``APOGEE_COLUMN_MAP`` or
                its columns are not in the APOGEE table.
        """
        if not self.fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {self.fits_path}")

        # Import astropy here to avoid import-time dependency
        from astropy.io import fits
        from astropy.table import Table

        with fits.open(self.fits_path) as hdul:
            # Validate required extensions
            required_extensions = ["SPECTRA", "SPEC_IVAR"]
            if require_labels:
                required_extensions.insert(0, "APOGEE")
            for ext in required_extensions:
                if ext not in hdul:
                    raise ValueError(f"Missing required extension: {ext}")

            # Load APOGEE data (optional when labels aren't required)
            apogee_table = Table(hdul["APOGEE"].data) if "APOGEE" in hdul else None
            gaia_ids = None
            if apogee_table is None and "GAIA_ID" in hdul:
                gaia_table = hdul["GAIA_ID"].data
                gaia_ids = np.array(gaia_table[gaia_table.columns.names[0]]).astype(str)

            # Load spectra data
            spectra_table = Table(hdul["SPECTRA"].data)
            ivar_table = Table(hdul["SPEC_IVAR"].data)

            # Load wavelength if available
            if "WAVELENGTH" in hdul:
                wavelength_table = Table(hdul["WAVELENGTH"].data)
                wavelength = np.array(wavelength_table["WAVELENGTH"], dtype=np.float64)
            else:
                # Create placeholder wavelength array
                n_wavelengths = len(spectra_table["FLUX"][0])
                wavelength = np.arange(n_wavelengths, dtype=np.float64)

        # Extract spectra
        n_samples = len(spectra_table)
        n_wavelengths = len(spectra_table["FLUX"][0])

        flux_array = np.zeros((n_samples, n_wavelengths), dtype=np.float32)
        ivar_array = np.zeros((n_samples, n_wavelengths), dtype=np.float32)

        flux_col = np.asarray(spectra_table["FLUX"])
        ivar_col = np.asarray(ivar_table["IVAR"])
        if flux_col.shape == ivar_col.shape == flux_array.shape and all(
            col.dtype.kind in "biuf" for col in (flux_col, ivar_col)
        ):
            flux_array[...] = flux_col
            ivar_array[...] = ivar_col
        else:
            # Irregular columns (e.g. variable-length arrays): row by row
            for i in range(n_samples):
                flux_array[i] = spectra_table["FLUX"][i]
                ivar_array[i] = ivar_table["IVAR"][i]
        # Spectra are processed independently, so one batched call gives the
        # same result as one call per spectrum
        if apply_normalization:
            normalize_spectra(flux_array, ivar_array, inplace=True)
        else:
            zero_invalid_pixels(flux_array, ivar_array, inplace=True)

        # Extract labels and errors
        has_labels = True
        try:
            if apogee_table is None:
                raise ValueError("No APOGEE extension")
            labels, errors = self._extract_labels(apogee_table)
        except ValueError:
            if require_labels:
                raise
            has_labels = False
            labels = np.full((n_samples, len(self.parameters)), np.nan, np.float32)
            errors = np.full_like(labels, np.nan)

        # Extract IDs
        names = apogee_table.colnames if apogee_table is not None else []
        if "APOGEE_ID" in names:
            ids = np.array(apogee_table["APOGEE_ID"], dtype=str)
        elif "TARGET_ID" in names:
            ids = np.array(apogee_table["TARGET_ID"], dtype=str)
        elif gaia_ids is not None and len(gaia_ids) == n_samples:
            ids = gaia_ids
        else:
            ids = np.array([str(i) for i in range(n_samples)])

        # Compute quality mask (needs labels)
        if apply_quality_filter and has_labels:
            quality_mask = self._compute_quality_mask(labels, errors, apogee_table)
        else:
            quality_mask = np.ones(n_samples, dtype=bool)

        # Compute spectral mask and error if requested
        spectral_mask = None
        spectral_error = None
        if compute_spectral_mask:
            spectral_error, spectral_mask = ivar_to_error_and_mask(ivar_array)

        return SpectralData(
            flux=flux_array,
            ivar=ivar_array,
            wavelength=wavelength,
            labels=labels,
            errors=errors,
            ids=ids,
            quality_mask=quality_mask,
            parameter_names=self.parameters,
            spectral_mask=spectral_mask,
            spectral_error=spectral_error,
        )

    def _extract_labels(
        self,
        apogee_table,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Extract stellar parameter labels and errors from APOGEE table."""
        n_samples = len(apogee_table)
        n_params = len(self.parameters)

        labels = np.zeros((n_samples, n_params), dtype=np.float32)
        errors = np.zeros((n_samples, n_params), dtype=np.float32)

        for i, param_name in enumerate(self.parameters):
            if param_name not in APOGEE_COLUMN_MAP:
                raise ValueError(f"Unknown parameter: {param_name}")

            label_col, err_col = APOGEE_COLUMN_MAP[param_name]

            if label_col not in apogee_table.colnames:
                raise ValueError(f"Column {label_col} not found in APOGEE table")
            if err_col not in apogee_table.colnames:
                raise ValueError(f"Column {err_col} not found in APOGEE table")

            labels[:, i] = np.array(apogee_table[label_col], dtype=np.float32)
            errors[:, i] = np.array(apogee_table[err_col], dtype=np.float32)

        return labels, errors

    def _compute_quality_mask(
        self,
        labels: NDArray,
        errors: NDArray,
        apogee_table,
    ) -> NDArray[np.bool_]:
        """
        Compute quality mask based on values and APOGEE flags.

        This combines:
        1. Basic value/error validation (NaN, Inf, physical bounds)
        2. APOGEE pipeline quality flags (PARAMFLAG, ASPCAPFLAG)
        """
        # Start with basic quality filter
        mask = apply_quality_filter(labels, errors, self.parameters)

        # Apply APOGEE flags if available
        if "ASPCAPFLAG" in apogee_table.colnames:
            aspcap_flags = np.array(apogee_table["ASPCAPFLAG"])

            # Critical ASPCAPFLAG bits
            star_bad = 1 << 23
            teff_bad = 1 << 16
            logg_bad = 1 << 17

            # Reject stars with these flags
            bad_flags = star_bad | teff_bad | logg_bad
            mask &= (aspcap_flags & bad_flags) == 0

        return mask
