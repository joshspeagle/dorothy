"""
FITS file loader for stellar spectra and labels.

This module handles loading spectroscopic data from FITS files containing
DESI, BOSS, or LAMOST data cross-matched with APOGEE stellar parameters.

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
    "feh": ("FE_H", "FE_H_ERR"),
    "mgfe": ("MG_FE", "MG_FE_ERR"),
    "cfe": ("C_FE", "C_FE_ERR"),
    "sife": ("SI_FE", "SI_FE_ERR"),
    "nife": ("NI_FE", "NI_FE_ERR"),
    "alfe": ("AL_FE", "AL_FE_ERR"),
    "cafe": ("CA_FE", "CA_FE_ERR"),
    "nfe": ("N_FE", "N_FE_ERR"),
    "mnfe": ("MN_FE", "MN_FE_ERR"),
}

# Physical bounds for quality filtering
PARAMETER_BOUNDS = {
    "teff": (2500, 10000),  # Kelvin
    "logg": (-1.0, 6.0),  # log cm/s^2
    "feh": (-5.0, 1.0),  # dex
    "mgfe": (-2.0, 2.0),  # dex
    "cfe": (-2.0, 2.0),
    "sife": (-2.0, 2.0),
    "nife": (-2.0, 2.0),
    "alfe": (-2.0, 2.0),
    "cafe": (-2.0, 2.0),
    "nfe": (-2.0, 2.0),
    "mnfe": (-2.0, 2.0),
}


@dataclass
class SpectralData:
    """
    Container for loaded spectral data.

    Attributes:
        flux: Normalized flux array of shape (n_samples, n_wavelengths).
        ivar: Scaled inverse variance array of shape (n_samples, n_wavelengths).
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
        """Get labels, optionally filtered by quality mask."""
        if apply_quality_mask:
            return self.labels[self.quality_mask]
        return self.labels

    def get_errors(self, apply_quality_mask: bool = True) -> NDArray[np.float32]:
        """Get errors, optionally filtered by quality mask."""
        if apply_quality_mask:
            return self.errors[self.quality_mask]
        return self.errors

    def get_combined_labels(
        self, apply_quality_mask: bool = True
    ) -> NDArray[np.float32]:
        """
        Get labels and errors combined for training.

        Returns:
            Array of shape (n_samples, 2 * n_parameters) where the first half
            contains labels and the second half contains errors.
        """
        labels = self.get_labels(apply_quality_mask)
        errors = self.get_errors(apply_quality_mask)
        return np.concatenate([labels, errors], axis=1).astype(np.float32)


def normalize_spectrum(flux: NDArray, ivar: NDArray) -> tuple[NDArray, NDArray]:
    """
    Normalize a single spectrum using median/IQR scaling.

    This applies per-star normalization:
        normalized_flux = (flux - median) / IQR
        scaled_ivar = IQR^2 * ivar

    Args:
        flux: Flux array of shape (n_wavelengths,).
        ivar: Inverse variance array of shape (n_wavelengths,).

    Returns:
        Tuple of (normalized_flux, scaled_ivar).
    """
    # Handle NaN/Inf in ivar
    ivar = np.where(np.isfinite(ivar), ivar, 0.0)

    # Compute normalization parameters
    median = np.median(flux)
    q25, q75 = np.percentile(flux, [25, 75])
    iqr = q75 - q25

    # Avoid division by zero
    if iqr <= 0:
        iqr = 1.0

    normalized_flux = (flux - median) / iqr
    scaled_ivar = iqr**2 * ivar

    return normalized_flux, scaled_ivar


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

    A star passes the quality filter if ALL parameters:
    - Are not NaN or Inf
    - Are within physical bounds
    - Have non-zero, non-NaN errors

    Args:
        labels: Array of shape (n_samples, n_parameters).
        errors: Array of shape (n_samples, n_parameters).
        parameter_names: List of parameter names matching columns.

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
        if param_name != "feh":  # [Fe/H] can legitimately be 0
            mask &= values != 0

    return mask


class FITSLoader:
    """
    Loader for FITS files containing stellar spectra and parameters.

    This class reads FITS files in the DOROTHY format, normalizes spectra,
    and applies quality filtering based on APOGEE flags.

    Attributes:
        fits_path: Path to the FITS file.
        parameters: List of stellar parameter names to load.

    Example:
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
            parameters: List of parameter names to load. If None, loads all
                11 stellar parameters in the default order.
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
    ) -> SpectralData:
        """
        Load spectral data from the FITS file.

        Args:
            apply_normalization: Whether to normalize spectra (recommended).
            apply_quality_filter: Whether to compute quality mask.
            compute_spectral_mask: Whether to compute spectral_mask and
                spectral_error from ivar. Required for 3-channel format.

        Returns:
            SpectralData object containing all loaded data.

        Raises:
            FileNotFoundError: If FITS file doesn't exist.
            ValueError: If required extensions are missing.
        """
        if not self.fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {self.fits_path}")

        # Import astropy here to avoid import-time dependency
        from astropy.io import fits
        from astropy.table import Table

        with fits.open(self.fits_path) as hdul:
            # Validate required extensions
            required_extensions = ["APOGEE", "SPECTRA", "SPEC_IVAR"]
            for ext in required_extensions:
                if ext not in hdul:
                    raise ValueError(f"Missing required extension: {ext}")

            # Load APOGEE data
            apogee_table = Table(hdul["APOGEE"].data)

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

        for i in range(n_samples):
            flux = np.array(spectra_table["FLUX"][i], dtype=np.float32)
            ivar = np.array(ivar_table["IVAR"][i], dtype=np.float32)

            if apply_normalization:
                flux, ivar = normalize_spectrum(flux, ivar)

            flux_array[i] = flux
            ivar_array[i] = ivar

        # Extract labels and errors
        labels, errors = self._extract_labels(apogee_table)

        # Extract IDs
        if "APOGEE_ID" in apogee_table.colnames:
            ids = np.array(apogee_table["APOGEE_ID"], dtype=str)
        elif "TARGET_ID" in apogee_table.colnames:
            ids = np.array(apogee_table["TARGET_ID"], dtype=str)
        else:
            ids = np.array([str(i) for i in range(n_samples)])

        # Compute quality mask
        if apply_quality_filter:
            quality_mask = self._compute_quality_mask(labels, errors, apogee_table)
        else:
            quality_mask = np.ones(n_samples, dtype=bool)

        # Compute spectral mask and error if requested
        spectral_mask = None
        spectral_error = None
        if compute_spectral_mask:
            spectral_error, spectral_mask = ivar_to_error_and_mask(ivar_array)
            # Zero out flux at masked wavelengths for consistency
            flux_array = flux_array * spectral_mask

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


def split_data(
    data: SpectralData,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
    apply_quality_mask: bool = True,
) -> tuple[
    tuple[NDArray, NDArray],
    tuple[NDArray, NDArray],
    tuple[NDArray, NDArray],
]:
    """
    Split data into train, validation, and test sets.

    Args:
        data: SpectralData object to split.
        train_ratio: Fraction for training (default 0.7).
        val_ratio: Fraction for validation (default 0.2).
        seed: Random seed for reproducibility.
        apply_quality_mask: Whether to filter by quality first.

    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test)).
        Each X has shape (n, 2, wavelengths), each y has shape (n, 2*n_params).
    """
    X = data.get_model_input(apply_quality_mask)
    y = data.get_combined_labels(apply_quality_mask)

    n_samples = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return (
        (X[train_idx], y[train_idx]),
        (X[val_idx], y[val_idx]),
        (X[test_idx], y[test_idx]),
    )
