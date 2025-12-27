"""
Loader for the DOROTHY super-catalogue HDF5 format.

This module provides utilities to load the unified multi-survey super-catalogue,
enabling training across multiple surveys with consistent label handling.

Example:
    >>> loader = CatalogueLoader("/path/to/super_catalogue.h5")
    >>> loader.info()  # Print catalogue summary
    >>> data = loader.load(surveys=["desi", "boss"], label_source="apogee")
    >>> print(f"Loaded {len(data.gaia_ids)} stars")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import h5py
import numpy as np


# 11 stellar parameters in standard order
PARAMETER_NAMES = [
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
class CatalogueData:
    """
    Container for loaded catalogue data.

    This is the output from CatalogueLoader.load(), containing spectral data
    from one or more surveys and labels from a selected source.

    Attributes:
        gaia_ids: Gaia DR3 source IDs for each star.
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        flux: Spectral flux values, shape (N, wavelength_bins).
        ivar: Inverse variance, shape (N, wavelength_bins).
        wavelength: Wavelength grid, shape (wavelength_bins,).
        snr: Signal-to-noise ratio for each spectrum.
        labels: Stellar parameter values, shape (N, 11).
        label_errors: Uncertainties on labels, shape (N, 11).
        label_flags: Per-parameter quality flags, shape (N, 11).
        survey_name: Name of the survey (for single-survey loading).
        label_source: Name of the label source used.
        has_spectrum: Boolean mask for stars with valid spectra.
        has_labels: Boolean mask for stars with valid labels.
    """

    gaia_ids: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    flux: np.ndarray
    ivar: np.ndarray
    wavelength: np.ndarray
    snr: np.ndarray
    labels: np.ndarray
    label_errors: np.ndarray
    label_flags: np.ndarray
    survey_name: str
    label_source: str
    has_spectrum: np.ndarray = field(default_factory=lambda: np.array([]))
    has_labels: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        """Compute derived masks if not provided."""
        if len(self.has_spectrum) == 0:
            self.has_spectrum = np.any(self.ivar > 0, axis=1)
        if len(self.has_labels) == 0:
            self.has_labels = np.any(self.label_errors > 0, axis=1)

    @property
    def n_stars(self) -> int:
        """Total number of stars in the dataset."""
        return len(self.gaia_ids)

    @property
    def n_with_spectra(self) -> int:
        """Number of stars with valid spectra."""
        return int(self.has_spectrum.sum())

    @property
    def n_with_labels(self) -> int:
        """Number of stars with valid labels."""
        return int(self.has_labels.sum())

    @property
    def n_complete(self) -> int:
        """Number of stars with both spectra and labels."""
        return int((self.has_spectrum & self.has_labels).sum())

    def filter_complete(self) -> CatalogueData:
        """Return a new CatalogueData with only complete (spectra + labels) stars."""
        mask = self.has_spectrum & self.has_labels
        return CatalogueData(
            gaia_ids=self.gaia_ids[mask],
            ra=self.ra[mask],
            dec=self.dec[mask],
            flux=self.flux[mask],
            ivar=self.ivar[mask],
            wavelength=self.wavelength,
            snr=self.snr[mask],
            labels=self.labels[mask],
            label_errors=self.label_errors[mask],
            label_flags=self.label_flags[mask],
            survey_name=self.survey_name,
            label_source=self.label_source,
            has_spectrum=self.has_spectrum[mask],
            has_labels=self.has_labels[mask],
        )

    def filter_by_flags(self, max_flag_bits: int = 0) -> CatalogueData:
        """
        Filter stars based on label quality flags.

        Args:
            max_flag_bits: Maximum number of set bits allowed in flags.
                0 = only stars with no flags set (highest quality).

        Returns:
            New CatalogueData with filtered stars.
        """
        # Count bits set in each flag
        flag_bits = np.zeros(len(self.gaia_ids), dtype=int)
        for i in range(11):
            flag_bits += np.bitwise_count(self.label_flags[:, i].astype(np.uint64))

        mask = flag_bits <= max_flag_bits
        return CatalogueData(
            gaia_ids=self.gaia_ids[mask],
            ra=self.ra[mask],
            dec=self.dec[mask],
            flux=self.flux[mask],
            ivar=self.ivar[mask],
            wavelength=self.wavelength,
            snr=self.snr[mask],
            labels=self.labels[mask],
            label_errors=self.label_errors[mask],
            label_flags=self.label_flags[mask],
            survey_name=self.survey_name,
            label_source=self.label_source,
            has_spectrum=self.has_spectrum[mask],
            has_labels=self.has_labels[mask],
        )


@dataclass
class CatalogueInfo:
    """Summary information about a catalogue."""

    n_stars: int
    parameter_names: list[str]
    survey_names: list[str]
    creation_date: str
    version: str
    surveys: dict[str, dict]
    labels: dict[str, dict]


class CatalogueLoader:
    """
    Loader for DOROTHY super-catalogue HDF5 files.

    This class provides methods to inspect and load data from the unified
    super-catalogue format. It supports loading from single or multiple
    surveys and selecting between APOGEE and GALAH labels.

    Example:
        >>> loader = CatalogueLoader("super_catalogue.h5")
        >>> loader.info()

        >>> # Load DESI spectra with APOGEE labels
        >>> data = loader.load(surveys="desi", label_source="apogee")
        >>> complete = data.filter_complete()  # Only stars with both
        >>> print(f"Training set: {complete.n_stars} stars")

        >>> # Load multiple surveys
        >>> data = loader.load(surveys=["desi", "boss"], label_source="apogee")
    """

    def __init__(self, catalogue_path: str | Path):
        """
        Initialize the catalogue loader.

        Args:
            catalogue_path: Path to the HDF5 super-catalogue file.

        Raises:
            FileNotFoundError: If the catalogue file does not exist.
        """
        self.path = Path(catalogue_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Catalogue not found: {self.path}")

    def get_info(self) -> CatalogueInfo:
        """
        Get summary information about the catalogue.

        Returns:
            CatalogueInfo object with catalogue metadata.
        """
        with h5py.File(self.path, "r") as f:
            # Read attributes
            n_stars = f.attrs["n_stars"]
            parameter_names = list(f.attrs["parameter_names"])
            survey_names = list(f.attrs["survey_names"])
            creation_date = f.attrs["creation_date"]
            version = f.attrs["version"]

            # Get survey info
            surveys = {}
            if "surveys" in f:
                for name in f["surveys"]:
                    grp = f["surveys"][name]
                    n_wave = grp["wavelength"].shape[0]
                    snr = grp["snr"][:]
                    n_with_data = int((snr > 0).sum())
                    surveys[name] = {
                        "wavelength_bins": n_wave,
                        "n_with_data": n_with_data,
                        "wavelength_range": (
                            float(grp["wavelength"][0]),
                            float(grp["wavelength"][-1]),
                        ),
                    }

            # Get label info
            labels = {}
            if "labels" in f:
                for name in f["labels"]:
                    grp = f["labels"][name]
                    errors = grp["errors"][:]
                    n_with_data = int(np.any(errors > 0, axis=1).sum())
                    labels[name] = {
                        "n_with_data": n_with_data,
                    }

        return CatalogueInfo(
            n_stars=n_stars,
            parameter_names=parameter_names,
            survey_names=survey_names,
            creation_date=creation_date,
            version=version,
            surveys=surveys,
            labels=labels,
        )

    def info(self) -> None:
        """Print a summary of the catalogue to stdout."""
        info = self.get_info()

        print(f"DOROTHY Super-Catalogue v{info.version}")
        print(f"Created: {info.creation_date}")
        print(f"Total stars: {info.n_stars:,}")
        print()

        print("Surveys:")
        for name, survey in info.surveys.items():
            print(f"  {name}:")
            print(f"    Stars with data: {survey['n_with_data']:,}")
            print(f"    Wavelength bins: {survey['wavelength_bins']}")
            print(
                f"    Wavelength range: {survey['wavelength_range'][0]:.1f} - "
                f"{survey['wavelength_range'][1]:.1f} A"
            )
        print()

        print("Label sources:")
        for name, label in info.labels.items():
            print(f"  {name}: {label['n_with_data']:,} stars")
        print()

        print(f"Parameters: {', '.join(info.parameter_names)}")

    def load(
        self,
        surveys: str | list[str] | None = None,
        label_source: Literal["apogee", "galah"] = "apogee",
        require_complete: bool = False,
    ) -> CatalogueData:
        """
        Load data from the catalogue.

        Args:
            surveys: Survey name(s) to load. If None, loads from first available.
                For multi-survey loading, pass a list.
            label_source: Which label source to use ("apogee" or "galah").
            require_complete: If True, only return stars with both spectra and labels.

        Returns:
            CatalogueData object with loaded data.

        Raises:
            ValueError: If specified survey or label source not in catalogue.

        Note:
            For multi-survey loading, this currently only supports loading from
            a single survey at a time. Multi-survey stacking is not yet implemented.
        """
        if surveys is None:
            info = self.get_info()
            for name in info.survey_names:
                if name in info.surveys and info.surveys[name]["n_with_data"] > 0:
                    surveys = name
                    break
            if surveys is None:
                raise ValueError("No surveys with data found in catalogue")

        # Handle list input (for now, just use first)
        if isinstance(surveys, list):
            if len(surveys) > 1:
                # Multi-survey loading could be implemented here
                pass
            surveys = surveys[0]

        with h5py.File(self.path, "r") as f:
            # Validate inputs
            if surveys not in f["surveys"]:
                raise ValueError(
                    f"Survey '{surveys}' not in catalogue. "
                    f"Available: {list(f['surveys'].keys())}"
                )
            if label_source not in f["labels"]:
                raise ValueError(
                    f"Label source '{label_source}' not in catalogue. "
                    f"Available: {list(f['labels'].keys())}"
                )

            # Load metadata
            gaia_ids = f["metadata"]["gaia_id"][:]
            ra = f["metadata"]["ra"][:]
            dec = f["metadata"]["dec"][:]

            # Load survey data
            survey_grp = f["surveys"][surveys]
            flux = survey_grp["flux"][:]
            ivar = survey_grp["ivar"][:]
            wavelength = survey_grp["wavelength"][:]
            snr = survey_grp["snr"][:]

            # Load labels
            label_grp = f["labels"][label_source]
            labels = label_grp["values"][:]
            label_errors = label_grp["errors"][:]
            label_flags = label_grp["flags"][:]

        data = CatalogueData(
            gaia_ids=gaia_ids,
            ra=ra,
            dec=dec,
            flux=flux,
            ivar=ivar,
            wavelength=wavelength,
            snr=snr,
            labels=labels,
            label_errors=label_errors,
            label_flags=label_flags,
            survey_name=surveys,
            label_source=label_source,
        )

        if require_complete:
            data = data.filter_complete()

        return data

    def load_for_training(
        self,
        survey: str,
        label_source: Literal["apogee", "galah"] = "apogee",
        max_flag_bits: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data ready for training.

        This is a convenience method that loads complete, high-quality data
        and returns it in the format expected by the DOROTHY trainer.

        Args:
            survey: Survey to load from.
            label_source: Label source to use.
            max_flag_bits: Maximum flag bits allowed (0 = highest quality).

        Returns:
            Tuple of (X, y, y_err, mask) where:
                - X: Spectral data, shape (N, 2, wavelength_bins)
                - y: Labels, shape (N, 11)
                - y_err: Label errors, shape (N, 11)
                - mask: Parameter mask, shape (N, 11) - True where valid
        """
        data = self.load(
            surveys=survey, label_source=label_source, require_complete=True
        )

        if max_flag_bits >= 0:
            data = data.filter_by_flags(max_flag_bits)

        # Stack flux and ivar into 2-channel format
        X = np.stack([data.flux, data.ivar], axis=1)

        # Create mask from flags and errors
        mask = (data.label_flags == 0) & (data.label_errors > 0)

        return X, data.labels, data.label_errors, mask

    def get_survey_wavelengths(self, survey: str) -> np.ndarray:
        """Get the wavelength grid for a survey."""
        with h5py.File(self.path, "r") as f:
            if survey not in f["surveys"]:
                raise ValueError(f"Survey '{survey}' not in catalogue")
            return f["surveys"][survey]["wavelength"][:]

    def get_parameter_names(self) -> list[str]:
        """Get the list of parameter names."""
        with h5py.File(self.path, "r") as f:
            return list(f.attrs["parameter_names"])
