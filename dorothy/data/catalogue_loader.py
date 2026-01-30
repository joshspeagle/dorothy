"""
Loader for the DOROTHY super-catalogue HDF5 format.

This module provides utilities to load the unified multi-survey super-catalogue,
enabling training across multiple surveys with consistent label handling.

Each survey in the catalogue has its own independent row structure and
corresponding labels. The loader automatically maps surveys to their
appropriate label sources.

Example:
    >>> loader = CatalogueLoader("/path/to/super_catalogue.h5")
    >>> loader.info()  # Print catalogue summary
    >>> data = loader.load(survey="boss")  # Loads BOSS with APOGEE labels
    >>> data = loader.load(survey="lamost_lrs")  # Loads LAMOST LRS with its labels
    >>> print(f"Loaded {len(data.gaia_ids)} stars")
"""

from __future__ import annotations

import gc
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

# Mapping from survey names to their label group names (v1 schema)
# For v2 schema, labels are stored directly as /labels/apogee/ and /labels/galah/
# The loader automatically detects schema version and uses appropriate label groups.
SURVEY_LABEL_MAP = {
    "boss": "apogee_boss",
    "lamost_lrs": "apogee_lamost_lrs",
    "lamost_mrs": "apogee_lamost_mrs",
    "desi": "apogee_desi",
}

# Available label sources (for v2 schema)
LABEL_SOURCES = ["apogee", "galah"]


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
class MergedCatalogueData:
    """
    Container for merged multi-survey data with outer join structure.

    This dataclass stores data from multiple surveys aligned by Gaia DR3 ID.
    Each survey's data is stored in a dictionary keyed by survey name, with
    missing data indicated by ivar=0 (natural masking).

    Attributes:
        gaia_ids: Gaia DR3 source IDs for each unique star (union of all surveys).
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        surveys: Dictionary mapping survey names to their data dictionaries.
            Each entry contains: 'flux', 'ivar', 'wavelength', 'snr', 'labels',
            'label_errors', 'label_flags', 'has_data' (bool mask).
        label_source: Label source used (typically 'apogee').
        primary_labels: Primary labels array, shape (N, 11), from best available source.
        primary_errors: Primary label errors array, shape (N, 11).
        primary_flags: Primary label flags array, shape (N, 11).

    Example:
        >>> merged = loader.load_merged(surveys=["boss", "lamost_lrs"])
        >>> # Access BOSS spectra for all stars (missing = ivar=0)
        >>> boss_flux = merged.surveys["boss"]["flux"]  # (N, wavelengths)
        >>> boss_ivar = merged.surveys["boss"]["ivar"]  # 0 where missing
        >>> # Check which stars have BOSS data
        >>> has_boss = merged.surveys["boss"]["has_data"]  # bool array
    """

    gaia_ids: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    surveys: dict[str, dict[str, np.ndarray]]
    label_source: str
    primary_labels: np.ndarray
    primary_errors: np.ndarray
    primary_flags: np.ndarray

    @property
    def n_stars(self) -> int:
        """Total number of unique stars."""
        return len(self.gaia_ids)

    @property
    def survey_names(self) -> list[str]:
        """List of survey names in this merged dataset."""
        return list(self.surveys.keys())

    def get_coverage_stats(self) -> dict[str, int]:
        """Get the number of stars with data from each survey."""
        return {
            survey: int(data["has_data"].sum()) for survey, data in self.surveys.items()
        }

    def get_overlap_matrix(self) -> dict[tuple[str, str], int]:
        """Get pairwise overlap counts between surveys."""
        overlaps = {}
        survey_list = list(self.surveys.keys())
        for i, s1 in enumerate(survey_list):
            for s2 in survey_list[i + 1 :]:
                has_both = self.surveys[s1]["has_data"] & self.surveys[s2]["has_data"]
                overlaps[(s1, s2)] = int(has_both.sum())
        return overlaps

    def filter_by_surveys(self, required_surveys: list[str]) -> MergedCatalogueData:
        """
        Return a new MergedCatalogueData with only stars that have data
        from all specified surveys.

        Args:
            required_surveys: List of survey names that must have data.

        Returns:
            Filtered MergedCatalogueData.
        """
        mask = np.ones(self.n_stars, dtype=bool)
        for survey in required_surveys:
            if survey not in self.surveys:
                raise ValueError(f"Survey '{survey}' not in merged data")
            mask &= self.surveys[survey]["has_data"]

        new_surveys = {}
        for survey, data in self.surveys.items():
            new_surveys[survey] = {
                key: arr[mask] if key != "wavelength" else arr
                for key, arr in data.items()
            }

        return MergedCatalogueData(
            gaia_ids=self.gaia_ids[mask],
            ra=self.ra[mask],
            dec=self.dec[mask],
            surveys=new_surveys,
            label_source=self.label_source,
            primary_labels=self.primary_labels[mask],
            primary_errors=self.primary_errors[mask],
            primary_flags=self.primary_flags[mask],
        )


@dataclass
class SparseMergedData:
    """
    Memory-efficient multi-survey data with sparse survey storage.

    Unlike MergedCatalogueData which stores dense arrays for each survey
    (with zeros for missing stars), this class stores only stars that have
    actual data for each survey. Index mappings allow efficient lookup
    between global star indices and survey-local indices.

    This reduces memory usage significantly for surveys with low coverage:
    - DESI: 0.8% coverage → stores only 1.3K rows instead of 155K
    - BOSS: 7.4% coverage → stores only 11K rows instead of 155K

    Attributes:
        flux: Per-survey sparse spectra. Maps survey name to array of shape
            (n_with_data[survey], n_wavelengths) containing only stars with data.
        ivar: Per-survey inverse variance arrays with same structure as flux.
        wavelengths: Per-survey wavelength grids (n_wavelengths,).
        snr: Per-survey SNR values for stars with data.
        global_to_local: Maps survey name to index array (n_total,) where
            value = local index into sparse array, or -1 if no data.
        local_to_global: Maps survey name to index array (n_with_data,) where
            value = global star index.
        labels: Dense labels array (n_total, 3, n_params) with [values, errors, mask].
        gaia_ids: Gaia DR3 source IDs for all stars (n_total,).
        ra: Right ascension in degrees (n_total,).
        dec: Declination in degrees (n_total,).
        surveys: List of survey names.
        n_total: Total number of unique stars across all surveys.
        n_params: Number of stellar parameters (typically 11).
    """

    flux: dict[str, np.ndarray]
    ivar: dict[str, np.ndarray]
    wavelengths: dict[str, np.ndarray]
    snr: dict[str, np.ndarray]
    global_to_local: dict[str, np.ndarray]
    local_to_global: dict[str, np.ndarray]
    labels: np.ndarray
    gaia_ids: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    surveys: list[str]
    n_total: int
    n_params: int

    # Multi-label support (optional) - for training with multiple label sources
    labels_dict: dict[str, np.ndarray] | None = None  # {source: (n_total, 3, n_params)}
    has_labels_dict: dict[str, np.ndarray] | None = None  # {source: (n_total,) bool}
    label_sources: list[str] | None = None  # List of label source names

    def has_data(self, survey: str) -> np.ndarray:
        """Return boolean mask of which global indices have data for survey."""
        return self.global_to_local[survey] >= 0

    def coverage(self, survey: str) -> float:
        """Return fraction of stars with data for this survey."""
        return np.sum(self.global_to_local[survey] >= 0) / self.n_total

    def n_with_data(self, survey: str) -> int:
        """Return number of stars with data for this survey."""
        return len(self.local_to_global[survey])

    def get_coverage_stats(self) -> dict[str, int]:
        """Get the number of stars with data from each survey."""
        return {survey: self.n_with_data(survey) for survey in self.surveys}

    def memory_usage_mb(self) -> dict[str, float]:
        """Estimate memory usage in MB for each component."""
        usage = {}
        for survey in self.surveys:
            flux_mb = self.flux[survey].nbytes / 1e6
            ivar_mb = self.ivar[survey].nbytes / 1e6
            usage[f"{survey}_flux"] = flux_mb
            usage[f"{survey}_ivar"] = ivar_mb
        usage["labels"] = self.labels.nbytes / 1e6
        usage["gaia_ids"] = self.gaia_ids.nbytes / 1e6
        usage["total"] = sum(usage.values())
        return usage

    def get_labels_for_source(self, source: str) -> np.ndarray:
        """Get labels for a specific source (or primary if single-source).

        Args:
            source: Label source name (e.g., 'apogee', 'galah').

        Returns:
            Labels array of shape (n_total, 3, n_params).
        """
        if self.labels_dict is not None and source in self.labels_dict:
            return self.labels_dict[source]
        return self.labels

    def has_labels_for_source(self, source: str) -> np.ndarray:
        """Get has_labels mask for a specific source.

        Args:
            source: Label source name.

        Returns:
            Boolean mask of shape (n_total,) indicating which stars have labels.

        Raises:
            ValueError: If has_labels_dict exists but source is not in it.
        """
        if self.has_labels_dict is not None:
            if source in self.has_labels_dict:
                return self.has_labels_dict[source]
            # Don't silently fall back - this indicates a bug in data setup
            raise ValueError(
                f"Label source '{source}' not found in has_labels_dict. "
                f"Available sources: {list(self.has_labels_dict.keys())}. "
                "Ensure duplicate_labels config properly populates has_labels_dict."
            )
        # Single-label mode: derive from primary labels mask
        return np.any(self.labels[:, 2, :] > 0, axis=1)

    def is_multi_label(self) -> bool:
        """Check if this data has multiple label sources."""
        return self.label_sources is not None and len(self.label_sources) > 1


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
        survey: str | None = None,
        label_source: str | None = None,
        require_complete: bool = False,
    ) -> CatalogueData:
        """
        Load data from a single survey in the catalogue.

        Each survey has its own independent row structure with row-aligned labels.
        The label source is automatically determined from the survey name unless
        explicitly overridden.

        Args:
            survey: Survey name to load (e.g., "boss", "lamost_lrs", "desi").
                If None, loads from first available survey.
            label_source: Label source to use. If None, automatically determined
                from survey name (e.g., "lamost_lrs" -> "apogee_lamost_lrs").
            require_complete: If True, only return stars with both spectra and labels.

        Returns:
            CatalogueData object with loaded data.

        Raises:
            ValueError: If specified survey or label source not in catalogue.

        Examples:
            >>> loader = CatalogueLoader("super_catalogue.h5")
            >>> # Load BOSS spectra with corresponding APOGEE labels
            >>> boss_data = loader.load(survey="boss")
            >>> # Load LAMOST LRS with its APOGEE labels
            >>> lamost_data = loader.load(survey="lamost_lrs")
        """
        if survey is None:
            info = self.get_info()
            for name in info.survey_names:
                if name in info.surveys and info.surveys[name]["n_with_data"] > 0:
                    survey = name
                    break
            if survey is None:
                raise ValueError(
                    f"No surveys with data found in catalogue at {self.path}. "
                    f"Available surveys (all empty): {info.survey_names}"
                )

        # Store original request for fallback logic
        original_label_source = label_source

        with h5py.File(self.path, "r") as f:
            # Validate inputs
            if survey not in f["surveys"]:
                raise ValueError(
                    f"Survey '{survey}' not in catalogue. "
                    f"Available: {list(f['surveys'].keys())}"
                )

            available_labels = list(f["labels"].keys())

            # Resolve label source with fallback logic
            # Supports v1 schema (apogee_boss, etc.) and v2 schema (apogee, galah)
            if label_source is None:
                # Auto-derive: try SURVEY_LABEL_MAP first (v1), then unified sources (v2)
                mapped_source = SURVEY_LABEL_MAP.get(survey)
                if mapped_source and mapped_source in f["labels"]:
                    label_source = mapped_source
                else:
                    # Try v2 unified sources first, then v1 fallback pattern
                    candidates = ["apogee", "galah", f"apogee_{survey}"]
                    for candidate in candidates:
                        if candidate in f["labels"]:
                            label_source = candidate
                            break
                    if label_source is None:
                        raise ValueError(
                            f"No label source found for survey '{survey}'. "
                            f"Available: {available_labels}"
                        )
            elif label_source not in f["labels"]:
                # User-specified source not found - try {label_source}_{survey} pattern (v1)
                survey_specific = f"{label_source}_{survey}"
                if survey_specific in f["labels"]:
                    label_source = survey_specific
                else:
                    raise ValueError(
                        f"Label source '{original_label_source}' not in catalogue. "
                        f"Available: {available_labels}"
                    )

            # Load survey data
            survey_grp = f["surveys"][survey]
            wavelength = survey_grp["wavelength"][:]
            snr = survey_grp["snr"][:]

            # Handle different flux storage formats
            if "flux" in survey_grp:
                flux = survey_grp["flux"][:]
                ivar = survey_grp["ivar"][:]
            elif "spectra" in survey_grp:
                # LAMOST MRS stores as (N, 4, wavelengths) - [flux_B, ivar_B, flux_R, ivar_R]
                spectra = survey_grp["spectra"][:]
                # Concatenate the two arms along wavelength dimension (preserves all info)
                flux_b = spectra[:, 0, :]
                flux_r = spectra[:, 2, :]
                ivar_b = spectra[:, 1, :]
                ivar_r = spectra[:, 3, :]
                flux = np.concatenate([flux_b, flux_r], axis=1)
                ivar = np.concatenate([ivar_b, ivar_r], axis=1)
                # Tile wavelength to match concatenated flux shape
                wavelength = np.tile(wavelength, 2)
            else:
                raise ValueError(f"Survey '{survey}' has no flux or spectra data")

            n_stars = flux.shape[0]

            # Load labels
            label_grp = f["labels"][label_source]
            labels = label_grp["values"][:]
            label_errors = label_grp["errors"][:]
            label_flags = label_grp["flags"][:]

            # Load IDs - try label group first, then metadata
            if "gaia_id" in label_grp:
                gaia_ids = label_grp["gaia_id"][:]
            elif "metadata" in f and "gaia_id" in f["metadata"]:
                gaia_ids = f["metadata"]["gaia_id"][:]
            else:
                # Create placeholder IDs
                gaia_ids = np.array([f"{survey}_{i}" for i in range(n_stars)])

            # Load RA/Dec - try survey group, then metadata
            if "ra" in survey_grp:
                ra = survey_grp["ra"][:]
                dec = survey_grp["dec"][:]
            elif "metadata" in f and "ra" in f["metadata"]:
                # Only use metadata if it matches this survey's row count
                meta_ra = f["metadata"]["ra"][:]
                if len(meta_ra) == n_stars:
                    ra = meta_ra
                    dec = f["metadata"]["dec"][:]
                else:
                    ra = np.zeros(n_stars, dtype=np.float64)
                    dec = np.zeros(n_stars, dtype=np.float64)
            else:
                ra = np.zeros(n_stars, dtype=np.float64)
                dec = np.zeros(n_stars, dtype=np.float64)

            # Validate array sizes match
            arrays_to_check = {
                "gaia_ids": len(gaia_ids),
                "ra": len(ra),
                "dec": len(dec),
                "labels": len(labels),
                "label_errors": len(label_errors),
                "label_flags": len(label_flags),
                "snr": len(snr),
            }
            mismatched = {k: v for k, v in arrays_to_check.items() if v != n_stars}
            if mismatched:
                raise ValueError(
                    f"Array size mismatch for survey '{survey}': "
                    f"flux has {n_stars} rows but {mismatched}. "
                    f"The catalogue may need to be rebuilt with aligned arrays."
                )

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
            survey_name=survey,
            label_source=label_source,
        )

        if require_complete:
            data = data.filter_complete()

        return data

    def load_for_training(
        self,
        survey: str,
        label_source: str | None = None,
        max_flag_bits: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load data ready for training in unified 3-channel format.

        This is a convenience method that loads complete, high-quality data
        and returns it in the format expected by the DOROTHY trainer.

        Args:
            survey: Survey to load from.
            label_source: Label source to use. If None, auto-determined from survey.
            max_flag_bits: Maximum flag bits allowed (0 = highest quality).

        Returns:
            Tuple of (X, y) where:
                - X: Spectral data, shape (N, 3, wavelength_bins)
                    Channel 0: flux values
                    Channel 1: sigma (std dev), converted from ivar
                    Channel 2: mask (1=valid, 0=invalid wavelength)
                - y: Labels, shape (N, 3, n_params)
                    Channel 0: label values
                    Channel 1: label errors (uncertainties)
                    Channel 2: mask (1=valid, 0=masked parameter)
        """
        data = self.load(
            survey=survey, label_source=label_source, require_complete=True
        )

        if max_flag_bits >= 0:
            data = data.filter_by_flags(max_flag_bits)

        # Create 3-channel spectral data: [flux, sigma, mask]
        # Convert ivar to sigma: sigma = 1/sqrt(ivar) where ivar > 0
        flux = data.flux
        ivar = data.ivar
        sigma = np.zeros_like(ivar)
        valid_ivar = ivar > 0
        sigma[valid_ivar] = 1.0 / np.sqrt(ivar[valid_ivar])
        spectral_mask = valid_ivar.astype(np.float32)

        X = np.stack([flux, sigma, spectral_mask], axis=1)

        # Create 3-channel label data: [values, errors, mask]
        label_mask = ((data.label_flags == 0) & (data.label_errors > 0)).astype(
            np.float32
        )
        y = np.stack([data.labels, data.label_errors, label_mask], axis=1)

        return X, y

    def load_raw_spectra(
        self,
        survey: str,
    ) -> dict[str, np.ndarray]:
        """
        Load raw spectral data without processing.

        This method loads the spectra exactly as stored in the HDF5 file,
        which may have different formats for different surveys (e.g.,
        LAMOST MRS has 4 channels for 2 arms).

        Args:
            survey: Survey to load from.

        Returns:
            Dictionary with survey-specific spectral arrays and wavelength grid.
        """
        with h5py.File(self.path, "r") as f:
            if survey not in f["surveys"]:
                raise ValueError(
                    f"Survey '{survey}' not in catalogue. "
                    f"Available: {list(f['surveys'].keys())}"
                )

            survey_grp = f["surveys"][survey]
            result = {"wavelength": survey_grp["wavelength"][:]}

            if "flux" in survey_grp:
                result["flux"] = survey_grp["flux"][:]
                result["ivar"] = survey_grp["ivar"][:]
            if "spectra" in survey_grp:
                result["spectra"] = survey_grp["spectra"][:]
                # Include channel names if available
                if "channel_names" in survey_grp.attrs:
                    result["channel_names"] = list(survey_grp.attrs["channel_names"])
            if "snr" in survey_grp:
                result["snr"] = survey_grp["snr"][:]

        return result

    def list_surveys(self) -> list[str]:
        """List available surveys in the catalogue."""
        with h5py.File(self.path, "r") as f:
            if "surveys" in f:
                return list(f["surveys"].keys())
            return []

    def list_label_sources(self) -> list[str]:
        """List available label sources in the catalogue."""
        with h5py.File(self.path, "r") as f:
            if "labels" in f:
                return list(f["labels"].keys())
            return []

    def get_label_source_for_survey(self, survey: str) -> str | None:
        """Get the default label source for a given survey."""
        return SURVEY_LABEL_MAP.get(survey)

    def load_multi(
        self,
        surveys: list[str],
        mode: Literal["union", "intersection"] = "union",
        deduplicate: bool = True,
    ) -> dict[str, CatalogueData]:
        """
        Load data from multiple surveys.

        This method loads data from multiple surveys and returns them as a
        dictionary. In "intersection" mode, only stars that appear in ALL
        specified surveys are returned (useful for cross-survey comparisons).
        In "union" mode, all stars from all surveys are returned.

        Args:
            surveys: List of survey names to load.
            mode: How to combine surveys:
                - "union": Return all stars from all surveys (default)
                - "intersection": Return only stars common to all surveys
            deduplicate: If True (default), keep only one observation per star
                (the first occurrence). If False, keep all observations.

        Returns:
            Dictionary mapping survey names to CatalogueData objects.
            In "intersection" mode, all CatalogueData objects are filtered
            to the same set of stars (ordered consistently by Gaia ID).

        Example:
            >>> loader = CatalogueLoader("super_catalogue.h5")
            >>> # Load stars observed by both BOSS and LAMOST LRS
            >>> data = loader.load_multi(
            ...     surveys=["boss", "lamost_lrs"],
            ...     mode="intersection"
            ... )
            >>> print(f"Stars in both: {data['boss'].n_stars}")
        """
        result = {}

        if mode == "union":
            # Simply load each survey independently
            for survey in surveys:
                data = self.load(survey=survey)
                if deduplicate:
                    data = self._deduplicate(data)
                result[survey] = data
        else:
            # intersection mode - find common stars
            # First, get IDs for each survey (unique IDs only)
            survey_data = {}
            survey_unique_ids = {}

            for survey in surveys:
                data = self.load(survey=survey)
                survey_data[survey] = data

                # Get unique IDs for intersection
                ids = data.gaia_ids
                # Handle int64 Gaia IDs
                if np.issubdtype(ids.dtype, np.integer):
                    unique_ids = {int(i) for i in ids if i > 0}
                else:
                    # String IDs (fallback)
                    if hasattr(ids[0], "decode"):
                        ids = np.array([i.decode("utf-8") for i in ids])
                    unique_ids = {str(i) for i in ids}
                survey_unique_ids[survey] = unique_ids

            # Find intersection of all unique IDs
            common_ids = survey_unique_ids[surveys[0]]
            for survey in surveys[1:]:
                common_ids &= survey_unique_ids[survey]

            if len(common_ids) == 0:
                raise ValueError(f"No common stars found across surveys: {surveys}")

            # Sort common IDs for consistent ordering
            common_ids_sorted = sorted(common_ids)

            for survey in surveys:
                data = survey_data[survey]
                ids = data.gaia_ids

                # Find first index of each common ID (deduplicate)
                first_indices = {}
                for idx, gid in enumerate(ids):
                    gid_key = (
                        int(gid) if np.issubdtype(ids.dtype, np.integer) else str(gid)
                    )
                    if gid_key in common_ids and gid_key not in first_indices:
                        first_indices[gid_key] = idx

                # Sort indices by common_ids_sorted order
                sorted_indices = np.array(
                    [first_indices[gid] for gid in common_ids_sorted]
                )

                # Create filtered data
                result[survey] = CatalogueData(
                    gaia_ids=data.gaia_ids[sorted_indices],
                    ra=data.ra[sorted_indices],
                    dec=data.dec[sorted_indices],
                    flux=data.flux[sorted_indices],
                    ivar=data.ivar[sorted_indices],
                    wavelength=data.wavelength,
                    snr=data.snr[sorted_indices],
                    labels=data.labels[sorted_indices],
                    label_errors=data.label_errors[sorted_indices],
                    label_flags=data.label_flags[sorted_indices],
                    survey_name=data.survey_name,
                    label_source=data.label_source,
                )

        return result

    def _resolve_label_source(
        self, survey: str, label_source: str | None, f: h5py.File
    ) -> str:
        """
        Resolve label source name with fallback logic.

        Supports both v1 schema (per-survey label groups like 'apogee_boss')
        and v2 schema (unified label groups like 'apogee', 'galah').

        Resolution order:
        1. If label_source specified and exists -> use it directly
        2. If label_source specified, try {label_source}_{survey} pattern (v1)
        3. If label_source is None:
           a. Try SURVEY_LABEL_MAP (v1 schema)
           b. Try 'apogee' directly (v2 schema)
           c. Try 'galah' directly (v2 schema)
           d. Try 'apogee_{survey}' pattern (v1 fallback)
        """
        available_labels = list(f["labels"].keys())

        if label_source is None:
            # Auto-derive: try multiple patterns
            # First try v1 SURVEY_LABEL_MAP
            mapped_source = SURVEY_LABEL_MAP.get(survey)
            if mapped_source and mapped_source in f["labels"]:
                return mapped_source
            # Then try v2 unified sources and v1 fallbacks
            for candidate in ["apogee", "galah", f"apogee_{survey}"]:
                if candidate in f["labels"]:
                    return candidate
            raise ValueError(
                f"No label source found for survey '{survey}'. "
                f"Available: {available_labels}"
            )
        elif label_source not in f["labels"]:
            # Try {label_source}_{survey} pattern (v1 schema)
            survey_specific = f"{label_source}_{survey}"
            if survey_specific in f["labels"]:
                return survey_specific
            raise ValueError(
                f"Label source '{label_source}' not in catalogue. "
                f"Available: {available_labels}"
            )
        return label_source

    def _load_survey_metadata(
        self, survey: str, label_source: str, f: h5py.File
    ) -> dict:
        """
        Load only metadata for a survey (no spectra).

        Returns dict with gaia_ids, snr, ra, dec, labels, label_errors, label_flags,
        wavelength shape info, and storage format.
        """
        survey_grp = f["surveys"][survey]
        label_grp = f["labels"][label_source]

        # Determine storage format and get wavelength info
        if "flux" in survey_grp:
            storage_format = "flux"
            n_wavelengths = survey_grp["flux"].shape[1]
        elif "spectra" in survey_grp:
            storage_format = "spectra"
            # LAMOST MRS: (N, 4, wavelengths) -> concatenated = 2 * wavelengths
            n_wavelengths = survey_grp["spectra"].shape[2] * 2
        else:
            raise ValueError(f"Survey '{survey}' has no flux or spectra data")

        # Load wavelength array(s)
        if "wavelength_b" in survey_grp and "wavelength_r" in survey_grp:
            # LAMOST MRS has separate blue/red wavelength arrays
            wavelength = np.concatenate(
                [survey_grp["wavelength_b"][:], survey_grp["wavelength_r"][:]]
            )
        else:
            wavelength = survey_grp["wavelength"][:]

        # Load metadata arrays (small compared to spectra)
        snr = survey_grp["snr"][:]
        n_stars = len(snr)

        # Load IDs
        if "gaia_id" in label_grp:
            gaia_ids = label_grp["gaia_id"][:]
        elif "metadata" in f and "gaia_id" in f["metadata"]:
            gaia_ids = f["metadata"]["gaia_id"][:]
        else:
            gaia_ids = np.arange(n_stars, dtype=np.int64)

        # Load RA/Dec
        if "ra" in survey_grp:
            ra = survey_grp["ra"][:]
            dec = survey_grp["dec"][:]
        elif "metadata" in f and "ra" in f["metadata"]:
            meta_ra = f["metadata"]["ra"][:]
            if len(meta_ra) == n_stars:
                ra = meta_ra
                dec = f["metadata"]["dec"][:]
            else:
                ra = np.zeros(n_stars, dtype=np.float64)
                dec = np.zeros(n_stars, dtype=np.float64)
        else:
            ra = np.zeros(n_stars, dtype=np.float64)
            dec = np.zeros(n_stars, dtype=np.float64)

        # Load labels
        labels = label_grp["values"][:]
        label_errors = label_grp["errors"][:]
        label_flags = label_grp["flags"][:]

        return {
            "gaia_ids": gaia_ids,
            "snr": snr,
            "ra": ra,
            "dec": dec,
            "labels": labels,
            "label_errors": label_errors,
            "label_flags": label_flags,
            "wavelength": wavelength,
            "n_wavelengths": n_wavelengths,
            "n_stars": n_stars,
            "storage_format": storage_format,
        }

    def _load_spectra_for_indices(
        self, survey: str, indices: np.ndarray, storage_format: str, f: h5py.File
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load flux and ivar for specific row indices using HDF5 fancy indexing.

        Args:
            survey: Survey name
            indices: Row indices to load (must be sorted for efficiency)
            storage_format: "flux" or "spectra"
            f: Open HDF5 file handle

        Returns:
            (flux, ivar) arrays for the requested indices
        """
        survey_grp = f["surveys"][survey]

        if storage_format == "flux":
            # Direct indexed read
            flux = survey_grp["flux"][indices]
            ivar = survey_grp["ivar"][indices]
        else:
            # LAMOST MRS: (N, 4, wavelengths) format
            spectra = survey_grp["spectra"][indices]
            flux_b = spectra[:, 0, :]
            flux_r = spectra[:, 2, :]
            ivar_b = spectra[:, 1, :]
            ivar_r = spectra[:, 3, :]
            flux = np.concatenate([flux_b, flux_r], axis=1)
            ivar = np.concatenate([ivar_b, ivar_r], axis=1)

        return flux.astype(np.float32), ivar.astype(np.float32)

    def _collect_survey_metadata_pass1(
        self,
        surveys: list[str],
        label_source: str,
        f: h5py.File,
    ) -> tuple[dict[str, dict], dict[str, str], np.ndarray, int, dict[int, int]]:
        """
        Pass 1: Load metadata only and identify duplicates per survey.

        This is shared logic between load_merged() and load_merged_sparse().

        Args:
            surveys: List of survey names to process.
            label_source: Primary label source to use.
            f: Open HDF5 file handle.

        Returns:
            Tuple of:
                - survey_meta: Dict mapping survey -> metadata dict (includes
                  duplicate_indices, unique_indices, duplicate_gaia_ids, unique_gaia_ids)
                - survey_resolved_labels: Dict mapping survey -> resolved label source name
                - all_gaia_ids_sorted: Sorted array of all unique Gaia IDs
                - n_total: Total number of unique stars
                - id_to_idx: Dict mapping Gaia ID -> merged array index
        """
        survey_meta: dict[str, dict] = {}
        survey_resolved_labels: dict[str, str] = {}
        all_gaia_ids: set[int] = set()

        for survey in surveys:
            print(f"Loading {survey}...")
            resolved_label = self._resolve_label_source(survey, label_source, f)
            survey_resolved_labels[survey] = resolved_label

            meta = self._load_survey_metadata(survey, resolved_label, f)
            survey_meta[survey] = meta

            # Identify duplicates vs unique stars
            # In v2 catalogues, all surveys have the same row count but SNR=0 for missing data
            gaia_ids = meta["gaia_ids"]
            snr = meta["snr"]

            # Valid = has valid gaia_id AND has actual data (SNR > 0)
            valid_mask = (gaia_ids > 0) & (snr > 0)
            valid_ids = gaia_ids[valid_mask].astype(np.int64)

            # Count occurrences
            unique_ids, counts = np.unique(valid_ids, return_counts=True)
            duplicate_gaia_ids = set(unique_ids[counts > 1])
            unique_gaia_ids = set(unique_ids[counts == 1])

            # Find indices for duplicates and unique stars (only where SNR > 0)
            duplicate_indices = []
            unique_indices = []
            for idx, (gid, s) in enumerate(zip(gaia_ids, snr, strict=False)):
                if gid <= 0 or s <= 0:
                    continue  # Skip invalid IDs or stars without data
                gid_int = int(gid)
                if gid_int in duplicate_gaia_ids:
                    duplicate_indices.append(idx)
                else:
                    unique_indices.append(idx)

            meta["duplicate_indices"] = np.array(duplicate_indices, dtype=np.intp)
            meta["unique_indices"] = np.array(unique_indices, dtype=np.intp)
            meta["duplicate_gaia_ids"] = duplicate_gaia_ids
            meta["unique_gaia_ids"] = unique_gaia_ids

            # Collect all valid Gaia IDs
            all_gaia_ids.update(unique_gaia_ids)
            all_gaia_ids.update(duplicate_gaia_ids)

            n_dups = len(duplicate_gaia_ids)
            n_unique = len(unique_gaia_ids)
            print(f"  {n_unique:,} unique stars, {n_dups:,} stars with duplicates")

        # Create sorted union of all Gaia IDs
        all_gaia_ids_sorted = np.array(sorted(all_gaia_ids), dtype=np.int64)
        n_total = len(all_gaia_ids_sorted)
        id_to_idx = {int(gid): i for i, gid in enumerate(all_gaia_ids_sorted)}

        print(f"Total unique stars: {n_total:,}")

        return (
            survey_meta,
            survey_resolved_labels,
            all_gaia_ids_sorted,
            n_total,
            id_to_idx,
        )

    def _check_for_duplicates(
        self,
        survey_meta: dict[str, dict],
        surveys: list[str],
    ) -> None:
        """
        Check for duplicate observations and raise an error if found.

        This method enforces that catalogues must be pre-deduplicated before
        training. If duplicates are found, it provides instructions for creating
        a clean catalogue.

        Args:
            survey_meta: Survey metadata from pass 1, containing duplicate info.
            surveys: List of survey names being processed.

        Raises:
            ValueError: If any survey contains duplicate observations.
        """
        total_duplicates = 0
        duplicate_details = []

        for survey in surveys:
            n_dups = len(survey_meta[survey].get("duplicate_gaia_ids", set()))
            if n_dups > 0:
                total_duplicates += n_dups
                n_obs = len(survey_meta[survey].get("duplicate_indices", []))
                duplicate_details.append(
                    f"  - {survey}: {n_dups:,} stars with {n_obs:,} total duplicate observations"
                )

        if total_duplicates > 0:
            details = "\n".join(duplicate_details)
            raise ValueError(
                f"\n{'='*60}\n"
                f"DUPLICATE OBSERVATIONS DETECTED\n"
                f"{'='*60}\n\n"
                f"Found {total_duplicates:,} stars with duplicate observations:\n"
                f"{details}\n\n"
                f"Training requires a deduplicated catalogue. To create one, run:\n\n"
                f"  python scripts/create_deduplicated_catalogue.py \\\n"
                f"      {self.path} \\\n"
                f"      data/super_catalogue_clean.h5 \\\n"
                f"      --surveys {' '.join(surveys)}\n\n"
                f"Then update your config to use the clean catalogue:\n"
                f"  data:\n"
                f"    catalogue_path: data/super_catalogue_clean.h5\n"
                f"{'='*60}\n"
            )

    def load_merged(
        self,
        surveys: list[str],
        label_source: str = "apogee",
    ) -> MergedCatalogueData:
        """
        Load and merge multiple surveys with outer join structure.

        This method uses memory-efficient two-pass loading:
        1. First pass: Load only metadata to check for duplicates
        2. Second pass: Load spectra selectively using HDF5 indexed access

        Stars that don't appear in a survey have ivar=0 (natural masking).

        Note:
            This method requires a pre-deduplicated catalogue. If duplicates are
            found, an error is raised with instructions to run the deduplication
            script: scripts/create_deduplicated_catalogue.py

        Args:
            surveys: List of survey names to load and merge.
            label_source: Primary label source to use for combined labels.

        Returns:
            MergedCatalogueData with outer join structure.

        Raises:
            ValueError: If duplicate observations are found in the catalogue.

        Example:
            >>> merged = loader.load_merged(["boss", "lamost_lrs", "desi"])
            >>> print(f"Total unique stars: {merged.n_stars}")
            >>> print(f"Coverage: {merged.get_coverage_stats()}")
        """
        n_params = 11

        with h5py.File(self.path, "r") as f:
            # PASS 1: Load metadata and identify duplicates
            (
                survey_meta,
                survey_resolved_labels,
                all_gaia_ids_sorted,
                n_total,
                id_to_idx,
            ) = self._collect_survey_metadata_pass1(surveys, label_source, f)

            # Always check for duplicates - require pre-deduplicated catalogue
            self._check_for_duplicates(survey_meta, surveys)

            # =========================================================
            # PASS 2: Process each survey with selective spectra loading
            # =========================================================
            merged_surveys: dict[str, dict[str, np.ndarray]] = {}

            # Initialize common arrays
            merged_ra = np.zeros(n_total, dtype=np.float64)
            merged_dec = np.zeros(n_total, dtype=np.float64)
            ra_set = np.zeros(n_total, dtype=bool)

            primary_labels = np.zeros((n_total, n_params), dtype=np.float32)
            primary_errors = np.zeros((n_total, n_params), dtype=np.float32)
            primary_flags = np.zeros((n_total, n_params), dtype=np.uint8)
            labels_set = np.zeros(n_total, dtype=bool)

            for survey in surveys:
                meta = survey_meta[survey]
                n_wave = meta["n_wavelengths"]
                storage_format = meta["storage_format"]

                # Initialize arrays for this survey
                flux = np.zeros((n_total, n_wave), dtype=np.float32)
                ivar = np.zeros((n_total, n_wave), dtype=np.float32)
                snr = np.zeros(n_total, dtype=np.float32)
                has_data = np.zeros(n_total, dtype=bool)
                labels = np.zeros((n_total, n_params), dtype=np.float32)
                label_errors = np.zeros((n_total, n_params), dtype=np.float32)
                label_flags = np.zeros((n_total, n_params), dtype=np.uint8)

                # --- Handle unique stars (load directly into final arrays) ---
                # With deduplicated catalogue, all stars should be unique
                unique_indices = meta["unique_indices"]
                if len(unique_indices) > 0:
                    # Load spectra for unique stars
                    sorted_unique = np.sort(unique_indices)
                    unique_flux, unique_ivar = self._load_spectra_for_indices(
                        survey, sorted_unique, storage_format, f
                    )

                    # Map back to original order
                    reorder = np.argsort(np.argsort(unique_indices))
                    unique_flux = unique_flux[reorder]
                    unique_ivar = unique_ivar[reorder]

                    # Get merged indices for unique stars
                    unique_gaia_ids = meta["gaia_ids"][unique_indices].astype(np.int64)
                    merged_indices = np.array(
                        [id_to_idx[int(gid)] for gid in unique_gaia_ids], dtype=np.intp
                    )

                    # Bulk copy
                    flux[merged_indices] = unique_flux
                    ivar[merged_indices] = unique_ivar
                    snr[merged_indices] = meta["snr"][unique_indices]
                    has_data[merged_indices] = True
                    labels[merged_indices] = meta["labels"][unique_indices]
                    label_errors[merged_indices] = meta["label_errors"][unique_indices]
                    label_flags[merged_indices] = meta["label_flags"][unique_indices]

                    # RA/Dec for unique stars
                    ra_not_set = ~ra_set[merged_indices]
                    if np.any(ra_not_set):
                        update_merged = merged_indices[ra_not_set]
                        update_orig = unique_indices[ra_not_set]
                        merged_ra[update_merged] = meta["ra"][update_orig]
                        merged_dec[update_merged] = meta["dec"][update_orig]
                        ra_set[update_merged] = True

                    # Primary labels for unique stars
                    labels_not_set = ~labels_set[merged_indices]
                    has_valid_errors = np.any(
                        meta["label_errors"][unique_indices] > 0, axis=1
                    )
                    should_set = labels_not_set & has_valid_errors
                    if np.any(should_set):
                        update_merged = merged_indices[should_set]
                        update_orig = unique_indices[should_set]
                        primary_labels[update_merged] = meta["labels"][update_orig]
                        primary_errors[update_merged] = meta["label_errors"][
                            update_orig
                        ]
                        primary_flags[update_merged] = meta["label_flags"][update_orig]
                        labels_set[update_merged] = True

                    del unique_flux, unique_ivar
                    gc.collect()

                # Store survey results
                merged_surveys[survey] = {
                    "flux": flux,
                    "ivar": ivar,
                    "wavelength": meta["wavelength"],
                    "snr": snr,
                    "has_data": has_data,
                    "labels": labels,
                    "label_errors": label_errors,
                    "label_flags": label_flags,
                }

                n_with_data = int(has_data.sum())
                print(
                    f"  {survey}: {n_with_data:,} stars ({100*n_with_data/n_total:.1f}%)"
                )

        # Clean up
        del survey_meta
        gc.collect()

        return MergedCatalogueData(
            gaia_ids=all_gaia_ids_sorted,
            ra=merged_ra,
            dec=merged_dec,
            surveys=merged_surveys,
            label_source=label_source,
            primary_labels=primary_labels,
            primary_errors=primary_errors,
            primary_flags=primary_flags,
        )

    def load_merged_sparse(
        self,
        surveys: list[str],
        label_source: str = "apogee",
        max_flag_bits: int = 0,
        label_sources: list[str] | None = None,
    ) -> SparseMergedData:
        """
        Load and merge multiple surveys with memory-efficient sparse storage.

        Unlike load_merged() which stores dense arrays (with zeros for missing
        stars), this method stores only stars that have actual data per survey.
        This reduces memory usage by 60-80% for multi-survey datasets.

        Memory comparison for 155K total stars:
        - Dense (load_merged): ~28 GB for spectra
        - Sparse (this method): ~7 GB for spectra

        Note:
            This method requires a pre-deduplicated catalogue. If duplicates are
            found, an error is raised with instructions to run the deduplication
            script: scripts/create_deduplicated_catalogue.py

        Args:
            surveys: List of survey names to load and merge.
            label_source: Primary label source to use for combined labels.
            max_flag_bits: Maximum flag bits allowed (0 = highest quality).
            label_sources: Optional list of label sources for multi-label training.
                If provided, labels_dict and has_labels_dict will be populated
                with labels from each source. The primary `labels` array uses
                the first source in the list (or label_source if not provided).

        Returns:
            SparseMergedData with memory-efficient sparse survey storage.
            Use with trainer's sparse batch construction for training.

        Raises:
            ValueError: If duplicate observations are found in the catalogue.

        Example:
            >>> sparse_data = loader.load_merged_sparse(["boss", "desi", "lamost_lrs"])
            >>> print(f"Memory: {sparse_data.memory_usage_mb()['total']:.1f} MB")
            >>> print(f"Coverage: {sparse_data.get_coverage_stats()}")
            >>>
            >>> # Multi-label loading
            >>> sparse_data = loader.load_merged_sparse(
            ...     ["lamost_lrs", "lamost_mrs"],
            ...     label_sources=["apogee", "galah"]
            ... )
            >>> print(f"Multi-label: {sparse_data.is_multi_label()}")
        """
        n_params = 11

        with h5py.File(self.path, "r") as f:
            # =========================================================
            # PASS 1: Fast scan - read only SNR arrays and gaia_ids
            # =========================================================
            # Load master gaia_ids (from metadata or first label source)
            if "metadata" in f and "gaia_id" in f["metadata"]:
                all_gaia_ids = f["metadata"]["gaia_id"][:]
            else:
                resolved_label = self._resolve_label_source(surveys[0], label_source, f)
                all_gaia_ids = f["labels"][resolved_label]["gaia_id"][:]

            n_total = len(all_gaia_ids)
            id_to_idx = {int(gid): i for i, gid in enumerate(all_gaia_ids)}

            # Find which rows have data in each survey (fast - only reads SNR)
            survey_valid_indices: dict[str, np.ndarray] = {}
            for survey in surveys:
                print(f"Loading {survey}...")
                snr = f["surveys"][survey]["snr"][:]
                valid_mask = snr > 0
                valid_indices = np.where(valid_mask)[0]
                survey_valid_indices[survey] = valid_indices
                print(f"  {len(valid_indices):,} stars with data")

            print(f"Total unique stars: {n_total:,}")

            # =========================================================
            # PASS 2: Load spectra only for valid indices (sparse)
            # =========================================================
            sparse_flux: dict[str, np.ndarray] = {}
            sparse_ivar: dict[str, np.ndarray] = {}
            sparse_snr: dict[str, np.ndarray] = {}
            wavelengths: dict[str, np.ndarray] = {}
            global_to_local: dict[str, np.ndarray] = {}
            local_to_global: dict[str, np.ndarray] = {}

            # Initialize label arrays at master gaia_id level
            primary_labels = np.zeros((n_total, n_params), dtype=np.float32)
            primary_errors = np.zeros((n_total, n_params), dtype=np.float32)
            primary_flags = np.zeros((n_total, n_params), dtype=np.uint8)

            # Load primary labels once (not per survey)
            resolved_label = self._resolve_label_source(surveys[0], label_source, f)
            label_grp = f["labels"][resolved_label]
            primary_labels[:] = label_grp["values"][:]
            primary_errors[:] = label_grp["errors"][:]
            primary_flags[:] = label_grp["flags"][:]

            # Load RA/Dec from metadata
            if "metadata" in f and "ra" in f["metadata"]:
                merged_ra = f["metadata"]["ra"][:]
                merged_dec = f["metadata"]["dec"][:]
            else:
                merged_ra = np.zeros(n_total, dtype=np.float64)
                merged_dec = np.zeros(n_total, dtype=np.float64)

            for survey in surveys:
                survey_grp = f["surveys"][survey]
                valid_indices = survey_valid_indices[survey]
                n_with_data = len(valid_indices)

                # Get wavelength info
                if "wavelength_b" in survey_grp and "wavelength_r" in survey_grp:
                    wavelength = np.concatenate(
                        [survey_grp["wavelength_b"][:], survey_grp["wavelength_r"][:]]
                    )
                else:
                    wavelength = survey_grp["wavelength"][:]
                wavelengths[survey] = wavelength

                # Determine storage format
                storage_format = "flux" if "flux" in survey_grp else "spectra"

                # Load spectra - for compressed HDF5, read all then filter is faster
                # than fancy indexing (gzip decompression overhead)
                if storage_format == "flux":
                    flux_all = survey_grp["flux"][:]
                    ivar_all = survey_grp["ivar"][:]
                    flux = flux_all[valid_indices]
                    ivar = ivar_all[valid_indices]
                    del flux_all, ivar_all
                else:
                    # LAMOST MRS: (N, 4, wavelengths) format - dual arm
                    # Spectra array is padded to max wavelength, but arms have different lengths
                    spectra_all = survey_grp["spectra"][:]
                    spectra = spectra_all[valid_indices]
                    del spectra_all

                    # Get actual wavelength lengths for each arm
                    n_wave_b = len(survey_grp["wavelength_b"][:])
                    n_wave_r = len(survey_grp["wavelength_r"][:])

                    # Extract and trim to actual wavelength lengths
                    flux_b = spectra[:, 0, :n_wave_b]
                    flux_r = spectra[:, 2, :n_wave_r]
                    ivar_b = spectra[:, 1, :n_wave_b]
                    ivar_r = spectra[:, 3, :n_wave_r]
                    flux = np.concatenate([flux_b, flux_r], axis=1)
                    ivar = np.concatenate([ivar_b, ivar_r], axis=1)
                    del spectra

                # Load SNR for valid indices
                snr_all = survey_grp["snr"][:]
                snr = snr_all[valid_indices]

                # Build index mappings (valid_indices ARE the global indices)
                g2l = np.full(n_total, -1, dtype=np.int32)
                g2l[valid_indices] = np.arange(n_with_data, dtype=np.int32)
                global_to_local[survey] = g2l
                local_to_global[survey] = valid_indices.astype(np.int32)

                # Store sparse arrays
                sparse_flux[survey] = flux
                sparse_ivar[survey] = ivar
                sparse_snr[survey] = snr

                print(
                    f"  {survey}: {n_with_data:,} stars ({100*n_with_data/n_total:.1f}%)"
                )

                del flux, ivar, snr
                gc.collect()

        # Build 3-channel label format: [values, errors, mask]
        # Apply flag filtering to label mask
        if max_flag_bits >= 0:
            flag_bits = np.sum(
                np.bitwise_count(primary_flags.astype(np.uint64)), axis=1
            )
            flag_ok = flag_bits <= max_flag_bits
        else:
            flag_ok = np.ones(n_total, dtype=bool)

        label_mask = (
            (primary_flags == 0) & (primary_errors > 0) & flag_ok[:, None]
        ).astype(np.float32)

        labels_3ch = np.stack([primary_labels, primary_errors, label_mask], axis=1)

        # Multi-label support: load additional label sources if requested
        labels_dict_out: dict[str, np.ndarray] | None = None
        has_labels_dict_out: dict[str, np.ndarray] | None = None
        label_sources_out: list[str] | None = None

        if label_sources is not None and len(label_sources) > 1:
            labels_dict_out = {}
            has_labels_dict_out = {}
            label_sources_out = label_sources

            with h5py.File(self.path, "r") as f:
                if "labels" not in f:
                    raise ValueError("Catalogue has no 'labels' group")

                available_labels = list(f["labels"].keys())

                for source in label_sources:
                    # Check if this is the primary source (already loaded)
                    if source == label_source:
                        # Use the already-loaded primary labels
                        labels_dict_out[source] = labels_3ch.copy()
                        has_labels_out = np.any(primary_errors > 0, axis=1)
                        has_labels_dict_out[source] = has_labels_out
                        continue

                    # Resolve source name with fallback logic
                    resolved_source = source
                    if source not in f["labels"]:
                        # Try {source}_{survey} patterns
                        found = False
                        for survey in surveys:
                            survey_specific = f"{source}_{survey}"
                            if survey_specific in f["labels"]:
                                resolved_source = survey_specific
                                found = True
                                break
                        if not found:
                            raise ValueError(
                                f"Label source '{source}' not in catalogue. "
                                f"Available: {available_labels}"
                            )

                    label_grp = f["labels"][resolved_source]

                    if "gaia_id" not in label_grp:
                        # Can't properly align from file; duplicate primary labels
                        labels_dict_out[source] = labels_3ch.copy()
                        has_labels_out = np.any(primary_errors > 0, axis=1)
                        has_labels_dict_out[source] = has_labels_out
                        continue

                    # Load label data from file
                    source_values = label_grp["values"][:]
                    source_errors = label_grp["errors"][:]
                    source_flags = label_grp["flags"][:]
                    source_gaia_ids = label_grp["gaia_id"][:]

                    # Initialize arrays aligned to merged gaia_ids
                    values = np.zeros((n_total, n_params), dtype=np.float32)
                    errors = np.zeros((n_total, n_params), dtype=np.float32)
                    flags = np.zeros((n_total, n_params), dtype=np.uint8)
                    has_labels_arr = np.zeros(n_total, dtype=bool)

                    # Map label source data to merged structure (vectorized)
                    source_gaia_ids_int = source_gaia_ids.astype(np.int64)
                    # Find which source IDs exist in merged set
                    valid_source_mask = np.isin(source_gaia_ids_int, all_gaia_ids)
                    valid_source_idx = np.where(valid_source_mask)[0]
                    valid_source_gids = source_gaia_ids_int[valid_source_mask]

                    # Get merged indices for matching IDs (use id_to_idx dict)
                    merged_idx = np.array(
                        [id_to_idx[int(gid)] for gid in valid_source_gids],
                        dtype=np.intp,
                    )

                    # Bulk copy
                    values[merged_idx] = source_values[valid_source_idx]
                    errors[merged_idx] = source_errors[valid_source_idx]
                    flags[merged_idx] = source_flags[valid_source_idx]
                    has_labels_arr[merged_idx] = np.any(
                        source_errors[valid_source_idx] > 0, axis=1
                    )

                    # Apply flag filtering (vectorized)
                    if max_flag_bits >= 0:
                        source_flag_bits = np.sum(
                            np.bitwise_count(flags.astype(np.uint64)), axis=1
                        )
                        source_flag_ok = source_flag_bits <= max_flag_bits
                    else:
                        source_flag_ok = np.ones(n_total, dtype=bool)

                    # Create label mask
                    source_label_mask = (
                        (flags == 0) & (errors > 0) & source_flag_ok[:, None]
                    ).astype(np.float32)

                    # Stack into 3-channel format
                    labels_dict_out[source] = np.stack(
                        [values, errors, source_label_mask], axis=1
                    )
                    has_labels_dict_out[source] = has_labels_arr

        return SparseMergedData(
            flux=sparse_flux,
            ivar=sparse_ivar,
            wavelengths=wavelengths,
            snr=sparse_snr,
            global_to_local=global_to_local,
            local_to_global=local_to_global,
            labels=labels_3ch,
            gaia_ids=all_gaia_ids,
            ra=merged_ra,
            dec=merged_dec,
            surveys=surveys,
            n_total=n_total,
            n_params=n_params,
            labels_dict=labels_dict_out,
            has_labels_dict=has_labels_dict_out,
            label_sources=label_sources_out,
        )

    def _deduplicate(self, data: CatalogueData) -> CatalogueData:
        """
        Remove duplicate observations, keeping only the first occurrence per star.

        Note: Rows with invalid Gaia IDs (0 or negative) are always kept since
        they may represent different stars that couldn't be cross-matched.
        """
        ids = data.gaia_ids

        if np.issubdtype(ids.dtype, np.integer):
            # Vectorized deduplication for integer IDs
            # Invalid IDs (<=0) are kept; for valid IDs, keep first occurrence
            invalid_mask = ids <= 0
            invalid_indices = np.where(invalid_mask)[0]

            # For valid IDs, use numpy.unique to find first occurrences
            valid_mask = ~invalid_mask
            valid_ids = ids[valid_mask]
            valid_original_indices = np.where(valid_mask)[0]

            # np.unique returns indices of first occurrences in the valid subset
            _, first_in_subset = np.unique(valid_ids, return_index=True)
            valid_first_indices = valid_original_indices[first_in_subset]

            # Combine invalid indices (all kept) with first occurrences of valid IDs
            indices = np.union1d(invalid_indices, valid_first_indices)
            indices = np.sort(indices)  # Maintain original order
        else:
            # Fallback for string IDs (less common)
            seen = set()
            keep_indices = []
            for idx, gid in enumerate(ids):
                gid_key = str(gid)
                if gid_key not in seen:
                    seen.add(gid_key)
                    keep_indices.append(idx)
            indices = np.array(keep_indices)
        return CatalogueData(
            gaia_ids=data.gaia_ids[indices],
            ra=data.ra[indices],
            dec=data.dec[indices],
            flux=data.flux[indices],
            ivar=data.ivar[indices],
            wavelength=data.wavelength,
            snr=data.snr[indices],
            labels=data.labels[indices],
            label_errors=data.label_errors[indices],
            label_flags=data.label_flags[indices],
            survey_name=data.survey_name,
            label_source=data.label_source,
        )

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

    def load_merged_for_training(
        self,
        surveys: list[str],
        label_source: str = "apogee",
        max_flag_bits: int = 0,
        label_sources: list[str] | None = None,
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray | dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray] | None,
    ]:
        """
        Load merged multi-survey data ready for training with MultiHeadMLP.

        This method loads data from multiple surveys using outer join alignment
        and returns it in the unified 3-channel format expected by the trainer.

        Supports multi-labelset training where different stars may have labels
        from different sources (e.g., APOGEE vs GALAH). When label_sources is
        provided, returns labels and masks for each source, enabling union-style
        training with per-source loss masking.

        Note:
            This method requires a pre-deduplicated catalogue. If duplicates are
            found, an error is raised with instructions to run the deduplication
            script: scripts/create_deduplicated_catalogue.py

        Args:
            surveys: List of survey names to load and merge.
            label_source: Primary label source to use for combined labels.
                Used when label_sources is None (single-source mode).
            max_flag_bits: Maximum flag bits allowed (0 = highest quality).
            label_sources: List of label sources for multi-labelset training.
                If None, uses single-source mode with label_source parameter.
                If provided, returns y_dict and has_labels_dict instead of y.

        Returns:
            Tuple of (X_dict, y, has_data_dict, has_labels_dict) where:
                - X_dict: Dict mapping survey name to spectral data array
                    Each array has shape (N, 3, wavelength_bins) with:
                    - Channel 0: flux values
                    - Channel 1: sigma (std dev), converted from ivar
                    - Channel 2: mask (1=valid, 0=invalid wavelength)
                - y: Labels. In single-source mode (label_sources=None):
                    Array of shape (N, 3, n_params) with:
                    - Channel 0: label values
                    - Channel 1: label errors (uncertainties)
                    - Channel 2: mask (1=valid, 0=masked parameter)
                  In multi-source mode (label_sources provided):
                    Dict mapping label source name to array (N, 3, n_params)
                - has_data_dict: Dict mapping survey name to boolean mask
                    indicating which stars have data from that survey.
                - has_labels_dict: Dict mapping label source to boolean mask
                    indicating which stars have labels from that source.
                    Only present in multi-source mode; None in single-source mode.

        Example (single-source):
            >>> loader = CatalogueLoader("super_catalogue.h5")
            >>> X_dict, y, has_data, _ = loader.load_merged_for_training(
            ...     surveys=["boss", "lamost_lrs"]
            ... )
            >>> # y.shape = (N, 3, 11)

        Example (multi-source):
            >>> X_dict, y_dict, has_data, has_labels = loader.load_merged_for_training(
            ...     surveys=["boss", "lamost_lrs"],
            ...     label_sources=["apogee", "galah"]
            ... )
            >>> # y_dict["apogee"].shape = (N, 3, 11)
            >>> # y_dict["galah"].shape = (N, 3, 11)
            >>> # has_labels["apogee"] = boolean array of shape (N,)
        """
        # Load merged data
        merged = self.load_merged(
            surveys=surveys,
            label_source=label_source,
        )

        # Convert to 3-channel format for each survey
        X_dict: dict[str, np.ndarray] = {}
        has_data_dict: dict[str, np.ndarray] = {}

        for survey in surveys:
            survey_data = merged.surveys[survey]
            flux = survey_data["flux"]
            ivar = survey_data["ivar"]

            # Convert ivar to sigma: sigma = 1/sqrt(ivar) where ivar > 0
            sigma = np.zeros_like(ivar)
            valid_ivar = ivar > 0
            sigma[valid_ivar] = 1.0 / np.sqrt(ivar[valid_ivar])

            # Create spectral mask
            spectral_mask = valid_ivar.astype(np.float32)

            # Stack into 3-channel format: [flux, sigma, mask]
            X_dict[survey] = np.stack([flux, sigma, spectral_mask], axis=1)
            has_data_dict[survey] = survey_data["has_data"]

        # Single-source mode: return single y array
        if label_sources is None:
            # Create 3-channel label data: [values, errors, mask]
            # Apply flag filtering to label mask (vectorized)
            if max_flag_bits >= 0:
                # Count bits set in each flag
                flag_bits = np.sum(
                    np.bitwise_count(merged.primary_flags.astype(np.uint64)), axis=1
                )
                flag_ok = flag_bits <= max_flag_bits
            else:
                flag_ok = np.ones(merged.n_stars, dtype=bool)

            # Label mask: valid if flag is ok, error > 0, and per-parameter flag is 0
            label_mask = (
                (merged.primary_flags == 0)
                & (merged.primary_errors > 0)
                & flag_ok[:, None]
            ).astype(np.float32)

            y = np.stack(
                [merged.primary_labels, merged.primary_errors, label_mask], axis=1
            )

            return X_dict, y, has_data_dict, None

        # Multi-source mode: load labels from each source and return dicts
        y_dict: dict[str, np.ndarray] = {}
        has_labels_dict: dict[str, np.ndarray] = {}

        n_params = 11
        n_stars = merged.n_stars

        with h5py.File(self.path, "r") as f:
            if "labels" not in f:
                raise ValueError("Catalogue has no 'labels' group")

            available_labels = list(f["labels"].keys())

            for source in label_sources:
                # Resolve source name with fallback logic (same as single-source mode)
                resolved_source = source
                if source not in f["labels"]:
                    # Try {source}_{survey} patterns for each survey being loaded
                    found = False
                    for survey in surveys:
                        survey_specific = f"{source}_{survey}"
                        if survey_specific in f["labels"]:
                            resolved_source = survey_specific
                            found = True
                            break
                    if not found:
                        raise ValueError(
                            f"Label source '{source}' not in catalogue. "
                            f"Available: {available_labels}"
                        )

                # Check if this source matches the primary label source used in merged
                # If so, use the already-aligned primary_labels from merged structure
                # This is more reliable than re-loading and re-aligning from the file
                label_grp = f["labels"][resolved_source]
                use_merged_primary = False

                if "gaia_id" not in label_grp:
                    # No gaia_id means we can't properly align from file
                    # Use the already-aligned labels from merged structure instead
                    use_merged_primary = True

                if use_merged_primary:
                    # Use the already-aligned labels from merged structure
                    values = merged.primary_labels.copy()
                    errors = merged.primary_errors.copy()
                    flags = merged.primary_flags.copy()
                    # Star has labels if at least one error is positive
                    has_labels = np.any(errors > 0, axis=1)
                else:
                    # Load label data from file and remap by gaia_id
                    source_values = label_grp["values"][:]
                    source_errors = label_grp["errors"][:]
                    source_flags = label_grp["flags"][:]

                    # Initialize arrays aligned to merged gaia_ids
                    values = np.zeros((n_stars, n_params), dtype=np.float32)
                    errors = np.zeros((n_stars, n_params), dtype=np.float32)
                    flags = np.zeros((n_stars, n_params), dtype=np.uint8)
                    has_labels = np.zeros(n_stars, dtype=bool)

                    source_gaia_ids = label_grp["gaia_id"][:]

                    # Create mapping from gaia_id to merged index
                    merged_id_to_idx = {
                        int(gid): i for i, gid in enumerate(merged.gaia_ids)
                    }

                    # Map label source data to merged structure (vectorized)
                    source_gaia_ids_int = source_gaia_ids.astype(np.int64)
                    # Find which source IDs exist in merged set
                    valid_source_mask = np.isin(source_gaia_ids_int, merged.gaia_ids)
                    valid_source_idx = np.where(valid_source_mask)[0]
                    valid_source_gids = source_gaia_ids_int[valid_source_mask]

                    # Get merged indices for matching IDs
                    merged_idx = np.array(
                        [merged_id_to_idx[int(gid)] for gid in valid_source_gids],
                        dtype=np.intp,
                    )

                    # Bulk copy
                    values[merged_idx] = source_values[valid_source_idx]
                    errors[merged_idx] = source_errors[valid_source_idx]
                    flags[merged_idx] = source_flags[valid_source_idx]
                    has_labels[merged_idx] = np.any(
                        source_errors[valid_source_idx] > 0, axis=1
                    )

                # Apply flag filtering (vectorized)
                if max_flag_bits >= 0:
                    flag_bits = np.sum(
                        np.bitwise_count(flags.astype(np.uint64)), axis=1
                    )
                    flag_ok = flag_bits <= max_flag_bits
                else:
                    flag_ok = np.ones(n_stars, dtype=bool)

                # Create label mask: valid if flag ok, error > 0, per-param flag is 0
                label_mask = ((flags == 0) & (errors > 0) & flag_ok[:, None]).astype(
                    np.float32
                )

                # Stack into 3-channel format
                y_dict[source] = np.stack([values, errors, label_mask], axis=1)
                has_labels_dict[source] = has_labels

        return X_dict, y_dict, has_data_dict, has_labels_dict

    def get_survey_wavelength_counts(self, surveys: list[str]) -> dict[str, int]:
        """
        Get the number of wavelength bins for each survey.

        This is useful for initializing MultiHeadMLP with correct input sizes.

        Args:
            surveys: List of survey names.

        Returns:
            Dictionary mapping survey name to number of wavelength bins.

        Example:
            >>> loader = CatalogueLoader("super_catalogue.h5")
            >>> counts = loader.get_survey_wavelength_counts(["boss", "lamost_lrs"])
            >>> # counts = {"boss": 4506, "lamost_lrs": 3473}
        """
        result = {}
        with h5py.File(self.path, "r") as f:
            for survey in surveys:
                if survey not in f["surveys"]:
                    raise ValueError(f"Survey '{survey}' not in catalogue")
                survey_grp = f["surveys"][survey]
                # LAMOST MRS has separate blue/red wavelength arrays
                if "wavelength_b" in survey_grp and "wavelength_r" in survey_grp:
                    n_wavelengths = (
                        survey_grp["wavelength_b"].shape[0]
                        + survey_grp["wavelength_r"].shape[0]
                    )
                else:
                    n_wavelengths = survey_grp["wavelength"].shape[0]
                result[survey] = n_wavelengths
        return result
