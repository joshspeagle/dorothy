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

# Mapping from survey names to their label group names
SURVEY_LABEL_MAP = {
    "boss": "apogee",
    "lamost_lrs": "apogee_lamost_lrs",
    "lamost_mrs": "apogee_lamost_mrs",
    "desi": "apogee_desi",
}


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
                raise ValueError("No surveys with data found in catalogue")

        # Determine label source from survey if not specified
        if label_source is None:
            label_source = SURVEY_LABEL_MAP.get(survey)
            if label_source is None:
                # Fall back to checking what's available
                info = self.get_info()
                # Try survey-specific first, then generic
                candidates = [f"apogee_{survey}", "apogee", "galah"]
                for candidate in candidates:
                    if candidate in info.labels:
                        label_source = candidate
                        break
                if label_source is None:
                    raise ValueError(
                        f"No label source found for survey '{survey}'. "
                        f"Available: {list(info.labels.keys())}"
                    )

        with h5py.File(self.path, "r") as f:
            # Validate inputs
            if survey not in f["surveys"]:
                raise ValueError(
                    f"Survey '{survey}' not in catalogue. "
                    f"Available: {list(f['surveys'].keys())}"
                )
            if label_source not in f["labels"]:
                raise ValueError(
                    f"Label source '{label_source}' not in catalogue. "
                    f"Available: {list(f['labels'].keys())}"
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
                # Average the two arms for a combined spectrum
                flux = (spectra[:, 0, :] + spectra[:, 2, :]) / 2
                ivar_b = spectra[:, 1, :]
                ivar_r = spectra[:, 3, :]
                # Combine inverse variances (sum of variances -> sum of 1/ivars -> 1/(1/ivar_b + 1/ivar_r))
                # For simplicity, use average ivar where both are valid
                ivar = np.zeros_like(flux)
                valid = (ivar_b > 0) & (ivar_r > 0)
                ivar[valid] = 2.0 / (1.0 / ivar_b[valid] + 1.0 / ivar_r[valid])
                single_b = (ivar_b > 0) & (ivar_r <= 0)
                single_r = (ivar_r > 0) & (ivar_b <= 0)
                ivar[single_b] = ivar_b[single_b]
                ivar[single_r] = ivar_r[single_r]
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

    def load_merged(
        self,
        surveys: list[str],
        smart_deduplicate: bool = True,
        chi2_threshold: float = 2.0,
        label_source: str = "apogee",
    ) -> MergedCatalogueData:
        """
        Load and merge multiple surveys with outer join structure.

        This method loads data from multiple surveys, deduplicates within each
        survey, then creates an outer join indexed by Gaia DR3 ID. Stars that
        don't appear in a survey have ivar=0 for that survey (natural masking).

        Args:
            surveys: List of survey names to load and merge.
            smart_deduplicate: If True (default), use consistency-checking
                deduplication (stack consistent observations, else take highest SNR).
                If False, simply take the first observation per star.
            chi2_threshold: Threshold for reduced chi-squared in smart deduplication.
            label_source: Primary label source to use for combined labels.

        Returns:
            MergedCatalogueData with outer join structure.

        Example:
            >>> merged = loader.load_merged(["boss", "lamost_lrs", "desi"])
            >>> print(f"Total unique stars: {merged.n_stars}")
            >>> print(f"Coverage: {merged.get_coverage_stats()}")
            >>> # Filter to stars with both BOSS and LAMOST
            >>> subset = merged.filter_by_surveys(["boss", "lamost_lrs"])
        """
        # Step 1: Load and deduplicate each survey
        survey_data: dict[str, CatalogueData] = {}
        all_gaia_ids: set[int] = set()

        for survey in surveys:
            print(f"Loading {survey}...")
            data = self.load(survey=survey, label_source=label_source)

            # Deduplicate within survey
            if smart_deduplicate:
                data = self._smart_deduplicate(data, chi2_threshold=chi2_threshold)
            else:
                data = self._deduplicate(data)

            survey_data[survey] = data

            # Collect valid Gaia IDs
            for gid in data.gaia_ids:
                gid_int = int(gid)
                if gid_int > 0:
                    all_gaia_ids.add(gid_int)

        # Step 2: Create sorted union of all Gaia IDs
        all_gaia_ids_sorted = np.array(sorted(all_gaia_ids), dtype=np.int64)
        n_total = len(all_gaia_ids_sorted)
        id_to_idx = {gid: i for i, gid in enumerate(all_gaia_ids_sorted)}

        print(f"Total unique stars: {n_total:,}")

        # Step 3: Build merged structure
        merged_surveys: dict[str, dict[str, np.ndarray]] = {}

        # Initialize common arrays
        merged_ra = np.zeros(n_total, dtype=np.float64)
        merged_dec = np.zeros(n_total, dtype=np.float64)
        ra_set = np.zeros(n_total, dtype=bool)

        # Primary labels (from specified label source, filled from first survey that has it)
        n_params = 11
        primary_labels = np.zeros((n_total, n_params), dtype=np.float32)
        primary_errors = np.zeros((n_total, n_params), dtype=np.float32)
        primary_flags = np.zeros((n_total, n_params), dtype=np.uint8)
        labels_set = np.zeros(n_total, dtype=bool)

        for survey in surveys:
            data = survey_data[survey]
            n_wave = data.flux.shape[1]

            # Initialize arrays for this survey
            flux = np.zeros((n_total, n_wave), dtype=np.float32)
            ivar = np.zeros((n_total, n_wave), dtype=np.float32)
            snr = np.zeros(n_total, dtype=np.float32)
            has_data = np.zeros(n_total, dtype=bool)
            labels = np.zeros((n_total, n_params), dtype=np.float32)
            label_errors = np.zeros((n_total, n_params), dtype=np.float32)
            label_flags = np.zeros((n_total, n_params), dtype=np.uint8)

            # Fill in data for stars in this survey
            for i, gid in enumerate(data.gaia_ids):
                gid_int = int(gid)
                if gid_int <= 0:
                    continue  # Skip invalid IDs

                idx = id_to_idx[gid_int]
                flux[idx] = data.flux[i]
                ivar[idx] = data.ivar[i]
                snr[idx] = data.snr[i]
                has_data[idx] = True
                labels[idx] = data.labels[i]
                label_errors[idx] = data.label_errors[i]
                label_flags[idx] = data.label_flags[i]

                # Set RA/Dec from first survey that has it
                if not ra_set[idx]:
                    merged_ra[idx] = data.ra[i]
                    merged_dec[idx] = data.dec[i]
                    ra_set[idx] = True

                # Set primary labels from first survey that has them
                if not labels_set[idx] and np.any(data.label_errors[i] > 0):
                    primary_labels[idx] = data.labels[i]
                    primary_errors[idx] = data.label_errors[i]
                    primary_flags[idx] = data.label_flags[i]
                    labels_set[idx] = True

            merged_surveys[survey] = {
                "flux": flux,
                "ivar": ivar,
                "wavelength": data.wavelength,
                "snr": snr,
                "has_data": has_data,
                "labels": labels,
                "label_errors": label_errors,
                "label_flags": label_flags,
            }

            n_with_data = int(has_data.sum())
            print(f"  {survey}: {n_with_data:,} stars ({100*n_with_data/n_total:.1f}%)")

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

    def _deduplicate(self, data: CatalogueData) -> CatalogueData:
        """
        Remove duplicate observations, keeping only the first occurrence per star.

        Note: Rows with invalid Gaia IDs (0 or negative) are always kept since
        they may represent different stars that couldn't be cross-matched.
        """
        ids = data.gaia_ids
        seen = set()
        keep_indices = []

        for idx, gid in enumerate(ids):
            if np.issubdtype(ids.dtype, np.integer):
                gid_int = int(gid)
                # Keep all invalid IDs (can't deduplicate unknown stars)
                if gid_int <= 0:
                    keep_indices.append(idx)
                    continue
                gid_key = gid_int
            else:
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

    def _smart_deduplicate(
        self,
        data: CatalogueData,
        chi2_threshold: float = 2.0,
    ) -> CatalogueData:
        """
        Deduplicate by stacking consistent observations or taking highest SNR.

        For each star with multiple observations:
        1. Compute inverse-variance weighted mean spectrum
        2. Compute reduced chi-squared across all observations
        3. If chi2_red < threshold: stack all observations (weighted average)
        4. Otherwise: take the observation with highest SNR

        Args:
            data: Input CatalogueData with possible duplicates.
            chi2_threshold: Maximum reduced chi-squared to consider consistent.

        Returns:
            Deduplicated CatalogueData with one row per unique star.
        """
        ids = data.gaia_ids

        # Group indices by Gaia ID
        id_to_indices: dict[int, list[int]] = {}
        invalid_indices = []

        for idx, gid in enumerate(ids):
            if np.issubdtype(ids.dtype, np.integer):
                gid_int = int(gid)
                if gid_int <= 0:
                    invalid_indices.append(idx)
                    continue
                gid_key = gid_int
            else:
                gid_key = hash(str(gid))

            if gid_key not in id_to_indices:
                id_to_indices[gid_key] = []
            id_to_indices[gid_key].append(idx)

        # Process each unique star
        n_unique = len(id_to_indices) + len(invalid_indices)
        n_wavelengths = data.flux.shape[1]
        n_params = data.labels.shape[1]

        # Pre-allocate output arrays
        out_gaia_ids = np.zeros(n_unique, dtype=data.gaia_ids.dtype)
        out_ra = np.zeros(n_unique, dtype=data.ra.dtype)
        out_dec = np.zeros(n_unique, dtype=data.dec.dtype)
        out_flux = np.zeros((n_unique, n_wavelengths), dtype=np.float32)
        out_ivar = np.zeros((n_unique, n_wavelengths), dtype=np.float32)
        out_snr = np.zeros(n_unique, dtype=np.float32)
        out_labels = np.zeros((n_unique, n_params), dtype=np.float32)
        out_errors = np.zeros((n_unique, n_params), dtype=np.float32)
        out_flags = np.zeros((n_unique, n_params), dtype=np.uint8)

        out_idx = 0
        n_stacked = 0
        n_highest_snr = 0

        # Handle invalid IDs first (keep as-is)
        for idx in invalid_indices:
            out_gaia_ids[out_idx] = data.gaia_ids[idx]
            out_ra[out_idx] = data.ra[idx]
            out_dec[out_idx] = data.dec[idx]
            out_flux[out_idx] = data.flux[idx]
            out_ivar[out_idx] = data.ivar[idx]
            out_snr[out_idx] = data.snr[idx]
            out_labels[out_idx] = data.labels[idx]
            out_errors[out_idx] = data.label_errors[idx]
            out_flags[out_idx] = data.label_flags[idx]
            out_idx += 1

        # Process each unique star
        for _gid_key, indices in id_to_indices.items():
            if len(indices) == 1:
                # Single observation - just copy
                idx = indices[0]
                out_gaia_ids[out_idx] = data.gaia_ids[idx]
                out_ra[out_idx] = data.ra[idx]
                out_dec[out_idx] = data.dec[idx]
                out_flux[out_idx] = data.flux[idx]
                out_ivar[out_idx] = data.ivar[idx]
                out_snr[out_idx] = data.snr[idx]
                out_labels[out_idx] = data.labels[idx]
                out_errors[out_idx] = data.label_errors[idx]
                out_flags[out_idx] = data.label_flags[idx]
            else:
                # Multiple observations - check consistency
                obs_flux = data.flux[indices]  # (N_obs, N_wave)
                obs_ivar = data.ivar[indices]  # (N_obs, N_wave)
                obs_snr = data.snr[indices]  # (N_obs,)
                obs_labels = data.labels[indices]  # (N_obs, N_params)
                obs_errors = data.label_errors[indices]  # (N_obs, N_params)

                # Compute inverse-variance weighted mean spectrum
                sum_weights = np.sum(obs_ivar, axis=0)  # (N_wave,)
                valid = sum_weights > 0
                weighted_flux = np.zeros(n_wavelengths, dtype=np.float32)
                weighted_flux[valid] = (
                    np.sum(obs_flux[:, valid] * obs_ivar[:, valid], axis=0)
                    / sum_weights[valid]
                )

                # Compute reduced chi-squared for spectra
                # chi2 = sum((flux_i - mean)^2 * ivar_i) / (N_obs - 1)
                residuals_sq = (obs_flux - weighted_flux) ** 2 * obs_ivar
                chi2_per_pixel = np.sum(residuals_sq, axis=0)  # sum over observations
                n_obs = len(indices)
                # Only count pixels with valid data from multiple observations
                multi_obs_pixels = np.sum(obs_ivar > 0, axis=0) > 1
                if multi_obs_pixels.sum() > 0:
                    chi2_red = chi2_per_pixel[multi_obs_pixels].sum() / (
                        multi_obs_pixels.sum() * (n_obs - 1)
                    )
                else:
                    chi2_red = 0.0  # Can't compute, assume consistent

                if chi2_red < chi2_threshold:
                    # Consistent - stack spectra
                    n_stacked += 1
                    out_flux[out_idx] = weighted_flux
                    out_ivar[out_idx] = sum_weights

                    # Stack labels (inverse-variance weighted, mask-aware)
                    # Only include labels where error > 0 (valid/unmasked)
                    valid_labels = obs_errors > 0
                    obs_weights = np.zeros_like(obs_errors)
                    obs_weights[valid_labels] = 1.0 / (obs_errors[valid_labels] ** 2)

                    sum_label_weights = np.sum(obs_weights, axis=0)
                    has_valid = sum_label_weights > 0

                    # Weighted average where we have valid labels
                    out_labels[out_idx, has_valid] = (
                        np.sum(
                            obs_labels[:, has_valid] * obs_weights[:, has_valid], axis=0
                        )
                        / sum_label_weights[has_valid]
                    )
                    out_errors[out_idx, has_valid] = 1.0 / np.sqrt(
                        sum_label_weights[has_valid]
                    )

                    # For parameters with no valid labels, keep them masked (error=0)
                    out_labels[out_idx, ~has_valid] = 0.0
                    out_errors[out_idx, ~has_valid] = 0.0

                    # Combine flags (OR across observations)
                    out_flags[out_idx] = np.bitwise_or.reduce(
                        data.label_flags[indices], axis=0
                    )

                    # SNR of stacked spectrum (approximate)
                    out_snr[out_idx] = np.sqrt(np.sum(obs_snr**2))
                else:
                    # Inconsistent - take highest SNR
                    n_highest_snr += 1
                    best_idx = indices[np.argmax(obs_snr)]
                    out_flux[out_idx] = data.flux[best_idx]
                    out_ivar[out_idx] = data.ivar[best_idx]
                    out_snr[out_idx] = data.snr[best_idx]
                    out_labels[out_idx] = data.labels[best_idx]
                    out_errors[out_idx] = data.label_errors[best_idx]
                    out_flags[out_idx] = data.label_flags[best_idx]

                # Common metadata (from first observation)
                out_gaia_ids[out_idx] = data.gaia_ids[indices[0]]
                out_ra[out_idx] = data.ra[indices[0]]
                out_dec[out_idx] = data.dec[indices[0]]

            out_idx += 1

        # Log statistics
        n_with_dups = sum(1 for indices in id_to_indices.values() if len(indices) > 1)
        if n_with_dups > 0:
            print(
                f"  Deduplication: {n_with_dups} stars with duplicates "
                f"({n_stacked} stacked, {n_highest_snr} took highest SNR)"
            )

        return CatalogueData(
            gaia_ids=out_gaia_ids,
            ra=out_ra,
            dec=out_dec,
            flux=out_flux,
            ivar=out_ivar,
            wavelength=data.wavelength,
            snr=out_snr,
            labels=out_labels,
            label_errors=out_errors,
            label_flags=out_flags,
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
        smart_deduplicate: bool = True,
        chi2_threshold: float = 2.0,
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

        Args:
            surveys: List of survey names to load and merge.
            smart_deduplicate: If True (default), use consistency-checking
                deduplication. If False, simply take the first observation.
            chi2_threshold: Threshold for reduced chi-squared in smart dedup.
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
            smart_deduplicate=smart_deduplicate,
            chi2_threshold=chi2_threshold,
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
            # Apply flag filtering to label mask
            if max_flag_bits >= 0:
                # Count bits set in each flag
                flag_bits = np.zeros(merged.n_stars, dtype=int)
                for i in range(11):
                    flag_bits += np.bitwise_count(
                        merged.primary_flags[:, i].astype(np.uint64)
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

        with h5py.File(self.path, "r") as f:
            if "labels" not in f:
                raise ValueError("Catalogue has no 'labels' group")

            for source in label_sources:
                if source not in f["labels"]:
                    raise ValueError(
                        f"Label source '{source}' not in catalogue. "
                        f"Available: {list(f['labels'].keys())}"
                    )

                # Load label data for this source
                label_grp = f["labels"][source]
                source_values = label_grp["values"][:]
                source_errors = label_grp["errors"][:]
                source_flags = label_grp["flags"][:]

                # Initialize arrays aligned to merged gaia_ids
                n_stars = merged.n_stars
                values = np.zeros((n_stars, n_params), dtype=np.float32)
                errors = np.zeros((n_stars, n_params), dtype=np.float32)
                flags = np.zeros((n_stars, n_params), dtype=np.uint8)
                has_labels = np.zeros(n_stars, dtype=bool)

                # Get gaia_ids from label source if available
                if "gaia_id" in label_grp:
                    source_gaia_ids = label_grp["gaia_id"][:]

                    # Create mapping from gaia_id to merged index
                    merged_id_to_idx = {
                        int(gid): i for i, gid in enumerate(merged.gaia_ids)
                    }

                    # Map label source data to merged structure
                    for i, gid in enumerate(source_gaia_ids):
                        gid_int = int(gid)
                        if gid_int in merged_id_to_idx:
                            idx = merged_id_to_idx[gid_int]
                            values[idx] = source_values[i]
                            errors[idx] = source_errors[i]
                            flags[idx] = source_flags[i]
                            # Star has labels if at least one error is positive
                            has_labels[idx] = np.any(source_errors[i] > 0)
                else:
                    # If no gaia_id, assume same ordering (fallback)
                    n_source = min(len(source_values), n_stars)
                    values[:n_source] = source_values[:n_source]
                    errors[:n_source] = source_errors[:n_source]
                    flags[:n_source] = source_flags[:n_source]
                    has_labels[:n_source] = np.any(source_errors[:n_source] > 0, axis=1)

                # Apply flag filtering
                if max_flag_bits >= 0:
                    flag_bits = np.zeros(n_stars, dtype=int)
                    for i in range(n_params):
                        flag_bits += np.bitwise_count(flags[:, i].astype(np.uint64))
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
                result[survey] = f["surveys"][survey]["wavelength"].shape[0]
        return result
