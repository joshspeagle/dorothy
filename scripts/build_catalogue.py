#!/usr/bin/env python
"""
Build a unified multi-survey super-catalogue for DOROTHY.

This script consolidates spectroscopic data from DESI, BOSS, LAMOST LRS, and LAMOST MRS,
cross-matched with stellar parameter labels from APOGEE and GALAH.

Usage:
    python scripts/build_catalogue.py --config catalogue_config.yaml
    python scripts/build_catalogue.py --config catalogue_config.yaml --dry-run
    python scripts/build_catalogue.py --config catalogue_config.yaml --surveys desi boss

Output:
    HDF5 file with row-matched structure (every star has a row, zeros for missing data).
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table
from pydantic import BaseModel, Field


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

VERSION = "1.0.0"

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

# Survey names
SURVEY_NAMES = ["desi", "boss", "lamost_lrs", "lamost_mrs"]

# Cross-match tolerance in arcseconds
CROSSMATCH_TOLERANCE_ARCSEC = 0.5

# APOGEE ASPCAPFLAG bitmask for STAR_BAD
# From https://www.sdss4.org/dr17/irspec/apogee-bitmasks/
APOGEE_STAR_BAD = 1 << 23  # STAR_BAD bit


# =============================================================================
# Pydantic Configuration Models
# =============================================================================


class SurveyName(str, Enum):
    """Supported survey names."""

    DESI = "desi"
    BOSS = "boss"
    LAMOST_LRS = "lamost_lrs"
    LAMOST_MRS = "lamost_mrs"


class LabelSourceName(str, Enum):
    """Supported label source names."""

    APOGEE = "apogee"
    GALAH = "galah"


class OutputConfig(BaseModel):
    """Configuration for output file."""

    path: Path = Field(..., description="Path to output HDF5 file")
    compression: str = Field(default="gzip", description="HDF5 compression")
    compression_opts: int = Field(default=4, description="Compression level (1-9)")


class GaiaConfig(BaseModel):
    """Configuration for Gaia cross-matching.

    Note: Gaia IDs are read directly from APOGEE/GALAH catalogues
    (GAIAEDR3_SOURCE_ID and gaiadr3_source_id columns respectively).
    No separate Gaia file is needed.
    """

    tolerance_arcsec: float = Field(
        default=0.5, description="Cross-match tolerance in arcseconds"
    )


class SurveyConfig(BaseModel):
    """Configuration for a single survey."""

    enabled: bool = Field(default=True, description="Whether to include this survey")
    data_path: Path = Field(..., description="Path to survey data file(s)")
    snr_threshold: float = Field(default=20.0, description="Minimum SNR threshold")
    extra_filters: dict[str, Any] = Field(
        default_factory=dict, description="Additional quality filters"
    )


class DESISurveyConfig(SurveyConfig):
    """DESI-specific configuration."""

    gaia_crossmatch_path: Path | None = Field(
        default=None, description="Path to DESI-Gaia cross-match table"
    )
    snr_column: str = Field(default="SN_B", description="SNR column name")


class BOSSSurveyConfig(SurveyConfig):
    """BOSS-specific configuration."""

    spectra_dir: Path | None = Field(
        default=None, description="Directory containing individual spectra"
    )
    snr_column: str = Field(default="SN_MEDIAN_ALL", description="SNR column name")


class LAMOSTLRSSurveyConfig(SurveyConfig):
    """LAMOST LRS-specific configuration."""

    snr_column: str = Field(default="snrg", description="SNR column name")


class LAMOSTMRSSurveyConfig(SurveyConfig):
    """LAMOST MRS-specific configuration."""

    snr_column: str = Field(default="snr", description="SNR column name")


class APOGEELabelConfig(BaseModel):
    """Configuration for APOGEE labels."""

    enabled: bool = Field(default=True, description="Whether to include APOGEE labels")
    data_path: Path = Field(..., description="Path to APOGEE allStar file")
    gaia_id_column: str = Field(
        default="GAIAEDR3_SOURCE_ID",
        description="Column containing Gaia DR3 source IDs",
    )
    snr_threshold: float = Field(
        default=50.0, description="Minimum SNR threshold for quality"
    )
    apply_star_bad_filter: bool = Field(
        default=True, description="Filter out stars with STAR_BAD flag"
    )


class GALAHLabelConfig(BaseModel):
    """Configuration for GALAH labels."""

    enabled: bool = Field(default=True, description="Whether to include GALAH labels")
    data_path: Path = Field(..., description="Path to GALAH allstar file")
    gaia_id_column: str = Field(
        default="gaiadr3_source_id", description="Column containing Gaia DR3 source IDs"
    )
    snr_threshold: float = Field(
        default=30.0, description="Minimum SNR threshold (snr_px_ccd3)"
    )
    apply_flag_sp_filter: bool = Field(
        default=True, description="Filter out stars with flag_sp != 0"
    )


class SurveysConfig(BaseModel):
    """Configuration for all surveys."""

    desi: DESISurveyConfig | None = None
    boss: BOSSSurveyConfig | None = None
    lamost_lrs: LAMOSTLRSSurveyConfig | None = None
    lamost_mrs: LAMOSTMRSSurveyConfig | None = None


class LabelsConfig(BaseModel):
    """Configuration for all label sources."""

    apogee: APOGEELabelConfig | None = None
    galah: GALAHLabelConfig | None = None


class CatalogueConfig(BaseModel):
    """Main configuration for catalogue building."""

    output: OutputConfig
    gaia: GaiaConfig = Field(default_factory=GaiaConfig)
    surveys: SurveysConfig
    labels: LabelsConfig

    @classmethod
    def from_yaml(cls, path: Path) -> CatalogueConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


# =============================================================================
# Data Classes for Intermediate Results
# =============================================================================


class SurveyData:
    """Container for loaded survey data."""

    def __init__(
        self,
        name: str,
        gaia_ids: np.ndarray,
        ra: np.ndarray,
        dec: np.ndarray,
        flux: np.ndarray,
        ivar: np.ndarray,
        wavelength: np.ndarray,
        snr: np.ndarray,
    ):
        self.name = name
        self.gaia_ids = gaia_ids  # int64
        self.ra = ra  # float64
        self.dec = dec  # float64
        self.flux = flux  # float32, shape (N, wavelength_bins) or (N, wavelength_bins) for each arm
        self.ivar = ivar  # float32
        self.wavelength = wavelength  # float32, shape (wavelength_bins,)
        self.snr = snr  # float32

    @property
    def n_stars(self) -> int:
        return len(self.gaia_ids)


class LabelData:
    """Container for loaded label data."""

    def __init__(
        self,
        name: str,
        gaia_ids: np.ndarray,
        ra: np.ndarray,
        dec: np.ndarray,
        values: np.ndarray,
        errors: np.ndarray,
        flags: np.ndarray,
    ):
        self.name = name
        self.gaia_ids = gaia_ids  # int64
        self.ra = ra  # float64
        self.dec = dec  # float64
        self.values = values  # float32, shape (N, 11)
        self.errors = errors  # float32, shape (N, 11)
        self.flags = flags  # uint8, shape (N, 11)

    @property
    def n_stars(self) -> int:
        return len(self.gaia_ids)


# =============================================================================
# Cross-Matching Functions
# =============================================================================


def crossmatch_to_gaia(
    ra: np.ndarray,
    dec: np.ndarray,
    gaia_ra: np.ndarray,
    gaia_dec: np.ndarray,
    gaia_source_ids: np.ndarray,
    tolerance_arcsec: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-match coordinates to Gaia DR3.

    Args:
        ra: Right ascension of sources (degrees)
        dec: Declination of sources (degrees)
        gaia_ra: Gaia right ascension (degrees)
        gaia_dec: Gaia declination (degrees)
        gaia_source_ids: Gaia source IDs
        tolerance_arcsec: Match tolerance in arcseconds

    Returns:
        Tuple of (matched_gaia_ids, match_mask)
        - matched_gaia_ids: Gaia source IDs for matched sources (-1 for unmatched)
        - match_mask: Boolean mask of successfully matched sources
    """
    source_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    gaia_coords = SkyCoord(ra=gaia_ra * u.deg, dec=gaia_dec * u.deg)

    idx, sep2d, _ = match_coordinates_sky(source_coords, gaia_coords)
    match_mask = sep2d.arcsec < tolerance_arcsec

    matched_gaia_ids = np.full(len(ra), -1, dtype=np.int64)
    matched_gaia_ids[match_mask] = gaia_source_ids[idx[match_mask]]

    return matched_gaia_ids, match_mask


def crossmatch_catalogues(
    cat1_ra: np.ndarray,
    cat1_dec: np.ndarray,
    cat2_ra: np.ndarray,
    cat2_dec: np.ndarray,
    tolerance_arcsec: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cross-match two catalogues by position.

    Args:
        cat1_ra, cat1_dec: Coordinates of first catalogue (degrees)
        cat2_ra, cat2_dec: Coordinates of second catalogue (degrees)
        tolerance_arcsec: Match tolerance in arcseconds

    Returns:
        Tuple of (idx1, idx2, separations)
        - idx1: Indices in cat1 that have matches
        - idx2: Corresponding indices in cat2
        - separations: Angular separations in arcseconds
    """
    coords1 = SkyCoord(ra=cat1_ra * u.deg, dec=cat1_dec * u.deg)
    coords2 = SkyCoord(ra=cat2_ra * u.deg, dec=cat2_dec * u.deg)

    idx, sep2d, _ = match_coordinates_sky(coords1, coords2)
    match_mask = sep2d.arcsec < tolerance_arcsec

    idx1 = np.where(match_mask)[0]
    idx2 = idx[match_mask]
    separations = sep2d.arcsec[match_mask]

    return idx1, idx2, separations


# =============================================================================
# Survey Loaders
# =============================================================================


class BaseSurveyLoader:
    """Base class for survey data loaders."""

    survey_name: str = ""
    wavelength_bins: int = 0

    def __init__(self, config: SurveyConfig, gaia_config: GaiaConfig):
        self.config = config
        self.gaia_config = gaia_config

    def load(self) -> SurveyData:
        """Load and filter survey data. Override in subclasses."""
        raise NotImplementedError

    def get_gaia_ids(
        self, ra: np.ndarray, dec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get Gaia IDs for sources.

        Note: In the simplified architecture, surveys get their Gaia IDs
        by inheriting from the label catalogues (APOGEE/GALAH) during
        cross-matching. This method returns placeholder values.

        Override in subclasses if the survey has direct Gaia IDs (e.g., BOSS).
        """
        # Return placeholders - actual Gaia IDs come from cross-matching with labels
        n_sources = len(ra)
        gaia_ids = np.full(n_sources, -1, dtype=np.int64)
        match_mask = np.zeros(n_sources, dtype=bool)
        return gaia_ids, match_mask


class DESILoader(BaseSurveyLoader):
    """Loader for DESI spectroscopic data."""

    survey_name = "desi"
    wavelength_bins = 7650

    def load(self) -> SurveyData:
        """Load DESI data from LOA training cube format."""
        logger.info(f"Loading DESI data from {self.config.data_path}")

        # TODO: Implement DESI-specific loading
        # This is a placeholder - actual implementation depends on data format
        raise NotImplementedError("DESI loader not yet implemented")


class BOSSLoader(BaseSurveyLoader):
    """Loader for BOSS spectroscopic data."""

    survey_name = "boss"
    wavelength_bins = 4506

    def load(self) -> SurveyData:
        """Load BOSS data from spAll-lite file."""
        logger.info(f"Loading BOSS data from {self.config.data_path}")

        # TODO: Implement BOSS-specific loading
        raise NotImplementedError("BOSS loader not yet implemented")

    def get_gaia_ids(
        self, ra: np.ndarray, dec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """BOSS has direct Gaia IDs in the catalogue."""
        # BOSS has GAIA_ID column directly
        # This will be handled in the load() method
        raise NotImplementedError("Handled in load()")


class LAMOSTLRSLoader(BaseSurveyLoader):
    """Loader for LAMOST LRS spectroscopic data."""

    survey_name = "lamost_lrs"
    wavelength_bins = 3800  # Approximate, depends on exact grid

    def load(self) -> SurveyData:
        """Load LAMOST LRS data."""
        logger.info(f"Loading LAMOST LRS data from {self.config.data_path}")

        # TODO: Implement LAMOST LRS-specific loading
        raise NotImplementedError("LAMOST LRS loader not yet implemented")


class LAMOSTMRSLoader(BaseSurveyLoader):
    """Loader for LAMOST MRS spectroscopic data (dual-arm)."""

    survey_name = "lamost_mrs"
    # MRS has two arms - wavelength_bins is per-arm
    wavelength_bins_b = 2000  # Blue arm (approximate)
    wavelength_bins_r = 2000  # Red arm (approximate)

    def load(self) -> SurveyData:
        """Load LAMOST MRS data (both arms)."""
        logger.info(f"Loading LAMOST MRS data from {self.config.data_path}")

        # TODO: Implement LAMOST MRS-specific loading
        # Note: Returns two flux/ivar arrays for blue and red arms
        raise NotImplementedError("LAMOST MRS loader not yet implemented")


# =============================================================================
# Label Loaders
# =============================================================================


class APOGEELabelLoader:
    """Loader for APOGEE stellar parameter labels."""

    # APOGEE column mappings to our 11 parameters
    COLUMN_MAP = {
        "teff": "TEFF",
        "logg": "LOGG",
        "fe_h": "M_H",  # APOGEE uses M_H for metallicity
        "mg_fe": "MG_FE",
        "c_fe": "C_FE",
        "si_fe": "SI_FE",
        "ni_fe": "NI_FE",
        "al_fe": "AL_FE",
        "ca_fe": "CA_FE",
        "n_fe": "N_FE",
        "mn_fe": "MN_FE",
    }

    # Error column suffix
    ERROR_SUFFIX = "_ERR"

    def __init__(self, config: APOGEELabelConfig, gaia_config: GaiaConfig):
        self.config = config
        self.gaia_config = gaia_config

    def load(self) -> LabelData:
        """Load and filter APOGEE labels."""
        logger.info(f"Loading APOGEE labels from {self.config.data_path}")

        # Load APOGEE allStar file
        table = Table.read(self.config.data_path)
        logger.info(f"  Loaded {len(table)} stars from APOGEE")

        # Apply quality filters
        mask = np.ones(len(table), dtype=bool)

        # SNR threshold
        if "SNR" in table.colnames:
            snr_mask = table["SNR"] > self.config.snr_threshold
            mask &= snr_mask
            logger.info(
                f"  After SNR > {self.config.snr_threshold}: {mask.sum()} stars"
            )

        # STAR_BAD filter
        if self.config.apply_star_bad_filter and "ASPCAPFLAG" in table.colnames:
            star_bad_mask = (table["ASPCAPFLAG"] & APOGEE_STAR_BAD) == 0
            mask &= star_bad_mask
            logger.info(f"  After STAR_BAD filter: {mask.sum()} stars")

        # Apply mask
        table = table[mask]
        logger.info(f"  Final APOGEE sample: {len(table)} stars")

        # Extract coordinates
        ra = np.array(table["RA"], dtype=np.float64)
        dec = np.array(table["DEC"], dtype=np.float64)

        # Read Gaia IDs directly from catalogue
        gaia_ids, gaia_mask = self._get_gaia_ids(table)
        logger.info(f"  Stars with valid Gaia IDs: {gaia_mask.sum()} / {len(table)}")

        # Extract values and errors
        values, errors = self._extract_parameters(table)

        # Encode per-parameter flags
        flags = self._encode_flags(table, values)

        return LabelData(
            name="apogee",
            gaia_ids=gaia_ids,
            ra=ra,
            dec=dec,
            values=values,
            errors=errors,
            flags=flags,
        )

    def _get_gaia_ids(self, table: Table) -> tuple[np.ndarray, np.ndarray]:
        """
        Read Gaia DR3 source IDs directly from APOGEE catalogue.

        APOGEE DR17 includes GAIAEDR3_SOURCE_ID column with ~99.2% coverage.
        Note: Gaia EDR3 and DR3 source IDs are identical.

        Args:
            table: APOGEE catalogue table (after quality filtering)

        Returns:
            Tuple of (gaia_ids, valid_mask):
                - gaia_ids: Gaia DR3 source IDs (int64), -1 for missing
                - valid_mask: Boolean mask of stars with valid Gaia IDs
        """
        gaia_col = self.config.gaia_id_column

        if gaia_col not in table.colnames:
            raise ValueError(
                f"Gaia ID column '{gaia_col}' not found in APOGEE catalogue. "
                f"Available columns: {table.colnames[:20]}..."
            )

        # Read Gaia IDs, handling potential masked/invalid values
        gaia_ids = np.array(table[gaia_col], dtype=np.int64)

        # Valid mask: Gaia ID > 0 (invalid entries are typically 0 or negative)
        valid_mask = gaia_ids > 0

        return gaia_ids, valid_mask

    def _extract_parameters(self, table: Table) -> tuple[np.ndarray, np.ndarray]:
        """Extract the 11 stellar parameters and their errors."""
        n_stars = len(table)
        values = np.zeros((n_stars, 11), dtype=np.float32)
        errors = np.zeros((n_stars, 11), dtype=np.float32)

        for i, param in enumerate(PARAMETER_NAMES):
            col_name = self.COLUMN_MAP[param]
            err_col_name = col_name + self.ERROR_SUFFIX

            if col_name in table.colnames:
                values[:, i] = np.array(table[col_name], dtype=np.float32)
            if err_col_name in table.colnames:
                errors[:, i] = np.array(table[err_col_name], dtype=np.float32)

        return values, errors

    def _encode_flags(self, table: Table, values: np.ndarray) -> np.ndarray:
        """
        Encode per-parameter quality flags.

        Flag bits (uint8):
            0: GRIDEDGE_BAD (PARAMFLAG bit 0)
            1: CALRANGE_BAD (PARAMFLAG bit 1)
            2: OTHER_BAD (PARAMFLAG bit 2)
            3: TEFF_CUT (PARAMFLAG bit 6)
            4: PARAM_SPECIFIC (TEFF_BAD/LOGG_BAD/etc from ASPCAPFLAG)
            5: VALUE_BOUNDS (outside physical bounds)
        """
        n_stars = len(table)
        flags = np.zeros((n_stars, 11), dtype=np.uint8)

        # PARAMFLAG-based flags (if available)
        if "PARAMFLAG" in table.colnames:
            paramflag = np.array(table["PARAMFLAG"])

            # PARAMFLAG has shape (n_stars, n_params) or similar
            # Map each bit to our flag array
            for i in range(min(11, paramflag.shape[1] if paramflag.ndim > 1 else 1)):
                pf = paramflag[:, i] if paramflag.ndim > 1 else paramflag

                flags[:, i] |= ((pf & (1 << 0)) != 0).astype(
                    np.uint8
                ) << 0  # GRIDEDGE_BAD
                flags[:, i] |= ((pf & (1 << 1)) != 0).astype(
                    np.uint8
                ) << 1  # CALRANGE_BAD
                flags[:, i] |= ((pf & (1 << 2)) != 0).astype(np.uint8) << 2  # OTHER_BAD
                flags[:, i] |= ((pf & (1 << 6)) != 0).astype(np.uint8) << 3  # TEFF_CUT

        # Physical bounds check
        bounds = {
            0: (2500, 10000),  # Teff
            1: (-1, 6),  # logg
            2: (-5, 1),  # [Fe/H]
        }
        for i, (low, high) in bounds.items():
            out_of_bounds = (values[:, i] < low) | (values[:, i] > high)
            flags[:, i] |= out_of_bounds.astype(np.uint8) << 5

        return flags


class GALAHLabelLoader:
    """Loader for GALAH stellar parameter labels."""

    # GALAH column mappings to our 11 parameters
    COLUMN_MAP = {
        "teff": "teff",
        "logg": "logg",
        "fe_h": "fe_h",
        "mg_fe": "Mg_fe",
        "c_fe": "C_fe",
        "si_fe": "Si_fe",
        "ni_fe": "Ni_fe",
        "al_fe": "Al_fe",
        "ca_fe": "Ca_fe",
        "n_fe": "N_fe",
        "mn_fe": "Mn_fe",
    }

    # Error column mapping (GALAH uses e_X format)
    ERROR_PREFIX = "e_"

    # Flag column mapping (GALAH uses flag_X_fe format)
    FLAG_PREFIX = "flag_"
    FLAG_SUFFIX = "_fe"

    def __init__(self, config: GALAHLabelConfig, gaia_config: GaiaConfig):
        self.config = config
        self.gaia_config = gaia_config

    def load(self) -> LabelData:
        """Load and filter GALAH labels."""
        logger.info(f"Loading GALAH labels from {self.config.data_path}")

        # Load GALAH allstar file
        table = Table.read(self.config.data_path)
        logger.info(f"  Loaded {len(table)} stars from GALAH")

        # Apply quality filters
        mask = np.ones(len(table), dtype=bool)

        # SNR threshold (use snr_px_ccd3)
        snr_col = "snr_px_ccd3" if "snr_px_ccd3" in table.colnames else "snr_c3_iraf"
        if snr_col in table.colnames:
            snr_mask = table[snr_col] > self.config.snr_threshold
            mask &= snr_mask
            logger.info(
                f"  After {snr_col} > {self.config.snr_threshold}: {mask.sum()} stars"
            )

        # flag_sp filter (IMPORTANT: this is the main quality flag)
        if self.config.apply_flag_sp_filter and "flag_sp" in table.colnames:
            flag_sp_mask = table["flag_sp"] == 0
            mask &= flag_sp_mask
            logger.info(f"  After flag_sp == 0: {mask.sum()} stars")

        # Apply mask
        table = table[mask]
        logger.info(f"  Final GALAH sample: {len(table)} stars")

        # Extract coordinates
        ra = np.array(table["ra"], dtype=np.float64)
        dec = np.array(table["dec"], dtype=np.float64)

        # Read Gaia IDs directly from catalogue
        gaia_ids, gaia_mask = self._get_gaia_ids(table)
        logger.info(f"  Stars with valid Gaia IDs: {gaia_mask.sum()} / {len(table)}")

        # Extract values and errors
        values, errors = self._extract_parameters(table)

        # Encode per-parameter flags
        flags = self._encode_flags(table, values)

        return LabelData(
            name="galah",
            gaia_ids=gaia_ids,
            ra=ra,
            dec=dec,
            values=values,
            errors=errors,
            flags=flags,
        )

    def _get_gaia_ids(self, table: Table) -> tuple[np.ndarray, np.ndarray]:
        """
        Read Gaia DR3 source IDs directly from GALAH catalogue.

        GALAH DR4 includes gaiadr3_source_id column with ~100% coverage.

        Args:
            table: GALAH catalogue table (after quality filtering)

        Returns:
            Tuple of (gaia_ids, valid_mask):
                - gaia_ids: Gaia DR3 source IDs (int64), -1 for missing
                - valid_mask: Boolean mask of stars with valid Gaia IDs
        """
        gaia_col = self.config.gaia_id_column

        if gaia_col not in table.colnames:
            raise ValueError(
                f"Gaia ID column '{gaia_col}' not found in GALAH catalogue. "
                f"Available columns: {table.colnames[:20]}..."
            )

        # Read Gaia IDs, handling potential masked/invalid values
        gaia_ids = np.array(table[gaia_col], dtype=np.int64)

        # Valid mask: Gaia ID > 0 (invalid entries are typically 0 or negative)
        valid_mask = gaia_ids > 0

        return gaia_ids, valid_mask

    def _extract_parameters(self, table: Table) -> tuple[np.ndarray, np.ndarray]:
        """Extract the 11 stellar parameters and their errors."""
        n_stars = len(table)
        values = np.zeros((n_stars, 11), dtype=np.float32)
        errors = np.zeros((n_stars, 11), dtype=np.float32)

        for i, param in enumerate(PARAMETER_NAMES):
            col_name = self.COLUMN_MAP[param]

            # Value column
            if col_name in table.colnames:
                values[:, i] = np.array(table[col_name], dtype=np.float32)

            # Error column (e_X format)
            err_col_name = self.ERROR_PREFIX + col_name
            if err_col_name in table.colnames:
                errors[:, i] = np.array(table[err_col_name], dtype=np.float32)

        return values, errors

    def _encode_flags(self, table: Table, values: np.ndarray) -> np.ndarray:
        """
        Encode per-parameter quality flags.

        Flag bits (uint8):
            0: FLAG_X_FE_NONZERO (flag_X_fe != 0 for this element)
            1: VALUE_BOUNDS (outside physical bounds)
            2-7: Reserved

        NOTE: flag_fe_h is BUGGED in GALAH DR4 - we use flag_sp for pre-filtering
              and only per-element flags here.
        """
        n_stars = len(table)
        flags = np.zeros((n_stars, 11), dtype=np.uint8)

        # Per-element flags from flag_X_fe columns
        # Skip flag_fe_h (index 2) as it's bugged - we rely on flag_sp pre-filter
        flag_columns = {
            3: "flag_Mg_fe",
            4: "flag_C_fe",
            5: "flag_Si_fe",
            6: "flag_Ni_fe",
            7: "flag_Al_fe",
            8: "flag_Ca_fe",
            9: "flag_N_fe",
            10: "flag_Mn_fe",
        }

        for i, col_name in flag_columns.items():
            if col_name in table.colnames:
                flag_nonzero = np.array(table[col_name]) != 0
                flags[:, i] |= flag_nonzero.astype(np.uint8) << 0

        # Physical bounds check
        bounds = {
            0: (2500, 10000),  # Teff
            1: (-1, 6),  # logg
            2: (-5, 1),  # [Fe/H]
        }
        for i, (low, high) in bounds.items():
            out_of_bounds = (values[:, i] < low) | (values[:, i] > high)
            flags[:, i] |= out_of_bounds.astype(np.uint8) << 1

        return flags


# =============================================================================
# Cross-Matcher
# =============================================================================


class CrossMatcher:
    """Cross-match surveys with label sources using UNION strategy."""

    def __init__(
        self,
        surveys: dict[str, SurveyData],
        label_sources: dict[str, LabelData],
        tolerance_arcsec: float = 0.5,
    ):
        self.surveys = surveys
        self.label_sources = label_sources
        self.tolerance_arcsec = tolerance_arcsec

    def build_master_catalogue(self) -> dict[str, Any]:
        """
        Build the master star list using UNION strategy.

        For each survey:
            1. Cross-match to APOGEE
            2. Cross-match to GALAH
            3. UNION: include if matched to EITHER

        Returns:
            Dictionary with master star list and cross-match indices.
        """
        logger.info("Building master catalogue with UNION strategy")

        # Collect all unique Gaia IDs from survey cross-matches
        all_gaia_ids = set()
        survey_matches = {}

        for survey_name, survey_data in self.surveys.items():
            survey_matches[survey_name] = {
                "apogee": {},
                "galah": {},
            }

            # Cross-match with each label source
            for label_name, label_data in self.label_sources.items():
                logger.info(f"  Cross-matching {survey_name} x {label_name}")

                idx1, idx2, seps = crossmatch_catalogues(
                    survey_data.ra,
                    survey_data.dec,
                    label_data.ra,
                    label_data.dec,
                    self.tolerance_arcsec,
                )

                survey_matches[survey_name][label_name] = {
                    "survey_idx": idx1,
                    "label_idx": idx2,
                    "separations": seps,
                }

                # Add Gaia IDs from the LABEL catalogue (not survey)
                # Labels (APOGEE/GALAH) have Gaia IDs directly from their catalogues
                matched_gaia_ids = label_data.gaia_ids[idx2]
                all_gaia_ids.update(matched_gaia_ids[matched_gaia_ids > 0])

                logger.info(f"    Found {len(idx1)} matches")

        # Build master list
        master_gaia_ids = np.array(sorted(all_gaia_ids), dtype=np.int64)
        logger.info(f"Master catalogue: {len(master_gaia_ids)} unique stars")

        # Assign Gaia IDs to surveys based on cross-match results
        self._assign_survey_gaia_ids(survey_matches)

        return {
            "gaia_ids": master_gaia_ids,
            "survey_matches": survey_matches,
        }

    def _assign_survey_gaia_ids(self, survey_matches: dict) -> None:
        """
        Assign Gaia IDs to surveys based on cross-match with label catalogues.

        Surveys inherit Gaia IDs from their matched label sources (APOGEE/GALAH).
        When a survey star matches multiple label sources, we use the first match.
        """
        logger.info("Assigning Gaia IDs to surveys from cross-match results")

        for survey_name, survey_data in self.surveys.items():
            n_assigned = 0
            matches = survey_matches.get(survey_name, {})

            for label_name, match_info in matches.items():
                label_data = self.label_sources.get(label_name)
                if label_data is None:
                    continue

                survey_idx = match_info["survey_idx"]
                label_idx = match_info["label_idx"]

                # Assign Gaia IDs from labels to surveys
                for si, li in zip(survey_idx, label_idx, strict=False):
                    if survey_data.gaia_ids[si] < 0:  # Not already assigned
                        survey_data.gaia_ids[si] = label_data.gaia_ids[li]
                        n_assigned += 1

            logger.info(f"  {survey_name}: assigned {n_assigned} Gaia IDs")


# =============================================================================
# HDF5 Writer
# =============================================================================


class CatalogueWriter:
    """Write the super-catalogue to HDF5."""

    def __init__(self, config: OutputConfig):
        self.config = config
        self.compression = config.compression
        self.compression_opts = config.compression_opts

    def write(
        self,
        output_path: Path,
        master_gaia_ids: np.ndarray,
        surveys: dict[str, SurveyData],
        labels: dict[str, LabelData],
        survey_matches: dict,
    ):
        """Write the full catalogue to HDF5."""
        logger.info(f"Writing catalogue to {output_path}")

        n_total = len(master_gaia_ids)

        with h5py.File(output_path, "w") as f:
            # Create metadata group
            meta = f.create_group("metadata")
            meta.create_dataset("gaia_id", data=master_gaia_ids)

            # RA/DEC will be filled from surveys (prefer first available)
            ra = np.zeros(n_total, dtype=np.float64)
            dec = np.zeros(n_total, dtype=np.float64)

            # Create surveys group
            surveys_grp = f.create_group("surveys")
            for survey_name, survey_data in surveys.items():
                self._write_survey(
                    surveys_grp, survey_name, survey_data, master_gaia_ids
                )

                # Fill RA/DEC from first available survey
                gaia_to_idx = {gid: i for i, gid in enumerate(master_gaia_ids)}
                for i, gid in enumerate(survey_data.gaia_ids):
                    if gid in gaia_to_idx:
                        master_idx = gaia_to_idx[gid]
                        if ra[master_idx] == 0:  # Only fill if not already set
                            ra[master_idx] = survey_data.ra[i]
                            dec[master_idx] = survey_data.dec[i]

            meta.create_dataset("ra", data=ra)
            meta.create_dataset("dec", data=dec)

            # Create labels group
            labels_grp = f.create_group("labels")
            for label_name, label_data in labels.items():
                self._write_labels(labels_grp, label_name, label_data, master_gaia_ids)

            # Write attributes
            f.attrs["n_stars"] = n_total
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = SURVEY_NAMES
            f.attrs["creation_date"] = datetime.now().isoformat()
            f.attrs["version"] = VERSION

        logger.info(f"  Wrote {n_total} stars to {output_path}")

    def _write_survey(
        self,
        parent: h5py.Group,
        name: str,
        data: SurveyData,
        master_gaia_ids: np.ndarray,
    ):
        """Write a single survey's data to HDF5."""
        grp = parent.create_group(name)
        n_total = len(master_gaia_ids)

        # Build index mapping
        gaia_to_master = {gid: i for i, gid in enumerate(master_gaia_ids)}
        gaia_to_survey = {gid: i for i, gid in enumerate(data.gaia_ids) if gid > 0}

        # Create row-matched arrays
        n_wave = data.flux.shape[1] if data.flux.ndim > 1 else len(data.wavelength)
        flux = np.zeros((n_total, n_wave), dtype=np.float32)
        ivar = np.zeros((n_total, n_wave), dtype=np.float32)
        snr = np.zeros(n_total, dtype=np.float32)

        # Fill from survey data
        for gid, survey_idx in gaia_to_survey.items():
            if gid in gaia_to_master:
                master_idx = gaia_to_master[gid]
                flux[master_idx] = data.flux[survey_idx]
                ivar[master_idx] = data.ivar[survey_idx]
                snr[master_idx] = data.snr[survey_idx]

        # Write datasets with compression
        grp.create_dataset(
            "flux",
            data=flux,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        grp.create_dataset(
            "ivar",
            data=ivar,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        grp.create_dataset("wavelength", data=data.wavelength)
        grp.create_dataset("snr", data=snr)

    def _write_labels(
        self,
        parent: h5py.Group,
        name: str,
        data: LabelData,
        master_gaia_ids: np.ndarray,
    ):
        """Write a single label source's data to HDF5."""
        grp = parent.create_group(name)
        n_total = len(master_gaia_ids)

        # Build index mapping
        gaia_to_master = {gid: i for i, gid in enumerate(master_gaia_ids)}
        gaia_to_label = {gid: i for i, gid in enumerate(data.gaia_ids) if gid > 0}

        # Create row-matched arrays
        values = np.zeros((n_total, 11), dtype=np.float32)
        errors = np.zeros((n_total, 11), dtype=np.float32)
        flags = np.zeros((n_total, 11), dtype=np.uint8)

        # Fill from label data
        for gid, label_idx in gaia_to_label.items():
            if gid in gaia_to_master:
                master_idx = gaia_to_master[gid]
                values[master_idx] = data.values[label_idx]
                errors[master_idx] = data.errors[label_idx]
                flags[master_idx] = data.flags[label_idx]

        # Write datasets with compression
        grp.create_dataset(
            "values",
            data=values,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        grp.create_dataset(
            "errors",
            data=errors,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        grp.create_dataset("flags", data=flags)


# =============================================================================
# Main Builder
# =============================================================================


class CatalogueBuilder:
    """Main orchestrator for building the super-catalogue."""

    def __init__(self, config: CatalogueConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.surveys: dict[str, SurveyData] = {}
        self.labels: dict[str, LabelData] = {}

    def build(self):
        """Execute the full build pipeline."""
        logger.info("=" * 60)
        logger.info("DOROTHY Super-Catalogue Builder")
        logger.info("=" * 60)

        if self.dry_run:
            logger.info("DRY RUN MODE - no files will be written")

        # Step 1: Load label sources
        self._load_labels()

        # Step 2: Load surveys
        self._load_surveys()

        # Step 3: Cross-match and build master list
        matcher = CrossMatcher(
            self.surveys,
            self.labels,
            self.config.gaia.tolerance_arcsec,
        )
        master = matcher.build_master_catalogue()

        # Step 4: Write output
        if not self.dry_run:
            writer = CatalogueWriter(self.config.output)
            writer.write(
                self.config.output.path,
                master["gaia_ids"],
                self.surveys,
                self.labels,
                master["survey_matches"],
            )

        logger.info("=" * 60)
        logger.info("Build complete!")
        logger.info("=" * 60)

    def _load_labels(self):
        """Load all enabled label sources."""
        logger.info("Loading label sources...")

        if self.config.labels.apogee and self.config.labels.apogee.enabled:
            loader = APOGEELabelLoader(self.config.labels.apogee, self.config.gaia)
            self.labels["apogee"] = loader.load()

        if self.config.labels.galah and self.config.labels.galah.enabled:
            loader = GALAHLabelLoader(self.config.labels.galah, self.config.gaia)
            self.labels["galah"] = loader.load()

        if not self.labels:
            raise ValueError("No label sources enabled")

    def _load_surveys(self):
        """Load all enabled surveys."""
        logger.info("Loading surveys...")

        loaders = {
            "desi": (DESILoader, self.config.surveys.desi),
            "boss": (BOSSLoader, self.config.surveys.boss),
            "lamost_lrs": (LAMOSTLRSLoader, self.config.surveys.lamost_lrs),
            "lamost_mrs": (LAMOSTMRSLoader, self.config.surveys.lamost_mrs),
        }

        for name, (loader_cls, survey_config) in loaders.items():
            if survey_config and survey_config.enabled:
                loader = loader_cls(survey_config, self.config.gaia)
                self.surveys[name] = loader.load()

        if not self.surveys:
            raise ValueError("No surveys enabled")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a unified multi-survey super-catalogue for DOROTHY.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/build_catalogue.py --config catalogue_config.yaml
    python scripts/build_catalogue.py --config catalogue_config.yaml --dry-run
    python scripts/build_catalogue.py --config catalogue_config.yaml --surveys desi boss
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Override output path from config",
    )

    parser.add_argument(
        "--surveys",
        nargs="+",
        choices=["desi", "boss", "lamost_lrs", "lamost_mrs"],
        help="Only process these surveys (overrides config)",
    )

    parser.add_argument(
        "--label-sources",
        nargs="+",
        choices=["apogee", "galah"],
        help="Only use these label sources (overrides config)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and report what would be done",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = CatalogueConfig.from_yaml(args.config)

    # Override output path if specified
    if args.output:
        config.output.path = args.output

    # Override surveys if specified
    if args.surveys:
        for survey_name in ["desi", "boss", "lamost_lrs", "lamost_mrs"]:
            survey_config = getattr(config.surveys, survey_name)
            if survey_config:
                survey_config.enabled = survey_name in args.surveys

    # Override label sources if specified
    if args.label_sources:
        if config.labels.apogee:
            config.labels.apogee.enabled = "apogee" in args.label_sources
        if config.labels.galah:
            config.labels.galah.enabled = "galah" in args.label_sources

    # Build catalogue
    builder = CatalogueBuilder(config, dry_run=args.dry_run)
    builder.build()


if __name__ == "__main__":
    main()
