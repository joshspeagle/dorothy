"""Data loading, normalization, and preprocessing utilities."""

from dorothy.data.augmentation import DynamicBlockMasking
from dorothy.data.catalogue_loader import CatalogueData, CatalogueInfo, CatalogueLoader
from dorothy.data.fits_loader import (
    APOGEE_COLUMN_MAP,
    PARAMETER_BOUNDS,
    FITSLoader,
    SpectralData,
    apply_quality_filter,
    ivar_to_error_and_mask,
    normalize_spectrum,
    split_data,
)
from dorothy.data.gaia_crossmatch import crossmatch_to_gaia_dr3
from dorothy.data.normalizer import LabelNormalizer, ParameterStats


__all__ = [
    # FITS loading
    "FITSLoader",
    "SpectralData",
    "normalize_spectrum",
    "apply_quality_filter",
    "split_data",
    "ivar_to_error_and_mask",
    "APOGEE_COLUMN_MAP",
    "PARAMETER_BOUNDS",
    # Catalogue loading
    "CatalogueLoader",
    "CatalogueData",
    "CatalogueInfo",
    # Gaia cross-matching
    "crossmatch_to_gaia_dr3",
    # Label normalization
    "LabelNormalizer",
    "ParameterStats",
    # Augmentation
    "DynamicBlockMasking",
]
