"""
Loader for the DOROTHY super-catalogue HDF5 format.

This module provides utilities to load the unified multi-survey super-catalogue,
enabling training across multiple surveys with consistent label handling.

Each survey in the catalogue has its own independent row structure and
corresponding labels. The loader automatically maps surveys to their
appropriate label sources.

Examples:
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

from dorothy.config.schema import STELLAR_PARAMETERS
from dorothy.data.fits_loader import normalize_spectra as _normalize_spectra
from dorothy.data.fits_loader import zero_invalid_pixels

# Default (v1/v2 [X/Fe]) parameter names; catalogues may define their own
PARAMETER_NAMES = list(STELLAR_PARAMETERS)

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


def _read_parameter_names(f: h5py.File) -> list[str]:
    """Read label parameter names from an open catalogue.

    Uses the ``parameter_names`` file attribute when present. Older files
    without it fall back to the standard names, sized to the label width.
    """
    if "parameter_names" in f.attrs:
        return [
            n.decode() if isinstance(n, bytes) else str(n)
            for n in f.attrs["parameter_names"]
        ]
    n_params = len(PARAMETER_NAMES)
    for source in f.get("labels", {}):
        if "values" in f["labels"][source]:
            n_params = f["labels"][source]["values"].shape[1]
            break
    if n_params > len(PARAMETER_NAMES):
        raise ValueError(
            f"Catalogue has {n_params} label columns but no parameter_names "
            "attribute; cannot infer parameter names"
        )
    return list(PARAMETER_NAMES[:n_params])


def _split_dual_arm_spectra(
    spectra: np.ndarray, survey_grp: h5py.Group
) -> tuple[np.ndarray, np.ndarray]:
    """Convert padded dual-arm (LAMOST MRS) spectra to concatenated flux/ivar.

    The catalogue stores MRS as (N, 4, max_wave) = [flux_b, ivar_b, flux_r,
    ivar_r], padded to the longer arm. Each arm is trimmed to its own
    wavelength grid and the arms are concatenated (blue then red), matching
    ``wavelength_b`` + ``wavelength_r``.
    """
    n_wave_b = survey_grp["wavelength_b"].shape[0]
    n_wave_r = survey_grp["wavelength_r"].shape[0]
    flux = np.concatenate([spectra[:, 0, :n_wave_b], spectra[:, 2, :n_wave_r]], axis=1)
    ivar = np.concatenate([spectra[:, 1, :n_wave_b], spectra[:, 3, :n_wave_r]], axis=1)
    return flux, ivar


def _survey_wavelength(survey_grp: h5py.Group) -> np.ndarray:
    """A survey's wavelength grid, one value per model input pixel.

    Dual-arm spectra (LAMOST MRS) have no ``wavelength`` dataset: their grid
    is ``wavelength_b`` then ``wavelength_r``, as ``_split_dual_arm_spectra``
    orders the pixels.
    """
    if "spectra" in survey_grp:
        return np.concatenate(
            [survey_grp["wavelength_b"][:], survey_grp["wavelength_r"][:]]
        )
    return survey_grp["wavelength"][:]


# Target size of one HDF5 read when selecting rows of a spectra dataset. Bounds
# the temporary buffers (chunked datasets are read in whole chunk-row bands).
_ROW_READ_BYTES = 64 * 1024**2


def _iter_row_blocks(
    dset: h5py.Dataset, rows: np.ndarray, out: np.ndarray | None = None
):
    """Read ``dset[rows]`` block by block, for sorted unique row indices.

    Yields ``(lo, hi, block)`` with ``block`` equal to ``dset[rows[lo:hi]]``.
    With ``out`` (shape ``(len(rows),) + dset.shape[1:]``), each block is the
    view ``out[lo:hi]``; without it, a view of a reusable buffer that is only
    valid until the next iteration.

    Only the requested rows are transferred and no full-dataset temporary is
    made. Contiguous datasets (as v2 catalogues are written) use HDF5 point
    selections. Chunked (e.g. gzip-compressed) datasets are read in
    chunk-aligned row bands, skipping bands without requested rows, so each
    chunk is decompressed once (point selections re-decompress chunks).
    """
    rows = np.asarray(rows, dtype=np.int64)
    n = len(rows)
    if n == 0:
        return
    row_shape = dset.shape[1:]
    row_bytes = max(1, int(np.prod(row_shape)) * dset.dtype.itemsize)

    if dset.chunks is None:
        step = max(1, _ROW_READ_BYTES // row_bytes)
        buf = (
            None
            if out is not None
            else np.empty((min(step, n),) + row_shape, dset.dtype)
        )
        for lo in range(0, n, step):
            hi = min(n, lo + step)
            if out is not None:
                dset.read_direct(out, source_sel=rows[lo:hi], dest_sel=np.s_[lo:hi])
                yield lo, hi, out[lo:hi]
            else:
                dset.read_direct(buf, source_sel=rows[lo:hi], dest_sel=np.s_[: hi - lo])
                yield lo, hi, buf[: hi - lo]
        return

    chunk_rows = dset.chunks[0]
    band = chunk_rows * max(1, _ROW_READ_BYTES // (row_bytes * chunk_rows))
    band_buf = np.empty((min(band, dset.shape[0]),) + row_shape, dset.dtype)
    buf = None if out is not None else np.empty_like(band_buf)
    band_ids = rows // band
    edges = np.flatnonzero(np.diff(band_ids)) + 1
    for lo, hi in zip(np.r_[0, edges].tolist(), np.r_[edges, n].tolist(), strict=True):
        start = int(band_ids[lo]) * band
        stop = min(start + band, dset.shape[0])
        dset.read_direct(
            band_buf, source_sel=np.s_[start:stop], dest_sel=np.s_[: stop - start]
        )
        block = out[lo:hi] if out is not None else buf[: hi - lo]
        np.take(band_buf, rows[lo:hi] - start, axis=0, out=block, mode="clip")
        yield lo, hi, block


def _read_rows(dset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    """Return ``dset[:][rows]`` (sorted unique ``rows``), reading only those rows."""
    out = np.empty((len(rows),) + dset.shape[1:], dtype=dset.dtype)
    for _ in _iter_row_blocks(dset, rows, out=out):
        pass
    return out


def _read_dual_arm_rows(
    survey_grp: h5py.Group, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Selected rows of padded dual-arm (LAMOST MRS) spectra as flux/ivar.

    Same result as ``_split_dual_arm_spectra(survey_grp["spectra"][:][rows],
    survey_grp)``, but only the requested rows are read, block by block.
    """
    dset = survey_grp["spectra"]
    n_wave = dset.shape[2]
    n_b = min(survey_grp["wavelength_b"].shape[0], n_wave)
    n_r = min(survey_grp["wavelength_r"].shape[0], n_wave)
    flux = np.empty((len(rows), n_b + n_r), dtype=dset.dtype)
    ivar = np.empty_like(flux)
    for lo, hi, block in _iter_row_blocks(dset, rows):
        flux[lo:hi, :n_b] = block[:, 0, :n_b]
        flux[lo:hi, n_b:] = block[:, 2, :n_r]
        ivar[lo:hi, :n_b] = block[:, 1, :n_b]
        ivar[lo:hi, n_b:] = block[:, 3, :n_r]
    return flux, ivar


def _preprocess_spectra(
    flux: np.ndarray, ivar: np.ndarray, normalize: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Float32 spectra with invalid pixels zeroed, normalized if ``normalize``.

    Works in place: float32 inputs are modified.
    """
    flux = np.asarray(flux, dtype=np.float32)
    ivar = np.asarray(ivar, dtype=np.float32)
    if normalize:
        return _normalize_spectra(flux, ivar, inplace=True)
    return zero_invalid_pixels(flux, ivar, inplace=True)


def _spectra_to_model_inputs(
    flux: np.ndarray, ivar: np.ndarray, block_rows: int = 256
) -> np.ndarray:
    """Stack ``[flux, sigma, mask]`` into (N, 3, n_wave) model inputs.

    ``sigma = 1 / sqrt(ivar)`` and ``mask = 1`` where ``ivar > 0``, both 0
    elsewhere. Same result as ``np.stack([flux, sigma, mask], axis=1)`` (with
    sigma computed in ivar's dtype), but written straight into the output a
    block of rows at a time, without boolean-indexed copies or full-size
    sigma/mask temporaries.
    """
    out = np.empty(
        (flux.shape[0], 3) + flux.shape[1:],
        dtype=np.result_type(flux.dtype, ivar.dtype, np.float32),
    )
    # Compute sigma in place when it already has the output dtype
    in_place = ivar.dtype == out.dtype and np.issubdtype(ivar.dtype, np.floating)
    for start in range(0, flux.shape[0], block_rows):
        rows = slice(start, start + block_rows)
        iv = ivar[rows]
        valid = iv > 0
        block = out[rows]
        block[:, 0] = flux[rows]
        if in_place:
            sigma = block[:, 1]
            # Warnings can only come from ivar <= 0 or NaN, zeroed below
            with np.errstate(divide="ignore", invalid="ignore"):
                np.sqrt(iv, out=sigma)
                np.divide(1.0, sigma, out=sigma)
            np.copyto(sigma, 0, where=~valid)
        else:
            sigma = np.zeros_like(iv)
            sigma[valid] = 1.0 / np.sqrt(iv[valid])
            block[:, 1] = sigma
        block[:, 2] = valid
    return out


def _match_ids(keys: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the positions of Gaia IDs in an array of IDs.

    Returns ``(found, idx)``: ``found[j]`` tells whether ``query[j]`` is one
    of the keys, and ``idx`` holds the key positions of the found queries, in
    query order. For repeated keys the last position wins.
    """
    keys = np.asarray(keys).astype(np.int64, copy=False)
    query = np.asarray(query).astype(np.int64, copy=False)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    # Last of any equal keys (stable sort keeps them in position order)
    pos = np.searchsorted(sorted_keys, query, side="right") - 1
    found = pos >= 0
    found[found] = sorted_keys[pos[found]] == query[found]
    return found, order[pos[found]].astype(np.intp, copy=False)


def _label_gaia_ids(label_grp: h5py.Group) -> np.ndarray:
    """The Gaia IDs of a label group's rows, which match its labels to stars."""
    if "gaia_id" not in label_grp:
        raise ValueError(
            f"{label_grp.name} has no gaia_id dataset, which is needed to "
            "match its labels to the catalogue stars"
        )
    return label_grp["gaia_id"][:]


def _flag_bit_count(flags: np.ndarray) -> np.ndarray:
    """Number of set flag bits of each star (row), over all its parameters."""
    return np.bitwise_count(flags.astype(np.uint64)).sum(axis=1)


def _label_array(
    values: np.ndarray, errors: np.ndarray, flags: np.ndarray, max_flag_bits: int
) -> np.ndarray:
    """Labels in the (N, 3, n_params) format [values, errors, mask].

    A label is valid (mask 1) when its flag is 0 and its error is positive.
    With ``max_flag_bits`` >= 0, every label of a star with more set flag
    bits than ``max_flag_bits`` is masked.
    """
    valid = (flags == 0) & (errors > 0)
    if max_flag_bits >= 0:
        valid &= (_flag_bit_count(flags) <= max_flag_bits)[:, None]
    return np.stack([values, errors, valid.astype(np.float32)], axis=1)


def _align_labels(
    label_grp: h5py.Group, gaia_ids: np.ndarray, max_flag_bits: int
) -> np.ndarray:
    """A label group's labels for the stars ``gaia_ids``, matched by Gaia ID.

    Returns the ``_label_array`` format (len(gaia_ids), 3, n_params); stars
    without a row in the group get zeros, so their mask is 0.
    """
    found, idx = _match_ids(gaia_ids, _label_gaia_ids(label_grp))
    rows = np.flatnonzero(found)
    shape = (len(gaia_ids), label_grp["values"].shape[1])
    values = np.zeros(shape, dtype=np.float32)
    errors = np.zeros(shape, dtype=np.float32)
    flags = np.zeros(shape, dtype=np.uint8)
    values[idx] = label_grp["values"][:][rows]
    errors[idx] = label_grp["errors"][:][rows]
    flags[idx] = label_grp["flags"][:][rows]
    return _label_array(values, errors, flags, max_flag_bits)


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
        mask = _flag_bit_count(self.label_flags) <= max_flag_bits
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

    Examples:
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
            Normalized if requested, and 0 at invalid pixels
            (:func:`~dorothy.data.fits_loader.zero_invalid_pixels`).
        ivar: Per-survey inverse variance arrays with same structure as flux,
            0 at invalid pixels.
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
        labels_dict: Per label source, labels (n_total, 3, n_params), when
            several sources are loaded.
        has_labels_dict: Per label source, whether each star has at least one
            valid label after the flag filter (mask 1 in ``labels_dict``).
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
    parameter_names: list[str] | None = None  # Label parameter names (n_params,)

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
            Boolean mask of shape (n_total,): stars with at least one valid
            label (mask 1) from the source.

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

    Examples:
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
            parameter_names = _read_parameter_names(f)
            survey_names = list(f.attrs["survey_names"])
            creation_date = f.attrs["creation_date"]
            version = f.attrs["version"]

            # Get survey info
            surveys = {}
            if "surveys" in f:
                for name in f["surveys"]:
                    grp = f["surveys"][name]
                    wavelength = _survey_wavelength(grp)
                    snr = grp["snr"][:]
                    n_with_data = int((snr > 0).sum())
                    surveys[name] = {
                        "wavelength_bins": len(wavelength),
                        "n_with_data": n_with_data,
                        "wavelength_range": (
                            float(wavelength[0]),
                            float(wavelength[-1]),
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

        with h5py.File(self.path, "r") as f:
            # Validate inputs
            if survey not in f["surveys"]:
                raise ValueError(
                    f"Survey '{survey}' not in catalogue. "
                    f"Available: {list(f['surveys'].keys())}"
                )
            label_source = self._resolve_label_source([survey], label_source, f)

            # Load survey data
            survey_grp = f["surveys"][survey]
            snr = survey_grp["snr"][:]

            # Handle different flux storage formats
            if "flux" in survey_grp:
                flux = survey_grp["flux"][:]
                ivar = survey_grp["ivar"][:]
            elif "spectra" in survey_grp:
                flux, ivar = _split_dual_arm_spectra(
                    survey_grp["spectra"][:], survey_grp
                )
            else:
                raise ValueError(f"Survey '{survey}' has no flux or spectra data")
            wavelength = _survey_wavelength(survey_grp)

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
        normalize_spectra: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load data ready for training in unified 3-channel format.

        This is a convenience method that loads complete, high-quality data
        and returns it in the format expected by the DOROTHY trainer.

        Args:
            survey: Survey to load from.
            label_source: Label source to use. If None, auto-determined from survey.
            max_flag_bits: Maximum flag bits allowed (0 = highest quality).
            normalize_spectra: Apply per-spectrum median/IQR normalization to
                flux (and rescale ivar) before building the model inputs.
                Training and inference must use the same setting.

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

        flux, ivar = _preprocess_spectra(data.flux, data.ivar, normalize_spectra)
        X = _spectra_to_model_inputs(flux, ivar)
        # filter_by_flags removed the stars with too many set flag bits
        y = _label_array(data.labels, data.label_errors, data.label_flags, -1)
        return X, y

    def load_raw_spectra(
        self,
        survey: str,
    ) -> dict[str, np.ndarray]:
        """
        Load a survey's datasets exactly as stored in the HDF5 file.

        LAMOST MRS stores padded ``spectra`` of shape (N, 4, max_wave) =
        [flux_b, ivar_b, flux_r, ivar_r] with ``wavelength_b`` and
        ``wavelength_r``; the other surveys store ``flux``, ``ivar`` and
        ``wavelength``.

        Args:
            survey: Survey to load from.

        Returns:
            Dictionary of the survey group's datasets, plus ``channel_names``
            when the group has that attribute.

        Raises:
            ValueError: If the survey is not in the catalogue.
        """
        with h5py.File(self.path, "r") as f:
            if survey not in f["surveys"]:
                raise ValueError(
                    f"Survey '{survey}' not in catalogue. "
                    f"Available: {list(f['surveys'].keys())}"
                )

            survey_grp = f["surveys"][survey]
            result = {name: survey_grp[name][:] for name in survey_grp}
            if "channel_names" in survey_grp.attrs:
                result["channel_names"] = list(survey_grp.attrs["channel_names"])

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
        dictionary. In "intersection" mode, only stars with data (snr > 0) in
        ALL specified surveys are returned (useful for cross-survey
        comparisons). In "union" mode, all stars from all surveys are returned.

        Args:
            surveys: List of survey names to load.
            mode: How to combine surveys:
                - "union": Return all stars from all surveys (default)
                - "intersection": Return only stars with data in every survey
            deduplicate: If True (default), keep only one observation per star
                (the first occurrence). If False, keep all observations.

        Returns:
            Dictionary mapping survey names to CatalogueData objects.
            In "intersection" mode, all CatalogueData objects are filtered
            to the same set of stars (ordered consistently by Gaia ID).

        Examples:
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
            survey_data = {}
            survey_rows = {}
            common_ids = None
            for survey in surveys:
                data = self.load(survey=survey)
                # A star belongs to a survey only where the survey has data;
                # v2 surveys have a row for every star in the catalogue
                has_data = data.snr > 0
                if np.issubdtype(data.gaia_ids.dtype, np.integer):
                    has_data &= data.gaia_ids > 0
                rows = np.flatnonzero(has_data)
                ids = data.gaia_ids[rows]
                common_ids = (
                    ids if common_ids is None else np.intersect1d(common_ids, ids)
                )
                survey_data[survey] = data
                survey_rows[survey] = rows

            if len(common_ids) == 0:
                raise ValueError(f"No common stars found across surveys: {surveys}")

            for survey in surveys:
                data = survey_data[survey]
                rows = survey_rows[survey]
                # First row with data of each common star, in Gaia ID order
                ids, first = np.unique(data.gaia_ids[rows], return_index=True)
                sorted_indices = rows[first[np.isin(ids, common_ids)]]

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
        self, surveys: list[str], label_source: str | None, f: h5py.File
    ) -> str:
        """
        Resolve label source name with fallback logic.

        Supports both v1 schema (per-survey label groups like 'apogee_boss')
        and v2 schema (unified label groups like 'apogee', 'galah').

        Resolution order:
        1. If label_source specified and exists -> use it directly
        2. If label_source specified, try {label_source}_{survey} (v1) for
           each survey in order
        3. If label_source is None, for the first survey:
           a. Try SURVEY_LABEL_MAP (v1 schema)
           b. Try 'apogee' directly (v2 schema)
           c. Try 'galah' directly (v2 schema)
           d. Try 'apogee_{survey}' pattern (v1 fallback)
        """
        if "labels" not in f:
            raise ValueError(f"Catalogue {self.path} has no 'labels' group")
        labels = f["labels"]
        if label_source is None:
            survey = surveys[0]
            candidates = [
                SURVEY_LABEL_MAP.get(survey),
                "apogee",
                "galah",
                f"apogee_{survey}",
            ]
            error = f"No label source found for survey '{survey}'"
        elif label_source in labels:
            return label_source
        else:
            candidates = [f"{label_source}_{survey}" for survey in surveys]
            error = f"Label source '{label_source}' not in catalogue"
        for candidate in candidates:
            if candidate is not None and candidate in labels:
                return candidate
        raise ValueError(f"{error}. Available: {list(labels)}")

    def _load_survey_metadata(
        self, survey: str, label_source: str, f: h5py.File
    ) -> dict:
        """
        Load only metadata for a survey (no spectra).

        Returns dict with gaia_ids, snr, ra, dec, labels, label_errors, label_flags,
        wavelength grid, and storage format.
        """
        survey_grp = f["surveys"][survey]
        label_grp = f["labels"][label_source]

        if "flux" in survey_grp:
            storage_format = "flux"
        elif "spectra" in survey_grp:
            storage_format = "spectra"
        else:
            raise ValueError(f"Survey '{survey}' has no flux or spectra data")
        wavelength = _survey_wavelength(survey_grp)

        # Load metadata arrays (small compared to spectra)
        snr = survey_grp["snr"][:]
        n_stars = len(snr)

        # Load IDs
        if "gaia_id" in label_grp:
            gaia_ids = label_grp["gaia_id"][:]
        elif "metadata" in f and "gaia_id" in f["metadata"]:
            gaia_ids = f["metadata"]["gaia_id"][:]
        else:
            raise ValueError(
                f"Catalogue {self.path} has no Gaia IDs (metadata/gaia_id or "
                f"labels/{label_source}/gaia_id), which dense loading needs to "
                "match stars across surveys"
            )

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
            "n_stars": n_stars,
            "storage_format": storage_format,
        }

    def _scatter_spectra(
        self,
        survey_grp: h5py.Group,
        rows: np.ndarray,
        dest_rows: np.ndarray,
        storage_format: str,
        flux: np.ndarray,
        ivar: np.ndarray,
    ) -> None:
        """
        Read the given (sorted) rows and write them to ``dest_rows`` of the
        dense ``flux``/``ivar`` arrays, block by block.

        The "spectra" (LAMOST MRS) layout gives [flux_b, flux_r] and
        [ivar_b, ivar_r], each arm trimmed to its own wavelength grid.
        """
        if storage_format == "flux":
            for name, dest in (("flux", flux), ("ivar", ivar)):
                for lo, hi, block in _iter_row_blocks(survey_grp[name], rows):
                    dest[dest_rows[lo:hi]] = block
        else:
            dset = survey_grp["spectra"]
            n_wave = dset.shape[2]
            n_b = min(survey_grp["wavelength_b"].shape[0], n_wave)
            n_r = min(survey_grp["wavelength_r"].shape[0], n_wave)
            for lo, hi, block in _iter_row_blocks(dset, rows):
                idx = dest_rows[lo:hi]
                flux[idx, :n_b] = block[:, 0, :n_b]
                flux[idx, n_b:] = block[:, 2, :n_r]
                ivar[idx, :n_b] = block[:, 1, :n_b]
                ivar[idx, n_b:] = block[:, 3, :n_r]

    def _collect_survey_metadata_pass1(
        self,
        surveys: list[str],
        label_source: str,
        f: h5py.File,
    ) -> tuple[dict[str, dict], dict[str, str], np.ndarray, int]:
        """
        Pass 1: Load metadata only and identify duplicates per survey.

        Args:
            surveys: List of survey names to process.
            label_source: Primary label source to use.
            f: Open HDF5 file handle.

        Returns:
            Tuple of:
                - survey_meta: Dict mapping survey -> metadata dict (includes
                  duplicate_indices, unique_indices, duplicate_gaia_ids,
                  unique_gaia_ids; indices ascending)
                - survey_resolved_labels: Dict mapping survey -> resolved label source name
                - all_gaia_ids_sorted: Sorted array of all unique Gaia IDs
                  (merged row i holds all_gaia_ids_sorted[i])
                - n_total: Total number of unique stars
        """
        survey_meta: dict[str, dict] = {}
        survey_resolved_labels: dict[str, str] = {}
        survey_valid_ids: list[np.ndarray] = []

        for survey in surveys:
            print(f"Loading {survey}...")
            resolved_label = self._resolve_label_source([survey], label_source, f)
            survey_resolved_labels[survey] = resolved_label

            meta = self._load_survey_metadata(survey, resolved_label, f)
            survey_meta[survey] = meta

            # Identify duplicates vs unique stars
            # In v2 catalogues, all surveys have the same row count but SNR=0 for missing data
            gaia_ids = meta["gaia_ids"]
            snr = meta["snr"]

            # Rows with a valid gaia_id and data: snr > 0, which excludes a NaN
            # SNR, as in the sparse loader
            rows = np.flatnonzero((gaia_ids > 0) & (snr > 0))
            valid_ids = gaia_ids[rows].astype(np.int64)

            # Count occurrences
            unique_ids, counts = np.unique(valid_ids, return_counts=True)
            duplicate_gaia_ids = unique_ids[counts > 1]
            unique_gaia_ids = unique_ids[counts == 1]

            # Split the rows to load into duplicates and unique stars
            is_dup = np.isin(gaia_ids[rows], duplicate_gaia_ids)

            meta["duplicate_indices"] = rows[is_dup].astype(np.intp)
            meta["unique_indices"] = rows[~is_dup].astype(np.intp)
            meta["duplicate_gaia_ids"] = duplicate_gaia_ids
            meta["unique_gaia_ids"] = unique_gaia_ids

            # Collect all valid Gaia IDs
            survey_valid_ids.append(unique_ids)

            n_dups = len(duplicate_gaia_ids)
            n_unique = len(unique_gaia_ids)
            print(f"  {n_unique:,} unique stars, {n_dups:,} stars with duplicates")

        # Create sorted union of all Gaia IDs
        if survey_valid_ids:
            all_gaia_ids_sorted = np.unique(np.concatenate(survey_valid_ids))
        else:
            all_gaia_ids_sorted = np.array([], dtype=np.int64)
        n_total = len(all_gaia_ids_sorted)

        print(f"Total unique stars: {n_total:,}")

        return (
            survey_meta,
            survey_resolved_labels,
            all_gaia_ids_sorted,
            n_total,
        )

    def _check_for_duplicates(
        self,
        survey_meta: dict[str, dict],
        surveys: list[str],
    ) -> None:
        """
        Check for duplicate observations and raise an error if found.

        Training needs one row per star. The error tells the user to rebuild
        the catalogue with ``scripts/build_catalogue_v2.py``.

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
                f"Training needs one row per star. Rebuild the catalogue with\n\n"
                f"  python scripts/build_catalogue_v2.py --output <new file>.h5\n\n"
                f"which keeps each star's highest-S/N spectrum, and set\n"
                f"data.catalogue_path to the new file.\n"
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
            This method needs one row per star, as catalogues built by
            ``scripts/build_catalogue_v2.py`` have; duplicate rows raise an error.

        Args:
            surveys: List of survey names to load and merge.
            label_source: Primary label source to use for combined labels.

        Returns:
            MergedCatalogueData with outer join structure.

        Raises:
            ValueError: If duplicate observations are found in the catalogue,
                or it has no Gaia IDs.

        Examples:
            >>> merged = loader.load_merged(["boss", "lamost_lrs", "desi"])
            >>> print(f"Total unique stars: {merged.n_stars}")
            >>> print(f"Coverage: {merged.get_coverage_stats()}")
        """
        with h5py.File(self.path, "r") as f:
            n_params = len(_read_parameter_names(f))
            # PASS 1: Load metadata and identify duplicates
            (
                survey_meta,
                survey_resolved_labels,
                all_gaia_ids_sorted,
                n_total,
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
                n_wave = len(meta["wavelength"])
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
                unique_indices = meta["unique_indices"]  # ascending rows
                if len(unique_indices) > 0:
                    # Get merged indices for unique stars
                    unique_gaia_ids = meta["gaia_ids"][unique_indices].astype(np.int64)
                    found, merged_indices = _match_ids(
                        all_gaia_ids_sorted, unique_gaia_ids
                    )
                    if not found.all():
                        raise KeyError(int(unique_gaia_ids[np.argmin(found)]))

                    # Read spectra straight into the merged arrays
                    self._scatter_spectra(
                        f["surveys"][survey],
                        unique_indices,
                        merged_indices,
                        storage_format,
                        flux,
                        ivar,
                    )

                    # Bulk copy
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
        normalize_spectra: bool = False,
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
            This method needs one row per star, as catalogues built by
            ``scripts/build_catalogue_v2.py`` have; duplicate rows raise an error.

        Args:
            surveys: List of survey names to load and merge.
            label_source: Primary label source to use for combined labels.
            max_flag_bits: Maximum flag bits allowed (0 = highest quality).
            label_sources: Optional list of label sources for multi-label training.
                If provided, labels_dict and has_labels_dict will be populated
                with labels from each source. The primary `labels` array uses
                the first source in the list (or label_source if not provided).
            normalize_spectra: Apply per-spectrum median/IQR normalization to
                flux (and rescale ivar) before building the model inputs.
                Training and inference must use the same setting.

        Returns:
            SparseMergedData with memory-efficient sparse survey storage.
            Use with trainer's sparse batch construction for training.

        Raises:
            ValueError: If duplicate observations are found in the catalogue,
                the primary label group's ``gaia_id`` differs from
                ``metadata/gaia_id``, or another label group in
                ``label_sources`` has no ``gaia_id``.

        Examples:
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
        with h5py.File(self.path, "r") as f:
            parameter_names = _read_parameter_names(f)
            n_params = len(parameter_names)
            # =========================================================
            # PASS 1: Fast scan - read only SNR arrays and gaia_ids
            # =========================================================
            # Load master gaia_ids (from metadata or first label source)
            resolved_label = self._resolve_label_source(surveys[:1], label_source, f)
            label_grp = f["labels"][resolved_label]
            if "metadata" in f and "gaia_id" in f["metadata"]:
                all_gaia_ids = f["metadata"]["gaia_id"][:]
                # The primary labels are read in catalogue row order
                if "gaia_id" in label_grp and not np.array_equal(
                    label_grp["gaia_id"][:], all_gaia_ids
                ):
                    raise ValueError(
                        f"{label_grp.name}/gaia_id does not match "
                        "metadata/gaia_id: label rows must be the catalogue "
                        "rows, as scripts/build_catalogue_v2.py writes them"
                    )
            else:
                all_gaia_ids = label_grp["gaia_id"][:]

            n_total = len(all_gaia_ids)

            # Find which rows have data in each survey (fast - only reads SNR)
            survey_valid_indices: dict[str, np.ndarray] = {}
            survey_snr: dict[str, np.ndarray] = {}
            for survey in surveys:
                print(f"Loading {survey}...")
                snr = f["surveys"][survey]["snr"][:]
                valid_mask = snr > 0
                valid_indices = np.where(valid_mask)[0]
                survey_valid_indices[survey] = valid_indices
                survey_snr[survey] = snr[valid_indices]
                print(f"  {len(valid_indices):,} stars with data")

            # Always check for duplicates - require pre-deduplicated catalogue
            dup_meta: dict[str, dict] = {}
            for survey in surveys:
                ids = all_gaia_ids[survey_valid_indices[survey]]
                ids = ids[ids > 0]
                unique_ids, counts = np.unique(ids, return_counts=True)
                dup_ids = unique_ids[counts > 1]
                dup_meta[survey] = {
                    "duplicate_gaia_ids": set(dup_ids.tolist()),
                    "duplicate_indices": np.where(np.isin(ids, dup_ids))[0],
                }
            self._check_for_duplicates(dup_meta, surveys)

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

                wavelengths[survey] = _survey_wavelength(survey_grp)

                # Read only the rows with data (no full-dataset temporaries)
                if "flux" in survey_grp:
                    flux = _read_rows(survey_grp["flux"], valid_indices)
                    ivar = _read_rows(survey_grp["ivar"], valid_indices)
                else:
                    # LAMOST MRS: (N, 4, wavelengths) format - dual arm
                    # Spectra array is padded to max wavelength, but arms have different lengths
                    flux, ivar = _read_dual_arm_rows(survey_grp, valid_indices)

                snr = survey_snr[survey]

                # Build index mappings (valid_indices ARE the global indices)
                g2l = np.full(n_total, -1, dtype=np.int32)
                g2l[valid_indices] = np.arange(n_with_data, dtype=np.int32)
                global_to_local[survey] = g2l
                local_to_global[survey] = valid_indices.astype(np.int32)

                flux, ivar = _preprocess_spectra(flux, ivar, normalize_spectra)

                # Store sparse arrays
                sparse_flux[survey] = flux
                sparse_ivar[survey] = ivar
                sparse_snr[survey] = snr

                print(
                    f"  {survey}: {n_with_data:,} stars ({100*n_with_data/n_total:.1f}%)"
                )

        labels_3ch = _label_array(
            primary_labels, primary_errors, primary_flags, max_flag_bits
        )

        # Multi-label support: load additional label sources if requested
        labels_dict_out: dict[str, np.ndarray] | None = None
        has_labels_dict_out: dict[str, np.ndarray] | None = None
        label_sources_out: list[str] | None = None

        if label_sources is not None and len(label_sources) > 1:
            labels_dict_out = {}
            label_sources_out = label_sources

            with h5py.File(self.path, "r") as f:
                if "labels" not in f:
                    raise ValueError("Catalogue has no 'labels' group")

                for source in label_sources:
                    # Check if this is the primary source (already loaded)
                    if source == label_source:
                        # Use the already-loaded primary labels
                        labels_dict_out[source] = labels_3ch.copy()
                        continue

                    label_grp = f["labels"][
                        self._resolve_label_source(surveys, source, f)
                    ]
                    labels_dict_out[source] = _align_labels(
                        label_grp, all_gaia_ids, max_flag_bits
                    )

            has_labels_dict_out = {
                source: (labels[:, 2, :] > 0).any(axis=1)
                for source, labels in labels_dict_out.items()
            }

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
            parameter_names=parameter_names,
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
        """Get the wavelength grid for a survey (LAMOST MRS: blue then red arm)."""
        with h5py.File(self.path, "r") as f:
            if survey not in f["surveys"]:
                raise ValueError(f"Survey '{survey}' not in catalogue")
            return _survey_wavelength(f["surveys"][survey])

    def load_spectrum(
        self,
        survey: str,
        index: int,
        normalize_spectra: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Load one star's spectrum as a 3-channel model input.

        Args:
            survey: Survey name (e.g. "boss", "lamost_mrs").
            index: Row index in the catalogue (v2 rows are aligned with
                ``metadata/gaia_id``).
            normalize_spectra: Apply per-spectrum median/IQR normalization,
                as during training (see ``data.normalize_spectra``).

        Returns:
            Tuple of (X, wavelength, gaia_id) where X has shape (3, n_wave)
            with channels [flux, sigma, mask], matching the training inputs.
        """
        with h5py.File(self.path, "r") as f:
            if survey not in f["surveys"]:
                raise ValueError(f"Survey '{survey}' not in catalogue")
            grp = f["surveys"][survey]
            if "flux" in grp:
                flux = grp["flux"][index : index + 1]
                ivar = grp["ivar"][index : index + 1]
            else:
                flux, ivar = _split_dual_arm_spectra(
                    grp["spectra"][index : index + 1], grp
                )
            wavelength = _survey_wavelength(grp)
            has_ids = "metadata" in f and "gaia_id" in f["metadata"]
            gaia_id = int(f["metadata"]["gaia_id"][index]) if has_ids else -1

        flux, ivar = _preprocess_spectra(flux, ivar, normalize_spectra)
        X = _spectra_to_model_inputs(flux, ivar)[0]
        return X, wavelength.astype(np.float32), gaia_id

    def load_labels_for_index(
        self,
        index: int,
        label_source: str = "apogee",
        survey: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Load one star's labels by catalogue row index.

        Args:
            index: Row index in the catalogue.
            label_source: Label source (e.g. "apogee", "galah").
            survey: Survey name, only needed to resolve v1 per-survey label
                groups (e.g. "apogee_boss").

        Returns:
            Tuple of (values, errors, flags), each of shape (n_params,), or
            None if the label source is not in the catalogue.
        """
        with h5py.File(self.path, "r") as f:
            if "labels" not in f:
                return None
            try:
                resolved = self._resolve_label_source([survey or ""], label_source, f)
            except ValueError:
                return None
            grp = f["labels"][resolved]
            return grp["values"][index], grp["errors"][index], grp["flags"][index]

    def load_labels(self, label_source: str, max_flag_bits: int = 0) -> np.ndarray:
        """Load one label source's labels for every catalogue star.

        The labels are matched to stars by Gaia ID, in the row order of
        ``metadata/gaia_id``, so the label group may store its rows in any
        order. A label is valid (mask 1) when its error is positive and its
        flag is 0; with ``max_flag_bits`` >= 0, every label of a star with
        more set flag bits than ``max_flag_bits`` is masked.

        Args:
            label_source: Label group name, such as "apogee" or "galah".
            max_flag_bits: Largest number of set flag bits, summed over the
                parameters of a star, for its labels to count; -1 sets no
                limit.

        Returns:
            Labels of shape (n_stars, 3, n_parameters) = [values, errors,
            mask], in the catalogue's units. Stars without a row in the label
            group have mask 0.

        Raises:
            ValueError: If max_flag_bits is not an integer >= -1, the catalogue
                has no labels, the label source is not in the catalogue, or
                the catalogue or its label group has no Gaia IDs.
        """
        if not isinstance(max_flag_bits, (int, np.integer)) or max_flag_bits < -1:
            raise ValueError(
                f"max_flag_bits must be an integer >= -1, got {max_flag_bits!r}"
            )
        with h5py.File(self.path, "r") as f:
            source = self._resolve_label_source([], label_source, f)
            if "gaia_id" not in f.get("metadata", {}):
                raise ValueError(
                    f"Catalogue {self.path} has no metadata/gaia_id, which "
                    "load_labels needs to match labels to stars"
                )
            return _align_labels(
                f["labels"][source], f["metadata"]["gaia_id"][:], max_flag_bits
            )

    def get_parameter_names(self) -> list[str]:
        """Get the label parameter names, in catalogue column order."""
        with h5py.File(self.path, "r") as f:
            return _read_parameter_names(f)

    def load_merged_for_training(
        self,
        surveys: list[str],
        label_source: str = "apogee",
        max_flag_bits: int = 0,
        label_sources: list[str] | None = None,
        normalize_spectra: bool = False,
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
            This method needs one row per star, as catalogues built by
            ``scripts/build_catalogue_v2.py`` have; duplicate rows raise an error.

        Args:
            surveys: List of survey names to load and merge.
            label_source: Primary label source to use for combined labels.
                Used when label_sources is None (single-source mode).
            max_flag_bits: Maximum flag bits allowed (0 = highest quality).
            label_sources: List of label sources for multi-labelset training.
                If None, uses single-source mode with label_source parameter.
                If provided, returns y_dict and has_labels_dict instead of y.
            normalize_spectra: Apply per-spectrum median/IQR normalization to
                flux (and rescale ivar) before building the model inputs.
                Training and inference must use the same setting.

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
                    of the stars with at least one valid label (mask 1)
                    from that source, after the flag filter. Only present in
                    multi-source mode; None in single-source mode.

        Raises:
            ValueError: If duplicate observations are found in the catalogue,
                it has no Gaia IDs, or a label group in ``label_sources``
                other than ``label_source`` has no ``gaia_id``.

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
            flux, ivar = _preprocess_spectra(
                survey_data["flux"], survey_data["ivar"], normalize_spectra
            )

            # 3-channel format: [flux, sigma = 1/sqrt(ivar), mask = ivar > 0]
            X_dict[survey] = _spectra_to_model_inputs(flux, ivar)
            has_data_dict[survey] = survey_data["has_data"]

        # Single-source mode: return single y array
        if label_sources is None:
            y = _label_array(
                merged.primary_labels,
                merged.primary_errors,
                merged.primary_flags,
                max_flag_bits,
            )
            return X_dict, y, has_data_dict, None

        # Multi-source mode: load labels from each source and return dicts
        y_dict: dict[str, np.ndarray] = {}
        has_labels_dict: dict[str, np.ndarray] = {}

        with h5py.File(self.path, "r") as f:
            if "labels" not in f:
                raise ValueError("Catalogue has no 'labels' group")

            for source in label_sources:
                label_grp = f["labels"][self._resolve_label_source(surveys, source, f)]
                if source == label_source and "gaia_id" not in label_grp:
                    # load_merged matched the primary labels to stars by
                    # catalogue row
                    y_dict[source] = _label_array(
                        merged.primary_labels,
                        merged.primary_errors,
                        merged.primary_flags,
                        max_flag_bits,
                    )
                else:
                    y_dict[source] = _align_labels(
                        label_grp, merged.gaia_ids, max_flag_bits
                    )
                # A star has labels if one survives the flag filter
                has_labels_dict[source] = (y_dict[source][:, 2, :] > 0).any(axis=1)

        return X_dict, y_dict, has_data_dict, has_labels_dict

    def get_survey_wavelength_counts(self, surveys: list[str]) -> dict[str, int]:
        """
        Get the number of wavelength bins for each survey.

        This is useful for initializing MultiHeadMLP with correct input sizes.

        Args:
            surveys: List of survey names.

        Returns:
            Dictionary mapping survey name to number of wavelength bins.

        Examples:
            >>> loader = CatalogueLoader("super_catalogue.h5")
            >>> counts = loader.get_survey_wavelength_counts(["boss", "lamost_lrs"])
            >>> # counts = {"boss": 4506, "lamost_lrs": 3473}
        """
        result = {}
        with h5py.File(self.path, "r") as f:
            for survey in surveys:
                if survey not in f["surveys"]:
                    raise ValueError(f"Survey '{survey}' not in catalogue")
                result[survey] = len(_survey_wavelength(f["surveys"][survey]))
        return result
