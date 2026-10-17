#!/usr/bin/env python
"""
Build DOROTHY super-catalogue v2 from raw FITS files.

This script creates a unified HDF5 catalogue from pre-cross-matched FITS files
containing survey spectra matched with APOGEE and GALAH stellar parameters.

Input files (in data/raw_catalogues/):
    - BOSSxAPOGEE_training.fits, BOSSxGALAH_training.fits
    - DESIxAPOGEE_training[_dr1].fits, DESIxGALAH_training[_dr1].fits
    - LAMOST_LRSxAPOGEE_training.fits, LAMOST_LRSxGALAH_training.fits
    - LAMOST_MRSxAPOGEE.fits, LAMOST_MRSxGALAH.fits

Output:
    - data/super_catalogue.h5 (unified multi-survey, multi-label catalogue)

Usage:
    python scripts/build_catalogue_v2.py
    python scripts/build_catalogue_v2.py --output data/super_catalogue.h5
    python scripts/build_catalogue_v2.py --dry-run  # Print stats only

    # 20 parameters (17 elements) with [X/H] abundances
    python scripts/build_catalogue_v2.py --elements extended --abundance-scale x_h \
        --output data/super_catalogue_xh_extended.h5
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from astropy.io import fits


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}hr"


# Abundance element sets (in output column order, after teff, logg, fe_h)
CORE_ELEMENTS = ["mg", "c", "si", "ni", "al", "ca", "n", "mn"]
EXTENDED_ELEMENTS = CORE_ELEMENTS + ["o", "na", "k", "ti", "v", "cr", "co", "ba", "sr"]
ELEMENT_SETS = {"core": CORE_ELEMENTS, "extended": EXTENDED_ELEMENTS}

# Column index of each element in the APOGEE DR17 allStar X_H / X_H_ERR arrays
APOGEE_X_H_INDEX = {
    "c": 0,
    "n": 2,
    "o": 3,
    "na": 4,
    "mg": 5,
    "al": 6,
    "si": 7,
    "k": 10,
    "ca": 11,
    "ti": 12,
    "v": 14,
    "cr": 15,
    "mn": 16,
    "co": 18,
    "ni": 19,
}


def get_parameter_names(
    elements: list[str] = CORE_ELEMENTS, abundance_scale: str = "x_fe"
) -> list[str]:
    """Output parameter names: teff, logg, fe_h, then one abundance per element.

    Args:
        elements: Elements to include, in output order.
        abundance_scale: "x_fe" for [X/Fe] or "x_h" for [X/H] abundances.
    """
    if abundance_scale not in ("x_fe", "x_h"):
        raise ValueError(
            f"abundance_scale must be 'x_fe' or 'x_h', got {abundance_scale}"
        )
    suffix = "fe" if abundance_scale == "x_fe" else "h"
    return ["teff", "logg", "fe_h"] + [f"{el}_{suffix}" for el in elements]


# Default (v2) parameter set: 11 parameters with [X/Fe] abundances
PARAMETER_NAMES = get_parameter_names()

# Input file candidates: (survey, label_source) -> filenames, first found is used
INPUT_FILE_CANDIDATES = {
    ("boss", "apogee"): ["BOSSxAPOGEE_training.fits"],
    ("boss", "galah"): ["BOSSxGALAH_training.fits"],
    ("desi", "apogee"): ["DESIxAPOGEE_training_dr1.fits", "DESIxAPOGEE_training.fits"],
    ("desi", "galah"): ["DESIxGALAH_training_dr1.fits", "DESIxGALAH_training.fits"],
    ("lamost_lrs", "apogee"): ["LAMOST_LRSxAPOGEE_training.fits"],
    ("lamost_lrs", "galah"): ["LAMOST_LRSxGALAH_training.fits"],
    ("lamost_mrs", "apogee"): ["LAMOST_MRSxAPOGEE.fits"],
    ("lamost_mrs", "galah"): ["LAMOST_MRSxGALAH.fits"],
}


def resolve_input_files(input_dir: Path) -> dict[tuple[str, str], str]:
    """Pick the first existing candidate filename for each (survey, label) pair.

    Pairs with no file present map to their first candidate (reported as
    missing and skipped later).
    """
    resolved = {}
    for key, candidates in INPUT_FILE_CANDIDATES.items():
        existing = [c for c in candidates if (input_dir / c).exists()]
        resolved[key] = existing[0] if existing else candidates[0]
    return resolved


# APOGEE ASPCAPFLAG bit positions for parameter-specific flags
# Reference: https://www.sdss4.org/dr17/irspec/apogee-bitmasks/
# We only flag teff/logg if their specific warning/error bits are set,
# NOT for unrelated warnings like VMICRO_WARN or N_M_WARN
ASPCAPFLAG_TEFF_BITS = (1 << 0) | (1 << 16)  # TEFF_WARN | TEFF_BAD
ASPCAPFLAG_LOGG_BITS = (1 << 1) | (1 << 17)  # LOGG_WARN | LOGG_BAD


def get_gaia_id_column(data: np.ndarray) -> str:
    """Determine the Gaia ID column name from the data."""
    names = data.dtype.names
    if "GAIAEDR3_SOURCE_ID" in names:
        return "GAIAEDR3_SOURCE_ID"
    elif "gaiadr3_source_id" in names:
        return "gaiadr3_source_id"
    elif "GAIA_ID" in names:
        return "GAIA_ID"
    else:
        raise ValueError(f"No Gaia ID column found. Available: {names}")


def collect_all_gaia_ids(
    input_dir: Path,
    input_files: dict[tuple[str, str], str],
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Scan all FITS files to collect union of all unique Gaia IDs.

    Returns:
        - Sorted array of unique Gaia IDs
        - Dict mapping (survey, label_source) -> array of Gaia IDs for that file
    """
    all_gaia_ids = set()
    file_gaia_ids = {}

    for (survey, label_source), filename in input_files.items():
        path = input_dir / filename
        if not path.exists():
            if verbose:
                print(f"    Warning: {filename} not found, skipping")
            continue

        t0 = time.time()
        with fits.open(path) as hdul:
            gaia_data = hdul["GAIA_ID"].data
            col_name = get_gaia_id_column(gaia_data)
            gaia_ids = gaia_data[col_name].astype(np.int64)

            # Filter out invalid IDs (0 or negative)
            valid_mask = gaia_ids > 0
            valid_ids = gaia_ids[valid_mask]

            all_gaia_ids.update(valid_ids)
            file_gaia_ids[(survey, label_source)] = gaia_ids

            if verbose:
                print(
                    f"    {filename}: {len(valid_ids):,} IDs ({format_time(time.time() - t0)})"
                )

    print("    Sorting and deduplicating...")
    sorted_ids = np.array(sorted(all_gaia_ids), dtype=np.int64)
    return sorted_ids, file_gaia_ids


def _finalize_labels(
    values: np.ndarray, errors: np.ndarray, flags: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero non-finite values/errors and flag them."""
    bad = ~np.isfinite(values) | ~np.isfinite(errors)
    values[bad] = 0.0
    errors[bad] = 0.0
    flags[bad] = 1
    return values, errors, flags


def _column(label_data: np.ndarray, name: str) -> np.ndarray | None:
    """Return a column as float64, or None if absent."""
    if name in label_data.dtype.names:
        return np.asarray(label_data[name], dtype=np.float64)
    return None


def extract_apogee_labels(
    label_data: np.ndarray,
    elements: list[str] = CORE_ELEMENTS,
    abundance_scale: str = "x_fe",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract APOGEE labels in standard order (teff, logg, fe_h, abundances).

    For ``abundance_scale="x_h"``, uses the allStar ``X_H`` / ``X_H_ERR``
    arrays when present, otherwise [X/Fe] + [Fe/H] with errors added in
    quadrature. Elements APOGEE does not measure are left with value and
    error 0 (masked, not flagged).
    """
    names = label_data.dtype.names
    n_stars = len(label_data)
    n_params = 3 + len(elements)
    values = np.zeros((n_stars, n_params), dtype=np.float64)
    errors = np.zeros((n_stars, n_params), dtype=np.float64)
    flags = np.zeros((n_stars, n_params), dtype=np.uint8)

    for i, col in enumerate(["TEFF", "LOGG", "FE_H"]):
        val, err = _column(label_data, col), _column(label_data, f"{col}_ERR")
        if val is not None:
            values[:, i] = val
        if err is not None:
            errors[:, i] = err

    # teff/logg: only their parameter-specific ASPCAPFLAG bits (not e.g. VMICRO_WARN)
    if "ASPCAPFLAG" in names:
        aspcapflag = label_data["ASPCAPFLAG"]
        flags[:, 0] = ((aspcapflag & ASPCAPFLAG_TEFF_BITS) != 0).astype(np.uint8)
        flags[:, 1] = ((aspcapflag & ASPCAPFLAG_LOGG_BITS) != 0).astype(np.uint8)
    if "FE_H_FLAG" in names:
        flags[:, 2] = (label_data["FE_H_FLAG"] != 0).astype(np.uint8)

    has_x_h = "X_H" in names
    has_fe_h = "FE_H" in names
    for k, el in enumerate(elements):
        i = 3 + k
        col = f"{el.upper()}_FE"
        x_fe, x_fe_err = _column(label_data, col), _column(label_data, f"{col}_ERR")
        flag_col = f"{col}_FLAG"
        if flag_col in names:
            flags[:, i] = (label_data[flag_col] != 0).astype(np.uint8)

        if abundance_scale == "x_h" and has_x_h and el in APOGEE_X_H_INDEX:
            idx = APOGEE_X_H_INDEX[el]
            values[:, i] = label_data["X_H"][:, idx]
            if "X_H_ERR" in names:
                errors[:, i] = label_data["X_H_ERR"][:, idx]
            elif x_fe_err is not None:
                errors[:, i] = np.hypot(x_fe_err, errors[:, 2])
        elif x_fe is not None:
            if abundance_scale == "x_h":
                if has_fe_h:  # Without [Fe/H], [X/H] is unknown (left masked)
                    values[:, i] = x_fe + values[:, 2]
                    errors[:, i] = np.hypot(
                        x_fe_err if x_fe_err is not None else 0.0, errors[:, 2]
                    )
                    flags[:, i] |= flags[:, 2]
            else:
                values[:, i] = x_fe
                if x_fe_err is not None:
                    errors[:, i] = x_fe_err
        # Elements APOGEE doesn't measure keep error 0: masked by the loader
        # (errors > 0) without being flagged, so they don't count against
        # max_flag_bits for every star.

    values, errors, flags = _finalize_labels(values, errors, flags)
    return values.astype(np.float32), errors.astype(np.float32), flags


def extract_galah_labels(
    label_data: np.ndarray,
    elements: list[str] = CORE_ELEMENTS,
    abundance_scale: str = "x_fe",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract GALAH labels in standard order (teff, logg, fe_h, abundances).

    For ``abundance_scale="x_h"``, abundances are [X/Fe] + [Fe/H] with errors
    added in quadrature, and are flagged whenever [Fe/H] is flagged. Elements
    GALAH does not measure are left with value and error 0 (masked, not
    flagged).
    """
    names = label_data.dtype.names
    n_stars = len(label_data)
    n_params = 3 + len(elements)
    values = np.zeros((n_stars, n_params), dtype=np.float64)
    errors = np.zeros((n_stars, n_params), dtype=np.float64)
    flags = np.zeros((n_stars, n_params), dtype=np.uint8)

    for i, col in enumerate(["teff", "logg", "fe_h"]):
        val, err = _column(label_data, col), _column(label_data, f"e_{col}")
        if val is not None:
            values[:, i] = val
        if err is not None:
            errors[:, i] = err

    # teff/logg use the stellar-parameter flag; fe_h its own flag
    if "flag_sp" in names:
        flag_sp = (label_data["flag_sp"] != 0).astype(np.uint8)
        flags[:, 0] = flag_sp
        flags[:, 1] = flag_sp
    if "flag_fe_h" in names:
        flags[:, 2] = (label_data["flag_fe_h"] != 0).astype(np.uint8)

    has_fe_h = "fe_h" in names
    for k, el in enumerate(elements):
        i = 3 + k
        col = f"{el}_fe"
        x_fe, x_fe_err = _column(label_data, col), _column(label_data, f"e_{col}")
        if f"flag_{col}" in names:
            flags[:, i] = (label_data[f"flag_{col}"] != 0).astype(np.uint8)
        if x_fe is None or (abundance_scale == "x_h" and not has_fe_h):
            # Not measured (or [X/H] unknown without [Fe/H]): error 0, so the
            # loader masks it without flagging
            flags[:, i] = 0
            continue
        if abundance_scale == "x_h":
            values[:, i] = x_fe + values[:, 2]
            errors[:, i] = np.hypot(
                x_fe_err if x_fe_err is not None else 0.0, errors[:, 2]
            )
            flags[:, i] |= flags[:, 2]
        else:
            values[:, i] = x_fe
            if x_fe_err is not None:
                errors[:, i] = x_fe_err

    values, errors, flags = _finalize_labels(values, errors, flags)
    return values.astype(np.float32), errors.astype(np.float32), flags


def build_catalogue(
    input_dir: Path,
    output_path: Path,
    verbose: bool = True,
    dry_run: bool = False,
    elements: list[str] = CORE_ELEMENTS,
    abundance_scale: str = "x_fe",
) -> dict[str, Any]:
    """
    Build unified super-catalogue from raw FITS files.

    Args:
        input_dir: Directory with the cross-matched FITS files.
        output_path: Output HDF5 path.
        verbose: Print per-file progress.
        dry_run: Print what would be processed without writing output.
        elements: Abundance elements to include (after teff, logg, fe_h).
        abundance_scale: "x_fe" ([X/Fe]) or "x_h" ([X/H]) abundances.

    Uses vectorized operations for efficiency:
    1. Build mapping from gaia_id to output index
    2. Load each file and use fancy indexing to place data
    3. Write to HDF5 in bulk operations
    """
    build_start = time.time()
    parameter_names = get_parameter_names(elements, abundance_scale)
    n_params = len(parameter_names)
    input_files = resolve_input_files(input_dir)
    print("=" * 70)
    print("DOROTHY Super-Catalogue Builder v2 (Vectorized)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Pass 1: Collect all unique Gaia IDs
    print("\n[Pass 1/3] Collecting Gaia IDs from all files...")
    t0 = time.time()
    all_gaia_ids, file_gaia_ids = collect_all_gaia_ids(input_dir, input_files, verbose)
    n_total = len(all_gaia_ids)

    # Build fast lookup: gaia_id -> output row index
    # Using searchsorted for O(log n) lookup instead of dict
    print("    Building index mapping...")

    print(f"\n  Total unique stars: {n_total:,}")
    print(f"  Pass 1 completed in {format_time(time.time() - t0)}")

    stats = {
        "n_total": n_total,
        "surveys": {},
        "labels": {},
    }

    if dry_run:
        print("\n[DRY RUN] Would process the following:")
        for survey in ["boss", "desi", "lamost_lrs", "lamost_mrs"]:
            apogee_key = (survey, "apogee")
            galah_key = (survey, "galah")
            n_apogee = len(file_gaia_ids.get(apogee_key, []))
            n_galah = len(file_gaia_ids.get(galah_key, []))
            print(f"  {survey}: APOGEE={n_apogee:,}, GALAH={n_galah:,}")
        return stats

    # Create output file
    print(f"\nCreating output file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Create metadata group
        meta = f.create_group("metadata")
        meta.create_dataset("gaia_id", data=all_gaia_ids)

        # Initialize RA/Dec
        ra = np.zeros(n_total, dtype=np.float64)
        dec = np.zeros(n_total, dtype=np.float64)

        # Create survey groups
        surveys_grp = f.create_group("surveys")

        # Pass 2: Process each survey
        print("\n[Pass 2/3] Processing surveys...")
        t0_pass2 = time.time()

        for survey_idx, survey in enumerate(
            ["boss", "desi", "lamost_lrs", "lamost_mrs"], 1
        ):
            print(f"\n  [{survey_idx}/4] Processing {survey}...")
            sys.stdout.flush()

            # Collect all files for this survey
            files_for_survey = []
            for label_source in ["apogee", "galah"]:
                key = (survey, label_source)
                if key in file_gaia_ids:
                    files_for_survey.append(
                        (label_source, input_files[key], file_gaia_ids[key])
                    )

            if not files_for_survey:
                print("    No files found, skipping")
                continue

            # Get wavelength info from first file
            first_path = input_dir / files_for_survey[0][1]
            with fits.open(first_path) as hdul:
                if survey == "lamost_mrs":
                    wavelength_b = (
                        hdul["WAVELENGTH_B"].data["WAVELENGTH_B"].astype(np.float32)
                    )
                    wavelength_r = (
                        hdul["WAVELENGTH_R"].data["WAVELENGTH_R"].astype(np.float32)
                    )
                    n_wave_b = len(wavelength_b)
                    n_wave_r = len(wavelength_r)
                    n_wave_max = max(n_wave_b, n_wave_r)
                else:
                    wavelength = (
                        hdul["WAVELENGTH"].data["WAVELENGTH"].astype(np.float32)
                    )
                    n_wave = len(wavelength)

            # Initialize output arrays in memory
            if survey == "lamost_mrs":
                out_spectra = np.zeros((n_total, 4, n_wave_max), dtype=np.float32)
            else:
                out_flux = np.zeros((n_total, n_wave), dtype=np.float32)
                out_ivar = np.zeros((n_total, n_wave), dtype=np.float32)
            out_snr = np.zeros(n_total, dtype=np.float32)

            # Process each file for this survey
            for _label_source, filename, gaia_ids_file in files_for_survey:
                path = input_dir / filename
                t0_file = time.time()
                print(f"    Loading {filename}...", end=" ", flush=True)

                with fits.open(path) as hdul:
                    # Get valid indices - stars with valid gaia IDs that exist in our master list
                    valid_mask = gaia_ids_file > 0
                    valid_gaia = gaia_ids_file[valid_mask]

                    # Find where these stars go in output (vectorized lookup)
                    out_indices = np.searchsorted(all_gaia_ids, valid_gaia)

                    # Verify the IDs actually match (handles IDs not in master list)
                    matches = (out_indices < n_total) & (
                        all_gaia_ids[np.minimum(out_indices, n_total - 1)] == valid_gaia
                    )

                    # Get source indices (rows in FITS file) for valid matches
                    source_indices = np.where(valid_mask)[0][matches]
                    target_indices = out_indices[matches]

                    # Only fill empty slots (SNR == 0)
                    empty_mask = out_snr[target_indices] == 0
                    source_idx = source_indices[empty_mask]
                    target_idx = target_indices[empty_mask]

                    n_to_add = len(source_idx)

                    if n_to_add > 0:
                        if survey == "lamost_mrs":
                            flux_b = hdul["SPECTRA_B"].data["FLUX_B"].astype(np.float32)
                            flux_r = hdul["SPECTRA_R"].data["FLUX_R"].astype(np.float32)
                            ivar_b = (
                                hdul["SPEC_IVAR_B"].data["IVAR_B"].astype(np.float32)
                            )
                            ivar_r = (
                                hdul["SPEC_IVAR_R"].data["IVAR_R"].astype(np.float32)
                            )

                            # Vectorized assignment
                            out_spectra[target_idx, 0, :n_wave_b] = flux_b[source_idx]
                            out_spectra[target_idx, 1, :n_wave_b] = ivar_b[source_idx]
                            out_spectra[target_idx, 2, :n_wave_r] = flux_r[source_idx]
                            out_spectra[target_idx, 3, :n_wave_r] = ivar_r[source_idx]

                            # Compute SNR from blue arm (vectorized)
                            snr_vals = np.zeros(n_to_add, dtype=np.float32)
                            for i, (si, _ti) in enumerate(
                                zip(source_idx, target_idx, strict=False)
                            ):
                                valid = ivar_b[si] > 0
                                if valid.sum() > 0:
                                    snr_vals[i] = np.median(
                                        np.abs(flux_b[si, valid])
                                        * np.sqrt(ivar_b[si, valid])
                                    )
                            out_snr[target_idx] = snr_vals
                        else:
                            flux_data = hdul["SPECTRA"].data["FLUX"].astype(np.float32)
                            ivar_data = (
                                hdul["SPEC_IVAR"].data["IVAR"].astype(np.float32)
                            )

                            # Vectorized assignment (fast!)
                            out_flux[target_idx] = flux_data[source_idx]
                            out_ivar[target_idx] = ivar_data[source_idx]

                            # Compute SNR (vectorized where possible)
                            snr_vals = np.zeros(n_to_add, dtype=np.float32)
                            for i, si in enumerate(source_idx):
                                valid = ivar_data[si] > 0
                                if valid.sum() > 0:
                                    snr_vals[i] = np.median(
                                        np.abs(flux_data[si, valid])
                                        * np.sqrt(ivar_data[si, valid])
                                    )
                            out_snr[target_idx] = snr_vals

                        # Load RA/Dec if available
                        if "RVTAB" in [h.name for h in hdul]:
                            rvtab = hdul["RVTAB"].data
                            names = rvtab.dtype.names
                            ra_col = next(
                                (n for n in names if n.upper() in ("RA", "TARGET_RA")),
                                None,
                            )
                            dec_col = next(
                                (
                                    n
                                    for n in names
                                    if n.upper() in ("DEC", "TARGET_DEC")
                                ),
                                None,
                            )
                            if ra_col and dec_col:
                                # Only fill where empty
                                empty_coords = (ra[target_idx] == 0) & (
                                    dec[target_idx] == 0
                                )
                                coord_target = target_idx[empty_coords]
                                coord_source = source_idx[empty_coords]
                                ra[coord_target] = rvtab[ra_col][coord_source]
                                dec[coord_target] = rvtab[dec_col][coord_source]

                    print(f"{n_to_add:,} stars ({format_time(time.time() - t0_file)})")

            # Write to HDF5 in bulk
            print("    Writing to HDF5...", end=" ", flush=True)
            t0_write = time.time()

            survey_grp = surveys_grp.create_group(survey)
            if survey == "lamost_mrs":
                survey_grp.create_dataset("spectra", data=out_spectra, compression=None)
                survey_grp.create_dataset("wavelength_b", data=wavelength_b)
                survey_grp.create_dataset("wavelength_r", data=wavelength_r)
                survey_grp.attrs["n_channels"] = 4
                survey_grp.attrs["channel_names"] = [
                    "flux_b",
                    "ivar_b",
                    "flux_r",
                    "ivar_r",
                ]
            else:
                survey_grp.create_dataset("flux", data=out_flux, compression=None)
                survey_grp.create_dataset("ivar", data=out_ivar, compression=None)
                survey_grp.create_dataset("wavelength", data=wavelength)
            survey_grp.create_dataset("snr", data=out_snr)

            n_with_data = int((out_snr > 0).sum())
            print(f"done ({format_time(time.time() - t0_write)})")
            print(f"    Stars with data: {n_with_data:,}")
            stats["surveys"][survey] = {"n_with_data": n_with_data}

        # Store RA/Dec
        meta.create_dataset("ra", data=ra)
        meta.create_dataset("dec", data=dec)
        print(f"\n  Pass 2 completed in {format_time(time.time() - t0_pass2)}")

        # Pass 3: Process labels
        print("\n[Pass 3/3] Processing labels...")
        t0_pass3 = time.time()
        labels_grp = f.create_group("labels")

        for ls_idx, label_source in enumerate(["apogee", "galah"], 1):
            print(f"\n  [{ls_idx}/2] Processing {label_source} labels...")
            sys.stdout.flush()

            # Initialize output arrays
            values = np.zeros((n_total, n_params), dtype=np.float32)
            errors = np.zeros((n_total, n_params), dtype=np.float32)
            flags = np.zeros((n_total, n_params), dtype=np.uint8)
            has_labels = np.zeros(n_total, dtype=bool)

            # Process each survey file with this label source
            for survey in ["boss", "desi", "lamost_lrs", "lamost_mrs"]:
                key = (survey, label_source)
                if key not in file_gaia_ids:
                    continue

                filename = input_files[key]
                path = input_dir / filename
                gaia_ids_file = file_gaia_ids[key]

                t0_file = time.time()
                print(f"    Loading {filename}...", end=" ", flush=True)

                with fits.open(path) as hdul:
                    # Get label data
                    label_hdu_name = label_source.upper()
                    if label_hdu_name not in [h.name for h in hdul]:
                        label_hdu_name = label_source.capitalize()
                    label_data = hdul[label_hdu_name].data

                    # Extract labels
                    if label_source == "apogee":
                        file_vals, file_errs, file_flags = extract_apogee_labels(
                            label_data, elements, abundance_scale
                        )
                    else:
                        file_vals, file_errs, file_flags = extract_galah_labels(
                            label_data, elements, abundance_scale
                        )

                    # Find valid mappings
                    valid_mask = gaia_ids_file > 0
                    valid_gaia = gaia_ids_file[valid_mask]

                    out_indices = np.searchsorted(all_gaia_ids, valid_gaia)
                    matches = (out_indices < n_total) & (
                        all_gaia_ids[np.minimum(out_indices, n_total - 1)] == valid_gaia
                    )

                    source_indices = np.where(valid_mask)[0][matches]
                    target_indices = out_indices[matches]

                    # Only fill empty slots
                    empty_mask = ~has_labels[target_indices]
                    source_idx = source_indices[empty_mask]
                    target_idx = target_indices[empty_mask]

                    n_added = len(source_idx)
                    if n_added > 0:
                        values[target_idx] = file_vals[source_idx]
                        errors[target_idx] = file_errs[source_idx]
                        flags[target_idx] = file_flags[source_idx]
                        has_labels[target_idx] = True

                    print(f"{n_added:,} stars ({format_time(time.time() - t0_file)})")

            # Write labels to HDF5
            print("    Writing to HDF5...", end=" ", flush=True)
            t0_write = time.time()

            label_grp = labels_grp.create_group(label_source)
            label_grp.create_dataset("values", data=values, compression=None)
            label_grp.create_dataset("errors", data=errors, compression=None)
            label_grp.create_dataset("flags", data=flags)
            label_grp.create_dataset("gaia_id", data=all_gaia_ids)

            n_with_labels = int(has_labels.sum())
            print(f"done ({format_time(time.time() - t0_write)})")
            print(f"    Total stars with {label_source} labels: {n_with_labels:,}")
            stats["labels"][label_source] = {"n_with_labels": n_with_labels}

        print(f"\n  Pass 3 completed in {format_time(time.time() - t0_pass3)}")

        # Set file attributes
        f.attrs["n_stars"] = n_total
        f.attrs["parameter_names"] = parameter_names
        f.attrs["abundance_scale"] = abundance_scale
        f.attrs["survey_names"] = ["boss", "desi", "lamost_lrs", "lamost_mrs"]
        f.attrs["label_sources"] = ["apogee", "galah"]
        f.attrs["creation_date"] = datetime.now().isoformat()
        f.attrs["version"] = "2.0.0"

    total_time = time.time() - build_start
    print("\n" + "=" * 70)
    print(f"Catalogue written to {output_path}")
    print(f"Total stars: {n_total:,}")
    for survey, sdata in stats.get("surveys", {}).items():
        print(f"  {survey}: {sdata['n_with_data']:,} stars with spectra")
    for label_source, ldata in stats.get("labels", {}).items():
        print(f"  {label_source}: {ldata['n_with_labels']:,} stars with labels")
    print(f"Total time: {format_time(total_time)}")
    print("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build DOROTHY super-catalogue v2 from raw FITS files"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw_catalogues"),
        help="Directory containing raw FITS files (default: data/raw_catalogues)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 file path (default: data/super_catalogue.h5, with "
        "_xh / _extended suffixes for [X/H] or extended builds)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without creating output",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less per-file progress output",
    )
    parser.add_argument(
        "--elements",
        choices=sorted(ELEMENT_SETS),
        default="core",
        help="Abundance set: core (8 elements, 11 params) or extended "
        "(17 elements, 20 params). Default: core",
    )
    parser.add_argument(
        "--abundance-scale",
        choices=["x_fe", "x_h"],
        default="x_fe",
        help="Store abundances as [X/Fe] (x_fe) or [X/H] (x_h). Default: x_fe",
    )
    parser.add_argument(
        "--X_H",
        "--x-h",
        dest="abundance_scale",
        action="store_const",
        const="x_h",
        help="Shorthand for --abundance-scale x_h",
    )

    args = parser.parse_args()
    if args.output is None:
        suffix = ("_xh" if args.abundance_scale == "x_h" else "") + (
            "_extended" if args.elements == "extended" else ""
        )
        args.output = Path(f"data/super_catalogue{suffix}.h5")

    # Resolve paths relative to script location if needed
    if not args.input_dir.is_absolute():
        if (Path.cwd() / args.input_dir).exists():
            args.input_dir = Path.cwd() / args.input_dir
        else:
            script_dir = Path(__file__).parent.parent
            args.input_dir = script_dir / args.input_dir

    if not args.output.is_absolute():
        if Path.cwd().name == "scripts":
            args.output = Path.cwd().parent / args.output
        else:
            args.output = Path.cwd() / args.output

    build_catalogue(
        args.input_dir,
        args.output,
        verbose=not args.quiet,
        dry_run=args.dry_run,
        elements=ELEMENT_SETS[args.elements],
        abundance_scale=args.abundance_scale,
    )


if __name__ == "__main__":
    main()
