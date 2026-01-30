#!/usr/bin/env python
"""
Build DOROTHY super-catalogue v2 from raw FITS files.

This script creates a unified HDF5 catalogue from pre-cross-matched FITS files
containing survey spectra matched with APOGEE and GALAH stellar parameters.

Input files (in data/raw_catalogues/):
    - BOSSxAPOGEE_training.fits, BOSSxGALAH_training.fits
    - DESIxAPOGEE_training.fits, DESIxGALAH_training.fits
    - LAMOST_LRSxAPOGEE_training.fits, LAMOST_LRSxGALAH_training.fits
    - LAMOST_MRSxAPOGEE.fits, LAMOST_MRSxGALAH.fits

Output:
    - data/super_catalogue.h5 (unified multi-survey, multi-label catalogue)

Usage:
    python scripts/build_catalogue_v2.py
    python scripts/build_catalogue_v2.py --output data/super_catalogue.h5
    python scripts/build_catalogue_v2.py --dry-run  # Print stats only
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

# Input file mapping: (survey, label_source) -> filename
INPUT_FILES = {
    ("boss", "apogee"): "BOSSxAPOGEE_training.fits",
    ("boss", "galah"): "BOSSxGALAH_training.fits",
    ("desi", "apogee"): "DESIxAPOGEE_training.fits",
    ("desi", "galah"): "DESIxGALAH_training.fits",
    ("lamost_lrs", "apogee"): "LAMOST_LRSxAPOGEE_training.fits",
    ("lamost_lrs", "galah"): "LAMOST_LRSxGALAH_training.fits",
    ("lamost_mrs", "apogee"): "LAMOST_MRSxAPOGEE.fits",
    ("lamost_mrs", "galah"): "LAMOST_MRSxGALAH.fits",
}

# APOGEE column mappings (UPPERCASE)
APOGEE_COLUMNS = {
    "value": [
        "TEFF",
        "LOGG",
        "FE_H",
        "MG_FE",
        "C_FE",
        "SI_FE",
        "NI_FE",
        "AL_FE",
        "CA_FE",
        "N_FE",
        "MN_FE",
    ],
    "error": [
        "TEFF_ERR",
        "LOGG_ERR",
        "FE_H_ERR",
        "MG_FE_ERR",
        "C_FE_ERR",
        "SI_FE_ERR",
        "NI_FE_ERR",
        "AL_FE_ERR",
        "CA_FE_ERR",
        "N_FE_ERR",
        "MN_FE_ERR",
    ],
    "flag": [
        None,
        None,
        "FE_H_FLAG",
        "MG_FE_FLAG",
        "C_FE_FLAG",
        "SI_FE_FLAG",
        "NI_FE_FLAG",
        "AL_FE_FLAG",
        "CA_FE_FLAG",
        "N_FE_FLAG",
        "MN_FE_FLAG",
    ],
}

# GALAH column mappings (lowercase)
GALAH_COLUMNS = {
    "value": [
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
    ],
    "error": [
        "e_teff",
        "e_logg",
        "e_fe_h",
        "e_mg_fe",
        "e_c_fe",
        "e_si_fe",
        "e_ni_fe",
        "e_al_fe",
        "e_ca_fe",
        "e_n_fe",
        "e_mn_fe",
    ],
    "flag": [
        None,
        None,
        "flag_fe_h",
        "flag_mg_fe",
        "flag_c_fe",
        "flag_si_fe",
        "flag_ni_fe",
        "flag_al_fe",
        "flag_ca_fe",
        "flag_n_fe",
        "flag_mn_fe",
    ],
}


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
    input_dir: Path, verbose: bool = True
) -> tuple[np.ndarray, dict]:
    """
    Scan all FITS files to collect union of all unique Gaia IDs.

    Returns:
        - Sorted array of unique Gaia IDs
        - Dict mapping (survey, label_source) -> array of Gaia IDs for that file
    """
    all_gaia_ids = set()
    file_gaia_ids = {}

    for (survey, label_source), filename in INPUT_FILES.items():
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


def extract_apogee_labels(
    label_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and format APOGEE labels to standard order."""
    n_stars = len(label_data)
    values = np.zeros((n_stars, 11), dtype=np.float32)
    errors = np.zeros((n_stars, 11), dtype=np.float32)
    flags = np.zeros((n_stars, 11), dtype=np.uint8)

    for i, col in enumerate(APOGEE_COLUMNS["value"]):
        if col in label_data.dtype.names:
            values[:, i] = label_data[col]

    for i, col in enumerate(APOGEE_COLUMNS["error"]):
        if col in label_data.dtype.names:
            errors[:, i] = label_data[col]

    # Handle flags
    aspcapflag = (
        label_data["ASPCAPFLAG"] if "ASPCAPFLAG" in label_data.dtype.names else None
    )

    for i, col in enumerate(APOGEE_COLUMNS["flag"]):
        if col is None:
            if aspcapflag is not None:
                flags[:, i] = (aspcapflag != 0).astype(np.uint8)
        elif col in label_data.dtype.names:
            flags[:, i] = (label_data[col] != 0).astype(np.uint8)

    # Handle NaN/inf values
    bad_values = ~np.isfinite(values)
    bad_errors = ~np.isfinite(errors)
    values[bad_values] = 0.0
    errors[bad_errors] = 0.0
    flags[bad_values | bad_errors] = 1

    return values, errors, flags


def extract_galah_labels(
    label_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and format GALAH labels to standard order."""
    n_stars = len(label_data)
    values = np.zeros((n_stars, 11), dtype=np.float32)
    errors = np.zeros((n_stars, 11), dtype=np.float32)
    flags = np.zeros((n_stars, 11), dtype=np.uint8)

    for i, col in enumerate(GALAH_COLUMNS["value"]):
        if col in label_data.dtype.names:
            values[:, i] = label_data[col]

    for i, col in enumerate(GALAH_COLUMNS["error"]):
        if col in label_data.dtype.names:
            errors[:, i] = label_data[col]

    # Handle flags
    flag_sp = label_data["flag_sp"] if "flag_sp" in label_data.dtype.names else None

    for i, col in enumerate(GALAH_COLUMNS["flag"]):
        if col is None:
            if flag_sp is not None:
                flags[:, i] = (flag_sp != 0).astype(np.uint8)
        elif col in label_data.dtype.names:
            flags[:, i] = (label_data[col] != 0).astype(np.uint8)

    # Handle NaN/inf values
    bad_values = ~np.isfinite(values)
    bad_errors = ~np.isfinite(errors)
    values[bad_values] = 0.0
    errors[bad_errors] = 0.0
    flags[bad_values | bad_errors] = 1

    return values, errors, flags


def build_catalogue(
    input_dir: Path,
    output_path: Path,
    verbose: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Build unified super-catalogue from raw FITS files.

    Uses vectorized operations for efficiency:
    1. Build mapping from gaia_id to output index
    2. Load each file and use fancy indexing to place data
    3. Write to HDF5 in bulk operations
    """
    build_start = time.time()
    print("=" * 70)
    print("DOROTHY Super-Catalogue Builder v2 (Vectorized)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Pass 1: Collect all unique Gaia IDs
    print("\n[Pass 1/3] Collecting Gaia IDs from all files...")
    t0 = time.time()
    all_gaia_ids, file_gaia_ids = collect_all_gaia_ids(input_dir, verbose)
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
                        (label_source, INPUT_FILES[key], file_gaia_ids[key])
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
                survey_grp.create_dataset(
                    "spectra", data=out_spectra, compression="gzip", compression_opts=4
                )
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
                survey_grp.create_dataset(
                    "flux", data=out_flux, compression="gzip", compression_opts=4
                )
                survey_grp.create_dataset(
                    "ivar", data=out_ivar, compression="gzip", compression_opts=4
                )
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
            values = np.zeros((n_total, 11), dtype=np.float32)
            errors = np.zeros((n_total, 11), dtype=np.float32)
            flags = np.zeros((n_total, 11), dtype=np.uint8)
            has_labels = np.zeros(n_total, dtype=bool)

            # Process each survey file with this label source
            for survey in ["boss", "desi", "lamost_lrs", "lamost_mrs"]:
                key = (survey, label_source)
                if key not in file_gaia_ids:
                    continue

                filename = INPUT_FILES[key]
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
                            label_data
                        )
                    else:
                        file_vals, file_errs, file_flags = extract_galah_labels(
                            label_data
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
            label_grp.create_dataset("values", data=values, compression="gzip")
            label_grp.create_dataset("errors", data=errors, compression="gzip")
            label_grp.create_dataset("flags", data=flags)
            label_grp.create_dataset("gaia_id", data=all_gaia_ids)

            n_with_labels = int(has_labels.sum())
            print(f"done ({format_time(time.time() - t0_write)})")
            print(f"    Total stars with {label_source} labels: {n_with_labels:,}")
            stats["labels"][label_source] = {"n_with_labels": n_with_labels}

        print(f"\n  Pass 3 completed in {format_time(time.time() - t0_pass3)}")

        # Set file attributes
        f.attrs["n_stars"] = n_total
        f.attrs["parameter_names"] = PARAMETER_NAMES
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
        default=Path("data/super_catalogue.h5"),
        help="Output HDF5 file path (default: data/super_catalogue.h5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without creating output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )

    args = parser.parse_args()

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

    build_catalogue(args.input_dir, args.output, args.verbose, args.dry_run)


if __name__ == "__main__":
    main()
