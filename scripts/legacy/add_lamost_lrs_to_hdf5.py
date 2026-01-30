#!/usr/bin/env python
"""
Add LAMOST LRS survey data to the super-catalogue HDF5 file.

This script loads pre-computed LAMOST Low Resolution Spectrograph training data
from the data/raw/lamost_lrs/ folder and adds it to the HDF5 super-catalogue structure.

NOTE: This script assumes the HDF5 file already exists (created by add_boss_to_hdf5.py).

Data sources:
    - data/raw/lamost_lrs/X_lamost.npy: (140765, 2, 3473) spectra [flux, ivar]
    - data/raw/lamost_lrs/y_lamost.npy: (140765, 22) labels [11 params + 11 errors]
    - data/raw/lamost_lrs/apogee_ids_lamost.npy: (140765,) APOGEE 2MASS IDs
    - data/raw/lamost_lrs/lamost_training.fits: WAVELENGTH extension for correct grid

Usage:
    python scripts/add_lamost_lrs_to_hdf5.py [--output data/super_catalogue.h5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits


def load_lamost_lrs_data(base_path: Path) -> dict:
    """Load LAMOST LRS training data from npy files."""
    lamost_path = base_path / "data" / "raw" / "lamost_lrs"

    print(f"Loading LAMOST LRS data from {lamost_path}")

    # Load spectra: (N, 2, 3473) -> channel 0 = flux, channel 1 = ivar
    X = np.load(lamost_path / "X_lamost.npy")
    print(f"  X_lamost shape: {X.shape}, dtype: {X.dtype}")

    # Load labels: (N, 22) -> first 11 = normalized params, last 11 = normalized errors
    y = np.load(lamost_path / "y_lamost.npy")
    print(f"  y_lamost shape: {y.shape}, dtype: {y.dtype}")

    # Load IDs: (N,) -> APOGEE 2MASS IDs as strings
    ids = np.load(lamost_path / "apogee_ids_lamost.npy")
    print(f"  apogee_ids_lamost shape: {ids.shape}, dtype: {ids.dtype}")

    n_stars = X.shape[0]
    n_wavelengths = X.shape[2]

    # Split labels and errors
    labels = y[:, :11].astype(np.float32)
    errors = y[:, 11:].astype(np.float32)

    # Create flags (all zeros since we don't have per-parameter flags in the npy)
    flags = np.zeros((n_stars, 11), dtype=np.uint8)

    # LAMOST LRS wavelength grid: read from FITS file (log-linear, ~4000-8898 A)
    fits_path = lamost_path / "lamost_training.fits"
    if fits_path.exists():
        with fits.open(fits_path) as hdul:
            wavelength = hdul["WAVELENGTH"].data["WAVELENGTH"].astype(np.float32)
        print(
            f"  Wavelength from FITS: {len(wavelength)} pts, {wavelength[0]:.2f}-{wavelength[-1]:.2f} A"
        )
    else:
        # Fallback: compute log-linear grid (LAMOST uses COEFF0=3.5682, COEFF1=0.0001)
        print("  Warning: FITS file not found, computing log-linear wavelength")
        coeff0, coeff1 = 3.5682, 0.0001
        n_native = 3901
        loglam = coeff0 + coeff1 * np.arange(n_native)
        wave_full = 10**loglam
        # Subset to 3629-9750 A as done in original preprocessing
        mask = (wave_full > 3629) & (wave_full < 9750)
        wavelength = wave_full[mask].astype(np.float32)

    if len(wavelength) != n_wavelengths:
        print(
            f"  Warning: wavelength has {len(wavelength)} pts, expected {n_wavelengths}"
        )

    # Convert to float32 and extract flux/ivar
    flux = X[:, 0, :].astype(np.float32)
    ivar = X[:, 1, :].astype(np.float32)

    # Simple SNR estimate: median(flux * sqrt(ivar)) where ivar > 0
    print("  Computing SNR estimates...")
    snr = np.zeros(n_stars, dtype=np.float32)
    for i in range(0, n_stars, 1000):
        end_idx = min(i + 1000, n_stars)
        batch_flux = flux[i:end_idx]
        batch_ivar = ivar[i:end_idx]
        for j in range(end_idx - i):
            good = batch_ivar[j] > 0
            if good.sum() > 0:
                snr[i + j] = np.median(
                    np.abs(batch_flux[j, good]) * np.sqrt(batch_ivar[j, good])
                )
        if (i + 1000) % 10000 == 0:
            print(f"    Processed {min(i + 1000, n_stars)}/{n_stars} stars")

    # Check for duplicate IDs
    unique_ids = np.unique(ids)
    n_unique = len(unique_ids)
    n_duplicates = n_stars - n_unique
    print(f"  Unique stars: {n_unique}, duplicate observations: {n_duplicates}")

    return {
        "flux": flux,
        "ivar": ivar,
        "wavelength": wavelength,
        "snr": snr,
        "labels": labels,
        "errors": errors,
        "flags": flags,
        "ids": ids,
        "n_stars": n_stars,
        "n_unique": n_unique,
        "n_duplicates": n_duplicates,
    }


def add_to_hdf5(output_path: Path, lamost_data: dict) -> None:
    """Add LAMOST LRS data to the HDF5 super-catalogue."""
    n_stars = lamost_data["n_stars"]

    if not output_path.exists():
        raise FileNotFoundError(
            f"HDF5 file not found: {output_path}\n"
            "Run add_boss_to_hdf5.py first to create the initial catalogue."
        )

    print(f"Updating HDF5 file: {output_path}")

    with h5py.File(output_path, "a") as f:
        # Check if LAMOST LRS already exists
        if "lamost_lrs" in f["surveys"]:
            print("  Warning: LAMOST LRS data already exists, skipping survey data")
        else:
            # Add LAMOST LRS survey data
            lrs_grp = f["surveys"].create_group("lamost_lrs")
            lrs_grp.create_dataset(
                "flux",
                data=lamost_data["flux"],
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
            )
            lrs_grp.create_dataset(
                "ivar",
                data=lamost_data["ivar"],
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
            )
            lrs_grp.create_dataset("wavelength", data=lamost_data["wavelength"])
            lrs_grp.create_dataset("snr", data=lamost_data["snr"])

            # Update survey_names attribute
            current_surveys = list(f.attrs.get("survey_names", []))
            if "lamost_lrs" not in current_surveys:
                current_surveys.append("lamost_lrs")
                f.attrs["survey_names"] = current_surveys

            print(
                f"  Added LAMOST LRS survey: {n_stars} spectra, "
                f"{len(lamost_data['wavelength'])} wavelength bins"
            )

        # Add LAMOST-specific APOGEE labels (separate from BOSS labels)
        label_key = "apogee_lamost_lrs"
        if label_key in f["labels"]:
            print(f"  Warning: {label_key} labels already exist, skipping")
        else:
            apogee_grp = f["labels"].create_group(label_key)
            apogee_grp.create_dataset(
                "values",
                data=lamost_data["labels"],
                dtype=np.float32,
                compression="gzip",
            )
            apogee_grp.create_dataset(
                "errors",
                data=lamost_data["errors"],
                dtype=np.float32,
                compression="gzip",
            )
            apogee_grp.create_dataset("flags", data=lamost_data["flags"])

            # Store the IDs associated with these labels
            dt = h5py.string_dtype(encoding="utf-8")
            ids_list = [s.encode("utf-8") for s in lamost_data["ids"]]
            apogee_grp.create_dataset("gaia_id", data=ids_list, dtype=dt)

            print(f"  Added {label_key} labels: {n_stars} stars, 11 parameters")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Add LAMOST LRS survey data to super-catalogue HDF5"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/super_catalogue.h5"),
        help="Output HDF5 file path (default: data/super_catalogue.h5)",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Base path of the repository",
    )

    args = parser.parse_args()

    # Load LAMOST LRS data
    lamost_data = load_lamost_lrs_data(args.base_path)

    # Add to HDF5
    add_to_hdf5(args.output, lamost_data)


if __name__ == "__main__":
    main()
