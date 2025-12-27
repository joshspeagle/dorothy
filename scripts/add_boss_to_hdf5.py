#!/usr/bin/env python
"""
Add BOSS survey data to the super-catalogue HDF5 file.

This script loads pre-computed BOSS training data from the data/raw/boss/ folder
and adds it to the HDF5 super-catalogue structure.

Data sources:
    - data/raw/boss/X_BOSS.npy: (12152, 2, 4506) spectra [flux, ivar]
    - data/raw/boss/y_BOSS.npy: (12152, 22) labels [11 params + 11 errors]
    - data/raw/boss/APOGEE_ids.npy: (12152,) APOGEE 2MASS IDs

Usage:
    python scripts/add_boss_to_hdf5.py [--output data/super_catalogue.h5]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

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


def load_boss_data(base_path: Path) -> dict:
    """Load BOSS training data from npy files."""
    boss_path = base_path / "data" / "raw" / "boss"

    print(f"Loading BOSS data from {boss_path}")

    # Load spectra: (N, 2, 4506) -> channel 0 = flux, channel 1 = ivar
    X = np.load(boss_path / "X_BOSS.npy")
    print(f"  X_BOSS shape: {X.shape}, dtype: {X.dtype}")

    # Load labels: (N, 22) -> first 11 = normalized params, last 11 = normalized errors
    y = np.load(boss_path / "y_BOSS.npy")
    print(f"  y_BOSS shape: {y.shape}, dtype: {y.dtype}")

    # Load IDs: (N,) -> APOGEE 2MASS IDs as strings
    ids = np.load(boss_path / "APOGEE_ids.npy")
    print(f"  APOGEE_ids shape: {ids.shape}, dtype: {ids.dtype}")

    n_stars = X.shape[0]
    n_wavelengths = X.shape[2]

    # Split labels and errors
    labels = y[:, :11].astype(np.float32)
    errors = y[:, 11:].astype(np.float32)

    # Create flags (all zeros since we don't have per-parameter flags in the npy)
    flags = np.zeros((n_stars, 11), dtype=np.uint8)

    # BOSS wavelength grid: 3650-10300 Angstroms, 4506 bins
    # This is an approximation based on the typical BOSS wavelength coverage
    wavelength = np.linspace(3650, 10300, n_wavelengths).astype(np.float32)

    # Compute SNR from ivar (approximate: mean of sqrt(ivar * flux^2) where ivar > 0)
    flux = X[:, 0, :].astype(np.float32)
    ivar = X[:, 1, :].astype(np.float32)

    # Simple SNR estimate: median(flux * sqrt(ivar)) where ivar > 0
    snr = np.zeros(n_stars, dtype=np.float32)
    for i in range(n_stars):
        good = ivar[i] > 0
        if good.sum() > 0:
            snr[i] = np.median(np.abs(flux[i, good]) * np.sqrt(ivar[i, good]))

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


def create_or_update_hdf5(output_path: Path, boss_data: dict) -> None:
    """Create or update the HDF5 super-catalogue with BOSS data."""
    n_stars = boss_data["n_stars"]

    # Check if file exists
    file_exists = output_path.exists()
    mode = "a" if file_exists else "w"

    print(f"{'Updating' if file_exists else 'Creating'} HDF5 file: {output_path}")

    with h5py.File(output_path, mode) as f:
        # If new file, create metadata and attributes
        if not file_exists:
            # Create metadata group
            meta = f.create_group("metadata")

            # Use APOGEE IDs as primary key (stored as gaia_id for compatibility)
            # Note: These are APOGEE 2MASS IDs, not actual Gaia IDs
            # Store as variable-length strings - convert to list of bytes for h5py
            dt = h5py.string_dtype(encoding="utf-8")
            ids_list = [s.encode("utf-8") for s in boss_data["ids"]]
            meta.create_dataset("gaia_id", data=ids_list, dtype=dt)

            # Placeholder RA/Dec (zeros - we don't have coordinates in the npy files)
            meta.create_dataset("ra", data=np.zeros(n_stars, dtype=np.float64))
            meta.create_dataset("dec", data=np.zeros(n_stars, dtype=np.float64))

            # Create surveys group
            f.create_group("surveys")

            # Create labels group
            f.create_group("labels")

            # Set file attributes
            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = ["boss"]
            f.attrs["creation_date"] = datetime.now().isoformat()
            f.attrs["version"] = "1.0.0"

            print(f"  Created new catalogue with {n_stars} stars")

        # Add BOSS survey data
        if "boss" in f["surveys"]:
            print("  Warning: BOSS data already exists, skipping")
        else:
            boss_grp = f["surveys"].create_group("boss")
            boss_grp.create_dataset(
                "flux",
                data=boss_data["flux"],
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
            )
            boss_grp.create_dataset(
                "ivar",
                data=boss_data["ivar"],
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
            )
            boss_grp.create_dataset("wavelength", data=boss_data["wavelength"])
            boss_grp.create_dataset("snr", data=boss_data["snr"])
            print(
                f"  Added BOSS survey: {n_stars} spectra, {len(boss_data['wavelength'])} wavelength bins"
            )

        # Add APOGEE labels
        if "apogee" in f["labels"]:
            print("  Warning: APOGEE labels already exist, skipping")
        else:
            apogee_grp = f["labels"].create_group("apogee")
            apogee_grp.create_dataset(
                "values",
                data=boss_data["labels"],
                dtype=np.float32,
                compression="gzip",
            )
            apogee_grp.create_dataset(
                "errors",
                data=boss_data["errors"],
                dtype=np.float32,
                compression="gzip",
            )
            apogee_grp.create_dataset("flags", data=boss_data["flags"])

            # Store IDs for cross-referencing (initially as strings, later converted to Gaia IDs)
            dt = h5py.string_dtype(encoding="utf-8")
            ids_list = [s.encode("utf-8") for s in boss_data["ids"]]
            apogee_grp.create_dataset("gaia_id", data=ids_list, dtype=dt)

            print(f"  Added APOGEE labels: {n_stars} stars, {11} parameters")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Add BOSS survey data to super-catalogue HDF5"
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

    # Load BOSS data
    boss_data = load_boss_data(args.base_path)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Create/update HDF5
    create_or_update_hdf5(args.output, boss_data)


if __name__ == "__main__":
    main()
