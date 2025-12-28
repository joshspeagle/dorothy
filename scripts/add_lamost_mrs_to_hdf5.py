#!/usr/bin/env python
"""
Add LAMOST MRS survey data to the super-catalogue HDF5 file.

This script loads pre-computed LAMOST Medium Resolution Spectrograph training data
from the data/raw/lamost_mrs/ folder and adds it to the HDF5 super-catalogue structure.

LAMOST MRS has 4 channels (2 arms x flux/ivar).

NOTE: There are two data files (X_lamost_mrs.npy and X_lamost_mrs2.npy).
      Batch 1 is completely contained within batch 2, so we only use batch 2
      which has 71,245 spectra (58,266 unique stars with some repeat observations).

Data sources:
    - data/raw/lamost_mrs/X_lamost_mrs2.npy: (71245, 4, 3375) spectra
    - data/raw/lamost_mrs/y_lamost_mrs2.npy: (71245, 22) labels
    - data/raw/lamost_mrs/apogee_ids_lamost_mrs2.npy: (71245,) IDs
    - data/raw/lamost_mrs/lamost_mrs_training2.fits: WAVELENGTH_B/R for correct grids

Usage:
    python scripts/add_lamost_mrs_to_hdf5.py [--output data/super_catalogue.h5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits


def load_lamost_mrs_data(base_path: Path) -> dict:
    """Load LAMOST MRS training data from npy files (batch 2 only)."""
    mrs_path = base_path / "data" / "raw" / "lamost_mrs"

    print(f"Loading LAMOST MRS data from {mrs_path}")

    # Only use batch 2 (batch 1 is completely contained within batch 2)
    print("  Loading batch 2 (superset of batch 1)...")
    X = np.load(mrs_path / "X_lamost_mrs2.npy").astype(np.float32)
    y = np.load(mrs_path / "y_lamost_mrs2.npy").astype(np.float32)
    ids = np.load(mrs_path / "apogee_ids_lamost_mrs2.npy")
    print(f"    X_lamost_mrs2: {X.shape}, y: {y.shape}, ids: {ids.shape}")

    n_stars = X.shape[0]
    n_channels = X.shape[1]  # 4 channels for MRS
    n_wavelengths = X.shape[2]

    # Check for duplicate IDs
    unique_ids = np.unique(ids)
    n_unique = len(unique_ids)
    n_duplicates = n_stars - n_unique
    print(
        f"  Loaded: {n_stars} spectra, {n_channels} channels, {n_wavelengths} wavelengths"
    )
    print(f"  Unique stars: {n_unique}, duplicate observations: {n_duplicates}")

    # Split labels and errors
    labels = y[:, :11]
    errors = y[:, 11:]

    # Create flags (all zeros)
    flags = np.zeros((n_stars, 11), dtype=np.uint8)

    # LAMOST MRS wavelength grids: separate blue and red arms with ~960 A gap
    # Blue arm: 4950-5350 A (3375 pts), Red arm: 6310-6800 A (3248 pts, padded to 3375)
    fits_path = mrs_path / "lamost_mrs_training2.fits"
    if fits_path.exists():
        with fits.open(fits_path) as hdul:
            wavelength_b = hdul["WAVELENGTH_B"].data["WAVELENGTH_B"].astype(np.float32)
            wavelength_r = hdul["WAVELENGTH_R"].data["WAVELENGTH_R"].astype(np.float32)
        print(
            f"  Blue wavelength from FITS: {len(wavelength_b)} pts, {wavelength_b[0]:.2f}-{wavelength_b[-1]:.2f} A"
        )
        print(
            f"  Red wavelength from FITS: {len(wavelength_r)} pts, {wavelength_r[0]:.2f}-{wavelength_r[-1]:.2f} A"
        )
    else:
        # Fallback: approximate log-linear grids
        print("  Warning: FITS file not found, using approximate wavelength grids")
        wavelength_b = np.linspace(4950, 5350, 3375).astype(np.float32)
        wavelength_r = np.linspace(6310, 6800, 3248).astype(np.float32)

    # For MRS, channels are typically [flux_B, ivar_B, flux_R, ivar_R]
    # or similar arrangement. Store all 4 channels.
    # SNR estimate using first flux channel
    print("  Computing SNR estimates...")
    flux_b = X[:, 0, :]
    ivar_b = X[:, 1, :]

    snr = np.zeros(n_stars, dtype=np.float32)
    for i in range(0, n_stars, 1000):
        end_idx = min(i + 1000, n_stars)
        batch_flux = flux_b[i:end_idx]
        batch_ivar = ivar_b[i:end_idx]
        for j in range(end_idx - i):
            good = batch_ivar[j] > 0
            if good.sum() > 0:
                snr[i + j] = np.median(
                    np.abs(batch_flux[j, good]) * np.sqrt(batch_ivar[j, good])
                )
        if (i + 1000) % 20000 == 0:
            print(f"    Processed {min(i + 1000, n_stars)}/{n_stars} stars")

    return {
        "spectra": X,  # Full 4-channel array
        "wavelength_b": wavelength_b,  # Blue arm wavelength
        "wavelength_r": wavelength_r,  # Red arm wavelength
        "snr": snr,
        "labels": labels,
        "errors": errors,
        "flags": flags,
        "ids": ids,
        "n_stars": n_stars,
        "n_channels": n_channels,
        "n_unique": n_unique,
        "n_duplicates": n_duplicates,
    }


def add_to_hdf5(output_path: Path, mrs_data: dict) -> None:
    """Add LAMOST MRS data to the HDF5 super-catalogue."""
    n_stars = mrs_data["n_stars"]

    if not output_path.exists():
        raise FileNotFoundError(
            f"HDF5 file not found: {output_path}\n"
            "Run add_boss_to_hdf5.py first to create the initial catalogue."
        )

    print(f"Updating HDF5 file: {output_path}")

    with h5py.File(output_path, "a") as f:
        # Check if LAMOST MRS already exists
        if "lamost_mrs" in f["surveys"]:
            print("  Warning: LAMOST MRS data already exists, skipping survey data")
        else:
            # Add LAMOST MRS survey data
            mrs_grp = f["surveys"].create_group("lamost_mrs")

            # Store the full 4-channel spectra
            # Shape: (n_stars, 4, n_wavelengths)
            mrs_grp.create_dataset(
                "spectra",
                data=mrs_data["spectra"],
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
            )
            # Store separate wavelength arrays for blue and red arms
            mrs_grp.create_dataset("wavelength_b", data=mrs_data["wavelength_b"])
            mrs_grp.create_dataset("wavelength_r", data=mrs_data["wavelength_r"])
            mrs_grp.create_dataset("snr", data=mrs_data["snr"])

            # Add metadata about channel layout and wavelength structure
            mrs_grp.attrs["n_channels"] = mrs_data["n_channels"]
            mrs_grp.attrs["channel_names"] = ["flux_B", "ivar_B", "flux_R", "ivar_R"]
            mrs_grp.attrs["wavelength_note"] = (
                "Blue and red arms have separate wavelength arrays. "
                "Blue: 4950-5350 A, Red: 6310-6800 A with ~960 A gap."
            )

            # Update survey_names attribute
            current_surveys = list(f.attrs.get("survey_names", []))
            if "lamost_mrs" not in current_surveys:
                current_surveys.append("lamost_mrs")
                f.attrs["survey_names"] = current_surveys

            print(
                f"  Added LAMOST MRS survey: {n_stars} spectra, "
                f"{mrs_data['n_channels']} channels"
            )
            print(
                f"    Blue: {len(mrs_data['wavelength_b'])} wavelength bins, "
                f"Red: {len(mrs_data['wavelength_r'])} wavelength bins"
            )

        # Add LAMOST MRS-specific APOGEE labels
        label_key = "apogee_lamost_mrs"
        if label_key in f["labels"]:
            print(f"  Warning: {label_key} labels already exist, skipping")
        else:
            apogee_grp = f["labels"].create_group(label_key)
            apogee_grp.create_dataset(
                "values",
                data=mrs_data["labels"],
                dtype=np.float32,
                compression="gzip",
            )
            apogee_grp.create_dataset(
                "errors",
                data=mrs_data["errors"],
                dtype=np.float32,
                compression="gzip",
            )
            apogee_grp.create_dataset("flags", data=mrs_data["flags"])

            # Store the IDs associated with these labels
            dt = h5py.string_dtype(encoding="utf-8")
            ids_list = [s.encode("utf-8") for s in mrs_data["ids"]]
            apogee_grp.create_dataset("gaia_id", data=ids_list, dtype=dt)

            print(f"  Added {label_key} labels: {n_stars} stars, 11 parameters")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Add LAMOST MRS survey data to super-catalogue HDF5"
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

    # Load LAMOST MRS data
    mrs_data = load_lamost_mrs_data(args.base_path)

    # Add to HDF5
    add_to_hdf5(args.output, mrs_data)


if __name__ == "__main__":
    main()
