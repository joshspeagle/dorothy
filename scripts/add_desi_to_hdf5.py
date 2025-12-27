#!/usr/bin/env python
"""
Add DESI survey data to the super-catalogue HDF5 file.

This script loads DESI spectra from a FITS file and matches them with
pre-computed APOGEE labels from the data/raw/desi/ folder.

Data sources:
    - data/raw/desi/DESI.fits: DESI spectra (1421, 7650)
    - data/raw/desi/y_BOSS_compare.npy: (1421, 22) APOGEE labels
    - data/raw/desi/APOGEE_ids.npy: (1421,) APOGEE 2MASS IDs

NOTE: These files are row-aligned. The 1421 stars have both DESI spectra
      and APOGEE labels from a cross-match.

Usage:
    python scripts/add_desi_to_hdf5.py [--output data/super_catalogue.h5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits


def load_desi_data(base_path: Path) -> dict:
    """Load DESI spectra and APOGEE labels from source files."""
    desi_path = base_path / "data" / "raw" / "desi"

    print(f"Loading DESI data from {desi_path}")

    # Load DESI spectra from FITS
    print("  Loading DESI.fits...")
    with fits.open(desi_path / "DESI.fits") as hdul:
        rvtab = hdul["RVTAB"].data
        spectra_data = hdul["SPECTRA"].data
        wavelength_data = hdul["WAVELENGTH"].data
        ivar_data = hdul["SPEC_IVAR"].data

        # Extract arrays from FITS columns
        flux = spectra_data[spectra_data.columns.names[0]].astype(np.float32)
        wavelength = wavelength_data[wavelength_data.columns.names[0]].astype(
            np.float32
        )
        ivar = ivar_data[ivar_data.columns.names[0]].astype(np.float32)

        # Get DESI TARGETIDs for reference
        targetids = rvtab["TARGETID"]

        # Get RA/Dec
        ra = rvtab["TARGET_RA"].astype(np.float64)
        dec = rvtab["TARGET_DEC"].astype(np.float64)

    n_stars = flux.shape[0]
    print(f"    DESI spectra: {flux.shape}")
    print(f"    Wavelength: {wavelength.shape}")

    # Load APOGEE labels (row-aligned with DESI spectra)
    print("  Loading APOGEE labels...")
    y = np.load(desi_path / "y_BOSS_compare.npy").astype(np.float32)
    print(f"    y_BOSS_compare: {y.shape}")

    # Load APOGEE IDs
    print("  Loading APOGEE IDs...")
    ids = np.load(desi_path / "APOGEE_ids.npy")
    print(f"    APOGEE_ids: {ids.shape}")

    # Verify alignment
    if len(ids) != n_stars or y.shape[0] != n_stars:
        raise ValueError(
            f"Row count mismatch: DESI={n_stars}, IDs={len(ids)}, labels={y.shape[0]}"
        )

    # Split labels and errors
    labels = y[:, :11]
    errors = y[:, 11:]

    # Create flags (all zeros)
    flags = np.zeros((n_stars, 11), dtype=np.uint8)

    # Compute SNR estimate
    print("  Computing SNR estimates...")
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
        "ra": ra,
        "dec": dec,
        "targetids": targetids,
        "labels": labels,
        "errors": errors,
        "flags": flags,
        "ids": ids,
        "n_stars": n_stars,
        "n_unique": n_unique,
        "n_duplicates": n_duplicates,
    }


def add_to_hdf5(output_path: Path, desi_data: dict) -> None:
    """Add DESI data to the HDF5 super-catalogue."""
    n_stars = desi_data["n_stars"]

    if not output_path.exists():
        raise FileNotFoundError(
            f"HDF5 file not found: {output_path}\n"
            "Run add_boss_to_hdf5.py first to create the initial catalogue."
        )

    print(f"Updating HDF5 file: {output_path}")

    with h5py.File(output_path, "a") as f:
        # Check if DESI already exists
        if "desi" in f["surveys"]:
            print("  Warning: DESI data already exists, skipping survey data")
        else:
            # Add DESI survey data
            desi_grp = f["surveys"].create_group("desi")
            desi_grp.create_dataset(
                "flux",
                data=desi_data["flux"],
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
            )
            desi_grp.create_dataset(
                "ivar",
                data=desi_data["ivar"],
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
            )
            desi_grp.create_dataset("wavelength", data=desi_data["wavelength"])
            desi_grp.create_dataset("snr", data=desi_data["snr"])

            # Store DESI TARGETIDs as additional metadata
            desi_grp.create_dataset("targetid", data=desi_data["targetids"])
            desi_grp.create_dataset("ra", data=desi_data["ra"])
            desi_grp.create_dataset("dec", data=desi_data["dec"])

            # Update survey_names attribute
            current_surveys = list(f.attrs.get("survey_names", []))
            if "desi" not in current_surveys:
                current_surveys.append("desi")
                f.attrs["survey_names"] = current_surveys

            print(
                f"  Added DESI survey: {n_stars} spectra, "
                f"{len(desi_data['wavelength'])} wavelength bins"
            )

        # Add DESI-specific APOGEE labels
        label_key = "apogee_desi"
        if label_key in f["labels"]:
            print(f"  Warning: {label_key} labels already exist, skipping")
        else:
            apogee_grp = f["labels"].create_group(label_key)
            apogee_grp.create_dataset(
                "values",
                data=desi_data["labels"],
                dtype=np.float32,
                compression="gzip",
            )
            apogee_grp.create_dataset(
                "errors",
                data=desi_data["errors"],
                dtype=np.float32,
                compression="gzip",
            )
            apogee_grp.create_dataset("flags", data=desi_data["flags"])

            # Store the IDs associated with these labels
            dt = h5py.string_dtype(encoding="utf-8")
            ids_list = [s.encode("utf-8") for s in desi_data["ids"]]
            apogee_grp.create_dataset("gaia_id", data=ids_list, dtype=dt)

            print(f"  Added {label_key} labels: {n_stars} stars, 11 parameters")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Add DESI survey data to super-catalogue HDF5"
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

    # Load DESI data
    desi_data = load_desi_data(args.base_path)

    # Add to HDF5
    add_to_hdf5(args.output, desi_data)


if __name__ == "__main__":
    main()
