#!/usr/bin/env python
"""
Replace 2MASS IDs with Gaia DR3 IDs in the super-catalogue HDF5 file.

This script reads the APOGEE DR17 catalogue to create a mapping from
APOGEE 2MASS IDs to Gaia EDR3 source IDs, then replaces the `gaia_id`
datasets in the super-catalogue with actual Gaia DR3 source IDs.

Usage:
    python scripts/add_gaia_ids_to_hdf5.py [--catalogue data/super_catalogue.h5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits


def load_apogee_gaia_mapping(apogee_path: Path) -> dict[str, int]:
    """
    Load APOGEE catalogue and create 2MASS ID -> Gaia ID mapping.

    Args:
        apogee_path: Path to allStar FITS file.

    Returns:
        Dictionary mapping 2MASS IDs to Gaia EDR3 source IDs.
    """
    print(f"Loading APOGEE catalogue from {apogee_path}...")

    with fits.open(apogee_path) as hdul:
        data = hdul[1].data
        apogee_ids = data["APOGEE_ID"]
        gaia_ids = data["GAIAEDR3_SOURCE_ID"]

    # Only include entries with valid Gaia IDs
    valid = gaia_ids > 0
    print(f"  Total entries: {len(apogee_ids):,}")
    print(f"  With valid Gaia IDs: {valid.sum():,}")

    # Create mapping (strip whitespace from 2MASS IDs)
    mapping = {}
    for tm_id, g_id in zip(apogee_ids[valid], gaia_ids[valid], strict=False):
        mapping[tm_id.strip()] = int(g_id)

    print(f"  Unique mappings: {len(mapping):,}")
    return mapping


def update_hdf5_with_gaia_ids(
    catalogue_path: Path, mapping: dict[str, int]
) -> dict[str, dict]:
    """
    Replace 2MASS IDs with Gaia DR3 IDs in the HDF5 super-catalogue.

    Replaces the 'gaia_id' dataset in each label group and metadata
    with actual Gaia DR3 source IDs (stored as int64).

    Args:
        catalogue_path: Path to super-catalogue HDF5 file.
        mapping: 2MASS ID -> Gaia ID mapping.

    Returns:
        Dictionary with update statistics for each group.
    """
    print(f"\nUpdating {catalogue_path}...")
    stats = {}

    with h5py.File(catalogue_path, "a") as f:
        # Update metadata/gaia_id
        if "metadata" in f and "gaia_id" in f["metadata"]:
            twomass_ids = f["metadata"]["gaia_id"][:]
            gaia_dr3_ids = np.zeros(len(twomass_ids), dtype=np.int64)

            for i, tm_id in enumerate(twomass_ids):
                if hasattr(tm_id, "decode"):
                    tm_id = tm_id.decode("utf-8")
                tm_id = str(tm_id).strip()
                gaia_dr3_ids[i] = mapping.get(tm_id, 0)

            # Replace the gaia_id dataset with Gaia DR3 IDs
            del f["metadata"]["gaia_id"]
            f["metadata"].create_dataset("gaia_id", data=gaia_dr3_ids)

            valid = gaia_dr3_ids > 0
            stats["metadata"] = {
                "total": len(twomass_ids),
                "matched": int(valid.sum()),
            }
            print(f"  metadata: {valid.sum():,}/{len(twomass_ids):,} matched")

        # Update each label group
        for label_name in f["labels"]:
            label_grp = f["labels"][label_name]

            if "gaia_id" in label_grp:
                twomass_ids = label_grp["gaia_id"][:]
            else:
                # Skip groups without IDs
                print(f"  {label_name}: no gaia_id dataset, skipping")
                continue

            gaia_dr3_ids = np.zeros(len(twomass_ids), dtype=np.int64)

            for i, tm_id in enumerate(twomass_ids):
                if hasattr(tm_id, "decode"):
                    tm_id = tm_id.decode("utf-8")
                tm_id = str(tm_id).strip()
                gaia_dr3_ids[i] = mapping.get(tm_id, 0)

            # Replace the gaia_id dataset with Gaia DR3 IDs
            del label_grp["gaia_id"]
            label_grp.create_dataset("gaia_id", data=gaia_dr3_ids)

            valid = gaia_dr3_ids > 0
            stats[label_name] = {
                "total": len(twomass_ids),
                "matched": int(valid.sum()),
            }
            print(f"  {label_name}: {valid.sum():,}/{len(twomass_ids):,} matched")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Add Gaia DR3 IDs to super-catalogue HDF5"
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=Path("data/super_catalogue.h5"),
        help="Path to super-catalogue HDF5 file",
    )
    parser.add_argument(
        "--apogee",
        type=Path,
        default=Path("data/raw/apogee/allStar-dr17-synspec_rev1.fits"),
        help="Path to APOGEE allStar FITS file",
    )

    args = parser.parse_args()

    if not args.apogee.exists():
        print(f"Error: APOGEE file not found: {args.apogee}")
        return 1

    if not args.catalogue.exists():
        print(f"Error: Catalogue file not found: {args.catalogue}")
        return 1

    # Load mapping
    mapping = load_apogee_gaia_mapping(args.apogee)

    # Update HDF5
    stats = update_hdf5_with_gaia_ids(args.catalogue, mapping)

    print("\nDone!")
    print("\nSummary:")
    for name, stat in stats.items():
        pct = 100 * stat["matched"] / stat["total"] if stat["total"] > 0 else 0
        print(f"  {name}: {stat['matched']:,}/{stat['total']:,} ({pct:.1f}%)")

    return 0


if __name__ == "__main__":
    exit(main())
