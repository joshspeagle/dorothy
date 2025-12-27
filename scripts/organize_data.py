#!/usr/bin/env python
"""
Organize raw data files into the centralized data/ folder structure.

This script moves/copies raw FITS files from d5martin/ to the new data/raw/ structure.

Usage:
    python scripts/organize_data.py --dry-run  # Show what would be done
    python scripts/organize_data.py --copy     # Copy files (keeps originals)
    python scripts/organize_data.py --move     # Move files (removes originals)
"""

import argparse
import shutil
from pathlib import Path


# Define source -> destination mappings for raw data files
# Format: (source_glob, destination_folder, description)
DATA_MAPPINGS = [
    # APOGEE
    (
        "d5martin/H Model/allStar*.fits",
        "data/raw/apogee",
        "APOGEE allStar catalogue",
    ),
    (
        "d5martin/H Model/allStarLite*.fits",
        "data/raw/apogee",
        "APOGEE allStarLite catalogue",
    ),
    # BOSS
    (
        "d5martin/BOSS_model/DOROTHY/spAll-lite*.fits*",
        "data/raw/boss",
        "BOSS spAll-lite catalogue",
    ),
    # DESI
    (
        "d5martin/BOSS_model/DOROTHY/DESI*.fits",
        "data/raw/desi",
        "DESI training cube",
    ),
    # LAMOST LRS
    (
        "d5martin/LAMOST/lamost_training*.fits",
        "data/raw/lamost_lrs",
        "LAMOST LRS training data",
    ),
    (
        "d5martin/LAMOST/dr11*.fits",
        "data/raw/lamost_lrs",
        "LAMOST LRS DR11 catalogue",
    ),
    # LAMOST MRS
    (
        "d5martin/LAMOST/LAMOST_MRS/lamost_mrs*.fits",
        "data/raw/lamost_mrs",
        "LAMOST MRS training data",
    ),
    (
        "d5martin/LAMOST/LAMOST_MRS/dr11*.fits",
        "data/raw/lamost_mrs",
        "LAMOST MRS DR11 catalogue",
    ),
    # GALAH (if present)
    (
        "d5martin/**/galah_dr4*.fits",
        "data/raw/galah",
        "GALAH DR4 catalogue",
    ),
    (
        "d5martin/**/GALAH*.fits",
        "data/raw/galah",
        "GALAH catalogue",
    ),
]


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024)


def find_files(base_path: Path, mapping: tuple) -> list[tuple[Path, Path, str]]:
    """Find files matching a mapping pattern."""
    pattern, dest_folder, description = mapping
    dest_path = base_path / dest_folder

    results = []
    for source in base_path.glob(pattern):
        if source.is_file():
            dest_file = dest_path / source.name
            results.append((source, dest_file, description))

    return results


def organize_data(base_path: Path, mode: str = "dry-run") -> None:
    """Organize data files according to mappings."""
    print(f"Base path: {base_path}")
    print(f"Mode: {mode}")
    print()

    total_size = 0
    file_count = 0

    for mapping in DATA_MAPPINGS:
        files = find_files(base_path, mapping)
        if files:
            print(f"[{mapping[2]}]")
            for source, dest, _ in files:
                size_mb = get_file_size_mb(source)
                total_size += size_mb
                file_count += 1

                status = ""
                if dest.exists():
                    status = " (already exists, skipping)"
                else:
                    if mode == "copy":
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, dest)
                        status = " -> COPIED"
                    elif mode == "move":
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(source, dest)
                        status = " -> MOVED"
                    else:
                        status = " (would copy)"

                print(f"  {source.name} ({size_mb:.1f} MB){status}")
            print()

    print(f"Total: {file_count} files, {total_size:.1f} MB ({total_size/1024:.2f} GB)")

    if mode == "dry-run":
        print()
        print("This was a dry run. Use --copy or --move to actually transfer files.")


def main():
    parser = argparse.ArgumentParser(
        description="Organize raw data files into data/ folder structure."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be done (default)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files (keeps originals in place)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files (removes originals)",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Base path of the repository",
    )

    args = parser.parse_args()

    if args.copy:
        mode = "copy"
    elif args.move:
        mode = "move"
    else:
        mode = "dry-run"

    organize_data(args.base_path, mode)


if __name__ == "__main__":
    main()
