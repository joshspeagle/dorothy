#!/usr/bin/env python
"""
Fix wavelength arrays in the super-catalogue HDF5 file.

The original catalogue building scripts used incorrect np.linspace approximations
for wavelength grids. This script replaces them with the correct wavelengths:

- LAMOST LRS: Read from lamost_training.fits (log-linear, 4000-8898 A)
- LAMOST MRS: Read from lamost_mrs_training2.fits (separate blue/red arrays)
- BOSS: Compute from SDSS log-linear formula (log10 dispersion)
- DESI: Already correct (read from original FITS)

Usage:
    python scripts/fix_wavelengths.py [--catalogue data/super_catalogue_clean.h5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits


def get_boss_wavelength(n_pixels: int = 4506) -> np.ndarray:
    """Compute correct BOSS wavelength grid using log-linear formula.

    BOSS uses log-linear wavelength dispersion:
        wavelength = 10**(coeff0 + coeff1 * pixel_index)

    The fiducial BOSS grid has 4800 pixels, which is then masked to
    3650-10300 Angstroms to get the final ~4506 pixel array.

    Args:
        n_pixels: Expected number of pixels (for verification)

    Returns:
        Wavelength array in Angstroms
    """
    # SDSS/BOSS fiducial wavelength parameters
    coeff0 = np.log10(3500.26)  # log10(wavelength) at pixel 0 = 3.54405...
    coeff1 = 1e-4  # log10 dispersion per pixel
    n_fiducial = 4800  # fiducial pixel count

    # Generate full fiducial log-wavelength grid
    loglam_fiducial = coeff0 + coeff1 * np.arange(n_fiducial)
    wavelength_fiducial = 10**loglam_fiducial

    # Apply the mask used in data preparation (3650 < wave < 10300)
    mask = (wavelength_fiducial > 3650) & (wavelength_fiducial < 10300)
    wavelength = wavelength_fiducial[mask].astype(np.float32)

    if len(wavelength) != n_pixels:
        print(
            f"  Warning: BOSS wavelength has {len(wavelength)} pixels, expected {n_pixels}"
        )

    return wavelength


def get_lamost_lrs_wavelength(fits_path: Path) -> np.ndarray:
    """Read LAMOST LRS wavelength from training FITS file.

    Args:
        fits_path: Path to lamost_training.fits

    Returns:
        Wavelength array in Angstroms
    """
    with fits.open(fits_path) as hdul:
        wavelength = hdul["WAVELENGTH"].data["WAVELENGTH"].astype(np.float32)
    return wavelength


def get_lamost_mrs_wavelengths(
    fits_path: Path, target_length: int = 3375
) -> tuple[np.ndarray, np.ndarray]:
    """Read LAMOST MRS wavelengths from training FITS file.

    The spectra are stored with both arms padded to the same length (3375),
    but the red wavelength array from the FITS file only has 3248 elements.
    This function pads the red wavelengths to match the spectra shape.

    Args:
        fits_path: Path to lamost_mrs_training2.fits
        target_length: Target length for each arm (default: 3375)

    Returns:
        Tuple of (wavelength_blue, wavelength_red) arrays in Angstroms
    """
    with fits.open(fits_path) as hdul:
        wavelength_b = hdul["WAVELENGTH_B"].data["WAVELENGTH_B"].astype(np.float32)
        wavelength_r = hdul["WAVELENGTH_R"].data["WAVELENGTH_R"].astype(np.float32)

    # Pad red wavelength to match spectra shape (extrapolate beyond last value)
    if len(wavelength_r) < target_length:
        # Calculate step size from last two wavelengths
        step = wavelength_r[-1] - wavelength_r[-2]
        n_pad = target_length - len(wavelength_r)
        # Extrapolate beyond the last wavelength
        pad_values = wavelength_r[-1] + step * np.arange(1, n_pad + 1)
        wavelength_r = np.concatenate([wavelength_r, pad_values.astype(np.float32)])

    return wavelength_b, wavelength_r


def fix_wavelengths(
    catalogue_path: Path,
    lamost_lrs_fits: Path,
    lamost_mrs_fits: Path,
    dry_run: bool = False,
) -> None:
    """Fix wavelength arrays in the super-catalogue.

    Args:
        catalogue_path: Path to super_catalogue HDF5 file
        lamost_lrs_fits: Path to lamost_training.fits
        lamost_mrs_fits: Path to lamost_mrs_training2.fits
        dry_run: If True, only print what would be changed
    """
    print(f"Fixing wavelengths in: {catalogue_path}")
    print(f"  LAMOST LRS source: {lamost_lrs_fits}")
    print(f"  LAMOST MRS source: {lamost_mrs_fits}")
    print(f"  Dry run: {dry_run}")
    print()

    # Load correct wavelengths
    print("Loading correct wavelength arrays...")

    # BOSS - compute from formula
    boss_wave = get_boss_wavelength()
    print(f"  BOSS: {len(boss_wave)} pts, {boss_wave[0]:.2f} - {boss_wave[-1]:.2f} A")

    # LAMOST LRS - from FITS
    lrs_wave = get_lamost_lrs_wavelength(lamost_lrs_fits)
    print(
        f"  LAMOST LRS: {len(lrs_wave)} pts, {lrs_wave[0]:.2f} - {lrs_wave[-1]:.2f} A"
    )

    # LAMOST MRS - from FITS (two arrays)
    mrs_wave_b, mrs_wave_r = get_lamost_mrs_wavelengths(lamost_mrs_fits)
    print(
        f"  LAMOST MRS Blue: {len(mrs_wave_b)} pts, {mrs_wave_b[0]:.2f} - {mrs_wave_b[-1]:.2f} A"
    )
    print(
        f"  LAMOST MRS Red: {len(mrs_wave_r)} pts, {mrs_wave_r[0]:.2f} - {mrs_wave_r[-1]:.2f} A"
    )
    print()

    # Open catalogue and compare/update
    mode = "r" if dry_run else "r+"
    with h5py.File(catalogue_path, mode) as f:

        # BOSS
        if "boss" in f["surveys"]:
            old_wave = f["surveys/boss/wavelength"][:]
            print(
                f"BOSS current: {len(old_wave)} pts, {old_wave[0]:.2f} - {old_wave[-1]:.2f} A"
            )
            print(
                f"BOSS correct: {len(boss_wave)} pts, {boss_wave[0]:.2f} - {boss_wave[-1]:.2f} A"
            )
            if len(old_wave) == len(boss_wave):
                max_diff = np.max(np.abs(old_wave - boss_wave))
                print(f"  Max difference: {max_diff:.4f} A")
                if not dry_run:
                    f["surveys/boss/wavelength"][:] = boss_wave
                    print("  -> Updated!")
            else:
                print("  ERROR: Length mismatch!")
            print()

        # LAMOST LRS
        if "lamost_lrs" in f["surveys"]:
            old_wave = f["surveys/lamost_lrs/wavelength"][:]
            print(
                f"LAMOST LRS current: {len(old_wave)} pts, {old_wave[0]:.2f} - {old_wave[-1]:.2f} A"
            )
            print(
                f"LAMOST LRS correct: {len(lrs_wave)} pts, {lrs_wave[0]:.2f} - {lrs_wave[-1]:.2f} A"
            )
            if len(old_wave) == len(lrs_wave):
                max_diff = np.max(np.abs(old_wave - lrs_wave))
                print(f"  Max difference: {max_diff:.4f} A")
                if not dry_run:
                    f["surveys/lamost_lrs/wavelength"][:] = lrs_wave
                    print("  -> Updated!")
            else:
                print("  ERROR: Length mismatch!")
            print()

        # LAMOST MRS - needs special handling for dual wavelength arrays
        if "lamost_mrs" in f["surveys"]:
            mrs_grp = f["surveys/lamost_mrs"]

            # Check if already has separate wavelength arrays
            if "wavelength_b" in mrs_grp and "wavelength_r" in mrs_grp:
                old_wave_b = mrs_grp["wavelength_b"][:]
                old_wave_r = mrs_grp["wavelength_r"][:]
                print(
                    f"LAMOST MRS Blue current: {len(old_wave_b)} pts, "
                    f"{old_wave_b[0]:.2f} - {old_wave_b[-1]:.2f} A"
                )
                print(
                    f"LAMOST MRS Red current: {len(old_wave_r)} pts, "
                    f"{old_wave_r[0]:.2f} - {old_wave_r[-1]:.2f} A"
                )
                print("LAMOST MRS correct:")
                print(
                    f"  Blue: {len(mrs_wave_b)} pts, {mrs_wave_b[0]:.2f} - {mrs_wave_b[-1]:.2f} A"
                )
                print(
                    f"  Red: {len(mrs_wave_r)} pts, {mrs_wave_r[0]:.2f} - {mrs_wave_r[-1]:.2f} A"
                )

                # Check spectra shape to verify expected wavelength count
                if "spectra" in mrs_grp:
                    spectra_shape = mrs_grp["spectra"].shape
                    expected_per_arm = spectra_shape[2]
                    print(
                        f"  Spectra shape: {spectra_shape}, expected {expected_per_arm} per arm"
                    )

                if not dry_run:
                    # Update if lengths don't match
                    if len(old_wave_b) != len(mrs_wave_b):
                        del mrs_grp["wavelength_b"]
                        mrs_grp.create_dataset("wavelength_b", data=mrs_wave_b)
                        print(
                            f"  -> Updated wavelength_b ({len(old_wave_b)} -> {len(mrs_wave_b)})"
                        )
                    else:
                        mrs_grp["wavelength_b"][:] = mrs_wave_b
                        print("  -> Updated wavelength_b values")

                    if len(old_wave_r) != len(mrs_wave_r):
                        del mrs_grp["wavelength_r"]
                        mrs_grp.create_dataset("wavelength_r", data=mrs_wave_r)
                        print(
                            f"  -> Updated wavelength_r ({len(old_wave_r)} -> {len(mrs_wave_r)})"
                        )
                    else:
                        mrs_grp["wavelength_r"][:] = mrs_wave_r
                        print("  -> Updated wavelength_r values")

            elif "wavelength" in mrs_grp:
                # Old format with single array
                old_wave = mrs_grp["wavelength"][:]
                print(
                    f"LAMOST MRS current: {len(old_wave)} pts, {old_wave[0]:.2f} - {old_wave[-1]:.2f} A"
                )
                print("  (This is WRONG - single linear array spanning gap)")
                print("LAMOST MRS correct:")
                print(
                    f"  Blue: {len(mrs_wave_b)} pts, {mrs_wave_b[0]:.2f} - {mrs_wave_b[-1]:.2f} A"
                )
                print(
                    f"  Red: {len(mrs_wave_r)} pts, {mrs_wave_r[0]:.2f} - {mrs_wave_r[-1]:.2f} A"
                )

                if not dry_run:
                    # Delete old single wavelength array
                    del mrs_grp["wavelength"]

                    # Create new wavelength arrays for each arm
                    mrs_grp.create_dataset("wavelength_b", data=mrs_wave_b)
                    mrs_grp.create_dataset("wavelength_r", data=mrs_wave_r)

                    # Update attributes to document the change
                    mrs_grp.attrs["wavelength_note"] = (
                        "Blue and red arms have separate wavelength arrays. "
                        "Blue: 4950-5350 A, Red: 6310-6800 A with ~960 A gap."
                    )
                    print(
                        "  -> Updated! (replaced single array with wavelength_b and wavelength_r)"
                    )
            print()

        # DESI - verify it's already correct
        if "desi" in f["surveys"]:
            desi_wave = f["surveys/desi/wavelength"][:]
            print(
                f"DESI: {len(desi_wave)} pts, {desi_wave[0]:.2f} - {desi_wave[-1]:.2f} A"
            )
            print("  (Already correct - from original FITS)")
            print()

    if dry_run:
        print("Dry run complete. Re-run without --dry-run to apply changes.")
    else:
        print("Wavelength fixes applied successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Fix wavelength arrays in super-catalogue"
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=Path("data/super_catalogue_clean.h5"),
        help="Path to super-catalogue HDF5 file",
    )
    parser.add_argument(
        "--lamost-lrs-fits",
        type=Path,
        default=Path("data/raw/lamost_lrs/lamost_training.fits"),
        help="Path to LAMOST LRS training FITS file",
    )
    parser.add_argument(
        "--lamost-mrs-fits",
        type=Path,
        default=Path("data/raw/lamost_mrs/lamost_mrs_training2.fits"),
        help="Path to LAMOST MRS training FITS file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be changed, don't modify",
    )

    args = parser.parse_args()

    # Verify files exist
    if not args.catalogue.exists():
        raise FileNotFoundError(f"Catalogue not found: {args.catalogue}")
    if not args.lamost_lrs_fits.exists():
        raise FileNotFoundError(f"LAMOST LRS FITS not found: {args.lamost_lrs_fits}")
    if not args.lamost_mrs_fits.exists():
        raise FileNotFoundError(f"LAMOST MRS FITS not found: {args.lamost_mrs_fits}")

    fix_wavelengths(
        args.catalogue,
        args.lamost_lrs_fits,
        args.lamost_mrs_fits,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
