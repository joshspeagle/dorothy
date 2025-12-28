#!/usr/bin/env python
"""
Create a deduplicated super-catalogue by removing duplicate observations.

This script processes a super-catalogue HDF5 file and removes duplicate
observations for stars that appear multiple times in a survey. For each
star with multiple observations, it either:
  - Stacks consistent observations (weighted average) if chi2 < threshold
  - Takes the observation with highest SNR if chi2 >= threshold

Usage:
    python scripts/create_deduplicated_catalogue.py \\
        data/super_catalogue.h5 \\
        data/super_catalogue_clean.h5 \\
        --surveys lamost_lrs lamost_mrs \\
        --chi2-threshold 2.0

After creating the deduplicated catalogue, update your config:
    data:
      catalogue_path: data/super_catalogue_clean.h5
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


# Mapping from survey names to their label group names
SURVEY_LABEL_MAP = {
    "boss": "apogee_boss",
    "lamost_lrs": "apogee_lamost_lrs",
    "lamost_mrs": "apogee_lamost_mrs",
    "desi": "apogee_desi",
}


def compute_weighted_spectrum(
    obs_flux: np.ndarray,
    obs_ivar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute inverse-variance weighted mean spectrum."""
    n_wave = obs_flux.shape[1]
    sum_weights = np.sum(obs_ivar, axis=0)
    valid = sum_weights > 0
    weighted_flux = np.zeros(n_wave, dtype=np.float32)
    weighted_flux[valid] = (
        np.sum(obs_flux[:, valid] * obs_ivar[:, valid], axis=0) / sum_weights[valid]
    )
    return weighted_flux, sum_weights


def compute_reduced_chi2(
    obs_flux: np.ndarray,
    obs_ivar: np.ndarray,
    weighted_flux: np.ndarray,
) -> float:
    """Compute reduced chi-squared across observations."""
    n_obs = obs_flux.shape[0]
    residuals = obs_flux - weighted_flux
    chi2 = np.sum(residuals**2 * obs_ivar)
    valid_count = np.sum(obs_ivar > 0)
    dof = max(1, valid_count - n_obs)
    return chi2 / dof


def stack_labels(
    obs_labels: np.ndarray,
    obs_errors: np.ndarray,
    obs_flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack labels using inverse-variance weighting."""
    n_params = obs_labels.shape[1]
    stacked_labels = np.zeros(n_params, dtype=np.float32)
    stacked_errors = np.zeros(n_params, dtype=np.float32)

    # Compute weights from errors (inverse variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        label_weights = np.where(obs_errors > 0, 1.0 / obs_errors**2, 0)
    sum_label_weights = np.sum(label_weights, axis=0)

    has_valid = sum_label_weights > 0
    stacked_labels[has_valid] = (
        np.sum(obs_labels[:, has_valid] * label_weights[:, has_valid], axis=0)
        / sum_label_weights[has_valid]
    )
    stacked_errors[has_valid] = 1.0 / np.sqrt(sum_label_weights[has_valid])

    # Combine flags (OR across observations)
    stacked_flags = np.bitwise_or.reduce(obs_flags, axis=0)

    return stacked_labels, stacked_errors, stacked_flags


def deduplicate_survey(
    f_in: h5py.File,
    f_out: h5py.File,
    survey: str,
    chi2_threshold: float,
    verbose: bool = True,
) -> dict:
    """
    Deduplicate a single survey and write to output file.

    Returns:
        Dict with statistics about deduplication.
    """
    if survey not in f_in["surveys"]:
        print(f"  Skipping {survey} - not found in catalogue")
        return {"skipped": True}

    survey_grp = f_in["surveys"][survey]

    # Get the label source for this survey
    label_source = SURVEY_LABEL_MAP.get(survey)
    if label_source is None or label_source not in f_in["labels"]:
        print(f"  Skipping {survey} - no matching label source found")
        return {"skipped": True}

    label_grp = f_in["labels"][label_source]

    # Determine storage format
    if "flux" in survey_grp:
        storage_format = "flux"
    elif "spectra" in survey_grp:
        storage_format = "spectra"
    else:
        print(f"  Skipping {survey} - unknown storage format")
        return {"skipped": True}

    # Load spectra based on format
    is_dual_arm = False
    if storage_format == "spectra":
        # LAMOST MRS format: (N, 4, wavelengths) - [flux_b, ivar_b, flux_r, ivar_r]
        spectra = survey_grp["spectra"][:]
        flux_b = spectra[:, 0, :]
        ivar_b = spectra[:, 1, :]
        flux_r = spectra[:, 2, :]
        ivar_r = spectra[:, 3, :]
        # Concatenate for deduplication
        flux = np.concatenate([flux_b, flux_r], axis=1)
        ivar = np.concatenate([ivar_b, ivar_r], axis=1)
        n_wave_b = flux_b.shape[1]
        is_dual_arm = True
    elif "flux_b" in survey_grp and "flux_r" in survey_grp:
        flux_b = survey_grp["flux_b"][:]
        flux_r = survey_grp["flux_r"][:]
        ivar_b = survey_grp["ivar_b"][:]
        ivar_r = survey_grp["ivar_r"][:]
        flux = np.concatenate([flux_b, flux_r], axis=1)
        ivar = np.concatenate([ivar_b, ivar_r], axis=1)
        n_wave_b = flux_b.shape[1]
        is_dual_arm = True
    else:
        flux = survey_grp["flux"][:]
        ivar = survey_grp["ivar"][:]

    snr = survey_grp["snr"][:]

    # Get gaia_ids from the labels group (this is where they're stored per survey)
    if "gaia_id" in label_grp:
        gaia_ids = label_grp["gaia_id"][:]
    elif "gaia_id" in survey_grp:
        gaia_ids = survey_grp["gaia_id"][:]
    else:
        print(f"  Error: No gaia_id found for {survey}")
        return {"skipped": True}

    # Load labels for this survey
    labels = label_grp["values"][:]
    label_errors = label_grp["errors"][:]
    label_flags = label_grp["flags"][:]

    n_orig = len(gaia_ids)

    # Verify dimensions match
    if len(flux) != n_orig:
        print(
            f"  Error: Dimension mismatch for {survey}: {len(flux)} spectra vs {n_orig} gaia_ids"
        )
        return {"skipped": True}

    # Find duplicates
    valid_mask = gaia_ids > 0
    valid_ids = gaia_ids[valid_mask].astype(np.int64)
    unique_ids, counts = np.unique(valid_ids, return_counts=True)
    duplicate_gaia_ids = set(unique_ids[counts > 1])

    n_with_dups = len(duplicate_gaia_ids)

    # Build index: gaia_id -> list of indices (for all stars)
    id_to_indices: dict[int, list[int]] = {}
    for idx, gid in enumerate(gaia_ids):
        if gid <= 0:
            continue
        gid_int = int(gid)
        if gid_int not in id_to_indices:
            id_to_indices[gid_int] = []
        id_to_indices[gid_int].append(idx)

    # Process and build output arrays
    n_unique = len(id_to_indices)
    n_wave = flux.shape[1]
    n_params = labels.shape[1]

    out_flux = np.zeros((n_unique, n_wave), dtype=np.float32)
    out_ivar = np.zeros((n_unique, n_wave), dtype=np.float32)
    out_snr = np.zeros(n_unique, dtype=np.float32)
    out_gaia_ids = np.zeros(n_unique, dtype=gaia_ids.dtype)
    out_labels = np.zeros((n_unique, n_params), dtype=np.float32)
    out_errors = np.zeros((n_unique, n_params), dtype=np.float32)
    out_flags = np.zeros((n_unique, n_params), dtype=np.uint8)

    n_stacked = 0
    n_highest_snr = 0

    for out_idx, (gid, indices) in enumerate(id_to_indices.items()):
        out_gaia_ids[out_idx] = gid

        if len(indices) == 1:
            # Single observation - just copy
            idx = indices[0]
            out_flux[out_idx] = flux[idx]
            out_ivar[out_idx] = ivar[idx]
            out_snr[out_idx] = snr[idx]
            out_labels[out_idx] = labels[idx]
            out_errors[out_idx] = label_errors[idx]
            out_flags[out_idx] = label_flags[idx]
        else:
            # Multiple observations - check consistency
            obs_flux = flux[indices]
            obs_ivar = ivar[indices]
            obs_snr = snr[indices]
            obs_labels = labels[indices]
            obs_errors = label_errors[indices]
            obs_flags = label_flags[indices]

            weighted_flux, sum_weights = compute_weighted_spectrum(obs_flux, obs_ivar)
            chi2_red = compute_reduced_chi2(obs_flux, obs_ivar, weighted_flux)

            if chi2_red < chi2_threshold:
                # Consistent - stack spectra and labels
                n_stacked += 1
                out_flux[out_idx] = weighted_flux
                out_ivar[out_idx] = sum_weights
                out_snr[out_idx] = np.sqrt(np.sum(obs_snr**2))
                stacked_l, stacked_e, stacked_f = stack_labels(
                    obs_labels, obs_errors, obs_flags
                )
                out_labels[out_idx] = stacked_l
                out_errors[out_idx] = stacked_e
                out_flags[out_idx] = stacked_f
            else:
                # Inconsistent - take highest SNR observation
                n_highest_snr += 1
                best_pos = np.argmax(obs_snr)
                best_idx = indices[best_pos]
                out_flux[out_idx] = flux[best_idx]
                out_ivar[out_idx] = ivar[best_idx]
                out_snr[out_idx] = snr[best_idx]
                out_labels[out_idx] = labels[best_idx]
                out_errors[out_idx] = label_errors[best_idx]
                out_flags[out_idx] = label_flags[best_idx]

    # Write deduplicated survey data
    out_grp = f_out.create_group(f"surveys/{survey}")

    if storage_format == "spectra":
        # LAMOST MRS: write back in spectra format
        n_wave_single = n_wave // 2
        out_spectra = np.zeros((n_unique, 4, n_wave_single), dtype=np.float32)
        out_spectra[:, 0, :] = out_flux[:, :n_wave_single]  # flux_b
        out_spectra[:, 1, :] = out_ivar[:, :n_wave_single]  # ivar_b
        out_spectra[:, 2, :] = out_flux[:, n_wave_single:]  # flux_r
        out_spectra[:, 3, :] = out_ivar[:, n_wave_single:]  # ivar_r
        out_grp.create_dataset("spectra", data=out_spectra, compression="gzip")
        out_grp.create_dataset("wavelength", data=survey_grp["wavelength"][:])
    elif is_dual_arm:
        out_grp.create_dataset(
            "flux_b", data=out_flux[:, :n_wave_b], compression="gzip"
        )
        out_grp.create_dataset(
            "flux_r", data=out_flux[:, n_wave_b:], compression="gzip"
        )
        out_grp.create_dataset(
            "ivar_b", data=out_ivar[:, :n_wave_b], compression="gzip"
        )
        out_grp.create_dataset(
            "ivar_r", data=out_ivar[:, n_wave_b:], compression="gzip"
        )
        out_grp.create_dataset("wavelength_b", data=survey_grp["wavelength_b"][:])
        out_grp.create_dataset("wavelength_r", data=survey_grp["wavelength_r"][:])
    else:
        out_grp.create_dataset("flux", data=out_flux, compression="gzip")
        out_grp.create_dataset("ivar", data=out_ivar, compression="gzip")
        out_grp.create_dataset("wavelength", data=survey_grp["wavelength"][:])

    out_grp.create_dataset("snr", data=out_snr)

    # Copy other survey datasets if present (ra, dec, etc.)
    for key in ["ra", "dec", "targetid"]:
        if key in survey_grp:
            # These need to be reindexed based on id_to_indices
            # For simplicity, we skip them - they can be recomputed from gaia_id
            pass

    # Copy survey attributes and add deduplication info
    for attr_name, attr_val in survey_grp.attrs.items():
        out_grp.attrs[attr_name] = attr_val

    out_grp.attrs["deduplicated"] = True
    out_grp.attrs["chi2_threshold"] = chi2_threshold
    out_grp.attrs["n_stacked"] = n_stacked
    out_grp.attrs["n_highest_snr"] = n_highest_snr
    out_grp.attrs["deduplication_date"] = datetime.now().isoformat()

    # Write deduplicated labels
    out_label_grp = f_out.create_group(f"labels/{label_source}")
    out_label_grp.create_dataset("values", data=out_labels, compression="gzip")
    out_label_grp.create_dataset("errors", data=out_errors, compression="gzip")
    out_label_grp.create_dataset("flags", data=out_flags, compression="gzip")
    out_label_grp.create_dataset("gaia_id", data=out_gaia_ids)

    # Copy label attributes
    for attr_name, attr_val in label_grp.attrs.items():
        out_label_grp.attrs[attr_name] = attr_val

    if verbose:
        if n_with_dups == 0:
            print(f"  {survey}: {n_orig:,} observations (no duplicates)")
        else:
            print(
                f"  {survey}: {n_orig:,} -> {n_unique:,} "
                f"({n_with_dups:,} stars with duplicates, "
                f"{n_stacked} stacked, {n_highest_snr} took highest SNR)"
            )

    return {
        "n_orig": n_orig,
        "n_dedup": n_unique,
        "n_with_dups": n_with_dups,
        "n_stacked": n_stacked,
        "n_highest_snr": n_highest_snr,
        "label_source": label_source,
    }


def create_deduplicated_catalogue(
    input_path: Path,
    output_path: Path,
    surveys: list[str] | None = None,
    chi2_threshold: float = 2.0,
    verbose: bool = True,
) -> None:
    """
    Create a deduplicated copy of the super-catalogue.

    Args:
        input_path: Path to input super-catalogue HDF5.
        output_path: Path to output deduplicated HDF5.
        surveys: List of surveys to deduplicate. If None, process all surveys.
        chi2_threshold: Threshold for reduced chi2 to consider observations consistent.
        verbose: Whether to print progress.
    """
    if verbose:
        print("Creating deduplicated catalogue")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Chi2 threshold: {chi2_threshold}")
        print()

    with h5py.File(input_path, "r") as f_in:
        # Get list of surveys
        available_surveys = list(f_in["surveys"].keys())
        if surveys is None:
            surveys = available_surveys
        else:
            for s in surveys:
                if s not in available_surveys:
                    print(f"Warning: Survey '{s}' not in catalogue, skipping")
            surveys = [s for s in surveys if s in available_surveys]

        if verbose:
            print(f"Processing surveys: {surveys}")
            print()

        with h5py.File(output_path, "w") as f_out:
            # Copy root attributes
            for attr_name, attr_val in f_in.attrs.items():
                f_out.attrs[attr_name] = attr_val

            # Add deduplication metadata
            f_out.attrs["deduplicated"] = True
            f_out.attrs["deduplication_date"] = datetime.now().isoformat()
            f_out.attrs["chi2_threshold"] = chi2_threshold
            f_out.attrs["source_catalogue"] = str(input_path)

            # Copy metadata group (will be outdated but kept for reference)
            if "metadata" in f_in:
                f_in.copy("metadata", f_out)

            # Create empty groups - surveys and labels will be populated by deduplicate_survey
            f_out.create_group("surveys")
            f_out.create_group("labels")

            # Process each survey (this creates surveys/{name} and labels/{label_source})
            print("Deduplicating surveys:")
            stats = {}
            for survey in surveys:
                stats[survey] = deduplicate_survey(
                    f_in, f_out, survey, chi2_threshold, verbose
                )

            # Copy any surveys not in the deduplicate list
            for survey in available_surveys:
                if survey not in surveys:
                    print(f"  {survey}: Copying without deduplication")
                    if survey not in f_out["surveys"]:
                        f_in.copy(f"surveys/{survey}", f_out["surveys"])
                        # Also copy the corresponding labels
                        label_source = SURVEY_LABEL_MAP.get(survey)
                        if (
                            label_source
                            and label_source in f_in["labels"]
                            and label_source not in f_out["labels"]
                        ):
                            f_in.copy(f"labels/{label_source}", f_out["labels"])

    # Summary
    if verbose:
        print()
        print("=" * 60)
        print("DEDUPLICATION SUMMARY")
        print("=" * 60)
        total_orig = sum(
            s.get("n_orig", 0) for s in stats.values() if not s.get("skipped")
        )
        total_dedup = sum(
            s.get("n_dedup", 0) for s in stats.values() if not s.get("skipped")
        )
        total_stacked = sum(s.get("n_stacked", 0) for s in stats.values())
        total_highest = sum(s.get("n_highest_snr", 0) for s in stats.values())

        print(f"Total observations: {total_orig:,} -> {total_dedup:,}")
        print(f"Reduction: {100 * (1 - total_dedup / total_orig):.1f}%")
        print(f"Stars stacked: {total_stacked:,}")
        print(f"Stars took highest SNR: {total_highest:,}")
        print()
        print(f"Output written to: {output_path}")
        print()
        print("Update your config to use the clean catalogue:")
        print("  data:")
        print(f"    catalogue_path: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a deduplicated super-catalogue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deduplicate all surveys with default settings
  python scripts/create_deduplicated_catalogue.py data/super_catalogue.h5 data/super_catalogue_clean.h5

  # Deduplicate specific surveys
  python scripts/create_deduplicated_catalogue.py data/super_catalogue.h5 data/super_catalogue_clean.h5 \\
      --surveys lamost_lrs lamost_mrs

  # Use custom chi2 threshold
  python scripts/create_deduplicated_catalogue.py data/super_catalogue.h5 data/super_catalogue_clean.h5 \\
      --chi2-threshold 3.0
""",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input super-catalogue HDF5 file",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output deduplicated HDF5 file",
    )
    parser.add_argument(
        "--surveys",
        nargs="+",
        help="Surveys to deduplicate (default: all surveys)",
    )
    parser.add_argument(
        "--chi2-threshold",
        type=float,
        default=2.0,
        help="Chi2 threshold for stacking (default: 2.0)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    if args.output.exists():
        response = input(f"Output file {args.output} exists. Overwrite? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return 1

    create_deduplicated_catalogue(
        input_path=args.input,
        output_path=args.output,
        surveys=args.surveys,
        chi2_threshold=args.chi2_threshold,
        verbose=not args.quiet,
    )

    return 0


if __name__ == "__main__":
    exit(main())
