"""Synthetic v2 super-catalogues for end-to-end tests.

The layout matches what ``scripts/build_catalogue_v2.py`` writes, including
LAMOST MRS as padded dual-arm ``spectra`` with ``wavelength_b`` /
``wavelength_r``. Labels are learnable functions of a latent vector that
also generates the spectra.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

PARAMS = ["teff", "logg", "fe_h", "mg_h"]
SURVEY_WAVES = {"boss": 40, "desi": 30}
MRS_ARMS = (12, 16)  # blue, red wavelength bins
COVERAGE = {"boss": 0.7, "desi": 0.4, "lamost_mrs": 0.3}


def _labels(rng, latent, noise):
    n = len(latent)
    values = np.zeros((n, len(PARAMS)), dtype=np.float32)
    values[:, 0] = 5000 + 700 * latent[:, 0]
    values[:, 1] = 3 + 0.7 * latent[:, 1]
    values[:, 2] = 0.3 * latent[:, 2]
    values[:, 3] = 0.2 * latent[:, 0] - 0.1 * latent[:, 2]
    errors = np.full_like(values, noise)
    errors[:, 0] = 50.0
    values += rng.normal(size=values.shape).astype(np.float32) * errors
    return values, errors


def build_catalogue(
    path: Path,
    n_stars: int = 240,
    seed: int = 0,
    apogee_fraction: float = 0.85,
    galah_fraction: float = 0.45,
) -> Path:
    """Write a synthetic v2 catalogue and return its path."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_stars, 3)).astype(np.float32)
    has = {s: rng.uniform(size=n_stars) < f for s, f in COVERAGE.items()}
    has["boss"] |= ~(has["boss"] | has["desi"] | has["lamost_mrs"])

    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        meta.create_dataset("gaia_id", data=np.arange(1, n_stars + 1, dtype=np.int64))
        meta.create_dataset("ra", data=rng.uniform(0, 360, n_stars))
        meta.create_dataset("dec", data=rng.uniform(-90, 90, n_stars))

        surveys = f.create_group("surveys")
        for survey, n_wave in SURVEY_WAVES.items():
            basis = rng.normal(size=(3, n_wave)).astype(np.float32)
            flux = (latent @ basis + 10.0) * rng.uniform(0.5, 2.0, (n_stars, 1))
            flux = flux.astype(np.float32)
            ivar = np.full_like(flux, 50.0)
            flux[~has[survey]] = 0.0
            ivar[~has[survey]] = 0.0
            ivar[:, :2] = 0.0  # always-masked pixels
            grp = surveys.create_group(survey)
            grp.create_dataset("flux", data=flux)
            grp.create_dataset("ivar", data=ivar)
            grp.create_dataset("wavelength", data=np.linspace(3600, 9000, n_wave))
            grp.create_dataset("snr", data=np.where(has[survey], 20.0, 0.0))

        n_b, n_r = MRS_ARMS
        spectra = np.zeros((n_stars, 4, max(n_b, n_r)), dtype=np.float32)
        for n_wave, offset in [(n_b, 0), (n_r, 2)]:
            basis = rng.normal(size=(3, n_wave)).astype(np.float32)
            spectra[:, offset, :n_wave] = latent @ basis + 5.0
            spectra[:, offset + 1, :n_wave] = 30.0
        spectra[~has["lamost_mrs"]] = 0.0
        mrs = surveys.create_group("lamost_mrs")
        mrs.create_dataset("spectra", data=spectra)
        mrs.create_dataset("wavelength_b", data=np.linspace(4950, 5350, n_b))
        mrs.create_dataset("wavelength_r", data=np.linspace(6300, 6800, n_r))
        mrs.create_dataset("snr", data=np.where(has["lamost_mrs"], 15.0, 0.0))

        labels = f.create_group("labels")
        for source, frac, noise in [
            ("apogee", apogee_fraction, 0.03),
            ("galah", galah_fraction, 0.06),
        ]:
            values, errors = _labels(rng, latent, noise)
            missing = rng.uniform(size=n_stars) >= frac
            values[missing] = 0.0
            errors[missing] = 0.0
            grp = labels.create_group(source)
            grp.create_dataset("values", data=values)
            grp.create_dataset("errors", data=errors)
            grp.create_dataset("flags", data=np.zeros(values.shape, dtype=np.uint8))
            grp.create_dataset("gaia_id", data=np.arange(1, n_stars + 1))

        f.attrs["n_stars"] = n_stars
        f.attrs["parameter_names"] = PARAMS
        f.attrs["survey_names"] = list(SURVEY_WAVES) + ["lamost_mrs"]
        f.attrs["label_sources"] = ["apogee", "galah"]
        f.attrs["creation_date"] = "2026-01-01"
        f.attrs["version"] = "2.0.0"
    return path


def write_config(
    path: Path,
    catalogue: Path,
    output_dir: Path,
    surveys: list[str],
    label_sources: list[str],
    name: str = "e2e",
    epochs: int = 3,
    extra_data: str = "",
    extra_training: str = "",
) -> Path:
    """Write a small training config YAML for the synthetic catalogue."""
    model_block = (
        "multi_head_model:\n"
        "  latent_dim: 16\n"
        "  encoder_hidden: [32]\n"
        "  trunk_hidden: [32]\n"
        "  output_hidden: [16]\n"
        if len(surveys) > 1
        else "model:\n  hidden_layers: [32, 16]\n"
    )
    path.write_text(
        f"name: {name}\n"
        f"output_dir: {output_dir}\n"
        "device: cpu\n"
        "data:\n"
        f"  catalogue_path: {catalogue}\n"
        f"  surveys: {surveys}\n"
        f"  label_sources: {label_sources}\n"
        "  max_flag_bits: -1\n"
        f"{extra_data}"
        f"{model_block}"
        "training:\n"
        f"  epochs: {epochs}\n"
        "  batch_size: 32\n"
        "  learning_rate: 0.003\n"
        f"{extra_training}"
    )
    return path
