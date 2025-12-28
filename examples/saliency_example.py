#!/usr/bin/env python
"""
Example: Saliency Analysis for Stellar Parameter Predictions.

This script demonstrates how to compute and visualize gradient-based saliency
maps showing which wavelengths are most important for predicting each stellar
parameter.

Prerequisites:
    - A trained model checkpoint (e.g., from variant5 training)
    - Input spectrum data from the super-catalogue (or use synthetic data for demo)

Usage:
    # Using real data from variant5 checkpoint
    python examples/saliency_example.py \
        --checkpoint outputs/variant5_all_surveys_masked/variant5_all_surveys_masked_final \
        --catalogue data/super_catalogue_clean.h5 \
        --survey boss \
        --index 0

    # Using synthetic data (demo mode)
    python examples/saliency_example.py --demo
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch
import yaml

from dorothy.analysis import (
    SaliencyAnalyzer,
    plot_parameter_saliency,
    plot_saliency_heatmap,
)
from dorothy.data.normalizer import LabelNormalizer
from dorothy.models import MultiHeadMLP


if TYPE_CHECKING:
    pass


def create_demo_model() -> MultiHeadMLP:
    """Create a demo model for testing without a checkpoint."""
    return MultiHeadMLP(
        survey_configs={"boss": 4506, "desi": 7650},
        n_parameters=11,
        latent_dim=64,
        encoder_hidden=[256, 128],
        trunk_hidden=[128, 64],
        output_hidden=[32],
    )


def create_synthetic_spectrum(
    n_wavelengths: int = 4506,
) -> tuple[torch.Tensor, np.ndarray]:
    """Create a synthetic spectrum for demonstration.

    Returns:
        Tuple of (X, wavelength) where X is shape (3, n_wavelengths).
    """
    wavelength = np.linspace(3600, 9000, n_wavelengths)

    # Base continuum
    continuum = 1.0 - 0.0001 * (wavelength - 6000)

    # Add some absorption lines
    flux = continuum.copy()
    line_centers = [4340, 4861, 6563, 8542]  # H lines
    line_depths = [0.3, 0.4, 0.5, 0.3]
    line_widths = [20, 25, 30, 25]

    for center, depth, width in zip(
        line_centers, line_depths, line_widths, strict=False
    ):
        flux -= depth * np.exp(-((wavelength - center) ** 2) / (2 * width**2))

    # Add noise
    noise_level = 0.02
    flux += np.random.randn(n_wavelengths) * noise_level

    # Error spectrum
    error = np.ones(n_wavelengths) * noise_level

    # Mask (all valid)
    mask = np.ones(n_wavelengths)

    X = torch.tensor(np.stack([flux, error, mask], axis=0), dtype=torch.float32)

    return X, wavelength.astype(np.float32)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: str = "cpu",
) -> tuple[MultiHeadMLP, LabelNormalizer | None]:
    """Load MultiHeadMLP model from checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory.
        device: Device to load model on.

    Returns:
        Tuple of (model, normalizer).
    """
    # Load config (use unsafe_load for Python objects like tuples)
    config_path = checkpoint_path / "config.yaml"
    with open(config_path) as f:
        config = yaml.unsafe_load(f)

    # Extract model config
    model_config = config["multi_head_model"]

    # Create model
    model = MultiHeadMLP(
        survey_configs=model_config["survey_wavelengths"],
        n_parameters=model_config["n_parameters"],
        latent_dim=model_config["latent_dim"],
        encoder_hidden=model_config["encoder_hidden"],
        trunk_hidden=model_config["trunk_hidden"],
        output_hidden=model_config["output_hidden"],
        combination_mode=model_config.get("combination_mode", "concat"),
        normalization=model_config.get("normalization", "batchnorm"),
        activation=model_config.get("activation", "relu"),
        dropout=model_config.get("dropout", 0.0),
    )

    # Load weights
    model_path = checkpoint_path / "best_model.pth"
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Load normalizer if available
    normalizer = None
    normalizer_path = checkpoint_path / "normalizer.pkl"
    if normalizer_path.exists():
        normalizer = LabelNormalizer.load(normalizer_path)

    return model, normalizer


def load_spectrum_from_catalogue(
    catalogue_path: Path,
    survey: str,
    index: int,
) -> tuple[torch.Tensor, np.ndarray, int]:
    """Load a spectrum from the super-catalogue.

    Args:
        catalogue_path: Path to HDF5 catalogue.
        survey: Survey name (e.g., "boss", "desi").
        index: Index of spectrum to load.

    Returns:
        Tuple of (X, wavelength, gaia_id).
    """
    with h5py.File(catalogue_path, "r") as f:
        # Load spectrum data
        flux = f[f"surveys/{survey}/flux"][index]
        ivar = f[f"surveys/{survey}/ivar"][index]
        wavelength = f[f"surveys/{survey}/wavelength"][:].astype(np.float32)

        # Convert ivar to error and mask
        error = np.where(ivar > 0, 1.0 / np.sqrt(ivar), 0.0).astype(np.float32)
        mask = (ivar > 0).astype(np.float32)

        # Get gaia_id from the aligned labels (spectra and labels are 1:1 by index)
        label_key = f"apogee_{survey}"
        if f"labels/{label_key}/gaia_id" in f:
            gaia_id = int(f[f"labels/{label_key}/gaia_id"][index])
        else:
            # Fallback to metadata (for backwards compatibility)
            gaia_id = int(f["metadata/gaia_id"][index])

    # Stack into input tensor
    X = torch.tensor(np.stack([flux, error, mask], axis=0), dtype=torch.float32)

    return X, wavelength, gaia_id


def load_labels_from_catalogue(
    catalogue_path: Path,
    survey: str,
    gaia_id: int,
    index: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Load labels for a given Gaia ID from the super-catalogue.

    Args:
        catalogue_path: Path to HDF5 catalogue.
        survey: Survey name (e.g., "boss", "desi").
        gaia_id: Gaia DR3 source ID to look up.
        index: Optional direct index (spectra and labels are aligned 1:1).

    Returns:
        Tuple of (label_values, label_errors) or (None, None) if not found.
    """
    # Labels are stored as apogee_{survey}
    label_key = f"apogee_{survey}"

    with h5py.File(catalogue_path, "r") as f:
        if f"labels/{label_key}" not in f:
            return None, None

        # If index provided, use it directly (spectra and labels are aligned)
        if index is not None:
            label_values = f[f"labels/{label_key}/values"][index]
            label_errors = f[f"labels/{label_key}/errors"][index]
            return label_values.astype(np.float32), label_errors.astype(np.float32)

        # Otherwise search by gaia_id
        label_gaia_ids = f[f"labels/{label_key}/gaia_id"][:]
        label_values = f[f"labels/{label_key}/values"][:]
        label_errors = f[f"labels/{label_key}/errors"][:]

    # Find index for this gaia_id
    matches = np.where(label_gaia_ids == gaia_id)[0]
    if len(matches) == 0:
        return None, None

    idx = matches[0]
    return label_values[idx].astype(np.float32), label_errors[idx].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Compute and visualize saliency maps for stellar parameters"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained model checkpoint directory",
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=None,
        help="Path to HDF5 super-catalogue for real data",
    )
    parser.add_argument(
        "--survey",
        type=str,
        default="boss",
        help="Survey name for the input spectrum",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of spectrum in catalogue to analyze",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use demo mode with synthetic data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/saliency"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for computation (auto, cuda, cpu)",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load model and normalizer
    normalizer = None
    if args.checkpoint is not None and not args.demo:
        print(f"Loading model from {args.checkpoint}")
        model, normalizer = load_model_from_checkpoint(args.checkpoint, device)
        print(f"Model loaded: surveys={model.survey_names}")
        if normalizer is not None:
            print("Normalizer loaded for denormalized predictions")
    else:
        print("Using demo model (no checkpoint provided or --demo specified)")
        model = create_demo_model()

    # Create analyzer
    analyzer = SaliencyAnalyzer(model, device=device)
    print(f"Saliency analyzer created on device: {analyzer.device}")

    # Load spectrum data
    gaia_id = None

    # Load spectrum data and labels
    label_values = None
    label_errors = None

    if args.catalogue is not None and not args.demo:
        print(
            f"Loading spectrum from {args.catalogue} (survey={args.survey}, index={args.index})"
        )
        X, wavelength, gaia_id = load_spectrum_from_catalogue(
            args.catalogue, args.survey, args.index
        )
        print(f"  Gaia DR3 {gaia_id}")

        # Load labels for this object (use index since spectra and labels are aligned)
        label_values, label_errors = load_labels_from_catalogue(
            args.catalogue, args.survey, gaia_id, index=args.index
        )
        if label_values is not None:
            print("  Labels loaded from catalogue")
        else:
            print("  No labels found for this object")
    else:
        # Get wavelength count for synthetic data
        if hasattr(model, "encoders") and args.survey in model.encoders:
            n_wavelengths = model.encoders[args.survey].n_wavelengths
        else:
            n_wavelengths = 4506
        print(f"Creating synthetic spectrum with {n_wavelengths} wavelengths")
        X, wavelength = create_synthetic_spectrum(n_wavelengths)

    print(
        f"Spectrum shape: {X.shape}, wavelength range: {wavelength.min():.0f}-{wavelength.max():.0f} Å"
    )

    # Compute saliency
    print("Computing saliency map...")
    result = analyzer.compute_saliency(
        X, wavelength, survey=args.survey, gaia_id=gaia_id
    )

    # Print predictions
    print("\nPredictions (normalized):")
    for name, pred, unc in zip(
        result.parameter_names, result.predictions, result.uncertainties, strict=False
    ):
        print(f"  {name}: {pred:.4f} ± {unc:.4f}")

    if normalizer is not None:
        pred_denorm, unc_denorm = normalizer.inverse_transform(
            result.predictions.reshape(1, -1),
            result.uncertainties.reshape(1, -1),
        )
        print("\nPredictions (physical units):")
        for i, name in enumerate(result.parameter_names):
            if name == "teff":
                print(f"  {name}: {pred_denorm[0, i]:.0f} ± {unc_denorm[0, i]:.0f} K")
            else:
                print(f"  {name}: {pred_denorm[0, i]:.4f} ± {unc_denorm[0, i]:.4f}")

    # Plot per-parameter saliency for key parameters
    key_params = ["teff", "logg", "fe_h", "mg_fe"]

    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    for param in key_params:
        output_path = args.output_dir / f"saliency_{param}.png"
        print(f"Plotting {param} saliency -> {output_path}")
        fig = plot_parameter_saliency(
            result,
            param,
            output_path=output_path,
            normalizer=normalizer,
            label_values=label_values,
            label_errors=label_errors,
        )
        plt.close(fig)

    # Plot heatmap
    heatmap_path = args.output_dir / "saliency_heatmap.png"
    print(f"Plotting heatmap -> {heatmap_path}")
    fig = plot_saliency_heatmap(result, output_path=heatmap_path)
    plt.close(fig)

    print(f"\nSaliency analysis complete! Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
