"""
Command-line interface for DOROTHY.

This module provides the main CLI entry point for training models,
making predictions, and managing experiments.

Commands:
    train     - Train a model from a configuration file
    predict   - Make predictions on new spectral data
    info      - Display information about a checkpoint

Example:
    $ dorothy train config.yaml
    $ dorothy predict --checkpoint ./outputs/my_exp --input spectra.fits --output predictions.csv
    $ dorothy info ./outputs/my_exp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="dorothy",
        description="DOROTHY: Deep learning for stellar parameter inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        description="Available commands",
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model from configuration",
        description="Train a DOROTHY model using the specified configuration file.",
    )
    train_parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training (default: auto)",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory from config",
    )
    train_parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from checkpoint",
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Make predictions on new data",
        description="Use a trained model to predict stellar parameters.",
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint directory",
    )
    predict_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input FITS file",
    )
    predict_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output CSV file (default: predictions.csv)",
    )
    predict_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference (default: auto)",
    )
    predict_parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for predictions (default: 1024)",
    )
    predict_parser.add_argument(
        "--denormalize",
        action="store_true",
        help="Output predictions in physical units",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Display checkpoint information",
        description="Show information about a trained model checkpoint.",
    )
    info_parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to checkpoint directory",
    )

    return parser


def cmd_train(args: argparse.Namespace) -> int:
    """Execute the train command."""
    import yaml

    from dorothy.config.schema import ExperimentConfig
    from dorothy.data import FITSLoader, split_data
    from dorothy.training import Trainer

    # Load configuration
    config_path = args.config
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return 1

    print(f"Loading configuration from {config_path}")
    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration: {e}")
        return 1

    # Handle Path conversion for fits_path
    if "data" in config_dict and "fits_path" in config_dict["data"]:
        config_dict["data"]["fits_path"] = Path(config_dict["data"]["fits_path"])

    # Override device if specified
    if args.device != "auto":
        config_dict["device"] = args.device

    # Override output directory if specified
    if args.output_dir is not None:
        config_dict["output_dir"] = args.output_dir

    # Create config object
    try:
        config = ExperimentConfig(**config_dict)
    except Exception as e:
        print(f"Error: Invalid configuration: {e}")
        return 1

    print(f"Experiment: {config.name}")
    print(f"Device: {config.device}")
    print(f"Output: {config.get_output_path()}")

    # Load data
    print(f"\nLoading data from {config.data.fits_path}")
    loader = FITSLoader.from_config(config.data)
    data = loader.load()

    print(f"  Total spectra: {data.n_samples}")
    print(f"  Good quality: {data.quality_mask.sum()}")
    print(f"  Wavelength bins: {data.n_wavelengths}")

    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
        data,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        seed=config.seed,
    )

    print(f"\n  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")

    # Create trainer and train
    print("\nInitializing trainer...")
    trainer = Trainer(config)

    print(f"  Model parameters: {trainer.model.count_parameters():,}")

    print("\nStarting training...")
    history = trainer.fit(X_train, y_train, X_val, y_val)

    print("\nTraining complete!")
    print(f"  Best validation loss: {history.best_val_loss:.6f}")
    print(f"  Best epoch: {history.best_epoch + 1}")

    # Save checkpoint
    print("\nSaving checkpoint...")
    checkpoint_path = trainer.save_checkpoint()
    print(f"  Saved to: {checkpoint_path}")

    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Execute the predict command."""
    from dorothy.inference import predict_from_fits

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1

    input_path = args.input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    output_path = args.output
    if output_path is None:
        output_path = Path("predictions.csv")

    print(f"Loading model from {checkpoint_path}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Device: {args.device}")

    try:
        predict_from_fits(
            fits_path=input_path,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            device=args.device,
        )
        print(f"\nPredictions saved to {output_path}")
        return 0
    except Exception as e:
        print(f"Error: Prediction failed: {e}")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command."""
    import pickle

    import torch

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1

    print(f"Checkpoint: {checkpoint_path}")
    print("-" * 50)

    # Check for model files
    model_files = list(checkpoint_path.glob("*.pth"))
    if model_files:
        print("\nModel files:")
        for f in model_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.2f} MB")

            # Try to infer architecture
            if f.name == "best_model.pth":
                try:
                    state_dict = torch.load(f, map_location="cpu", weights_only=True)
                    # Count parameters
                    n_params = sum(p.numel() for p in state_dict.values())
                    print(f"    Parameters: {n_params:,}")

                    # Find architecture
                    linear_weights = [
                        k
                        for k, v in state_dict.items()
                        if "weight" in k and v.dim() == 2
                    ]
                    if linear_weights:
                        first_shape = state_dict[linear_weights[0]].shape
                        last_shape = state_dict[linear_weights[-1]].shape
                        print(f"    Input features: {first_shape[1]}")
                        print(f"    Output features: {last_shape[0]}")
                except Exception as e:
                    print(f"    Error loading model: {e}")

    # Check for history
    history_file = checkpoint_path / "history_train_val.pkl"
    if history_file.exists():
        try:
            with open(history_file, "rb") as f:
                history = pickle.load(f)

            print("\nTraining history:")
            if "history_train" in history:
                train_losses = history["history_train"]
                print(f"  Epochs: {len(train_losses)}")
                print(f"  Final train loss: {train_losses[-1]:.6f}")
            if "history_val" in history:
                val_losses = history["history_val"]
                best_epoch = val_losses.index(min(val_losses))
                print(
                    f"  Best val loss: {min(val_losses):.6f} (epoch {best_epoch + 1})"
                )
        except Exception as e:
            print(f"  Error loading history: {e}")

    # Check for normalizer
    normalizer_file = checkpoint_path / "normalizer.pkl"
    if normalizer_file.exists():
        print("\nNormalizer: present")
        try:
            with open(normalizer_file, "rb") as f:
                params = pickle.load(f)
            print(f"  Parameters: {list(params.keys())}")
        except Exception as e:
            print(f"  Error loading normalizer: {e}")

    # Check for learning rates
    lr_file = checkpoint_path / "learning_rates.pkl"
    if lr_file.exists():
        try:
            with open(lr_file, "rb") as f:
                lrs = pickle.load(f)
            print(f"\nLearning rates: {len(lrs)} recorded steps")
            print(f"  Range: {min(lrs):.2e} - {max(lrs):.2e}")
        except Exception as e:
            print(f"  Error loading learning rates: {e}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the DOROTHY CLI.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "train":
        return cmd_train(args)
    elif args.command == "predict":
        return cmd_predict(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
