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
    from dorothy.data import CatalogueLoader
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

    # Handle Path conversion for catalogue_path
    if "data" in config_dict and "catalogue_path" in config_dict["data"]:
        config_dict["data"]["catalogue_path"] = Path(
            config_dict["data"]["catalogue_path"]
        )

    # Override device if specified
    if args.device != "auto":
        config_dict["device"] = args.device

    # Override output directory if specified
    if args.output_dir is not None:
        config_dict["output_dir"] = args.output_dir

    # Auto-populate multi_head_model.survey_wavelengths for multi-survey training
    data_config = config_dict.get("data", {})
    surveys = data_config.get("surveys", [])
    label_sources = data_config.get("label_sources", ["apogee"])

    if len(surveys) > 1:
        # Multi-survey training requires multi_head_model with survey_wavelengths
        catalogue_path = data_config.get("catalogue_path")
        if catalogue_path is None:
            print("Error: catalogue_path is required for multi-survey training")
            return 1

        # Get wavelength counts from catalogue
        print("Querying catalogue for survey wavelengths...")
        loader = CatalogueLoader(catalogue_path)
        try:
            survey_wavelengths = loader.get_survey_wavelength_counts(surveys)
        except Exception as e:
            print(f"Error: Could not get survey wavelengths: {e}")
            return 1

        print(f"  Survey wavelengths: {survey_wavelengths}")

        # Create or update multi_head_model config
        if "multi_head_model" not in config_dict:
            config_dict["multi_head_model"] = {}

        # Always set survey_wavelengths from catalogue (authoritative source)
        config_dict["multi_head_model"]["survey_wavelengths"] = survey_wavelengths

        # Set label_sources for multi-label training
        if len(label_sources) > 1:
            config_dict["multi_head_model"]["label_sources"] = label_sources

    # Create config object
    try:
        config = ExperimentConfig(**config_dict)
    except Exception as e:
        print(f"Error: Invalid configuration: {e}")
        return 1

    print(f"Experiment: {config.name}")
    print(f"Device: {config.device}")
    print(f"Output: {config.get_output_path()}")
    print(f"Surveys: {config.data.surveys}")
    print(f"Label sources: {config.data.label_sources}")

    # Load data from super-catalogue
    print(f"\nLoading data from {config.data.catalogue_path}")
    loader = CatalogueLoader(config.data.catalogue_path)

    import numpy as np

    if config.data.is_multi_survey:
        # Determine if multi-labelset training
        is_multi_label = config.data.is_multi_label

        if is_multi_label:
            # Multi-labelset mode: load labels from multiple sources
            # Determine which label sources to actually load from catalogue
            # (excluding duplicated ones that will be copied from source)
            duplicate_labels = config.data.duplicate_labels or {}
            sources_to_load = [
                s for s in config.data.label_sources if s not in duplicate_labels
            ]

            X_dict, y_dict, has_data_dict, has_labels_dict = (
                loader.load_merged_for_training(
                    surveys=config.data.surveys,
                    smart_deduplicate=config.data.smart_deduplicate,
                    chi2_threshold=config.data.chi2_threshold,
                    label_source=sources_to_load[0],
                    max_flag_bits=config.data.max_flag_bits,
                    label_sources=sources_to_load,
                )
            )

            # Handle duplicate_labels: copy labels from source to target
            if duplicate_labels:
                print(f"  Duplicating labels: {duplicate_labels}")
                for target, source in duplicate_labels.items():
                    if source not in y_dict:
                        print(
                            f"Error: Cannot duplicate from '{source}' - not in loaded sources"
                        )
                        return 1
                    y_dict[target] = y_dict[source].copy()
                    has_labels_dict[target] = has_labels_dict[source].copy()

            print(f"  Surveys loaded: {list(X_dict.keys())}")
            for survey_name, X_survey in X_dict.items():
                n_with_data = has_data_dict[survey_name].sum()
                print(
                    f"    {survey_name}: {X_survey.shape[2]} wavelengths, "
                    f"{n_with_data:,} stars"
                )

            print(f"  Label sources: {config.data.label_sources}")
            first_source = config.data.label_sources[0]
            n_total = y_dict[first_source].shape[0]
            print(f"  Total unique stars: {n_total:,}")

            for source in config.data.label_sources:
                n_with_labels = has_labels_dict[source].sum()
                print(f"    {source}: {n_with_labels:,} stars")

            # Filter to samples with at least one survey AND at least one label source
            has_any_survey = np.zeros(n_total, dtype=bool)
            for survey_name in X_dict:
                has_any_survey |= has_data_dict[survey_name]

            has_any_labels = np.zeros(n_total, dtype=bool)
            for source in config.data.label_sources:
                has_any_labels |= has_labels_dict[source]

            valid_samples = has_any_survey & has_any_labels
            print(
                f"  Valid samples (at least one survey + one label source): "
                f"{valid_samples.sum():,}"
            )

            # Filter all arrays
            X_dict = {survey: arr[valid_samples] for survey, arr in X_dict.items()}
            y_dict = {source: arr[valid_samples] for source, arr in y_dict.items()}
            has_data_dict = {
                survey: arr[valid_samples] for survey, arr in has_data_dict.items()
            }
            has_labels_dict = {
                source: arr[valid_samples] for source, arr in has_labels_dict.items()
            }

            # Split data
            n_samples = valid_samples.sum()
            indices = np.random.RandomState(config.seed).permutation(n_samples)

            n_train = int(n_samples * config.data.train_ratio)
            n_val = int(n_samples * config.data.val_ratio)

            train_idx = indices[:n_train]
            val_idx = indices[n_train : n_train + n_val]
            test_idx = indices[n_train + n_val :]

            X_train = {survey: arr[train_idx] for survey, arr in X_dict.items()}
            X_val = {survey: arr[val_idx] for survey, arr in X_dict.items()}
            y_train = {source: arr[train_idx] for source, arr in y_dict.items()}
            y_val = {source: arr[val_idx] for source, arr in y_dict.items()}
            has_data_train = {
                survey: arr[train_idx] for survey, arr in has_data_dict.items()
            }
            has_data_val = {
                survey: arr[val_idx] for survey, arr in has_data_dict.items()
            }
            has_labels_train = {
                source: arr[train_idx] for source, arr in has_labels_dict.items()
            }
            has_labels_val = {
                source: arr[val_idx] for source, arr in has_labels_dict.items()
            }

            print(f"\n  Training samples: {len(train_idx):,}")
            print(f"  Validation samples: {len(val_idx):,}")
            print(f"  Test samples: {len(test_idx):,}")

            # Create trainer (uses MultiHeadMLP with multi-label support)
            print("\nInitializing trainer...")
            trainer = Trainer(config)

            print(f"  Model parameters: {trainer.model.count_parameters():,}")
            print("  Architecture: MultiHeadMLP (multi-labelset)")
            print(
                f"  Combination mode: {config.multi_head_model.combination_mode.value}"
            )
            print(f"  Output heads: {config.multi_head_model.label_sources}")

            # Train with multi-survey + multi-labelset data
            print("\nStarting multi-labelset training...")
            history = trainer.fit_multi_labelset(
                X_train,
                y_train,
                X_val,
                y_val,
                has_data_train,
                has_data_val,
                has_labels_train,
                has_labels_val,
            )

        else:
            # Single-label mode: use existing multi-survey code
            X_dict, y, has_data_dict, _ = loader.load_merged_for_training(
                surveys=config.data.surveys,
                smart_deduplicate=config.data.smart_deduplicate,
                chi2_threshold=config.data.chi2_threshold,
                label_source=config.data.label_sources[0],
                max_flag_bits=config.data.max_flag_bits,
            )

            print(f"  Surveys loaded: {list(X_dict.keys())}")
            for survey_name, X_survey in X_dict.items():
                n_with_data = has_data_dict[survey_name].sum()
                print(
                    f"    {survey_name}: {X_survey.shape[2]} wavelengths, "
                    f"{n_with_data:,} stars"
                )
            print(f"  Total unique stars: {y.shape[0]:,}")
            print(f"  Parameters: {y.shape[2]}")

            # Filter to samples with at least one survey having data AND all valid labels
            has_any_survey = np.zeros(y.shape[0], dtype=bool)
            for survey_name in X_dict:
                has_any_survey |= has_data_dict[survey_name]

            label_mask = y[:, 2, :]  # (N, n_params)
            valid_labels = label_mask.all(axis=1)

            valid_samples = has_any_survey & valid_labels
            print(
                f"  Valid samples (at least one survey + all labels): "
                f"{valid_samples.sum():,}"
            )

            # Filter all arrays
            X_dict = {survey: arr[valid_samples] for survey, arr in X_dict.items()}
            y = y[valid_samples]
            has_data_dict = {
                survey: arr[valid_samples] for survey, arr in has_data_dict.items()
            }

            # Split data
            n_samples = y.shape[0]
            indices = np.random.RandomState(config.seed).permutation(n_samples)

            n_train = int(n_samples * config.data.train_ratio)
            n_val = int(n_samples * config.data.val_ratio)

            train_idx = indices[:n_train]
            val_idx = indices[n_train : n_train + n_val]
            test_idx = indices[n_train + n_val :]

            X_train = {survey: arr[train_idx] for survey, arr in X_dict.items()}
            X_val = {survey: arr[val_idx] for survey, arr in X_dict.items()}
            y_train, y_val = y[train_idx], y[val_idx]
            has_data_train = {
                survey: arr[train_idx] for survey, arr in has_data_dict.items()
            }
            has_data_val = {
                survey: arr[val_idx] for survey, arr in has_data_dict.items()
            }

            print(f"\n  Training samples: {len(train_idx):,}")
            print(f"  Validation samples: {len(val_idx):,}")
            print(f"  Test samples: {len(test_idx):,}")

            # Create trainer and train (uses MultiHeadMLP via config.is_multi_head)
            print("\nInitializing trainer...")
            trainer = Trainer(config)

            print(f"  Model parameters: {trainer.model.count_parameters():,}")
            print("  Architecture: MultiHeadMLP")
            print(
                f"  Combination mode: {config.multi_head_model.combination_mode.value}"
            )

            # Train with multi-survey data
            print("\nStarting multi-survey training...")
            history = trainer.fit_multi_survey(
                X_train, y_train, X_val, y_val, has_data_train, has_data_val
            )

        print("\nTraining complete!")
        print(f"  Best validation loss: {history.best_val_loss:.6f}")
        print(f"  Best epoch: {history.best_epoch + 1}")

        # Save checkpoint
        print("\nSaving checkpoint...")
        checkpoint_path = trainer.save_checkpoint()
        print(f"  Saved to: {checkpoint_path}")

        # Generate training report plots
        print("\nGenerating training plots...")
        try:
            from dorothy.visualization import generate_training_report

            plots_dir = checkpoint_path / "plots"
            plots = generate_training_report(
                history=history,
                output_dir=plots_dir,
                experiment_name=config.name,
            )
            if plots:
                print(f"  Generated {len(plots)} plots in: {plots_dir}")
                for p in plots:
                    print(f"    - {p.name}")
            else:
                print("  No plots generated (matplotlib may not be installed)")
        except ImportError:
            print("  Skipping plots (matplotlib not available)")
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")

    else:
        # Single survey: use load_for_training
        # Returns 3-channel format: X=(N, 3, wavelengths), y=(N, 3, n_params)
        survey = config.data.surveys[0]
        X, y = loader.load_for_training(
            survey=survey,
            max_flag_bits=config.data.max_flag_bits,
        )

        print(f"  Survey: {survey}")
        print(f"  Spectra: {X.shape[0]:,}")
        print(f"  Wavelength bins: {X.shape[2]}")
        print(f"  Parameters: {y.shape[2]}")

        # Filter to samples with all valid labels
        # Mask is in channel 2 of y: y[:, 2, :] is (N, n_params)
        mask = y[:, 2, :]  # (N, n_params)
        valid_samples = mask.all(axis=1)
        print(f"  Valid samples (all labels): {valid_samples.sum():,}")

        X = X[valid_samples]
        y = y[valid_samples]

        # Split data
        n_samples = X.shape[0]
        indices = np.random.RandomState(config.seed).permutation(n_samples)

        n_train = int(n_samples * config.data.train_ratio)
        n_val = int(n_samples * config.data.val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        print(f"\n  Training samples: {X_train.shape[0]:,}")
        print(f"  Validation samples: {X_val.shape[0]:,}")
        print(f"  Test samples: {len(test_idx):,}")

        # X is already 3-channel: (N, 3, wavelengths) -> flattened to (N, 3*wavelengths)
        # for MLP input
        input_features = X_train.shape[1] * X_train.shape[2]  # 3 * wavelengths

        # Update model input features based on actual data shape
        # Only needed for standard MLP (multi-head gets wavelengths from survey_wavelengths)
        if config.model is not None:
            config.model = config.model.model_copy(
                update={"input_features": input_features}
            )

        # Create trainer and train
        print("\nInitializing trainer...")
        trainer = Trainer(config)

        print(f"  Model parameters: {trainer.model.count_parameters():,}")
        if config.model is not None:
            print(f"  Input features: {config.model.input_features}")
        else:
            print("  Architecture: MultiHeadMLP")

        # Pass 3-channel data directly to trainer
        # X: (N, 3, wavelengths), y: (N, 3, n_params)
        print("\nStarting training...")
        history = trainer.fit(X_train, y_train, X_val, y_val)

        print("\nTraining complete!")
        print(f"  Best validation loss: {history.best_val_loss:.6f}")
        print(f"  Best epoch: {history.best_epoch + 1}")

        # Save checkpoint
        print("\nSaving checkpoint...")
        checkpoint_path = trainer.save_checkpoint()
        print(f"  Saved to: {checkpoint_path}")

        # Generate training report plots
        print("\nGenerating training plots...")
        try:
            from dorothy.visualization import generate_training_report

            plots_dir = checkpoint_path / "plots"
            plots = generate_training_report(
                history=history,
                output_dir=plots_dir,
                experiment_name=config.name,
            )
            if plots:
                print(f"  Generated {len(plots)} plots in: {plots_dir}")
                for p in plots:
                    print(f"    - {p.name}")
            else:
                print("  No plots generated (matplotlib may not be installed)")
        except ImportError:
            print("  Skipping plots (matplotlib not available)")
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")

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
