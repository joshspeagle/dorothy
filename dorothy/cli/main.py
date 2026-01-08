"""
Command-line interface for DOROTHY.

This module provides the main CLI entry point for training models,
making predictions, and managing experiments.

Commands:
    train     - Train a model from a configuration file
    predict   - Make predictions on new spectral data
    info      - Display information about a checkpoint
    evaluate  - Evaluate model on held-out test set

Example:
    $ dorothy train config.yaml
    $ dorothy predict --checkpoint ./outputs/my_exp --input spectra.fits --output predictions.csv
    $ dorothy info ./outputs/my_exp
    $ dorothy evaluate ./outputs/my_exp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dorothy import __version__


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
        version=f"%(prog)s {__version__}",
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

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model on held-out test set",
        description="Evaluate a trained model on the held-out test set from training.",
    )
    evaluate_parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to checkpoint directory",
    )
    evaluate_parser.add_argument(
        "--model",
        type=str,
        default="best_model.pth",
        help="Model file to load (default: best_model.pth)",
    )
    evaluate_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference (default: auto)",
    )
    evaluate_parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for predictions (default: 1024)",
    )
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save evaluation results (JSON or Markdown)",
    )
    evaluate_parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "markdown", "json"],
        help="Output format (default: text)",
    )

    return parser


def _get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1e6
    except ImportError:
        return 0.0


def _print_memory(label: str) -> None:
    """Print current memory usage with a label."""
    mem = _get_memory_mb()
    if mem > 0:
        print(f"  [Memory] {label}: {mem:.0f} MB ({mem/1024:.2f} GB)")


def _save_data_split(
    checkpoint_path: Path,
    test_idx,
    train_idx,
    val_idx,
    split_mode: str,
    n_total: int,
    valid_indices=None,
) -> None:
    """Save data split indices for later evaluation.

    Args:
        checkpoint_path: Path to checkpoint directory.
        test_idx: Test set indices.
        train_idx: Training set indices.
        val_idx: Validation set indices.
        split_mode: Type of splitting used ('dense_multi_label', 'sparse_multi_survey',
            or 'single_survey').
        n_total: Total number of samples before splitting.
        valid_indices: For sparse mode, the indices of valid samples in the full
            catalogue (test_idx indexes into this).
    """
    import pickle

    import numpy as np

    split_info = {
        "test_idx": np.array(test_idx),
        "train_idx": np.array(train_idx),
        "val_idx": np.array(val_idx),
        "split_mode": split_mode,
        "n_total": n_total,
    }

    # For sparse mode, we need valid_indices to map back to catalogue
    if valid_indices is not None:
        split_info["valid_indices"] = np.array(valid_indices)

    split_file = checkpoint_path / "data_split.pkl"
    with open(split_file, "wb") as f:
        pickle.dump(split_info, f)

    print(f"  Saved data split to: {split_file}")


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

    # Auto-populate multi_head_model.survey_wavelengths if multi_head_model is used
    data_config = config_dict.get("data", {})
    surveys = data_config.get("surveys", [])
    label_sources = data_config.get("label_sources", ["apogee"])

    if "multi_head_model" in config_dict or len(surveys) > 1:
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
    resolved_device = config.device
    if resolved_device == "auto":
        import torch

        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {resolved_device}")
    print(f"Output: {config.get_output_path()}")
    print(f"Surveys: {config.data.surveys}")
    print(f"Label sources: {config.data.label_sources}")
    _print_memory("After config")

    # Load data from super-catalogue
    print(f"\nLoading data from {config.data.catalogue_path}")
    loader = CatalogueLoader(config.data.catalogue_path)

    import numpy as np

    if config.data.is_multi_survey:
        # Determine if multi-labelset training
        is_multi_label = config.data.is_multi_label

        if is_multi_label and config.data.use_dense_loading:
            # Multi-labelset mode with DENSE loading (high memory, ~40GB)
            # Only use when use_dense_loading: true is set in config
            duplicate_labels = config.data.duplicate_labels or {}
            sources_to_load = [
                s for s in config.data.label_sources if s not in duplicate_labels
            ]

            _print_memory("Before load_merged_for_training (DENSE)")
            X_dict, y_dict, has_data_dict, has_labels_dict = (
                loader.load_merged_for_training(
                    surveys=config.data.surveys,
                    label_source=sources_to_load[0],
                    max_flag_bits=config.data.max_flag_bits,
                    label_sources=sources_to_load,
                )
            )
            _print_memory("After load_merged_for_training (DENSE)")

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

            # Track split mode for saving later
            _split_mode = "dense_multi_label"
            _split_n_total = n_samples
            _split_valid_indices = None

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
            print("\nStarting multi-labelset training (dense)...")
            _print_memory("Before training")
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
            # Memory-efficient sparse loading (default for all multi-survey training)
            # Supports both single-label and multi-label configurations
            _print_memory("Before load_merged_sparse (SPARSE)")

            # Determine which label sources to load
            duplicate_labels = config.data.duplicate_labels or {}
            sources_to_load = [
                s for s in config.data.label_sources if s not in duplicate_labels
            ]

            # Load sparse data with multi-label support if needed
            if is_multi_label and len(sources_to_load) > 1:
                sparse_data = loader.load_merged_sparse(
                    surveys=config.data.surveys,
                    label_source=sources_to_load[0],
                    max_flag_bits=config.data.max_flag_bits,
                    label_sources=sources_to_load,
                )
            else:
                sparse_data = loader.load_merged_sparse(
                    surveys=config.data.surveys,
                    label_source=(
                        sources_to_load[0]
                        if sources_to_load
                        else config.data.label_sources[0]
                    ),
                    max_flag_bits=config.data.max_flag_bits,
                )
            _print_memory("After load_merged_sparse (SPARSE)")

            # Handle duplicate_labels: copy labels from source to target
            if duplicate_labels:
                print(f"  Duplicating labels: {duplicate_labels}")
                for target, source in duplicate_labels.items():
                    if sparse_data.labels_dict is not None:
                        # Multi-label mode: copy in labels_dict
                        if source not in sparse_data.labels_dict:
                            print(
                                f"Error: Cannot duplicate from '{source}' - not in loaded sources"
                            )
                            return 1
                        sparse_data.labels_dict[target] = sparse_data.labels_dict[
                            source
                        ].copy()
                        sparse_data.has_labels_dict[target] = (
                            sparse_data.has_labels_dict[source].copy()
                        )
                        if target not in sparse_data.label_sources:
                            sparse_data.label_sources.append(target)
                    else:
                        # Single-label mode: initialize multi-label structure
                        sparse_data.labels_dict = {
                            source: sparse_data.labels.copy(),
                            target: sparse_data.labels.copy(),
                        }
                        has_labels = np.any(sparse_data.labels[:, 2, :] > 0, axis=1)
                        sparse_data.has_labels_dict = {
                            source: has_labels.copy(),
                            target: has_labels.copy(),
                        }
                        sparse_data.label_sources = [source, target]

            print(f"  Surveys loaded: {sparse_data.surveys}")
            for survey_name in sparse_data.surveys:
                n_with_data = sparse_data.n_with_data(survey_name)
                n_wave = sparse_data.wavelengths[survey_name].shape[0]
                print(
                    f"    {survey_name}: {n_wave} wavelengths, "
                    f"{n_with_data:,} stars"
                )
            print(f"  Total unique stars: {sparse_data.n_total:,}")
            print(f"  Parameters: {sparse_data.n_params}")

            # Report label sources for multi-label
            if sparse_data.is_multi_label():
                print(f"  Label sources: {sparse_data.label_sources}")
                for source in sparse_data.label_sources:
                    n_with_labels = sparse_data.has_labels_for_source(source).sum()
                    print(f"    {source}: {n_with_labels:,} stars")

            # Filter to samples with at least one survey having data AND labels
            has_any_survey = np.zeros(sparse_data.n_total, dtype=bool)
            for survey_name in sparse_data.surveys:
                has_any_survey |= sparse_data.has_data(survey_name)

            # For multi-label: require at least one label source
            # For single-label: require all labels valid
            if sparse_data.is_multi_label():
                has_any_labels = np.zeros(sparse_data.n_total, dtype=bool)
                for source in sparse_data.label_sources:
                    has_any_labels |= sparse_data.has_labels_for_source(source)
                valid_labels = has_any_labels
                label_desc = "at least one label source"
            else:
                label_mask = sparse_data.labels[:, 2, :]  # (N, n_params)
                valid_labels = label_mask.all(axis=1)
                label_desc = "all labels"

            valid_samples = has_any_survey & valid_labels
            valid_indices = np.where(valid_samples)[0]
            print(
                f"  Valid samples (at least one survey + {label_desc}): "
                f"{len(valid_indices):,}"
            )

            # Report memory usage
            mem_usage = sparse_data.memory_usage_mb()
            print(f"  Memory usage: {mem_usage['total']:.1f} MB (sparse storage)")

            # Split indices for train/val/test
            n_samples = len(valid_indices)
            perm = np.random.RandomState(config.seed).permutation(n_samples)

            n_train = int(n_samples * config.data.train_ratio)
            n_val = int(n_samples * config.data.val_ratio)

            train_idx = valid_indices[perm[:n_train]]
            val_idx = valid_indices[perm[n_train : n_train + n_val]]
            test_idx = valid_indices[perm[n_train + n_val :]]

            # Track split mode for saving later
            _split_mode = "sparse_multi_survey"
            _split_n_total = n_samples
            _split_valid_indices = valid_indices

            print(f"\n  Training samples: {len(train_idx):,}")
            print(f"  Validation samples: {len(val_idx):,}")
            print(f"  Test samples: {len(test_idx):,}")

            # Create trainer (uses MultiHeadMLP via config.is_multi_head)
            print("\nInitializing trainer...")
            trainer = Trainer(config)

            print(f"  Model parameters: {trainer.model.count_parameters():,}")
            if sparse_data.is_multi_label():
                print("  Architecture: MultiHeadMLP (multi-labelset)")
                print(f"  Output heads: {config.multi_head_model.label_sources}")
            else:
                print("  Architecture: MultiHeadMLP")
            print(
                f"  Combination mode: {config.multi_head_model.combination_mode.value}"
            )

            # Train with sparse multi-survey data (memory-efficient)
            training_mode = (
                "multi-labelset" if sparse_data.is_multi_label() else "multi-survey"
            )
            print(f"\nStarting {training_mode} training (sparse)...")
            _print_memory("Before training")
            history = trainer.fit_multi_survey_sparse(sparse_data, train_idx, val_idx)
            _print_memory("After training")

        print("\nTraining complete!")
        print(f"  Best validation loss: {history.best_val_loss:.6f}")
        print(f"  Best epoch: {history.best_epoch + 1}")

        # Save checkpoint
        print("\nSaving checkpoint...")
        checkpoint_path = trainer.save_checkpoint()
        print(f"  Saved to: {checkpoint_path}")

        # Save data split for evaluation
        _save_data_split(
            checkpoint_path=checkpoint_path,
            test_idx=test_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            split_mode=_split_mode,
            n_total=_split_n_total,
            valid_indices=_split_valid_indices,
        )

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

        # Track split mode for saving later
        _split_mode = "single_survey"
        _split_n_total = n_samples
        _split_valid_indices = None

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

        # Save data split for evaluation
        _save_data_split(
            checkpoint_path=checkpoint_path,
            test_idx=test_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            split_mode=_split_mode,
            n_total=_split_n_total,
            valid_indices=_split_valid_indices,
        )

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


def _recreate_data_split(config, loader):
    """Recreate data split from config using the same logic as training.

    This is the fallback when data_split.pkl is not found. It reproduces
    the exact same splitting logic used during training.

    Args:
        config: ExperimentConfig loaded from the checkpoint.
        loader: CatalogueLoader for the data.

    Returns:
        Tuple of (test_idx, split_mode, valid_indices, sparse_data_or_filtered_data)
        where sparse_data_or_filtered_data is either a SparseData object (for sparse mode)
        or a tuple of (X, y) filtered arrays (for dense/single modes).
    """
    import numpy as np

    print("  Recreating split from config seed...")

    if config.data.is_multi_survey:
        is_multi_label = config.data.is_multi_label

        if is_multi_label and config.data.use_dense_loading:
            # Dense multi-label mode
            duplicate_labels = config.data.duplicate_labels or {}
            sources_to_load = [
                s for s in config.data.label_sources if s not in duplicate_labels
            ]

            X_dict, y_dict, has_data_dict, has_labels_dict = (
                loader.load_merged_for_training(
                    surveys=config.data.surveys,
                    label_source=sources_to_load[0],
                    max_flag_bits=config.data.max_flag_bits,
                    label_sources=sources_to_load,
                )
            )

            # Handle duplicate_labels
            if duplicate_labels:
                for target, source in duplicate_labels.items():
                    y_dict[target] = y_dict[source].copy()
                    has_labels_dict[target] = has_labels_dict[source].copy()

            first_source = config.data.label_sources[0]
            n_total = y_dict[first_source].shape[0]

            # Filter to valid samples
            has_any_survey = np.zeros(n_total, dtype=bool)
            for survey_name in X_dict:
                has_any_survey |= has_data_dict[survey_name]

            has_any_labels = np.zeros(n_total, dtype=bool)
            for source in config.data.label_sources:
                has_any_labels |= has_labels_dict[source]

            valid_samples = has_any_survey & has_any_labels

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

            test_idx = indices[n_train + n_val :]

            return (
                test_idx,
                "dense_multi_label",
                None,
                (X_dict, y_dict, has_data_dict, has_labels_dict),
            )

        else:
            # Sparse multi-survey mode
            duplicate_labels = config.data.duplicate_labels or {}
            sources_to_load = [
                s for s in config.data.label_sources if s not in duplicate_labels
            ]

            if is_multi_label and len(sources_to_load) > 1:
                sparse_data = loader.load_merged_sparse(
                    surveys=config.data.surveys,
                    label_source=sources_to_load[0],
                    max_flag_bits=config.data.max_flag_bits,
                    label_sources=sources_to_load,
                )
            else:
                sparse_data = loader.load_merged_sparse(
                    surveys=config.data.surveys,
                    label_source=(
                        sources_to_load[0]
                        if sources_to_load
                        else config.data.label_sources[0]
                    ),
                    max_flag_bits=config.data.max_flag_bits,
                )

            # Handle duplicate_labels
            if duplicate_labels:
                for target, source in duplicate_labels.items():
                    if sparse_data.labels_dict is not None:
                        sparse_data.labels_dict[target] = sparse_data.labels_dict[
                            source
                        ].copy()
                        sparse_data.has_labels_dict[target] = (
                            sparse_data.has_labels_dict[source].copy()
                        )
                        if target not in sparse_data.label_sources:
                            sparse_data.label_sources.append(target)
                    else:
                        sparse_data.labels_dict = {
                            source: sparse_data.labels.copy(),
                            target: sparse_data.labels.copy(),
                        }
                        has_labels = np.any(sparse_data.labels[:, 2, :] > 0, axis=1)
                        sparse_data.has_labels_dict = {
                            source: has_labels.copy(),
                            target: has_labels.copy(),
                        }
                        sparse_data.label_sources = [source, target]

            # Filter to valid samples
            has_any_survey = np.zeros(sparse_data.n_total, dtype=bool)
            for survey_name in sparse_data.surveys:
                has_any_survey |= sparse_data.has_data(survey_name)

            if sparse_data.is_multi_label():
                has_any_labels = np.zeros(sparse_data.n_total, dtype=bool)
                for source in sparse_data.label_sources:
                    has_any_labels |= sparse_data.has_labels_for_source(source)
                valid_labels = has_any_labels
            else:
                label_mask = sparse_data.labels[:, 2, :]
                valid_labels = label_mask.all(axis=1)

            valid_samples = has_any_survey & valid_labels
            valid_indices = np.where(valid_samples)[0]

            # Split indices
            n_samples = len(valid_indices)
            perm = np.random.RandomState(config.seed).permutation(n_samples)

            n_train = int(n_samples * config.data.train_ratio)
            n_val = int(n_samples * config.data.val_ratio)

            test_idx = valid_indices[perm[n_train + n_val :]]

            return test_idx, "sparse_multi_survey", valid_indices, sparse_data

    else:
        # Single survey mode
        survey = config.data.surveys[0]
        X, y = loader.load_for_training(
            survey=survey,
            max_flag_bits=config.data.max_flag_bits,
        )

        # Filter to valid samples
        mask = y[:, 2, :]
        valid_samples = mask.all(axis=1)

        X = X[valid_samples]
        y = y[valid_samples]

        # Split data
        n_samples = X.shape[0]
        indices = np.random.RandomState(config.seed).permutation(n_samples)

        n_train = int(n_samples * config.data.train_ratio)
        n_val = int(n_samples * config.data.val_ratio)

        test_idx = indices[n_train + n_val :]

        return test_idx, "single_survey", None, (X, y)


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Execute the evaluate command."""
    import json
    import pickle

    import numpy as np
    import torch
    import yaml

    from dorothy.config.schema import ExperimentConfig
    from dorothy.data import CatalogueLoader

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1

    print(f"Evaluating checkpoint: {checkpoint_path}")

    # Load config from checkpoint
    config_file = checkpoint_path / "config.yaml"
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        return 1

    with open(config_file) as f:
        # Use full_load to handle Python tuples in saved config
        config_dict = yaml.full_load(f)

    # Handle Path conversion
    if "data" in config_dict and "catalogue_path" in config_dict["data"]:
        config_dict["data"]["catalogue_path"] = Path(
            config_dict["data"]["catalogue_path"]
        )

    config = ExperimentConfig(**config_dict)
    print(f"  Experiment: {config.name}")
    print(f"  Surveys: {config.data.surveys}")
    print(f"  Label sources: {config.data.label_sources}")

    # Load data split
    split_file = checkpoint_path / "data_split.pkl"
    loader = CatalogueLoader(config.data.catalogue_path)

    if split_file.exists():
        print(f"\nLoading saved data split from {split_file}")
        with open(split_file, "rb") as f:
            split_info = pickle.load(f)

        test_idx = split_info["test_idx"]
        split_mode = split_info["split_mode"]
        valid_indices = split_info.get("valid_indices")

        print(f"  Split mode: {split_mode}")
        print(f"  Test samples: {len(test_idx):,}")

        # Load data based on split mode
        if split_mode == "sparse_multi_survey":
            # Reload sparse data
            duplicate_labels = config.data.duplicate_labels or {}
            sources_to_load = [
                s for s in config.data.label_sources if s not in duplicate_labels
            ]
            is_multi_label = config.data.is_multi_label

            if is_multi_label and len(sources_to_load) > 1:
                sparse_data = loader.load_merged_sparse(
                    surveys=config.data.surveys,
                    label_source=sources_to_load[0],
                    max_flag_bits=config.data.max_flag_bits,
                    label_sources=sources_to_load,
                )
            else:
                sparse_data = loader.load_merged_sparse(
                    surveys=config.data.surveys,
                    label_source=(
                        sources_to_load[0]
                        if sources_to_load
                        else config.data.label_sources[0]
                    ),
                    max_flag_bits=config.data.max_flag_bits,
                )

            # Handle duplicate_labels
            if duplicate_labels:
                for target, source in duplicate_labels.items():
                    if sparse_data.labels_dict is not None:
                        sparse_data.labels_dict[target] = sparse_data.labels_dict[
                            source
                        ].copy()
                        sparse_data.has_labels_dict[target] = (
                            sparse_data.has_labels_dict[source].copy()
                        )
                        if target not in sparse_data.label_sources:
                            sparse_data.label_sources.append(target)

            data_for_eval = ("sparse", sparse_data)

        elif split_mode == "dense_multi_label":
            # Reload dense data
            duplicate_labels = config.data.duplicate_labels or {}
            sources_to_load = [
                s for s in config.data.label_sources if s not in duplicate_labels
            ]

            X_dict, y_dict, has_data_dict, has_labels_dict = (
                loader.load_merged_for_training(
                    surveys=config.data.surveys,
                    label_source=sources_to_load[0],
                    max_flag_bits=config.data.max_flag_bits,
                    label_sources=sources_to_load,
                )
            )

            # Handle duplicate_labels
            if duplicate_labels:
                for target, source in duplicate_labels.items():
                    y_dict[target] = y_dict[source].copy()
                    has_labels_dict[target] = has_labels_dict[source].copy()

            # Filter to valid samples
            first_source = config.data.label_sources[0]
            n_total = y_dict[first_source].shape[0]

            has_any_survey = np.zeros(n_total, dtype=bool)
            for survey_name in X_dict:
                has_any_survey |= has_data_dict[survey_name]

            has_any_labels = np.zeros(n_total, dtype=bool)
            for source in config.data.label_sources:
                has_any_labels |= has_labels_dict[source]

            valid_samples = has_any_survey & has_any_labels

            X_dict = {survey: arr[valid_samples] for survey, arr in X_dict.items()}
            y_dict = {source: arr[valid_samples] for source, arr in y_dict.items()}
            has_data_dict = {
                survey: arr[valid_samples] for survey, arr in has_data_dict.items()
            }

            data_for_eval = ("dense", X_dict, y_dict, has_data_dict)

        else:  # single_survey
            survey = config.data.surveys[0]
            X, y = loader.load_for_training(
                survey=survey,
                max_flag_bits=config.data.max_flag_bits,
            )

            # Filter to valid samples
            mask = y[:, 2, :]
            valid_samples = mask.all(axis=1)

            X = X[valid_samples]
            y = y[valid_samples]

            data_for_eval = ("single", X, y)

    else:
        print(f"\nNo saved data split found. Recreating from config seed={config.seed}")
        test_idx, split_mode, valid_indices, data = _recreate_data_split(config, loader)

        print(f"  Split mode: {split_mode}")
        print(f"  Test samples: {len(test_idx):,}")

        if split_mode == "sparse_multi_survey":
            data_for_eval = ("sparse", data)
        elif split_mode == "dense_multi_label":
            X_dict, y_dict, has_data_dict, _ = data
            data_for_eval = ("dense", X_dict, y_dict, has_data_dict)
        else:
            X, y = data
            data_for_eval = ("single", X, y)

    # Load model
    print("\nLoading model...")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Determine model type and load accordingly
    model_file = args.model
    model_path = checkpoint_path / model_file
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    print(f"  Model: {model_file}")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Check if it's a multi-head model by looking for survey encoders or trunk
    is_multi_head = any(
        k.startswith("encoders.") or k.startswith("trunk.") or "survey_encoders" in k
        for k in state_dict
    )

    if is_multi_head:
        from dorothy.models import MultiHeadMLP

        # Create multi-head model from config and load state dict
        model = MultiHeadMLP(
            survey_configs=config.multi_head_model.survey_wavelengths,
            n_parameters=config.multi_head_model.n_parameters,
            latent_dim=config.multi_head_model.latent_dim,
            encoder_hidden=list(config.multi_head_model.encoder_hidden),
            trunk_hidden=list(config.multi_head_model.trunk_hidden),
            output_hidden=list(config.multi_head_model.output_hidden),
            combination_mode=config.multi_head_model.combination_mode.value,
            normalization=config.multi_head_model.normalization.value,
            activation=config.multi_head_model.activation.value,
            dropout=config.multi_head_model.dropout,
            label_sources=config.multi_head_model.label_sources,
        )
        model.load_state_dict(state_dict)
    else:
        from dorothy.models import MLP

        # Use config hidden_layers if available, otherwise infer from state dict
        if config.model.hidden_layers:
            hidden_layers = list(config.model.hidden_layers)
            input_features = config.model.input_features
            output_features = config.model.output_features

            # Get normalization and activation from config
            normalization = (
                config.model.normalization.value
                if config.model.normalization
                else "layernorm"
            )
            activation = (
                config.model.activation.value if config.model.activation else "gelu"
            )
            dropout = config.model.dropout if config.model.dropout else 0.0

            model = MLP(
                input_features=input_features,
                output_features=output_features,
                hidden_layers=hidden_layers,
                normalization=normalization,
                activation=activation,
                dropout=dropout,
            )
        else:
            # Fallback: infer architecture from state dict
            linear_weights = [
                k for k, v in state_dict.items() if "weight" in k and v.dim() == 2
            ]
            linear_weights.sort()
            first_weight = state_dict[linear_weights[0]]
            last_weight = state_dict[linear_weights[-1]]
            input_features = first_weight.shape[1]
            output_features = last_weight.shape[0]
            inferred_hidden = [state_dict[w].shape[0] for w in linear_weights[:-1]]

            model = MLP(
                input_features=input_features,
                output_features=output_features,
                hidden_layers=inferred_hidden,
            )
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load normalizer
    normalizer_file = checkpoint_path / "normalizer.pkl"
    normalizer = None
    if normalizer_file.exists():
        from dorothy.data.normalizer import LabelNormalizer

        normalizer = LabelNormalizer.load(normalizer_file)
        print("  Normalizer: loaded")

    # Make predictions on test set
    print("\nMaking predictions on test set...")

    if data_for_eval[0] == "sparse":
        # Use shared evaluation utility that matches training validation exactly
        from dorothy.inference import evaluate_on_test_set

        sparse_data = data_for_eval[1]
        print("  Using shared evaluation pipeline (matches training validation)")

        result, result_physical = evaluate_on_test_set(
            model=model,
            data=sparse_data,
            test_indices=test_idx,
            normalizer=normalizer,
            config=config,
            batch_size=args.batch_size,
            device=device,
        )

    elif data_for_eval[0] == "dense":
        # Use shared evaluation utility that matches training validation exactly
        from dorothy.inference import evaluate_on_dense_data

        X_dict, y_dict, has_data_dict = (
            data_for_eval[1],
            data_for_eval[2],
            data_for_eval[3],
        )
        print("  Using shared evaluation pipeline (matches training validation)")

        result, result_physical = evaluate_on_dense_data(
            model=model,
            X_dict=X_dict,
            y_dict=y_dict,
            has_data_dict=has_data_dict,
            test_indices=test_idx,
            normalizer=normalizer,
            config=config,
            batch_size=args.batch_size,
            device=device,
        )

    else:  # single
        # Use shared evaluation utility that matches training validation exactly
        from dorothy.inference import evaluate_on_single_survey_data

        X, y = data_for_eval[1], data_for_eval[2]
        print("  Using shared evaluation pipeline (matches training validation)")

        result, result_physical = evaluate_on_single_survey_data(
            model=model,
            X=X,
            y=y,
            test_indices=test_idx,
            normalizer=normalizer,
            config=config,
            batch_size=args.batch_size,
            device=device,
        )

    # Display results
    print("\n" + "=" * 80)

    if normalizer is not None:
        # Show normalized space metrics (comparable to training validation)
        print("NORMALIZED SPACE METRICS (comparable to training validation)")
        print("=" * 80)
        if args.format == "markdown":
            summary = result.summary(format="markdown")
        else:
            summary = result.summary(format="text")
        print(summary)

        # Show physical space metrics for interpretability
        print("\n" + "=" * 80)
        print("PHYSICAL SPACE METRICS (interpretable units)")
        print("=" * 80)
        if args.format == "markdown":
            summary_phys = result_physical.summary(format="markdown")
        else:
            summary_phys = result_physical.summary(format="text")
        print(summary_phys)

        # Combined summary for saving
        full_summary = (
            "NORMALIZED SPACE METRICS (comparable to training validation)\n"
            + "=" * 80
            + "\n"
            + summary
            + "\n\n"
            + "PHYSICAL SPACE METRICS (interpretable units)\n"
            + "=" * 80
            + "\n"
            + summary_phys
        )
    else:
        if args.format == "markdown":
            summary = result.summary(format="markdown")
        else:
            summary = result.summary(format="text")
        print(summary)
        full_summary = summary

    # Save results if output path specified
    if args.output is not None:
        output_path = args.output
        if args.format == "json":
            # Include both normalized and physical metrics if available
            output_dict = {"normalized_space": result.to_dict()}
            if result_physical is not None:
                output_dict["physical_space"] = result_physical.to_dict()
            with open(output_path, "w") as f:
                json.dump(output_dict, f, indent=2)
        else:
            with open(output_path, "w") as f:
                f.write(full_summary)
        print(f"\nResults saved to: {output_path}")

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
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
