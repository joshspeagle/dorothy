"""
Command-line interface for DOROTHY.

This module provides the main CLI entry point for training models,
making predictions, and managing experiments.

Commands:
    train     - Train a model from a configuration file
    predict   - Make predictions on new spectral data
    info      - Display information about a checkpoint
    evaluate  - Evaluate model on held-out test set

Examples:
    ```bash
    dorothy train config.yaml
    dorothy predict --checkpoint ./outputs/my_exp --input spectra.fits --output predictions.csv
    dorothy info ./outputs/my_exp
    dorothy evaluate ./outputs/my_exp
    ```
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
        choices=["auto", "cpu", "cuda", "mps"],
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
        help="Initialize model weights from a previous checkpoint directory "
        "(warm start; the optimizer and LR schedule start fresh)",
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
        choices=["auto", "cpu", "cuda", "mps"],
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
        help="Output predictions in physical units (the default)",
    )
    predict_parser.add_argument(
        "--normalized",
        action="store_true",
        help="Output predictions in normalized units instead of physical units",
    )
    predict_parser.add_argument(
        "--survey",
        type=str,
        default=None,
        help="Survey the spectra come from (required for multi-survey models)",
    )
    predict_parser.add_argument(
        "--label-source",
        type=str,
        default=None,
        help="Output head to predict from, for multi-head models "
        "(default: the first label source)",
    )
    predict_parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Write the raw predicted scatter, without the checkpoint's "
        "calibration factors",
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
        choices=["auto", "cpu", "cuda", "mps"],
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
    evaluate_parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory for evaluation plots (default: <checkpoint>/plots/evaluation)",
    )
    evaluate_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating evaluation plots",
    )
    evaluate_parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Evaluate the raw predicted scatter, without the checkpoint's "
        "calibration factors",
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


def _generate_training_plots(history, checkpoint_path: Path, name: str) -> None:
    """Write the training report plots next to a checkpoint."""
    print("\nGenerating training plots...")
    try:
        from dorothy.visualization import generate_training_report

        plots_dir = checkpoint_path / "plots"
        plots = generate_training_report(
            history=history, output_dir=plots_dir, experiment_name=name
        )
        if plots:
            print(f"  Generated {len(plots)} plots in: {plots_dir}")
        else:
            print("  No plots generated (matplotlib may not be installed)")
    except Exception as e:  # plotting must never fail a finished training run
        print(f"  Warning: Could not generate plots: {e}")


def cmd_train(args: argparse.Namespace) -> int:
    """Execute the train command."""
    import yaml

    from dorothy.config.schema import ExperimentConfig
    from dorothy.data.training_data import complete_config_from_catalogue
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

    # Fill in data-dependent settings from the catalogue (survey wavelength
    # counts, output heads, parameter names)
    try:
        config_dict = complete_config_from_catalogue(config_dict, verbose=True)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    # Create config object
    try:
        config = ExperimentConfig(**config_dict)
    except Exception as e:
        print(f"Error: Invalid configuration: {e}")
        return 1

    from dorothy.data.training_data import load_training_data
    from dorothy.utils import resolve_device

    print(f"Experiment: {config.name}")
    print(f"Device: {resolve_device(config.device)}")
    print(f"Output: {config.get_output_path()}")
    print(f"Surveys: {config.data.surveys}")
    print(f"Label sources: {config.data.label_sources}")
    _print_memory("After config")

    # Load data, keep stars with >= 1 survey and >= 1 valid label, and split
    print(f"\nLoading data from {config.data.catalogue_path}")
    try:
        data = load_training_data(config)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    _print_memory("After loading data")
    data.configure_model(config)
    split = data.split(config.data.train_ratio, config.data.val_ratio, config.seed)
    print(f"\n  Training samples: {len(split.train):,}")
    print(f"  Validation samples: {len(split.val):,}")
    print(f"  Test samples: {len(split.test):,}")
    if len(split.train) == 0 or len(split.val) == 0:
        print("Error: Not enough valid samples to train")
        return 1

    print("\nInitializing trainer...")
    trainer = Trainer(config)
    print(f"  Model parameters: {trainer.model.count_parameters():,}")
    if config.is_multi_head:
        print(f"  Architecture: MultiHeadMLP, output heads: {trainer.label_keys}")
    else:
        print(f"  Architecture: MLP, input features: {config.model.input_features}")

    if args.resume is not None:
        import torch

        weights = args.resume / "best_model.pth"
        if not weights.exists():
            print(f"Error: No best_model.pth in {args.resume}")
            return 1
        resumed = args.resume / "config.yaml"
        if (
            resumed.exists()
            and not ExperimentConfig.load_checkpoint_config(
                resumed
            ).data.precision_encoding
        ):
            print(
                f"Error: {args.resume} was trained on sigma inputs, before the "
                "precision encoding, so its weights cannot start a new run"
            )
            return 1
        try:
            trainer.model.load_state_dict(
                torch.load(weights, map_location=trainer.device, weights_only=True)
            )
        except RuntimeError as e:
            print(f"Error: Checkpoint weights don't match the configured model: {e}")
            return 1
        print(f"  Initialized weights from {weights}")

    print(f"\nStarting training ({data.kind})...")
    _print_memory("Before training")
    history = data.fit(trainer, split)
    _print_memory("After training")

    print("\nTraining complete!")
    print(f"  Best validation loss: {history.best_val_loss:.6f}")
    print(f"  Best epoch: {history.best_epoch + 1}")
    for source, epoch in history.per_labelset_best_epoch.items():
        print(f"    {source}: best at epoch {epoch + 1}")

    print("\nSaving checkpoint...")
    checkpoint_path = trainer.save_checkpoint()
    print(f"  Saved to: {checkpoint_path}")
    split_path = split.save(checkpoint_path, data)
    print(f"  Saved data split to: {split_path}")

    _generate_training_plots(history, checkpoint_path, config.name)
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
            batch_size=args.batch_size,
            denormalize=not args.normalized,
            survey=args.survey,
            label_source=args.label_source,
            calibrate=not args.no_calibration,
        )
        units = "normalized" if args.normalized else "physical"
        print(f"\nPredictions ({units} units) saved to {output_path}")
        return 0
    except Exception as e:
        print(f"Error: Prediction failed: {e}")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command."""
    import pickle

    import torch

    from dorothy.inference.calibration import CALIBRATION_FILE, load_calibration

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

                except Exception as e:
                    print(f"    Error loading model: {e}")

    # Architecture and data settings from the saved config
    config_file = checkpoint_path / "config.yaml"
    if config_file.exists():
        try:
            from dorothy.config.schema import ExperimentConfig

            config = ExperimentConfig.load_checkpoint_config(config_file)
            print("\nConfiguration:")
            print(f"  Experiment: {config.name}")
            if config.is_multi_head:
                mh = config.multi_head_model
                print("  Architecture: MultiHeadMLP")
                print(f"    Surveys (wavelength bins): {dict(mh.survey_wavelengths)}")
                print(f"    Output heads: {mh.label_sources}")
                print(
                    f"    Encoder {list(mh.encoder_hidden)} -> latent {mh.latent_dim}, "
                    f"trunk {list(mh.trunk_hidden)}, head {list(mh.output_hidden)}"
                )
            else:
                print("  Architecture: MLP")
                print(f"    Input features: {config.model.input_features}")
                print(f"    Hidden layers: {list(config.model.hidden_layers)}")
            model_config = config.multi_head_model or config.model
            uncertainty = (
                "label-aware"
                if model_config.label_aware_uncertainty
                else "additive, sigma_y^2 + s^2 (saved before label-aware uncertainty)"
            )
            print(f"  Uncertainty model: {uncertainty}")
            names = config.parameter_names
            print(f"  Parameters ({len(names)}): {', '.join(names)}")
            print(f"  Spectral normalization: {config.data.normalize_spectra}")
            print(f"  Precision encoding: {config.data.precision_encoding}")
        except Exception as e:
            print(f"  Error loading config: {e}")

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
            if history.get("history_val"):
                val_losses = history["history_val"]
                best_epoch = history.get(
                    "best_epoch", val_losses.index(min(val_losses))
                )
                print(
                    f"  Best val loss: {val_losses[best_epoch]:.6f} "
                    f"(epoch {best_epoch + 1})"
                )
            for source, epoch in history.get("per_labelset_best_epoch", {}).items():
                loss = history["per_labelset_best_val_loss"][source]
                print(f"    {source}: best val loss {loss:.6f} (epoch {epoch + 1})")
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

    # Calibration factors, one column per head
    try:
        calibration = load_calibration(checkpoint_path)
        if calibration is None:
            print("\nCalibration: none (the uncertainties are the raw scatter)")
        else:
            heads = list(calibration)
            print(f"\nCalibration factors of best_model.pth ({CALIBRATION_FILE}):")
            print(f"  {'Parameter':<12}" + "".join(f"{h:>10}" for h in heads))
            for p in calibration[heads[0]]:
                factors = "".join(f"{calibration[h][p]:>10.4f}" for h in heads)
                print(f"  {p:<12}{factors}")
    except Exception as e:
        print(f"\nError loading calibration factors: {e}")

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


def _calibration_table(head, fmt: str) -> str:
    """One head's calibration factors and its raw and calibrated z-score spreads."""
    rows = [
        (
            p,
            head.calibration[p],
            head.uncalibrated[p].z_robust_scatter,
            head.normalized[p].z_robust_scatter,
        )
        for p in head.parameter_names
    ]
    header = ["Parameter", "Factor", "z_scatter raw", "z_scatter calibrated"]
    if fmt == "markdown":
        lines = [f"| {' | '.join(header)} |", "|---" * len(header) + "|"]
        lines += [
            f"| {p} | {c:.4f} | {raw:.4f} | {cal:.4f} |" for p, c, raw, cal in rows
        ]
    else:
        lines = ["{:<12} {:>10} {:>14} {:>21}".format(*header), "-" * 80]
        lines += [
            f"{p:<12} {c:>10.4f} {raw:>14.4f} {cal:>21.4f}" for p, c, raw, cal in rows
        ]
    return "\n".join(lines)


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Execute the evaluate command."""
    import json

    from dorothy.data.training_data import SPLIT_FILE, DataSplit, load_training_data
    from dorothy.inference import evaluate_model, load_checkpoint
    from dorothy.inference.calibration import CALIBRATION_FILE

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1
    if not (checkpoint_path / "config.yaml").exists():
        print(f"Error: Config file not found: {checkpoint_path / 'config.yaml'}")
        return 1

    print(f"Evaluating checkpoint: {checkpoint_path}")
    try:
        ckpt = load_checkpoint(
            checkpoint_path, model_file=args.model, device=args.device
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    config = ckpt.config
    print(f"  Experiment: {config.name}")
    print(f"  Surveys: {config.data.surveys}")
    print(f"  Label sources: {config.data.label_sources}")
    print(f"  Model: {args.model} on {ckpt.device}")
    print(f"  Normalizer: {'loaded' if ckpt.normalizer is not None else 'none'}")
    calibration = None if args.no_calibration else ckpt.calibration
    if calibration is not None:
        print(f"  Calibration: factors from {CALIBRATION_FILE}")
    elif args.no_calibration:
        print("  Calibration: off (--no-calibration)")
    else:
        print(f"  Calibration: none for {args.model}")

    print(f"\nLoading data from {config.data.catalogue_path}")
    try:
        data = load_training_data(config)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    split_file = checkpoint_path / SPLIT_FILE
    if split_file.exists():
        print(f"  Loading saved data split from {split_file}")
        split = DataSplit.load(checkpoint_path, data)
    else:
        print(f"  No saved data split found. Recreating from config seed={config.seed}")
        split = data.split(config.data.train_ratio, config.data.val_ratio, config.seed)
    print(f"  Test samples: {len(split.test):,}")

    print("\nMaking predictions on test set...")
    print("  Using shared evaluation pipeline (as in training validation)")
    results = evaluate_model(
        ckpt.model,
        ckpt.normalizer,
        config,
        data,
        split.test,
        batch_size=args.batch_size,
        device=ckpt.device,
        calibration=calibration,
    )
    # Text / markdown summaries, one section per head
    fmt = "markdown" if args.format == "markdown" else "text"
    sections = []
    for source_name, head in results.items():
        source = f"[{source_name}]"
        if head.physical is not None:
            note = (
                "calibrated uncertainties"
                if head.uncalibrated is not None
                else "comparable to training validation"
            )
            sections.append(
                f"NORMALIZED SPACE METRICS ({note}) {source}\n"
                + "=" * 80
                + "\n"
                + head.normalized.summary(format=fmt)
            )
            sections.append(
                f"PHYSICAL SPACE METRICS (interpretable units) {source}\n"
                + "=" * 80
                + "\n"
                + head.physical.summary(format=fmt)
            )
        else:
            sections.append(
                f"METRICS {source}\n"
                + "=" * 80
                + "\n"
                + head.normalized.summary(format=fmt)
            )
        if head.uncalibrated is not None:
            sections.append(
                f"CALIBRATION {source}\n"
                + "=" * 80
                + "\n"
                + _calibration_table(head, fmt)
            )
    full_summary = "\n\n".join(sections)

    # JSON: primary head at the top level (backward compatible) + every head
    primary = next(iter(results.values()))
    output_dict = primary.to_dict()
    output_dict["label_sources"] = {k: head.to_dict() for k, head in results.items()}

    if args.format == "json" and args.output is None:
        print(json.dumps(output_dict, indent=2))
    else:
        print("\n" + "=" * 80)
        print(full_summary)

    if args.output is not None:
        with open(args.output, "w") as f:
            if args.format == "json":
                json.dump(output_dict, f, indent=2)
            else:
                f.write(full_summary)
        print(f"\nResults saved to: {args.output}")

    if not args.no_plots:
        try:
            from dorothy.visualization import generate_evaluation_plots

            plots_dir = args.plots_dir or checkpoint_path / "plots" / "evaluation"
            n_plots = 0
            for source_name, head in results.items():
                n_plots += len(
                    generate_evaluation_plots(
                        plots_dir,
                        head.y_pred,
                        head.y_true,
                        head.pred_scatter,
                        head.z_label_errors,
                        head.mask,
                        head.parameter_names,
                        label_source=source_name,
                    )
                )
            print(f"\nGenerated {n_plots} evaluation plots in: {plots_dir}")
        except Exception as e:  # plots are optional
            print(f"\nWarning: Could not generate evaluation plots: {e}")

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
