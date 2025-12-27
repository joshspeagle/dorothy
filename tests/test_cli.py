"""
Tests for the command-line interface.

These tests verify:
1. CLI parser creation and argument parsing
2. Command-line argument validation
3. Basic command execution paths
4. Help text generation
"""

import tempfile
from pathlib import Path

import pytest

from dorothy.cli.main import create_parser, main


class TestCLIParser:
    """Tests for CLI argument parsing."""

    def test_parser_creation(self):
        """Test that parser can be created."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "dorothy"

    def test_help_flag(self):
        """Test that --help works without error."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        # SystemExit 0 means help was shown successfully
        assert exc_info.value.code == 0

    def test_version_flag(self):
        """Test that --version works."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_no_command_returns_none(self):
        """Test that no command sets command to None."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_train_command_parses(self):
        """Test train command parsing."""
        parser = create_parser()
        args = parser.parse_args(["train", "config.yaml"])

        assert args.command == "train"
        assert args.config == Path("config.yaml")
        assert args.device == "auto"
        assert args.output_dir is None
        assert args.resume is None

    def test_train_command_with_options(self):
        """Test train command with all options."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "train",
                "config.yaml",
                "--device",
                "cuda",
                "--output-dir",
                "/outputs",
                "--resume",
                "checkpoint/",
            ]
        )

        assert args.command == "train"
        assert args.device == "cuda"
        assert args.output_dir == Path("/outputs")
        assert args.resume == Path("checkpoint/")

    def test_predict_command_parses(self):
        """Test predict command parsing."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "predict",
                "--checkpoint",
                "./model",
                "--input",
                "spectra.fits",
            ]
        )

        assert args.command == "predict"
        assert args.checkpoint == Path("./model")
        assert args.input == Path("spectra.fits")
        assert args.output is None
        assert args.device == "auto"
        assert args.batch_size == 1024
        assert args.denormalize is False

    def test_predict_command_with_options(self):
        """Test predict command with all options."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "predict",
                "--checkpoint",
                "./model",
                "--input",
                "spectra.fits",
                "--output",
                "results.csv",
                "--device",
                "cpu",
                "--batch-size",
                "512",
                "--denormalize",
            ]
        )

        assert args.output == Path("results.csv")
        assert args.device == "cpu"
        assert args.batch_size == 512
        assert args.denormalize is True

    def test_predict_requires_checkpoint(self):
        """Test that predict requires --checkpoint."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["predict", "--input", "spectra.fits"])

    def test_predict_requires_input(self):
        """Test that predict requires --input."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["predict", "--checkpoint", "./model"])

    def test_info_command_parses(self):
        """Test info command parsing."""
        parser = create_parser()
        args = parser.parse_args(["info", "./checkpoint"])

        assert args.command == "info"
        assert args.checkpoint == Path("./checkpoint")

    def test_device_choices_validated(self):
        """Test that device choices are validated."""
        parser = create_parser()

        # Valid choices should work
        for device in ["auto", "cpu", "cuda"]:
            args = parser.parse_args(["train", "config.yaml", "--device", device])
            assert args.device == device

        # Invalid choice should fail
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "config.yaml", "--device", "invalid"])


class TestCLIMain:
    """Tests for the main CLI entry point."""

    def test_no_command_shows_help(self, capsys):
        """Test that no command shows help and returns 0."""
        result = main([])
        assert result == 0

    def test_train_missing_config(self, capsys):
        """Test that train with missing config returns error."""
        result = main(["train", "nonexistent.yaml"])
        assert result == 1

        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()

    def test_predict_missing_checkpoint(self, capsys):
        """Test that predict with missing checkpoint returns error."""
        result = main(
            [
                "predict",
                "--checkpoint",
                "/nonexistent/path",
                "--input",
                "dummy.fits",
            ]
        )
        assert result == 1

        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()

    def test_info_missing_checkpoint(self, capsys):
        """Test that info with missing checkpoint returns error."""
        result = main(["info", "/nonexistent/path"])
        assert result == 1


class TestCLIInfoCommand:
    """Tests for the info command."""

    @pytest.fixture
    def temp_checkpoint(self):
        """Create a temporary checkpoint directory."""
        import pickle

        import torch

        from dorothy.models import MLP

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir)

            # Create a simple model and save it
            model = MLP(
                input_features=100,
                output_features=6,
                hidden_layers=[32, 16],
            )
            torch.save(model.state_dict(), checkpoint_path / "best_model.pth")

            # Create training history
            history = {
                "history_train": [1.0, 0.8, 0.6, 0.5, 0.4],
                "history_val": [1.1, 0.9, 0.7, 0.55, 0.45],
            }
            with open(checkpoint_path / "history_train_val.pkl", "wb") as f:
                pickle.dump(history, f)

            # Create learning rates
            lrs = [1e-3] * 100
            with open(checkpoint_path / "learning_rates.pkl", "wb") as f:
                pickle.dump(lrs, f)

            yield checkpoint_path

    def test_info_displays_model(self, temp_checkpoint, capsys):
        """Test that info displays model information."""
        result = main(["info", str(temp_checkpoint)])
        assert result == 0

        captured = capsys.readouterr()
        assert "best_model.pth" in captured.out
        assert "Parameters" in captured.out

    def test_info_displays_history(self, temp_checkpoint, capsys):
        """Test that info displays training history."""
        result = main(["info", str(temp_checkpoint)])
        assert result == 0

        captured = capsys.readouterr()
        assert "Training history" in captured.out
        assert "Epochs" in captured.out

    def test_info_displays_learning_rates(self, temp_checkpoint, capsys):
        """Test that info displays learning rate info."""
        result = main(["info", str(temp_checkpoint)])
        assert result == 0

        captured = capsys.readouterr()
        assert "Learning rates" in captured.out


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_train_invalid_config(self, capsys):
        """Test train with invalid YAML config."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("invalid:\n  yaml: [not valid\n")
            config_path = Path(f.name)

        try:
            result = main(["train", str(config_path)])
            # Should fail due to invalid YAML or missing required fields
            assert result == 1
        finally:
            config_path.unlink()

    def test_train_incomplete_config(self, capsys):
        """Test train with incomplete config."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("name: test\n")  # Missing required 'data' field
            config_path = Path(f.name)

        try:
            result = main(["train", str(config_path)])
            assert result == 1

            captured = capsys.readouterr()
            assert "Error" in captured.out or "error" in captured.out.lower()
        finally:
            config_path.unlink()
