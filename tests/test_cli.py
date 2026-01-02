"""
Tests for the command-line interface.

These tests verify:
1. CLI parser creation and argument parsing
2. Command-line argument validation
3. Basic command execution paths
4. Help text generation
5. Full training workflow integration tests
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from dorothy.cli.main import create_parser, main
from dorothy.data.catalogue_loader import PARAMETER_NAMES


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


class TestCLITrainIntegration:
    """Full integration tests for the train command with mock HDF5 data."""

    @pytest.fixture
    def mock_catalogue(self, tmp_path):
        """Create a complete mock HDF5 super-catalogue for training."""
        path = tmp_path / "test_catalogue.h5"
        n_stars = 100
        n_wave_boss = 100  # Small for fast tests

        with h5py.File(path, "w") as f:
            # Metadata
            meta = f.create_group("metadata")
            meta.create_dataset("gaia_id", data=np.arange(n_stars, dtype=np.int64))
            meta.create_dataset("ra", data=np.random.uniform(0, 360, n_stars))
            meta.create_dataset("dec", data=np.random.uniform(-90, 90, n_stars))

            # Surveys
            surveys = f.create_group("surveys")

            boss = surveys.create_group("boss")
            boss.create_dataset(
                "flux", data=np.random.randn(n_stars, n_wave_boss).astype(np.float32)
            )
            boss.create_dataset(
                "ivar",
                data=np.abs(np.random.randn(n_stars, n_wave_boss)).astype(np.float32)
                + 0.1,
            )
            boss.create_dataset(
                "wavelength",
                data=np.linspace(3600, 10400, n_wave_boss).astype(np.float32),
            )
            boss.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            # Labels - APOGEE
            # Generate realistic stellar parameter values
            # Params: Teff, logg, [Fe/H], [Mg/Fe], [C/Fe], [Si/Fe], [Ni/Fe], [Al/Fe], [Ca/Fe], [N/Fe], [Mn/Fe]
            labels = f.create_group("labels")
            apogee = labels.create_group("apogee")
            label_values = np.random.randn(n_stars, 11).astype(np.float32)
            # Teff must be positive (typically 3500-7000 K)
            label_values[:, 0] = np.random.uniform(4000, 6000, n_stars).astype(
                np.float32
            )
            # logg typically 0-5
            label_values[:, 1] = np.random.uniform(2, 4.5, n_stars).astype(np.float32)
            # [Fe/H] typically -2 to 0.5
            label_values[:, 2] = np.random.uniform(-1, 0.3, n_stars).astype(np.float32)
            apogee.create_dataset("values", data=label_values)
            apogee.create_dataset(
                "errors",
                data=np.abs(np.random.randn(n_stars, 11)).astype(np.float32) + 0.01,
            )
            apogee.create_dataset("flags", data=np.zeros((n_stars, 11), dtype=np.uint8))

            # Attributes
            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = ["boss"]
            f.attrs["creation_date"] = "2024-01-01T00:00:00"
            f.attrs["version"] = "1.0.0"

        return path

    @pytest.fixture
    def single_survey_config(self, mock_catalogue, tmp_path):
        """Create a minimal single-survey training config."""
        config = {
            "name": "test_single_survey",
            "data": {
                "catalogue_path": str(mock_catalogue),
                "surveys": ["boss"],
                "label_sources": ["apogee"],
                "train_ratio": 0.7,
                "val_ratio": 0.2,
            },
            "model": {
                "hidden_layers": [32, 16],
                "normalization": "layernorm",
                "activation": "gelu",
            },
            "training": {
                "epochs": 2,
                "batch_size": 32,
                "learning_rate": 0.001,
            },
            "output_dir": str(tmp_path / "outputs"),
            "seed": 42,
            "device": "cpu",
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path

    def test_train_single_survey_completes(self, single_survey_config, capsys):
        """Test that single-survey training completes successfully."""
        result = main(["train", str(single_survey_config)])

        captured = capsys.readouterr()
        assert "Training complete" in captured.out
        assert result == 0

    def test_train_creates_checkpoint(self, single_survey_config, tmp_path):
        """Test that training creates a checkpoint."""
        result = main(["train", str(single_survey_config)])
        assert result == 0

        # Check that checkpoint was created
        outputs_dir = tmp_path / "outputs"
        assert outputs_dir.exists()

        # Find all checkpoint directories (may be nested)
        model_files = list(outputs_dir.glob("**/best_model.pth"))
        assert len(model_files) >= 1, "Should create best_model.pth checkpoint"


class TestCLIMultiSurveyIntegration:
    """Integration tests for multi-survey training."""

    @pytest.fixture
    def multi_survey_catalogue(self, tmp_path):
        """Create a mock HDF5 with multiple surveys."""
        path = tmp_path / "multi_survey_catalogue.h5"
        n_stars = 100
        n_wave_boss = 80
        n_wave_desi = 100

        with h5py.File(path, "w") as f:
            # Metadata
            meta = f.create_group("metadata")
            meta.create_dataset("gaia_id", data=np.arange(n_stars, dtype=np.int64))
            meta.create_dataset("ra", data=np.random.uniform(0, 360, n_stars))
            meta.create_dataset("dec", data=np.random.uniform(-90, 90, n_stars))

            surveys = f.create_group("surveys")

            # BOSS: stars 0-69
            boss = surveys.create_group("boss")
            boss_flux = np.zeros((n_stars, n_wave_boss), dtype=np.float32)
            boss_ivar = np.zeros((n_stars, n_wave_boss), dtype=np.float32)
            boss_flux[:70] = np.random.randn(70, n_wave_boss).astype(np.float32)
            boss_ivar[:70] = (
                np.abs(np.random.randn(70, n_wave_boss)).astype(np.float32) + 0.1
            )
            boss.create_dataset("flux", data=boss_flux)
            boss.create_dataset("ivar", data=boss_ivar)
            boss.create_dataset(
                "wavelength", data=np.linspace(3600, 10400, n_wave_boss)
            )
            boss.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            # DESI: stars 30-99
            desi = surveys.create_group("desi")
            desi_flux = np.zeros((n_stars, n_wave_desi), dtype=np.float32)
            desi_ivar = np.zeros((n_stars, n_wave_desi), dtype=np.float32)
            desi_flux[30:] = np.random.randn(70, n_wave_desi).astype(np.float32)
            desi_ivar[30:] = (
                np.abs(np.random.randn(70, n_wave_desi)).astype(np.float32) + 0.1
            )
            desi.create_dataset("flux", data=desi_flux)
            desi.create_dataset("ivar", data=desi_ivar)
            desi.create_dataset("wavelength", data=np.linspace(3600, 9800, n_wave_desi))
            desi.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            # Labels with realistic values
            labels = f.create_group("labels")
            apogee = labels.create_group("apogee")
            label_values = np.random.randn(n_stars, 11).astype(np.float32)
            label_values[:, 0] = np.random.uniform(4000, 6000, n_stars).astype(
                np.float32
            )  # Teff
            label_values[:, 1] = np.random.uniform(2, 4.5, n_stars).astype(
                np.float32
            )  # logg
            label_values[:, 2] = np.random.uniform(-1, 0.3, n_stars).astype(
                np.float32
            )  # [Fe/H]
            apogee.create_dataset("values", data=label_values)
            apogee.create_dataset(
                "errors",
                data=np.abs(np.random.randn(n_stars, 11)).astype(np.float32) + 0.01,
            )
            apogee.create_dataset("flags", data=np.zeros((n_stars, 11), dtype=np.uint8))

            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = ["boss", "desi"]
            f.attrs["creation_date"] = "2024-01-01T00:00:00"
            f.attrs["version"] = "1.0.0"

        return path

    @pytest.fixture
    def multi_survey_config(self, multi_survey_catalogue, tmp_path):
        """Create a multi-survey training config."""
        config = {
            "name": "test_multi_survey",
            "data": {
                "catalogue_path": str(multi_survey_catalogue),
                "surveys": ["boss", "desi"],
                "label_sources": ["apogee"],
                "train_ratio": 0.7,
                "val_ratio": 0.2,
            },
            # Note: multi_head_model.survey_wavelengths will be auto-populated
            "multi_head_model": {
                "latent_dim": 32,
                "encoder_hidden": [64],
                "trunk_hidden": [32],
                "output_hidden": [16],
                "combination_mode": "concat",
            },
            "training": {
                "epochs": 2,
                "batch_size": 32,
                "learning_rate": 0.001,
            },
            "output_dir": str(tmp_path / "outputs"),
            "seed": 42,
            "device": "cpu",
        }
        config_path = tmp_path / "multi_survey_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path

    def test_train_multi_survey_completes(self, multi_survey_config, capsys):
        """Test that multi-survey training completes."""
        result = main(["train", str(multi_survey_config)])

        captured = capsys.readouterr()
        assert "Training complete" in captured.out or result == 0
        # Check that survey wavelengths were auto-populated
        assert "Survey wavelengths" in captured.out

    def test_train_auto_populates_wavelengths(self, multi_survey_config, capsys):
        """Test that CLI auto-populates survey wavelengths from catalogue."""
        main(["train", str(multi_survey_config)])

        captured = capsys.readouterr()
        # Should show wavelength counts from catalogue
        assert "boss" in captured.out.lower()
        assert "desi" in captured.out.lower()


class TestCLIMultiLabelsetIntegration:
    """Integration tests for multi-labelset training."""

    @pytest.fixture
    def multi_labelset_catalogue(self, tmp_path):
        """Create a mock HDF5 with multiple label sources."""
        path = tmp_path / "multi_labelset_catalogue.h5"
        n_stars = 100
        n_wave = 80

        with h5py.File(path, "w") as f:
            meta = f.create_group("metadata")
            meta.create_dataset("gaia_id", data=np.arange(n_stars, dtype=np.int64))
            meta.create_dataset("ra", data=np.random.uniform(0, 360, n_stars))
            meta.create_dataset("dec", data=np.random.uniform(-90, 90, n_stars))

            surveys = f.create_group("surveys")
            boss = surveys.create_group("boss")
            boss.create_dataset(
                "flux", data=np.random.randn(n_stars, n_wave).astype(np.float32)
            )
            boss.create_dataset(
                "ivar",
                data=np.abs(np.random.randn(n_stars, n_wave)).astype(np.float32) + 0.1,
            )
            boss.create_dataset("wavelength", data=np.linspace(3600, 10400, n_wave))
            boss.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            desi = surveys.create_group("desi")
            desi.create_dataset(
                "flux", data=np.random.randn(n_stars, n_wave).astype(np.float32)
            )
            desi.create_dataset(
                "ivar",
                data=np.abs(np.random.randn(n_stars, n_wave)).astype(np.float32) + 0.1,
            )
            desi.create_dataset("wavelength", data=np.linspace(3600, 9800, n_wave))
            desi.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            # Multiple label sources with realistic values
            labels = f.create_group("labels")

            # APOGEE: all stars
            apogee = labels.create_group("apogee")
            apogee_values = np.random.randn(n_stars, 11).astype(np.float32)
            apogee_values[:, 0] = np.random.uniform(4000, 6000, n_stars).astype(
                np.float32
            )  # Teff
            apogee_values[:, 1] = np.random.uniform(2, 4.5, n_stars).astype(
                np.float32
            )  # logg
            apogee_values[:, 2] = np.random.uniform(-1, 0.3, n_stars).astype(
                np.float32
            )  # [Fe/H]
            apogee.create_dataset("values", data=apogee_values)
            apogee.create_dataset(
                "errors",
                data=np.abs(np.random.randn(n_stars, 11)).astype(np.float32) + 0.01,
            )
            apogee.create_dataset("flags", data=np.zeros((n_stars, 11), dtype=np.uint8))

            # GALAH: stars 20-99 only
            galah = labels.create_group("galah")
            galah_values = np.random.randn(n_stars, 11).astype(np.float32)
            galah_values[:, 0] = np.random.uniform(4000, 6000, n_stars).astype(
                np.float32
            )  # Teff
            galah_values[:, 1] = np.random.uniform(2, 4.5, n_stars).astype(
                np.float32
            )  # logg
            galah_values[:, 2] = np.random.uniform(-1, 0.3, n_stars).astype(
                np.float32
            )  # [Fe/H]
            galah_values[:20] = np.nan  # No labels for first 20 stars
            galah.create_dataset("values", data=galah_values)
            galah.create_dataset(
                "errors",
                data=np.abs(np.random.randn(n_stars, 11)).astype(np.float32) + 0.01,
            )
            galah.create_dataset("flags", data=np.zeros((n_stars, 11), dtype=np.uint8))

            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = ["boss", "desi"]
            f.attrs["creation_date"] = "2024-01-01T00:00:00"
            f.attrs["version"] = "1.0.0"

        return path

    @pytest.fixture
    def multi_labelset_config(self, multi_labelset_catalogue, tmp_path):
        """Create a multi-labelset training config."""
        config = {
            "name": "test_multi_labelset",
            "data": {
                "catalogue_path": str(multi_labelset_catalogue),
                "surveys": ["boss", "desi"],
                "label_sources": ["apogee", "galah"],
                "train_ratio": 0.7,
                "val_ratio": 0.2,
            },
            "multi_head_model": {
                "latent_dim": 32,
                "encoder_hidden": [64],
                "trunk_hidden": [32],
                "output_hidden": [16],
                "combination_mode": "concat",
            },
            "training": {
                "epochs": 2,
                "batch_size": 32,
                "learning_rate": 0.001,
            },
            "output_dir": str(tmp_path / "outputs"),
            "seed": 42,
            "device": "cpu",
        }
        config_path = tmp_path / "multi_labelset_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path

    def test_train_multi_labelset_completes(self, multi_labelset_config, capsys):
        """Test that multi-labelset training completes."""
        main(["train", str(multi_labelset_config)])

        captured = capsys.readouterr()
        # Should mention multiple label sources
        assert "apogee" in captured.out.lower()
        assert "galah" in captured.out.lower()

    @pytest.fixture
    def duplicate_labels_config(self, multi_labelset_catalogue, tmp_path):
        """Create a config with duplicate_labels for testing."""
        config = {
            "name": "test_duplicate_labels",
            "data": {
                "catalogue_path": str(multi_labelset_catalogue),
                "surveys": ["boss", "desi"],
                "label_sources": ["apogee", "fake_galah"],
                "duplicate_labels": {"fake_galah": "apogee"},
                "train_ratio": 0.7,
                "val_ratio": 0.2,
            },
            "multi_head_model": {
                "latent_dim": 32,
                "encoder_hidden": [64],
                "trunk_hidden": [32],
                "output_hidden": [16],
                "combination_mode": "concat",
            },
            "training": {
                "epochs": 2,
                "batch_size": 32,
                "learning_rate": 0.001,
            },
            "output_dir": str(tmp_path / "outputs"),
            "seed": 42,
            "device": "cpu",
        }
        config_path = tmp_path / "duplicate_labels_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path

    def test_train_duplicate_labels(self, duplicate_labels_config, capsys):
        """Test that duplicate_labels works to copy labels."""
        main(["train", str(duplicate_labels_config)])

        captured = capsys.readouterr()
        # Should show duplication message
        assert "Duplicating labels" in captured.out


class TestCLIEvaluateParser:
    """Tests for evaluate command parsing."""

    def test_evaluate_command_parses(self):
        """Test evaluate command parsing."""
        parser = create_parser()
        args = parser.parse_args(["evaluate", "./checkpoint"])

        assert args.command == "evaluate"
        assert args.checkpoint == Path("./checkpoint")
        assert args.model == "best_model.pth"
        assert args.device == "auto"
        assert args.batch_size == 1024
        assert args.output is None
        assert args.format == "text"

    def test_evaluate_command_with_options(self):
        """Test evaluate command with all options."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "evaluate",
                "./checkpoint",
                "--model",
                "final_model.pth",
                "--device",
                "cpu",
                "--batch-size",
                "512",
                "--output",
                "results.json",
                "--format",
                "json",
            ]
        )

        assert args.command == "evaluate"
        assert args.checkpoint == Path("./checkpoint")
        assert args.model == "final_model.pth"
        assert args.device == "cpu"
        assert args.batch_size == 512
        assert args.output == Path("results.json")
        assert args.format == "json"

    def test_evaluate_format_choices(self):
        """Test that format choices are validated."""
        parser = create_parser()

        # Valid choices should work
        for fmt in ["text", "markdown", "json"]:
            args = parser.parse_args(["evaluate", "./checkpoint", "--format", fmt])
            assert args.format == fmt

        # Invalid choice should fail
        with pytest.raises(SystemExit):
            parser.parse_args(["evaluate", "./checkpoint", "--format", "invalid"])


class TestCLIEvaluateCommand:
    """Tests for the evaluate command."""

    def test_evaluate_missing_checkpoint(self, capsys):
        """Test that evaluate with missing checkpoint returns error."""
        result = main(["evaluate", "/nonexistent/path"])
        assert result == 1

        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()


class TestCLIEvaluateIntegration:
    """Integration tests for the evaluate command."""

    @pytest.fixture
    def trained_checkpoint(self, tmp_path):
        """Create a trained checkpoint for evaluation tests."""
        # Create mock catalogue
        path = tmp_path / "test_catalogue.h5"
        n_stars = 100
        n_wave = 80

        with h5py.File(path, "w") as f:
            meta = f.create_group("metadata")
            meta.create_dataset("gaia_id", data=np.arange(n_stars, dtype=np.int64))
            meta.create_dataset("ra", data=np.random.uniform(0, 360, n_stars))
            meta.create_dataset("dec", data=np.random.uniform(-90, 90, n_stars))

            surveys = f.create_group("surveys")
            boss = surveys.create_group("boss")
            boss.create_dataset(
                "flux", data=np.random.randn(n_stars, n_wave).astype(np.float32)
            )
            boss.create_dataset(
                "ivar",
                data=np.abs(np.random.randn(n_stars, n_wave)).astype(np.float32) + 0.1,
            )
            boss.create_dataset("wavelength", data=np.linspace(3600, 10400, n_wave))
            boss.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            labels = f.create_group("labels")
            apogee = labels.create_group("apogee")
            label_values = np.random.randn(n_stars, 11).astype(np.float32)
            label_values[:, 0] = np.random.uniform(4000, 6000, n_stars).astype(
                np.float32
            )
            label_values[:, 1] = np.random.uniform(2, 4.5, n_stars).astype(np.float32)
            label_values[:, 2] = np.random.uniform(-1, 0.3, n_stars).astype(np.float32)
            apogee.create_dataset("values", data=label_values)
            apogee.create_dataset(
                "errors",
                data=np.abs(np.random.randn(n_stars, 11)).astype(np.float32) + 0.01,
            )
            apogee.create_dataset("flags", data=np.zeros((n_stars, 11), dtype=np.uint8))

            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES

        # Create config
        config = {
            "name": "test_for_eval",
            "data": {
                "catalogue_path": str(path),
                "surveys": ["boss"],
                "label_sources": ["apogee"],
                "train_ratio": 0.7,
                "val_ratio": 0.2,
            },
            "model": {
                "hidden_layers": [32, 16],
                "normalization": "layernorm",
                "activation": "gelu",
            },
            "training": {
                "epochs": 2,
                "batch_size": 32,
                "learning_rate": 0.001,
            },
            "output_dir": str(tmp_path / "outputs"),
            "seed": 42,
            "device": "cpu",
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Train the model
        result = main(["train", str(config_path)])
        assert result == 0

        # Find the checkpoint directory
        outputs_dir = tmp_path / "outputs"
        checkpoint_dirs = list(outputs_dir.glob("**/best_model.pth"))
        assert len(checkpoint_dirs) >= 1
        checkpoint_path = checkpoint_dirs[0].parent

        return checkpoint_path

    def test_train_creates_data_split(self, trained_checkpoint):
        """Test that training creates data_split.pkl."""
        split_file = trained_checkpoint / "data_split.pkl"
        assert split_file.exists(), "data_split.pkl should be created during training"

        import pickle

        with open(split_file, "rb") as f:
            split_info = pickle.load(f)

        assert "test_idx" in split_info
        assert "train_idx" in split_info
        assert "val_idx" in split_info
        assert "split_mode" in split_info
        assert "n_total" in split_info
        assert len(split_info["test_idx"]) > 0

    def test_evaluate_with_saved_split(self, trained_checkpoint, capsys):
        """Test evaluate command with saved data split."""
        result = main(["evaluate", str(trained_checkpoint)])
        assert result == 0

        captured = capsys.readouterr()
        assert "Loading saved data split" in captured.out
        assert "Evaluation Results" in captured.out
        assert "RMSE" in captured.out

    def test_evaluate_output_formats(self, trained_checkpoint, tmp_path, capsys):
        """Test evaluate with different output formats."""
        # Text format
        result = main(["evaluate", str(trained_checkpoint), "--format", "text"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Evaluation Results" in captured.out

        # Markdown format
        result = main(["evaluate", str(trained_checkpoint), "--format", "markdown"])
        assert result == 0
        captured = capsys.readouterr()
        assert "## Evaluation Results" in captured.out

    def test_evaluate_saves_output(self, trained_checkpoint, tmp_path, capsys):
        """Test that evaluate saves output to file."""
        output_file = tmp_path / "results.txt"
        result = main(
            [
                "evaluate",
                str(trained_checkpoint),
                "--output",
                str(output_file),
            ]
        )
        assert result == 0

        assert output_file.exists()
        content = output_file.read_text()
        assert "RMSE" in content

    def test_evaluate_json_output(self, trained_checkpoint, tmp_path):
        """Test that evaluate saves JSON output."""
        import json

        output_file = tmp_path / "results.json"
        result = main(
            [
                "evaluate",
                str(trained_checkpoint),
                "--output",
                str(output_file),
                "--format",
                "json",
            ]
        )
        assert result == 0

        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)

        # JSON output now includes both normalized and physical space metrics
        assert "normalized_space" in data
        assert "physical_space" in data
        assert "teff" in data["normalized_space"]
        assert "rmse" in data["normalized_space"]["teff"]
        assert "teff" in data["physical_space"]
        assert "rmse" in data["physical_space"]["teff"]

    def test_evaluate_recreates_split(self, trained_checkpoint, capsys):
        """Test that evaluate can recreate split from seed when data_split.pkl is missing."""
        import os

        # Remove the data split file
        split_file = trained_checkpoint / "data_split.pkl"
        if split_file.exists():
            os.remove(split_file)

        result = main(["evaluate", str(trained_checkpoint)])
        assert result == 0

        captured = capsys.readouterr()
        assert "Recreating from config seed" in captured.out
        assert "Evaluation Results" in captured.out


class TestCLIEvaluateSingleSurvey:
    """Tests for single-survey (variant1-style) evaluation."""

    @pytest.fixture
    def single_survey_checkpoint(self, tmp_path):
        """Create a single-survey trained checkpoint for evaluation tests.

        This uses the single-survey config format (survey: "boss" instead of surveys: ["boss"])
        which triggers the standard MLP architecture.
        """
        # Create mock catalogue
        path = tmp_path / "test_catalogue.h5"
        n_stars = 100
        n_wave = 80

        with h5py.File(path, "w") as f:
            meta = f.create_group("metadata")
            meta.create_dataset("gaia_id", data=np.arange(n_stars, dtype=np.int64))
            meta.create_dataset("ra", data=np.random.uniform(0, 360, n_stars))
            meta.create_dataset("dec", data=np.random.uniform(-90, 90, n_stars))

            surveys = f.create_group("surveys")
            boss = surveys.create_group("boss")
            boss.create_dataset(
                "flux", data=np.random.randn(n_stars, n_wave).astype(np.float32)
            )
            boss.create_dataset(
                "ivar",
                data=np.abs(np.random.randn(n_stars, n_wave)).astype(np.float32) + 0.1,
            )
            boss.create_dataset("wavelength", data=np.linspace(3600, 10400, n_wave))
            boss.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            labels = f.create_group("labels")
            apogee = labels.create_group("apogee")
            label_values = np.random.randn(n_stars, 11).astype(np.float32)
            label_values[:, 0] = np.random.uniform(4000, 6000, n_stars).astype(
                np.float32
            )
            label_values[:, 1] = np.random.uniform(2, 4.5, n_stars).astype(np.float32)
            label_values[:, 2] = np.random.uniform(-1, 0.3, n_stars).astype(np.float32)
            apogee.create_dataset("values", data=label_values)
            apogee.create_dataset(
                "errors",
                data=np.abs(np.random.randn(n_stars, 11)).astype(np.float32) + 0.01,
            )
            apogee.create_dataset("flags", data=np.zeros((n_stars, 11), dtype=np.uint8))

            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES

        # Create single-survey config (variant1-style)
        config = {
            "name": "test_single_survey",
            "data": {
                "catalogue_path": str(path),
                "survey": "boss",  # Singular, not plural
                "label_source": "apogee",  # Singular, not plural
                "train_ratio": 0.7,
                "val_ratio": 0.2,
            },
            "model": {
                "hidden_layers": [32, 16],
                "normalization": "layernorm",
                "activation": "gelu",
            },
            "training": {
                "epochs": 2,
                "batch_size": 32,
                "learning_rate": 0.001,
            },
            "output_dir": str(tmp_path / "outputs"),
            "seed": 42,
            "device": "cpu",
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Train the model
        result = main(["train", str(config_path)])
        assert result == 0

        # Find the checkpoint directory
        outputs_dir = tmp_path / "outputs"
        checkpoint_dirs = list(outputs_dir.glob("**/best_model.pth"))
        assert len(checkpoint_dirs) >= 1
        checkpoint_path = checkpoint_dirs[0].parent

        return checkpoint_path

    def test_evaluate_single_survey_works(self, single_survey_checkpoint, capsys):
        """Test that evaluate works on single-survey (variant1-style) checkpoint."""
        result = main(["evaluate", str(single_survey_checkpoint)])
        assert result == 0

        captured = capsys.readouterr()
        # Should use shared evaluation pipeline
        assert "Using shared evaluation pipeline" in captured.out
        assert "Evaluation Results" in captured.out
        assert "RMSE" in captured.out
        # Should have both normalized and physical space metrics
        assert "NORMALIZED SPACE METRICS" in captured.out
        assert "PHYSICAL SPACE METRICS" in captured.out

    def test_evaluate_single_survey_json_output(
        self, single_survey_checkpoint, tmp_path
    ):
        """Test JSON output has both normalized and physical metrics for single-survey."""
        import json

        output_file = tmp_path / "results.json"
        result = main(
            [
                "evaluate",
                str(single_survey_checkpoint),
                "--output",
                str(output_file),
                "--format",
                "json",
            ]
        )
        assert result == 0

        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)

        # Verify both spaces are present
        assert "normalized_space" in data
        assert "physical_space" in data

        # Verify metrics for key parameters
        assert "teff" in data["normalized_space"]
        assert "rmse" in data["normalized_space"]["teff"]
        assert "z_robust_scatter" in data["normalized_space"]["teff"]

        assert "teff" in data["physical_space"]
        assert "rmse" in data["physical_space"]["teff"]

    def test_evaluate_single_survey_z_scores(self, single_survey_checkpoint, tmp_path):
        """Test that z-scores are reasonable for single-survey evaluation."""
        import json

        output_file = tmp_path / "results.json"
        result = main(
            [
                "evaluate",
                str(single_survey_checkpoint),
                "--output",
                str(output_file),
                "--format",
                "json",
            ]
        )
        assert result == 0

        with open(output_file) as f:
            data = json.load(f)

        # Z-scores should be in a reasonable range (0.5-3.0 for a trained model)
        # With random data they may be higher but should still be finite
        for param in ["teff", "logg", "fe_h"]:
            z_scatter = data["normalized_space"][param]["z_robust_scatter"]
            assert (
                0 < z_scatter < 100
            ), f"z_robust_scatter for {param} should be reasonable"
