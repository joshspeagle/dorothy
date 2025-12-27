"""
Tests for the configuration schema module.

These tests verify:
1. Default values are sensible and match documented conventions
2. Validation catches invalid inputs with clear error messages
3. Computed properties work correctly
4. Model validators enforce cross-field constraints
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from dorothy.config.schema import (
    STELLAR_PARAMETERS,
    ActivationType,
    DataConfig,
    ExperimentConfig,
    LossType,
    MaskingConfig,
    ModelConfig,
    NormalizationType,
    SchedulerConfig,
    SchedulerType,
    SurveyType,
    TrainingConfig,
)


class TestDataConfig:
    """Tests for DataConfig validation and properties."""

    def test_default_values(self):
        """Test that default values match DOROTHY conventions."""
        config = DataConfig(fits_path=Path("/data/test.fits"))

        assert config.survey == SurveyType.DESI
        assert config.input_channels == 2
        assert config.wavelength_bins == 7650
        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.2
        assert config.quality_filter is True

    def test_test_ratio_computed_correctly(self):
        """Test that test_ratio is computed as 1 - train - val."""
        config = DataConfig(
            fits_path=Path("/data/test.fits"),
            train_ratio=0.6,
            val_ratio=0.3,
        )

        assert abs(config.test_ratio - 0.1) < 1e-10

    def test_ratios_must_sum_to_less_than_one(self):
        """Test that train + val ratios cannot equal or exceed 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            DataConfig(
                fits_path=Path("/data/test.fits"),
                train_ratio=0.7,
                val_ratio=0.3,  # Sum = 1.0, no room for test
            )

        assert "must be less than 1.0" in str(exc_info.value)

    def test_ratios_exceeding_one_rejected(self):
        """Test that train + val > 1.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DataConfig(
                fits_path=Path("/data/test.fits"),
                train_ratio=0.8,
                val_ratio=0.5,
            )

        assert "must be less than 1.0" in str(exc_info.value)

    def test_invalid_survey_type_rejected(self):
        """Test that invalid survey types are rejected."""
        with pytest.raises(ValidationError):
            DataConfig(
                fits_path=Path("/data/test.fits"),
                survey="invalid_survey",
            )

    def test_wavelength_bins_bounds(self):
        """Test wavelength_bins validation bounds."""
        # Too small
        with pytest.raises(ValidationError):
            DataConfig(fits_path=Path("/data/test.fits"), wavelength_bins=100)

        # Too large
        with pytest.raises(ValidationError):
            DataConfig(fits_path=Path("/data/test.fits"), wavelength_bins=50000)

        # Valid BOSS value
        config = DataConfig(fits_path=Path("/data/test.fits"), wavelength_bins=4506)
        assert config.wavelength_bins == 4506


class TestModelConfig:
    """Tests for ModelConfig validation."""

    def test_default_architecture(self):
        """Test that default architecture matches DOROTHY standard."""
        config = ModelConfig()

        assert config.hidden_layers == [5000, 2000, 1000, 500, 200, 100]
        assert config.normalization == NormalizationType.LAYERNORM
        assert config.activation == ActivationType.GELU
        assert config.dropout == 0.0
        assert config.input_features == 15300  # 2 * 7650
        assert config.output_features == 22  # 11 params * 2

    def test_n_parameters_computed(self):
        """Test that n_parameters is correctly computed from output_features."""
        config = ModelConfig(output_features=22)
        assert config.n_parameters == 11

        config = ModelConfig(output_features=10)
        assert config.n_parameters == 5

    def test_negative_hidden_layer_size_rejected(self):
        """Test that negative hidden layer sizes are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(hidden_layers=[5000, -100, 500])

        assert "must be positive" in str(exc_info.value)

    def test_zero_hidden_layer_size_rejected(self):
        """Test that zero hidden layer size is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(hidden_layers=[5000, 0, 500])

        assert "must be positive" in str(exc_info.value)

    def test_empty_hidden_layers_rejected(self):
        """Test that empty hidden layers list is rejected."""
        with pytest.raises(ValidationError):
            ModelConfig(hidden_layers=[])

    def test_dropout_bounds(self):
        """Test dropout probability bounds."""
        # Valid dropout
        config = ModelConfig(dropout=0.5)
        assert config.dropout == 0.5

        # Negative dropout rejected
        with pytest.raises(ValidationError):
            ModelConfig(dropout=-0.1)

        # Dropout >= 1.0 rejected
        with pytest.raises(ValidationError):
            ModelConfig(dropout=1.0)


class TestSchedulerConfig:
    """Tests for SchedulerConfig validation."""

    def test_default_one_cycle_scheduler(self):
        """Test default one_cycle scheduler configuration."""
        config = SchedulerConfig()

        assert config.type == SchedulerType.ONE_CYCLE
        assert config.max_lr == 1e-3
        assert config.pct_start == 0.3
        assert config.div_factor == 25.0
        assert config.final_div_factor == 1e4
        assert config.anneal_strategy == "cos"

    def test_base_lr_cannot_exceed_max_lr(self):
        """Test that base_lr must be <= max_lr."""
        with pytest.raises(ValidationError) as exc_info:
            SchedulerConfig(base_lr=1e-2, max_lr=1e-4)

        assert "must be <=" in str(exc_info.value)

    def test_equal_base_and_max_lr_allowed(self):
        """Test that base_lr == max_lr is allowed (constant LR)."""
        config = SchedulerConfig(base_lr=1e-3, max_lr=1e-3)
        assert config.base_lr == config.max_lr


class TestTrainingConfig:
    """Tests for TrainingConfig validation."""

    def test_default_values(self):
        """Test default training configuration."""
        config = TrainingConfig()

        assert config.epochs == 300
        assert config.batch_size == 1024
        assert config.learning_rate == 1e-3
        assert config.loss == LossType.HETEROSCEDASTIC
        assert config.gradient_clip == 10.0
        assert config.scatter_floor == 0.01

    def test_scatter_floor_for_heteroscedastic_loss(self):
        """Test that scatter_floor (s_0) is validated correctly."""
        # Valid scatter floor
        config = TrainingConfig(scatter_floor=0.05)
        assert config.scatter_floor == 0.05

        # Negative rejected
        with pytest.raises(ValidationError):
            TrainingConfig(scatter_floor=-0.01)

        # Too large rejected
        with pytest.raises(ValidationError):
            TrainingConfig(scatter_floor=2.0)


class TestMaskingConfig:
    """Tests for MaskingConfig validation."""

    def test_disabled_by_default(self):
        """Test that masking is disabled by default."""
        config = MaskingConfig()
        assert config.enabled is False

    def test_block_size_validation(self):
        """Test that min_block_size is valid."""
        # Valid configuration
        config = MaskingConfig(min_block_size=10, max_block_size=100)
        assert config.min_block_size == 10
        assert config.max_block_size == 100

        # Invalid: min_block_size < 1
        with pytest.raises(ValidationError):
            MaskingConfig(min_block_size=0)

    def test_fraction_validation(self):
        """Test min_fraction <= max_fraction."""
        # Valid fraction range
        config = MaskingConfig(min_fraction=0.2, max_fraction=0.5)
        assert config.min_fraction == 0.2
        assert config.max_fraction == 0.5

        # Invalid: min > max
        with pytest.raises(ValidationError) as exc_info:
            MaskingConfig(min_fraction=0.6, max_fraction=0.4)

        assert "must be <=" in str(exc_info.value)

        # Valid: full range allowed
        config = MaskingConfig(min_fraction=0.0, max_fraction=1.0)
        assert config.min_fraction == 0.0
        assert config.max_fraction == 1.0

    def test_fraction_choices_validation(self):
        """Test fraction_choices validation."""
        # Valid fraction choices
        config = MaskingConfig(fraction_choices=[0.1, 0.3, 0.5])
        assert config.fraction_choices == [0.1, 0.3, 0.5]

        # Invalid: fraction choice > 1
        with pytest.raises(ValidationError):
            MaskingConfig(fraction_choices=[0.1, 1.5])


class TestExperimentConfig:
    """Tests for the top-level ExperimentConfig."""

    def test_minimal_valid_config(self):
        """Test creating a minimal valid experiment configuration."""
        config = ExperimentConfig(
            name="test_experiment",
            data=DataConfig(fits_path=Path("/data/test.fits")),
        )

        assert config.name == "test_experiment"
        assert config.seed == 42
        assert config.device == "auto"

    def test_input_features_synced_with_data(self):
        """Test that model input_features is synced with data config."""
        config = ExperimentConfig(
            name="test",
            data=DataConfig(
                fits_path=Path("/data/test.fits"),
                input_channels=2,
                wavelength_bins=4506,  # BOSS wavelengths
            ),
        )

        # Model should have 2 * 4506 = 9012 input features
        assert config.model.input_features == 9012

    def test_output_path_construction(self):
        """Test output path is constructed correctly."""
        config = ExperimentConfig(
            name="my_experiment",
            data=DataConfig(fits_path=Path("/data/test.fits")),
            output_dir=Path("/outputs"),
        )

        assert config.get_output_path() == Path("/outputs/my_experiment")

    def test_checkpoint_path_construction(self):
        """Test checkpoint path construction for epochs and final."""
        config = ExperimentConfig(
            name="my_experiment",
            data=DataConfig(fits_path=Path("/data/test.fits")),
            output_dir=Path("/outputs"),
        )

        # Final checkpoint
        assert config.get_checkpoint_path() == Path(
            "/outputs/my_experiment/my_experiment_final"
        )

        # Epoch checkpoint
        assert config.get_checkpoint_path(epoch=20) == Path(
            "/outputs/my_experiment/epoch_20"
        )

    def test_empty_name_rejected(self):
        """Test that empty experiment name is rejected."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                name="",
                data=DataConfig(fits_path=Path("/data/test.fits")),
            )

    def test_seed_must_be_non_negative(self):
        """Test that seed must be >= 0."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                name="test",
                data=DataConfig(fits_path=Path("/data/test.fits")),
                seed=-1,
            )


class TestStellarParameters:
    """Tests for the stellar parameters constant."""

    def test_eleven_parameters(self):
        """Test that we have exactly 11 stellar parameters."""
        assert len(STELLAR_PARAMETERS) == 11

    def test_teff_is_first(self):
        """Test that Teff is the first parameter (important for log normalization)."""
        assert STELLAR_PARAMETERS[0] == "teff"

    def test_expected_parameters_present(self):
        """Test that all expected parameters are in the list."""
        expected = {
            "teff",
            "logg",
            "feh",
            "mgfe",
            "cfe",
            "sife",
            "nife",
            "alfe",
            "cafe",
            "nfe",
            "mnfe",
        }
        assert set(STELLAR_PARAMETERS) == expected
