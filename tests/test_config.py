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
    ModelConfig,
    NormalizationType,
    SchedulerConfig,
    SchedulerType,
    TrainingConfig,
)


class TestDataConfig:
    """Tests for DataConfig validation and properties."""

    def test_default_values(self):
        """Test that default values match DOROTHY conventions."""
        config = DataConfig(catalogue_path=Path("/data/super_catalogue.h5"))

        assert config.surveys == ["boss"]
        assert config.label_sources == ["apogee"]
        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.2
        assert config.max_flag_bits == 0

    def test_test_ratio_computed_correctly(self):
        """Test that test_ratio is computed as 1 - train - val."""
        config = DataConfig(
            catalogue_path=Path("/data/super_catalogue.h5"),
            train_ratio=0.6,
            val_ratio=0.3,
        )

        assert abs(config.test_ratio - 0.1) < 1e-10

    def test_ratios_must_sum_to_less_than_one(self):
        """Test that train + val ratios cannot equal or exceed 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            DataConfig(
                catalogue_path=Path("/data/super_catalogue.h5"),
                train_ratio=0.7,
                val_ratio=0.3,  # Sum = 1.0, no room for test
            )

        assert "must be less than 1.0" in str(exc_info.value)

    def test_ratios_exceeding_one_rejected(self):
        """Test that train + val > 1.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DataConfig(
                catalogue_path=Path("/data/super_catalogue.h5"),
                train_ratio=0.8,
                val_ratio=0.5,
            )

        assert "must be less than 1.0" in str(exc_info.value)

    def test_multiple_surveys(self):
        """Test multi-survey configuration."""
        config = DataConfig(
            catalogue_path=Path("/data/super_catalogue.h5"),
            surveys=["boss", "lamost_lrs", "desi"],
        )

        assert config.surveys == ["boss", "lamost_lrs", "desi"]
        assert config.is_multi_survey is True

    def test_single_survey(self):
        """Test single survey configuration."""
        config = DataConfig(
            catalogue_path=Path("/data/super_catalogue.h5"),
            surveys=["boss"],
        )

        assert config.is_multi_survey is False

    def test_multiple_label_sources(self):
        """Test multi-label source configuration."""
        config = DataConfig(
            catalogue_path=Path("/data/super_catalogue.h5"),
            label_sources=["apogee", "galah"],
        )

        assert config.label_sources == ["apogee", "galah"]
        assert config.is_multi_label is True

    def test_single_label_source(self):
        """Test single label source configuration."""
        config = DataConfig(
            catalogue_path=Path("/data/super_catalogue.h5"),
            label_sources=["apogee"],
        )

        assert config.is_multi_label is False

    def test_empty_surveys_rejected(self):
        """Test that empty surveys list is rejected."""
        with pytest.raises(ValidationError):
            DataConfig(
                catalogue_path=Path("/data/super_catalogue.h5"),
                surveys=[],
            )

    def test_empty_label_sources_rejected(self):
        """Test that empty label_sources list is rejected."""
        with pytest.raises(ValidationError):
            DataConfig(
                catalogue_path=Path("/data/super_catalogue.h5"),
                label_sources=[],
            )


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


class TestExperimentConfig:
    """Tests for the top-level ExperimentConfig."""

    def test_minimal_valid_config(self):
        """Test creating a minimal valid experiment configuration."""
        config = ExperimentConfig(
            name="test_experiment",
            data=DataConfig(catalogue_path=Path("/data/super_catalogue.h5")),
        )

        assert config.name == "test_experiment"
        assert config.seed == 42
        assert config.device == "auto"

    def test_output_path_construction(self):
        """Test output path is constructed correctly."""
        config = ExperimentConfig(
            name="my_experiment",
            data=DataConfig(catalogue_path=Path("/data/super_catalogue.h5")),
            output_dir=Path("/outputs"),
        )

        assert config.get_output_path() == Path("/outputs/my_experiment")

    def test_checkpoint_path_construction(self):
        """Test checkpoint path construction for epochs and final."""
        config = ExperimentConfig(
            name="my_experiment",
            data=DataConfig(catalogue_path=Path("/data/super_catalogue.h5")),
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
                data=DataConfig(catalogue_path=Path("/data/super_catalogue.h5")),
            )

    def test_seed_must_be_non_negative(self):
        """Test that seed must be >= 0."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                name="test",
                data=DataConfig(catalogue_path=Path("/data/super_catalogue.h5")),
                seed=-1,
            )

    def test_multi_survey_config(self):
        """Test experiment with multiple surveys requires multi_head_model."""
        from dorothy.config import MultiHeadModelConfig

        config = ExperimentConfig(
            name="multi_survey_experiment",
            data=DataConfig(
                catalogue_path=Path("/data/super_catalogue.h5"),
                surveys=["boss", "lamost_lrs", "desi"],
                label_sources=["apogee"],
            ),
            multi_head_model=MultiHeadModelConfig(
                survey_wavelengths={"boss": 4506, "lamost_lrs": 3700, "desi": 7650},
            ),
        )

        assert config.data.is_multi_survey is True
        assert config.data.is_multi_label is False
        assert config.is_multi_head is True
        assert config.model is None

    def test_multi_survey_without_multi_head_raises(self):
        """Test that multi-survey without multi_head_model raises error."""
        with pytest.raises(ValueError, match="Multi-survey training requires"):
            ExperimentConfig(
                name="multi_survey_experiment",
                data=DataConfig(
                    catalogue_path=Path("/data/super_catalogue.h5"),
                    surveys=["boss", "lamost_lrs"],
                    label_sources=["apogee"],
                ),
            )

    def test_both_model_configs_raises(self):
        """Test that specifying both model configs raises error."""
        from dorothy.config import MultiHeadModelConfig

        with pytest.raises(ValueError, match="Cannot specify both"):
            ExperimentConfig(
                name="conflicting_config",
                data=DataConfig(
                    catalogue_path=Path("/data/super_catalogue.h5"),
                    surveys=["boss"],
                    label_sources=["apogee"],
                ),
                model=ModelConfig(),
                multi_head_model=MultiHeadModelConfig(
                    survey_wavelengths={"boss": 4506},
                ),
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
