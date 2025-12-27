"""Tests for multi-head MLP architecture."""

import pytest
import torch

from dorothy.config import CombinationMode, MultiHeadModelConfig
from dorothy.models import MultiHeadMLP, OutputHead, SharedTrunk, SurveyEncoder


class TestSurveyEncoder:
    """Tests for individual survey encoders."""

    def test_encoder_creation(self):
        """Test encoder can be created with default settings."""
        encoder = SurveyEncoder(
            survey_name="boss",
            n_wavelengths=4506,
            latent_dim=256,
        )
        assert encoder.survey_name == "boss"
        assert encoder.n_wavelengths == 4506
        assert encoder.latent_dim == 256

    def test_encoder_forward_shape(self):
        """Test encoder produces correct output shape."""
        encoder = SurveyEncoder(
            survey_name="test",
            n_wavelengths=100,
            latent_dim=64,
        )
        x = torch.randn(32, 3, 100)
        output = encoder(x)
        assert output.shape == (32, 64)

    def test_encoder_custom_hidden_layers(self):
        """Test encoder with custom hidden layers."""
        encoder = SurveyEncoder(
            survey_name="test",
            n_wavelengths=100,
            latent_dim=64,
            hidden_layers=[256, 128],
        )
        x = torch.randn(16, 3, 100)
        output = encoder(x)
        assert output.shape == (16, 64)

    def test_encoder_batchnorm(self):
        """Test encoder with batch normalization."""
        encoder = SurveyEncoder(
            survey_name="test",
            n_wavelengths=100,
            latent_dim=64,
            normalization="batchnorm",
        )
        x = torch.randn(32, 3, 100)
        encoder.train()
        output = encoder(x)
        assert output.shape == (32, 64)

    def test_encoder_dropout(self):
        """Test encoder with dropout."""
        encoder = SurveyEncoder(
            survey_name="test",
            n_wavelengths=100,
            latent_dim=64,
            dropout=0.2,
        )
        x = torch.randn(32, 3, 100)
        encoder.train()
        output = encoder(x)
        assert output.shape == (32, 64)

    def test_encoder_invalid_activation(self):
        """Test encoder raises error for invalid activation."""
        with pytest.raises(ValueError, match="Unknown activation"):
            SurveyEncoder(
                survey_name="test",
                n_wavelengths=100,
                latent_dim=64,
                activation="invalid",
            )

    def test_encoder_invalid_normalization(self):
        """Test encoder raises error for invalid normalization."""
        with pytest.raises(ValueError, match="Unknown normalization"):
            SurveyEncoder(
                survey_name="test",
                n_wavelengths=100,
                latent_dim=64,
                normalization="invalid",
            )


class TestSharedTrunk:
    """Tests for shared trunk network."""

    def test_trunk_creation(self):
        """Test trunk can be created with default settings."""
        trunk = SharedTrunk(input_dim=256, output_dim=128)
        assert trunk.input_dim == 256
        assert trunk.output_dim == 128

    def test_trunk_forward_shape(self):
        """Test trunk produces correct output shape."""
        trunk = SharedTrunk(input_dim=256, output_dim=128)
        x = torch.randn(32, 256)
        output = trunk(x)
        assert output.shape == (32, 128)

    def test_trunk_custom_hidden_layers(self):
        """Test trunk with custom hidden layers."""
        trunk = SharedTrunk(
            input_dim=256,
            output_dim=128,
            hidden_layers=[512, 256, 128],
        )
        x = torch.randn(16, 256)
        output = trunk(x)
        assert output.shape == (16, 128)


class TestOutputHead:
    """Tests for output head."""

    def test_output_head_creation(self):
        """Test output head can be created with default settings."""
        head = OutputHead(input_dim=128, n_parameters=11)
        assert head.input_dim == 128
        assert head.n_parameters == 11
        assert head.output_features == 22

    def test_output_head_forward_shape(self):
        """Test output head produces correct output shape."""
        head = OutputHead(input_dim=128, n_parameters=11)
        x = torch.randn(32, 128)
        output = head(x)
        assert output.shape == (32, 2, 11)

    def test_output_head_custom_parameters(self):
        """Test output head with custom number of parameters."""
        head = OutputHead(input_dim=64, n_parameters=5)
        x = torch.randn(16, 64)
        output = head(x)
        assert output.shape == (16, 2, 5)


class TestMultiHeadMLP:
    """Tests for the full multi-head MLP architecture."""

    @pytest.fixture
    def single_survey_model(self):
        """Create model with single survey."""
        return MultiHeadMLP(
            survey_configs={"boss": 100},
            n_parameters=11,
            latent_dim=64,
            encoder_hidden=[128],
            trunk_hidden=[64],
            output_hidden=[32],
        )

    @pytest.fixture
    def multi_survey_model_mean(self):
        """Create model with multiple surveys using mean combination."""
        return MultiHeadMLP(
            survey_configs={"boss": 100, "lamost": 80},
            n_parameters=11,
            latent_dim=64,
            encoder_hidden=[128],
            trunk_hidden=[64],
            output_hidden=[32],
            combination_mode="mean",
        )

    @pytest.fixture
    def multi_survey_model_concat(self):
        """Create model with multiple surveys using concat combination."""
        return MultiHeadMLP(
            survey_configs={"boss": 100, "lamost": 80},
            n_parameters=11,
            latent_dim=64,
            encoder_hidden=[128],
            trunk_hidden=[64],
            output_hidden=[32],
            combination_mode="concat",
        )

    def test_model_creation_single_survey(self, single_survey_model):
        """Test model creation with single survey."""
        assert single_survey_model.survey_names == ["boss"]
        assert single_survey_model.n_surveys == 1
        assert "boss" in single_survey_model.encoders

    def test_model_creation_multi_survey(self, multi_survey_model_mean):
        """Test model creation with multiple surveys."""
        assert set(multi_survey_model_mean.survey_names) == {"boss", "lamost"}
        assert multi_survey_model_mean.n_surveys == 2
        assert "boss" in multi_survey_model_mean.encoders
        assert "lamost" in multi_survey_model_mean.encoders

    def test_model_empty_configs_raises(self):
        """Test model raises error for empty survey configs."""
        with pytest.raises(ValueError, match="survey_configs cannot be empty"):
            MultiHeadMLP(survey_configs={})

    def test_model_invalid_combination_mode(self):
        """Test model raises error for invalid combination mode."""
        with pytest.raises(ValueError, match="combination_mode must be"):
            MultiHeadMLP(
                survey_configs={"boss": 100},
                combination_mode="invalid",
            )

    def test_forward_single_survey(self, single_survey_model):
        """Test forward pass with single survey."""
        x = torch.randn(32, 3, 100)
        output = single_survey_model.forward_single(x, "boss")
        assert output.shape == (32, 2, 11)

    def test_forward_single_unknown_survey(self, single_survey_model):
        """Test forward_single raises error for unknown survey."""
        x = torch.randn(32, 3, 100)
        with pytest.raises(KeyError, match="Unknown survey"):
            single_survey_model.forward_single(x, "unknown")

    def test_forward_single_inferred_survey(self, single_survey_model):
        """Test forward() infers survey when only one available."""
        x = torch.randn(32, 3, 100)
        output = single_survey_model.forward(x)
        assert output.shape == (32, 2, 11)

    def test_forward_multi_requires_survey_name(self, multi_survey_model_mean):
        """Test forward() with tensor requires survey for multi-survey model."""
        x = torch.randn(32, 3, 100)
        with pytest.raises(ValueError, match="Must specify survey"):
            multi_survey_model_mean.forward(x)

    def test_forward_multi_mean_combination(self, multi_survey_model_mean):
        """Test forward_multi with mean combination."""
        inputs = {
            "boss": torch.randn(32, 3, 100),
            "lamost": torch.randn(32, 3, 80),
        }
        has_data = {
            "boss": torch.ones(32, dtype=torch.bool),
            "lamost": torch.ones(32, dtype=torch.bool),
        }
        output = multi_survey_model_mean.forward_multi(inputs, has_data)
        assert output.shape == (32, 2, 11)

    def test_forward_multi_concat_combination(self, multi_survey_model_concat):
        """Test forward_multi with concat combination."""
        inputs = {
            "boss": torch.randn(32, 3, 100),
            "lamost": torch.randn(32, 3, 80),
        }
        has_data = {
            "boss": torch.ones(32, dtype=torch.bool),
            "lamost": torch.ones(32, dtype=torch.bool),
        }
        output = multi_survey_model_concat.forward_multi(inputs, has_data)
        assert output.shape == (32, 2, 11)

    def test_forward_multi_partial_data_mean(self, multi_survey_model_mean):
        """Test forward_multi handles partial data with mean combination."""
        inputs = {
            "boss": torch.randn(32, 3, 100),
            "lamost": torch.randn(32, 3, 80),
        }
        # Only half the samples have boss data, all have lamost
        has_data = {
            "boss": torch.cat([torch.ones(16), torch.zeros(16)]).bool(),
            "lamost": torch.ones(32, dtype=torch.bool),
        }
        output = multi_survey_model_mean.forward_multi(inputs, has_data)
        assert output.shape == (32, 2, 11)

    def test_forward_multi_partial_data_concat(self, multi_survey_model_concat):
        """Test forward_multi handles partial data with concat combination."""
        inputs = {
            "boss": torch.randn(32, 3, 100),
            "lamost": torch.randn(32, 3, 80),
        }
        has_data = {
            "boss": torch.cat([torch.ones(16), torch.zeros(16)]).bool(),
            "lamost": torch.ones(32, dtype=torch.bool),
        }
        output = multi_survey_model_concat.forward_multi(inputs, has_data)
        assert output.shape == (32, 2, 11)

    def test_forward_unified_with_dict(self, multi_survey_model_mean):
        """Test forward() with dict input dispatches to forward_multi."""
        inputs = {
            "boss": torch.randn(32, 3, 100),
            "lamost": torch.randn(32, 3, 80),
        }
        output = multi_survey_model_mean.forward(inputs)
        assert output.shape == (32, 2, 11)

    def test_forward_unified_with_tensor(self, multi_survey_model_mean):
        """Test forward() with tensor and survey name dispatches to forward_single."""
        x = torch.randn(32, 3, 100)
        output = multi_survey_model_mean.forward(x, survey="boss")
        assert output.shape == (32, 2, 11)

    def test_get_embeddings(self, single_survey_model):
        """Test get_embeddings returns correct shape."""
        x = torch.randn(32, 3, 100)
        embeddings = single_survey_model.get_embeddings(x, "boss")
        assert embeddings.shape == (32, 64)

    def test_count_parameters(self, single_survey_model):
        """Test count_parameters returns positive integer."""
        count = single_survey_model.count_parameters()
        assert count > 0
        assert isinstance(count, int)

    def test_count_parameters_by_component(self, multi_survey_model_mean):
        """Test count_parameters_by_component returns breakdown."""
        counts = multi_survey_model_mean.count_parameters_by_component()
        assert "encoder_boss" in counts
        assert "encoder_lamost" in counts
        assert "trunk" in counts
        assert "output_head" in counts
        assert "total" in counts
        assert counts["total"] == multi_survey_model_mean.count_parameters()

    def test_extra_repr(self, multi_survey_model_mean):
        """Test extra_repr contains key info."""
        repr_str = multi_survey_model_mean.extra_repr()
        assert "boss" in repr_str
        assert "lamost" in repr_str
        assert "mean" in repr_str

    def test_single_survey_concat_mode(self):
        """Test single survey with concat mode works correctly."""
        model = MultiHeadMLP(
            survey_configs={"boss": 100},
            n_parameters=11,
            latent_dim=64,
            combination_mode="concat",
        )
        x = torch.randn(32, 3, 100)
        output = model.forward_single(x, "boss")
        assert output.shape == (32, 2, 11)

    def test_gradient_flow(self, single_survey_model):
        """Test gradients flow through the model."""
        x = torch.randn(32, 3, 100, requires_grad=True)
        output = single_survey_model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_train_eval_modes(self, single_survey_model):
        """Test model behaves correctly in train and eval modes."""
        x = torch.randn(32, 3, 100)

        single_survey_model.train()
        train_output = single_survey_model(x)

        single_survey_model.eval()
        with torch.no_grad():
            eval_output = single_survey_model(x)

        assert train_output.shape == eval_output.shape


class TestMultiHeadModelConfig:
    """Tests for MultiHeadModelConfig."""

    def test_config_defaults(self):
        """Test config creates with defaults."""
        config = MultiHeadModelConfig(
            survey_wavelengths={"boss": 4506},
        )
        assert config.n_parameters == 11
        assert config.latent_dim == 256
        assert config.combination_mode == CombinationMode.CONCAT

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = MultiHeadModelConfig(
            survey_wavelengths={"boss": 4506, "lamost": 3700},
            n_parameters=5,
            latent_dim=128,
            encoder_hidden=[512, 256],
            trunk_hidden=[256, 128],
            output_hidden=[32],
            combination_mode=CombinationMode.CONCAT,
            dropout=0.1,
        )
        assert config.n_parameters == 5
        assert config.latent_dim == 128
        assert config.combination_mode == CombinationMode.CONCAT
        assert config.dropout == 0.1

    def test_config_to_model(self):
        """Test creating model from config."""
        config = MultiHeadModelConfig(
            survey_wavelengths={"boss": 100, "lamost": 80},
            n_parameters=11,
            latent_dim=64,
            encoder_hidden=[128],
            trunk_hidden=[64],
            output_hidden=[32],
            combination_mode=CombinationMode.MEAN,
        )
        model = MultiHeadMLP(
            survey_configs=config.survey_wavelengths,
            n_parameters=config.n_parameters,
            latent_dim=config.latent_dim,
            encoder_hidden=config.encoder_hidden,
            trunk_hidden=config.trunk_hidden,
            output_hidden=config.output_hidden,
            combination_mode=config.combination_mode.value,
            normalization=config.normalization.value,
            activation=config.activation.value,
            dropout=config.dropout,
        )
        x = torch.randn(32, 3, 100)
        output = model.forward(x, survey="boss")
        assert output.shape == (32, 2, 11)


class TestMultiHeadMLPWithHeteroscedasticLoss:
    """Tests for MultiHeadMLP integration with heteroscedastic loss."""

    def test_output_compatible_with_loss(self):
        """Test model output is compatible with heteroscedastic loss."""
        from dorothy.losses import HeteroscedasticLoss

        model = MultiHeadMLP(
            survey_configs={"boss": 100},
            n_parameters=11,
        )
        loss_fn = HeteroscedasticLoss(n_parameters=11)

        x = torch.randn(32, 3, 100)
        output = model(x)

        # Create target in 3-channel format
        target = torch.randn(32, 3, 11)
        target[:, 2, :] = 1.0  # All valid

        loss = loss_fn(output, target)
        assert loss.shape == ()
        assert torch.isfinite(loss)


class TestMultiLabelsetOutputs:
    """Tests for multi-labelset output head functionality."""

    @pytest.fixture
    def single_label_model(self):
        """Create model with single label source (default)."""
        return MultiHeadMLP(
            survey_configs={"boss": 100},
            n_parameters=11,
            latent_dim=64,
            encoder_hidden=[128],
            trunk_hidden=[64],
            output_hidden=[32],
        )

    @pytest.fixture
    def multi_label_model(self):
        """Create model with multiple label sources."""
        return MultiHeadMLP(
            survey_configs={"boss": 100, "lamost": 80},
            n_parameters=11,
            latent_dim=64,
            encoder_hidden=[128],
            trunk_hidden=[64],
            output_hidden=[32],
            label_sources=["apogee", "galah"],
        )

    def test_single_label_defaults(self, single_label_model):
        """Test single label source uses default."""
        assert single_label_model.label_sources == ["default"]
        assert single_label_model.n_label_sources == 1
        assert not single_label_model.is_multi_label
        assert "default" in single_label_model.output_heads

    def test_multi_label_creation(self, multi_label_model):
        """Test multi-label model has correct output heads."""
        assert multi_label_model.label_sources == ["apogee", "galah"]
        assert multi_label_model.n_label_sources == 2
        assert multi_label_model.is_multi_label
        assert "apogee" in multi_label_model.output_heads
        assert "galah" in multi_label_model.output_heads

    def test_forward_for_label_source_single_survey(self, multi_label_model):
        """Test forward_for_label_source with single survey input."""
        x = torch.randn(32, 3, 100)
        output = multi_label_model.forward_for_label_source(x, "apogee", survey="boss")
        assert output.shape == (32, 2, 11)

    def test_forward_for_label_source_all_sources(self, multi_label_model):
        """Test forward_for_label_source works for each label source."""
        x = torch.randn(32, 3, 100)
        for source in ["apogee", "galah"]:
            output = multi_label_model.forward_for_label_source(
                x, source, survey="boss"
            )
            assert output.shape == (32, 2, 11)

    def test_forward_for_label_source_unknown_raises(self, multi_label_model):
        """Test forward_for_label_source raises for unknown label source."""
        x = torch.randn(32, 3, 100)
        with pytest.raises(KeyError, match="Unknown label source"):
            multi_label_model.forward_for_label_source(x, "unknown", survey="boss")

    def test_forward_for_label_source_dict_input(self, multi_label_model):
        """Test forward_for_label_source with dict input (multi-survey)."""
        inputs = {
            "boss": torch.randn(32, 3, 100),
            "lamost": torch.randn(32, 3, 80),
        }
        output = multi_label_model.forward_for_label_source(inputs, "galah")
        assert output.shape == (32, 2, 11)

    def test_forward_all_label_sources_single_survey(self, multi_label_model):
        """Test forward_all_label_sources returns dict of outputs."""
        x = torch.randn(32, 3, 100)
        outputs = multi_label_model.forward_all_label_sources(x, survey="boss")
        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == {"apogee", "galah"}
        for _source, output in outputs.items():
            assert output.shape == (32, 2, 11)

    def test_forward_all_label_sources_dict_input(self, multi_label_model):
        """Test forward_all_label_sources with dict input."""
        inputs = {
            "boss": torch.randn(32, 3, 100),
            "lamost": torch.randn(32, 3, 80),
        }
        outputs = multi_label_model.forward_all_label_sources(inputs)
        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == {"apogee", "galah"}

    def test_forward_all_single_label_model(self, single_label_model):
        """Test forward_all_label_sources on single label model."""
        x = torch.randn(32, 3, 100)
        outputs = single_label_model.forward_all_label_sources(x)
        assert set(outputs.keys()) == {"default"}
        assert outputs["default"].shape == (32, 2, 11)

    def test_default_forward_uses_first_label_source(self, multi_label_model):
        """Test default forward() uses first label source (apogee)."""
        x = torch.randn(32, 3, 100)
        default_output = multi_label_model.forward(x, survey="boss")
        apogee_output = multi_label_model.forward_for_label_source(
            x, "apogee", survey="boss"
        )
        assert torch.allclose(default_output, apogee_output)

    def test_count_parameters_multi_label(self, multi_label_model):
        """Test count_parameters_by_component with multiple output heads."""
        counts = multi_label_model.count_parameters_by_component()
        assert "output_head_apogee" in counts
        assert "output_head_galah" in counts
        assert "trunk" in counts
        assert "total" in counts
        # Both output heads should have same number of params
        assert counts["output_head_apogee"] == counts["output_head_galah"]

    def test_count_parameters_single_label(self, single_label_model):
        """Test count_parameters_by_component with single output head."""
        counts = single_label_model.count_parameters_by_component()
        assert "output_head" in counts
        assert "trunk" in counts
        assert "total" in counts

    def test_extra_repr_multi_label(self, multi_label_model):
        """Test extra_repr shows label sources."""
        repr_str = multi_label_model.extra_repr()
        assert "apogee" in repr_str
        assert "galah" in repr_str

    def test_gradient_flow_multi_label(self, multi_label_model):
        """Test gradients flow through all label output heads."""
        x = torch.randn(32, 3, 100, requires_grad=True)
        outputs = multi_label_model.forward_all_label_sources(x, survey="boss")

        # Sum all outputs and backprop
        total_loss = sum(out.sum() for out in outputs.values())
        total_loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_output_heads_independent(self, multi_label_model):
        """Test output heads produce different outputs (not shared weights)."""
        x = torch.randn(32, 3, 100)
        outputs = multi_label_model.forward_all_label_sources(x, survey="boss")

        # Different output heads should produce different outputs
        # (unless by extreme coincidence)
        assert not torch.allclose(outputs["apogee"], outputs["galah"])


class TestMultiHeadModelConfigLabelSources:
    """Tests for MultiHeadModelConfig with label_sources."""

    def test_config_no_label_sources(self):
        """Test config without label_sources."""
        config = MultiHeadModelConfig(
            survey_wavelengths={"boss": 4506},
        )
        assert config.label_sources is None
        assert not config.is_multi_label

    def test_config_single_label_source(self):
        """Test config with single label source."""
        config = MultiHeadModelConfig(
            survey_wavelengths={"boss": 4506},
            label_sources=["apogee"],
        )
        assert config.label_sources == ["apogee"]
        assert not config.is_multi_label

    def test_config_multi_label_sources(self):
        """Test config with multiple label sources."""
        config = MultiHeadModelConfig(
            survey_wavelengths={"boss": 4506},
            label_sources=["apogee", "galah"],
        )
        assert config.label_sources == ["apogee", "galah"]
        assert config.is_multi_label

    def test_config_to_model_with_label_sources(self):
        """Test creating model from config with label_sources."""
        config = MultiHeadModelConfig(
            survey_wavelengths={"boss": 100},
            n_parameters=11,
            latent_dim=64,
            label_sources=["apogee", "galah"],
        )
        model = MultiHeadMLP(
            survey_configs=config.survey_wavelengths,
            n_parameters=config.n_parameters,
            latent_dim=config.latent_dim,
            label_sources=config.label_sources,
        )
        assert model.label_sources == ["apogee", "galah"]
        assert model.is_multi_label
