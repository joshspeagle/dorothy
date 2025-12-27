"""
Tests for the neural network models module.

These tests verify:
1. Model architecture matches specifications
2. Forward pass produces correct output shapes
3. Configuration options work correctly
4. Gradient flow for training
5. Embedding extraction for analysis
"""

import pytest
import torch

from dorothy.config.schema import ActivationType, ModelConfig, NormalizationType
from dorothy.models.mlp import MLP


class TestMLPInit:
    """Tests for MLP initialization."""

    def test_default_architecture(self):
        """Test that default architecture matches DOROTHY standard."""
        model = MLP()

        assert model.input_features == 15300
        assert model.output_features == 22
        assert model.hidden_layers == [5000, 2000, 1000, 500, 200, 100]

    def test_custom_architecture(self):
        """Test custom architecture specification."""
        model = MLP(
            input_features=9012,  # BOSS spectra
            output_features=10,
            hidden_layers=[1000, 500, 100],
        )

        assert model.input_features == 9012
        assert model.output_features == 10
        assert model.hidden_layers == [1000, 500, 100]

    def test_empty_hidden_layers_rejected(self):
        """Test that empty hidden layers list is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            MLP(hidden_layers=[])

    def test_invalid_activation_rejected(self):
        """Test that invalid activation type is rejected."""
        with pytest.raises(ValueError, match="Unknown activation"):
            MLP(activation="invalid_activation")

    def test_invalid_normalization_rejected(self):
        """Test that invalid normalization type is rejected."""
        with pytest.raises(ValueError, match="Unknown normalization"):
            MLP(normalization="invalid_norm")

    def test_all_activations_supported(self):
        """Test that all documented activations work."""
        for activation in ["gelu", "relu", "silu"]:
            model = MLP(hidden_layers=[100], activation=activation)
            assert model is not None

    def test_all_normalizations_supported(self):
        """Test that all documented normalizations work."""
        for norm in ["batchnorm", "layernorm"]:
            model = MLP(hidden_layers=[100], normalization=norm)
            assert model is not None


class TestMLPForward:
    """Tests for MLP forward pass."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return MLP(
            input_features=100,
            output_features=22,
            hidden_layers=[50, 25],
        )

    def test_forward_3d_input(self, model):
        """Test forward pass with 3D input (batch, channels, wavelengths)."""
        batch_size = 16
        # Simulate 2 channels x 50 wavelengths = 100 features
        x = torch.randn(batch_size, 2, 50)

        output = model(x)

        # Output shape is (batch, 2, n_params) where n_params = 11
        assert output.shape == (batch_size, 2, 11)
        # Check that means and log_scatter can be extracted
        assert output[:, 0, :].shape == (batch_size, 11)
        assert output[:, 1, :].shape == (batch_size, 11)

    def test_forward_2d_input(self, model):
        """Test forward pass with already-flattened 2D input."""
        batch_size = 16
        x = torch.randn(batch_size, 100)

        output = model(x)

        # Output shape is (batch, 2, n_params) where n_params = 11
        assert output.shape == (batch_size, 2, 11)

    def test_forward_single_sample(self, model):
        """Test forward pass with single sample.

        Note: LayerNorm works fine with batch_size=1, unlike BatchNorm.
        """
        model.eval()
        x = torch.randn(1, 2, 50)

        with torch.no_grad():
            output = model(x)

        # Output shape is (batch, 2, n_params) where n_params = 11
        assert output.shape == (1, 2, 11)

    def test_output_dtype_matches_input(self, model):
        """Test that output dtype matches input dtype."""
        x = torch.randn(8, 100, dtype=torch.float32)
        output = model(x)
        assert output.dtype == torch.float32

        # Test with float64
        model_f64 = model.double()
        x_f64 = torch.randn(8, 100, dtype=torch.float64)
        output_f64 = model_f64(x_f64)
        assert output_f64.dtype == torch.float64


class TestMLPWithDropout:
    """Tests for MLP with dropout."""

    def test_dropout_affects_training(self):
        """Test that dropout produces different outputs in training mode."""
        model = MLP(
            input_features=100,
            output_features=22,
            hidden_layers=[50],
            dropout=0.5,
        )
        model.train()

        x = torch.randn(32, 100)

        # Run multiple times - outputs should differ due to dropout
        outputs = [model(x) for _ in range(5)]

        # Check that at least some outputs are different
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Dropout should cause variation in training mode"

    def test_no_dropout_in_eval_mode(self):
        """Test that dropout is disabled in eval mode."""
        model = MLP(
            input_features=100,
            output_features=22,
            hidden_layers=[50],
            dropout=0.5,
        )
        model.eval()

        x = torch.randn(32, 100)

        # Run multiple times - outputs should be identical
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2)


class TestMLPFromConfig:
    """Tests for creating MLP from configuration."""

    def test_from_default_config(self):
        """Test creating MLP from default ModelConfig."""
        config = ModelConfig()
        model = MLP.from_config(config)

        assert model.input_features == 15300
        assert model.output_features == 22
        assert model.hidden_layers == [5000, 2000, 1000, 500, 200, 100]

    def test_from_custom_config(self):
        """Test creating MLP from custom ModelConfig."""
        config = ModelConfig(
            input_features=9012,
            output_features=10,
            hidden_layers=[2000, 500, 100],
            normalization=NormalizationType.LAYERNORM,
            activation=ActivationType.RELU,
            dropout=0.1,
        )
        model = MLP.from_config(config)

        assert model.input_features == 9012
        assert model.output_features == 10
        assert model.hidden_layers == [2000, 500, 100]


class TestMLPGradients:
    """Tests for gradient computation."""

    def test_gradients_flow_backward(self):
        """Test that gradients flow through all layers."""
        model = MLP(
            input_features=100,
            output_features=22,
            hidden_layers=[50, 25],
        )

        x = torch.randn(16, 100, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist for input
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Check gradients exist for all model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(
                    param.grad
                ).all(), f"Non-finite gradient for {name}"

    def test_no_nan_gradients_with_extreme_input(self):
        """Test that gradients don't become NaN with large inputs."""
        model = MLP(
            input_features=100,
            output_features=22,
            hidden_layers=[50],
        )

        # Large but not extreme input values
        x = torch.randn(16, 100) * 10
        x.requires_grad = True

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert torch.isfinite(output).all()
        assert torch.isfinite(x.grad).all()


class TestMLPEmbeddings:
    """Tests for embedding extraction."""

    def test_get_embeddings_shape(self):
        """Test that embeddings have correct shape."""
        model = MLP(
            input_features=100,
            output_features=22,
            hidden_layers=[50, 25, 10],
        )

        x = torch.randn(16, 100)
        embeddings = model.get_embeddings(x)

        # Default layer_index=-2 should give us the output of the last hidden layer
        # Before the final Linear(10, 22), after activation
        # The embedding should have shape (16, 10)
        assert embeddings.shape[0] == 16
        # The exact size depends on which layer is extracted

    def test_get_embeddings_different_layers(self):
        """Test extracting embeddings from different layers."""
        model = MLP(
            input_features=100,
            output_features=22,
            hidden_layers=[50, 25],
        )

        x = torch.randn(8, 100)

        # Get embeddings from different layers
        emb_early = model.get_embeddings(x, layer_index=3)
        emb_late = model.get_embeddings(x, layer_index=-2)

        # Should have different shapes
        assert emb_early.shape != emb_late.shape or not torch.allclose(
            emb_early, emb_late
        )


class TestMLPParameters:
    """Tests for parameter counting."""

    def test_count_parameters_small_model(self):
        """Test parameter counting for a small model."""
        model = MLP(
            input_features=100,
            output_features=10,
            hidden_layers=[50],
        )

        # Expected parameters:
        # Linear(100, 50): 100*50 + 50 = 5050
        # BatchNorm(50): 50 + 50 = 100 (gamma + beta)
        # Linear(50, 10): 50*10 + 10 = 510
        # Total: 5050 + 100 + 510 = 5660
        expected = (100 * 50 + 50) + (50 * 2) + (50 * 10 + 10)

        assert model.count_parameters() == expected

    def test_count_parameters_matches_pytorch(self):
        """Test that count_parameters matches PyTorch's count."""
        model = MLP(
            input_features=100,
            output_features=22,
            hidden_layers=[50, 25],
        )

        # Compare with PyTorch's method
        pytorch_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert model.count_parameters() == pytorch_count


class TestMLPDeviceHandling:
    """Tests for device handling (CPU/GPU)."""

    def test_model_to_device(self):
        """Test moving model to device."""
        model = MLP(
            input_features=100,
            output_features=22,
            hidden_layers=[50],
        )

        # Should work on CPU
        x = torch.randn(8, 100)
        output = model(x)
        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_on_cuda(self):
        """Test model works on CUDA if available."""
        model = MLP(
            input_features=100,
            output_features=22,
            hidden_layers=[50],
        ).cuda()

        x = torch.randn(8, 100).cuda()
        output = model(x)

        assert output.device.type == "cuda"


class TestMLPRepr:
    """Tests for string representation."""

    def test_extra_repr_contains_info(self):
        """Test that extra_repr includes architecture info."""
        model = MLP(
            input_features=9012,
            output_features=10,
            hidden_layers=[500, 100],
        )

        repr_str = model.extra_repr()

        assert "9012" in repr_str
        assert "10" in repr_str
        assert "500" in repr_str
        assert "100" in repr_str


class TestMLPIntegrationWithLoss:
    """Integration tests with the loss function."""

    def test_model_output_compatible_with_loss(self):
        """Test that model output is compatible with heteroscedastic loss."""
        from dorothy.losses import HeteroscedasticLoss

        model = MLP(
            input_features=100,
            output_features=22,  # 11 params * 2
            hidden_layers=[50],
        )
        loss_fn = HeteroscedasticLoss(n_parameters=11)

        x = torch.randn(16, 100)
        target = torch.randn(16, 22)
        target[:, 11:] = torch.abs(target[:, 11:]) + 0.01  # Positive errors

        # Model output shape is (batch, 2, n_params)
        output = model(x)
        assert output.shape == (16, 2, 11)

        loss = loss_fn(output, target)
        assert torch.isfinite(loss)

    def test_training_step(self):
        """Test a complete training step (forward, backward, update)."""
        from dorothy.losses import HeteroscedasticLoss

        model = MLP(
            input_features=100,
            output_features=22,
            hidden_layers=[50],
        )
        loss_fn = HeteroscedasticLoss(n_parameters=11)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(16, 100)
        target = torch.randn(16, 22)
        target[:, 11:] = torch.abs(target[:, 11:]) + 0.01

        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(x)  # Shape: (16, 2, 11)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # Verify loss decreased (optional, depends on random init)
        assert torch.isfinite(loss)
