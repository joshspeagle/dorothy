"""
Tests for the loss functions module.

These tests verify:
1. Mathematical correctness of the heteroscedastic loss
2. Proper handling of edge cases
3. Correct behavior of reduction modes
4. Gradient flow for backpropagation
"""

import math

import pytest
import torch

from dorothy.losses.heteroscedastic import HeteroscedasticLoss


class TestHeteroscedasticLossInit:
    """Tests for HeteroscedasticLoss initialization."""

    def test_default_initialization(self):
        """Test default initialization matches DOROTHY conventions."""
        loss_fn = HeteroscedasticLoss()

        assert loss_fn.scatter_floor == 0.01
        assert loss_fn.n_parameters == 11
        assert loss_fn.reduction == "mean"

    def test_custom_initialization(self):
        """Test custom initialization."""
        loss_fn = HeteroscedasticLoss(
            scatter_floor=0.05,
            n_parameters=5,
            reduction="sum",
        )

        assert loss_fn.scatter_floor == 0.05
        assert loss_fn.n_parameters == 5
        assert loss_fn.reduction == "sum"

    def test_invalid_scatter_floor_rejected(self):
        """Test that non-positive scatter floor is rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            HeteroscedasticLoss(scatter_floor=0.0)

        with pytest.raises(ValueError, match="must be positive"):
            HeteroscedasticLoss(scatter_floor=-0.01)

    def test_invalid_n_parameters_rejected(self):
        """Test that non-positive n_parameters is rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            HeteroscedasticLoss(n_parameters=0)

        with pytest.raises(ValueError, match="must be positive"):
            HeteroscedasticLoss(n_parameters=-1)

    def test_invalid_reduction_rejected(self):
        """Test that invalid reduction mode is rejected."""
        with pytest.raises(ValueError, match="must be 'none', 'mean', or 'sum'"):
            HeteroscedasticLoss(reduction="average")


class TestHeteroscedasticLossForward:
    """Tests for HeteroscedasticLoss forward pass."""

    @pytest.fixture
    def loss_fn(self):
        """Create a loss function for testing."""
        return HeteroscedasticLoss(scatter_floor=0.01, n_parameters=3)

    def test_output_shape_mean_reduction(self, loss_fn):
        """Test output is scalar with mean reduction."""
        output = torch.randn(32, 6)  # 3 means + 3 log-scatters
        target = torch.randn(32, 6)  # 3 labels + 3 errors (use abs for errors)
        target[:, 3:] = torch.abs(target[:, 3:]) + 0.01

        loss = loss_fn(output, target)

        assert loss.shape == ()
        assert loss.ndim == 0

    def test_output_shape_sum_reduction(self):
        """Test output is scalar with sum reduction."""
        loss_fn = HeteroscedasticLoss(n_parameters=3, reduction="sum")
        output = torch.randn(32, 6)
        target = torch.randn(32, 6)
        target[:, 3:] = torch.abs(target[:, 3:]) + 0.01

        loss = loss_fn(output, target)

        assert loss.shape == ()
        assert loss.ndim == 0

    def test_output_shape_no_reduction(self):
        """Test output shape with no reduction."""
        loss_fn = HeteroscedasticLoss(n_parameters=3, reduction="none")
        batch_size = 32
        output = torch.randn(batch_size, 6)
        target = torch.randn(batch_size, 6)
        target[:, 3:] = torch.abs(target[:, 3:]) + 0.01

        loss = loss_fn(output, target)

        assert loss.shape == (batch_size, 3)

    def test_invalid_output_dimension_rejected(self, loss_fn):
        """Test that incorrect output dimension is rejected."""
        output = torch.randn(32, 8)  # Wrong size, should be 6
        target = torch.randn(32, 6)

        with pytest.raises(ValueError, match="Output last dimension"):
            loss_fn(output, target)

    def test_invalid_target_dimension_rejected(self, loss_fn):
        """Test that incorrect target dimension is rejected."""
        output = torch.randn(32, 6)
        target = torch.randn(32, 8)  # Wrong size, should be 6

        with pytest.raises(ValueError, match="Target last dimension"):
            loss_fn(output, target)


class TestHeteroscedasticLossMathematics:
    """Tests for mathematical correctness of the loss."""

    def test_perfect_prediction_low_loss(self):
        """Test that perfect predictions with low uncertainty give low loss."""
        loss_fn = HeteroscedasticLoss(scatter_floor=0.01, n_parameters=2)

        # Perfect prediction: mu = y
        output = torch.tensor([[1.0, 2.0, -3.0, -3.0]])  # means=1,2, ln_s=-3,-3
        target = torch.tensor([[1.0, 2.0, 0.1, 0.1]])  # y=1,2, sigma=0.1

        loss = loss_fn(output, target)

        # With perfect prediction (mu=y), loss should be just the log-variance term
        # which should be relatively small
        assert loss.item() < 1.0

    def test_wrong_prediction_high_loss(self):
        """Test that wrong predictions give higher loss."""
        loss_fn = HeteroscedasticLoss(scatter_floor=0.01, n_parameters=2)

        # Perfect prediction
        output_correct = torch.tensor([[1.0, 2.0, -2.0, -2.0]])
        target = torch.tensor([[1.0, 2.0, 0.1, 0.1]])
        loss_correct = loss_fn(output_correct, target)

        # Wrong prediction (off by 1.0)
        output_wrong = torch.tensor([[2.0, 3.0, -2.0, -2.0]])
        loss_wrong = loss_fn(output_wrong, target)

        assert loss_wrong.item() > loss_correct.item()

    def test_higher_uncertainty_reduces_loss_for_wrong_prediction(self):
        """Test that higher model uncertainty reduces loss for wrong predictions."""
        loss_fn = HeteroscedasticLoss(scatter_floor=0.01, n_parameters=1)

        target = torch.tensor([[1.0, 0.1]])  # y=1, sigma=0.1

        # Wrong prediction with low uncertainty
        output_confident = torch.tensor([[2.0, -2.0]])  # mu=2, ln_s=-2 (s~0.13)
        loss_confident = loss_fn(output_confident, target)

        # Same wrong prediction with high uncertainty
        output_uncertain = torch.tensor([[2.0, 1.0]])  # mu=2, ln_s=1 (s~2.7)
        loss_uncertain = loss_fn(output_uncertain, target)

        # High uncertainty should reduce loss for wrong predictions
        assert loss_uncertain.item() < loss_confident.item()

    def test_scatter_floor_prevents_numerical_issues(self):
        """Test that scatter floor prevents division by near-zero."""
        loss_fn = HeteroscedasticLoss(scatter_floor=0.01, n_parameters=1)

        # Very small log-scatter that would cause issues without floor
        output = torch.tensor([[1.0, -100.0]])  # ln_s=-100 -> exp(-200) ≈ 0
        target = torch.tensor([[1.0, 0.001]])  # Small label error too

        loss = loss_fn(output, target)

        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    def test_manual_calculation_verification(self):
        """Verify loss computation against manual calculation."""
        scatter_floor = 0.1
        loss_fn = HeteroscedasticLoss(scatter_floor=scatter_floor, n_parameters=1)

        mu = 2.0
        ln_s = 0.0  # exp(0) = 1, so s^2 = 1 + 0.01 ≈ 1.01
        y = 1.5
        sigma_label = 0.5

        output = torch.tensor([[mu, ln_s]])
        target = torch.tensor([[y, sigma_label]])

        loss = loss_fn(output, target)

        # Manual calculation
        s_squared = math.exp(2 * ln_s) + scatter_floor**2
        total_var = sigma_label**2 + s_squared
        expected_loss = (mu - y) ** 2 / total_var + math.log(total_var)

        assert abs(loss.item() - expected_loss) < 1e-5


class TestHeteroscedasticLossGradients:
    """Tests for gradient computation."""

    def test_gradients_flow_through_loss(self):
        """Test that gradients flow properly through the loss."""
        loss_fn = HeteroscedasticLoss(n_parameters=3)

        output = torch.randn(16, 6, requires_grad=True)
        target = torch.randn(16, 6)
        target[:, 3:] = torch.abs(target[:, 3:]) + 0.01

        loss = loss_fn(output, target)
        loss.backward()

        assert output.grad is not None
        assert output.grad.shape == output.shape
        assert torch.isfinite(output.grad).all()

    def test_gradient_direction_correct(self):
        """Test that gradient points in the right direction for improvement."""
        loss_fn = HeteroscedasticLoss(n_parameters=1)

        # Start with wrong prediction
        mu = torch.tensor([[2.0]], requires_grad=True)
        ln_s = torch.tensor([[-1.0]], requires_grad=True)
        output = torch.cat([mu, ln_s], dim=1)

        target = torch.tensor([[1.0, 0.1]])  # True value is 1.0

        loss = loss_fn(output, target)
        loss.backward()

        # Gradient should push mu toward y=1.0
        # Since mu=2.0 > y=1.0, gradient should be positive
        assert mu.grad.item() > 0  # Positive gradient means decreasing mu reduces loss


class TestHeteroscedasticLossGetPredictedScatter:
    """Tests for the get_predicted_scatter helper method."""

    def test_scatter_extraction(self):
        """Test extracting predicted scatter from output."""
        scatter_floor = 0.01
        loss_fn = HeteroscedasticLoss(scatter_floor=scatter_floor, n_parameters=2)

        # Known log-scatter values
        ln_s = torch.tensor([[0.0, 1.0]])  # exp(0)=1, exp(1)≈2.718
        output = torch.cat([torch.zeros(1, 2), ln_s], dim=1)

        scatter = loss_fn.get_predicted_scatter(output)

        # s = sqrt(exp(2*ln_s) + s_0^2)
        expected_0 = math.sqrt(math.exp(0) + scatter_floor**2)
        expected_1 = math.sqrt(math.exp(2) + scatter_floor**2)

        assert abs(scatter[0, 0].item() - expected_0) < 1e-5
        assert abs(scatter[0, 1].item() - expected_1) < 1e-4

    def test_scatter_always_positive(self):
        """Test that scatter is always positive regardless of ln_s."""
        loss_fn = HeteroscedasticLoss(scatter_floor=0.01, n_parameters=3)

        # Various ln_s values including very negative
        output = torch.tensor([[0, 0, 0, -10, 0, 10.0]])

        scatter = loss_fn.get_predicted_scatter(output)

        assert (scatter > 0).all()


class TestHeteroscedasticLossRepr:
    """Tests for string representation."""

    def test_extra_repr(self):
        """Test that extra_repr includes all relevant info."""
        loss_fn = HeteroscedasticLoss(
            scatter_floor=0.05,
            n_parameters=7,
            reduction="sum",
        )

        repr_str = loss_fn.extra_repr()

        assert "0.05" in repr_str
        assert "7" in repr_str
        assert "sum" in repr_str
