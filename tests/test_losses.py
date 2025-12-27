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
        # Output shape is (batch, 2, n_params): [means, log_scatter]
        output = torch.randn(32, 2, 3)
        # Target shape is (batch, 3, n_params): [values, errors, mask]
        target = make_target_3ch(32, 3)

        loss = loss_fn(output, target)

        assert loss.shape == ()
        assert loss.ndim == 0

    def test_output_shape_sum_reduction(self):
        """Test output is scalar with sum reduction."""
        loss_fn = HeteroscedasticLoss(n_parameters=3, reduction="sum")
        output = torch.randn(32, 2, 3)
        target = make_target_3ch(32, 3)

        loss = loss_fn(output, target)

        assert loss.shape == ()
        assert loss.ndim == 0

    def test_output_shape_no_reduction(self):
        """Test output shape with no reduction."""
        loss_fn = HeteroscedasticLoss(n_parameters=3, reduction="none")
        batch_size = 32
        output = torch.randn(batch_size, 2, 3)
        target = make_target_3ch(batch_size, 3)

        loss = loss_fn(output, target)

        assert loss.shape == (batch_size, 3)

    def test_invalid_output_dimension_rejected(self, loss_fn):
        """Test that incorrect output dimension is rejected."""
        # Wrong shape - should be (batch, 2, 3)
        output = torch.randn(32, 2, 4)  # 4 params instead of 3
        target = make_target_3ch(32, 3)

        with pytest.raises(ValueError, match="Output shape"):
            loss_fn(output, target)

    def test_invalid_target_dimension_rejected(self, loss_fn):
        """Test that incorrect target dimension is rejected."""
        output = torch.randn(32, 2, 3)
        # Wrong n_params in target
        target = make_target_3ch(32, 4)

        with pytest.raises(ValueError, match="Target shape"):
            loss_fn(output, target)


def make_target_3ch(batch_size, n_params, values=None, errors=None, mask=None):
    """Helper to create 3-channel target: [values, errors, mask]."""
    if values is None:
        values = torch.randn(batch_size, n_params)
    if errors is None:
        errors = torch.abs(torch.randn(batch_size, n_params)) + 0.01
    if mask is None:
        mask = torch.ones(batch_size, n_params)
    return torch.stack([values, errors, mask], dim=1)  # (batch, 3, n_params)


class TestHeteroscedasticLossMathematics:
    """Tests for mathematical correctness of the loss."""

    def test_perfect_prediction_low_loss(self):
        """Test that perfect predictions with low uncertainty give low loss."""
        loss_fn = HeteroscedasticLoss(scatter_floor=0.01, n_parameters=2)

        # Perfect prediction: mu = y
        # Output shape: (1, 2, 2) = [means=[1,2], log_scatter=[-3,-3]]
        output = torch.tensor([[[1.0, 2.0], [-3.0, -3.0]]])
        # Target shape: (1, 3, 2) = [values, errors, mask]
        target = torch.tensor([[[1.0, 2.0], [0.1, 0.1], [1.0, 1.0]]])

        loss = loss_fn(output, target)

        # With perfect prediction (mu=y), loss should be just the log-variance term
        # which should be relatively small
        assert loss.item() < 1.0

    def test_wrong_prediction_high_loss(self):
        """Test that wrong predictions give higher loss."""
        loss_fn = HeteroscedasticLoss(scatter_floor=0.01, n_parameters=2)

        # Perfect prediction: (1, 2, 2)
        output_correct = torch.tensor([[[1.0, 2.0], [-2.0, -2.0]]])
        # Target shape: (1, 3, 2) = [values, errors, mask]
        target = torch.tensor([[[1.0, 2.0], [0.1, 0.1], [1.0, 1.0]]])
        loss_correct = loss_fn(output_correct, target)

        # Wrong prediction (off by 1.0)
        output_wrong = torch.tensor([[[2.0, 3.0], [-2.0, -2.0]]])
        loss_wrong = loss_fn(output_wrong, target)

        assert loss_wrong.item() > loss_correct.item()

    def test_higher_uncertainty_reduces_loss_for_wrong_prediction(self):
        """Test that higher model uncertainty reduces loss for wrong predictions."""
        loss_fn = HeteroscedasticLoss(scatter_floor=0.01, n_parameters=1)

        # Target shape: (1, 3, 1) = [values, errors, mask]
        target = torch.tensor([[[1.0], [0.1], [1.0]]])  # y=1, sigma=0.1, mask=1

        # Wrong prediction with low uncertainty: (1, 2, 1)
        output_confident = torch.tensor([[[2.0], [-2.0]]])  # mu=2, ln_s=-2 (s~0.13)
        loss_confident = loss_fn(output_confident, target)

        # Same wrong prediction with high uncertainty
        output_uncertain = torch.tensor([[[2.0], [1.0]]])  # mu=2, ln_s=1 (s~2.7)
        loss_uncertain = loss_fn(output_uncertain, target)

        # High uncertainty should reduce loss for wrong predictions
        assert loss_uncertain.item() < loss_confident.item()

    def test_scatter_floor_prevents_numerical_issues(self):
        """Test that scatter floor prevents division by near-zero."""
        loss_fn = HeteroscedasticLoss(scatter_floor=0.01, n_parameters=1)

        # Very small log-scatter that would cause issues without floor
        # Output shape: (1, 2, 1)
        output = torch.tensor([[[1.0], [-100.0]]])  # ln_s=-100 -> exp(-200) ≈ 0
        # Target shape: (1, 3, 1) = [values, errors, mask]
        target = torch.tensor([[[1.0], [0.001], [1.0]]])  # Small label error too

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

        # Output shape: (1, 2, 1)
        output = torch.tensor([[[mu], [ln_s]]])
        # Target shape: (1, 3, 1) = [values, errors, mask]
        target = torch.tensor([[[y], [sigma_label], [1.0]]])

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

        # Output shape: (batch, 2, n_params)
        output = torch.randn(16, 2, 3, requires_grad=True)
        # Target shape: (batch, 3, n_params) = [values, errors, mask]
        target = make_target_3ch(16, 3)

        loss = loss_fn(output, target)
        loss.backward()

        assert output.grad is not None
        assert output.grad.shape == output.shape
        assert torch.isfinite(output.grad).all()

    def test_gradient_direction_correct(self):
        """Test that gradient points in the right direction for improvement."""
        loss_fn = HeteroscedasticLoss(n_parameters=1)

        # Start with wrong prediction
        # Output shape: (1, 2, 1) - need to create it as 3D for the loss
        mu = torch.tensor([[[2.0]]], requires_grad=True)
        ln_s = torch.tensor([[[-1.0]]], requires_grad=True)
        # Stack along dim 1 to get (1, 2, 1)
        output = torch.cat([mu, ln_s], dim=1)

        # Target shape: (1, 3, 1) = [values, errors, mask]
        target = torch.tensor([[[1.0], [0.1], [1.0]]])  # True value is 1.0

        loss = loss_fn(output, target)
        loss.backward()

        # Gradient should push mu toward y=1.0
        # Since mu=2.0 > y=1.0, gradient should be positive
        assert (
            mu.grad.flatten().item() > 0
        )  # Positive gradient means decreasing mu reduces loss


class TestHeteroscedasticLossGetPredictedScatter:
    """Tests for the get_predicted_scatter helper method."""

    def test_scatter_extraction(self):
        """Test extracting predicted scatter from output."""
        scatter_floor = 0.01
        loss_fn = HeteroscedasticLoss(scatter_floor=scatter_floor, n_parameters=2)

        # Known log-scatter values
        # Output shape: (1, 2, 2) = [means, log_scatter]
        ln_s = torch.tensor([[0.0, 1.0]])  # exp(0)=1, exp(1)≈2.718
        output = torch.stack([torch.zeros(1, 2), ln_s], dim=1)  # (1, 2, 2)

        scatter = loss_fn.get_predicted_scatter(output)

        # s = sqrt(exp(2*ln_s) + s_0^2)
        expected_0 = math.sqrt(math.exp(0) + scatter_floor**2)
        expected_1 = math.sqrt(math.exp(2) + scatter_floor**2)

        assert abs(scatter[0, 0].item() - expected_0) < 1e-5
        assert abs(scatter[0, 1].item() - expected_1) < 1e-4

    def test_scatter_always_positive(self):
        """Test that scatter is always positive regardless of ln_s."""
        loss_fn = HeteroscedasticLoss(scatter_floor=0.01, n_parameters=3)

        # Output shape: (1, 2, 3) = [means, log_scatter]
        means = torch.zeros(1, 3)
        ln_s = torch.tensor([[-10.0, 0.0, 10.0]])
        output = torch.stack([means, ln_s], dim=1)  # (1, 2, 3)

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


class TestHeteroscedasticLossMasking:
    """Tests for mask handling in HeteroscedasticLoss (mask embedded in target channel 2)."""

    @pytest.fixture
    def loss_fn(self):
        """Create a loss function for testing."""
        return HeteroscedasticLoss(scatter_floor=0.01, n_parameters=3)

    def test_forward_with_all_ones_mask(self, loss_fn):
        """Test that all-ones mask computes loss for all elements."""
        output = torch.randn(16, 2, 3)
        # Target with all-ones mask
        target = make_target_3ch(16, 3)

        loss = loss_fn(output, target)

        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_forward_mask_excludes_samples(self, loss_fn):
        """Test that masked samples don't contribute to loss."""
        # Output shape: (batch, 2, n_params)
        output = torch.zeros(4, 2, 3)

        # Create target with values, errors, mask
        values = torch.zeros(4, 3)
        errors = torch.ones(4, 3) * 0.1
        mask = torch.ones(4, 3)

        # Make first sample have high loss (means channel)
        output[0, 0, :] = 10.0  # Large prediction error

        # Without masking first sample, loss is high
        target_no_mask = torch.stack([values, errors, mask], dim=1)
        loss_no_mask = loss_fn(output, target_no_mask)

        # Mask out first sample entirely
        mask_with_exclusion = mask.clone()
        mask_with_exclusion[0, :] = 0
        target_masked = torch.stack([values, errors, mask_with_exclusion], dim=1)

        loss_masked = loss_fn(output, target_masked)

        # Masked loss should be lower (no contribution from bad sample)
        assert loss_masked < loss_no_mask

    def test_forward_mask_per_parameter(self, loss_fn):
        """Test that masking works per parameter."""
        output = torch.zeros(4, 2, 3)

        values = torch.zeros(4, 3)
        errors = torch.ones(4, 3) * 0.1
        mask = torch.ones(4, 3)

        # Make first parameter of first sample have high loss
        output[0, 0, 0] = 10.0  # First sample, means channel, first param

        # Mask only the first parameter of first sample
        mask_partial = mask.clone()
        mask_partial[0, 0] = 0
        target = torch.stack([values, errors, mask_partial], dim=1)

        loss = loss_fn(output, target)

        # Compare with loss where that value is corrected (not masked)
        output_corrected = output.clone()
        output_corrected[0, 0, 0] = 0.0
        target_full_mask = torch.stack([values, errors, mask], dim=1)
        loss_corrected = loss_fn(output_corrected, target_full_mask)

        # Should be very close since we masked out the bad element
        assert torch.allclose(loss, loss_corrected, rtol=1e-4)

    def test_forward_all_masked_returns_zero(self, loss_fn):
        """Test that fully masked batch returns zero loss."""
        output = torch.randn(4, 2, 3)

        values = torch.randn(4, 3)
        errors = torch.abs(torch.randn(4, 3)) + 0.01
        mask = torch.zeros(4, 3)  # All masked
        target = torch.stack([values, errors, mask], dim=1)

        loss = loss_fn(output, target)

        assert loss.item() == 0.0

    def test_forward_detailed_with_mask(self, loss_fn):
        """Test forward_detailed with mask returns correct components."""
        output = torch.randn(8, 2, 3)

        values = torch.randn(8, 3)
        errors = torch.abs(torch.randn(8, 3)) + 0.01
        mask = torch.ones(8, 3)
        mask[:4, 0] = 0  # Mask half of first parameter
        target = torch.stack([values, errors, mask], dim=1)

        result = loss_fn.forward_detailed(output, target)

        assert "loss" in result
        assert "mean_component" in result
        assert "scatter_component" in result
        assert "residuals" in result
        assert result["mean_component"].shape == (3,)
        assert result["scatter_component"].shape == (3,)

    def test_forward_detailed_mask_affects_components(self, loss_fn):
        """Test that mask properly affects per-parameter component averages."""
        output = torch.zeros(4, 2, 3)

        values = torch.zeros(4, 3)
        errors = torch.ones(4, 3) * 0.1

        # Add large error only to masked samples (first 2 samples, first param)
        output[:2, 0, 0] = 10.0

        # Mask out those samples for first parameter
        mask = torch.ones(4, 3)
        mask[:2, 0] = 0
        target = torch.stack([values, errors, mask], dim=1)

        result = loss_fn.forward_detailed(output, target)

        # Mean component for first param should be ~0 (masked samples excluded)
        # Other params should have normal values
        assert result["mean_component"][0] < 0.01  # Near zero

    def test_forward_reduction_none_with_mask(self):
        """Test that reduction='none' with mask zeros out masked elements."""
        loss_fn = HeteroscedasticLoss(n_parameters=3, reduction="none")
        output = torch.randn(4, 2, 3)

        values = torch.randn(4, 3)
        errors = torch.abs(torch.randn(4, 3)) + 0.01
        mask = torch.zeros(4, 3)
        mask[0, 0] = 1
        mask[1, 1] = 1
        target = torch.stack([values, errors, mask], dim=1)

        loss = loss_fn(output, target)

        # Only (0,0) and (1,1) should be non-zero
        assert loss[0, 0] != 0
        assert loss[1, 1] != 0
        assert loss[0, 1] == 0
        assert loss[2, :].sum() == 0

    def test_forward_reduction_sum_with_mask(self):
        """Test that reduction='sum' with mask only sums valid elements."""
        loss_fn = HeteroscedasticLoss(n_parameters=2, reduction="sum")
        output = torch.zeros(4, 2, 2)

        values = torch.zeros(4, 2)
        errors = torch.ones(4, 2) * 0.1
        mask_all = torch.ones(4, 2)

        target_all = torch.stack([values, errors, mask_all], dim=1)
        loss_all = loss_fn(output, target_all)

        # Mask out half
        mask_half = mask_all.clone()
        mask_half[:2, :] = 0
        target_half = torch.stack([values, errors, mask_half], dim=1)
        loss_half = loss_fn(output, target_half)

        # Sum should be half (since all elements have same loss)
        assert torch.allclose(loss_half, loss_all / 2, rtol=1e-4)

    def test_gradients_with_mask(self, loss_fn):
        """Test that gradients flow correctly with mask."""
        output = torch.randn(8, 2, 3, requires_grad=True)

        values = torch.randn(8, 3)
        errors = torch.abs(torch.randn(8, 3)) + 0.01
        mask = torch.ones(8, 3)
        mask[:4, :] = 0  # Mask first half
        target = torch.stack([values, errors, mask], dim=1)

        loss = loss_fn(output, target)
        loss.backward()

        assert output.grad is not None
        assert torch.isfinite(output.grad).all()

        # Gradients for masked samples should be zero
        assert (output.grad[:4, :, :] == 0).all()

        # Gradients for unmasked samples should be non-zero
        assert (output.grad[4:, :, :] != 0).any()
