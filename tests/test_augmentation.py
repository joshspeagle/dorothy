"""
Tests for data augmentation utilities.

These tests verify:
1. DynamicBlockMasking initialization and validation
2. Block mask generation
3. Mask application to 3-channel input
4. Proper handling of existing masks
"""

import numpy as np
import pytest
import torch

from dorothy.data.augmentation import DynamicBlockMasking


class TestDynamicBlockMaskingInit:
    """Tests for DynamicBlockMasking initialization."""

    def test_default_init(self):
        """Test default initialization."""
        aug = DynamicBlockMasking()

        assert aug.min_fraction == 0.1
        assert aug.max_fraction == 0.8
        assert aug.min_block_size == 5
        assert aug.fraction_choices is None

    def test_custom_fractions(self):
        """Test custom fraction range."""
        aug = DynamicBlockMasking(min_fraction=0.2, max_fraction=0.5)

        assert aug.min_fraction == 0.2
        assert aug.max_fraction == 0.5

    def test_fraction_choices(self):
        """Test using specific fraction choices."""
        choices = [0.1, 0.3, 0.5]
        aug = DynamicBlockMasking(fraction_choices=choices)

        assert aug.fraction_choices == choices
        assert aug.min_fraction == 0.1
        assert aug.max_fraction == 0.5

    def test_invalid_min_fraction_raises(self):
        """Test that invalid min_fraction raises error."""
        with pytest.raises(ValueError, match="min_fraction"):
            DynamicBlockMasking(min_fraction=-0.1)

        with pytest.raises(ValueError, match="min_fraction"):
            DynamicBlockMasking(min_fraction=1.5)

    def test_invalid_max_fraction_raises(self):
        """Test that invalid max_fraction raises error."""
        with pytest.raises(ValueError, match="max_fraction"):
            DynamicBlockMasking(max_fraction=-0.1)

        with pytest.raises(ValueError, match="max_fraction"):
            DynamicBlockMasking(max_fraction=1.5)

    def test_min_greater_than_max_raises(self):
        """Test that min > max raises error."""
        with pytest.raises(ValueError, match="min_fraction.*max_fraction"):
            DynamicBlockMasking(min_fraction=0.6, max_fraction=0.4)

    def test_invalid_fraction_choices_raises(self):
        """Test that invalid fraction_choices raise error."""
        with pytest.raises(ValueError, match="fraction_choices"):
            DynamicBlockMasking(fraction_choices=[0.1, 1.5, 0.3])

    def test_invalid_min_block_size_raises(self):
        """Test that invalid min_block_size raises error."""
        with pytest.raises(ValueError, match="min_block_size"):
            DynamicBlockMasking(min_block_size=0)

    def test_repr(self):
        """Test string representation."""
        aug = DynamicBlockMasking(min_fraction=0.2, max_fraction=0.5)
        repr_str = repr(aug)

        assert "DynamicBlockMasking" in repr_str
        assert "0.2" in repr_str
        assert "0.5" in repr_str


class TestDynamicBlockMaskingCall:
    """Tests for DynamicBlockMasking __call__."""

    def test_output_shape(self):
        """Test that output has same shape as input."""
        aug = DynamicBlockMasking(min_fraction=0.2, max_fraction=0.3)
        X = torch.ones(16, 3, 100)

        X_out = aug(X)

        assert X_out.shape == X.shape

    def test_output_dtype(self):
        """Test that output preserves dtype."""
        aug = DynamicBlockMasking()
        X = torch.ones(8, 3, 50, dtype=torch.float32)

        X_out = aug(X)

        assert X_out.dtype == X.dtype

    def test_mask_channel_updated(self):
        """Test that mask channel is modified."""
        aug = DynamicBlockMasking(min_fraction=0.3, max_fraction=0.5)
        X = torch.ones(4, 3, 100)

        X_out = aug(X)

        # At least some values in mask channel should be 0
        assert (X_out[:, 2, :] == 0).any()

    def test_flux_zeroed_at_masked_positions(self):
        """Test that flux is zeroed where mask is 0."""
        aug = DynamicBlockMasking(min_fraction=0.3, max_fraction=0.5)
        X = torch.ones(4, 3, 100)

        X_out = aug(X)

        # Flux should be 0 where mask is 0
        mask = X_out[:, 2, :]
        flux = X_out[:, 0, :]
        assert (flux[mask == 0] == 0).all()

    def test_error_zeroed_at_masked_positions(self):
        """Test that error is zeroed where mask is 0."""
        aug = DynamicBlockMasking(min_fraction=0.3, max_fraction=0.5)
        X = torch.ones(4, 3, 100)

        X_out = aug(X)

        # Error should be 0 where mask is 0
        mask = X_out[:, 2, :]
        error = X_out[:, 1, :]
        assert (error[mask == 0] == 0).all()

    def test_existing_mask_preserved(self):
        """Test that existing masked positions stay masked."""
        aug = DynamicBlockMasking(min_fraction=0.1, max_fraction=0.2)
        X = torch.ones(4, 3, 100)
        # Pre-mask first 10 wavelengths
        X[:, 2, :10] = 0

        X_out = aug(X)

        # First 10 should still be masked
        assert (X_out[:, 2, :10] == 0).all()

    def test_masking_fraction_approximate(self):
        """Test that masking fraction is approximately in range."""
        torch.manual_seed(42)
        aug = DynamicBlockMasking(min_fraction=0.2, max_fraction=0.4)
        X = torch.ones(100, 3, 1000)

        X_out = aug(X)

        # Calculate masked fraction per sample
        mask = X_out[:, 2, :]
        masked_fractions = (mask == 0).float().mean(dim=1)

        # Average should be roughly in the range
        avg_fraction = masked_fractions.mean().item()
        assert 0.1 < avg_fraction < 0.6  # Allow some variance

    def test_invalid_input_dims_raises(self):
        """Test that wrong input dimensions raise error."""
        aug = DynamicBlockMasking()
        X_2d = torch.ones(4, 100)
        X_4d = torch.ones(4, 3, 100, 2)

        with pytest.raises(ValueError, match="3D tensor"):
            aug(X_2d)

        with pytest.raises(ValueError, match="3D tensor"):
            aug(X_4d)

    def test_invalid_n_channels_raises(self):
        """Test that wrong number of channels raises error."""
        aug = DynamicBlockMasking()
        X = torch.ones(4, 2, 100)  # Only 2 channels

        with pytest.raises(ValueError, match="3 channels"):
            aug(X)

    def test_does_not_modify_input(self):
        """Test that original input is not modified."""
        aug = DynamicBlockMasking(min_fraction=0.3, max_fraction=0.5)
        X = torch.ones(4, 3, 100)
        X_original = X.clone()

        _ = aug(X)

        assert torch.equal(X, X_original)

    def test_zero_fraction_no_masking(self):
        """Test that zero fraction doesn't mask anything."""
        aug = DynamicBlockMasking(min_fraction=0.0, max_fraction=0.0)
        X = torch.ones(4, 3, 100)

        X_out = aug(X)

        # Should be unchanged
        assert torch.equal(X_out, X)


class TestBlockMaskGeneration:
    """Tests for block mask generation."""

    def test_mask_shape(self):
        """Test that generated mask has correct shape."""
        aug = DynamicBlockMasking()
        mask = aug._create_block_mask(100, 0.3, 50)

        assert mask.shape == (100,)
        assert mask.dtype == np.float32

    def test_mask_values_binary(self):
        """Test that mask only contains 0 and 1."""
        aug = DynamicBlockMasking()
        mask = aug._create_block_mask(100, 0.5, 30)

        unique_values = np.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique_values)

    def test_mask_contiguous_blocks(self):
        """Test that masked regions are contiguous blocks."""
        np.random.seed(42)
        aug = DynamicBlockMasking(min_block_size=5)
        mask = aug._create_block_mask(100, 0.3, 20)

        # Find transitions from 1 to 0 (block starts)
        transitions = np.diff(mask)
        block_starts = np.where(transitions == -1)[0] + 1

        # Each block should be at least min_block_size
        for start in block_starts:
            # Find end of this block
            block = mask[start:]
            end = start + np.argmax(block == 1) if 1 in block else len(mask)
            block_size = end - start
            assert block_size >= 5, f"Block size {block_size} < min_block_size 5"

    def test_approximate_target_fraction(self):
        """Test that masked fraction is approximately target."""
        np.random.seed(42)
        aug = DynamicBlockMasking()

        # Run multiple times and check average
        fractions = []
        for _ in range(100):
            mask = aug._create_block_mask(1000, 0.3, 100)
            masked_fraction = 1 - mask.mean()
            fractions.append(masked_fraction)

        avg_fraction = np.mean(fractions)
        # Should be close to target, but allow variance
        assert 0.2 < avg_fraction < 0.4


class TestDynamicBlockMaskingFractionChoices:
    """Tests for fraction_choices functionality."""

    def test_fraction_choices_used(self):
        """Test that specified fraction choices are used."""
        torch.manual_seed(42)
        np.random.seed(42)

        choices = [0.2, 0.4, 0.6]
        aug = DynamicBlockMasking(fraction_choices=choices)
        X = torch.ones(100, 3, 500)

        X_out = aug(X)

        # Compute masked fractions
        mask = X_out[:, 2, :]
        masked_fractions = (mask == 0).float().mean(dim=1).numpy()

        # Each should be close to one of the choices
        # (allowing for overlap and some variance)
        for frac in masked_fractions:
            closest = min(choices, key=lambda c: abs(c - frac))
            assert abs(frac - closest) < 0.15  # Allow some variance


class TestDynamicBlockMaskingDeviceCompat:
    """Tests for device compatibility."""

    def test_cpu_input(self):
        """Test with CPU tensor."""
        aug = DynamicBlockMasking(min_fraction=0.2, max_fraction=0.3)
        X = torch.ones(4, 3, 100)

        X_out = aug(X)

        assert X_out.device == X.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_input(self):
        """Test with CUDA tensor."""
        aug = DynamicBlockMasking(min_fraction=0.2, max_fraction=0.3)
        X = torch.ones(4, 3, 100, device="cuda")

        X_out = aug(X)

        assert X_out.device == X.device
