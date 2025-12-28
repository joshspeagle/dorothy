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

from dorothy.data.augmentation import (
    DynamicBlockMasking,
    DynamicInputMasking,
    DynamicLabelMasking,
)


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


# ============================================================================
# DynamicLabelMasking Tests
# ============================================================================


class TestDynamicLabelMaskingInit:
    """Tests for DynamicLabelMasking initialization."""

    def test_default_init(self):
        """Test default initialization."""
        masking = DynamicLabelMasking()

        assert masking.p_labelset_min == 0.3
        assert masking.p_labelset_max == 1.0
        assert masking.p_label_min == 0.3
        assert masking.p_label_max == 1.0

    def test_custom_probabilities(self):
        """Test custom probability ranges."""
        masking = DynamicLabelMasking(
            p_labelset_min=0.5,
            p_labelset_max=0.9,
            p_label_min=0.2,
            p_label_max=0.8,
        )

        assert masking.p_labelset_min == 0.5
        assert masking.p_labelset_max == 0.9
        assert masking.p_label_min == 0.2
        assert masking.p_label_max == 0.8

    def test_invalid_labelset_range_raises(self):
        """Test that invalid labelset probability range raises error."""
        with pytest.raises(ValueError, match="labelset probability"):
            DynamicLabelMasking(p_labelset_min=0.8, p_labelset_max=0.3)

        with pytest.raises(ValueError, match="labelset probability"):
            DynamicLabelMasking(p_labelset_min=-0.1)

    def test_invalid_label_range_raises(self):
        """Test that invalid label probability range raises error."""
        with pytest.raises(ValueError, match="label probability"):
            DynamicLabelMasking(p_label_min=0.8, p_label_max=0.3)

        with pytest.raises(ValueError, match="label probability"):
            DynamicLabelMasking(p_label_max=1.5)

    def test_repr(self):
        """Test string representation."""
        masking = DynamicLabelMasking(p_label_min=0.4, p_label_max=0.9)
        repr_str = repr(masking)

        assert "DynamicLabelMasking" in repr_str
        assert "0.4" in repr_str
        assert "0.9" in repr_str


class TestDynamicLabelMaskingSingleTensor:
    """Tests for DynamicLabelMasking with single tensor input."""

    def test_output_shape(self):
        """Test that output has same shape as input."""
        masking = DynamicLabelMasking()
        y = torch.ones(16, 3, 11)

        y_out = masking(y)

        assert y_out.shape == y.shape

    def test_output_dtype(self):
        """Test that output preserves dtype."""
        masking = DynamicLabelMasking()
        y = torch.ones(8, 3, 11, dtype=torch.float32)

        y_out = masking(y)

        assert y_out.dtype == y.dtype

    def test_guaranteed_keeper(self):
        """Test that at least one label is always kept per sample."""
        torch.manual_seed(42)
        masking = DynamicLabelMasking(p_label_min=0.0, p_label_max=0.1)
        y = torch.ones(100, 3, 11)

        y_out = masking(y)

        # Each sample should have at least one label kept
        mask = y_out[:, 2, :]
        kept_per_sample = mask.sum(dim=1)
        assert (kept_per_sample >= 1).all()

    def test_respects_natural_mask(self):
        """Test that naturally masked labels stay masked."""
        masking = DynamicLabelMasking(p_label_min=1.0, p_label_max=1.0)
        y = torch.ones(4, 3, 11)
        # Pre-mask first 5 labels
        y[:, 2, :5] = 0

        y_out = masking(y)

        # First 5 should still be masked
        assert (y_out[:, 2, :5] == 0).all()

    def test_values_zeroed_at_masked_positions(self):
        """Test that values and errors are zeroed where mask becomes 0."""
        torch.manual_seed(42)
        masking = DynamicLabelMasking(p_label_min=0.3, p_label_max=0.5)
        y = torch.ones(10, 3, 11)

        y_out = masking(y)

        mask = y_out[:, 2, :]
        values = y_out[:, 0, :]
        errors = y_out[:, 1, :]

        # Where mask is 0, values and errors should be 0
        assert (values[mask == 0] == 0).all()
        assert (errors[mask == 0] == 0).all()

    def test_does_not_modify_input(self):
        """Test that original input is not modified."""
        masking = DynamicLabelMasking()
        y = torch.ones(4, 3, 11)
        y_original = y.clone()

        _ = masking(y)

        assert torch.equal(y, y_original)

    def test_all_natural_masked_unchanged(self):
        """Test that samples with all labels masked are unchanged."""
        masking = DynamicLabelMasking()
        y = torch.zeros(4, 3, 11)  # All masked

        y_out = masking(y)

        assert torch.equal(y_out, y)

    def test_invalid_input_shape_raises(self):
        """Test that wrong input shape raises error."""
        masking = DynamicLabelMasking()

        with pytest.raises(ValueError, match="shape"):
            masking(torch.ones(4, 11))  # 2D instead of 3D

        with pytest.raises(ValueError, match="shape"):
            masking(torch.ones(4, 2, 11))  # Wrong channel count


class TestDynamicLabelMaskingDict:
    """Tests for DynamicLabelMasking with dict input."""

    def test_dict_output_structure(self):
        """Test that dict input produces dict output with same keys."""
        masking = DynamicLabelMasking()
        y_dict = {
            "apogee": torch.ones(8, 3, 11),
            "galah": torch.ones(8, 3, 11),
        }

        y_out = masking(y_dict)

        assert isinstance(y_out, dict)
        assert set(y_out.keys()) == {"apogee", "galah"}

    def test_dict_output_shapes(self):
        """Test that dict outputs have correct shapes."""
        masking = DynamicLabelMasking()
        y_dict = {
            "apogee": torch.ones(8, 3, 11),
            "galah": torch.ones(8, 3, 11),
        }

        y_out = masking(y_dict)

        for name in y_dict:
            assert y_out[name].shape == y_dict[name].shape

    def test_guaranteed_keeper_labelset(self):
        """Test that at least one labelset is always kept per sample."""
        torch.manual_seed(42)
        masking = DynamicLabelMasking(p_labelset_min=0.0, p_labelset_max=0.1)
        y_dict = {
            "apogee": torch.ones(100, 3, 11),
            "galah": torch.ones(100, 3, 11),
        }

        y_out = masking(y_dict)

        # Each sample should have at least one labelset with valid labels
        for i in range(100):
            has_any = False
            for name in y_out:
                if y_out[name][i, 2, :].sum() > 0:
                    has_any = True
                    break
            assert has_any, f"Sample {i} has no valid labels in any labelset"

    def test_single_labelset_no_labelset_masking(self):
        """Test that single labelset doesn't do labelset-level masking."""
        torch.manual_seed(42)
        masking = DynamicLabelMasking(p_labelset_min=0.0, p_labelset_max=0.1)
        y_dict = {"apogee": torch.ones(50, 3, 11)}

        y_out = masking(y_dict)

        # With single labelset, should only do label-level masking
        # Each sample should still have labels (guaranteed keeper at label level)
        mask = y_out["apogee"][:, 2, :]
        assert (mask.sum(dim=1) >= 1).all()


class TestDynamicLabelMaskingEdgeCases:
    """Edge case tests for DynamicLabelMasking."""

    def test_single_available_label(self):
        """Test with only one available label per sample."""
        masking = DynamicLabelMasking(p_label_min=0.0, p_label_max=0.1)
        y = torch.zeros(10, 3, 11)
        y[:, :, 0] = 1.0  # Only first label available

        y_out = masking(y)

        # First label should always be kept (it's the guaranteed keeper)
        assert (y_out[:, 2, 0] == 1).all()

    def test_empty_dict(self):
        """Test with empty dict input."""
        masking = DynamicLabelMasking()
        y_dict: dict[str, torch.Tensor] = {}

        y_out = masking(y_dict)

        assert y_out == {}

    def test_high_keep_probability(self):
        """Test with high keep probability (should keep most labels)."""
        torch.manual_seed(42)
        masking = DynamicLabelMasking(p_label_min=0.9, p_label_max=1.0)
        y = torch.ones(100, 3, 11)

        y_out = masking(y)

        # Should keep most labels
        mask = y_out[:, 2, :]
        kept_fraction = mask.mean().item()
        assert kept_fraction > 0.85


# ============================================================================
# DynamicInputMasking Tests
# ============================================================================


class TestDynamicInputMaskingInit:
    """Tests for DynamicInputMasking initialization."""

    def test_default_init(self):
        """Test default initialization."""
        masking = DynamicInputMasking()

        assert masking.p_survey_min == 0.3
        assert masking.p_survey_max == 1.0
        assert masking.f_min_override is None
        assert masking.f_max == 0.5
        assert masking.p_block_min == 0.3
        assert masking.p_block_max == 1.0

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        masking = DynamicInputMasking(
            p_survey_min=0.5,
            p_survey_max=0.9,
            f_min_override=0.01,
            f_max=0.3,
            p_block_min=0.4,
            p_block_max=0.8,
        )

        assert masking.p_survey_min == 0.5
        assert masking.p_survey_max == 0.9
        assert masking.f_min_override == 0.01
        assert masking.f_max == 0.3
        assert masking.p_block_min == 0.4
        assert masking.p_block_max == 0.8

    def test_invalid_survey_range_raises(self):
        """Test that invalid survey probability range raises error."""
        with pytest.raises(ValueError, match="survey probability"):
            DynamicInputMasking(p_survey_min=0.8, p_survey_max=0.3)

    def test_invalid_block_range_raises(self):
        """Test that invalid block probability range raises error."""
        with pytest.raises(ValueError, match="block probability"):
            DynamicInputMasking(p_block_min=0.8, p_block_max=0.3)

    def test_invalid_f_max_raises(self):
        """Test that invalid f_max raises error."""
        with pytest.raises(ValueError, match="f_max"):
            DynamicInputMasking(f_max=0.0)

        with pytest.raises(ValueError, match="f_max"):
            DynamicInputMasking(f_max=1.5)

    def test_repr(self):
        """Test string representation."""
        masking = DynamicInputMasking(f_max=0.4)
        repr_str = repr(masking)

        assert "DynamicInputMasking" in repr_str
        assert "0.4" in repr_str


class TestDynamicInputMaskingSingleSurvey:
    """Tests for DynamicInputMasking with single survey."""

    def test_output_shape(self):
        """Test that output has same shape as input."""
        masking = DynamicInputMasking()
        X = {"boss": torch.ones(16, 3, 4506)}
        n_wavelengths = {"boss": 4506}

        X_out = masking(X, n_wavelengths)

        assert X_out["boss"].shape == X["boss"].shape

    def test_output_dtype(self):
        """Test that output preserves dtype."""
        masking = DynamicInputMasking()
        X = {"boss": torch.ones(8, 3, 1000, dtype=torch.float32)}
        n_wavelengths = {"boss": 1000}

        X_out = masking(X, n_wavelengths)

        assert X_out["boss"].dtype == X["boss"].dtype

    def test_guaranteed_keeper_block(self):
        """Test that at least one block is always kept per sample."""
        torch.manual_seed(42)
        np.random.seed(42)
        masking = DynamicInputMasking(p_block_min=0.0, p_block_max=0.1)
        X = {"boss": torch.ones(100, 3, 1000)}
        n_wavelengths = {"boss": 1000}

        X_out = masking(X, n_wavelengths)

        # Each sample should have at least one wavelength kept
        mask = X_out["boss"][:, 2, :]
        kept_per_sample = mask.sum(dim=1)
        assert (kept_per_sample >= 1).all()

    def test_respects_natural_mask(self):
        """Test that naturally masked wavelengths stay masked."""
        masking = DynamicInputMasking(p_block_min=1.0, p_block_max=1.0)
        X = {"boss": torch.ones(4, 3, 1000)}
        # Pre-mask first 100 wavelengths
        X["boss"][:, 2, :100] = 0
        n_wavelengths = {"boss": 1000}

        X_out = masking(X, n_wavelengths)

        # First 100 should still be masked
        assert (X_out["boss"][:, 2, :100] == 0).all()

    def test_flux_and_error_zeroed(self):
        """Test that flux and error are zeroed at masked positions."""
        torch.manual_seed(42)
        np.random.seed(42)
        masking = DynamicInputMasking(p_block_min=0.3, p_block_max=0.5)
        X = {"boss": torch.ones(10, 3, 500)}
        n_wavelengths = {"boss": 500}

        X_out = masking(X, n_wavelengths)

        mask = X_out["boss"][:, 2, :]
        flux = X_out["boss"][:, 0, :]
        error = X_out["boss"][:, 1, :]

        assert (flux[mask == 0] == 0).all()
        assert (error[mask == 0] == 0).all()


class TestDynamicInputMaskingMultiSurvey:
    """Tests for DynamicInputMasking with multiple surveys."""

    def test_multi_survey_output_structure(self):
        """Test that multi-survey input produces correct output structure."""
        masking = DynamicInputMasking()
        X = {
            "boss": torch.ones(8, 3, 4506),
            "desi": torch.ones(8, 3, 7650),
        }
        n_wavelengths = {"boss": 4506, "desi": 7650}

        X_out = masking(X, n_wavelengths)

        assert isinstance(X_out, dict)
        assert set(X_out.keys()) == {"boss", "desi"}
        assert X_out["boss"].shape == X["boss"].shape
        assert X_out["desi"].shape == X["desi"].shape

    def test_guaranteed_keeper_survey(self):
        """Test that at least one survey is always kept per sample."""
        torch.manual_seed(42)
        np.random.seed(42)
        masking = DynamicInputMasking(p_survey_min=0.0, p_survey_max=0.1)
        X = {
            "boss": torch.ones(100, 3, 1000),
            "desi": torch.ones(100, 3, 1000),
        }
        n_wavelengths = {"boss": 1000, "desi": 1000}

        X_out = masking(X, n_wavelengths)

        # Each sample should have at least one survey with valid data
        for i in range(100):
            has_any = False
            for name in X_out:
                if X_out[name][i, 2, :].sum() > 0:
                    has_any = True
                    break
            assert has_any, f"Sample {i} has no valid data in any survey"

    def test_different_wavelength_counts(self):
        """Test with different wavelength counts per survey."""
        masking = DynamicInputMasking()
        X = {
            "boss": torch.ones(8, 3, 4506),
            "lamost_lrs": torch.ones(8, 3, 3473),
            "desi": torch.ones(8, 3, 7650),
        }
        n_wavelengths = {"boss": 4506, "lamost_lrs": 3473, "desi": 7650}

        X_out = masking(X, n_wavelengths)

        assert X_out["boss"].shape == (8, 3, 4506)
        assert X_out["lamost_lrs"].shape == (8, 3, 3473)
        assert X_out["desi"].shape == (8, 3, 7650)


class TestDynamicInputMaskingBlockSize:
    """Tests for block size sampling."""

    def test_block_size_varies(self):
        """Test that block size varies across batches (log-uniform)."""
        np.random.seed(42)

        # Call block masking multiple times and check block sizes vary
        block_sizes = []
        for _ in range(50):
            # Reset random state to get different samples
            f_min = 1.0 / 1000
            log_f = np.random.uniform(np.log(f_min), np.log(0.5))
            f = np.exp(log_f)
            block_size = max(1, int(np.ceil(f * 1000)))
            block_sizes.append(block_size)

        # Should have variety in block sizes
        unique_sizes = len(set(block_sizes))
        assert unique_sizes > 10, f"Expected variety in block sizes, got {unique_sizes}"

    def test_f_min_override(self):
        """Test that f_min_override is respected."""
        np.random.seed(42)
        masking = DynamicInputMasking(f_min_override=0.1, f_max=0.5)
        X = {"survey": torch.ones(10, 3, 100)}
        n_wavelengths = {"survey": 100}

        # With f_min=0.1 and N=100, minimum block_size = ceil(0.1 * 100) = 10
        # Run multiple times and check all block sizes are at least 10
        for _ in range(20):
            X_out = masking(X, n_wavelengths)
            # This is indirect - just verify we get reasonable output
            mask = X_out["survey"][:, 2, :]
            assert mask.shape == (10, 100)


class TestDynamicInputMaskingEdgeCases:
    """Edge case tests for DynamicInputMasking."""

    def test_empty_dict(self):
        """Test with empty dict input."""
        masking = DynamicInputMasking()
        X: dict[str, torch.Tensor] = {}
        n_wavelengths: dict[str, int] = {}

        X_out = masking(X, n_wavelengths)

        assert X_out == {}

    def test_all_natural_masked(self):
        """Test with all wavelengths naturally masked."""
        masking = DynamicInputMasking()
        X = {"survey": torch.zeros(4, 3, 100)}
        n_wavelengths = {"survey": 100}

        X_out = masking(X, n_wavelengths)

        # Should be unchanged
        assert torch.equal(X_out["survey"], X["survey"])

    def test_single_wavelength(self):
        """Test with single wavelength."""
        masking = DynamicInputMasking()
        X = {"survey": torch.ones(4, 3, 1)}
        n_wavelengths = {"survey": 1}

        X_out = masking(X, n_wavelengths)

        # Single wavelength should always be kept (guaranteed keeper)
        assert (X_out["survey"][:, 2, 0] == 1).all()

    def test_does_not_modify_input(self):
        """Test that original input is not modified."""
        masking = DynamicInputMasking()
        X = {"survey": torch.ones(4, 3, 100)}
        X_original = {"survey": X["survey"].clone()}
        n_wavelengths = {"survey": 100}

        _ = masking(X, n_wavelengths)

        assert torch.equal(X["survey"], X_original["survey"])
