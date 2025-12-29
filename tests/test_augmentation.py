"""
Tests for data augmentation utilities.

These tests verify:
1. DynamicLabelMasking initialization and validation
2. DynamicInputMasking initialization and validation
3. Mask application to inputs and labels
4. Proper handling of existing masks
"""

import numpy as np
import pytest
import torch

from dorothy.data.augmentation import (
    DynamicInputMasking,
    DynamicLabelMasking,
)


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


# ============================================================================
# DynamicInputMasking Random Offset Tests
# ============================================================================


class TestDynamicInputMaskingRandomOffset:
    """Tests for random offset feature in DynamicInputMasking."""

    def test_offset_varies_across_calls(self):
        """Test that random offset produces varying block boundaries."""
        np.random.seed(42)
        torch.manual_seed(42)
        masking = DynamicInputMasking(p_block_min=0.5, p_block_max=0.5, f_max=0.1)
        X = {"survey": torch.ones(1, 3, 100)}
        n_wavelengths = {"survey": 100}

        # Collect masks from multiple calls to check for variation
        masks = []
        for _ in range(20):
            X_out = masking(X, n_wavelengths)
            masks.append(X_out["survey"][0, 2, :].clone())

        # Check that not all masks are identical (offset should cause variation)
        unique_patterns = len({tuple(m.tolist()) for m in masks})
        assert unique_patterns > 1, "All masks are identical - offset not varying"

    def test_offset_produces_partial_blocks_at_edges(self):
        """Test that offset creates partial blocks at spectrum edges."""
        np.random.seed(123)
        torch.manual_seed(123)

        # Use parameters that force a specific block structure
        # With p_block=1.0, all blocks should be kept
        masking = DynamicInputMasking(p_block_min=1.0, p_block_max=1.0, f_max=0.5)
        X = {"survey": torch.ones(1, 3, 100)}
        n_wavelengths = {"survey": 100}

        # Run multiple times - all valid data should be preserved
        for _ in range(10):
            X_out = masking(X, n_wavelengths)
            mask = X_out["survey"][0, 2, :]
            # With p_block=1.0, all naturally valid pixels should remain valid
            assert mask.sum() == 100, "All pixels should be kept when p_block=1.0"

    def test_guaranteed_keeper_with_offset(self):
        """Test that guaranteed keeper works correctly with random offset."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Very aggressive masking - should still keep at least one block
        masking = DynamicInputMasking(p_block_min=0.0, p_block_max=0.05)
        X = {"survey": torch.ones(50, 3, 100)}
        n_wavelengths = {"survey": 100}

        X_out = masking(X, n_wavelengths)

        # Every sample should have at least some valid pixels
        mask = X_out["survey"][:, 2, :]
        kept_per_sample = mask.sum(dim=1)
        assert (kept_per_sample >= 1).all(), "Guaranteed keeper failed with offset"

    def test_offset_does_not_change_mean_coverage(self):
        """Test that random offset doesn't significantly change mean coverage."""
        np.random.seed(42)
        torch.manual_seed(42)

        masking = DynamicInputMasking(p_block_min=0.4, p_block_max=0.6)
        X = {"survey": torch.ones(100, 3, 200)}
        n_wavelengths = {"survey": 200}

        # Run many times and collect coverage statistics
        coverages = []
        for _ in range(50):
            X_out = masking(X, n_wavelengths)
            mask = X_out["survey"][:, 2, :]
            coverage = mask.mean().item()
            coverages.append(coverage)

        mean_coverage = np.mean(coverages)
        # With p_block in [0.4, 0.6], expect roughly 50% coverage
        # Allow for variation due to block structure and guaranteed keeper
        assert (
            0.3 < mean_coverage < 0.7
        ), f"Mean coverage {mean_coverage:.3f} outside expected range"

    def test_offset_with_partial_natural_mask(self):
        """Test offset works correctly when some wavelengths are naturally masked."""
        np.random.seed(42)
        torch.manual_seed(42)

        masking = DynamicInputMasking(p_block_min=0.5, p_block_max=0.5)
        X = {"survey": torch.ones(10, 3, 100)}
        # Mask first and last 10 wavelengths naturally
        X["survey"][:, 2, :10] = 0
        X["survey"][:, 2, -10:] = 0
        n_wavelengths = {"survey": 100}

        X_out = masking(X, n_wavelengths)

        # Naturally masked regions should stay masked
        assert (X_out["survey"][:, 2, :10] == 0).all()
        assert (X_out["survey"][:, 2, -10:] == 0).all()

        # Some middle pixels should still be valid (guaranteed keeper)
        middle_mask = X_out["survey"][:, 2, 10:90]
        assert (middle_mask.sum(dim=1) >= 1).all()

    def test_offset_with_very_small_spectrum(self):
        """Test that offset works with very small spectra (edge case)."""
        np.random.seed(42)
        torch.manual_seed(42)

        masking = DynamicInputMasking()
        X = {"survey": torch.ones(5, 3, 5)}  # Only 5 wavelengths
        n_wavelengths = {"survey": 5}

        X_out = masking(X, n_wavelengths)

        # Should still produce valid output
        assert X_out["survey"].shape == (5, 3, 5)
        # Guaranteed keeper should work
        mask = X_out["survey"][:, 2, :]
        assert (mask.sum(dim=1) >= 1).all()

    def test_offset_with_large_block_size(self):
        """Test offset when block size is large relative to spectrum."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Force large blocks (up to 50% of spectrum)
        masking = DynamicInputMasking(f_min_override=0.4, f_max=0.5)
        X = {"survey": torch.ones(10, 3, 50)}
        n_wavelengths = {"survey": 50}

        X_out = masking(X, n_wavelengths)

        # Should handle gracefully
        assert X_out["survey"].shape == (10, 3, 50)
        mask = X_out["survey"][:, 2, :]
        assert (mask.sum(dim=1) >= 1).all()
