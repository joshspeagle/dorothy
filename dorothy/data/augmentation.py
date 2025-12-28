"""
Data augmentation for spectral data.

This module implements augmentation techniques used during training to improve
model robustness:
- DynamicLabelMasking: Hierarchical label masking for missing label robustness
- DynamicInputMasking: Hierarchical input masking for missing data robustness
"""

from __future__ import annotations

import numpy as np
import torch


class DynamicLabelMasking:
    """
    Training augmentation that randomly masks labels hierarchically.

    This augmentation implements a two-level hierarchical masking scheme:
    1. Labelset level: Which label sources (apogee, galah, etc.) to include
    2. Label level: Which parameters (teff, logg, fe_h, etc.) within each set

    All probabilities are sampled fresh per batch from configurable uniform
    ranges to ensure the model sees diverse masking conditions during training.

    A guaranteed keeper mechanism ensures at least one labelset and one label
    per labelset are always kept, preventing complete masking.

    Attributes:
        p_labelset_min: Minimum probability of keeping each labelset.
        p_labelset_max: Maximum probability of keeping each labelset.
        p_label_min: Minimum probability of keeping each label.
        p_label_max: Maximum probability of keeping each label.

    Example:
        >>> masking = DynamicLabelMasking(p_label_min=0.3, p_label_max=1.0)
        >>> y_masked = masking(y_batch)  # y has shape (batch, 3, n_params)
    """

    def __init__(
        self,
        p_labelset_min: float = 0.3,
        p_labelset_max: float = 1.0,
        p_label_min: float = 0.3,
        p_label_max: float = 1.0,
    ) -> None:
        """
        Initialize the dynamic label masking augmentation.

        Args:
            p_labelset_min: Minimum probability of keeping each labelset.
            p_labelset_max: Maximum probability of keeping each labelset.
            p_label_min: Minimum probability of keeping each label.
            p_label_max: Maximum probability of keeping each label.

        Raises:
            ValueError: If probability ranges are invalid.
        """
        if not 0 <= p_labelset_min <= p_labelset_max <= 1:
            raise ValueError(
                f"Invalid labelset probability range: [{p_labelset_min}, {p_labelset_max}]"
            )
        if not 0 <= p_label_min <= p_label_max <= 1:
            raise ValueError(
                f"Invalid label probability range: [{p_label_min}, {p_label_max}]"
            )

        self.p_labelset_min = p_labelset_min
        self.p_labelset_max = p_labelset_max
        self.p_label_min = p_label_min
        self.p_label_max = p_label_max

    def __call__(
        self, y: torch.Tensor | dict[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Apply hierarchical label masking to label tensor(s).

        Args:
            y: Either a single tensor of shape (batch, 3, n_params) containing:
                - Channel 0: Label values
                - Channel 1: Label uncertainties
                - Channel 2: Mask (1=valid, 0=masked)
               Or a dict mapping labelset names to such tensors.

        Returns:
            Same structure as input with channel 2 (mask) modified.
            Values and uncertainties are zeroed where mask becomes 0.
        """
        # Sample probabilities fresh for this batch
        p_keep_labelset = np.random.uniform(self.p_labelset_min, self.p_labelset_max)
        p_keep_label = np.random.uniform(self.p_label_min, self.p_label_max)

        if isinstance(y, dict):
            return self._apply_hierarchical(y, p_keep_labelset, p_keep_label)
        else:
            return self._apply_single(y, p_keep_label)

    def _apply_single(self, y: torch.Tensor, p_keep: float) -> torch.Tensor:
        """
        Apply label masking to a single labelset (vectorized).

        Args:
            y: Tensor of shape (batch, 3, n_params).
            p_keep: Probability of keeping each label.

        Returns:
            Masked tensor with same shape.
        """
        if y.dim() != 3 or y.shape[1] != 3:
            raise ValueError(f"Expected shape (batch, 3, n_params), got {y.shape}")

        batch_size, _, n_params = y.shape
        mask = y[:, 2, :]  # (batch, n_params)
        available = mask > 0

        # Check if any samples have available labels
        any_available = available.any(dim=1)  # (batch,)
        if not any_available.any():
            # No samples have any available labels, return unchanged
            return y

        # Generate random values for vectorized selection
        rand = torch.rand(batch_size, n_params, device=y.device, dtype=y.dtype)

        # Guaranteed keeper: select one available label per sample
        # Use highest random value among available as the guaranteed keeper
        rand_for_guaranteed = rand.clone()
        rand_for_guaranteed[~available] = -float("inf")
        guaranteed_idx = rand_for_guaranteed.argmax(dim=1)  # (batch,)

        # Bernoulli mask: keep if random < p_keep AND available
        keep = (rand < p_keep) & available

        # Force-keep the guaranteed label for each sample
        batch_indices = torch.arange(batch_size, device=y.device)
        keep[batch_indices, guaranteed_idx] = True

        # Only apply to samples that have available labels
        keep = keep | ~any_available.unsqueeze(1)  # Don't modify samples with no labels

        # Apply masking
        y_out = y.clone()
        keep_float = keep.float()
        y_out[:, 2, :] = mask * keep_float  # Update mask
        y_out[:, 0, :] = y[:, 0, :] * keep_float  # Zero masked values
        y_out[:, 1, :] = y[:, 1, :] * keep_float  # Zero masked uncertainties

        return y_out

    def _apply_hierarchical(
        self,
        y_dict: dict[str, torch.Tensor],
        p_keep_labelset: float,
        p_keep_label: float,
    ) -> dict[str, torch.Tensor]:
        """
        Apply hierarchical masking to multiple labelsets.

        First masks at the labelset level, then within each kept labelset
        applies label-level masking.

        Args:
            y_dict: Dict mapping labelset names to tensors of shape (batch, 3, n_params).
            p_keep_labelset: Probability of keeping each labelset.
            p_keep_label: Probability of keeping each label within kept sets.

        Returns:
            Dict with same keys, with masking applied.
        """
        labelsets = list(y_dict.keys())
        n_labelsets = len(labelsets)

        if n_labelsets == 0:
            return y_dict

        if n_labelsets == 1:
            # Single labelset, just apply label-level masking
            return {
                labelsets[0]: self._apply_single(y_dict[labelsets[0]], p_keep_label)
            }

        # Get batch size from first labelset
        first_tensor = y_dict[labelsets[0]]
        batch_size = first_tensor.shape[0]
        device = first_tensor.device
        dtype = first_tensor.dtype

        # Determine which labelsets have any valid labels per sample
        # Shape: (batch, n_labelsets)
        labelset_available = torch.zeros(
            batch_size, n_labelsets, dtype=torch.bool, device=device
        )
        for i, name in enumerate(labelsets):
            mask = y_dict[name][:, 2, :]  # (batch, n_params)
            labelset_available[:, i] = mask.any(dim=1)  # Has at least one valid label

        # Check if any samples have available labelsets
        any_labelset_available = labelset_available.any(dim=1)  # (batch,)

        # Generate random values for labelset selection
        rand_labelset = torch.rand(batch_size, n_labelsets, device=device, dtype=dtype)

        # Guaranteed keeper: select one available labelset per sample
        rand_for_guaranteed = rand_labelset.clone()
        rand_for_guaranteed[~labelset_available] = -float("inf")
        guaranteed_labelset_idx = rand_for_guaranteed.argmax(dim=1)  # (batch,)

        # Bernoulli mask for labelsets
        keep_labelset = (rand_labelset < p_keep_labelset) & labelset_available

        # Force-keep the guaranteed labelset for each sample
        batch_indices = torch.arange(batch_size, device=device)
        keep_labelset[batch_indices, guaranteed_labelset_idx] = True

        # Don't modify samples with no available labelsets
        keep_labelset = keep_labelset | ~any_labelset_available.unsqueeze(1)

        # Apply masking to each labelset
        y_out = {}
        for i, name in enumerate(labelsets):
            y_tensor = y_dict[name]
            labelset_kept = keep_labelset[:, i]  # (batch,)

            if not labelset_kept.any():
                # This labelset is completely masked for all samples
                y_out[name] = torch.zeros_like(y_tensor)
                continue

            # Apply label-level masking
            y_masked = self._apply_single(y_tensor, p_keep_label)

            # Zero out samples where this labelset is not kept
            labelset_mask = labelset_kept.float().view(-1, 1, 1)  # (batch, 1, 1)
            y_out[name] = y_masked * labelset_mask

        return y_out

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DynamicLabelMasking("
            f"p_labelset_min={self.p_labelset_min}, "
            f"p_labelset_max={self.p_labelset_max}, "
            f"p_label_min={self.p_label_min}, "
            f"p_label_max={self.p_label_max})"
        )


class DynamicInputMasking:
    """
    Training augmentation that randomly masks input spectra hierarchically.

    This augmentation implements a two-level hierarchical masking scheme:
    1. Survey level: Which surveys to include (for multi-survey training)
    2. Block level: Which wavelength blocks within each kept survey

    Block sizes are sampled log-uniformly per batch to explore all scales
    from single pixels up to large spectral regions. All probabilities are
    sampled fresh per batch from configurable uniform ranges.

    A guaranteed keeper mechanism ensures at least one survey and one block
    per survey are always kept, preventing complete masking.

    Attributes:
        p_survey_min: Minimum probability of keeping each survey.
        p_survey_max: Maximum probability of keeping each survey.
        f_min_override: Optional override for minimum block size fraction.
        f_max: Maximum block size as fraction of spectrum.
        p_block_min: Minimum probability of keeping each block.
        p_block_max: Maximum probability of keeping each block.

    Example:
        >>> masking = DynamicInputMasking(p_block_min=0.3, p_block_max=1.0)
        >>> X_masked = masking(X_batch, n_wavelengths={"boss": 4506, "desi": 7650})
    """

    def __init__(
        self,
        p_survey_min: float = 0.3,
        p_survey_max: float = 1.0,
        f_min_override: float | None = None,
        f_max: float = 0.5,
        p_block_min: float = 0.3,
        p_block_max: float = 1.0,
    ) -> None:
        """
        Initialize the dynamic input masking augmentation.

        Args:
            p_survey_min: Minimum probability of keeping each survey.
            p_survey_max: Maximum probability of keeping each survey.
            f_min_override: Optional override for minimum block size fraction.
                If None, defaults to 1/N_wavelengths (single pixel).
            f_max: Maximum block size as fraction of spectrum.
            p_block_min: Minimum probability of keeping each block.
            p_block_max: Maximum probability of keeping each block.

        Raises:
            ValueError: If probability or fraction ranges are invalid.
        """
        if not 0 <= p_survey_min <= p_survey_max <= 1:
            raise ValueError(
                f"Invalid survey probability range: [{p_survey_min}, {p_survey_max}]"
            )
        if not 0 <= p_block_min <= p_block_max <= 1:
            raise ValueError(
                f"Invalid block probability range: [{p_block_min}, {p_block_max}]"
            )
        if f_min_override is not None and not 0 < f_min_override <= f_max <= 1:
            raise ValueError(
                f"Invalid block fraction range: [{f_min_override}, {f_max}]"
            )
        if f_max <= 0 or f_max > 1:
            raise ValueError(f"f_max must be in (0, 1], got {f_max}")

        self.p_survey_min = p_survey_min
        self.p_survey_max = p_survey_max
        self.f_min_override = f_min_override
        self.f_max = f_max
        self.p_block_min = p_block_min
        self.p_block_max = p_block_max

    def __call__(
        self,
        X: dict[str, torch.Tensor],
        n_wavelengths: dict[str, int],
    ) -> dict[str, torch.Tensor]:
        """
        Apply hierarchical input masking to multi-survey data.

        Args:
            X: Dict mapping survey names to tensors of shape (batch, 3, N_wavelengths).
                Each tensor has channels [flux, error, mask].
            n_wavelengths: Dict mapping survey names to number of wavelengths.

        Returns:
            Dict with same keys, with masking applied to each survey.
        """
        surveys = list(X.keys())
        n_surveys = len(surveys)

        if n_surveys == 0:
            return X

        # Sample survey-level probability
        p_keep_survey = np.random.uniform(self.p_survey_min, self.p_survey_max)

        # Get batch size and device from first survey
        first_tensor = X[surveys[0]]
        batch_size = first_tensor.shape[0]
        device = first_tensor.device
        dtype = first_tensor.dtype

        if n_surveys == 1:
            # Single survey, just apply block-level masking
            return {
                surveys[0]: self._apply_block_masking(
                    X[surveys[0]], n_wavelengths[surveys[0]]
                )
            }

        # Determine which surveys have any valid data per sample
        # Shape: (batch, n_surveys)
        survey_available = torch.zeros(
            batch_size, n_surveys, dtype=torch.bool, device=device
        )
        for i, name in enumerate(surveys):
            mask = X[name][:, 2, :]  # (batch, N_wavelengths)
            survey_available[:, i] = mask.any(dim=1)  # Has at least one valid pixel

        # Check if any samples have available surveys
        any_survey_available = survey_available.any(dim=1)  # (batch,)

        # Generate random values for survey selection
        rand_survey = torch.rand(batch_size, n_surveys, device=device, dtype=dtype)

        # Guaranteed keeper: select one available survey per sample
        rand_for_guaranteed = rand_survey.clone()
        rand_for_guaranteed[~survey_available] = -float("inf")
        guaranteed_survey_idx = rand_for_guaranteed.argmax(dim=1)  # (batch,)

        # Bernoulli mask for surveys
        keep_survey = (rand_survey < p_keep_survey) & survey_available

        # Force-keep the guaranteed survey for each sample
        batch_indices = torch.arange(batch_size, device=device)
        keep_survey[batch_indices, guaranteed_survey_idx] = True

        # Don't modify samples with no available surveys
        keep_survey = keep_survey | ~any_survey_available.unsqueeze(1)

        # Apply masking to each survey
        X_out = {}
        for i, name in enumerate(surveys):
            X_tensor = X[name]
            survey_kept = keep_survey[:, i]  # (batch,)

            if not survey_kept.any():
                # This survey is completely masked for all samples
                X_out[name] = torch.zeros_like(X_tensor)
                continue

            # Apply block-level masking
            X_masked = self._apply_block_masking(X_tensor, n_wavelengths[name])

            # Zero out samples where this survey is not kept
            survey_mask = survey_kept.float().view(-1, 1, 1)  # (batch, 1, 1)
            X_out[name] = X_masked * survey_mask

        return X_out

    def _apply_block_masking(self, X: torch.Tensor, N: int) -> torch.Tensor:
        """
        Apply block-based wavelength masking to a single survey (vectorized).

        Args:
            X: Tensor of shape (batch, 3, N_wavelengths).
            N: Number of wavelengths in this survey.

        Returns:
            Masked tensor with same shape.
        """
        if X.dim() != 3 or X.shape[1] != 3:
            raise ValueError(f"Expected shape (batch, 3, N), got {X.shape}")

        batch_size = X.shape[0]
        device = X.device
        dtype = X.dtype

        # Sample block size fraction (log-uniform)
        f_min = self.f_min_override if self.f_min_override is not None else (1.0 / N)
        # Ensure f_min is valid
        f_min = max(f_min, 1.0 / N)

        log_f = np.random.uniform(np.log(f_min), np.log(self.f_max))
        f = np.exp(log_f)
        block_size = max(1, int(np.ceil(f * N)))
        n_blocks = int(np.ceil(N / block_size))

        # Sample p_keep for blocks
        p_keep_block = np.random.uniform(self.p_block_min, self.p_block_max)

        # Get natural mask
        mask = X[:, 2, :]  # (batch, N)

        # Pad to multiple of block_size if needed
        pad_size = n_blocks * block_size - N
        if pad_size > 0:
            mask_padded = torch.nn.functional.pad(mask, (0, pad_size), value=0)
        else:
            mask_padded = mask

        # Reshape to (batch, n_blocks, block_size) and check if any valid per block
        mask_blocks = mask_padded.view(batch_size, n_blocks, block_size)
        block_available = mask_blocks.any(dim=2)  # (batch, n_blocks)

        # Check if any samples have available blocks
        any_block_available = block_available.any(dim=1)  # (batch,)
        if not any_block_available.any():
            # No samples have any available blocks, return unchanged
            return X

        # Generate random values for block selection
        rand_block = torch.rand(batch_size, n_blocks, device=device, dtype=dtype)

        # Guaranteed keeper: select one available block per sample
        rand_for_guaranteed = rand_block.clone()
        rand_for_guaranteed[~block_available] = -float("inf")
        guaranteed_block_idx = rand_for_guaranteed.argmax(dim=1)  # (batch,)

        # Bernoulli mask for blocks
        keep_block = (rand_block < p_keep_block) & block_available

        # Force-keep the guaranteed block for each sample
        batch_indices = torch.arange(batch_size, device=device)
        keep_block[batch_indices, guaranteed_block_idx] = True

        # Don't modify samples with no available blocks
        keep_block = keep_block | ~any_block_available.unsqueeze(1)

        # Expand block mask to wavelength mask
        # (batch, n_blocks) -> (batch, n_blocks, block_size) -> (batch, N)
        wavelength_keep = keep_block.unsqueeze(2).expand(-1, -1, block_size)
        wavelength_keep = wavelength_keep.reshape(batch_size, -1)[:, :N]

        # Apply masking
        X_out = X.clone()
        keep_float = wavelength_keep.float()
        X_out[:, 2, :] = mask * keep_float  # Update mask
        X_out[:, 0, :] = X[:, 0, :] * keep_float  # Zero masked flux
        X_out[:, 1, :] = X[:, 1, :] * keep_float  # Zero masked error

        return X_out

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DynamicInputMasking("
            f"p_survey_min={self.p_survey_min}, "
            f"p_survey_max={self.p_survey_max}, "
            f"f_min_override={self.f_min_override}, "
            f"f_max={self.f_max}, "
            f"p_block_min={self.p_block_min}, "
            f"p_block_max={self.p_block_max})"
        )
