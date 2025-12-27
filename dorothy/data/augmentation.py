"""
Data augmentation for spectral data.

This module implements augmentation techniques used during training to improve
model robustness, including dynamic block masking that simulates missing data.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray


class DynamicBlockMasking:
    """
    Training augmentation that randomly masks contiguous blocks of spectrum.

    This augmentation simulates missing data by randomly selecting contiguous
    blocks of wavelengths to mask out. The fraction of masked wavelengths and
    the block sizes are randomized to create diverse training examples.

    The augmentation works on 3-channel input [flux | error | mask] and updates
    the mask channel by combining the original mask with the augmentation mask.

    Attributes:
        min_fraction: Minimum fraction of wavelengths to mask (default 0.1).
        max_fraction: Maximum fraction of wavelengths to mask (default 0.8).
        fraction_choices: Optional list of specific fractions to choose from.
        min_block_size: Minimum size of each masked block (default 5).
        max_block_size: Maximum size of each masked block (default n_wavelengths//2).

    Example:
        >>> augment = DynamicBlockMasking(min_fraction=0.1, max_fraction=0.5)
        >>> X_augmented = augment(X_train)  # X has shape (batch, 3, n_wavelengths)
    """

    def __init__(
        self,
        min_fraction: float = 0.1,
        max_fraction: float = 0.8,
        fraction_choices: list[float] | None = None,
        min_block_size: int = 5,
        max_block_size: int | None = None,
    ) -> None:
        """
        Initialize the dynamic block masking augmentation.

        Args:
            min_fraction: Minimum fraction of wavelengths to mask (0 to 1).
            max_fraction: Maximum fraction of wavelengths to mask (0 to 1).
            fraction_choices: Optional list of specific fractions to choose from.
                If provided, min_fraction and max_fraction are ignored.
            min_block_size: Minimum size of each masked block.
            max_block_size: Maximum size of each masked block. If None, defaults
                to n_wavelengths // 2 at runtime.

        Raises:
            ValueError: If fractions are invalid or block size is invalid.
        """
        if fraction_choices is not None:
            if not all(0 <= f <= 1 for f in fraction_choices):
                raise ValueError("fraction_choices must all be between 0 and 1")
            self.fraction_choices = fraction_choices
            self.min_fraction = min(fraction_choices)
            self.max_fraction = max(fraction_choices)
        else:
            if not 0 <= min_fraction <= 1:
                raise ValueError(
                    f"min_fraction must be between 0 and 1, got {min_fraction}"
                )
            if not 0 <= max_fraction <= 1:
                raise ValueError(
                    f"max_fraction must be between 0 and 1, got {max_fraction}"
                )
            if min_fraction > max_fraction:
                raise ValueError(
                    f"min_fraction ({min_fraction}) must be <= max_fraction ({max_fraction})"
                )
            self.fraction_choices = None
            self.min_fraction = min_fraction
            self.max_fraction = max_fraction

        if min_block_size < 1:
            raise ValueError(f"min_block_size must be >= 1, got {min_block_size}")

        self.min_block_size = min_block_size
        self.max_block_size = max_block_size

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply random block masking to a batch.

        Args:
            X: Input tensor of shape (batch_size, 3, n_wavelengths) containing:
                - Channel 0: Flux
                - Channel 1: Error
                - Channel 2: Mask (1=valid, 0=masked)

        Returns:
            Augmented tensor with same shape where:
                - Channels 0-1 are zeroed at newly masked positions
                - Channel 2 is updated: new_mask = old_mask AND block_mask
        """
        if X.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor (batch, channels, wavelengths), got {X.dim()}D"
            )

        batch_size, n_channels, n_wavelengths = X.shape

        if n_channels != 3:
            raise ValueError(f"Expected 3 channels, got {n_channels}")

        # Work on a copy
        X_out = X.clone()

        # Get max block size
        max_block = (
            self.max_block_size
            if self.max_block_size is not None
            else n_wavelengths // 2
        )
        max_block = min(max_block, n_wavelengths)

        # Process each sample independently
        for i in range(batch_size):
            # Determine masking fraction for this sample
            if self.fraction_choices is not None:
                fraction = np.random.choice(self.fraction_choices)
            else:
                fraction = np.random.uniform(self.min_fraction, self.max_fraction)

            # Create augmentation mask (1 = keep, 0 = mask)
            aug_mask = self._create_block_mask(n_wavelengths, fraction, max_block)

            # Convert to tensor on same device
            aug_mask_tensor = torch.tensor(aug_mask, dtype=X.dtype, device=X.device)

            # Combine with existing mask: new_mask = old_mask AND aug_mask
            old_mask = X_out[i, 2, :]
            new_mask = old_mask * aug_mask_tensor

            # Update channels
            X_out[i, 2, :] = new_mask
            X_out[i, 0, :] = X_out[i, 0, :] * aug_mask_tensor  # Flux
            X_out[i, 1, :] = X_out[i, 1, :] * aug_mask_tensor  # Error

        return X_out

    def _create_block_mask(
        self,
        n_wavelengths: int,
        target_fraction: float,
        max_block_size: int,
    ) -> NDArray[np.float32]:
        """
        Create a random block mask.

        Args:
            n_wavelengths: Total number of wavelength bins.
            target_fraction: Target fraction of wavelengths to mask.
            max_block_size: Maximum size of each block.

        Returns:
            Mask array of shape (n_wavelengths,) with 1=keep, 0=mask.
        """
        target_masked = int(n_wavelengths * target_fraction)

        if target_masked == 0:
            return np.ones(n_wavelengths, dtype=np.float32)

        mask = np.ones(n_wavelengths, dtype=np.float32)
        masked_count = 0

        # Keep adding blocks until we reach target
        max_attempts = 100  # Prevent infinite loops
        attempts = 0

        while masked_count < target_masked and attempts < max_attempts:
            attempts += 1

            # Random block size
            remaining = target_masked - masked_count
            max_block_for_this = min(max_block_size, remaining, n_wavelengths)

            # Ensure valid range for randint
            if max_block_for_this < self.min_block_size:
                # If remaining is less than min_block_size, use remaining or min_block_size
                block_size = min(self.min_block_size, n_wavelengths)
            else:
                block_size = np.random.randint(
                    self.min_block_size,
                    max_block_for_this + 1,
                )

            # Random start position
            max_start = n_wavelengths - block_size
            if max_start < 0:
                break

            start = np.random.randint(0, max_start + 1)

            # Count how many new wavelengths we're masking
            new_masked = np.sum(mask[start : start + block_size])

            # Apply mask
            mask[start : start + block_size] = 0
            masked_count += int(new_masked)

        return mask

    def __repr__(self) -> str:
        """Return string representation."""
        if self.fraction_choices is not None:
            frac_str = f"fraction_choices={self.fraction_choices}"
        else:
            frac_str = (
                f"min_fraction={self.min_fraction}, max_fraction={self.max_fraction}"
            )
        return (
            f"DynamicBlockMasking({frac_str}, "
            f"min_block_size={self.min_block_size}, "
            f"max_block_size={self.max_block_size})"
        )
