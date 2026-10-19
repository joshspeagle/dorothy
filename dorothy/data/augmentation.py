"""Training augmentations that mask labels and input spectra.

- :class:`DynamicLabelMasking`: hierarchical label masking.
- :class:`DynamicInputMasking`: hierarchical input masking.

Both draw their random numbers from the global numpy and torch generators.
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

    For each star, one valid labelset and one valid label per labelset are
    always kept. Stars without a valid label are unchanged.

    Attributes:
        p_labelset_min: Minimum probability of keeping each labelset.
        p_labelset_max: Maximum probability of keeping each labelset.
        p_label_min: Minimum probability of keeping each label.
        p_label_max: Maximum probability of keeping each label.

    Examples:
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
            ValueError: If a range does not satisfy 0 <= min <= max <= 1.
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
            y: Either a single tensor of shape (batch, 3, n_params) with
                channels [values, errors, mask], where mask > 0 is valid, or
                a dict mapping labelset names to such tensors. Not modified.

        Returns:
            Same structure as y. Values, errors and mask are zeroed for each
            dropped label; a dropped labelset is zeroed whole. A single
            tensor without any valid label is returned as is.

        Raises:
            ValueError: If y is a tensor, or a one-entry dict of tensors, not
                of shape (batch, 3, n_params).
        """
        # Sample probabilities fresh for this batch
        p_keep_labelset = np.random.uniform(self.p_labelset_min, self.p_labelset_max)
        p_keep_label = np.random.uniform(self.p_label_min, self.p_label_max)

        if isinstance(y, dict):
            return self._apply_hierarchical(y, p_keep_labelset, p_keep_label)
        else:
            return self._apply_single(y, p_keep_label)

    def _apply_single(
        self, y: torch.Tensor, p_keep: float, has_any_label: bool | None = None
    ) -> torch.Tensor:
        """
        Apply label masking to a single labelset (vectorized).

        Args:
            y: Tensor of shape (batch, 3, n_params).
            p_keep: Probability of keeping each label.
            has_any_label: Whether any entry of y is a valid label
                (``(y[:, 2] > 0).any()``), if already known on the host.

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
        if has_any_label is None:
            has_any_label = bool(any_available.any())
        if not has_any_label:
            # No samples have any available labels, return unchanged
            return y

        # Generate random values for vectorized selection
        rand = torch.rand(batch_size, n_params, device=y.device, dtype=y.dtype)

        # Guaranteed keeper: select one available label per sample
        # Use highest random value among available as the guaranteed keeper
        rand_for_guaranteed = rand.masked_fill(~available, -float("inf"))
        guaranteed_idx = rand_for_guaranteed.argmax(dim=1)  # (batch,)

        # Bernoulli mask: keep if random < p_keep AND available
        keep = (rand < p_keep) & available

        # Force-keep the guaranteed label for each sample
        batch_indices = torch.arange(batch_size, device=y.device)
        keep[batch_indices, guaranteed_idx] = True

        # Only apply to samples that have available labels
        keep = keep | ~any_available.unsqueeze(1)  # Don't modify samples with no labels

        # Zero the values, uncertainties and mask of the dropped labels
        return y.masked_fill(~keep.unsqueeze(1), 0.0)

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
            labelset_available[:, i] = (mask > 0).any(dim=1)

        # Check if any samples have available labelsets
        any_labelset_available = labelset_available.any(dim=1)  # (batch,)

        # Generate random values for labelset selection
        rand_labelset = torch.rand(batch_size, n_labelsets, device=device, dtype=dtype)

        # Guaranteed keeper: select one available labelset per sample
        rand_for_guaranteed = rand_labelset.masked_fill(
            ~labelset_available, -float("inf")
        )
        guaranteed_labelset_idx = rand_for_guaranteed.argmax(dim=1)  # (batch,)

        # Bernoulli mask for labelsets
        keep_labelset = (rand_labelset < p_keep_labelset) & labelset_available

        # Force-keep the guaranteed labelset for each sample
        batch_indices = torch.arange(batch_size, device=device)
        keep_labelset[batch_indices, guaranteed_labelset_idx] = True

        # Don't modify samples with no available labelsets
        keep_labelset = keep_labelset | ~any_labelset_available.unsqueeze(1)

        # Per labelset: does any sample keep it, and does it have any valid
        # label? Read in one host transfer instead of one sync per check.
        flags = torch.cat(
            [keep_labelset.any(dim=0), labelset_available.any(dim=0)]
        ).tolist()
        kept_any, has_any_label = flags[:n_labelsets], flags[n_labelsets:]

        # Apply masking to each labelset
        y_out = {}
        for i, name in enumerate(labelsets):
            y_tensor = y_dict[name]
            labelset_kept = keep_labelset[:, i]  # (batch,)

            if not kept_any[i]:
                # This labelset is completely masked for all samples
                y_out[name] = torch.zeros_like(y_tensor)
                continue

            # Apply label-level masking
            y_masked = self._apply_single(
                y_tensor, p_keep_label, has_any_label=has_any_label[i]
            )

            # Zero out samples where this labelset is not kept
            y_out[name] = y_masked.masked_fill(~labelset_kept.view(-1, 1, 1), 0.0)

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


def _zero_dropped(X: torch.Tensor, keep: torch.Tensor, inplace: bool) -> torch.Tensor:
    """Zero all three channels of X (batch, 3, N) where ``keep`` is False.

    ``keep`` has shape (batch, N) or (batch, 1). Dropped entries are selected,
    not multiplied by the mask, so they become 0 even where X holds NaN.
    """
    drop = ~keep.unsqueeze(1)
    return X.masked_fill_(drop, 0.0) if inplace else X.masked_fill(drop, 0.0)


class DynamicInputMasking:
    """
    Training augmentation that randomly masks input spectra hierarchically.

    This augmentation implements a two-level hierarchical masking scheme:
    1. Survey level: Which surveys to include (for multi-survey training)
    2. Block level: Which wavelength blocks within each kept survey

    Block sizes are sampled log-uniformly per batch to explore all scales
    from single pixels up to large spectral regions. All probabilities are
    sampled fresh per batch from configurable uniform ranges.

    For each star, one valid survey and one valid block per survey are always
    kept. Stars without a valid pixel are unchanged. A dropped block or survey
    has flux, error and mask zeroed.

    Attributes:
        p_survey_min: Minimum probability of keeping each survey.
        p_survey_max: Maximum probability of keeping each survey.
        f_min_override: Optional override for minimum block size fraction.
        f_max: Maximum block size as fraction of spectrum.
        p_block_min: Minimum probability of keeping each block.
        p_block_max: Maximum probability of keeping each block.

    Examples:
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
            f_min_override: Minimum block size as a fraction of the spectrum,
                raised to 1/N_wavelengths (one pixel) if smaller. None uses
                one pixel.
            f_max: Maximum block size as fraction of spectrum.
            p_block_min: Minimum probability of keeping each block.
            p_block_max: Maximum probability of keeping each block.

        Raises:
            ValueError: If a probability range does not satisfy
                0 <= min <= max <= 1, f_max is not in (0, 1], or
                f_min_override is not in (0, f_max].
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
        inplace: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Apply hierarchical input masking to multi-survey data.

        Args:
            X: Dict mapping survey names to tensors of shape (batch, 3, N_wavelengths).
                Each tensor has channels [flux, error, mask]; mask != 0 is
                valid.
            n_wavelengths: Survey name -> N_wavelengths, which must equal
                X[name].shape[2].
            inplace: Mask the tensors in X in place instead of returning
                masked copies (the result is identical). Saves a copy of
                every spectrum when the caller owns X.

        Returns:
            Dict with same keys, with masking applied to each survey.

        Raises:
            ValueError: If a tensor is not of shape (batch, 3, N_wavelengths).
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
                    X[surveys[0]], n_wavelengths[surveys[0]], inplace=inplace
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
        rand_for_guaranteed = rand_survey.masked_fill(~survey_available, -float("inf"))
        guaranteed_survey_idx = rand_for_guaranteed.argmax(dim=1)  # (batch,)

        # Bernoulli mask for surveys
        keep_survey = (rand_survey < p_keep_survey) & survey_available

        # Force-keep the guaranteed survey for each sample
        batch_indices = torch.arange(batch_size, device=device)
        keep_survey[batch_indices, guaranteed_survey_idx] = True

        # Don't modify samples with no available surveys
        keep_survey = keep_survey | ~any_survey_available.unsqueeze(1)

        # Per survey: does any sample keep it, and does it have any valid
        # pixel? Read in one host transfer instead of one sync per check.
        flags = torch.cat(
            [
                keep_survey.any(dim=0),
                torch.stack([(X[name][:, 2, :] != 0).any() for name in surveys]),
            ]
        ).tolist()
        kept_any, has_any_pixel = flags[:n_surveys], flags[n_surveys:]

        # Apply masking to each survey
        X_out = {}
        for i, name in enumerate(surveys):
            X_tensor = X[name]
            survey_kept = keep_survey[:, i]  # (batch,)

            if not kept_any[i]:
                # This survey is completely masked for all samples
                X_out[name] = (
                    X_tensor.zero_() if inplace else torch.zeros_like(X_tensor)
                )
                continue

            # Block-level masking, then zero out samples where this survey is
            # not kept. The masks are combined first and applied to the
            # spectra in a single pass.
            keep = self._block_keep_mask(
                X_tensor, n_wavelengths[name], has_any_pixel=has_any_pixel[i]
            )
            survey_kept = survey_kept.unsqueeze(1)  # (batch, 1)
            keep = survey_kept if keep is None else keep & survey_kept
            X_out[name] = _zero_dropped(X_tensor, keep, inplace)

        return X_out

    def _apply_block_masking(
        self, X: torch.Tensor, N: int, inplace: bool = False
    ) -> torch.Tensor:
        """
        Apply block-based wavelength masking to a single survey (vectorized).

        See :meth:`_block_keep_mask` for how blocks are drawn. Flux, error
        and mask are zeroed where a block is dropped.

        Args:
            X: Tensor of shape (batch, 3, N_wavelengths).
            N: Number of wavelengths in this survey.
            inplace: Modify X in place instead of returning a masked copy.

        Returns:
            Masked tensor with same shape (X itself if no sample has any
            valid pixel).
        """
        keep = self._block_keep_mask(X, N)
        if keep is None:
            return X
        return _zero_dropped(X, keep, inplace)

    def _block_keep_mask(
        self, X: torch.Tensor, N: int, has_any_pixel: bool | None = None
    ) -> torch.Tensor | None:
        """
        Draw the block-based wavelength keep mask for a single survey.

        Uses random offset to shift block boundaries, creating partial blocks
        at the beginning and end of the spectrum. This prevents the model from
        learning fixed positional patterns.

        Example with N=100, block_size=50, offset=20:
            Without offset: Block 0: [0:50], Block 1: [50:100] (2 blocks)
            With offset=20: Block 0 (partial): [0:30], Block 1: [30:80],
                            Block 2 (partial): [80:100] (3 blocks)

        Args:
            X: Tensor of shape (batch, 3, N_wavelengths).
            N: Number of wavelengths in this survey.
            has_any_pixel: Whether any pixel of X is valid
                (``(X[:, 2] != 0).any()``), if already known on the host.

        Returns:
            Boolean keep mask of shape (batch, N), or None if no sample has
            any valid pixel (X is then left unchanged).
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

        # Sample random offset in [0, block_size) to shift block boundaries
        offset = np.random.randint(0, block_size)

        # With offset, we need to account for the shifted structure:
        # - First partial block: [0, block_size - offset) if offset > 0
        # - Full blocks in the middle
        # - Last partial block: remainder
        # Total blocks = ceil((N + offset) / block_size)
        total_length = offset + N
        n_blocks = int(np.ceil(total_length / block_size))
        post_pad = n_blocks * block_size - total_length

        # Sample p_keep for blocks
        p_keep_block = np.random.uniform(self.p_block_min, self.p_block_max)

        # Natural mask: which pixels are valid (non-zero mask channel)
        valid = X[:, 2, :] != 0  # (batch, N)

        # Pad: add `offset` invalid pixels at front, then pad to a multiple of
        # block_size. This naturally shifts the block boundaries
        valid_padded = torch.nn.functional.pad(valid, (offset, post_pad), value=False)

        # Reshape to (batch, n_blocks, block_size) and check if any valid per block
        valid_blocks = valid_padded.view(batch_size, n_blocks, block_size)
        block_available = valid_blocks.any(dim=2)  # (batch, n_blocks)

        # Check if any samples have available blocks
        any_block_available = block_available.any(dim=1)  # (batch,)
        if has_any_pixel is None:
            has_any_pixel = bool(any_block_available.any())
        if not has_any_pixel:
            # No samples have any available blocks, leave X unchanged
            return None

        # Generate random values for block selection
        rand_block = torch.rand(batch_size, n_blocks, device=device, dtype=dtype)

        # Guaranteed keeper: select one available block per sample
        rand_for_guaranteed = rand_block.masked_fill(~block_available, -float("inf"))
        guaranteed_block_idx = rand_for_guaranteed.argmax(dim=1)  # (batch,)

        # Bernoulli mask for blocks
        keep_block = (rand_block < p_keep_block) & block_available

        # Force-keep the guaranteed block for each sample
        batch_indices = torch.arange(batch_size, device=device)
        keep_block[batch_indices, guaranteed_block_idx] = True

        # Don't modify samples with no available blocks
        keep_block = keep_block | ~any_block_available.unsqueeze(1)

        # Expand block mask to wavelength mask: wavelength j lies in padded
        # position offset + j, i.e. in block (offset + j) // block_size
        block_of_wavelength = (
            torch.arange(offset, offset + N, device=device) // block_size
        )
        return keep_block.index_select(1, block_of_wavelength)

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
