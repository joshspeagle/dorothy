"""
Batch construction for training and evaluation.

Every training mode (single-survey, dense multi-survey, sparse multi-survey,
multi-labelset) is expressed as a :class:`BatchSource` that yields
:class:`Batch` objects with the same structure, so the trainer runs a single
train/validation loop:

- ``X``: survey name -> spectra (batch, 3, n_wavelengths) = [flux, sigma, mask]
- ``has_data``: survey name -> (batch,) bool, which stars have that survey
- ``y``: label source name -> labels (batch, 3, n_params) = [values, errors,
  mask]; stars without labels from a source have an all-zero mask.

Single-survey inputs use the survey key ``"default"``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from dorothy.data.catalogue_loader import SparseMergedData

# Survey key used for single-survey (plain MLP) inputs
SINGLE_SURVEY_KEY = "default"


@dataclass
class Batch:
    """One batch of model inputs and targets (see module docstring)."""

    X: dict[str, torch.Tensor]
    has_data: dict[str, torch.Tensor]
    y: dict[str, torch.Tensor]

    def __len__(self) -> int:
        return next(iter(self.y.values())).shape[0]


class TensorBatchSource:
    """Batches from in-memory tensors (single- or multi-survey).

    Args:
        X: Survey name -> spectra tensor (n_samples, 3, n_wavelengths).
        y: Label source name -> labels tensor (n_samples, 3, n_params).
        has_data: Survey name -> (n_samples,) bool tensor. Defaults to all True.
        device: Device batches are moved to.
    """

    def __init__(
        self,
        X: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        has_data: dict[str, torch.Tensor] | None,
        device: torch.device,
    ) -> None:
        self.X = X
        self.y = y
        self.n_samples = len(next(iter(y.values())))
        if has_data is None:
            has_data = {
                survey: torch.ones(self.n_samples, dtype=torch.bool) for survey in X
            }
        self.has_data = has_data
        self.device = device
        for name, tensor in [*X.items(), *y.items(), *has_data.items()]:
            if len(tensor) != self.n_samples:
                raise ValueError(
                    f"'{name}' has {len(tensor)} samples, expected {self.n_samples}"
                )

    def batches(
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> Iterator[Batch]:
        """Yield batches, shuffled with ``rng`` if given, else in order."""
        order = (
            rng.permutation(self.n_samples)
            if rng is not None
            else np.arange(self.n_samples)
        )
        for start in range(0, self.n_samples, batch_size):
            idx = torch.from_numpy(order[start : start + batch_size])
            yield Batch(
                X={s: x[idx].to(self.device) for s, x in self.X.items()},
                has_data={s: h[idx].to(self.device) for s, h in self.has_data.items()},
                y={k: v[idx].to(self.device) for k, v in self.y.items()},
            )


class _StagingBuffers:
    """Per-survey flux/ivar host buffers, pinned on CUDA for async copies."""

    def __init__(self, data: SparseMergedData, pinned: bool) -> None:
        self.data = data
        self.pinned = pinned
        self.size = 0
        self.buffers: dict[str, dict[str, torch.Tensor]] = {}

    def get(self, batch_size: int) -> dict[str, dict[str, torch.Tensor]]:
        """Return buffers with room for batch_size rows, growing if needed."""
        if batch_size > self.size:
            self.buffers = {
                survey: {
                    key: torch.zeros(
                        (batch_size, self.data.wavelengths[survey].shape[0]),
                        pin_memory=self.pinned,
                    )
                    for key in ("flux", "ivar")
                }
                for survey in self.data.surveys
            }
            self.size = batch_size
        return self.buffers


class SparseBatchSource:
    """Batches built on the fly from memory-efficient sparse survey storage.

    Spectra are stored only for stars that have them (see
    :class:`~dorothy.data.catalogue_loader.SparseMergedData`); dense batches
    are assembled per step. On CUDA, pinned staging buffers allow
    asynchronous host-to-device copies and sigma/mask are computed on the GPU.

    Args:
        data: Sparse survey data (spectra and index mappings).
        indices: Global star indices belonging to this source (e.g. the
            training split).
        labels: Label source name -> labels (n_total, 3, n_params), indexed by
            global star index (typically normalized copies of the data labels).
        device: Device batches are moved to.
        share_buffers_with: Another source over the same data whose staging
            buffers to reuse (e.g. validation reusing the training buffers),
            to avoid allocating extra page-locked memory.
    """

    def __init__(
        self,
        data: SparseMergedData,
        indices: np.ndarray,
        labels: dict[str, np.ndarray],
        device: torch.device,
        share_buffers_with: SparseBatchSource | None = None,
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices)
        self.labels = labels
        self.device = device
        self.n_samples = len(self.indices)
        self._staging = (
            share_buffers_with._staging
            if share_buffers_with is not None
            else _StagingBuffers(data, pinned=device.type == "cuda")
        )

    def build_batch(self, global_indices: np.ndarray) -> Batch:
        """Assemble a dense batch for the given global star indices."""
        n = len(global_indices)
        buffers = self._staging.get(n)
        X, has_data = {}, {}
        for survey in self.data.surveys:
            local_idx = self.data.global_to_local[survey][global_indices]
            present = local_idx >= 0
            flux = buffers[survey]["flux"][:n]
            ivar = buffers[survey]["ivar"][:n]
            flux.zero_()
            ivar.zero_()
            if present.any():
                rows = np.where(present)[0]
                flux.numpy()[rows] = self.data.flux[survey][local_idx[present]]
                ivar.numpy()[rows] = self.data.ivar[survey][local_idx[present]]

            # The copies below must not alias the reused staging buffers
            if self.device.type == "cpu":
                flux_dev, ivar_dev = flux.clone(), ivar.clone()
            else:
                flux_dev = flux.to(self.device, non_blocking=True)
                ivar_dev = ivar.to(self.device, non_blocking=True)
            valid = ivar_dev > 0
            sigma = torch.zeros_like(ivar_dev)
            sigma[valid] = 1.0 / torch.sqrt(ivar_dev[valid])
            X[survey] = torch.stack([flux_dev, sigma, valid.float()], dim=1)
            has_data[survey] = torch.from_numpy(present).to(self.device)

        y = {
            source: torch.from_numpy(labels[global_indices]).to(self.device)
            for source, labels in self.labels.items()
        }
        return Batch(X=X, has_data=has_data, y=y)

    def batches(
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> Iterator[Batch]:
        """Yield batches, shuffled with ``rng`` if given, else in order."""
        order = (
            self.indices[rng.permutation(self.n_samples)]
            if rng is not None
            else self.indices
        )
        for start in range(0, self.n_samples, batch_size):
            yield self.build_batch(order[start : start + batch_size])
