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
from dorothy.models.mlp import MLP
from dorothy.models.multi_head_mlp import MultiHeadMLP

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

    @property
    def label_keys(self) -> list[str]:
        """Label sources this source provides targets for."""
        return list(self.y)

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
                X={s: _rows_to(x, idx, self.device) for s, x in self.X.items()},
                has_data={
                    s: _rows_to(h, idx, self.device) for s, h in self.has_data.items()
                },
                y={k: _rows_to(v, idx, self.device) for k, v in self.y.items()},
            )


def _rows_to(
    tensor: torch.Tensor, idx: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Rows ``idx`` of a tensor, as a new tensor on ``device``.

    Host rows bound for CUDA are gathered into pinned memory and copied
    asynchronously. PyTorch's caching host allocator keeps a pinned block
    from being reused until its copy has completed, so no host sync is needed.
    """
    if tensor.device.type != "cpu":
        return tensor[idx].to(device)
    pin = device.type == "cuda"
    out = torch.empty((len(idx), *tensor.shape[1:]), dtype=tensor.dtype, pin_memory=pin)
    torch.index_select(tensor, 0, idx, out=out)
    return out.to(device, non_blocking=pin)


def _numpy_to(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """A host array as a tensor on ``device`` (async pinned copy on CUDA)."""
    tensor = torch.from_numpy(array)
    if device.type != "cuda":
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


class SparseBatchSource:
    """Batches built on the fly from memory-efficient sparse survey storage.

    Spectra are stored only for stars that have them (see
    :class:`~dorothy.data.catalogue_loader.SparseMergedData`); dense batches
    are assembled per step. Only the rows of stars that have a survey are
    gathered, moved to the device and given a mask; the other
    rows stay zero. On CUDA the gathered rows go through pinned memory with
    asynchronous copies, and nothing in batch assembly waits for the GPU.

    Args:
        data: Sparse survey data (spectra and index mappings).
        indices: Global star indices belonging to this source (e.g. the
            training split).
        labels: Label source name -> labels (n_total, 3, n_params), indexed by
            global star index (typically normalized copies of the data labels).
        device: Device batches are moved to.
    """

    def __init__(
        self,
        data: SparseMergedData,
        indices: np.ndarray,
        labels: dict[str, np.ndarray],
        device: torch.device,
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices)
        self.labels = labels
        self.device = device
        self.n_samples = len(self.indices)

    def _gather(self, array: np.ndarray, local: np.ndarray) -> torch.Tensor:
        """Rows ``local`` of a sparse survey array, as a device tensor."""
        source = torch.from_numpy(array)
        index = torch.from_numpy(np.asarray(local, dtype=np.int64))
        dtype = torch.get_default_dtype()
        if source.dtype != dtype:
            return source.index_select(0, index).to(self.device, dtype)
        return _rows_to(source, index, self.device)

    @property
    def label_keys(self) -> list[str]:
        """Label sources this source provides targets for."""
        return list(self.labels)

    def build_batch(self, global_indices: np.ndarray) -> Batch:
        """Assemble a dense batch for the given global star indices.

        Each survey gives X (batch, 3, n_wavelengths) = [flux, sigma, mask]
        with mask = 1 where sigma > 0, and all-zero rows for stars without
        that survey.
        """
        n = len(global_indices)
        X, has_data = {}, {}
        for survey in self.data.surveys:
            local_idx = self.data.global_to_local[survey][global_indices]
            present = local_idx >= 0
            rows = np.flatnonzero(present)
            n_wave = self.data.wavelengths[survey].shape[0]
            x = torch.zeros((n, 3, n_wave), device=self.device)
            if len(rows):
                flux = self._gather(self.data.flux[survey], local_idx[rows])
                sigma = self._gather(self.data.sigma[survey], local_idx[rows])
                valid = sigma > 0
                if len(rows) == n:
                    x[:, 0], x[:, 1], x[:, 2] = flux, sigma, valid
                else:
                    rows_dev = _numpy_to(rows, self.device)
                    x[:, 0].index_copy_(0, rows_dev, flux)
                    x[:, 1].index_copy_(0, rows_dev, sigma)
                    x[:, 2].index_copy_(0, rows_dev, valid.to(x.dtype))
            X[survey] = x
            has_data[survey] = _numpy_to(present, self.device)

        y = {
            source: _numpy_to(labels[global_indices], self.device)
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


def forward_label_sources(
    model: MLP | MultiHeadMLP, batch: Batch, label_keys: list[str]
) -> dict[str, torch.Tensor]:
    """Run the model on a batch, returning (batch, 2, n_params) per label source.

    A multi-label MultiHeadMLP returns every head; a single-head model's
    output is keyed by ``label_keys[0]``.
    """
    if isinstance(model, MultiHeadMLP):
        if model.is_multi_label:
            return model.forward_all_label_sources(batch.X, has_data=batch.has_data)
        output = model(batch.X, has_data=batch.has_data)
    else:
        output = model(batch.X[SINGLE_SURVEY_KEY])
    return {label_keys[0]: output}
