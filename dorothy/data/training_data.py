"""Load, filter and split experiment data from a config.

``dorothy train`` and ``dorothy evaluate`` both go through
:func:`load_training_data` and :class:`DataSplit`, so the stars used for
training and the held-out test set are defined in exactly one place.

Three layouts are produced, chosen by the config:

- ``single_survey``: one survey and one label source, dense arrays ``X``
  (N, 3, W) and ``y`` (N, 3, P), for a plain MLP (or a single-survey
  MultiHeadMLP).
- ``dense_multi_label``: several label sources as dense dicts
  (``data.use_dense_loading: true``; high memory).
- ``sparse_multi_survey``: every other configuration, in memory-efficient
  :class:`~dorothy.data.catalogue_loader.SparseMergedData` (the default).

Rows are the loaded arrays' rows (global catalogue indices in sparse mode).
A star is kept if it has at least one survey and at least one valid label
(any parameter, any label source); per-parameter masks handle the rest.
Splits are stored as row indices.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from dorothy.config.schema import DataConfig, ExperimentConfig
from dorothy.data.catalogue_loader import CatalogueLoader, SparseMergedData

if TYPE_CHECKING:
    import torch

    from dorothy.training.trainer import Trainer, TrainingHistory

logger = logging.getLogger(__name__)

DataKind = Literal["single_survey", "dense_multi_label", "sparse_multi_survey"]
SPLIT_FILE = "data_split.pkl"


@dataclass
class DataSplit:
    """Train/validation/test row indices.

    Attributes:
        train: Training row indices.
        val: Validation row indices.
        test: Test row indices.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def save(self, checkpoint_path: str | Path, data: TrainingData) -> Path:
        """Save the split next to a checkpoint (``data_split.pkl``).

        Args:
            checkpoint_path: Checkpoint directory.
            data: The data the split indexes; its layout and valid rows are
                saved with the split.

        Returns:
            Path of the written file.
        """
        path = Path(checkpoint_path) / SPLIT_FILE
        split_info = {
            "train_idx": np.asarray(self.train),
            "val_idx": np.asarray(self.val),
            "test_idx": np.asarray(self.test),
            "split_mode": data.kind,
            "index_space": "rows",
            "n_total": len(data.valid_indices),
            "valid_indices": np.asarray(data.valid_indices),
        }
        with open(path, "wb") as f:
            pickle.dump(split_info, f)
        return path

    @classmethod
    def load(cls, checkpoint_path: str | Path, data: TrainingData) -> DataSplit:
        """Load a saved split.

        Split files without ``index_space: rows`` (single-survey and dense
        multi-label layouts) hold indices into the filtered arrays. They are
        mapped to rows with :meth:`TrainingData.legacy_filtered_rows`.

        Args:
            checkpoint_path: Checkpoint directory holding ``data_split.pkl``.
            data: The data the split indexes.

        Returns:
            The split, as row indices.

        Raises:
            FileNotFoundError: If the directory has no ``data_split.pkl``.
            ValueError: If the split was saved for one survey and one label
                source, and ``data`` has another layout.
        """
        with open(Path(checkpoint_path) / SPLIT_FILE, "rb") as f:
            info = pickle.load(f)
        if info["split_mode"] == "single_survey" and data.kind != "single_survey":
            raise ValueError(
                f"The split in {checkpoint_path} indexes the single_survey layout, "
                f"but the data now load as {data.kind}. Older versions trained "
                "only the first label source of one survey with several label "
                "sources; retrain the model."
            )
        idx = {k: np.asarray(info[f"{k}_idx"]) for k in ("train", "val", "test")}
        if info.get("index_space") != "rows" and info["split_mode"] != (
            "sparse_multi_survey"
        ):
            rows = data.legacy_filtered_rows()
            idx = {k: rows[v] for k, v in idx.items()}
        return cls(**idx)


@dataclass
class TrainingData:
    """Experiment data loaded according to a config.

    Only the fields for ``kind`` are set (see module docstring).

    Attributes:
        kind: Data layout.
        label_sources: Label source names, in config order.
        valid_indices: Rows kept for training/evaluation.
        X, y: Single-survey spectra (N, 3, W) and labels (N, 3, P).
        X_dict, y_dict, has_data, has_labels: Dense multi-survey arrays keyed
            by survey (X_dict, has_data) or label source (y_dict, has_labels).
        sparse: Sparse multi-survey data.
    """

    kind: DataKind
    label_sources: list[str]
    valid_indices: np.ndarray
    X: np.ndarray | None = None
    y: np.ndarray | None = None
    X_dict: dict[str, np.ndarray] | None = None
    y_dict: dict[str, np.ndarray] | None = None
    has_data: dict[str, np.ndarray] | None = None
    has_labels: dict[str, np.ndarray] | None = None
    sparse: SparseMergedData | None = None

    @property
    def n_rows(self) -> int:
        """Number of loaded rows (stars)."""
        if self.kind == "single_survey":
            return len(self.y)
        if self.kind == "dense_multi_label":
            return len(next(iter(self.y_dict.values())))
        return self.sparse.n_total

    def labels(self, source: str | None = None) -> np.ndarray:
        """Physical-unit labels for a label source.

        Args:
            source: Label source name. None, or a name the sparse data does
                not hold separately, gives the primary labels.

        Returns:
            Array of shape (n_rows, 3, n_params) = [values, errors, mask].
        """
        if self.kind == "single_survey":
            return self.y
        if self.kind == "dense_multi_label":
            return self.y_dict[source or self.label_sources[0]]
        data = self.sparse
        if data.labels_dict is not None and source in data.labels_dict:
            return data.labels_dict[source]
        return data.labels

    def split(self, train_ratio: float, val_ratio: float, seed: int) -> DataSplit:
        """Randomly split the valid rows into train/validation/test.

        Permutes the valid rows with ``numpy.random.RandomState(seed)``. The
        first ``int(n * train_ratio)`` go to train, the next
        ``int(n * val_ratio)`` to validation and the rest to test, where n is
        the number of valid rows.

        Args:
            train_ratio: Fraction of valid rows for training.
            val_ratio: Fraction of valid rows for validation.
            seed: Seed of the permutation.

        Returns:
            DataSplit of row indices.
        """
        n = len(self.valid_indices)
        perm = np.random.RandomState(seed).permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        rows = self.valid_indices
        return DataSplit(
            train=rows[perm[:n_train]],
            val=rows[perm[n_train : n_train + n_val]],
            test=rows[perm[n_train + n_val :]],
        )

    def configure_model(self, config: ExperimentConfig) -> None:
        """Set the MLP input features of a single-survey config, in place.

        Args:
            config: Experiment config, modified in place. Other layouts and
                configs without ``model`` are left unchanged.
        """
        if self.kind == "single_survey" and config.model is not None:
            config.model = config.model.model_copy(
                update={"input_features": self.X.shape[1] * self.X.shape[2]}
            )

    def fit(self, trainer: Trainer, split: DataSplit) -> TrainingHistory:
        """Train with the fit method matching this data layout.

        Args:
            trainer: The trainer.
            split: Row indices; only train and val are used.

        Returns:
            The training history.
        """
        tr, va = split.train, split.val
        if self.kind == "single_survey":
            return trainer.fit(self.X[tr], self.y[tr], self.X[va], self.y[va])
        if self.kind == "dense_multi_label":

            def take(d, idx):
                return {k: v[idx] for k, v in d.items()}

            return trainer.fit_multi_labelset(
                take(self.X_dict, tr),
                take(self.y_dict, tr),
                take(self.X_dict, va),
                take(self.y_dict, va),
                take(self.has_data, tr),
                take(self.has_data, va),
                take(self.has_labels, tr),
                take(self.has_labels, va),
            )
        return trainer.fit_multi_survey_sparse(self.sparse, tr, va)

    def batch_source(
        self,
        rows: np.ndarray,
        labels: dict[str, np.ndarray],
        device: torch.device,
        survey_key: str | None = None,
    ):
        """Batch source over the given rows, for evaluation.

        Args:
            rows: Row indices to iterate over.
            labels: Label source name -> labels (n_rows, 3, P), e.g. normalized.
            device: Device batches are moved to.
            survey_key: Input key for single-survey data (the survey name for
                a MultiHeadMLP, otherwise the default key).

        Returns:
            A sparse or tensor batch source over rows, with ``batches(size)``.
        """
        import torch

        from dorothy.training.batches import (
            SINGLE_SURVEY_KEY,
            SparseBatchSource,
            TensorBatchSource,
        )

        if self.kind == "sparse_multi_survey":
            return SparseBatchSource(self.sparse, rows, labels, device)
        y = {k: torch.as_tensor(v[rows]) for k, v in labels.items()}
        if self.kind == "single_survey":
            key = survey_key or SINGLE_SURVEY_KEY
            return TensorBatchSource(
                {key: torch.as_tensor(self.X[rows])}, y, None, device
            )
        return TensorBatchSource(
            {s: torch.as_tensor(x[rows]) for s, x in self.X_dict.items()},
            y,
            {s: torch.as_tensor(h[rows]) for s, h in self.has_data.items()},
            device,
        )

    def legacy_filtered_rows(self) -> np.ndarray:
        """Rows kept by the filter that old split files indexed into.

        Returns:
            Row indices. Single-survey: rows with a valid label. Dense
            multi-label: rows with a survey and a positive label error in any
            source. Sparse: ``valid_indices``.
        """
        if self.kind == "single_survey":
            return np.where(self.y[:, 2, :].any(axis=1))[0]
        if self.kind == "dense_multi_label":
            has_survey = np.any(list(self.has_data.values()), axis=0)
            # That filter counted a star as labelled when a label error was
            # positive, whether or not the flag filter kept the label
            has_error = [(y[:, 1, :] > 0).any(axis=1) for y in self.y_dict.values()]
            return np.where(has_survey & np.any(has_error, axis=0))[0]
        return self.valid_indices


def complete_config_from_catalogue(config_dict: dict, verbose: bool = False) -> dict:
    """Fill in a raw config's data-dependent settings from its catalogue.

    Sets (from the catalogue, the authoritative source):

    - ``multi_head_model.survey_wavelengths`` for multi-head / multi-survey
      configs, and ``multi_head_model.label_sources`` when several label
      sources are configured;
    - ``data.parameter_names`` (an explicit value must match the catalogue).

    Args:
        config_dict: Raw config mapping (e.g. parsed YAML); not modified.
        verbose: Print what was filled in.

    Returns:
        A completed copy of the config mapping, ready for ExperimentConfig.

    Raises:
        FileNotFoundError: If the catalogue does not exist.
        ValueError: If ``data.catalogue_path`` is missing, the catalogue
            cannot provide the settings, or explicit parameter names disagree
            with it.
    """
    import copy

    config_dict = copy.deepcopy(config_dict)
    data_cfg = config_dict.setdefault("data", {})
    catalogue_path = data_cfg.get("catalogue_path")
    if catalogue_path is None:
        raise ValueError("data.catalogue_path is required")
    # Validating resolves the survey/label_source aliases and the defaults
    data = DataConfig.model_validate(data_cfg)
    surveys, label_sources = data.surveys, data.label_sources
    loader = CatalogueLoader(catalogue_path)

    if "multi_head_model" in config_dict or len(surveys) > 1 or len(label_sources) > 1:
        try:
            wavelengths = loader.get_survey_wavelength_counts(surveys)
        except Exception as e:
            raise ValueError(f"Could not get survey wavelengths: {e}") from e
        mh = config_dict.get("multi_head_model") or {}
        mh["survey_wavelengths"] = wavelengths
        if len(label_sources) > 1:
            mh["label_sources"] = list(label_sources)
        config_dict["multi_head_model"] = mh
        if verbose:
            print(f"  Survey wavelengths: {wavelengths}")

    try:
        catalogue_params = loader.get_parameter_names()
    except Exception as e:
        raise ValueError(f"Could not read parameter names from catalogue: {e}") from e
    user_params = data_cfg.get("parameter_names")
    if user_params is not None and list(user_params) != catalogue_params:
        raise ValueError(
            f"data.parameter_names {list(user_params)} does not match the "
            f"catalogue's parameters {catalogue_params}"
        )
    data_cfg["parameter_names"] = catalogue_params
    if verbose:
        print(f"  Parameters ({len(catalogue_params)}): {', '.join(catalogue_params)}")
    return config_dict


def _valid_rows(has_survey: np.ndarray, label_arrays: list[np.ndarray]) -> np.ndarray:
    """Rows with at least one survey and at least one valid label."""
    has_label = np.zeros(len(has_survey), dtype=bool)
    for y in label_arrays:
        has_label |= (y[:, 2, :] > 0).any(axis=1)
    return np.where(has_survey & has_label)[0]


def _apply_duplicate_labels(
    data: SparseMergedData, duplicate_labels: dict[str, str]
) -> None:
    """Copy one label source to another (testing aid, ``data.duplicate_labels``)."""
    for target, source in duplicate_labels.items():
        if data.labels_dict is None:
            has_labels = np.any(data.labels[:, 2, :] > 0, axis=1)
            data.labels_dict = {source: data.labels.copy()}
            data.has_labels_dict = {source: has_labels}
            data.label_sources = [source]
        if source not in data.labels_dict:
            raise ValueError(f"Cannot duplicate from '{source}': not in loaded sources")
        data.labels_dict[target] = data.labels_dict[source].copy()
        data.has_labels_dict[target] = data.has_labels_dict[source].copy()
        if target not in data.label_sources:
            data.label_sources.append(target)


def load_training_data(
    config: ExperimentConfig,
    loader: CatalogueLoader | None = None,
    verbose: bool = True,
) -> TrainingData:
    """Load the data an experiment config describes.

    Args:
        config: Experiment configuration.
        loader: Catalogue loader (created from ``data.catalogue_path`` if None).
        verbose: Print a short summary.

    Returns:
        TrainingData with the valid rows identified.

    Raises:
        FileNotFoundError: If no loader is given and the catalogue does not
            exist.
        ValueError: If the catalogue's parameters don't match the config, or a
            label source to duplicate from is not loaded.
    """
    data_cfg = config.data
    loader = loader or CatalogueLoader(data_cfg.catalogue_path)
    catalogue_params = loader.get_parameter_names()
    if data_cfg.parameter_names is not None:
        if list(data_cfg.parameter_names) != catalogue_params:
            raise ValueError(
                f"Config parameters {list(data_cfg.parameter_names)} do not match the "
                f"catalogue's parameters {catalogue_params}"
            )
    elif len(config.parameter_names) != len(catalogue_params):
        raise ValueError(
            f"Model predicts {len(config.parameter_names)} parameters but the "
            f"catalogue has {len(catalogue_params)} ({catalogue_params})"
        )
    duplicate = data_cfg.duplicate_labels or {}
    if duplicate and verbose:
        print(f"  Duplicating labels: {duplicate}")
    sources = list(data_cfg.label_sources)
    to_load = [s for s in sources if s not in duplicate] or sources[:1]
    common = {
        "max_flag_bits": data_cfg.max_flag_bits,
        "normalize_spectra": data_cfg.normalize_spectra,
        "precision_encoding": data_cfg.precision_encoding,
    }

    if not data_cfg.is_multi_survey and not data_cfg.is_multi_label:
        survey = data_cfg.surveys[0]
        X, y = loader.load_for_training(
            survey=survey, label_source=to_load[0], **common
        )
        valid = _valid_rows(np.ones(len(y), dtype=bool), [y])
        result = TrainingData("single_survey", sources, valid, X=X, y=y)
        if verbose:
            print(f"  Survey: {survey} ({X.shape[2]} wavelength bins)")

    elif data_cfg.is_multi_label and data_cfg.use_dense_loading:
        X_dict, y_dict, has_data, has_labels = loader.load_merged_for_training(
            surveys=data_cfg.surveys,
            label_source=to_load[0],
            label_sources=to_load,
            **common,
        )
        for target, source in duplicate.items():
            if source not in y_dict:
                raise ValueError(
                    f"Cannot duplicate from '{source}': not in loaded sources"
                )
            y_dict[target] = y_dict[source].copy()
            has_labels[target] = has_labels[source].copy()
        has_survey = np.any(list(has_data.values()), axis=0)
        valid = _valid_rows(has_survey, [y_dict[s] for s in sources])
        result = TrainingData(
            "dense_multi_label",
            sources,
            valid,
            X_dict=X_dict,
            y_dict=y_dict,
            has_data=has_data,
            has_labels=has_labels,
        )

    else:
        multi = data_cfg.is_multi_label and len(to_load) > 1
        sparse = loader.load_merged_sparse(
            surveys=data_cfg.surveys,
            label_source=to_load[0],
            label_sources=to_load if multi else None,
            **common,
        )
        if duplicate:
            _apply_duplicate_labels(sparse, duplicate)
        has_survey = np.zeros(sparse.n_total, dtype=bool)
        for survey in sparse.surveys:
            has_survey |= sparse.has_data(survey)
        label_arrays = (
            [sparse.labels_dict[s] for s in sources if s in sparse.labels_dict]
            if sparse.labels_dict is not None
            else [sparse.labels]
        )
        valid = _valid_rows(has_survey, label_arrays)
        result = TrainingData("sparse_multi_survey", sources, valid, sparse=sparse)

    if verbose:
        print(f"  Loaded {result.n_rows:,} stars ({result.kind})")
        print(
            f"  Valid (>= 1 survey and >= 1 valid label): {len(result.valid_indices):,}"
        )
    return result
