"""
Test-set evaluation that matches training validation exactly.

:func:`evaluate_model` runs a trained model over a set of rows using the same
batch construction, label normalization and predicted-scatter definition as
the trainer, and scores every output head (label source) in both normalized
space (directly comparable to the training validation metrics) and physical
units. It also returns the physical-unit predictions for plotting.

The older per-layout helpers (``evaluate_on_test_set``,
``evaluate_on_dense_data``, ``evaluate_on_single_survey_data`` and the
``normalize_*`` / ``build_batch_from_sparse`` utilities) are kept as thin
wrappers for backward compatibility; they report the primary head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from dorothy.data.catalogue_loader import SparseMergedData
from dorothy.data.training_data import TrainingData
from dorothy.inference.evaluator import EvaluationResult, Evaluator
from dorothy.losses.heteroscedastic import predicted_scatter
from dorothy.models.multi_head_mlp import MultiHeadMLP
from dorothy.training.batches import SINGLE_SURVEY_KEY, forward_label_sources
from dorothy.utils import resolve_device

if TYPE_CHECKING:
    from dorothy.config.schema import ExperimentConfig
    from dorothy.data.normalizer import LabelNormalizer


@dataclass
class HeadEvaluation:
    """Evaluation of one output head (label source) on a set of stars.

    Attributes:
        label_source: Label source name of the head.
        normalized: Metrics in normalized space (comparable to training
            validation), or in raw model units without a normalizer.
        physical: Metrics in physical units (None without a normalizer).
        parameter_names: Parameter names, in column order.
        y_pred, pred_scatter, y_true, y_err, mask: Physical-unit arrays of
            shape (n_stars, n_params) (model units without a normalizer).
        z_label_errors: The label errors that the z-scores add to the
            predicted scatter: ``y_err`` for a model with an AdditiveOutput,
            else None.
    """

    label_source: str
    normalized: EvaluationResult
    physical: EvaluationResult | None
    parameter_names: list[str]
    y_pred: np.ndarray
    pred_scatter: np.ndarray
    y_true: np.ndarray
    y_err: np.ndarray
    mask: np.ndarray
    z_label_errors: np.ndarray | None

    def to_dict(self) -> dict:
        """Metrics as a JSON-serializable dict."""
        out = {"normalized_space": self.normalized.to_dict()}
        if self.physical is not None:
            out["physical_space"] = self.physical.to_dict()
        return out


def _head_names(model: nn.Module, config: ExperimentConfig) -> list[str]:
    """Label source of each output head (the configured one for a single head)."""
    if isinstance(model, MultiHeadMLP) and model.label_sources != ["default"]:
        return list(model.label_sources)
    return list(config.data.label_sources[:1])


def evaluate_model(
    model: nn.Module,
    normalizer: LabelNormalizer | None,
    config: ExperimentConfig,
    data: TrainingData,
    rows: np.ndarray,
    batch_size: int = 1024,
    device: str | torch.device = "auto",
) -> dict[str, HeadEvaluation]:
    """Evaluate a trained model on the given rows of a dataset.

    The model reads no label errors, so it predicts with its missing vector.
    The z-scores divide the residuals by s for a model with label-aware
    uncertainty, and by sqrt(sigma_y^2 + s^2) for the additive model of
    checkpoints saved before it.

    Args:
        model: Trained MLP or MultiHeadMLP (moved to ``device``, eval mode).
        normalizer: Label normalizer from training (None if not used).
        config: The checkpoint's experiment config.
        data: Dataset loaded with :func:`~dorothy.data.training_data.load_training_data`.
        rows: Row indices to evaluate (e.g. the test split).
        batch_size: Batch size for inference.
        device: Device for inference.

    Returns:
        Label source name -> HeadEvaluation, in the model's head order. A
        single-head model's head is named after the configured label source.
    """
    device = resolve_device(device)
    model.to(device)
    model.eval()
    rows = np.asarray(rows)
    keys = _head_names(model, config)
    parameter_names = (
        list(normalizer.parameters)
        if normalizer is not None
        else config.parameter_names
    )
    floor = config.training.scatter_floor

    # Physical labels per head; a single-head model uses the primary labels
    physical = {k: data.labels(k if len(keys) > 1 else None) for k in keys}
    model_labels = (
        {k: normalizer.transform_labels(v) for k, v in physical.items()}
        if normalizer is not None
        else physical
    )
    survey_key = None
    if isinstance(model, MultiHeadMLP) and data.kind == "single_survey":
        survey_key = model.survey_names[0]
    source = data.batch_source(rows, model_labels, device, survey_key=survey_key)

    outputs = {k: [] for k in keys}
    with torch.no_grad():
        for batch in source.batches(batch_size):
            batch_out = forward_label_sources(model, batch, keys)
            for k in keys:
                outputs[k].append(batch_out[k].float().cpu())

    evaluator = Evaluator(parameter_names, teff_in_log=False, scatter_floor=floor)
    additive = not model.label_aware_uncertainty
    results = {}
    for k in keys:
        out = torch.cat(outputs[k])
        pred = out[:, 0, :].numpy()
        scatter = predicted_scatter(out[:, 1, :], floor).numpy()
        target = model_labels[k][rows]
        mask = target[:, 2, :]
        normalized = evaluator.evaluate(
            y_pred=pred,
            y_true=target[:, 0, :],
            pred_scatter=scatter,
            label_errors=target[:, 1, :] if additive else None,
            mask=mask,
        )
        result_physical = None
        y_true, y_err = target[:, 0, :], target[:, 1, :]
        if normalizer is not None:
            pred, scatter = normalizer.inverse_transform(pred, scatter)
            y_true, y_err = physical[k][rows, 0, :], physical[k][rows, 1, :]
            result_physical = evaluator.evaluate(
                y_pred=pred,
                y_true=y_true,
                pred_scatter=scatter,
                label_errors=y_err if additive else None,
                mask=mask,
            )
        results[k] = HeadEvaluation(
            label_source=k,
            normalized=normalized,
            physical=result_physical,
            parameter_names=parameter_names,
            y_pred=pred,
            pred_scatter=scatter,
            y_true=y_true,
            y_err=y_err,
            mask=mask,
            z_label_errors=y_err if additive else None,
        )
    return results


# ---------------------------------------------------------------------------
# Backward-compatible wrappers (primary head only)
# ---------------------------------------------------------------------------


def _primary(results: dict[str, HeadEvaluation]):
    head = next(iter(results.values()))
    return head.normalized, head.physical


def evaluate_on_test_set(
    model: nn.Module,
    data: SparseMergedData,
    test_indices: np.ndarray,
    normalizer: LabelNormalizer | None,
    config: ExperimentConfig,
    batch_size: int = 1024,
    device: str | torch.device = "auto",
) -> tuple[EvaluationResult, EvaluationResult | None]:
    """Evaluate on sparse multi-survey data (primary head).

    Returns:
        Tuple of (normalized_space_result, physical_space_result).
    """
    td = TrainingData(
        "sparse_multi_survey",
        list(data.label_sources or config.data.label_sources),
        np.asarray(test_indices),
        sparse=data,
    )
    return _primary(
        evaluate_model(model, normalizer, config, td, test_indices, batch_size, device)
    )


def evaluate_on_dense_data(
    model: nn.Module,
    X_dict: dict[str, np.ndarray],
    y_dict: dict[str, np.ndarray],
    has_data_dict: dict[str, np.ndarray],
    test_indices: np.ndarray,
    normalizer: LabelNormalizer | None,
    config: ExperimentConfig,
    batch_size: int = 1024,
    device: str | torch.device = "auto",
) -> tuple[EvaluationResult, EvaluationResult | None]:
    """Evaluate on dense multi-survey dict data (primary head).

    Returns:
        Tuple of (normalized_space_result, physical_space_result).
    """
    td = TrainingData(
        "dense_multi_label",
        list(y_dict),
        np.asarray(test_indices),
        X_dict=X_dict,
        y_dict=y_dict,
        has_data=has_data_dict,
    )
    return _primary(
        evaluate_model(model, normalizer, config, td, test_indices, batch_size, device)
    )


def evaluate_on_single_survey_data(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    test_indices: np.ndarray,
    normalizer: LabelNormalizer | None,
    config: ExperimentConfig,
    batch_size: int = 1024,
    device: str | torch.device = "auto",
) -> tuple[EvaluationResult, EvaluationResult | None]:
    """Evaluate on single-survey arrays (X (N, 3, W), y (N, 3, P)).

    Returns:
        Tuple of (normalized_space_result, physical_space_result).
    """
    td = TrainingData(
        "single_survey",
        config.data.label_sources[:1],
        np.asarray(test_indices),
        X=X,
        y=y,
    )
    return _primary(
        evaluate_model(model, normalizer, config, td, test_indices, batch_size, device)
    )


def build_batch_from_sparse(
    data: SparseMergedData,
    global_indices: np.ndarray,
    device: torch.device | str = "cpu",
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor | dict[str, torch.Tensor],
]:
    """Build one dense batch from sparse storage (same as training).

    Returns:
        (X_batch, has_data_batch, y_batch); y_batch is a dict per label
        source when the data has several, else the primary labels tensor.
    """
    from dorothy.training.batches import SparseBatchSource

    if data.labels_dict is not None and data.label_sources is not None:
        labels = {s: data.labels_dict[s] for s in data.label_sources}
    else:
        labels = {SINGLE_SURVEY_KEY: data.labels}
    batch = SparseBatchSource(
        data, global_indices, labels, resolve_device(device)
    ).build_batch(np.asarray(global_indices))
    y = batch.y if data.labels_dict is not None else batch.y[SINGLE_SURVEY_KEY]
    return batch.X, batch.has_data, y


def normalize_sparse_data(
    data: SparseMergedData,
    normalizer: LabelNormalizer,
) -> SparseMergedData:
    """Copy of sparse data with labels normalized as in training."""
    from dataclasses import replace

    labels_dict = (
        {s: normalizer.transform_labels(v) for s, v in data.labels_dict.items()}
        if data.labels_dict is not None
        else None
    )
    return replace(
        data, labels=normalizer.transform_labels(data.labels), labels_dict=labels_dict
    )


def normalize_dense_data(
    X_dict: dict[str, np.ndarray],
    y_dict: dict[str, np.ndarray],
    has_data_dict: dict[str, np.ndarray],
    normalizer: LabelNormalizer,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Normalize dense dict labels as in training (X and has_data unchanged)."""
    return (
        X_dict,
        {s: normalizer.transform_labels(y) for s, y in y_dict.items()},
        has_data_dict,
    )


def normalize_single_survey_data(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: LabelNormalizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize single-survey labels as in training (X unchanged)."""
    return X, normalizer.transform_labels(y)
