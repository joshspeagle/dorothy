"""
Shared evaluation utilities for consistent metrics between training and evaluation.

This module provides functions that replicate the exact evaluation pipeline used
during training validation, ensuring that the evaluate CLI command produces
identical metrics to those logged during training.

The key functions are:
- build_batch_from_sparse: Constructs dense batches from sparse storage
- normalize_sparse_data: Applies label normalization to sparse data
- evaluate_on_test_set: Main evaluation function matching trainer validation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from dorothy.data.catalogue_loader import SparseMergedData
from dorothy.inference.evaluator import EvaluationResult, Evaluator


if TYPE_CHECKING:
    from dorothy.config.schema import ExperimentConfig
    from dorothy.data.normalizer import LabelNormalizer


def build_batch_from_sparse(
    data: SparseMergedData,
    global_indices: np.ndarray,
    device: torch.device | str = "cpu",
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor | dict[str, torch.Tensor],
]:
    """
    Build dense batch tensors from sparse survey data.

    This function mirrors trainer._build_batch_from_sparse without requiring
    a Trainer instance. It uses the CPU-based approach (no pinned memory buffers).

    Args:
        data: SparseMergedData with sparse survey arrays.
        global_indices: Which stars to include in batch (global indices).
        device: Device to place tensors on.

    Returns:
        X_batch: Dict[survey -> (batch_size, 3, n_wave)] dense tensor on device.
        has_data_batch: Dict[survey -> (batch_size,)] bool tensor on device.
        y_batch: Either (batch_size, 3, n_params) labels tensor for single-label,
            or Dict[source -> (batch_size, 3, n_params)] for multi-label.
    """
    if isinstance(device, str):
        device = torch.device(device)

    batch_size = len(global_indices)
    X_batch = {}
    has_data_batch = {}

    for survey in data.surveys:
        n_wave = data.wavelengths[survey].shape[0]

        # Map global indices to local (survey-specific) indices
        local_idx = data.global_to_local[survey][global_indices]
        has_data = local_idx >= 0  # (batch_size,) bool

        # CPU-based batch construction
        flux_batch = np.zeros((batch_size, n_wave), dtype=np.float32)
        ivar_batch = np.zeros((batch_size, n_wave), dtype=np.float32)

        if has_data.any():
            valid_batch_idx = np.where(has_data)[0]
            valid_local = local_idx[has_data]
            flux_batch[valid_batch_idx] = data.flux[survey][valid_local]
            ivar_batch[valid_batch_idx] = data.ivar[survey][valid_local]

        # Convert ivar to sigma
        sigma_batch = np.zeros_like(ivar_batch)
        valid_ivar = ivar_batch > 0
        sigma_batch[valid_ivar] = 1.0 / np.sqrt(ivar_batch[valid_ivar])

        # Build 3-channel format: [flux, sigma, mask]
        mask = valid_ivar.astype(np.float32)
        X_survey = np.stack([flux_batch, sigma_batch, mask], axis=1)

        X_batch[survey] = torch.from_numpy(X_survey).to(device)
        has_data_batch[survey] = torch.from_numpy(has_data).to(device)

    # Labels - handle both single-label and multi-label modes
    if data.labels_dict is not None and data.label_sources is not None:
        # Multi-label mode: return per-source labels dict
        y_batch = {}
        for source in data.label_sources:
            source_labels = data.labels_dict[source][global_indices]
            y_batch[source] = torch.from_numpy(source_labels).to(device)
    else:
        # Single-label mode: return single tensor
        y_batch = torch.from_numpy(data.labels[global_indices]).to(device)

    return X_batch, has_data_batch, y_batch


def normalize_sparse_data(
    data: SparseMergedData,
    normalizer: LabelNormalizer,
) -> SparseMergedData:
    """
    Create a normalized copy of SparseMergedData.

    Applies normalizer.transform() to all labels and errors, creating a new
    SparseMergedData with normalized values matching training data preparation.

    This mirrors the normalization done in trainer.fit_multi_survey_sparse
    (lines 1835-1852) where ALL labels are normalized at once.

    Args:
        data: SparseMergedData with labels in physical units.
        normalizer: Fitted LabelNormalizer from training.

    Returns:
        New SparseMergedData with normalized labels.
    """
    # Make copies of label arrays to avoid modifying original
    y_labels = data.labels.copy()

    # Normalize primary labels
    all_labels = y_labels[:, 0, :]
    all_errors = y_labels[:, 1, :]
    all_labels_norm, all_errors_norm = normalizer.transform(all_labels, all_errors)
    y_labels[:, 0, :] = all_labels_norm
    y_labels[:, 1, :] = all_errors_norm

    # Also normalize labels_dict if multi-label mode
    y_labels_dict = None
    if data.labels_dict is not None:
        y_labels_dict = {}
        for source in data.labels_dict:
            source_labels = data.labels_dict[source].copy()
            labels_2d = source_labels[:, 0, :]
            errors_2d = source_labels[:, 1, :]
            norm_labels, norm_errors = normalizer.transform(labels_2d, errors_2d)
            source_labels[:, 0, :] = norm_labels
            source_labels[:, 1, :] = norm_errors
            y_labels_dict[source] = source_labels

    # Create new SparseMergedData with normalized labels
    return SparseMergedData(
        flux=data.flux,
        ivar=data.ivar,
        wavelengths=data.wavelengths,
        snr=data.snr,
        global_to_local=data.global_to_local,
        local_to_global=data.local_to_global,
        labels=y_labels,
        gaia_ids=data.gaia_ids,
        ra=data.ra,
        dec=data.dec,
        surveys=data.surveys,
        n_total=data.n_total,
        n_params=data.n_params,
        labels_dict=y_labels_dict,
        has_labels_dict=data.has_labels_dict,
        label_sources=data.label_sources,
    )


def normalize_dense_data(
    X_dict: dict[str, np.ndarray],
    y_dict: dict[str, np.ndarray],
    has_data_dict: dict[str, np.ndarray],
    normalizer: LabelNormalizer,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Normalize labels in dense dict format (matching training).

    This applies normalizer.transform() to all label arrays, ensuring evaluation
    uses the same normalization approach as training.

    Args:
        X_dict: Dict[survey -> (N, 3, n_wave)] feature arrays. Passed through unchanged.
        y_dict: Dict[source -> (N, 3, n_params)] label arrays in physical units.
        has_data_dict: Dict[survey -> (N,)] boolean masks. Passed through unchanged.
        normalizer: Fitted LabelNormalizer from training.

    Returns:
        Tuple of (X_dict, y_dict_normalized, has_data_dict).
        X_dict and has_data_dict are the same objects (unchanged).
        y_dict_normalized has labels and errors transformed to normalized space.
    """
    y_dict_normalized = {}
    for source, y in y_dict.items():
        y_copy = y.copy()
        labels = y_copy[:, 0, :]
        errors = y_copy[:, 1, :]
        labels_norm, errors_norm = normalizer.transform(labels, errors)
        y_copy[:, 0, :] = labels_norm
        y_copy[:, 1, :] = errors_norm
        y_dict_normalized[source] = y_copy
    return X_dict, y_dict_normalized, has_data_dict


def normalize_single_survey_data(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: LabelNormalizer,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize labels for single-survey format (matching training).

    This applies normalizer.transform() to the label array, ensuring evaluation
    uses the same normalization approach as training.

    Args:
        X: (N, 3, n_wave) or (N, n_features) feature array. Passed through unchanged.
        y: (N, 3, n_params) label array in physical units.
        normalizer: Fitted LabelNormalizer from training.

    Returns:
        Tuple of (X, y_normalized).
        X is the same object (unchanged).
        y_normalized has labels and errors transformed to normalized space.
    """
    y_copy = y.copy()
    labels = y_copy[:, 0, :]
    errors = y_copy[:, 1, :]
    labels_norm, errors_norm = normalizer.transform(labels, errors)
    y_copy[:, 0, :] = labels_norm
    y_copy[:, 1, :] = errors_norm
    return X, y_copy


def evaluate_on_test_set(
    model: nn.Module,
    data: SparseMergedData,
    test_indices: np.ndarray,
    normalizer: LabelNormalizer | None,
    config: ExperimentConfig,
    batch_size: int = 1024,
    device: str | torch.device = "auto",
) -> tuple[EvaluationResult, EvaluationResult | None]:
    """
    Evaluate model on test set using the same logic as training validation.

    This function replicates the exact evaluation pipeline from
    trainer._validate_detailed_multi_survey_sparse to ensure consistent metrics.

    Args:
        model: Trained model (MultiHeadMLP or MLP).
        data: SparseMergedData with labels in PHYSICAL units (will be normalized).
        test_indices: Global indices of test samples in data.
        normalizer: Fitted LabelNormalizer from training (can be None).
        config: ExperimentConfig loaded from checkpoint.
        batch_size: Number of samples per batch for streaming.
        device: Device for inference ("auto", "cpu", or "cuda").

    Returns:
        Tuple of (normalized_space_result, physical_space_result).
        normalized_space_result: EvaluationResult in normalized space (matches training).
        physical_space_result: EvaluationResult in physical space (for interpretability),
            or None if normalizer is not available.
    """
    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    # Normalize data (like training does at fit_multi_survey_sparse lines 1835-1852)
    data_normalized = (
        normalize_sparse_data(data, normalizer) if normalizer is not None else data
    )

    # Set model to eval mode
    model.eval()

    # Determine multi-label mode
    label_sources = getattr(model, "label_sources", ["default"])
    is_multi_label = len(label_sources) > 1
    primary_label_source = label_sources[0] if is_multi_label else None

    # Collect all predictions and targets
    all_outputs = []
    all_targets = []

    # Stream test data in batches
    with torch.no_grad():
        for start in range(0, len(test_indices), batch_size):
            end = min(start + batch_size, len(test_indices))
            global_indices = test_indices[start:end]

            X_batch, has_data_batch, y_batch = build_batch_from_sparse(
                data_normalized, global_indices, device
            )

            # Forward pass
            output = model(X_batch, has_data=has_data_batch)

            all_outputs.append(output.cpu())
            # Handle multi-label mode where y_batch is a dict
            if is_multi_label and isinstance(y_batch, dict):
                all_targets.append(y_batch[primary_label_source].cpu())
            else:
                all_targets.append(y_batch.cpu())

    # Concatenate all batches
    outputs = torch.cat(all_outputs)
    targets = torch.cat(all_targets)

    # Extract predictions and labels (all in normalized space)
    y_true_norm = targets[:, 0, :].numpy()
    y_err_norm = targets[:, 1, :].numpy()
    y_mask = targets[:, 2, :].numpy()
    y_pred_norm = outputs[:, 0, :].numpy()
    log_scatter = outputs[:, 1, :].numpy()

    # Compute predicted scatter (same as trainer line 2276-2278)
    scatter_floor = config.training.scatter_floor
    pred_scatter_norm = np.sqrt(np.exp(2 * log_scatter) + scatter_floor**2)

    # Get parameter names
    from dorothy.config.schema import STELLAR_PARAMETERS

    n_params = y_pred_norm.shape[1]
    if normalizer is not None:
        parameter_names = normalizer.parameters[:n_params]
    else:
        parameter_names = list(STELLAR_PARAMETERS[:n_params])

    # Create evaluator with same settings as training (teff_in_log=False, scatter_floor)
    evaluator = Evaluator(
        parameter_names=parameter_names,
        teff_in_log=False,  # Training uses False
        scatter_floor=scatter_floor,
    )

    # Evaluate in normalized space (matches training validation)
    result_normalized = evaluator.evaluate(
        y_pred=y_pred_norm,
        y_true=y_true_norm,
        pred_scatter=pred_scatter_norm,
        label_errors=y_err_norm,
        mask=y_mask,
    )

    # Also evaluate in physical space for interpretability
    result_physical = None
    if normalizer is not None:
        # Denormalize predictions and scatter
        y_pred_phys, pred_scatter_phys = normalizer.inverse_transform(
            y_pred_norm, pred_scatter_norm
        )

        # Get physical labels from original (non-normalized) data
        if is_multi_label and data.labels_dict is not None:
            y_test_phys = data.labels_dict[primary_label_source][test_indices]
        else:
            y_test_phys = data.labels[test_indices]

        y_true_phys = y_test_phys[:, 0, :]
        y_err_phys = y_test_phys[:, 1, :]

        result_physical = evaluator.evaluate(
            y_pred=y_pred_phys,
            y_true=y_true_phys,
            pred_scatter=pred_scatter_phys,
            label_errors=y_err_phys,
            mask=y_mask,
        )

    return result_normalized, result_physical


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
    """
    Evaluate model on test set using dense dict format.

    This function replicates the evaluation pipeline with early normalization
    matching training behavior. Used for multi-survey configurations with
    dict-format data (X_dict, y_dict, has_data_dict).

    Args:
        model: Trained MultiHeadMLP model.
        X_dict: Dict[survey -> (N, 3, n_wave)] feature arrays.
        y_dict: Dict[source -> (N, 3, n_params)] label arrays in PHYSICAL units.
        has_data_dict: Dict[survey -> (N,)] boolean masks.
        test_indices: Indices of test samples.
        normalizer: Fitted LabelNormalizer from training (can be None).
        config: ExperimentConfig loaded from checkpoint.
        batch_size: Number of samples per batch for streaming.
        device: Device for inference ("auto", "cpu", or "cuda").

    Returns:
        Tuple of (normalized_space_result, physical_space_result).
        normalized_space_result: EvaluationResult in normalized space (matches training).
        physical_space_result: EvaluationResult in physical space (for interpretability),
            or None if normalizer is not available.
    """
    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    # Get scatter floor
    scatter_floor = config.training.scatter_floor

    # Normalize labels early (matching training behavior)
    if normalizer is not None:
        _, y_dict_norm, _ = normalize_dense_data(
            X_dict, y_dict, has_data_dict, normalizer
        )
    else:
        y_dict_norm = y_dict

    # Set model to eval mode
    model.eval()

    # Get primary label source
    label_sources = config.data.label_sources
    first_source = label_sources[0] if label_sources else list(y_dict.keys())[0]

    # Slice data to test indices
    X_test = {survey: arr[test_indices] for survey, arr in X_dict.items()}
    has_data_test = {survey: arr[test_indices] for survey, arr in has_data_dict.items()}
    y_test_norm = y_dict_norm[first_source][test_indices]
    y_test_phys = y_dict[first_source][test_indices]  # Keep physical for later

    # Make predictions in batches
    n_test = len(test_indices)
    all_preds = []
    all_scatters = []

    with torch.no_grad():
        # Pre-convert to tensors
        X_tensors = {
            survey: torch.tensor(X_test[survey], dtype=torch.float32, device=device)
            for survey in X_test
        }
        has_data_tensors = {
            survey: torch.tensor(has_data_test[survey], dtype=torch.bool, device=device)
            for survey in has_data_test
        }

        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            batch_X = {s: X_tensors[s][start:end] for s in X_tensors}
            batch_has_data = {
                s: has_data_tensors[s][start:end] for s in has_data_tensors
            }

            output = model(batch_X, batch_has_data)

            # Handle multi-head output (dict)
            if isinstance(output, dict):
                output = output[first_source]

            pred_mean = output[:, 0, :].cpu().numpy()
            pred_ln_s = output[:, 1, :].cpu().numpy()
            batch_scatter = np.sqrt(np.exp(2 * pred_ln_s) + scatter_floor**2)

            all_preds.append(pred_mean)
            all_scatters.append(batch_scatter)

    predictions = np.vstack(all_preds)
    pred_scatter = np.vstack(all_scatters)

    # Extract normalized labels
    y_true_norm = y_test_norm[:, 0, :]
    y_err_norm = y_test_norm[:, 1, :]
    y_mask = y_test_norm[:, 2, :]

    # Get parameter names
    from dorothy.config.schema import STELLAR_PARAMETERS

    n_params = predictions.shape[1]
    if normalizer is not None:
        parameter_names = normalizer.parameters[:n_params]
    else:
        parameter_names = list(STELLAR_PARAMETERS[:n_params])

    # Create evaluator
    evaluator = Evaluator(
        parameter_names=parameter_names,
        teff_in_log=False,
        scatter_floor=scatter_floor,
    )

    # Evaluate in normalized space
    result_normalized = evaluator.evaluate(
        y_pred=predictions,
        y_true=y_true_norm,
        pred_scatter=pred_scatter,
        label_errors=y_err_norm,
        mask=y_mask,
    )

    # Also evaluate in physical space for interpretability
    result_physical = None
    if normalizer is not None:
        y_pred_phys, pred_scatter_phys = normalizer.inverse_transform(
            predictions, pred_scatter
        )

        y_true_phys = y_test_phys[:, 0, :]
        y_err_phys = y_test_phys[:, 1, :]

        result_physical = evaluator.evaluate(
            y_pred=y_pred_phys,
            y_true=y_true_phys,
            pred_scatter=pred_scatter_phys,
            label_errors=y_err_phys,
            mask=y_mask,
        )

    return result_normalized, result_physical


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
    """
    Evaluate model on test set using single-survey array format.

    This function replicates the evaluation pipeline with early normalization
    matching training behavior. Used for single-survey configurations with
    standard MLP models.

    Args:
        model: Trained MLP model.
        X: (N, 3, n_wave) or (N, n_features) feature array.
        y: (N, 3, n_params) label array in PHYSICAL units.
        test_indices: Indices of test samples.
        normalizer: Fitted LabelNormalizer from training (can be None).
        config: ExperimentConfig loaded from checkpoint.
        batch_size: Number of samples per batch for streaming.
        device: Device for inference ("auto", "cpu", or "cuda").

    Returns:
        Tuple of (normalized_space_result, physical_space_result).
        normalized_space_result: EvaluationResult in normalized space (matches training).
        physical_space_result: EvaluationResult in physical space (for interpretability),
            or None if normalizer is not available.
    """
    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    # Get scatter floor
    scatter_floor = config.training.scatter_floor

    # Normalize labels early (matching training behavior)
    if normalizer is not None:
        _, y_norm = normalize_single_survey_data(X, y, normalizer)
    else:
        y_norm = y

    # Set model to eval mode
    model.eval()

    # Slice data to test indices
    X_test = X[test_indices]
    y_test_norm = y_norm[test_indices]
    y_test_phys = y[test_indices]  # Keep physical for later

    # Flatten 3-channel input if needed (for standard MLP)
    X_flat = X_test.reshape(X_test.shape[0], -1) if X_test.ndim == 3 else X_test

    # Make predictions in batches
    n_test = len(test_indices)
    all_preds = []
    all_scatters = []

    with torch.no_grad():
        X_tensor = torch.tensor(X_flat, dtype=torch.float32, device=device)

        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            batch_X = X_tensor[start:end]

            output = model(batch_X)

            pred_mean = output[:, 0, :].cpu().numpy()
            pred_ln_s = output[:, 1, :].cpu().numpy()
            batch_scatter = np.sqrt(np.exp(2 * pred_ln_s) + scatter_floor**2)

            all_preds.append(pred_mean)
            all_scatters.append(batch_scatter)

    predictions = np.vstack(all_preds)
    pred_scatter = np.vstack(all_scatters)

    # Extract normalized labels
    y_true_norm = y_test_norm[:, 0, :]
    y_err_norm = y_test_norm[:, 1, :]
    y_mask = y_test_norm[:, 2, :]

    # Get parameter names
    from dorothy.config.schema import STELLAR_PARAMETERS

    n_params = predictions.shape[1]
    if normalizer is not None:
        parameter_names = normalizer.parameters[:n_params]
    else:
        parameter_names = list(STELLAR_PARAMETERS[:n_params])

    # Create evaluator
    evaluator = Evaluator(
        parameter_names=parameter_names,
        teff_in_log=False,
        scatter_floor=scatter_floor,
    )

    # Evaluate in normalized space
    result_normalized = evaluator.evaluate(
        y_pred=predictions,
        y_true=y_true_norm,
        pred_scatter=pred_scatter,
        label_errors=y_err_norm,
        mask=y_mask,
    )

    # Also evaluate in physical space for interpretability
    result_physical = None
    if normalizer is not None:
        y_pred_phys, pred_scatter_phys = normalizer.inverse_transform(
            predictions, pred_scatter
        )

        y_true_phys = y_test_phys[:, 0, :]
        y_err_phys = y_test_phys[:, 1, :]

        result_physical = evaluator.evaluate(
            y_pred=y_pred_phys,
            y_true=y_true_phys,
            pred_scatter=pred_scatter_phys,
            label_errors=y_err_phys,
            mask=y_mask,
        )

    return result_normalized, result_physical
