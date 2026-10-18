"""
Stellar parameter prediction from trained DOROTHY models.

This module provides inference capabilities for trained models, including:
- Loading trained model checkpoints
- Chunk-based prediction for memory efficiency with large datasets
- Denormalization of predictions to physical units
- Calibrated uncertainties from heteroscedastic outputs

Examples:
    >>> predictor = Predictor.load("path/to/checkpoint")
    >>> result = predictor.predict(spectra, denormalize=True)
    >>> print(result.predictions.shape, result.uncertainties.shape)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray

from dorothy.data.normalizer import LabelNormalizer
from dorothy.inference.calibration import calibrate_scatter
from dorothy.inference.checkpoint import (
    head_label_sources,
    infer_mlp_architecture,
    load_checkpoint,
)
from dorothy.losses.heteroscedastic import predicted_scatter
from dorothy.models import MLP, MultiHeadMLP
from dorothy.utils import resolve_device

if TYPE_CHECKING:
    from dorothy.config.schema import ExperimentConfig


@dataclass
class PredictionResult:
    """
    Container for prediction results.

    Attributes:
        predictions: Predicted stellar parameters of shape (n_samples, n_params).
        uncertainties: Predicted uncertainties of shape (n_samples, n_params).
        raw_output: Raw model output of shape (n_samples, 2, n_params) where
            raw_output[:, 0, :] contains predicted means and
            raw_output[:, 1, :] contains predicted log-scatter.
        parameter_names: List of parameter names in order.
        is_normalized: Whether predictions are in normalized space.
        calibrated: Whether the uncertainties are scaled by calibration
            factors.
    """

    predictions: NDArray[np.float32]
    uncertainties: NDArray[np.float32]
    raw_output: NDArray[np.float32]
    parameter_names: list[str]
    is_normalized: bool = True
    calibrated: bool = False

    @property
    def n_samples(self) -> int:
        """Number of predicted samples."""
        return self.predictions.shape[0]

    @property
    def n_parameters(self) -> int:
        """Number of stellar parameters."""
        return self.predictions.shape[1]

    def to_dict(self) -> dict[str, NDArray[np.float32]]:
        """Convert to dictionary mapping parameter names to values."""
        result = {}
        for i, name in enumerate(self.parameter_names):
            result[name] = self.predictions[:, i]
            result[f"{name}_err"] = self.uncertainties[:, i]
        return result


class Predictor:
    """
    Predictor for stellar parameters from spectroscopic data.

    This class provides inference capabilities for trained DOROTHY models
    (single-survey MLP or multi-survey MultiHeadMLP), including loading
    checkpoints, making predictions, and denormalizing results to physical
    units.

    Inputs use the training format: arrays of shape (n_samples, 3,
    n_wavelengths) with channels [flux, sigma, mask]. For a MultiHeadMLP,
    pass either one survey's array (set ``survey``) or a dict mapping survey
    names to arrays (optionally with ``has_data`` masks).

    Attributes:
        model: The trained neural network model.
        normalizer: Label normalizer for denormalization (optional).
        device: Device to run inference on.
        parameter_names: Names of stellar parameters being predicted.
        scatter_floor: Scatter floor used to convert log-scatter to scatter.
        survey: Survey of single-array inputs (MultiHeadMLP only).
        label_source: Output head to predict from (MultiHeadMLP with several
            label sources; None uses the first / primary head).
        calibration: Calibration factor per parameter name of the head
            predicted from, which scales the predicted scatter (see
            :func:`~dorothy.inference.calibration.calibrate_scatter`), or
            None for the raw scatter.
        config: Experiment config of the checkpoint, if loaded from one.

    Examples:
        >>> predictor = Predictor.load("checkpoint/")
        >>> result = predictor.predict(X, denormalize=True)
        >>> print(result.predictions.shape)  # (n_samples, n_params)
    """

    def __init__(
        self,
        model: MLP | MultiHeadMLP,
        normalizer: LabelNormalizer | None = None,
        device: str | torch.device = "cpu",
        parameter_names: list[str] | None = None,
        scatter_floor: float = 0.01,
        survey: str | None = None,
        label_source: str | None = None,
        calibration: dict[str, float] | None = None,
    ) -> None:
        """
        Initialize the predictor.

        Args:
            model: Trained MLP or MultiHeadMLP model.
            normalizer: Label normalizer for denormalization.
            device: Device for inference ("auto", "cpu", "cuda", "mps" or
                torch.device).
            parameter_names: Names of parameters in prediction order.
            scatter_floor: Scatter floor (s_0) the model was trained with.
            survey: For a MultiHeadMLP, the survey of single-array inputs.
            label_source: For a multi-label MultiHeadMLP, which output head
                to use (default: the first label source).
            calibration: Calibration factor per parameter name of that head,
                which scales the predicted scatter; None reports the raw
                scatter.
        """
        self.model = model
        self.normalizer = normalizer
        self.scatter_floor = scatter_floor
        self.device = resolve_device(device)
        self.calibration = calibration
        self.config = None

        if isinstance(model, MultiHeadMLP):
            if survey is not None and survey not in model.survey_names:
                raise ValueError(
                    f"Unknown survey '{survey}'. Model surveys: {model.survey_names}"
                )
            if label_source is not None and label_source not in model.label_sources:
                raise ValueError(
                    f"Unknown label source '{label_source}'. "
                    f"Model label sources: {model.label_sources}"
                )
        elif survey is not None or label_source is not None:
            raise ValueError("survey/label_source only apply to a MultiHeadMLP")
        self.survey = survey
        self.label_source = label_source

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        # Set parameter names
        if parameter_names is None:
            if normalizer is not None:
                parameter_names = list(normalizer.parameters)
            else:
                from dorothy.config.schema import STELLAR_PARAMETERS

                n_params = model.n_parameters
                parameter_names = list(STELLAR_PARAMETERS[:n_params])
        self.parameter_names = parameter_names

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        model_file: str = "best_model.pth",
        normalizer_file: str | None = "normalizer.pkl",
        device: str = "auto",
        survey: str | None = None,
        label_source: str | None = None,
        calibrate: bool = True,
    ) -> Predictor:
        """
        Load a predictor from a checkpoint directory.

        The model is rebuilt from the checkpoint's ``config.yaml`` (MLP or
        MultiHeadMLP with the trained normalization/activation); checkpoints
        without a config fall back to inferring an MLP from the weights.

        Args:
            checkpoint_path: Path to checkpoint directory.
            model_file: Name of model weights file.
            normalizer_file: Name of normalizer file (None to skip).
            device: Device for inference ("auto", "cpu", "cuda", "mps").
            survey: For a MultiHeadMLP, the survey of single-array inputs.
            label_source: For a multi-label model, which output head to use.
            calibrate: Scale the predicted scatter by the checkpoint's
                calibration factors for that head, if it has them.

        Returns:
            Configured Predictor instance.

        Raises:
            FileNotFoundError: If required files are not found.
            ValueError: If a checkpoint without config.yaml holds a model with
                label-aware uncertainty.
        """
        ckpt = load_checkpoint(checkpoint_path, model_file, normalizer_file, device)
        calibration = None
        if calibrate and ckpt.calibration is not None:
            # A single head is filed under the configured label source, while
            # the model may call it "default"
            heads = head_label_sources(ckpt.model, ckpt.config)
            head = label_source if label_source in heads else heads[0]
            calibration = ckpt.calibration.get(head)
        predictor = cls(
            model=ckpt.model,
            normalizer=ckpt.normalizer,
            device=ckpt.device,
            parameter_names=ckpt.parameter_names,
            scatter_floor=ckpt.scatter_floor,
            survey=survey,
            label_source=label_source,
            calibration=calibration,
        )
        predictor.config = ckpt.config
        return predictor

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        checkpoint_path: str | Path | None = None,
    ) -> Predictor:
        """
        Create a predictor from experiment configuration.

        Args:
            config: Experiment configuration.
            checkpoint_path: Path to checkpoint (uses config default if None).

        Returns:
            Configured Predictor instance.
        """
        if checkpoint_path is None:
            checkpoint_path = config.get_checkpoint_path()
        return cls.load(checkpoint_path, device=config.device)

    @staticmethod
    def _infer_architecture(
        state_dict: dict,
    ) -> tuple[int, int, list[int]]:
        """
        Infer MLP architecture from state dict keys.

        Returns:
            Tuple of (input_features, output_features, hidden_layers).
        """
        arch = infer_mlp_architecture(state_dict)
        return arch["input_features"], arch["output_features"], arch["hidden_layers"]

    @property
    def is_multi_head(self) -> bool:
        """Whether the model is a MultiHeadMLP."""
        return isinstance(self.model, MultiHeadMLP)

    def _forward(
        self,
        X: torch.Tensor | dict[str, torch.Tensor],
        has_data: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if not self.is_multi_head:
            return self.model(X)
        if self.label_source is not None:
            return self.model.forward_for_label_source(
                X, self.label_source, survey=self.survey, has_data=has_data
            )
        return self.model(X, survey=self.survey, has_data=has_data)

    def predict(
        self,
        X: NDArray[np.float32] | torch.Tensor | dict[str, NDArray[np.float32]],
        batch_size: int = 1024,
        denormalize: bool = False,
        has_data: dict[str, NDArray[np.bool_]] | None = None,
    ) -> PredictionResult:
        """
        Make predictions on input spectra.

        Args:
            X: Input spectra of shape (n_samples, 3, n_wavelengths), or for a
                MultiHeadMLP a dict mapping survey names to such arrays.
            batch_size: Batch size for inference.
            denormalize: Whether to denormalize predictions.
            has_data: For dict inputs, survey name -> (n_samples,) bool array
                of which stars have that survey (default: all).

        Returns:
            PredictionResult containing predictions and uncertainties: the
            predicted scatter, times the calibration factors if the predictor
            has them.
        """

        def as_tensor(a):
            return a if isinstance(a, torch.Tensor) else torch.as_tensor(a)

        if isinstance(X, dict):
            X = {s: as_tensor(a).float() for s, a in X.items()}
            n_samples = len(next(iter(X.values())))
            if has_data is not None:
                has_data = {s: as_tensor(h).bool() for s, h in has_data.items()}
        else:
            X = as_tensor(X).float()
            n_samples = X.shape[0]

        outputs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                sl = slice(start, start + batch_size)
                if isinstance(X, dict):
                    X_batch = {s: a[sl].to(self.device) for s, a in X.items()}
                    hd = (
                        {s: h[sl].to(self.device) for s, h in has_data.items()}
                        if has_data is not None
                        else None
                    )
                else:
                    X_batch, hd = X[sl].to(self.device), None
                outputs.append(self._forward(X_batch, hd).float().cpu())

        raw = (
            torch.cat(outputs)
            if outputs
            else torch.zeros(0, 2, self.model.n_parameters)
        )
        predictions = raw[:, 0, :].numpy().copy()
        uncertainties = predicted_scatter(raw[:, 1, :], self.scatter_floor).numpy()
        if self.calibration is not None:
            uncertainties = calibrate_scatter(
                uncertainties, self.calibration, self.parameter_names
            )

        # Denormalize if requested and normalizer available
        is_normalized = True
        if denormalize and self.normalizer is not None:
            predictions, uncertainties = self.normalizer.inverse_transform(
                predictions, uncertainties
            )
            is_normalized = False

        return PredictionResult(
            predictions=predictions.astype(np.float32),
            uncertainties=uncertainties.astype(np.float32),
            raw_output=raw.numpy(),
            parameter_names=self.parameter_names,
            is_normalized=is_normalized,
            calibrated=self.calibration is not None,
        )

    def predict_chunked(
        self,
        X_iterator: Iterator[NDArray[np.float32]],
        denormalize: bool = False,
        batch_size: int = 1024,
    ) -> Iterator[PredictionResult]:
        """
        Make predictions on data provided by an iterator.

        This is memory-efficient for large datasets that don't fit in memory.

        Args:
            X_iterator: Iterator yielding batches of input spectra.
            denormalize: Whether to denormalize predictions.
            batch_size: Batch size for inference within each chunk.

        Yields:
            PredictionResult for each batch.
        """
        for X_batch in X_iterator:
            yield self.predict(X_batch, batch_size=batch_size, denormalize=denormalize)

    def predict_all_chunked(
        self,
        X_iterator: Iterator[NDArray[np.float32]],
        denormalize: bool = False,
        batch_size: int = 1024,
    ) -> PredictionResult:
        """
        Make predictions on all data from iterator and concatenate.

        Args:
            X_iterator: Iterator yielding batches of input spectra.
            denormalize: Whether to denormalize predictions.
            batch_size: Batch size for inference within each chunk.

        Returns:
            Combined PredictionResult for all batches.
        """
        results = list(
            self.predict_chunked(
                X_iterator, denormalize=denormalize, batch_size=batch_size
            )
        )
        return PredictionResult(
            predictions=np.vstack([r.predictions for r in results]),
            uncertainties=np.vstack([r.uncertainties for r in results]),
            raw_output=np.vstack([r.raw_output for r in results]),
            parameter_names=self.parameter_names,
            is_normalized=not (denormalize and self.normalizer is not None),
            calibrated=self.calibration is not None,
        )

    def get_embeddings(
        self,
        X: NDArray[np.float32] | torch.Tensor,
        layer_index: int = -2,
        batch_size: int = 1024,
    ) -> NDArray[np.float32]:
        """
        Extract intermediate embeddings from the model.

        Useful for k-NN anomaly detection and dimensionality reduction. For an
        MLP, ``layer_index`` selects the hidden layer. For a MultiHeadMLP, the
        shared trunk features of ``self.survey`` (comparable across surveys)
        are returned and ``layer_index`` is ignored.

        Args:
            X: Input spectra of shape (n_samples, 3, n_wavelengths).
            layer_index: Which MLP layer to extract embeddings from.
            batch_size: Batch size for processing.

        Returns:
            Embeddings array of shape (n_samples, embedding_dim).
        """
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X, dtype=torch.float32)
        survey = self.survey
        if self.is_multi_head and survey is None:
            if len(self.model.survey_names) != 1:
                raise ValueError("Set Predictor.survey to embed MultiHeadMLP inputs")
            survey = self.model.survey_names[0]

        chunks = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                X_batch = X[start : start + batch_size].to(self.device)
                if self.is_multi_head:
                    embed = self.model.get_embeddings(X_batch, survey, level="trunk")
                else:
                    embed = self.model.get_embeddings(X_batch, layer_index)
                chunks.append(embed.float().cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32)


def predict_from_fits(
    fits_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    chunk_size: int = 1000,
    device: str = "auto",
    batch_size: int = 1024,
    denormalize: bool = True,
    survey: str | None = None,
    label_source: str | None = None,
    calibrate: bool = True,
) -> PredictionResult | None:
    """
    Predict stellar parameters for every spectrum in a FITS file.

    Spectra are preprocessed exactly as in training: per-spectrum flux
    normalization when the checkpoint's ``data.normalize_spectra`` is set,
    the 3-channel [flux, sigma, mask] input format, and the precision
    encoding of sigma when ``data.precision_encoding`` is set.

    Args:
        fits_path: Path to input FITS file (SPECTRA / SPEC_IVAR extensions).
        checkpoint_path: Path to model checkpoint directory.
        output_path: Path to save predictions (CSV): the star ID, each
            parameter and its uncertainty, and a ``calibrated`` column that
            says whether the uncertainties are calibrated. If None, returns
            results.
        chunk_size: Number of spectra to process at a time.
        device: Device for inference.
        batch_size: Batch size for inference.
        denormalize: Output physical units (requires a normalizer).
        survey: Survey of the spectra, for multi-survey models.
        label_source: Output head to use, for multi-label models.
        calibrate: Scale the uncertainties by the checkpoint's calibration
            factors, if it has them.

    Returns:
        PredictionResult if output_path is None, otherwise None.

    Raises:
        ValueError: If the spectra don't match the model's wavelength grid.
    """
    from dorothy.data import FITSLoader, encode_precision

    predictor = Predictor.load(
        checkpoint_path,
        device=device,
        survey=survey,
        label_source=label_source,
        calibrate=calibrate,
    )
    normalize, encode = (
        (
            predictor.config.data.normalize_spectra,
            predictor.config.data.precision_encoding,
        )
        if predictor.config is not None
        else (True, False)
    )

    data = FITSLoader(fits_path).load(
        apply_normalization=normalize,
        apply_quality_filter=False,
        compute_spectral_mask=True,
        require_labels=False,
    )
    X = data.get_model_input_3channel(apply_quality_mask=False)
    if encode:
        encode_precision(X, inplace=True)

    model = predictor.model
    if isinstance(model, MultiHeadMLP):
        target = predictor.survey or (
            model.survey_names[0] if len(model.survey_names) == 1 else None
        )
        if target is None:
            raise ValueError(
                f"Model was trained on surveys {model.survey_names}; "
                "specify which survey the spectra are from"
            )
        expected = model.encoders[target].n_wavelengths
    else:
        expected = model.input_features // 3
    if X.shape[2] != expected:
        raise ValueError(
            f"Spectra have {X.shape[2]} wavelength bins but the model expects "
            f"{expected}"
        )

    def chunk_iterator():
        for start in range(0, X.shape[0], chunk_size):
            yield X[start : start + chunk_size]

    result = predictor.predict_all_chunked(
        chunk_iterator(), denormalize=denormalize, batch_size=batch_size
    )

    # Save if output path provided
    if output_path is not None:
        import pandas as pd

        df = pd.DataFrame(result.to_dict())
        df.insert(0, "star_id", data.ids)
        df["calibrated"] = result.calibrated
        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
        return None

    return result
