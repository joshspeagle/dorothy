"""
Stellar parameter prediction from trained DOROTHY models.

This module provides inference capabilities for trained models, including:
- Loading trained model checkpoints
- Chunk-based prediction for memory efficiency with large datasets
- Denormalization of predictions to physical units
- Uncertainty extraction from heteroscedastic outputs

Example:
    >>> predictor = Predictor.load("path/to/checkpoint")
    >>> preds, uncerts = predictor.predict(spectra)
    >>> preds_physical = predictor.denormalize(preds, uncerts)
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
from dorothy.models import MLP

if TYPE_CHECKING:
    from dorothy.config.schema import ExperimentConfig


@dataclass
class PredictionResult:
    """
    Container for prediction results.

    Attributes:
        predictions: Predicted stellar parameters of shape (n_samples, n_params).
        uncertainties: Predicted uncertainties of shape (n_samples, n_params).
        raw_output: Raw model output of shape (n_samples, 2 * n_params).
        parameter_names: List of parameter names in order.
        is_normalized: Whether predictions are in normalized space.
    """

    predictions: NDArray[np.float32]
    uncertainties: NDArray[np.float32]
    raw_output: NDArray[np.float32]
    parameter_names: list[str]
    is_normalized: bool = True

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

    This class provides inference capabilities for trained DOROTHY models,
    including loading checkpoints, making predictions, and denormalizing
    results to physical units.

    Attributes:
        model: The trained neural network model.
        normalizer: Label normalizer for denormalization (optional).
        device: Device to run inference on.
        parameter_names: Names of stellar parameters being predicted.
        scatter_floor: Minimum scatter floor for uncertainty extraction.

    Example:
        >>> predictor = Predictor.load("checkpoint/")
        >>> result = predictor.predict(X)
        >>> print(result.predictions.shape)  # (n_samples, 11)
    """

    def __init__(
        self,
        model: MLP,
        normalizer: LabelNormalizer | None = None,
        device: str | torch.device = "cpu",
        parameter_names: list[str] | None = None,
        scatter_floor: float = 0.01,
    ) -> None:
        """
        Initialize the predictor.

        Args:
            model: Trained MLP model.
            normalizer: Label normalizer for denormalization.
            device: Device for inference ("cpu", "cuda", or torch.device).
            parameter_names: Names of parameters in prediction order.
            scatter_floor: Minimum scatter floor (s_0) for uncertainty.
        """
        self.model = model
        self.normalizer = normalizer
        self.scatter_floor = scatter_floor

        # Set device
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        # Set parameter names
        if parameter_names is None:
            from dorothy.config.schema import STELLAR_PARAMETERS

            n_params = model.output_features // 2
            parameter_names = list(STELLAR_PARAMETERS[:n_params])
        self.parameter_names = parameter_names

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        model_file: str = "best_model.pth",
        normalizer_file: str | None = "normalizer.pkl",
        device: str = "auto",
    ) -> Predictor:
        """
        Load a predictor from a checkpoint directory.

        Args:
            checkpoint_path: Path to checkpoint directory.
            model_file: Name of model weights file.
            normalizer_file: Name of normalizer file (None to skip).
            device: Device for inference ("auto", "cpu", or "cuda").

        Returns:
            Configured Predictor instance.

        Raises:
            FileNotFoundError: If required files are not found.
        """
        checkpoint_path = Path(checkpoint_path)

        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model weights
        model_path = checkpoint_path / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load state dict to determine architecture
        state_dict = torch.load(model_path, map_location=device, weights_only=True)

        # Infer architecture from state dict
        input_features, output_features, hidden_layers = cls._infer_architecture(
            state_dict
        )

        # Create model
        model = MLP(
            input_features=input_features,
            output_features=output_features,
            hidden_layers=hidden_layers,
        )
        model.load_state_dict(state_dict)

        # Load normalizer if available
        normalizer = None
        if normalizer_file is not None:
            normalizer_path = checkpoint_path / normalizer_file
            if normalizer_path.exists():
                normalizer = LabelNormalizer.load(normalizer_path)

        return cls(model=model, normalizer=normalizer, device=device)

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

        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        return cls.load(checkpoint_path, device=device)

    @staticmethod
    def _infer_architecture(
        state_dict: dict,
    ) -> tuple[int, int, list[int]]:
        """
        Infer model architecture from state dict keys.

        Returns:
            Tuple of (input_features, output_features, hidden_layers).
        """
        # Find all Linear layer weights (2D tensors only, not BatchNorm 1D weights)
        linear_weights = []
        for k, v in state_dict.items():
            if "weight" in k and v.dim() == 2:
                linear_weights.append(k)
        linear_weights.sort()

        # First linear layer gives input features
        first_weight = state_dict[linear_weights[0]]
        input_features = first_weight.shape[1]

        # Last linear layer gives output features
        last_weight = state_dict[linear_weights[-1]]
        output_features = last_weight.shape[0]

        # Middle layers give hidden sizes
        hidden_layers = []
        for w_key in linear_weights[:-1]:
            weight = state_dict[w_key]
            hidden_layers.append(weight.shape[0])

        return input_features, output_features, hidden_layers

    def predict(
        self,
        X: NDArray[np.float32] | torch.Tensor,
        batch_size: int = 1024,
        denormalize: bool = False,
    ) -> PredictionResult:
        """
        Make predictions on input spectra.

        Args:
            X: Input spectra of shape (n_samples, n_features) or
               (n_samples, 2, n_wavelengths).
            batch_size: Batch size for inference.
            denormalize: Whether to denormalize predictions.

        Returns:
            PredictionResult containing predictions and uncertainties.
        """
        # Convert to tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        X = X.to(self.device)
        n_samples = X.shape[0]
        n_params = self.model.output_features // 2

        # Preallocate output arrays
        predictions = np.zeros((n_samples, n_params), dtype=np.float32)
        uncertainties = np.zeros((n_samples, n_params), dtype=np.float32)
        raw_output = np.zeros((n_samples, 2 * n_params), dtype=np.float32)

        # Process in batches
        self.model.eval()
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X[start:end]

                output = self.model(X_batch)
                output_np = output.cpu().numpy()

                raw_output[start:end] = output_np
                predictions[start:end] = output_np[:, :n_params]

                # Extract uncertainties from log-scatter
                ln_s = output_np[:, n_params:]
                s = np.sqrt(np.exp(2 * ln_s) + self.scatter_floor**2)
                uncertainties[start:end] = s

        # Denormalize if requested and normalizer available
        is_normalized = True
        if denormalize and self.normalizer is not None:
            predictions, uncertainties = self.normalizer.inverse_transform(
                predictions, uncertainties
            )
            is_normalized = False

        return PredictionResult(
            predictions=predictions,
            uncertainties=uncertainties,
            raw_output=raw_output,
            parameter_names=self.parameter_names,
            is_normalized=is_normalized,
        )

    def predict_chunked(
        self,
        X_iterator: Iterator[NDArray[np.float32]],
        denormalize: bool = False,
    ) -> Iterator[PredictionResult]:
        """
        Make predictions on data provided by an iterator.

        This is memory-efficient for large datasets that don't fit in memory.

        Args:
            X_iterator: Iterator yielding batches of input spectra.
            denormalize: Whether to denormalize predictions.

        Yields:
            PredictionResult for each batch.
        """
        for X_batch in X_iterator:
            yield self.predict(X_batch, denormalize=denormalize)

    def predict_all_chunked(
        self,
        X_iterator: Iterator[NDArray[np.float32]],
        denormalize: bool = False,
    ) -> PredictionResult:
        """
        Make predictions on all data from iterator and concatenate.

        Args:
            X_iterator: Iterator yielding batches of input spectra.
            denormalize: Whether to denormalize predictions.

        Returns:
            Combined PredictionResult for all batches.
        """
        all_predictions = []
        all_uncertainties = []
        all_raw_output = []

        for result in self.predict_chunked(X_iterator, denormalize=denormalize):
            all_predictions.append(result.predictions)
            all_uncertainties.append(result.uncertainties)
            all_raw_output.append(result.raw_output)

        return PredictionResult(
            predictions=np.vstack(all_predictions),
            uncertainties=np.vstack(all_uncertainties),
            raw_output=np.vstack(all_raw_output),
            parameter_names=self.parameter_names,
            is_normalized=not denormalize,
        )

    def get_embeddings(
        self,
        X: NDArray[np.float32] | torch.Tensor,
        layer_index: int = -2,
        batch_size: int = 1024,
    ) -> NDArray[np.float32]:
        """
        Extract intermediate embeddings from the model.

        Useful for k-NN anomaly detection and dimensionality reduction.

        Args:
            X: Input spectra.
            layer_index: Which layer to extract embeddings from.
            batch_size: Batch size for processing.

        Returns:
            Embeddings array of shape (n_samples, embedding_dim).
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        X = X.to(self.device)
        n_samples = X.shape[0]

        # Get embedding dimension from a single sample
        with torch.no_grad():
            sample_embed = self.model.get_embeddings(X[:1], layer_index)
            embed_dim = sample_embed.shape[1]

        embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X[start:end]

                embed = self.model.get_embeddings(X_batch, layer_index)
                embeddings[start:end] = embed.cpu().numpy()

        return embeddings


def predict_from_fits(
    fits_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    chunk_size: int = 1000,
    device: str = "auto",
) -> PredictionResult | None:
    """
    Convenience function to predict stellar parameters from a FITS file.

    Args:
        fits_path: Path to input FITS file.
        checkpoint_path: Path to model checkpoint directory.
        output_path: Path to save predictions (CSV). If None, returns results.
        chunk_size: Number of spectra to process at a time.
        device: Device for inference.

    Returns:
        PredictionResult if output_path is None, otherwise None.
    """
    from dorothy.data import FITSLoader

    # Load predictor
    predictor = Predictor.load(checkpoint_path, device=device)

    # Load data
    loader = FITSLoader(fits_path)
    data = loader.load()

    # Get model input
    X = data.get_model_input(apply_quality_mask=True)

    # Predict with chunking for memory efficiency
    def chunk_iterator():
        for start in range(0, X.shape[0], chunk_size):
            end = min(start + chunk_size, X.shape[0])
            yield X[start:end]

    result = predictor.predict_all_chunked(
        chunk_iterator(),
        denormalize=predictor.normalizer is not None,
    )

    # Save if output path provided
    if output_path is not None:
        import pandas as pd

        df = pd.DataFrame(result.to_dict())

        # Add star IDs
        good_ids = data.ids[data.quality_mask]
        df.insert(0, "star_id", good_ids)

        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
        return None

    return result
