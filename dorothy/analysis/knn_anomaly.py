"""
k-NN based anomaly detection for stellar spectra.

This module provides anomaly detection by comparing embedding-space distances
to a threshold computed from the training distribution. Stars with k-NN distances
exceeding the threshold are flagged as potential anomalies.

The approach:
1. Extract penultimate-layer embeddings from the trained model
2. L2-normalize embeddings to unit vectors
3. Build a BallTree for efficient nearest neighbor queries
4. Compute k-th nearest neighbor distances
5. Flag stars exceeding the percentile threshold as anomalies

Example:
    >>> detector = AnomalyDetector.from_predictor(predictor)
    >>> detector.fit(X_train, k=10, percentile=99)
    >>> result = detector.detect(X_test)
    >>> anomaly_indices = np.where(result.is_anomaly)[0]
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sklearn.neighbors import BallTree

    from dorothy.inference.predictor import Predictor


def l2_normalize(X: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Normalize each row of X to unit L2 norm.

    Args:
        X: Array of shape (n_samples, n_features).

    Returns:
        Normalized array with each row having unit L2 norm.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Add small epsilon to avoid division by zero
    return X / (norms + 1e-10)


@dataclass
class AnomalyResult:
    """
    Container for anomaly detection results.

    Attributes:
        distances: k-th nearest neighbor distances for each sample.
        is_anomaly: Boolean mask indicating anomalous samples.
        threshold: Distance threshold used for detection.
        k: Number of neighbors used.
        percentile: Percentile used to compute threshold.
    """

    distances: NDArray[np.float32]
    is_anomaly: NDArray[np.bool_]
    threshold: float
    k: int
    percentile: float

    @property
    def n_samples(self) -> int:
        """Number of samples evaluated."""
        return len(self.distances)

    @property
    def n_anomalies(self) -> int:
        """Number of detected anomalies."""
        return int(self.is_anomaly.sum())

    @property
    def anomaly_fraction(self) -> float:
        """Fraction of samples flagged as anomalies."""
        return self.n_anomalies / self.n_samples if self.n_samples > 0 else 0.0

    def get_anomaly_indices(self) -> NDArray[np.int64]:
        """Get indices of anomalous samples."""
        return np.where(self.is_anomaly)[0]


class AnomalyDetector:
    """
    k-NN based anomaly detector for stellar spectra.

    Uses embedding-space distances to identify outliers that differ
    significantly from the training distribution.

    Attributes:
        predictor: The DOROTHY predictor for extracting embeddings.
        tree: BallTree for efficient nearest neighbor queries.
        threshold: Distance threshold for anomaly detection.
        k: Number of neighbors to consider.
        percentile: Percentile used to compute threshold.
        train_embeddings: Normalized training embeddings.

    Example:
        >>> detector = AnomalyDetector.from_predictor(predictor)
        >>> detector.fit(X_train, k=10, percentile=99)
        >>> result = detector.detect(X_test)
    """

    def __init__(
        self,
        predictor: Predictor,
        layer_index: int = -2,
    ) -> None:
        """
        Initialize the anomaly detector.

        Args:
            predictor: Trained DOROTHY predictor for embedding extraction.
            layer_index: Which layer to extract embeddings from (-2 = penultimate).
        """
        self.predictor = predictor
        self.layer_index = layer_index

        # State set during fit()
        self.tree: BallTree | None = None
        self.threshold: float | None = None
        self.k: int | None = None
        self.percentile: float | None = None
        self.train_embeddings: NDArray[np.float32] | None = None

    @classmethod
    def from_predictor(
        cls,
        predictor: Predictor,
        layer_index: int = -2,
    ) -> AnomalyDetector:
        """
        Create an anomaly detector from a predictor.

        Args:
            predictor: Trained DOROTHY predictor.
            layer_index: Layer index for embedding extraction.

        Returns:
            Configured AnomalyDetector instance.
        """
        return cls(predictor=predictor, layer_index=layer_index)

    def fit(
        self,
        X_train: NDArray[np.float32],
        k: int = 10,
        percentile: float = 99.0,
        batch_size: int = 1024,
    ) -> AnomalyDetector:
        """
        Fit the anomaly detector on training data.

        Computes embeddings, builds the BallTree, and determines the
        distance threshold from the training distribution.

        Args:
            X_train: Training spectra of shape (n_samples, n_features).
            k: Number of nearest neighbors to consider.
            percentile: Percentile of training distances for threshold.
            batch_size: Batch size for embedding computation.

        Returns:
            Self for method chaining.
        """
        from sklearn.neighbors import BallTree

        self.k = k
        self.percentile = percentile

        # Extract and normalize embeddings
        embeddings = self.predictor.get_embeddings(
            X_train,
            layer_index=self.layer_index,
            batch_size=batch_size,
        )
        self.train_embeddings = l2_normalize(embeddings)

        # Build BallTree for efficient queries
        self.tree = BallTree(self.train_embeddings, metric="euclidean")

        # Query k+1 neighbors (includes self) for training data
        distances, _ = self.tree.query(self.train_embeddings, k=k + 1)
        kth_distances = distances[:, -1]  # k-th neighbor (excluding self)

        # Compute threshold
        self.threshold = float(np.percentile(kth_distances, percentile))

        return self

    def detect(
        self,
        X: NDArray[np.float32],
        batch_size: int = 1024,
    ) -> AnomalyResult:
        """
        Detect anomalies in new data.

        Args:
            X: Input spectra of shape (n_samples, n_features).
            batch_size: Batch size for embedding computation.

        Returns:
            AnomalyResult containing distances and anomaly flags.

        Raises:
            RuntimeError: If detector has not been fitted.
        """
        if self.tree is None or self.threshold is None:
            raise RuntimeError(
                "Detector must be fitted before detection. Call fit() first."
            )

        # Extract and normalize embeddings
        embeddings = self.predictor.get_embeddings(
            X,
            layer_index=self.layer_index,
            batch_size=batch_size,
        )
        embeddings_norm = l2_normalize(embeddings)

        # Query k neighbors (no self-distance for new data)
        distances, _ = self.tree.query(embeddings_norm, k=self.k)
        kth_distances = distances[:, -1].astype(np.float32)

        # Flag anomalies
        is_anomaly = kth_distances > self.threshold

        return AnomalyResult(
            distances=kth_distances,
            is_anomaly=is_anomaly,
            threshold=self.threshold,
            k=self.k,
            percentile=self.percentile,
        )

    def get_training_distances(self) -> NDArray[np.float32]:
        """
        Get k-th neighbor distances for training data.

        Useful for comparing training and test distributions.

        Returns:
            Array of k-th neighbor distances for training samples.

        Raises:
            RuntimeError: If detector has not been fitted.
        """
        if self.tree is None or self.train_embeddings is None:
            raise RuntimeError("Detector must be fitted first. Call fit().")

        distances, _ = self.tree.query(self.train_embeddings, k=self.k + 1)
        return distances[:, -1].astype(np.float32)

    def save(self, path: str | Path) -> None:
        """
        Save the fitted detector to disk.

        Args:
            path: Path to save the detector.

        Raises:
            RuntimeError: If detector has not been fitted.
        """
        if self.tree is None:
            raise RuntimeError("Cannot save unfitted detector. Call fit() first.")

        path = Path(path)
        state = {
            "train_embeddings": self.train_embeddings,
            "threshold": self.threshold,
            "k": self.k,
            "percentile": self.percentile,
            "layer_index": self.layer_index,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, path: str | Path) -> AnomalyDetector:
        """
        Load a previously saved detector state.

        Note: The predictor must be provided separately during __init__
        as models cannot be pickled safely.

        Args:
            path: Path to the saved state.

        Returns:
            Self for method chaining.
        """
        from sklearn.neighbors import BallTree

        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.train_embeddings = state["train_embeddings"]
        self.threshold = state["threshold"]
        self.k = state["k"]
        self.percentile = state["percentile"]
        self.layer_index = state["layer_index"]

        # Rebuild BallTree
        self.tree = BallTree(self.train_embeddings, metric="euclidean")

        return self
