"""k-NN anomaly detection for stellar spectra.

A star is an anomaly when its k-th nearest training neighbour in embedding
space is farther than a percentile of the same distance over the training set.

The steps:
1. Extract embeddings from the trained model (see ``Predictor.get_embeddings``).
2. L2-normalize the embeddings to unit vectors.
3. Find the k nearest training embeddings by brute-force search.
4. Take the Euclidean distance to the k-th neighbour.
5. Flag the stars whose distance exceeds the percentile threshold.

Examples:
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
    from dorothy.inference.predictor import Predictor


def l2_normalize(X: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Normalize each row of X to unit L2 norm.

    Each row is divided by its norm plus 1e-10, so a zero row stays zero.

    Args:
        X: Array of shape (n_samples, n_features).

    Returns:
        Array of shape (n_samples, n_features).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Add small epsilon to avoid division by zero
    return X / (norms + 1e-10)


@dataclass
class AnomalyResult:
    """
    Result of ``AnomalyDetector.detect``.

    Attributes:
        distances: Distance to the k-th nearest training neighbour, shape
            (n_samples,). The embeddings are unit vectors, so it is at most 2.
        is_anomaly: True where ``distances`` exceeds ``threshold``, shape
            (n_samples,).
        threshold: Distance threshold.
        k: Number of neighbours.
        percentile: Percentile of the training distances that gave the
            threshold.
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
        """Indices of the anomalous samples, shape (n_anomalies,)."""
        return np.where(self.is_anomaly)[0]


class AnomalyDetector:
    """
    k-NN anomaly detector for stellar spectra.

    Attributes:
        predictor: The predictor that extracts embeddings.
        layer_index: Layer of the embeddings.
        threshold: Distance threshold; None before ``fit``.
        k: Number of neighbours; None before ``fit``.
        percentile: Percentile of the training distances that gave the
            threshold; None before ``fit``.
        train_embeddings: Unit-norm training embeddings, shape
            (n_train, embedding_dim); None before ``fit``.

    Examples:
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
            predictor: Trained predictor that extracts the embeddings.
            layer_index: Layer of the embeddings, as in
                ``Predictor.get_embeddings`` (-2 is the penultimate MLP layer).
        """
        self.predictor = predictor
        self.layer_index = layer_index

        # State set during fit()
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
            predictor: Trained predictor that extracts the embeddings.
            layer_index: Layer of the embeddings, as in ``__init__``.

        Returns:
            An unfitted detector.
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
        Fit the detector on training spectra.

        The threshold is the given percentile of the distance of each training
        star to its k-th nearest other training star. The brute-force search
        takes O(n_train^2 * embedding_dim) time.

        Args:
            X_train: Training spectra, shape (n_train, 3, n_wavelengths).
            k: Number of nearest neighbours.
            percentile: Percentile of the training distances, in [0, 100].
            batch_size: Batch size for the embeddings.

        Returns:
            The detector.
        """
        self.k = k
        self.percentile = percentile

        # Extract and normalize embeddings
        embeddings = self.predictor.get_embeddings(
            X_train,
            layer_index=self.layer_index,
            batch_size=batch_size,
        )
        self.train_embeddings = l2_normalize(embeddings)

        # Query k+1 neighbors (includes self) for training data
        kth_distances = self._kth_neighbor_distances(self.train_embeddings, k + 1)

        # Compute threshold
        self.threshold = float(np.percentile(kth_distances, percentile))

        return self

    def detect(
        self,
        X: NDArray[np.float32],
        batch_size: int = 1024,
    ) -> AnomalyResult:
        """
        Flag the anomalies among new spectra.

        The brute-force search takes O(n_samples * n_train * embedding_dim)
        time.

        Args:
            X: Spectra, shape (n_samples, 3, n_wavelengths).
            batch_size: Batch size for the embeddings.

        Returns:
            Distances and anomaly flags.

        Raises:
            RuntimeError: If the detector is not fitted.
        """
        if self.train_embeddings is None or self.threshold is None:
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
        kth_distances = self._kth_neighbor_distances(embeddings_norm, self.k)
        kth_distances = kth_distances.astype(np.float32)

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
        Distance of each training star to its k-th nearest other training star.

        This takes O(n_train^2 * embedding_dim) time.

        Returns:
            Distances, shape (n_train,).

        Raises:
            RuntimeError: If the detector is not fitted.
        """
        if self.train_embeddings is None:
            raise RuntimeError("Detector must be fitted first. Call fit().")

        distances = self._kth_neighbor_distances(self.train_embeddings, self.k + 1)
        return distances.astype(np.float32)

    def _kth_neighbor_distances(
        self, X: NDArray[np.float32], n_neighbors: int
    ) -> NDArray[np.float64]:
        from sklearn.neighbors import NearestNeighbors

        # Brute force: a ball tree cannot prune in ~100-dimensional embedding
        # space, so its queries are far slower than blocked BLAS distances.
        search = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute")
        distances, _ = search.fit(self.train_embeddings).kneighbors(X)
        return distances[:, -1]

    def save(self, path: str | Path) -> None:
        """
        Pickle the fitted state to a file.

        The state is the embeddings, threshold, k, percentile and layer index.
        The predictor is not saved.

        Args:
            path: Output file.

        Raises:
            RuntimeError: If the detector is not fitted.
        """
        if self.train_embeddings is None:
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
        Load a state written by ``save``, replacing the fitted state.

        The predictor comes from ``__init__``. Unpickling runs code, so load
        only files that you trust.

        Args:
            path: File written by ``save``.

        Returns:
            The detector.
        """
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.train_embeddings = state["train_embeddings"]
        self.threshold = state["threshold"]
        self.k = state["k"]
        self.percentile = state["percentile"]
        self.layer_index = state["layer_index"]

        return self
