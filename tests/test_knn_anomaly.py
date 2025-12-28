"""
Tests for the k-NN anomaly detection module.

These tests verify:
1. L2 normalization
2. AnomalyDetector initialization and fitting
3. Anomaly detection on new data
4. AnomalyResult properties
5. Serialization (save/load)
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dorothy.analysis.knn_anomaly import (
    AnomalyDetector,
    AnomalyResult,
    l2_normalize,
)
from dorothy.inference.predictor import Predictor
from dorothy.models import MLP


class TestL2Normalize:
    """Tests for L2 normalization function."""

    def test_normalize_shape_preserved(self):
        """Test that normalization preserves shape."""
        X = np.random.randn(100, 50).astype(np.float32)
        X_norm = l2_normalize(X)
        assert X_norm.shape == X.shape

    def test_normalize_unit_norms(self):
        """Test that normalized rows have unit L2 norm."""
        X = np.random.randn(100, 50).astype(np.float32)
        X_norm = l2_normalize(X)
        norms = np.linalg.norm(X_norm, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_normalize_zero_row(self):
        """Test that zero rows don't cause NaN."""
        X = np.array(
            [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        X_norm = l2_normalize(X)
        assert not np.any(np.isnan(X_norm))

    def test_normalize_single_row(self):
        """Test normalization of single row."""
        X = np.array([[3.0, 4.0]], dtype=np.float32)
        X_norm = l2_normalize(X)
        np.testing.assert_allclose(X_norm, [[0.6, 0.8]], rtol=1e-5)


class TestAnomalyResult:
    """Tests for the AnomalyResult container."""

    def test_properties(self):
        """Test that properties return correct values."""
        distances = np.array([0.1, 0.2, 0.3, 0.5, 0.8], dtype=np.float32)
        is_anomaly = np.array([False, False, False, False, True])

        result = AnomalyResult(
            distances=distances,
            is_anomaly=is_anomaly,
            threshold=0.6,
            k=10,
            percentile=99.0,
        )

        assert result.n_samples == 5
        assert result.n_anomalies == 1
        assert result.anomaly_fraction == 0.2

    def test_get_anomaly_indices(self):
        """Test getting anomaly indices."""
        is_anomaly = np.array([True, False, True, False, True])
        result = AnomalyResult(
            distances=np.zeros(5, dtype=np.float32),
            is_anomaly=is_anomaly,
            threshold=0.5,
            k=10,
            percentile=99.0,
        )

        indices = result.get_anomaly_indices()
        np.testing.assert_array_equal(indices, [0, 2, 4])

    def test_empty_anomalies(self):
        """Test with no anomalies."""
        result = AnomalyResult(
            distances=np.array([0.1, 0.2], dtype=np.float32),
            is_anomaly=np.array([False, False]),
            threshold=0.5,
            k=10,
            percentile=99.0,
        )

        assert result.n_anomalies == 0
        assert result.anomaly_fraction == 0.0
        assert len(result.get_anomaly_indices()) == 0


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    model = MLP(
        input_features=100,
        output_features=6,
        hidden_layers=[32, 16],
    )
    model.eval()
    return model


@pytest.fixture
def simple_predictor(simple_model):
    """Create a simple predictor for testing."""
    return Predictor(
        simple_model,
        device="cpu",
        parameter_names=["teff", "logg", "fe_h"],
    )


@pytest.fixture
def training_data():
    """Create synthetic training data."""
    np.random.seed(42)
    # Create clustered data - most points near origin
    X_normal = np.random.randn(100, 100).astype(np.float32) * 0.5
    return X_normal


@pytest.fixture
def test_data_with_anomalies():
    """Create test data with some anomalies."""
    np.random.seed(123)
    # Normal points
    X_normal = np.random.randn(80, 100).astype(np.float32) * 0.5
    # Anomalous points (far from training distribution)
    X_anomaly = np.random.randn(20, 100).astype(np.float32) * 5.0 + 10.0
    return np.vstack([X_normal, X_anomaly])


class TestAnomalyDetectorInit:
    """Tests for AnomalyDetector initialization."""

    def test_detector_creation(self, simple_predictor):
        """Test that detector can be created."""
        detector = AnomalyDetector(simple_predictor)
        assert detector.predictor is simple_predictor
        assert detector.layer_index == -2
        assert detector.tree is None
        assert detector.threshold is None

    def test_from_predictor(self, simple_predictor):
        """Test factory method."""
        detector = AnomalyDetector.from_predictor(simple_predictor, layer_index=-3)
        assert detector.predictor is simple_predictor
        assert detector.layer_index == -3


class TestAnomalyDetectorFit:
    """Tests for fitting the anomaly detector."""

    def test_fit_sets_attributes(self, simple_predictor, training_data):
        """Test that fit sets all required attributes."""
        detector = AnomalyDetector(simple_predictor)
        detector.fit(training_data, k=5, percentile=95.0)

        assert detector.tree is not None
        assert detector.threshold is not None
        assert detector.threshold > 0
        assert detector.k == 5
        assert detector.percentile == 95.0
        assert detector.train_embeddings is not None
        assert detector.train_embeddings.shape[0] == 100

    def test_fit_returns_self(self, simple_predictor, training_data):
        """Test that fit returns self for chaining."""
        detector = AnomalyDetector(simple_predictor)
        result = detector.fit(training_data, k=5)
        assert result is detector

    def test_fit_normalizes_embeddings(self, simple_predictor, training_data):
        """Test that embeddings are L2 normalized."""
        detector = AnomalyDetector(simple_predictor)
        detector.fit(training_data, k=5)

        norms = np.linalg.norm(detector.train_embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


class TestAnomalyDetectorDetect:
    """Tests for anomaly detection."""

    def test_detect_unfitted_raises(self, simple_predictor, training_data):
        """Test that detect raises if not fitted."""
        detector = AnomalyDetector(simple_predictor)
        with pytest.raises(RuntimeError, match="must be fitted"):
            detector.detect(training_data)

    def test_detect_returns_result(self, simple_predictor, training_data):
        """Test that detect returns AnomalyResult."""
        detector = AnomalyDetector(simple_predictor)
        detector.fit(training_data, k=5, percentile=95.0)

        result = detector.detect(training_data[:20])
        assert isinstance(result, AnomalyResult)
        assert result.n_samples == 20
        assert result.threshold == detector.threshold
        assert result.k == 5
        assert result.percentile == 95.0

    def test_detect_distances_positive(self, simple_predictor, training_data):
        """Test that distances are positive."""
        detector = AnomalyDetector(simple_predictor)
        detector.fit(training_data, k=5)

        result = detector.detect(training_data[:20])
        assert np.all(result.distances >= 0)

    def test_detect_finds_anomalies(
        self, simple_predictor, training_data, test_data_with_anomalies
    ):
        """Test that detector finds obvious anomalies."""
        detector = AnomalyDetector(simple_predictor)
        # Use 90th percentile for a more robust threshold with random embeddings
        detector.fit(training_data, k=5, percentile=90.0)

        result = detector.detect(test_data_with_anomalies)

        # The anomalous points (last 20) should have higher detection rate than
        # normal points (first 80). With a 90th percentile threshold, we expect
        # ~10% of normal data to be flagged as anomalies.
        normal_anomaly_rate = result.is_anomaly[:80].mean()
        outlier_anomaly_rate = result.is_anomaly[80:].mean()

        # Outliers should be detected at a higher rate than normal data
        # Even with random embeddings, extreme outliers (5σ + offset) should
        # generally have larger k-NN distances than normal data (0.5σ)
        assert outlier_anomaly_rate > normal_anomaly_rate


class TestAnomalyDetectorTrainingDistances:
    """Tests for getting training distances."""

    def test_get_training_distances(self, simple_predictor, training_data):
        """Test getting training k-NN distances."""
        detector = AnomalyDetector(simple_predictor)
        detector.fit(training_data, k=5)

        distances = detector.get_training_distances()
        assert distances.shape == (100,)
        assert np.all(distances >= 0)

    def test_get_training_distances_unfitted_raises(self, simple_predictor):
        """Test that unfitted detector raises."""
        detector = AnomalyDetector(simple_predictor)
        with pytest.raises(RuntimeError, match="must be fitted"):
            detector.get_training_distances()


class TestAnomalyDetectorSerialization:
    """Tests for saving and loading detector state."""

    def test_save_and_load(self, simple_predictor, training_data):
        """Test saving and loading detector."""
        detector = AnomalyDetector(simple_predictor)
        detector.fit(training_data, k=5, percentile=95.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "detector.pkl"
            detector.save(save_path)

            # Create new detector and load state
            new_detector = AnomalyDetector(simple_predictor)
            new_detector.load_state(save_path)

            assert new_detector.threshold == detector.threshold
            assert new_detector.k == detector.k
            assert new_detector.percentile == detector.percentile
            np.testing.assert_array_equal(
                new_detector.train_embeddings, detector.train_embeddings
            )

    def test_save_unfitted_raises(self, simple_predictor):
        """Test that saving unfitted detector raises."""
        detector = AnomalyDetector(simple_predictor)
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(RuntimeError, match="Cannot save unfitted"),
        ):
            detector.save(Path(tmpdir) / "detector.pkl")

    def test_loaded_detector_can_detect(self, simple_predictor, training_data):
        """Test that loaded detector can make predictions."""
        detector = AnomalyDetector(simple_predictor)
        detector.fit(training_data, k=5)

        original_result = detector.detect(training_data[:10])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "detector.pkl"
            detector.save(save_path)

            new_detector = AnomalyDetector(simple_predictor)
            new_detector.load_state(save_path)

            loaded_result = new_detector.detect(training_data[:10])

            np.testing.assert_array_almost_equal(
                original_result.distances, loaded_result.distances
            )
            np.testing.assert_array_equal(
                original_result.is_anomaly, loaded_result.is_anomaly
            )
