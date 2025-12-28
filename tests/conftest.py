"""
Shared pytest fixtures for DOROTHY tests.

This module provides commonly used fixtures to reduce duplication across test files.
All fixtures are automatically available to tests without explicit imports.

Fixtures defined here:
- simple_config: Basic ExperimentConfig for training tests
- training_data_3ch: Synthetic 3-channel spectral data (X, y)
- simple_model: Small MLP model for quick tests
- simple_input: Random input data
- simple_normalizer: Fitted LabelNormalizer
"""

from pathlib import Path

import numpy as np
import pytest

from dorothy.config.schema import (
    DataConfig,
    ExperimentConfig,
    LossType,
    ModelConfig,
    TrainingConfig,
)
from dorothy.data.normalizer import LabelNormalizer
from dorothy.models import MLP


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def simple_config():
    """Create a simple configuration for testing.

    Returns an ExperimentConfig with minimal settings suitable for unit tests:
    - 3-channel input (flux, sigma, mask)
    - 3 output parameters
    - Small hidden layers for fast execution
    - 5 epochs, batch_size=16
    """
    return ExperimentConfig(
        name="test_experiment",
        data=DataConfig(
            catalogue_path=Path("/fake/catalogue.h5"),
        ),
        model=ModelConfig(
            input_features=3000,  # 3 channels * 1000 wavelengths
            output_features=6,  # 3 params * 2 (value + uncertainty)
            hidden_layers=[32, 16],
        ),
        training=TrainingConfig(
            epochs=5,
            batch_size=16,
            learning_rate=1e-3,
            loss=LossType.HETEROSCEDASTIC,
            gradient_clip=10.0,
            scatter_floor=0.01,
        ),
        seed=42,
    )


# =============================================================================
# Data Fixtures
# =============================================================================


@pytest.fixture
def training_data_3ch():
    """Create synthetic 3-channel training data.

    Returns a tuple (X, y) where:
    - X: shape (100, 3, 1000) = [flux, sigma, mask]
    - y: shape (100, 3, 3) = [values, errors, mask] for 3 params

    Labels simulate Teff (~4500-6500K), logg (~2-5), [Fe/H] (~-1 to 0.5).
    """
    np.random.seed(42)
    n_samples = 100
    n_wavelengths = 1000
    n_params = 3

    # 3-channel spectral data: [flux, sigma, mask]
    flux = np.random.randn(n_samples, n_wavelengths).astype(np.float32)
    sigma = (np.abs(np.random.randn(n_samples, n_wavelengths)) * 0.1 + 0.01).astype(
        np.float32
    )
    spec_mask = np.ones((n_samples, n_wavelengths), dtype=np.float32)
    X = np.stack([flux, sigma, spec_mask], axis=1)

    # 3-channel labels: [values, errors, mask]
    labels = np.column_stack(
        [
            np.random.uniform(4500, 6500, n_samples),  # teff
            np.random.uniform(2.0, 5.0, n_samples),  # logg
            np.random.uniform(-1.0, 0.5, n_samples),  # feh
        ]
    ).astype(np.float32)
    errors = np.column_stack(
        [
            np.abs(np.random.randn(n_samples)) * 50 + 10,  # teff_err
            np.abs(np.random.randn(n_samples)) * 0.1 + 0.05,  # logg_err
            np.abs(np.random.randn(n_samples)) * 0.05 + 0.02,  # feh_err
        ]
    ).astype(np.float32)
    label_mask = np.ones((n_samples, n_params), dtype=np.float32)
    y = np.stack([labels, errors, label_mask], axis=1)

    return X, y


@pytest.fixture
def simple_input():
    """Create simple input data for prediction tests.

    Returns array of shape (50, 100) with random values.
    """
    np.random.seed(42)
    return np.random.randn(50, 100).astype(np.float32)


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def simple_model():
    """Create a simple MLP model for testing.

    Returns a small model with:
    - 100 input features
    - 6 output features (3 params * 2)
    - Hidden layers: [32, 16]

    Model is set to eval mode.
    """
    model = MLP(
        input_features=100,
        output_features=6,
        hidden_layers=[32, 16],
    )
    model.eval()
    return model


# =============================================================================
# Normalizer Fixtures
# =============================================================================


@pytest.fixture
def simple_normalizer():
    """Create a fitted LabelNormalizer for testing.

    Returns a normalizer fitted on synthetic data for teff, logg, feh.
    """
    np.random.seed(42)
    y = np.column_stack(
        [
            np.random.uniform(4000, 6000, 100),  # teff
            np.random.uniform(2, 5, 100),  # logg
            np.random.uniform(-1, 0.5, 100),  # feh
        ]
    )
    normalizer = LabelNormalizer(parameters=["teff", "logg", "feh"])
    normalizer.fit(y)
    return normalizer


# =============================================================================
# Helper Functions (not fixtures, but commonly used)
# =============================================================================


def split_3ch_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Split 3-channel data into train/val/test sets.

    Args:
        X: Spectral data of shape (N, 3, wavelengths)
        y: Labels of shape (N, 3, n_params)
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        seed: Random seed for reproducibility

    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    n_samples = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return (
        (X[train_idx], y[train_idx]),
        (X[val_idx], y[val_idx]),
        (X[test_idx], y[test_idx]),
    )
