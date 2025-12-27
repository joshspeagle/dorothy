"""
Label normalization for stellar parameter prediction.

This module implements the median/IQR normalization scheme used by DOROTHY models.
Labels are normalized to have zero median and unit IQR (interquartile range),
which provides robustness to outliers compared to standard scaling.

Special handling for Teff:
    Temperature is normalized in log10 space to better capture the distribution
    of stellar temperatures across the HR diagram.

The normalizer also handles uncertainty (error) columns, scaling them by the
same IQR factor as their corresponding labels.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dorothy.config.schema import STELLAR_PARAMETERS


@dataclass
class ParameterStats:
    """
    Statistics for a single stellar parameter.

    Attributes:
        name: Parameter name (e.g., "teff", "logg", "feh").
        median: Median value for normalization.
        iqr: Interquartile range (75th - 25th percentile).
        use_log: Whether to apply log10 before normalization (for Teff).
        log_median: Median of log10 values (only if use_log=True).
        log_iqr: IQR of log10 values (only if use_log=True).
    """

    name: str
    median: float
    iqr: float
    use_log: bool = False
    log_median: float | None = None
    log_iqr: float | None = None

    def __post_init__(self) -> None:
        """Validate that log stats are provided if use_log is True."""
        if self.use_log and (self.log_median is None or self.log_iqr is None):
            raise ValueError(
                f"log_median and log_iqr required when use_log=True for {self.name}"
            )


@dataclass
class LabelNormalizer:
    """
    Normalizer for stellar parameter labels using median/IQR scaling.

    This normalizer:
    1. Fits on training data to compute median and IQR for each parameter
    2. Normalizes labels to ~zero median and unit IQR
    3. Applies log10 transformation for Teff before normalization
    4. Handles error columns by scaling with the same IQR

    The normalization formula is:
        normalized = (value - median) / IQR

    For Teff:
        log_value = log10(value)
        normalized = (log_value - log_median) / log_IQR

    Attributes:
        parameters: List of parameter names to normalize.
        stats: Dictionary mapping parameter names to their statistics.
        is_fitted: Whether the normalizer has been fitted to data.

    Example:
        >>> normalizer = LabelNormalizer()
        >>> normalizer.fit(y_train)
        >>> y_train_norm = normalizer.transform(y_train)
        >>> y_pred_denorm = normalizer.inverse_transform(y_pred)
    """

    parameters: list[str] = field(default_factory=lambda: list(STELLAR_PARAMETERS))
    stats: dict[str, ParameterStats] = field(default_factory=dict)
    is_fitted: bool = False

    def __post_init__(self) -> None:
        """Validate parameter list."""
        if not self.parameters:
            raise ValueError("parameters list cannot be empty")

    @property
    def n_parameters(self) -> int:
        """Number of parameters being normalized."""
        return len(self.parameters)

    def fit(
        self,
        y: np.ndarray,
        errors: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> LabelNormalizer:
        """
        Fit the normalizer on training labels.

        Computes median and IQR for each parameter. For Teff (first parameter
        by convention), computes statistics in log10 space.

        Args:
            y: Labels array of shape (n_samples, n_parameters).
            errors: Optional error array of shape (n_samples, n_parameters).
                Not used for fitting, but validates shapes if provided.
            mask: Optional mask array of shape (n_samples, n_parameters).
                Values of 1 indicate valid labels, 0 indicates masked.
                When provided, statistics are computed only on unmasked values
                for each parameter.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If y has wrong shape or contains invalid values.
        """
        if y.ndim != 2:
            raise ValueError(f"y must be 2D, got shape {y.shape}")

        if y.shape[1] != self.n_parameters:
            raise ValueError(
                f"y has {y.shape[1]} columns but expected {self.n_parameters} parameters"
            )

        if errors is not None and errors.shape != y.shape:
            raise ValueError(
                f"errors shape {errors.shape} doesn't match y shape {y.shape}"
            )

        if mask is not None and mask.shape != y.shape:
            raise ValueError(f"mask shape {mask.shape} doesn't match y shape {y.shape}")

        self.stats = {}

        for i, param_name in enumerate(self.parameters):
            # Get mask for this parameter if provided
            if mask is not None:
                param_mask = mask[:, i].astype(bool)
                values = y[param_mask, i]
                if len(values) == 0:
                    raise ValueError(f"All values masked for parameter {param_name}")
            else:
                values = y[:, i]

            # Check for invalid values in unmasked data
            if np.any(np.isnan(values)):
                raise ValueError(f"NaN values found in {param_name}")
            if np.any(np.isinf(values)):
                raise ValueError(f"Inf values found in {param_name}")

            # Teff uses log10 space (by convention, it's the first parameter)
            use_log = param_name == "teff"

            if use_log:
                if np.any(values <= 0):
                    raise ValueError(
                        f"Non-positive values found in {param_name} (log space)"
                    )

                log_values = np.log10(values)
                log_median = float(np.median(log_values))
                log_iqr = float(
                    np.percentile(log_values, 75) - np.percentile(log_values, 25)
                )

                if log_iqr == 0:
                    raise ValueError(f"Zero IQR in log space for {param_name}")

                self.stats[param_name] = ParameterStats(
                    name=param_name,
                    median=float(np.median(values)),
                    iqr=float(np.percentile(values, 75) - np.percentile(values, 25)),
                    use_log=True,
                    log_median=log_median,
                    log_iqr=log_iqr,
                )
            else:
                median = float(np.median(values))
                iqr = float(np.percentile(values, 75) - np.percentile(values, 25))

                if iqr == 0:
                    raise ValueError(f"Zero IQR for {param_name}")

                self.stats[param_name] = ParameterStats(
                    name=param_name,
                    median=median,
                    iqr=iqr,
                    use_log=False,
                )

        self.is_fitted = True
        return self

    def transform(
        self,
        y: np.ndarray,
        errors: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Normalize labels (and optionally errors).

        Args:
            y: Labels array of shape (n_samples, n_parameters).
            errors: Optional error array of shape (n_samples, n_parameters).
            mask: Optional mask array of shape (n_samples, n_parameters).
                Values of 1 indicate valid labels, 0 indicates masked.
                The mask is not modified; masked positions are still normalized
                but their values may be meaningless.

        Returns:
            If errors is None: normalized labels of shape (n_samples, n_parameters).
            If errors is provided: tuple of (normalized_labels, normalized_errors).

        Raises:
            RuntimeError: If normalizer has not been fitted.
            ValueError: If input shape doesn't match fitted parameters.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")

        if y.ndim != 2 or y.shape[1] != self.n_parameters:
            raise ValueError(
                f"y shape {y.shape} doesn't match expected (n_samples, {self.n_parameters})"
            )

        if errors is not None and errors.shape != y.shape:
            raise ValueError(
                f"errors shape {errors.shape} doesn't match y shape {y.shape}"
            )

        if mask is not None and mask.shape != y.shape:
            raise ValueError(f"mask shape {mask.shape} doesn't match y shape {y.shape}")

        y_norm = np.zeros_like(y, dtype=np.float64)
        errors_norm = np.zeros_like(y, dtype=np.float64) if errors is not None else None

        for i, param_name in enumerate(self.parameters):
            stats = self.stats[param_name]
            values = y[:, i]

            if stats.use_log:
                log_values = np.log10(values)
                y_norm[:, i] = (log_values - stats.log_median) / stats.log_iqr

                if errors is not None:
                    # Error propagation for log transform: d(log10(x))/dx = 1/(x*ln(10))
                    # So error in log space = error / (value * ln(10))
                    log_errors = errors[:, i] / (values * np.log(10))
                    errors_norm[:, i] = log_errors / stats.log_iqr
            else:
                y_norm[:, i] = (values - stats.median) / stats.iqr

                if errors is not None:
                    errors_norm[:, i] = errors[:, i] / stats.iqr

        if errors is not None:
            return y_norm, errors_norm
        return y_norm

    def inverse_transform(
        self,
        y_norm: np.ndarray,
        errors_norm: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Convert normalized predictions back to physical units.

        Args:
            y_norm: Normalized predictions of shape (n_samples, n_parameters).
            errors_norm: Optional normalized errors of shape (n_samples, n_parameters).

        Returns:
            If errors_norm is None: denormalized values of shape (n_samples, n_parameters).
            If errors_norm is provided: tuple of (denormalized_values, denormalized_errors).

        Raises:
            RuntimeError: If normalizer has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")

        if y_norm.ndim != 2 or y_norm.shape[1] != self.n_parameters:
            raise ValueError(
                f"y_norm shape {y_norm.shape} doesn't match expected "
                f"(n_samples, {self.n_parameters})"
            )

        y = np.zeros_like(y_norm, dtype=np.float64)
        errors = (
            np.zeros_like(y_norm, dtype=np.float64) if errors_norm is not None else None
        )

        for i, param_name in enumerate(self.parameters):
            stats = self.stats[param_name]
            values_norm = y_norm[:, i]

            if stats.use_log:
                # Inverse: log_value = normalized * log_iqr + log_median
                #          value = 10^log_value
                log_values = values_norm * stats.log_iqr + stats.log_median
                y[:, i] = 10**log_values

                if errors_norm is not None:
                    # Error in log space
                    log_errors = errors_norm[:, i] * stats.log_iqr
                    # Inverse of log transform error propagation
                    errors[:, i] = y[:, i] * np.log(10) * log_errors
            else:
                y[:, i] = values_norm * stats.iqr + stats.median

                if errors_norm is not None:
                    errors[:, i] = errors_norm[:, i] * stats.iqr

        if errors_norm is not None:
            return y, errors
        return y

    def get_params_dict(self) -> dict[str, dict[str, Any]]:
        """
        Get normalization parameters as a dictionary.

        Returns a dictionary compatible with the original DOROTHY format
        for saving to pickle files.

        Returns:
            Dictionary mapping parameter names to their normalization stats.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")

        params = {}
        for param_name, stats in self.stats.items():
            params[param_name] = {
                "median": stats.median,
                "IQR": stats.iqr,
            }
            if stats.use_log:
                params[param_name]["log_median"] = stats.log_median
                params[param_name]["log_IQR"] = stats.log_iqr

        return params

    def save(self, path: str | Path) -> None:
        """
        Save normalizer parameters to a pickle file.

        Args:
            path: Path to save the pickle file.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")

        params = self.get_params_dict()
        with open(path, "wb") as f:
            pickle.dump(params, f)

    @classmethod
    def load(
        cls, path: str | Path, parameters: list[str] | None = None
    ) -> LabelNormalizer:
        """
        Load normalizer from a pickle file.

        Supports loading from the original DOROTHY params format.

        Args:
            path: Path to the pickle file.
            parameters: Optional list of parameter names. If not provided,
                uses the parameters found in the saved file.

        Returns:
            Fitted LabelNormalizer instance.
        """
        with open(path, "rb") as f:
            params = pickle.load(f)

        if parameters is None:
            # Use parameters from saved file
            parameters = list(params.keys())

        normalizer = cls(parameters=parameters)

        for param_name in parameters:
            if param_name not in params:
                raise ValueError(f"Parameter {param_name} not found in saved params")

            param_dict = params[param_name]
            use_log = "log_median" in param_dict or "log_IQR" in param_dict

            normalizer.stats[param_name] = ParameterStats(
                name=param_name,
                median=param_dict["median"],
                iqr=param_dict["IQR"],
                use_log=use_log,
                log_median=param_dict.get("log_median"),
                log_iqr=param_dict.get("log_IQR"),
            )

        normalizer.is_fitted = True
        return normalizer
